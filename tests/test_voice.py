from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from adanvil.audio import audio_duration, rms_per_second, voice_music_mix
from adanvil.ffmpeg import probe
from adanvil.voice import worker
from adanvil.voice.engines import FakeEngine, VoiceUnavailableError, WorkerEngine
from adanvil.voice.generate import VoiceSpec, VoiceTrack, generate_voice
from adanvil.voice.setup import voice_setup


def _fixture(media: dict[str, dict[str, object]], name: str) -> Path:
    return Path(str(media[name]["path"]))


def test_fake_voice_track_is_trimmed_and_round_trips(tmp_path: Path) -> None:
    spec = VoiceSpec(
        engine="fake",
        language="fr",
        lines=["Bonjour", "Une phrase courte", "À bientôt"],
        gap=0.15,
    )
    track = generate_voice(tmp_path, spec, "demo", engine=FakeEngine())

    assert track.language == "fr"
    assert len(track.lines) == 3
    for line in track.lines:
        path = tmp_path / line.file
        assert path.is_file()
        assert probe(path).audio_sample_rate == 48_000
        assert probe(path).audio_channels == 1
        expected = 0.06 * len(line.text) + spec.lead + spec.tail
        assert line.duration == pytest.approx(expected, abs=0.05)
    assert track.lines[1].start == pytest.approx(
        track.lines[0].start + track.lines[0].duration + spec.gap, abs=0.01
    )
    assert track.total_duration == pytest.approx(
        track.lines[-1].start + track.lines[-1].duration, abs=0.01
    )
    loaded = VoiceTrack.model_validate_json((tmp_path / track.manifest).read_text(encoding="utf-8"))
    assert loaded == track
    assert audio_duration(tmp_path / "voice" / "demo" / "voice_full.wav") == pytest.approx(
        track.total_duration, abs=0.01
    )


def test_voice_progress_reports_tts_and_checked_lines(tmp_path: Path) -> None:
    events: list[dict[str, object]] = []
    generate_voice(
        tmp_path,
        VoiceSpec(engine="fake", lines=["one", "two"]),
        "progress",
        engine=FakeEngine(),
        progress=events.append,
    )

    assert any(event["stage"] == "tts" for event in events)
    assert events[-1] == {
        "stage": "complete",
        "done": 2,
        "total": 2,
        "unit": "lines",
        "message": "voice generation complete",
    }


def test_voice_cache_avoids_second_synthesis(tmp_path: Path) -> None:
    spec = VoiceSpec(engine="fake", lines=["Cache me"])
    engine = FakeEngine()
    generate_voice(tmp_path, spec, "cached", engine=engine)
    calls = engine.synthesis_calls
    generate_voice(tmp_path, spec, "cached", engine=engine)
    assert calls == 1
    assert engine.synthesis_calls == calls


def test_fake_mispronunciation_retries_and_is_reported(tmp_path: Path) -> None:
    spec = VoiceSpec(engine="fake", language="fr", lines=["Bonjour [mispronounce] monde"])
    engine = FakeEngine()
    track = generate_voice(tmp_path, spec, "suspect", engine=engine)

    assert engine.synthesis_calls == 3
    assert track.lines[0].attempts == 3
    assert track.lines[0].status == "suspect"
    assert track.warnings


def test_worker_protocol_without_heavy_dependencies(tmp_path: Path) -> None:
    worker = Path(__file__).parents[1] / "src" / "adanvil" / "voice" / "worker.py"
    for jobs, expected in [({"ops": []}, []), ({"ops": [{"op": "nope"}]}, [False])]:
        jobs_path = tmp_path / "jobs.json"
        result_path = tmp_path / "result.json"
        jobs_path.write_text(json.dumps(jobs), encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(worker), str(jobs_path), str(result_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0
        results = json.loads(result_path.read_text(encoding="utf-8"))["results"]
        expected_values = [True] * len(expected) if not expected else expected
        assert [item["ok"] for item in results] == expected_values


def test_worker_import_failure_explains_install(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __import__

    def fail_chatterbox(name: str, *args: object, **kwargs: object):
        if name.startswith("chatterbox"):
            raise ModuleNotFoundError("no chatterbox")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fail_chatterbox)
    result = worker._tts(
        {"text": "hello", "language": "fr", "reference": "ref.wav", "out": "out.wav"},
        {},
    )
    assert result["ok"] is False
    assert "pip install chatterbox-tts faster-whisper" in result["error"]


def test_worker_align_passes_null_language_for_auto_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    class FakeModel:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def transcribe(self, _audio: object, **kwargs: object):
            calls.append(kwargs["language"])
            return [SimpleNamespace(text="", words=[])], None

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeModel))
    monkeypatch.setattr(worker, "_load_16k", lambda path: path)

    result = worker._align({"audio": "reference.wav", "language": None, "model": "small"}, {})

    assert result["ok"] is True
    assert calls == [None]


def test_worker_missing_interpreter_mentions_setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # No voice environment, whatever is installed on this machine.
    monkeypatch.setenv("ADANVIL_CACHE", str(tmp_path / "no-voice-environment"))
    monkeypatch.delenv("ADANVIL_VOICE_PYTHON", raising=False)
    with pytest.raises(VoiceUnavailableError, match="voice_setup"):
        WorkerEngine(python="definitely-not-an-interpreter").synthesize([])


def test_voice_setup_plan_is_network_free() -> None:
    commands = voice_setup(plan=True)
    rendered = " ".join(" ".join(command) for command in commands)
    assert "chatterbox-tts" in rendered
    assert "faster-whisper" in rendered


def test_voice_music_mix_has_voice_and_music_tail(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    spec = VoiceSpec(engine="fake", lines=["Voice first"])
    track = generate_voice(tmp_path, spec, "mix", engine=FakeEngine())
    duration = track.total_duration + 1.8
    output = voice_music_mix(
        tmp_path / track.manifest.replace("manifest.json", "voice_full.wav"),
        duration,
        tmp_path / "mix.m4a",
        music=_fixture(media, "music.m4a"),
    )
    assert audio_duration(output) == pytest.approx(duration, abs=0.04)
    levels = rms_per_second(output)
    assert all(level > -45 for level in levels[:-1])
    assert levels[-1] > -40
    assert max(levels[:-1]) > levels[-1]

    with pytest.raises(ValueError, match="cut the voice"):
        voice_music_mix(
            tmp_path / track.manifest.replace("manifest.json", "voice_full.wav"),
            track.total_duration - 0.1,
            tmp_path / "too-short.m4a",
        )


def test_similarity_ignores_numbers_and_currency_but_not_words() -> None:
    from adanvil.voice.generate import _similarity

    assert _similarity("Il coûte quarante-neuf euros", "il coûte 49 euros") == 1.0
    assert _similarity("quarante-neuf euros quatre-vingt-dix", "49,90 €") == 1.0
    assert _similarity("Seulement vingt et un dollars", "seulement 21 $") == 1.0
    assert _similarity("only ninety nine dollars", "only 79.90 dollars") == 1.0
    assert _similarity("Une soupe légère", "Une soupe légère") == 1.0
    assert _similarity("Il coûte quarante-neuf euros", "elle vend ce produit 49 euros") < 0.85
    assert _similarity("Une soupe légère", "Une soupe lourde") < 0.85


def test_say_show_lines_are_stored_and_replace_same_name(tmp_path: Path) -> None:
    from adanvil.storyboard import VoiceCaptionsOverlay

    spec = VoiceSpec(
        engine="fake",
        language="fr",
        lines=["Bonjour", {"say": "quarante-neuf euros quatre-vingt-dix", "show": "49,90 €"}],
    )
    engine = FakeEngine()
    track = generate_voice(tmp_path, spec, "price", engine=engine)
    assert track.lines[0].display is None
    assert track.lines[1].text == "quarante-neuf euros quatre-vingt-dix"
    assert track.lines[1].display == "49,90 €"
    assert track.lines[1].status == "ok"

    sentence = VoiceCaptionsOverlay(manifest=track.manifest).expand(track)
    assert [" ".join(c.lines) for c in sentence] == ["Bonjour", "49,90 €"]
    chunks = VoiceCaptionsOverlay(manifest=track.manifest, mode="chunks", max_words=1).expand(track)
    assert [" ".join(c.lines) for c in chunks] == ["Bonjour", "49,90 €"]

    calls = engine.synthesis_calls
    again = generate_voice(tmp_path, spec, "price", engine=engine)
    assert again == track
    changed = VoiceSpec(
        engine="fake",
        language="fr",
        lines=["Bonjour", {"say": "quarante-neuf euros quatre-vingt-dix", "show": "49,90 EUR"}],
    )
    replaced = generate_voice(tmp_path, changed, "price", engine=engine)
    assert replaced.lines[1].display == "49,90 EUR"
    assert engine.synthesis_calls == calls


class _LanguageSpy(FakeEngine):
    """A fake engine that remembers the language each line was synthesised in."""

    def __init__(self) -> None:
        super().__init__()
        self.languages: list[str] = []

    def synthesize(self, lines, *, progress=None):
        self.languages.extend(line.language for line in lines)
        return super().synthesize(lines, progress=progress)


def _brand_language(root: Path, language: str | None) -> None:
    from adanvil.brand import Brand, save_brand

    save_brand(Brand(name="Acme", language=language), root / "brand.yaml")


def test_voice_language_is_optional_and_defaults_to_english(tmp_path: Path) -> None:
    assert VoiceSpec(engine="fake", lines=["Hello"]).language is None

    engine = _LanguageSpy()
    spec = VoiceSpec(engine="fake", lines=["Hello there"])
    track = generate_voice(tmp_path, spec, "a", engine=engine)
    assert track.language == "en"
    assert engine.languages == ["en"]

    # A brand profile without a language changes nothing.
    _brand_language(tmp_path, None)
    spec = VoiceSpec(engine="fake", lines=["Hello again"])
    again = generate_voice(tmp_path, spec, "b", engine=engine)
    assert again.language == "en"
    # A broken brand profile never stops a voice job.
    (tmp_path / "brand.yaml").write_text("name: [unclosed", encoding="utf-8")
    spec = VoiceSpec(engine="fake", lines=["Hello third"])
    broken = generate_voice(tmp_path, spec, "c", engine=engine)
    assert broken.language == "en"


def test_voice_language_follows_the_brand_profile_unless_set(tmp_path: Path) -> None:
    engine = _LanguageSpy()
    _brand_language(tmp_path, "fr-FR")
    spec = VoiceSpec(engine="fake", lines=["Bonjour"])
    track = generate_voice(tmp_path, spec, "a", engine=engine)
    assert track.language == "fr"
    assert engine.languages == ["fr"]
    stored = VoiceTrack.model_validate_json((tmp_path / track.manifest).read_text(encoding="utf-8"))
    assert stored.language == "fr"

    _brand_language(tmp_path, "Spanish")
    named = generate_voice(tmp_path, VoiceSpec(engine="fake", lines=["Hola"]), "b", engine=engine)
    assert named.language == "es"

    # An explicit language wins over the brand's, and is used as written.
    explicit = generate_voice(
        tmp_path, VoiceSpec(engine="fake", language="de", lines=["Hallo"]), "c", engine=engine
    )
    assert explicit.language == "de"
    assert engine.languages == ["fr", "es", "de"]

    # Changing the brand language makes a cached voice stale: it is generated again.
    spec = VoiceSpec(engine="fake", lines=["Bonjour"])
    _brand_language(tmp_path, "en")
    regenerated = generate_voice(tmp_path, spec, "a", engine=engine)
    assert regenerated.language == "en"
    assert engine.languages[-1] == "en"


def test_old_manifest_without_display_still_loads(tmp_path: Path) -> None:
    track = generate_voice(
        tmp_path,
        VoiceSpec(engine="fake", language="fr", lines=["Bonjour"]),
        "old",
        engine=FakeEngine(),
    )
    path = tmp_path / track.manifest
    data = json.loads(path.read_text(encoding="utf-8"))
    for line in data["lines"]:
        del line["display"]
    path.write_text(json.dumps(data), encoding="utf-8")
    assert VoiceTrack.model_validate_json(path.read_text(encoding="utf-8")).lines[0].display is None
