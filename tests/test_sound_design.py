"""Sound design: the sfx pack, sfx overlays and sfx_auto, the beat grid and cuts on the beat."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import wave
import zipfile
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import pytest
from mcp import Client

from adanvil import audio, sfx
from adanvil.cli import main
from adanvil.ffmpeg import probe, run
from adanvil.project import init_project
from adanvil.render import compose
from adanvil.server import build_server
from adanvil.storyboard import Storyboard, storyboard_json_schema, validate_storyboard
from adanvil.storyboard.sfx import all_sfx, expand_sfx_auto
from adanvil.storyboard.snap import snap_storyboard
from adanvil.voice import FakeEngine, VoiceSpec, generate_voice

SR = 48_000
FRAME = 1 / 30


# ----------------------------------------------------------------------------- helpers


def _write(path: Path, samples: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def _click_times(bpm: float, first: float, seconds: float) -> list[float]:
    period = 60.0 / bpm
    times = []
    moment = first
    while moment < seconds - 0.3:
        times.append(round(moment * SR) / SR)
        moment += period
    return times


def _click_track(path: Path, bpm: float, *, first: float = 0.0, seconds: float = 20.0) -> Path:
    """A click (a 1 ms burst) on every beat."""
    samples = np.zeros(int(seconds * SR))
    burst = 0.9 * np.hanning(48) * np.sign(np.sin(np.arange(48) * 1.3))
    for moment in _click_times(bpm, first, seconds):
        index = round(moment * SR)
        samples[index : index + 48] += burst
    return _write(path, samples)


def _decode(path: Path, channels: int = 1) -> np.ndarray:
    decoded = run(
        [
            *("-i", str(path), "-map", "0:a:0", "-ac", str(channels)),
            *("-ar", str(SR), "-f", "f32le", "-"),
        ],
        capture_stdout=True,
    )
    return np.frombuffer(decoded.stdout or b"", dtype="<f4")


def _clicks_in(samples: np.ndarray, threshold: float = 0.3) -> list[float]:
    """Times of the clicks of a signal (a click is a spike of the second difference)."""
    spikes = np.abs(np.diff(samples.astype(float), 2))
    found: list[float] = []
    index = 0
    while index < len(spikes):
        if spikes[index] > threshold:
            window = spikes[index : index + int(0.05 * SR)]
            found.append((index + int(np.argmax(window))) / SR)
            index += int(0.1 * SR)
        else:
            index += 1
    return found


def _project(tmp_path: Path, media: dict[str, dict[str, object]] | None = None) -> Path:
    root = tmp_path / "project"
    init_project(root, "sound")
    if media is not None:
        for name in ("sdr_portrait.mp4", "cta.png", "music_short.m4a"):
            shutil.copy2(Path(str(media[name]["path"])), root / "footage" / name)
    return root


def _images(root: Path, media: dict[str, dict[str, object]]) -> None:
    shutil.copy2(Path(str(media["cta.png"]["path"])), root / "footage" / "cta.png")


def _image_ad(durations: list[float], **extra: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "name": "snap",
        "emoji": "none",
        "segments": [
            {"type": "image", "source": "footage/cta.png", "duration": duration}
            for duration in durations
        ],
        "audio": {"music": "music/beat.wav"},
        "output": {"quality": "preview"},
    }
    raw.update(extra)
    return raw


# ----------------------------------------------------------------------------- the pack


def test_pack_manifest_lists_pinned_cc0_sources_and_a_fallback_for_every_sound() -> None:
    entries = sfx.load_manifest()
    names = [entry["name"] for entry in entries]
    assert len(names) == len(set(names))
    assert 15 <= len(names) <= 25
    wanted = {
        "whoosh_short", "whoosh_long", "swoosh", "swipe", "pop", "click", "tap", "ding",
        "chime", "notification", "cash_register", "camera_shutter", "record_scratch",
        "typing", "riser_1s", "riser_2s", "impact", "boom",
    }  # fmt: skip
    assert wanted <= set(names)
    downloads = [entry["download"] for entry in entries if "download" in entry]
    assert downloads
    for download in downloads:
        assert download["url"].startswith("https://kenney.nl/media/")
        assert len(download["sha1"]) == 40 and set(download["sha1"]) <= set("0123456789abcdef")
        assert download["licence"] == "CC0-1.0"
        assert download["licence_url"] == "https://creativecommons.org/publicdomain/zero/1.0/"
        assert download["page"].startswith("https://kenney.nl/assets/")
    for entry in entries:  # every sound can be synthesised when its download fails
        sfx.generate_sound(entry["name"])


def test_the_readme_and_third_party_notice_list_every_sound_and_source() -> None:
    repo = Path(__file__).resolve().parents[1]
    readme = (repo / "examples" / "sfx" / "README.md").read_text(encoding="utf-8")
    notice = (repo / "THIRD_PARTY.md").read_text(encoding="utf-8")
    for entry in sfx.load_manifest():
        assert f"`{entry['name']}`" in readme, entry["name"]
        if "download" in entry:
            assert entry["download"]["member"] in readme
            assert entry["download"]["sha1"] in readme
    assert "CC0" in readme and "generated" in readme
    assert "kenney.nl" in notice and "sfx" in notice.lower()


def test_synthesis_is_deterministic_and_leveled(tmp_path: Path) -> None:
    first = sfx.fetch_pack(tmp_path / "a", offline=True)
    second = sfx.fetch_pack(tmp_path / "b", offline=True)
    assert first["downloaded"] == [] and len(first["generated"]) == len(sfx.load_manifest())
    assert second["warnings"] == []
    for entry in sfx.load_manifest():
        name = entry["name"]
        one = (tmp_path / "a" / "sfx" / f"{name}.wav").read_bytes()
        two = (tmp_path / "b" / "sfx" / f"{name}.wav").read_bytes()
        assert hashlib.sha1(one).hexdigest() == hashlib.sha1(two).hexdigest(), name
        with wave.open(str(tmp_path / "a" / "sfx" / f"{name}.wav"), "rb") as handle:
            assert (handle.getnchannels(), handle.getframerate()) == (1, SR)
            data = np.frombuffer(handle.readframes(handle.getnframes()), "<i2") / 32768.0
        assert 0.0 < len(data) / SR <= 2.0
        assert audio_peak(data) <= sfx.PEAK_CEILING_DB + 0.05, name
        assert sfx.momentary_max_lufs(data) <= sfx.TARGET_LUFS + 0.1, name
        assert abs(data[0]) < 0.05 and abs(data[-1]) < 0.01, name  # no click at either end
    # Risers keep their nominal length: their end is placed on the moment they build to.
    assert sfx.generate_sound("riser_1s")[1]["duration"] == pytest.approx(1.0, abs=0.01)
    assert sfx.generate_sound("riser_2s")[1]["duration"] == pytest.approx(2.0, abs=0.01)


def audio_peak(data: np.ndarray) -> float:
    return 20 * np.log10(float(np.max(np.abs(data))))


def test_fetch_pack_writes_licences_and_keeps_existing_files(tmp_path: Path) -> None:
    root = tmp_path / "p"
    sfx.fetch_pack(root, offline=True, names=["pop", "ding"])
    folder = root / "sfx"
    assert sorted(item.name for item in folder.glob("*.wav")) == ["ding.wav", "pop.wav"]
    licences = (folder / "LICENCES.txt").read_text(encoding="utf-8")
    assert "pop: generated by AdAnvil" in licences
    stamp = (folder / "pop.wav").stat().st_mtime_ns
    again = sfx.fetch_pack(root, offline=True, names=["pop", "swoosh"])
    assert again["kept"] == ["pop"] and again["built"] == ["swoosh"]
    assert (folder / "pop.wav").stat().st_mtime_ns == stamp
    pack = json.loads((folder / "pack.json").read_text(encoding="utf-8"))
    assert {item["name"] for item in pack["sounds"]} == {"pop", "ding", "swoosh"}
    with pytest.raises(sfx.SfxError, match="unknown sound"):
        sfx.fetch_pack(root, names=["nope"])


def _fake_archive(destination: Path, members: list[str]) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        source = io.BytesIO()
        with wave.open(source, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(44_100)
            t = np.arange(4410) / 44_100
            tone = 0.5 * np.sin(2 * np.pi * 880 * t) * np.exp(-t / 0.03)
            handle.writeframes((tone * 32767).astype("<i2").tobytes())
        for member in members:
            bundle.writestr(member, source.getvalue())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(buffer.getvalue())


def test_downloads_are_verified_decoded_and_labelled_cc0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    members = [e["download"]["member"] for e in sfx.load_manifest() if "download" in e]
    calls: list[str] = []

    def fake_download(url: str, sha1: str, destination: Path, **_kw: Any) -> Path:
        calls.append(url)
        _fake_archive(destination, members)
        return destination

    monkeypatch.setattr(sfx, "download_file", fake_download)
    result = sfx.fetch_pack(tmp_path / "p")
    assert result["warnings"] == []
    downloaded = [e["name"] for e in sfx.load_manifest() if "download" in e]
    assert result["downloaded"] == sorted(downloaded)
    assert len(calls) == len(set(calls)) == 2  # one archive per pack, however many sounds
    pack = json.loads((tmp_path / "p" / "sfx" / "pack.json").read_text(encoding="utf-8"))
    by_name = {item["name"]: item for item in pack["sounds"]}
    assert by_name["ding"]["licence"] == "CC0-1.0" and by_name["ding"]["source"] == "download"
    assert by_name["pop"]["source"] == "generated"
    licences = (tmp_path / "p" / "sfx" / "LICENCES.txt").read_text(encoding="utf-8")
    assert "kenney.nl" in licences and "CC0-1.0" in licences


def test_a_failed_download_is_replaced_by_the_generated_sound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_args: Any, **_kw: Any) -> Path:
        raise sfx.SfxError("no network")

    monkeypatch.setattr(sfx, "download_file", broken)
    result = sfx.fetch_pack(tmp_path / "p", names=["ding", "pop"])
    assert result["downloaded"] == [] and result["generated"] == ["ding", "pop"]
    assert any("no network" in warning for warning in result["warnings"])
    pack = json.loads((tmp_path / "p" / "sfx" / "pack.json").read_text(encoding="utf-8"))
    ding = next(item for item in pack["sounds"] if item["name"] == "ding")
    assert ding["source"] == "generated" and ding["note"] == "download failed"


def test_download_file_discards_a_file_with_the_wrong_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_fetch(url: str, destination: Path, **_kw: Any) -> None:
        Path(destination).write_bytes(b"payload")

    monkeypatch.setattr(sfx.netguard, "fetch_to_file", fake_fetch)
    target = tmp_path / "x.zip"
    with pytest.raises(sfx.SfxError, match="SHA-1 mismatch"):
        sfx.download_file("https://example.test/x.zip", "0" * 40, target)
    assert not target.exists() and not (tmp_path / "x.zip.part").exists()
    good = hashlib.sha1(b"payload").hexdigest()
    assert sfx.download_file("https://example.test/x.zip", good, target) == target


def test_list_sfx_shows_the_pack_and_the_users_own_files(tmp_path: Path) -> None:
    root = _project(tmp_path)
    empty = sfx.list_sfx(root)
    assert empty["count"] == 0 and "sfx-fetch" in empty["hint"]
    sfx.fetch_pack(root, offline=True, names=["pop"])
    _write(root / "sfx" / "mine.wav", 0.5 * np.sin(2 * np.pi * 440 * np.arange(4800) / SR))
    listed = sfx.list_sfx(root)
    by_name = {item["name"]: item for item in listed["sounds"]}
    assert by_name["pop"]["origin"] == "pack" and by_name["pop"]["source"] == "generated"
    assert by_name["mine"]["origin"] == "user" and by_name["mine"]["duration"] == 0.1
    assert by_name["pop"]["path"] == "sfx/pop.wav"
    assert "whoosh_short" in listed["missing_from_pack"]
    fetched = sfx.list_sfx(root, fetch=True, offline=True)
    assert fetched["missing_from_pack"] == [] and fetched["fetch"]["built"]


def test_sfx_names_resolve_to_files_of_the_sfx_folder(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _write(root / "sfx" / "a.wav", np.zeros(480) + 0.1)
    (root / "sfx" / "b.mp3").write_bytes(b"x")
    assert sfx.sfx_source_path(root, "a") == "sfx/a.wav"
    assert sfx.sfx_source_path(root, "a.wav") == "sfx/a.wav"
    assert sfx.sfx_source_path(root, "sfx/a.wav") == "sfx/a.wav"
    assert sfx.sfx_source_path(root, "b") == "sfx/b.mp3"
    assert sfx.sfx_source_path(root, "missing") == "sfx/missing.wav"
    with pytest.raises(ValueError):
        sfx.resolve_sfx(root, "../outside.wav")


def test_cli_sfx_fetch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _project(tmp_path)
    assert main(["sfx-fetch", str(root), "--offline", "--only", "pop", "click", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["built"] == ["pop", "click"] and payload["generated"] == ["click", "pop"]
    assert (root / "sfx" / "LICENCES.txt").is_file()
    assert main(["sfx-fetch", str(root), "--offline", "--only", "pop"]) == 0
    assert "1 kept" in capsys.readouterr().out
    assert main(["sfx-fetch", str(root), "--only", "nope"]) == 1


# ----------------------------------------------------------------------------- the overlay


def test_sfx_overlay_and_sfx_auto_in_the_schema() -> None:
    overlay = Storyboard.model_validate(_image_ad([2.0], overlays=[
        {"type": "sfx", "source": "sfx/pop.wav", "at": 1.0}
    ])).overlays[0]
    assert overlay.volume == 0.8 and overlay.at == 1.0
    for bad in (
        {"type": "sfx", "source": "sfx/pop.wav", "at": -1},
        {"type": "sfx", "source": "sfx/pop.wav", "at": 1, "volume": 3},
        {"type": "sfx", "source": "../x.wav", "at": 1},
        {"type": "sfx", "source": "", "at": 1},
        {"type": "sfx", "source": "pop", "at": 1, "start": 2},
    ):
        with pytest.raises(ValueError):
            Storyboard.model_validate(_image_ad([2.0], overlays=[bad]))
    with pytest.raises(ValueError, match="unexpected field"):
        Storyboard.model_validate(_image_ad([2.0], audio={"sfx_auto": {"cut": "pop"}}))
    with pytest.raises(ValueError):
        Storyboard.model_validate(_image_ad([2.0], audio={"sfx_auto": {"every": 0}}))
    schema = storyboard_json_schema()
    assert "SfxOverlay" in schema["$defs"] and "SfxAuto" in schema["$defs"]
    assert "sfx_auto" in schema["$defs"]["AudioSpec"]["properties"]


def _timeline(model: Storyboard) -> tuple[list[float], float]:
    return [start for start, _ in model.timeline()], model.total_duration()


def test_sfx_auto_places_sounds_at_cuts_and_caption_starts() -> None:
    raw = _image_ad(
        [1.0, 1.0, 1.0, 1.0],
        audio={"sfx_auto": {"cuts": "whoosh_short", "captions": "pop", "volume": 0.4}},
        overlays=[
            {"type": "caption", "style": "pill", "lines": ["One"], "start": 0.0, "end": 0.9},
            {"type": "caption", "style": "pill", "lines": ["Two"], "start": 1.3, "end": 1.9},
            {"type": "caption", "style": "pill", "lines": ["Three"], "start": 3.7},
            {"type": "sfx", "source": "ding", "at": 2.0},
        ],
    )
    model = Storyboard.model_validate(raw)
    starts, total = _timeline(model)
    auto = expand_sfx_auto(model, starts, total)
    assert [(item["auto"], item["source"], item["at"]) for item in auto] == [
        ("captions", "sfx/pop.wav", 0.0),
        ("cuts", "sfx/whoosh_short.wav", 1.0),
        ("cuts", "sfx/whoosh_short.wav", 2.0),
        ("cuts", "sfx/whoosh_short.wav", 3.0),
        ("captions", "sfx/pop.wav", 3.7),
    ]  # the caption at 1.3 is within 0.6 s of the cut at 1.0, so it gets no sound
    assert all(item["volume"] == 0.4 for item in auto)
    everything = all_sfx(model, starts, total)
    assert [item["auto"] for item in everything if item["at"] == 2.0] == [None, "cuts"]
    assert len(everything) == len(auto) + 1


def test_sfx_auto_every_min_gap_and_the_end() -> None:
    def places(auto: dict[str, Any], durations: list[float]) -> list[float]:
        model = Storyboard.model_validate(_image_ad(durations, audio={"sfx_auto": auto}))
        starts, total = _timeline(model)
        return [item["at"] for item in expand_sfx_auto(model, starts, total)]

    seven = [1.0] * 7
    assert places({"cuts": "x"}, seven) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert places({"cuts": "x", "every": 2}, seven) == [1.0, 3.0, 5.0]
    assert places({"cuts": "x", "every": 3}, seven) == [1.0, 4.0]
    assert places({"cuts": "x", "min_gap": 1.5}, seven) == [1.0, 3.0, 5.0]
    assert places({"cuts": "x", "min_gap": 0.4}, [0.5, 0.3, 0.5, 2.0]) == [0.5, 1.3]
    assert places({"captions": "x"}, seven) == []  # no caption written
    assert places({}, seven) == []


def test_sfx_overlay_validation_summary_and_errors(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _project(tmp_path)
    _images(root, media)
    sfx.fetch_pack(root, offline=True, names=["pop", "whoosh_short"])
    raw = _image_ad(
        [1.5, 1.5],
        audio={"sfx_auto": {"cuts": "whoosh_short"}},
        overlays=[{"type": "sfx", "source": "pop", "at": 0.5, "volume": 0.6}],
    )
    report = validate_storyboard(raw, root=root)
    assert report.ok, report.errors
    assert report.summary["sfx"] == [
        {"type": "sfx", "source": "sfx/pop.wav", "at": 0.5, "volume": 0.6, "auto": None},
        {"type": "sfx", "source": "sfx/whoosh_short.wav", "at": 1.5, "volume": 0.5,
         "auto": "cuts"},
    ]  # fmt: skip
    assert report.summary["overlay_count"] == 1

    missing = _image_ad(
        [1.5, 1.5],
        audio={"sfx_auto": {"captions": "ghost"}},
        overlays=[
            {"type": "sfx", "source": "sfx/nothing.wav", "at": 1.0},
            {"type": "sfx", "source": "pop", "at": 3.0},
        ],
    )
    bad = validate_storyboard(missing, root=root)
    codes = sorted((issue.code, issue.where) for issue in bad.errors)
    assert codes == [
        ("sfx_after_end", "overlays[1].at"),
        ("sfx_missing", "audio.sfx_auto.captions"),
        ("sfx_missing", "overlays[0].source"),
    ]
    assert "sfx-fetch" in bad.errors[0].message + bad.errors[1].message + bad.errors[2].message

    (root / "sfx" / "notes.txt").write_text("not a sound", encoding="utf-8")
    not_audio = _image_ad(
        [1.5], audio={}, overlays=[{"type": "sfx", "source": "sfx/notes.txt", "at": 0}]
    )
    assert [i.code for i in validate_storyboard(not_audio, root=root).errors] == ["sfx_not_audio"]

    empty = _image_ad([1.5], audio={"sfx_auto": {"every": 2}})
    assert [w.code for w in validate_storyboard(empty, root=root).warnings] == ["sfx_auto_empty"]


def test_old_storyboards_are_unchanged(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _project(tmp_path, media)
    raw = {
        "name": "old",
        "emoji": "none",
        "segments": [
            {"type": "clip", "source": "footage/sdr_portrait.mp4", "in": 0.0, "out": 1.0},
            {"type": "clip", "source": "footage/sdr_portrait.mp4", "in": 1.0, "out": 2.0},
        ],
        "audio": {"music": "footage/music_short.m4a"},
        "output": {"quality": "preview"},
    }
    model = Storyboard.model_validate(raw)
    assert model.audio.sfx_auto is None
    assert "sfx_auto" in model.model_dump(mode="json")["audio"]  # None: nothing is placed
    report = validate_storyboard(raw, root=root)
    assert report.ok and "sfx" not in report.summary
    result = compose(model, root=root)
    assert result.sfx is None
    assert not (root / "cache" / "audio-sfx").exists()
    assert (result.path).is_file()


# ----------------------------------------------------------------------------- the mix


def test_mix_sfx_puts_each_sound_at_its_time_and_keeps_the_length(tmp_path: Path) -> None:
    bed = audio.silence(3.0, tmp_path / "bed.m4a")
    burst = 0.9 * np.hanning(48) * np.sign(np.sin(np.arange(48) * 1.3))
    click = _write(tmp_path / "click.wav", np.concatenate([burst, np.zeros(480)]))
    out = audio.mix_sfx(
        bed,
        [audio.SfxHit(click, 0.5), audio.SfxHit(click, 1.5, 0.5), audio.SfxHit(click, 2.9)],
        3.0,
        tmp_path / "mix.wav",
    )
    info = probe(out)
    assert info.audio_duration == pytest.approx(3.0, abs=0.001)
    assert (info.audio_sample_rate, info.audio_channels) == (48_000, 2)
    found = _clicks_in(_decode(out), threshold=0.05)
    assert len(found) == 3
    for got, wanted in zip(found, [0.5, 1.5, 2.9], strict=True):
        assert abs(got - wanted) <= 0.001  # to the millisecond, not just to the frame
    samples = _decode(out)
    assert np.max(np.abs(samples[: int(0.49 * SR)])) < 1e-4  # nothing before the first sound
    assert np.max(np.abs(samples[int(1.5 * SR) : int(1.6 * SR)])) < np.max(
        np.abs(samples[int(0.5 * SR) : int(0.6 * SR)])
    )  # half the volume


def test_mix_sfx_does_not_shift_or_change_the_bed(tmp_path: Path) -> None:
    t = np.arange(int(3.0 * SR)) / SR
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)
    bed = _write(tmp_path / "bed.wav", tone)
    click = _write(tmp_path / "tick.wav", 0.2 * np.sin(2 * np.pi * 900 * t[:2400]))
    out = audio.mix_sfx(bed, [audio.SfxHit(click, 2.5)], 3.0, tmp_path / "mix.wav")
    heard = _decode(out)
    assert len(heard) == 3 * SR
    # Before the sound the mix is the bed itself, sample for sample (the limiter is idle).
    assert np.max(np.abs(heard[: int(2.4 * SR)] - _decode(bed)[: int(2.4 * SR)])) < 0.003


def test_peak_limit_only_touches_what_is_over_the_ceiling() -> None:
    t = np.arange(SR) / SR
    quiet = np.stack([0.3 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1)
    assert audio.peak_limit(quiet, 0.84) is quiet  # nothing to do: the very same samples
    loud = quiet.copy()
    loud[SR // 2 : SR // 2 + 4800] *= 3.2  # 0.96 for 100 ms
    limited = audio.peak_limit(loud, 0.84)
    assert np.max(np.abs(limited)) <= 0.84 + 1e-9
    assert np.array_equal(limited[: SR // 4], loud[: SR // 4])  # far from the peak: untouched
    assert np.max(np.abs(limited[SR // 2 : SR // 2 + 4800])) > 0.7  # no pumping flat-line


def test_mix_sfx_never_runs_past_the_end_and_never_clips(tmp_path: Path) -> None:
    t = np.arange(int(4.0 * SR)) / SR
    loud_bed = _write(tmp_path / "bed.wav", 0.95 * np.sin(2 * np.pi * 220 * t))
    sound = _write(tmp_path / "long.wav", 0.95 * np.sin(2 * np.pi * 1000 * t[: int(1.5 * SR)]))
    out = audio.mix_sfx(
        loud_bed,
        [
            audio.SfxHit(sound, 3.5, 2.0),
            audio.SfxHit(sound, 0.2, 2.0),
            audio.SfxHit(sound, 0.2, 2.0),
        ],
        4.0,
        tmp_path / "mix.wav",
    )
    assert probe(out).audio_duration == pytest.approx(4.0, abs=0.001)
    peak = audio.true_peak(out)
    assert peak is not None and peak <= -1.0
    samples = _decode(out, channels=2)
    assert np.max(np.abs(samples)) <= 10 ** (audio.SFX_CEILING_DB / 20) + 1e-3
    # A sound cut by the end fades out instead of stopping dead.
    quiet = audio.silence(4.0, tmp_path / "quiet.m4a")
    cut = _decode(audio.mix_sfx(quiet, [audio.SfxHit(sound, 3.5)], 4.0, tmp_path / "cut.wav"))
    assert len(cut) == 4 * SR
    middle = np.max(np.abs(cut[int(3.6 * SR) : int(3.9 * SR)]))
    assert np.max(np.abs(cut[-96:])) < 0.25 * middle  # the last 2 ms are the end of the fade
    with pytest.raises(ValueError, match="at least one"):
        audio.mix_sfx(loud_bed, [], 4.0, tmp_path / "none.wav")
    with pytest.raises(ValueError, match="not negative"):
        audio.mix_sfx(loud_bed, [audio.SfxHit(sound, -1.0)], 4.0, tmp_path / "bad.wav")


def test_compose_mixes_sfx_overlays_and_sfx_auto_at_the_right_frames(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _project(tmp_path, media)
    burst = np.concatenate([0.9 * np.hanning(48) * np.sign(np.sin(np.arange(48) * 1.3)),
                            np.zeros(960)])  # fmt: skip
    _write(root / "sfx" / "tick.wav", burst)
    raw = {
        "name": "with_sfx",
        "emoji": "none",
        "segments": [
            {"type": "clip", "source": "footage/sdr_portrait.mp4", "in": 0.2, "out": 1.7},
            {"type": "clip", "source": "footage/sdr_portrait.mp4", "in": 0.0, "out": 1.5},
            {"type": "image", "source": "footage/cta.png", "duration": 1.0},
        ],
        "overlays": [
            {"type": "sfx", "source": "tick", "at": 0.8},
            {"type": "sfx", "source": "sfx/tick.wav", "at": 3.9, "volume": 2.0},
        ],
        "audio": {"music": "footage/music_short.m4a", "sfx_auto": {"cuts": "tick", "volume": 1.0}},
        "output": {"quality": "preview"},
    }
    model = Storyboard.model_validate(raw)
    assert validate_storyboard(raw, root=root).ok
    report = compose(model, root=root)
    info = probe(report.path)
    assert info.audio_duration == pytest.approx(info.duration, abs=0.05)
    assert report.sfx is not None
    assert [(item["at"], item["auto"]) for item in report.sfx] == [
        (0.8, None), (1.5, "cuts"), (3.0, "cuts"), (3.9, None),
    ]  # fmt: skip
    found = _clicks_in(_decode(report.path), threshold=0.15)
    assert len(found) == 4, found
    for got, wanted in zip(found, [0.8, 1.5, 3.0, 3.9], strict=True):
        assert abs(got - wanted) <= FRAME, (got, wanted)
    peak = audio.true_peak(report.path)
    assert peak is not None and peak <= -1.0
    # The same render without the sounds has the same length and the same bed level.
    plain = json.loads(json.dumps(raw))
    plain["name"] = "without_sfx"
    plain["overlays"] = []
    plain["audio"].pop("sfx_auto")
    quiet = compose(Storyboard.model_validate(plain), root=root)
    assert probe(quiet.path).duration == pytest.approx(info.duration, abs=0.001)
    assert quiet.sfx is None


def _energy(samples: np.ndarray, start: float, end: float) -> float:
    return float(np.sum(samples[int(start * SR) : int(end * SR)].astype(float) ** 2))


@pytest.mark.parametrize("mode", ["silence", "clips", "voice"])
def test_sfx_works_with_every_audio_mode(
    tmp_path: Path, media: dict[str, dict[str, object]], mode: str
) -> None:
    root = _project(tmp_path, media)
    shutil.copy2(Path(str(media["music_short.m4a"]["path"])), root / "music" / "bed.m4a")
    tone = 0.6 * np.sin(2 * np.pi * 500 * np.arange(int(0.4 * SR)) / SR)
    _write(root / "sfx" / "beep.wav", tone)
    audio_spec: dict[str, Any] = {}
    if mode == "clips":
        audio_spec = {"clips": True, "music": "music/bed.m4a", "music_volume": 0.3}
    if mode == "voice":
        track = generate_voice(
            root, VoiceSpec(engine="fake", lines=["Hello there"]), "narration", engine=FakeEngine()
        )
        audio_spec = {"voice": track.manifest, "music": "music/bed.m4a"}
    raw = {
        "name": f"mode_{mode}",
        "emoji": "none",
        "segments": [
            {"type": "clip", "source": "footage/sdr_portrait.mp4", "in": 0.0, "out": 3.0},
            {"type": "image", "source": "footage/cta.png", "duration": 1.0},
        ],
        "overlays": [{"type": "sfx", "source": "beep", "at": 2.5, "volume": 0.8}],
        "audio": audio_spec,
        "output": {"quality": "preview"},
    }
    with_sound = compose(Storyboard.model_validate(raw), root=root)
    plain = json.loads(json.dumps(raw))
    plain["name"] += "_plain"
    plain["overlays"] = []
    without = compose(Storyboard.model_validate(plain), root=root)
    assert with_sound.sfx == [{"source": "sfx/beep.wav", "at": 2.5, "volume": 0.8, "auto": None}]
    assert probe(with_sound.path).audio_duration == pytest.approx(
        probe(without.path).audio_duration, abs=0.05
    )
    heard = _decode(with_sound.path)
    base = _decode(without.path)
    size = min(len(heard), len(base))
    difference = heard[:size] - base[:size]
    after = _energy(difference, 2.5, 2.9)
    before = _energy(difference, 0.0, 2.3)
    assert after > 1.0
    assert before < 0.01 * after  # nothing changes before the sound
    peak = audio.true_peak(with_sound.path)
    assert peak is not None and peak <= -1.0


# ----------------------------------------------------------------------------- the beat grid


@pytest.mark.parametrize("bpm", [90, 120, 128])
def test_beat_grid_finds_the_beats_of_a_click_track_within_a_frame(
    tmp_path: Path, bpm: int
) -> None:
    track = _click_track(tmp_path / "clicks.wav", bpm, seconds=20.0)
    grid = audio.beat_grid(track)
    assert grid.tempo == pytest.approx(bpm, abs=0.5)
    truth = _click_times(bpm, 0.0, 20.0)
    assert abs(len(grid.beats) - len(truth)) <= 1
    for moment in truth:
        assert min(abs(beat - moment) for beat in grid.beats) <= FRAME, moment
    assert all(0.0 <= beat <= grid.analysed for beat in grid.beats)


def test_beat_grid_with_a_first_beat_that_is_not_at_zero_and_with_start(tmp_path: Path) -> None:
    track = _click_track(tmp_path / "clicks.wav", 120, first=0.31, seconds=20.0)
    grid = audio.beat_grid(track)
    truth = _click_times(120, 0.31, 20.0)
    assert grid.beats[0] == pytest.approx(0.31, abs=FRAME)
    for moment in truth:
        assert min(abs(beat - moment) for beat in grid.beats) <= FRAME
    # As the music plays from start=1.2 s, the times are measured from there.
    shifted = audio.beat_grid(track, start=1.2)
    for moment in truth:
        if moment >= 1.2:
            assert min(abs(beat - (moment - 1.2)) for beat in shifted.beats) <= FRAME
    assert shifted.beats[0] < 0.5
    limited = audio.beat_grid(track, duration=5.0)
    assert limited.analysed == pytest.approx(5.0, abs=0.01) and limited.beats[-1] < 5.0


def test_beat_grid_prefers_the_kick_over_the_hats_and_has_no_beat_for_a_pad(
    tmp_path: Path,
) -> None:
    samples = np.zeros(20 * SR)
    rng = np.random.default_rng(5)
    for k in range(40):
        index = round(k * 0.5 * SR)
        t = np.arange(int(0.15 * SR)) / SR
        kick = 0.8 * np.sin(2 * np.pi * (50 + 90 * np.exp(-t / 0.03)) * t) * np.exp(-t / 0.07)
        samples[index : index + len(kick)] += kick
        hat_at = index + int(0.25 * SR)
        m = int(0.05 * SR)
        tm = np.arange(m) / SR
        hat = 0.3 * np.diff(rng.standard_normal(m + 1)) * np.exp(-tm / 0.012)
        samples[hat_at : hat_at + m] += hat
    grid = audio.beat_grid(_write(tmp_path / "kick_hats.wav", samples))
    assert grid.tempo == pytest.approx(120, abs=0.5)
    assert min(abs(grid.beats[3] - 0.5 * k) for k in range(40)) <= FRAME  # on a kick, not a hat
    t = np.arange(10 * SR) / SR
    pad = _write(tmp_path / "pad.wav", 0.3 * np.sin(2 * np.pi * 220 * t))
    silent = audio.beat_grid(pad)
    assert silent.beats == [] and silent.tempo is None
    with pytest.raises(ValueError, match="No audio remains"):
        audio.beat_grid(pad, start=60)


# ----------------------------------------------------------------------------- cuts on the beat


def _snap_project(tmp_path: Path, media: dict[str, dict[str, object]]) -> Path:
    root = _project(tmp_path, media)
    _click_track(root / "music" / "beat.wav", 120, seconds=30.0)  # a beat every 0.5 s
    return root


def _cut_times(raw: dict[str, Any]) -> list[float]:
    model = Storyboard.model_validate(raw)
    return [round(start, 4) for start, _ in model.timeline()][1:]


def test_snap_moves_cuts_to_the_nearest_beat_and_keeps_the_total(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _snap_project(tmp_path, media)
    sfx.fetch_pack(root, offline=True, names=["pop"])
    raw = _image_ad(
        [1.62, 1.45, 1.1, 1.83],
        overlays=[
            {"type": "caption", "style": "pill", "lines": ["A"], "start": 1.62, "end": 3.07},
            {"type": "caption", "style": "pill", "lines": ["B"], "start": 0.0, "end": 1.62},
            {"type": "sfx", "source": "pop", "at": 3.07},
            {"type": "caption", "style": "pill", "lines": ["C"], "start": 2.0, "end": 2.5},
        ],
    )
    before = validate_storyboard(raw, root=root)
    assert before.summary["beat_alignment"]["on_beat"] == 0
    assert "storyboard_snap" in before.info[0].message and before.info[0].code == "beat_alignment"
    result = snap_storyboard(root, raw)
    assert result["changed"] and result["on_beat_before"] == 0 and result["on_beat_after"] == 3
    assert [move["to"] for move in result["moves"]] == pytest.approx([1.5, 3.0, 4.0], abs=0.02)
    snapped = result["storyboard"]
    assert [round(s["duration"], 2) for s in snapped["segments"]] == pytest.approx(
        [1.5, 1.5, 1.0, 2.0], abs=0.02
    )
    assert result["total_duration"] == pytest.approx(6.0, abs=1e-6)
    assert Storyboard.model_validate(snapped).total_duration() == pytest.approx(6.0, abs=1e-3)
    assert raw["segments"][0]["duration"] == 1.62  # the input is not touched
    # The captions that started or ended on a moved cut, and the sound on it, move with it.
    first, second, sound, loose = snapped["overlays"]
    assert first["start"] == pytest.approx(1.5, abs=0.02)
    assert first["end"] == pytest.approx(3.0, abs=0.02)
    assert second["end"] == pytest.approx(1.5, abs=0.02) and second["start"] == 0.0
    assert sound["at"] == pytest.approx(3.0, abs=0.02)
    assert (loose["start"], loose["end"]) == (2.0, 2.5)
    after = validate_storyboard(snapped, root=root)
    assert after.summary["beat_alignment"] == {
        "cuts": 3, "on_beat": 3, "share": 1.0, "tempo_bpm": pytest.approx(120, abs=0.5),
    }  # fmt: skip
    assert "3 of 3 cuts" in after.info[0].message


def test_snap_keeps_in_points_and_order_of_clips(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _snap_project(tmp_path, media)
    raw = _image_ad([1.0])
    raw["segments"] = [
        {"type": "clip", "source": "footage/sdr_portrait.mp4", "in": 0.4, "out": 2.3, "speed": 1.0},
        {"type": "clip", "source": "footage/sdr_portrait.mp4", "in": 0.7, "out": 2.45},
        {"type": "image", "source": "footage/cta.png", "duration": 2.0},
    ]
    # cuts at 1.9 and 3.65: the beats are at 2.0 and 3.5
    result = snap_storyboard(root, raw)
    snapped = result["storyboard"]["segments"]
    assert [item["to"] for item in result["moves"]] == pytest.approx([2.0, 3.5], abs=0.02)
    assert [s["in"] for s in snapped[:2]] == [0.4, 0.7]
    assert snapped[0]["out"] == pytest.approx(2.4, abs=0.02)
    assert snapped[1]["out"] == pytest.approx(0.7 + 1.5, abs=0.03)  # 2.0 to 3.5
    assert snapped[2]["duration"] == pytest.approx(2.15, abs=0.03)
    assert result["total_duration"] == pytest.approx(5.65, abs=1e-6)
    assert [s["source"] for s in snapped] == [s["source"] for s in raw["segments"]]
    # A clip is never made to run past the end of its source (3.0 s long).
    tight = _image_ad([1.0])
    tight["segments"] = [
        {"type": "clip", "source": "footage/sdr_portrait.mp4", "in": 0.15, "out": 3.0},
        {"type": "image", "source": "footage/cta.png", "duration": 2.0},
    ]  # the cut at 2.85 would go to 3.0: the clip would need out 3.15
    refused = snap_storyboard(root, tight)
    assert refused["moves"] == [] and "past its source" in refused["kept"][0]["reason"]
    assert refused["storyboard"] == tight


def test_snap_respects_the_minimum_segment_length_and_the_distance(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _snap_project(tmp_path, media)
    raw = _image_ad([1.9, 0.55, 2.0, 1.0])
    result = snap_storyboard(root, raw)
    # Cut 1 (1.9) would go to 2.0, leaving 0.45 s: it stays. Cut 2 (2.45) moves to 2.5.
    kept = {item["cut"]: item["reason"] for item in result["kept"]}
    assert "too short" in kept[1]
    assert [move["cut"] for move in result["moves"]] == [2, 3]
    assert [move["to"] for move in result["moves"]] == pytest.approx([2.5, 4.5], abs=0.01)
    durations = [s["duration"] for s in result["storyboard"]["segments"]]
    assert min(durations) >= 0.5 and sum(durations) == pytest.approx(5.45, abs=1e-6)
    far = snap_storyboard(root, _image_ad([1.75, 2.0]), max_shift=0.2)
    assert far["moves"] == [] and "no beat within" in far["kept"][0]["reason"]
    near = snap_storyboard(root, _image_ad([1.72, 2.0]), max_shift=0.3)
    assert [move["to"] for move in near["moves"]] == pytest.approx([1.5], abs=0.02)
    with pytest.raises(ValueError, match="max_shift"):
        snap_storyboard(root, raw, max_shift=0)
    with pytest.raises(ValueError, match="to must be"):
        snap_storyboard(root, raw, to="bars")


def test_snap_never_moves_the_cuts_of_talking_segments(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _snap_project(tmp_path, media)
    raw = _image_ad([1.0])
    raw["segments"] = [
        {"type": "clip", "source": "footage/sdr_portrait.mp4", "in": 0.0, "out": 1.62},
        {"type": "clip", "source": "footage/sdr_portrait.mp4", "in": 0.0, "out": 1.45},
        {"type": "image", "source": "footage/cta.png", "duration": 1.1},
        {"type": "image", "source": "footage/cta.png", "duration": 1.5},
    ]
    raw["audio"] = {"clips": True, "music": "music/beat.wav"}
    result = snap_storyboard(root, raw)
    reasons = {item["cut"]: item["reason"] for item in result["kept"]}
    assert reasons[1].startswith("talking segment") and reasons[2].startswith("talking segment")
    assert [move["cut"] for move in result["moves"]] == [3]  # image to image: free to move
    assert result["storyboard"]["segments"][0] == raw["segments"][0]


def test_snap_needs_music_and_a_beat(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _snap_project(tmp_path, media)
    silent = _image_ad([1.6, 1.4])
    silent["audio"] = {}
    with pytest.raises(ValueError, match="no music"):
        snap_storyboard(root, silent)
    pad = np.sin(2 * np.pi * 220 * np.arange(10 * SR) / SR) * 0.3
    _write(root / "music" / "pad.wav", pad)
    calm = _image_ad([1.6, 1.4], audio={"music": "music/pad.wav"})
    result = snap_storyboard(root, calm)
    assert result["moves"] == [] and result["changed"] is False
    assert "no steady beat" in result["warnings"][0]
    report = validate_storyboard(calm, root=root)
    assert report.info[0].code == "beat_alignment" and "no steady beat" in report.info[0].message


def test_snap_start_offset_shifts_the_beats_of_the_music(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _snap_project(tmp_path, media)
    # The music plays from 0.25 s, so on the timeline its beats are at 0.25, 0.75, 1.25...
    raw = _image_ad([0.9, 1.0, 1.0], audio={"music": "music/beat.wav", "start": 0.25})
    result = snap_storyboard(root, raw)
    assert [move["to"] for move in result["moves"]] == pytest.approx([0.75, 1.75], abs=0.02)
    plain = snap_storyboard(root, _image_ad([0.9, 1.0, 1.0]))
    assert [move["to"] for move in plain["moves"]] == pytest.approx([1.0, 2.0], abs=0.02)


# ----------------------------------------------------------------------------- the tools


def test_mcp_tools_for_sound_design(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _snap_project(tmp_path, media)

    def payload(result: Any) -> dict[str, Any]:
        assert not result.is_error, [c.text for c in result.content if c.type == "text"]
        return json.loads([c.text for c in result.content if c.type == "text"][0])

    async def run_tools() -> None:
        async with Client(build_server()) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            assert {"sfx_list", "music_beats", "storyboard_snap"} <= set(tools)
            assert {"fetch", "offline"} <= set(tools["sfx_list"].input_schema["properties"])
            assert {"start", "duration"} <= set(tools["music_beats"].input_schema["properties"])
            assert {"to", "max_shift", "save_as"} <= set(
                tools["storyboard_snap"].input_schema["properties"]
            )
            empty = payload(await client.call_tool("sfx_list", {"project_dir": str(root)}))
            assert empty["count"] == 0 and "hint" in empty
            built = payload(
                await client.call_tool(
                    "sfx_list", {"project_dir": str(root), "fetch": True, "offline": True}
                )
            )
            assert built["count"] == len(sfx.load_manifest()) and built["fetch"]["warnings"] == []
            beats = payload(
                await client.call_tool(
                    "music_beats",
                    {"project_dir": str(root), "path": "music/beat.wav", "start": 0.0},
                )
            )
            assert beats["tempo_bpm"] == pytest.approx(120, abs=0.5)
            assert beats["beats"][:3] == pytest.approx([0.0, 0.5, 1.0], abs=0.02)
            missing = await client.call_tool(
                "music_beats", {"project_dir": str(root), "path": "music/none.wav"}
            )
            assert missing.is_error
            escape = await client.call_tool(
                "music_beats", {"project_dir": str(root), "path": "../x.wav"}
            )
            assert escape.is_error

            raw = _image_ad(
                [1.62, 1.45, 1.1, 1.83],
                audio={
                    "music": "music/beat.wav",
                    "sfx_auto": {"cuts": "whoosh_short", "captions": "pop"},
                },
                overlays=[
                    {"type": "caption", "style": "pill", "lines": ["Hi"], "start": 0.0, "end": 1.0},
                    {"type": "sfx", "source": "ding", "at": 4.0, "volume": 0.7},
                ],
            )
            saved = payload(
                await client.call_tool(
                    "storyboard_save", {"project_dir": str(root), "storyboard": raw, "name": "ad"}
                )
            )
            assert [(i["at"], i["auto"]) for i in saved["sfx"]] == [
                (0.0, "captions"), (1.62, "cuts"), (3.07, "cuts"), (4.0, None), (4.17, "cuts"),
            ]  # fmt: skip
            plain = payload(
                await client.call_tool(
                    "storyboard_save",
                    {"project_dir": str(root), "storyboard": _image_ad([1.0]), "name": "plain"},
                )
            )
            assert "sfx" not in plain
            validated = payload(
                await client.call_tool(
                    "storyboard_validate",
                    {"project_dir": str(root), "storyboard": "storyboards/ad.json"},
                )
            )
            assert validated["ok"] and len(validated["summary"]["sfx"]) == 5
            assert validated["info"][0]["code"] == "beat_alignment"
            snapped = payload(
                await client.call_tool(
                    "storyboard_snap",
                    {"project_dir": str(root), "storyboard": "storyboards/ad.json"},
                )
            )
            assert snapped["on_beat_after"] == 3 and "storyboard" in snapped
            kept = payload(
                await client.call_tool(
                    "storyboard_snap",
                    {
                        "project_dir": str(root),
                        "storyboard": "storyboards/ad.json",
                        "save_as": "ad_beats",
                    },
                )
            )
            assert kept["saved"] == "storyboards/ad_beats.json" and "storyboard" not in kept
            again = payload(
                await client.call_tool(
                    "storyboard_validate",
                    {"project_dir": str(root), "storyboard": "storyboards/ad_beats.json"},
                )
            )
            assert again["summary"]["beat_alignment"]["on_beat"] == 3
            invalid = await client.call_tool(
                "storyboard_snap",
                {"project_dir": str(root), "storyboard": {"name": "x"}},
            )
            assert invalid.is_error

    anyio.run(run_tools)


# ----------------------------------------------------------------------------- the guides


def test_the_guides_describe_sound_design() -> None:
    repo = Path(__file__).resolve().parents[1]
    texts = {
        name: (repo / name).read_text(encoding="utf-8")
        for name in ("AGENTS.md", "skill/SKILL.md", "docs/EDITING_GUIDE.md", "README.md")
    }
    for name in ("AGENTS.md", "skill/SKILL.md"):
        for word in ("Tools (43)", "sfx_list", "music_beats", "storyboard_snap"):
            assert word in texts[name], (name, word)
    for name in ("AGENTS.md", "skill/SKILL.md", "docs/EDITING_GUIDE.md"):
        for word in (
            "Sound design", "sfx_auto", "min_gap", "beat_alignment", "sfx-fetch", "CC0",
            "every second cut", "-16 LUFS",
        ):  # fmt: skip
            assert word in texts[name], (name, word)
    assert "Tools (43)" in texts["README.md"] and "sfx_list" in texts["README.md"]
    assert (repo / "src" / "adanvil" / "assets" / "EDITING_GUIDE.md").read_bytes() == (
        repo / "docs" / "EDITING_GUIDE.md"
    ).read_bytes()
    agents = texts["AGENTS.md"].split("### Sound design", 1)[1].split("###", 1)[0]
    skill = texts["skill/SKILL.md"].split("### Sound design", 1)[1].split("###", 1)[0]
    assert agents == skill  # one text, two entry points
