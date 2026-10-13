"""Speech in the library: indexing, phrases, fillers, restarts, clean segments, takes."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest
from mcp import Client

from adanvil import jobs
from adanvil.brand import Brand, Speech, load_brand, save_brand
from adanvil.library import entry_line, index_library, index_summary, load_catalog, search
from adanvil.library.catalog import EntrySpeech
from adanvil.project import init_project
from adanvil.server import build_server
from adanvil.speech import store
from adanvil.speech import transcribe as speech_transcribe
from adanvil.speech.analyze import (
    FillerLists,
    brand_filler_lists,
    clean_segments,
    find_fillers,
    find_restarts,
    read_speech,
    split_phrases,
)
from adanvil.speech.library import SpeechLibrary
from adanvil.speech.model import SpeechData, SpeechWord
from adanvil.speech.takes import ClipSpeech, rank_takes
from adanvil.voice.engines import Alignment
from tests.speech_helpers import ScriptedEngine, make_clip, spoken, words


@pytest.fixture(autouse=True)
def _speech_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """The suite turns ``transcribe="auto"`` off; these tests inject their own engine."""

    monkeypatch.setenv("ADANVIL_SPEECH", "auto")


# ---------------------------------------------------------------- phrases, fillers, restarts


def test_phrases_split_on_long_pauses_and_sentence_punctuation() -> None:
    items = words(
        (0.0, 0.3, "Bonjour"),
        (0.35, 0.6, "à"),
        (0.65, 0.9, "tous."),
        (0.95, 1.3, "Aujourd'hui"),
        (1.35, 1.6, "je"),
        (2.4, 2.8, "teste"),  # 0.8 s of silence before it
        (2.85, 3.1, "ceci"),
    )
    phrases = split_phrases(items, max_pause=0.35)
    assert [phrase.text for phrase in phrases] == [
        "Bonjour à tous.",
        "Aujourd'hui je",
        "teste ceci",
    ]
    spans = [(phrase.start, phrase.end) for phrase in phrases]
    assert spans == [(0.0, 0.9), (0.95, 1.6), (2.4, 3.1)]
    # A larger tolerance keeps the two phrases without punctuation apart only by the pause.
    assert len(split_phrases(items, max_pause=1.0)) == 2


def test_fillers_flag_hesitations_and_isolated_words_only() -> None:
    items = words(
        (0.0, 0.2, "Je"),
        (0.25, 0.5, "like"),  # "I like it": a real verb, no pause around it
        (0.55, 0.7, "it"),
        (1.5, 1.7, "Euh,"),
        (1.75, 1.95, "ce"),
        (2.0, 2.3, "genre"),  # "ce genre de crème": kept
        (2.35, 2.45, "de"),
        (2.5, 2.9, "crème"),
        (3.8, 3.95, "Ben"),
        (4.0, 4.5, "euhhhh"),
        (5.4, 5.6, "du"),
        (5.65, 5.9, "coup,"),
        (5.95, 6.2, "voilà"),
        (6.25, 6.5, "I,"),
        (6.55, 6.85, "like,"),  # commas around it: a filler
        (6.9, 7.2, "love"),
    )
    flagged = {items[index].text for index in find_fillers(items)}
    assert flagged == {"Euh,", "Ben", "euhhhh", "du", "coup,", "like,"}
    assert "like" not in flagged and "genre" not in flagged


def test_fillers_use_the_lists_of_the_brand(tmp_path: Path) -> None:
    items = words(
        (0.0, 0.3, "alors"), (0.35, 0.6, "voilà"), (0.65, 0.9, "euh"), (0.95, 1.2, "bref")
    )
    assert {items[i].text for i in find_fillers(items)} == {"euh"}
    custom = FillerLists(always=("voilà", "bref"), isolated=())
    assert {items[i].text for i in find_fillers(items, custom)} == {"voilà", "bref"}

    brand = Brand(name="Ember & Oak", speech=Speech(fillers=["voilà"], isolated_fillers=[]))
    save_brand(brand, tmp_path / "brand.yaml")
    assert load_brand(tmp_path / "brand.yaml").speech == brand.speech
    lists = brand_filler_lists(tmp_path)
    assert lists == FillerLists(always=("voilà",), isolated=())
    # Without a brand file or a speech section the defaults apply.
    assert "euh" in brand_filler_lists(tmp_path / "missing").always
    save_brand(Brand(name="Plain"), tmp_path / "brand.yaml")
    assert "genre" in brand_filler_lists(tmp_path).isolated


def test_restart_is_a_phrase_repeated_by_the_next_one() -> None:
    items = words(
        *spoken("Bonjour je m'appelle Léa et", 0.0),
        *spoken("Bonjour je m'appelle Léa et je teste", 3.0),
        *spoken("Un autre sujet complètement.", 6.0),
        *spoken("Un autre", 9.0),
        *spoken("truc différent maintenant.", 11.0),
    )
    phrases = split_phrases(items, max_pause=0.35)
    assert len(phrases) == 5
    # Only the first phrase restarts (in the second). "Un autre sujet complètement." is
    # followed by its own first two words, which is not a retake of it.
    assert find_restarts(items, phrases, set()) == [0]


def test_a_filler_does_not_hide_a_restart() -> None:
    items = words(
        *spoken("Voici ma routine du matin", 0.0),
        *spoken("euh voici ma routine du matin complète", 3.0),
    )
    phrases = split_phrases(items, max_pause=0.35)
    fillers = find_fillers(items)
    assert find_restarts(items, phrases, fillers) == [0]


# --------------------------------------------------------------------------- clean segments


def _inside_a_word(items, instant: float, tolerance: float = 0.0011) -> bool:
    return any(word.start + tolerance < instant < word.end - tolerance for word in items)


def test_clean_segments_drop_pauses_and_fillers_and_keep_margins() -> None:
    items = words(
        (1.00, 1.30, "Voici"),
        (1.35, 1.60, "mon"),
        (1.65, 2.00, "produit."),
        (3.00, 3.30, "euh"),  # a filler between two pauses
        (4.20, 4.50, "Il"),
        (4.55, 4.90, "change"),
        (4.95, 5.20, "tout"),
    )
    reading = read_speech(items, "footage/a.mp4", duration=6.0)
    assert [item["source"] for item in reading.segments] == ["footage/a.mp4"] * 2
    first, second = reading.segments
    assert (first["in"], first["out"]) == (0.94, 2.1)  # 0.06 s before, 0.10 s after
    assert (second["in"], second["out"]) == (4.14, 5.3)
    assert reading.fillers == {3}


def test_a_filler_inside_a_phrase_is_cut_even_when_its_gaps_are_tiny() -> None:
    items = words(
        (0.0, 0.3, "Je"),
        (0.32, 0.6, "suis"),
        (0.62, 0.72, "euh"),  # short, glued to its neighbours
        (0.74, 1.1, "contente"),
    )
    reading = read_speech(items, "footage/a.mp4")
    assert len(reading.segments) == 2
    first, second = reading.segments
    # No cut falls inside the filler or the words: both edges sit on the filler's borders.
    assert first["out"] <= 0.62 + 1e-9 and second["in"] >= 0.72 - 1e-9


def test_short_pauses_between_segments_are_merged() -> None:
    items = words((0.0, 0.3, "Un"), (0.55, 0.9, "deux"))  # 0.25 s pause
    # With max_pause 0.2 the pause is cut, but the margins (0.16 s) leave 0.09 s: merged.
    segments = clean_segments(items, "a.mp4", max_pause=0.2)
    assert len(segments) == 1
    assert segments[0]["in"] == 0.0 and segments[0]["out"] == 1.0
    # A real pause stays a cut.
    items = words((0.0, 0.3, "Un"), (1.5, 1.9, "deux"))
    assert len(clean_segments(items, "a.mp4")) == 2


def test_clean_segments_never_cut_inside_a_word() -> None:
    randomiser = random.Random(7)
    for _ in range(40):
        cursor = round(randomiser.uniform(0.0, 1.0), 2)
        items = []
        for number in range(45):
            length = round(randomiser.uniform(0.1, 0.6), 2)
            text = "euh" if number % 7 == 3 else f"w{number}"
            items.append((cursor, round(cursor + length, 2), text))
            gap = randomiser.choice([0.0, 0.02, 0.05, 0.1, 0.3, 0.5, 1.2])
            cursor = round(cursor + length + gap, 2)
        timed = words(*items)
        reading = read_speech(timed, "a.mp4", max_pause=randomiser.choice([0.2, 0.35, 0.6]))
        edges = [(seg["in"], seg["out"]) for seg in reading.segments]
        for start, end in edges:
            assert start < end
            assert not _inside_a_word(timed, start), (start, end)
            assert not _inside_a_word(timed, end), (start, end)
        # Segments are ordered and never overlap.
        assert all(a[1] <= b[0] for a, b in zip(edges, edges[1:], strict=False))
        # Every non-filler word is inside a segment.
        for index, word in enumerate(timed):
            if index in reading.fillers:
                continue
            assert any(a <= word.start and word.end <= b for a, b in edges), word


def test_drop_restarts_removes_the_aborted_phrase_from_clean_segments() -> None:
    items = words(
        *spoken("Bonjour je m'appelle Léa et", 0.0),
        *spoken("Bonjour je m'appelle Léa et je teste", 3.0),
    )
    kept = read_speech(items, "a.mp4")
    dropped = read_speech(items, "a.mp4", drop_restarts=True)
    assert kept.restarts == [0] and len(kept.segments) == 2
    assert len(dropped.segments) == 1 and dropped.segments[0]["in"] >= 2.9


# -------------------------------------------------------------------------------------- takes


def _clip(name: str, script, duration: float, *, probability: float = 0.95) -> ClipSpeech:
    data = SpeechData(
        words=[SpeechWord(start, end, text, probability) for start, end, text in script],
        duration=duration,
        levels_db=[-22.0] * int(duration / 0.1 + 1),
    )
    return ClipSpeech(name, f"footage/{name}.mov", duration, data)


def test_takes_rank_a_clean_complete_take_first() -> None:
    line = "Mes matinées sont enfin calmes grâce à Brio"
    clips = [
        _clip("HOOK1", spoken("Mes matinées sont euh enfin calmes grâce à Brio", 0.5), 4.5),
        _clip("HOOK1.1", spoken("Mes matinées sont enfin calmes grâce à", 0.5), 3.1),  # cut
        _clip("HOOK1.2", spoken(line, 0.5), 4.2),
    ]
    lines = rank_takes(clips)
    assert len(lines) == 1
    best = lines[0]["best"]
    assert best["entry_id"] == "HOOK1.2"
    assert best["source"] == "footage/HOOK1.2.mov"
    assert "no filler" in best["why"] and "complete ending" in best["why"]
    assert best["clean_segments"] and all(
        seg["source"] == "footage/HOOK1.2.mov" for seg in best["clean_segments"]
    )
    others = {item["entry_id"]: item for item in lines[0]["others"]}
    assert "1 filler" in others["HOOK1"]["issues"]
    assert any("ending cut" in issue for issue in others["HOOK1.1"]["issues"])
    assert best["score"] > others["HOOK1"]["score"] > 0
    assert best["score"] > others["HOOK1.1"]["score"]


def test_takes_penalise_restarts_and_group_different_lines() -> None:
    clips = [
        _clip(
            "A",
            [
                *spoken("Je teste Brio depuis un mois", 0.5),
                *spoken("Je teste Brio depuis un mois et ça change tout.", 4.0),
                *spoken("Commandez maintenant sur le site", 8.0),
            ],
            11.0,
        ),
        _clip("B", spoken("Commandez maintenant sur le site", 0.5), 3.5),
    ]
    lines = rank_takes(clips)
    texts = [line["best"]["text"] for line in lines]
    assert len(lines) == 2
    assert "ça change tout" in texts[0]  # the complete retake wins over the aborted start
    assert lines[0]["takes"] == 2
    assert lines[1]["takes"] == 2 and {lines[1]["best"]["entry_id"]} <= {"A", "B"}
    aborted = next(item for item in lines[0]["others"])
    assert any("restart" in issue for issue in aborted["issues"])


def test_takes_prefer_a_natural_speech_rate_and_a_good_level() -> None:
    sentence = "Une phrase assez longue pour mesurer le débit"
    fast = _clip("FAST", spoken(sentence, 0.5, rate=0.1, gap=0.0), 3.0)
    steady = _clip("OK", spoken(sentence, 0.5), 5.0)
    quiet = _clip("QUIET", spoken(sentence, 0.5), 5.0)
    quiet.data.levels_db = [-48.0] * len(quiet.data.levels_db)
    ranked = rank_takes([fast, quiet, steady])[0]
    assert ranked["best"]["entry_id"] == "OK"
    assert [item["entry_id"] for item in ranked["others"]] == ["QUIET", "FAST"]


# --------------------------------------------------------------------- indexing with speech


def _project_with_clips(tmp_path: Path) -> Path:
    init_project(tmp_path, "Ember & Oak")
    footage = tmp_path / "footage"
    make_clip(footage / "HOOK1.mov", 3.0, [(0.3, 1.2, 300), (1.8, 2.7, 320)])
    make_clip(footage / "HOOK1.1.mov", 3.0, [(0.3, 2.5, 300)])
    make_clip(footage / "broll.mov", 2.0, audio=False)
    make_clip(footage / "quiet.mov", 2.0, [])  # an audio stream that is silent
    return tmp_path


SCRIPTS = {
    "HOOK1": [
        (0.3, 0.6, "Bonjour"),
        (0.65, 0.9, "je"),
        (0.95, 1.2, "teste."),
        (1.8, 2.1, "Brio"),
        (2.15, 2.7, "enfin."),
    ],
    "HOOK1.1": [
        (0.3, 0.7, "Bonjour"),
        (0.75, 0.95, "je"),
        (1.0, 1.4, "teste."),
        (1.9, 2.2, "Brio"),
        (2.25, 2.6, "enfin."),
    ],
    "quiet": [(0.2, 0.5, "rien")],
}


def test_index_transcribes_in_one_batch_and_keeps_a_sidecar(tmp_path: Path) -> None:
    root = _project_with_clips(tmp_path)
    engine = ScriptedEngine(SCRIPTS)

    result = index_library(root, "footage", transcribe="auto", engine=engine)
    assert not result.errors
    # One call for the whole job: both clips with sound, the silent one skipped, B-roll untouched.
    assert len(engine.calls) == 1
    assert sorted(engine.calls[0][0]) == ["HOOK1", "HOOK1.1"]
    assert engine.calls[0][2] == "base"  # the default model: fast, good enough to search

    catalog = load_catalog(root)
    by_name = {Path(entry.path).name: entry for entry in catalog.entries.values()}
    hook = by_name["HOOK1.mov"]
    assert hook.speech is not None and hook.speech.status == "done"
    assert hook.speech.has_speech and hook.speech.language == "fr"
    assert hook.speech.text == "Bonjour je teste. Brio enfin."
    assert hook.speech.words_file is not None
    assert hook.speech.words_file.startswith("cache/speech/") and hook.speech.words_file.endswith(
        ".json"
    )
    assert by_name["broll.mov"].speech is not None
    assert by_name["broll.mov"].speech.status == "not attempted: no audio"
    assert by_name["quiet.mov"].speech is not None
    assert by_name["quiet.mov"].speech.status == "silent_audio"
    assert not by_name["quiet.mov"].speech.has_speech

    # The catalogue keeps only the summary: the timed words are in the sidecar.
    assert set(json.loads((root / "catalog.json").read_text())["entries"][hook.id]["speech"]) == {
        "status",
        "language",
        "text",
        "words_file",
        "has_speech",
    }
    side = store.read_sidecar(root, hook.speech.words_file)
    assert side is not None
    assert [(w.start, w.end, w.text) for w in side.words] == [
        (item[0], item[1], item[2]) for item in SCRIPTS["HOOK1"]
    ]
    assert side.language == "fr" and side.levels_db and side.duration == pytest.approx(3.0, abs=0.1)
    # The levels follow the sound: the tone bursts are far above the silence between them.
    assert side.level(0.3, 1.2) > side.level(1.3, 1.7) + 30

    # Reused: nothing is transcribed again, and no scratch audio is left behind.
    again = index_library(root, "footage", engine=engine)
    assert len(engine.calls) == 1 and not again.added and not again.updated
    assert not list((root / "cache" / "speech").glob(".tmp-*"))

    # Search reaches what the clips say, and the line tells which clips speak.
    found = search(root, text="ENFIN")
    assert sorted(Path(entry.path).name for entry in found) == ["HOOK1.1.mov", "HOOK1.mov"]
    assert entry_line(found[0]).endswith("| speech")
    assert not search(root, text="introuvable")


def test_transcribe_true_forces_silent_clips_and_false_does_no_speech_work(
    tmp_path: Path,
) -> None:
    root = _project_with_clips(tmp_path)
    engine = ScriptedEngine(SCRIPTS)

    off = index_library(root, "footage", transcribe=False, engine=engine)
    assert not off.speech and not engine.calls
    assert all(entry.speech is None for entry in load_catalog(root).entries.values())

    forced = index_library(root, "footage", transcribe=True, engine=engine)
    assert len(engine.calls) == 1
    assert sorted(engine.calls[0][0]) == ["HOOK1", "HOOK1.1", "quiet"]
    quiet = next(e for e in load_catalog(root).entries.values() if e.path.endswith("quiet.mov"))
    assert quiet.speech is not None and quiet.speech.status == "done"
    summary = index_summary(root, forced)
    assert summary["speech"]["done"] == 3

    with pytest.raises(ValueError, match="transcribe must be"):
        index_library(root, "footage", transcribe="sometimes")  # type: ignore[arg-type]


def test_auto_follows_the_environment_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project_with_clips(tmp_path)
    engine = ScriptedEngine(SCRIPTS)
    monkeypatch.setenv("ADANVIL_SPEECH", "off")
    index_library(root, "footage", engine=engine)
    assert not engine.calls
    monkeypatch.delenv("ADANVIL_SPEECH")
    index_library(root, "footage", engine=engine)
    assert len(engine.calls) == 1


def test_no_transcriber_is_a_status_never_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project_with_clips(tmp_path)
    monkeypatch.setenv("ADANVIL_SPEECH", "auto")
    monkeypatch.setenv("ADANVIL_CACHE", str(tmp_path / "no-voice-environment"))
    monkeypatch.delenv("ADANVIL_VOICE_PYTHON", raising=False)
    monkeypatch.setitem(sys.modules, "faster_whisper", None)

    result = index_library(root, "footage")
    assert not result.errors
    entries = {Path(e.path).name: e for e in load_catalog(root).entries.values()}
    assert entries["HOOK1.mov"].speech is not None
    assert entries["HOOK1.mov"].speech.status.startswith("unavailable")
    assert "adanvil[transcribe]" in entries["HOOK1.mov"].speech.status
    assert "voice-setup" in entries["HOOK1.mov"].speech.status
    assert entries["broll.mov"].speech.status == "not attempted: no audio"
    assert not list((root / "cache" / "speech").glob("*.json"))

    # A transcriber that appears later fills the same entries, without re-measuring them.
    engine = ScriptedEngine(SCRIPTS)
    later = index_library(root, "footage", engine=engine)
    assert not later.added and not later.updated
    assert sorted(engine.calls[0][0]) == ["HOOK1", "HOOK1.1"]
    assert load_catalog(root).entries[entries["HOOK1.mov"].id].speech.status == "done"


def test_a_changed_file_is_transcribed_again(tmp_path: Path) -> None:
    root = _project_with_clips(tmp_path)
    engine = ScriptedEngine(SCRIPTS)
    index_library(root, "footage", engine=engine)
    first = next(e for e in load_catalog(root).entries.values() if e.path.endswith("HOOK1.mov"))
    old_sidecar = first.speech.words_file
    make_clip(root / "footage" / "HOOK1.mov", 3.5, [(0.3, 1.0, 300)])

    result = index_library(root, "footage", engine=engine)
    assert len(result.updated) == 1
    assert [call[0] for call in engine.calls[1:]] == [["HOOK1"]]
    second = load_catalog(root).entries[first.id]
    assert second.speech.words_file != old_sidecar
    assert not (root / old_sidecar).exists()


def test_a_failing_clip_does_not_lose_the_others(tmp_path: Path) -> None:
    root = _project_with_clips(tmp_path)

    class Flaky(ScriptedEngine):
        def align(self, paths, language, model, **options):
            results = super().align(paths, language, model, **options)
            results[0].error = "model crashed on this file"
            return results

    result = index_library(root, "footage", engine=Flaky(SCRIPTS))
    statuses = {Path(e.path).name: e.speech.status for e in load_catalog(root).entries.values()}
    assert statuses["HOOK1.1.mov"].startswith("failed: model crashed")
    assert statuses["HOOK1.mov"] == "done"
    assert any("speech failed" in error["message"] for error in result.errors)


def test_language_comes_from_the_brand_profile(tmp_path: Path) -> None:
    root = _project_with_clips(tmp_path)
    save_brand(Brand(name="Ember & Oak", language="French"), root / "brand.yaml")
    engine = ScriptedEngine(SCRIPTS)
    index_library(root, "footage", engine=engine)
    assert engine.calls[0][1] == "fr"

    other = ScriptedEngine(SCRIPTS)
    index_library(root, "footage", engine=other, transcribe=True, language="en")
    assert other.calls[0][1] == "en"  # an explicit language wins


def test_model_comes_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project_with_clips(tmp_path)
    monkeypatch.setenv("ADANVIL_SPEECH_MODEL", "tiny")
    engine = ScriptedEngine(SCRIPTS)
    index_library(root, "footage", engine=engine)
    assert engine.calls[0][2] == "tiny"


def test_in_process_whisper_loads_its_model_once_for_the_whole_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project_with_clips(tmp_path)
    loads: list[tuple] = []
    heard: list[dict] = []

    class FakeModel:
        def __init__(self, model: str, *, device: str, compute_type: str) -> None:
            loads.append((model, device, compute_type))

        def transcribe(self, audio: str, **options):
            heard.append({"audio": Path(audio).stem, **options})
            word = SimpleNamespace(start=0.3, end=0.6, word=" Salut", probability=0.9)
            return [SimpleNamespace(words=[word])], SimpleNamespace(language="fr")

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeModel))
    monkeypatch.setenv("ADANVIL_SPEECH", "auto")
    progress: list[str] = []
    index_library(root, "footage", progress=progress.append)

    assert loads == [("base", "cpu", "int8")]  # one model load, two clips
    assert sorted(item["audio"] for item in heard) == ["HOOK1", "HOOK1.1"]
    assert all(item["word_timestamps"] and item["vad_filter"] for item in heard)
    assert progress.count("speech 2/2 transcribing") == 1
    entry = next(e for e in load_catalog(root).entries.values() if e.path.endswith("HOOK1.mov"))
    assert entry.speech.text == "Salut" and entry.speech.language == "fr"


def test_the_voice_worker_receives_every_file_in_a_single_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project_with_clips(tmp_path)
    runs: list[dict] = []

    class FakeWorker:
        def __init__(self, *, project_dir: Path) -> None:
            assert project_dir == root.resolve()

        def _python(self, _paths) -> str:
            return sys.executable

        def align(self, paths, language, model, *, vad, prompt, strict, progress, on_result):
            runs.append({"paths": [Path(p).stem for p in paths], "vad": vad, "strict": strict})
            if progress:
                progress(len(paths))
            return [
                Alignment(path=p, words=[(0.3, 0.6, "Salut", 0.9)], language="fr") for p in paths
            ]

    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setattr(speech_transcribe, "WorkerEngine", FakeWorker)
    monkeypatch.setenv("ADANVIL_SPEECH", "auto")
    progress: list[str] = []
    index_library(root, "footage", progress=progress.append)

    assert len(runs) == 1 and sorted(runs[0]["paths"]) == ["HOOK1", "HOOK1.1"]
    assert runs[0]["vad"] is True and runs[0]["strict"] is False
    assert "speech 2/2 transcribing" in progress


def test_the_index_job_reports_transcription_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project_with_clips(tmp_path)
    engine = ScriptedEngine(SCRIPTS)
    real_subprocess = jobs.subprocess
    no_runner = SimpleNamespace(
        Popen=lambda *args, **kwargs: SimpleNamespace(pid=0), DEVNULL=real_subprocess.DEVNULL
    )
    monkeypatch.setattr(jobs, "subprocess", no_runner)  # run the job here, not in a child
    record = jobs.start_job(
        root,
        "index",
        {"source": "footage", "transcribe": "auto"},
        progress={"total": 4, "unit": "files"},
    )
    monkeypatch.setattr(jobs, "subprocess", real_subprocess)
    monkeypatch.setattr(
        speech_transcribe,
        "_resolve_backend",
        lambda _root, _engine: ("engine", engine),
    )
    stages: list[tuple[str, dict]] = []
    original = jobs.update_job

    def spy(job_file, **updates):
        if "stage" in updates:
            stages.append((updates["stage"], updates.get("progress", {})))
        return original(job_file, **updates)

    monkeypatch.setattr(jobs, "update_job", spy)
    jobs.run_job(record["job_file"])

    final = jobs.read_job(record["job_file"])
    assert final["status"] == "done", final
    assert ("transcribing", {"done": 2, "total": 2, "unit": "clips"}) in stages
    assert final["index"]["speech"]["done"] == 2
    assert any(stage == "indexing" for stage, _ in stages)


# -------------------------------------------------------------------------------- tools


def _texts(result):
    return [item.text for item in result.content if item.type == "text"]


def _json(result):
    return json.loads(_texts(result)[0])


async def _speech_tools(root: Path) -> None:
    server = build_server()
    async with Client(server) as client:
        listed = {tool.name for tool in (await client.list_tools()).tools}
        assert {"speech_view", "speech_takes"} <= listed

        view = await client.call_tool(
            "speech_view", {"project_dir": str(root), "source": "footage/HOOK1.mov"}
        )
        assert not view.is_error, _texts(view)
        payload = _json(view)
        assert payload["speech_status"] == "done" and payload["has_speech"]
        assert [phrase["text"] for phrase in payload["phrases"]] == [
            "Bonjour je teste.",
            "Brio enfin.",
        ]
        # Compact by default: start, end, text and flags, no per-word data.
        assert payload["phrases"][0] == {
            "index": 0,
            "start": 0.3,
            "end": 1.2,
            "text": "Bonjour je teste.",
            "flags": [],
        }
        assert payload["phrase_count"] == 2 and "note" not in payload
        assert payload["model"] == "base"
        detailed = await client.call_tool(
            "speech_view",
            {"project_dir": str(root), "source": "footage/HOOK1.mov", "words": True},
        )
        words_payload = _json(detailed)
        assert words_payload["phrases"][0]["words"][0] == {
            "start": 0.3,
            "end": 0.6,
            "text": "Bonjour",
        }
        assert payload["fillers"] == [] and payload["restarts"] == []
        assert payload["clean_segments"] == [
            {"source": "footage/HOOK1.mov", "in": 0.24, "out": 1.3},
            {"source": "footage/HOOK1.mov", "in": 1.74, "out": 2.8},
        ]
        assert payload["clean_duration"] == pytest.approx(2.12, abs=0.01)

        by_id = await client.call_tool(
            "speech_view",
            {"project_dir": str(root), "entry_id": payload["entry_id"], "max_pause": 5.0},
        )
        assert len(_json(by_id)["clean_segments"]) == 1

        missing = await client.call_tool(
            "speech_view", {"project_dir": str(root), "source": "footage/nothing.mov"}
        )
        assert missing.is_error and "not indexed" in _texts(missing)[0]
        neither = await client.call_tool("speech_view", {"project_dir": str(root)})
        assert neither.is_error

        broll = await client.call_tool(
            "speech_view", {"project_dir": str(root), "source": "footage/broll.mov"}
        )
        broll_payload = _json(broll)
        assert broll_payload["speech_status"] == "not attempted: no audio"
        assert broll_payload["phrases"] == [] and "hint" in broll_payload

        takes = await client.call_tool(
            "speech_takes", {"project_dir": str(root), "source": "footage"}
        )
        assert not takes.is_error, _texts(takes)
        result = _json(takes)
        assert result["files"] == 2
        assert {item["source"] for item in result["skipped"]} == {
            "footage/broll.mov",
            "footage/quiet.mov",
        }
        first = result["lines"][0]
        assert first["takes"] == 2 and first["best"]["clean_segments"]
        nothing = await client.call_tool("speech_takes", {"project_dir": str(root)})
        assert nothing.is_error


def test_speech_tools_through_mcp(tmp_path: Path) -> None:
    root = _project_with_clips(tmp_path)
    index_library(root, "footage", engine=ScriptedEngine(SCRIPTS))
    anyio.run(_speech_tools, root)


def test_library_index_tool_accepts_transcribe(tmp_path: Path) -> None:
    root = _project_with_clips(tmp_path)

    async def run() -> None:
        async with Client(build_server()) as client:
            indexed = await client.call_tool(
                "library_index",
                {"project_dir": str(root), "source": "footage", "transcribe": False},
            )
            assert not indexed.is_error, _texts(indexed)
            summary = _json(indexed)
            assert summary["counts"]["added"] == 4 and summary["speech"] == {}
            bad = await client.call_tool(
                "library_index",
                {"project_dir": str(root), "source": "footage", "transcribe": "maybe"},
            )
            assert bad.is_error

    anyio.run(run)


def test_entry_speech_model_is_strict() -> None:
    with pytest.raises(ValueError):
        EntrySpeech(status="done", unexpected=True)  # type: ignore[call-arg]


def test_speech_library_reads_only_done_transcripts(tmp_path: Path) -> None:
    root = _project_with_clips(tmp_path)
    index_library(root, "footage", engine=ScriptedEngine(SCRIPTS))
    library = SpeechLibrary(root)
    assert library.data("footage/HOOK1.mov") is not None
    assert library.data("./footage\\HOOK1.mov") is not None  # same path, other spelling
    assert library.data("footage/broll.mov") is None
    assert library.data("footage/unknown.mov") is None
    assert library.fillers("footage/unknown.mov") == set()


def test_alignment_defaults_keep_the_voice_contract() -> None:
    alignment = Alignment(path=Path("x.wav"))
    assert alignment.language is None and alignment.error is None and alignment.words == []


def test_worker_align_passes_vad_and_prompt_and_returns_the_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adanvil.voice import worker

    seen: list[dict] = []

    class FakeModel:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def transcribe(self, _audio: object, **options: object):
            seen.append(options)
            word = SimpleNamespace(start=0.1, end=0.4, word=" Salut", probability=0.9)
            return [SimpleNamespace(text="Salut", words=[word])], SimpleNamespace(language="fr")

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeModel))
    monkeypatch.setattr(worker, "_load_16k", lambda path: path)

    speech = worker._align(
        {"audio": "a.wav", "language": None, "model": "small", "vad": True, "prompt": "Euh, bon"},
        {},
    )
    assert speech["ok"] and speech["language"] == "fr"
    assert speech["words"] == [[0.1, 0.4, "Salut", 0.9]]
    assert seen[-1] == {
        "language": None,
        "word_timestamps": True,
        "vad_filter": True,
        "initial_prompt": "Euh, bon",
    }
    # The voice checks keep their old call: no VAD, no prompt.
    worker._align({"audio": "a.wav", "language": "fr", "model": "small"}, {})
    assert seen[-1] == {"language": "fr", "word_timestamps": True}


def test_worker_engine_batches_speech_and_reports_per_file_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adanvil.voice.engines import WorkerEngine

    engine = WorkerEngine(python="python")
    captured: dict = {}

    def fake_run(operations, paths, *, progress=None, strict=True):
        captured.update(operations=operations, strict=strict)
        return [
            {"ok": True, "words": [[0.0, 0.5, "a", 0.9]], "text": "a", "language": "fr"},
            {"ok": False, "error": "boom"},
        ]

    monkeypatch.setattr(engine, "_run", fake_run)
    first, second = engine.align(
        [Path("a.wav"), Path("b.wav")], "fr", "small", vad=True, prompt="p", strict=False
    )
    assert captured["strict"] is False and len(captured["operations"]) == 2
    assert captured["operations"][0] == {
        "op": "align",
        "audio": "a.wav",
        "language": "fr",
        "model": "small",
        "vad": True,
        "prompt": "p",
    }
    assert first.language == "fr" and first.error is None and first.words == [(0.0, 0.5, "a", 0.9)]
    assert second.error == "boom" and second.words == []
    # The voice pipeline's plain call is unchanged: no extra operation field.
    captured.clear()
    monkeypatch.setattr(
        engine, "_run", lambda operations, paths, **kw: captured.update(ops=operations) or []
    )
    engine.align([], "en", "small")
    assert captured["ops"] == []
