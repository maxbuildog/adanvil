"""Talking-head UGC, second iteration: better words, progressive indexing, audio files,
compact outputs, recipe roles and four new validation checks."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest
import yaml
from mcp import Client
from pydantic import ValidationError

from adanvil import jobs
from adanvil.brand import Brand, Product, Speech, load_brand, save_brand
from adanvil.library import (
    entry_line,
    index_library,
    load_catalog,
    refine_speech,
    refine_targets,
    search,
)
from adanvil.library.separate_audio import separate_audio_candidates
from adanvil.onboard.site import ProductFact, SiteFacts
from adanvil.project import init_project
from adanvil.render import compose
from adanvil.server import build_server
from adanvil.speech import store
from adanvil.speech import transcribe as speech_transcribe
from adanvil.speech.corrections import apply_corrections, closeness, correction_problem
from adanvil.speech.library import SpeechLibrary
from adanvil.speech.model import SpeechWord
from adanvil.storyboard import (
    SpeechCaptionsOverlay,
    Storyboard,
    get_recipe,
    list_recipes,
    validate_storyboard,
)
from adanvil.vocab import ROLE_VALUES
from tests.speech_helpers import ScriptedEngine, make_audio, make_clip, spoken
from tests.test_speech_edit import _expand, _jump_cut_board, _project, _talk_project, _valid_ad


@pytest.fixture(autouse=True)
def _speech_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADANVIL_SPEECH", "auto")


def _texts(result):
    return [item.text for item in result.content if item.type == "text"]


def _json(result):
    return json.loads(_texts(result)[0])


# ------------------------------------------------------------------------ the project

HEARD = {
    "HOOK1": [
        (0.3, 0.6, "Mes"),
        (0.65, 0.9, "tasks"),
        (0.95, 1.2, "changent."),
        (1.8, 2.3, "Brio."),
    ],
    "HOOK2": [(0.3, 0.7, "Bonjour"), (0.75, 1.1, "tout.")],
}
BETTER = {
    "HOOK1": [
        (0.3, 0.6, "Mes"),
        (0.65, 0.9, "tasses"),
        (0.95, 1.2, "changent."),
        (1.8, 2.3, "Brio."),
    ],
}


def _footage(tmp_path: Path) -> Path:
    init_project(tmp_path, "Ember & Oak")
    footage = tmp_path / "footage"
    make_clip(footage / "HOOK1.mov", 3.0, [(0.3, 1.2, 300), (1.8, 2.3, 320)])
    make_clip(footage / "HOOK2.mov", 3.0, [(0.3, 1.1, 300)])
    make_clip(footage / "quiet.mov", 2.0, [])  # an audio stream that is silent
    make_clip(footage / "broll.mov", 2.0, audio=False)
    return tmp_path


def _entry(root: Path, name: str):
    return next(e for e in load_catalog(root).entries.values() if e.path.endswith(name))


# ------------------------------------------------------------------------ 1. better words


def test_brand_vocabulary_is_the_initial_prompt_of_the_transcriber(tmp_path: Path) -> None:
    root = _footage(tmp_path)
    save_brand(
        Brand(
            name="Ember & Oak",
            product=Product(name="Cinder Kettle"),
            speech=Speech(vocabulary=["gooseneck", "pour-over", "ember & oak"]),
        ),
        root / "brand.yaml",
    )
    assert load_brand(root / "brand.yaml").speech.vocabulary[0] == "gooseneck"
    assert speech_transcribe.speech_vocabulary(root) == [
        "Ember & Oak",
        "Cinder Kettle",
        "gooseneck",
        "pour-over",
    ]  # no duplicate: "ember & oak" is the brand name again

    engine = ScriptedEngine(HEARD)
    index_library(root, "footage", engine=engine)
    prompt = engine.options[0]["prompt"]
    assert prompt == "Glossary: Ember & Oak, Cinder Kettle, gooseneck, pour-over."


def test_the_prompt_keeps_the_environment_text_and_works_without_a_brand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _footage(tmp_path)
    (root / "brand.yaml").unlink(missing_ok=True)
    assert speech_transcribe.speech_prompt(root) is None
    assert speech_transcribe.speech_vocabulary(root) == []
    monkeypatch.setenv("ADANVIL_SPEECH_PROMPT", "Euh, bon, alors.")
    assert speech_transcribe.speech_prompt(root) == "Euh, bon, alors."
    save_brand(Brand(name="Ember & Oak"), root / "brand.yaml")
    assert speech_transcribe.speech_prompt(root) == "Euh, bon, alors. Glossary: Ember & Oak."
    # A very long vocabulary is cut at the front: the end of a prompt weighs the most.
    long = Brand(name="Ember & Oak", speech=Speech(vocabulary=[f"term{n:03d}" for n in range(200)]))
    save_brand(long, root / "brand.yaml")
    prompt = speech_transcribe.speech_prompt(root)
    assert prompt is not None and len(prompt) <= 400 and prompt.endswith("term199.")


def test_in_process_whisper_gets_the_vocabulary_as_initial_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _footage(tmp_path)
    brand = Brand(name="Ember & Oak", speech=Speech(vocabulary=["gooseneck"]))
    save_brand(brand, root / "brand.yaml")
    prompts: list[str | None] = []

    class FakeModel:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def transcribe(self, _audio: str, **options):
            prompts.append(options.get("initial_prompt"))
            word = SimpleNamespace(start=0.3, end=0.6, word=" Salut", probability=0.9)
            return [SimpleNamespace(words=[word])], SimpleNamespace(language="fr")

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeModel))
    index_library(root, "footage")
    assert prompts == ["Glossary: Ember & Oak, gooseneck."] * 2


def test_refine_replaces_the_words_with_a_stronger_model(tmp_path: Path) -> None:
    root = _footage(tmp_path)
    engine = ScriptedEngine(HEARD, by_model={"medium": BETTER})
    index_library(root, "footage", engine=engine)
    before = _entry(root, "HOOK1.mov")
    assert before.speech.text == "Mes tasks changent. Brio."
    sidecar = before.speech.words_file
    assert store.read_sidecar(root, sidecar).model == "base"
    assert search(root, text="tasses") == []

    result = refine_speech(root, [before.id], engine=engine)
    assert result.model == "medium" and result.refined == [before.id]
    assert engine.calls[-1] == (["HOOK1"], None, "medium")  # the stronger model, one clip
    after = _entry(root, "HOOK1.mov")
    assert after.speech.text == "Mes tasses changent. Brio."
    assert after.speech.words_file == sidecar  # replaced in place
    data = store.read_sidecar(root, sidecar)
    assert data.model == "medium" and [w.text for w in data.words][1] == "tasses"
    assert [entry.path for entry in search(root, text="TASSES")] == ["footage/HOOK1.mov"]
    # The other clips are untouched, and indexing again does not undo the refinement.
    assert _entry(root, "HOOK2.mov").speech.text == "Bonjour tout."
    index_library(root, "footage", engine=engine)
    assert _entry(root, "HOOK1.mov").speech.text == "Mes tasses changent. Brio."
    assert store.read_sidecar(root, sidecar).model == "medium"


def test_refine_gives_the_vocabulary_and_never_loses_old_words(tmp_path: Path) -> None:
    root = _footage(tmp_path)
    save_brand(Brand(name="Ember & Oak", speech=Speech(vocabulary=["tasses"])), root / "brand.yaml")
    engine = ScriptedEngine(HEARD, by_model={"medium": BETTER})
    index_library(root, "footage", engine=engine)
    hook = _entry(root, "HOOK1.mov")

    class Failing(ScriptedEngine):
        def align(self, paths, language, model, **options):
            results = super().align(paths, language, model, **options)
            results[0].error = "model crashed"
            return results

    failing = Failing(HEARD, by_model={"medium": BETTER})
    result = refine_speech(root, [hook.id], engine=failing, model="small")
    assert not result.refined and result.errors == [
        {"path": "footage/HOOK1.mov", "message": "speech failed: model crashed"}
    ]
    assert failing.options[0]["prompt"] == "Glossary: Ember & Oak, tasses."
    kept = _entry(root, "HOOK1.mov")
    assert kept.speech.status == "done" and kept.speech.text == "Mes tasks changent. Brio."


def test_refine_targets_take_paths_folders_and_ids_and_say_why_not(tmp_path: Path) -> None:
    root = _footage(tmp_path)
    index_library(root, "footage", engine=ScriptedEngine(HEARD))
    hook = _entry(root, "HOOK1.mov")
    entries, skipped = refine_targets(
        root, [hook.id, "footage/HOOK2.mov", "footage/quiet.mov", "footage/broll.mov", "nope.mov"]
    )
    assert [e.path for e in entries] == ["footage/HOOK1.mov", "footage/HOOK2.mov"]
    reasons = {item["path"]: item["reason"] for item in skipped}
    assert "silent" in reasons["footage/quiet.mov"]
    assert reasons["footage/broll.mov"] == "no sound track"
    assert "not indexed" in reasons["nope.mov"]
    folder, _ = refine_targets(root, ["footage/"])
    assert [e.path for e in folder] == ["footage/HOOK1.mov", "footage/HOOK2.mov"]


def test_refine_without_a_transcriber_is_an_error_not_a_lost_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _footage(tmp_path)
    index_library(root, "footage", engine=ScriptedEngine(HEARD))
    hook = _entry(root, "HOOK1.mov")
    monkeypatch.setenv("ADANVIL_CACHE", str(tmp_path / "no-voice-environment"))
    monkeypatch.delenv("ADANVIL_VOICE_PYTHON", raising=False)
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(RuntimeError, match="unavailable"):
        refine_speech(root, [hook.id])
    assert _entry(root, "HOOK1.mov").speech.text == "Mes tasks changent. Brio."


def test_refine_runs_as_a_detached_job_with_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _footage(tmp_path)
    engine = ScriptedEngine(HEARD, by_model={"medium": BETTER})
    index_library(root, "footage", engine=engine)
    ids = [_entry(root, "HOOK1.mov").id, _entry(root, "HOOK2.mov").id]
    real_subprocess = jobs.subprocess
    no_runner = SimpleNamespace(
        Popen=lambda *args, **kwargs: SimpleNamespace(pid=0), DEVNULL=real_subprocess.DEVNULL
    )
    monkeypatch.setattr(jobs, "subprocess", no_runner)  # run the job here, not in a child
    record = jobs.start_job(
        root,
        "refine",
        {"entry_ids": ids, "model": "medium", "language": None},
        progress={"total": 2, "unit": "clips"},
    )
    monkeypatch.setattr(jobs, "subprocess", real_subprocess)
    monkeypatch.setattr(
        speech_transcribe, "_resolve_backend", lambda _root, _engine: ("engine", engine)
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
    assert final["refine"]["model"] == "medium"
    assert sorted(final["refine"]["refined"]) == sorted(ids)
    assert ("transcribing", {"done": 2, "total": 2, "unit": "clips"}) in stages
    assert engine.calls[-1][2] == "medium"
    assert _entry(root, "HOOK1.mov").speech.text == "Mes tasses changent. Brio."
    # A refine job never waits for the project lock (like an index job).
    assert jobs._has_earlier_pending_job(Path(record["job_file"]), final) is False


def test_speech_refine_tool_validates_and_queues_a_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _footage(tmp_path)
    index_library(root, "footage", engine=ScriptedEngine(HEARD))
    started: list[tuple] = []

    def fake_start(project_dir, kind, inputs, *, progress=None):
        started.append((kind, dict(inputs), progress))
        return {"job_id": "job-refine-test", "job_file": str(root / "renders" / "jobs" / "x.json")}

    monkeypatch.setattr("adanvil.server.start_job", fake_start)

    async def run() -> None:
        async with Client(build_server()) as client:
            listed = {tool.name: tool for tool in (await client.list_tools()).tools}
            properties = set(listed["speech_refine"].input_schema["properties"])
            assert {"project_dir", "sources", "model", "language"} <= properties
            assert "medium" in listed["speech_refine"].description
            empty = await client.call_tool(
                "speech_refine", {"project_dir": str(root), "sources": []}
            )
            assert empty.is_error
            unknown = await client.call_tool(
                "speech_refine", {"project_dir": str(root), "sources": ["footage/ghost.mov"]}
            )
            assert unknown.is_error and "No clip to refine" in _texts(unknown)[0]
            escape = await client.call_tool(
                "speech_refine", {"project_dir": str(root), "sources": ["../outside.mov"]}
            )
            assert escape.is_error
            ok = await client.call_tool(
                "speech_refine",
                {"project_dir": str(root), "sources": ["footage", "footage/quiet.mov"]},
            )
            assert not ok.is_error, _texts(ok)
            payload = _json(ok)
            assert payload["status"] == "queued" and payload["job_id"] == "job-refine-test"
            assert payload["clips"] == 2 and payload["model"] == "medium"
            assert [item["path"] for item in payload["skipped"]] == ["footage/quiet.mov"] * 1

    anyio.run(run)
    assert started[0][0] == "refine"
    assert started[0][1]["model"] == "medium" and len(started[0][1]["entry_ids"]) == 2
    assert started[0][2] == {"total": 2, "unit": "clips"}


def test_a_correction_must_stay_close_to_the_heard_word() -> None:
    assert closeness("tasks", "tasses") >= 0.5
    assert correction_problem("tasks", "tasses") is None
    assert correction_problem("100", "10") is None
    assert correction_problem("Glowly", "glowly") is None
    # A different word is not a spelling fix: it would put words in the speaker's mouth.
    problem = correction_problem("temps", "sucre")
    assert problem is not None and "speech_refine" in problem and "0.20" in problem
    assert "speech_refine" in correction_problem("we", "guaranteed result")
    assert "one word" in correction_problem("glow ly", "Glowly")
    assert correction_problem("", "x") and correction_problem("x", "  ")

    with pytest.raises(ValidationError, match="speech_corrections.*speech_refine"):
        SpeechCaptionsOverlay.model_validate({"corrections": {"temps": "sucre"}})
    ok = SpeechCaptionsOverlay.model_validate({"corrections": {"tasks": "tasses", "100": "10"}})
    assert ok.corrections == {"tasks": "tasses", "100": "10"}
    with pytest.raises(ValidationError, match="same heard word"):
        SpeechCaptionsOverlay.model_validate({"corrections": {"Tasks": "tasses", "tasks": "tass"}})


def test_corrections_apply_word_by_word_ignoring_case_and_keeping_punctuation() -> None:
    words = [SpeechWord(0.0, 0.2, text) for text in ["Tasks,", "des", "TASKS", "(tasks)", "btasks"]]
    fixed, used = apply_corrections(words, {"tasks": "tasses", "des": "dès", "never": "nevers"})
    assert [word.text for word in fixed] == ["Tasses,", "dès", "TASSES", "(tasses)", "btasks"]
    assert used == {"tasks": 3, "des": 1, "never": 0}
    # A replacement that has capitals is shown as written.
    fixed, _ = apply_corrections(words[:1], {"tasks": "TaSses"})
    assert fixed[0].text == "TaSses,"
    assert apply_corrections(words, {})[1] == {}


MISHEARD = [
    (0.5, 0.8, "Mes"),
    (0.85, 1.2, "Tasks"),
    (1.25, 1.9, "changent."),
    (3.0, 3.3, "Voici"),
    (3.35, 3.7, "tasks"),
    (3.75, 4.3, "pour"),
    (4.35, 4.7, "vous."),
]


def test_corrections_fix_the_captions_and_are_listed_for_review(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _project(tmp_path, media)
    make_clip(root / "footage" / "talk.mov", 6.0, [(0.5, 4.7, 300)])
    index_library(root, "footage", engine=ScriptedEngine({"talk": MISHEARD}), transcribe=True)
    board = _jump_cut_board()
    overlay = {
        "type": "speech_captions",
        "mode": "sentence",
        "corrections": {"tasks": "tasses", "ghost": "ghosts"},
    }
    shown = [" ".join(c.lines) for c in _expand(root, board, overlay)]
    assert shown[0] == "Mes Tasses changent."  # the capital of "Tasks" is kept
    assert any("Voici tasses" in text for text in shown)
    plain = [" ".join(c.lines) for c in _expand(root, board, {**overlay, "corrections": {}})]
    assert plain[0] == "Mes Tasks changent."

    ad = _valid_ad(
        {"overlays": [{"type": "speech_captions", "mode": "chunks", "max_words": 3,
                       "corrections": {"tasks": "tasses", "ghost": "ghosts"}}]}
    )  # fmt: skip
    report = validate_storyboard(ad, root=root)
    assert report.ok, report.errors
    listed = [w for w in report.warnings if w.code == "speech_corrections"]
    assert len(listed) == 1 and listed[0].where == "overlays[0].corrections"
    assert "'tasks' -> 'tasses' (2x)" in listed[0].message
    assert "'ghost' -> 'ghosts' (0x)" in listed[0].message
    assert "speech_refine" in listed[0].message
    # No corrections, no warning.
    clean = validate_storyboard(_valid_ad(), root=root)
    assert not [w for w in clean.warnings if w.code == "speech_corrections"]
    # A correction that changes the word too much is an error with its own code.
    bad = _valid_ad({"overlays": [{"type": "speech_captions", "corrections": {"vous": "demain"}}]})
    refused = validate_storyboard(bad, root=root)
    assert not refused.ok
    error = next(e for e in refused.errors if e.code == "speech_corrections")
    assert "speech_refine" in error.message
    # Old storyboards (no corrections field) stay valid.
    assert Storyboard.model_validate(_valid_ad()).overlays[0].corrections == {}


# ------------------------------------------------------------- 2. progressive indexing

STUB_WHISPER = '''
import os
import time
from types import SimpleNamespace

_calls = {"n": 0}


class WhisperModel:
    def __init__(self, name, device="cpu", compute_type="int8", cpu_threads=1):
        pass

    def transcribe(self, audio, **options):
        _calls["n"] += 1
        release = os.environ.get("ADANVIL_STUB_RELEASE")
        if release and _calls["n"] >= 2:  # the second file waits for the parent to look
            deadline = time.time() + 40
            while not os.path.exists(release) and time.time() < deadline:
                time.sleep(0.05)
        stem = os.path.splitext(os.path.basename(str(audio)))[0].lower()
        word = SimpleNamespace(start=0.3, end=0.6, word=" " + stem + "spoken", probability=0.9)
        segment = SimpleNamespace(text=stem + "spoken", words=[word])
        return [segment], SimpleNamespace(language="fr")
'''


def test_the_worker_stores_each_file_while_the_next_one_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    init_project(root, "Ember & Oak")
    for name in ("alpha", "bravo", "charlie"):
        make_clip(root / "footage" / f"{name}.mov", 2.0, [(0.3, 1.5, 300)])
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "faster_whisper.py").write_text(STUB_WHISPER, encoding="utf-8")
    release = tmp_path / "release"
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join(filter(None, [str(stub), os.environ.get("PYTHONPATH")]))
    )
    monkeypatch.setenv("ADANVIL_STUB_RELEASE", str(release))
    monkeypatch.setenv("ADANVIL_VOICE_PYTHON", sys.executable)
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # no in-process transcriber

    seen: list[tuple[int, list[str]]] = []

    def progress(message: str) -> None:
        if not message.startswith("speech ") or "transcribing" not in message:
            return
        done = int(message.split()[1].split("/")[0])
        stored = sorted(
            entry.path
            for entry in load_catalog(root).entries.values()
            if entry.speech is not None and entry.speech.status == "done"
        )
        seen.append((done, stored))
        if done == 1:
            # The worker is stuck on the second file until we release it: the first
            # file must already be searchable.
            assert search(root, text="alphaspoken")[0].path == "footage/alpha.mov"
            assert not search(root, text="bravospoken")
            release.write_text("go", encoding="utf-8")

    result = index_library(root, "footage", progress=progress)
    assert not result.errors, result.errors
    assert result.speech and set(result.speech.values()) == {"done"}
    assert (1, ["footage/alpha.mov"]) in seen  # progressive: one file stored, two to go
    assert seen[-1][0] == 3 and len(seen[-1][1]) == 3
    assert [entry.path for entry in search(root, text="charliespoken")] == ["footage/charlie.mov"]


def test_a_streaming_worker_that_fails_midway_keeps_the_files_already_done(
    tmp_path: Path,
) -> None:
    root = _footage(tmp_path)
    stored: list[int] = []

    class Streaming:
        def align(self, paths, language, model, **options):
            # Only the plain call is used for a non-worker engine; nothing is streamed.
            raise RuntimeError("worker died")

    outcomes = speech_transcribe.transcribe_clips(
        [root / "footage" / "HOOK1.mov", root / "footage" / "HOOK2.mov"],
        project_root=root,
        engine=Streaming(),
        on_result=lambda index, outcome: stored.append(index),
    )
    assert [o.status for o in outcomes] == ["failed: worker died"] * 2
    assert sorted(stored) == [0, 1]


def test_audio_extraction_of_a_batch_does_not_change_the_results(tmp_path: Path) -> None:
    root = _footage(tmp_path)
    for number in range(5):
        make_clip(root / "footage" / f"extra{number}.mov", 1.5, [(0.2, 1.2, 300)])
    scripts = {f"extra{number}": [(0.3, 0.6, f"mot{number}")] for number in range(5)}
    engine = ScriptedEngine({**HEARD, **scripts})
    index_library(root, "footage", engine=engine)
    assert len(engine.calls) == 1  # one batch, one model load
    for number in range(5):
        assert _entry(root, f"extra{number}.mov").speech.text == f"mot{number}"


# ------------------------------------------------------------------ 3. audio files


def _voices(root: Path) -> None:
    footage = root / "footage"
    make_clip(footage / "HOOKS" / "HOOK1.mov", 3.0, [])  # the sound track is silent
    make_clip(footage / "HOOKS" / "HOOK2.mov", 3.0, [])
    make_clip(footage / "HOOKS" / "demo.mov", 2.0, audio=False)
    make_audio(footage / "HOOKS" / "VOIX OFF" / "HOOK1.1.m4a", 3.0, [(0.3, 2.0, 300)])
    make_audio(footage / "VOIX OFF" / "HOOK2.wav", 3.0, [(0.3, 2.0, 300)])
    make_audio(footage / "VOIX OFF" / "intro.mp3", 3.0, [(0.3, 2.0, 300)])
    make_audio(footage / "VOIX OFF" / "outro.aac", 2.0, [(0.3, 1.5, 300)])


VOICE_SCRIPTS = {
    "HOOK1.1": [(0.3, 0.7, "Bonjour"), (0.8, 1.5, "tout")],
    "HOOK2": [(0.3, 0.7, "Salut")],
    "intro": [(0.3, 0.7, "Alors")],
    "outro": [(0.3, 0.7, "Voila")],
}


def test_audio_files_are_indexed_as_audio_entries_with_speech(tmp_path: Path) -> None:
    root = tmp_path
    init_project(root, "Ember & Oak")
    _voices(root)
    engine = ScriptedEngine(VOICE_SCRIPTS)
    result = index_library(root, "footage", engine=engine)
    assert not result.errors, result.errors
    assert sorted(engine.calls[0][0]) == ["HOOK1.1", "HOOK2", "intro", "outro"]

    entries = {Path(e.path).name: e for e in load_catalog(root).entries.values()}
    for name in ("HOOK1.1.m4a", "HOOK2.wav", "intro.mp3", "outro.aac"):
        entry = entries[name]
        assert entry.kind == "audio" and entry.has_audio and entry.sheet is None
        assert entry.duration and entry.duration > 1.5
        assert entry.speech is not None and entry.speech.status == "done"
        assert entry.speech.has_speech and entry.speech.words_file
    voice = entries["HOOK1.1.m4a"]
    assert voice.speech.text == "Bonjour tout"
    data = store.read_sidecar(root, voice.speech.words_file)
    assert data.levels_db and data.level(0.3, 2.0) > data.level(2.3, 2.9) + 30  # levels measured
    assert [e.path for e in search(root, text="bonjour")] == ["footage/HOOKS/VOIX OFF/HOOK1.1.m4a"]
    assert len(search(root, kind="audio")) == 4
    assert "| audio |" in entry_line(voice) and entry_line(voice).endswith("| speech")
    assert SpeechLibrary(root).data("footage/HOOKS/VOIX OFF/HOOK1.1.m4a") is not None
    # Indexed once: nothing is transcribed again.
    index_library(root, "footage", engine=engine)
    assert len(engine.calls) == 1


def test_a_video_with_silent_sound_lists_the_voice_files_that_may_belong_to_it(
    tmp_path: Path,
) -> None:
    root = tmp_path
    init_project(root, "Ember & Oak")
    _voices(root)
    index_library(root, "footage", engine=ScriptedEngine(VOICE_SCRIPTS))
    entries = {Path(e.path).name: e for e in load_catalog(root).entries.values()}
    hook1, hook2 = entries["HOOK1.mov"], entries["HOOK2.mov"]
    assert hook1.speech.status == "silent_audio" and not hook1.speech.has_speech
    assert hook1.speech.language is None and hook1.speech.words_file is None
    # HOOK1.1.m4a sits in a folder below the video's, HOOK2.wav in a sibling folder.
    assert hook1.speech.separate_audio_candidates == ["footage/HOOKS/VOIX OFF/HOOK1.1.m4a"]
    assert hook2.speech.separate_audio_candidates == ["footage/VOIX OFF/HOOK2.wav"]
    assert "silent_audio" in entry_line(hook1) and "voice_files=1" in entry_line(hook1)
    # B-roll without a sound track and no name in common gets no candidate.
    assert entries["demo.mov"].speech.status == "not attempted: no audio"
    assert entries["demo.mov"].speech.separate_audio_candidates == []
    # The catalogue stays short: the field is only written when it has something to say.
    raw = json.loads((root / "catalog.json").read_text(encoding="utf-8"))["entries"]
    assert "separate_audio_candidates" in raw[hook1.id]["speech"]
    assert "separate_audio_candidates" not in raw[entries["demo.mov"].id]["speech"]


def test_candidates_follow_audio_files_indexed_later(tmp_path: Path) -> None:
    root = tmp_path
    init_project(root, "Ember & Oak")
    make_clip(root / "footage" / "HOOK3.mov", 2.0, [])
    engine = ScriptedEngine({"HOOK3": [(0.3, 0.6, "x")], "HOOK3.take2": [(0.3, 0.6, "Oui")]})
    index_library(root, "footage", engine=engine)
    assert _entry(root, "HOOK3.mov").speech.separate_audio_candidates == []
    make_audio(root / "footage" / "voices" / "HOOK3.take2.m4a", 2.0, [(0.3, 1.5, 300)])
    index_library(root, "footage/voices", engine=engine)  # another folder, another run
    assert _entry(root, "HOOK3.mov").speech.separate_audio_candidates == [
        "footage/voices/HOOK3.take2.m4a"
    ]


def test_candidate_matching_is_about_shared_words_nearby() -> None:
    audio = [
        "footage/hooks/VOIX OFF/HOOK1.1.m4a",
        "footage/hooks/VOIX OFF/HOOK12.m4a",
        "footage/hooks/VOIX OFF/HOOK2.m4a",
        "footage/hooks/VOIX OFF/voice 1.m4a",
        "footage/far/away/HOOK1.m4a",
        "footage/hooks/Interview Anna.m4a",
        "footage/hooks/Interview Marie.m4a",
    ]
    assert separate_audio_candidates("footage/hooks/HOOK1.mov", audio) == [
        "footage/hooks/VOIX OFF/HOOK1.1.m4a"
    ]  # not HOOK12, not HOOK2, not a bare "1", not a folder too far away
    assert separate_audio_candidates("footage/hooks/Interview Anna.mov", audio) == [
        "footage/hooks/Interview Anna.m4a",
        "footage/hooks/Interview Marie.m4a",
    ]  # the same file first (all words in common)
    nearby = ["footage/hooks/eclair 3.wav"]
    assert separate_audio_candidates("footage/hooks/Éclair 3.mov", nearby) == nearby  # accents


def test_a_forced_silent_clip_that_yields_no_word_is_silent_audio_not_done(
    tmp_path: Path,
) -> None:
    root = _footage(tmp_path)
    engine = ScriptedEngine(HEARD)  # nothing scripted for "quiet": no word
    result = index_library(root, "footage", engine=engine, transcribe=True)
    quiet = _entry(root, "quiet.mov")
    assert quiet.speech.status == "silent_audio" and quiet.speech.language is None
    assert result.speech[quiet.id] == "silent_audio"
    assert not quiet.speech.has_speech and quiet.speech.words_file is None
    # A clip with sound but no word (music, noise) is done, without an invented language.
    music = tmp_path / "footage" / "music_only.mov"
    make_clip(music, 2.0, [(0.0, 2.0, 440)])
    index_library(root, "footage", engine=engine)
    assert _entry(root, "music_only.mov").speech.status == "done"
    assert _entry(root, "music_only.mov").speech.language is None


def test_audio_entries_through_the_tools(tmp_path: Path) -> None:
    root = tmp_path
    init_project(root, "Ember & Oak")
    _voices(root)
    index_library(root, "footage", engine=ScriptedEngine(VOICE_SCRIPTS))
    voice = _entry(root, "HOOK1.1.m4a")
    video = _entry(root, "HOOK1.mov")

    async def run() -> None:
        async with Client(build_server()) as client:
            viewed = await client.call_tool(
                "library_view", {"project_dir": str(root), "entry_id": voice.id}
            )
            assert not viewed.is_error, _texts(viewed)
            assert _json(viewed)["kind"] == "audio"
            assert not any(item.type == "image" for item in viewed.content)
            found = await client.call_tool(
                "library_search", {"project_dir": str(root), "kind": "audio", "text": "tout"}
            )
            assert _json(found)["count"] == 1
            heard = await client.call_tool(
                "speech_view", {"project_dir": str(root), "entry_id": voice.id}
            )
            payload = _json(heard)
            assert payload["kind"] == "audio" and payload["phrases"][0]["text"] == "Bonjour tout"
            assert payload["clean_segments"][0]["source"] == voice.path
            silent = await client.call_tool(
                "speech_view", {"project_dir": str(root), "entry_id": video.id}
            )
            payload = _json(silent)
            assert payload["speech_status"] == "silent_audio" and payload["phrases"] == []
            assert payload["separate_audio_candidates"] == [voice.path]
            assert "audio_from" in payload["hint"] and "sync" in payload["hint"]
            takes = await client.call_tool(
                "speech_takes", {"project_dir": str(root), "source": "footage/VOIX OFF"}
            )
            assert _json(takes)["files"] == 3  # audio files take part in the comparison

    anyio.run(run)


def test_an_audio_file_is_an_audio_from_source(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = tmp_path
    init_project(root, "Ember & Oak")
    make_clip(root / "footage" / "HOOK1.mov", 3.0, [])
    make_clip(root / "footage" / "broll.mp4", 3.0, audio=False)
    make_audio(root / "footage" / "voices" / "HOOK1.1.m4a", 3.0, [(0.3, 2.5, 300)])
    index_library(root, "footage", engine=ScriptedEngine(VOICE_SCRIPTS))
    board = {
        "name": "voiced",
        "segments": [
            {
                "type": "clip",
                "source": "footage/HOOK1.mov",
                "in": 0.0,
                "out": 1.5,
                "audio_from": {"source": "footage/voices/HOOK1.1.m4a", "in": 0.2},
            },
            {
                "type": "clip",
                "source": "footage/broll.mp4",
                "in": 0.0,
                "out": 1.5,
                "audio_from": {"source": "footage/voices/HOOK1.1.m4a", "in": 1.4},
            },
        ],
        "overlays": [{"type": "speech_captions", "mode": "chunks", "max_words": 2}],
        "audio": {"clips": True},
    }
    report = validate_storyboard(board, root=root)
    assert report.ok, report.errors
    rendered = compose(Storyboard.model_validate(board), root=root, quality="preview")
    assert rendered.duration == pytest.approx(3.0, abs=0.1)
    from adanvil.audio import audio_duration

    assert audio_duration(rendered.path) == pytest.approx(rendered.duration, abs=0.1)


# ------------------------------------------------------------------ 4. compact outputs


def test_speech_view_is_compact_and_paginated(tmp_path: Path) -> None:
    root = tmp_path
    init_project(root, "Ember & Oak")
    make_clip(root / "footage" / "long.mov", 16.0, [(0.3, 15.0, 300)])
    script = spoken(" ".join("one two |0.5" for _ in range(12)), 0.3)
    script[2] = (script[2][0], script[2][1], "euh")  # a filler in phrase 1
    low = list(script)
    index_library(root, "footage", engine=ScriptedEngine({"long": low}), transcribe=True)

    async def run() -> None:
        async with Client(build_server()) as client:
            full = _json(
                await client.call_tool(
                    "speech_view", {"project_dir": str(root), "source": "footage/long.mov"}
                )
            )
            assert full["phrase_count"] == 12 and len(full["phrases"]) == 12
            assert "note" not in full and "next_first_phrase" not in full
            first = full["phrases"][0]
            assert set(first) == {"index", "start", "end", "text", "flags"}  # no word list
            assert "words" not in json.dumps(full["phrases"])

            page = _json(
                await client.call_tool(
                    "speech_view",
                    {"project_dir": str(root), "source": "footage/long.mov", "max_phrases": 5},
                )
            )
            assert [p["index"] for p in page["phrases"]] == [0, 1, 2, 3, 4]
            assert page["next_first_phrase"] == 5
            assert "phrases 1 to 5 of 12" in page["note"] and "first_phrase=5" in page["note"]
            last = page["phrases"][-1]["end"]
            assert all(segment["in"] < last for segment in page["clean_segments"])
            assert len(page["clean_segments"]) < len(full["clean_segments"])

            nxt = _json(
                await client.call_tool(
                    "speech_view",
                    {
                        "project_dir": str(root),
                        "source": "footage/long.mov",
                        "max_phrases": 5,
                        "first_phrase": 10,
                        "words": True,
                    },
                )
            )
            assert [p["index"] for p in nxt["phrases"]] == [10, 11]
            assert "next_first_phrase" not in nxt and "first_phrase" not in nxt["note"]
            assert nxt["phrases"][0]["words"][0]["text"] == "one"
            beyond = _json(
                await client.call_tool(
                    "speech_view",
                    {"project_dir": str(root), "source": "footage/long.mov", "first_phrase": 40},
                )
            )
            assert beyond["phrases"] == [] and "no phrase" in beyond["note"]
            bad = await client.call_tool(
                "speech_view",
                {"project_dir": str(root), "source": "footage/long.mov", "max_phrases": 0},
            )
            assert bad.is_error

    anyio.run(run)


def test_speech_view_flags_say_what_to_check() -> None:
    from adanvil.speech.analyze import read_speech, reading_payload

    def word(start: float, text: str, probability: float = 0.95) -> SpeechWord:
        return SpeechWord(start, start + 0.3, text, probability)

    words = [
        word(0.0, "Mon"),
        word(0.35, "matin"),  # phrase 0: said again by the next phrase, a retake
        word(1.2, "Mon"),
        word(1.55, "matin"),
        word(1.9, "change."),  # phrase 1: clean
        word(3.5, "euh"),
        word(3.85, "bon", 0.2),  # phrase 2: a hesitation and a word the model doubted
    ]
    payload = reading_payload(read_speech(words, "footage/a.mov"), source="footage/a.mov")
    assert [phrase["flags"] for phrase in payload["phrases"]] == [
        ["restart"],
        [],
        ["filler", "uncertain"],
    ]
    assert [phrase["text"] for phrase in payload["phrases"]][2] == "euh bon"
    assert payload["phrase_count"] == 3


def _big_site() -> SiteFacts:
    products = [
        ProductFact(
            title=f"Product number {number} " + "with a very long title " * 4,
            handle=f"product-{number}",
            url=f"https://example.test/products/product-{number}",
            price=f"{10 + number}.00",
            currency="USD",
            images=[f"https://example.test/img/{number}-{k}.jpg" for k in range(6)],
            description="A long description of the product. " * 60,
            tags=[f"tag{k}" for k in range(20)],
        )
        for number in range(60)
    ]
    return SiteFacts(
        url="https://example.test",
        final_url="https://example.test/",
        title="Ember & Oak | Small-batch coffee",
        description="Small-batch coffee roasted to order. " * 20,
        language="en",
        logo_candidates=[f"https://example.test/logo{k}.png" for k in range(8)],
        headings=[f"Heading number {k}" for k in range(20)],
        prices=[f"${k}.00" for k in range(20)],
        guarantee_phrases=["30-day guarantee " * 30] * 7,
        review_snippets=["Great coffee " * 40] * 9,
        is_shopify=True,
        products=products,
        warnings=[f"warning {k}" for k in range(9)],
    )


def test_brand_onboard_returns_a_compact_summary_and_saves_the_full_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_project(tmp_path, "Ember & Oak")
    big = _big_site()
    monkeypatch.setattr("adanvil.onboard.build.analyze_site", lambda url, **_kw: big)

    async def run() -> None:
        async with Client(build_server()) as client:
            result = await client.call_tool(
                "brand_onboard", {"project_dir": str(tmp_path), "url": "https://example.test"}
            )
            assert not result.is_error, _texts(result)
            raw = _texts(result)[0]
            payload = json.loads(raw)
            facts = payload["facts"]
            assert facts["facts_file"] == "brand/site_facts.json"
            assert facts["product_count"] == 60 and len(facts["products"]) == 8
            assert set(facts["products"][0]) <= {"title", "price", "currency", "url",
                                                  "compare_at_price"}  # fmt: skip
            assert facts["more"]["products"] == 52 and facts["more"]["headings"] == 12
            assert len(facts["headings"]) == 8 and len(facts["logo_candidates"]) == 3
            assert len(facts["description"]) <= 240 and "description" not in facts["products"][0]
            # Compact: a few kilobytes for the whole answer (facts and questionnaire).
            assert len(raw) < 14_000, len(raw)
            assert "A long description of the product." not in raw
            assert payload["questionnaire"] and payload["open_questions"] == []
            # The full facts are in the file, whole.
            saved = json.loads((tmp_path / "brand" / "site_facts.json").read_text("utf-8"))
            assert len(saved["products"]) == 60 and saved["products"][0]["images"]
            summary_only = len(json.dumps(facts))
            assert summary_only < 6_000, summary_only

            # Step two with a url saves the facts it read, so the reference stays valid.
            (tmp_path / "brand" / "site_facts.json").unlink()
            second = await client.call_tool(
                "brand_onboard",
                {
                    "project_dir": str(tmp_path),
                    "url": "https://example.test",
                    "answers": {"brand_name": "Ember & Oak"},
                },
            )
            assert not second.is_error, _texts(second)
            assert _json(second)["facts"]["facts_file"] == "brand/site_facts.json"
            assert (tmp_path / "brand" / "site_facts.json").is_file()

    anyio.run(run)


# ---------------------------------------------------------------------- 5. recipe roles


def test_every_recipe_slot_names_the_role_of_its_segments() -> None:
    from importlib import resources

    folder = resources.files("adanvil.storyboard").joinpath("recipes")
    raw_files = [entry for entry in folder.iterdir() if entry.name.endswith(".yaml")]
    assert len(raw_files) == len(list_recipes()) == 9
    for entry in raw_files:
        document = yaml.safe_load(entry.read_text(encoding="utf-8"))
        for slot in document["structure"]:
            # A slot whose name is not a role says explicitly which role it plays.
            assert slot["slot"] in ROLE_VALUES or "role" in slot, (document["id"], slot["slot"])
            assert "purpose" in slot  # the prose moved from ``role`` to ``purpose``
    for recipe in list_recipes():
        for slot in recipe.structure:
            assert slot.role in ROLE_VALUES, (recipe.id, slot.slot, slot.role)
            assert slot.purpose
    ugc = {slot.slot: slot.role for slot in get_recipe("ugc_talking").structure}
    assert ugc["result"] == "payoff" and ugc["end_card"] == "cta" and ugc["hook"] == "hook"


def test_recipe_get_shows_the_role_and_the_example_uses_it() -> None:
    async def run() -> None:
        async with Client(build_server()) as client:
            recipe = _json(await client.call_tool("recipe_get", {"recipe_id": "ugc_talking"}))
            slots = {slot["slot"]: slot for slot in recipe["structure"]}
            assert slots["result"]["role"] == "payoff" and slots["result"]["purpose"]
            assert all(slot["role"] in ROLE_VALUES for slot in recipe["structure"])

    anyio.run(run)
    recipe = get_recipe("ugc_talking")
    by_label = {s["label"]: s["role"] for s in recipe.example["segments"]}
    slot_roles = {slot.slot: slot.role for slot in recipe.structure}
    for label, role in by_label.items():
        assert role == slot_roles[label.split(":")[0]]  # the segment role is the slot's role
        assert role in ROLE_VALUES
    assert Storyboard.model_validate(recipe.example).segments[5].role == "payoff"


def test_a_slot_name_that_is_not_a_role_needs_a_role() -> None:
    from adanvil.storyboard.recipes import Slot

    base = {"purpose": "x", "count": 1, "shot_duration": [1.0, 2.0], "footage": "f", "caption": "c"}
    assert Slot.model_validate({"slot": "hook", **base}).role == "hook"  # defaults to its name
    with pytest.raises(ValidationError):
        Slot.model_validate({"slot": "result", **base})
    with pytest.raises(ValidationError, match="slot role must be one of"):
        Slot.model_validate({"slot": "result", "role": "winner", **base})


# ------------------------------------------------- 6. cut_mid_word and contiguous sound


def _contiguous_board(second_in: float, **second: object) -> dict:
    return _valid_ad(
        {
            "overlays": [],
            "segments": [
                {"type": "clip", "source": "footage/talk.mov", "in": 0.44, "out": 1.7},
                {
                    "type": "clip",
                    "source": "footage/talk.mov",
                    "in": second_in,
                    "out": 2.9,
                    **second,
                },
            ],
        }
    )


def _mid_word(report) -> list[str]:
    return [w.where for w in report.warnings if w.code == "cut_mid_word"]


def test_a_voice_that_goes_on_across_the_cut_is_not_cut_mid_word(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _talk_project(tmp_path, media)
    # 1.70 s is inside "monde." (1.55 to 1.90). Alone, the cut is flagged.
    alone = _valid_ad({"overlays": [], "segments": [
        {"type": "clip", "source": "footage/talk.mov", "in": 0.44, "out": 1.7},
        {"type": "clip", "source": "footage/talk.mov", "in": 3.0, "out": 3.32},
    ]})  # fmt: skip
    assert _mid_word(validate_storyboard(alone, root=root)) == ["segments[0].out"]

    # The second segment resumes the same source exactly there: the sound is continuous.
    assert _mid_word(validate_storyboard(_contiguous_board(1.7), root=root)) == []
    # Within one frame (1/30 s) is still continuous; two frames are not.
    assert _mid_word(validate_storyboard(_contiguous_board(1.7 + 0.03), root=root)) == []
    assert _mid_word(validate_storyboard(_contiguous_board(1.7 + 0.07), root=root)) == [
        "segments[0].out",
        "segments[1].in",
    ]
    # B-roll over the voice: the picture changes mid-word while the voice goes on.
    broll = _valid_ad(
        {
            "overlays": [],
            "segments": [
                {"type": "clip", "source": "footage/talk.mov", "in": 0.44, "out": 1.7},
                {
                    "type": "clip",
                    "source": "footage/mute.mp4",
                    "in": 0.0,
                    "out": 1.2,
                    "audio_from": {"source": "footage/talk.mov", "in": 1.7},
                },
            ],
        }
    )
    assert _mid_word(validate_storyboard(broll, root=root)) == []
    # ...and a cut of that voice inside a word still counts: 1.7 + 1.2 = 2.9 s is a pause,
    # 3.4 s is inside "Brio" (3.35 to 3.70).
    broll["segments"][1]["out"] = 1.2
    broll["segments"][1]["audio_from"]["in"] = 1.7
    longer = json.loads(json.dumps(broll))
    longer["segments"][1]["out"] = 1.7  # the voice runs to 3.4 s
    assert _mid_word(validate_storyboard(longer, root=root)) == ["segments[1]"]
    # Not contiguous when another source, a gap of segments, or a different place.
    other = _contiguous_board(1.7)
    other["segments"][1]["source"] = "footage/tone440.mp4"
    assert "segments[0].out" in _mid_word(validate_storyboard(other, root=root))
    gap = _valid_ad(
        {
            "overlays": [],
            "segments": [
                {"type": "clip", "source": "footage/talk.mov", "in": 0.44, "out": 1.7},
                {"type": "image", "source": "brand/cta.png", "duration": 1.0},
                {"type": "clip", "source": "footage/talk.mov", "in": 1.7, "out": 2.9},
            ],
        }
    )
    assert _mid_word(validate_storyboard(gap, root=root)) == [
        "segments[0].out",
        "segments[2].in",
    ]  # an image between the two is silence: the voice is interrupted


# --------------------------------------------- 7 and 8. end card captions, CTA wording


def _ending(**changes: object) -> dict:
    board = _valid_ad(
        {
            "segments": [
                {"type": "clip", "source": "footage/talk.mov", "in": 0.44, "out": 2.0},
                {"type": "clip", "source": "footage/talk.mov", "in": 2.9, "out": 4.8},
                {
                    "type": "image",
                    "source": "brand/cta.png",
                    "duration": 3.0,
                    "motion": "zoom_in",
                    "label": "end_card",
                },
            ],
            "overlays": [],
            "audio": {"clips": True, "music": "music/bed.m4a"},
        }
    )
    board.update(changes)
    return board


def _caption(start: float, end: float | None, *lines: str) -> dict:
    return {
        "type": "caption",
        "style": "box",
        "lines": list(lines),
        "start": start,
        "end": end,
        "y": 1280,
        "size": 46,
    }


def _on_card(report) -> list[str]:
    return [w.where for w in report.warnings if w.code == "caption_on_end_card"]


def test_a_caption_running_onto_the_end_card_image_is_a_warning(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _talk_project(tmp_path, media)
    # The end card image lasts from 3.46 s to 6.46 s.
    assert _on_card(validate_storyboard(_ending(), root=root)) == []
    spilled = _ending(overlays=[_caption(3.0, 6.0, "Learn more")])
    found = validate_storyboard(spilled, root=root)
    assert _on_card(found) == ["overlays[0]"]
    message = next(w for w in found.warnings if w.code == "caption_on_end_card").message
    assert "2.5s" in message and "segments[2]" in message and "Learn more" in message
    # Until the end of the ad (end null) counts as well.
    assert _on_card(validate_storyboard(_ending(overlays=[_caption(5.0, None, "Hi")]), root=root))
    # More than 0.2 s is a warning, 0.2 s and less is not.
    def flagged(start: float, end: float | None) -> list[str]:
        board = _ending(overlays=[_caption(start, end, "A")])
        return _on_card(validate_storyboard(board, root=root))

    assert flagged(3.0, 3.6) == []  # 0.14 s on the card
    assert flagged(3.0, 3.8) == ["overlays[0]"]  # 0.34 s
    assert flagged(2.0, 3.4) == []  # ends before the card
    assert flagged(0.0, 3.0) == []  # only on the clips


def test_the_end_card_rule_covers_speech_captions_voice_captions_and_the_cta_image(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    from adanvil.voice import FakeEngine, VoiceSpec, generate_voice

    root = _talk_project(tmp_path, media)
    # A speech caption that lingers 0.2 s past its last word stays under the limit by design.
    speech = _ending(overlays=[{"type": "speech_captions", "mode": "chunks", "max_words": 3}])
    assert _on_card(validate_storyboard(speech, root=root)) == []

    # A voice caption whose chunks run over the end card image.
    track = generate_voice(
        root, VoiceSpec(engine="fake", lines=["One two three four"]), "v", engine=FakeEngine()
    )
    voiced = _ending(
        overlays=[{"type": "voice_captions", "manifest": track.manifest, "mode": "sentence"}],
        audio={"voice": track.manifest, "voice_offset": 3.0, "music": "music/bed.m4a"},
    )
    report = validate_storyboard(voiced, root=root)
    assert report.ok, report.errors
    assert _on_card(report) == ["overlays[0.0]"]

    # The brand's CTA image counts as an end card wherever it is used, label or not.
    brand = Brand(name="Ember & Oak", cta_image="brand/cta.png")
    unlabelled = _ending(overlays=[_caption(3.2, 5.0, "Learn more")])
    unlabelled["segments"][2].pop("label")
    assert _on_card(validate_storyboard(unlabelled, root=root, brand=brand)) == ["overlays[0]"]
    # An end_card that is a video clip (a product shot with its own box text) is by design.
    clip_card = _ending(overlays=[_caption(3.5, 6.0, "Learn more")])
    clip_card["segments"][2] = {
        "type": "clip", "source": "footage/tone880.mp4", "in": 0.0, "out": 3.0, "label": "end_card",
    }  # fmt: skip
    assert _on_card(validate_storyboard(clip_card, root=root)) == []


def _cta_missing(report) -> list[str]:
    return [w.message for w in report.warnings if w.code == "cta_text_missing"]


def test_the_brand_cta_wording_must_appear_in_a_caption(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _project(tmp_path, media)
    make_clip(root / "footage" / "talk.mov", 6.0, [(0.5, 4.7, 300)])
    heard = [(0.5, 0.8, "Découvrez"), (0.85, 1.1, "l'offre"), (1.15, 1.5, "maintenant.")]
    index_library(root, "footage", engine=ScriptedEngine({"talk": heard}), transcribe=True)
    brand = Brand(name="Ember & Oak", product=Product(cta_text="Découvrez l’offre"))
    nothing = validate_storyboard(_ending(), root=root, brand=brand)
    message = _cta_missing(nothing)
    assert len(message) == 1
    assert "Découvrez l’offre" in message[0]
    assert "add it as a caption on the end, unless your end card image" in message[0]
    assert "already shows it" in message[0]
    # Case, accents, quotes and punctuation do not matter.
    shown = _ending(overlays=[_caption(0.0, 2.0, "DECOUVREZ L'OFFRE !")])
    assert _cta_missing(validate_storyboard(shown, root=root, brand=brand)) == []
    # Across two lines of one caption.
    two_lines = _ending(overlays=[_caption(0.0, 2.0, "Tap to", "découvrez l'offre now")])
    assert _cta_missing(validate_storyboard(two_lines, root=root, brand=brand)) == []
    # Across the chunks of a speech caption.
    spoken_board = _ending(overlays=[{"type": "speech_captions", "mode": "chunks", "max_words": 1}])
    assert _cta_missing(validate_storyboard(spoken_board, root=root, brand=brand)) == []
    # Another wording: warned.
    other = _ending(overlays=[_caption(0.0, 2.0, "Découvrez notre histoire")])
    assert len(_cta_missing(validate_storyboard(other, root=root, brand=brand))) == 1
    # No CTA in the brand, no brand: nothing to say.
    assert _cta_missing(validate_storyboard(_ending(), root=root, brand=Brand(name="X"))) == []
    assert _cta_missing(validate_storyboard(_ending(), root=root)) == []
    # A brand CTA image may not show the exact wording: still warned, the agent judges.
    with_image = Brand(
        name="Ember & Oak", cta_image="brand/cta.png", product=Product(cta_text="Découvrez l'offre")
    )
    assert len(_cta_missing(validate_storyboard(_ending(), root=root, brand=with_image))) == 1


# ------------------------------------------------------------- the documentation matches


def test_the_guides_describe_the_new_features() -> None:
    repo = Path(__file__).resolve().parents[1]
    agents = (repo / "AGENTS.md").read_text(encoding="utf-8")
    skill = (repo / "skill" / "SKILL.md").read_text(encoding="utf-8")
    guide = (repo / "docs" / "EDITING_GUIDE.md").read_text(encoding="utf-8")
    for text in (agents, skill):
        for word in (
            "Tools (43)",
            "storyboard_save",
            "storyboard_patch",
            "karaoke",
            "Animated captions",
            "highlight_box",
            "speech_refine",
            "speech.vocabulary",
            "corrections",
            "speech_corrections",
            "silent_audio",
            "separate_audio_candidates",
            "caption_on_end_card",
            "cta_text_missing",
            "max_phrases",
            "`role`",
            "audio_from",
        ):
            assert word in text, word
    for word in (
        "speech_refine",
        "caption_on_end_card",
        "cta_text_missing",
        "silent_audio",
        "karaoke",
        "Keyword emphasis",
    ):
        assert word in guide, word
    # Both guides carry the same instructions after their front matter.
    shared = agents.split("## 4. The workflow", 1)[1].split("## 5. Storyboard cheat-sheet", 1)[0]
    assert "Talking-head UGC" in shared and "speech_refine" in shared

    def section(text: str) -> str:
        return text.split("### Talking-head UGC", 1)[1].split("### Step 7.", 1)[0]

    assert section(agents) == section(skill)  # one text, two entry points
    assert (repo / "src" / "adanvil" / "assets" / "EDITING_GUIDE.md").read_bytes() == (
        repo / "docs" / "EDITING_GUIDE.md"
    ).read_bytes()
