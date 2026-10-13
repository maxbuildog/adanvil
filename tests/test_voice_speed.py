"""Persistent voice worker, shared transcription and cheaper pronunciation checks (V2.6)."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from adanvil.voice import worker
from adanvil.voice.engines import FakeEngine, WorkerEngine
from adanvil.voice.generate import VoiceSpec, generate_voice

# A stand-in for worker.py: the real protocol loop with fake operations, so the tests need
# neither torch nor whisper. It counts "model loads" in a file shared by every process.
FAKE_WORKER = """
import json, os, sys, time
from pathlib import Path
from adanvil.voice import worker

LOADS = Path(os.environ["FAKE_LOADS"])


def load(name):
    # one file per process and model: concurrent appends to a shared file can be lost
    LOADS.with_name(f"{LOADS.name}.{os.getpid()}.{name}").write_text("x", encoding="utf-8")


def tts(op, models):
    if "tts" not in models:
        load("tts")
        models["tts"] = True
    if op["text"] == "CRASH" and "--serve" in sys.argv:
        os._exit(3)
    if op["text"] == "SLOW":
        time.sleep(60)
    Path(op["out"]).write_text(op["text"], encoding="utf-8")
    return {"ok": True, "out": op["out"], "sample_rate": 48000}


def align(op, models):
    key = op.get("model", "small")
    if key not in models:
        load(key)
        models[key] = True
    return {"ok": True, "audio": op["audio"], "words": [[0.0, 1.0, "w", 0.9]], "text": "w"}


worker._tts = tts
worker._align = align
raise SystemExit(worker.main(sys.argv[1:]))
"""


@pytest.fixture
def fake_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER, encoding="utf-8")
    loads = tmp_path / "loads.txt"
    monkeypatch.setenv("FAKE_LOADS", str(loads))
    monkeypatch.delenv("ADANVIL_VOICE_PERSISTENT", raising=False)

    def make(**kw: object) -> WorkerEngine:
        engine = WorkerEngine(python=sys.executable, project_dir=tmp_path, **kw)  # type: ignore[arg-type]
        engine.worker = script
        return engine

    def counted() -> dict[str, int]:
        counts: dict[str, int] = {}
        for path in loads.parent.glob(f"{loads.name}.*"):
            model = path.name.rsplit(".", 1)[-1]
            counts[model] = counts.get(model, 0) + 1
        return counts

    return SimpleNamespace(make=make, counted=counted, tmp=tmp_path)


def _tts_op(text: str, out: Path) -> dict:
    return {"op": "tts", "text": text, "out": str(out), "language": "fr", "reference": "r.wav"}


def test_session_loads_each_model_once_for_all_operations(fake_engine) -> None:
    engine = fake_engine.make()
    audio = [fake_engine.tmp / f"{i}.wav" for i in range(3)]
    with engine.session():
        first = engine._run([_tts_op(f"line {i}", path) for i, path in enumerate(audio)], audio)
        small = engine.align(audio, "fr", "small")
        medium = engine.align(audio, "fr", "medium")
        again = engine.align(audio[:1], "fr", "small")

    assert all(item["ok"] for item in first)
    assert [a.text for a in small] == ["w"] * 3
    assert [a.text for a in medium] == ["w"] * 3
    assert len(again) == 1
    assert fake_engine.counted() == {"tts": 1, "small": 1, "medium": 1}
    assert engine.fallbacks == []
    assert audio[2].read_text(encoding="utf-8") == "line 2"


def test_without_a_session_each_call_starts_its_own_worker(fake_engine) -> None:
    engine = fake_engine.make()
    audio = [fake_engine.tmp / "a.wav"]
    engine.align(audio, "fr", "small")
    engine.align(audio, "fr", "small")
    assert fake_engine.counted() == {"small": 2}
    assert engine._session is None


def test_results_stream_while_the_session_runs(fake_engine) -> None:
    engine = fake_engine.make()
    audio = [fake_engine.tmp / f"{i}.wav" for i in range(4)]
    seen: list[int] = []
    progress: list[int] = []
    engine.align(
        audio,
        "fr",
        "small",
        on_result=lambda index, _alignment: seen.append(index),
        progress=progress.append,
    )
    assert seen == [0, 1, 2, 3]
    assert progress[-1] == 4


def test_crash_falls_back_to_one_process_per_batch(fake_engine) -> None:
    engine = fake_engine.make()
    outs = [fake_engine.tmp / f"{i}.wav" for i in range(4)]
    texts = ["one", "two", "CRASH", "four"]
    done: list[int] = []
    with engine.session():
        results = engine._run(
            [_tts_op(text, out) for text, out in zip(texts, outs, strict=True)],
            outs,
            progress=done.append,
        )

    assert [item["ok"] for item in results] == [True] * 4
    assert [out.read_text(encoding="utf-8") for out in outs] == ["one", "two", "CRASH", "four"]
    assert engine.persistent is False
    assert engine.fallbacks and "persistent voice worker stopped" in engine.fallbacks[0]
    assert done[-1] == 4
    # later calls keep using the per-batch mode without trying the session again
    again = engine._run([_tts_op("five", outs[0])], outs[:1])
    assert again[0]["ok"] is True


def test_missing_serve_support_falls_back(fake_engine) -> None:
    engine = fake_engine.make()
    old = fake_engine.tmp / "old_worker.py"
    old.write_text(
        "import sys, json\n"
        "jobs = json.load(open(sys.argv[1]))\n"
        "json.dump({'results': [{'ok': True, 'out': 'x'} for _ in jobs['ops']]}, "
        "open(sys.argv[2], 'w'))\n",
        encoding="utf-8",
    )
    engine.worker = old
    results = engine._run([_tts_op("a", fake_engine.tmp / "a.wav")], [fake_engine.tmp / "a.wav"])
    assert results[0]["ok"] is True
    assert engine.fallbacks


def test_timeout_kills_the_persistent_worker(fake_engine) -> None:
    engine = fake_engine.make(timeout=2.0)
    out = fake_engine.tmp / "slow.wav"
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        with engine.session():
            engine._run([_tts_op("SLOW", out)], [out])
    assert time.monotonic() - started < 30
    assert engine._session is None


def test_persistent_mode_can_be_switched_off(fake_engine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADANVIL_VOICE_PERSISTENT", "0")
    engine = fake_engine.make()
    audio = [fake_engine.tmp / "a.wav"]
    with engine.session():
        engine.align(audio, "fr", "small")
        engine.align(audio, "fr", "small")
    assert engine.persistent is False
    assert fake_engine.counted() == {"small": 2}


def test_serve_loop_answers_requests_and_keeps_models() -> None:
    seen: list[dict] = []

    def fake(op: dict, models: dict) -> dict:
        models["n"] = int(models.get("n", 0)) + 1
        seen.append(op)
        return {"ok": True, "calls": models["n"]}

    requests = [
        {"id": 1, "op": {"op": "tts"}},
        {"id": 2, "op": {"op": "nope"}},
        {"id": 3, "op": {"op": "align"}},
        {"cmd": "quit"},
        {"id": 4, "op": {"op": "align"}},
    ]
    source = io.StringIO("\n".join(json.dumps(r) for r in requests) + "\nnot json\n")
    sink = io.StringIO()
    original = (worker._tts, worker._align)
    worker._tts = worker._align = fake  # type: ignore[assignment]
    try:
        assert worker.serve(source, sink) == 0
    finally:
        worker._tts, worker._align = original
    lines = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert lines[0] == {"ready": True}
    assert [line["id"] for line in lines[1:]] == [1, 2, 3]
    assert lines[1]["result"]["calls"] == 1
    assert lines[2]["result"]["ok"] is False  # unknown operation
    assert lines[3]["result"]["calls"] == 2  # the model state survived between requests


def test_real_worker_serves_over_stdin_and_stdout() -> None:
    script = Path(worker.__file__)
    process = subprocess.Popen(
        [sys.executable, str(script), "--serve"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, _ = process.communicate(
        json.dumps({"id": 7, "op": {"op": "nope"}}) + "\n" + json.dumps({"cmd": "quit"}) + "\n",
        timeout=60,
    )
    lines = [json.loads(line) for line in out.splitlines()]
    assert lines[0] == {"ready": True}
    assert lines[1]["id"] == 7 and lines[1]["result"]["ok"] is False
    assert process.returncode == 0


def test_tts_prepares_the_reference_voice_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared: list[tuple[str, float]] = []
    generated: list[dict] = []

    class FakeTTS:
        sr = 24000

        @classmethod
        def from_pretrained(cls, device: str) -> FakeTTS:
            return cls()

        def prepare_conditionals(self, path: str, exaggeration: float = 0.5) -> None:
            prepared.append((path, exaggeration))

        def generate(self, text: str, **kwargs: object) -> str:
            generated.append({"text": text, **kwargs})
            return "wav"

    torch = ModuleType("torch")
    torch.set_num_threads = lambda _n: None  # type: ignore[attr-defined]
    torch.manual_seed = lambda _n: None  # type: ignore[attr-defined]
    torchaudio = ModuleType("torchaudio")
    torchaudio.save = lambda path, _wav, _sr: Path(path).write_bytes(b"x")  # type: ignore[attr-defined]
    chatterbox = ModuleType("chatterbox")
    mtl = ModuleType("chatterbox.mtl_tts")
    mtl.ChatterboxMultilingualTTS = FakeTTS  # type: ignore[attr-defined]
    for name, module in {
        "torch": torch,
        "torchaudio": torchaudio,
        "chatterbox": chatterbox,
        "chatterbox.mtl_tts": mtl,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    reference = tmp_path / "voice.wav"
    reference.write_bytes(b"ref")
    models: dict = {}
    for text in ("un", "deux", "trois"):
        op = _tts_op(text, tmp_path / f"{text}.wav")
        op["reference"] = str(reference)
        assert worker._tts(op, models)["ok"] is True
    assert len(prepared) == 1
    assert all("audio_prompt_path" not in call for call in generated)
    assert [call["text"] for call in generated] == ["un", "deux", "trois"]

    reference.write_bytes(b"another reference voice")  # a changed file is analysed again
    op = _tts_op("quatre", tmp_path / "quatre.wav")
    op["reference"] = str(reference)
    worker._tts(op, models)
    assert len(prepared) == 2


def test_threads_follow_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADANVIL_VOICE_THREADS", "3")
    assert worker._cpu_threads() == 3
    monkeypatch.setenv("ADANVIL_VOICE_THREADS", "nonsense")
    assert worker._cpu_threads() >= 1


# --- the pronunciation check -----------------------------------------------------------


def _run(tmp_path: Path, lines: list[str], engine: FakeEngine, **spec: object):
    return generate_voice(
        tmp_path,
        VoiceSpec(engine="fake", language="fr", lines=lines, **spec),  # type: ignore[arg-type]
        "v",
        engine=engine,
    )


def test_confident_alignment_replaces_the_medium_check(tmp_path: Path) -> None:
    engine = FakeEngine()
    track = _run(tmp_path, ["Bonjour tout le monde", "Une autre phrase claire"], engine)

    assert engine.align_models == ["small"]  # no medium pass
    assert [line.status for line in track.lines] == ["ok", "ok"]
    for line in track.lines:
        assert line.transcript
        first, last = line.words[0], line.words[-1]
        # the words were moved onto the trimmed file: they start near the lead, inside it
        assert 0.0 <= first[0] <= 0.2
        assert last[1] <= line.duration + 1e-6


def test_low_confidence_keeps_the_medium_check(tmp_path: Path) -> None:
    engine = FakeEngine()
    engine.confidence = 0.3
    _run(tmp_path, ["Bonjour tout le monde"], engine)
    assert engine.align_models == ["small", "medium"]


def test_check_confidence_can_be_switched_off_by_spec_or_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = FakeEngine()
    _run(tmp_path / "a", ["Bonjour tout le monde"], engine, check_confidence=None)
    assert engine.align_models == ["small", "medium"]

    monkeypatch.setenv("ADANVIL_VOICE_CHECK_CONFIDENCE", "off")
    other = FakeEngine()
    _run(tmp_path / "b", ["Bonjour tout le monde"], other)
    assert other.align_models == ["small", "medium"]

    monkeypatch.setenv("ADANVIL_VOICE_CHECK_CONFIDENCE", "0.2")
    strict = FakeEngine()
    strict.confidence = 0.3
    _run(tmp_path / "c", ["Bonjour tout le monde"], strict)
    assert strict.align_models == ["small"]


def test_flagged_line_goes_to_the_medium_check_and_the_rest_do_not(tmp_path: Path) -> None:
    engine = FakeEngine()
    track = _run(tmp_path, ["Une phrase correcte", "Bonjour [mispronounce] monde"], engine)

    # attempt 1: small for both, medium only for the line that does not read as written
    assert engine.align_models[:3] == ["small", "medium", "small"]
    assert track.lines[0].status == "ok" and track.lines[0].attempts == 1
    assert track.lines[1].status == "suspect" and track.lines[1].attempts == 3


def test_same_model_shares_one_transcription(tmp_path: Path) -> None:
    engine = FakeEngine()
    track = _run(
        tmp_path,
        ["Bonjour [mispronounce] monde"],
        engine,
        align_model="small",
        check_model="small",
    )
    # three attempts, one transcription each: the check reuses the alignment's
    assert engine.align_models == ["small", "small", "small"]
    assert track.lines[0].status == "suspect"


def test_the_engine_session_wraps_the_whole_generation(tmp_path: Path) -> None:
    events: list[str] = []

    class Spy(FakeEngine):
        def session(self):
            from contextlib import contextmanager

            @contextmanager
            def manager():
                events.append("open")
                try:
                    yield self
                finally:
                    events.append("close")

            return manager()

        def synthesize(self, lines, *, progress=None):
            events.append("tts")
            return super().synthesize(lines, progress=progress)

    _run(tmp_path, ["Bonjour tout le monde"], Spy())
    assert events == ["open", "tts", "close"]


# --- the kokoro engine -----------------------------------------------------------------


def test_kokoro_spec_rules(tmp_path: Path) -> None:
    assert VoiceSpec(engine="kokoro", lines=["Bonjour"]).voice is None
    assert VoiceSpec(engine="kokoro", voice="ff_siwis", lines=["Bonjour"]).voice == "ff_siwis"
    with pytest.raises(ValueError, match="cannot clone"):
        VoiceSpec(engine="kokoro", reference="voices/a.mp3", lines=["Bonjour"])
    with pytest.raises(ValueError, match="only used by the kokoro"):
        VoiceSpec(engine="chatterbox", reference="a.mp3", voice="ff_siwis", lines=["Bonjour"])
    with pytest.raises(ValueError, match=r"French \(fr\) or English \(en\)"):
        generate_voice(
            tmp_path,
            VoiceSpec(engine="kokoro", language="de", lines=["Hallo"]),
            "v",
            engine=FakeEngine(),
        )


def test_kokoro_makes_one_attempt_because_a_retry_would_sound_the_same(tmp_path: Path) -> None:
    engine = FakeEngine()
    track = generate_voice(
        tmp_path,
        VoiceSpec(engine="kokoro", language="fr", lines=["Bonjour [mispronounce] monde"]),
        "v",
        engine=engine,
    )
    assert engine.synthesis_calls == 1
    assert track.engine == "kokoro"
    assert track.lines[0].attempts == 1


def test_kokoro_worker_engine_sends_voice_and_needs_no_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from adanvil.voice.engines import LineJob, get_engine

    engine = get_engine("kokoro", project_dir=tmp_path)
    assert isinstance(engine, WorkerEngine) and engine.engine == "kokoro"
    captured: list[dict] = []

    def fake_run(ops: list[dict], paths: list[Path], **_kw: object) -> list[dict]:
        captured.extend(ops)
        return [{"ok": True, "out": op["out"]} for op in ops]

    monkeypatch.setattr(engine, "_run", fake_run)
    out = tmp_path / "a.wav"
    engine.synthesize([LineJob(index=0, text="Salut", language="fr", out=out, voice="ff_siwis")])
    assert captured == [
        {
            "op": "tts",
            "engine": "kokoro",
            "text": "Salut",
            "language": "fr",
            "voice": "ff_siwis",
            "out": str(out),
        }
    ]


def test_worker_kokoro_writes_a_wav_and_picks_default_voices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import numpy

    built: list[str] = []
    voices: list[str] = []

    class FakePipeline:
        def __init__(self, lang_code: str, repo_id: str) -> None:
            built.append(lang_code)

        def __call__(self, text: str, voice: str):
            voices.append(voice)
            yield "g", "p", numpy.full(2400, 0.25, dtype="float32")
            yield "g", "p", numpy.full(2400, -0.25, dtype="float32")

    torch = ModuleType("torch")
    torch.set_num_threads = lambda _n: None  # type: ignore[attr-defined]
    kokoro = ModuleType("kokoro")
    kokoro.KPipeline = FakePipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "kokoro", kokoro)

    models: dict = {}
    out = tmp_path / "k.wav"
    result = worker._tts(
        {"engine": "kokoro", "text": "Bonjour", "language": "fr", "out": str(out)}, models
    )
    assert result == {"ok": True, "out": str(out), "sample_rate": 24000}
    import wave

    with wave.open(str(out), "rb") as source:
        assert (source.getframerate(), source.getnchannels(), source.getnframes()) == (
            24000,
            1,
            4800,
        )
    worker._tts({"engine": "kokoro", "text": "Hello", "language": "en", "out": str(out)}, models)
    worker._tts(
        {
            "engine": "kokoro",
            "text": "Hi",
            "language": "en-GB",
            "voice": "bf_emma",
            "out": str(out),
        },
        models,
    )
    worker._tts({"engine": "kokoro", "text": "Again", "language": "fr", "out": str(out)}, models)
    assert voices == ["ff_siwis", "af_heart", "bf_emma", "ff_siwis"]
    assert built == ["f", "a", "b"]  # one pipeline per language code, built once
    refused = worker._tts(
        {"engine": "kokoro", "text": "Hallo", "language": "de", "out": str(out)}, models
    )
    assert refused["ok"] is False and "French (fr) or English (en)" in refused["error"]


def test_worker_kokoro_missing_install_explains_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "kokoro", None)  # makes ``import kokoro`` fail
    result = worker._tts({"engine": "kokoro", "text": "a", "language": "fr", "out": "x.wav"}, {})
    assert result["ok"] is False
    assert "voice-setup --engine kokoro" in result["error"]


def test_voice_setup_plan_for_kokoro() -> None:
    from adanvil.voice.setup import voice_setup

    rendered = " ".join(" ".join(command) for command in voice_setup(plan=True, engine="kokoro"))
    assert "kokoro" in rendered and "faster-whisper" in rendered
    assert "chatterbox" not in rendered
    with pytest.raises(ValueError, match="engine must be one of"):
        voice_setup(plan=True, engine="nope")
