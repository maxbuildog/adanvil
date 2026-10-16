"""Standalone worker for the optional voice environment.

This file deliberately imports neither ``adanvil`` nor any project dependency.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import sys
from pathlib import Path


def _cpu_threads() -> int:
    """Threads for torch and whisper: ``ADANVIL_VOICE_THREADS`` or the usable cores."""
    try:
        override = int(os.environ.get("ADANVIL_VOICE_THREADS", ""))
    except ValueError:
        override = 0
    if override >= 1:
        return override
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # Windows, macOS
        return max(1, os.cpu_count() or 1)


def _tune_threads() -> int:
    """Give the math libraries the same thread count as torch and whisper (before torch loads)."""
    threads = _cpu_threads()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, str(threads))
    return threads


def _missing(exc: BaseException) -> str:
    return (
        "Voice worker dependencies are unavailable: "
        f"{exc}. Install them in the voice environment with "
        "pip install chatterbox-tts faster-whisper"
    )


_KOKORO_CODES = {"fr": "f", "en": "a"}  # kokoro language codes ("a" American, "b" British)
_KOKORO_VOICES = {"fr": "ff_siwis", "en": "af_heart"}
_KOKORO_RATE = 24_000
# Kokoro loads a voice ending in ".pt" as a local file and a comma as a blend of voices: only a
# plain voice name such as af_heart or ff_siwis is accepted (the parent checks it too).
_KOKORO_VOICE = re.compile(r"[a-z]{2}_[a-z0-9]+")

# Model weights are fetched at fixed revisions of their repositories (the ones the models were
# tested with), not from the moving branch.
KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
CHATTERBOX_REPO = "ResembleAI/chatterbox"
CHATTERBOX_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
CHATTERBOX_VERSION = "0.1.7"  # the library whose file list is below
# The files ``ChatterboxMultilingualTTS.from_pretrained`` of chatterbox-tts 0.1.7 downloads.
_CHATTERBOX_FILES = [
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
]
# Whisper model names that may be asked for, and the revisions of the Systran repositories that
# were tested (the same table is in ``adanvil.speech.transcribe``: a test keeps them equal). The
# operator's own choice (``ADANVIL_SPEECH_MODEL``, ``ADANVIL_VOICE_CHECK_MODEL``) stays free.
WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo")
WHISPER_REVISIONS = {
    "base": "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66",
    "small": "536b0662742c02347bc0e980a01041f333bce120",
    "medium": "08e178d48790749d25932bbc082711ddcfdfbc4f",
}
_OPERATOR_MODEL_VARIABLES = ("ADANVIL_SPEECH_MODEL", "ADANVIL_VOICE_CHECK_MODEL")


def _model_allowed(name: str) -> bool:
    if name in WHISPER_MODELS:
        return True
    chosen = {os.environ.get(variable, "").strip() for variable in _OPERATOR_MODEL_VARIABLES}
    return bool(name) and name in chosen


def _accepts(function: object, name: str) -> bool:
    """Whether ``function`` takes the keyword ``name`` (or any keyword)."""
    try:
        parameters = inspect.signature(function).parameters  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _library_version(distribution: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _kokoro_pinned(models: dict, code: str, voice: str):
    """``(pipeline, voice tensor)`` made from the pinned revision of the Kokoro repository, or
    None when the installed libraries cannot do it (the caller then lets Kokoro load the voice
    by name, which it takes from the repository's moving branch).

    Kokoro 0.9 can load its model from local files (``KModel(config=..., model=...)``) and takes
    a voice as a tensor, so the snapshot of one revision is fetched here and loaded from disk.
    """
    try:
        import torch
        from huggingface_hub import snapshot_download
        from kokoro import KModel, KPipeline
    except ImportError:
        return None
    if "kokoro-model" not in models:
        folder = Path(
            snapshot_download(
                KOKORO_REPO,
                revision=KOKORO_REVISION,
                allow_patterns=["config.json", "kokoro-v1_0.pth"],
            )
        )
        models["kokoro-model"] = KModel(
            repo_id=KOKORO_REPO,
            config=str(folder / "config.json"),
            model=str(folder / "kokoro-v1_0.pth"),
        ).eval()
    key = f"kokoro-pinned-{code}"
    if key not in models:
        models[key] = KPipeline(lang_code=code, repo_id=KOKORO_REPO, model=models["kokoro-model"])
    voice_key = f"kokoro-voice-{voice}"
    if voice_key not in models:
        folder = Path(
            snapshot_download(
                KOKORO_REPO, revision=KOKORO_REVISION, allow_patterns=[f"voices/{voice}.pt"]
            )
        )
        path = folder / "voices" / f"{voice}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Kokoro has no voice named {voice!r}")
        models[voice_key] = torch.load(path, weights_only=True)
    return models[key], models[voice_key]


def _tts_kokoro(op: dict, models: dict) -> dict:
    """Fixed-voice synthesis with Kokoro (Apache-2.0): no reference, no sampling settings."""
    try:
        import numpy
        from kokoro import KPipeline
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        return {
            "ok": False,
            "error": "Kokoro is unavailable: "
            f"{exc}. Install it in the voice environment with adanvil voice-setup --engine kokoro",
        }
    try:
        language = str(op["language"]).lower().split("-")[0]
        if language not in _KOKORO_CODES:
            return {
                "ok": False,
                "error": "The kokoro voice engine speaks French (fr) or English (en)",
            }
        voice = str(op.get("voice") or _KOKORO_VOICES[language])
        if not _KOKORO_VOICE.fullmatch(voice):
            return {
                "ok": False,
                "error": f"{voice!r} is not a Kokoro voice name such as af_heart or ff_siwis",
            }
        code = voice[0] if language == "en" and voice[:1] in {"a", "b"} else _KOKORO_CODES[language]
        if "kokoro-threads" not in models:
            import torch

            torch.set_num_threads(_cpu_threads())
            models["kokoro-threads"] = True
        pinned = _kokoro_pinned(models, code, voice)
        if pinned is not None:
            pipeline, voice_argument = pinned
        else:
            key = f"kokoro-{code}"
            if key not in models:
                models[key] = KPipeline(lang_code=code, repo_id=KOKORO_REPO)
            pipeline, voice_argument = models[key], voice
        pieces = []
        for _graphemes, _phonemes, audio in pipeline(str(op["text"]), voice=voice_argument):
            if audio is not None:
                pieces.append(numpy.asarray(audio, dtype="float32").reshape(-1))
        if not pieces:
            return {"ok": False, "error": "Kokoro produced no audio for this line"}
        samples = numpy.clip(numpy.concatenate(pieces), -1.0, 1.0)
        output = Path(str(op["out"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        import wave

        with wave.open(str(output), "wb") as destination:
            destination.setnchannels(1)
            destination.setsampwidth(2)
            destination.setframerate(_KOKORO_RATE)
            destination.writeframes((samples * 32767).astype("<i2").tobytes())
        return {"ok": True, "out": str(output), "sample_rate": _KOKORO_RATE}
    except Exception as exc:  # the parent receives the useful operation error
        return {"ok": False, "error": f"TTS operation failed: {exc}"}


def _chatterbox_model(model_class):
    """The multilingual model, loaded from the pinned revision of its repository.

    ``from_pretrained`` of chatterbox-tts 0.1.7 downloads the branch ``main``; ``from_local``
    loads a folder, so the snapshot of one revision is fetched here and loaded from disk. With
    another version of the library (other file names) or without ``from_local``, the library
    loads the model as it always did.
    """
    if not hasattr(model_class, "from_local") or (
        _library_version("chatterbox-tts") != CHATTERBOX_VERSION
    ):
        return model_class.from_pretrained(device="cpu")
    from huggingface_hub import snapshot_download

    folder = snapshot_download(
        repo_id=CHATTERBOX_REPO,
        repo_type="model",
        revision=CHATTERBOX_REVISION,
        allow_patterns=_CHATTERBOX_FILES,
        token=os.getenv("HF_TOKEN"),
    )
    return model_class.from_local(folder, "cpu")


def _tts(op: dict, models: dict) -> dict:
    if op.get("engine") == "kokoro":
        return _tts_kokoro(op, models)
    try:
        import torch
        import torchaudio
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        return {"ok": False, "error": _missing(exc)}

    try:
        torch.set_num_threads(_cpu_threads())
        if "tts" not in models:
            models["tts"] = _chatterbox_model(ChatterboxMultilingualTTS)
        if op.get("seed") is not None:
            torch.manual_seed(int(op["seed"]))
        model = models["tts"]
        # The reference voice is analysed once (it costs several seconds per line otherwise);
        # ``generate`` then only adjusts the exaggeration, so the audio is the same.
        reference = str(op["reference"])
        stamp = _file_stamp(reference)
        if models.get("tts_reference") != (reference, stamp):
            model.prepare_conditionals(reference, exaggeration=float(op.get("exaggeration", 0.5)))
            models["tts_reference"] = (reference, stamp)
        wav = model.generate(
            str(op["text"]),
            language_id=str(op["language"]),
            exaggeration=float(op.get("exaggeration", 0.5)),
            cfg_weight=float(op.get("cfg_weight", 0.5)),
            temperature=float(op.get("temperature", 0.7)),
        )
        output = Path(str(op["out"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output), wav, int(model.sr))
        return {"ok": True, "out": str(output), "sample_rate": int(model.sr)}
    except Exception as exc:  # the parent receives the useful operation error
        return {"ok": False, "error": f"TTS operation failed: {exc}"}


def _file_stamp(path: str) -> tuple[int, int] | None:
    try:
        info = os.stat(path)
    except OSError:
        return None
    return info.st_size, info.st_mtime_ns


def _align(op: dict, models: dict) -> dict:
    try:
        from faster_whisper import WhisperModel
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        return {"ok": False, "error": _missing(exc)}

    try:
        model_name = str(op.get("model", "small"))
        if not _model_allowed(model_name):
            return {
                "ok": False,
                "error": f"Whisper model {model_name!r} is not allowed: use one of "
                + ", ".join(WHISPER_MODELS),
            }
        if model_name not in models:
            load: dict = {"device": "cpu", "compute_type": "int8", "cpu_threads": _cpu_threads()}
            if model_name in WHISPER_REVISIONS and _accepts(WhisperModel, "revision"):
                load["revision"] = WHISPER_REVISIONS[model_name]
            models[model_name] = WhisperModel(model_name, **load)
        language = op.get("language") or None
        options: dict = {"language": language, "word_timestamps": True}
        if op.get("vad"):
            options["vad_filter"] = True  # speech rushes: drop room noise and silence
        if op.get("prompt"):
            options["initial_prompt"] = str(op["prompt"])
        segments, info = models[model_name].transcribe(_load_16k(str(op["audio"])), **options)
        words = []
        text_parts = []
        for segment in segments:
            if getattr(segment, "text", ""):
                text_parts.append(str(segment.text).strip())
            for word in getattr(segment, "words", None) or []:
                words.append(
                    [
                        float(word.start),
                        float(word.end),
                        str(word.word).strip(),
                        float(getattr(word, "probability", 0.0) or 0.0),
                    ]
                )
        return {
            "ok": True,
            "audio": str(op["audio"]),
            "words": words,
            "text": " ".join(text_parts),
            "language": getattr(info, "language", None),
        }
    except Exception as exc:
        return {"ok": False, "error": f"Alignment operation failed: {exc}"}


def _load_16k(path: str):
    """Decode to 16 kHz mono float32 without PyAV.

    faster-whisper decodes files with PyAV, whose newer releases broke its call
    (``open() got an unexpected keyword argument 'metadata_errors'``). librosa ships
    with the TTS stack, so decode here and pass the samples.
    """
    try:
        import librosa

        samples, _ = librosa.load(path, sr=16000, mono=True)
        return samples
    except ImportError:
        return path


def _save(path: Path, document: dict) -> None:
    """Write a JSON document so a reader never sees half of it (temp file, then replace)."""

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    try:
        os.replace(temporary, path)
    except OSError:  # a reader holds the file for a moment (Windows): write in place instead
        path.write_text(temporary.read_text(encoding="utf-8"), encoding="utf-8")
        temporary.unlink(missing_ok=True)


def run_jobs(jobs: dict, progress: Path | None = None) -> dict:
    """Run a JSON job document and return its JSON result document.

    With ``progress``, the results written so far are saved after every operation, so
    a caller that times out on a slow CPU still recovers the lines already made, and a caller
    that reads the file while the worker runs gets each result as soon as it exists.
    """
    _tune_threads()
    results = []
    models: dict[str, object] = {}
    for op in jobs.get("ops", []):
        results.append(run_op(op, models))
        if progress is not None:
            _save(progress, {"results": results})
    return {"results": results}


def run_op(op: dict, models: dict) -> dict:
    """Run one operation with the models already loaded in ``models``."""
    kind = op.get("op")
    if kind == "tts":
        return _tts(op, models)
    if kind == "align":
        return _align(op, models)
    return {"ok": False, "error": f"Unknown voice worker operation: {kind!r}"}


def serve(source, sink) -> int:
    """Persistent mode: one JSON request per line in, one JSON reply per line out.

    ``{"id": 3, "op": {...}}`` is answered by ``{"id": 3, "result": {...}}`` once the operation
    is done, and the models stay loaded for the next request. ``{"cmd": "quit"}`` or the end of
    the input stops the worker. A ``{"ready": true}`` line is written first.
    """

    def send(document: dict) -> None:
        sink.write(json.dumps(document, ensure_ascii=False) + "\n")
        sink.flush()

    models: dict[str, object] = {}
    send({"ready": True})
    for line in source:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError as exc:
            send({"id": None, "result": {"ok": False, "error": f"Invalid request: {exc}"}})
            continue
        if request.get("cmd") == "quit":
            break
        try:
            result = run_op(request.get("op") or {}, models)
        except Exception as exc:  # a bug in one operation must not end the session
            result = {"ok": False, "error": f"Voice worker operation crashed: {exc}"}
        send({"id": request.get("id"), "result": result})
    return 0


def main_serve() -> int:
    """Serve on the real stdin/stdout, keeping library prints off the protocol channel."""
    _tune_threads()
    sink = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)  # anything a library prints goes to the log, not into the protocol
    sys.stdout = sys.stderr
    return serve(sys.stdin, sink)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--serve"]:
        return main_serve()
    if len(argv) != 2:
        print("usage: python worker.py JOBS.json RESULTS.json | --serve", file=sys.stderr)
        return 2
    try:
        jobs = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
        result = run_jobs(jobs, progress=Path(argv[1]))
        Path(argv[1]).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return 0
    except Exception as exc:
        Path(argv[1]).write_text(
            json.dumps({"results": [{"ok": False, "error": str(exc)}]}), encoding="utf-8"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
