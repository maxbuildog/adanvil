"""Transcribe the speech of clips with word timestamps, without ever failing hard.

The transcriber is found the same way as for reference decoding: faster-whisper in this
process when it is installed, else the voice environment worker (``adanvil voice-setup``).
When neither exists, every clip gets a clear ``unavailable`` status instead of an error.
A batch is one model load: in-process the model is built once for the whole batch, and the
worker receives all the files in a single run and reports each file as soon as it is done
(``on_result``), so a caller can store the speech of a file while the next one is running.
"""

from __future__ import annotations

import inspect
import math
import os
import shutil
import uuid
import wave
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from adanvil import config, ffmpeg
from adanvil.voice.engines import VoiceUnavailableError, WorkerEngine

from .model import SpeechWord

DONE = "done"
UNAVAILABLE = "unavailable: pip install 'adanvil[transcribe]' or run adanvil voice-setup"
NO_AUDIO = "not attempted: no audio"
SILENT = "silent_audio"  # the audio stream exists but carries no sound: nothing to transcribe

DEFAULT_MODEL = "base"  # indexing: fast and good enough to search what a clip says
REFINE_MODEL = "medium"  # captions: ``speech_refine`` on the clips that are used
# The model names an agent may ask for. The operator's own choice, ``ADANVIL_SPEECH_MODEL``,
# stays free (a local model folder, a distilled model...).
WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo")
# Revisions of the Systran repositories that were tested; the other models follow their branch.
# The voice worker holds the same table (``adanvil.voice.worker``): a test keeps them equal.
WHISPER_REVISIONS = {
    "base": "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66",
    "small": "536b0662742c02347bc0e980a01041f333bce120",
    "medium": "08e178d48790749d25932bbc082711ddcfdfbc4f",
}
LEVEL_STEP = 0.1  # seconds per level value kept in the sidecar
_SAMPLE_RATE = 16_000
_SILENCE_DB = -60.0  # every one-second window under this: the clip is silent

# Hesitations are dropped by most transcribers; a prompt written like speech with fillers
# makes them keep them. Off unless ADANVIL_SPEECH_PROMPT carries the text.
_OFF_VALUES = {"off", "0", "false", "no"}
_PROMPT_LIMIT = 400  # characters: whisper only reads the last ~224 tokens of a prompt


@dataclass
class SpeechOutcome:
    """The result of transcribing one clip."""

    status: str
    language: str | None = None
    words: list[SpeechWord] = field(default_factory=list)
    levels_db: list[float] = field(default_factory=list)
    duration: float | None = None

    @property
    def done(self) -> bool:
        return self.status == DONE


def speech_model() -> str:
    """The indexing model: ``ADANVIL_SPEECH_MODEL`` or ``base``."""

    return os.environ.get("ADANVIL_SPEECH_MODEL", "").strip() or DEFAULT_MODEL


def check_model_name(name: str) -> str:
    """``name`` when a transcription may load that model, else a ``ValueError`` listing the
    names that are allowed. The model of ``ADANVIL_SPEECH_MODEL`` is the operator's and passes."""

    operator = os.environ.get("ADANVIL_SPEECH_MODEL", "").strip()
    if name in WHISPER_MODELS or (name and name == operator):
        return name
    raise ValueError(
        f"the transcription model {name!r} is not allowed: use one of {', '.join(WHISPER_MODELS)}"
    )


def auto_enabled() -> bool:
    """Whether ``transcribe="auto"`` may transcribe (``ADANVIL_SPEECH=off`` turns it off)."""

    return os.environ.get("ADANVIL_SPEECH", "").strip().casefold() not in _OFF_VALUES


def speech_vocabulary(root: Path) -> list[str]:
    """The words to spell right: brand name, product name, then ``speech.vocabulary``.

    Read from the brand profile of the project (nothing when it has none or cannot be read),
    without duplicates, in that order.
    """

    from adanvil.brand import load_brand
    from adanvil.project import BRAND_FILE

    path = Path(root) / BRAND_FILE
    if not path.is_file():
        return []
    try:
        brand = load_brand(path)
    except Exception:  # an unreadable brand profile must not stop the transcription
        return []
    candidates = [brand.name]
    if brand.product is not None:
        candidates.append(brand.product.name or "")
    if brand.speech is not None:
        candidates.extend(brand.speech.vocabulary or [])
    terms: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        term = " ".join(str(candidate).split())
        if term and term.casefold() not in seen:
            seen.add(term.casefold())
            terms.append(term)
    return terms


def speech_prompt(root: Path | None = None) -> str | None:
    """The initial prompt of the transcriber: ``ADANVIL_SPEECH_PROMPT`` plus the brand vocabulary.

    The vocabulary (see :func:`speech_vocabulary`) is given as a glossary sentence so the model
    spells the brand and product names, and the terms of the brand, the way they are written.
    """

    parts: list[str] = []
    configured = os.environ.get("ADANVIL_SPEECH_PROMPT", "").strip()
    if configured:
        parts.append(configured)
    terms = speech_vocabulary(root) if root is not None else []
    if terms:
        parts.append("Glossary: " + ", ".join(terms) + ".")
    prompt = " ".join(parts).strip()
    if len(prompt) > _PROMPT_LIMIT:  # keep the end: it is what the model weighs most
        prompt = prompt[-_PROMPT_LIMIT:].split(" ", 1)[-1]
    return prompt or None


def resolve_language(root: Path, language: str | None = None) -> str | None:
    """The language to transcribe in: the argument, else the brand profile's, else auto."""

    from adanvil.voice.generate import brand_language, language_code

    if language is not None and language.strip():
        return language_code(language)
    return brand_language(Path(root))


def _resolve_backend(project_root: Path, engine: Any | None) -> tuple[str, Any] | None:
    """``("engine", obj)``, ``("whisper", WhisperModel)``, ``("worker", WorkerEngine)`` or None."""

    if engine is not None:
        return "engine", engine
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        worker = WorkerEngine(project_dir=project_root)
        resolver = getattr(worker, "_python", None)
        if callable(resolver):
            try:
                resolver([])
            except VoiceUnavailableError:
                return None
        return "worker", worker
    return "whisper", WhisperModel


def backend_available(project_root: Path, engine: Any | None = None) -> bool:
    """Whether a transcriber can be used (cheap: it loads nothing and runs no ffmpeg)."""

    return _resolve_backend(Path(project_root), engine) is not None


def _extract_wav(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg.run(
        [
            "-i", str(source), "-map", "0:a:0", "-vn",
            "-ar", str(_SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(destination),
        ]
    )  # fmt: skip


def _levels(wav: Path) -> tuple[list[float], float]:
    """RMS level in dBFS per ``LEVEL_STEP`` and the duration (seconds) of a 16 kHz mono WAV."""

    with wave.open(str(wav), "rb") as handle:
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if rate <= 0 or samples.size == 0:
        return [], 0.0
    window = max(1, round(rate * LEVEL_STEP))
    count = math.ceil(samples.size / window)
    levels: list[float] = []
    for index in range(count):
        chunk = samples[index * window : (index + 1) * window]
        rms = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
        levels.append(20 * math.log10(rms) if rms > 1e-6 else -120.0)
    return levels, samples.size / rate


def _is_silent(levels: list[float]) -> bool:
    """True when no one-second window of the clip rises above the silence floor."""

    if not levels:
        return True
    per_second = max(1, round(1.0 / LEVEL_STEP))
    for first in range(0, len(levels), per_second):
        values = levels[first : first + per_second]
        power = sum(10 ** (value / 10) for value in values) / len(values)
        if power > 0 and 10 * math.log10(power) > _SILENCE_DB:
            return False
    return True


def _words(items: Any) -> list[SpeechWord]:
    words: list[SpeechWord] = []
    for item in items:
        start, end, text = float(item[0]), float(item[1]), str(item[2]).strip()
        probability = float(item[3]) if len(item) > 3 else 1.0
        if text and end >= start >= 0:
            words.append(SpeechWord(start, end, text, probability))
    words.sort(key=lambda word: (word.start, word.end))
    return words


def _whisper_words(segments: Any) -> list[SpeechWord]:
    items: list[tuple[float, float, str, float]] = []
    for segment in segments:
        for word in getattr(segment, "words", None) or []:
            text = getattr(word, "word", None)
            if text is None:
                text = getattr(word, "text", "")
            items.append(
                (
                    float(getattr(word, "start", getattr(segment, "start", 0.0))),
                    float(getattr(word, "end", getattr(segment, "end", 0.0))),
                    str(text),
                    float(getattr(word, "probability", 0.0) or 0.0),
                )
            )
    return _words(items)


def _accepts(function: Any, name: str) -> bool:
    """Whether ``function`` takes the keyword ``name`` (or any keyword)."""

    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _prepare(index: int, source: Path, scratch: Path) -> tuple[list[float], float, Path]:
    """Extract the audio of one clip as a 16 kHz mono WAV and measure its level."""

    wav = scratch / f"{index:04d}" / f"{Path(source).stem}.wav"
    _extract_wav(Path(source), wav)
    levels, duration = _levels(wav)
    return levels, duration, wav


def transcribe_clips(
    sources: list[Path],
    *,
    project_root: Path,
    language: str | None = None,
    model: str | None = None,
    engine: Any | None = None,
    skip_silent: bool = True,
    prompt: str | None = None,
    progress: Callable[[int, int], None] | None = None,
    on_result: Callable[[int, SpeechOutcome], None] | None = None,
) -> list[SpeechOutcome]:
    """Transcribe the first audio stream of each clip (video or audio file), in one batch.

    ``engine`` is any object with ``align(paths, language, model)`` like the voice engines
    (tests inject a scripted one); without it the transcriber is found as described in the
    module docs. With ``skip_silent`` a clip whose audio never rises above -60 dBFS is not
    transcribed and gets the status ``silent_audio``; a forced silent clip that yields no word
    gets the same status instead of a ``done`` with an invented language. ``prompt`` is the
    initial prompt of the transcriber: by default the brand vocabulary of the project
    (:func:`speech_prompt`). ``progress(done, total)`` counts the clips that reach the
    transcriber and ``on_result(index, outcome)`` fires as soon as a clip has its outcome (a
    clip skipped or failed at once, a transcribed one as the transcriber yields it, so a
    caller can store each file while the next one runs). Never raises for a missing
    transcriber or a failing clip: the outcome carries the status.
    """

    project_root = Path(project_root).resolve()
    model_name = check_model_name(model or speech_model())
    outcomes: list[SpeechOutcome | None] = [None] * len(sources)

    def finish(index: int, outcome: SpeechOutcome) -> None:
        outcomes[index] = outcome
        if on_result is not None:
            on_result(index, outcome)

    backend = _resolve_backend(project_root, engine)
    if backend is None:
        for index in range(len(sources)):
            finish(index, SpeechOutcome(UNAVAILABLE))
        return [outcome for outcome in outcomes if outcome is not None]

    scratch = project_root / "cache" / "speech" / f".tmp-{uuid.uuid4().hex}"
    try:
        wavs: dict[int, Path] = {}
        measured: dict[int, tuple[list[float], float]] = {}
        # The audio of every clip is extracted first (about a second each, in parallel), so the
        # transcriber then runs without interruption.
        workers = max(1, min(config.workers(), len(sources) or 1))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            prepared = [
                pool.submit(_prepare, index, source, scratch)
                for index, source in enumerate(sources)
            ]
            for index, future in enumerate(prepared):
                try:
                    levels, duration, wav = future.result()
                except Exception as exc:  # one unreadable clip must not stop the batch
                    finish(index, SpeechOutcome(f"failed: {exc}"))
                    continue
                if skip_silent and _is_silent(levels):
                    finish(index, SpeechOutcome(SILENT, levels_db=levels, duration=duration))
                    wav.unlink(missing_ok=True)
                    continue
                wavs[index] = wav
                measured[index] = (levels, duration)

        order = sorted(wavs)
        total = len(order)
        if progress is not None:
            progress(0, total)

        def result(index: int, words: list[SpeechWord], detected: str | None) -> None:
            levels, duration = measured[index]
            if not words and _is_silent(levels):
                finish(index, SpeechOutcome(SILENT, levels_db=levels, duration=duration))
                return
            finish(
                index,
                SpeechOutcome(
                    DONE,
                    language=(detected or language) if words else None,
                    words=words,
                    levels_db=levels,
                    duration=duration,
                ),
            )

        kind, handle = backend
        if prompt is None:
            prompt = speech_prompt(project_root)
        if kind == "whisper" and order:
            load: dict[str, Any] = {"device": "cpu", "compute_type": "int8"}
            if model_name in WHISPER_REVISIONS and _accepts(handle, "revision"):
                load["revision"] = WHISPER_REVISIONS[model_name]
            try:
                whisper = handle(model_name, **load)
            except Exception as exc:
                for index in order:
                    finish(index, SpeechOutcome(f"failed: {exc}"))
            else:
                for done, index in enumerate(order, start=1):
                    options: dict[str, Any] = {
                        "word_timestamps": True,
                        "language": language,
                        "vad_filter": True,
                    }
                    if prompt:
                        options["initial_prompt"] = prompt
                    try:
                        segments, info = whisper.transcribe(str(wavs[index]), **options)
                        words = _whisper_words(segments)
                        detected = getattr(info, "language", None)
                    except Exception as exc:
                        finish(index, SpeechOutcome(f"failed: {exc}"))
                    else:
                        result(index, words, detected if isinstance(detected, str) else None)
                    if progress is not None:
                        progress(done, total)
        elif order:
            paths = [wavs[index] for index in order]

            def settle(position: int, alignment: Any) -> None:
                index = order[position]
                if outcomes[index] is not None:
                    return  # already reported while the batch was running
                error = getattr(alignment, "error", None)
                if error:
                    finish(index, SpeechOutcome(f"failed: {error}"))
                    return
                result(index, _words(alignment.words), getattr(alignment, "language", None))

            try:
                if kind == "worker":
                    alignments = handle.align(
                        paths,
                        language,
                        model_name,
                        vad=True,
                        prompt=prompt,
                        strict=False,
                        progress=(lambda done: progress(done, total)) if progress else None,
                        on_result=settle,
                    )
                elif prompt and _accepts(handle.align, "prompt"):
                    alignments = handle.align(paths, language, model_name, prompt=prompt)
                else:
                    alignments = handle.align(paths, language, model_name)
            except Exception as exc:  # a worker that cannot start or times out
                status = UNAVAILABLE if isinstance(exc, VoiceUnavailableError) else f"failed: {exc}"
                for index in order:
                    if outcomes[index] is None:  # files finished before the failure are kept
                        finish(index, SpeechOutcome(status))
            else:
                for position, alignment in enumerate(alignments):
                    settle(position, alignment)
                if progress is not None:
                    progress(total, total)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return [
        outcome if outcome is not None else SpeechOutcome("failed: no result")
        for outcome in outcomes
    ]


__all__ = [
    "DONE",
    "NO_AUDIO",
    "REFINE_MODEL",
    "SILENT",
    "UNAVAILABLE",
    "WHISPER_MODELS",
    "WHISPER_REVISIONS",
    "SpeechOutcome",
    "auto_enabled",
    "backend_available",
    "check_model_name",
    "resolve_language",
    "speech_model",
    "speech_prompt",
    "speech_vocabulary",
    "transcribe_clips",
]
