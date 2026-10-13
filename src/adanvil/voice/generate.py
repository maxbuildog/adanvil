"""Sentence-by-sentence voice generation, checking, trimming and manifests."""

from __future__ import annotations

import contextlib
import difflib
import inspect
import os
import re
import shutil
import unicodedata
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from adanvil.brand import load_brand
from adanvil.ffmpeg import probe, run
from adanvil.media.cache import Cache
from adanvil.project import BRAND_FILE, resolve_in_project
from adanvil.text.animate import strip_markers

from .engines import Alignment, FilesEngine, LineJob, VoiceEngine, get_engine


class VoiceLineText(BaseModel):
    """A line whose spoken text (``say``) differs from the caption text (``show``)."""

    model_config = ConfigDict(extra="forbid")

    say: str = Field(min_length=1)
    show: str | None = None


def _say(item: str | VoiceLineText) -> str:
    """The text the voice speaks: ``*word*`` emphasis markers are for the captions only."""

    return strip_markers(item if isinstance(item, str) else item.say)[0]


def _show(item: str | VoiceLineText) -> str | None:
    """The caption text when it differs from the spoken one (a marked word counts)."""

    if isinstance(item, str):
        return item if strip_markers(item)[1] else None
    shown = item.show or (item.say if strip_markers(item.say)[1] else None)
    return shown if shown and shown != _say(item) else None


DEFAULT_LANGUAGE = "en"
_LANGUAGE_NAMES = {
    "english": "en",
    "french": "fr",
    "francais": "fr",
    "français": "fr",
    "spanish": "es",
    "espanol": "es",
    "español": "es",
    "german": "de",
    "deutsch": "de",
    "italian": "it",
    "italiano": "it",
    "portuguese": "pt",
    "portugues": "pt",
    "português": "pt",
    "dutch": "nl",
    "nederlands": "nl",
}


def language_code(value: str | None) -> str | None:
    """Reduce a language setting ("fr", "fr-FR", "French") to the code the TTS expects."""

    if not value or not value.strip():
        return None
    text = value.strip().casefold()
    if text in _LANGUAGE_NAMES:
        return _LANGUAGE_NAMES[text]
    return text.replace("_", "-").split("-")[0] or None


def brand_language(root: Path) -> str | None:
    """The language of the project's ``brand.yaml``, or ``None`` when it sets none."""

    path = Path(root) / BRAND_FILE
    if not path.is_file():
        return None
    try:
        return language_code(load_brand(path).language)
    except Exception:  # an unreadable brand profile must not stop a voice job
        return None


class VoiceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: Literal["chatterbox", "kokoro", "files", "fake"] = "chatterbox"
    reference: str | None = None
    # kokoro only: the fixed voice (default ``ff_siwis`` for French, ``af_heart`` for English).
    voice: str | None = None
    # Optional: without it the language of ``brand.yaml`` is used, else ``en``.
    language: str | None = None
    lines: list[str | VoiceLineText] = Field(min_length=1)
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.7
    check_model: Literal["small", "medium"] = "medium"
    # A line whose alignment transcript already passes, with every word at least this
    # confident, is not transcribed again by ``check_model`` (None: always run the check).
    check_confidence: float | None = Field(default=0.6, ge=0.0, le=1.0)
    align_model: Literal["small", "medium"] = "small"
    min_similarity: float = Field(default=0.85, ge=0.0, le=1.0)
    lead: float = Field(default=0.08, ge=0.0)
    tail: float = Field(default=0.22, ge=0.0)
    gap: float = Field(default=0.0, ge=0.0)
    files: dict[int, str] | None = None

    @model_validator(mode="after")
    def validate_engine_inputs(self) -> VoiceSpec:
        if self.engine == "chatterbox" and not self.reference:
            raise ValueError("reference is required for the chatterbox voice engine")
        if self.engine == "files" and not self.files:
            raise ValueError("files is required for the files voice engine")
        if self.engine == "kokoro" and self.reference:
            raise ValueError(
                "the kokoro voice engine has fixed voices and cannot clone a reference: "
                "use voice, or the chatterbox engine to clone a voice"
            )
        if self.voice and self.engine != "kokoro":
            raise ValueError("voice is only used by the kokoro voice engine")
        return self


class VoiceLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    text: str
    display: str | None = None
    file: str
    duration: float
    start: float
    words: list[tuple[float, float, str]]
    transcript: str
    similarity: float
    status: Literal["ok", "suspect"]
    attempts: int
    unmatched_words: list[str] = Field(default_factory=list)


class VoiceTrack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    language: str
    engine: str
    lines: list[VoiceLine]
    total_duration: float
    manifest: str
    warnings: list[str] = Field(default_factory=list)


@dataclass
class _BestLine:
    similarity: float = -1.0
    transcript: str = ""
    words: list[tuple[float, float, str]] | None = None
    path: Path | None = None
    attempts: int = 0
    unmatched: list[str] = field(default_factory=list)

    def passes(self, min_similarity: float) -> bool:
        return self.similarity >= min_similarity and len(self.unmatched) < _SUSPECT_UNMATCHED_WORDS


_NUMBER_WORDS = frozenset(
    """
    zero deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze
    quinze seize vingt vingts trente quarante cinquante soixante cent cents mille
    million millions milliard milliards
    two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen
    seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety
    hundred thousand billion
    euro euros eur dollar dollars usd
    """.split()
)
_WEAK_NUMBER_WORDS = frozenset({"un", "une", "one"})  # also articles: only inside numbers
_CURRENCY_SYMBOLS = str.maketrans({"€": " euro ", "$": " dollar "})


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _normalise(text: str) -> str:
    value = "".join(char for char in text.casefold() if char.isalnum() or char.isspace())
    return " ".join(value.split())


def _normalise_numbers(text: str) -> str:
    """Normalise text so spoken and digit forms of numbers and prices compare equal.

    Digits, decimal numbers, French and English number words, "et"/"and" and
    currency words are dropped: "soixante-cinq euros" and "65 euros" both become
    empty, while ordinary words are kept untouched.
    """

    value = _strip_accents(text.casefold()).translate(_CURRENCY_SYMBOLS)
    value = re.sub(r"[-‐-―]", " ", value)
    value = re.sub(r"(?<=\d)[.,](?=\d)", "", value)
    tokens = _normalise(value).split()
    numeric = [token.isdigit() or token in _NUMBER_WORDS for token in tokens]
    for index, token in enumerate(tokens):
        if token in _WEAK_NUMBER_WORDS:
            before = index > 0 and numeric[index - 1]
            if index > 1 and tokens[index - 1] in {"et", "and"} and numeric[index - 2]:
                before = True
            after = index + 1 < len(tokens) and numeric[index + 1]
            numeric[index] = numeric[index] or before or after
    kept: list[str] = []
    for index, token in enumerate(tokens):
        if numeric[index]:
            continue
        # "et"/"and" only disappear inside a number ("vingt et un", "one hundred and five").
        if token in {"et", "and"} and 0 < index < len(tokens) - 1:
            if numeric[index - 1] and numeric[index + 1]:
                continue
        kept.append(token)
    return " ".join(kept)


def _similarity(expected: str, actual: str) -> float:
    left, right = _normalise_numbers(expected), _normalise_numbers(actual)
    if not left and not right:
        return 1.0
    return difflib.SequenceMatcher(None, left, right).ratio()


_KEY_WORD_LETTERS = 4  # expected words at least this long must be heard
_KEY_WORD_RATIO = 0.8
_SUSPECT_UNMATCHED_WORDS = 2  # a line with this many unmatched words is suspect


def _word_heard(word: str, heard: list[str], *, fuzzy: bool = True) -> bool:
    """Whether ``word`` is in the transcript, allowing small spelling drift and merges."""

    candidates = [*heard, *(left + right for left, right in zip(heard, heard[1:], strict=False))]
    # Containment covers elisions and hyphen merges ("lapplication" for "application"),
    # so allow at most 2 extra letters: "brioche" must not count as "brio".
    return any(
        (word in candidate and len(candidate) - len(word) <= 2)
        or (fuzzy and difflib.SequenceMatcher(None, word, candidate).ratio() >= _KEY_WORD_RATIO)
        for candidate in candidates
    )


def _unmatched_words(expected: str, actual: str) -> list[str]:
    """Expected words of 4+ letters that the transcript does not contain.

    Both texts are normalised like ``_similarity`` (numbers and currency neutral). A word
    is heard when a transcript word has a character ratio of at least 0.8, or contains it
    with at most 2 extra letters (elisions such as ``lapplication``), alone or merged with
    the expected word next to it (``peutetre`` for ``peut-être``).
    """

    heard = _normalise_numbers(actual).split()
    words = _normalise_numbers(expected).split()
    missing: list[str] = []
    for index, word in enumerate(words):
        if len(word) < _KEY_WORD_LETTERS:
            continue
        merged = []
        if index > 0:
            merged.append(words[index - 1] + word)
        if index + 1 < len(words):
            merged.append(word + words[index + 1])
        # Merged forms count only when found whole: a fuzzy match on a long merged string
        # would accept "briochest" for "brioche est".
        if not _word_heard(word, heard) and not any(
            _word_heard(form, heard, fuzzy=False) for form in merged
        ):
            missing.append(word)
    return missing


def _project_path(root: Path, value: str, field: str) -> Path:
    try:
        return resolve_in_project(root, value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a project-relative path: {value!r}") from exc


def _cached_path(cache: Cache, key: dict[str, object]) -> Path:
    return cache.path_for("voice", key, suffix=".wav")


def _trim_line(raw: Path, output: Path, alignment: Alignment, lead: float, tail: float) -> float:
    """Cut ``raw`` around the words of ``alignment``; returns where the cut starts (seconds)."""

    info = probe(raw)
    duration = float(info.audio_duration or 0.0)
    if duration <= 0:
        raise ValueError(f"Voice source has no usable duration: {raw}")
    if alignment.words:
        first = float(alignment.words[0][0])
        last = float(alignment.words[-1][1])
        start = max(0.0, first - lead)
        end = min(duration, last + tail)
    else:
        start, end = 0.0, duration
    if end <= start:
        start, end = 0.0, duration
    trimmed_duration = end - start
    fade = min(0.1, trimmed_duration)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{output}.tmp.wav")
    temporary.unlink(missing_ok=True)
    run(
        [
            "-ss",
            f"{start:.9f}",
            "-i",
            str(raw),
            "-t",
            f"{trimmed_duration:.9f}",
            "-af",
            (
                "asetpts=PTS-STARTPTS,"
                f"afade=t=out:st={max(0.0, trimmed_duration - fade):.9f}:d={fade:.9f}"
            ),
            "-vn",
            "-c:a",
            "pcm_s16le",
            "-ar",
            "48000",
            "-ac",
            "1",
            str(temporary),
        ]
    )
    os.replace(temporary, output)
    return start


def _line_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as source:
        if source.getframerate() != 48_000 or source.getnchannels() != 1:
            raise ValueError(f"Voice line is not 48 kHz mono PCM: {path}")
        return source.getnframes() / source.getframerate()


def _write_full(paths: list[Path], gap: float, output: Path) -> float:
    output.parent.mkdir(parents=True, exist_ok=True)
    silence_frames = int(round(gap * 48_000))
    temporary = Path(f"{output}.tmp.wav")
    temporary.unlink(missing_ok=True)
    total_frames = 0
    with wave.open(str(temporary), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(48_000)
        for index, path in enumerate(paths):
            with wave.open(str(path), "rb") as source:
                if (
                    source.getframerate() != 48_000
                    or source.getnchannels() != 1
                    or source.getsampwidth() != 2
                ):
                    raise ValueError(f"Voice line is not 48 kHz mono PCM: {path}")
                data = source.readframes(source.getnframes())
                destination.writeframes(data)
                total_frames += len(data) // 2
            if index != len(paths) - 1 and silence_frames:
                destination.writeframes(b"\x00\x00" * silence_frames)
                total_frames += silence_frames
    os.replace(temporary, output)
    return total_frames / 48_000


def _relative(root: Path, path: Path) -> str:
    return Path(path).relative_to(Path(root)).as_posix()


def _load_cached_track(
    root: Path, directory: Path, spec: VoiceSpec, name: str
) -> VoiceTrack | None:
    manifest = directory / "manifest.json"
    full = directory / "voice_full.wav"
    if not manifest.is_file() or not full.is_file():
        return None
    try:
        track = VoiceTrack.model_validate_json(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if track.name != name or track.engine != spec.engine or track.language != spec.language:
        return None
    if [line.text for line in track.lines] != [_say(item) for item in spec.lines]:
        return None
    if [line.display for line in track.lines] != [_show(item) for item in spec.lines]:
        return None
    if not all((_project_path(root, line.file, "voice line")).is_file() for line in track.lines):
        return None
    return track


def _check_confidence(spec: VoiceSpec) -> float | None:
    """Word confidence above which the alignment transcript is the pronunciation check.

    ``ADANVIL_VOICE_CHECK_CONFIDENCE`` (a number from 0 to 1, or ``off``) applies unless the
    spec sets ``check_confidence`` itself.
    """

    if "check_confidence" in spec.model_fields_set:
        return spec.check_confidence
    value = os.environ.get("ADANVIL_VOICE_CHECK_CONFIDENCE", "").strip().lower()
    if not value:
        return spec.check_confidence
    if value in {"off", "none", "no", "false"}:
        return None
    try:
        return min(1.0, max(0.0, float(value)))
    except ValueError:
        return spec.check_confidence


def _shifted(alignment: Alignment, path: Path, offset: float) -> Alignment:
    """The alignment of an uncut file expressed on the trimmed file that starts at ``offset``."""

    words = [
        (max(0.0, float(w[0]) - offset), max(0.0, float(w[1]) - offset), str(w[2]), float(w[3]))
        for w in alignment.words
    ]
    return Alignment(path=path, words=words, text=alignment.text, language=alignment.language)


def _confident_enough(alignment: Alignment, expected: str, floor: float, minimum: float) -> bool:
    """Whether the alignment transcript alone shows the line was spoken as written."""

    if alignment.error or not alignment.words or not alignment.text:
        return False
    if min(float(word[3]) for word in alignment.words) < floor:
        return False
    return _similarity(expected, alignment.text) >= minimum and not _unmatched_words(
        expected, alignment.text
    )


def _synthesize_with_progress(
    engine: VoiceEngine,
    jobs: list[LineJob],
    progress: Callable[[int], None],
) -> None:
    """Pass progress to current engines while retaining old engine compatibility."""

    try:
        parameters = inspect.signature(engine.synthesize).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    supports_progress = any(
        parameter.name == "progress" or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if supports_progress:
        engine.synthesize(jobs, progress=progress)
    else:
        engine.synthesize(jobs)


def generate_voice(
    root: Path,
    spec: VoiceSpec,
    name: str,
    *,
    cache: Cache | None = None,
    engine: VoiceEngine | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> VoiceTrack:
    """Generate a checked, trimmed sentence track and its JSON manifest.

    ``progress`` receives dictionaries with ``stage``, ``done``, ``total``,
    ``unit`` and ``message``.  The callback is deliberately JSON-shaped so it
    can be persisted by the detached job runner.
    """
    if not isinstance(spec, VoiceSpec):
        spec = VoiceSpec.model_validate(spec)
    root = Path(root).resolve()
    # An explicit language is used as written; else the brand's, else English.
    spec = spec.model_copy(
        update={"language": spec.language or brand_language(root) or DEFAULT_LANGUAGE}
    )
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("voice name must be a single safe directory name")
    directory = root / "voice" / name
    says = [_say(item) for item in spec.lines]
    shows = [_show(item) for item in spec.lines]
    cached = _load_cached_track(root, directory, spec, name)
    if cached is not None:
        return cached

    cache = cache or Cache(root / "cache")
    reference = _project_path(root, spec.reference, "reference") if spec.reference else None
    if reference is not None and not reference.is_file():
        raise FileNotFoundError(reference)
    mapping = {
        index: str(_project_path(root, value, f"files[{index}]"))
        for index, value in (spec.files or {}).items()
    }
    if spec.engine == "kokoro" and language_code(spec.language) not in {"fr", "en"}:
        raise ValueError("the kokoro voice engine speaks French (fr) or English (en)")
    selected_engine = engine or get_engine(
        spec.engine,
        project_dir=root,
        mapping=mapping,
    )
    directory.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    def report(stage: str, done: int, total: int, message: str) -> None:
        if progress is not None:
            progress(
                {
                    "stage": stage,
                    "done": min(max(done, 0), total),
                    "total": total,
                    "unit": "lines",
                    "message": message,
                }
            )

    # ``ADANVIL_VOICE_CHECK_MODEL`` picks the whisper model of the pronunciation check
    # unless the spec sets ``check_model`` itself.
    check_model: str = spec.check_model
    if "check_model" not in spec.model_fields_set:
        check_model = os.environ.get("ADANVIL_VOICE_CHECK_MODEL", "").strip() or check_model

    states = [_BestLine() for _ in says]
    temperatures = [spec.temperature, 0.6, 0.8]
    if spec.engine == "kokoro":
        temperatures = temperatures[:1]  # fixed voice, no sampling: a retry says the same thing

    session_factory = getattr(selected_engine, "session", None)
    confidence = _check_confidence(spec)
    # One worker process for the whole job: its models load once, not once per step.
    with session_factory() if callable(session_factory) else contextlib.nullcontext():
        for attempt, temperature in enumerate(temperatures):
            pending = [
                index for index, state in enumerate(states) if not state.passes(spec.min_similarity)
            ]
            if not pending:
                break
            jobs: list[LineJob] = []
            staged: list[tuple[Path, Path]] = []
            attempt_paths: list[Path] = []
            for index in pending:
                reference_key = Cache.file_key(reference) if reference is not None else None
                key = {
                    "engine": spec.engine,
                    "text": says[index],
                    "reference": reference_key,
                    "language": spec.language,
                    "exaggeration": spec.exaggeration,
                    "cfg_weight": spec.cfg_weight,
                    "temperature": temperature,
                    "seed": None,
                    "attempt": attempt,
                }
                if spec.engine == "kokoro":
                    key["voice"] = spec.voice
                raw = _cached_path(cache, key)
                attempt_paths.append(raw)
                if hasattr(selected_engine, "register_audio"):
                    # Test engines map audio to text by path; cached and renamed files included.
                    selected_engine.register_audio(raw, says[index])  # type: ignore[attr-defined]
                if not raw.is_file():
                    temporary_raw = Path(f"{raw}.tmp.wav")
                    temporary_raw.unlink(missing_ok=True)
                    staged.append((temporary_raw, raw))
                    jobs.append(
                        LineJob(
                            index=index,
                            text=says[index],
                            language=spec.language,
                            reference=reference,
                            out=temporary_raw,
                            exaggeration=spec.exaggeration,
                            cfg_weight=spec.cfg_weight,
                            temperature=temperature,
                            voice=spec.voice,
                        )
                    )
            if jobs:
                try:
                    total_jobs = len(jobs)

                    def tts_progress(done: int, total_jobs: int = total_jobs) -> None:
                        report("tts", done, total_jobs, f"tts {done}/{total_jobs}")

                    _synthesize_with_progress(selected_engine, jobs, tts_progress)
                    for temporary_raw, raw in staged:
                        os.replace(temporary_raw, raw)
                except BaseException:
                    for temporary_raw, _raw in staged:
                        temporary_raw.unlink(missing_ok=True)
                    raise
            alignments = selected_engine.align(attempt_paths, spec.language, spec.align_model)
            trimmed_paths: list[Path] = []
            offsets: list[float] = []
            for index, raw, alignment in zip(pending, attempt_paths, alignments, strict=True):
                trimmed = directory / f".line_{index + 1:02d}.attempt_{attempt}.wav"
                offsets.append(_trim_line(raw, trimmed, alignment, spec.lead, spec.tail))
                if hasattr(selected_engine, "register_audio"):
                    selected_engine.register_audio(trimmed, says[index])  # type: ignore[attr-defined]
                trimmed_paths.append(trimmed)
            # The alignment already transcribed each line: when the check uses the same model,
            # or the transcript is confident and correct, it is the check (no second pass).
            shared = spec.align_model == check_model
            checked_at: dict[int, Alignment] = {}
            for position, (index, alignment) in enumerate(zip(pending, alignments, strict=True)):
                if alignment.error or not (
                    shared
                    or (
                        confidence is not None
                        and _confident_enough(
                            alignment, says[index], confidence, spec.min_similarity
                        )
                    )
                ):
                    continue
                if alignment.text or alignment.words:
                    checked_at[position] = _shifted(
                        alignment, trimmed_paths[position], offsets[position]
                    )
            recheck = [position for position in range(len(pending)) if position not in checked_at]
            if recheck:
                rechecked = selected_engine.align(
                    [trimmed_paths[position] for position in recheck],
                    spec.language,
                    check_model,
                )
                checked_at.update(zip(recheck, rechecked, strict=True))
            checked = [checked_at[position] for position in range(len(pending))]
            for index, trimmed, checked_alignment in zip(
                pending, trimmed_paths, checked, strict=True
            ):
                transcript = checked_alignment.text
                if (
                    not transcript
                    and isinstance(selected_engine, FilesEngine)
                    and not checked_alignment.words
                ):
                    transcript = says[index]
                similarity = _similarity(says[index], transcript)
                unmatched = _unmatched_words(says[index], transcript)
                if not transcript and isinstance(selected_engine, FilesEngine):
                    similarity = 1.0
                    unmatched = []
                word_values = [
                    (float(word[0]), float(word[1]), str(word[2]))
                    for word in checked_alignment.words
                ]
                state = states[index]
                state.attempts += 1
                if state.path is None or (-len(unmatched), similarity) > (
                    -len(state.unmatched),
                    state.similarity,
                ):
                    state.similarity = similarity
                    state.unmatched = unmatched
                    state.transcript = transcript
                    state.words = word_values
                    state.path = trimmed
                done_lines = sum(state.path is not None for state in states)
                report(
                    "check",
                    done_lines,
                    len(states),
                    f"checked line {index + 1}/{len(states)}",
                )

    final_paths: list[Path] = []
    voice_lines: list[VoiceLine] = []
    for index, (_text, state) in enumerate(zip(says, states, strict=True)):
        if state.path is None or state.words is None:
            raise RuntimeError(f"Voice line {index} did not produce audio")
        if not state.passes(spec.min_similarity):
            detail = f"similarity {state.similarity:.3f}"
            if state.unmatched:
                detail += f", unmatched words: {', '.join(state.unmatched)}"
            warnings.append(
                f"Line {index + 1} pronunciation remains suspect after {state.attempts} attempts "
                f"({detail})."
            )
        final = directory / f"line_{index + 1:02d}.wav"
        shutil.copyfile(state.path, final)
        final_paths.append(final)

    starts: list[float] = []
    cursor = 0.0
    durations: list[float] = []
    for path in final_paths:
        starts.append(cursor)
        duration = _line_duration(path)
        durations.append(duration)
        cursor += duration + spec.gap
    total = cursor - spec.gap if final_paths else 0.0
    full = directory / "voice_full.wav"
    full_duration = _write_full(final_paths, spec.gap, full)
    voice_lines = [
        VoiceLine(
            index=index,
            text=text,
            display=shows[index],
            file=_relative(root, path),
            duration=duration,
            start=start,
            words=states[index].words or [],
            transcript=states[index].transcript,
            similarity=states[index].similarity,
            status="ok" if states[index].passes(spec.min_similarity) else "suspect",
            attempts=states[index].attempts,
            unmatched_words=states[index].unmatched,
        )
        for index, (text, path, duration, start) in enumerate(
            zip(says, final_paths, durations, starts, strict=True)
        )
    ]
    if abs(full_duration - total) > 1 / 48_000:
        warnings.append(
            f"voice_full.wav duration {full_duration:.6f}s differs from manifest total {total:.6f}s"
        )
    if isinstance(selected_engine, FilesEngine):
        warnings.extend(selected_engine.warnings)
    manifest_path = directory / "manifest.json"
    track = VoiceTrack(
        name=name,
        language=spec.language,
        engine=spec.engine,
        lines=voice_lines,
        total_duration=full_duration,
        manifest=_relative(root, manifest_path),
        warnings=list(dict.fromkeys(warnings)),
    )
    temporary = Path(f"{manifest_path}.tmp")
    temporary.write_text(track.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    for path in directory.glob("line_*.wav"):
        if path not in final_paths:
            path.unlink(missing_ok=True)
    for path in directory.glob(".line_*.attempt_*.wav"):
        path.unlink(missing_ok=True)
    report("complete", len(spec.lines), len(spec.lines), "voice generation complete")
    return track


def load_track(path: Path) -> VoiceTrack:
    """Load and validate a generated voice manifest from ``path``."""

    path = Path(path)
    try:
        return VoiceTrack.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not parse voice manifest {path}: {exc}") from exc


__all__ = [
    "DEFAULT_LANGUAGE",
    "VoiceLine",
    "VoiceLineText",
    "VoiceSpec",
    "VoiceTrack",
    "brand_language",
    "generate_voice",
    "language_code",
    "load_track",
]
