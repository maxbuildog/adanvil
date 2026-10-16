"""Incremental media measurement and contact-sheet indexing."""

from __future__ import annotations

import io
import os
import shutil
import unicodedata
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from adanvil import config, ffmpeg
from adanvil.ffmpeg import MediaInfo
from adanvil.media.cache import Cache
from adanvil.media.sheet import contact_sheet
from adanvil.project import resolve_in_project
from adanvil.safeio import replace_bytes
from adanvil.speech import store as speech_store
from adanvil.speech import transcribe as speech_transcribe
from adanvil.speech.model import SpeechData
from adanvil.textremoval import detect_text_boxes

from .annotate import entry_line
from .catalog import (
    Catalog,
    Entry,
    EntrySpeech,
    Shot,
    entry_id,
    load_catalog,
    merge_into_catalog,
    update_catalog,
)
from .onepass import KeyframePass, keyframe_pass
from .separate_audio import refresh_separate_audio
from .vision_index import vision_phase, vision_plan

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".aac", ".ogg"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS | AUDIO_EXTENSIONS


@dataclass
class IndexResult:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    speech: dict[str, str] = field(default_factory=dict)  # entry id -> speech status of this run
    vision: dict[str, str] = field(default_factory=dict)  # entry id -> vision status of this run


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _relative_path(root: Path, path: Path) -> str:
    relative = path.relative_to(root).as_posix()
    return unicodedata.normalize("NFC", relative)


def _iter_media(root: Path) -> list[Path]:
    if root.is_file():  # ``source`` may be a single file
        return [root] if root.suffix.casefold() in MEDIA_EXTENSIONS else []
    if not root.is_dir():
        return []
    found: list[Path] = []
    visited: set[str] = set()
    for directory, directories, filenames in os.walk(root, followlinks=True):
        real_directory = os.path.normcase(os.path.realpath(directory))
        if real_directory in visited:
            directories[:] = []
            continue
        visited.add(real_directory)
        directories[:] = sorted(
            directory_name
            for directory_name in directories
            if not directory_name.startswith(".")
            and directory_name != "cache"
            and os.path.normcase(
                os.path.realpath(os.path.join(directory, directory_name))
            )
            not in visited
        )
        for filename in sorted(filenames):
            if filename.startswith("."):
                continue
            path = Path(directory) / filename
            if path.suffix.casefold() in MEDIA_EXTENSIONS and path.is_file():
                found.append(path)
    return found


def _merge_short_shots(boundaries: list[float], duration: float) -> list[tuple[float, float]]:
    shots = [(start, end) for start, end in zip(boundaries, boundaries[1:], strict=False)]
    while len(shots) > 1:
        short_index = next(
            (index for index, (start, end) in enumerate(shots) if end - start < 0.3), None
        )
        if short_index is None:
            break
        if short_index == 0:
            shots[1] = (shots[0][0], shots[1][1])
            del shots[0]
        else:
            shots[short_index - 1] = (shots[short_index - 1][0], shots[short_index][1])
            del shots[short_index]
    if not shots:
        return [(0.0, duration)]
    shots[0] = (0.0, shots[0][1])
    shots[-1] = (shots[-1][0], duration)
    return shots


def _video_shots(
    info: MediaInfo,
    cuts: list[float],
    yavg_samples: list[float],
    *,
    sample_fps: int = 4,
) -> list[Shot]:
    if info.duration is None or info.duration <= 0:
        raise ValueError("video has no positive duration")
    duration = float(info.duration)
    filtered_cuts = [float(cut) for cut in cuts if 0.0 < float(cut) < duration]
    boundaries = [0.0, *sorted(set(filtered_cuts)), duration]
    shots = _merge_short_shots(boundaries, duration)
    result: list[Shot] = []
    for start, end in shots:
        midpoint = (start + end) / 2
        sample_start = max(start, midpoint - 0.5)
        sample_end = min(end, midpoint + 0.5)
        first = max(0, int(sample_start * sample_fps))
        last = min(len(yavg_samples), max(first + 1, int(sample_end * sample_fps) + 1))
        values = yavg_samples[first:last]
        yavg = sum(values) / len(values) if values else 128.0
        result.append(Shot(start=start, end=end, yavg=yavg))
    return result


def _sheet_timestamps(duration: float, shots: list[Shot]) -> tuple[list[float], dict[int, str]]:
    """Pick sheet frames: one per shot midpoint, the rest spread evenly over the clip.

    Frames are returned in chronological order; the first frame inside each shot is
    labelled ``S<n>`` so the agent can read the cut structure off the sheet.
    """
    if duration <= 20:
        count = 12
    elif duration <= 60:
        count = 16
    else:
        count = 24

    midpoints = [(shot.start + shot.end) / 2 for shot in shots][:count]
    remaining = count - len(midpoints)
    min_gap = duration / (count * 4)
    fill: list[float] = []
    for index in range(remaining):
        candidate = (index + 0.5) * duration / remaining
        if all(abs(candidate - taken) >= min_gap for taken in midpoints + fill):
            fill.append(candidate)
    # Top up by farthest-point sampling: always fill the largest remaining gap, so
    # extra frames never bunch up at the start of the clip.
    grid = [(index + 0.5) * duration / (count * 8) for index in range(count * 8)]
    while len(midpoints) + len(fill) < count:
        taken = midpoints + fill
        best = max(grid, key=lambda t: min(abs(t - other) for other in taken) if taken else t)
        if taken and min(abs(best - other) for other in taken) < 1e-3:
            break
        fill.append(best)

    timestamps = sorted(round(t, 3) for t in midpoints + fill)
    labels: dict[int, str] = {}
    for shot_number, shot in enumerate(shots):
        for position, timestamp in enumerate(timestamps):
            if shot.start <= timestamp < shot.end or (
                shot_number == len(shots) - 1 and timestamp == shot.end
            ):
                labels.setdefault(position, f"S{shot_number + 1}")
                break
    return timestamps, labels


BIG_SOURCE_SIDE = 1920  # above this long side, analysis decodes keyframes only
SHEET_THUMB_WIDTH = 270
CATALOG_BATCH = 1  # finished entries merged per catalog write: progress and crash safety per file


def _is_big(info: MediaInfo) -> bool:
    return max(info.width or 0, info.height or 0) > BIG_SOURCE_SIDE


def _measure(
    path: Path,
    info: MediaInfo,
    scene_threshold: float,
    keyframes: KeyframePass | None = None,
) -> tuple[list[float], list[float]]:
    """Cuts and luma samples; keyframes only for big sources, full decode if too few.

    ``keyframes`` is the shared keyframe pass of the file when it already ran.
    """

    if keyframes is not None:
        if keyframes.decoded >= 3:
            return keyframes.cuts, keyframes.yavg
        return ffmpeg.measure_video(path, threshold=scene_threshold)
    if _is_big(info):
        cuts, yavg, decoded = ffmpeg.measure_video_keyframes(path, threshold=scene_threshold)
        if decoded >= 3:
            return cuts, yavg
    return ffmpeg.measure_video(path, threshold=scene_threshold)


def _keyframe_windows(
    timestamps: list[float], shots: list[Shot], tolerance: float = 0.75
) -> list[tuple[float, float]]:
    """Where a keyframe may stand in for each sheet timestamp: near it and in its shot."""

    windows: list[tuple[float, float]] = []
    for timestamp in timestamps:
        shot = next(
            (shot for shot in shots if shot.start <= timestamp < shot.end),
            shots[-1] if shots else Shot(start=0.0, end=timestamp),
        )
        windows.append(
            (max(shot.start, timestamp - tolerance), min(shot.end, timestamp + tolerance))
        )
    return windows


def _image_sheet(path: Path, output: Path) -> None:
    from PIL import Image

    encoded = io.BytesIO()
    with Image.open(path) as image:
        width = 540
        height = max(1, round(image.height * width / image.width))
        image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS).save(
            encoded, format="JPEG", quality=90, optimize=True
        )
    replace_bytes(output, encoded.getvalue())  # a link planted at ``output`` is replaced


def _index_file(
    path: Path,
    relative: str,
    identifier: str,
    *,
    scene_threshold: float,
    text_scan: bool,
    project_root: Path,
) -> Entry:
    info = ffmpeg.probe(path)
    indexed = _now()
    sheet_relative: str | None = None
    sheet_path: Path | None = None
    if info.kind in {"video", "image"}:
        sheet_relative = f"cache/sheets/{identifier}.jpg"
        sheet_path = resolve_in_project(project_root, sheet_relative)
        sheet_path.parent.mkdir(parents=True, exist_ok=True)

    shots: list[Shot] = []
    text_boxes: list[tuple[int, int, int, int]] = []
    yavg: float | None = None
    if info.kind == "video":
        if info.duration is None:
            raise ValueError("video has no positive duration")
        keyframes: KeyframePass | None = None
        scratch = sheet_path.parent / f".adanvil-keyframes-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            if _is_big(info):
                # One keyframe decode feeds the cut/luma measure, the text scan and the sheet.
                try:
                    keyframes = keyframe_pass(
                        path,
                        info,
                        threshold=scene_threshold,
                        thumb_dir=scratch,
                        thumb_width=SHEET_THUMB_WIDTH,
                        text_seconds=min(float(info.duration), 6.0) if text_scan else None,
                    )
                except Exception:  # the separate passes below still work
                    keyframes = None
            cuts, yavg_samples = _measure(path, info, scene_threshold, keyframes)
            yavg = sum(yavg_samples) / len(yavg_samples) if yavg_samples else 128.0
            shots = _video_shots(info, cuts, yavg_samples)
            if text_scan:
                text_boxes = [
                    tuple(int(value) for value in box)
                    for box in detect_text_boxes(
                        path,
                        duration=min(float(info.duration), 6.0),
                        fps=2,
                        fast=_is_big(info),
                        keyframe_frames=keyframes.text_frames if keyframes else None,
                    )
                ]
            timestamps, shot_labels = _sheet_timestamps(float(info.duration), shots)
            assert sheet_path is not None
            contact_sheet(
                path,
                sheet_path,
                n=len(timestamps),
                cols=4,
                thumb_width=SHEET_THUMB_WIDTH,
                labels=True,
                timestamps=timestamps,
                shot_labels=shot_labels,
                keyframe_windows=(
                    _keyframe_windows(timestamps, shots) if _is_big(info) else None
                ),
                keyframes=keyframes.thumbs if keyframes else None,
            )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    elif info.kind == "image":
        assert sheet_path is not None
        _image_sheet(path, sheet_path)

    return Entry(
        id=identifier,
        path=relative,
        kind=info.kind,
        file_key=Cache.file_key(path),
        duration=info.duration,
        width=info.width,
        height=info.height,
        fps=info.fps,
        rotation=info.rotation,
        hdr=info.is_hdr,
        has_audio=info.has_audio,
        yavg=yavg,
        shots=shots,
        text_boxes=text_boxes,
        sheet=sheet_relative,
        indexed=indexed,
    )


def pending_files(root: Path, source: str) -> tuple[list[Path], int]:
    """The media files under ``source`` and how many of them need (re)indexing."""

    project_root = Path(root).resolve()
    files = _iter_media(resolve_in_project(project_root, source))
    catalog = load_catalog(project_root)
    pending = 0
    for path in files:
        previous = catalog.entries.get(entry_id(_relative_path(project_root, path)))
        try:
            if previous is None or previous.file_key != Cache.file_key(path):
                pending += 1
        except OSError:
            pending += 1
    return files, pending


def index_summary(root: Path, result: IndexResult) -> dict[str, object]:
    """Compact report of an indexing run: counts, one line per new or updated entry, errors."""

    catalog = load_catalog(Path(root).resolve())
    changed = sorted([*result.added, *result.updated])
    speech: dict[str, int] = {}
    for status in result.speech.values():
        key = status.split(":")[0]
        speech[key] = speech.get(key, 0) + 1
    vision: dict[str, int] = {}
    for status in result.vision.values():
        key = status.split(":")[0]
        vision[key] = vision.get(key, 0) + 1
    summary: dict[str, object] = {
        "counts": {
            "added": len(result.added),
            "updated": len(result.updated),
            "unchanged": len(result.unchanged),
            "removed": len(result.removed),
            "errors": len(result.errors),
            "total": len(catalog.entries),
        },
        "entries": [
            entry_line(catalog.entries[item]) for item in changed if item in catalog.entries
        ],
        "speech": speech,
        "errors": result.errors,
    }
    if vision:  # only when the vision pass ran or said why it did not
        summary["vision"] = vision
        failed = sorted({status for status in result.vision.values() if status != "done"})
        if failed:
            summary["vision_note"] = failed[0]
    return summary


SpeechRequest = bool | Literal["auto"]


def _speech_plan(transcribe: SpeechRequest) -> SpeechRequest | None:
    """``True`` (force), ``"auto"``, or ``None`` when no transcription is wanted."""

    if transcribe not in {True, False, "auto"}:
        raise ValueError("transcribe must be true, false, or 'auto'")
    if transcribe is False:
        return None
    if transcribe == "auto":
        return "auto" if speech_transcribe.auto_enabled() else None
    return True


def _needs_speech(project_root: Path, entry: Entry, mode: SpeechRequest) -> bool:
    """Whether a clip with audio still has to be transcribed (or retried)."""

    speech = entry.speech
    if speech is None:
        return True
    if speech.status == speech_transcribe.DONE:
        try:
            return not resolve_in_project(project_root, speech.words_file or "").is_file()
        except (TypeError, ValueError):
            return True
    if speech.status == speech_transcribe.SILENT:
        return mode is True  # a silent clip is transcribed only when asked for
    return True  # unavailable, failed, or marked as having no audio before it had some


def _set_speech(project_root: Path, updates: dict[str, tuple[dict[str, Any], EntrySpeech]]) -> None:
    """Store speech summaries on the entries that still describe the same file."""

    def change(catalog: Catalog) -> None:
        for identifier, (file_key, info) in updates.items():
            stored = catalog.entries.get(identifier)
            if stored is not None and stored.file_key == file_key:
                catalog.entries[identifier] = stored.model_copy(update={"speech": info})

    if updates:
        update_catalog(project_root, change)


def _has_sound_track(entry: Entry) -> bool:
    """Audio files always have a track; a video has one when its probe found an audio stream."""

    return entry.kind == "audio" or (entry.kind == "video" and entry.has_audio)


def store_speech_outcome(
    project_root: Path,
    entry: Entry,
    outcome: speech_transcribe.SpeechOutcome,
    *,
    model: str,
    keep_on_failure: bool = False,
) -> EntrySpeech | None:
    """Write the sidecar and the catalogue summary of one transcribed clip, at once.

    A transcript replaces the previous one (the same sidecar file). A clip that was not
    transcribed gets its status; with ``keep_on_failure`` (a refinement) the transcript it
    already has is left untouched instead and ``None`` is returned.
    """

    previous = entry.speech.words_file if entry.speech is not None else None
    if outcome.done:
        relative = speech_store.sidecar_relative(entry.file_key)
        speech_store.write_sidecar(
            project_root,
            relative,
            SpeechData(
                words=outcome.words,
                language=outcome.language,
                duration=outcome.duration,
                level_step=speech_transcribe.LEVEL_STEP,
                levels_db=outcome.levels_db,
                model=model,
            ),
            source=entry.path,
            model=model,
        )
        info = EntrySpeech(
            status=outcome.status,
            language=outcome.language,
            text=" ".join(word.text for word in outcome.words),
            words_file=relative,
            has_speech=bool(outcome.words),
        )
        if previous and previous != relative:
            speech_store.remove_sidecar(project_root, previous)
    elif keep_on_failure:
        return None
    else:
        info = EntrySpeech(status=outcome.status[:300])
    _set_speech(project_root, {entry.id: (entry.file_key, info)})
    return info


def _speech_phase(
    project_root: Path,
    identifiers: set[str],
    mode: SpeechRequest,
    result: IndexResult,
    *,
    language: str | None,
    engine: Any | None,
    progress: Callable[[str], None] | None,
) -> None:
    """Transcribe the clips of this run that need it, in one batch (one model load).

    Each file is written to the catalogue as soon as the transcriber has finished it, so
    ``library_search(text=...)`` finds what a clip says while the job is still running.
    """

    catalog = load_catalog(project_root)
    marks: dict[str, tuple[dict[str, Any], EntrySpeech]] = {}
    wanted: list[Entry] = []
    for identifier in sorted(identifiers):
        entry = catalog.entries.get(identifier)
        if entry is None or entry.kind not in {"video", "audio"}:
            continue
        if not _has_sound_track(entry):
            if entry.speech is None or entry.speech.status != speech_transcribe.NO_AUDIO:
                marks[identifier] = (
                    entry.file_key,
                    EntrySpeech(status=speech_transcribe.NO_AUDIO),
                )
            result.speech[identifier] = speech_transcribe.NO_AUDIO
        elif _needs_speech(project_root, entry, mode):
            wanted.append(entry)
    if wanted and not speech_transcribe.backend_available(project_root, engine):
        for entry in wanted:
            status = speech_transcribe.UNAVAILABLE
            if entry.speech is None or entry.speech.status != status:
                marks[entry.id] = (entry.file_key, EntrySpeech(status=status))
            result.speech[entry.id] = status
        wanted = []
    _set_speech(project_root, marks)
    if not wanted:
        return

    model = speech_transcribe.speech_model()
    code = speech_transcribe.resolve_language(project_root, language)
    paths = [resolve_in_project(project_root, entry.path) for entry in wanted]
    if progress:
        progress(f"speech 0/{len(wanted)} preparing audio")

    def persist(index: int, outcome: speech_transcribe.SpeechOutcome) -> None:
        entry = wanted[index]
        info = store_speech_outcome(project_root, entry, outcome, model=model)
        assert info is not None
        if outcome.status.startswith("failed"):
            result.errors.append({"path": entry.path, "message": f"speech {outcome.status}"})
        result.speech[entry.id] = info.status

    def transcribing(done: int, total: int) -> None:
        if progress:
            progress(f"speech {done}/{total} transcribing")

    speech_transcribe.transcribe_clips(
        paths,
        project_root=project_root,
        language=code,
        model=model,
        engine=engine,
        skip_silent=mode == "auto",
        progress=transcribing,
        on_result=persist,
    )


@dataclass
class RefineResult:
    """What ``refine_speech`` did: the clips redone, the ones left alone and why."""

    model: str
    refined: list[str] = field(default_factory=list)  # entry ids
    skipped: list[dict[str, str]] = field(default_factory=list)  # {"path", "reason"}
    errors: list[dict[str, str]] = field(default_factory=list)


def refine_targets(
    root: Path, sources: list[str]
) -> tuple[list[Entry], list[dict[str, str]]]:
    """The indexed clips named by ``sources`` (project paths, folders or entry ids).

    Returns the clips that can be transcribed again and, for the others, ``{"path", "reason"}``
    (not indexed, no sound track, silent audio). A folder only brings the clips that can be
    refined: its B-roll without sound is not reported.
    """

    project_root = Path(root).resolve()
    catalog = load_catalog(project_root)
    chosen: dict[str, Entry] = {}
    skipped: dict[str, str] = {}
    for written in sources:
        explicit = True
        if written in catalog.entries:
            matches = [catalog.entries[written]]
        else:
            prefix = unicodedata.normalize("NFC", str(written)).replace("\\", "/")
            prefix = prefix.removeprefix("./").rstrip("/")
            matches = [
                entry
                for entry in catalog.entries.values()
                if entry.path == prefix or entry.path.startswith(f"{prefix}/")
            ]
            explicit = len(matches) == 1 and matches[0].path == prefix
        if not matches:
            skipped[str(written)] = "not indexed: run library_index first"
        for entry in matches:
            reason = None
            if entry.kind not in {"video", "audio"} or not _has_sound_track(entry):
                reason = "no sound track"
            elif entry.speech is not None and entry.speech.status == speech_transcribe.SILENT:
                reason = "silent audio, nothing to refine"
            if reason is None:
                chosen[entry.id] = entry
            elif explicit:
                skipped[entry.path] = reason
    ordered = sorted(chosen.values(), key=lambda entry: entry.path)
    return ordered, [{"path": path, "reason": reason} for path, reason in skipped.items()]


def refine_speech(
    root: Path,
    entry_ids: list[str],
    *,
    model: str | None = None,
    language: str | None = None,
    engine: Any | None = None,
    progress: Callable[[str], None] | None = None,
) -> RefineResult:
    """Transcribe the chosen clips again with a stronger model and the brand vocabulary.

    The new words replace the old ones (same sidecar, same catalogue entry); a clip that fails
    keeps the transcript it had. ``model`` defaults to ``medium``. The transcriber is the one
    of indexing (faster-whisper, else the voice environment): without any, this raises.
    """

    project_root = Path(root).resolve()
    model_name = (model or speech_transcribe.REFINE_MODEL).strip()
    if not model_name:
        raise ValueError("model must not be empty")
    result = RefineResult(model=model_name)
    catalog = load_catalog(project_root)
    entries = [
        catalog.entries[identifier] for identifier in entry_ids if identifier in catalog.entries
    ]
    if not entries:
        return result
    if not speech_transcribe.backend_available(project_root, engine):
        raise RuntimeError(speech_transcribe.UNAVAILABLE)
    code = speech_transcribe.resolve_language(project_root, language)
    paths = [resolve_in_project(project_root, entry.path) for entry in entries]
    if progress:
        progress(f"speech 0/{len(entries)} preparing audio")

    def persist(index: int, outcome: speech_transcribe.SpeechOutcome) -> None:
        entry = entries[index]
        info = store_speech_outcome(
            project_root, entry, outcome, model=model_name, keep_on_failure=True
        )
        if info is not None:
            result.refined.append(entry.id)
        elif outcome.status == speech_transcribe.SILENT:
            result.skipped.append({"path": entry.path, "reason": "silent audio, nothing to refine"})
        else:
            result.errors.append({"path": entry.path, "message": f"speech {outcome.status}"})

    def transcribing(done: int, total: int) -> None:
        if progress:
            progress(f"speech {done}/{total} transcribing")

    speech_transcribe.transcribe_clips(
        paths,
        project_root=project_root,
        language=code,
        model=model_name,
        engine=engine,
        skip_silent=True,
        progress=transcribing,
        on_result=persist,
    )
    return result


def pending_speech(root: Path, source: str, transcribe: SpeechRequest = "auto") -> int:
    """How many clips under ``source`` an index run would transcribe (0 when speech is off)."""

    mode = _speech_plan(transcribe)
    project_root = Path(root).resolve()
    if mode is None or not speech_transcribe.backend_available(project_root):
        return 0
    catalog = load_catalog(project_root)
    count = 0
    for path in _iter_media(resolve_in_project(project_root, source)):
        if path.suffix.casefold() not in VIDEO_EXTENSIONS | AUDIO_EXTENSIONS:
            continue
        previous = catalog.entries.get(entry_id(_relative_path(project_root, path)))
        try:
            changed = previous is None or previous.file_key != Cache.file_key(path)
        except OSError:
            changed = True
        if changed or (previous.has_audio and _needs_speech(project_root, previous, mode)):
            count += 1
    return count


def _belongs_to_source(path: str, source: str) -> bool:
    source = source.rstrip("/")
    return path == source or path.startswith(f"{source}/")


def index_library(
    root: Path,
    source: str = "footage",
    *,
    scene_threshold: float = 0.3,
    text_scan: bool = True,
    progress: Callable[[str], None] | None = None,
    transcribe: SpeechRequest = "auto",
    language: str | None = None,
    engine: Any | None = None,
    vision: bool | Literal["auto"] = "auto",
    vision_engine: Any | None = None,
) -> IndexResult:
    """Measure new/changed media below ``root/source`` and persist the catalog.

    Clips with sound are also transcribed (word timestamps, kept next to the catalogue in
    ``cache/speech/``): ``transcribe`` is ``"auto"`` (clips with an audio stream that is not
    near-silent), ``True`` (every clip with an audio stream) or ``False`` (no speech work;
    entries keep what they have). The model comes from ``ADANVIL_SPEECH_MODEL``, the language
    from ``language``, else the brand profile, else auto-detection. Without a transcriber
    the entries get an ``unavailable`` speech status: indexing never fails for it. Clips
    already transcribed are not redone; ``engine`` injects a transcriber (tests).

    With the vision environment installed (``adanvil vision-setup``) the frames of the contact
    sheet are also embedded and searched for faces, and each entry gets automatic tags and a
    product version guess: ``vision`` is ``"auto"`` (when installed), ``True`` (redo every
    entry) or ``False`` (no vision work). Without the environment the entries get a vision
    status ``unavailable: ...`` and indexing still succeeds; ``vision_engine`` injects an
    engine (tests).
    """

    plan = _speech_plan(transcribe)
    vision_mode = vision_plan(vision)

    project_root = Path(root).resolve()
    source_root = resolve_in_project(project_root, source)
    source_relative = _relative_path(project_root, source_root)
    catalog = load_catalog(project_root)
    result = IndexResult()
    seen: set[str] = set()

    todo: list[tuple[Path, str, str, Entry | None]] = []
    for path in _iter_media(source_root):
        relative = _relative_path(project_root, path)
        identifier = entry_id(relative)
        seen.add(identifier)
        previous = catalog.entries.get(identifier)
        try:
            key = Cache.file_key(path)
            unchanged = previous is not None and previous.file_key == key
        except Exception as exc:  # one bad rush must not abort the library scan
            result.errors.append({"path": relative, "message": str(exc)})
            if progress:
                progress(f"error {relative}: {exc}")
            continue
        if unchanged:
            result.unchanged.append(identifier)
            if progress:
                progress(f"unchanged {relative}")
            continue
        todo.append((path, relative, identifier, previous))

    def analyse(item: tuple[Path, str, str, Entry | None]) -> Entry:
        path, relative, identifier, _previous = item
        return _index_file(
            path,
            relative,
            identifier,
            scene_threshold=scene_threshold,
            text_scan=text_scan,
            project_root=project_root,
        )

    finished: list[Entry] = []
    stale_speech: list[str] = []  # sidecars of files that changed: their words are obsolete
    saved_messages: list[str] = []  # "indexed" is reported once the entry is in the catalog
    failures: dict[int, dict[str, str]] = {}

    def flush() -> None:
        # Read-merge-write under the catalog lock: other index jobs and annotations
        # may have changed the file since it was loaded.
        if finished:
            merge_into_catalog(project_root, list(finished))
            finished.clear()
        if progress:
            for message in saved_messages:
                progress(message)
        saved_messages.clear()

    pool = ThreadPoolExecutor(max_workers=max(1, min(config.workers(), len(todo))))
    try:
        futures = {pool.submit(analyse, item): number for number, item in enumerate(todo)}
        for future in as_completed(futures):
            number = futures[future]
            _path, relative, identifier, previous = todo[number]
            try:
                indexed = future.result()
            except Exception as exc:  # one bad rush must not abort the library scan
                failures[number] = {"path": relative, "message": str(exc)}
                if progress:
                    progress(f"error {relative}: {exc}")
                continue
            if previous is not None:
                indexed = indexed.model_copy(update={"annotation": previous.annotation}, deep=True)
                result.updated.append(identifier)
                if previous.speech is not None and previous.speech.words_file:
                    stale_speech.append(previous.speech.words_file)
            else:
                result.added.append(identifier)
            finished.append(indexed)
            saved_messages.append(f"indexed {relative}")
            if len(finished) >= CATALOG_BATCH:
                flush()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        flush()  # entries already measured are never lost, even if the job dies here
    result.errors.extend(failures[number] for number in sorted(failures))

    for words_file in stale_speech:
        speech_store.remove_sidecar(project_root, words_file)
    if plan is not None:
        _speech_phase(
            project_root,
            seen,
            plan,
            result,
            language=language,
            engine=engine,
            progress=progress,
        )
    if vision_mode is not None:
        vision_phase(
            project_root, seen, vision_mode, result, engine=vision_engine, progress=progress
        )
    refresh_separate_audio(project_root)  # voices recorded apart, for videos without sound

    gone: list[str] = []
    for identifier, previous in catalog.entries.items():
        if identifier not in seen and _belongs_to_source(previous.path, source_relative):
            gone.append(identifier)
            result.removed.append(identifier)
            result.errors.append(
                {"path": previous.path, "message": "file is missing from the source folder"}
            )
            if progress:
                progress(f"removed {previous.path}")

    result.added.sort()
    result.updated.sort()
    result.unchanged.sort()
    result.removed.sort()
    merge_into_catalog(project_root, removed=gone)
    if progress:
        progress("catalog saved")
    return result


__all__ = [
    "IndexResult",
    "RefineResult",
    "index_library",
    "index_summary",
    "pending_files",
    "pending_speech",
    "refine_speech",
    "refine_targets",
    "store_speech_outcome",
]
