"""Decode a reference ad into the assets an editing agent needs to read it."""

from __future__ import annotations

import json
import math
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from adanvil import ffmpeg
from adanvil.audio import profile_audio, rms_per_second
from adanvil.media.sheet import contact_sheet, reference_shot_sheets
from adanvil.project import resolve_in_project
from adanvil.textremoval import detect_text_boxes
from adanvil.voice.engines import VoiceUnavailableError, WorkerEngine

BBox = tuple[int, int, int, int]


class Word(BaseModel):
    """One word from an optional word-timestamped transcript."""

    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    text: str
    probability: float


class RefShot(BaseModel):
    """The measured contents of one scene in a reference ad."""

    model_config = ConfigDict(extra="forbid")

    index: int
    start: float
    end: float
    duration: float
    keyframe: str
    text_boxes: list[BBox] = Field(default_factory=list)
    spoken: str = ""
    text_regions: list[dict[str, str | int]] = Field(default_factory=list)
    yavg: float

    @model_validator(mode="after")
    def valid_range(self) -> RefShot:
        if self.index < 1:
            raise ValueError("reference shot indexes start at one")
        if self.start < 0 or self.end < self.start:
            raise ValueError("reference shot must have a non-negative range")
        if self.duration < 0:
            raise ValueError("reference shot duration must not be negative")
        return self


class ReferenceDecode(BaseModel):
    """The JSON-compatible manifest produced by :func:`decode_reference`."""

    model_config = ConfigDict(extra="forbid")

    source: str
    duration: float
    width: int
    height: int
    fps: float | None
    has_audio: bool
    shots: list[RefShot] = Field(default_factory=list)
    sheet: str
    soundtrack: str | None
    music_profile: dict[str, object] | None = None
    rms_per_second: list[float] = Field(default_factory=list)
    transcript: list[Word] | None
    transcript_status: str
    pacing: dict[str, object] = Field(default_factory=dict)
    shot_sheets: list[str] = Field(default_factory=list)
    breakdown_template: list[dict[str, object]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _slug(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    result = re.sub(r"[^A-Za-z0-9]+", "-", ascii_value).strip("-").lower()
    return result or "reference"


def _relative(root: Path, path: Path) -> str:
    return Path(path).relative_to(Path(root)).as_posix()


def _merge_short_shots(boundaries: list[float], duration: float) -> list[tuple[float, float]]:
    """Remove sub-frame scene detector noise while preserving useful short shots."""

    shots = [
        (start, end) for start, end in zip(boundaries, boundaries[1:], strict=False) if end > start
    ]
    while len(shots) > 1:
        short_index = next(
            (index for index, (start, end) in enumerate(shots) if end - start < 0.3),
            None,
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


def _make_shot_ranges(source: Path, duration: float, threshold: float) -> list[tuple[float, float]]:
    cuts = [
        float(cut)
        for cut in ffmpeg.detect_cuts(source, threshold=threshold)
        if 0.0 < float(cut) < duration
    ]
    boundaries = [0.0, *sorted(set(cuts)), duration]
    return _merge_short_shots(boundaries, duration)


def _shot_timestamps(
    duration: float, shots: list[tuple[float, float]]
) -> tuple[list[float], dict[int, str]]:
    rate = 3 if duration <= 12.0 else 2
    count = max(1, math.ceil(duration * rate))
    if count <= 48:
        timestamps = [min(index / rate, max(0.0, duration - 0.001)) for index in range(count)]
    else:
        timestamps = [
            min((index + 0.5) * duration / 48, max(0.0, duration - 0.001)) for index in range(48)
        ]

    labels: dict[int, str] = {}
    for shot_number, (start, end) in enumerate(shots, start=1):
        for frame_index, timestamp in enumerate(timestamps):
            if start <= timestamp < end or (shot_number == len(shots) and timestamp == end):
                labels.setdefault(frame_index, f"S{shot_number}")
                break
    return timestamps, labels


def _same_area(first: BBox, other: BBox) -> bool:
    intersection_width = max(0, min(first[2], other[2]) - max(first[0], other[0]))
    intersection_height = max(0, min(first[3], other[3]) - max(first[1], other[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    other_area = max(0, other[2] - other[0]) * max(0, other[3] - other[1])
    union = first_area + other_area - intersection
    return union > 0 and intersection / union >= 0.5


def _format_shot_list(indices: list[int]) -> str:
    if len(indices) == 1:
        return f"shot {indices[0]}"
    return "shots " + ", ".join(str(index) for index in indices[:-1]) + f" and {indices[-1]}"


def _text_notes(shots: list[RefShot]) -> list[str]:
    present = [shot for shot in shots if shot.text_boxes]
    if not present:
        return []
    anchor = present[0].text_boxes[0]
    same_area = all(any(_same_area(anchor, box) for box in shot.text_boxes) for shot in present[1:])
    suffix = " (same area)" if same_area else ""
    return [
        f"burnt-in text present in {_format_shot_list([shot.index for shot in present])}"
        f"{suffix}: candidate for remove_text"
    ]


def _silence_notes(rms: list[float]) -> list[str]:
    """Describe contiguous one-second windows that are effectively silent."""

    notes: list[str] = []
    start: int | None = None
    for index, level in enumerate([*rms, -120.0]):
        silent = level <= -60.0
        if silent and start is None:
            start = index
        elif not silent and start is not None:
            notes.append(f"audio is near-silent for {start:.1f}-{index:.1f} s")
            start = None
    return notes


def _text_regions(
    boxes: list[BBox], *, height: int
) -> list[dict[str, str | int]]:
    """Merge detected text boxes into top, middle and bottom bands."""

    bands: dict[str, list[BBox]] = {"top": [], "middle": [], "bottom": []}
    for box in boxes:
        center = (box[1] + box[3]) / 2
        ratio = center / max(1, height)
        band = "top" if ratio < 1 / 3 else "middle" if ratio < 2 / 3 else "bottom"
        bands[band].append(box)
    return [
        {"band": band, "height": max(box[3] for box in values) - min(box[1] for box in values)}
        for band, values in bands.items()
        if values
    ]


def _spoken_for_shot(words: list[Word] | None, start: float, end: float) -> str:
    if not words:
        return ""
    return " ".join(
        word.text.strip()
        for word in words
        if start <= (word.start + word.end) / 2 < end
    ).strip()


def _pacing(ranges: list[tuple[float, float]], duration: float) -> dict[str, object]:
    lengths = [end - start for start, end in ranges]
    cuts = [start for start, _ in ranges[1:]]
    rate = len(cuts) * 10 / duration if duration > 0 else 0.0
    quarter = duration / 4 if duration > 0 else 0.0
    by_quarter = []
    for index in range(4):
        start = index * quarter
        end = duration if index == 3 else (index + 1) * quarter
        count = sum(start <= cut < end for cut in cuts)
        by_quarter.append(count * 10 / (end - start) if end > start else 0.0)
    return {
        "shot_count": len(ranges),
        "mean": statistics.mean(lengths) if lengths else 0.0,
        "median": statistics.median(lengths) if lengths else 0.0,
        "min": min(lengths) if lengths else 0.0,
        "max": max(lengths) if lengths else 0.0,
        "cuts_per_10s": rate,
        "by_quarter": by_quarter,
    }


def _transcribe(
    source: Path,
    *,
    requested: bool,
    language: str | None,
    model: str,
    output_dir: Path,
    project_root: Path,
) -> tuple[list[Word] | None, str]:
    if not requested:
        return None, "not requested"
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        engine = WorkerEngine(project_dir=project_root)
        resolver = getattr(engine, "_python", None)
        if callable(resolver):
            try:
                resolver([])
            except VoiceUnavailableError:
                return None, "pip install 'adanvil[transcribe]' or run adanvil voice-setup"
        audio_path = output_dir / "transcript-16k.wav"
        try:
            ffmpeg.run(
                [
                    "-i",
                    str(source),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(audio_path),
                ]
            )
            alignment = engine.align([audio_path], language, model)[0]
            words = [
                Word(
                    start=float(word[0]),
                    end=float(word[1]),
                    text=str(word[2]),
                    probability=float(word[3]),
                )
                for word in alignment.words
            ]
            return words, "done (voice environment)"
        except VoiceUnavailableError:
            audio_path.unlink(missing_ok=True)
            return None, "pip install 'adanvil[transcribe]' or run adanvil voice-setup"
        except Exception as exc:  # the manifest should still be useful if a model fails
            return None, str(exc)

    try:
        whisper = WhisperModel(model, device="cpu", compute_type="int8")
        segments, _ = whisper.transcribe(
            str(source),
            word_timestamps=True,
            language=language,
            vad_filter=True,
        )
        words: list[Word] = []
        for segment in segments:
            for word in getattr(segment, "words", None) or []:
                text = getattr(word, "word", None)
                if text is None:
                    text = getattr(word, "text", "")
                probability = getattr(word, "probability", 0.0)
                words.append(
                    Word(
                        start=float(getattr(word, "start", getattr(segment, "start", 0.0))),
                        end=float(getattr(word, "end", getattr(segment, "end", 0.0))),
                        text=str(text),
                        probability=float(probability),
                    )
                )
        return words, "done"
    except Exception as exc:  # the manifest should still be useful if a model fails
        return None, str(exc)


def decode_reference(
    root: Path,
    video: str,
    *,
    out: str | None = None,
    scene_threshold: float = 0.25,
    transcribe: bool | Literal["auto"] = "auto",
    language: str | None = None,
    model: str = "small",
) -> ReferenceDecode:
    """Decode ``video`` and persist a reference manifest and reading assets."""

    project_root = Path(root).resolve()
    source_path = resolve_in_project(project_root, video)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    info = ffmpeg.probe(source_path)
    if info.kind != "video" or info.duration is None or info.duration <= 0:
        raise ValueError(f"reference must be a video with a positive duration: {source_path}")
    if info.width is None or info.height is None:
        raise ValueError(f"reference has no display dimensions: {source_path}")

    duration = float(info.duration)
    output_relative = out or f"reference/{_slug(source_path.stem)}"
    output_dir = resolve_in_project(project_root, output_relative)
    output_dir.mkdir(parents=True, exist_ok=True)
    keyframe_dir = output_dir / "keyframes"
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    ranges = _make_shot_ranges(source_path, duration, float(scene_threshold))

    soundtrack_path: Path | None = None
    rms: list[float] = []
    music: dict[str, object] | None = None
    if info.has_audio:
        soundtrack_path = output_dir / "soundtrack.m4a"
        ffmpeg.run(
            [
                "-i",
                str(source_path),
                "-map",
                "0:a:0",
                "-vn",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(soundtrack_path),
            ]
        )
        rms = rms_per_second(source_path)
        try:
            music = profile_audio(soundtrack_path)
        except Exception:  # the profile is a hint, never a reason to fail the decode
            music = None
        if music is not None:
            music["path"] = _relative(project_root, soundtrack_path)
            music["note"] = (
                "the whole soundtrack: it mixes voice and music when the reference speaks"
            )

    if transcribe not in {True, False, "auto"}:
        raise ValueError("transcribe must be true, false, or 'auto'")
    if transcribe == "auto" and not info.has_audio:
        transcript, transcript_status = None, "not attempted: no audio"
    elif transcribe == "auto" and (not rms or max(rms) <= -60.0):
        transcript, transcript_status = None, "not attempted: audio is near-silent"
    else:
        transcript, transcript_status = _transcribe(
            source_path,
            requested=transcribe is not False,
            language=language,
            model=model,
            output_dir=output_dir,
            project_root=project_root,
        )

    shots: list[RefShot] = []
    for index, (start, end) in enumerate(ranges, start=1):
        midpoint = min((start + end) / 2, max(0.0, duration - 0.001))
        keyframe_path = keyframe_dir / f"shot-{index:02d}.png"
        ffmpeg.extract_frame(source_path, midpoint, keyframe_path)
        text_boxes = [
            tuple(int(value) for value in box)
            for box in detect_text_boxes(source_path, start=start, duration=max(0.001, end - start))
        ]
        sample_start = max(start, midpoint - 0.5)
        sample_duration = max(0.001, min(end, midpoint + 0.5) - sample_start)
        shots.append(
            RefShot(
                index=index,
                start=start,
                end=end,
                duration=end - start,
                keyframe=_relative(project_root, keyframe_path),
                text_boxes=text_boxes,
                spoken=_spoken_for_shot(transcript, start, end),
                text_regions=_text_regions(text_boxes, height=info.height),
                yavg=ffmpeg.measure_yavg(
                    source_path,
                    start=sample_start,
                    duration=sample_duration,
                ),
            )
        )

    timestamps, shot_labels = _shot_timestamps(duration, ranges)
    sheet_path = output_dir / "sheet.jpg"
    contact_sheet(
        source_path,
        sheet_path,
        n=len(timestamps),
        cols=6,
        thumb_width=180,
        labels=True,
        timestamps=timestamps,
        shot_labels=shot_labels,
    )

    notes: list[str] = []
    if info.height > info.width:
        notes.append(f"portrait {info.width}x{info.height}")
    elif info.width > info.height:
        notes.append("landscape: will be cover-cropped")
    if not info.has_audio:
        notes.append("no audio")
    notes.extend(_text_notes(shots))
    notes.extend(
        f"shot {shot.index} is shorter than 0.8 s" for shot in shots if shot.duration < 0.8
    )
    notes.extend(_silence_notes(rms))

    shot_sheets = reference_shot_sheets(
        source_path,
        shots,
        output_dir,
        spoken=[shot.spoken for shot in shots],
    )
    pacing = _pacing(ranges, duration)
    breakdown_template = [
        {
            "shots": [shot.index],
            "start": shot.start,
            "end": shot.end,
            "type": None,
            "scale": None,
            "subject": None,
            "on_screen_text": None,
            "spoken": shot.spoken,
            "role": None,
            "motion": None,
            "replacement": None,
            "text_speaker": None,
            "text_device": None,
            "emotion": None,
            "action": None,
        }
        for shot in shots
    ]

    result = ReferenceDecode(
        source=_relative(project_root, source_path),
        duration=duration,
        width=info.width,
        height=info.height,
        fps=info.fps,
        has_audio=info.has_audio,
        shots=shots,
        sheet=_relative(project_root, sheet_path),
        soundtrack=_relative(project_root, soundtrack_path) if soundtrack_path else None,
        music_profile=music,
        rms_per_second=rms,
        transcript=transcript,
        transcript_status=transcript_status,
        pacing=pacing,
        shot_sheets=[_relative(project_root, path) for path in shot_sheets],
        breakdown_template=breakdown_template,
        notes=notes,
    )
    (output_dir / "reference.json").write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = ["BBox", "RefShot", "ReferenceDecode", "Word", "decode_reference"]
