"""Export a storyboard as an editable timeline for DaVinci Resolve or Premiere Pro.

``export_timeline`` writes ``exports/<name>/`` (``exports/<name>-relative/`` for relative paths,
``exports/<name>-portable/`` with copied media, so one mode never overwrites another):

* ``<name>.otio``: OpenTimelineIO JSON, written by hand (no dependency). Video track ``V1`` with
  the segments (clips with their source range and speed as a ``LinearTimeWarp``, images as
  stills), ``V2`` for the second picture of a split or reveal, an ``overlays`` track (the rendered
  PNGs, which are real media) and the audio tracks (clip sound, voice, music, and one track per
  sound effect). Captions are markers on ``V1`` (and in the SRT): there is no caption track of
  offline clips. A clip's ``available_range`` is the real media (its own frame rate, duration and
  start timecode); its ``source_range`` starts at the source ``in`` and lasts as long as the clip
  does on the timeline; ``time_scalar`` is the speed. ``target_url`` is a ``file://`` URL.
* ``<name>.xml``: Final Cut Pro 7 XML (xmeml version 4), read by Premiere Pro and DaVinci Resolve.
  Each ``<file>`` carries the real rate, size, duration and start timecode of its media; a clip
  item's ``in``/``out`` count frames of that media. A remapped clip has ``out - in = end - start``
  and a ``Time Remap`` filter that carries the speed. Music and sound effects have Audio Levels
  (a constant level, or keyframes for a fade).
* ``<name>.edl``: CMX 3600 of the video cuts only (the fallback). Record timecode starts at
  01:00:00:00 (Resolve's default timeline start); reel names are 8 characters at most; source
  timecodes start at the file's own start timecode. For a remapped clip the source out is the
  source in plus the record duration and the ``M2`` line carries the speed in frames per second.
* ``<name>.srt``: every caption (written, voice and speech captions) at its final time.
* ``overlays/`` and ``overlays.json``: each rendered caption, sticker and card as a full-canvas
  transparent PNG named after its storyboard overlay index (``overlay_03_caption.png``), with its
  start, end and ink box.
* ``README.txt``: the files, the numbering, and the ``not_exported`` list; ``relink.txt`` when the
  media paths are relative.

All times are on the frame grid of the storyboard (``format.fps``), exactly as the render snaps
them. Everything is a plain cut: the storyboard has no transitions.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import posixpath
import re
import shutil
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import quote
from xml.etree import ElementTree as ET

from PIL import Image

from adanvil.ffmpeg import probe, run_probe
from adanvil.media import (
    Cache,
    ClipSegment,
    RevealSegment,
    SplitSegment,
    segment_duration,
)
from adanvil.project import resolve_in_project
from adanvil.safeio import replace_bytes, replace_text
from adanvil.storyboard import Storyboard, load_storyboard
from adanvil.voice import load_track

FORMATS = ("otio", "fcp7xml", "edl", "srt")
EDL_START = "01:00:00:00"
EDL_START_FRAMES_HOUR = 3600
RELINK_ROOT = "RELINK_ROOT"
MEDIA_MODES = ("absolute", "relative", "portable")


class ExportError(ValueError):
    """The export cannot be made (unknown format, missing media, bad name)."""


# ------------------------------------------------------------------------------ the model


@dataclass
class Media:
    """What the file itself says: its real frame rate, size, duration and start timecode."""

    path: Path
    kind: str  # "video", "image" or "audio"
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    duration: float | None = None
    has_audio: bool = False
    timecode: str | None = None
    timecode_seconds: float = 0.0
    average_fps: float | None = None


@dataclass
class Item:
    """One placed piece: ``start`` and ``length`` on the timeline (frames of the storyboard
    rate), ``src_in`` in seconds of the source file and ``speed`` (the media used lasts
    ``length / fps * speed`` seconds)."""

    name: str
    media: Path | None
    start: int
    length: int
    label: str = ""
    src_in: float = 0.0
    speed: float = 1.0
    still: bool = False
    freeze: bool = False
    has_audio: bool = False
    info: Media | None = None
    level: float | None = None
    exact: float | None = None  # exact length in frames, when the file is shorter than its frames
    fade_in: int = 0
    fade_out: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Caption:
    text: str
    start: int
    end: int
    overlay: int | None = None


@dataclass
class OverlayFile:
    index: int
    kind: str
    start: int
    end: int
    png: str | None
    box: list[int] | None
    overlay: int | None
    note: str | None = None
    animated: bool = False


@dataclass
class Timeline:
    name: str
    fps: int
    width: int
    height: int
    total: int
    video: list[Item] = field(default_factory=list)
    second: list[Item] = field(default_factory=list)
    clip_audio: list[Item] = field(default_factory=list)
    music: list[Item] = field(default_factory=list)
    voice: list[Item] = field(default_factory=list)
    sfx: list[Item] = field(default_factory=list)
    captions: list[Caption] = field(default_factory=list)
    overlays: list[OverlayFile] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    segments: list[tuple[int, str, int, int]] = field(default_factory=list)
    licences: list[str] = field(default_factory=list)
    not_exported: list[str] = field(default_factory=list)

    def audio_groups(self) -> list[tuple[str, list[Item]]]:
        """The audio tracks in order: ``(label, items)``, one track per entry (each sound effect
        has its own track)."""

        groups: list[tuple[str, list[Item]]] = []
        if self.clip_audio:
            groups.append(("clip sound", self.clip_audio))
        if self.voice:
            groups.append(("voice", self.voice))
        if self.music:
            groups.append(("music", self.music))
        for number, item in enumerate(self.sfx, start=1):
            groups.append((f"sfx {item.label or number}: {item.name}", [item]))
        return groups


def _frames(seconds: float, fps: int) -> int:
    return int(round(seconds * fps))


def _timecode(frames: int, fps: int, hour: bool = False) -> str:
    if hour:
        frames += EDL_START_FRAMES_HOUR * fps
    seconds, frame = divmod(max(frames, 0), fps)
    minutes, second = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02d}:{minute:02d}:{second:02d}:{frame:02d}"


def _lanes(spans: list[tuple[int, int]]) -> list[int]:
    """The lane (0, 1, ...) of each span so that no two spans of a lane overlap."""

    ends: list[int] = []
    lanes: list[int] = []
    order = sorted(range(len(spans)), key=lambda number: spans[number])
    assigned = {}
    for number in order:
        start, end = spans[number]
        for lane, last in enumerate(ends):
            if last <= start:
                ends[lane] = end
                assigned[number] = lane
                break
        else:
            ends.append(end)
            assigned[number] = len(ends) - 1
    for number in range(len(spans)):
        lanes.append(assigned[number])
    return lanes


def _src_rate(item: Item, seq_fps: float) -> float:
    """The frame rate the item's in/out are counted in: the real rate of a video file, else the
    rate of the sequence (a still, an audio file)."""

    info = item.info
    if info is not None and info.kind == "video" and info.fps and not item.still:
        return float(info.fps)
    return float(seq_fps)


def _clip_range(item: Item, seq_fps: float) -> tuple[int, int]:
    """``(in, out)`` in frames of the item's own media rate. ``out - in`` is the record length in
    those frames, so for a file at the sequence rate it equals ``end - start`` (also when the clip
    is remapped: the speed is carried by the speed filter, as in a Premiere export)."""

    if item.still:
        return 0, item.length
    rate = _src_rate(item, seq_fps)
    first = int(round(item.src_in * rate))
    span = int(round(item.length / seq_fps * rate))
    return first, first + max(1, span)


def _media_frames(item: Item, seq_fps: float, total: int) -> int:
    """The length of the file in frames of its own rate."""

    info = item.info
    if item.still or info is None or info.kind == "image":
        return max(total, item.length, 1)
    rate = _src_rate(item, seq_fps)
    seconds = info.duration if info.duration else (item.src_in + item.length / seq_fps)
    return max(int(math.ceil(seconds * rate - 1e-6)), 1)


# --------------------------------------------------------------------------------- building

_TIMECODE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})[:;.](\d{2,3})$")


STANDARD_RATES = (24000 / 1001, 24.0, 25.0, 30000 / 1001, 30.0, 48.0, 50.0, 60000 / 1001, 60.0)


def _nominal_rate(average: float | None, base: float | None) -> float | None:
    """The frame rate of a file, by one rule that agrees with the library catalog (which keeps
    the probe's average rate): the average snapped to a standard rate when it is within 0.05 fps
    of one (29.99999 is 30, 29.97003 is 29.97). A variable-rate phone clip whose average is far
    from every standard rate (29.87) takes its declared base rate when that is standard, else the
    average rounded."""

    if average:
        near = min(STANDARD_RATES, key=lambda rate: abs(average - rate))
        if abs(average - near) <= 0.05:
            return near
    if base:
        near = min(STANDARD_RATES, key=lambda rate: abs(base - rate))
        if abs(base - near) / near < 0.005:
            return near
    if average:
        return round(average, 3)
    return base


def _fraction_rate(text: Any) -> float | None:
    try:
        top, _slash, bottom = str(text).partition("/")
        value = float(top) / float(bottom or 1)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def _read_stream_facts(path: Path, fps: float | None) -> tuple[str | None, float, float | None]:
    """``(timecode text, timecode seconds, base frame rate)`` of a video file: the start
    timecode is a ``timecode`` tag on a stream (including the ``tmcd`` track of a camera or phone
    file) or on the container."""

    try:
        raw = run_probe(
            [
                "-show_entries",
                "stream=codec_type,r_frame_rate:stream_tags=timecode:format_tags=timecode",
                "-of",
                "json",
                str(path),
            ]
        )
        data = json.loads(raw)
    except Exception:
        return None, 0.0, None
    candidates: list[str] = []
    base: float | None = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and base is None:
            base = _fraction_rate(stream.get("r_frame_rate"))
        value = (stream.get("tags") or {}).get("timecode")
        if value:
            candidates.append(str(value))
    value = (data.get("format", {}).get("tags") or {}).get("timecode")
    if value:
        candidates.append(str(value))
    rate = _nominal_rate(fps, base) or 30.0
    for text in candidates:
        found = _TIMECODE.match(text.strip())
        if found:
            hours, minutes, seconds, frames = (int(part) for part in found.groups())
            return text.strip(), hours * 3600 + minutes * 60 + seconds + frames / round(rate), base
    return None, 0.0, base


def _media_info(root: Path, source: str, cache: dict[str, Any]) -> tuple[Path, Media]:
    if source not in cache:
        path = resolve_in_project(root, source)
        try:
            info = probe(path)
        except Exception as exc:
            raise ExportError(f"cannot read media {source!r}: {exc}") from exc
        text, seconds, rate = (None, 0.0, info.fps)
        if info.kind == "video":
            text, seconds, base = _read_stream_facts(path, info.fps)
            rate = _nominal_rate(info.fps, base)
        media = Media(
            path=path,
            kind=info.kind,
            fps=rate,
            width=info.width,
            height=info.height,
            duration=info.duration,
            has_audio=bool(info.has_audio),
            timecode=text,
            timecode_seconds=seconds,
            average_fps=info.fps,
        )
        cache[source] = (path, media)
    return cache[source]


def _reframed(segment: Any) -> bool:
    return bool(
        getattr(segment, "face_focus", False)
        or tuple(segment.focus) != (0.5, 0.5)
        or segment.crop is not None
    )


def _list_numbers(numbers: list[int]) -> str:
    return ", ".join(str(number) for number in numbers)


def _fit_to_file(item: Item, fps: int, timeline: Timeline, number: int) -> list[Item]:
    """The item, or the item clamped to its file followed by a hold of the last frame. The render
    holds the last frame when a clip needs more source than the file has (``tpad`` clone); the
    export does the same with a freeze item, so no clip points past the end of its media."""

    info = item.info
    if item.still or item.freeze or info is None or info.kind != "video" or not info.duration:
        return [item]
    need = item.length / fps * item.speed
    room = info.duration - item.src_in
    if need <= room + 0.5 / fps:
        return [item]
    usable = max(1, int(math.floor(max(room, 0.0) / item.speed * fps + 1e-6)))
    if usable >= item.length:
        return [item]
    hold = item.length - usable
    held = Item(
        name=f"{item.name} (hold)",
        media=item.media,
        start=item.start + usable,
        length=hold,
        src_in=max(0.0, info.duration - 1.0 / fps),
        speed=0.0,
        freeze=True,
        info=info,
        meta={**item.meta, "hold_of_last_frame": True},
    )
    timeline.warnings.append(
        f"segment {number}: '{item.name}' needs {need:.3f}s of {Path(item.media).name} from "
        f"{item.src_in:g}s but the file has {max(room, 0.0):.3f}s: the render holds the last "
        f"frame for {hold / fps:.3f}s; the export clamps the clip to the file ({usable} frames) "
        f"and holds its last frame for {hold} frames (a freeze)"
    )
    return [replace(item, length=usable), held]


def _segment_items(
    root: Path, sb: Storyboard, timeline: Timeline, infos: dict[str, Any]
) -> list[dict[str, float]]:
    """Fill the video tracks and return the timeline (start, end in seconds) of each segment."""

    fps = sb.format.fps
    cursor = 0
    spans: list[dict[str, float]] = []
    flags: dict[str, list[int]] = {}

    def flag(feature: str, number: int) -> None:
        flags.setdefault(feature, []).append(number)

    for number, segment in enumerate(sb.segments):
        length = max(1, _frames(segment_duration(segment), fps))
        label = getattr(segment, "label", None)
        spans.append({"start": cursor / fps, "end": (cursor + length) / fps})
        timeline.segments.append((number, str(label or segment.type), cursor, length))
        if isinstance(segment, ClipSegment):
            path, info = _media_info(root, segment.source, infos)
            hold = _frames(segment.freeze.hold, fps) if segment.freeze else 0
            main = length - hold
            item = Item(
                name=label or Path(segment.source).stem,
                media=path,
                start=cursor,
                length=main,
                src_in=float(segment.in_),
                speed=float(segment.speed),
                has_audio=info.has_audio,
                info=info,
                meta={
                    "segment": number,
                    "in": segment.in_,
                    "out": segment.out,
                    "speed": segment.speed,
                },
            )
            timeline.video.extend(_fit_to_file(item, fps, timeline, number))
            if hold:
                last = max(0.0, segment.out - 1.0 / fps)
                timeline.video.append(
                    Item(
                        name=f"{item.name} (freeze)",
                        media=path,
                        start=cursor + main,
                        length=hold,
                        src_in=last,
                        speed=0.0,
                        freeze=True,
                        info=info,
                        meta={"segment": number, "freeze_hold": segment.freeze.hold},
                    )
                )
                flag("freeze zoom", number)
            if segment.gamma is not None:
                flag("gamma", number)
            if _reframed(segment):
                flag("focus", number)
            if segment.zoom is not None:
                flag("zoom", number)
            if segment.punch:
                flag("punch", number)
            if segment.remove_text is not None:
                flag("remove_text", number)
            if segment.audio_from is not None:
                flag("audio_from", number)
        elif isinstance(segment, (SplitSegment, RevealSegment)):
            kind = "split" if isinstance(segment, SplitSegment) else "reveal"
            panels = (
                list(segment.panels)
                if isinstance(segment, SplitSegment)
                else [segment.before, segment.after]
            )
            for slot, panel in enumerate(panels):
                path, info = _media_info(root, panel.source, infos)
                still = info.kind == "image"
                item = Item(
                    name=f"{label or kind} {'AB'[slot]}: {Path(panel.source).stem}",
                    media=path,
                    start=cursor,
                    length=length,
                    src_in=0.0 if still else float(panel.in_),
                    speed=1.0 if still else float(panel.speed),
                    still=still,
                    info=info,
                    meta={"segment": number, "composed": kind, "panel": slot},
                )
                target = timeline.video if slot == 0 else timeline.second
                target.extend(_fit_to_file(item, fps, timeline, number))
            timeline.warnings.append(
                f"segment {number} is a {kind}: its two pictures are on V1 and V2, "
                "the layout (split or sweep) has to be rebuilt in the editor"
            )
            flag(kind, number)
        else:  # an image segment
            path, info = _media_info(root, segment.source, infos)
            timeline.video.append(
                Item(
                    name=label or Path(segment.source).stem,
                    media=path,
                    start=cursor,
                    length=length,
                    still=True,
                    info=info,
                    meta={"segment": number, "motion": segment.motion},
                )
            )
            if segment.motion != "static":
                flag("image motion", number)
            if segment.fit == "contain" and segment.background == "blur" and info.width:
                ratio = (info.width / max(info.height or 1, 1)) / (
                    sb.format.width / max(sb.format.height, 1)
                )
                if abs(ratio - 1.0) > 0.02:
                    flag("blurred background", number)
            if _reframed(segment):
                flag("focus", number)
        cursor += length
    timeline.total = cursor
    descriptions = {
        "freeze zoom": "the zoom of a freeze frame is not applied (the frame is held as it is)",
        "gamma": "gamma and automatic grading are not applied (the editor shows the raw footage)",
        "focus": "focus, face and crop reframing: the editor scales each source to the frame "
        "without the render's smart crop",
        "zoom": "slow zoom on a clip",
        "punch": "punch-in zooms",
        "remove_text": "remove_text (mask, delogo, crop_out): the burnt-in text is still there",
        "audio_from": "audio_from (the sound of another clip under this picture): the clip "
        "sound of the segment is its own",
        "split": "split layout (two panels, gap and labels): panel A on V1 and B on V2 only",
        "reveal": "reveal sweep (divider, labels, hold): the two sides on V1 and V2 only",
        "image motion": "end-card or image zoom (motion zoom_in)",
        "blurred background": "blurred background behind an image shown whole (contain)",
    }
    for feature, numbers in flags.items():
        timeline.not_exported.append(f"{descriptions[feature]}: segments {_list_numbers(numbers)}")
    return spans


def _audio_items(
    root: Path, sb: Storyboard, timeline: Timeline, infos: dict[str, Any], spans: list[Any]
) -> None:
    from adanvil.sfx import resolve_sfx
    from adanvil.storyboard.sfx import all_sfx

    fps = sb.format.fps
    audio = sb.audio
    if audio.clips:
        for item in timeline.video:
            if item.media is not None and item.has_audio and not item.freeze and not item.still:
                timeline.clip_audio.append(item)
    written = audio.music
    shaped = bool(audio.voice or audio.clips)
    if written:
        path = resolve_in_project(root, written)
        if not path.is_file():
            timeline.warnings.append(f"music file {written} does not exist; left out")
        else:
            info = probe(path)
            begin = audio.start
            available = _frames(max((info.duration or 0.0) - begin, 0.0), fps)
            length = min(timeline.total, available)
            if length <= 0:
                timeline.warnings.append(f"no music remains at audio.start in {written}")
            else:
                if length < timeline.total:
                    timeline.warnings.append(
                        f"music {written} is shorter than the ad: the render loops it with a "
                        "crossfade, the export plays it once"
                    )
                level = audio.music_volume if shaped else audio.volume
                fade_in = 0.8 if shaped else audio.fade_in
                fade_out = 1.5 if shaped else audio.fade_out
                timeline.music.append(
                    Item(
                        name=Path(written).stem,
                        media=path,
                        start=0,
                        length=length,
                        src_in=float(begin),
                        info=Media(path, "audio", duration=info.duration),
                        level=float(level),
                        fade_in=min(_frames(fade_in, fps), length),
                        fade_out=min(_frames(fade_out, fps), length),
                        meta={
                            "level": level,
                            "fade_in": fade_in,
                            "fade_out": fade_out,
                            "loudness": audio.loudness,
                        },
                    )
                )
                timeline.not_exported.append(
                    "music volume and fades in OpenTimelineIO and EDL (the FCP7 XML carries them "
                    f"as Audio Levels: level {level:g}, fade in {fade_in:g}s, out {fade_out:g}s; "
                    "OTIO keeps them in the clip metadata)"
                )
                if audio.loudness is not None and not shaped:
                    timeline.not_exported.append(
                        f"music loudness normalisation to {audio.loudness:g} LUFS "
                        "(applied by the render after the mix)"
                    )
                if shaped:
                    timeline.not_exported.append(
                        "music reference level (the render brings the bed to -11 LUFS before "
                        "music_volume) and the ducking against the voice"
                    )
    if written:
        timeline.licences.extend(_music_licence_lines(root, written))
    if audio.from_reference:
        timeline.not_exported.append(
            f"audio.from_reference soundtrack ({audio.from_reference}): not placed in the timeline"
        )
    if audio.voice:
        try:
            track = load_track(resolve_in_project(root, audio.voice))
        except Exception as exc:
            timeline.warnings.append(f"voice manifest {audio.voice} not readable: {exc}")
            track = None
        if track is not None:
            offset = _frames(audio.voice_offset, fps)
            for line in track.lines:
                file = resolve_in_project(root, line.file)
                if not file.is_file():
                    timeline.warnings.append(f"voice line {line.index} file is missing")
                    continue
                length = max(1, _frames(line.duration, fps))
                timeline.voice.append(
                    Item(
                        name=f"voice {line.index}",
                        media=file,
                        start=offset + _frames(line.start, fps),
                        length=length,
                        info=Media(file, "audio", duration=line.duration),
                        meta={"text": line.text},
                    )
                )
            timeline.not_exported.append(
                f"voice loudness target ({audio.voice_loudness:g} LUFS): the lines are placed "
                "as generated"
            )
    # sound effects: one audio track each, at their time, with their volume
    starts = [float(span["start"]) for span in spans]
    total_seconds = timeline.total / fps
    try:
        sounds = all_sfx(sb, starts, total_seconds, root)
    except Exception as exc:
        timeline.warnings.append(f"sound effects could not be listed ({exc}); left out")
        sounds = []
    explicit = [
        (number, item)
        for number, item in enumerate(sb.overlays)
        if getattr(item, "type", "") == "sfx"
    ]
    for entry in sounds:
        at = float(entry["at"])
        owner = None
        if entry.get("auto") is None:
            for position, (number, item) in enumerate(explicit):
                if abs(item.at - at) < 1e-6 and abs(item.volume - float(entry["volume"])) < 1e-9:
                    owner = number
                    del explicit[position]
                    break
        if at >= total_seconds:
            continue
        try:
            file = resolve_sfx(root, entry["source"])
        except Exception:
            file = resolve_in_project(root, entry["source"])
        if not file.is_file():
            timeline.warnings.append(f"sound effect {entry['source']} is missing; left out")
            continue
        info = probe(file)
        begin = _frames(at, fps)
        wanted = max(1, int(math.floor((info.duration or 0.1) * fps + 1e-6)))
        length = min(wanted, timeline.total - begin)
        if length <= 0:
            continue
        timeline.sfx.append(
            Item(
                name=Path(file).stem,
                label=(
                    f"overlay {owner}" if owner is not None else f"auto {entry.get('auto')}"
                ),
                media=file,
                start=begin,
                length=length,
                info=Media(file, "audio", duration=info.duration),
                exact=min((info.duration or 0.0) * fps, timeline.total - begin),
                level=float(entry["volume"]),
                meta={"at": at, "volume": entry["volume"], "auto": entry.get("auto")},
            )
        )
    if timeline.sfx:
        timeline.not_exported.append(
            "sound effect limiter and the loudness the pack was set to (each sound is placed "
            "as the file is, at its volume); in OpenTimelineIO and the EDL the volume is "
            "metadata only (EDL: no audio at all)"
        )


def _music_licence_lines(root: Path, relative: str) -> list[str]:
    from adanvil.music import music_licence

    try:
        record = music_licence(root, relative)
    except Exception as exc:
        return [f"music {relative}: licence record not readable ({exc}); check it by hand"]
    if record is None:
        return []
    if record.get("status") != "recorded":
        return [f"music {relative}: NO LICENCE RECORD; check by hand that it may be used"]
    parts = (record.get("title"), record.get("author"), record.get("source"))
    who = ", ".join(str(part) for part in parts if part)
    line = f"music {relative}: {record.get('licence')}" + (f" ({who})" if who else "")
    if record.get("licence_url"):
        line += f" {record['licence_url']}"
    if record.get("attribution_required"):
        text = record.get("attribution_text")
        line += f". ATTRIBUTION REQUIRED, put in the ad description: {text}"
    return [line]


def _sfx_licence_lines(root: Path, timeline: Timeline) -> list[str]:
    names = sorted({Path(item.media).name for item in timeline.sfx if item.media is not None})
    ledger = root / "sfx" / "LICENCES.txt"
    if not names or not ledger.is_file():
        return [f"sound effect {name}: see sfx/LICENCES.txt of the project" for name in names]
    lines = ledger.read_text(encoding="utf-8", errors="replace").splitlines()
    found: list[str] = []
    for name in names:
        stem = Path(name).stem
        match = next((x.strip() for x in lines if stem in x or name in x), None)
        found.append(f"sound effect {name}: {match or 'no line in sfx/LICENCES.txt'}")
    return found


def _overlay_files(
    root: Path,
    sb: Storyboard,
    timeline: Timeline,
    spans: list[dict[str, float]],
    cache: Cache,
    out_dir: Path,
    write_pngs: bool,
) -> None:
    from adanvil.render.compose import build_overlays, snap_time

    fps = sb.format.fps
    total = timeline.total / fps
    warnings: list[str] = []
    _final, entries, _voice = build_overlays(
        sb, root=root, cache=cache, timeline=spans, total=total, warnings=warnings
    )
    timeline.warnings.extend(warnings)
    cuts = [span["start"] for span in spans[1:]]
    folder = out_dir / "overlays"
    if write_pngs:
        folder.mkdir(parents=True, exist_ok=True)
    per_overlay: dict[int, int] = {}
    for position, entry in enumerate(entries):
        key = entry.get("overlay")
        key = position if not isinstance(key, int) else key
        per_overlay[key] = per_overlay.get(key, 0) + 1
    seen: dict[int, int] = {}
    animated: dict[int, str] = {}
    inset: dict[int, str] = {}
    for number, entry in enumerate(entries):
        start = _frames(snap_time(float(entry["start"]), fps, cuts), fps)
        end = _frames(snap_time(float(entry["end"]), fps, cuts), fps)
        end = min(max(end, start + 1), timeline.total)
        text = entry.get("text")
        overlay_index = entry.get("overlay")
        label = overlay_index if isinstance(overlay_index, int) else number
        if text:
            timeline.captions.append(Caption(str(text), start, end, overlay_index))
        kind = str(entry.get("type", "overlay"))
        picture = _state_picture(entry, (start + end) / 2.0 / fps)
        if picture is None:
            timeline.overlays.append(
                OverlayFile(
                    number, kind, start, end, None, entry.get("bbox"), overlay_index,
                    note="no picture: drawn by the renderer (video inset or progress bar)",
                )
            )  # fmt: skip
            inset.setdefault(label, kind)
            continue
        states = entry.get("states")
        moving = isinstance(states, list) and len({str(s.get("png")) for s in states}) > 1
        if moving:
            animated.setdefault(label, kind)
        seen[label] = seen.get(label, 0) + 1
        suffix = f"_{seen[label]:02d}" if per_overlay.get(label, 1) > 1 else ""
        png, offset = picture
        name = f"overlay_{label:02d}_{kind}{suffix}.png"
        box = None
        if write_pngs:
            box = _write_overlay_png(png, offset, sb, folder / name)
        timeline.overlays.append(
            OverlayFile(
                number, kind, start, end, f"overlays/{name}", box, overlay_index, animated=moving
            )
        )
    if animated:
        timeline.not_exported.append(
            "entry animations (pop, karaoke, word, slide, fade, draw): each caption, sticker "
            "and card is flattened to its final state; overlays "
            + _list_numbers(sorted(animated))
        )
    if inset:
        timeline.not_exported.append(
            "video insets and progress bars (drawn by the renderer, no picture): overlays "
            + _list_numbers(sorted(inset))
        )


def _state_picture(entry: dict[str, Any], middle: float) -> tuple[Path, tuple[int, int]] | None:
    """The picture of a report entry on screen at ``middle`` (seconds) and its offset."""

    states = entry.get("states")
    if isinstance(states, list) and states:
        chosen = states[-1]
        for state in states:
            if float(state["start"]) <= middle < float(state["end"]):
                chosen = state
                break
        offset = chosen.get("offset") or [0, 0]
        return Path(chosen["png"]), (int(offset[0]), int(offset[1]))
    png = entry.get("png")
    if png is None:
        return None
    offset = entry.get("offset") or [0, 0]
    return Path(png), (int(offset[0]), int(offset[1]))


def _write_overlay_png(
    source: Path, offset: tuple[int, int], sb: Storyboard, target: Path
) -> list[int] | None:
    canvas = Image.new("RGBA", (sb.format.width, sb.format.height), (0, 0, 0, 0))
    with Image.open(source) as opened:
        picture = opened.convert("RGBA")
    canvas.paste(picture, offset, picture)
    box = canvas.getchannel("A").getbbox()
    encoded = io.BytesIO()
    canvas.save(encoded, format="PNG")
    replace_bytes(target, encoded.getvalue())  # never writes through a link planted at target
    picture.close()
    canvas.close()
    return list(box) if box else None


def _note_unmapped(sb: Storyboard, timeline: Timeline) -> None:
    """The things no editor format here can carry, listed once for the result and the README."""

    timeline.not_exported.append(
        "EDL: video cuts only (no audio, captions, overlays or second pictures)"
    )
    timeline.not_exported.append(
        "caption placement in OpenTimelineIO: captions are markers on V1 (and the SRT); the "
        "styled caption pictures are on the overlays track"
    )
    missing = sorted(
        {
            Path(item.media).name
            for item in timeline.video
            if item.info is not None
            and item.info.kind == "video"
            and item.info.timecode is None
        }
    )
    if missing:
        timeline.warnings.append(
            "no start timecode in "
            + ", ".join(missing[:6])
            + (f" and {len(missing) - 6} more" if len(missing) > 6 else "")
            + ": the EDL source times count from 00:00:00:00"
        )
    variable = sorted(
        {
            f"{Path(item.media).name} (average {item.info.average_fps:.2f} fps, declared "
            f"{item.info.fps:g})"
            for item in [*timeline.video, *timeline.second]
            if item.info is not None
            and item.info.kind == "video"
            and item.info.fps
            and item.info.average_fps
            and abs(item.info.average_fps - item.info.fps) / item.info.fps > 0.0002
        }
    )
    if variable:
        timeline.warnings.append(
            "variable frame rate (VFR) sources: "
            + ", ".join(variable)
            + ". An editor may drift on these over a long clip; for editing, transcode them to a "
            "constant frame rate first, at the rate declared for each file (for example "
            "ffmpeg -i in.mov -vf fps=<declared rate> -c:a copy out.mov)"
        )
    trimmed = [
        f"{item.name} at {item.start / timeline.fps:.3f}s ends "
        f"{(item.exact - item.length) / timeline.fps * 1000:.0f} ms early"
        for item in timeline.sfx
        if item.exact is not None and item.exact - item.length > 0.05
    ]
    if trimmed:
        timeline.warnings.append(
            "sound effects are whole frames in the XML (the OTIO keeps their exact length): "
            + "; ".join(trimmed)
        )
    odd = sorted(
        {
            f"{Path(item.media).name} ({item.info.fps:g} fps)"
            for item in [*timeline.video, *timeline.second]
            if item.info is not None
            and item.info.kind == "video"
            and item.info.fps
            and abs(item.info.fps - timeline.fps) > 0.001
        }
    )
    if odd:
        timeline.warnings.append(
            "sources at another frame rate than the sequence: "
            + ", ".join(odd)
            + ". The XML and OTIO count their in/out in the file's own frames; the EDL counts "
            "source times at the EDL rate"
        )


def build_timeline(
    root: Path,
    sb: Storyboard,
    *,
    out_dir: Path,
    cache: Cache,
    write_pngs: bool = True,
) -> Timeline:
    infos: dict[str, Any] = {}
    timeline = Timeline(
        name=sb.name,
        fps=sb.format.fps,
        width=sb.format.width,
        height=sb.format.height,
        total=0,
    )
    spans = _segment_items(root, sb, timeline, infos)
    _audio_items(root, sb, timeline, infos, spans)
    _overlay_files(root, sb, timeline, spans, cache, out_dir, write_pngs)
    _note_unmapped(sb, timeline)
    timeline.licences.extend(_sfx_licence_lines(root, timeline))
    return timeline


# ------------------------------------------------------------------------------- media refs


def _file_uri(text: str, host: str = "") -> str:
    """``file://`` URL of an absolute path written with forward slashes."""

    if len(text) > 1 and text[1] == ":":  # a Windows drive: file:///C:/dir/x
        return f"file://{host}/" + text[0] + ":" + quote(text[2:], safe="/")
    return f"file://{host}" + quote(text, safe="/")


class _Paths:
    """How a media path is written. ``absolute``: the project path as the project sees it (a
    symlink such as ``footage/`` is not resolved); ``relative``: relative to the export folder;
    ``portable``: the sources are copied into ``media/`` and written relative to it. With
    ``media_root`` a relative export is written as absolute paths under that root."""

    def __init__(
        self,
        out_dir: Path,
        relative: bool,
        copy_media: bool,
        media_root: str | None = None,
        media: list[Path] | None = None,
    ) -> None:
        self.out_dir = Path(os.path.abspath(out_dir))
        self.relative = relative or copy_media
        self.copy_media = copy_media
        self.media_root = media_root.replace("\\", "/").rstrip("/") if media_root else None
        self.copied: dict[Path, Path] = {}
        self.names: dict[Path, str] = {}
        by_name: dict[str, set[Path]] = {}
        for path in media or []:
            absolute = Path(os.path.abspath(path))
            by_name.setdefault(absolute.name.casefold(), set()).add(absolute)
        for _name, group in by_name.items():
            for absolute in group:
                if len(group) > 1:
                    digest = hashlib.sha1(absolute.as_posix().encode("utf-8")).hexdigest()[:6]
                    self.names[absolute] = f"{digest}_{absolute.name}"
                else:
                    self.names[absolute] = absolute.name

    def local(self, path: Path) -> Path:
        """Where the file is in the export: the original, or its copy in ``media/``."""

        absolute = Path(os.path.abspath(path))
        if self.copy_media and not self._inside(absolute):
            if absolute not in self.copied:
                folder = self.out_dir / "media"
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / self.names.get(absolute, absolute.name)
                if os.path.islink(target):
                    target.unlink()  # a planted link: the copy is made next to it, not through it
                if not os.path.lexists(target):
                    _copy_new_file(absolute, target)
                self.copied[absolute] = target
            return self.copied[absolute]
        return absolute

    def _inside(self, absolute: Path) -> bool:
        try:
            absolute.relative_to(self.out_dir)
        except ValueError:
            return False
        return True

    def _relative_text(self, path: Path) -> str:
        try:
            return Path(os.path.relpath(self.local(path), self.out_dir)).as_posix()
        except ValueError:  # another drive on Windows
            return self.local(path).as_posix()

    def of(self, path: Path) -> str:
        """The path text: absolute, or relative to the export folder."""

        if self.relative:
            relative = self._relative_text(path)
            if self.media_root:
                return posixpath.normpath(self.media_root + "/" + relative)
            return relative
        return self.local(path).as_posix()

    def _absolute_text(self, path: Path) -> str | None:
        if not self.relative:
            return self.local(path).as_posix()
        if self.media_root:
            return posixpath.normpath(self.media_root + "/" + self._relative_text(path))
        return None

    def otio_url(self, path: Path) -> str:
        """``target_url``: a ``file://`` URL; a relative export without ``media_root`` writes a
        relative URL reference (resolved against the folder of the OTIO file)."""

        text = self._absolute_text(path)
        if text is None:
            return quote(self._relative_text(path), safe="/")
        return _file_uri(text)

    def xml_url(self, path: Path) -> str:
        """``pathurl``: ``file://localhost/...``. Premiere Pro does not take a relative
        ``pathurl`` (clips come in offline), so a relative export writes a placeholder root
        (``RELINK_ROOT``) to replace, see ``relink.txt``."""

        text = self._absolute_text(path)
        if text is None:
            tail = quote(self._relative_text(path), safe="/")
            return f"file://localhost/{RELINK_ROOT}/{tail}"
        return _file_uri(text, "localhost")


# --------------------------------------------------------------------------------- OpenTimelineIO


def _rt(value: float, rate: float) -> dict[str, Any]:
    return {"OTIO_SCHEMA": "RationalTime.1", "rate": float(rate), "value": float(value)}


def _tr(start: float, duration: float, rate: float, duration_rate: float | None = None) -> dict:
    return {
        "OTIO_SCHEMA": "TimeRange.1",
        "duration": _rt(duration, duration_rate if duration_rate is not None else rate),
        "start_time": _rt(start, rate),
    }


def _otio_clip(item: Item, rate: int, paths: _Paths, total: int, *, audio: bool = False) -> dict:
    effects: list[dict[str, Any]] = []
    if item.freeze:
        effects.append(
            {
                "OTIO_SCHEMA": "FreezeFrame.1",
                "name": "Freeze",
                "effect_name": "FreezeFrame",
                "time_scalar": 0.0,
                "metadata": {},
            }
        )
    elif abs(item.speed - 1.0) > 1e-9 and not item.still:
        effects.append(
            {
                "OTIO_SCHEMA": "LinearTimeWarp.1",
                "name": "Speed",
                "effect_name": "LinearTimeWarp",
                "time_scalar": item.speed,
                "metadata": {},
            }
        )
    start_value: float
    start_rate: float = rate
    if item.media is None:
        reference: dict[str, Any] = {
            "OTIO_SCHEMA": "MissingReference.1",
            "name": "",
            "available_range": None,
            "available_image_bounds": None,
            "metadata": {},
        }
        start_value = 0
    else:
        media_rate = _src_rate(item, rate)
        count = _media_frames(item, rate, total)
        if item.info is not None and item.info.kind == "audio" and item.info.duration:
            count = item.info.duration * media_rate  # the real length, not rounded up
        begin = 0.0
        if item.info is not None and item.info.kind == "video" and not item.still:
            begin = item.info.timecode_seconds * media_rate
        reference = {
            "OTIO_SCHEMA": "ExternalReference.1",
            "name": item.media.name,
            "available_range": _tr(round(begin), count, media_rate),
            "available_image_bounds": None,
            "target_url": paths.otio_url(item.media),
            "metadata": {},
        }
        if item.still:
            start_value = 0
        else:
            first, _last = _clip_range(item, rate)
            start_value = round(begin) + first
            start_rate = media_rate
    metadata: dict[str, Any] = {"adanvil": dict(item.meta)}
    if item.info is not None and item.info.kind == "video" and item.info.fps:
        metadata["adanvil"]["source_fps"] = item.info.fps
        metadata["adanvil"]["source_size"] = [item.info.width, item.info.height]
        if item.info.timecode:
            metadata["adanvil"]["source_timecode"] = item.info.timecode
    if audio:
        metadata["adanvil"]["audio"] = True
    if item.level is not None:
        metadata["adanvil"]["level"] = item.level
        if item.fade_in or item.fade_out:
            metadata["adanvil"]["fade_in_frames"] = item.fade_in
            metadata["adanvil"]["fade_out_frames"] = item.fade_out
    return {
        "OTIO_SCHEMA": "Clip.1",
        "name": item.name,
        "source_range": _tr(
            start_value, item.exact if item.exact is not None else item.length, start_rate, rate
        ),
        "effects": effects,
        "markers": [],
        "media_reference": reference,
        "metadata": metadata,
        "enabled": True,
    }


def _gap(length: int, rate: int) -> dict[str, Any]:
    return {
        "OTIO_SCHEMA": "Gap.1",
        "name": "",
        "source_range": _tr(0, length, rate),
        "effects": [],
        "markers": [],
        "metadata": {},
        "enabled": True,
    }


def _track(
    name: str, kind: str, children: list[dict[str, Any]], markers: list[dict] | None = None
) -> dict[str, Any]:
    return {
        "OTIO_SCHEMA": "Track.1",
        "name": name,
        "kind": kind,
        "children": children,
        "source_range": None,
        "effects": [],
        "markers": markers or [],
        "metadata": {},
        "enabled": True,
    }


def _packed(items: list[tuple[int, int, dict[str, Any]]], rate: int) -> list[list[dict[str, Any]]]:
    """Lanes of children with gaps between, from ``(start, end, child)`` triples."""

    lanes = _lanes([(start, end) for start, end, _child in items])
    count = max(lanes, default=-1) + 1
    result: list[list[dict[str, Any]]] = []
    for lane in range(count):
        mine = sorted(
            (item for item, number in zip(items, lanes, strict=True) if number == lane),
            key=lambda item: item[0],
        )
        children: list[dict[str, Any]] = []
        cursor = 0
        for start, end, child in mine:
            if start > cursor:
                children.append(_gap(start - cursor, rate))
            children.append(child)
            cursor = end
        result.append(children)
    return result


def _overlay_name(overlay: OverlayFile) -> str:
    label = overlay.overlay if isinstance(overlay.overlay, int) else overlay.index
    return f"overlay {label} {overlay.kind}"


def otio_document(timeline: Timeline, paths: _Paths) -> dict[str, Any]:
    rate = timeline.fps
    total = timeline.total
    tracks: list[dict[str, Any]] = []

    def clips_track(
        name: str,
        kind: str,
        items: list[Item],
        *,
        audio: bool = False,
        markers: list[dict[str, Any]] | None = None,
    ) -> None:
        triples = [
            (
                item.start,
                item.start + item.length,
                _otio_clip(item, rate, paths, total, audio=audio),
            )
            for item in items
        ]
        for number, children in enumerate(_packed(triples, rate)):
            suffix = "" if number == 0 else f" {number + 1}"
            tracks.append(_track(name + suffix, kind, children, markers if number == 0 else None))

    # captions are markers on the first video track (and the SRT), never offline clips
    markers: list[dict[str, Any]] = []
    for caption in sorted(timeline.captions, key=lambda item: (item.start, item.end)):
        label = caption.overlay if isinstance(caption.overlay, int) else None
        markers.append(
            {
                "OTIO_SCHEMA": "Marker.2",
                "name": caption.text,
                "color": "GREEN",
                "comment": caption.text
                if label is None
                else f"overlay {label}: {caption.text}",
                "marked_range": _tr(caption.start, caption.end - caption.start, rate),
                "metadata": {"adanvil": {"text": caption.text, "overlay": caption.overlay}},
            }
        )
    # a single main track must be continuous: its items are already end to end
    clips_track("V1", "Video", timeline.video, markers=markers)
    if timeline.second:
        clips_track("V2 (second pictures)", "Video", timeline.second)
    ordered = sorted(timeline.overlays, key=lambda item: item.start)
    overlay_items: list[tuple[int, int, dict[str, Any]]] = []
    for overlay in ordered:
        if overlay.png is None:
            continue
        file = paths.out_dir / overlay.png
        child = {
            "OTIO_SCHEMA": "Clip.1",
            "name": _overlay_name(overlay),
            "source_range": _tr(0, overlay.end - overlay.start, rate),
            "effects": [],
            "markers": [],
            "media_reference": {
                "OTIO_SCHEMA": "ExternalReference.1",
                "name": Path(overlay.png).name,
                "available_range": _tr(0, overlay.end - overlay.start, rate),
                "available_image_bounds": None,
                "target_url": paths.otio_url(file),
                "metadata": {},
            },
            "metadata": {
                "adanvil": {
                    "overlay": overlay.overlay,
                    "box": overlay.box,
                    "canvas": [timeline.width, timeline.height],
                }
            },
            "enabled": True,
        }
        overlay_items.append((overlay.start, overlay.end, child))
    for number, children in enumerate(_packed(overlay_items, rate)):
        tracks.append(
            _track("overlays" + ("" if number == 0 else f" {number + 1}"), "Video", children)
        )
    for number, (label, items) in enumerate(timeline.audio_groups(), start=1):
        clips_track(f"A{number} ({label})", "Audio", items, audio=True)
    return {
        "OTIO_SCHEMA": "Timeline.1",
        "name": timeline.name,
        "global_start_time": _rt(EDL_START_FRAMES_HOUR * rate, rate),
        "metadata": {
            "adanvil": {
                "fps": rate,
                "width": timeline.width,
                "height": timeline.height,
                "duration_frames": timeline.total,
                "warnings": timeline.warnings,
                "not_exported": timeline.not_exported,
            }
        },
        "tracks": {
            "OTIO_SCHEMA": "Stack.1",
            "name": "tracks",
            "children": tracks,
            "source_range": None,
            "effects": [],
            "markers": [],
            "metadata": {},
            "enabled": True,
        },
    }


# ---------------------------------------------------------------------------- Final Cut XML


def _text(parent: ET.Element, tag: str, value: Any) -> ET.Element:
    element = ET.SubElement(parent, tag)
    element.text = str(value)
    return element


def _timebase(fps: float) -> tuple[int, bool]:
    """``(timebase, ntsc)`` of a frame rate: 29.97 is timebase 30 with ``ntsc`` TRUE."""

    near = int(round(fps))
    if abs(fps - near) < 0.01:
        return near, False
    if abs(fps - near * 1000 / 1001) < 0.01:
        return near, True
    return near, False


def _rate(parent: ET.Element, fps: float) -> None:
    rate = ET.SubElement(parent, "rate")
    base, ntsc = _timebase(fps)
    _text(rate, "timebase", base)
    _text(rate, "ntsc", "TRUE" if ntsc else "FALSE")


def _sample(
    parent: ET.Element, fps: float, width: int | None, height: int | None
) -> None:
    sample = ET.SubElement(parent, "samplecharacteristics")
    _rate(sample, fps)
    _text(sample, "width", width or 0)
    _text(sample, "height", height or 0)
    _text(sample, "anamorphic", "FALSE")
    _text(sample, "pixelaspectratio", "square")
    _text(sample, "fielddominance", "none")


class _Files:
    """``<file>`` elements: the first use of a path writes it, the next ones point to it."""

    def __init__(self, timeline: Timeline, paths: _Paths) -> None:
        self.timeline = timeline
        self.paths = paths
        self.ids: dict[Path, str] = {}

    def add(self, parent: ET.Element, item: Item, path: Path) -> None:
        if path in self.ids:
            ET.SubElement(parent, "file", {"id": self.ids[path]})
            return
        seq = self.timeline
        info = item.info
        identifier = f"file-{len(self.ids) + 1}"
        self.ids[path] = identifier
        file = ET.SubElement(parent, "file", {"id": identifier})
        _text(file, "name", path.name)
        _text(file, "pathurl", self.paths.xml_url(path))
        is_video = info is not None and info.kind == "video" and not item.still
        rate = _src_rate(item, seq.fps)
        _rate(file, rate)
        _text(file, "duration", _media_frames(item, seq.fps, seq.total))
        code = ET.SubElement(file, "timecode")
        _rate(code, rate)
        seconds = info.timecode_seconds if info is not None and is_video else 0.0
        frames = int(round(seconds * rate))
        base = max(_timebase(rate)[0], 1)
        whole, rest = divmod(frames, base)
        minutes, second = divmod(whole, 60)
        hours, minute = divmod(minutes, 60)
        _text(code, "string", f"{hours:02d}:{minute:02d}:{second:02d}:{rest:02d}")
        _text(code, "frame", frames)
        _text(code, "displayformat", "NDF")
        media = ET.SubElement(file, "media")
        if info is None or info.kind in {"video", "image"}:
            width = info.width if info is not None and info.width else seq.width
            height = info.height if info is not None and info.height else seq.height
            _sample(ET.SubElement(media, "video"), rate, width, height)
        if info is None or info.kind == "audio" or (info.kind == "video" and info.has_audio):
            node = ET.SubElement(media, "audio")
            sample_audio = ET.SubElement(node, "samplecharacteristics")
            _text(sample_audio, "depth", 16)
            _text(sample_audio, "samplerate", 48000)
            _text(node, "channelcount", 2)


def _level_effect(parent: ET.Element, item: Item, fps: int) -> None:
    """Audio Levels of a clip item: a constant level, or keyframes when it fades."""

    filter_node = ET.SubElement(parent, "filter")
    effect = ET.SubElement(filter_node, "effect")
    _text(effect, "name", "Audio Levels")
    _text(effect, "effectid", "audiolevels")
    _text(effect, "effectcategory", "audiolevels")
    _text(effect, "effecttype", "audiolevels")
    _text(effect, "mediatype", "audio")
    parameter = ET.SubElement(effect, "parameter", {"authoringApp": "PremierePro"})
    _text(parameter, "parameterid", "level")
    _text(parameter, "name", "Level")
    _text(parameter, "valuemin", 0)
    _text(parameter, "valuemax", 3.98109)
    level = float(item.level if item.level is not None else 1.0)
    _text(parameter, "value", round(level, 6))
    points: list[tuple[int, float]] = []
    if item.fade_in:
        points += [(0, 0.0), (item.fade_in, level)]
    if item.fade_out:
        points += [(max(item.length - item.fade_out, item.fade_in), level), (item.length, 0.0)]
    for when, value in points:
        key = ET.SubElement(parameter, "keyframe")
        _text(key, "when", when)
        _text(key, "value", round(value, 6))


def _clipitem(
    track: ET.Element,
    item: Item,
    number: int,
    files: _Files,
    timeline: Timeline,
    *,
    audio: bool,
    name: str | None = None,
) -> None:
    node = ET.SubElement(track, "clipitem", {"id": f"{'a' if audio else 'v'}clip-{number}"})
    _text(node, "name", name or item.name)
    _text(node, "enabled", "TRUE")
    if "segment" in item.meta:
        comments = ET.SubElement(node, "comments")
        _text(comments, "mastercomment1", f"segment {item.meta['segment']} (counted from 0)")
    _text(node, "duration", _media_frames(item, timeline.fps, timeline.total))
    _rate(node, _src_rate(item, timeline.fps))
    _text(node, "start", item.start)
    _text(node, "end", item.start + item.length)
    first, last = _clip_range(item, timeline.fps)
    _text(node, "in", first)
    _text(node, "out", last)
    if item.still:
        _text(node, "stillframe", "TRUE")
    if item.media is not None:
        files.add(node, item, files.paths.local(item.media))
    if not audio and (item.freeze or (abs(item.speed - 1.0) > 1e-9 and not item.still)):
        filter_node = ET.SubElement(node, "filter")
        effect = ET.SubElement(filter_node, "effect")
        _text(effect, "name", "Time Remap")
        _text(effect, "effectid", "timeremap")
        _text(effect, "effectcategory", "motion")
        _text(effect, "effecttype", "motion")
        _text(effect, "mediatype", "video")
        for identifier, value in (
            ("speed", round(item.speed * 100, 4)),
            ("reverse", "FALSE"),
            ("frameblending", "FALSE"),
        ):
            parameter = ET.SubElement(effect, "parameter")
            _text(parameter, "parameterid", identifier)
            _text(parameter, "name", identifier)
            _text(parameter, "value", value)
    if audio:
        source = ET.SubElement(node, "sourcetrack")
        _text(source, "mediatype", "audio")
        _text(source, "trackindex", 1)
        if item.level is not None:
            _level_effect(node, item, timeline.fps)


def xml_document(timeline: Timeline, paths: _Paths) -> str:
    root = ET.Element("xmeml", {"version": "4"})
    sequence = ET.SubElement(root, "sequence", {"id": "sequence-1"})
    _text(sequence, "name", timeline.name)
    _text(sequence, "duration", timeline.total)
    _rate(sequence, timeline.fps)
    code = ET.SubElement(sequence, "timecode")
    _rate(code, timeline.fps)
    _text(code, "string", EDL_START)
    _text(code, "frame", EDL_START_FRAMES_HOUR * timeline.fps)
    _text(code, "displayformat", "NDF")
    media = ET.SubElement(sequence, "media")
    video = ET.SubElement(media, "video")
    _sample(ET.SubElement(video, "format"), timeline.fps, timeline.width, timeline.height)
    files = _Files(timeline, paths)
    counter = 0

    def video_track(items: list[Item]) -> None:
        nonlocal counter
        track = ET.SubElement(video, "track")
        for item in items:
            counter += 1
            _clipitem(track, item, counter, files, timeline, audio=False)

    video_track(timeline.video)
    if timeline.second:
        video_track(timeline.second)
    # overlay pictures: each is its own clip item on the first track it fits, as a still
    pictures = [overlay for overlay in timeline.overlays if overlay.png]
    lanes = _lanes([(overlay.start, overlay.end) for overlay in pictures])
    for lane in range(max(lanes, default=-1) + 1):
        track = ET.SubElement(video, "track")
        for overlay, number in zip(pictures, lanes, strict=True):
            if number != lane:
                continue
            counter += 1
            assert overlay.png is not None
            _clipitem(
                track,
                Item(
                    name=_overlay_name(overlay),
                    media=paths.out_dir / overlay.png,
                    start=overlay.start,
                    length=overlay.end - overlay.start,
                    still=True,
                    info=Media(
                        paths.out_dir / overlay.png,
                        "image",
                        width=timeline.width,
                        height=timeline.height,
                    ),
                ),
                counter,
                files,
                timeline,
                audio=False,
            )
    audio = ET.SubElement(media, "audio")
    _text(audio, "numOutputChannels", 2)
    format_node = ET.SubElement(audio, "format")
    sample = ET.SubElement(format_node, "samplecharacteristics")
    _text(sample, "depth", 16)
    _text(sample, "samplerate", 48000)
    for _label, group in timeline.audio_groups():
        track = ET.SubElement(audio, "track")
        for item in group:
            counter += 1
            _clipitem(track, item, counter, files, timeline, audio=True)
    for caption in sorted(timeline.captions, key=lambda item: item.start):
        marker = ET.SubElement(sequence, "marker")
        _text(marker, "name", caption.text)
        _text(
            marker,
            "comment",
            caption.text
            if not isinstance(caption.overlay, int)
            else f"overlay {caption.overlay}: {caption.text}",
        )
        _text(marker, "in", caption.start)
        _text(marker, "out", caption.end)
    ET.indent(root)
    body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n' + body + "\n"


# ------------------------------------------------------------------------------------ EDL


def _reel_names(timeline: Timeline) -> dict[Path, str]:
    names: dict[Path, str] = {}
    used: set[str] = set()
    for item in [*timeline.video]:
        if item.media is None or item.media in names:
            continue
        stem = "".join(ch for ch in item.media.stem.upper() if ch.isalnum() or ch == "_")
        reel = (stem or "AX")[:8]
        number = 1
        while reel in used:
            number += 1
            suffix = str(number)
            reel = (stem or "AX")[: 8 - len(suffix)] + suffix
        used.add(reel)
        names[item.media] = reel
    return names


def edl_document(timeline: Timeline, paths: _Paths | None = None) -> str:
    fps = timeline.fps
    reels = _reel_names(timeline)
    lines = [f"TITLE: {timeline.name}", "FCM: NON-DROP FRAME", ""]
    for number, item in enumerate(timeline.video, start=1):
        reel = reels.get(item.media, "AX") if item.media else "AX"
        # source timecode: the file's own start timecode plus the in point, counted at the EDL
        # rate; the source out is the source in plus the record duration (the CMX convention
        # for a clip with a speed change, whose M2 line carries the speed)
        if item.still:
            source_in = 0
        else:
            offset = item.info.timecode_seconds if item.info is not None else 0.0
            source_in = int(round((offset + item.src_in) * fps))
        source_out = source_in + item.length
        record_in = item.start
        record_out = item.start + item.length
        lines.append(
            f"{number:03d}  {reel:<8} V     C        "
            f"{_timecode(source_in, fps)} {_timecode(source_out, fps)} "
            f"{_timecode(record_in, fps, True)} {_timecode(record_out, fps, True)}"
        )
        if item.freeze:
            lines.append(f"M2   {reel:<8} 000.0                {_timecode(source_in, fps)}")
        elif abs(item.speed - 1.0) > 1e-9 and not item.still:
            speed_fps = item.speed * _src_rate(item, fps)
            lines.append(
                f"M2   {reel:<8} {speed_fps:05.1f}                {_timecode(source_in, fps)}"
            )
        if item.media is not None:
            lines.append(f"* FROM CLIP NAME: {item.media.name}")
            where = paths.of(item.media) if paths is not None else item.media.as_posix()
            lines.append(f"* SOURCE FILE: {where}")
        rate = _src_rate(item, fps)
        if abs(rate - fps) > 0.001:
            lines.append(
                f"* ADANVIL: source is {rate:g} fps; source times are counted at {fps} fps"
            )
        if "segment" in item.meta:
            lines.append(f"* ADANVIL: segment {item.meta['segment']} (counted from 0)")
        if item.meta.get("composed"):
            lines.append(f"* ADANVIL: {item.meta['composed']} segment, panel A only")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------ SRT


def _srt_time(frames: int, fps: int) -> str:
    milliseconds = int(round(frames * 1000 / fps))
    seconds, milli = divmod(milliseconds, 1000)
    minutes, second = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02d}:{minute:02d}:{second:02d},{milli:03d}"


def _ordered_captions(timeline: Timeline) -> list[Caption]:
    return sorted(timeline.captions, key=lambda item: (item.start, item.end))


def srt_document(timeline: Timeline) -> str:
    blocks = []
    for number, caption in enumerate(_ordered_captions(timeline), start=1):
        blocks.append(
            f"{number}\n{_srt_time(caption.start, timeline.fps)} --> "
            f"{_srt_time(caption.end, timeline.fps)}\n{caption.text}\n"
        )
    return "\n".join(blocks)


# --------------------------------------------------------------------------------- README


def readme_document(timeline: Timeline, mode: str, written: dict[str, str]) -> str:
    fps = timeline.fps
    lines = [
        f"AdAnvil timeline export: {timeline.name}",
        "=" * (26 + len(timeline.name)),
        "",
        f"Media mode: {mode}",
        f"Sequence: {timeline.width}x{timeline.height}, {fps} fps, "
        f"{timeline.total} frames ({timeline.total / fps:.3f} s)",
        "",
        "Files",
        "-----",
    ]
    names = {
        "otio": "OpenTimelineIO (DaVinci Resolve: File > Import > Timeline)",
        "fcp7xml": "Final Cut Pro 7 XML (Premiere Pro and DaVinci Resolve)",
        "edl": "CMX 3600 EDL, video cuts only (the fallback)",
        "srt": "every caption at its final time",
        "overlays": "position, time and storyboard overlay of each caption picture",
        "relink": "how to relink the media of this folder (Premiere needs one relink)",
    }
    for kind, file in written.items():
        lines.append(f"  {file}: {names.get(kind, kind)}")
    lines += [
        "  overlays/: each caption, sticker and card as a full-canvas transparent PNG "
        "(place at 0, 0)",
        "",
        "Numbering",
        "---------",
        "Overlays carry the index they have in the storyboard (`overlays[N]`, counted from 0, "
        "sound effects included): overlays/overlay_NN_<type>.png, overlays.json, the names of "
        "the clips and markers in the OTIO and the XML, and the QC crops (overlay_N_<text>.jpg). "
        "One overlay can have several pictures (voice captions come in chunks): "
        "overlay_NN_<type>_01.png, _02.png...",
        "",
    ]
    captions = _ordered_captions(timeline)
    if captions:
        lines += ["SRT block -> storyboard overlay", "--------------------------------"]
        for number, caption in enumerate(captions, start=1):
            overlay = "?" if caption.overlay is None else str(caption.overlay)
            lines.append(f"  {number}: overlay {overlay}: {caption.text.replace(chr(10), ' / ')}")
        lines.append("")
    lines += [
        "Segments (counted from 0 everywhere: warnings, EDL comments, XML comments, OTIO metadata)",
        "-" * 88,
    ]
    events: dict[int, list[int]] = {}
    for event, item in enumerate(timeline.video, start=1):
        if "segment" in item.meta:
            events.setdefault(int(item.meta["segment"]), []).append(event)
    for number, label, start, length in timeline.segments:
        found = events.get(number)
        where = (
            "EDL event " + "-".join(f"{event:03d}" for event in (found[0], found[-1]))
            if found and len(found) > 1
            else f"EDL event {found[0]:03d}"
            if found
            else "not in the EDL"
        )
        lines.append(
            f"  segment {number}: {label} (record frames {start}-{start + length}, {where})"
        )
    lines += [
        "",
        f"Timecode: the sequence, the EDL, the XML and the OTIO all start at {EDL_START}.",
        "Sound-effect files: the XML declares each file's duration rounded up to whole frames "
        "(the format counts in frames); the clips themselves stay inside the file.",
        "",
    ]
    lines += ["Audio licences and attribution (keep this with the finished ad)", "-" * 62]
    lines += [f"  - {item}" for item in timeline.licences] or ["  (no audio file)"]
    lines.append("")
    lines += ["NOT EXPORTED (the edit in the editor will differ here)", "-" * 54]
    lines += [f"  - {item}" for item in timeline.not_exported] or ["  (nothing)"]
    lines += ["", "Warnings", "--------"]
    lines += [f"  - {item}" for item in timeline.warnings] or ["  (none)"]
    lines.append("")
    return "\n".join(lines)


def relink_document(paths: _Paths) -> str:
    return (
        "Relinking the media of this export\n"
        "==================================\n\n"
        "The media paths in the OpenTimelineIO file and in the EDL (* SOURCE FILE lines) are\n"
        "relative to this folder.\n"
        "The Final Cut XML needs absolute URLs: no Adobe documentation describes relative\n"
        "pathurl support in Premiere Pro, whose projects store absolute paths and relink with\n"
        "Link Media (Adobe Community, 'Relative paths in Premiere Pro': relink one file and\n"
        "Premiere finds the rest relative to it). So the XML carries\n"
        f"file://localhost/{RELINK_ROOT}/<path> instead, and the clips come in offline until\n"
        "you do one of the following.\n\n"
        f"1. Either replace {RELINK_ROOT} in the .xml with the absolute path of this folder\n"
        "   (for example Users/me/exports/name, with no leading slash after localhost/),\n"
        "   with a text editor, before importing;\n"
        "2. or import as is and use Link Media (Premiere Pro) / Relink Selected Clips (Resolve)\n"
        "   and point it at this folder;\n"
        "3. or export again with the media_root argument (CLI: --media-root) set to the\n"
        "   absolute folder where the export will live.\n\n"
        "Option 2 is one operation: Link Media on a single clip, then 'Locate files' finds the\n"
        "others in the same folder structure.\n"
        "Not verified in Premiere Pro or Resolve on the build machine.\n"
    )


# --------------------------------------------------------------------------------- the call


MANIFEST = "export_manifest.json"
_LEGACY_FILES = (".otio", ".xml", ".edl", ".srt")


def _real_inside(path: Path, folder: Path) -> bool:
    """Whether the real path of ``path`` is ``folder`` or below it (links resolved)."""

    inner = os.path.normcase(os.path.realpath(path))
    outer = os.path.normcase(os.path.realpath(folder))
    try:
        return os.path.commonpath((inner, outer)) == outer
    except ValueError:  # another drive
        return False


def _is_link(path: Path) -> bool:
    junction = getattr(os.path, "isjunction", None)
    return os.path.islink(path) or bool(junction and junction(path))


def _plain_relative(relative: object) -> tuple[str, ...] | None:
    """The parts of a name the manifest lists, or ``None`` when it is not a plain relative path:
    absolute, a drive or UNC prefix, a ``..`` or hidden component, a control character."""

    if not isinstance(relative, str) or not relative or any(ord(ch) < 32 for ch in relative):
        return None
    posix = PurePosixPath(relative.replace("\\", "/"))
    windows = PureWindowsPath(relative)
    if posix.is_absolute() or windows.is_absolute() or windows.drive or windows.root:
        return None
    parts = posix.parts
    if not parts or any(part.startswith(".") or ":" in part for part in parts):
        return None
    return parts


def _remove_file(folder: Path, relative: str) -> None:
    """Delete one file an earlier export wrote, named by its manifest. The manifest is a file of
    the project anyone can plant, so an entry that is not a plain relative path inside the export
    folder deletes nothing; a link is removed itself, never the file it points to."""

    parts = _plain_relative(relative)
    if parts is None:
        return
    target = Path(os.path.abspath(folder)).joinpath(*parts)
    if not _real_inside(target.parent, folder):
        return  # a link in the middle of the path leads out of the export folder
    if target.is_symlink() or target.is_file():
        try:
            target.unlink()
        except OSError:
            pass  # e.g. a link to a folder on Windows: left alone


def _prune(folder: Path) -> None:
    for name in ("overlays", "media"):
        sub_folder = folder / name
        if _is_link(sub_folder):
            continue
        if sub_folder.is_dir() and not any(sub_folder.iterdir()):
            sub_folder.rmdir()


def _copy_new_file(source: Path, target: Path) -> None:
    """Copy ``source`` to ``target`` through a new file renamed into place: a link planted at
    ``target`` is replaced, never followed. Media can be large, so it is copied, not read."""

    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _prepare_folder(folder: Path, root: Path, mode: str, overwrite: bool) -> None:
    """Clean what an earlier export of this folder wrote (the files its manifest lists). A
    folder that holds files but no manifest (made by an older version, or not an export) is
    refused unless ``overwrite``, which clears only the files an export writes (the known
    names, ``overlays/*.png`` and the files of ``media/``), nothing else."""

    if not folder.is_dir() or not any(folder.iterdir()):
        return
    for protected in (root, root.parent, folder.anchor and Path(folder.anchor)):
        if protected and folder.absolute() == Path(protected).absolute():
            raise ExportError(f"{folder} is not an export folder")
    manifest = folder / MANIFEST
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        previous = data.get("mode")
        if previous != mode and not overwrite:
            raise ExportError(
                f"{folder} already holds a {previous!r} export: choose another folder or pass "
                "overwrite=true to replace it"
            )
        listed = data.get("files")
        for relative in listed if isinstance(listed, list) else []:
            _remove_file(folder, relative)
        _prune(folder)
        return
    if not overwrite:
        raise ExportError(
            f"{folder} holds files but no {MANIFEST} (an export of an older version, or "
            "another folder): choose an empty folder or pass overwrite=true to clear its export "
            "files (.otio, .xml, .edl, .srt, overlays.json, README.txt, relink.txt, overlays/*.png "
            "and media/)"
        )
    for entry in list(folder.iterdir()):
        if entry.is_file() and (
            entry.suffix in _LEGACY_FILES
            or entry.name in {"overlays.json", "README.txt", "relink.txt"}
        ):
            entry.unlink()  # a link is removed itself, not the file it points to
    for name, only in (("overlays", ".png"), ("media", None)):
        sub_folder = folder / name
        # a sub-folder that is a link, or leads out of the export folder, is not cleared
        if sub_folder.is_dir() and not _is_link(sub_folder) and _real_inside(sub_folder, folder):
            for entry in list(sub_folder.iterdir()):
                if entry.is_file() and (only is None or entry.suffix == only):
                    entry.unlink()
    _prune(folder)


def _files_in(folder: Path) -> set[str]:
    return {
        path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()
    }


def _mode_of(folder: Path) -> str | None:
    readme = folder / "README.txt"
    if not readme.is_file():
        return None
    for line in readme.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Media mode:"):
            return line.split(":", 1)[1].strip()
    return None


def _all_media(timeline: Timeline) -> list[Path]:
    seen: list[Path] = []
    for item in [*timeline.video, *timeline.second, *timeline.music, *timeline.voice]:
        if item.media is not None and item.media not in seen:
            seen.append(item.media)
    for item in timeline.sfx:
        if item.media is not None and item.media not in seen:
            seen.append(item.media)
    return seen


def export_timeline(
    root: Path,
    storyboard: dict[str, Any] | Storyboard,
    *,
    formats: list[str] | None = None,
    relative: bool = False,
    copy_media: bool = False,
    media_root: str | None = None,
    overwrite: bool = False,
    out_dir: Path | None = None,
    cache_dir: Path | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Write ``exports/<name>/`` (``-relative`` or ``-portable`` for the other media modes, or
    ``out_dir``) for ``storyboard``; returns what was written, the ``warnings`` and the
    ``not_exported`` list. An existing export of another media mode in the same folder is only
    replaced with ``overwrite=True``."""

    root = Path(root)
    chosen = list(formats) if formats else list(FORMATS)
    unknown = [item for item in chosen if item not in FORMATS]
    if unknown:
        raise ExportError(f"unknown format(s) {unknown}: choose from {list(FORMATS)}")
    sb = storyboard if isinstance(storyboard, Storyboard) else load_storyboard(storyboard)
    title = name or sb.name
    if not title or any(ch in title for ch in '/\\:*?"<>|') or title in {".", ".."}:
        raise ExportError(f"storyboard name {title!r} cannot name a folder")
    mode = "portable" if copy_media else ("relative" if relative else "absolute")
    if out_dir is not None:
        folder = Path(out_dir)
    else:
        suffix = {"absolute": "", "relative": "-relative", "portable": "-portable"}[mode]
        folder = root / "exports" / f"{title}{suffix}"
    if (
        out_dir is None
        and os.path.lexists(folder)
        and (_is_link(folder) or not _real_inside(folder, root / "exports"))
    ):
        raise ExportError(
            f"{folder} is a link or leads out of the project's exports/ folder: remove it or "
            "choose another storyboard name"
        )
    _prepare_folder(folder, root, mode, overwrite)
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("overlays", "media"):
        sub_folder = folder / name
        if os.path.lexists(sub_folder) and (
            _is_link(sub_folder) or not _real_inside(sub_folder, folder)
        ):
            raise ExportError(f"{sub_folder} is a link: remove it, an export writes real files")
    foreign = _files_in(folder)
    cache = Cache(Path(cache_dir) if cache_dir is not None else root / "cache")
    wants_pictures = "otio" in chosen or "fcp7xml" in chosen
    timeline = build_timeline(root, sb, out_dir=folder, cache=cache, write_pngs=wants_pictures)
    timeline.name = title
    paths = _Paths(folder, relative, copy_media, media_root, _all_media(timeline))
    written: dict[str, str] = {}
    if "otio" in chosen:
        target = folder / f"{title}.otio"
        replace_text(
            target,
            json.dumps(otio_document(timeline, paths), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written["otio"] = target.name
    if "fcp7xml" in chosen:
        target = folder / f"{title}.xml"
        replace_text(target, xml_document(timeline, paths), encoding="utf-8")
        written["fcp7xml"] = target.name
    if "edl" in chosen:
        target = folder / f"{title}.edl"
        replace_text(target, edl_document(timeline, paths), encoding="utf-8")
        written["edl"] = target.name
    if "srt" in chosen:
        target = folder / f"{title}.srt"
        replace_text(target, srt_document(timeline), encoding="utf-8")
        written["srt"] = target.name
    if wants_pictures:
        fps = timeline.fps
        listing = {
            "canvas": [timeline.width, timeline.height],
            "fps": fps,
            "numbering": "overlay = the index of the overlay in the storyboard (from 0, sound "
            "effects included)",
            "overlays": [
                {
                    "overlay": overlay.overlay if isinstance(overlay.overlay, int) else None,
                    "name": _overlay_name(overlay),
                    "file": overlay.png,
                    "type": overlay.kind,
                    "start": round(overlay.start / fps, 4),
                    "end": round(overlay.end / fps, 4),
                    "start_frame": overlay.start,
                    "end_frame": overlay.end,
                    "x": overlay.box[0] if overlay.box else 0,
                    "y": overlay.box[1] if overlay.box else 0,
                    "width": overlay.box[2] - overlay.box[0] if overlay.box else 0,
                    "height": overlay.box[3] - overlay.box[1] if overlay.box else 0,
                    "position_note": "x, y, width, height are the ink box on the canvas; the "
                    "PNG itself is full canvas, place it at 0, 0",
                    "storyboard_overlay": overlay.overlay,
                    "animated": overlay.animated,
                    **({"note": overlay.note} if overlay.note else {}),
                }
                for overlay in timeline.overlays
            ],
        }
        replace_text(
            folder / "overlays.json",
            json.dumps(listing, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written["overlays"] = "overlays.json"
    replace_text(folder / "README.txt", readme_document(timeline, mode, written), encoding="utf-8")
    written["readme"] = "README.txt"
    if paths.relative and not media_root:
        replace_text(folder / "relink.txt", relink_document(paths), encoding="utf-8")
        written["relink"] = "relink.txt"
    owned = sorted(_files_in(folder) - foreign - {MANIFEST})
    replace_text(
        folder / MANIFEST,
        json.dumps(
            {"adanvil_export": 2, "name": title, "mode": mode, "files": owned},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    result: dict[str, Any] = {
        "folder": str(folder),
        "files": written,
        "fps": timeline.fps,
        "duration": round(timeline.total / timeline.fps, 4),
        "segments": len(sb.segments),
        "captions": len(timeline.captions),
        "overlay_pictures": sum(1 for item in timeline.overlays if item.png),
        "sound_effects": len(timeline.sfx),
        "media": mode,
        "warnings": timeline.warnings,
        "not_exported": timeline.not_exported,
        "licences": timeline.licences,
    }
    return result


__all__ = ["FORMATS", "MEDIA_MODES", "ExportError", "export_timeline", "build_timeline"]
