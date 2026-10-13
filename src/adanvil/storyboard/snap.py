"""Cuts on the beat: ``storyboard_snap`` moves the cuts of a storyboard onto the beats of its music.

Cut ``k`` sits between segment ``k-1`` and segment ``k``. A cut is moved by changing the length
of the segments on each side of it: the ``out`` of a clip (``in`` points and segment order never
change) or the ``duration`` of an image. Every other cut keeps its time, so the total duration,
the end card and everything written after the last moved cut stay where they were, and the
captions, images and sounds that started on a moved cut move with it.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from adanvil.audio import beat_grid
from adanvil.ffmpeg import probe
from adanvil.media import ClipSegment, RevealSegment, segment_duration
from adanvil.project import resolve_in_project

from .schema import Storyboard

MIN_SEGMENT = 0.5  # a moved cut never leaves a segment shorter than this (or shorter than before)
_MAX_SHIFT_LIMIT = 1.0
MIN_ANALYSIS = 20.0  # seconds of music looked at: a short window gives a poor tempo


def _source_length(root: Path, source: str, cache: dict[str, float | None]) -> float | None:
    if source not in cache:
        try:
            cache[source] = probe(resolve_in_project(root, source)).duration
        except Exception:
            cache[source] = None
    return cache[source]


def snap_storyboard(
    root: Path,
    raw: dict[str, Any],
    *,
    to: str = "beats",
    max_shift: float = 0.2,
) -> dict[str, Any]:
    """The storyboard ``raw`` with its cuts moved to the nearest beat of its music.

    A cut moves only when a beat lies within ``max_shift`` seconds and the segments on both
    sides stay at least 0.5 s long (or as long as they were). With ``audio.clips`` the cuts of
    talking segments (any cut touching a clip) never move: they follow the speech. Returns
    ``{"storyboard", "moves", "kept", "changed", ...}``; ``raw`` itself is left untouched.
    """

    if to != "beats":
        raise ValueError(f"to must be 'beats', got {to!r}")
    max_shift = float(max_shift)
    if not math.isfinite(max_shift) or not 0 < max_shift <= _MAX_SHIFT_LIMIT:
        raise ValueError("max_shift must be a number of seconds above 0 and up to 1")
    model = Storyboard.model_validate(raw)
    written = model.audio.music or model.audio.from_reference
    if written is None:
        raise ValueError("the storyboard has no music to snap to: set audio.music first")
    music = resolve_in_project(Path(root), written)
    if not music.is_file():
        raise ValueError(f"music file does not exist: {written}")

    segments = model.segments
    durations = [segment_duration(segment) for segment in segments]
    old = [0.0]
    for duration in durations:
        old.append(old[-1] + duration)
    total = old[-1]
    fps = model.format.fps
    frame = 1.0 / fps
    grid = beat_grid(
        music, start=model.audio.start, duration=max(total + max_shift + 1.0, MIN_ANALYSIS)
    )
    result: dict[str, Any] = {
        "tempo_bpm": grid.tempo,
        "beats_used": len(grid.beats),
        "max_shift": max_shift,
    }
    warnings: list[str] = []
    if model.audio.voice is not None:
        warnings.append(
            "audio.voice is set: the cuts follow the voice lines, check after snapping that no "
            "cut now lands on the wrong sentence (snapping is meant for music-led ads)"
        )

    def on_beat(moment: float) -> bool:
        beat = grid.nearest(moment)
        return beat is not None and abs(moment - beat) <= frame + 1e-6

    cuts = list(range(1, len(segments)))
    result["on_beat_before"] = sum(1 for k in cuts if on_beat(old[k]))
    result["cuts"] = len(cuts)
    if not grid.beats:
        result.update(
            storyboard=copy.deepcopy(raw),
            moves=[],
            kept=[],
            changed=False,
            warnings=[*warnings, "the music has no steady beat: nothing to snap to"],
            on_beat_after=result["on_beat_before"],
            total_duration=round(total, 4),
        )
        return result

    new = list(old)
    lengths: dict[str, float | None] = {}

    def fits(index: int, duration: float) -> bool:
        """Whether segment ``index`` can last ``duration`` seconds."""
        original = durations[index]
        if duration < min(MIN_SEGMENT, original) - 1e-9:
            return False
        segment = segments[index]
        if isinstance(segment, RevealSegment) and (
            duration < segment.hold_before + segment.sweep - 1e-9
        ):
            return False  # the sweep would not reach the after picture
        if isinstance(segment, ClipSegment):
            hold = segment.freeze.hold if segment.freeze else 0.0
            if duration - hold <= 0.05:
                return False
            if any(item.at + item.hold > duration + 1e-6 for item in segment.punch):
                return False  # a punch-in would end after the segment
            out = segment.in_ + (duration - hold) * segment.speed
            length = _source_length(Path(root), segment.source, lengths)
            if length is not None and out > length + 1e-6 and out > segment.out + 1e-6:
                return False
        return True

    moves: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    for k in cuts:
        at = old[k]
        before, after = segments[k - 1], segments[k]
        if model.audio.clips and (
            isinstance(before, ClipSegment) or isinstance(after, ClipSegment)
        ):
            kept.append({"cut": k, "at": round(at, 4), "reason": "talking segment (audio.clips)"})
            continue
        if on_beat(at):
            kept.append({"cut": k, "at": round(at, 4), "reason": "already on a beat"})
            continue
        candidates = sorted(
            (beat for beat in grid.beats if abs(beat - at) <= max_shift + 1e-9),
            key=lambda beat: abs(beat - at),
        )
        if not candidates:
            kept.append(
                {"cut": k, "at": round(at, 4), "reason": f"no beat within {max_shift:g}s"}
            )
            continue
        following = old[k + 1]  # the next cut is where it was (it may move after this one)
        chosen = None
        for beat in candidates:
            if fits(k - 1, beat - new[k - 1]) and fits(k, following - beat):
                chosen = beat
                break
        if chosen is None:
            kept.append(
                {
                    "cut": k,
                    "at": round(at, 4),
                    "reason": "a segment would become too short (under 0.5 s) or run past its "
                    "source",
                }
            )
            continue
        new[k] = chosen
        moves.append(
            {
                "cut": k,
                "from": round(at, 4),
                "to": round(chosen, 4),
                "shift": round(chosen - at, 4),
            }
        )

    document = copy.deepcopy(raw)
    changed_segments: list[int] = []
    for index, segment in enumerate(segments):
        duration = new[index + 1] - new[index]
        if abs(duration - durations[index]) < 1e-6:
            continue
        written_segment = document["segments"][index]
        if isinstance(segment, ClipSegment):
            hold = segment.freeze.hold if segment.freeze else 0.0
            written_segment["out"] = round(segment.in_ + (duration - hold) * segment.speed, 4)
        else:
            written_segment["duration"] = round(duration, 4)
        changed_segments.append(index)

    shifts = {k: new[k] - old[k] for k in cuts if abs(new[k] - old[k]) > 1e-9}
    carried: list[str] = []

    def carried_shift(moment: float) -> float:
        for k, shift in shifts.items():
            if abs(moment - old[k]) <= frame + 1e-6:
                return shift
        return 0.0

    for index, overlay in enumerate(document.get("overlays") or []):
        kind = overlay.get("type")
        if kind in {"caption", "comment", "image", "video", "sticker", "progress"}:
            start = float(overlay.get("start", 0.0))
            shift = carried_shift(start)
            end = overlay.get("end")
            end_shift = carried_shift(float(end)) if end is not None else 0.0
            if shift == 0.0 and end_shift == 0.0:
                continue
            new_start = start + shift
            new_end = None if end is None else float(end) + end_shift
            if new_start < 0 or (new_end is not None and new_end <= new_start + 0.05):
                continue
            if shift:
                overlay["start"] = round(new_start, 4)
                if kind == "caption" and overlay.get("word_times"):
                    overlay["word_times"] = [
                        [round(a + shift, 4), round(b + shift, 4)]
                        for a, b in overlay["word_times"]
                    ]
            if end_shift and new_end is not None:
                overlay["end"] = round(new_end, 4)
            carried.append(f"overlays[{index}] ({kind})")
        elif kind == "sfx":
            shift = carried_shift(float(overlay.get("at", 0.0)))
            if shift:
                overlay["at"] = round(float(overlay["at"]) + shift, 4)
                carried.append(f"overlays[{index}] (sfx)")

    result.update(
        storyboard=document,
        moves=moves,
        kept=kept,
        changed=bool(moves),
        changed_segments=changed_segments,
        moved_with_cuts=carried,
        warnings=warnings,
        on_beat_after=sum(1 for k in cuts if on_beat(new[k])),
        total_duration=round(new[-1], 4),
    )
    return result


__all__ = ["MIN_SEGMENT", "snap_storyboard"]
