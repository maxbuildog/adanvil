"""Timing of animated captions: which state is on screen when, and keyword markers.

An animated caption is a chunk of words shown for some time. ``karaoke`` and ``word`` light the
word being spoken (one state per word, each lit from the start of its word to the start of the
next, the last until the end of the chunk); ``pop`` scales the chunk in over a few frames. This
module only decides the timing and the scales; the pictures come from ``text.render_states``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import ceil

from .chunking import is_mark, is_opening_mark

POP_SECONDS = 0.12  # the scale-in of a chunk
BOUNCE_SECONDS = 0.18  # the same with a bounce (an overshoot, then a settle)
POP_FROM = 0.85
BOUNCE_PEAK = 1.08
WORD_SIZE = 1.3  # the ``word`` animation draws its words this much larger

_MARKER = re.compile(r"^([^\w*]*?)(\*+)?(.*?)(\*+)?([^\w*]*)$", re.DOTALL)


def strip_markers(text: str) -> tuple[str, set[int]]:
    """``text`` without its ``*word*`` markers, and the indexes of the marked words.

    Indexes count the whitespace separated tokens of the clean text. A marker may span several
    words (``*very fast*``, ``*65,90 €*``), may wrap text with no letter (``*-20%*``, ``*€*``)
    and may be followed by punctuation (``*now*,``). A text without any marker comes back
    unchanged; a star that has no partner (``49€*``) is plain text.
    """

    if "*" not in text:
        return text, set()
    tokens = text.split()
    clean: list[str] = []
    marked: set[int] = set()
    inside = False
    group: list[int] = []
    opened_at = 0
    for index, token in enumerate(tokens):
        match = _MARKER.match(token)
        assert match is not None
        before, opening, core, closing, after = match.groups()
        has_text = token.strip("*") != ""
        opens = bool(opening) and has_text
        closes = bool(closing) and has_text
        if opens and not inside:
            inside, group, opened_at = True, [], index
        if inside:
            group.append(index)
            clean.append(f"{before}{core}{after}" if (opens or closes) else token)
            if closes:
                marked.update(group)
                inside = False
        else:
            clean.append(token)
    if inside:  # a marker that never closes (a footnote star) is plain text
        clean[opened_at] = tokens[opened_at]
    if not marked:
        return text, set()
    return " ".join(clean), marked


def marked_entries(tokens: list[str], marked: set[int]) -> list[str]:
    """The emphasis entries of the marked ``tokens``: a mark with no letter (``€``, ``%``) next
    to a marked word joins its entry, so ``*65,90 €*`` is the one entry ``65,90 €``."""

    entries: list[str] = []
    last = -2
    pending = ""
    for index in sorted(marked):
        if index >= len(tokens):
            continue
        token = tokens[index]
        if is_mark(token):
            if is_opening_mark(token):
                pending += token + " "
                continue
            if entries and last == index - 1:
                entries[-1] += " " + token
            else:
                entries.append(token)
        else:
            entries.append(pending + token)
            pending = ""
        last = index
    return entries


def _ease_out(progress: float) -> float:
    return 1.0 - (1.0 - progress) ** 3


def pop_scales(fps: int, bounce: bool = False) -> list[float]:
    """The scale of each frame of a chunk's scale-in, one frame after the other.

    The plain pop goes from 0.85 to 1.0 in 120 ms with an ease-out; the bounce takes 180 ms,
    rises to 1.08 and settles on 1.0. The caption stays at scale 1.0 after the last frame.
    """

    seconds = BOUNCE_SECONDS if bounce else POP_SECONDS
    frames = max(1, ceil(seconds * fps - 1e-9))
    scales: list[float] = []
    for index in range(frames):
        progress = min(1.0, index / (seconds * fps))
        if not bounce:
            value = POP_FROM + (1.0 - POP_FROM) * _ease_out(progress)
        elif progress < 0.55:
            value = POP_FROM + (BOUNCE_PEAK - POP_FROM) * _ease_out(progress / 0.55)
        else:
            settle = (progress - 0.55) / 0.45
            value = BOUNCE_PEAK - (BOUNCE_PEAK - 1.0) * settle * settle * (3 - 2 * settle)
        scales.append(round(value, 4))
    return scales


def spread_times(words: list[str], start: float, end: float) -> list[tuple[float, float]]:
    """Time ``words`` over ``start`` to ``end``, a longer word taking a longer share."""

    if not words:
        return []
    weights = [len(word) + 1 for word in words]
    total = float(sum(weights))
    times: list[tuple[float, float]] = []
    cursor = start
    for weight in weights:
        span = (end - start) * weight / total
        times.append((cursor, cursor + span))
        cursor += span
    return times


@dataclass(frozen=True)
class StatePlan:
    """One picture of an animated caption and the time it is on screen."""

    start: float
    end: float
    active: int | None  # the lit spoken word, if any
    scale: float = 1.0  # the pop scale of the picture


def plan_states(
    animation: str,
    start: float,
    end: float,
    fps: int,
    word_times: list[tuple[float, float]],
    *,
    bounce: bool = False,
) -> list[StatePlan]:
    """The states of a chunk shown from ``start`` to ``end``, in order, with no gap between.

    ``word_times`` are the spoken words (start, end) on the timeline. A state lasting less than
    half a frame could never be seen and is left out; the state before it simply runs on.
    """

    if end <= start:
        return []
    if animation == "pop":
        plans: list[StatePlan] = []
        scales = pop_scales(fps, bounce)
        for index, scale in enumerate(scales):
            begin = start + index / fps
            finish = min(end, start + (index + 1) / fps)
            if begin >= end:
                break
            plans.append(StatePlan(begin, finish, None, scale))
        settled = start + len(scales) / fps
        if settled < end:
            plans.append(StatePlan(settled, end, None, 1.0))
        elif plans:
            plans[-1] = StatePlan(plans[-1].start, end, None, plans[-1].scale)
        return plans
    if animation in {"karaoke", "word"} and word_times:
        starts = [start] + [max(start, min(end, begin)) for begin, _ in word_times[1:]]
        plans = []
        half = 0.5 / fps
        for index, begin in enumerate(starts):
            finish = starts[index + 1] if index + 1 < len(starts) else end
            if finish - begin < half and index + 1 < len(starts):
                continue
            if plans and plans[-1].end < begin:
                plans[-1] = StatePlan(plans[-1].start, begin, plans[-1].active, 1.0)
            plans.append(StatePlan(begin, finish, index, 1.0))
        return plans
    return [StatePlan(start, end, None, 1.0)]


__all__ = [
    "BOUNCE_SECONDS",
    "POP_SECONDS",
    "WORD_SIZE",
    "StatePlan",
    "plan_states",
    "pop_scales",
    "spread_times",
    "marked_entries",
    "strip_markers",
]
