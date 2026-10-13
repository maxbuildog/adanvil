"""Pictures and placements of animated captions for the final encode.

An animated caption is not one overlay but a run of them: one picture per state (the lit word of
a ``karaoke`` or ``word`` caption, one frame of the scale-in of a ``pop``), each shown for exactly
its span, one right after the other. Every picture is cropped to the ink of the whole caption (the
same box for all the states, so the text never moves) and placed at its own offset, which keeps
the overlay filter working on a small area instead of the whole 1080x1920 frame.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adanvil.media import Cache
from adanvil.storyboard import CaptionOverlay, Storyboard
from adanvil.text import (
    StateImage,
    TextSpec,
    render_states,
    scale_state,
    spoken_words,
)
from adanvil.text.animate import StatePlan, plan_states, spread_times

PLACEMENT_VERSION = 1  # bump when the pictures of animated captions change


@dataclass(frozen=True)
class Placement:
    """One overlay picture of the final encode: shown on ``[start, end)`` at ``(x, y)``."""

    png: Path
    start: float
    end: float
    x: int = 0
    y: int = 0
    exact: bool = False  # the span is exact (``[start, end)``): states follow each other
    x_expr: str | None = None  # an ``x`` that moves with time (the fill of a progress bar)


def word_times_of(caption: CaptionOverlay, start: float, end: float) -> list[tuple[float, float]]:
    """The timeline span of each spoken word of ``caption`` (evenly spread when not given)."""

    words = spoken_words(caption.lines)
    if caption.word_times is not None and len(caption.word_times) == len(words):
        return [(float(a), float(b)) for a, b in caption.word_times]
    return spread_times(words, start, end)


def plan_caption(
    caption: CaptionOverlay, start: float, end: float, fps: int
) -> tuple[list[StatePlan], list[tuple[float, float]]]:
    """The states of an animated chunk on ``[start, end)`` and the span of its words."""

    times = word_times_of(caption, start, end)
    plans = plan_states(caption.animation, start, end, fps, times, bounce=caption.bounce)
    return plans, times


def state_images(
    caption: CaptionOverlay,
    plans: list[StatePlan],
    provider: object,
    canvas: tuple[int, int],
    highlight: str | None,
) -> tuple[dict[tuple[int | None, float], StateImage], TextSpec, Any]:
    """The picture of each (lit word, pop scale) the plans need, rendered together."""

    spec = caption.to_text_spec(highlight)
    actives: list[int | None] = []
    for plan in plans:
        if plan.active not in actives:
            actives.append(plan.active)
    rendered = render_states(spec, provider, actives or [None], canvas)  # type: ignore[arg-type]
    by_active = {state.active: state for state in rendered.states}
    images: dict[tuple[int | None, float], StateImage] = {}
    for plan in plans:
        key = (plan.active, plan.scale)
        if key not in images:
            images[key] = scale_state(by_active[plan.active], plan.scale, canvas)
    return images, spec, rendered


def animated_placements(
    caption: CaptionOverlay,
    start: float,
    end: float,
    *,
    storyboard: Storyboard,
    provider: object,
    cache: Cache,
    warnings: list[str],
    highlight: str | None,
) -> tuple[list[Placement], dict[str, Any]]:
    """Cached pictures and placements of an animated chunk shown on ``[start, end)``.

    Returns the placements for the encode and the report entry of the chunk (its first picture,
    and every state with its span and offset).
    """

    canvas = (storyboard.format.width, storyboard.format.height)
    plans, _ = plan_caption(caption, start, end, storyboard.format.fps)
    images, spec, rendered = state_images(caption, plans, provider, canvas, highlight)
    if rendered.overflow:
        warnings.append(f"caption overlay at {start:g}s overflowed its minimum text size")
    base = {
        "spec": spec.model_dump(mode="json"),
        "emoji": storyboard.emoji,
        "canvas": list(canvas),
        "version": PLACEMENT_VERSION,
    }
    placements: list[Placement] = []
    states: list[dict[str, Any]] = []
    for plan in plans:
        state = images[(plan.active, plan.scale)]
        key = {**base, "active": plan.active, "scale": plan.scale}
        png = cache.path_for("overlays", key, suffix=".png")
        if not png.is_file():
            png.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(f"{png}.tmp{png.suffix}")
            try:
                state.image.save(temporary, format="PNG")
                os.replace(temporary, png)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        placements.append(Placement(png, plan.start, plan.end, state.x, state.y, exact=True))
        states.append(
            {
                "png": png,
                "start": plan.start,
                "end": plan.end,
                "offset": [state.x, state.y],
                "active": plan.active,
            }
        )
    for state in images.values():
        state.image.close()
    if rendered.states:
        for state in rendered.states:
            state.image.close()
    entry: dict[str, Any] = {
        "type": "caption",
        "start": start,
        "end": end,
        "png": placements[0].png if placements else None,
        "offset": states[0]["offset"] if states else [0, 0],
        "canvas": list(canvas),
        "animation": caption.animation,
        "states": states,
    }
    return placements, entry


__all__ = ["Placement", "animated_placements", "plan_caption", "state_images", "word_times_of"]
