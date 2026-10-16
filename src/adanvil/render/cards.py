"""Image overlays as cards: a picture with rounded corners, a border, a shadow and a small
rotation, that comes in (and optionally goes out) with an animation.

A card is drawn with Pillow (corners and rotation supersampled, so it is anti-aliased) and each
animation state is its own cached picture, placed by the overlay filter on an exact span, like an
animated caption or a sticker: ``plan_card`` turns the overlay into the run of states (scale,
opacity, offset), ``card_placements`` draws and caches them. The states are a pure function of
the overlay and the frame rate, so the same storyboard always gives the same video.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFilter

from adanvil.media import Cache
from adanvil.storyboard import ImageOverlay

from .animated import Placement
from .stickers import _even_clip

CARD_VERSION = 2  # bump when the pictures of cards change
ENTRY_SECONDS = 0.25  # an entry (and an exit) lasts this long
SHADOW_PAD = 40  # room around the card for the blurred shadow
_SS = 3  # supersampling factor of the shapes


@dataclass(frozen=True)
class CardState:
    """One look of the card, shown on ``[start, end)``."""

    start: float
    end: float
    scale: float
    alpha: float
    dx: float
    dy: float


def _rounded_mask(width: int, height: int, radius: float) -> Image.Image:
    """A grey mask of a rounded rectangle, anti-aliased (255 inside)."""

    big = Image.new("L", (width * _SS, height * _SS), 0)
    ImageDraw.Draw(big).rounded_rectangle(
        (0, 0, width * _SS - 1, height * _SS - 1),
        radius=max(0.0, radius) * _SS,
        fill=255,
    )
    return big.resize((width, height), Image.Resampling.LANCZOS)


def card_size(overlay: ImageOverlay, source_width: int, source_height: int) -> tuple[int, int]:
    """The outer size of the card (border included): ``width`` wide when given, else the
    picture's own width; the height follows the picture."""

    width = overlay.width if overlay.width is not None else source_width
    border = overlay.frame.border_width if overlay.frame is not None else 0
    inner_width = max(1, width - 2 * border)
    inner_height = max(1, round(source_height * inner_width / max(source_width, 1)))
    return width, inner_height + 2 * border


def card_art(
    overlay: ImageOverlay, source: Path, canvas: tuple[int, int]
) -> tuple[Image.Image, tuple[int, int], tuple[int, int, int, int]]:
    """The finished card (with its shadow, rotated) as an RGBA picture, the position of its
    top-left corner on the frame, and the box the card itself covers (rotation included, shadow
    left out)."""

    with Image.open(source) as opened:
        picture = opened.convert("RGBA")
    frame = overlay.frame
    border = frame.border_width if frame is not None else 0
    radius = float(frame.radius) if frame is not None else 0.0
    width, height = card_size(overlay, picture.width, picture.height)
    inner_width, inner_height = width - 2 * border, height - 2 * border
    if picture.size != (inner_width, inner_height):
        picture = picture.resize((inner_width, inner_height), Image.Resampling.LANCZOS)

    card = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    outer = _rounded_mask(width, height, radius)
    if border > 0:
        colour = ImageColor.getcolor(frame.border_color, "RGBA")  # type: ignore[union-attr]
        layer = Image.new("RGBA", (width, height), colour)
        layer.putalpha(outer.point(lambda value: value * colour[3] // 255))
        card.alpha_composite(layer)
    inner_mask = _rounded_mask(inner_width, inner_height, radius - border)
    picture.putalpha(ImageChops.multiply(picture.getchannel("A"), inner_mask))
    card.alpha_composite(picture, (border, border))
    picture.close()

    pad = SHADOW_PAD
    layer = Image.new("RGBA", (width + 2 * pad, height + 2 * pad), (0, 0, 0, 0))
    if frame is not None and frame.shadow:
        shade = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        shade_alpha = Image.new("L", layer.size, 0)
        shade_alpha.paste(outer.point(lambda value: value * 120 // 255), (pad, pad + 8))
        shade.putalpha(shade_alpha)
        layer.alpha_composite(shade.filter(ImageFilter.GaussianBlur(11)))
    layer.alpha_composite(card, (pad, pad))
    card.close()

    centre = (overlay.x + width / 2, overlay.y + height / 2)
    if overlay.rotate:
        layer = layer.rotate(
            -overlay.rotate, resample=Image.Resampling.BICUBIC, expand=True
        )
        angle = math.radians(overlay.rotate)
        half_w = (abs(width * math.cos(angle)) + abs(height * math.sin(angle))) / 2
        half_h = (abs(width * math.sin(angle)) + abs(height * math.cos(angle))) / 2
    else:
        half_w, half_h = width / 2, height / 2
    box = (
        round(centre[0] - half_w),
        round(centre[1] - half_h),
        round(centre[0] + half_w),
        round(centre[1] + half_h),
    )
    left = round(centre[0] - layer.width / 2)
    top = round(centre[1] - layer.height / 2)
    return layer, (left, top), box


def _ease_out(u: float) -> float:
    return 1 - (1 - u) ** 3


def _state_values(
    animation: str, u: float, canvas: tuple[int, int]
) -> tuple[float, float, float, float]:
    """``(scale, alpha, dx, dy)`` of the card at ``u`` of its entry (1.0 is settled). The
    offsets are fractions of a pixel: the card is drawn at that exact position."""

    e = _ease_out(u)
    if animation == "fade":
        return 1.0, round(e, 4), 0, 0
    if animation == "pop":
        if u < 0.6:
            scale = 0.6 + (1.06 - 0.6) * _ease_out(u / 0.6)
        else:
            settle = (u - 0.6) / 0.4
            scale = 1.06 - 0.06 * settle * settle * (3 - 2 * settle)
        return round(scale, 4), round(min(1.0, u / 0.35), 4), 0, 0
    distance_x = round(canvas[0] * 0.45)
    distance_y = round(canvas[1] * 0.25)
    if animation == "slide_left":  # comes in from the right, moving left
        return 1.0, round(e, 4), round(distance_x * (1 - e), 3), 0
    if animation == "slide_right":  # comes in from the left, moving right
        return 1.0, round(e, 4), -round(distance_x * (1 - e), 3), 0
    if animation == "slide_up":  # comes in from below, moving up
        return 1.0, round(e, 4), 0, round(distance_y * (1 - e), 3)
    raise ValueError(f"unknown image animation {animation!r}")


def plan_card(
    overlay: ImageOverlay, start: float, end: float, fps: int, canvas: tuple[int, int]
) -> list[CardState]:
    """The states of a card shown on ``[start, end)``, one right after the other: the entry
    (one per frame over 250 ms), the settled card, then the exit when ``exit`` is on."""

    settled = CardState(start, end, 1.0, 1.0, 0, 0)
    if end <= start or overlay.animation == "none":
        return [settled] if end > start else []
    frames = max(2, ceil(ENTRY_SECONDS * fps - 1e-9))
    states: list[CardState] = []
    for number in range(frames):
        begin = start + number / fps
        if begin >= end - 1e-9:
            break
        scale, alpha, dx, dy = _state_values(overlay.animation, (number + 1) / frames, canvas)
        states.append(CardState(begin, min(end, start + (number + 1) / fps), scale, alpha, dx, dy))
    entry_end = start + frames / fps
    exit_start = end
    exits: list[CardState] = []
    if overlay.exit:
        exit_start = max(entry_end, end - frames / fps)
        count = max(0, ceil((end - exit_start) * fps - 1e-9))
        for number in range(count):
            begin = exit_start + number / fps
            u = 1 - (number + 1) / (count + 1)
            scale, alpha, dx, dy = _state_values(overlay.animation, u, canvas)
            finish = end if number == count - 1 else min(end, exit_start + (number + 1) / fps)
            exits.append(CardState(begin, finish, scale, alpha, dx, dy))
    if entry_end < exit_start - 1e-9:
        states.append(CardState(entry_end, exit_start, 1.0, 1.0, 0, 0))
    return [*states, *exits]


def _transformed(
    art: Image.Image, state: CardState, centre: tuple[float, float]
) -> tuple[Image.Image, int, int]:
    """The card as it looks in ``state`` and the integer position of its top-left corner.

    The scale and the offset are floats, and the picture is resampled onto the pixel grid with
    that sub-pixel phase (``resize`` with a float ``box``), so an entry that slides or pops moves
    and grows smoothly instead of by whole-pixel steps. A settled card is the art as it is.
    """

    scale = state.scale if abs(state.scale - 1.0) >= 0.005 else 1.0
    width, height = art.width * scale, art.height * scale
    left = centre[0] + state.dx - width / 2
    top = centre[1] + state.dy - height / 2
    x0, y0 = math.floor(left + 1e-6), math.floor(top + 1e-6)
    if scale == 1.0 and abs(left - round(left)) < 1e-6 and abs(top - round(top)) < 1e-6:
        picture = art
        x0, y0 = round(left), round(top)
    else:
        w = max(1, math.ceil(left + width - 1e-6) - x0)
        h = max(1, math.ceil(top + height - 1e-6) - y0)
        pad = 3  # transparent margin, so the box of the resize stays inside the source
        padded = Image.new("RGBA", (art.width + 2 * pad, art.height + 2 * pad), (0, 0, 0, 0))
        padded.paste(art, (pad, pad))
        box = (
            pad + (x0 - left) / scale,
            pad + (y0 - top) / scale,
            pad + (x0 + w - left) / scale,
            pad + (y0 + h - top) / scale,
        )
        picture = padded.resize((w, h), Image.Resampling.LANCZOS, box=box)
        padded.close()
    if state.alpha < 0.999:
        picture = picture.copy()
        channel = picture.getchannel("A").point(lambda value: round(value * state.alpha))
        picture.putalpha(channel)
    return picture, x0, y0


def card_placements(
    overlay: ImageOverlay,
    source: Path,
    start: float,
    end: float,
    *,
    canvas: tuple[int, int],
    fps: int,
    cache: Cache,
) -> tuple[list[Placement], dict[str, Any]]:
    """Cached pictures and placements of a card shown on ``[start, end)``, and its report entry
    (the states with their spans and offsets, like an animated caption)."""

    states = plan_card(overlay, start, end, fps, canvas)
    art, (left, top), box = card_art(overlay, source, canvas)
    centre = (left + art.width / 2, top + art.height / 2)
    base = {
        "overlay": overlay.model_dump(mode="json", exclude={"start", "end"}),
        "source": Cache.file_key(source),
        "canvas": list(canvas),
        "version": CARD_VERSION,
    }
    placements: list[Placement] = []
    entries: list[dict[str, Any]] = []
    for state in states:
        key = {
            **base,
            "state": [state.scale, state.alpha, state.dx, state.dy],
        }
        picture, x, y = _transformed(art, state, centre)
        clipped, x, y = _even_clip(picture, x, y, canvas)
        png = cache.path_for("overlays", key, suffix=".png")
        if not png.is_file():
            png.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(f"{png}.tmp{png.suffix}")
            temporary.unlink(missing_ok=True)
            try:
                clipped.save(temporary, format="PNG")
                os.replace(temporary, png)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        placements.append(Placement(png, state.start, state.end, x, y, exact=True))
        entries.append(
            {
                "png": png,
                "start": state.start,
                "end": state.end,
                "offset": [x, y],
                "active": None,
                "scale": state.scale,
                "alpha": state.alpha,
            }
        )
    art.close()
    settled = [state for state in states if state.alpha >= 0.999 and state.scale == 1.0]
    entry: dict[str, Any] = {
        "type": "image",
        "start": start,
        "end": end,
        "png": placements[0].png if placements else None,
        "offset": entries[0]["offset"] if entries else [0, 0],
        "canvas": list(canvas),
        "animation": overlay.animation,
        "bbox": list(box),
        "states": entries,
        "settled": [min(s.start for s in settled), max(s.end for s in settled)]
        if settled
        else [start, end],
    }
    return placements, entry


__all__ = [
    "CardState",
    "card_art",
    "card_placements",
    "card_size",
    "plan_card",
]
