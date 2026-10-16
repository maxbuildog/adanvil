"""Stickers: vector arrows, hand-drawn circles and underlines, badges, emoji and image stickers.

A sticker is drawn with Pillow at 3x and brought back to the frame size, so it is deterministic
and anti-aliased. ``sticker_art`` draws one state (how much of it is drawn, at which scale);
``sticker_placements`` turns a sticker overlay into the run of cached pictures the final encode
needs (one per state, each cropped to its ink and placed by the overlay filter), like an
animated caption.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

from PIL import Image, ImageColor, ImageDraw, ImageFilter

from adanvil.media import Cache
from adanvil.project import resolve_in_project
from adanvil.storyboard import StickerOverlay
from adanvil.text import DEFAULT_HIGHLIGHT
from adanvil.text.emoji import split_runs
from adanvil.text.fonts import load_font

from .animated import Placement

STICKER_VERSION = 2  # bump when the pictures of stickers change
DRAW_SECONDS = 0.3  # an arrow, circle or underline is drawn over this long
POP_SECONDS = 0.24
SS = 3  # supersampling factor
_DIRECTIONS = {
    "up": (0.0, -1.0),
    "down": (0.0, 1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "up_left": (-0.7071, -0.7071),
    "up_right": (0.7071, -0.7071),
    "down_left": (-0.7071, 0.7071),
    "down_right": (0.7071, 0.7071),
}


@dataclass
class StickerArt:
    """One drawn state: an RGBA picture cropped to its ink and its top-left on the frame."""

    image: Image.Image
    x: int
    y: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.image.width, self.y + self.image.height)


BADGE_FALLBACK = "#111827"  # a badge with no colour and no brand accent: neutral dark, white text


def sticker_color(
    overlay: StickerOverlay, accent: str | None, badge_color: str | None = None
) -> tuple[int, int, int, int]:
    """The colour of a sticker: its own ``color``, else for a badge the brand accent colour
    (``badge_color``) or a neutral dark, for any other kind the highlight ``accent`` or yellow."""

    if overlay.kind == "badge":
        return ImageColor.getcolor(overlay.color or badge_color or BADGE_FALLBACK, "RGBA")
    return ImageColor.getcolor(overlay.color or accent or DEFAULT_HIGHLIGHT, "RGBA")


def _ink_box(image: Image.Image) -> tuple[int, int, int, int]:
    return image.getchannel("A").getbbox() or (0, 0, 1, 1)


def _halo(image: Image.Image, radius: int = 2) -> Image.Image:
    """A thin dark halo under a line drawing, so it reads on any picture."""

    alpha = image.getchannel("A")
    grown = alpha.filter(ImageFilter.MaxFilter(radius * 2 + 1)).filter(
        ImageFilter.GaussianBlur(1.2)
    )
    shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shadow.putalpha(grown.point(lambda value: int(value * 0.55)))
    shadow.alpha_composite(image)
    return shadow


def _draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    widths: list[float],
    fill: tuple[int, int, int, int],
) -> None:
    for number in range(len(points) - 1):
        width = max(1, round((widths[number] + widths[number + 1]) / 2))
        draw.line([points[number], points[number + 1]], fill=fill, width=width)
    for point, width in zip(points, widths, strict=True):
        radius = max(0.5, width / 2)
        draw.ellipse(
            [point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius],
            fill=fill,
        )


def _arrow_points(overlay: StickerOverlay) -> tuple[tuple[float, float], tuple[float, float]]:
    if overlay.from_ is not None and overlay.to is not None:
        return (
            (float(overlay.from_[0]), float(overlay.from_[1])),
            (float(overlay.to[0]), float(overlay.to[1])),
        )
    assert overlay.x is not None and overlay.y is not None and overlay.direction is not None
    dx, dy = _DIRECTIONS[overlay.direction]
    tip = (float(overlay.x), float(overlay.y))
    return (tip[0] - dx * overlay.length, tip[1] - dy * overlay.length), tip


def _draw_arrow(
    overlay: StickerOverlay, colour: tuple[int, int, int, int], progress: float, pad: int
) -> tuple[Image.Image, tuple[int, int]]:
    (x0, y0), (x1, y1) = _arrow_points(overlay)
    length = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    stroke = overlay.stroke
    head_length = min(stroke * 4.0, length * 0.6)
    head_half = stroke * 2.0
    base = (x1 - ux * head_length, y1 - uy * head_length)
    shaft_progress = min(1.0, progress / 0.8)
    end = (x0 + (base[0] - x0) * shaft_progress, y0 + (base[1] - y0) * shaft_progress)
    head_progress = 0.0 if progress < 0.8 else min(1.0, (progress - 0.8) / 0.2)
    left = math.floor(min(x0, x1, base[0] - head_half, base[0] + head_half)) - pad
    top = math.floor(min(y0, y1, base[1] - head_half, base[1] + head_half)) - pad
    right = math.ceil(max(x0, x1, base[0] - head_half, base[0] + head_half)) + pad
    bottom = math.ceil(max(y0, y1, base[1] - head_half, base[1] + head_half)) + pad
    image = Image.new("RGBA", ((right - left) * SS, (bottom - top) * SS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    def local(point: tuple[float, float]) -> tuple[float, float]:
        return ((point[0] - left) * SS, (point[1] - top) * SS)

    if shaft_progress > 0:
        _draw_polyline(draw, [local((x0, y0)), local(end)], [stroke * SS, stroke * SS], colour)
    if head_progress > 0:
        scale = head_progress
        hl, hw = head_length * scale, head_half * scale
        apex = (base[0] + ux * hl, base[1] + uy * hl)
        corner_a = (base[0] - uy * hw, base[1] + ux * hw)
        corner_b = (base[0] + uy * hw, base[1] - ux * hw)
        draw.polygon([local(apex), local(corner_a), local(corner_b)], fill=colour)
    return image, (left, top)


def _draw_circle(
    overlay: StickerOverlay, colour: tuple[int, int, int, int], progress: float, pad: int
) -> tuple[Image.Image, tuple[int, int]]:
    assert overlay.x is not None and overlay.y is not None and overlay.width is not None
    rx = overlay.width / 2
    ry = (overlay.height if overlay.height is not None else overlay.width * 0.62) / 2
    stroke = overlay.stroke
    sweep = 395.0 * progress
    start = -105.0
    steps = max(2, round(sweep / 3))
    points: list[tuple[float, float]] = []
    widths: list[float] = []
    for number in range(steps + 1):
        degrees = sweep * number / steps
        angle = math.radians(start + degrees)
        # A slightly irregular, slowly shrinking loop that overshoots its start, like a pen.
        wobble = 1 + 0.03 * math.sin(2 * angle + 0.6) + 0.018 * math.sin(3 * angle)
        shrink = 1 - 0.07 * (degrees / 395.0)
        factor = wobble * shrink
        points.append(
            (overlay.x + rx * factor * math.cos(angle), overlay.y + ry * factor * math.sin(angle))
        )
        taper = 0.75 + 0.25 * math.sin(math.pi * min(1.0, degrees / 395.0) * 0.9 + 0.3)
        widths.append(stroke * taper * SS)
    extent = stroke + pad
    left = math.floor(overlay.x - rx * 1.12) - extent
    top = math.floor(overlay.y - ry * 1.12) - extent
    right = math.ceil(overlay.x + rx * 1.12) + extent
    bottom = math.ceil(overlay.y + ry * 1.12) + extent
    image = Image.new("RGBA", ((right - left) * SS, (bottom - top) * SS), (0, 0, 0, 0))
    scaled = [((px - left) * SS, (py - top) * SS) for px, py in points]
    _draw_polyline(ImageDraw.Draw(image), scaled, widths, colour)
    return image, (left, top)


def _draw_underline(
    overlay: StickerOverlay, colour: tuple[int, int, int, int], progress: float, pad: int
) -> tuple[Image.Image, tuple[int, int]]:
    assert overlay.x is not None and overlay.y is not None and overlay.width is not None
    stroke = overlay.stroke
    amplitude = stroke * 0.45
    steps = max(2, round(overlay.width / 6))
    visible = max(1, round(steps * progress))
    points: list[tuple[float, float]] = []
    widths: list[float] = []
    for number in range(visible + 1):
        u = number / steps
        points.append(
            (
                overlay.x + overlay.width * u,
                overlay.y - overlay.width * 0.012 * u + amplitude * math.sin(u * 7.5 + 0.4),
            )
        )
        widths.append(stroke * (0.8 + 0.2 * math.sin(u * math.pi)) * SS)
    extent = ceil(stroke + amplitude + pad)
    left = overlay.x - extent
    top = math.floor(overlay.y - overlay.width * 0.012 - extent)
    right = overlay.x + overlay.width + extent
    bottom = math.ceil(overlay.y + extent)
    image = Image.new("RGBA", ((right - left) * SS, (bottom - top) * SS), (0, 0, 0, 0))
    scaled = [((px - left) * SS, (py - top) * SS) for px, py in points]
    _draw_polyline(ImageDraw.Draw(image), scaled, widths, colour)
    return image, (left, top)


def _on_colour(colour: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    luminance = (0.299 * colour[0] + 0.587 * colour[1] + 0.114 * colour[2]) / 255
    return (24, 24, 27, 255) if luminance > 0.6 else (255, 255, 255, 255)


def _badge(
    overlay: StickerOverlay, colour: tuple[int, int, int, int]
) -> tuple[Image.Image, tuple[int, int]]:
    assert overlay.x is not None and overlay.y is not None and overlay.text is not None
    size = overlay.size or 44
    font = load_font("poppins", 700, size * SS)
    text = overlay.text.strip()
    ink = font.getbbox(text, anchor="ls")
    cap = -font.getbbox("H", anchor="ls")[1]
    pad_x, pad_y = round(size * 0.6) * SS, round(size * 0.34) * SS
    width = (ink[2] - ink[0]) + 2 * pad_x
    height = cap + 2 * pad_y
    margin = 14 * SS
    image = Image.new("RGBA", (width + 2 * margin, height + 2 * margin), (0, 0, 0, 0))
    shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        [margin, margin + 4 * SS, margin + width, margin + height + 4 * SS],
        radius=round(height * 0.3),
        fill=(0, 0, 0, 90),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(5 * SS))
    image.alpha_composite(shadow)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        [margin, margin, margin + width, margin + height],
        radius=round(height * 0.3),
        fill=colour,
    )
    foreground = (
        ImageColor.getcolor(overlay.fg, "RGBA") if overlay.fg is not None else _on_colour(colour)
    )
    draw.text(
        (margin + pad_x - ink[0], margin + pad_y + cap),
        text,
        font=font,
        fill=foreground,
        anchor="ls",
    )
    return image, (overlay.x - 14, overlay.y - 14)


def _emoji(
    overlay: StickerOverlay, provider: Any | None
) -> tuple[Image.Image, tuple[int, int]]:
    assert overlay.x is not None and overlay.y is not None and overlay.emoji is not None
    size = overlay.size or 120
    cluster = next((text for is_emoji, text in split_runs(overlay.emoji) if is_emoji), None)
    cluster = cluster or overlay.emoji.strip()[:1]
    picture = None if provider is None else provider.get(cluster, size * SS)
    if picture is None:
        if provider is not None:
            raise ValueError(f"emoji {overlay.emoji!r} has no picture in the emoji set")
        picture = Image.new("RGBA", (size * SS, size * SS), (0, 0, 0, 1))  # layout only
    return picture, (overlay.x, overlay.y)


def _image(
    overlay: StickerOverlay, root: Path | None
) -> tuple[Image.Image, tuple[int, int]]:
    assert overlay.x is not None and overlay.y is not None and overlay.source is not None
    if root is None:
        raise ValueError("an image sticker needs the project folder")
    path = resolve_in_project(root, overlay.source)
    with Image.open(path) as source:
        picture = source.convert("RGBA")
    if overlay.width is not None and picture.width != overlay.width:
        height = max(1, round(picture.height * overlay.width / picture.width))
        picture = picture.resize((overlay.width * SS, height * SS), Image.Resampling.LANCZOS)
    else:
        picture = picture.resize(
            (picture.width * SS, picture.height * SS), Image.Resampling.LANCZOS
        )
    return picture, (overlay.x, overlay.y)


def sticker_art(
    overlay: StickerOverlay,
    *,
    accent: str | None = None,
    badge_color: str | None = None,
    provider: Any | None = None,
    root: Path | None = None,
    progress: float = 1.0,
) -> StickerArt:
    """The sticker drawn with ``progress`` of its stroke done (1.0 is the finished sticker).

    Without ``provider`` an emoji sticker is a transparent square of the right size and place
    (what validation uses to measure it without fetching pictures).
    """

    if overlay.x == "center":  # used outside a Storyboard: centre on the usual 1080 px frame
        overlay = overlay.model_copy(update={"x": centered_x(overlay, 1080)})
    colour = sticker_color(overlay, accent, badge_color)
    if overlay.kind == "arrow":
        big, (left, top) = _draw_arrow(overlay, colour, progress, 3)
    elif overlay.kind == "circle":
        big, (left, top) = _draw_circle(overlay, colour, progress, 3)
    elif overlay.kind == "underline":
        big, (left, top) = _draw_underline(overlay, colour, progress, 3)
    elif overlay.kind == "badge":
        big, (left, top) = _badge(overlay, colour)
    elif overlay.kind == "emoji":
        big, (left, top) = _emoji(overlay, provider)
    else:
        big, (left, top) = _image(overlay, root)
    image = big.resize(
        (max(1, round(big.width / SS)), max(1, round(big.height / SS))),
        Image.Resampling.LANCZOS,
    )
    big.close()
    if overlay.kind in {"arrow", "circle", "underline"}:
        image = _halo(image)
    box = _ink_box(image)
    cropped = image.crop(box)
    image.close()
    return StickerArt(cropped, left + box[0], top + box[1])


def centered_x(overlay: StickerOverlay, frame_width: int) -> int:
    """The ``x`` that centres a sticker with ``x: "center"`` across a frame ``frame_width`` wide:
    the tip of an arrow and the centre of a circle go to the middle, an underline, an image
    sticker (``width``), a badge and an emoji are centred by their drawn width."""

    kind = overlay.kind
    if kind in {"arrow", "circle"}:
        return round(frame_width / 2)
    if kind in {"underline", "image"} and overlay.width is not None:
        return round((frame_width - overlay.width) / 2)
    probe = overlay.model_copy(update={"x": 0, "anchor": None})
    art = sticker_art(probe)
    wide = art.image.width
    art.image.close()
    return round((frame_width - wide) / 2 - art.x)


def sticker_scales(fps: int) -> list[float]:
    """The scale of each frame of a ``pop``: from 0.5 up to an overshoot of 1.12, then 1.0."""

    frames = max(2, ceil(POP_SECONDS * fps - 1e-9))
    scales: list[float] = []
    for number in range(frames):
        progress = min(1.0, number / (POP_SECONDS * fps))
        if progress < 0.6:
            local = progress / 0.6
            value = 0.5 + (1.12 - 0.5) * (1 - (1 - local) ** 3)
        else:
            settle = (progress - 0.6) / 0.4
            value = 1.12 - 0.12 * settle * settle * (3 - 2 * settle)
        scales.append(round(value, 4))
    return scales


def draw_progress(fps: int) -> list[float]:
    """The fraction drawn on each frame of a ``draw`` (the last one is complete)."""

    frames = max(2, ceil(DRAW_SECONDS * fps - 1e-9))
    return [(number + 1) / frames for number in range(frames)]


def _even_clip(
    image: Image.Image, x: int, y: int, canvas: tuple[int, int]
) -> tuple[Image.Image, int, int]:
    """``image`` placed at ``(x, y)`` and clipped to the canvas, with an even left and top."""

    left, top = max(0, x), max(0, y)
    right, bottom = min(canvas[0], x + image.width), min(canvas[1], y + image.height)
    if right <= left or bottom <= top:
        return Image.new("RGBA", (2, 2)), 0, 0
    clipped = image.crop((left - x, top - y, right - x, bottom - y))
    pad_x, pad_y = left % 2, top % 2
    if pad_x or pad_y:
        padded = Image.new("RGBA", (clipped.width + pad_x, clipped.height + pad_y), (0, 0, 0, 0))
        padded.paste(clipped, (pad_x, pad_y))
        clipped, left, top = padded, left - pad_x, top - pad_y
    return clipped, left, top


@dataclass(frozen=True)
class _State:
    start: float
    end: float
    progress: float
    scale: float


def plan_sticker(
    overlay: StickerOverlay, start: float, end: float, fps: int
) -> list[_State]:
    """The states of a sticker shown on ``[start, end)``, one right after the other."""

    if end <= start:
        return []
    if overlay.animation == "pop":
        scales = sticker_scales(fps)
        steps = [(1.0, scale) for scale in scales]
    elif overlay.animation == "draw":
        steps = [(progress, 1.0) for progress in draw_progress(fps)]
    else:
        return [_State(start, end, 1.0, 1.0)]
    states: list[_State] = []
    for number, (progress, scale) in enumerate(steps):
        begin = start + number / fps
        if begin >= end:
            break
        # one state at a time: a state never runs past the start of the next one. Only a
        # ``draw`` holds its last (complete) state to the end; a ``pop`` settles after its
        # last frame, below.
        finish = min(end, start + (number + 1) / fps)
        if number == len(steps) - 1 and overlay.animation == "draw":
            finish = end
        states.append(_State(begin, finish, progress, scale))
    if overlay.animation == "pop" and states and states[-1].scale != 1.0:
        settled = start + len(steps) / fps
        if settled < end:
            states.append(_State(settled, end, 1.0, 1.0))
        else:
            states[-1] = _State(states[-1].start, end, 1.0, 1.0)
    elif overlay.animation == "pop" and states:
        states[-1] = _State(states[-1].start, end, 1.0, 1.0)
    return states


def sticker_placements(
    overlay: StickerOverlay,
    start: float,
    end: float,
    *,
    canvas: tuple[int, int],
    fps: int,
    emoji_name: str,
    provider: Any,
    cache: Cache,
    root: Path,
    accent: str | None,
    badge_color: str | None = None,
) -> tuple[list[Placement], dict[str, Any]]:
    """Cached pictures and placements of a sticker shown on ``[start, end)``, and its report
    entry (the states with their spans and offsets, like an animated caption)."""

    states = plan_sticker(overlay, start, end, fps)
    source_key = None
    if overlay.kind == "image" and overlay.source is not None:
        source_key = Cache.file_key(resolve_in_project(root, overlay.source))
    base = {
        "overlay": overlay.model_dump(mode="json", by_alias=True, exclude={"start", "end"}),
        "accent": accent,
        "badge_color": badge_color,
        "emoji": emoji_name,
        "source": source_key,
        "canvas": list(canvas),
        "version": STICKER_VERSION,
    }
    drawn: dict[float, StickerArt] = {}
    placements: list[Placement] = []
    entries: list[dict[str, Any]] = []
    for state in states:
        if state.progress not in drawn:
            drawn[state.progress] = sticker_art(
                overlay,
                accent=accent,
                badge_color=badge_color,
                provider=provider,
                root=root,
                progress=state.progress,
            )
        art = drawn[state.progress]
        if abs(state.scale - 1.0) >= 0.005:
            width = max(1, round(art.image.width * state.scale))
            height = max(1, round(art.image.height * state.scale))
            resized = art.image.resize((width, height), Image.Resampling.LANCZOS)
            picture, x, y = _even_clip(
                resized,
                round(art.x + art.image.width / 2 - width / 2),
                round(art.y + art.image.height / 2 - height / 2),
                canvas,
            )
        else:
            picture, x, y = _even_clip(art.image, art.x, art.y, canvas)
        key = {**base, "progress": state.progress, "scale": state.scale}
        png = cache.path_for("overlays", key, suffix=".png")
        if not png.is_file():
            png.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(f"{png}.tmp{png.suffix}")
            temporary.unlink(missing_ok=True)
            try:
                picture.save(temporary, format="PNG")
                os.replace(temporary, png)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        placements.append(Placement(png, state.start, state.end, x, y, exact=True))
        entries.append(
            {"png": png, "start": state.start, "end": state.end, "offset": [x, y], "active": None}
        )
    final_art = drawn.get(1.0) or next(iter(drawn.values()), None)
    bbox = None if final_art is None else list(final_art.bbox)
    entry: dict[str, Any] = {
        "type": "sticker",
        "kind": overlay.kind,
        "start": start,
        "end": end,
        "png": placements[0].png if placements else None,
        "offset": entries[0]["offset"] if entries else [0, 0],
        "canvas": list(canvas),
        "animation": overlay.animation,
        "bbox": bbox,
        "states": entries,
    }
    return placements, entry


__all__ = [
    "StickerArt",
    "plan_sticker",
    "sticker_art",
    "sticker_placements",
    "sticker_scales",
]
