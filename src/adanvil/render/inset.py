"""Video insets (picture-in-picture) and progress bars: geometry, masks and the pictures that
frame them.

An inset is a clip played inside the frame. Its outer footprint (``x``, ``y``, ``width`` and
the height) includes the border; the clip fills the shape inside the border. The shape of the
clip is a grey mask picture (``alphamerge`` in the final encode), the border and the shadow are
one RGBA picture drawn under the clip. Both are deterministic and cached.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFilter

from adanvil.media import Cache
from adanvil.storyboard import ProgressOverlay, VideoOverlay

INSET_VERSION = 1  # bump when the pictures of insets change
SHADOW_PAD = 36  # room around the footprint for the blurred shadow
_SS = 4


@dataclass(frozen=True)
class InsetLayer:
    """One video inset of the final encode, ready for ``-i`` and the filter graph."""

    source: Path
    in_: float
    start: float
    end: float
    x: int
    y: int
    width: int  # inner (video) size
    height: int
    mask: Path | None  # grey picture, white inside the shape; None for a plain rectangle
    hdr_filter: str = ""


def inset_span(
    overlay: VideoOverlay, source_duration: float | None, total: float
) -> tuple[float, float]:
    """The ``(start, end)`` of an inset on the timeline: ``end`` when given, else as long as the
    clip lasts from ``in``, never past the end of the ad."""

    end = total if overlay.end is None else min(overlay.end, total)
    if overlay.end is None and source_duration is not None:
        end = min(end, overlay.start + max(0.0, source_duration - overlay.in_))
    return overlay.start, end


def _even(value: int) -> int:
    return max(2, value - value % 2)


def inset_size(overlay: VideoOverlay, source_width: int, source_height: int) -> tuple[int, int]:
    """The outer ``(width, height)`` of an inset: ``height`` when given (square for a circle),
    else the clip's aspect, both rounded down to even numbers."""

    width = _even(overlay.width)
    if overlay.shape == "circle":
        return width, width
    if overlay.height is not None:
        return width, _even(overlay.height)
    return width, _even(round(overlay.width * source_height / max(source_width, 1)))


def inset_footprint(overlay: VideoOverlay, height: int) -> tuple[int, int, int, int]:
    """The box ``(x0, y0, x1, y1)`` the inset covers on the frame (border included)."""

    width = _even(overlay.width)
    x, y = overlay.x - overlay.x % 2, overlay.y - overlay.y % 2
    return (x, y, x + width, y + height)


def _border_width(overlay: VideoOverlay) -> int:
    return overlay.border.width if overlay.border is not None else 0


def inner_size(overlay: VideoOverlay, height: int) -> tuple[int, int]:
    """The size of the clip inside the border (even)."""

    border = _border_width(overlay)
    return _even(_even(overlay.width) - 2 * border), _even(height - 2 * border)


def _radius(overlay: VideoOverlay, width: int, height: int) -> float:
    if overlay.shape == "circle":
        return min(width, height) / 2
    if overlay.shape == "rounded":
        if overlay.radius is not None:
            return float(overlay.radius)
        return float(round(min(width, height) * 0.12))
    return 0.0


def _shape(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, ...],
    shape: str,
    radius: float,
    fill: object,
) -> None:
    if shape == "circle":
        draw.ellipse(box, fill=fill)
    elif shape == "rounded" and radius > 0:
        draw.rounded_rectangle(box, radius=radius, fill=fill)
    else:
        draw.rectangle(box, fill=fill)


def _save(image: Image.Image, final: Path) -> Path:
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{final}.tmp{final.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, final)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return final


def inset_mask(overlay: VideoOverlay, height: int, cache: Cache) -> Path | None:
    """The grey mask of the clip inside the border (``None`` for a plain rectangle)."""

    if overlay.shape == "rect":
        return None
    width, inner_height = inner_size(overlay, height)
    border = _border_width(overlay)
    radius = max(0.0, _radius(overlay, _even(overlay.width), height) - border)
    key = {
        "kind": "inset-mask",
        "shape": overlay.shape,
        "size": [width, inner_height],
        "radius": radius,
        "version": INSET_VERSION,
    }
    final = cache.path_for("overlays", key, suffix=".png")
    if final.is_file():
        return final
    big = Image.new("L", (width * _SS, inner_height * _SS), 0)
    _shape(
        ImageDraw.Draw(big),
        (0, 0, width * _SS - 1, inner_height * _SS - 1),
        overlay.shape,
        radius * _SS,
        255,
    )
    mask = big.resize((width, inner_height), Image.Resampling.LANCZOS)
    return _save(mask, final)


def inset_decor(
    overlay: VideoOverlay, height: int, cache: Cache
) -> tuple[Path, tuple[int, int]] | None:
    """The border and shadow picture of an inset and the offset of its top-left corner from the
    inset's own top-left (negative: the shadow reaches outside the footprint); ``None`` when
    the inset has neither."""

    border = _border_width(overlay)
    if border == 0 and not overlay.shadow:
        return None
    width = _even(overlay.width)
    pad = SHADOW_PAD if overlay.shadow else 0
    radius = _radius(overlay, width, height)
    colour = overlay.border.color if overlay.border is not None else "#FFFFFF"
    key = {
        "kind": "inset-decor",
        "shape": overlay.shape,
        "size": [width, height],
        "radius": radius,
        "border": [border, colour],
        "shadow": overlay.shadow,
        "version": INSET_VERSION,
    }
    final = cache.path_for("overlays", key, suffix=".png")
    if final.is_file():
        return final, (-pad, -pad)
    size = (width + 2 * pad, height + 2 * pad)
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    if overlay.shadow:
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        _shape(
            ImageDraw.Draw(layer),
            (pad, pad + 8, pad + width - 1, pad + height + 7),
            overlay.shape,
            radius,
            (0, 0, 0, 120),
        )
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(11)))
    if border > 0:
        big = Image.new("RGBA", (size[0] * _SS, size[1] * _SS), (0, 0, 0, 0))
        fill = ImageColor.getcolor(colour, "RGBA")
        _shape(
            ImageDraw.Draw(big),
            (pad * _SS, pad * _SS, (pad + width) * _SS - 1, (pad + height) * _SS - 1),
            overlay.shape,
            radius * _SS,
            fill,
        )
        image.alpha_composite(big.resize(size, Image.Resampling.LANCZOS))
    return _save(image, final), (-pad, -pad)


def progress_pictures(
    overlay: ProgressOverlay,
    width: int,
    colour: str,
    cache: Cache,
) -> tuple[Path, Path | None]:
    """The fill picture (a solid bar the width of the frame) and the track under it."""

    def solid(value: str) -> Path:
        key = {
            "kind": "progress",
            "color": value,
            "size": [width, overlay.height],
            "version": INSET_VERSION,
        }
        final = cache.path_for("overlays", key, suffix=".png")
        if final.is_file():
            return final
        picture = Image.new("RGBA", (width, overlay.height), ImageColor.getcolor(value, "RGBA"))
        return _save(picture, final)

    return solid(colour), (solid(overlay.background) if overlay.background else None)


def progress_x_expression(start: float, end: float, fps: int) -> str:
    """The ``x`` of the fill picture over time: its right edge follows the progress, which is
    full on the last frame of the span."""

    span = max(end - start - 1.0 / fps, 1.0 / fps)
    return f"W*min(1,max(0,(t-{format(start, '.9g')})/{format(span, '.9g')}))-w"


__all__ = [
    "InsetLayer",
    "inner_size",
    "inset_decor",
    "inset_footprint",
    "inset_mask",
    "inset_size",
    "inset_span",
    "progress_pictures",
    "progress_x_expression",
]
