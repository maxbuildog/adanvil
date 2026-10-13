"""Platform safe zones: where the apps draw their own interface over a vertical ad.

Each platform covers part of a 9:16 frame with its own buttons and text: a top bar, a column of
buttons on the right edge, and a caption / description / call-to-action band at the bottom. A
caption or a key sticker under that interface is hidden or fights with it. The boxes below are
``(left, top, right, bottom)`` in the pixels of a 1080x1920 frame (they scale to any other size
with the same shape). They are guidelines drawn from the interfaces of the three placements an
ad usually runs in, not rules: the apps change them, and a long caption description or a
sponsored label can grow the bottom band. Treat a finding as a question and look at the preview.

========  ===========================================  =================================
Platform  Box (left, top, right, bottom)               What the app draws there
========  ===========================================  =================================
tiktok    top_bar        (0, 0, 1080, 150)             search and Following / For You tabs
tiktok    right_buttons  (900, 760, 1080, 1480)        avatar, like, comment, share, sound
tiktok    bottom_band    (0, 1480, 1080, 1920)         account, description, sound, CTA
reels     top_bar        (0, 0, 1080, 250)             Reels title, camera, account row
reels     right_buttons  (940, 1000, 1080, 1580)       like, comment, share, menu
reels     bottom_band    (0, 1580, 1080, 1920)         account, caption, audio, CTA button
stories   top_bar        (0, 0, 1080, 250)             progress bars and account row
stories   bottom_band    (0, 1580, 1080, 1920)         reply bar and call-to-action
========  ===========================================  =================================

Stories have no button column. A box counts as inside a zone when the overlap is at least
``MIN_OVERLAP`` pixels (at 1080 wide) in both directions.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

Box = tuple[float, float, float, float]

REFERENCE_SIZE = (1080, 1920)
PLATFORMS = ("tiktok", "reels", "stories")
MIN_OVERLAP = 24  # pixels at 1080 wide, in both directions, to count as inside a zone

SAFE_ZONES: dict[str, dict[str, tuple[int, int, int, int]]] = {
    "tiktok": {
        "top_bar": (0, 0, 1080, 150),
        "right_buttons": (900, 760, 1080, 1480),
        "bottom_band": (0, 1480, 1080, 1920),
    },
    "reels": {
        "top_bar": (0, 0, 1080, 250),
        "right_buttons": (940, 1000, 1080, 1580),
        "bottom_band": (0, 1580, 1080, 1920),
    },
    "stories": {
        "top_bar": (0, 0, 1080, 250),
        "bottom_band": (0, 1580, 1080, 1920),
    },
    # Feed (4:5 and 1:1): the video sits between the account row and the call-to-action bar of
    # the post, so only a thin band at each edge is covered (the sound button, the duration and
    # the "Sponsored" label). Like the others the boxes are the 1080x1920 shape and scale with
    # the frame height, so a 1350 px tall 4:5 frame covers its top 70 and bottom 98 px.
    "feed": {
        "top_bar": (0, 0, 1080, 100),
        "bottom_band": (0, 1780, 1080, 1920),
    },
}

# The placement a rendered aspect ratio is meant for: 9:16 runs in Stories, Reels and TikTok,
# 4:5 and 1:1 in the Feed.
PLACEMENT_BY_RATIO: dict[str, tuple[str, ...]] = {
    "9:16": ("tiktok", "reels", "stories"),
    "4:5": ("feed",),
    "1:1": ("feed",),
}

_COLOURS = {
    "top_bar": (255, 170, 0),
    "right_buttons": (255, 60, 90),
    "bottom_band": (60, 150, 255),
}


def platforms_for_size(width: int, height: int) -> tuple[str, ...] | None:
    """The platforms of a frame whose shape is a known placement ratio (4:5 and 1:1 are the
    Feed), or ``None`` for any other shape (the caller keeps its default)."""

    if width <= 0 or height <= 0:
        return None
    ratio = width / height
    for name, platforms in PLACEMENT_BY_RATIO.items():
        across, down = (int(part) for part in name.split(":"))
        if abs(ratio - across / down) < 0.01:
            return platforms
    return None


def check_platforms(platforms: Iterable[str] | None) -> tuple[str, ...]:
    """The platforms asked for (all three by default), validated."""

    if platforms is None:
        return PLATFORMS
    chosen = tuple(dict.fromkeys(str(name) for name in platforms))
    unknown = [name for name in chosen if name not in SAFE_ZONES]
    if unknown or not chosen:
        raise ValueError(
            f"platforms must be a non-empty list of: {', '.join(PLATFORMS)} (got {list(platforms)})"
        )
    return chosen


def safe_zones(platform: str, size: tuple[int, int] = REFERENCE_SIZE) -> dict[str, Box]:
    """The zones of ``platform`` scaled to a frame of ``size`` (width, height)."""

    if platform not in SAFE_ZONES:
        raise ValueError(f"unknown platform {platform!r}: use one of {', '.join(PLATFORMS)}")
    scale_x = size[0] / REFERENCE_SIZE[0]
    scale_y = size[1] / REFERENCE_SIZE[1]
    return {
        name: (box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y)
        for name, box in SAFE_ZONES[platform].items()
    }


def overlap(box: Box, zone: Box) -> tuple[float, float]:
    """Width and height of the overlap of two boxes (0 when they do not meet)."""

    return (
        max(0.0, min(box[2], zone[2]) - max(box[0], zone[0])),
        max(0.0, min(box[3], zone[3]) - max(box[1], zone[1])),
    )


def collisions(
    box: Box, platforms: Iterable[str], size: tuple[int, int] = REFERENCE_SIZE
) -> list[dict[str, Any]]:
    """The zones ``box`` falls into: ``{platform, zone, share}`` where ``share`` is the part of
    the box (0 to 1) inside the zone. A box that only grazes a zone (under ``MIN_OVERLAP``
    pixels at 1080 wide in either direction) does not count."""

    minimum = MIN_OVERLAP * size[0] / REFERENCE_SIZE[0]
    area = max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
    found: list[dict[str, Any]] = []
    for platform in platforms:
        for name, zone in safe_zones(platform, size).items():
            across, down = overlap(box, zone)
            if across >= minimum and down >= minimum:
                found.append(
                    {"platform": platform, "zone": name, "share": round(across * down / area, 3)}
                )
    return found


def zones_table(platforms: Iterable[str] | None = None) -> dict[str, dict[str, list[int]]]:
    """The documented boxes (1080x1920 pixels) as JSON-ready lists."""

    return {
        platform: {name: list(box) for name, box in SAFE_ZONES[platform].items()}
        for platform in check_platforms(platforms)
    }


def draw_preview(
    frame: Image.Image,
    platforms: Iterable[str],
    items: Iterable[dict[str, Any]],
    out: Path,
    *,
    panel_height: int = 720,
) -> Path:
    """Draw the zones of each platform over ``frame`` (one panel per platform, side by side) and
    outline each ``item`` (``{"box": (l, t, r, b), "label": str, "inside": bool}``, boxes in the
    pixels of ``frame``), red when it falls inside a zone and green when it is clear. Saved as a
    JPEG at ``out``."""

    chosen = tuple(platforms)
    rgb = frame.convert("RGB")
    scale = panel_height / rgb.height
    panel_width = max(1, round(rgb.width * scale))
    base = rgb.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
    strip = 36
    gap = 10
    canvas = Image.new(
        "RGB",
        (len(chosen) * panel_width + (len(chosen) - 1) * gap, panel_height + strip),
        (20, 20, 20),
    )
    listed = list(items)
    for number, platform in enumerate(chosen):
        panel = base.convert("RGBA")
        shade = Image.new("RGBA", panel.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(shade)
        for name, zone in safe_zones(platform, rgb.size).items():
            box = [value * scale for value in zone]
            colour = _COLOURS.get(name, (255, 255, 255))
            draw.rectangle(box, fill=(*colour, 90), outline=(*colour, 255), width=2)
            draw.text((box[0] + 6, box[1] + 4), name, fill=(255, 255, 255, 255))
        panel = Image.alpha_composite(panel, shade).convert("RGB")
        pen = ImageDraw.Draw(panel)
        for item in listed:
            left, top, right, bottom = (value * scale for value in item["box"])
            pen.rectangle(
                [left, top, right, bottom],
                outline=(255, 40, 40) if item.get("inside") else (40, 230, 90),
                width=3,
            )
        x = number * (panel_width + gap)
        canvas.paste(panel, (x, strip))
        ImageDraw.Draw(canvas).text((x + 8, 10), platform.upper(), fill=(255, 255, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, format="JPEG", quality=88)
    return out


__all__ = [
    "MIN_OVERLAP",
    "PLACEMENT_BY_RATIO",
    "PLATFORMS",
    "SAFE_ZONES",
    "check_platforms",
    "collisions",
    "draw_preview",
    "overlap",
    "platforms_for_size",
    "safe_zones",
    "zones_table",
]
