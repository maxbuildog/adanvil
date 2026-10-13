"""Split-screen segments: two pictures played together in one frame (before/after, problem and
solution, two products), rendered as one ordinary mezzanine.

Each panel is first rendered on its own, at the size of its half, by the same code as a plain
clip or still image (cover crop around ``focus``, speed, freeze, tone-mapping, auto gamma: a clip
panel behaves exactly like a clip segment of that size, and a clip that is shorter than the
segment holds its last frame). Then the two panels, the gap colour and the optional labels are
put together in one encode of exactly ``duration * fps`` frames.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageColor
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from adanvil import config
from adanvil.ffmpeg import probe, run
from adanvil.text import TextSpec, get_provider
from adanvil.text import render as render_text
from adanvil.vocab import (
    EMOTION_VALUES,
    ROLE_VALUES,
    SCALE_VALUES,
    TYPE_VALUES,
    check_vocabulary,
    normalize_shot_type,
)

from .cache import Cache
from .facefocus import FaceFocusModel
from .filters import encode_args, finalize
from .segment import (
    ClipSegment,
    Crop,
    Format,
    Freeze,
    ImageSegment,
    _bt709_encoder_args,
    _fmt_key,
    _resolve_source,
    render_segment,
)

SPLIT_VERSION = 1  # bump when the pictures of a split change
LABEL_STYLES = Literal["label", "pill", "outline"]


class PanelFreeze(BaseModel):
    """Hold the last frame of a panel clip for ``hold`` seconds while zooming towards ``center``
    (pixels of the panel; default its middle), like the ``freeze`` of a clip segment."""

    model_config = ConfigDict(extra="forbid")

    hold: float = Field(default=1.0, gt=0)
    zoom: float = Field(default=1.35, ge=1.0)
    center: tuple[int, int] | None = None


class SplitPanel(FaceFocusModel):
    """One half of a split: a clip (``in``, ``focus``, ``crop``, ``speed``, ``freeze``, ``gamma``)
    or a still image (``fit``, and ``crop`` to keep one part of a composite picture). Whether it
    is a clip or an image is read from the file."""

    model_config = ConfigDict(populate_by_name=True)

    source: str
    in_: float = Field(default=0.0, alias="in", ge=0)
    focus: tuple[float, float] = (0.5, 0.5)
    crop: Crop | None = None
    speed: float = 1.0
    gamma: float | Literal["auto"] | None = "auto"
    freeze: PanelFreeze | None = None
    fit: Literal["contain", "cover"] = "cover"

    @model_validator(mode="after")
    def validate_speed(self) -> SplitPanel:
        if not 0.25 <= self.speed <= 4:
            raise ValueError("speed must be between 0.25 and 4")
        return self


class SplitSegment(BaseModel):
    """Two panels playing together for ``duration`` seconds.

    ``layout`` is ``side_by_side`` (left and right) or ``top_bottom``; ``gap`` pixels of
    ``gap_color`` separate them (the gap grows by up to three pixels so the panels have even sizes).
    ``labels`` is a pair of short texts, one per panel (first and second), drawn in
    ``label_style`` near the top of each panel, or ``None`` for no label.
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["split"] = "split"
    layout: Literal["side_by_side", "top_bottom"] = "side_by_side"
    duration: float = Field(gt=0)
    panels: list[SplitPanel] = Field(min_length=2, max_length=2)
    gap: int = Field(default=8, ge=0, le=200)
    gap_color: str = "#000000"
    labels: list[str] | None = None
    label_style: LABEL_STYLES = "label"
    label: str | None = None
    shot_type: str | None = None
    scale: str | None = None
    subject: str | None = None
    emotion: str | None = None
    role: str | None = None

    @property
    def source(self) -> str:
        """The source of the first panel (what a report shows for the segment)."""

        return self.panels[0].source

    @field_validator("shot_type", mode="before")
    @classmethod
    def rename_old_shot_type(cls, value: Any) -> Any:
        return normalize_shot_type(value)

    @field_validator("shot_type")
    @classmethod
    def check_shot_type(cls, value: str | None) -> str | None:
        return check_vocabulary(value, "shot_type", TYPE_VALUES)

    @field_validator("scale")
    @classmethod
    def check_scale(cls, value: str | None) -> str | None:
        return check_vocabulary(value, "scale", SCALE_VALUES)

    @field_validator("emotion")
    @classmethod
    def check_emotion(cls, value: str | None) -> str | None:
        return check_vocabulary(value, "emotion", EMOTION_VALUES)

    @field_validator("role")
    @classmethod
    def check_role(cls, value: str | None) -> str | None:
        return check_vocabulary(value, "role", ROLE_VALUES)

    @field_validator("gap_color")
    @classmethod
    def check_gap_color(cls, value: str) -> str:
        try:
            ImageColor.getrgb(value)
        except ValueError as exc:
            raise ValueError(f"invalid colour {value!r}") from exc
        return value

    @field_validator("labels")
    @classmethod
    def check_labels(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and (len(value) != 2 or not all(item.strip() for item in value)):
            raise ValueError("labels must be a list of two non-empty texts (or null)")
        return value

    @model_validator(mode="after")
    def validate_freeze(self) -> SplitSegment:
        for number, panel in enumerate(self.panels):
            if panel.freeze is not None and panel.freeze.hold > self.duration - 0.1:
                raise ValueError(
                    f"panels[{number}].freeze.hold must leave at least 0.1 s of motion "
                    f"(duration is {self.duration:g}s)"
                )
        return self


def split_geometry(
    layout: str, width: int, height: int, gap: int
) -> tuple[list[tuple[int, int, int, int]], int]:
    """The boxes ``(x, y, w, h)`` of the two panels and the real gap in pixels.

    Panel sizes are even (yuv420p), so the gap is the asked ``gap`` plus the few pixels left
    over by the rounding. The two panels and the gap fill the frame exactly.
    """

    if layout == "side_by_side":
        panel = (width - gap) // 2
        panel -= panel % 2
        real = width - 2 * panel
        return [(0, 0, panel, height), (panel + real, 0, panel, height)], real
    panel = (height - gap) // 2
    panel -= panel % 2
    real = height - 2 * panel
    return [(0, 0, width, panel), (0, panel + real, width, panel)], real


def label_specs(seg: SplitSegment, width: int, height: int) -> list[TextSpec]:
    """The text spec of each label, centred near the top of its panel (empty without labels)."""

    if seg.labels is None:
        return []
    boxes, _ = split_geometry(seg.layout, width, height, seg.gap)
    specs: list[TextSpec] = []
    for text, (x, y, w, h) in zip(seg.labels, boxes, strict=True):
        size = max(24, min(56, round(w * 0.075)))
        max_width = w - 40 if seg.layout == "side_by_side" else min(w - 120, 900)
        specs.append(
            TextSpec(
                style=seg.label_style,
                lines=[text],
                x_center=x + w // 2,
                y=y + max(40, round(h * 0.06)),
                size=size,
                max_width=max(80, max_width),
            )
        )
    return specs


def render_labels(
    seg: SplitSegment, width: int, height: int, emoji: Any | None = None
) -> tuple[Image.Image, list[bool]]:
    """The labels of ``seg`` on one transparent full-frame picture, and the overflow flag of
    each (a label wider than its panel after shrinking)."""

    provider = emoji if emoji is not None else get_provider("twemoji")
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    overflow: list[bool] = []
    for spec in label_specs(seg, width, height):
        result = render_text(spec, provider, canvas=(width, height))
        canvas.alpha_composite(result.image)
        result.image.close()
        overflow.append(result.overflow)
    return canvas, overflow


def panel_segment(
    panel: SplitPanel, source_kind: str, duration: float, panel_width: int, panel_height: int
) -> ClipSegment | ImageSegment:
    """The clip or still image segment a panel is rendered as (at the size of its half)."""

    if source_kind == "image":
        return ImageSegment(
            source=panel.source,
            duration=duration,
            fit=panel.fit,
            motion="static",
            crop=panel.crop,
            face_y=panel.face_y,
            face_focus=panel.face_focus,
        )
    hold = panel.freeze.hold if panel.freeze else 0.0
    freeze = None
    if panel.freeze is not None:
        centre = panel.freeze.center or (panel_width // 2, panel_height // 2)
        freeze = Freeze(hold=panel.freeze.hold, zoom=panel.freeze.zoom, center=centre)
    return ClipSegment(
        source=panel.source,
        in_=panel.in_,
        out=panel.in_ + (duration - hold) * panel.speed,
        crop=panel.crop,
        focus=panel.focus,
        face_y=panel.face_y,
        face_focus=panel.face_focus,
        speed=panel.speed,
        gamma=panel.gamma,
        freeze=freeze,
    )


def _gap_hex(colour: str) -> str:
    red, green, blue = ImageColor.getrgb(colour)[:3]
    return f"{red:02X}{green:02X}{blue:02X}"


def _build_split(
    panels: list[Path],
    boxes: list[tuple[int, int, int, int]],
    seg: SplitSegment,
    fmt: Format,
    labels: Path | None,
    output: Path,
) -> None:
    frames = round(seg.duration * fmt.fps)
    if frames < 1:
        raise ValueError("split duration must produce at least one frame")
    args: list[str] = []
    for path in panels:
        args.extend(["-i", str(path)])
    if labels is not None:
        args.extend(["-loop", "1", "-framerate", str(fmt.fps), "-i", str(labels)])
    steps = [
        f"color=c=0x{_gap_hex(seg.gap_color)}:s={fmt.width}x{fmt.height}:r={fmt.fps},"
        "format=gbrp,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p[bg]",
        f"[bg][0:v]overlay={boxes[0][0]}:{boxes[0][1]}:eof_action=repeat[a]",
        f"[a][1:v]overlay={boxes[1][0]}:{boxes[1][1]}:eof_action=repeat[b]",
    ]
    last = "b"
    if labels is not None:
        steps.append("[b][2:v]overlay=0:0:format=auto[c]")
        last = "c"
    steps.append(f"[{last}]{finalize(fmt.fps)},tpad=stop_mode=clone:stop_duration=1[v]")
    run(
        [
            *config.thread_args(),
            *args,
            "-filter_complex",
            ";".join(steps),
            "-map",
            "[v]",
            "-frames:v",
            str(frames),
            "-map_metadata",
            "-1",
            "-metadata:s:v:0",
            "rotate=0",
            *config.thread_args(),
            *encode_args(),
            *_bt709_encoder_args(),
            str(output),
        ]
    )


def _label_picture(seg: SplitSegment, fmt: Format, cache: Cache) -> Path | None:
    if seg.labels is None:
        return None
    key = {
        "labels": seg.labels,
        "style": seg.label_style,
        "layout": seg.layout,
        "gap": seg.gap,
        "format": _fmt_key(fmt),
        "version": SPLIT_VERSION,
    }
    final = cache.path_for("split-label", key, suffix=".png")
    with cache.building(final):
        if final.is_file():
            return final
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(f"{final}.tmp{final.suffix}")
        try:
            picture, _ = render_labels(seg, fmt.width, fmt.height)
            picture.save(temporary, format="PNG")
            picture.close()
            os.replace(temporary, final)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    return final


def render_split(seg: SplitSegment, *, root: Path, cache: Cache, fmt: Format) -> Path:
    """Render a split segment into the normalised mezzanine format (exact frame count)."""

    boxes, _ = split_geometry(seg.layout, fmt.width, fmt.height, seg.gap)
    pieces: list[Path] = []
    for panel, (_x, _y, width, height) in zip(seg.panels, boxes, strict=True):
        source = _resolve_source(root, panel.source)
        kind = probe(source).kind
        if kind == "audio":
            raise ValueError(f"split panel source '{panel.source}' is an audio file")
        pieces.append(
            render_segment(
                panel_segment(panel, kind, seg.duration, width, height),
                root=root,
                cache=cache,
                fmt=Format(width=width, height=height, fps=fmt.fps),
            )
        )
    labels = _label_picture(seg, fmt, cache)
    key = {
        "panels": [Cache.file_key(path) for path in pieces],
        "boxes": boxes,
        "duration": seg.duration,
        "gap_color": _gap_hex(seg.gap_color),
        "labels": None if labels is None else Cache.file_key(labels),
        "format": _fmt_key(fmt),
        "version": SPLIT_VERSION,
    }
    return cache.get_or_build(
        "split", key, lambda output: _build_split(pieces, boxes, seg, fmt, labels, output)
    )


__all__ = [
    "PanelFreeze",
    "SplitPanel",
    "SplitSegment",
    "label_specs",
    "panel_segment",
    "render_labels",
    "render_split",
    "split_geometry",
]
