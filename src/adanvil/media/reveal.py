"""Reveal segments: a before picture that a divider line sweeps away to show the after picture.

The segment is full frame. Each side is first rendered on its own at the size of the frame by
the same code as a plain clip or still image (cover crop around ``focus``, speed, tone-mapping,
auto gamma, a clip shorter than the segment holds its last frame). Then one encode of exactly
``duration * fps`` frames shows the before side for ``hold_before`` seconds, moves a divider
across the frame over ``sweep`` seconds (eased) to uncover the after side, and holds the after
side until the end. The optional labels are part of their own side, so a label disappears or
appears exactly as the divider passes it. A slow zoom (``motion: zoom_in``) is applied to both
sides with the same expression, so they stay registered.

The position of the divider at any time is a closed formula (``reveal_progress`` and
``reveal_boundary``): the filter graph evaluates the same expression per frame, so the picture
is deterministic and the tests can check it to the pixel.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageColor, ImageDraw, ImageFilter
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from adanvil import config
from adanvil.ffmpeg import MAX_SECONDS, filter_script_option, probe, run
from adanvil.safeio import replace_text
from adanvil.text import TextSpec, get_provider
from adanvil.text import render as render_text
from adanvil.vision.framing import DEFAULT_FACE_Y, Framing, align_zooms
from adanvil.vocab import (
    EMOTION_VALUES,
    ROLE_VALUES,
    SCALE_VALUES,
    TYPE_VALUES,
    check_vocabulary,
    normalize_shot_type,
)

from .cache import Cache
from .facefocus import FaceFocusModel, segment_track
from .filters import encode_args, finalize, zoom
from .segment import (
    Format,
    ImageSegment,
    _bt709_encoder_args,
    _fmt_key,
    _resolve_source,
    render_segment,
)
from .split import LABEL_STYLES, SplitPanel, panel_segment

REVEAL_VERSION = 2  # bump when the pictures of a reveal change
SHADOW_PAD = 18  # room on each side of the divider line for its blurred shadow
DIRECTIONS = Literal["left_to_right", "right_to_left", "top_to_bottom"]
# With ``back_and_forth`` the sweep is three eased legs: all the way across, back to the
# middle, then across again. (share of the sweep, from, to)
_BACK_AND_FORTH_LEGS = ((0.4, 0.0, 1.0), (0.3, 1.0, 0.5), (0.3, 0.5, 1.0))


class RevealDivider(BaseModel):
    """The line that sweeps across: ``width`` pixels of ``color``, with a soft ``shadow``."""

    model_config = ConfigDict(extra="forbid")

    width: int = Field(default=6, ge=1, le=40)
    color: str = "#FFFFFF"
    shadow: bool = True

    @field_validator("color")
    @classmethod
    def check_color(cls, value: str) -> str:
        try:
            ImageColor.getrgb(value)
        except ValueError as exc:
            raise ValueError(f"invalid colour {value!r}") from exc
        return value


class RevealSegment(FaceFocusModel):
    """A before picture and an after picture, revealed by a sweeping divider.

    ``before`` and ``after`` are clips (``in``, ``focus``, ``crop``, ``speed``, ``gamma``) or
    still images (``fit``, and ``crop`` in source pixels to cut one half out of a side-by-side
    composite), like the panels of a split. ``focus``, when given, is used for both
    sides (so the faces line up) instead of their own. The before side shows alone for
    ``hold_before`` seconds, then the divider crosses the frame in ``direction`` over ``sweep``
    seconds (eased) and the after side stays until ``duration``. ``back_and_forth`` makes the
    sweep go all the way across, back to the middle and across again (a slider being dragged),
    all within ``sweep``. ``labels`` is a pair of short texts (before, after) drawn in
    ``label_style`` near the top, each on its own side. ``divider`` is ``null`` for a plain
    wipe. ``motion`` ``zoom_in`` zooms both sides slowly, in sync, up to ``zoom_to``.

    ``focus: "face"`` keeps the main face of each side centred, ``face_y`` of the way down.
    ``align_faces`` (default true) goes further when both sides show exactly one face (the
    vision environment finds them): both faces are put at the same place AND at the same size
    (the smaller face is enlarged to the bigger one, up to 1.6x), so the divider crosses one
    face that changes instead of two that jump.
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["reveal"] = "reveal"
    duration: float = Field(gt=0, le=MAX_SECONDS)
    before: SplitPanel
    after: SplitPanel
    direction: DIRECTIONS = "left_to_right"
    hold_before: float = Field(default=0.8, ge=0, le=MAX_SECONDS)
    sweep: float = Field(default=0.9, gt=0, le=MAX_SECONDS)
    back_and_forth: bool = False
    focus: tuple[float, float] | None = None
    align_faces: bool = True
    labels: list[str] | None = None
    label_style: LABEL_STYLES = "label"
    divider: RevealDivider | None = Field(default_factory=RevealDivider)
    motion: Literal["static", "zoom_in"] = "static"
    zoom_to: float = Field(default=1.04, ge=1.0, le=1.3)
    label: str | None = None
    shot_type: str | None = None
    scale: str | None = None
    subject: str | None = None
    emotion: str | None = None
    role: str | None = None

    @property
    def source(self) -> str:
        """The source of the before side (what a report shows for the segment)."""

        return self.before.source

    @property
    def sides(self) -> tuple[SplitPanel, SplitPanel]:
        return (self.before, self.after)

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

    @field_validator("focus")
    @classmethod
    def check_focus(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        if value is not None and not all(0.0 <= item <= 1.0 for item in value):
            raise ValueError("focus must be two numbers between 0 and 1")
        return value

    @field_validator("labels")
    @classmethod
    def check_labels(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and (len(value) != 2 or not all(item.strip() for item in value)):
            raise ValueError("labels must be a list of two non-empty texts (or null)")
        return value

    @model_validator(mode="after")
    def validate_timing(self) -> RevealSegment:
        if self.hold_before + self.sweep > self.duration + 1e-9:
            raise ValueError(
                f"hold_before ({self.hold_before:g}s) + sweep ({self.sweep:g}s) must fit in "
                f"duration ({self.duration:g}s): the after picture needs to be reached"
            )
        for name, side in (("before", self.before), ("after", self.after)):
            if side.freeze is not None and side.freeze.hold > self.duration - 0.1:
                raise ValueError(
                    f"{name}.freeze.hold must leave at least 0.1 s of motion "
                    f"(duration is {self.duration:g}s)"
                )
        return self

    def side_focus(self, side: SplitPanel) -> tuple[float, float]:
        """The focus a side is cropped around: the segment's, else its own."""

        return self.focus if self.focus is not None else side.focus

    def side_face(self, side: SplitPanel) -> tuple[bool, float]:
        """Whether a side is framed on its face, and at which height: the segment's choice
        when it has one, else the side's own."""

        if self.face_focus:
            return True, self.face_y
        return side.face_focus, side.face_y

    def zoom_centre(self) -> tuple[float, float]:
        """The point of the frame the slow zoom (``motion: zoom_in``) heads for."""

        if self.face_focus:
            return 0.5, self.face_y
        return self.side_focus(self.before)


# --------------------------------------------------------------------------- the motion


def _ease(u: float) -> float:
    """Smoothstep, an ease in and out from 0 to 1 (slow start, fast middle, slow end)."""

    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def reveal_legs(
    hold_before: float, sweep: float, back_and_forth: bool
) -> list[tuple[float, float, float, float]]:
    """The eased legs of the sweep as ``(start, end, from, to)``: seconds in the segment and
    the share of the frame uncovered (0 is all before, 1 is all after)."""

    if not back_and_forth:
        return [(hold_before, hold_before + sweep, 0.0, 1.0)]
    legs: list[tuple[float, float, float, float]] = []
    cursor = hold_before
    for share, start, end in _BACK_AND_FORTH_LEGS:
        legs.append((cursor, cursor + sweep * share, start, end))
        cursor += sweep * share
    return legs


def reveal_progress(t: float, hold_before: float, sweep: float, back_and_forth: bool) -> float:
    """How much of the after picture is uncovered at ``t`` seconds into the segment."""

    legs = reveal_legs(hold_before, sweep, back_and_forth)
    for start, end, source, target in legs[:-1]:
        if t < end:
            return source + (target - source) * _ease((t - start) / (end - start))
    start, end, source, target = legs[-1]
    return source + (target - source) * _ease((t - start) / (end - start))


def divider_margin(divider: RevealDivider | None) -> int:
    """Half the size of the divider picture across its travel: the divider starts and ends
    this far outside the frame, so it is never seen while a side is alone on screen."""

    return 0 if divider is None else _divider_extent(divider) // 2


def _divider_extent(divider: RevealDivider) -> int:
    extent = divider.width + 2 * (SHADOW_PAD if divider.shadow else 0)
    return extent + extent % 2


def reveal_boundary(
    progress: float, direction: str, width: int, height: int, margin: int
) -> float:
    """The coordinate of the divider (its centre) along its travel for ``progress``: x for a
    horizontal sweep, y for ``top_to_bottom``."""

    size = height if direction == "top_to_bottom" else width
    travel = size + 2 * margin
    if direction == "right_to_left":
        return size + margin - progress * travel
    return -margin + progress * travel


def reveal_position(
    t: float, seg: RevealSegment, width: int, height: int
) -> int:
    """The pixel the divider centre is at, ``t`` seconds into the segment (what the render
    shows, to the pixel)."""

    progress = reveal_progress(t, seg.hold_before, seg.sweep, seg.back_and_forth)
    value = reveal_boundary(progress, seg.direction, width, height, divider_margin(seg.divider))
    return math.floor(value + 0.5)


def _n(value: float) -> str:
    return format(float(value), ".9g")


def _ease_expression(u: str) -> str:
    return f"({u})*({u})*(3-2*({u}))"


def progress_expression(hold_before: float, sweep: float, back_and_forth: bool) -> str:
    """The ffmpeg expression of ``reveal_progress`` in the time variable ``t``."""

    def leg(start: float, end: float, source: float, target: float) -> str:
        u = f"clip((t-{_n(start)})/{_n(end - start)},0,1)"
        return f"({_n(source)}+({_n(target - source)})*{_ease_expression(u)})"

    legs = reveal_legs(hold_before, sweep, back_and_forth)
    expression = leg(*legs[-1])
    for start, end, source, target in reversed(legs[:-1]):
        expression = f"if(lt(t,{_n(end)}),{leg(start, end, source, target)},{expression})"
    return expression


def boundary_expression(seg: RevealSegment, width: int, height: int) -> str:
    """The ffmpeg expression (rounded to a pixel) of ``reveal_boundary`` in ``t``."""

    progress = progress_expression(seg.hold_before, seg.sweep, seg.back_and_forth)
    margin = divider_margin(seg.divider)
    size = height if seg.direction == "top_to_bottom" else width
    travel = size + 2 * margin
    if seg.direction == "right_to_left":
        return f"round({_n(size + margin)}-({progress})*{_n(travel)})"
    return f"round({_n(-margin)}+({progress})*{_n(travel)})"


# ------------------------------------------------------------------------ the pictures


def label_specs(seg: RevealSegment, width: int, height: int) -> list[TextSpec]:
    """The text spec of the before label and the after label.

    Both sit near the top. For a horizontal sweep each is placed on the side where its picture
    stays longest (the before label at the far end, the after label at the start), so the two
    never overlap while the divider crosses the middle; for a vertical sweep both are centred.
    """

    if seg.labels is None:
        return []
    size = max(24, min(60, round(width * 0.05)))
    y = max(60, round(height * 0.07))
    if seg.direction == "left_to_right":
        centres = [round(width * 0.72), round(width * 0.28)]
    elif seg.direction == "right_to_left":
        centres = [round(width * 0.28), round(width * 0.72)]
    else:
        centres = [width // 2, width // 2]
    max_width = round(width * 0.4) if seg.direction != "top_to_bottom" else round(width * 0.8)
    return [
        TextSpec(
            style=seg.label_style,
            lines=[text],
            x_center=centre,
            y=y,
            size=size,
            max_width=max(80, max_width),
        )
        for text, centre in zip(seg.labels, centres, strict=True)
    ]


def render_labels(
    seg: RevealSegment, width: int, height: int, emoji: Any | None = None
) -> tuple[list[Image.Image], list[bool]]:
    """The two labels, each on its own transparent full-frame picture, and the overflow flag of
    each (a label wider than its room after shrinking)."""

    provider = emoji if emoji is not None else get_provider("twemoji")
    pictures: list[Image.Image] = []
    overflow: list[bool] = []
    for spec in label_specs(seg, width, height):
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        result = render_text(spec, provider, canvas=(width, height))
        canvas.alpha_composite(result.image)
        result.image.close()
        pictures.append(canvas)
        overflow.append(result.overflow)
    return pictures, overflow


def render_divider(
    divider: RevealDivider, direction: str, width: int, height: int
) -> Image.Image:
    """The divider picture: a line (and its shadow) the full height of the frame for a
    horizontal sweep, the full width for a vertical one."""

    extent = _divider_extent(divider)
    pad = (extent - divider.width) // 2
    vertical = direction == "top_to_bottom"
    size = (width, extent) if vertical else (extent, height)
    line_box = (
        (0, pad, width, pad + divider.width) if vertical else (pad, 0, pad + divider.width, height)
    )
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    if divider.shadow:
        shade = Image.new("RGBA", size, (0, 0, 0, 0))
        ImageDraw.Draw(shade).rectangle(
            (line_box[0], line_box[1], line_box[2] - 1, line_box[3] - 1), fill=(0, 0, 0, 150)
        )
        image.alpha_composite(shade.filter(ImageFilter.GaussianBlur(SHADOW_PAD / 3)))
    colour = ImageColor.getcolor(divider.color, "RGBA")
    ImageDraw.Draw(image).rectangle(
        (line_box[0], line_box[1], line_box[2] - 1, line_box[3] - 1), fill=colour
    )
    return image


def _save_png(builder: Any, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{final}.tmp{final.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        picture = builder()
        picture.save(temporary, format="PNG")
        picture.close()
        os.replace(temporary, final)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _label_pictures(seg: RevealSegment, fmt: Format, cache: Cache) -> list[Path]:
    if seg.labels is None:
        return []
    paths = [
        cache.path_for(
            "reveal-label",
            {
                "labels": seg.labels,
                "which": number,
                "style": seg.label_style,
                "direction": seg.direction,
                "format": _fmt_key(fmt),
                "version": REVEAL_VERSION,
            },
            suffix=".png",
        )
        for number in range(2)
    ]
    if all(path.is_file() for path in paths):
        return paths
    pictures, _ = render_labels(seg, fmt.width, fmt.height)
    for path, picture in zip(paths, pictures, strict=True):
        with cache.building(path):
            if path.is_file():
                picture.close()
            else:
                _save_png(lambda picture=picture: picture, path)
    return paths


def _divider_picture(seg: RevealSegment, fmt: Format, cache: Cache) -> Path | None:
    divider = seg.divider
    if divider is None:
        return None
    key = {
        "divider": divider.model_dump(mode="json"),
        "direction": "vertical" if seg.direction == "top_to_bottom" else "horizontal",
        "format": _fmt_key(fmt),
        "version": REVEAL_VERSION,
    }
    final = cache.path_for("reveal-divider", key, suffix=".png")
    with cache.building(final):
        if not final.is_file():
            _save_png(
                lambda: render_divider(divider, seg.direction, fmt.width, fmt.height), final
            )
    return final


# ------------------------------------------------------------------------- the encode


def _side_segment(
    seg: RevealSegment,
    side: SplitPanel,
    kind: str,
    fmt: Format,
    framing: Framing | None = None,
) -> Any:
    """The clip or still image segment a side is rendered as (at the size of the frame).

    ``framing`` is the face framing worked out for the pair (see ``_aligned_framings``)."""

    focus = seg.side_focus(side)
    face, face_y = seg.side_face(side)
    face = face or framing is not None
    if kind == "image":
        built: Any = ImageSegment(
            source=side.source,
            duration=seg.duration,
            fit=side.fit,
            focus=focus,
            crop=side.crop,
            motion="static",
            face_y=face_y,
            face_focus=face,
        )
    else:
        built = panel_segment(
            side.model_copy(update={"focus": focus, "face_y": face_y, "face_focus": face}),
            kind,
            seg.duration,
            fmt.width,
            fmt.height,
        )
    return built.with_framing_value(framing) if framing is not None else built


def _aligned_framings(
    seg: RevealSegment, root: Path, fmt: Format, kinds: list[str]
) -> list[Framing | None]:
    """Put the face of both sides at the same place and the same size, when each side shows
    exactly one face; ``[None, None]`` (the sides are framed on their own) otherwise."""

    none: list[Framing | None] = [None, None]
    if not seg.align_faces:
        return none
    tracks = []
    for side, kind in zip(seg.sides, kinds, strict=True):
        if kind == "image" and side.fit == "contain":
            return none
        source = _resolve_source(root, side.source)
        window = _side_segment(seg, side, kind, fmt)
        track = segment_track(window, root, source, "image" if kind == "image" else "video")
        if track is None or not track.single:
            return none
        tracks.append(track)
    zooms = align_zooms(tracks[0], tracks[1], fmt.width, fmt.height)
    height = seg.face_y if seg.face_focus else DEFAULT_FACE_Y
    return [Framing(track, height, zoom) for track, zoom in zip(tracks, zooms, strict=True)]


def mask_white_size(seg: RevealSegment, width: int, height: int) -> tuple[int, int]:
    """The size of the white picture that is slid over the black frame to make the mask: the
    frame plus the divider margin along the travel, so it still reaches the far edge when the
    divider has gone past it."""

    margin = divider_margin(seg.divider)
    if seg.direction == "top_to_bottom":
        return width, height + margin
    return width + margin, height


def mask_positions(seg: RevealSegment, width: int, height: int) -> tuple[str, str]:
    """The ``x`` and ``y`` expressions that place the white picture over a black frame so that
    the white part is exactly the uncovered (after) area."""

    boundary = boundary_expression(seg, width, height)
    white_w, white_h = mask_white_size(seg, width, height)
    if seg.direction == "left_to_right":
        return f"{boundary}-{white_w}", "0"
    if seg.direction == "right_to_left":
        return boundary, "0"
    return "0", f"{boundary}-{white_h}"


def divider_positions(seg: RevealSegment, width: int, height: int) -> tuple[str, str]:
    """The ``x`` and ``y`` expressions of the divider picture, centred on the boundary."""

    assert seg.divider is not None
    half = _divider_extent(seg.divider) // 2
    boundary = boundary_expression(seg, width, height)
    if seg.direction == "top_to_bottom":
        return "0", f"{boundary}-{half}"
    return f"{boundary}-{half}", "0"


def reveal_graph(
    seg: RevealSegment, fmt: Format, *, labels: bool, divider: bool
) -> str:
    """The filter graph of the encode. Inputs: 0 before, 1 after, then the two label pictures
    (when ``labels``), then the divider picture (when ``divider``)."""

    width, height, fps = fmt.width, fmt.height, fmt.fps
    pad = f"tpad=stop_mode=clone:stop_duration={_n(seg.duration + 1)}"
    zoom_filter = ""
    if seg.motion == "zoom_in":
        centre = seg.zoom_centre()
        zoom_filter = "," + zoom(
            1.0, seg.zoom_to, seg.duration, *centre, width=width, height=height, fps=fps
        )
    steps = [
        f"[0:v]setpts=PTS-STARTPTS,{pad}{zoom_filter},format=yuv420p[b0]",
        f"[1:v]setpts=PTS-STARTPTS,{pad}{zoom_filter},format=yuv420p[a0]",
    ]
    index = 2
    before, after = "b0", "a0"
    if labels:
        steps.append(f"[b0][{index}:v]overlay=0:0:format=auto,format=yuv420p[b1]")
        steps.append(f"[a0][{index + 1}:v]overlay=0:0:format=auto,format=yuv420p[a1]")
        before, after = "b1", "a1"
        index += 2
    mask_x, mask_y = mask_positions(seg, width, height)
    white_w, white_h = mask_white_size(seg, width, height)
    steps += [
        # The mask is built in 4:4:4: overlay snaps its position to the chroma grid (even
        # pixels in 4:2:0), and the edge of the reveal has to land on the exact pixel.
        f"color=c=black:s={width}x{height}:r={fps},format=yuv444p[mb]",
        f"color=c=white:s={white_w}x{white_h}:r={fps},format=yuv444p[mw]",
        f"[mb][mw]overlay=x='{mask_x}':y='{mask_y}':eof_action=repeat:format=yuv444,"
        "format=gray[mask]",
        f"[{after}][mask]alphamerge[am]",
        f"[{before}][am]overlay=0:0:format=auto:eof_action=repeat[c]",
    ]
    last = "c"
    if divider:
        div_x, div_y = divider_positions(seg, width, height)
        steps.append(
            f"[c][{index}:v]overlay=x='{div_x}':y='{div_y}':format=auto:eof_action=repeat[d]"
        )
        last = "d"
    steps.append(f"[{last}]{finalize(fps)},tpad=stop_mode=clone:stop_duration=1[v]")
    return ";".join(steps)


def _build_reveal(
    sides: list[Path],
    labels: list[Path],
    divider: Path | None,
    seg: RevealSegment,
    fmt: Format,
    output: Path,
) -> None:
    frames = round(seg.duration * fmt.fps)
    if frames < 1:
        raise ValueError("reveal duration must produce at least one frame")
    args: list[str] = []
    for path in sides:
        args.extend(["-i", str(path)])
    for path in (*labels, *([divider] if divider is not None else [])):
        args.extend(["-loop", "1", "-framerate", str(fmt.fps), "-i", str(path)])
    graph = Path(f"{output}.filters.txt")
    replace_text(
        graph,
        reveal_graph(seg, fmt, labels=bool(labels), divider=divider is not None),
        encoding="utf-8",
    )
    try:
        run(
            [
                *config.thread_args(),
                *args,
                filter_script_option(),
                str(graph),
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
    finally:
        graph.unlink(missing_ok=True)


def render_reveal(seg: RevealSegment, *, root: Path, cache: Cache, fmt: Format) -> Path:
    """Render a reveal segment into the normalised mezzanine format (exact frame count)."""

    pieces: list[Path] = []
    kinds: list[str] = []
    for name, side in (("before", seg.before), ("after", seg.after)):
        source = _resolve_source(root, side.source)
        kind = probe(source).kind
        if kind == "audio":
            raise ValueError(f"reveal {name} source '{side.source}' is an audio file")
        kinds.append(kind)
    framings = _aligned_framings(seg, root, fmt, kinds)
    for side, kind, framing in zip(seg.sides, kinds, framings, strict=True):
        pieces.append(
            render_segment(
                _side_segment(seg, side, kind, fmt, framing), root=root, cache=cache, fmt=fmt
            )
        )
    labels = _label_pictures(seg, fmt, cache)
    divider = _divider_picture(seg, fmt, cache)
    key = {
        "sides": [Cache.file_key(path) for path in pieces],
        "labels": [Cache.file_key(path) for path in labels],
        "divider": None if divider is None else Cache.file_key(divider),
        "duration": seg.duration,
        "direction": seg.direction,
        "hold_before": seg.hold_before,
        "sweep": seg.sweep,
        "back_and_forth": seg.back_and_forth,
        "motion": seg.motion,
        "zoom_to": seg.zoom_to,
        "focus": list(seg.side_focus(seg.before)),
        "format": _fmt_key(fmt),
        "version": REVEAL_VERSION,
    }
    return cache.get_or_build(
        "reveal",
        key,
        lambda output: _build_reveal(pieces, labels, divider, seg, fmt, output),
    )


__all__ = [
    "RevealDivider",
    "RevealSegment",
    "divider_margin",
    "label_specs",
    "progress_expression",
    "render_divider",
    "render_labels",
    "render_reveal",
    "reveal_boundary",
    "reveal_graph",
    "reveal_legs",
    "reveal_position",
    "reveal_progress",
]
