"""Pydantic segment models and ffmpeg-backed segment rendering."""

from __future__ import annotations

import os
from array import array
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from adanvil import config
from adanvil.ffmpeg import measure_yavg, probe, run
from adanvil.project import resolve_in_project
from adanvil.textremoval import RemoveText, build_mask, removal_filter
from adanvil.vision.framing import face_cover
from adanvil.vocab import (
    EMOTION_VALUES,
    ROLE_VALUES,
    SCALE_VALUES,
    TYPE_VALUES,
    check_vocabulary,
    normalize_shot_type,
)

from .cache import Cache
from .facefocus import FaceFocusModel, with_framing
from .filters import (
    auto_gamma,
    cover,
    encode_args,
    finalize,
    freeze_zoom,
    punch_in,
    source_crop,
    tonemap,
)
from .filters import (
    gamma as gamma_filter,
)
from .filters import (
    speed as speed_filter,
)
from .filters import (
    zoom as zoom_filter,
)

if TYPE_CHECKING:
    from .reveal import RevealSegment
    from .split import SplitSegment


class Zoom(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: float = Field(1.0, alias="from")
    to: float = 1.04
    focus: tuple[float, float] = (0.5, 0.5)


class Freeze(BaseModel):
    hold: float = 1.0
    zoom: float = 1.35
    center: tuple[int, int] = (540, 960)


class Crop(BaseModel):
    x: int
    y: int
    w: int
    h: int


class Punch(BaseModel):
    """An instant zoom on a clip: from ``at`` (seconds in the segment) for ``hold`` seconds.

    There is no ramp: the picture jumps to ``zoom`` and jumps back. It is centred on ``center``
    (output pixels), else on the segment's ``focus``. Only a punch above 1.25 counts as digital
    zoom for the ``digital_zoom`` warning.
    """

    model_config = ConfigDict(extra="forbid")

    at: float = Field(ge=0)
    zoom: float = Field(default=1.15, gt=1.0, le=3.0)
    hold: float = Field(default=0.5, gt=0)
    center: tuple[int, int] | None = None


class AudioFrom(BaseModel):
    """Play the audio of another clip, from ``in`` (seconds), under this segment's picture.

    The segment shows its own video for its own duration and plays ``source`` at normal
    speed from ``in`` for that long: an L-cut (the creator keeps talking over a product
    shot). Only meaningful with ``audio.clips``.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    source: str
    in_: float = Field(alias="in", ge=0)


class ClipSegment(FaceFocusModel):
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["clip"] = "clip"
    source: str
    in_: float = Field(alias="in")
    out: float
    crop: Crop | None = None
    # Where the cover crop keeps the picture (0 to 1 on each axis), or "face": keep the main
    # face centred, ``face_y`` of the way down (needs the vision environment or an indexed clip).
    focus: tuple[float, float] = (0.5, 0.5)
    speed: float = 1.0
    gamma: float | Literal["auto"] | None = "auto"
    zoom: Zoom | None = None
    freeze: Freeze | None = None
    remove_text: RemoveText | None = None
    label: str | None = None
    # Reference-remake annotations (breakdown vocabularies, checked by Storyboard).
    shot_type: str | None = None
    scale: str | None = None
    subject: str | None = None
    emotion: str | None = None
    role: str | None = None
    # With ``audio.clips``: play the audio of another clip instead of this clip's own.
    audio_from: AudioFrom | None = None
    # Instant zooms on a strong word (see ``Punch``); times are inside the segment.
    punch: list[Punch] = Field(default_factory=list)

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

    @model_validator(mode="after")
    def validate_timing_and_speed(self) -> ClipSegment:
        if self.out <= self.in_:
            raise ValueError("out must be greater than in")
        if not 0.25 <= self.speed <= 4:
            raise ValueError("speed must be between 0.25 and 4")
        if self.punch:
            length = (self.out - self.in_) / self.speed + (self.freeze.hold if self.freeze else 0)
            ordered = sorted(self.punch, key=lambda item: item.at)
            for index, item in enumerate(ordered):
                if item.at + item.hold > length + 1e-6:
                    raise ValueError(
                        f"punch at {item.at:g}s for {item.hold:g}s ends after the segment "
                        f"({length:g}s): shorten hold or move at"
                    )
                if index and item.at < ordered[index - 1].at + ordered[index - 1].hold - 1e-9:
                    raise ValueError(
                        f"punch at {item.at:g}s starts before the previous punch ends "
                        f"({ordered[index - 1].at + ordered[index - 1].hold:g}s): punches "
                        "must not overlap"
                    )
        return self


class ImageSegment(FaceFocusModel):
    type: Literal["image"] = "image"
    source: str
    duration: float
    motion: Literal["static", "zoom_in"] = "static"
    zoom_to: float = 1.04
    background: str = "blur"
    fit: Literal["contain", "cover"] = "contain"
    # Where a ``cover`` crop keeps the picture (0 to 1 on each axis); it does nothing for
    # ``contain``. ``reveal`` uses it so both of its sides line up on the same subject.
    focus: tuple[float, float] = (0.5, 0.5)
    # A rectangle of the source picture, in source pixels, kept before anything else: it lets a
    # side of a ``reveal`` or a ``split`` be cut out of a brand's side-by-side composite (the
    # before half, the after half) without preparing files.
    crop: Crop | None = None
    label: str | None = None
    shot_type: str | None = None
    scale: str | None = None
    subject: str | None = None
    emotion: str | None = None
    role: str | None = None

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
    def check_focus(cls, value: tuple[float, float]) -> tuple[float, float]:
        if not all(0.0 <= item <= 1.0 for item in value):
            raise ValueError("focus must be two numbers between 0 and 1")
        return value


class Format(BaseModel):
    width: int = 1080
    height: int = 1920
    fps: int = 30


Segment = ClipSegment | ImageSegment
_DEFAULT_FORMAT = Format()


ZOOM_ENGINE = 2  # bump when the way a slow zoom is drawn changes (it is part of cache keys)


def _number(value: float | int) -> str:
    return format(float(value), ".12g")


def segment_duration(seg: Segment) -> float:
    if isinstance(seg, ClipSegment):
        return (seg.out - seg.in_) / seg.speed + (seg.freeze.hold if seg.freeze else 0)
    return seg.duration


def _resolve_source(root: Path, source: str) -> Path:
    try:
        return resolve_in_project(root, source)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"segment source: {exc}") from exc


def _join_filters(*filters: str) -> str:
    return ",".join(filter(None, filters))


def _fmt_key(fmt: Format) -> dict[str, int]:
    return {"width": fmt.width, "height": fmt.height, "fps": fmt.fps}


def _clip_base_key(source: Path, seg: ClipSegment, fmt: Format) -> dict[str, Any]:
    return {
        "source": Cache.file_key(source),
        "in": seg.in_,
        "out": seg.out,
        "crop": seg.crop.model_dump(mode="json") if seg.crop else None,
        "focus": list(seg.focus),
        **({"face": seg.framing.key()} if seg.framing is not None else {}),
        "speed": seg.speed,
        "remove_text": seg.remove_text.model_dump(mode="json") if seg.remove_text else None,
        "format": _fmt_key(fmt),
    }


def _clip_post_filter(seg: ClipSegment, fmt: Format) -> str:
    source_crop_filter = ""
    if seg.crop:
        source_crop_filter = source_crop(seg.crop.x, seg.crop.y, seg.crop.w, seg.crop.h)
    framed = (
        face_cover(fmt.width, fmt.height, seg.framing)
        if seg.framing is not None
        else cover(fmt.width, fmt.height, *seg.focus)
    )
    return _join_filters(
        source_crop_filter,
        framed,
        speed_filter(seg.speed),
        finalize(fmt.fps),
    )


def _clip_base_filter(seg: ClipSegment, fmt: Format, source_info: Any) -> str:
    return _join_filters(
        tonemap(source_info.color_transfer or "") if source_info.is_hdr else "",
        _clip_post_filter(seg, fmt),
    )


def _cached_mask(
    source: Path,
    seg: ClipSegment,
    cache: Cache,
) -> Path:
    if seg.remove_text is None:
        raise ValueError("mask requested without remove_text")
    key = {
        "source": Cache.file_key(source),
        "in": seg.in_,
        "out": seg.out,
        "remove_text": seg.remove_text.model_dump(mode="json"),
    }
    final = cache.path_for("mask", key, suffix=".png")
    with cache.building(final):
        if final.is_file():
            return final
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(f"{final}.tmp{final.suffix}")
        temporary.unlink(missing_ok=True)
        try:
            build_mask(source, seg.in_, seg.out - seg.in_, seg.remove_text, temporary)
            os.replace(temporary, final)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    return final


def _exact_frames(filters: str, frames: int) -> tuple[str, list[str]]:
    """Pad with clones of the last frame, then cut, so a stage outputs exactly ``frames``.

    Trimming, filter graphs and zoompan can each drop one frame; over a whole ad
    those losses add up and shift every overlay, so every encode is pinned.
    """
    padded = _join_filters(filters, "tpad=stop_mode=clone:stop_duration=1")
    return padded, ["-frames:v", str(frames)]


def _build_clip_base(
    source: Path,
    seg: ClipSegment,
    fmt: Format,
    output: Path,
    mask_path: Path | None = None,
) -> None:
    source_info = probe(source)
    duration = seg.out - seg.in_
    base_frames = round(duration / seg.speed * fmt.fps)
    if seg.remove_text is None:
        base_filter, frame_args = _exact_frames(
            _clip_base_filter(seg, fmt, source_info), base_frames
        )
        args = [
            *config.thread_args(),
            "-ss",
            str(seg.in_),
            "-t",
            str(duration),
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            base_filter,
            *frame_args,
            "-map_metadata",
            "-1",
            "-metadata:s:v:0",
            "rotate=0",
            *config.thread_args(),
            *encode_args(),
            *_bt709_encoder_args(),
            str(output),
        ]
        run(args)
        return

    if seg.remove_text.method == "mask" and mask_path is None:
        raise ValueError("mask removal requires a generated mask")
    tone = tonemap(source_info.color_transfer or "") if source_info.is_hdr else ""
    tone_input = f"[0:v]{tone or 'null'}[vin]"
    removal, needs_mask = removal_filter(
        seg.remove_text,
        "[1:v]" if mask_path is not None else None,
        source_info.width or 0,
        source_info.height or 0,
        mask_path=mask_path,
    )
    post, frame_args = _exact_frames(_clip_post_filter(seg, fmt), base_frames)
    graph = f"{tone_input};{removal};[vout]{post}[v]"
    args = [
        *config.thread_args(),
        "-ss",
        str(seg.in_),
        "-t",
        str(duration),
        "-i",
        str(source),
    ]
    if needs_mask:
        args.extend(["-loop", "1", "-i", str(mask_path)])
    args.extend(
        [
            "-filter_complex",
            graph,
            "-map",
            "[v]",
            *frame_args,
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
    run(args)


def _resolved_gamma(seg: ClipSegment, base: Path) -> float:
    if seg.gamma == "auto":
        return auto_gamma(measure_yavg(base))
    if seg.gamma is None:
        return 1.0
    return float(seg.gamma)


def _clip_final_key(
    base_key: dict[str, Any],
    resolved_gamma: float,
    seg: ClipSegment,
) -> dict[str, Any]:
    return {
        "base": base_key,
        "gamma": resolved_gamma,
        "zoom": seg.zoom.model_dump(mode="json", by_alias=True) if seg.zoom else None,
        "freeze": seg.freeze.model_dump(mode="json") if seg.freeze else None,
        **({"punch": [item.model_dump(mode="json") for item in seg.punch]} if seg.punch else {}),
        # A slow zoom is drawn at sub-pixel precision since ZOOM_ENGINE 2.
        **({"zoom_engine": ZOOM_ENGINE} if seg.zoom or seg.freeze else {}),
    }


def punch_centre(seg: ClipSegment, fmt: Format) -> tuple[float, float]:
    """The output pixel a punch without ``center`` zooms on: the segment's ``focus``."""

    if seg.face_focus:
        return 0.5 * fmt.width, seg.face_y * fmt.height
    return seg.focus[0] * fmt.width, seg.focus[1] * fmt.height


def _build_clip_final(
    base: Path,
    seg: ClipSegment,
    fmt: Format,
    resolved_gamma: float,
    output: Path,
) -> None:
    moving_duration = (seg.out - seg.in_) / seg.speed
    if seg.freeze:
        motion = freeze_zoom(
            moving_duration,
            seg.freeze.hold,
            seg.freeze.zoom,
            seg.freeze.center[0],
            seg.freeze.center[1],
            width=fmt.width,
            height=fmt.height,
            fps=fmt.fps,
        )
    elif seg.zoom:
        motion = zoom_filter(
            seg.zoom.from_,
            seg.zoom.to,
            moving_duration,
            *seg.zoom.focus,
            width=fmt.width,
            height=fmt.height,
            fps=fmt.fps,
        )
    else:
        motion = ""
    hold = seg.freeze.hold if seg.freeze else 0.0
    punch = punch_in(
        [
            (
                item.at,
                item.zoom,
                item.hold,
                *(item.center if item.center is not None else punch_centre(seg, fmt)),
            )
            for item in sorted(seg.punch, key=lambda item: item.at)
        ],
        fmt.width,
        fmt.height,
    )
    filters, frame_args = _exact_frames(
        _join_filters(gamma_filter(resolved_gamma), motion, punch, finalize(fmt.fps)),
        round((moving_duration + hold) * fmt.fps),
    )
    args = [
        "-i",
        str(base),
        "-map",
        "0:v:0",
        "-vf",
        filters,
        *frame_args,
        "-map_metadata",
        "-1",
        "-metadata:s:v:0",
        "rotate=0",
        *config.thread_args(),
        *encode_args(),
        *_bt709_encoder_args(),
        str(output),
    ]
    run(args)


def _colour_filter(background: str, width: int, height: int, fps: int) -> str:
    if background == "blur":
        return ""
    if len(background) != 7 or background[0] != "#":
        raise ValueError("background must be 'blur' or a #RRGGBB colour")
    try:
        int(background[1:], 16)
    except ValueError as exc:
        raise ValueError("background must be 'blur' or a #RRGGBB colour") from exc
    return f"color=c=0x{background[1:]}:s={width}x{height}:r={fps}"


def _bt709_encoder_args() -> list[str]:
    """Make libx264 write all three BT.709 colour fields for untagged stills."""
    return ["-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709"]


def _image_composition(
    seg: ImageSegment,
    width: int,
    height: int,
    fps: int,
) -> str:
    comp_width = width
    comp_height = height
    background = _colour_filter(seg.background, comp_width, comp_height, fps)
    foreground_fit = seg.fit
    first = "null"
    if seg.crop is not None:
        first = f"crop={seg.crop.w}:{seg.crop.h}:{seg.crop.x}:{seg.crop.y}"
    if background:
        background_graph = f"{background}[bg]"
        input_graph = f"[0:v]{first}[fgsrc];"
    else:
        background_graph = (
            f"[bgsrc]{cover(comp_width, comp_height)},gblur=sigma=40,"
            "eq=brightness=-0.06[bg]"
        )
        input_graph = f"[0:v]{first},split=2[bgsrc][fgsrc];"
    foreground_scale = (
        f"scale={comp_width}:{comp_height}:force_original_aspect_ratio="
        f"{'decrease' if foreground_fit == 'contain' else 'increase'}"
    )
    if foreground_fit == "contain":
        # No pad here: the overlay centres the smaller foreground on the background.
        foreground_graph = f"[fgsrc]{foreground_scale}[fg]"
    elif seg.framing is not None:
        foreground_graph = f"[fgsrc]{face_cover(comp_width, comp_height, seg.framing)}[fg]"
    else:
        foreground_graph = (
            f"[fgsrc]{foreground_scale},crop={comp_width}:{comp_height}:"
            f"(iw-{comp_width})*{_number(seg.focus[0])}:"
            f"(ih-{comp_height})*{_number(seg.focus[1])}[fg]"
        )
    return (
        f"{input_graph}{background_graph};{foreground_graph};"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2:format=auto"
    )


def _build_image(seg: ImageSegment, source: Path, fmt: Format, output: Path) -> None:
    frames = round(seg.duration * fmt.fps)
    if frames < 1:
        raise ValueError("image duration must produce at least one frame")
    # Both motions read the picture as a looped still at the frame rate: the zoom is driven
    # by the frame number and moves by fractions of a pixel (see ``filters.zoom``).
    motion = ""
    if seg.motion != "static":
        motion = (
            zoom_filter(
                1.0, seg.zoom_to, seg.duration, width=fmt.width, height=fmt.height, fps=fmt.fps
            )
            + ","
        )
    filters = (
        f"{_image_composition(seg, fmt.width, fmt.height, fmt.fps)},"
        f"{motion}{finalize(fmt.fps)},tpad=stop_mode=clone:stop_duration=1[v]"
    )
    args = [
        "-loop",
        "1",
        "-framerate",
        str(fmt.fps),
        "-i",
        str(source),
        "-filter_complex",
        filters,
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
    run(args)


def _image_key(source: Path, seg: ImageSegment, fmt: Format) -> dict[str, Any]:
    # Annotations do not change pixels: keep them out of the cache key. So does a centred
    # focus (the only crop there was before ``focus`` existed).
    exclude = {"label", "shot_type", "scale", "subject", "emotion", "role", "face_y", "face_focus"}
    if tuple(seg.focus) == (0.5, 0.5):
        exclude.add("focus")
    if seg.crop is None:
        exclude.add("crop")  # no crop: the key of an image rendered before crop existed
    return {
        "source": Cache.file_key(source),
        "segment": seg.model_dump(mode="json", exclude=exclude),
        **({"face": seg.framing.key()} if seg.framing is not None else {}),
        "format": _fmt_key(fmt),
        **({"zoom_engine": ZOOM_ENGINE} if seg.motion != "static" else {}),
    }


def render_segment(
    seg: Segment | SplitSegment | RevealSegment,
    *,
    root: Path,
    cache: Cache,
    fmt: Format = _DEFAULT_FORMAT,
) -> Path:
    """Render one clip, still image, split screen or reveal into the normalised mezzanine
    format."""
    if seg.type == "split":
        from .split import render_split

        return render_split(seg, root=root, cache=cache, fmt=fmt)
    if seg.type == "reveal":
        from .reveal import render_reveal

        return render_reveal(seg, root=root, cache=cache, fmt=fmt)
    source = _resolve_source(root, seg.source)
    seg = with_framing(seg, root, source)  # focus "face": find the face the crop follows
    if isinstance(seg, ClipSegment):
        mask_path = None
        if seg.remove_text is not None and seg.remove_text.method == "mask":
            mask_path = _cached_mask(source, seg, cache)
        base_key = _clip_base_key(source, seg, fmt)
        base = cache.get_or_build(
            "base",
            base_key,
            lambda output: _build_clip_base(source, seg, fmt, output, mask_path),
        )
        resolved_gamma = _resolved_gamma(seg, base)
        if (
            not gamma_filter(resolved_gamma)
            and seg.zoom is None
            and seg.freeze is None
            and not seg.punch
        ):
            return base
        final_key = _clip_final_key(base_key, resolved_gamma, seg)
        return cache.get_or_build(
            "clip",
            final_key,
            lambda output: _build_clip_final(base, seg, fmt, resolved_gamma, output),
        )

    key = _image_key(source, seg, fmt)
    return cache.get_or_build(
        "image",
        key,
        lambda output: _build_image(seg, source, fmt, output),
    )


def measure_motion(path: Path, n: int = 12) -> list[float]:
    """Measure mean absolute grayscale differences between sampled frames."""
    if n < 2:
        return []
    info = probe(path)
    duration = info.duration or 0.0
    sample_rate = n / max(duration, 1e-9) if duration else float(n)
    result = run(
        [
            "-i",
            str(path),
            "-vf",
            f"fps={sample_rate},scale=108:192,format=gray",
            "-an",
            "-frames:v",
            str(n),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        capture_stdout=True,
    )
    raw = result.stdout or b""
    frame_size = 108 * 192
    frames = [raw[offset : offset + frame_size] for offset in range(0, len(raw), frame_size)]
    frames = [frame for frame in frames if len(frame) == frame_size]
    differences: list[float] = []
    previous: array[int] | None = None
    for frame in frames:
        current = array("B", frame)
        if previous is not None:
            differences.append(
                sum(abs(a - b) for a, b in zip(current, previous, strict=True)) / frame_size
            )
        previous = current
    return differences
