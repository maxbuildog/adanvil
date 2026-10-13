"""Pure ffmpeg filter and encoder argument builders for media segments."""

from __future__ import annotations

import math


def _number(value: float | int) -> str:
    """Use a stable, compact representation in filter expressions."""
    return format(float(value), ".12g")


def tonemap(transfer: str) -> str:
    """Return the architecture-prescribed HDR-to-SDR filter, if applicable."""
    if transfer not in {"arib-std-b67", "smpte2084"}:
        return ""
    return (
        f"zscale=tin={transfer}:pin=bt2020:min=bt2020nc:t=linear:npl=100,"
        "format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,"
        "zscale=t=bt709:m=bt709:r=tv"
    )


def source_crop(x: int, y: int, w: int, h: int) -> str:
    return f"crop={w}:{h}:{x}:{y}"


def cover(width: int, height: int, fx: float = 0.5, fy: float = 0.5) -> str:
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height}:'(iw-{width})*{_number(fx)}':'(ih-{height})*{_number(fy)}'"
    )


def speed(factor: float) -> str:
    return "" if factor == 1 else f"setpts=PTS/{_number(factor)}"


def gamma(g: float) -> str:
    return "" if g <= 1.01 else f"eq=gamma={_number(g)}:saturation=1.06"


def auto_gamma(yavg: float) -> float:
    if yavg >= 135:
        return 1.0
    return min(1.6, max(1.0, math.log(max(yavg, 20) / 255) / math.log(0.55)))


def _zoom_expression(z0: float, z1: float, duration: float, fps: int = 30) -> str:
    """The zoom factor at the current frame (``in`` is the frame number of the input)."""
    frames = max(float(duration) * fps, 1e-9)
    return (
        f"({_number(z0)}+({_number(z1)}-{_number(z0)})*"
        f"min(in/{_number(frames)},1))"
    )


def _smooth_window(z: str, left: str, top: str, width: int, height: int) -> str:
    """Show the part of the picture that is ``width/z`` by ``height/z`` wide, from
    ``(left, top)``, enlarged to the whole frame.

    ``perspective`` maps the four corners of that window onto the frame with float
    coordinates and resamples with a cubic kernel at 1/256 of a pixel, so a slow zoom or move
    advances by fractions of a pixel instead of the whole-pixel steps of a scale with an even
    integer size followed by an integer crop (which makes a still tremble). It behaves the same
    on ffmpeg 6.1 and 9, and the picture keeps its size.
    """

    right = f"({left}+{width}/{z})"
    bottom = f"({top}+{height}/{z})"
    return (
        f"perspective=x0='{left}':y0='{top}':x1='{right}':y1='{top}':"
        f"x2='{left}':y2='{bottom}':x3='{right}':y3='{bottom}':"
        "interpolation=cubic:eval=frame"
    )


def zoom(
    z0: float,
    z1: float,
    duration: float,
    fx: float = 0.5,
    fy: float = 0.5,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
) -> str:
    """Build the frame-driven, sub-pixel video zoom (``fps`` is the rate of the input)."""
    z = _zoom_expression(z0, z1, duration, fps)
    left = f"({width}*(1-1/{z})*{_number(fx)})"
    top = f"({height}*(1-1/{z})*{_number(fy)})"
    return _smooth_window(z, left, top, width, height)


def freeze_zoom(
    moving_duration: float,
    hold: float,
    zoom: float,
    cx: int,
    cy: int,
    ramp: float = 0.45,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
) -> str:
    """Clone the last frame and zoom toward an output-pixel focus point."""
    moving_frames = round(moving_duration * fps)
    ramp_frames = max(float(ramp) * fps, 1e-9)
    z = (
        f"(1+({_number(zoom)}-1)*min(max((in-{moving_frames})/"
        f"{_number(ramp_frames)},0),1))"
    )
    # Keep the window inside the picture, a pixel from the far edge so the cubic kernel never
    # reads outside it.
    left = f"max(0,min({_number(cx)}-{width}/(2*{z}),{width}-{width}/{z}-1))"
    top = f"max(0,min({_number(cy)}-{height}/(2*{z}),{height}-{height}/{z}-1))"
    return (
        f"tpad=stop_mode=clone:stop_duration={_number(hold)},"
        f"{_smooth_window(z, left, top, width, height)}"
    )


def punch_in(
    punches: list[tuple[float, float, float, float, float]],
    width: int = 1080,
    height: int = 1920,
) -> str:
    """Instant zooms: each ``(at, zoom, hold, cx, cy)`` enlarges the picture by ``zoom`` from
    ``at`` for ``hold`` seconds (segment time), centred on the output pixel ``(cx, cy)``.

    There is no ramp: the first frame at or after ``at`` is already at full zoom and the first
    frame at or after ``at + hold`` is back to 1.0. Punches never overlap (the storyboard model
    refuses that), so at most one gate is on at a time.
    """

    if not punches:
        return ""
    gates = [
        f"gte(t,{_number(at)})*lt(t,{_number(at + hold)})" for at, _z, hold, _x, _y in punches
    ]
    boosts = "".join(
        f"+({_number(p[1])}-1)*{g}" for p, g in zip(punches, gates, strict=True)
    )
    z = f"(1{boosts})"

    def offset(centre: float, size: int) -> str:
        return f"max(0,min({_number(centre)}*{z}-{size}/2,{size}*{z}-{size}-2))"

    x = "0"
    y = "0"
    for (_at, _z, _hold, cx, cy), gate in reversed(list(zip(punches, gates, strict=True))):
        x = f"if({gate},{offset(cx, width)},{x})"
        y = f"if({gate},{offset(cy, height)},{y})"
    return (
        f"scale=w='trunc({width}*{z}/2)*2':h='trunc({height}*{z}/2)*2':eval=frame,"
        f"crop={width}:{height}:'{x}':'{y}'"
    )


def finalize(fps: int = 30) -> str:
    """Constant frame rate, square pixels, and 8-bit 4:2:0 converted *to* BT.709 limited range.

    The conversion is explicit because the encoder only writes the BT.709 tags: ffmpeg 7.1 and
    later also convert the pixels to match them, ffmpeg 6.1 does not, and an untagged source
    (or an RGB picture) would come out with the right tags on the wrong colours there.
    """
    return f"fps={fps},setsar=1,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"


def encode_args(crf: int = 16, preset: str = "veryfast") -> list[str]:
    return [
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-an",
        "-movflags",
        "+faststart",
    ]
