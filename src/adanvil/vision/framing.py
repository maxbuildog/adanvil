"""Face-aware framing: keep the main face centred at a chosen height.

``focus: "face"`` on a clip or image segment (or a side of a split or a reveal) crops the
cover picture so that the main face sits at the middle of the frame, ``face_y`` of the way down
(default 0.38, a little above the middle, as a creator is usually framed). The face comes from
the boxes stored by the vision pass (or from a detection made on the spot). For a clip the face
position is smoothed over time (median, then moving average; a face that hardly moves gives a
still frame) and the crop follows it with a piecewise linear expression of the frame time.

The maths is kept free of ffmpeg: ``face_track`` turns stored boxes into a ``FaceTrack``,
``crop_offset`` and ``track_expression`` say where the crop goes, ``align_zooms`` gives the
zoom that makes two faces the same size, and ``face_cover`` writes the filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .engine import Face
from .store import VisionData

DEFAULT_FACE_Y = 0.38
STILL_RANGE = 0.03  # a face that moves less than this share of the frame stays still
MAX_KNOTS = 10
MAX_ALIGN_ZOOM = 1.6  # the most a side is enlarged to match the face size of the other


@dataclass(frozen=True)
class FaceTrack:
    """The main face over time, in 0..1 of the picture that is being cropped.

    ``knots`` are ``(seconds from the start of the window, centre x, centre y)``; ``height``
    is the median face height as a share of the picture height, ``aspect`` the picture's
    width over height, and ``single`` whether no analysed frame of the window had two faces.
    """

    knots: tuple[tuple[float, float, float], ...]
    height: float
    aspect: float
    single: bool = True

    @property
    def still(self) -> bool:
        return len(self.knots) == 1


@dataclass(frozen=True)
class Framing:
    """How a segment is framed on a face: the track, the target height and the extra zoom."""

    track: FaceTrack
    y: float = DEFAULT_FACE_Y
    zoom: float = 1.0

    def key(self) -> dict[str, Any]:
        """What changes the picture, for cache keys."""

        return {
            "knots": [[round(value, 4) for value in knot] for knot in self.track.knots],
            "y": round(self.y, 4),
            "zoom": round(self.zoom, 4),
        }


def main_face(faces: list[Face]) -> Face | None:
    """The biggest face (the one the viewer is looking at)."""

    return max(faces, key=lambda face: face.area * (0.5 + 0.5 * face.score), default=None)


def _median3(values: list[float]) -> list[float]:
    if len(values) < 3:
        return list(values)
    padded = [values[0], *values, values[-1]]
    return [float(np.median(padded[index : index + 3])) for index in range(len(values))]


def _mean3(values: list[float]) -> list[float]:
    if len(values) < 3:
        return list(values)
    padded = [values[0], *values, values[-1]]
    return [sum(padded[index : index + 3]) / 3 for index in range(len(values))]


def face_track(
    data: VisionData,
    *,
    start: float = 0.0,
    end: float | None = None,
    crop: tuple[float, float, float, float] | None = None,
    source_size: tuple[float, float] | None = None,
) -> FaceTrack | None:
    """The smoothed main-face track of the window ``start``..``end`` (seconds of the file).

    ``crop`` is ``(x, y, w, h)`` in source pixels of the part of the picture that is kept (the
    face centre is expressed inside it) and ``source_size`` the picture size in pixels; both
    are needed to follow a crop. Returns ``None`` when no frame near the window has a face.
    """

    if data.frames == 0:
        return None
    last = float(data.times.max()) if end is None else float(end)
    first = float(start)
    moments = [float(value) for value in data.times]
    inside = [index for index, moment in enumerate(moments) if first <= moment <= last]
    with_face = [index for index in range(data.frames) if data.faces(index)]
    chosen = [index for index in inside if index in with_face]
    if not chosen:  # no sampled frame of the window has a face: take the nearest ones that do
        middle = (first + last) / 2
        chosen = sorted(with_face, key=lambda index: abs(moments[index] - middle))[:2]
        chosen.sort()
    if not chosen:
        return None
    width_px, height_px = (
        (float(source_size[0]), float(source_size[1]))
        if source_size
        else (
            float(data.sizes[0][0]),
            float(data.sizes[0][1]),
        )
    )
    left, top, crop_w, crop_h = crop if crop is not None else (0.0, 0.0, width_px, height_px)
    times: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    heights: list[float] = []
    multiple = False
    for index in chosen:
        faces = data.faces(index)
        multiple = multiple or len(faces) > 1
        face = main_face(faces)
        if face is None:
            continue
        centre_x, centre_y = face.centre
        relative_x = (centre_x * width_px - left) / crop_w
        relative_y = (centre_y * height_px - top) / crop_h
        times.append(min(max(moments[index], first), last) - first)
        xs.append(min(max(relative_x, 0.0), 1.0))
        ys.append(min(max(relative_y, 0.0), 1.0))
        heights.append(face.height * height_px / crop_h)
    if not times:
        return None
    xs, ys = _mean3(_median3(xs)), _mean3(_median3(ys))
    if max(xs) - min(xs) < STILL_RANGE and max(ys) - min(ys) < STILL_RANGE:
        knots = ((0.0, float(np.mean(xs)), float(np.mean(ys))),)
    else:
        order = list(range(len(times)))
        if len(order) > MAX_KNOTS:
            spread = np.linspace(0, len(order) - 1, MAX_KNOTS)
            order = sorted({round(float(position)) for position in spread})
        knots = tuple((times[index], xs[index], ys[index]) for index in order)
    return FaceTrack(
        knots=knots,
        height=float(np.median(heights)),
        aspect=crop_w / crop_h,
        single=not multiple,
    )


# --------------------------------------------------------------------------- the maths


def _number(value: float) -> str:
    return format(float(value), ".6g")


def track_expression(track: FaceTrack, axis: int) -> str:
    """The face centre along ``axis`` (1 is x, 2 is y) as an ffmpeg expression of ``t``."""

    knots = track.knots
    if len(knots) == 1:
        return _number(knots[0][axis])
    expression = _number(knots[-1][axis])
    for before, after in zip(reversed(knots[:-1]), reversed(knots[1:]), strict=True):
        span = max(after[0] - before[0], 1e-3)
        segment = (
            f"{_number(before[axis])}+({_number(after[axis] - before[axis])})"
            f"*(t-{_number(before[0])})/{_number(span)}"
        )
        expression = f"if(lt(t,{_number(after[0])}),{segment},{expression})"
    return f"if(lt(t,{_number(knots[0][0])}),{_number(knots[0][axis])},{expression})"


def crop_offset(centre: float, anchor: float, scaled: float, frame: float) -> float:
    """Where a crop of ``frame`` pixels starts in a picture ``scaled`` pixels wide so that the
    point at ``centre`` (0..1 of the picture) lands at ``anchor`` (0..1 of the frame); kept
    inside the picture."""

    return min(max(centre * scaled - anchor * frame, 0.0), max(scaled - frame, 0.0))


def face_cover(width: int, height: int, framing: Framing) -> str:
    """The filter that scales to cover ``width`` x ``height`` (enlarged by ``framing.zoom``) and
    crops so the face is centred across and ``framing.y`` of the way down."""

    zoom = max(1.0, framing.zoom)
    scaled_w = 2 * round(width * zoom / 2)
    scaled_h = 2 * round(height * zoom / 2)
    x = track_expression(framing.track, 1)
    y = track_expression(framing.track, 2)
    return (
        f"scale={scaled_w}:{scaled_h}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height}:"
        f"'min(max(({x})*iw-{width}/2,0),max(iw-{width},0))':"
        f"'min(max(({y})*ih-{_number(framing.y)}*{height},0),max(ih-{height},0))'"
    )


def face_share(track: FaceTrack, width: int, height: int) -> float:
    """The height of the face as a share of the output frame once the picture covers it."""

    scale_to_height = max(width / track.aspect, height) / height
    return track.height * scale_to_height


def align_zooms(
    first: FaceTrack,
    second: FaceTrack,
    width: int,
    height: int,
    *,
    limit: float = MAX_ALIGN_ZOOM,
) -> tuple[float, float]:
    """The zoom of each side that gives both faces the same size in the frame: the smaller
    face is enlarged to the bigger one, never beyond ``limit``."""

    a, b = face_share(first, width, height), face_share(second, width, height)
    target = max(a, b)
    return (
        min(max(target / a, 1.0), limit) if a > 0 else 1.0,
        min(max(target / b, 1.0), limit) if b > 0 else 1.0,
    )


__all__ = [
    "DEFAULT_FACE_Y",
    "MAX_ALIGN_ZOOM",
    "FaceTrack",
    "Framing",
    "align_zooms",
    "crop_offset",
    "face_cover",
    "face_share",
    "face_track",
    "main_face",
    "track_expression",
]
