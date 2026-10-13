"""Captions and faces: where a face is on the output frame, so that no caption sits on it.

Two ways to know where the faces are:

* ``frame_faces`` works from the SOURCE of the segment under a given moment of a storyboard:
  the face boxes stored by the vision pass (``cache/vision``, V2.7) or, when there are none, a
  detection made now by the vision environment's worker; the boxes are then carried to the
  output frame through the same crop the renderer applies (the segment's ``crop``, the cover
  scale and its ``focus`` or ``focus: "face"``). ``render_formats`` uses it to place captions
  before anything is rendered. A segment that is not a clip or a still (a split, a reveal) is
  not looked at; a slow zoom or a punch is not followed (the boxes are padded for it).
* ``rendered_faces`` detects the faces of a frame of a RENDERED video (``qc_run`` and
  ``ad_score``, every format).

ONE test answers "is a caption on a face?" everywhere (``render_formats`` notes, ``qc_run``,
``ad_score``): ``caption_on_face(box, faces)`` measures the caption against the FACE CORE, the
detector's box from the eye line to the chin with a small margin (``face_core``): a caption
above the hair or on the forehead above the eyes does not count, one on the eyes, nose, mouth
or chin does. ``frame_faces`` and ``rendered_faces`` both return cores, already filtered by
``is_real_face``.

Faces that are not people in the footage are dropped (``is_real_face``): a drawing or diagram
(an anatomy chart on a tablet, a poster) is found by the detector with a lower confidence than
a person, and a tiny face is a background figure. Thresholds, tuned on a real test ad
(the anatomy diagram scores 0.76 to 0.85 in the footage; every real face of the ad scores 0.90
to 0.95): ``MIN_SCORE`` 0.88 and ``MIN_WIDTH`` 0.06 (the face width as a share of the picture
width).

All boxes are ``(left, top, right, bottom)`` in pixels of the output frame.
"""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

Box = tuple[float, float, float, float]

CORE_TOP = 0.25  # of the face height: the detector box starts at the brow, the eyes are here
CORE_BOTTOM = 0.04  # of the face height, below the chin
CORE_SIDE = 0.05  # of the face width
MARGIN = 4.0  # pixels around the core
MIN_SCORE = 0.88  # a lower detector confidence is a drawing, not a person
MIN_WIDTH = 0.06  # a narrower face (share of the picture width) is a background figure


def face_core(box: Box) -> Box:
    """A face box (pixels) reduced to its core: from the eye line to the chin, with a margin."""

    left, top, right, bottom = box
    width, height = right - left, bottom - top
    return (
        left - CORE_SIDE * width - MARGIN,
        top + CORE_TOP * height - MARGIN,
        right + CORE_SIDE * width + MARGIN,
        bottom + CORE_BOTTOM * height + MARGIN,
    )


def is_real_face(score: float, width_share: float) -> bool:
    """False for a detection that is not a person in the footage (see the module docstring)."""

    return score >= MIN_SCORE and width_share >= MIN_WIDTH


def caption_on_face(box: Box, faces: list[Box]) -> bool:
    """True when the caption ``box`` covers any of the face ``faces`` (cores): the one test used
    by the format layout, ``qc_run`` and ``ad_score``."""

    left, top, right, bottom = box
    return any(
        left < f_right and right > f_left and top < f_bottom and bottom > f_top
        for f_left, f_top, f_right, f_bottom in faces
    )


FACE_STEP = 0.2  # seconds between two looks at the faces under a caption
FACE_LIMIT = 40  # at most this many looks per caption (a long caption is looked at more widely)

Sample = tuple[float, list[Box]]  # a moment of the storyboard and the face cores found then


def sample_moments(
    start: float, end: float, total: float, *, count: int | None = None
) -> list[float]:
    """The moments at which a caption shown on ``start``..``end`` is checked against the faces:
    every ``FACE_STEP`` seconds (at most ``FACE_LIMIT``), the first and the last included; with
    ``count`` that many evenly spread (``qc_run``, where each look is a detection)."""

    last = max(total - 1e-3, 0.0)
    # a little inside the range: a caption that starts on a cut must not be looked at on the
    # picture before the cut (float sums of the durations differ by a hair)
    inside = min(0.02, max(end - start, 0.0) / 4)
    start, end = start + inside, max(start + inside, end - inside)
    if count is None:
        count = min(FACE_LIMIT, max(2, int((end - start) / FACE_STEP) + 1))
    if count <= 1 or end - start < 1e-6:
        return [min(max(start, 0.0), last)]
    return [
        min(max(start + (end - start) * number / (count - 1), 0.0), last)
        for number in range(count)
    ]


def first_covered(box: Box, samples: list[Sample]) -> float | None:
    """The first sampled moment at which the caption ``box`` covers a face (``caption_on_face``
    at each one), ``None`` when it covers none: the check of a FINAL position, the same for the
    layout notes and for ``qc_run``."""

    for moment, faces in samples:
        if caption_on_face(box, faces):
            return moment
    return None


def _interpolate(track: Any, moment: float) -> tuple[float, float]:
    knots = track.knots
    if len(knots) == 1 or moment <= knots[0][0]:
        return knots[0][1], knots[0][2]
    for before, after in zip(knots[:-1], knots[1:], strict=True):
        if moment <= after[0]:
            span = max(after[0] - before[0], 1e-3)
            mix = (moment - before[0]) / span
            return (
                before[1] + (after[1] - before[1]) * mix,
                before[2] + (after[2] - before[2]) * mix,
            )
    return knots[-1][1], knots[-1][2]


_FRESH: dict[tuple[str, float], list[Any] | None] = {}


def _fresh_faces(source: Path, seen: float) -> list[Any] | None:
    """The faces (detector boxes in 0..1) of ``source`` at ``seen`` seconds, detected now by the
    vision engine; ``None`` without an engine or when the frame cannot be read. Kept in memory
    per (file, moment)."""

    from adanvil.ffmpeg import extract_frame
    from adanvil.vision import client as vision_client

    try:
        stamp = Path(source).stat().st_mtime
    except OSError:
        stamp = 0.0
    key = (f"{source}|{stamp}", round(seen, 2))
    if key in _FRESH:
        return _FRESH[key]
    found: list[Any] | None = None
    try:
        engine = vision_client.get_engine()
        if engine is not None:
            with tempfile.TemporaryDirectory(prefix="adanvil-faces-") as folder:
                path = Path(folder) / "frame.png"
                extract_frame(Path(source), max(0.0, seen), path, width=540)
                pictures = engine.detect_faces([path])
            found = list(pictures[0].faces) if pictures else []
    except Exception:
        found = None
    if len(_FRESH) > 4096:
        _FRESH.clear()
    _FRESH[key] = found
    return found


@lru_cache(maxsize=128)
def _picture_size_cached(path: str, stamp: float) -> tuple[float, float]:
    from adanvil.ffmpeg import probe

    info = probe(Path(path))
    return float(info.width or 0), float(info.height or 0)


def _picture_size(source: Path) -> tuple[float, float]:
    try:
        stamp = Path(source).stat().st_mtime
    except OSError:
        stamp = 0.0
    return _picture_size_cached(str(source), stamp)


def frame_faces(
    document: dict[str, Any], root: Path, moment: float, *, engine: Any | None = None
) -> list[Box] | None:
    """The face cores on the output frame at ``moment`` (seconds of the storyboard).

    ``None`` when there is no face data for the picture under that moment and none can be made
    (no vision environment); an empty list when there is no face, or the segment is not a clip
    or a still."""

    from adanvil.media.facefocus import face_data
    from adanvil.project import resolve_in_project
    from adanvil.variants import VariantError, durations
    from adanvil.vision.framing import DEFAULT_FACE_Y, main_face

    segments = list(document.get("segments") or [])
    try:
        spans = durations(segments)
    except VariantError:
        return []
    start = 0.0
    chosen: tuple[dict[str, Any], float] | None = None
    for segment, span in zip(segments, spans, strict=True):
        if start - 1e-6 <= moment < start + span + 1e-6 and isinstance(segment, dict):
            chosen = (segment, start)
            break
        start += span
    if chosen is None:
        return []
    segment, begin = chosen
    kind = segment.get("type")
    if kind not in {"clip", "image"}:
        return []
    fmt = document.get("format") or {}
    frame_w, frame_h = int(fmt.get("width", 1080)), int(fmt.get("height", 1920))
    try:
        source = resolve_in_project(Path(root), str(segment.get("source")))
        data = face_data(Path(root), source, kind, engine=engine)
    except Exception:
        return None
    if data is None or data.frames == 0:
        return None
    if kind == "clip":
        speed = float(segment.get("speed", 1.0) or 1.0)
        first, last = float(segment.get("in", 0.0)), float(segment.get("out", 0.0))
        seen = min(max(first + (moment - begin) * speed, first), last)
    else:
        seen = 0.0
    index = min(range(data.frames), key=lambda i: abs(float(data.times[i]) - seen))
    faces = data.faces(index)
    if kind == "clip":
        # the stored frames are about a second apart and a head moves faster than that, and their
        # boxes come from another pass than the one ``qc_run`` uses on the rendered frames: look at
        # the picture of this very moment, with the same detector and frame size as QC
        fresh = _fresh_faces(source, seen)
        if fresh is not None:
            faces = fresh
    if not faces:
        return []
    try:
        pic_w, pic_h = _picture_size(source)
    except Exception:
        pic_w = pic_h = 0.0
    if pic_w <= 0 or pic_h <= 0:
        pic_w, pic_h = float(data.sizes[index][0]), float(data.sizes[index][1])
    crop = segment.get("crop")
    if isinstance(crop, dict) and all(key in crop for key in ("x", "y", "w", "h")):
        cx, cy, cw, ch = (float(crop[key]) for key in ("x", "y", "w", "h"))
    else:
        cx, cy, cw, ch = 0.0, 0.0, pic_w, pic_h
    if cw <= 0 or ch <= 0:
        return []
    contain = kind == "image" and segment.get("fit", "contain") == "contain"
    scale = min(frame_w / cw, frame_h / ch) if contain else max(frame_w / cw, frame_h / ch)
    scaled_w, scaled_h = cw * scale, ch * scale
    if contain:
        off_x, off_y = (scaled_w - frame_w) / 2, (scaled_h - frame_h) / 2
    else:
        focus = segment.get("focus", (0.5, 0.5))
        if focus == "face":
            main = main_face(faces)
            assert main is not None
            fx = (main.centre[0] * pic_w - cx) / cw
            fy = (main.centre[1] * pic_h - cy) / ch
            face_y = float(segment.get("face_y", DEFAULT_FACE_Y))
            off_x = min(max(fx * scaled_w - frame_w / 2, 0.0), max(scaled_w - frame_w, 0.0))
            off_y = min(max(fy * scaled_h - face_y * frame_h, 0.0), max(scaled_h - frame_h, 0.0))
        else:
            try:
                ax, ay = float(focus[0]), float(focus[1])
            except (TypeError, ValueError, IndexError):
                ax = ay = 0.5
            off_x, off_y = (scaled_w - frame_w) * ax, (scaled_h - frame_h) * ay
    found: list[Box] = []
    for face in faces:
        if not is_real_face(face.score, face.width):
            continue
        left, top, right, bottom = face.box
        box = (
            (left * pic_w - cx) * scale - off_x,
            (top * pic_h - cy) * scale - off_y,
            (right * pic_w - cx) * scale - off_x,
            (bottom * pic_h - cy) * scale - off_y,
        )
        if box[2] <= 0 or box[3] <= 0 or box[0] >= frame_w or box[1] >= frame_h:
            continue
        found.append(face_core(box))
    return found


def rendered_faces(
    video: Path, moment: float, engine: Any, *, width: int = 540
) -> tuple[list[Box], tuple[int, int]]:
    """The face cores found by ``engine`` on the frame of ``video`` at ``moment``, and the size
    of the frame they are measured on (the video scaled to ``width`` wide)."""

    from PIL import Image

    from adanvil.ffmpeg import extract_frame

    with tempfile.TemporaryDirectory(prefix="adanvil-faces-") as folder:
        path = Path(folder) / "frame.png"
        extract_frame(Path(video), max(0.0, moment), path, width=width, on_screen=True)
        with Image.open(path) as opened:
            size = opened.size
        pictures = engine.detect_faces([path])
    found: list[Box] = []
    for face in pictures[0].faces if pictures else []:
        if not is_real_face(face.score, face.width):
            continue
        left, top, right, bottom = face.box
        found.append(face_core((left * size[0], top * size[1], right * size[0], bottom * size[1])))
    return found, size


def scale_boxes(boxes: list[Box], factor_x: float, factor_y: float) -> list[Box]:
    return [(a * factor_x, b * factor_y, c * factor_x, d * factor_y) for a, b, c, d in boxes]


__all__ = [
    "Box",
    "FACE_STEP",
    "Sample",
    "caption_on_face",
    "face_core",
    "first_covered",
    "frame_faces",
    "is_real_face",
    "rendered_faces",
    "sample_moments",
    "scale_boxes",
]
