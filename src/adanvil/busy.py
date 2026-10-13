"""Where a still picture is busy, and where it is empty.

A picture shown ``contain`` (an end card, any still) keeps its own lettering, buttons and
product shots. A caption laid over it must not hide them. ``picture_busy`` finds, on the output
frame, the rectangles of the picture that carry something, and ``free_bands`` turns them into
the horizontal bands of the frame where an overlay can go.

The busy region is the union of:

* the pixels that stand out from the picture's own blurred copy (its local surround): a small
  blur finds edges and thin lettering, a large blur finds flat fills such as a button, whose
  inside has no edge; both are compared on the three colour channels;
* the text boxes of ``qc.score.detect_text`` (white against black lettering).

The mask is cut into runs of rows; each run becomes one rectangle (its columns are the span of
the mask in the run) and is padded by a small margin. Smooth gradients are not busy.
No dependency beyond Pillow and numpy.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

Box = tuple[float, float, float, float]

_WORK = 384  # long side of the picture the mask is computed on
_SMALL_BLUR = 4.0  # at the working size: edges and thin lettering
_SMALL_THRESHOLD = 22.0
_BIG_BLUR_SHARE = 0.12  # of the long side: flat fills
_BIG_THRESHOLD = 45.0
_MARGIN = 10.0  # frame pixels added around each busy rectangle
_MERGE_GAP = 3  # rows (working size) that still join two runs


def picture_rect(
    picture_size: tuple[int, int], frame: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Where a picture shown ``contain`` lands on ``frame``: (left, top, right, bottom)."""

    pw, ph = picture_size
    fw, fh = frame
    scale = min(fw / max(pw, 1), fh / max(ph, 1))
    width, height = pw * scale, ph * scale
    left, top = (fw - width) / 2.0, (fh - height) / 2.0
    return left, top, left + width, top + height


def _mask(rgb: np.ndarray) -> np.ndarray:
    image = Image.fromarray(rgb)
    long_side = max(rgb.shape[:2])
    data = rgb.astype(np.float32)
    flagged = np.zeros(rgb.shape[:2], dtype=bool)
    for radius, threshold in (
        (_SMALL_BLUR, _SMALL_THRESHOLD),
        (long_side * _BIG_BLUR_SHARE, _BIG_THRESHOLD),
    ):
        blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
        flagged |= np.abs(data - blurred).max(axis=-1) > threshold
    grown = Image.fromarray((flagged * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(7))
    return np.asarray(grown) > 0


def _runs(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """(left, top, right, bottom) in mask pixels of each run of busy rows."""

    rows = np.nonzero(mask.any(axis=1))[0]
    if rows.size == 0:
        return []
    groups: list[list[int]] = [[int(rows[0]), int(rows[0])]]
    for row in rows[1:]:
        if row - groups[-1][1] <= _MERGE_GAP + 1:
            groups[-1][1] = int(row)
        else:
            groups.append([int(row), int(row)])
    found = []
    for top, bottom in groups:
        columns = np.nonzero(mask[top : bottom + 1].any(axis=0))[0]
        found.append((int(columns[0]), top, int(columns[-1]) + 1, bottom + 1))
    return found


@lru_cache(maxsize=64)
def _busy_cached(
    path: str,
    stamp: int,
    frame: tuple[int, int],
    crop: tuple[int, int, int, int] | None,
) -> tuple[Box, tuple[Box, ...]]:
    del stamp  # only part of the cache key: a changed file is read again
    from adanvil.qc.score import detect_text

    with Image.open(path) as opened:
        picture = opened.convert("RGB")
    if crop is not None:
        x, y, w, h = crop
        picture = picture.crop((x, y, x + w, y + h))
    rect = picture_rect(picture.size, frame)
    scale = min(1.0, _WORK / max(picture.size))
    work = picture.resize(
        (max(1, round(picture.width * scale)), max(1, round(picture.height * scale))),
        Image.LANCZOS,
    )
    rgb = np.asarray(work)
    mask = _mask(rgb)
    to_frame_x = (rect[2] - rect[0]) / work.width
    to_frame_y = (rect[3] - rect[1]) / work.height
    boxes: list[Box] = []
    for left, top, right, bottom in _runs(mask):
        boxes.append(
            (
                rect[0] + left * to_frame_x - _MARGIN,
                rect[1] + top * to_frame_y - _MARGIN,
                rect[0] + right * to_frame_x + _MARGIN,
                rect[1] + bottom * to_frame_y + _MARGIN,
            )
        )
    for left, top, right, bottom in detect_text(rgb):
        boxes.append(
            (
                rect[0] + left * (rect[2] - rect[0]) - _MARGIN,
                rect[1] + top * (rect[3] - rect[1]) - _MARGIN,
                rect[0] + right * (rect[2] - rect[0]) + _MARGIN,
                rect[1] + bottom * (rect[3] - rect[1]) + _MARGIN,
            )
        )
    return rect, tuple(boxes)


def picture_busy(
    path: str | Path,
    frame: tuple[int, int],
    *,
    crop: tuple[int, int, int, int] | None = None,
) -> tuple[Box, list[Box]]:
    """The rectangle of ``path`` shown ``contain`` on ``frame`` and its busy rectangles, both in
    frame pixels. ``crop`` is the (x, y, w, h) the segment takes of the picture, if any."""

    file = Path(path)
    rect, boxes = _busy_cached(str(file), file.stat().st_mtime_ns, frame, crop)
    return rect, list(boxes)


def overlaps(box: Box, busy: list[Box]) -> bool:
    """True when ``box`` covers any of the ``busy`` rectangles."""

    left, top, right, bottom = box
    return any(
        left < b_right and right > b_left and top < b_bottom and bottom > b_top
        for b_left, b_top, b_right, b_bottom in busy
    )


def free_bands(busy: list[Box], top: float, bottom: float) -> list[tuple[float, float]]:
    """The horizontal bands between ``top`` and ``bottom`` that no busy rectangle crosses,
    largest first (full width: a band is free only when nothing busy lies at its height)."""

    edges = sorted(
        (max(top, b[1]), min(bottom, b[3])) for b in busy if b[3] > top and b[1] < bottom
    )
    bands: list[tuple[float, float]] = []
    cursor = top
    for start, end in edges:
        if start > cursor:
            bands.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < bottom:
        bands.append((cursor, bottom))
    return sorted(bands, key=lambda band: band[1] - band[0], reverse=True)


__all__ = ["free_bands", "overlaps", "picture_busy", "picture_rect"]
