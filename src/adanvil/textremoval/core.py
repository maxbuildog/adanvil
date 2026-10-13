"""Pure helpers for finding and removing static, burnt-in text."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageFilter
from pydantic import BaseModel, Field, model_validator

from adanvil.ffmpeg import probe, run

Box = tuple[int, int, int, int]


class RemoveText(BaseModel):
    """Description of one static text-removal operation."""

    method: Literal["delogo", "mask", "crop_out"]
    box: Box | None = None
    extra_boxes: list[Box] = Field(default_factory=list)
    threshold: int = 200
    dilate: int = 7
    blur: float = 7.0

    @model_validator(mode="after")
    def require_box(self) -> RemoveText:
        if self.box is None:
            raise ValueError("remove_text.box is required for every removal method")
        return self


def _box_in_frame(box: Box, width: int, height: int) -> Box | None:
    x0, y0, x1, y1 = (int(value) for value in box)
    x0 = max(0, min(width, x0))
    y0 = max(0, min(height, y0))
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _require_box(box: Box | None, width: int, height: int) -> Box:
    if box is None:
        raise ValueError("remove_text.box is required for every removal method")
    clipped = _box_in_frame(box, width, height)
    if clipped is None:
        raise ValueError("remove_text.box must overlap the source frame")
    return clipped


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a boolean mask with a square Pillow max-filter."""
    if radius <= 0 or not mask.any():
        return mask.copy()
    size = 2 * int(radius) + 1
    image = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")
    filtered = image.filter(ImageFilter.MaxFilter(size=size))
    return np.asarray(filtered, dtype=np.uint8) > 0


def scaled_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    """Even-sized dimensions that bring the longest side down to ``long_side``."""

    ratio = long_side / max(width, height)
    return max(2, round(width * ratio / 2) * 2), max(2, round(height * ratio / 2) * 2)


def sample_frames(
    path: str | Path,
    start: float,
    duration: float | None,
    fps: int = 10,
    box: Box | None = None,
    *,
    long_side: int | None = None,
    keyframes_only: bool = False,
) -> np.ndarray:
    """Decode sampled RGB frames, optionally cropping in ffmpeg first.

    ``long_side`` scales the frames down in ffmpeg (before the RGB conversion) so their
    longest side is at most that many pixels; ``keyframes_only`` decodes only keyframes
    (``-skip_frame nokey``, one output frame per keyframe). Both are for the cheap
    analysis of large sources and cannot be combined with ``box``.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if box is not None and (long_side is not None or keyframes_only):
        raise ValueError("box cannot be combined with long_side or keyframes_only")
    info = probe(path)
    if info.width is None or info.height is None:
        raise ValueError(f"source has no display video dimensions: {path}")
    width, height = info.width, info.height
    filters: list[str] = []
    if long_side is not None and max(width, height) > long_side:
        width, height = scaled_size(width, height, long_side)
        filters.append(f"scale={width}:{height}")
    # Convert before cropping so odd source-pixel box widths/heights are not
    # rounded down by a subsampled input pixel format.
    filters.append("format=rgb24")
    output_width, output_height = width, height
    if box is not None:
        x0, y0, x1, y1 = _require_box(box, width, height)
        output_width, output_height = x1 - x0, y1 - y0
        filters.append(f"crop={output_width}:{output_height}:{x0}:{y0}")
    if not keyframes_only:
        filters.append(f"fps={fps}")
    args = ["-ss", str(start)]
    if keyframes_only:
        args.extend(["-skip_frame", "nokey"])
    args.extend(["-i", str(path)])
    if duration is not None:
        args.extend(["-t", str(duration)])
    if keyframes_only:
        args.extend(["-fps_mode", "passthrough"])
    args.extend(
        [
            "-vf",
            ",".join(filters),
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
    )
    result = run(args, capture_stdout=True)
    raw = result.stdout or b""
    frame_size = output_width * output_height * 3
    if frame_size <= 0:
        raise ValueError("sampled frame has no pixels")
    if len(raw) % frame_size:
        raise ValueError(
            f"ffmpeg returned a partial RGB frame ({len(raw)} bytes, expected a multiple of "
            f"{frame_size})"
        )
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, output_height, output_width, 3).copy()


def _connected_components(mask: np.ndarray, min_area: int) -> list[Box]:
    """Return bounding boxes for 4-connected true components."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes: list[Box] = []
    for y, x in np.argwhere(mask):
        y = int(y)
        x = int(x)
        if visited[y, x]:
            continue
        visited[y, x] = True
        stack = [(y, x)]
        area = 0
        min_x = max_x = x
        min_y = max_y = y
        while stack:
            current_y, current_x = stack.pop()
            area += 1
            min_x = min(min_x, current_x)
            max_x = max(max_x, current_x)
            min_y = min(min_y, current_y)
            max_y = max(max_y, current_y)
            for next_y, next_x in (
                (current_y - 1, current_x),
                (current_y + 1, current_x),
                (current_y, current_x - 1),
                (current_y, current_x + 1),
            ):
                if (
                    0 <= next_y < height
                    and 0 <= next_x < width
                    and mask[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    stack.append((next_y, next_x))
        if area >= min_area:
            boxes.append((min_x, min_y, max_x + 1, max_y + 1))
    return boxes


def _component_fill_ratio(mask: np.ndarray, box: Box) -> float:
    x0, y0, x1, y1 = box
    area = max(1, (x1 - x0) * (y1 - y0))
    return float(mask[y0:y1, x0:x1].sum()) / area


def _is_in_horizontal_row(box: Box, boxes: list[Box]) -> bool:
    """Whether a component belongs to a row of at least three similar glyphs."""

    x0, y0, x1, y1 = box
    height = y1 - y0
    center_y = (y0 + y1) / 2
    row_count = 1
    for other in boxes:
        if other == box:
            continue
        other_height = other[3] - other[1]
        if not 0.6 <= other_height / max(height, 1) <= 1.67:
            continue
        other_center_y = (other[1] + other[3]) / 2
        if abs(other_center_y - center_y) > max(height, other_height) * 0.75:
            continue
        # A letter row may have modest gaps, but a separate vertical object should
        # not be grouped into it.
        gap = max(other[0] - x1, x0 - other[2], 0)
        if gap > max(height * 2, 24):
            continue
        row_count += 1
    return row_count >= 3


def _merge_boxes(boxes: list[Box]) -> list[Box]:
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        result: list[Box] = []
        while merged:
            current = merged.pop()
            index = 0
            while index < len(merged):
                other = merged[index]
                overlaps = not (
                    current[2] < other[0]
                    or other[2] < current[0]
                    or current[3] < other[1]
                    or other[3] < current[1]
                )
                if overlaps:
                    current = (
                        min(current[0], other[0]),
                        min(current[1], other[1]),
                        max(current[2], other[2]),
                        max(current[3], other[3]),
                    )
                    merged.pop(index)
                    changed = True
                    index = 0
                else:
                    index += 1
            result.append(current)
        merged = result
    return sorted(merged, key=lambda box: (box[1], box[0]))


MIN_TEXT_FRAMES = 2  # a box needs at least this many sampled frames to count as text
MAX_BOX_AREA = 0.6  # a box over this share of the frame area is a bright background, not text
MAX_BOX_SIDE = 0.9  # so is a box over this share of both the width and the height


def _covers_frame(box: Box, width: int, height: int) -> bool:
    """Whether ``box`` is too big to be burnt-in text: most of the frame (a white wall, a
    bright sky that stays bright over the whole window)."""

    x0, y0, x1, y1 = box
    box_width, box_height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        return False
    return (
        box_width * box_height > MAX_BOX_AREA * width * height
        or (box_width > MAX_BOX_SIDE * width and box_height > MAX_BOX_SIDE * height)
    )


def detect_text_boxes(
    path: str | Path,
    *,
    start: float = 0.0,
    duration: float | None = None,
    fps: int = 2,
    bright: int = 235,
    min_area: int = 150,
    merge_gap: int = 24,
    fast: bool = False,
    keyframe_frames: np.ndarray | None = None,
) -> list[Box]:
    """Report static bright connected regions likely to be burnt-in text.

    With ``fast`` (for sources above 1920 px) the frames are the keyframes of the window,
    scaled down while decoding, and the boxes are mapped back to source pixels; it falls
    back to every sampled frame (still scaled down) when fewer than 3 keyframes exist.
    ``keyframe_frames`` hands the fast path keyframes already decoded by the caller
    (same scaling and window as ``sample_frames`` would use) instead of decoding them again.
    """
    scale_back = (1.0, 1.0)
    if fast:
        info = probe(path)
        frames = (
            keyframe_frames
            if keyframe_frames is not None
            else sample_frames(path, start, duration, fps=fps, long_side=1280, keyframes_only=True)
        )
        if len(frames) < 3:
            frames = sample_frames(path, start, duration, fps=fps, long_side=1280)
        if info.width and info.height and len(frames):
            scale_back = (info.width / frames.shape[2], info.height / frames.shape[1])
    else:
        frames = sample_frames(path, start, duration, fps=fps)
    if len(frames) < MIN_TEXT_FRAMES:  # text is what stays over several frames: one proves nothing
        return []
    bright_on_frame = frames.min(axis=3) > bright
    persistent = bright_on_frame.mean(axis=0) >= 0.8
    # Inspect smaller pre-merge glyph components as well.  ``min_area`` still
    # applies to the final, dilated detection, while a row of small letters
    # needs its individual components available for the row test.
    raw_boxes = _connected_components(persistent, max(1, min_area // 4))
    kept_boxes: list[Box] = []
    for box in raw_boxes:
        x0, y0, x1, y1 = box
        width = x1 - x0
        height = y1 - y0
        if _component_fill_ratio(persistent, box) > 0.85:
            continue
        if width >= 1.6 * height or _is_in_horizontal_row(box, raw_boxes):
            kept_boxes.append(box)
    filtered = np.zeros_like(persistent)
    for x0, y0, x1, y1 in kept_boxes:
        filtered[y0:y1, x0:x1] = persistent[y0:y1, x0:x1]
    dilated = _dilate(filtered, int(math.ceil(merge_gap / 2)))
    boxes = _connected_components(dilated, min_area)
    height, width = persistent.shape
    padded: list[Box] = []
    for x0, y0, x1, y1 in boxes:
        padded_box = _box_in_frame((x0 - 6, y0 - 6, x1 + 6, y1 + 6), width, height)
        if padded_box is not None:
            padded.append(padded_box)
    merged = [
        box for box in _merge_boxes(padded) if not _covers_frame(box, width, height)
    ]
    if scale_back != (1.0, 1.0):
        source_width, source_height = round(width * scale_back[0]), round(height * scale_back[1])
        merged = [
            box
            for box in (
                _box_in_frame(
                    (
                        math.floor(x0 * scale_back[0]),
                        math.floor(y0 * scale_back[1]),
                        math.ceil(x1 * scale_back[0]),
                        math.ceil(y1 * scale_back[1]),
                    ),
                    source_width,
                    source_height,
                )
                for x0, y0, x1, y1 in merged
            )
            if box is not None
        ]
    return merged


def build_mask(
    path: str | Path,
    start: float,
    duration: float | None,
    rt: RemoveText,
    out: Path,
) -> Path:
    """Build a full-frame grayscale mask for a white static text region."""
    info = probe(path)
    if info.width is None or info.height is None:
        raise ValueError(f"source has no display video dimensions: {path}")
    width, height = info.width, info.height
    box = _require_box(rt.box, width, height)
    frames = sample_frames(path, start, duration, fps=10, box=box)
    if not len(frames):
        raise ValueError("text-removal mask is empty: no frames were sampled")
    inside = np.all(frames.min(axis=3) > rt.threshold, axis=0)
    mask = np.zeros((height, width), dtype=bool)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = inside
    for extra_box in rt.extra_boxes:
        clipped = _box_in_frame(extra_box, width, height)
        if clipped is not None:
            ex0, ey0, ex1, ey1 = clipped
            mask[ey0:ey1, ex0:ex1] = True
    mask = _dilate(mask, rt.dilate)
    if not mask.any():
        raise ValueError(
            "text-removal mask is empty: the box contains no near-white pixels "
            "in every sampled frame"
        )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(out)
    return out


def ffmpeg_filter_path(path: str | Path) -> str:
    """Escape a filesystem path for an ffmpeg filter option value."""
    escaped = str(path).replace("\\", "\\\\")
    escaped = escaped.replace(":", "\\:")
    escaped = escaped.replace("'", "\\'")
    escaped = escaped.replace(" ", "\\ ")
    return f"'{escaped}'"


def _filter_label(label: str) -> str:
    return label if label.startswith("[") else f"[{label}]"


def _delogo_box(box: Box, width: int, height: int) -> Box:
    """Clamp a delogo rectangle so it stays one pixel inside every edge."""
    x0, y0, x1, y1 = _require_box(box, width, height)
    if width < 3 or height < 3:
        raise ValueError("delogo requires a source at least 3x3 pixels")
    x0 = max(1, min(width - 2, x0))
    y0 = max(1, min(height - 2, y0))
    x1 = max(x0 + 1, min(width - 1, x1))
    y1 = max(y0 + 1, min(height - 1, y1))
    return x0, y0, x1, y1


def removal_filter(
    rt: RemoveText,
    mask_input_label: str | None,
    src_w: int,
    src_h: int,
    *,
    mask_path: str | Path | None = None,
) -> tuple[str, bool]:
    """Build the filter graph that consumes ``[vin]`` and produces ``[vout]``.

    ``mask_path`` is optional for the pure filter-builder API.  For convenience,
    a non-label ``mask_input_label`` is also accepted as the mask path and uses
    the conventional second-input label ``[1:v]``.
    """
    box = _require_box(rt.box, src_w, src_h)
    x0, y0, x1, y1 = box
    if rt.method == "delogo":
        x0, y0, x1, y1 = _delogo_box(box, src_w, src_h)
        return f"[vin]delogo=x={x0}:y={y0}:w={x1 - x0}:h={y1 - y0}[vout]", False
    if rt.method == "crop_out":
        if y0 >= src_h - y1:
            return f"[vin]crop=iw:{y0}:0:0[vout]", False
        return f"[vin]crop=iw:ih-{y1}:0:{y1}[vout]", False
    if mask_input_label is None:
        raise ValueError("mask removal requires a second mask input label")
    if mask_path is None and not mask_input_label.startswith("[") and ":v" not in mask_input_label:
        mask_path = mask_input_label
        mask_input_label = "[1:v]"
    if mask_path is None:
        raise ValueError("mask removal requires the mask path")
    mask_input = _filter_label(mask_input_label)
    graph = (
        f"[vin]format=yuv420p,removelogo=f={ffmpeg_filter_path(mask_path)},split[a][b];"
        f"[b]gblur=sigma={rt.blur:g}[bl];"
        f"{mask_input}format=gray,gblur=sigma=5[mk];"
        "[bl][mk]alphamerge[bm];"
        "[a][bm]overlay=shortest=1,format=yuv420p[vout]"
    )
    return graph, True
