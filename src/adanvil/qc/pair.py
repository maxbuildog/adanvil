"""How different two pictures look: the score that ranks candidate before/after pairs.

A before/after only works when the viewer sees the difference at a glance. ``compare_pair``
measures it on two pictures (still images, or frames of clips at given times), and draws them
side by side so an agent can LOOK at the pair it is about to use.

The measure, in the order it is applied (every number is a guideline, not a rule):

1. Each side may be cut with a ``crop`` (``x``, ``y``, ``w``, ``h`` in source pixels), which is
   how one half of a brand's side-by-side composite is taken without preparing files.
2. A face is looked for in each picture. With the vision environment installed
   (``adanvil vision-setup``) the YuNet face detector finds it (``face_source`` is
   ``detector``). Without it a skin-region heuristic is used (``find_face``): the skin pixels
   (YCbCr range, the classic one) are binned on a coarse grid, the largest connected blob that
   has a plausible size, shape and fill is the face. No model, no new dependency, and it is
   only a heuristic: a beige wall can pass for skin and a very dark or very pale skin can be
   missed, so the result says which region was used (``region``) and the preview shows it.
3. When a face is found in BOTH pictures, the two faces are compared (a square around each
   face, resized to ``FACE_SIZE`` pixels, so a pair taken a little closer or further still
   lines up). Otherwise the whole frames are compared after aligning sizes: the second picture
   is cover-cropped to the aspect of the first and both are resized to ``FRAME_WIDTH`` pixels.
4. Both are blurred slightly (noise and compression do not count as a difference) and a small
   translation (``SHIFT`` pixels either way) is searched, so a pair that is a few pixels off is
   not mistaken for a different picture.
5. ``visible_difference`` (0 to 100) mixes the mean colour difference and the share of pixels
   that changed by more than ``CHANGED_LEVEL`` (out of 255):
   ``100 * (0.55 * min(1, changed_share / 0.30) + 0.45 * min(1, mean_delta / 0.12))``.

Bands (``verdict``): under 12 ``not_visible``, under 30 ``subtle``, under 55 ``visible``,
from 55 ``obvious``. Two near-identical frames land near 0; a messy and a tidy room, or a
clearly different face, land above 55. Rank candidate pairs with the score, then look at the
previews: the number does not know what the viewer cares about.
"""

from __future__ import annotations

import hashlib
import tempfile
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from adanvil.ffmpeg import extract_frame, probe
from adanvil.text.styles import save_image

FACE_SIZE = 128  # pixels: the square the two faces are resized to
FRAME_WIDTH = 128  # pixels: the width both frames are resized to when no face is compared
BLUR_RADIUS = 1.5
SHIFT = 3  # pixels at the comparison size: the translation searched either way
CHANGED_LEVEL = 40  # a pixel counts as changed above this difference (out of 255)
BANDS = ((12.0, "not_visible"), (30.0, "subtle"), (55.0, "visible"))
_GRID_WIDTH = 64  # cells across for the skin blob search
_MIN_FACE_SHARE = 0.012  # of the picture
_MAX_FACE_SHARE = 0.70
_FACE_ASPECT = (0.45, 1.6)  # width / height of the blob
_MIN_FILL = 0.5  # blob area / bounding box area (an ellipse is 0.79)


class PairError(ValueError):
    """The two pictures cannot be compared (unreadable file, bad crop, bad time)."""


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    """Boolean mask of the pixels whose colour is in the usual skin range (YCbCr)."""

    red = rgb[..., 0].astype(np.float32)
    green = rgb[..., 1].astype(np.float32)
    blue = rgb[..., 2].astype(np.float32)
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = 128.0 - 0.168736 * red - 0.331264 * green + 0.5 * blue
    cr = 128.0 + 0.5 * red - 0.418688 * green - 0.081312 * blue
    return (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173) & (luma >= 40)


def find_face(image: Image.Image) -> dict[str, Any] | None:
    """The most plausible face of ``image`` as ``{"box": (left, top, right, bottom), "share"}``
    in the pixels of ``image``, or ``None`` when no skin blob looks like a face.

    A skin heuristic, not a detector: see the module docstring for its limits."""

    rgb = image.convert("RGB")
    scale = _GRID_WIDTH / max(rgb.width, 1)
    small = rgb.resize(
        (_GRID_WIDTH, max(1, round(rgb.height * scale))), Image.Resampling.BOX
    )
    cells = skin_mask(np.asarray(small)).astype(np.uint8)
    rows, cols = cells.shape
    seen = np.zeros_like(cells, dtype=bool)
    best: tuple[float, int, tuple[int, int, int, int]] | None = None
    for row in range(rows):
        for col in range(cols):
            if not cells[row, col] or seen[row, col]:
                continue
            queue = deque([(row, col)])
            seen[row, col] = True
            area = 0
            top, left, bottom, right = row, col, row, col
            while queue:
                y, x = queue.popleft()
                area += 1
                top, bottom = min(top, y), max(bottom, y)
                left, right = min(left, x), max(right, x)
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < rows and 0 <= nx < cols and cells[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            width, height = right - left + 1, bottom - top + 1
            share = area / (rows * cols)
            fill = area / (width * height)
            aspect = width / height
            if not _MIN_FACE_SHARE <= share <= _MAX_FACE_SHARE:
                continue
            if not _FACE_ASPECT[0] <= aspect <= _FACE_ASPECT[1] or fill < _MIN_FILL:
                continue
            if best is None or area > best[1]:
                best = (share, area, (left, top, right + 1, bottom + 1))
    if best is None:
        return None
    share, _area, (left, top, right, bottom) = best
    step_x, step_y = rgb.width / cols, rgb.height / rows
    box = (
        round(left * step_x),
        round(top * step_y),
        min(rgb.width, round(right * step_x)),
        min(rgb.height, round(bottom * step_y)),
    )
    return {"box": box, "share": round(share, 4)}


# ----------------------------------------------------------------------------- the pictures


def _check_crop(crop: dict[str, Any] | None, size: tuple[int, int], who: str) -> tuple[int, ...]:
    if crop is None:
        return (0, 0, size[0], size[1])
    try:
        x, y, w, h = (int(crop[key]) for key in ("x", "y", "w", "h"))
    except (KeyError, TypeError, ValueError) as exc:
        raise PairError(f"{who} crop must be an object with integer x, y, w, h") from exc
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > size[0] or y + h > size[1]:
        raise PairError(f"{who} crop {x},{y} {w}x{h} leaves the {size[0]}x{size[1]} picture")
    return (x, y, x + w, y + h)


def load_picture(
    path: Path,
    *,
    t: float | None = None,
    crop: dict[str, Any] | None = None,
    who: str = "picture",
    scratch: Path | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """The picture of ``path``: the image itself, or the frame of a clip at ``t`` seconds
    (the middle of the clip without ``t``), then cut by ``crop`` when given."""

    try:
        info = probe(path)
    except Exception as exc:
        raise PairError(f"cannot read {who} {path.name}: {exc}") from exc
    detail: dict[str, Any] = {"kind": info.kind}
    if info.kind == "audio":
        raise PairError(f"{who} {path.name} is an audio file")
    if info.kind == "video":
        moment = float(t) if t is not None else (info.duration or 0.0) / 2.0
        if moment < 0 or (info.duration is not None and moment > info.duration):
            raise PairError(
                f"{who} time {moment:g}s is outside the clip ({info.duration or 0:.2f}s)"
            )
        detail["t"] = round(moment, 3)
        if scratch is not None:
            scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="adanvil-pair-", dir=scratch) as temporary:
            frame = Path(temporary) / "frame.png"
            extract_frame(path, moment, frame)
            with Image.open(frame) as opened:
                picture = opened.convert("RGB")
    else:
        try:
            with Image.open(path) as opened:
                picture = ImageOps.exif_transpose(opened).convert("RGB")
        except (OSError, ValueError) as exc:
            raise PairError(f"cannot open {who} {path.name}: {exc}") from exc
    detail["size"] = list(picture.size)
    box = _check_crop(crop, picture.size, who)
    if crop is not None:
        picture = picture.crop(box)
        detail["crop"] = {"x": box[0], "y": box[1], "w": box[2] - box[0], "h": box[3] - box[1]}
    return picture, detail


def _cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """``image`` scaled to cover ``size`` and cropped in the middle."""

    scale = max(size[0] / image.width, size[1] / image.height)
    scaled = image.resize(
        (max(size[0], round(image.width * scale)), max(size[1], round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = (scaled.width - size[0]) // 2
    top = (scaled.height - size[1]) // 2
    return scaled.crop((left, top, left + size[0], top + size[1]))


def _face_square(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """A square around a face box with some margin, moved inside the picture."""

    centre_x, centre_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = min(max(box[2] - box[0], box[3] - box[1]) * 1.3, min(image.size))
    left = min(max(0.0, centre_x - side / 2), image.width - side)
    top = min(max(0.0, centre_y - side / 2), image.height - side)
    return (round(left), round(top), round(left + side), round(top + side))


def _aligned_difference(first: Image.Image, second: Image.Image) -> tuple[float, float]:
    """Mean colour difference (0 to 1) and share of changed pixels, after the best small shift."""

    a = np.asarray(first.filter(ImageFilter.GaussianBlur(BLUR_RADIUS)), dtype=np.float32)
    b = np.asarray(second.filter(ImageFilter.GaussianBlur(BLUR_RADIUS)), dtype=np.float32)
    height, width = a.shape[:2]
    best: tuple[float, np.ndarray] | None = None
    for dy in range(-SHIFT, SHIFT + 1):
        for dx in range(-SHIFT, SHIFT + 1):
            ys = slice(max(0, dy), height + min(0, dy))
            xs = slice(max(0, dx), width + min(0, dx))
            yb = slice(max(0, -dy), height + min(0, -dy))
            xb = slice(max(0, -dx), width + min(0, -dx))
            difference = np.abs(a[ys, xs] - b[yb, xb])
            mean = float(difference.mean())
            if best is None or mean < best[0]:
                best = (mean, difference)
    assert best is not None
    difference = best[1]
    mean_delta = float(difference.mean() / 255.0)
    changed = float((difference.max(axis=-1) >= CHANGED_LEVEL).mean())
    return mean_delta, changed


def visible_difference(mean_delta: float, changed_share: float) -> float:
    """The 0 to 100 score from the two measures (see the module docstring)."""

    score = 0.55 * min(1.0, changed_share / 0.30) + 0.45 * min(1.0, mean_delta / 0.12)
    return round(100.0 * score, 1)


def verdict(score: float) -> str:
    """``not_visible``, ``subtle``, ``visible`` or ``obvious``."""

    for limit, name in BANDS:
        if score < limit:
            return name
    return "obvious"


FaceFinder = Callable[[Image.Image], "dict[str, Any] | None"]


def detector_face_finder(engine: Any) -> FaceFinder:
    """A face finder that asks the vision engine's face detector (the biggest face wins)."""

    def find(image: Image.Image) -> dict[str, Any] | None:
        with tempfile.TemporaryDirectory(prefix="adanvil-face-") as folder:
            path = Path(folder) / "picture.png"
            image.convert("RGB").save(path)
            pictures = engine.detect_faces([path])
        if not pictures or not pictures[0].faces:
            return None
        face = max(pictures[0].faces, key=lambda item: item.area)
        left, top, right, bottom = face.box
        box = (
            round(left * image.width),
            round(top * image.height),
            round(right * image.width),
            round(bottom * image.height),
        )
        return {"box": box, "share": round(face.area, 4)}

    return find


def compare_images(
    before: Image.Image, after: Image.Image, *, face_finder: FaceFinder | None = None
) -> dict[str, Any]:
    """Compare two pictures: the score, the region used and the face boxes found.

    ``face_finder`` finds the face of a picture (``{"box", "share"}`` or ``None``); the skin
    heuristic ``find_face`` is used without one."""

    notes: list[str] = []
    finder = face_finder or find_face
    face_before, face_after = finder(before), finder(after)
    if face_before and face_after:
        region = "face"
        square_before = _face_square(before, face_before["box"])
        square_after = _face_square(after, face_after["box"])
        first = before.crop(square_before).resize((FACE_SIZE, FACE_SIZE), Image.Resampling.LANCZOS)
        second = after.crop(square_after).resize((FACE_SIZE, FACE_SIZE), Image.Resampling.LANCZOS)
    else:
        region = "frame"
        if face_before or face_after:
            notes.append(
                "a face was found in only one of the pictures: the whole frames were compared"
            )
        size = (FRAME_WIDTH, max(8, round(FRAME_WIDTH * before.height / before.width)))
        first = before.resize(size, Image.Resampling.LANCZOS)
        second = _cover(after, size)
        if abs(before.width / before.height - after.width / after.height) > 0.05:
            notes.append(
                "the pictures have different shapes: the second was cover-cropped to the shape "
                "of the first before comparing"
            )
    mean_delta, changed = _aligned_difference(first, second)
    score = visible_difference(mean_delta, changed)
    return {
        "visible_difference": score,
        "verdict": verdict(score),
        "region": region,
        "face_source": "detector" if face_finder is not None else "skin heuristic",
        "metrics": {"mean_delta": round(mean_delta, 4), "changed_share": round(changed, 4)},
        "faces": {
            "before": face_before["box"] if face_before else None,
            "after": face_after["box"] if face_after else None,
        },
        "notes": notes,
    }


def pair_preview(
    before: Image.Image,
    after: Image.Image,
    result: dict[str, Any],
    out: Path,
    *,
    height: int = 640,
) -> Path:
    """The two pictures side by side at one height, labelled, with the compared face boxes
    drawn when the comparison was made on the faces. Saved as a JPEG at ``out``."""

    panels: list[Image.Image] = []
    for name, image in (("before", before), ("after", after)):
        scale = height / image.height
        panel = image.resize((max(1, round(image.width * scale)), height), Image.Resampling.LANCZOS)
        box = result["faces"][name] if result["region"] == "face" else None
        if box is not None:
            square = _face_square(image, box)
            ImageDraw.Draw(panel).rectangle(
                [round(value * scale) for value in square], outline=(255, 64, 64), width=4
            )
        panels.append(panel)
    gap, strip = 12, 44
    canvas = Image.new(
        "RGB",
        (panels[0].width + panels[1].width + gap, height + strip),
        (24, 24, 24),
    )
    canvas.paste(panels[0], (0, strip))
    canvas.paste(panels[1], (panels[0].width + gap, strip))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), "BEFORE", fill=(255, 255, 255))
    draw.text((panels[0].width + gap + 10, 12), "AFTER", fill=(255, 255, 255))
    summary = (
        f"visible_difference {result['visible_difference']:g} ({result['verdict']}), "
        f"compared on the {result['region']}"
    )
    draw.text((canvas.width - 10 - 7 * len(summary), 12), summary, fill=(255, 220, 90))
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(canvas, out, format="JPEG", quality=88)
    return out


def compare_pair(
    root: Path,
    before: Path,
    after: Path,
    *,
    before_t: float | None = None,
    after_t: float | None = None,
    before_crop: dict[str, Any] | None = None,
    after_crop: dict[str, Any] | None = None,
    out_dir: Path | None = None,
    face_finder: FaceFinder | None = None,
) -> tuple[dict[str, Any], Path]:
    """Compare two project pictures; returns the result and the path of the preview image.

    The faces come from the vision environment's detector when it is installed (or from
    ``face_finder``), else from the skin heuristic."""

    if face_finder is None:
        from adanvil.vision import get_engine

        engine = get_engine()
        if engine is not None:
            face_finder = detector_face_finder(engine)

    folder = out_dir if out_dir is not None else Path(root) / "cache" / "compare_pair"
    first, first_detail = load_picture(
        before, t=before_t, crop=before_crop, who="before", scratch=folder
    )
    second, second_detail = load_picture(
        after, t=after_t, crop=after_crop, who="after", scratch=folder
    )
    try:
        result = compare_images(first, second, face_finder=face_finder)
    except Exception:
        if face_finder is None:
            raise
        result = compare_images(first, second)  # the detector failed: use the heuristic
    result["before"] = first_detail
    result["after"] = second_detail
    key = hashlib.sha1(
        repr(
            (str(before), str(after), before_t, after_t, before_crop, after_crop)
        ).encode("utf-8")
    ).hexdigest()[:12]
    preview = pair_preview(first, second, result, folder / f"pair_{key}.jpg")
    first.close()
    second.close()
    return result, preview


__all__ = [
    "BANDS",
    "PairError",
    "compare_images",
    "compare_pair",
    "detector_face_finder",
    "find_face",
    "load_picture",
    "pair_preview",
    "skin_mask",
    "verdict",
    "visible_difference",
]
