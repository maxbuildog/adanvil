"""Technical checks and control images for rendered advertisements."""

from __future__ import annotations

import math
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from adanvil.audio import rms_per_second
from adanvil.ffmpeg import FFmpegError, extract_frame, probe, run
from adanvil.media import Format, contact_sheet
from adanvil.render.compose import RenderReport
from adanvil.text.styles import save_image

CheckStatus = Literal["pass", "warn", "fail"]


@dataclass
class Check:
    """One automatic quality-control result."""

    id: str
    status: CheckStatus
    message: str
    value: Any = None


@dataclass
class QCReport:
    """All checks and visual artefacts produced for one video."""

    ok: bool
    checks: list[Check]
    images: list[Path]
    summary: dict[str, Any]


_DEFAULT_FORMAT = Format()
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_BLACK_FIELD = re.compile(
    rf"black_start:\s*(?P<start>{_NUMBER}).*?"
    rf"black_end:\s*(?P<end>{_NUMBER}).*?"
    rf"black_duration:\s*(?P<duration>{_NUMBER})"
)
_FREEZE_START = re.compile(rf"freeze_start:\s*(?P<value>{_NUMBER})")
_FREEZE_END = re.compile(rf"freeze_end:\s*(?P<value>{_NUMBER})")
_LUFS = re.compile(rf"^\s*I:\s*(?P<value>{_NUMBER}|-inf|inf)\s+LUFS", re.MULTILINE)


def _check(
    checks: list[Check], check_id: str, status: CheckStatus, message: str, value: Any = None
) -> None:
    checks.append(Check(check_id, status, message, value))


def _duration_from_report(report: RenderReport | None) -> float | None:
    if report is None:
        return None
    duration = getattr(report, "duration", None)
    if duration is not None:
        try:
            parsed = float(duration)
        except (TypeError, ValueError):
            parsed = 0.0
        if math.isfinite(parsed) and parsed > 0:
            return parsed
    timeline = getattr(report, "timeline", [])
    ends: list[float] = []
    for item in timeline:
        if isinstance(item, dict):
            try:
                end = float(item["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(end):
                ends.append(end)
    return max(ends, default=None)


def _first_error_lines(error: FFmpegError) -> list[str]:
    return [line.strip() for line in error.stderr_tail.splitlines() if line.strip()][:3]


def _decode_check(video: Path, checks: list[Check]) -> None:
    stderr = ""
    error_lines: list[str] = []
    try:
        result = run(["-v", "error", "-i", str(video), "-f", "null", "-"])
        stderr = str(result.stderr or "")
    except FFmpegError as error:
        error_lines = _first_error_lines(error)
    lines = error_lines or [line.strip() for line in stderr.splitlines() if line.strip()]
    if lines:
        _check(
            checks,
            "decode",
            "fail",
            "ffmpeg decode errors: " + " | ".join(lines[:3]),
            lines[:3],
        )
    else:
        _check(checks, "decode", "pass", "video decodes without ffmpeg errors")


def _run_video_filter(video: Path, filter_name: str) -> str:
    result = run(
        [
            "-loglevel",
            "info",
            "-i",
            str(video),
            "-vf",
            filter_name,
            "-an",
            "-f",
            "null",
            "-",
        ]
    )
    return str(result.stderr or "")


def _black_intervals(stderr: str) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for match in _BLACK_FIELD.finditer(stderr):
        start = float(match.group("start"))
        end = float(match.group("end"))
        if end >= start:
            intervals.append((round(start, 3), round(end, 3)))
    return intervals


def _freeze_intervals(stderr: str, duration: float | None) -> list[tuple[float, float]]:
    starts = [float(match.group("value")) for match in _FREEZE_START.finditer(stderr)]
    ends = [float(match.group("value")) for match in _FREEZE_END.finditer(stderr)]
    intervals: list[tuple[float, float]] = []
    for index, start in enumerate(starts):
        end = ends[index] if index < len(ends) else duration
        if end is not None and end >= start:
            intervals.append((round(start, 3), round(end, 3)))
    return intervals


def _image_difference(frame: Image.Image, overlay: Image.Image) -> float | None:
    frame_rgb = np.asarray(frame.convert("RGB"), dtype=np.int16)
    overlay_rgba = np.asarray(overlay.convert("RGBA"), dtype=np.int16)
    if frame_rgb.shape[:2] != overlay_rgba.shape[:2]:
        return None
    mask = overlay_rgba[:, :, 3] > 200
    if not np.any(mask):
        return None
    difference = np.abs(frame_rgb[mask] - overlay_rgba[:, :, :3][mask])
    return float(np.mean(difference))


def _overlay_bounds(overlay: Image.Image) -> tuple[int, int, int, int]:
    """The box of everything the overlay draws (any non-transparent pixel: a translucent caption
    box and its shadow count), so a crop around it holds the whole caption."""

    alpha = overlay.convert("RGBA").getchannel("A")
    return alpha.getbbox() or (0, 0, overlay.width, overlay.height)


def _clamp_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int] | None:
    """Clamp a PIL crop box and reject rectangles with no pixels."""

    left = max(0, min(width, int(round(box[0]))))
    top = max(0, min(height, int(round(box[1]))))
    right = max(0, min(width, int(round(box[2]))))
    bottom = max(0, min(height, int(round(box[3]))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _scaled_overlay(source: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Put an overlay PNG on the actual video's pixel grid."""

    overlay = source.convert("RGBA")
    if overlay.size == size:
        return overlay
    return overlay.resize(size, Image.Resampling.LANCZOS)


def _scaled_report_bbox(
    item: dict[str, Any], source_size: tuple[int, int], frame_size: tuple[int, int]
) -> tuple[float, float, float, float] | None:
    raw = item.get("bbox")
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    source_width, source_height = source_size
    frame_width, frame_height = frame_size
    return (
        x0 * frame_width / max(source_width, 1),
        y0 * frame_height / max(source_height, 1),
        x1 * frame_width / max(source_width, 1),
        y1 * frame_height / max(source_height, 1),
    )


def _overlay_crop_indexes(count: int) -> set[int]:
    if count <= 4:
        return set(range(count))
    indexes = {0, count - 1, round((count - 1) / 3), round(2 * (count - 1) / 3)}
    return indexes


def _overlay_path(item: dict[str, Any]) -> Path:
    value = item.get("png")
    if value is None:
        raise ValueError("render report overlay has no png path")
    return Path(value)


def _open_overlay(item: dict[str, Any], when: float) -> Image.Image:
    """The overlay picture on screen at ``when`` as a full-canvas RGBA image.

    An animated caption lists its ``states`` (a picture cropped to the ink of the caption, with
    its ``offset`` on the ``canvas``): the one whose span holds ``when`` is put back on a
    transparent canvas. Any other overlay is a full-canvas picture already.
    """

    states = item.get("states")
    path = _overlay_path(item)
    offset = item.get("offset")
    if isinstance(states, list) and states:
        chosen = states[-1]
        for state in states:
            if float(state["start"]) <= when < float(state["end"]):
                chosen = state
                break
        path = Path(chosen["png"])
        offset = chosen.get("offset")
    with Image.open(path) as source:
        picture = source.convert("RGBA")
    canvas = item.get("canvas")
    if isinstance(offset, (list, tuple)) and isinstance(canvas, (list, tuple)):
        full = Image.new("RGBA", (int(canvas[0]), int(canvas[1])), (0, 0, 0, 0))
        full.paste(picture, (int(offset[0]), int(offset[1])))
        picture.close()
        return full
    return picture


def _overlay_times(item: dict[str, Any], duration: float) -> tuple[float, float]:
    start = max(0.0, float(item.get("start", 0.0)))
    raw_end = item.get("end")
    end = duration if raw_end is None else min(duration, float(raw_end))
    if end < start:
        end = start
    middle = (start + end) / 2.0
    ending = max(0.0, end - 0.05)
    settled = item.get("settled")
    if isinstance(settled, (list, tuple)) and len(settled) == 2:
        # An animated card is checked once it has come in and before it goes out.
        first, last = float(settled[0]), float(settled[1])
        middle = min(max(middle, first), max(first, last - 0.02))
        ending = min(ending, max(first, last - 0.02))
    return middle, ending


def _save_jpeg(image: Image.Image, path: Path) -> None:
    save_image(image.convert("RGB"), path, format="JPEG", quality=92)


def _slug(text: Any, limit: int = 30) -> str:
    """A short ASCII slug of a caption text for a file name (empty when there is none)."""

    if not isinstance(text, str):
        return ""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    words = re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")
    return words[:limit].strip("-")


def _crop_path(out_dir: Path, label: int, taken: list[Path], text: Any = None) -> Path:
    """``overlay_<storyboard index>_<short text>.jpg``; a storyboard overlay that shows as several
    report entries (animated chunks, voice captions) gets ``..._2.jpg`` and so on."""

    slug = _slug(text)
    stem = f"overlay_{label}" + (f"_{slug}" if slug else "")
    path = out_dir / f"{stem}.jpg"
    count = 1
    while path in taken:
        count += 1
        path = out_dir / f"{stem}_{count}.jpg"
    return path


Box = tuple[float, float, float, float]
CROP_MARGIN = 80


def _overlaps(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _inside(inner: Box, outer: Box) -> bool:
    return (
        inner[0] >= outer[0]
        and inner[1] >= outer[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def _crop_around(own: Box, neighbours: list[Box], margin: float = CROP_MARGIN) -> Box:
    """The crop box around ``own`` (a margin all round) that never cuts a neighbouring overlay
    in half: a neighbour that touches ``own`` is included whole, any other one that the margin
    would clip is left out by stopping the crop at its edge (tighter to the box)."""

    left, top, right, bottom = own[0] - margin, own[1] - margin, own[2] + margin, own[3] + margin
    for _round in range(3):
        for other in neighbours:
            if not _overlaps((left, top, right, bottom), other):
                continue
            if _inside(other, (left, top, right, bottom)):
                continue
            if _overlaps(own, other):  # it touches the box: take it whole
                left, top = min(left, other[0]), min(top, other[1])
                right, bottom = max(right, other[2]), max(bottom, other[3])
            elif other[1] >= own[3]:
                bottom = min(bottom, max(other[1] - 1, own[3]))
            elif other[3] <= own[1]:
                top = max(top, min(other[3] + 1, own[1]))
            elif other[0] >= own[2]:
                right = min(right, max(other[0] - 1, own[2]))
            elif other[2] <= own[0]:
                left = max(left, min(other[2] + 1, own[0]))
    return left, top, right, bottom


def _report_boxes_at(
    overlays: list[Any], skip: int, moment: float, frame_width: int
) -> list[Box]:
    """The boxes (in pixels of a frame ``frame_width`` wide) of the other report overlays on
    screen at ``moment``."""

    boxes: list[Box] = []
    for number, raw in enumerate(overlays):
        if number == skip:
            continue
        other = dict(raw)
        box = other.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        begin = float(other.get("start", 0.0))
        finish = other.get("end")
        if moment < begin or (finish is not None and moment >= float(finish)):
            continue
        canvas = other.get("canvas") or [frame_width]
        scale = frame_width / max(float(canvas[0]), 1.0)
        boxes.append(tuple(float(value) * scale for value in box))  # type: ignore[arg-type]
    return boxes


def _box_crop(
    item: dict[str, Any],
    index: int,
    frame_path: Path,
    out_dir: Path,
    checks: list[Check],
    images: list[Path],
    crop_indexes: set[int],
    label: int | None = None,
    neighbours: list[Box] | None = None,
) -> None:
    """Overlays that are not one still picture (a video inset, a progress bar) have no picture
    to compare with the frame: their crop is the reference box of the report, plus a margin.
    A video inset is always cropped, so it is always looked at."""

    if index not in crop_indexes and item.get("type") != "video":
        return
    label = index if label is None else label
    with Image.open(frame_path) as opened:
        frame = opened.convert("RGB")
    raw = item.get("bbox")
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        _check(checks, "overlay_crop", "warn", f"overlay {label} has no box to crop; skipped")
        return
    scale = frame.width / max(float(item.get("canvas", [frame.width])[0]), 1.0)
    own = tuple(float(value) * scale for value in raw)
    box = _clamp_box(
        _crop_around(own, neighbours or []),  # type: ignore[arg-type]
        frame.width,
        frame.height,
    )
    if box is None:
        _check(checks, "overlay_crop", "warn", f"overlay {label} crop is empty; skipped", raw)
        return
    path = _crop_path(out_dir, label, images, item.get("text"))
    _save_jpeg(frame.crop(box), path)
    images.append(path)


def _split_frames(
    video: Path, report: RenderReport, out_dir: Path, temporary: Path, images: list[Path]
) -> None:
    """One frame, in the middle of each split screen segment, so both panels and their labels
    are looked at; for each reveal segment, one frame in the middle of the sweep (the divider
    and both pictures) and one on the settled after picture."""

    for number, item in enumerate(getattr(report, "timeline", [])):
        if not isinstance(item, dict) or item.get("type") not in {"split", "reveal"}:
            continue
        start, end = float(item["start"]), float(item["end"])
        if item["type"] == "split":
            shots = [("split", (start + end) / 2.0)]
        else:
            sweep_start = start + float(item.get("hold_before", 0.0))
            sweep_end = sweep_start + float(item.get("sweep", 0.0))
            shots = [
                ("reveal_sweep", min(end - 0.04, (sweep_start + sweep_end) / 2.0)),
                ("reveal_after", max(start, end - 0.1)),
            ]
        for name, moment in shots:
            raw = temporary / f"{name}-{number}.png"
            extract_frame(video, moment, raw, width=540, on_screen=True)
            with Image.open(raw) as opened:
                path = out_dir / f"{name}_{number}.jpg"
                _save_jpeg(opened, path)
            images.append(path)


_STILL_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _end_card_frame(
    video: Path,
    report: RenderReport,
    out_dir: Path,
    temporary: Path,
    images: list[Path],
    duration: float | None = None,
) -> None:
    """One frame near the end of the last still segment (the end card), in the format of the
    render, so a caption over its button is seen in every format (``endcard.jpg``). When the
    still runs to the last frame of the video, ``end.jpg`` is that same picture and nothing is
    saved twice."""

    stills = [
        item
        for item in getattr(report, "timeline", [])
        if isinstance(item, dict)
        and Path(str(item.get("source", ""))).suffix.lower() in _STILL_SUFFIXES
    ]
    if not stills:
        return
    last = stills[-1]
    if duration is not None and float(last["end"]) >= duration - 0.1:
        return  # the last frame (end.jpg) already is the end card
    moment = max(float(last["start"]), float(last["end"]) - 0.3)
    raw = temporary / "endcard.png"
    extract_frame(video, moment, raw, width=540, on_screen=True)
    with Image.open(raw) as opened:
        path = out_dir / "endcard.jpg"
        _save_jpeg(opened, path)
    images.append(path)


def _name_entries(overlays: list[Any], model: Any) -> list[dict[str, Any]]:
    """The report entries with their storyboard ``overlay`` index and ``text``. A report written
    before the entries carried them (an older render) is named from the storyboard: its pictures
    are the non-sound overlays in order, so when the counts agree entry ``n`` is that overlay."""

    stored = getattr(model, "overlays", None) if model is not None else None
    visual = [
        number for number, item in enumerate(stored or []) if getattr(item, "type", "") != "sfx"
    ]
    aligned = stored is not None and len(visual) == len(overlays)
    named: list[dict[str, Any]] = []
    for number, raw in enumerate(overlays):
        item = dict(raw)
        if not isinstance(item.get("overlay"), int) and aligned:
            item["overlay"] = visual[number]
        label = item.get("overlay")
        if not item.get("text") and stored is not None and isinstance(label, int):
            if label < len(stored):
                lines = getattr(stored[label], "lines", None)
                if isinstance(lines, list):
                    item["text"] = " ".join(str(line) for line in lines)
        named.append(item)
    return named


def _overlay_checks_and_images(
    video: Path,
    duration: float,
    report: RenderReport,
    out_dir: Path,
    checks: list[Check],
    images: list[Path],
    temporary: Path,
    model: Any = None,
) -> None:
    overlays = _name_entries(getattr(report, "overlays", []), model)
    crop_indexes = _overlay_crop_indexes(len(overlays))
    for index, raw_item in enumerate(overlays):
        item = dict(raw_item)
        label = int(item["overlay"]) if isinstance(item.get("overlay"), int) else index
        middle, ending = _overlay_times(item, duration)
        middle_frame_path = temporary / f"overlay-{index}-middle.png"
        ending_frame_path = temporary / f"overlay-{index}-end.png"
        extract_frame(video, middle, middle_frame_path)
        if item.get("png") is None:
            with Image.open(middle_frame_path) as opened:
                frame_width = opened.width
            _box_crop(
                item,
                index,
                middle_frame_path,
                out_dir,
                checks,
                images,
                crop_indexes,
                label,
                _report_boxes_at(overlays, index, middle, frame_width),
            )
            continue
        extract_frame(video, ending, ending_frame_path)

        with (
            _open_overlay(item, middle) as source_overlay,
            Image.open(middle_frame_path) as middle_frame,
        ):
            frame = middle_frame.convert("RGB")
            overlay = _scaled_overlay(source_overlay, frame.size)
            report_bbox = _scaled_report_bbox(item, source_overlay.size, frame.size)
            bounds = report_bbox or tuple(float(value) for value in _overlay_bounds(overlay))
            crop_box = _clamp_box(
                _crop_around(
                    bounds, _report_boxes_at(overlays, index, middle, frame.width)
                ),
                frame.width,
                frame.height,
            )
            if index in crop_indexes:
                if crop_box is None:
                    _check(
                        checks,
                        "overlay_crop",
                        "warn",
                        f"overlay {label} crop is empty; skipped",
                        [*bounds],
                    )
                else:
                    crop_path = _crop_path(out_dir, label, images, item.get("text"))
                    _save_jpeg(frame.crop(crop_box), crop_path)
                    images.append(crop_path)
            middle_difference = _image_difference(frame, overlay)
            overlay.close()

        with (
            _open_overlay(item, ending) as source_overlay,
            Image.open(ending_frame_path) as ending_frame,
        ):
            ending_overlay = _scaled_overlay(source_overlay, ending_frame.size)
            ending_difference = _image_difference(ending_frame.convert("RGB"), ending_overlay)
            ending_overlay.close()

        differences = {
            "mid_pixel_difference_0_255": middle_difference,
            "end_pixel_difference_0_255": ending_difference,
            "mid_time_s": round(middle, 4),
            "end_time_s": round(ending, 4),
            "meaning": "mean absolute difference per colour channel between the rendered frame "
            "and the overlay picture, on a 0-255 scale; under 30 means the overlay is on screen",
        }
        scores = (middle_difference, ending_difference)
        visible = all(value is not None and value < 30 for value in scores)
        shown = _slug(item.get("text"), 20)
        name = f"overlay {label}" + (f" ({shown})" if shown else "")
        if visible:
            message = f"{name} is on screen at its mid-time and end-time"
            _check(checks, "overlay_visible", "pass", message, differences)
        else:
            _check(
                checks,
                "overlay_visible",
                "fail",
                f"{name}: caption disappears before its end",
                differences,
            )


def _report_matches(report: RenderReport, size: tuple[int, int]) -> bool:
    """Whether the overlay pictures of ``report`` were drawn on a canvas of the video's size
    (a report of another format places its boxes elsewhere)."""

    for item in getattr(report, "overlays", []):
        canvas = dict(item).get("canvas")
        if isinstance(canvas, (list, tuple)) and len(canvas) == 2:
            return (int(canvas[0]), int(canvas[1])) == tuple(size)
    return True


def _text_boxes(storyboard: dict[str, Any], root: Path) -> tuple[Any, dict[int, tuple[float, ...]]]:
    from adanvil.formats import _boxes
    from adanvil.storyboard.schema import Storyboard

    model = Storyboard.model_validate(storyboard)
    return model, _boxes(model, root)


def _overlay_range(overlay: Any, total: float) -> tuple[float, float]:
    start = float(getattr(overlay, "start", 0.0) or 0.0)
    end = getattr(overlay, "end", None)
    return start, min(total, float(end)) if end is not None else total


def _layout_crops(
    video: Path,
    duration: float,
    model: Any,
    boxes: dict[int, tuple[float, ...]],
    out_dir: Path,
    checks: list[Check],
    images: list[Path],
    temporary: Path,
) -> None:
    """The overlay crops from the boxes of the effective storyboard of THIS render (the
    re-laid positions of a format), so no caption is cut in half."""

    indexes = sorted(boxes)
    wanted = _overlay_crop_indexes(len(indexes))
    for position, index in enumerate(indexes):
        if position not in wanted:
            continue
        start, end = _overlay_range(model.overlays[index], duration)
        moment = max(0.0, min((start + end) / 2.0, duration - 0.05))
        raw = temporary / f"layout-{index}.png"
        extract_frame(video, moment, raw, on_screen=True)
        with Image.open(raw) as opened:
            frame = opened.convert("RGB")
        factor = frame.width / max(model.format.width, 1)
        left, top, right, bottom = boxes[index]
        others: list[Box] = []
        for other_index, other_box in boxes.items():
            if other_index == index:
                continue
            other_start, other_end = _overlay_range(model.overlays[other_index], duration)
            if other_start <= moment < other_end:
                others.append(tuple(value * factor for value in other_box))  # type: ignore[arg-type]
        crop = _clamp_box(
            _crop_around((left * factor, top * factor, right * factor, bottom * factor), others),
            frame.width,
            frame.height,
        )
        if crop is None:
            _check(checks, "overlay_crop", "warn", f"overlay {index} crop is empty; skipped")
            continue
        lines = getattr(model.overlays[index], "lines", None)
        words = " ".join(str(line) for line in lines) if isinstance(lines, list) else None
        path = _crop_path(out_dir, index, images, words)
        _save_jpeg(frame.crop(crop), path)
        images.append(path)


QC_FACE_LOOKS = 4  # frames looked at per caption (each is a detection)


def _caption_faces_check(
    video: Path,
    duration: float,
    model: Any,
    boxes: dict[int, tuple[float, ...]],
    checks: list[Check],
    document: dict[str, Any] | None = None,
    root: Path | None = None,
) -> None:
    """``caption_on_face``: a caption that covers a face on the rendered frame (``QC_FACE_LOOKS``
    frames spread over each caption's time range, faces found by the vision engine).

    With the effective ``document`` and the project ``root``, the same final position is also
    checked against the SOURCE face data at the same moments (what ``render_formats`` noted at
    layout time). When the two disagree, both verdicts are reported."""

    from adanvil.captionfaces import first_covered, frame_faces, rendered_faces, sample_moments
    from adanvil.storyboard.schema import CaptionOverlay, CommentOverlay
    from adanvil.vision import get_engine

    texts = [
        index
        for index in sorted(boxes)
        if isinstance(model.overlays[index], (CaptionOverlay, CommentOverlay))
    ]
    if not texts:
        return
    try:
        engine = get_engine()
    except Exception:
        engine = None
    if engine is None:
        _check(
            checks,
            "caption_on_face",
            "pass",
            "skipped: no vision engine (install it with adanvil vision-setup)",
        )
        return
    clashes: list[str] = []
    for index in texts[:16]:
        start, end = _overlay_range(model.overlays[index], duration)
        moments = sample_moments(start, min(end, duration - 0.05), duration, count=QC_FACE_LOOKS)
        left, top, right, bottom = boxes[index]
        rendered_at = None
        for moment in moments:
            try:
                faces, size = rendered_faces(video, moment, engine)
            except Exception:
                continue
            fx, fy = size[0] / model.format.width, size[1] / model.format.height
            where = (left * fx, top * fy, right * fx, bottom * fy)
            if first_covered(where, [(moment, faces)]) is not None:
                rendered_at = moment
                break
        source_at = None
        if document is not None and root is not None:
            looks: list[tuple[float, list[Any]]] = []
            for moment in moments:
                try:
                    found = frame_faces(document, Path(root), moment)
                except Exception:
                    found = None
                if found is not None:
                    looks.append((moment, found))
            source_at = first_covered((left, top, right, bottom), looks) if looks else None
        if rendered_at is not None and (source_at is not None or document is None):
            clashes.append(f"overlay {index} at {rendered_at:.1f}s")
        elif rendered_at is not None:
            clashes.append(
                f"overlay {index} at {rendered_at:.1f}s (rendered-frame detection finds a face "
                "there; the source face data finds none)"
            )
        elif source_at is not None:
            clashes.append(
                f"overlay {index} at {source_at:.1f}s (source face data says it covers a face; "
                "rendered-frame detection finds none: check the frame by eye)"
            )
    if clashes:
        _check(
            checks,
            "caption_on_face",
            "warn",
            "caption_on_face: a caption covers a face (" + ", ".join(clashes) + ")",
            clashes,
        )
    else:
        _check(checks, "caption_on_face", "pass", "no caption covers a face")


def _loudness(video: Path) -> float | None:
    stderr = str(
        run(
            [
                "-loglevel",
                "info",
                "-i",
                str(video),
                "-map",
                "0:a:0",
                "-af",
                "ebur128=framelog=verbose",
                "-vn",
                "-f",
                "null",
                "-",
            ]
        ).stderr
        or ""
    )
    values = _LUFS.findall(stderr)
    if not values or values[-1].lower() in {"inf", "-inf"}:
        return None
    return float(values[-1])


def qc_report(
    video: Path,
    *,
    render_report: RenderReport | None = None,
    expected_duration: float | None = None,
    fmt: Format = _DEFAULT_FORMAT,
    out_dir: Path | None = None,
    silent_ok: bool = False,
    storyboard: dict[str, Any] | None = None,
    root: Path | None = None,
) -> QCReport:
    """Run the automatic delivery checks and save visual control images.

    ``storyboard`` (with the project ``root``) is the EFFECTIVE storyboard of this render, the
    one of its format after the re-layout (``renders/<batch>/_storyboards/<name>.json``). The
    overlay crops are then taken where its overlays really are, when the render report is
    missing or belongs to another format, and the ``caption_on_face`` check runs on the
    rendered frames (vision engine)."""

    video = Path(video)
    if render_report is not None and render_report.quality == "preview":
        fmt = Format(width=540, height=960)
    output_dir = Path(out_dir) if out_dir is not None else video.with_name(f"{video.stem}_qc")
    output_dir.mkdir(parents=True, exist_ok=True)
    checks: list[Check] = []
    images: list[Path] = []
    info = probe(video)
    duration = info.duration

    _decode_check(video, checks)

    dimensions_ok = (info.width, info.height) == (fmt.width, fmt.height)
    _check(
        checks,
        "resolution",
        "pass" if dimensions_ok else "fail",
        f"display size is {info.width}x{info.height}; expected {fmt.width}x{fmt.height}"
        if not dimensions_ok
        else f"display size is {fmt.width}x{fmt.height}",
        [info.width, info.height],
    )

    fps_ok = info.fps is not None and abs(info.fps - 30.0) <= 0.01
    _check(
        checks,
        "fps",
        "pass" if fps_ok else "fail",
        f"frame rate is {info.fps:g} fps" if info.fps is not None else "frame rate is missing",
        info.fps,
    )

    pixel_ok = info.pix_fmt == "yuv420p"
    _check(
        checks,
        "pixel_format",
        "pass" if pixel_ok else "fail",
        f"pixel format is {info.pix_fmt or 'missing'}"
        + ("; expected yuv420p" if not pixel_ok else ""),
        info.pix_fmt,
    )

    tags = (info.color_space, info.color_primaries, info.color_transfer)
    colors_ok = tags == ("bt709", "bt709", "bt709")
    _check(
        checks,
        "color_tags",
        "pass" if colors_ok else "fail",
        (
            "color tags are bt709/bt709/bt709"
            if colors_ok
            else "color tags must be bt709/bt709/bt709; players will show a red/dark cast otherwise"
        ),
        {
            "color_space": info.color_space,
            "color_primaries": info.color_primaries,
            "color_transfer": info.color_transfer,
        },
    )

    target_duration = (
        expected_duration
        if expected_duration is not None
        else _duration_from_report(render_report)
    )
    if target_duration is None:
        _check(checks, "duration", "pass", "skipped: expected duration is unknown")
    else:
        duration_ok = duration is not None and abs(duration - float(target_duration)) <= 0.05
        _check(
            checks,
            "duration",
            "pass" if duration_ok else "fail",
            (
                f"duration is {duration:.3f}s; expected {float(target_duration):.3f}s"
                if duration is not None
                else "video duration is missing"
            ),
            duration,
        )

    if info.has_audio:
        _check(checks, "audio_present", "pass", "audio stream is present", True)
    else:
        _check(
            checks,
            "audio_present",
            "pass" if silent_ok else "fail",
            "audio stream is absent (silent_ok)" if silent_ok else "audio stream is absent",
            False,
        )

    audio_duration_ok = (
        info.has_audio
        and duration is not None
        and info.audio_duration is not None
        and abs(info.audio_duration - duration) <= 0.05
    )
    _check(
        checks,
        "audio_duration",
        "pass" if audio_duration_ok else "fail",
        (
            f"audio duration is {info.audio_duration:.3f}s; video duration is {duration:.3f}s"
            if info.audio_duration is not None and duration is not None
            else "audio and video durations cannot be compared"
        ),
        info.audio_duration,
    )

    rms = rms_per_second(video) if info.has_audio else []
    if silent_ok:
        _check(checks, "audio_tail", "pass", "skipped: silent_ok")
    elif (
        info.audio_duration is not None
        and duration is not None
        and info.audio_duration < duration - 0.5
    ):
        # The tail is judged on the video timeline: a track that ends early still has
        # sound in its own last second, which would hide the bug this check exists for.
        gap = duration - info.audio_duration
        _check(checks, "audio_tail", "warn", f"audio stops {gap:.1f}s before the end", gap)
    elif not rms:
        _check(checks, "audio_tail", "pass", "skipped: no complete audio-second window")
    elif rms[-1] <= -50:
        _check(checks, "audio_tail", "warn", "audio stops before the end", rms[-1])
    else:
        _check(
            checks,
            "audio_tail",
            "pass",
            "audio remains present in the last full second",
            rms[-1],
        )

    gap_seconds = [index for index, value in enumerate(rms[1:-1], start=1) if value < -60]
    if gap_seconds:
        _check(
            checks,
            "audio_gaps",
            "warn",
            "silent whole seconds inside the audio: "
            + ", ".join(str(index) for index in gap_seconds),
            gap_seconds,
        )
    else:
        _check(checks, "audio_gaps", "pass", "no silent whole-second gaps", gap_seconds)

    black = _black_intervals(_run_video_filter(video, "blackdetect=d=0.2:pix_th=0.10"))
    _check(
        checks,
        "black_frames",
        "warn" if black else "pass",
        "black intervals: " + ", ".join(f"{start:.2f}-{end:.2f}s" for start, end in black)
        if black
        else "no black intervals detected",
        black,
    )

    frozen = _freeze_intervals(
        _run_video_filter(video, "freezedetect=n=0.001:d=1.5"), duration
    )
    _check(
        checks,
        "frozen_frames",
        "warn" if frozen else "pass",
        "frozen intervals: " + ", ".join(f"{start:.2f}-{end:.2f}s" for start, end in frozen)
        if frozen
        else "no frozen intervals detected",
        frozen,
    )

    sheet_path = output_dir / "sheet.jpg"
    contact_sheet(video, sheet_path, n=16)
    images.append(sheet_path)

    with tempfile.TemporaryDirectory(prefix="adanvil-qc-", dir=output_dir) as temporary_dir:
        temporary = Path(temporary_dir)
        laid_out: tuple[Any, dict[int, tuple[float, ...]]] | None = None
        if storyboard is not None and root is not None and duration is not None:
            try:
                laid_out = _text_boxes(storyboard, Path(root))
            except Exception as exc:
                _check(checks, "overlay_crop", "warn", f"storyboard not usable for crops: {exc}")
        report_fits = render_report is not None and _report_matches(
            render_report, (info.width or 0, info.height or 0)
        )
        if render_report is not None and duration is not None:
            if report_fits or laid_out is None:
                _overlay_checks_and_images(
                    video,
                    duration,
                    render_report,
                    output_dir,
                    checks,
                    images,
                    temporary,
                    laid_out[0] if laid_out is not None else None,
                )
            _split_frames(video, render_report, output_dir, temporary, images)
            _end_card_frame(video, render_report, output_dir, temporary, images, duration)
        if laid_out is not None and duration is not None:
            model, boxes = laid_out
            if render_report is None or not report_fits:
                _layout_crops(video, duration, model, boxes, output_dir, checks, images, temporary)
            _caption_faces_check(video, duration, model, boxes, checks, storyboard, root)

        if duration is not None and duration > 0:
            last_frame_path = temporary / "last-frame.png"
            last_time = max(0.0, duration - 1.0 / max(info.fps or 30.0, 1.0))
            extract_frame(video, last_time, last_frame_path, width=540, on_screen=True)
            with Image.open(last_frame_path) as last_frame:
                end_path = output_dir / "end.jpg"
                _save_jpeg(last_frame, end_path)
            images.append(end_path)

    loudness = _loudness(video) if info.has_audio else None
    # Feeds normalise around -14 LUFS; far quieter ads sound weak next to other posts.
    too_quiet = loudness is not None and not silent_ok and loudness < -24
    too_loud = loudness is not None and loudness > -9
    _check(
        checks,
        "loudness",
        "warn" if too_quiet or too_loud else "pass",
        (
            f"integrated loudness is {loudness:.2f} LUFS"
            + (" (too quiet: aim for about -14 LUFS)" if too_quiet else "")
            + (" (too loud: aim for about -14 LUFS)" if too_loud else "")
            if loudness is not None
            else "integrated loudness is unavailable"
        ),
        loudness,
    )

    failures = sum(check.status == "fail" for check in checks)
    warnings = sum(check.status == "warn" for check in checks)
    summary = {
        "video": str(video),
        "duration": duration,
        "expected_duration": target_duration,
        "pass": len(checks) - failures - warnings,
        "warn": warnings,
        "fail": failures,
        "statuses": [check.status for check in checks],
    }
    return QCReport(ok=failures == 0, checks=checks, images=images, summary=summary)


__all__ = ["Check", "QCReport", "qc_report"]
