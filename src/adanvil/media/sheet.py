"""Contact-sheet generation for rendered videos."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import textwrap
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from adanvil import config
from adanvil.ffmpeg import extract_frame, probe, run
from adanvil.text.styles import save_image

from .cache import Cache

_PTS_TIME_RE = re.compile(r"pts_time:\s*(-?\d+(?:\.\d+)?)")


def _load_frame(path: Path, width: int) -> Image.Image:
    """Open a frame as RGB and normalize its width for a tiled sheet."""

    with Image.open(path) as source:
        image = source.convert("RGB")
    height = max(1, round(image.height * width / image.width))
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    *,
    fill: tuple[int, int, int] = (245, 245, 245),
) -> None:
    """Draw a compact caption in a fixed-width sheet cell."""

    x0, y0, x1, y1 = box
    chars = max(1, (x1 - x0) // 7)
    lines = textwrap.wrap(text, width=chars, break_long_words=False) or [""]
    line_height = max(12, draw.textbbox((0, 0), "Ag", font=font)[3])
    for index, line in enumerate(lines):
        y = y0 + index * (line_height + 3)
        if y + line_height > y1:
            break
        draw.text((x0, y), line, font=font, fill=fill)


def reference_shot_sheets(
    video: str | Path,
    shots: Sequence[object],
    out_dir: str | Path,
    *,
    spoken: Sequence[str] | None = None,
    shots_per_sheet: int = 6,
    frame_width: int = 240,
) -> list[Path]:
    """Build readable three-frame sheets for consecutive reference shots.

    ``shots`` may contain objects with ``start``/``end``/``index`` attributes
    or mappings with the same keys.  Keeping this helper independent of the
    reference models also lets the media package use it without a circular
    import.
    """

    if shots_per_sheet <= 0 or frame_width <= 0:
        raise ValueError("shots_per_sheet and frame_width must be positive")
    video = Path(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = probe(video)
    if info.duration is None or info.duration <= 0:
        raise ValueError(f"video has no positive duration: {video}")

    def value(shot: object, key: str) -> object:
        if isinstance(shot, Mapping):
            return shot[key]
        return getattr(shot, key)

    sheets: list[Path] = []
    font = ImageFont.load_default()
    caption_width = 360
    row_gap = 8
    for sheet_number, start_index in enumerate(range(0, len(shots), shots_per_sheet), start=1):
        group = shots[start_index : start_index + shots_per_sheet]
        rows: list[tuple[list[Image.Image], str]] = []
        for offset, shot in enumerate(group):
            start = float(value(shot, "start"))
            end = float(value(shot, "end"))
            index = int(value(shot, "index"))
            frame_times = [
                max(start, min(end, start + 0.08)),
                max(start, min(end, (start + end) / 2)),
                max(start, min(end, end - 0.08)),
            ]
            frames: list[Image.Image] = []
            for frame_number, timestamp in enumerate(frame_times):
                timestamp = min(timestamp, max(0.0, info.duration - 0.001))
                frame_path = out_dir / f".shot-{sheet_number:02d}-{offset:02d}-{frame_number}.png"
                extract_frame(video, timestamp, frame_path, width=frame_width)
                try:
                    frames.append(_load_frame(frame_path, frame_width))
                finally:
                    frame_path.unlink(missing_ok=True)
            spoken_text = ""
            if spoken is not None and 0 <= index - 1 < len(spoken):
                spoken_text = str(spoken[index - 1])[:80]
            label = f"S{index}  {start:.2f}-{end:.2f} s ({end - start:.2f} s)"
            if spoken_text:
                label += f"\n{spoken_text}"
            rows.append((frames, label))

        row_height = max(max(frame.height for frame in frames) for frames, _ in rows)
        canvas = Image.new(
            "RGB",
            (frame_width * 3 + caption_width, row_height * len(rows) + row_gap * (len(rows) - 1)),
            "#202020",
        )
        draw = ImageDraw.Draw(canvas)
        for row_index, (frames, label) in enumerate(rows):
            y = row_index * (row_height + row_gap)
            for frame_index, frame in enumerate(frames):
                canvas.paste(frame, (frame_index * frame_width, y))
                frame.close()
            draw.rectangle(
                (frame_width * 3, y, canvas.width, y + row_height),
                fill="#202020",
            )
            _draw_wrapped(
                draw,
                label,
                (frame_width * 3 + 14, y + 14, canvas.width - 14, y + row_height - 10),
                font,
            )
        output = out_dir / f"shots-{sheet_number:02d}.jpg"
        save_image(canvas, output, format="JPEG", quality=90, optimize=True)
        sheets.append(output)
    return sheets


BIG_SOURCE_SIDE = 1920  # above this long side a mosaic thumbnail is cut from a keyframe
_THUMB_VERSION = 1


def _mosaic_thumbnail(
    path: Path, thumb_width: int, cache_dir: Path | None, scratch: Path
) -> Image.Image:
    """One mosaic cell: a cached thumbnail, or a fresh one stored under its file key."""

    cached: Path | None = None
    if cache_dir is not None:
        try:
            key = json.dumps([Cache.file_key(path), thumb_width, _THUMB_VERSION], sort_keys=True)
            cached = cache_dir / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()[:20]}.png"
        except OSError:
            cached = None
    if cached is not None and cached.is_file():
        try:
            with Image.open(cached) as stored:
                return stored.convert("RGB")
        except (OSError, ValueError):
            cached.unlink(missing_ok=True)

    info = probe(path)
    if info.kind == "video":
        timestamp = 0.4 * float(info.duration or 0.0)
        frame_path = scratch / f".mosaic-{os.getpid()}-{uuid.uuid4().hex}.png"
        big = max(info.width or 0, info.height or 0) > BIG_SOURCE_SIDE
        threads = config.ffmpeg_threads()
        try:
            if big:
                try:
                    extract_frame(
                        path, timestamp, frame_path, width=thumb_width, keyframe=True,
                        threads=threads,
                    )  # fmt: skip
                except Exception:
                    frame_path.unlink(missing_ok=True)
            if not frame_path.is_file():
                extract_frame(path, timestamp, frame_path, width=thumb_width, threads=threads)
            image = _load_frame(frame_path, thumb_width)
        finally:
            frame_path.unlink(missing_ok=True)
    elif info.kind == "image":
        image = _load_frame(path, thumb_width)
    else:
        return Image.new("RGB", (thumb_width, 120), "#505050")
    if cached is not None:
        temporary = cached.with_name(f".{cached.stem}.{os.getpid()}.{uuid.uuid4().hex}.png")
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            image.save(temporary, format="PNG")
            os.replace(temporary, cached)
        except OSError:
            pass
        finally:
            temporary.unlink(missing_ok=True)
    return image


def media_mosaic(
    media: Sequence[Path],
    out: str | Path,
    *,
    columns: int = 6,
    thumb_width: int = 180,
    labels: Sequence[str] | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Create a six-column thumbnail mosaic for video and image files.

    Thumbnails are cut in parallel (``ADANVIL_WORKERS``); with ``cache_dir`` each one is
    stored under the key of its file, so another page or a second call reuses it.
    """

    if not media:
        raise ValueError("media must not be empty")
    if columns <= 0 or thumb_width <= 0:
        raise ValueError("columns and thumb_width must be positive")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    thumbs = Path(cache_dir) if cache_dir is not None else None
    with ThreadPoolExecutor(max_workers=max(1, min(config.workers(), len(media)))) as pool:
        prepared = list(
            pool.map(
                lambda item: _mosaic_thumbnail(Path(item), thumb_width, thumbs, out.parent),
                media,
            )
        )
    cell_height = max(image.height for image in prepared)

    label_height = 30
    rows = math.ceil(len(prepared) / columns)
    canvas = Image.new(
        "RGB",
        (columns * thumb_width, rows * (cell_height + label_height)),
        "#202020",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, image in enumerate(prepared):
        x = (index % columns) * thumb_width
        y = (index // columns) * (cell_height + label_height)
        canvas.paste(image, (x, y))
        image.close()
        label = labels[index] if labels is not None else Path(media[index]).name
        if len(label) > 22:
            label = f"{label[:19]}..."
        draw.rectangle(
            (x, y + cell_height, x + thumb_width, y + cell_height + label_height),
            fill="#202020",
        )
        draw.text((x + 4, y + cell_height + 8), label, font=font, fill=(245, 245, 245))
    save_image(canvas, out, format="JPEG", quality=88, optimize=True)
    return out


def _keyframe_thumbnails(
    video: Path, directory: Path, thumb_width: int
) -> list[tuple[float, Path]]:
    """Decode only the keyframes of ``video`` as small JPEGs, with their timestamps."""

    pattern = directory / "key-%05d.jpg"
    result = run(
        [
            "-loglevel",
            "info",
            "-skip_frame",
            "nokey",
            "-i",
            str(video),
            "-vf",
            f"scale={thumb_width}:-2,showinfo",
            "-fps_mode",
            "passthrough",
            "-an",
            "-q:v",
            "4",
            str(pattern),
        ]
    )
    times = [float(match.group(1)) for match in _PTS_TIME_RE.finditer(result.stderr or "")]
    found: list[tuple[float, Path]] = []
    for number, timestamp in enumerate(times, start=1):
        path = directory / f"key-{number:05d}.jpg"
        if path.is_file():
            found.append((timestamp, path))
    return found


def contact_sheet(
    video: str | Path,
    out: str | Path,
    *,
    n: int = 16,
    cols: int = 8,
    thumb_width: int = 240,
    labels: bool = True,
    timestamps: Sequence[float] | None = None,
    shot_labels: Mapping[int, str] | None = None,
    keyframe_windows: Sequence[tuple[float, float]] | None = None,
    keyframes: Sequence[tuple[float, Path]] | None = None,
) -> Path:
    """Extract frames and tile them into a JPEG contact sheet.

    ``timestamps`` and ``shot_labels`` are optional extensions used by the
    footage library. Existing callers continue to get evenly spaced frames.

    ``keyframe_windows`` (one ``(start, end)`` per timestamp) is for sources above
    1920 px: the keyframes are decoded in one cheap pass and each timestamp takes the
    keyframe of its window nearest to it (and is labelled with that keyframe's time);
    only a timestamp with no keyframe in its window gets an exact-timestamp extraction.
    ``keyframes`` passes keyframe thumbnails (time, JPEG) the caller already decoded at
    ``thumb_width``, in place of that pass; the caller keeps ownership of the files.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    if cols <= 0:
        raise ValueError("cols must be positive")
    if thumb_width <= 0:
        raise ValueError("thumb_width must be positive")

    video = Path(video)
    out = Path(out)
    info = probe(video)
    if info.duration is None or info.duration <= 0:
        raise ValueError(f"video has no positive duration: {video}")
    if timestamps is not None:
        if not timestamps:
            raise ValueError("timestamps must not be empty")
        n = len(timestamps)
    if keyframe_windows is not None and (
        timestamps is None or len(keyframe_windows) != len(timestamps)
    ):
        raise ValueError("keyframe_windows needs one window per timestamp")
    rows = math.ceil(n / cols)
    out.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = out.parent / f".adanvil-sheet-{os.getpid()}-{uuid.uuid4().hex}"
    temporary_path.mkdir(parents=True, exist_ok=False)
    try:
        frames: list[tuple[float, Image.Image]] = []
        frame_times = (
            list(timestamps)
            if timestamps is not None
            else [(index + 0.5) * info.duration / n for index in range(n)]
        )
        available: list[tuple[float, Path]] = list(keyframes or [])
        if keyframe_windows is not None and keyframes is None:
            try:
                available = _keyframe_thumbnails(video, temporary_path, thumb_width)
            except Exception:  # fall back to exact extraction of every frame
                available = []
        for index, timestamp in enumerate(frame_times):
            timestamp = max(0.0, min(float(timestamp), max(0.0, info.duration - 0.001)))
            if available and keyframe_windows is not None:
                low, high = keyframe_windows[index]
                inside = [item for item in available if low <= item[0] <= high]
                if inside:
                    key_time, key_path = min(inside, key=lambda item: abs(item[0] - timestamp))
                    with Image.open(key_path) as image:
                        frames.append((key_time, image.convert("RGB")))
                    continue
            frame_path = temporary_path / f"frame-{index:03d}.png"
            # A seek within the last frame of a short or variable-rate file can yield
            # no picture: step back a few frames before giving up on this thumbnail.
            for back_off in (0.0, 0.1, 0.25, 0.5):
                extract_frame(video, max(0.0, timestamp - back_off), frame_path, width=thumb_width)
                if frame_path.is_file():
                    break
            if not frame_path.is_file():
                continue
            with Image.open(frame_path) as image:
                frames.append((timestamp, image.convert("RGB")))
        if not frames:
            raise ValueError(f"could not extract any frame from {video}")

        thumb_height = frames[0][1].height
        canvas = Image.new("RGB", (cols * thumb_width, rows * thumb_height), "#202020")
        font = ImageFont.load_default()
        draw = ImageDraw.Draw(canvas)
        for index, (timestamp, image) in enumerate(frames):
            if image.size != (thumb_width, thumb_height):
                image = image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
            x = (index % cols) * thumb_width
            y = (index // cols) * thumb_height
            canvas.paste(image, (x, y))
            if labels:
                label = f"{timestamp:.1f}s" if timestamps is not None else f"{timestamp:.2f}s"
                if shot_labels and index in shot_labels:
                    label = f"{label} {shot_labels[index]}"
                bbox = draw.textbbox((0, 0), label, font=font)
                padding = 3
                draw.rectangle(
                    (
                        x,
                        y,
                        x + bbox[2] - bbox[0] + padding * 2,
                        y + bbox[3] - bbox[1] + padding * 2,
                    ),
                    fill=(0, 0, 0),
                )
                draw.text((x + padding, y + padding), label, fill=(255, 255, 255), font=font)

        save_image(canvas, out, format="JPEG", quality=90, optimize=True)
    finally:
        for temporary_frame in temporary_path.iterdir():
            temporary_frame.unlink(missing_ok=True)
        temporary_path.rmdir()
    return out


__all__ = ["contact_sheet"]
