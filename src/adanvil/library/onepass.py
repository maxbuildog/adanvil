"""One keyframe decode of a big source for everything indexing needs from it.

Measuring cuts and luma, scanning the first seconds for burnt-in text and cutting the
contact-sheet thumbnails each used to decode the keyframes of the file on their own.
For sources above 1920 px that decode dominates, so this runs it once and feeds the
three branches from the same decoded frames. Every branch keeps the filter chain and
output options of the separate pass it replaces, so the results are the same.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from adanvil import config
from adanvil.ffmpeg import MediaInfo, run
from adanvil.textremoval.core import scaled_size

_PTS_TIME_RE = re.compile(r"pts_time:\s*(-?\d+(?:\.\d+)?)")
_YAVG_RE = re.compile(r"YAVG=([-+]?(?:\d+(?:\.\d*)?|\.\d+))")
_PTS_RE = re.compile(r"pts_time:([-+]?(?:\d+(?:\.\d*)?|\.\d+))")

TEXT_LONG_SIDE = 1280


@dataclass
class KeyframePass:
    cuts: list[float]
    yavg: list[float]
    decoded: int  # keyframes decoded
    thumbs: list[tuple[float, Path]] = field(default_factory=list)
    text_frames: np.ndarray | None = None


def keyframe_pass(
    path: Path,
    info: MediaInfo,
    *,
    threshold: float,
    thumb_dir: Path,
    thumb_width: int,
    text_seconds: float | None,
    fps: int = 4,
    width: int = 360,
) -> KeyframePass:
    """Cuts, luma samples, thumbnails and (when ``text_seconds``) text-scan frames in one pass."""

    thumb_dir.mkdir(parents=True, exist_ok=True)
    pattern = thumb_dir / "key-%05d.jpg"
    graph = (
        "[0:v]showinfo@kf,split=3[a][b][c];"
        f"[a]fps={fps},scale={width}:-2,format=yuv420p,split=2[scene][luma];"
        f"[scene]select='gt(scene,{threshold})',showinfo@cut,nullsink;"
        "[luma]signalstats,metadata=print:key=lavfi.signalstats.YAVG[lumas];"
        f"[b]scale={thumb_width}:-2,showinfo@th[thumbs]"
    )
    source_width, source_height = info.width or 0, info.height or 0
    frame_width, frame_height = source_width, source_height
    if text_seconds is not None:
        if max(source_width, source_height) > TEXT_LONG_SIDE:
            frame_width, frame_height = scaled_size(source_width, source_height, TEXT_LONG_SIDE)
            graph += f";[c]scale={frame_width}:{frame_height},format=rgb24[text]"
        else:
            graph += ";[c]format=rgb24[text]"
    else:
        graph += ";[c]nullsink"
    args = [
        "-loglevel", "info", "-skip_frame", "nokey", *config.thread_args(),
        "-i", str(path),
        "-filter_complex", graph,
        "-map", "[lumas]", "-an", "-f", "null", "-",
        "-map", "[thumbs]", "-fps_mode", "passthrough", "-an", "-q:v", "4", str(pattern),
    ]  # fmt: skip
    if text_seconds is not None:
        args += [
            "-map", "[text]", "-t", str(text_seconds), "-fps_mode", "passthrough", "-an",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]  # fmt: skip
    result = run(args, capture_stdout=text_seconds is not None)
    stderr = result.stderr or ""
    lines = stderr.splitlines()
    cuts = sorted(
        {
            round(float(match.group(1)), 3)
            for line in lines
            if "showinfo@cut" in line
            for match in _PTS_RE.finditer(line)
        }
    )
    yavg = [float(match.group(1)) for match in _YAVG_RE.finditer(stderr)]
    decoded = sum(1 for line in lines if "showinfo@kf" in line and "pts_time" in line)
    times = [
        float(match.group(1))
        for line in lines
        if "showinfo@th" in line
        for match in _PTS_TIME_RE.finditer(line)
    ]
    thumbs: list[tuple[float, Path]] = []
    for number, timestamp in enumerate(times, start=1):
        thumb = thumb_dir / f"key-{number:05d}.jpg"
        if thumb.is_file():
            thumbs.append((timestamp, thumb))
    frames: np.ndarray | None = None
    if text_seconds is not None:
        raw = result.stdout or b""
        frame_size = frame_width * frame_height * 3
        if frame_size <= 0 or len(raw) % frame_size:
            raise ValueError("ffmpeg returned a partial text-scan frame")
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, frame_height, frame_width, 3).copy()
    return KeyframePass(cuts, yavg, decoded, thumbs, frames)
