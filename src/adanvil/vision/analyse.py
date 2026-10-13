"""Look at the frames of one file: embeddings and faces, ready to store."""

from __future__ import annotations

import shutil
import tempfile
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from adanvil import config
from adanvil.ffmpeg import extract_frame

from .engine import Face, VisionEngine
from .store import VisionData

FRAME_WIDTH = 480  # pixels: the width frames are cut at (faces and tags need no more)
BIG_SOURCE_SIDE = 1920  # above this long side, frames are cut from keyframes (cheap)


def extract_frames(
    path: Path,
    times: Sequence[float],
    directory: Path,
    *,
    big: bool = False,
    width: int = FRAME_WIDTH,
) -> list[Path]:
    """One picture per time, in order (a time past the end gives the last frame)."""

    directory.mkdir(parents=True, exist_ok=True)
    threads = config.ffmpeg_threads()

    def cut(item: tuple[int, float]) -> Path:
        number, moment = item
        output = directory / f"frame-{number:03d}.png"
        if big:
            try:
                extract_frame(path, moment, output, width=width, keyframe=True, threads=threads)
                if output.is_file():
                    return output
            except Exception:
                output.unlink(missing_ok=True)
        # A time after the start of the last frame gives no picture: step back until one comes.
        for back in (0.0, 0.15, 0.4, 1.0):
            try:
                extract_frame(path, max(0.0, moment - back), output, width=width, threads=threads)
            except Exception:
                output.unlink(missing_ok=True)
            if output.is_file():
                return output
        raise OSError(f"no frame could be cut from {path.name} near {moment:.2f}s")

    with ThreadPoolExecutor(max_workers=max(1, min(config.workers(), len(times) or 1))) as pool:
        return list(pool.map(cut, list(enumerate(times))))


def analyse_pictures(
    engine: VisionEngine,
    pictures: Sequence[Path],
    times: Sequence[float],
    *,
    embeddings: bool = True,
) -> VisionData:
    """Embeddings (unless ``embeddings`` is false) and faces of ``pictures``, taken at ``times``."""

    vectors = engine.embed_images(list(pictures)) if embeddings and pictures else None
    found = engine.detect_faces(list(pictures)) if pictures else []
    faces: list[list[Face]] = [list(item.faces) for item in found]
    sizes = [item.size for item in found]
    return VisionData.build(engine.model, list(times), sizes, vectors, faces)


def analyse_media(
    engine: VisionEngine,
    path: Path,
    *,
    kind: str,
    times: Sequence[float],
    big: bool = False,
    embeddings: bool = True,
    scratch: Path | None = None,
) -> VisionData:
    """Analyse a video (frames cut at ``times``) or an image (the picture itself)."""

    if kind == "image":
        return analyse_pictures(engine, [path], [0.0], embeddings=embeddings)
    folder = Path(tempfile.mkdtemp(prefix=f"adanvil-vision-{uuid.uuid4().hex[:8]}-", dir=scratch))
    try:
        pictures = extract_frames(path, times, folder, big=big)
        return analyse_pictures(engine, pictures, times, embeddings=embeddings)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


__all__ = ["BIG_SOURCE_SIDE", "FRAME_WIDTH", "analyse_media", "analyse_pictures", "extract_frames"]
