"""Fast footage tree overview and thumbnail mosaic, without indexing."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from adanvil import ffmpeg
from adanvil.media.cache import Cache
from adanvil.media.sheet import media_mosaic

from .index import IMAGE_EXTENSIONS, MEDIA_EXTENSIONS, VIDEO_EXTENSIONS

MOSAIC_LIMIT = 48  # thumbnails per mosaic page
_SKIPPED = {"cache", "renders"}


def _walk(root: Path) -> list[tuple[Path, list[Path]]]:
    """Return ``(directory, media files directly inside)`` for every directory."""

    found: list[tuple[Path, list[Path]]] = []
    visited: set[str] = set()
    for directory, directories, filenames in os.walk(root, followlinks=True):
        real_directory = os.path.normcase(os.path.realpath(directory))
        if real_directory in visited:
            directories[:] = []
            continue
        visited.add(real_directory)
        directories[:] = sorted(
            name
            for name in directories
            if not name.startswith(".")
            and name not in _SKIPPED
            and os.path.normcase(os.path.realpath(os.path.join(directory, name))) not in visited
        )
        files = [
            Path(directory) / name
            for name in sorted(filenames)
            if not name.startswith(".")
            and Path(name).suffix.casefold() in MEDIA_EXTENSIONS
            and (Path(directory) / name).is_file()
        ]
        found.append((Path(directory), files))
    return found


def _durations(root: Path, videos: list[Path]) -> dict[Path, float]:
    cache_path = root / "cache" / "overview.json"
    try:
        stored = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(stored, dict):
            stored = {}
    except (OSError, ValueError):
        stored = {}

    keys: dict[Path, str] = {}
    for video in videos:
        try:
            keys[video] = json.dumps(Cache.file_key(video), sort_keys=True)
        except OSError:
            continue
    missing = [video for video, key in keys.items() if key not in stored]

    def measure(video: Path) -> float:
        try:
            return float(ffmpeg.probe(video).duration or 0.0)
        except Exception:
            return 0.0

    if missing:
        with ThreadPoolExecutor(max_workers=min(8, len(missing))) as pool:
            for video, seconds in zip(missing, pool.map(measure, missing), strict=True):
                stored[keys[video]] = seconds
        alive = set(keys.values())
        pruned = {key: value for key, value in stored.items() if key in alive}
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(pruned), encoding="utf-8")
        except OSError:
            pass
    return {video: float(stored.get(key, 0.0)) for video, key in keys.items()}


def library_overview(
    root: Path, path: str = "footage", depth: int = 2, mosaic: bool = True, page: int = 1
) -> dict[str, Any]:
    """Describe the folder tree under ``path`` and optionally draw a mosaic.

    Mosaics show ``MOSAIC_LIMIT`` files per page, in path order; ``page``
    selects which one (1-based), so a flat folder of hundreds of B-rolls can
    be browsed page by page.
    """

    from adanvil.project import resolve_in_project

    base = resolve_in_project(root, path)
    if not base.is_dir():
        raise FileNotFoundError(f"not a directory: {path}")
    depth = max(0, int(depth))
    walked = _walk(base)
    files_by_dir = dict(walked)
    every = [f for _, files in walked for f in files]
    all_videos = [f for f in every if f.suffix.casefold() in VIDEO_EXTENSIONS]
    seconds = _durations(root, all_videos)

    def node(directory: Path, level: int) -> dict[str, Any]:
        below = [
            (d, files)
            for d, files in walked
            if d == directory or _is_under(d, directory)
        ]
        files = [f for _, group in below for f in group]
        videos = [f for f in files if f.suffix.casefold() in VIDEO_EXTENSIONS]
        images = [f for f in files if f.suffix.casefold() in IMAGE_EXTENSIONS]
        result: dict[str, Any] = {
            "path": _rel(root, directory),
            "files": len(files),
            "videos": len(videos),
            "images": len(images),
            "total_video_seconds": round(sum(seconds.get(v, 0.0) for v in videos), 2),
            "first_files": [f.name for f in files_by_dir.get(directory, [])[:5]]
            or [f.name for f in files[:5]],
        }
        if level < depth:
            result["folders"] = [
                node(child, level + 1) for child in _children(directory, walked)
            ]
        return result

    tree = node(base, 0)
    visual = [f for f in every if f.suffix.casefold() in VIDEO_EXTENSIONS | IMAGE_EXTENSIONS]
    payload: dict[str, Any] = {"path": _rel(root, base), "tree": tree}
    if not mosaic:
        return payload
    if not visual:
        payload["note"] = "no video or image files under this path; no mosaic generated"
        return payload
    pages = (len(visual) + MOSAIC_LIMIT - 1) // MOSAIC_LIMIT
    page = min(max(1, int(page)), pages)
    visual = visual[(page - 1) * MOSAIC_LIMIT : page * MOSAIC_LIMIT]
    payload["page"] = page
    payload["pages"] = pages
    if pages > 1:
        payload["note"] = (
            f"page {page} of {pages} ({MOSAIC_LIMIT} files per mosaic): call again with "
            f"page={page + 1 if page < pages else 1}, or on a subfolder"
        )
    key = hashlib.sha256(
        json.dumps(
            [json.dumps(Cache.file_key(f), sort_keys=True) for f in visual]
        ).encode("utf-8")
    ).hexdigest()[:16]
    out = root / "cache" / "overview" / f"{key}.jpg"
    if not out.is_file():
        media_mosaic(visual, out, cache_dir=root / "cache" / "thumbs")
    payload["mosaic"] = _rel(root, out)
    payload["mosaic_files"] = [_rel(root, f) for f in visual]
    return payload


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _children(directory: Path, walked: list[tuple[Path, list[Path]]]) -> list[Path]:
    return [d for d, _ in walked if d.parent == directory]


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


__all__ = ["library_overview"]
