"""Timing benchmark for the slow stages of AdAnvil (not shipped in the package).

Generates short 4K HEVC fixtures in a throwaway project (or reuses ``--project``) and times:
indexing N files, one overview mosaic page, a cold preview render, a final render after
that preview, and a cold final render.

    uv run python scripts/bench.py --files 12 --seconds 4
    ADANVIL_WORKERS=1 uv run python scripts/bench.py   # sequential reference

Needs ffmpeg on PATH (or ADANVIL_FFMPEG). Prints one JSON document with the timings in seconds.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from adanvil import config
from adanvil.ffmpeg import run
from adanvil.library.index import index_library
from adanvil.library.overview import library_overview
from adanvil.render import compose
from adanvil.storyboard import Storyboard

_PATTERNS = ("testsrc2", "smptehdbars", "gradients", "rgbtestsrc")


def make_fixture(path: Path, seed: int, seconds: float, size: str) -> None:
    """A two-shot 4K HEVC clip (a cut at the half), with a keyframe every second."""

    half = seconds / 2
    first = _PATTERNS[seed % len(_PATTERNS)]
    second = _PATTERNS[(seed + 1) % len(_PATTERNS)]
    run(
        [
            "-f", "lavfi", "-i", f"{first}=s={size}:r=30:d={half}",
            "-f", "lavfi", "-i", f"{second}=s={size}:r=30:d={half}",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p10le[v]",
            "-map", "[v]",
            "-c:v", "libx265", "-preset", "ultrafast", "-crf", "30", "-g", "30",
            "-x265-params", "log-level=error",
            "-tag:v", "hvc1",
            str(path),
        ]
    )  # fmt: skip


def timed(label: str, results: dict[str, float], action) -> object:
    start = time.perf_counter()
    value = action()
    results[label] = round(time.perf_counter() - start, 2)
    print(f"{label}: {results[label]} s", flush=True)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--files", type=int, default=12, help="fixture files to index")
    parser.add_argument("--seconds", type=float, default=4.0, help="length of each fixture")
    parser.add_argument("--size", default="3840x2160", help="fixture size (landscape)")
    parser.add_argument("--segments", type=int, default=6, help="clip segments in the render")
    parser.add_argument("--project", type=Path, help="existing project root (skips fixtures)")
    parser.add_argument("--keep", action="store_true", help="keep the temporary project")
    args = parser.parse_args()

    owned = args.project is None
    root = Path(tempfile.mkdtemp(prefix="adanvil-bench-")) if owned else args.project.resolve()
    results: dict[str, float] = {}
    try:
        footage = root / "footage"
        if owned:
            footage.mkdir(parents=True)
            timed(
                "fixtures",
                results,
                lambda: [
                    make_fixture(footage / f"clip{i:02d}.mp4", i, args.seconds, args.size)
                    for i in range(args.files)
                ],
            )
        files = sorted(footage.glob("*.mp4"))
        print(
            f"{len(files)} files, workers={config.workers()}, "
            f"ffmpeg threads={config.ffmpeg_threads()}, cpus={os.cpu_count()}",
            flush=True,
        )

        shutil.rmtree(root / "cache", ignore_errors=True)
        (root / "catalog.json").unlink(missing_ok=True)
        timed("index", results, lambda: index_library(root, "footage"))
        timed("index_again", results, lambda: index_library(root, "footage"))

        shutil.rmtree(root / "cache" / "overview", ignore_errors=True)
        for thumbs in (root / "cache").glob("thumbs"):
            shutil.rmtree(thumbs, ignore_errors=True)
        timed("overview_page", results, lambda: library_overview(root, "footage", page=1))

        segments = []
        for i in range(args.segments):
            source = files[i % len(files)]
            clip = {"type": "clip", "source": f"footage/{source.name}", "in": 0.0, "out": 1.5}
            if i % 2:
                clip["zoom"] = {"from": 1.0, "to": 1.06}
            segments.append(clip)
        board = {
            "name": "bench",
            "format": {"width": 1080, "height": 1920, "fps": 30},
            "emoji": "none",
            "segments": segments,
            "overlays": [
                {
                    "type": "caption", "style": "pill", "lines": ["Bench"],
                    "start": 0, "end": 2, "y": 330,
                }
            ],
        }  # fmt: skip

        def render(quality: str) -> object:
            return compose(Storyboard.model_validate(board), root=root, quality=quality)

        mezz = ("base", "clip", "image", "concat-video")

        def wipe() -> None:
            for kind in mezz:
                shutil.rmtree(root / "cache" / kind, ignore_errors=True)

        wipe()
        timed("preview_cold", results, lambda: render("preview"))
        timed("final_after_preview", results, lambda: render("final"))
        wipe()
        timed("final_cold", results, lambda: render("final"))
        results["workers"] = config.workers()
        results["ffmpeg_threads"] = config.ffmpeg_threads()
        print(json.dumps(results, indent=2))
    finally:
        if owned and not args.keep:
            shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
