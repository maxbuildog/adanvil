from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from adanvil import config
from adanvil.library import index_library, load_catalog
from adanvil.media.sheet import media_mosaic


def test_workers_and_threads_follow_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADANVIL_WORKERS", "3")
    monkeypatch.setenv("ADANVIL_FFMPEG_THREADS", "5")
    assert config.workers() == 3
    assert config.ffmpeg_threads() == 5
    assert config.thread_args() == ["-threads", "5"]
    monkeypatch.setenv("ADANVIL_WORKERS", "nonsense")
    monkeypatch.delenv("ADANVIL_FFMPEG_THREADS")
    assert 1 <= config.workers() <= 4
    assert config.ffmpeg_threads(1) >= 1


def test_default_workers_stay_safe_on_a_small_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ADANVIL_WORKERS", raising=False)
    monkeypatch.setattr(config.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(config, "total_ram_bytes", lambda: 8 * 1024**3)
    assert config.workers() == 2
    monkeypatch.setattr(config, "total_ram_bytes", lambda: 22 * 1024**3)
    assert config.workers() == 4
    monkeypatch.setattr(config, "total_ram_bytes", lambda: None)
    assert config.workers() == 2


def test_parallel_index_matches_sequential(
    media: dict[str, dict[str, object]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = ["sdr_portrait.mp4", "hlg_portrait.mp4", "burnt_text.mp4", "cta.png"]
    snapshots = []
    for workers in ("1", "3"):
        monkeypatch.setenv("ADANVIL_WORKERS", workers)
        root = tmp_path / f"w{workers}"
        (root / "footage").mkdir(parents=True)
        for name in names:
            shutil.copy2(Path(str(media[name]["path"])), root / "footage" / name)
        result = index_library(root, "footage")
        assert not result.errors
        entries = load_catalog(root).entries
        snapshots.append(
            {
                key: entry.model_dump(exclude={"indexed", "file_key"})
                for key, entry in entries.items()
            }
        )
        assert sorted(result.added) == sorted(entries)
    assert snapshots[0] == snapshots[1]


def test_mosaic_thumbnails_are_cached_by_file(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    clip = tmp_path / "clip.mp4"
    shutil.copy2(Path(str(media["sdr_portrait.mp4"]["path"])), clip)
    cache = tmp_path / "thumbs"
    first = media_mosaic([clip, clip], tmp_path / "a.jpg", cache_dir=cache)
    stored = sorted(cache.glob("*.png"))
    assert len(stored) == 1
    stamp = stored[0].stat().st_mtime_ns
    second = media_mosaic([clip], tmp_path / "b.jpg", cache_dir=cache)
    assert first.is_file() and second.is_file()
    assert sorted(cache.glob("*.png")) == stored
    assert stored[0].stat().st_mtime_ns == stamp


def _big_clip(path: Path, keyint: int) -> Path:
    from adanvil.ffmpeg import run

    run(
        [
            "-f", "lavfi", "-i", "testsrc2=s=2048x1152:r=30:d=2",
            "-c:v", "libx264", "-preset", "ultrafast", "-g", str(keyint), "-pix_fmt", "yuv420p",
            str(path),
        ]
    )  # fmt: skip
    return path


def test_keyframe_pass_matches_the_separate_passes(tmp_path: Path) -> None:
    import numpy as np

    from adanvil import ffmpeg
    from adanvil.library.onepass import keyframe_pass
    from adanvil.media.sheet import _keyframe_thumbnails
    from adanvil.textremoval import sample_frames

    clip = _big_clip(tmp_path / "big.mp4", 15)
    info = ffmpeg.probe(clip)
    cuts, yavg, decoded = ffmpeg.measure_video_keyframes(clip, threshold=0.3)
    assert decoded >= 3
    shared = keyframe_pass(
        clip,
        info,
        threshold=0.3,
        thumb_dir=tmp_path / "shared",
        thumb_width=270,
        text_seconds=min(float(info.duration or 0), 6.0),
    )
    assert (shared.cuts, shared.yavg, shared.decoded) == (cuts, yavg, decoded)
    (tmp_path / "alone").mkdir()
    thumbs = _keyframe_thumbnails(clip, tmp_path / "alone", 270)
    assert [t for t, _ in shared.thumbs] == [t for t, _ in thumbs]
    assert all(
        a.read_bytes() == b.read_bytes()
        for (_, a), (_, b) in zip(shared.thumbs, thumbs, strict=True)
    )
    frames = sample_frames(
        clip, 0.0, min(float(info.duration or 0), 6.0), fps=2, long_side=1280, keyframes_only=True
    )
    assert shared.text_frames is not None
    assert np.array_equal(frames, shared.text_frames)


@pytest.mark.parametrize("keyint", [15, 999])
def test_big_sources_index_through_the_shared_pass_or_its_fallback(
    tmp_path: Path, keyint: int
) -> None:
    root = tmp_path / "project"
    (root / "footage").mkdir(parents=True)
    _big_clip(root / "footage" / "big.mp4", keyint)
    result = index_library(root, "footage")
    assert not result.errors
    entry = next(iter(load_catalog(root).entries.values()))
    assert entry.shots and entry.sheet
    assert (root / entry.sheet).is_file()
    assert not list((root / "cache" / "sheets").glob(".adanvil-keyframes-*"))
