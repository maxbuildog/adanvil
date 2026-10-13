from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from PIL import Image

from adanvil.ffmpeg import (
    FFmpegError,
    detect_cuts,
    extract_frame,
    measure_yavg,
    probe,
    run,
)


def test_probe_matches_manifest(media: dict[str, dict[str, object]]) -> None:
    for name, expected in media.items():
        info = probe(Path(str(expected["path"])))
        if name.endswith(".png"):
            assert info.kind == "image"
        elif name == "music.m4a":
            assert info.kind == "audio"
            assert info.has_audio
        elif name.endswith(".mp4"):
            assert info.kind == "video"
            if "width" in expected:
                assert info.width == expected["width"]
                assert info.height == expected["height"]
            else:
                assert info.stored_width == expected["stored_width"]
                assert info.stored_height == expected["stored_height"]
            assert info.duration == pytest.approx(float(expected["duration"]), abs=0.05)
            assert info.fps == pytest.approx(float(expected["fps"]))
            assert info.has_audio is bool(expected["has_audio"])
            if "color_transfer" in expected:
                assert info.color_transfer == expected["color_transfer"]
        if name == "rotated.mp4":
            assert (info.width, info.height) == (1080, 1920)
            assert info.rotation in {90, 270}
        if name == "hlg_portrait.mp4":
            assert info.is_hdr


def test_probe_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        probe(tmp_path / "missing.mp4")


def test_probe_invalid_media_raises(tmp_path: Path) -> None:
    path = tmp_path / "not-media.mp4"
    path.write_text("this is not a media file", encoding="utf-8")
    with pytest.raises(FFmpegError, match=path.name):
        probe(path)


def test_run_timeout() -> None:
    started = time.monotonic()
    with pytest.raises(FFmpegError) as caught:
        run(
            [
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=1920x1080:rate=30",
                "-t",
                "600",
                "-f",
                "null",
                "-",
            ],
            timeout=1,
        )
    assert time.monotonic() - started < 10
    assert caught.value.returncode == -1


@pytest.mark.parametrize("returncode", [3221225794, 3221225477])
def test_windows_process_start_error_message(returncode: int) -> None:
    error = FFmpegError(["ffprobe.exe", "-version"], returncode, "")

    assert "Windows could not start ffmpeg/ffprobe (0xC0000142)." in str(error)
    assert "close other applications and retry" in str(error)
    assert "stderr tail" in str(error)
    assert "ffprobe.exe -version" in str(error)


def test_measure_yavg(media: dict[str, dict[str, object]]) -> None:
    assert measure_yavg(Path(str(media["dark_portrait.mp4"]["path"]))) < 60
    assert measure_yavg(Path(str(media["sdr_portrait.mp4"]["path"]))) > 60


def test_detect_cuts(media: dict[str, dict[str, object]]) -> None:
    actual = detect_cuts(Path(str(media["ref_ad.mp4"]["path"])))
    assert len(actual) == 2
    for found, expected in zip(actual, [2.0, 4.0], strict=True):
        assert found == pytest.approx(expected, abs=0.1)


def test_extract_frame(media: dict[str, dict[str, object]], tmp_path: Path) -> None:
    out = extract_frame(
        Path(str(media["sdr_landscape.mp4"]["path"])),
        1.0,
        tmp_path / "frame.png",
        width=320,
    )
    assert out == tmp_path / "frame.png"
    with Image.open(out) as image:
        assert image.format == "PNG"
        assert image.size == (320, 180)


def test_media_info_is_json_serialisable(media: dict[str, dict[str, object]]) -> None:
    info = probe(Path(str(media["sdr_portrait.mp4"]["path"])))
    json.dumps(info.model_dump(mode="json"))
