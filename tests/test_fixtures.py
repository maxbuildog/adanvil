from __future__ import annotations

import json
import re
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from adanvil._bin import ffmpeg_path, ffprobe_path
from tests.fixtures.make import build

VIDEO_NAMES = (
    "sdr_landscape.mp4",
    "sdr_portrait.mp4",
    "hlg_portrait.mp4",
    "dark_portrait.mp4",
    "rotated.mp4",
    "burnt_text.mp4",
    "ref_ad.mp4",
)


def _probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            ffprobe_path(),
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _video_stream(probe: dict[str, Any]) -> dict[str, Any]:
    return next(stream for stream in probe["streams"] if stream.get("codec_type") == "video")


def _duration(probe: dict[str, Any], stream: dict[str, Any]) -> float:
    return float(stream.get("duration") or probe["format"]["duration"])


def _fps(stream: dict[str, Any]) -> float:
    return float(Fraction(stream["avg_frame_rate"]))


def _rotation(stream: dict[str, Any]) -> float:
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            return float(side_data["rotation"])
    rotate = stream.get("tags", {}).get("rotate")
    if rotate is not None:
        return float(rotate)
    raise AssertionError("video stream has no reported rotation")


def _assert_video_properties(properties: dict[str, Any], path: Path) -> dict[str, Any]:
    probe = _probe(path)
    stream = _video_stream(probe)
    assert stream["width"] == properties.get("width", properties.get("stored_width"))
    assert stream["height"] == properties.get("height", properties.get("stored_height"))
    assert abs(_duration(probe, stream) - properties["duration"]) <= 0.05
    assert _fps(stream) == properties["fps"]
    if "pix_fmt" in properties:
        assert stream["pix_fmt"] == properties["pix_fmt"]
    if "color_transfer" in properties:
        assert stream["color_transfer"] == properties["color_transfer"]
    has_audio = any(item.get("codec_type") == "audio" for item in probe["streams"])
    assert has_audio is properties["has_audio"]
    if has_audio:
        audio = next(item for item in probe["streams"] if item.get("codec_type") == "audio")
        assert audio["sample_rate"] == "48000"
        assert audio["channels"] == 2
    if "rotation" in properties:
        actual = _rotation(stream)
        assert abs(actual) == 90
        assert actual * properties["rotation"] > 0
    return probe


def _signalstats_yavg(path: Path) -> float:
    result = subprocess.run(
        [
            ffmpeg_path(),
            "-nostdin",
            "-loglevel",
            "info",
            "-i",
            str(path),
            "-vf",
            "signalstats,metadata=mode=print:file=-",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    match = re.search(r"lavfi\.signalstats\.YAVG=([0-9]+(?:\.[0-9]+)?)", output)
    if match is None:
        raise AssertionError(f"signalstats did not report YAVG:\n{output}")
    return float(match.group(1))


def _scene_cut_times(path: Path) -> list[float]:
    result = subprocess.run(
        [
            ffmpeg_path(),
            "-nostdin",
            "-loglevel",
            "info",
            "-i",
            str(path),
            "-vf",
            "select='gt(scene,0.3)',showinfo",
            "-an",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [float(value) for value in re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", result.stderr)]


def _gray_frame(path: Path, timestamp: str) -> Image.Image:
    result = subprocess.run(
        [
            ffmpeg_path(),
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            timestamp,
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return Image.frombytes("L", (1080, 1920), result.stdout)


def test_video_files_match_manifest(media: dict[str, dict[str, object]]) -> None:
    for name in VIDEO_NAMES:
        properties = media[name]
        _assert_video_properties(properties, Path(str(properties["path"])))


def test_dark_video_is_dark(media: dict[str, dict[str, object]]) -> None:
    properties = media["dark_portrait.mp4"]
    assert _signalstats_yavg(Path(str(properties["path"]))) < properties["expected_max_yavg"]


def test_reference_ad_has_two_hard_cuts(media: dict[str, dict[str, object]]) -> None:
    properties = media["ref_ad.mp4"]
    actual = _scene_cut_times(Path(str(properties["path"])))
    expected = properties["cuts"]
    assert len(actual) == 2
    for found, wanted in zip(actual, expected, strict=True):
        assert abs(found - wanted) <= 0.1


def test_burnt_text_is_brighter_than_background(media: dict[str, dict[str, object]]) -> None:
    properties = media["burnt_text.mp4"]
    frame = _gray_frame(Path(str(properties["path"])), "1.0")
    bbox = tuple(properties["text_bbox"])
    text_mean = ImageStat.Stat(frame.crop(bbox)).mean[0]
    x0, y0, x1, y1 = bbox
    comparison = frame.crop((x0, 1200, x1, 1200 + y1 - y0))
    comparison_mean = ImageStat.Stat(comparison).mean[0]
    assert text_mean > comparison_mean + 20


def test_png_sizes(media: dict[str, dict[str, object]]) -> None:
    for name in ("product.png", "cta.png"):
        properties = media[name]
        probe = _probe(Path(str(properties["path"])))
        stream = probe["streams"][0]
        assert stream["width"] == properties["width"]
        assert stream["height"] == properties["height"]
        with Image.open(Path(str(properties["path"]))) as image:
            assert image.size == (properties["width"], properties["height"])


def test_audio_files_match_manifest(media: dict[str, dict[str, object]]) -> None:
    for name in ("music.m4a", "music_short.m4a"):
        properties = media[name]
        probe = _probe(Path(str(properties["path"])))
        audio = next(item for item in probe["streams"] if item.get("codec_type") == "audio")
        duration = float(audio.get("duration") or probe["format"]["duration"])
        assert abs(duration - properties["duration"]) <= 0.05
        assert audio["sample_rate"] == "48000"
        assert audio["channels"] == 2


def test_build_is_idempotent(media: dict[str, dict[str, object]]) -> None:
    out_dir = Path(str(media["product.png"]["path"])).parent
    before = {
        name: (out_dir / name).stat().st_mtime_ns
        for name in media
    }
    second = build(out_dir)
    after = {name: (out_dir / name).stat().st_mtime_ns for name in media}
    assert second == json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert before == after
