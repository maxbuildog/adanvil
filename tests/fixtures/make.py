"""Build deterministic synthetic media used by AdAnvil's test suite."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from adanvil._bin import ffmpeg_path, ffprobe_path

WIDTH = 1080
HEIGHT = 1920
FPS = 30


def _run_ffmpeg(*args: str | Path) -> None:
    command = [
        ffmpeg_path(),
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        *(str(arg) for arg in args),
    ]
    subprocess.run(command, check=True)


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


def _rotation(path: Path) -> int:
    data = _probe(path)
    stream = next(stream for stream in data["streams"] if stream.get("codec_type") == "video")
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            return int(round(float(side_data["rotation"])))
    rotate = stream.get("tags", {}).get("rotate")
    if rotate is not None:
        return int(round(float(rotate)))
    raise RuntimeError(f"ffprobe did not report display rotation for {path}")


def _write_text_overlay(path: Path) -> list[int]:
    image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=72)
    text = "BURNT TEXT\nLINE TWO"
    stroke_width = 4
    spacing = 8
    bbox = draw.multiline_textbbox(
        (0, 0),
        text,
        font=font,
        spacing=spacing,
        stroke_width=stroke_width,
    )
    text_width = bbox[2] - bbox[0]
    x = (WIDTH - text_width) // 2 - bbox[0]
    y = 300 - bbox[1]
    draw.multiline_text(
        (x, y),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        spacing=spacing,
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0, 255),
    )
    alpha_bbox = image.getchannel("A").getbbox()
    if alpha_bbox is None:
        raise RuntimeError("text overlay unexpectedly has no non-transparent pixels")
    image.save(path)
    return list(alpha_bbox)


def _write_product(path: Path) -> None:
    image = Image.new("RGB", (1080, 1350), (246, 242, 233))
    draw = ImageDraw.Draw(image)
    draw.ellipse((180, 180, 900, 900), fill=(222, 235, 224), outline=(47, 91, 67), width=8)
    draw.rounded_rectangle(
        (315, 280, 765, 1160),
        radius=95,
        fill=(72, 137, 108),
        outline=(29, 67, 48),
        width=10,
    )
    draw.rounded_rectangle((425, 195, 655, 330), radius=32, fill=(29, 67, 48))
    draw.rectangle((375, 610, 705, 830), fill=(246, 242, 233))
    image.save(path)


def _write_cta(path: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), (35, 55, 91))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=120)
    text = "SHOP NOW"
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (WIDTH - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (HEIGHT - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=(255, 224, 112))
    image.save(path)


def _video_common(*, size: str, duration: str, output: Path, audio: bool = False) -> None:
    args: list[str | Path] = [
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={size}:rate={FPS}",
    ]
    if audio:
        args.extend(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"])
    args.extend(["-t", duration, "-map", "0:v:0"])
    if audio:
        args.extend(["-map", "1:a:0"])
    args.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-x264-params",
            "colorprim=bt709:transfer=bt709:colormatrix=bt709",
        ]
    )
    if audio:
        args.extend(["-c:a", "aac", "-ar", "48000", "-ac", "2", "-shortest"])
    else:
        args.append("-an")
    args.append(output)
    _run_ffmpeg(*args)


def _write_music(path: Path, duration: str) -> None:
    filter_graph = (
        "[0:a][1:a][2:a]amix=inputs=3:duration=longest:normalize=0,"
        "volume=0.25,tremolo=f=2:d=0.6[a]"
    )
    _run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=220:sample_rate=48000:duration={duration}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=277.18:sample_rate=48000:duration={duration}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=329.63:sample_rate=48000:duration={duration}",
        "-filter_complex",
        filter_graph,
        "-map",
        "[a]",
        "-t",
        duration,
        "-c:a",
        "aac",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-preset",
        "ultrafast",
        path,
    )


def _write_burnt_text(path: Path, overlay: Path) -> None:
    _run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={WIDTH}x{HEIGHT}:rate={FPS}",
        "-loop",
        "1",
        "-framerate",
        str(FPS),
        "-i",
        overlay,
        "-filter_complex",
        "[0:v]gblur=sigma=20,eq=brightness=-0.5[bg];"
        "[bg][1:v]overlay=0:0:format=auto[v]",
        "-map",
        "[v]",
        "-t",
        "3",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-x264-params",
        "colorprim=bt709:transfer=bt709:colormatrix=bt709",
        "-an",
        path,
    )


def _write_ref_ad(path: Path, overlay: Path, music: Path) -> None:
    _run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={WIDTH}x{HEIGHT}:rate={FPS}:duration=2",
        "-f",
        "lavfi",
        "-i",
        f"color=c=0x3060C0:size={WIDTH}x{HEIGHT}:rate={FPS}:duration=2",
        "-f",
        "lavfi",
        "-i",
        f"smptebars=size={WIDTH}x{HEIGHT}:rate={FPS}:duration=2",
        "-loop",
        "1",
        "-framerate",
        str(FPS),
        "-i",
        overlay,
        "-i",
        music,
        "-filter_complex",
        "[0:v][1:v][2:v]concat=n=3:v=1:a=0[shots];"
        "[shots][3:v]overlay=0:0:format=auto[v]",
        "-map",
        "[v]",
        "-map",
        "4:a:0",
        "-t",
        "6",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-x264-params",
        "colorprim=bt709:transfer=bt709:colormatrix=bt709",
        "-c:a",
        "aac",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-af",
        "atrim=duration=6,asetpts=PTS-STARTPTS",
        path,
    )


def _write_hlg(path: Path) -> None:
    _run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={WIDTH}x{HEIGHT}:rate={FPS}",
        "-t",
        "2",
        "-c:v",
        "libx264",
        "-profile:v",
        "high10",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p10le",
        "-colorspace",
        "bt2020nc",
        "-color_primaries",
        "bt2020",
        "-color_trc",
        "arib-std-b67",
        "-x264-params",
        "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc",
        "-an",
        path,
    )


def _write_dark(path: Path) -> None:
    _run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={WIDTH}x{HEIGHT}:rate={FPS}",
        "-vf",
        "format=yuv420p,lutyuv=y=val*0.2",  # underexposed but above video black
        "-t",
        "2",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-x264-params",
        "colorprim=bt709:transfer=bt709:colormatrix=bt709",
        "-an",
        path,
    )


def _write_rotated(path: Path) -> int:
    source = path.with_name("_rotated_source.mp4")
    try:
        _video_common(size="1920x1080", duration="3", output=source)
        _run_ffmpeg(
            "-display_rotation",
            "90",
            "-i",
            source,
            "-c",
            "copy",
            "-preset",
            "ultrafast",
            path,
        )
    finally:
        source.unlink(missing_ok=True)
    return _rotation(path)


def build(out_dir: Path, force: bool = False) -> dict[str, dict[str, Any]]:
    """Build all synthetic media, returning the JSON-compatible manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all((out_dir / name).exists() for name in existing):
            return existing

    overlay = out_dir / "_burnt_text.png"
    text_bbox = _write_text_overlay(overlay)
    _write_product(out_dir / "product.png")
    _write_cta(out_dir / "cta.png")
    _write_music(out_dir / "music.m4a", "8")
    _write_music(out_dir / "music_short.m4a", "2.5")
    _video_common(size="1920x1080", duration="3", output=out_dir / "sdr_landscape.mp4", audio=True)
    _video_common(size="1080x1920", duration="3", output=out_dir / "sdr_portrait.mp4")
    _write_hlg(out_dir / "hlg_portrait.mp4")
    _write_dark(out_dir / "dark_portrait.mp4")
    rotation = _write_rotated(out_dir / "rotated.mp4")
    _write_burnt_text(out_dir / "burnt_text.mp4", overlay)
    _write_ref_ad(out_dir / "ref_ad.mp4", overlay, out_dir / "music.m4a")

    manifest: dict[str, dict[str, Any]] = {
        "sdr_landscape.mp4": {
            "width": 1920,
            "height": 1080,
            "duration": 3.0,
            "fps": 30,
            "has_audio": True,
            "color_transfer": "bt709",
        },
        "sdr_portrait.mp4": {
            "width": 1080,
            "height": 1920,
            "duration": 3.0,
            "fps": 30,
            "has_audio": False,
            "color_transfer": "bt709",
        },
        "hlg_portrait.mp4": {
            "width": 1080,
            "height": 1920,
            "duration": 2.0,
            "fps": 30,
            "pix_fmt": "yuv420p10le",
            "color_transfer": "arib-std-b67",
            "has_audio": False,
        },
        "dark_portrait.mp4": {
            "width": 1080,
            "height": 1920,
            "duration": 2.0,
            "fps": 30,
            "has_audio": False,
            "color_transfer": "bt709",
            "expected_max_yavg": 60,
        },
        "rotated.mp4": {
            "stored_width": 1920,
            "stored_height": 1080,
            "duration": 3.0,
            "fps": 30,
            "has_audio": False,
            "rotation": rotation,
        },
        "burnt_text.mp4": {
            "width": WIDTH,
            "height": HEIGHT,
            "duration": 3.0,
            "fps": FPS,
            "has_audio": False,
            "color_transfer": "bt709",
            "text_bbox": text_bbox,
        },
        "ref_ad.mp4": {
            "width": WIDTH,
            "height": HEIGHT,
            "duration": 6.0,
            "fps": FPS,
            "has_audio": True,
            "color_transfer": "bt709",
            "cuts": [2.0, 4.0],
            "text_bbox": text_bbox,
        },
        "music.m4a": {"duration": 8.0},
        "music_short.m4a": {"duration": 2.5},
        "product.png": {"width": 1080, "height": 1350},
        "cta.png": {"width": WIDTH, "height": HEIGHT},
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    overlay.unlink()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures/out"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.out, force=args.force)


if __name__ == "__main__":
    main()
