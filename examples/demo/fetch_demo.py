"""Create a complete AdAnvil demo project from the small Wikimedia pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

# ``python examples/demo/fetch_demo.py ...`` is intentionally supported from a
# checkout, where the ``src`` directory is not normally on sys.path yet.
if __package__ in {None, ""}:  # pragma: no cover - exercised by the CLI smoke test
    _src = Path(__file__).resolve().parents[2] / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from PIL import Image, ImageDraw, ImageFilter
from PIL.Image import UnidentifiedImageError

from adanvil import __version__
from adanvil._bin import ffmpeg_path
from adanvil.brand import Brand, Claims, EndCard, Product, Tone, Visual, save_brand
from adanvil.project import init_project
from adanvil.text.fonts import load_font

PACK_PATH = Path(__file__).with_name("pack.json")
PACK: list[dict[str, Any]] = json.loads(PACK_PATH.read_text(encoding="utf-8"))["files"]
# A named map keeps the tests able to substitute fixture hashes while still
# exercising the real streaming and verification code.
EXPECTED_SHA1 = {entry["target"]: entry["sha1"] for entry in PACK}
USER_AGENT = f"AdAnvil-demo/{__version__} (https://github.com/maxbuildog/adanvil)"
CANVAS = (1080, 1920)


class DemoPackError(RuntimeError):
    """An actionable error while assembling the demo project."""


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_sha1(entry: dict[str, Any]) -> str:
    return str(EXPECTED_SHA1.get(entry["target"], entry["sha1"]))


def download_file(entry: dict[str, Any], destination: Path, *, force: bool = False) -> bool:
    """Stream one manifest entry, returning whether a download was performed."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = _expected_sha1(entry)
    if target.is_file() and not force and _sha1(target) == expected:
        return False

    partial = target.with_name(target.name + ".part")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(str(entry["url"]), headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise DemoPackError(
            f"Could not download {entry['target']}: {exc}. "
            "Check your network connection and retry the demo command."
        ) from exc

    actual = _sha1(partial)
    if actual != expected:
        partial.unlink(missing_ok=True)
        raise DemoPackError(
            f"SHA-1 mismatch for {entry['target']}: expected {expected}, got {actual}. "
            "The download was discarded; retry the demo command."
        )

    # os.replace is atomic on the same filesystem and replaces a stale target
    # only after the new bytes have passed verification.
    os.replace(partial, target)
    return True


def _run_ffmpeg(*args: str | Path) -> None:
    command = [ffmpeg_path(), "-nostdin", "-y", "-loglevel", "error", *(str(arg) for arg in args)]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DemoPackError(
            f"Could not create the synthetic demo music: {exc}. "
            "Install ffmpeg and retry the demo command."
        ) from exc


def write_synthetic_music(path: Path) -> None:
    """Write a deterministic, 24-second four-chord AAC loop with ffmpeg."""

    path.parent.mkdir(parents=True, exist_ok=True)
    chords = (
        (261.63, 329.63, 392.00, 523.25),  # I: C
        (196.00, 246.94, 392.00, 493.88),  # V: G
        (220.00, 261.63, 329.63, 440.00),  # vi: Am
        (174.61, 220.00, 261.63, 349.23),  # IV: F
    )
    args: list[str | Path] = []
    for chord in chords:
        for frequency in chord:
            args.extend(
                [
                    "-f",
                    "lavfi",
                    "-i",
                    f"sine=frequency={frequency}:sample_rate=48000:duration=24",
                ]
            )

    graph: list[str] = []
    chord_windows = ((0, 6), (6, 12), (12, 18), (18, 24))
    for index, window in enumerate(chord_windows):
        start, end = window
        for note in range(4):
            stream = index * 4 + note
            graph.append(
                f"[{stream}:a]volume=0.075:enable='between(t\\,{start}\\,{end})'[n{stream}]"
            )
    graph.append(
        "".join(f"[n{index}]" for index in range(16))
        + "amix=inputs=16:duration=longest:normalize=0,lowpass=f=2400,"
        "afade=t=in:st=0:d=0.45,afade=t=out:st=23.55:d=0.45[chords]"
    )
    graph.append(
        "[16:a]highpass=f=35,volume=0.45[kick]"
    )
    graph.append(
        "[17:a]highpass=f=6000,volume='if(between(mod(t\\,0.666667)\\,0.333333\\,0.45)\\,1\\,0)'[hat]"
    )
    graph.append(
        "[chords][kick][hat]amix=inputs=3:duration=longest:normalize=0,"
        "volume=0.8,alimiter=limit=0.9[mix]"
    )
    args.extend(
        [
            "-f",
            "lavfi",
            "-i",
            "aevalsrc=0.22*sin(2*PI*72*t)*exp(-28*mod(t\\,0.666667))*"
            "if(lt(mod(t\\,0.666667)\\,0.12)\\,1\\,0):s=48000:d=24",
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=color=white:amplitude=0.07:duration=24:sample_rate=48000:seed=42",
            "-filter_complex",
            ";".join(graph),
            "-map",
            "[mix]",
            "-t",
            "24",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ar",
            "48000",
            "-ac",
            "2",
            path,
        ]
    )
    _run_ffmpeg(*args)


def _placeholder_jar() -> Image.Image:
    image = Image.new("RGBA", (700, 700), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((100, 90, 600, 600), fill="#D9B28C", outline="#8B5A3C", width=12)
    draw.rounded_rectangle(
        (150, 180, 550, 570), radius=55, fill="#C08457", outline="#8B5A3C", width=12
    )
    draw.rounded_rectangle((220, 125, 480, 230), radius=25, fill="#6F452F")
    return image


def _contain(image: Image.Image, box: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(box, Image.Resampling.LANCZOS)
    return copy


def _centered(draw: ImageDraw.ImageDraw, text: str, y: int, font: Any, fill: str) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (CANVAS[0] - (bbox[2] - bbox[0])) // 2 - bbox[0]
    draw.text((x, y), text, font=font, fill=fill)


def write_end_card(project_root: Path) -> None:
    """Create the portrait Ember & Oak end card from the downloaded jar."""

    destination = project_root / "brand" / "end_card.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(project_root / "brand" / "jar.jpg") as source:
            jar = source.convert("RGBA")
    except (FileNotFoundError, OSError, UnidentifiedImageError):
        # A placeholder keeps the offline fixture test and a partially fetched
        # demo usable; a real pack always uses the verified Commons image.
        jar = _placeholder_jar()

    canvas = Image.new("RGB", CANVAS, "#2B1D14")
    draw = ImageDraw.Draw(canvas)
    name_font = load_font("plus-jakarta-sans", 800, 110)
    tagline_font = load_font("plus-jakarta-sans", 500, 60)
    cta_font = load_font("plus-jakarta-sans", 700, 48)
    _centered(draw, "Ember & Oak", 170, name_font, "#F5E6D3")
    _centered(draw, "Hand-poured soy candles", 365, tagline_font, "#F5E6D3")

    jar = _contain(jar, (720, 600))
    x = (CANVAS[0] - jar.width) // 2
    y = 790 + (600 - jar.height) // 2
    shadow = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.ellipse(
        (x + 35, y + jar.height - 20, x + jar.width - 35, y + jar.height + 45),
        fill=(0, 0, 0, 135),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(28))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), shadow)
    canvas.alpha_composite(jar, (x, y))

    draw = ImageDraw.Draw(canvas)
    label = "Shop now"
    bbox = draw.textbbox((0, 0), label, font=cta_font)
    pill_width = bbox[2] - bbox[0] + 120
    pill_height = 118
    pill_x = (CANVAS[0] - pill_width) // 2
    pill_y = 1450
    draw.rounded_rectangle(
        (pill_x, pill_y, pill_x + pill_width, pill_y + pill_height),
        radius=pill_height // 2,
        fill="#E07A3F",
    )
    text_x = (CANVAS[0] - (bbox[2] - bbox[0])) // 2 - bbox[0]
    text_y = pill_y + (pill_height - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((text_x, text_y), label, font=cta_font, fill="white")
    canvas.convert("RGB").save(destination, format="PNG")


def write_brand(project_root: Path) -> None:
    save_brand(
        Brand(
            name="Ember & Oak",
            website="https://example.com",
            language="en",
            markets=["US", "GB"],
            niche="home fragrance, candles",
            tone=Tone(address="you", voice="warm, cosy, honest"),
            product=Product(
                name="Ember & Oak soy candle",
                offer=None,
                guarantee=None,
                cta_text="Shop now",
            ),
            claims=Claims(
                banned_words=["toxic-free", "detox", "cure", "therapeutic"],
                guidance="Describe scent, mood and craft. No health or wellness claims.",
            ),
            visual=Visual(
                primary_color="#2B1D14",
                accent_color="#E07A3F",
                caption_family="plus-jakarta-sans",
                caption_weight=700,
                emoji="twemoji",
            ),
            end_card=EndCard(source="brand/end_card.png", duration=3.0),
        ),
        project_root / "brand.yaml",
    )


def build_demo(project_root: Path, *, music: str = "synthetic", force: bool = False) -> Path:
    """Build the demo project in *project_root* and return its resolved path."""

    root = Path(project_root).resolve()
    init_project(root, "Ember & Oak demo")
    for entry in PACK:
        destination = root / Path(str(entry["target"]))
        downloaded = download_file(entry, destination, force=force)
        print(f"{'Downloaded' if downloaded else 'Verified'} {entry['target']}")

    write_brand(root)
    write_end_card(root)
    if music == "synthetic":
        write_synthetic_music(root / "music" / "demo_loop.m4a")

    print(f"\nDemo ready: {root}")
    print("Next steps:")
    print(f"  adanvil index {root} --source footage")
    print("  adanvil recipes text_hook")
    print("  Write a storyboard using the text_hook slots, then:")
    print(f"  adanvil validate {root} storyboard.json --recipe text_hook")
    print(f"  adanvil render {root} storyboard.json")
    if music == "synthetic":
        print("  You may replace music/demo_loop.m4a with any track you are licensed to use.")
    print("\nLicence summary (CC0; attribution is appreciated but not required):")
    for entry in PACK:
        print(f"  {entry['target']} - {entry['license']} - {entry['page']}")
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo_dir", type=Path, metavar="DEMO_DIR")
    parser.add_argument("--music", choices=("synthetic", "none"), default="synthetic")
    parser.add_argument("--force", action="store_true", help="redownload even verified files")
    args = parser.parse_args(argv)
    build_demo(args.demo_dir, music=args.music, force=args.force)
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except DemoPackError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
