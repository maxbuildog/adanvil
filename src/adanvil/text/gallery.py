from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from .emoji import DirectoryProvider, NoEmojiProvider, TwemojiProvider
from .styles import TextSpec, render

CANVAS = (1080, 1920)


def _default_background() -> Image.Image:
    image = Image.new("RGB", CANVAS)
    draw = ImageDraw.Draw(image)
    top = (76, 76, 80)
    bottom = (158, 158, 164)
    for y in range(CANVAS[1]):
        ratio = y / max(1, CANVAS[1] - 1)
        draw.line(
            (0, y, CANVAS[0], y),
            fill=tuple(round(top[i] * (1 - ratio) + bottom[i] * ratio) for i in range(3)),
        )
    return image.convert("RGBA")


def _background(path: Path | None) -> Image.Image:
    if path is None:
        return _default_background()
    with Image.open(path) as source:
        image = source.convert("RGBA")
    return image.resize(CANVAS, Image.Resampling.LANCZOS)


def _provider(value: str):
    if value == "none":
        return NoEmojiProvider()
    if value == "twemoji":
        return TwemojiProvider()
    return DirectoryProvider(Path(value))


def _samples() -> list[tuple[str, TextSpec]]:
    return [
        (
            "pill",
            TextSpec(style="pill", lines=["C'est votre signe d'essayer 👀✨"], y=280),
        ),
        (
            "meme",
            TextSpec(style="meme", lines=["Retours gratuits|30 JOURS ✅"], y=820),
        ),
        (
            "outline",
            TextSpec(style="outline", lines=["Essayez, vous verrez ✨"], y=350),
        ),
        (
            "outline-soft",
            TextSpec(style="outline_soft", lines=["Un matin plus simple, enfin"], y=570),
        ),
        (
            "box",
            TextSpec(style="box", lines=["Essayez aujourd'hui", "sans engagement"], y=880),
        ),
        (
            "label",
            TextSpec(style="label", lines=["Une offre pensée pour vous"], y=1180),
        ),
        (
            "comment-facebook",
            TextSpec(
                style="comment",
                look="facebook",
                lines=["Réponse au commentaire d'un client"],
                reactions=12,
                y=1420,
            ),
        ),
        (
            "comment-tiktok",
            TextSpec(
                style="comment",
                look="tiktok",
                header="Réponse au commentaire d'un client",
                lines=["Oui, il reste chaud toute la journée ✅"],
                y=1420,
            ),
        ),
        (
            "pill-uniform",
            TextSpec(
                style="pill",
                uniform_width=True,
                lines=["Retours gratuits", "30 JOURS ✅"],
                y=1040,
            ),
        ),
        (
            "overflow",
            TextSpec(
                style="pill",
                lines=["a very very long line that must shrink to fit inside the frame"],
                y=1740,
                max_width=620,
            ),
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render an AdAnvil text-style gallery")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--background", type=Path)
    parser.add_argument("--emoji", default="twemoji", help="twemoji, none, or a PNG directory")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    background = _background(args.background)
    provider = _provider(args.emoji)
    rendered: list[Image.Image] = []
    for name, spec in _samples():
        result = render(spec, provider, CANVAS)
        composited = background.copy()
        composited.alpha_composite(result.image)
        rendered.append(composited)
        composited.convert("RGB").save(args.out_dir / f"{name}.jpg", quality=95)

    thumb_size = (270, 480)
    columns = 4
    rows = (len(rendered) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * thumb_size[1]), (35, 35, 38))
    for index, image in enumerate(rendered):
        sheet.paste(
            image.convert("RGB").resize(thumb_size, Image.Resampling.LANCZOS),
            ((index % columns) * 270, (index // columns) * 480),
        )
    sheet.save(args.out_dir / "gallery.jpg", quality=95)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
