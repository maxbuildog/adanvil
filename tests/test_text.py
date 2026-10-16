from __future__ import annotations

import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image

from adanvil import netguard
from adanvil.text import TextSpec, get_provider, load_font, render
from adanvil.text.emoji import (
    DirectoryProvider,
    TwemojiProvider,
    cluster_codepoints,
    split_runs,
)
from adanvil.text.layout import balance_lines, fit_size, wrap


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    path = Path.cwd() / ".adanvil-text-assets"
    path.mkdir(exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
    try:
        yield path
    finally:
        for child in path.iterdir():
            if child.is_file():
                child.unlink()


def _emoji_dir(tmp_path: Path) -> DirectoryProvider:
    colours = {
        "1f440": (220, 40, 40, 255),
        "2728": (40, 180, 60, 255),
        "2705": (40, 80, 220, 255),
        "1f469-200d-1f9b0": (220, 150, 30, 255),
    }
    for name, colour in colours.items():
        Image.new("RGBA", (72, 72), colour).save(tmp_path / f"{name}.png")
    return DirectoryProvider(tmp_path)


def _rgba_pixels(image: Image.Image) -> list[tuple[int, int, int, int]]:
    raw = image.tobytes()
    return [tuple(raw[index : index + 4]) for index in range(0, len(raw), 4)]  # type: ignore[misc]


def test_split_runs_plain_accents_and_typographic_punctuation() -> None:
    assert split_runs("Été « déjà »… l’été – oui") == [(False, "Été « déjà »… l’été – oui")]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("👀", [(True, "👀")]),
        ("✨️", [(True, "✨️")]),
        ("👩‍🦰", [(True, "👩‍🦰")]),
        ("👍🏽", [(True, "👍🏽")]),
        ("🇫🇷", [(True, "🇫🇷")]),
        ("1️⃣", [(True, "1️⃣")]),
    ],
)
def test_split_runs_emoji_clusters(value: str, expected: list[tuple[bool, str]]) -> None:
    assert split_runs(value) == expected


def test_cluster_codepoints() -> None:
    assert cluster_codepoints("👩‍🦰") == "1f469-200d-1f9b0"
    assert cluster_codepoints("✨️", strip_fe0f=True) == "2728"


def test_directory_provider_and_provider_factory(tmp_path: Path) -> None:
    provider = _emoji_dir(tmp_path)
    image = provider.get("👀", 36)
    assert image is not None and image.size == (36, 36)
    assert get_provider("none").get("👀", 36) is None
    assert get_provider("directory", path=tmp_path).get("✅", 36) is not None


def test_twemoji_strips_fe0f_and_warns_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def fail(url: str, destination: object, **_kw: object) -> None:
        calls.append(url)
        raise OSError("offline")

    monkeypatch.setattr(netguard, "fetch_to_file", fail)
    provider = TwemojiProvider(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert provider.get("✨️", 72) is None
        assert provider.get("✨️", 72) is None
    assert len(caught) == 1
    assert any("2728.png" in url for url in calls)


def test_measurement_and_fit() -> None:
    used, overflow = fit_size(
        ["un texte extrêmement long qui doit rétrécir"],
        "plus-jakarta-sans",
        700,
        60,
        max_width=900,
    )
    assert used < 60 and not overflow
    used, overflow = fit_size(["WWWWWWWWWWWW"], "montserrat", 700, 60, 50, min_size=28)
    assert used == 28 and overflow


def test_wrap_and_balance() -> None:
    font = load_font("plus-jakarta-sans", 700, 40)
    greedy = wrap("un texte français assez long pour plusieurs lignes", font, 330)
    balanced = balance_lines("un texte français assez long pour plusieurs lignes", 3, font)
    assert len(greedy) == 3
    assert len(balanced) == 3
    assert max(font.getlength(line) for line in balanced) <= max(
        font.getlength(line) for line in greedy
    )


def test_variable_font_weight_changes_width_and_unknown_family() -> None:
    regular = load_font("plus-jakarta-sans", 300, 60)
    bold = load_font("plus-jakarta-sans", 800, 60)
    assert regular.getlength("AdAnvil") != bold.getlength("AdAnvil")
    with pytest.raises(ValueError, match="plus-jakarta-sans"):
        load_font("not-a-family", 400, 40)


@pytest.mark.parametrize("style", ["pill", "meme", "outline", "outline_soft", "box", "label"])
def test_all_caption_styles_render_inside_canvas(style: str, tmp_path: Path) -> None:
    provider = _emoji_dir(tmp_path)
    result = render(TextSpec(style=style, lines=["C'est votre signe 👀✨"], y=500), provider)
    assert result.image.mode == "RGBA"
    assert result.image.size == (1080, 1920)
    assert result.bbox[0] >= 0 and result.bbox[1] >= 0
    assert result.bbox[2] <= 1080 and result.bbox[3] <= 1920
    assert result.bbox[2] - result.bbox[0] <= 1060


@pytest.mark.parametrize("look", ["facebook", "tiktok"])
def test_comment_styles_render_and_hide_author(look: str) -> None:
    result = render(
        TextSpec(
            style="comment",
            look=look,
            header="Réponse au commentaire d'un client",
            lines=["Réponse au commentaire d'un client"],
            y=900,
            reactions=12,
        ),
        get_provider("none"),
    )
    assert result.bbox[2] - result.bbox[0] <= 960
    assert result.image.getchannel("A").getbbox() == result.bbox
    assert result.image.size == (1080, 1920)


def test_anchor_bottom_and_pipe_validator() -> None:
    spec = TextSpec(style="pill", lines="première ligne|deuxième ligne", anchor="bottom", y=1000)
    assert spec.lines == ["première ligne", "deuxième ligne"]
    assert render(spec, get_provider("none")).bbox[3] == 1000


def test_pipe_validator_splits_each_list_item() -> None:
    spec = TextSpec(style="pill", lines=["a|b", "c"])
    assert spec.lines == ["a", "b", "c"]
    assert "|" not in render(spec, get_provider("none")).lines


def test_facebook_meta_is_inside_outer_card() -> None:
    result = render(
        TextSpec(style="comment", look="facebook", lines=["A comment"], y=500),
        get_provider("none"),
    )
    meta_colour = (101, 103, 107, 255)
    meta_pixels = [
        (x, y)
        for y in range(result.image.height)
        for x in range(result.image.width)
        if result.image.getpixel((x, y)) == meta_colour
    ]
    assert meta_pixels
    meta_bottom = max(y for _, y in meta_pixels)
    card_bottom = max(
        y
        for y in range(result.image.height)
        if result.image.getpixel((result.image.width // 2, y))[:3] == (255, 255, 255)
    )
    assert meta_bottom < card_bottom
    assert card_bottom - meta_bottom >= 20


def test_facebook_avatar_is_circular() -> None:
    result = render(
        TextSpec(style="comment", look="facebook", lines=["A comment"], y=500),
        get_provider("none"),
    )
    alpha = result.image.getchannel("A")
    opaque = alpha.load()
    # The avatar starts 18 px from the card's top-left, with the card inset by 8 px.
    left, top = result.bbox[0] + 26, result.bbox[1] + 26
    assert result.image.getpixel((left, top))[:3] == (255, 255, 255)
    assert result.image.getpixel((left + 87, top))[:3] == (255, 255, 255)
    assert opaque[left + 44, top + 44] > 0
    assert result.image.getpixel((left + 44, top + 44))[:3] != (255, 255, 255)


def test_facebook_comment_lines_are_left_aligned() -> None:
    result = render(
        TextSpec(
            style="comment",
            look="facebook",
            lines=["Short line has a much longer second line"],
            y=500,
            size=28,
            max_width=500,
        ),
        get_provider("none"),
    )
    pixels = result.image.load()
    bubble = [
        (x, y)
        for y in range(result.image.height)
        for x in range(result.image.width)
        if pixels[x, y][:3] == (240, 242, 245)
    ]
    bubble_left = min(x for x, _ in bubble)
    bubble_right = max(x for x, _ in bubble)
    bubble_top = min(y for _, y in bubble)
    bubble_bottom = max(y for _, y in bubble)
    text_rows = sorted(
        {
            y
            for y in range(bubble_top + 50, bubble_bottom)
            if any(
                pixels[x, y][3] > 0 and max(pixels[x, y][:3]) < 80
                for x in range(bubble_left + 10, bubble_right - 9)
            )
        }
    )
    groups: list[list[int]] = []
    for y in text_rows:
        if not groups or y > groups[-1][-1] + 1:
            groups.append([])
        groups[-1].append(y)
    starts = [
        min(
            x
            for y in group
            for x in range(bubble_left + 10, bubble_right - 9)
            if pixels[x, y][3] > 0 and max(pixels[x, y][:3]) < 80
        )
        for group in groups
    ]
    assert len(starts) >= 2
    assert max(starts) - min(starts) <= 2


def test_facebook_reactions_contain_white_icons() -> None:
    result = render(
        TextSpec(style="comment", look="facebook", lines=["A comment"], reactions=12, y=500),
        get_provider("none"),
    )
    blue = []
    red = []
    for y in range(result.image.height):
        for x in range(result.image.width):
            pixel = result.image.getpixel((x, y))[:3]
            if pixel == (24, 119, 242):
                blue.append((x, y))
            if pixel == (243, 62, 88):
                red.append((x, y))
    assert blue and red
    blue_box = (
        min(x for x, _ in blue),
        min(y for _, y in blue),
        max(x for x, _ in blue),
        max(y for _, y in blue),
    )
    red_box = (
        min(x for x, _ in red),
        min(y for _, y in red),
        max(x for x, _ in red),
        max(y for _, y in red),
    )
    assert any(
        result.image.getpixel((x, y))[:3] == (255, 255, 255)
        for x in range(blue_box[0], blue_box[2] + 1)
        for y in range(blue_box[1], blue_box[3] + 1)
    )
    assert any(
        result.image.getpixel((x, y))[:3] == (255, 255, 255)
        for x in range(red_box[0], red_box[2] + 1)
        for y in range(red_box[1], red_box[3] + 1)
    )


def test_pill_has_white_box_dark_text_and_emoji_pixels(tmp_path: Path) -> None:
    result = render(TextSpec(style="pill", lines=["Test 👀"], y=300), _emoji_dir(tmp_path))
    pixels = [pixel for pixel in _rgba_pixels(result.image) if pixel[3] > 0]
    assert any(pixel[:3] == (255, 255, 255) for pixel in pixels)
    assert any(max(pixel[:3]) < 50 for pixel in pixels)
    assert any(pixel[0] > 150 and pixel[1] < 100 and pixel[2] < 100 for pixel in pixels)


def test_outline_has_white_and_near_black_pixels() -> None:
    result = render(TextSpec(style="outline", lines=["Contraste"], y=300), get_provider("none"))
    pixels = [pixel for pixel in _rgba_pixels(result.image) if pixel[3] > 0]
    assert any(min(pixel[:3]) > 240 for pixel in pixels)
    assert any(max(pixel[:3]) < 40 for pixel in pixels)


_BOX_TEXTS = [
    "Brio : 49,90 € au lieu de 69,90 €",
    "Retours gratuits sous 30 jours",
    "Découvre la « carte du jour » 👇",
    "gjpqy Agy",
]


def _white_emoji_dir(tmp_path: Path) -> DirectoryProvider:
    Image.new("RGBA", (72, 72), (255, 255, 255, 255)).save(tmp_path / "1f447.png")
    return DirectoryProvider(tmp_path)


def _ink_margins(result, ink) -> tuple[int, int, int, int]:
    """Left, right, top, bottom gap between the box and the ink drawn inside it."""
    box = result.image.getchannel("A").getbbox()
    assert box is not None
    pixels = result.image.load()
    xs: list[int] = []
    ys: list[int] = []
    for y in range(box[1], box[3]):
        for x in range(box[0], box[2]):
            if ink(pixels[x, y]):
                xs.append(x)
                ys.append(y)
    assert xs
    return min(xs) - box[0], box[2] - 1 - max(xs), min(ys) - box[1], box[3] - 1 - max(ys)


@pytest.mark.parametrize("text", _BOX_TEXTS)
def test_box_text_is_optically_centred(text: str, tmp_path: Path) -> None:
    result = render(
        TextSpec(style="box", lines=[text], y=800), _white_emoji_dir(tmp_path)
    )
    left, right, top, bottom = _ink_margins(
        result, lambda p: p[3] > 200 and min(p[:3]) > 140
    )
    assert abs(left - right) <= 2, (left, right)
    assert abs(top - bottom) <= 3, (top, bottom)


def test_box_geometry_follows_the_font_size() -> None:
    result = render(TextSpec(style="box", lines=["Hello"], y=800, size=56), get_provider("none"))
    left, right, top, bottom = _ink_margins(result, lambda p: p[3] > 200 and min(p[:3]) > 140)
    assert 24 <= left <= 34
    assert 12 <= top <= 22


def test_text_only_captions_keep_the_same_baseline_from_chunk_to_chunk() -> None:
    from adanvil.text import render
    from adanvil.text.styles import TextSpec

    def ink_bottom(text: str, anchor: str) -> int:
        spec = TextSpec(style="outline", lines=[text], y=1290, anchor=anchor, size=56)
        return render(spec, None).bbox[3]

    for anchor in ("top", "bottom"):
        # No descender in either: the letters must sit on the same line, accent or not.
        bottoms = {ink_bottom(text, anchor) for text in ("TOUT", "ÂME", "MÂCHOIRE ET LE", "OK")}
        assert max(bottoms) - min(bottoms) <= 1, (anchor, bottoms)
