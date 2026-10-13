from __future__ import annotations

import json
import shutil
from pathlib import Path

import anyio
import pytest
from mcp import Client
from PIL import Image

from adanvil.brand import Claims, load_brand, save_brand
from adanvil.cli import main
from adanvil.endcard import compose_end_card, parse_spec, render_end_card
from adanvil.project import init_project
from adanvil.server import build_server

ACCENT = (37, 99, 235)  # the neutral default accent, #2563EB


def _spec(**changes) -> dict:
    spec = {
        "product_image": "footage/product.png",
        "pill": "OFFRE DU MOMENT",
        "title": [{"text": "Votre premier café est "}, {"text": "offert", "accent": True}],
        "subline": "avec votre Ember & Oak",
        "badges": [
            {"text": "Retours gratuits sous 30 jours", "icon": "check"},
            {"text": "4,8/5 sur plus de 1 800 avis", "icon": "star"},
        ],
        "button": {"text": "Lien juste en dessous", "arrow": "down"},
        "emoji": "none",
    }
    spec.update(changes)
    return spec


@pytest.fixture
def project(tmp_path: Path, media: dict[str, dict[str, object]]) -> Path:
    init_project(tmp_path, "Ember & Oak")
    shutil.copy2(str(media["product.png"]["path"]), tmp_path / "footage" / "product.png")
    shutil.copy2(str(media["ref_ad.mp4"]["path"]), tmp_path / "footage" / "ref_ad.mp4")
    return tmp_path


def _inside(box: list[int], width: int, height: int, margin: int = 40) -> bool:
    left, top, right, bottom = box
    return (
        left >= margin and top >= margin and right <= width - margin and bottom <= height - margin
    )


def test_spec_validation_explains_each_problem() -> None:
    parsed = parse_spec({"product_image": "a.png", "title": "Hello there"})
    assert [run.text for run in parsed.title] == ["Hello there"]
    assert parsed.badges == [] and parsed.button is None

    bad = {
        "product_image": "a.png",
        "title": "",
        "badges": [{"text": "x", "icon": "heart"}] * 4,
        "accent": "not-a-colour",
        "family": "comic-sans",
        "colour": "red",
    }
    with pytest.raises(ValueError) as error:
        parse_spec(bad)
    message = str(error.value)
    for fragment in ("badges", "accent", "family", "colour", "title"):
        assert fragment in message


def test_render_sizes_and_everything_inside_the_frame(project: Path) -> None:
    out = compose_end_card(project, _spec(), project / "brand" / "endcards" / "a.png")
    with Image.open(out) as image:
        assert image.size == (1080, 1920)
    layout = render_end_card(project, _spec()).layout
    stack = [layout["pill"], layout["title"], layout["photo_card"], layout["subline"]]
    stack += [*layout["badges"], layout["button"]]
    assert len(layout["badges"]) == 2
    assert all(_inside(box, 1080, 1920) for box in stack)
    assert layout["photo_scale"] <= 1.25
    # Blocks are stacked in order without overlapping, and centred on the frame.
    assert all(a[3] <= b[1] for a, b in zip(stack, stack[1:], strict=False))
    for box in stack:
        assert abs((box[0] + box[2]) / 2 - 540) <= 2
    # The same space above the first block and below the last one.
    assert abs(layout["pill"][1] - (1920 - layout["button"][3])) <= 4


def test_other_format_scales_the_layout(project: Path) -> None:
    layout = render_end_card(project, _spec(format={"width": 1080, "height": 1080})).layout
    assert (layout["width"], layout["height"]) == (1080, 1080)
    assert layout["button"][3] <= 1080 - 40
    assert layout["pill"][1] >= 40


def test_long_title_wraps_to_two_lines_and_shrinks(project: Path) -> None:
    short = render_end_card(project, _spec(title="Offre limitée")).layout
    assert len(short["title_lines"]) == 1
    long_title = "Livraison offerte sur toute votre première commande Brio aujourd hui seulement"
    layout = render_end_card(project, _spec(title=long_title)).layout
    assert len(layout["title_lines"]) == 2
    assert layout["title_size"] < short["title_size"]
    assert layout["title"][0] >= 60 and layout["title"][2] <= 1020
    assert layout["title_overflow"] is False


def test_photo_is_never_upscaled_beyond_the_cap(project: Path) -> None:
    Image.new("RGB", (120, 90), (30, 90, 160)).save(project / "footage" / "tiny.png")
    layout = render_end_card(project, _spec(product_image="footage/tiny.png")).layout
    assert layout["photo_scale"] == 1.25
    assert _inside(layout["photo_card"], 1080, 1920)


def test_text_is_optically_centred_in_the_button(project: Path) -> None:
    rendered = render_end_card(project, _spec(button={"text": "Commander", "arrow": "none"}))
    x0, y0, x1, y1 = rendered.layout["button"]
    inner = rendered.image.convert("RGB").crop((x0 + 40, y0 + 8, x1 - 40, y1 - 8))
    # White text on the accent button: the green channel is high only on the text.
    bbox = inner.getchannel("G").point(lambda value: 255 if value > 200 else 0).getbbox()
    assert bbox is not None
    centre_x = x0 + 40 + (bbox[0] + bbox[2]) / 2
    centre_y = y0 + 8 + (bbox[1] + bbox[3]) / 2
    assert abs(centre_x - (x0 + x1) / 2) <= 3
    assert abs(centre_y - (y0 + y1) / 2) <= 5


def test_brand_defaults_and_overrides(project: Path) -> None:
    rendered = render_end_card(project, _spec())
    x0, y0, _, y1 = rendered.layout["button"]
    probe = (x0 + 12, (y0 + y1) // 2)
    assert rendered.image.convert("RGB").getpixel(probe) == ACCENT

    brand = load_brand(project / "brand.yaml")
    brand.visual.accent_color = "#1166CC"
    save_brand(brand, project / "brand.yaml")
    blue = render_end_card(project, _spec())
    assert blue.image.convert("RGB").getpixel(probe) == (17, 102, 204)
    forced = render_end_card(project, _spec(accent="#00AA00"))
    assert forced.image.convert("RGB").getpixel(probe) == (0, 170, 0)


def test_default_accent_is_neutral_and_a_brand_colour_wins(project: Path) -> None:
    from adanvil.brand import default_brand
    from adanvil.endcard import DEFAULT_ACCENT

    assert DEFAULT_ACCENT == "#2563EB"
    visual = default_brand("Acme").visual
    assert visual is not None
    assert visual.accent_color == visual.primary_color == "#2563EB"

    # Without a brand profile the end card falls back to the same neutral accent.
    (project / "brand.yaml").unlink()
    rendered = render_end_card(project, _spec())
    x0, y0, _, y1 = rendered.layout["button"]
    assert rendered.image.convert("RGB").getpixel((x0 + 12, (y0 + y1) // 2)) == ACCENT


def test_banned_words_of_the_brand_are_an_error(project: Path) -> None:
    brand = load_brand(project / "brand.yaml")
    brand.claims = Claims(banned_words=["miracle"])
    save_brand(brand, project / "brand.yaml")
    with pytest.raises(ValueError, match="Banned words.*miracle"):
        render_end_card(project, _spec(subline="Un résultat miracle"))
    render_end_card(project, _spec())


def test_product_can_be_a_video_frame_and_inputs_are_checked(project: Path) -> None:
    crop = {"x": 0, "y": 0, "w": 200, "h": 300}
    layout = render_end_card(project, _spec(product_image="footage/ref_ad.mp4", t=0.5, crop=crop))
    assert layout.layout["photo_card"]
    with pytest.raises(ValueError, match="exceeds the image"):
        render_end_card(project, _spec(crop={"x": 1000, "y": 0, "w": 500, "h": 300}))
    with pytest.raises(ValueError, match="does not exist"):
        render_end_card(project, _spec(product_image="footage/missing.png"))
    with pytest.raises(ValueError, match="escapes"):
        render_end_card(project, _spec(product_image="../x.png"))


def test_emoji_in_texts_is_supported(project: Path) -> None:
    folder = project / "emoji"
    folder.mkdir()
    Image.new("RGBA", (72, 72), (255, 120, 0, 255)).save(folder / "1f525.png")
    spec = _spec(emoji="emoji", pill="Offre \U0001f525", badges=["Top \U0001f525"])
    layout = render_end_card(project, spec).layout
    assert layout["pill"] and len(layout["badges"]) == 1


def test_mcp_tool_returns_path_layout_and_image(project: Path) -> None:
    async def run() -> None:
        async with Client(build_server()) as client:
            result = await client.call_tool(
                "end_card", {"project_dir": str(project), "spec": _spec(), "name": "offer"}
            )
            assert not result.is_error
            assert any(item.type == "image" for item in result.content)
            payload = json.loads([i.text for i in result.content if i.type == "text"][0])
            assert payload["path"] == "brand/endcards/offer.png"
            assert payload["button"] and payload["photo_card"]
            bad = await client.call_tool(
                "end_card",
                {"project_dir": str(project), "spec": {"product_image": "x.png"}, "name": "bad"},
            )
            assert bad.is_error
            assert "title" in bad.content[0].text
            wrong_name = await client.call_tool(
                "end_card", {"project_dir": str(project), "spec": _spec(), "name": "../x"}
            )
            assert wrong_name.is_error

    anyio.run(run)
    assert (project / "brand" / "endcards" / "offer.png").is_file()


def test_cli_end_card(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    spec_path = project / "spec.json"
    spec_path.write_text(json.dumps(_spec()), encoding="utf-8")
    code = main(["end-card", str(spec_path), "--project", str(project), "--name", "cli", "--json"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["path"] == "brand/endcards/cli.png"
    assert (project / "brand" / "endcards" / "cli.png").is_file()
    assert main(["end-card", "spec.json", "--project", str(project), "--name", "again"]) == 0
    spec_path.write_text(json.dumps({"product_image": "x"}), encoding="utf-8")
    assert main(["end-card", str(spec_path), "--project", str(project), "--name", "no"]) == 1
