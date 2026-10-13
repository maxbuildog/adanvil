"""V2.8b: text scale per format and captions kept off the content of a still."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image, ImageDraw

from adanvil.busy import free_bands, overlaps, picture_busy
from adanvil.formats import (
    FORMATS,
    MIN_TEXT_SIZE,
    card_content_clashes,
    reformat,
    text_scale,
)
from adanvil.project import init_project
from adanvil.qc.score import _card_content_findings
from adanvil.storyboard import Storyboard
from adanvil.storyboard.craft import Checks
from adanvil.storyboard.validate import validate_storyboard
from adanvil.text.styles import default_size


def _card(path: Path) -> None:
    """A 9:16 end card: a gradient, a title block and a yellow button, nothing else."""

    image = Image.new("RGB", (1080, 1920))
    draw = ImageDraw.Draw(image)
    for y in range(1920):
        draw.line([(0, y), (1080, y)], fill=(40 + y // 30, 30 + y // 45, 110 + y // 25))
    draw.rectangle((240, 620, 840, 720), fill=(250, 250, 250))
    draw.rounded_rectangle((330, 1390, 750, 1500), radius=55, fill=(246, 200, 70))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    init_project(root, "Card")
    _card(root / "footage" / "card.png")
    return root


def _document(y: int = 1380) -> dict[str, Any]:
    return {
        "name": "card",
        "format": {"width": 1080, "height": 1920, "fps": 10},
        "emoji": "none",
        "segments": [
            {"type": "image", "source": "footage/card.png", "duration": 3, "label": "end_card"}
        ],
        "overlays": [
            {
                "type": "caption",
                "style": "pill",
                "lines": ["Order today, free shipping"],
                "start": 0.2,
                "end": 2.8,
                "y": y,
            }
        ],
    }


def test_text_scale_formula() -> None:
    assert text_scale((1080, 1920)) == 1.0
    assert text_scale((1080, 1350)) == pytest.approx(1350 / 1920)  # same share of the height
    assert text_scale((1080, 1080)) == pytest.approx(0.8 * 1080 / 1920)  # 0.8 of that share
    assert text_scale((1080, 2400)) == 1.0  # never enlarged
    # a 4:5 source going to 1:1 is measured against its own frame
    assert text_scale((1080, 1080), (1080, 1350)) == pytest.approx(0.8 * 1080 / 1350)


@pytest.mark.parametrize("ratio", ["4:5", "1:1"])
def test_text_is_scaled_and_never_below_the_readable_minimum(ratio: str) -> None:
    document = _document()
    document["overlays"][0]["size"] = 100
    document["overlays"].append(
        {"type": "caption", "style": "pill", "lines": ["small"], "size": 30, "y": 400}
    )
    document["overlays"].append(
        {"type": "caption", "style": "pill", "lines": ["default size"], "y": 500}
    )
    moved, _ = reformat(document, ratio)
    factor = text_scale((1080, FORMATS[ratio].height))
    assert moved["overlays"][0]["size"] == max(MIN_TEXT_SIZE, round(100 * factor))
    assert moved["overlays"][1]["size"] == 30  # already under the minimum: not touched
    assert moved["overlays"][2]["size"] == max(MIN_TEXT_SIZE, round(default_size("pill") * factor))
    assert reformat(document, "9:16")[0]["overlays"][0]["size"] == 100


def test_stickers_and_cards_scale_too() -> None:
    document = _document()
    document["overlays"] = [
        {"type": "sticker", "kind": "badge", "text": "NEW", "x": 100, "y": 300},
        {"type": "sticker", "kind": "circle", "x": 500, "y": 500, "width": 400, "stroke": 12},
        {
            "type": "image",
            "source": "footage/card.png",
            "x": 100,
            "y": 900,
            "width": 400,
            "rotate": 3,
        },
    ]
    moved, _ = reformat(document, "1:1")
    factor = text_scale((1080, 1080))
    badge, circle, card = moved["overlays"]
    assert badge["size"] == round(44 * factor)
    assert circle["width"] == round(400 * factor) and circle["stroke"] == round(12 * factor)
    assert card["width"] == round(400 * factor) and card["x"] > 100


def test_busy_region_finds_the_title_and_the_button(tmp_path: Path) -> None:
    root = _project(tmp_path)
    rect, busy = picture_busy(root / "footage" / "card.png", (1080, 1350))
    assert rect[1] == 0 and rect[3] == 1350  # contain: the card fills the height
    scale = 1350 / 1920
    title = (240 * scale + rect[0], 620 * scale, 840 * scale + rect[0], 720 * scale)
    button = (330 * scale + rect[0], 1390 * scale, 750 * scale + rect[0], 1500 * scale)
    assert overlaps(title, busy) and overlaps(button, busy)
    empty = (300.0, 600.0, 800.0, 900.0)
    assert not overlaps(empty, busy)
    bands = free_bands(busy, 0, 1350)
    assert bands[0][1] - bands[0][0] > 300  # the largest empty band


@pytest.mark.parametrize("ratio", ["4:5", "1:1"])
def test_caption_over_a_card_is_moved_off_its_content(tmp_path: Path, ratio: str) -> None:
    root = _project(tmp_path)
    spec = FORMATS[ratio]
    moved, notes = reformat(_document(), ratio, root=root)
    assert card_content_clashes(moved, root) == []
    assert any("out of the content" in note for note in notes)
    assert not any("caption_over_card_content" in note for note in notes)
    # without the picture check the caption sits on the button, which is the defect
    plain, _ = reformat(_document(), ratio)
    assert card_content_clashes(plain, root)
    assert 0 <= moved["overlays"][0]["y"] < spec.height


def test_a_card_with_no_room_shrinks_then_warns(tmp_path: Path) -> None:
    root = _project(tmp_path)
    busy_card = Image.new("RGB", (1080, 1920))
    draw = ImageDraw.Draw(busy_card)
    for y in range(0, 1920, 40):  # stripes everywhere: no free band at all
        draw.rectangle((0, y, 1080, y + 20), fill=(250, 250, 250) if y % 80 else (10, 10, 10))
    busy_card.save(root / "footage" / "card.png")
    _moved, notes = reformat(_document(), "1:1", root=root)
    assert any(note.startswith("caption_over_card_content") for note in notes)


def test_validation_and_score_report_a_caption_over_the_card(tmp_path: Path) -> None:
    root = _project(tmp_path)
    document = _document()
    report = validate_storyboard(document, root=root)
    codes = [issue.code for issue in report.warnings]
    assert "caption_over_card_content" in codes
    clear = validate_storyboard(_document(y=900), root=root)
    assert "caption_over_card_content" not in [issue.code for issue in clear.warnings]
    checks = Checks()
    context = SimpleNamespace(model=Storyboard.model_validate(document), root=root)
    _card_content_findings(context, checks)  # type: ignore[arg-type]
    assert [finding.id for finding in checks.findings] == ["caption_over_card_content"]
    assert checks.findings[0].area == "safe_zones"
