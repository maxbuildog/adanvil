"""V2.4b fixes found by a full ad: emphasis on numbers and prices, snapped storyboards that keep
their own name and render path, the badge default colour, one emotion vocabulary, and text boxes
that are not text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import pytest
from mcp import Client

from adanvil import ffmpeg
from adanvil.brand import brand_accent_color, load_brand, save_brand
from adanvil.library import Annotation, annotate, index_library, load_catalog
from adanvil.library import search as library_search
from adanvil.media import ClipSegment
from adanvil.project import init_project
from adanvil.reference.breakdown import BreakdownEntry
from adanvil.render import compose
from adanvil.render.stickers import BADGE_FALLBACK, sticker_art
from adanvil.server import build_server, render_caption_preview
from adanvil.storyboard import CaptionOverlay, StickerOverlay, Storyboard, validate_storyboard
from adanvil.storyboard.patch import follow_rename, rename_storyboard, snapped_name
from adanvil.text import TextSpec, get_provider, spoken_words
from adanvil.text.animate import marked_entries, strip_markers
from adanvil.text.chunking import emphasis_key, match_emphasis
from adanvil.textremoval import core as textremoval_core
from adanvil.textremoval import detect_text_boxes
from adanvil.vocab import EMOTION_VALUES, check_emotions
from tests.test_animated_captions import GREEN, _green_columns, _project
from tests.test_sound_design import _image_ad, _snap_project
from tests.test_visual_grammar import _board as _grid_board
from tests.test_visual_grammar import _frames, _near
from tests.test_visual_grammar import _project as _grid_project

NONE = get_provider("none")


# ------------------------------------------------------------ 1. emphasis on numbers and prices


def test_emphasis_key_normalises_prices_and_rates() -> None:
    assert emphasis_key("65,90 €") == emphasis_key("65.90€") == emphasis_key("65,90€.") == "65.90€"
    assert emphasis_key("-20%") == emphasis_key("- 20 %,") == "-20%"
    assert emphasis_key("–20%") == "-20%"  # an en dash used as a minus
    assert emphasis_key("Éclat!") == "eclat"
    assert emphasis_key("$49") == "$49"
    assert emphasis_key("à-la-carte") == "a-la-carte"  # inner hyphens stay


def test_match_emphasis_finds_numbers_prices_and_phrases() -> None:
    words = ["Only", "65,90 €", "today,", "-20%", "livraison", "offerte.", "l'éclat"]
    assert match_emphasis(words, ["65,90"]) == ({1}, [])
    assert match_emphasis(words, ["65,90 €"]) == ({1}, [])
    assert match_emphasis(words, ["65.90€"]) == ({1}, [])
    assert match_emphasis(words, ["€"]) == (set(), ["€"])  # a lone sign matches no word
    assert match_emphasis(words, ["-20%"]) == ({3}, [])
    assert match_emphasis(words, ["20%"]) == (set(), ["20%"])  # the sign matters
    assert match_emphasis(words, ["Livraison offerte"]) == ({4, 5}, [])
    assert match_emphasis(words, ["offerte livraison"]) == (set(), ["offerte livraison"])
    assert match_emphasis(words, ["éclat"]) == ({6}, [])  # a part of an elided word
    assert match_emphasis(words, ["65", "gratuit"]) == (set(), ["65", "gratuit"])
    assert match_emphasis(words, ["", "   "]) == (set(), [])  # nothing to look for: ignored


def test_strip_markers_works_around_prices_and_numbers() -> None:
    assert strip_markers("now *65,90 €* only") == ("now 65,90 € only", {1, 2})
    assert strip_markers("*-20%* today") == ("-20% today", {0})
    assert strip_markers("only *49€*, really") == ("only 49€, really", {1})
    assert strip_markers("only *€ 49* now") == ("only € 49 now", {1, 2})
    assert strip_markers("*65,90 €*") == ("65,90 €", {0, 1})
    assert strip_markers("*Livraison offerte*.") == ("Livraison offerte.", {0, 1})
    assert strip_markers("49€* only *terms") == ("49€* only *terms", set())


def test_marked_entries_keep_a_price_in_one_entry() -> None:
    text, marked = strip_markers("only *65,90 €* and *fast*")
    assert marked_entries(text.split(), marked) == ["65,90 €", "fast"]
    caption = CaptionOverlay(style="outline", lines=["Now *65,90 €* only", "*-20%* today"])
    assert caption.lines == ["Now 65,90 € only", "-20% today"]
    assert caption.emphasis == ["65,90 €", "-20%"]
    assert not any("*" in line for line in caption.lines)
    assert caption.unmatched_emphasis() == []


def test_a_price_is_drawn_in_the_highlight_colour() -> None:
    lines = ["Now only 65,90 € today"]
    base = {"style": "outline", "lines": lines, "y": 500, "highlight": GREEN}
    assert _green_columns(TextSpec(**base)) is None
    same = {
        _green_columns(TextSpec(**base, emphasis=[entry]))
        for entry in ("65,90", "65,90 €", "65.90€", "65,90€")
    }
    assert len(same) == 1 and None not in same
    marked = CaptionOverlay(
        style="outline", lines=["Now only *65,90 €* today"], y=500, highlight=GREEN
    )
    assert _green_columns(marked.to_text_spec()) == same.pop()  # the stars are never drawn
    percent = _green_columns(
        TextSpec(
            style="outline", lines=["Save -20% now"], y=500, highlight=GREEN, emphasis=["-20%"]
        )
    )
    assert percent is not None


def test_emphasis_unmatched_warning_in_validate(tmp_path: Path) -> None:
    root = _project(tmp_path)
    raw: dict[str, Any] = {
        "name": "emph",
        "segments": [{"type": "clip", "source": "footage/grey.mp4", "in": 0, "out": 3}],
        "overlays": [
            {
                "type": "caption",
                "style": "pill",
                "lines": ["Only 65,90 € today"],
                "start": 0,
                "end": 2,
                "emphasis": ["65,90", "99,90", "today"],
            },
            {"type": "caption", "style": "pill", "lines": ["Only *65,90 €* today"], "start": 2},
        ],
    }
    report = validate_storyboard(raw, root=root)
    warned = [w for w in report.warnings if w.code == "emphasis_unmatched"]
    assert len(warned) == 1
    assert "'99,90'" in warned[0].message and warned[0].where == "overlays[0].emphasis"
    raw["overlays"][0]["emphasis"] = ["65,90 €", "today"]
    again = validate_storyboard(raw, root=root)
    assert not [w for w in again.warnings if w.code == "emphasis_unmatched"]


def test_emphasis_unmatched_in_caption_preview(tmp_path: Path) -> None:
    root = _project(tmp_path)
    overlay = {
        "type": "caption",
        "style": "outline",
        "lines": ["Only 65,90 € today"],
        "emphasis": ["65,90", "nothing"],
    }
    payload, _ = render_caption_preview(root, "footage/grey.mp4", 0.5, overlay)
    assert payload["emphasis_unmatched"] == ["nothing"]
    assert "emphasis_unmatched" in payload["warnings"][0]
    overlay["emphasis"] = ["65,90"]
    clean, _ = render_caption_preview(root, "footage/grey.mp4", 0.5, overlay)
    assert "emphasis_unmatched" not in clean


def test_spoken_words_group_a_price_with_its_sign() -> None:
    assert spoken_words(["Only 65,90 € today"]) == ["Only", "65,90 €", "today"]


# ------------------------------------------------------ 2. snapped storyboards keep their own name


def test_rename_helpers() -> None:
    document: dict[str, Any] = {
        "name": "ad",
        "output": {"path": "renders/ad.mp4", "quality": "preview"},
    }
    rename_storyboard(document, "ad_b")
    assert document == {
        "name": "ad_b",
        "output": {"path": "renders/ad_b.mp4", "quality": "preview"},
    }
    bare = rename_storyboard({"name": "ad"}, "ad_b")
    assert bare["output"] == {"path": "renders/ad_b.mp4"}
    assert snapped_name("ad") == "ad_snapped" and len(snapped_name("x" * 80)) == 80
    assert snapped_name(None) == "storyboard_snapped"
    # A path chosen by hand is not touched by a rename.
    for path, moved in (
        ("renders/ad.mp4", True),
        ("renders/{name}.mp4", True),
        ("renders/custom.mp4", False),
    ):
        before = {"name": "ad"}
        after = {"name": "ad2", "output": {"path": path, "quality": "final"}}
        assert follow_rename(before, after) is moved
        assert after["output"]["path"] == ("renders/ad2.mp4" if moved else path)
    unchanged = {"name": "ad", "output": {"path": "renders/ad.mp4"}}
    assert follow_rename({"name": "ad"}, unchanged) is False
    assert follow_rename({"name": "ad"}, {"name": "ad2"}) is False  # no output: the template


def _text(result: Any) -> dict[str, Any]:
    return json.loads([item.text for item in result.content if item.type == "text"][0])


def test_storyboard_snap_gives_the_new_version_its_own_name_and_output(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _snap_project(tmp_path, media)
    raw = _image_ad([1.62, 1.45, 1.1, 1.83])
    raw["output"] = {"path": "renders/snap.mp4", "quality": "preview"}

    async def run() -> None:
        async with Client(build_server()) as client:
            args = {"project_dir": str(root), "storyboard": raw}
            saved = _text(await client.call_tool("storyboard_snap", {**args, "save_as": "snap_b"}))
            assert saved["saved"] == "storyboards/snap_b.json" and "storyboard" not in saved
            on_disk = json.loads((root / saved["saved"]).read_text("utf-8"))
            assert on_disk["name"] == "snap_b"
            assert on_disk["output"] == {"path": "renders/snap_b.mp4", "quality": "preview"}
            assert Storyboard.model_validate(on_disk).output.path == "renders/snap_b.mp4"

            inline = _text(await client.call_tool("storyboard_snap", args))
            board = inline["storyboard"]
            assert board["name"] == "snap_snapped"
            assert board["output"] == {"path": "renders/snap_snapped.mp4", "quality": "preview"}

            # With no output at all, the render path still follows the new name.
            plain = {k: v for k, v in raw.items() if k != "output"}
            other = _text(
                await client.call_tool(
                    "storyboard_snap", {"project_dir": str(root), "storyboard": plain}
                )
            )
            assert other["storyboard"]["output"]["path"] == "renders/snap_snapped.mp4"

    anyio.run(run)
    assert raw["name"] == "snap" and raw["output"]["path"] == "renders/snap.mp4"


def test_storyboard_patch_renaming_moves_the_default_output_path(tmp_path: Path) -> None:
    root = _project(tmp_path)
    base = {
        "version": 1,
        "name": "first",
        "segments": [{"type": "clip", "source": "footage/grey.mp4", "in": 0, "out": 2}],
    }

    async def run() -> None:
        async with Client(build_server()) as client:
            async def patch(name: str, changes: list[dict[str, Any]]) -> dict[str, Any]:
                result = _text(
                    await client.call_tool(
                        "storyboard_patch",
                        {"project_dir": str(root), "name": name, "changes": changes},
                    )
                )
                return json.loads((root / result["path"]).read_text("utf-8"))

            await client.call_tool(
                "storyboard_save",
                {
                    "project_dir": str(root),
                    "name": "a",
                    "storyboard": {
                        **base,
                        "output": {"path": "renders/first.mp4", "quality": "preview"},
                    },
                },
            )
            renamed = await patch("a", [{"path": "name", "value": "second"}])
            assert renamed["name"] == "second"
            assert renamed["output"] == {"path": "renders/second.mp4", "quality": "preview"}

            await client.call_tool(
                "storyboard_save",
                {
                    "project_dir": str(root),
                    "name": "b",
                    "storyboard": {**base, "output": {"path": "renders/hand_made.mp4"}},
                },
            )
            kept = await patch("b", [{"path": "name", "value": "second"}])
            assert kept["output"]["path"] == "renders/hand_made.mp4"

            await client.call_tool(
                "storyboard_save", {"project_dir": str(root), "name": "c", "storyboard": base}
            )
            template = await patch("c", [{"path": "name", "value": "third"}])
            assert "output" not in template  # the default template follows the name by itself

            untouched = await patch("a", [{"path": "segments.0.out", "value": 1.5}])
            assert untouched["output"]["path"] == "renders/second.mp4"

    anyio.run(run)


# --------------------------------------------------------------------- 3. badge default colour


def _badge(**extra: Any) -> StickerOverlay:
    return StickerOverlay.model_validate(
        {"type": "sticker", "kind": "badge", "text": "NEW", "x": 20, "y": 20, "size": 40, **extra}
    )


def _fill(art: Any) -> tuple[int, int, int, int]:
    """The most common fully opaque colour of the sticker: the badge body."""

    pixels = np.asarray(art.image).reshape(-1, 4)
    opaque = pixels[pixels[:, 3] == 255]
    colours, counts = np.unique(opaque, axis=0, return_counts=True)
    return tuple(int(v) for v in colours[counts.argmax()])  # type: ignore[return-value]


def test_badge_default_colour_is_neutral_dark_with_white_text() -> None:
    art = sticker_art(_badge())
    assert BADGE_FALLBACK == "#111827"
    assert _fill(art)[:3] == (0x11, 0x18, 0x27)
    pixels = np.asarray(art.image)
    assert (pixels[..., :3].min(axis=2) > 240).any()  # white text
    # The highlight accent (yellow, red...) no longer paints a badge.
    assert _fill(sticker_art(_badge(), accent="#FFD400"))[:3] == (0x11, 0x18, 0x27)


def test_badge_uses_the_brand_accent_colour_then_its_own_colour() -> None:
    branded = sticker_art(_badge(), badge_color="#2563EB")
    assert _fill(branded)[:3] == (0x25, 0x63, 0xEB)
    own = sticker_art(_badge(color="#00AA00"), badge_color="#2563EB")
    assert _fill(own)[:3] == (0x00, 0xAA, 0x00)
    # Other stickers keep the highlight colour.
    arrow = StickerOverlay.model_validate(
        {"type": "sticker", "kind": "underline", "x": 10, "y": 50, "width": 100}
    )
    pixels = np.asarray(sticker_art(arrow, accent="#FFD400", badge_color="#2563EB").image)
    assert (pixels[..., 0] > 240).any() and (pixels[..., 2] < 40).any()


def test_brand_accent_colour_is_read_from_the_brand_profile(tmp_path: Path) -> None:
    assert brand_accent_color(tmp_path) is None  # no profile
    init_project(tmp_path, "Sunbrew kettles")
    brand = load_brand(tmp_path / "brand.yaml")
    brand.visual.accent_color = None
    save_brand(brand, tmp_path / "brand.yaml")
    assert brand_accent_color(tmp_path) is None
    brand.visual.accent_color = "#2563EB"
    brand.visual.highlight_color = "#FFD400"
    save_brand(brand, tmp_path / "brand.yaml")
    assert brand_accent_color(tmp_path) == "#2563EB"


def test_a_rendered_badge_follows_the_brand_accent_and_is_never_red(tmp_path: Path) -> None:
    root = _grid_project(tmp_path)
    overlays = [
        {
            "type": "sticker", "kind": "badge", "text": "NEW", "x": 20, "y": 30, "size": 28,
            "start": 0.0, "end": 1.0,
        }
    ]  # fmt: skip
    segments = [{"type": "clip", "source": "footage/grey.mp4", "in": 0, "out": 2}]
    brand = load_brand(root / "brand.yaml")
    brand.visual.accent_color = None  # a new project starts with a blue accent
    save_brand(brand, root / "brand.yaml")
    dark = _frames(compose(_grid_board(segments, overlays), root=root).path)
    assert int(_near(dark[10], (0x11, 0x18, 0x27), 14).sum()) > 300
    assert int(_near(dark[10], (255, 0, 0), 60).sum()) == 0
    brand.visual.accent_color = "#2563EB"
    save_brand(brand, root / "brand.yaml")
    blue = _frames(compose(_grid_board(segments, overlays), root=root).path)
    assert int(_near(blue[10], (0x25, 0x63, 0xEB), 20).sum()) > 300
    assert int(_near(blue[10], (0x11, 0x18, 0x27), 14).sum()) == 0


# ----------------------------------------------------------------- 4. one emotion vocabulary


def test_every_emotion_entry_point_accepts_and_rejects_the_same_values(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    clip = {"source": "footage/a.mp4", "in": 0, "out": 1}
    checks = {
        "annotation": lambda value: Annotation(emotion=[value]),
        "annotation_str": lambda value: Annotation(emotion=value),
        "filter": lambda value: check_emotions([value]),
        "segment": lambda value: ClipSegment(**clip, emotion=value),
        "breakdown": lambda value: BreakdownEntry(shots=[1], emotion=value),
    }
    assert len(EMOTION_VALUES) == 13
    for value in (*EMOTION_VALUES, "smug", "Happy", "", "sadness"):
        outcomes = {}
        for name, build in checks.items():
            try:
                build(value)
                outcomes[name] = True
            except ValueError:  # pydantic's ValidationError is one
                outcomes[name] = False
        assert len(set(outcomes.values())) == 1, (value, outcomes)
        assert all(outcomes.values()) == (value in EMOTION_VALUES), value


def test_library_search_refuses_an_unknown_emotion_like_annotate(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _snap_project(tmp_path, media)
    with pytest.raises(ValueError, match="emotion must be one of"):
        library_search(root, emotion=["smug"])
    assert library_search(root, emotion=["tired", "closed", "sad"]) == []
    entry_id = next(iter(load_catalog(root).entries), None)
    if entry_id is not None:  # the project of the helper may have no catalogue yet
        with pytest.raises(ValueError):
            annotate(root, entry_id, emotion=["smug"])


def test_emotion_lists_in_the_docs_are_the_whole_vocabulary() -> None:
    repo = Path(__file__).resolve().parents[1]
    for name in (
        "AGENTS.md",
        "skill/SKILL.md",
        "docs/EDITING_GUIDE.md",
        "src/adanvil/assets/EDITING_GUIDE.md",
    ):
        text = (repo / name).read_text(encoding="utf-8")
        listed = [line for line in text.splitlines() if "`none`, `neutral`, `worried`" in line]
        assert listed, name
        for line in listed:
            for value in EMOTION_VALUES:
                assert f"`{value}`" in line, (name, value)


# ------------------------------------------------------- 5. text boxes that are not text


def _patch_frames(monkeypatch: pytest.MonkeyPatch, frames: np.ndarray) -> None:
    monkeypatch.setattr(textremoval_core, "sample_frames", lambda *args, **kwargs: frames)


def _text_like(frames: int, size: tuple[int, int] = (400, 300)) -> np.ndarray:
    """A row of hollow letter-like glyphs: static, bright, wide: burnt-in text."""

    width, height = size
    shots = np.zeros((frames, height, width, 3), dtype=np.uint8)
    for left in range(60, 340, 22):
        shots[:, 40:64, left : left + 16, :] = 255
        shots[:, 46:58, left + 4 : left + 12, :] = 0
    return shots


def test_a_text_band_is_still_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_frames(monkeypatch, _text_like(4))
    boxes = detect_text_boxes("synthetic.mp4")
    assert len(boxes) == 1
    x0, y0, x1, y1 = boxes[0]
    assert x0 <= 60 and x1 >= 340 and y0 <= 40 and y1 >= 70


def test_a_full_frame_box_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    white = np.full((4, 300, 400, 3), 255, dtype=np.uint8)  # a white wall over the whole frame
    _patch_frames(monkeypatch, white)
    assert detect_text_boxes("synthetic.mp4") == []
    # A bright block over most of the area (here 70 %) is a background too, text or not.
    wide = np.zeros((4, 300, 400, 3), dtype=np.uint8)
    wide[:, 40:260, 20:380, :] = 255
    _patch_frames(monkeypatch, wide)
    assert detect_text_boxes("synthetic.mp4") == []
    # Nearly the full width and height, even a thin ring of it, is not text either.
    assert textremoval_core._covers_frame((0, 0, 400, 300), 400, 300)
    assert textremoval_core._covers_frame((5, 5, 395, 295), 400, 300)
    assert not textremoval_core._covers_frame((0, 0, 400, 160), 400, 300)  # 53 % of the area
    assert textremoval_core._covers_frame((0, 0, 400, 190), 400, 300)  # 63 % of the area
    assert not textremoval_core._covers_frame((0, 0, 400, 100), 400, 300)
    assert not textremoval_core._covers_frame((60, 40, 340, 70), 400, 300)


def test_a_box_seen_on_fewer_than_two_frames_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_frames(monkeypatch, _text_like(1))
    assert detect_text_boxes("synthetic.mp4") == []
    _patch_frames(monkeypatch, _text_like(2))
    assert len(detect_text_boxes("synthetic.mp4")) == 1
    _patch_frames(monkeypatch, np.zeros((0, 300, 400, 3), dtype=np.uint8))
    assert detect_text_boxes("synthetic.mp4") == []


def test_indexing_reports_no_full_frame_text_box(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_project(tmp_path, "Whitewall")
    clip = tmp_path / "footage" / "white.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg.run(
        [
            "-f", "lavfi", "-i", "color=c=white:s=320x568:r=12:d=2",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(clip),
        ]
    )  # fmt: skip
    index_library(tmp_path, "footage", transcribe=False)
    entries = [e for e in load_catalog(tmp_path).entries.values() if e.path.endswith("white.mp4")]
    assert entries and entries[0].text_boxes == []
