"""V2.8c: the usability fixes found by a real test of variants and formats (12 items)."""

from __future__ import annotations

import copy
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from adanvil._bin import ffmpeg_path
from adanvil.batch import copy_texts, write_manifest
from adanvil.busy import overlaps
from adanvil.captionfaces import frame_faces
from adanvil.formats import _boxes, reformat
from adanvil.project import init_project
from adanvil.storyboard import Storyboard
from adanvil.variants import (
    VariantError,
    expand_matrix,
    replace_slot,
    tail_preview,
    variants_sheet,
)
from adanvil.vision import client as vision_client
from adanvil.vision.engine import FAKE_FACE_RGB, FakeVisionEngine

FACE = FAKE_FACE_RGB


def _clip(path: Path, seconds: float = 3.0, face: tuple[int, int] | None = None) -> None:
    """A 360x640 clip, with a face-coloured 60x60 square at ``face`` (x, y) when asked."""

    path.parent.mkdir(parents=True, exist_ok=True)
    colour = f"0x{FACE[0]:02X}{FACE[1]:02X}{FACE[2]:02X}"
    graph = f"color=c=0x203040:s=360x640:r=10:d={seconds},format=yuv420p[bg]"
    if face is None:
        graph = f"color=c=0x305070:s=360x640:r=10:d={seconds},format=yuv420p[v]"
    else:
        graph += (
            f";color=c={colour}:s=60x60:r=10:d={seconds},format=yuv420p[face];"
            f"[bg][face]overlay=x={face[0]}:y={face[1]}:shortest=1[v]"
        )
    subprocess.run(
        [
            ffmpeg_path(),
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-filter_complex",
            graph,
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "5",
            str(path),
        ],
        check=True,
    )


def _project(tmp_path: Path, face: tuple[int, int] | None = None) -> Path:
    root = tmp_path / "project"
    init_project(root, "Usability")
    _clip(root / "footage" / "hook_a.mp4", 2.0, face)
    _clip(root / "footage" / "hook_b.mp4", 2.0)
    _clip(root / "footage" / "body.mp4", 3.0)
    Image.new("RGB", (216, 384), (230, 230, 230)).save(root / "footage" / "card.png")
    return root


def _base() -> dict[str, Any]:
    return {
        "name": "demo",
        "format": {"width": 1080, "height": 1920, "fps": 10},
        "emoji": "none",
        "segments": [
            {"type": "clip", "source": "footage/hook_a.mp4", "in": 0, "out": 2, "label": "hook"},
            {"type": "clip", "source": "footage/body.mp4", "in": 0, "out": 3, "label": "body"},
            {"type": "image", "source": "footage/card.png", "duration": 1.5, "label": "end_card"},
        ],
        "overlays": [
            {"type": "caption", "style": "pill", "lines": ["Old hook"], "start": 0.2, "end": 1.8},
            {"type": "caption", "style": "outline", "lines": ["Body"], "start": 2.5, "end": 4},
            {"type": "caption", "style": "pill", "lines": ["Shop now"], "start": 5.0, "end": 6.4},
        ],
    }


def _hook(length: float, text: str, extra: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "replace_label": "hook",
        "segments": [
            {
                "type": "clip",
                "source": "footage/hook_b.mp4",
                "in": 0,
                "out": length,
                "label": "hook",
            }
        ],
        "overlays": [
            {"type": "caption", "style": "pill", "lines": [text], "start": 0.3, "end": 1.0},
            *(extra or []),
        ],
    }


# 1 ------------------------------------------------------------------------ empty changes


def test_empty_changes_are_the_control() -> None:
    variants = {
        "hook": [
            {"id": "h1", "changes": []},
            {"id": "h2"},
            {"id": "h3", "changes": [_hook(1.0, "New")]},
        ]
    }
    built = expand_matrix(_base(), variants)
    assert [v.name for v in built] == ["demo_h1", "demo_h2", "demo_h3"]
    for control in built[:2]:
        document = copy.deepcopy(control.document)
        document["name"] = "demo"
        document["output"] = _base().get("output", document["output"])
        assert document["segments"] == _base()["segments"]
        assert document["overlays"] == _base()["overlays"]
    assert built[2].duration == pytest.approx(5.5)


# 2 --------------------------------------------------------------------- float tolerance


def test_an_overlay_equal_to_the_slot_up_to_rounding_is_inside() -> None:
    base = _base()
    # a range written with 4 decimals, as a tool would: 0.0005 s off the summed durations
    base["segments"][0]["out"] = 2.0007
    base["overlays"][0].update(start=0.0, end=2.0012)
    built, _ = replace_slot(base, _hook(1.0, "Fresh"))
    assert [o["lines"] for o in built["overlays"] if o["type"] == "caption"][0] == ["Fresh"]


# 3 ----------------------------------------------------------------- sounds follow the slot


def test_explicit_sfx_follow_the_slot_and_the_build_says_what_moved() -> None:
    base = _base()
    base["overlays"].append({"type": "sfx", "source": "whoosh_short", "at": 0.5})  # in the slot
    base["overlays"].append({"type": "sfx", "source": "riser", "at": 3.0})  # after the hook
    base["overlays"].append({"type": "sfx", "source": "ding", "at": 6.0})
    variants = {
        "hook": [
            {"id": "longer", "changes": [_hook(2.6, "Longer")]},
            {
                "id": "own",
                "changes": [
                    _hook(
                        1.0,
                        "Own",
                        [{"type": "sfx", "source": "pop", "at": 0.1}],
                    )
                ],
            },
        ]
    }
    longer, own = expand_matrix(base, variants)
    by_source = {o["source"]: o for o in longer.document["overlays"] if o["type"] == "sfx"}
    assert by_source["riser"]["at"] == pytest.approx(3.6)
    assert by_source["ding"]["at"] == pytest.approx(6.6)
    assert by_source["whoosh_short"]["at"] == pytest.approx(0.5)  # kept, same offset
    assert any("riser" in line and "moved" in line for line in longer.moved)
    sources = {o["source"]: o for o in own.document["overlays"] if o["type"] == "sfx"}
    assert "whoosh_short" not in sources  # the new slot brings its own sound
    assert sources["pop"]["at"] == pytest.approx(0.1)
    assert sources["riser"]["at"] == pytest.approx(2.0)  # 3.0 - 1.0
    assert any("whoosh_short" in line and "dropped" in line for line in own.moved)
    for variant in (longer, own):
        Storyboard.model_validate(variant.document)


# 4 ------------------------------------------------------------------- all errors at once


def test_every_error_of_every_variant_comes_back_together() -> None:
    variants = {
        "hook": [
            {"id": "x", "changes": [{"path": "segments.0.scale", "value": "medium_closeup"}]},
            {"id": "ok", "changes": []},
            {"id": "y", "changes": [{"path": "segments.9.out", "value": 1}]},
        ],
        "cta": [{"id": "a", "changes": []}, {"id": "b", "changes": []}],
    }
    with pytest.raises(VariantError) as caught:
        expand_matrix(_base(), variants)
    errors = caught.value.errors
    assert {e["variant"] for e in errors} == {
        "hook=x, cta=a",
        "hook=x, cta=b",
        "hook=y, cta=a",
        "hook=y, cta=b",
    }
    assert all(e["message"] for e in errors)
    text = str(caught.value)
    assert "hook=x, cta=a" in text and "hook=y, cta=b" in text and "4 errors" in text


# 5 --------------------------------------------------------------------- CTA text column


def _cta_document(text: str, extra: str | None = None) -> dict[str, Any]:
    document = _base()
    document["overlays"][2]["lines"] = [text]
    if extra is not None:
        document["overlays"].append(
            {"type": "caption", "style": "pill", "lines": [extra], "start": 5.2, "end": 6.4}
        )
    return document


def test_cta_text_tells_two_ctas_apart(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _, a = copy_texts(root, _cta_document("Click now"))
    _, b = copy_texts(root, _cta_document("Satisfied or refunded 60 days", extra="Click now"))
    assert a == "caption: Click now"
    assert b == "caption: Satisfied or refunded 60 days | caption: Click now"
    assert a != b
    # the hook text is unchanged
    hook, _ = copy_texts(root, _cta_document("Click now"))
    assert hook == "Old hook"


# 6 ------------------------------------------------------------------- manifest merge


def _entry(name: str, ratio: str, ids: dict[str, str], score: int | None = 80) -> dict[str, Any]:
    return {
        "status": "done",
        "file": f"renders/b/{name}.mp4",
        "ids": ids,
        "format": ratio,
        "duration": 5.0,
        "score": score,
        "hook_text": "h",
        "cta_text": "c",
    }


def test_manifest_merges_by_file_name(tmp_path: Path) -> None:
    ids = {"hook": "h2", "cta": "ca"}
    write_manifest(
        tmp_path, "b", [_entry("ad_9x16", "9:16", ids), _entry("ad_4x5", "4:5", ids, 70)]
    )
    path = write_manifest(
        tmp_path, "b", [_entry("ad_1x1", "1:1", ids), _entry("ad_4x5", "4:5", ids, 91)]
    )
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    assert [r[0] for r in rows[1:]] == ["ad_9x16.mp4", "ad_4x5.mp4", "ad_1x1.mp4"]
    assert rows[2][4] == "91"  # replaced in place
    assert rows[1][1] == "hook=h2;cta=ca"


def test_a_variant_of_a_variant_keeps_its_parent_ids(tmp_path: Path) -> None:
    import anyio
    from mcp import Client

    from adanvil.server import build_server

    root = _project(tmp_path)

    async def scenario() -> dict[str, Any]:
        async with Client(build_server()) as client:
            await client.call_tool(
                "variants_build",
                {
                    "project_dir": str(root),
                    "base": _base(),
                    "variants": {"hook": [{"id": "h2", "changes": []}], "cta": [{"id": "ca"}]},
                    "contact_sheet": False,
                },
            )
            fixed = await client.call_tool(
                "variants_build",
                {
                    "project_dir": str(root),
                    "base": "storyboards/demo_h2_ca.json",
                    "variants": {"fix": [{"id": "sq", "changes": []}]},
                    "contact_sheet": False,
                },
            )
            return json.loads(fixed.content[0].text)

    payload = anyio.run(scenario)
    assert payload["variants"][0]["ids"] == {"hook": "h2", "cta": "ca", "fix": "sq"}
    side = json.loads((root / "storyboards" / "demo_h2_ca_sq.variant.json").read_text("utf-8"))
    assert side["ids"] == {"hook": "h2", "cta": "ca", "fix": "sq"}


# 7 ----------------------------------------------------------------- captions never on a face


def _face_document() -> dict[str, Any]:
    return {
        "name": "faces",
        "format": {"width": 1080, "height": 1920, "fps": 10},
        "emoji": "none",
        "segments": [
            {"type": "clip", "source": "footage/hook_a.mp4", "in": 0, "out": 2, "label": "hook"}
        ],
        "overlays": [
            {
                "type": "caption",
                "style": "pill",
                "lines": ["Plus maintenant !"],
                "start": 0.2,
                "end": 1.8,
                "y": 660,
            }
        ],
    }


@pytest.mark.parametrize("ratio", ["1:1", "4:5"])
def test_a_caption_never_sits_on_a_face_in_another_format(tmp_path: Path, ratio: str) -> None:
    root = _project(tmp_path, face=(150, 215))  # the face is at x 450-630, y 645-825 (9:16)
    document = _face_document()
    engine = FakeVisionEngine()
    with vision_client.use_engine(engine):
        moved, notes = reformat(document, ratio, root=root)
        faces = frame_faces(moved, root, 1.0)
        model = Storyboard.model_validate(moved)
        box = _boxes(model, root)[0]
    assert faces, "the face is carried to the new frame"
    assert not overlaps(box, faces), (box, faces, notes)
    assert any("off a face" in note for note in notes) or any("caption_on_face" in n for n in notes)
    # sanity: without the face check the caption would have been on the face
    from adanvil import formats

    original = formats._avoid_faces
    formats._avoid_faces = lambda *args, **kwargs: []
    try:
        with vision_client.use_engine(engine):
            plain, _ = reformat(document, ratio, root=root)
            clash = _boxes(Storyboard.model_validate(plain), root)[0]
            assert overlaps(clash, frame_faces(plain, root, 1.0) or [])
    finally:
        formats._avoid_faces = original


def test_without_face_data_the_check_is_skipped_with_a_note(tmp_path: Path, monkeypatch) -> None:
    root = _project(tmp_path, face=(150, 215))
    monkeypatch.setenv("ADANVIL_VISION", "off")
    _, notes = reformat(_face_document(), "1:1", root=root)
    assert any("no face data" in note for note in notes)


def test_ad_score_flags_a_caption_on_a_face_in_every_format(tmp_path: Path) -> None:
    import numpy as np

    class RenderedFake(FakeVisionEngine):
        """h264 shifts the face colour (bt709): a wider tolerance on encoded frames."""

        def _face_mask(self, picture: Image.Image) -> np.ndarray:
            small = picture.copy()
            small.thumbnail((128, 128))
            pixels = np.asarray(small, dtype=np.int16)
            return np.abs(pixels - np.array((255, 200, 165), dtype=np.int16)).max(axis=-1) <= 25

    from adanvil.qc import ad_score
    from adanvil.render import compose
    from adanvil.storyboard import load_storyboard

    root = _project(tmp_path, face=(150, 215))
    document = _face_document()
    document["output"] = {"path": "renders/faces.mp4", "quality": "preview"}
    document["overlays"][0]["y"] = 760  # over the chin in 9:16 (the fake detector sees the rest)
    report = compose(load_storyboard(document), root=root, quality="preview")
    with vision_client.use_engine(RenderedFake()):
        score = ad_score(root, report.path, storyboard=document)
    ids = [finding["id"] for finding in score.findings]
    assert "caption_on_face" in ids, ids
    # the same check in qc_run's report, with the effective storyboard
    from adanvil.media import Format
    from adanvil.qc import qc_report

    with vision_client.use_engine(RenderedFake()):
        qc = qc_report(
            report.path, fmt=Format(width=540, height=960, fps=10), storyboard=document, root=root
        )
    flagged = next(check for check in qc.checks if check.id == "caption_on_face")
    assert flagged.status == "warn"
    assert any(path.name == "overlay_0.jpg" for path in qc.images)


# 8 ------------------------------------------------------------------ per-format override


def test_per_format_override_is_applied_after_the_layout_and_validated(tmp_path: Path) -> None:
    root = _project(tmp_path)
    document = _base()
    document["overlays"][0]["formats"] = {"1:1": {"y": 900, "size": 50}, "4:5": {"hidden": True}}
    document["overlays"][1]["formats"] = {"1:1": {"x": 300, "anchor": "bottom"}}
    Storyboard.model_validate(document)
    square, notes = reformat(document, "1:1", root=root)
    assert square["overlays"][0]["y"] == 900 and square["overlays"][0]["size"] == 50
    assert square["overlays"][1]["x_center"] == 300 and square["overlays"][1]["anchor"] == "bottom"
    assert all("formats" not in overlay for overlay in square["overlays"])
    feed, feed_notes = reformat(document, "4:5", root=root)
    assert len(feed["overlays"]) == 2 and any("hidden" in n for n in feed_notes)
    plain = copy.deepcopy(document)
    plain["overlays"][0].pop("formats")
    assert (
        "formats"
        not in Storyboard.model_validate(plain).model_dump(exclude_none=True)["overlays"][0]
    )
    for bad in ({"2:1": {"y": 1}}, {"1:1": {"colour": "red"}}, {"1:1": {"size": 0}}):
        broken = copy.deepcopy(document)
        broken["overlays"][0]["formats"] = bad
        with pytest.raises(ValueError):
            Storyboard.model_validate(broken)


# 9 ---------------------------------------------------------------------- one 9:16


def test_the_9x16_of_a_9x16_storyboard_is_the_plain_render(tmp_path: Path) -> None:
    root = _project(tmp_path)
    document = _base()
    document["overlays"][0]["y"] = 40  # above the Reels band: the remap would have moved it
    document["overlays"][2]["y"] = 1850
    out, _ = reformat(document, "9:16", root=root)
    assert out["overlays"] == document["overlays"]
    assert out["segments"] == document["segments"]


# 10 --------------------------------------------------------- QC crops follow moved overlays


def test_qc_crops_follow_the_relaid_overlay(tmp_path: Path) -> None:
    from adanvil.qc.report import _layout_crops, _text_boxes

    root = _project(tmp_path)
    document = _base()
    document["overlays"][0]["y"] = 1500
    moved, _ = reformat(document, "1:1", root=root)
    model, boxes = _text_boxes(moved, root)
    clip = tmp_path / "square.mp4"
    subprocess.run(
        [
            ffmpeg_path(),
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x305070:s=216x216:r=10:d=7",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
    )
    out = tmp_path / "qc"
    out.mkdir()
    images: list[Path] = []
    work = tmp_path / "work"
    work.mkdir()
    _layout_crops(clip, 7.0, model, boxes, out, [], images, work)
    assert images
    first = next(p for p in images if p.name == "overlay_0.jpg")
    left, top, right, bottom = boxes[0]
    factor = 216 / model.format.width
    expected_w = min(216, round(right * factor + 80)) - max(0, round(left * factor - 80))
    expected_h = min(216, round(bottom * factor + 80)) - max(0, round(top * factor - 80))
    with Image.open(first) as crop:
        assert crop.size == pytest.approx((expected_w, expected_h), abs=2)
    assert 0 < top * factor < 216  # the moved caption is inside the 1:1 frame


# 11 -------------------------------------------------------- the sheet shows the end too


def test_tail_preview_keeps_the_end_and_the_sheet_has_two_rows(tmp_path: Path) -> None:
    root = _project(tmp_path)
    document = _base()
    tail = tail_preview(document, 3.0)
    # the ad ends on an image card: the row shows the 3 s BEFORE it (V2.8d)
    assert [s["label"] for s in tail["segments"]] == ["body"]
    assert [o["lines"] for o in tail["overlays"]] == [["Body"]]  # "Shop now" is on the card
    assert tail["overlays"][0]["start"] == pytest.approx(0.5)  # 2.5 - 2.0 (the hook is cut)
    built = expand_matrix(
        document,
        {
            "cta": [
                {"id": "a", "changes": []},
                {"id": "b", "changes": [{"path": "overlays.2.lines.0", "value": "Other"}]},
            ]
        },
        root=root,
    )
    out = root / "renders" / "variants" / "sheet.jpg"
    path, notes = variants_sheet(root, built, out)
    assert not notes, notes
    with Image.open(path) as sheet:
        assert sheet.height > 2 * 150 * 16 / 9 * 0.9  # two strips per variant


# 12 ---------------------------------------------------------------------- centered badge


@pytest.mark.parametrize("how", [{"x": "center"}, {"anchor": "center"}])
def test_badge_can_be_centered(how: dict[str, str]) -> None:
    from adanvil.render.stickers import sticker_art

    document = _base()
    document["overlays"].append(
        {"type": "sticker", "kind": "badge", "text": "60 jours", "y": 900, "start": 0, **how}
    )
    model = Storyboard.model_validate(document)
    badge = model.overlays[-1]
    assert isinstance(badge.x, int)
    art = sticker_art(badge)
    centre = (art.bbox[0] + art.bbox[2]) / 2
    assert centre == pytest.approx(540, abs=2)
    other = copy.deepcopy(document)
    other["format"]["width"] = 720
    assert Storyboard.model_validate(other).overlays[-1].x < badge.x
