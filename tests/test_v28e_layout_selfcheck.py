"""V2.8e: a move is checked, notes and QC agree, stickers keep their centre, CTA text with roles,
no silent overwrite, ``draft`` as a quality."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import pytest

from adanvil import formats
from adanvil.batch import copy_parts, write_manifest
from adanvil.captionfaces import caption_on_face, first_covered, sample_moments
from adanvil.formats import _boxes, reformat
from adanvil.qc import report as qc_report
from adanvil.render.stickers import sticker_art
from adanvil.storyboard import Storyboard
from adanvil.storyboard.schema import OutputSpec, StickerOverlay, normalise_quality
from adanvil.vision import client as vision_client
from adanvil.vision.engine import FakeVisionEngine
from tests.test_v28c_variants_usability import _base, _face_document, _project

# The face cores of a real segment of a test ad (1:1, "Pas forcément !" from 7.9 to 8.9 s,
# measured on 09/10 with the detector on the frames of the source clip): the head rises and turns
# while the caption stays, so one frame is not enough.
SAMPLE_LOOKS = [
    (7.92, [(176.0, 219.0, 769.0, 948.0)]),
    (8.16, [(237.0, 171.0, 856.0, 854.0)]),
    (8.40, [(269.0, 88.0, 854.0, 739.0)]),
    (8.64, [(345.0, 84.0, 797.0, 516.0)]),
    (8.88, [(372.0, 148.0, 752.0, 510.0)]),
]


def _layout(
    root: Path, document: dict[str, Any], looks: list[Any], ratio: str = "1:1"
) -> tuple[dict[str, Any], list[str]]:
    original = formats.caption_faces
    formats.caption_faces = lambda *args, **kwargs: looks
    try:
        return reformat(document, ratio, root=root)
    finally:
        formats.caption_faces = original


def _final_box(root: Path, moved: dict[str, Any], index: int = 0) -> Any:
    return _boxes(Storyboard.model_validate(moved), root)[index]


# 1 ---------------------------------------------------------------- a move is checked


def test_a_move_is_never_noted_off_a_face_when_the_check_rejects_it(tmp_path: Path) -> None:
    """The real segment: the head covers the whole free band at one moment or another."""

    root = _project(tmp_path)
    document = _face_document()
    document["overlays"][0].update({"lines": ["Pas forcément !"], "y": 1400})
    moved, notes = _layout(root, document, SAMPLE_LOOKS)
    box = _final_box(root, moved)
    covered_at = first_covered(box, SAMPLE_LOOKS)
    off_a_face = [n for n in notes if "off a face" in n]
    if covered_at is not None:
        assert not off_a_face, (box, notes)  # never "moved off a face" on a rejected position
        assert any("caption_on_face" in n and "face" in n for n in notes), notes
    else:
        assert off_a_face, notes
    # the old layout looked at four moments only and put the caption across the eyes
    # (y 160 to 195 at 8.4 s); the position kept must not be on the face at 8.4 s
    at_840 = [look for look in SAMPLE_LOOKS if look[0] == 8.40]
    assert first_covered(box, at_840) is None, box


def test_a_free_spot_on_the_other_side_is_taken_after_the_check(tmp_path: Path) -> None:
    root = _project(tmp_path)
    document = _face_document()  # a caption at 660 in 9:16
    face = (380.0, 380.0, 700.0, 640.0)
    looks = [(0.4, [face]), (1.0, [(face[0], face[1] + 90, face[2], face[3] + 90)])]
    moved, notes = _layout(root, document, looks)
    box = _final_box(root, moved)
    assert first_covered(box, looks) is None, (box, notes)
    assert any("off a face" in n for n in notes), notes
    assert not any("cannot avoid" in n for n in notes), notes


def test_every_sampled_moment_counts_not_four() -> None:
    moments = sample_moments(7.9, 8.9, 22.0)
    assert len(moments) >= 5 and moments[0] > 7.9 and moments[-1] < 8.9
    # a face that is only there for a moment between four coarse samples is still found
    boxes = (100.0, 100.0, 300.0, 200.0)
    looks = [(m, [(150.0, 120.0, 250.0, 180.0)] if abs(m - 8.4) < 0.11 else []) for m in moments]
    assert first_covered(boxes, looks) is not None
    assert caption_on_face(boxes, [(150.0, 120.0, 250.0, 180.0)])


# 2 ---------------------------------------------------------------- notes and QC agree


def test_notes_and_qc_report_both_sources_when_they_disagree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path, face=(150, 215))
    document = _face_document()
    document["overlays"][0]["y"] = 700
    model = Storyboard.model_validate(document)
    boxes = _boxes(model, root)
    box = boxes[0]
    on_it = (box[0] - 10, box[1] - 10, box[2] + 10, box[3] + 10)
    import adanvil.captionfaces as captionfaces

    # the source face data says a face is under the caption, the rendered frame shows none
    monkeypatch.setattr(captionfaces, "frame_faces", lambda *a, **k: [on_it])
    monkeypatch.setattr(captionfaces, "rendered_faces", lambda *a, **k: ([], (540, 960)))
    checks: list[Any] = []
    with vision_client.use_engine(FakeVisionEngine()):
        qc_report._caption_faces_check(
            root / "footage" / "hook_a.mp4", 2.0, model, boxes, checks, document, root
        )
    verdict = next(c for c in checks if c.id == "caption_on_face")
    assert verdict.status == "warn"
    text = verdict.message
    assert "source face data" in text and "rendered-frame detection finds none" in text, text
    # and the other way round
    monkeypatch.setattr(captionfaces, "frame_faces", lambda *a, **k: [])
    monkeypatch.setattr(captionfaces, "rendered_faces", lambda *a, **k: ([on_it], (1080, 1920)))
    checks = []
    with vision_client.use_engine(FakeVisionEngine()):
        qc_report._caption_faces_check(
            root / "footage" / "hook_a.mp4", 2.0, model, boxes, checks, document, root
        )
    text = next(c for c in checks if c.id == "caption_on_face").message
    assert "rendered-frame detection finds a face" in text and "source face data finds none" in text


def test_the_final_note_is_made_on_the_final_position(tmp_path: Path) -> None:
    root = _project(tmp_path)
    document = _face_document()
    document["overlays"][0]["lines"] = ["T'es sûre ?"]
    # a face over the whole frame: the caption cannot avoid it, the note says when
    looks = [(0.5, [(0.0, 0.0, 1080.0, 1080.0)]), (1.5, [(0.0, 0.0, 1080.0, 1080.0)])]
    moved, notes = _layout(root, document, looks)
    final = [n for n in notes if "cannot avoid" in n]
    assert len(final) == 1 and "0.5s" in final[0], notes
    assert not any("off a face" in n for n in notes)
    # with an override that keeps it on the face the note still comes, after the override
    document["overlays"][0]["formats"] = {"1:1": {"y": 500}}
    _moved, notes = _layout(root, document, looks)
    assert any("still covers a face" in n and "formats" in n for n in notes), notes


# 3 ---------------------------------------------------------------- stickers keep their centre


def _badge_centre(document: dict[str, Any], index: int = 0) -> float:
    overlay = StickerOverlay.model_validate(document["overlays"][index])
    art = sticker_art(overlay)
    centre = art.x + art.image.width / 2
    art.image.close()
    return centre


def test_a_shrinking_badge_keeps_its_centre() -> None:
    document = {
        "name": "badge",
        "format": {"width": 1080, "height": 1920, "fps": 10},
        "emoji": "none",
        "segments": [{"type": "clip", "source": "footage/hook_a.mp4", "in": 0, "out": 2}],
        "overlays": [
            {
                "type": "sticker",
                "kind": "badge",
                "text": "Satisfaite ou remboursée 60 jours",
                "x": 190,
                "y": 250,
                "size": 40,
            }
        ],
    }
    before = _badge_centre(document)
    moved, _ = reformat(document, "1:1")
    after_overlay = moved["overlays"][0]
    assert after_overlay["size"] < 40 and after_overlay["x"] > 190  # smaller, moved to the right
    after = _badge_centre(moved)
    assert abs(after - before) <= 2, (before, after)


# 4 ---------------------------------------------------------------- CTA text with roles


def _cta_variant(caption: str, badge: str) -> dict[str, Any]:
    document = copy.deepcopy(_base())
    document["overlays"][2]["lines"] = [caption]
    document["overlays"].append(
        {
            "type": "sticker",
            "kind": "badge",
            "text": badge,
            "x": 100,
            "y": 300,
            "start": 5.0,
            "end": 6.4,
        }
    )
    return document


def test_cta_text_shows_the_role_of_each_text(tmp_path: Path) -> None:
    root = _project(tmp_path)
    price, guarantee = "65,90 € au lieu de 89,90 €", "Satisfaite ou remboursée 60 jours"
    _, a = copy_parts(root, _cta_variant(price, guarantee))
    _, b = copy_parts(root, _cta_variant(guarantee + " ✅", price))
    assert a == [f"caption: {price}", f"badge: {guarantee}"]
    assert b == [f"caption: {guarantee} ✅", f"badge: {price}"]

    def entry(name: str, cta: str, pieces: list[str]) -> dict[str, Any]:
        return {
            "status": "done",
            "file": f"renders/b/{name}.mp4",
            "ids": {"hook": "h1", "cta": cta},
            "format": "9:16",
            "duration": 5.0,
            "score": 80,
            "hook_text": "h",
            "cta_text": " | ".join(pieces),
            "cta_pieces": pieces,
        }

    path = write_manifest(tmp_path, "b", [entry("a", "ca", a), entry("b", "cb", b)])
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    texts = {row[6]: row[7] for row in rows[1:]}
    assert texts["ca"] == f"caption: {price} | badge: {guarantee}"
    assert texts["cb"] == f"caption: {guarantee} ✅ | badge: {price}"
    assert texts["ca"] != texts["cb"]


# 5 ---------------------------------------------------------------- no silent overwrite


def test_variants_build_refuses_to_overwrite_an_existing_file(tmp_path: Path) -> None:
    import anyio
    from mcp import Client

    from adanvil.server import build_server

    root = _project(tmp_path)
    own = root / "storyboards" / "demo_h1_ca.json"
    own.parent.mkdir(parents=True, exist_ok=True)
    own.write_text(json.dumps({"name": "mine"}), encoding="utf-8")
    variants = {"hook": [{"id": "h1", "changes": []}], "cta": [{"id": "ca"}]}

    async def build(**extra: Any) -> Any:
        async with Client(build_server()) as client:
            return await client.call_tool(
                "variants_build",
                {
                    "project_dir": str(root),
                    "base": _base(),
                    "variants": variants,
                    "contact_sheet": False,
                    **extra,
                },
            )

    refused = anyio.run(build)
    assert refused.is_error
    assert "storyboards/demo_h1_ca.json" in refused.content[0].text
    assert json.loads(own.read_text("utf-8")) == {"name": "mine"}  # untouched
    anyio.run(lambda: build(overwrite=True))
    assert json.loads(own.read_text("utf-8"))["name"] == "demo_h1_ca"
    # since V2.9b its own earlier output is refused too: a storyboard is never replaced silently
    again = anyio.run(build)
    assert again.is_error and "overwrite=true" in again.content[0].text


# 6 ---------------------------------------------------------------- quality names


def test_draft_is_an_alias_of_preview(tmp_path: Path) -> None:
    import anyio
    from mcp import Client

    from adanvil.server import build_server

    assert normalise_quality("draft") == "preview" and normalise_quality("final") == "final"
    assert OutputSpec(quality="draft").quality == "preview"
    root = _project(tmp_path)

    async def scenario() -> tuple[Any, Any, Any]:
        async with Client(build_server()) as client:
            started = await client.call_tool(
                "render_start",
                {"project_dir": str(root), "storyboard": _base(), "quality": "draft"},
            )
            job = json.loads(started.content[0].text)
            status = await client.call_tool(
                "job_status", {"job_id": job["job_id"], "wait_seconds": 100}
            )
            return started, job, json.loads(status.content[0].text)

    started, job, status = anyio.run(scenario)
    assert not started.is_error and status["status"] == "done", status
    assert Path(status["report"]["path"]).name == "demo.preview.mp4"


def test_a_preview_render_is_scored_in_the_manifest(tmp_path: Path) -> None:
    from adanvil.batch import plan_items, run_batch

    root = _project(tmp_path)
    items, _ = plan_items(root, [_base()], None, "scored")
    result = run_batch(
        root, {"batch": "scored", "items": items, "quality": "draft", "score": True}
    )
    entry = result["renders"][0]
    assert entry["status"] == "done" and result["quality"] == "preview", entry
    assert isinstance(entry["score"], int), entry
    rows = list(csv.reader((root / result["manifest"]).read_text("utf-8").splitlines()))
    assert rows[1][4].isdigit() and rows[1][0].endswith(".preview.mp4")
