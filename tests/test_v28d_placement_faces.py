"""V2.8d: minimal caption moves, one face test with fewer false positives, CTA in review."""

from __future__ import annotations

import copy
import csv
import inspect
from pathlib import Path
from typing import Any

import pytest

from adanvil import formats
from adanvil.batch import MANIFEST_COLUMNS, write_manifest
from adanvil.captionfaces import (
    MIN_SCORE,
    caption_on_face,
    face_core,
    frame_faces,
    is_real_face,
    rendered_faces,
)
from adanvil.formats import _boxes, _nearest_free_spot, reformat
from adanvil.storyboard import Storyboard
from adanvil.variants import tail_preview
from adanvil.vision import client as vision_client
from adanvil.vision.engine import Face, FacePicture, FakeVisionEngine
from tests.test_v28c_variants_usability import _base, _face_document, _project

# 1 ------------------------------------------------------------------------ minimal move


def test_nearest_free_spot_stays_when_free_and_prefers_the_same_side() -> None:
    face = (300.0, 400.0, 800.0, 900.0)
    # a hook at the top, free: it stays where it is (cost 0)
    assert _nearest_free_spot((200.0, 80.0, 880.0, 200.0), [face], 70.0, 1250.0, 650.0) == 80.0
    # a caption on the face: the nearest free spot flush against it, on its own side
    top_half = _nearest_free_spot((200.0, 420.0, 880.0, 540.0), [face], 70.0, 1250.0, 650.0)
    assert top_half is not None and top_half + 120 <= face[1]
    low_half = _nearest_free_spot((200.0, 760.0, 880.0, 880.0), [face], 70.0, 1250.0, 650.0)
    assert low_half is not None and low_half >= face[3]
    # no room on its side: it crosses to the other one rather than staying on the face
    tight = _nearest_free_spot((200.0, 420.0, 880.0, 540.0), [face], 380.0, 1250.0, 650.0)
    assert tight is not None and tight >= face[3]
    assert _nearest_free_spot((200.0, 420.0, 880.0, 540.0), [face], 380.0, 950.0, 650.0) is None


def _layout(
    root: Path, document: dict[str, Any], ratio: str
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """The re-laid storyboard without the face step, the one with it, and its notes."""

    original = formats._avoid_faces
    formats._avoid_faces = lambda *args, **kwargs: []
    try:
        plain, _ = reformat(document, ratio, root=root)
    finally:
        formats._avoid_faces = original
    placed, notes = reformat(document, ratio, root=root)
    return plain, placed, notes


@pytest.mark.parametrize("ratio", ["4:5", "1:1"])
def test_a_free_caption_keeps_its_place_and_a_moved_one_moves_least(
    tmp_path: Path, ratio: str
) -> None:
    root = _project(tmp_path, face=(150, 215))  # the face core is mid frame
    with vision_client.use_engine(FakeVisionEngine()):
        free_doc = _face_document()
        free_doc["overlays"][0]["y"] = 60
        plain, placed, notes = _layout(root, free_doc, ratio)
        assert placed["overlays"][0]["y"] == plain["overlays"][0]["y"]  # nothing moved
        assert not any("off a face" in note for note in notes), notes

        on_face = _face_document()  # y 660: on the face in 9:16 and in the other formats
        plain, placed, notes = _layout(root, on_face, ratio)
        cores = frame_faces(placed, root, 1.0)
        box = _boxes(Storyboard.model_validate(placed), root)[0]
        assert cores and not caption_on_face(box, cores), (box, cores, notes)
        moved = placed["overlays"][0]["y"] - plain["overlays"][0]["y"]
        assert any(f"moved {moved:+d}px off a face" in note for note in notes), (moved, notes)
        # no free spot nearer than the one chosen (searched pixel by pixel)
        height = box[3] - box[1]
        top, bottom = formats._band(ratio, formats.FORMATS[ratio].height)
        plain_box = _boxes(Storyboard.model_validate(plain), root)[0]
        free = [
            y
            for y in range(int(top), int(bottom - height))
            if not caption_on_face((box[0], y, box[2], y + height), cores)
        ]
        best = min(abs(y - plain_box[1]) for y in free)
        assert abs(moved) <= best + 12, (moved, best)


# 2 ---- one face test, fewer false positives


def test_the_face_test_measures_the_core_not_the_hair() -> None:
    detector_box = (300.0, 400.0, 700.0, 900.0)  # brow to chin
    core = face_core(detector_box)
    assert core[1] > detector_box[1] and core[3] < detector_box[3] + 40
    above_hair = (250.0, 250.0, 750.0, 340.0)
    on_forehead = (250.0, 380.0, 750.0, 440.0)
    on_eyes = (250.0, 480.0, 750.0, 560.0)
    on_chin = (250.0, 880.0, 750.0, 940.0)
    assert not caption_on_face(above_hair, [core])
    assert not caption_on_face(on_forehead, [core])
    assert caption_on_face(on_eyes, [core]) and caption_on_face(on_chin, [core])


def test_drawings_and_tiny_faces_are_not_people() -> None:
    assert is_real_face(0.94, 0.3) and is_real_face(0.9, 0.25)  # real face boxes from a test ad
    assert not is_real_face(0.85, 0.34)  # the anatomy diagram on a tablet (0.76 to 0.85)
    assert not is_real_face(0.76, 0.34)
    assert not is_real_face(0.95, 0.03)  # a background figure
    assert MIN_SCORE > 0.85


def test_rendered_faces_skip_the_diagram(tmp_path: Path) -> None:
    root = _project(tmp_path)

    class Diagram(FakeVisionEngine):
        def detect_faces(self, paths):  # a person (0.94) and a drawing (0.80)
            return [
                FacePicture(
                    (216, 384),
                    [Face((0.1, 0.1, 0.4, 0.4), 0.94), Face((0.55, 0.5, 0.9, 0.8), 0.8)],
                )
                for _ in paths
            ]

    cores, size = rendered_faces(root / "footage" / "body.mp4", 1.0, Diagram())
    assert len(cores) == 1 and cores[0][0] < size[0] * 0.4


def test_render_notes_qc_and_score_use_the_same_function() -> None:
    from adanvil import formats as layout
    from adanvil.qc import report, score

    for function in (layout._avoid_faces, report._caption_faces_check, score._face_findings):
        source = inspect.getsource(function)
        assert "caption_on_face" in source
        assert "overlaps(" not in source.replace("def overlaps", "")


# 3 ------------------------------------------------------------------------ CTA in review


def _variant_doc(text: str) -> dict[str, Any]:
    document = copy.deepcopy(_base())
    document["overlays"][1]["lines"] = [text]  # a CTA caption on the body part
    return document


def test_the_sheet_bottom_row_is_the_cta_part_before_the_end_card() -> None:
    a, b = tail_preview(_variant_doc("Price A"), 3.0), tail_preview(_variant_doc("Price B"), 3.0)
    assert [s["label"] for s in a["segments"]] == ["body"]  # no end card in the row
    assert a["overlays"][0]["lines"] == ["Price A"]
    assert a["overlays"][0]["lines"] != b["overlays"][0]["lines"]
    # an ad that does not end on an image keeps its last seconds as they were
    clip_end = _base()
    clip_end["segments"].pop()
    clip_end["overlays"].pop()
    assert [s["label"] for s in tail_preview(clip_end, 3.0)["segments"]] == ["body"]


def _entry(name: str, cta: str | None, pieces: list[str], ratio: str = "9:16") -> dict[str, Any]:
    return {
        "status": "done",
        "file": f"renders/b/{name}.mp4",
        "ids": {"hook": "h2", **({"cta": cta} if cta else {})},
        "format": ratio,
        "duration": 5.0,
        "score": 80,
        "hook_text": "h",
        "cta_text": " | ".join(pieces),
        "cta_pieces": pieces,
    }


def _rows(path: Path) -> list[list[str]]:
    return list(csv.reader(path.read_text(encoding="utf-8").splitlines()))


def test_manifest_has_a_cta_id_and_only_the_texts_that_differ(tmp_path: Path) -> None:
    shared = ["Clique ici", "24,90 EUR"]
    path = write_manifest(
        tmp_path,
        "b",
        [
            _entry("a", "ca", [*shared, "Satisfait ou rembourse"]),
            _entry("b", "cb", [*shared, "Garantie 30 jours"]),
        ],
    )
    rows = _rows(path)
    assert rows[0] == MANIFEST_COLUMNS and "CTA id" in rows[0]
    by_file = {row[0]: row for row in rows[1:]}
    assert by_file["a.mp4"][6:8] == ["ca", "Satisfait ou rembourse"]
    assert by_file["b.mp4"][6:8] == ["cb", "Garantie 30 jours"]


def test_manifest_falls_back_to_the_whole_cta_text(tmp_path: Path) -> None:
    # no CTA axis: the current text
    path = write_manifest(tmp_path, "one", [_entry("x", None, ["Clique ici", "24,90 EUR"])])
    assert _rows(path)[1][6:8] == ["", "Clique ici | 24,90 EUR"]
    # two CTAs with the same texts in another order: nothing differs, the whole text is kept
    path = write_manifest(
        tmp_path,
        "same",
        [_entry("a", "ca", ["x", "y"]), _entry("b", "cb", ["y", "x"])],
    )
    assert [row[7] for row in _rows(path)[1:]] == ["x | y", "y | x"]


def test_an_older_manifest_is_migrated_when_merged(tmp_path: Path) -> None:
    folder = tmp_path / "renders" / "old"
    folder.mkdir(parents=True)
    legacy = [c for c in MANIFEST_COLUMNS if c not in ("CTA id", "music licence", "quality")]
    with (folder / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(legacy)
        writer.writerow(["old.mp4", "hook=h1;cta=ca", "9:16", "5.00", "80", "h", "Old CTA"])
    path = write_manifest(tmp_path, "old", [_entry("new", "cb", ["New CTA"])])
    rows = _rows(path)
    expected = ["old.mp4", "hook=h1;cta=ca", "9:16", "5.00", "80", "h", "ca", "Old CTA", "", ""]
    assert rows[1] == expected
    assert rows[2][6] == "cb"
