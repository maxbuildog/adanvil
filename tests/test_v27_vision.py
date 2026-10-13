"""V2.7: the optional vision environment, indexing with it, search by sentence, product versions
from reference photos, and face-aware framing.

Everything runs with ``FakeVisionEngine`` (deterministic: colours and a flat-coloured "face" decide
the embeddings and the detections), so no model is needed. One test talks to the real worker and is
skipped when the vision environment is not installed.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import pytest
import yaml
from mcp import Client
from PIL import Image

from adanvil._bin import ffmpeg_path
from adanvil.brand import Brand, ProductVersion
from adanvil.ffmpeg import run
from adanvil.library import (
    annotate,
    index_library,
    index_summary,
    load_catalog,
    search,
    search_mosaic,
    search_shots,
)
from adanvil.library.vision_index import sheet_times
from adanvil.media import Cache, ClipSegment, Format, ImageSegment, RevealSegment, render_segment
from adanvil.project import init_project
from adanvil.qc import compare_pair
from adanvil.qc.pair import detector_face_finder
from adanvil.server import build_server
from adanvil.storyboard import Storyboard, validate_storyboard
from adanvil.vision import client as vision_client
from adanvil.vision.engine import FAKE_FACE_RGB, FakeVisionEngine, probabilities
from adanvil.vision.framing import (
    DEFAULT_FACE_Y,
    FaceTrack,
    Framing,
    align_zooms,
    crop_offset,
    face_cover,
    face_share,
    face_track,
    track_expression,
)
from adanvil.vision.labels import LABELS, labels_signature, resolve_tag
from adanvil.vision.setup import vision_setup
from adanvil.vision.store import VisionData, load_vision, save_vision, vision_relative
from adanvil.vision.versions import (
    ProductReference,
    brand_versions,
    match_frames,
    references_signature,
    version_photos,
)

FACE = FAKE_FACE_RGB
W, H = 216, 384
FORMAT = Format(width=W, height=H, fps=10)


@pytest.fixture
def fake() -> Any:
    engine = FakeVisionEngine()
    with vision_client.use_engine(engine):
        yield engine


def _hex(colour: tuple[int, int, int]) -> str:
    return f"0x{colour[0]:02X}{colour[1]:02X}{colour[2]:02X}"


def _clip(
    path: Path, parts: list[tuple[tuple[int, int, int], float, tuple[int, ...] | None]]
) -> None:
    """A clip made of flat colour parts, each with an optional face box ``(x, y, w, h)``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg_path(), "-nostdin", "-y", "-loglevel", "error"]
    chains = []
    for number, (colour, seconds, face) in enumerate(parts):
        command += [
            "-f",
            "lavfi",
            "-t",
            str(seconds),
            "-i",
            f"color=c={_hex(colour)}:s={W}x{H}:r=10",
        ]
        chain = f"[{number}:v]"
        if face is not None:
            x, y, w, h = face
            chain += f"drawbox=x={x}:y={y}:w={w}:h={h}:color={_hex(FACE)}:t=fill,"
        chains.append(f"{chain}format=yuv420p[p{number}]")
    joined = "".join(f"[p{number}]" for number in range(len(parts)))
    graph = ";".join([*chains, f"{joined}concat=n={len(parts)}:v=1:a=0[v]"])
    command += ["-filter_complex", graph, "-map", "[v]", "-c:v", "libx264", "-g", "5", str(path)]
    subprocess.run(command, check=True)


def _moving_face_clip(path: Path, seconds: float = 4.0) -> None:
    """A landscape clip whose face slides from the left to the right."""

    path.parent.mkdir(parents=True, exist_ok=True)
    graph = (
        "color=c=0x203040:s=640x360:r=10,format=yuv420p[bg];"
        f"color=c={_hex(FACE)}:s=60x60:r=10,format=yuv420p[face];"
        "[bg][face]overlay=x='140+t*80':y=107:shortest=1[v]"
    )
    subprocess.run(
        [
            ffmpeg_path(), "-nostdin", "-y", "-loglevel", "error", "-t", str(seconds),
            "-filter_complex", graph, "-map", "[v]", "-c:v", "libx264", "-g", "5", str(path),
        ],
        check=True,
    )  # fmt: skip


def _picture(
    path: Path,
    colour: tuple[int, int, int] = (40, 60, 90),
    face: tuple[int, int, int, int] | None = None,
    size: tuple[int, int] = (W, H),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    picture = Image.new("RGB", size, colour)
    if face is not None:
        x, y, w, h = face
        picture.paste(Image.new("RGB", (w, h), FACE), (x, y))
    picture.save(path)
    return path


def _frame_at(video: Path, index: int) -> np.ndarray:
    raw = run(
        ["-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_stdout=True
    ).stdout
    frames = np.frombuffer(raw or b"", dtype=np.uint8).reshape(-1, H, W, 3)
    return frames[min(index, len(frames) - 1)]


def _blob(frame: np.ndarray) -> tuple[float, float, float] | None:
    """Centre (0..1) and height (0..1) of the face-coloured pixels of a rendered frame."""

    mask = np.all(np.abs(frame.astype(int) - np.array(FACE)) <= 45, axis=-1)
    if mask.sum() < 20:
        return None
    rows, columns = np.nonzero(mask)
    return (
        (columns.min() + columns.max() + 1) / 2 / frame.shape[1],
        (rows.min() + rows.max() + 1) / 2 / frame.shape[0],
        (rows.max() - rows.min() + 1) / frame.shape[0],
    )


def _project(tmp_path: Path) -> Path:
    init_project(tmp_path, "Vision")
    return tmp_path


def _indexed_library(tmp_path: Path, engine: FakeVisionEngine) -> Path:
    root = _project(tmp_path)
    _clip(
        root / "footage" / "two_rooms.mp4",
        [((255, 0, 0), 2.0, None), ((0, 0, 255), 2.0, None)],
    )
    _clip(root / "footage" / "talking.mp4", [((30, 30, 30), 3.0, (80, 90, 60, 70))])
    _picture(root / "footage" / "portrait.png", (240, 240, 240), (70, 80, 70, 80))
    index_library(root, "footage", transcribe=False, vision=True, vision_engine=engine)
    return root


def _entry(root: Path, name: str) -> Any:
    return next(
        entry for entry in load_catalog(root).entries.values() if Path(entry.path).name == name
    )


# ------------------------------------------------------------------- the engine and the store


def test_fake_engine_is_deterministic_and_finds_faces(tmp_path: Path) -> None:
    engine = FakeVisionEngine()
    red = _picture(tmp_path / "red.png", (220, 10, 10))
    face = _picture(tmp_path / "face.png", (10, 10, 10), (40, 50, 30, 40))
    two = _picture(tmp_path / "two.png", (10, 10, 10), (20, 50, 30, 40))
    with Image.open(two) as opened:
        drawing = opened.copy()
    drawing.paste(Image.new("RGB", (30, 40), FACE), (150, 100))
    drawing.save(two)

    images = engine.embed_images([red, face])
    texts = engine.embed_texts(["a red wall", "a face"])
    scores = probabilities(engine, images, texts)
    assert scores[0, 0] > 0.9 > 0.1 > scores[0, 1]
    assert scores[1, 1] > 0.9 > 0.1 > scores[1, 0]
    assert np.allclose(images, engine.embed_images([red, face]))

    found = engine.detect_faces([red, face, two])
    assert [len(item.faces) for item in found] == [0, 1, 2]
    left, top, right, bottom = found[1].faces[0].box
    assert (left, right) == pytest.approx((40 / W, 70 / W), abs=0.02)
    assert (top, bottom) == pytest.approx((50 / H, 90 / H), abs=0.02)


def test_vision_data_round_trips_and_ignores_other_versions(tmp_path: Path) -> None:
    from adanvil.vision.engine import Face

    data = VisionData.build(
        "fake-vision-1",
        [0.5, 1.5],
        [(216, 384), (216, 384)],
        np.eye(2, 8, dtype=np.float32),
        [[Face((0.1, 0.2, 0.3, 0.4), 0.9)], []],
    )
    relative = vision_relative({"path": "a.mp4", "size": 1, "mtime_ns": 2})
    assert relative.startswith("cache/vision/") and relative.endswith(".npz")
    save_vision(tmp_path, relative, data)
    loaded = load_vision(tmp_path, relative)
    assert loaded is not None and loaded.model == "fake-vision-1"
    assert loaded.frames == 2 and loaded.face_counts() == [1, 0]
    assert loaded.faces(0)[0].box == pytest.approx((0.1, 0.2, 0.3, 0.4), abs=1e-6)
    assert np.allclose(loaded.embeddings, data.embeddings, atol=1e-3)
    assert load_vision(tmp_path, "cache/vision/missing.npz") is None
    assert load_vision(tmp_path, None) is None


def test_labels_are_aligned_with_prompts_and_aliases() -> None:
    assert len({label.tag for label in LABELS}) == len(LABELS)
    assert resolve_tag("Salle de bain") == "bathroom"
    assert resolve_tag("smiling") == "smiling"
    assert resolve_tag("nothing like that") is None
    assert labels_signature() == labels_signature()


def test_vision_setup_plan_is_isolated_and_prefetches_the_models(tmp_path: Path) -> None:
    plan = vision_setup(tmp_path / "env", plan=True)
    assert isinstance(plan, list) and len(plan) >= 4
    flat = [" ".join(step) for step in plan]
    assert any("transformers" in step and "opencv-python-headless" in step for step in flat)
    assert flat[-1].endswith("worker.py --prefetch")
    assert all(str(tmp_path / "env") in step or "worker.py" in step for step in flat)


# ------------------------------------------------------------------------- indexing with vision


def test_index_with_vision_stores_frames_faces_and_auto_tags(
    tmp_path: Path, fake: FakeVisionEngine
) -> None:
    root = _project(tmp_path)
    _clip(root / "footage" / "talking.mp4", [((30, 30, 30), 3.0, (80, 90, 60, 70))])
    _picture(root / "footage" / "wall.png", (200, 10, 10))
    result = index_library(root, "footage", transcribe=False, vision="auto")
    assert set(result.vision.values()) == {"done"}

    talking = _entry(root, "talking.mp4")
    assert talking.vision is not None and talking.vision.status == "done"
    assert talking.vision.frames == len(sheet_times(talking)) > 3
    assert talking.vision.faces_max == 1 and talking.vision.face_frames == talking.vision.frames
    assert "person" in talking.auto_tags and talking.auto_tags["person"] > 0.35
    assert not talking.annotation.tags  # automatic tags never touch the human ones
    data = load_vision(root, talking.vision.file)
    assert data is not None and data.has_embeddings and data.model == "fake-vision-1"

    wall = _entry(root, "wall.png")
    assert wall.vision is not None and wall.vision.frames == 1 and wall.vision.faces_max == 0
    assert "person" not in wall.auto_tags
    line = search(root, auto_tags=["person"])
    assert [Path(entry.path).name for entry in line] == ["talking.mp4"]
    assert [Path(e.path).name for e in search(root, auto_tags=["personne"])] == ["talking.mp4"]

    dumped = load_catalog(root).model_dump(mode="json")
    assert "auto_tags" in dumped["entries"][talking.id]


def test_second_index_does_no_vision_work_and_new_labels_retag_without_frames(
    tmp_path: Path, fake: FakeVisionEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _indexed_library(tmp_path, fake)
    fake.calls.clear()
    again = index_library(root, "footage", transcribe=False, vision="auto", vision_engine=fake)
    assert not again.added and not again.updated
    assert fake.calls == []  # nothing changed: the engine was not even asked

    import adanvil.vision.labels as labels

    monkeypatch.setattr(labels, "labels_signature", lambda: "other-labels")
    monkeypatch.setattr("adanvil.library.vision_index.labels_signature", lambda: "other-labels")
    index_library(root, "footage", transcribe=False, vision="auto", vision_engine=fake)
    assert "detect_faces" not in fake.calls  # re-tagged from the stored embeddings
    assert _entry(root, "talking.mp4").vision.tags_sig == "other-labels"


def test_without_the_environment_indexing_succeeds_and_says_why(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _picture(root / "footage" / "wall.png", (200, 10, 10))
    result = index_library(root, "footage", transcribe=False)  # ADANVIL_VISION=off in the suite
    assert result.added and not result.errors
    entry = _entry(root, "wall.png")
    assert entry.vision is not None and entry.vision.status.startswith("unavailable")
    summary = index_summary(root, result)
    assert summary["vision"] == {"unavailable": 1}
    assert "vision-setup" in summary["vision_note"] or "switched off" in summary["vision_note"]
    skipped = index_library(root, "footage", transcribe=False, vision=False)
    assert skipped.vision == {}


# ----------------------------------------------------------------------- search by sentence


def test_search_by_sentence_ranks_shots_and_respects_filters(
    tmp_path: Path, fake: FakeVisionEngine
) -> None:
    root = _indexed_library(tmp_path, fake)
    entries = search(root)
    hits, missing = search_shots(root, "a blue room", entries, engine=fake, limit=3)
    assert missing == 0
    top = hits[0]
    assert Path(top.entry.path).name == "two_rooms.mp4"
    assert top.start >= 1.5 and top.score > 0.9  # the second shot is the blue one
    assert top.shot == 2

    red, _ = search_shots(root, "a red room", entries, engine=fake, limit=1)
    assert red[0].end <= 2.5 and red[0].shot == 1

    face, _ = search_shots(root, "a woman face", entries, engine=fake, limit=2)
    assert {Path(hit.entry.path).name for hit in face} == {"talking.mp4", "portrait.png"}

    annotate(root, _entry(root, "two_rooms.mp4").id, tags=["rooms"])
    only, _ = search_shots(root, "face", search(root, tags=["rooms"]), engine=fake)
    assert all(Path(hit.entry.path).name == "two_rooms.mp4" for hit in only)

    mosaic = search_mosaic(root, hits, "a blue room")
    assert mosaic is not None and mosaic.is_file()
    with Image.open(mosaic) as picture:
        assert picture.width == 4 * 270 and picture.height > 100
    assert search_mosaic(root, [], "x") is None


def test_search_counts_entries_without_vision_data(tmp_path: Path, fake: FakeVisionEngine) -> None:
    root = _indexed_library(tmp_path, fake)
    _picture(root / "footage" / "late.png", (10, 200, 10))
    index_library(root, "footage", transcribe=False, vision=False)  # indexed, never looked at
    hits, missing = search_shots(root, "green", search(root), engine=fake, limit=5)
    assert missing == 1
    assert all(hit.entry.path != "footage/late.png" for hit in hits)
    with pytest.raises(ValueError, match="empty"):
        search_shots(root, "  ", search(root), engine=fake)


def test_library_search_tool_returns_results_and_a_mosaic(
    tmp_path: Path, fake: FakeVisionEngine
) -> None:
    root = _indexed_library(tmp_path, fake)

    async def scenario() -> None:
        async with Client(build_server()) as client:
            found = await client.call_tool(
                "library_search", {"project_dir": str(root), "query": "a blue room", "limit": 4}
            )
            assert not found.is_error
            answer = json.loads(found.content[0].text)
            assert answer["results"][0]["path"] == "footage/two_rooms.mp4"
            assert answer["lines"][0].startswith("#1 ")
            assert answer["mosaic"].startswith("cache/vision/search/")
            assert any(item.type == "image" for item in found.content)

            plain = await client.call_tool("library_search", {"project_dir": str(root)})
            assert json.loads(plain.content[0].text)["count"] == 3
            assert not any(item.type == "image" for item in plain.content)

    anyio.run(scenario)


def test_library_search_by_sentence_needs_the_environment(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _picture(root / "footage" / "wall.png")
    index_library(root, "footage", transcribe=False)

    async def scenario() -> None:
        async with Client(build_server()) as client:
            found = await client.call_tool(
                "library_search", {"project_dir": str(root), "query": "a wall"}
            )
            assert found.is_error and "vision" in found.content[0].text

    anyio.run(scenario)


# ------------------------------------------------------------------------------ product versions


def _brand_with_versions(root: Path) -> None:
    _picture(root / "brand" / "current.png", (255, 0, 0))
    _picture(root / "brand" / "old.png", (0, 255, 0))
    brand = Brand(
        name="Acme",
        product_versions=[
            ProductVersion(name="current", photos=["brand/current.png"]),
            ProductVersion(name="old", photos=["brand/old.png", "brand/missing.png"]),
        ],
    )
    (root / "brand.yaml").write_text(
        yaml.safe_dump(brand.model_dump(mode="json", exclude_none=True)), encoding="utf-8"
    )


def test_brand_product_versions_accept_text_or_photos(tmp_path: Path) -> None:
    assert Brand(name="A", product_versions="the old box is white").version_references() == []
    assert Brand(name="A", product_versions="the old box is white").version_text() == (
        "the old box is white"
    )
    brand = Brand.model_validate(
        {"name": "A", "product_versions": [{"name": "v2", "photos": ["a.jpg", "b.jpg"]}]}
    )
    assert brand.version_references()[0].photos == ["a.jpg", "b.jpg"]
    assert brand.version_text() == "v2 (2 photos)"
    root = _project(tmp_path)
    _brand_with_versions(root)
    versions = brand_versions(root)
    assert [item.name for item in versions] == ["current", "old"]
    photos = version_photos(root, versions)
    assert {name: len(files) for name, files in photos.items()} == {"current": 1, "old": 1}
    assert references_signature(photos) != ""


def test_match_frames_picks_the_closest_version_with_a_confidence() -> None:
    references = [
        ProductReference("current", np.array([[1.0, 0.0, 0.0]], dtype=np.float32)),
        ProductReference("old", np.array([[0.0, 1.0, 0.0]], dtype=np.float32)),
    ]
    frames = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.7, 0.7, 0.0]], dtype=np.float32
    )
    frames[3] /= np.linalg.norm(frames[3])
    matches = match_frames(frames, references)
    assert matches[0].version == "current" and matches[0].confidence == 1.0
    assert matches[1].version == "old"
    assert matches[2] is None
    assert matches[3] is not None and matches[3].confidence == pytest.approx(0.5, abs=0.05)
    assert match_frames(frames, [])[0] is None


def test_product_version_is_guessed_per_shot_and_a_human_annotation_wins(
    tmp_path: Path, fake: FakeVisionEngine
) -> None:
    root = _project(tmp_path)
    _brand_with_versions(root)
    _clip(
        root / "footage" / "unboxing.mp4",
        [((255, 0, 0), 2.0, None), ((0, 255, 0), 2.0, None), ((10, 10, 10), 2.0, None)],
    )
    _picture(root / "footage" / "pack.png", (255, 0, 0))
    index_library(root, "footage", transcribe=False, vision=True, vision_engine=fake)

    clip = _entry(root, "unboxing.mp4")
    guesses = [(shot.product_version, shot.product_confidence) for shot in clip.shots]
    assert [name for name, _ in guesses] == ["current", "old", None]
    assert all(confidence >= 0.9 for name, confidence in guesses if name)
    assert clip.auto_product is not None and clip.auto_product.frames > 0
    assert _entry(root, "pack.png").auto_product.version == "current"

    names = lambda entries: sorted(Path(entry.path).name for entry in entries)  # noqa: E731
    assert names(search(root, product_version="current")) == ["pack.png", "unboxing.mp4"]
    assert names(search(root, product_version="old")) == ["unboxing.mp4"]

    # A human says the clip shows the old product: that wins over the photos.
    annotate(root, clip.id, product_version="old")
    assert names(search(root, product_version="current")) == ["pack.png"]
    assert names(search(root, product_version="old")) == ["unboxing.mp4"]
    annotate(root, clip.id, product_version="v9")
    assert names(search(root, product_version="old")) == []

    # By sentence, the shots of the wanted version only (no human annotation on that entry).
    annotate(root, clip.id, product_version="current")
    hits, _ = search_shots(root, "red", search(root), engine=fake, product_version="current")
    assert {hit.entry.path for hit in hits} == {"footage/pack.png", "footage/unboxing.mp4"}


# ------------------------------------------------------------------------------- face framing


def test_face_track_smooths_follows_a_crop_and_reads_the_window() -> None:
    from adanvil.vision.engine import Face

    times = [float(index) for index in range(6)]
    centres = [0.30, 0.31, 0.90, 0.32, 0.33, 0.34]  # one wild detection
    faces = [[Face((x - 0.05, 0.2, x + 0.05, 0.4))] for x in centres]
    data = VisionData.build("m", times, [(100, 200)] * 6, None, faces)

    track = face_track(data, start=0.0, end=5.0)
    assert track is not None and track.single and track.height == pytest.approx(0.2)
    assert all(0.28 < knot[1] < 0.5 for knot in track.knots)  # the outlier did not drag the crop

    still = face_track(data, start=0.0, end=1.0)
    assert still is not None and still.still  # barely moves inside the window

    cropped = face_track(data, start=0.0, end=1.0, crop=(20, 0, 50, 200), source_size=(100, 200))
    assert cropped is not None
    assert cropped.knots[0][1] == pytest.approx((0.305 * 100 - 20) / 50, abs=0.05)
    assert cropped.height == pytest.approx(0.2)

    two = VisionData.build(
        "m", [0.0], [(100, 200)], None, [[Face((0.1, 0.1, 0.3, 0.3)), Face((0.6, 0.1, 0.8, 0.3))]]
    )
    track_two = face_track(two)
    assert track_two is not None and not track_two.single
    assert face_track(VisionData.build("m", [0.0], [(1, 1)], None, [[]])) is None


def test_crop_offset_and_expressions() -> None:
    # face at 70% of a 1000 px wide picture, wanted at the middle of a 400 px frame: x = 500
    assert crop_offset(0.7, 0.5, 1000, 400) == pytest.approx(500)
    assert crop_offset(0.1, 0.5, 1000, 400) == 0.0  # clamped to the picture
    assert crop_offset(0.95, 0.5, 1000, 400) == pytest.approx(600)
    assert crop_offset(0.5, 0.38, 1000, 1000) == 0.0  # picture as big as the frame

    track = FaceTrack(((0.0, 0.2, 0.3), (2.0, 0.6, 0.5)), 0.2, 0.5)
    expression = track_expression(track, 1)
    assert expression.startswith("if(lt(t,0),") and "(t-0)/2" in expression
    assert track_expression(FaceTrack(((0.0, 0.4, 0.3),), 0.2, 0.5), 2) == "0.3"
    graph = face_cover(1080, 1920, Framing(track, 0.38, 1.2))
    assert graph.startswith("scale=1296:2304:force_original_aspect_ratio=increase,crop=1080:1920:")
    assert "0.38*1920" in graph


def test_align_zooms_enlarges_the_smaller_face_up_to_a_limit() -> None:
    big = FaceTrack(((0.0, 0.5, 0.4),), 0.30, 0.5625)
    small = FaceTrack(((0.0, 0.5, 0.4),), 0.20, 0.5625)
    zooms = align_zooms(big, small, 1080, 1920)
    assert zooms[0] == 1.0 and zooms[1] == pytest.approx(1.5, abs=0.01)
    tiny = FaceTrack(((0.0, 0.5, 0.4),), 0.05, 0.5625)
    assert align_zooms(big, tiny, 1080, 1920)[1] == pytest.approx(1.6)
    assert face_share(big, 1080, 1920) == pytest.approx(0.30)
    wide = FaceTrack(((0.0, 0.5, 0.4),), 0.3, 1.7778)  # a landscape picture is cut at the sides
    assert face_share(wide, 1080, 1920) == pytest.approx(0.3)


def test_face_focus_text_is_accepted_dumped_and_validated() -> None:
    clip = ClipSegment.model_validate(
        {"type": "clip", "source": "a.mp4", "in": 0, "out": 2, "focus": "face"}
    )
    assert clip.face_focus and clip.focus == (0.5, 0.5) and clip.face_y == DEFAULT_FACE_Y
    dumped = clip.model_dump(mode="json", by_alias=True)
    assert dumped["focus"] == "face" and "face_focus" not in dumped
    assert ClipSegment.model_validate(dumped).face_focus
    plain = ClipSegment(source="a.mp4", in_=0, out=2)
    assert "face_y" not in plain.model_dump(mode="json") and not plain.face_focus
    with pytest.raises(ValueError, match="face"):
        ClipSegment.model_validate({"source": "a.mp4", "in": 0, "out": 2, "focus": "centre"})
    with pytest.raises(ValueError):
        ClipSegment.model_validate({"source": "a.mp4", "in": 0, "out": 2, "face_y": 1.5})

    board = Storyboard.model_validate(
        {
            "name": "faces",
            "format": {"width": W, "height": H, "fps": 10},
            "segments": [
                {
                    "type": "clip",
                    "source": "a.mp4",
                    "in": 0,
                    "out": 2,
                    "focus": "face",
                    "face_y": 0.4,
                },
                {"type": "image", "source": "p.png", "duration": 2, "focus": "face"},
                {
                    "type": "reveal",
                    "duration": 3,
                    "before": {"source": "p.png", "focus": "face"},
                    "after": {"source": "q.png"},
                    "align_faces": False,
                },
            ],
        }
    )
    assert board.segments[0].face_y == 0.4 and board.segments[1].face_focus
    assert board.segments[2].before.face_focus and board.segments[2].align_faces is False
    reveal = RevealSegment.model_validate(
        {
            "duration": 3,
            "before": {"source": "a.png"},
            "after": {"source": "b.png"},
            "focus": "face",
        }
    )
    assert reveal.zoom_centre() == (0.5, DEFAULT_FACE_Y)
    assert validate_storyboard is not None


def test_clip_focus_face_centres_a_moving_face(tmp_path: Path, fake: FakeVisionEngine) -> None:
    root = _project(tmp_path)
    _moving_face_clip(root / "footage" / "walk.mp4")
    seg = ClipSegment.model_validate(
        {"type": "clip", "source": "footage/walk.mp4", "in": 0, "out": 3.5, "focus": "face"}
    )
    cache = Cache(root / "cache")
    output = render_segment(seg, root=root, cache=cache, fmt=FORMAT)
    for index in (1, 17, 33):
        blob = _blob(_frame_at(output, index))
        assert blob is not None
        assert blob[0] == pytest.approx(0.5, abs=0.09), index  # centred while it walks
        assert blob[1] == pytest.approx(DEFAULT_FACE_Y, abs=0.09), index
    # the same clip with a plain centred focus shows the face sliding across the frame
    plain = ClipSegment(source="footage/walk.mp4", in_=0, out=3.5)
    plain_output = render_segment(plain, root=root, cache=cache, fmt=FORMAT)
    seen = [_blob(_frame_at(plain_output, index)) for index in (1, 17, 33)]
    assert any(blob is None or abs(blob[0] - 0.5) > 0.2 for blob in seen)


def test_clip_focus_face_uses_face_y_and_the_crop_window(
    tmp_path: Path, fake: FakeVisionEngine
) -> None:
    root = _project(tmp_path)
    # a wide clip: the face sits at the left third, high in the frame
    wide = root / "footage" / "wide.mp4"
    subprocess.run(
        [
            ffmpeg_path(), "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi", "-t", "2",
            "-i", f"color=c=0x203040:s=640x360:r=10,format=yuv420p,"
            f"drawbox=x=150:y=40:w=60:h=70:color={_hex(FACE)}:t=fill",
            "-c:v", "libx264", str(wide),
        ],
        check=True,
    )  # fmt: skip
    seg = ClipSegment.model_validate(
        {"type": "clip", "source": "footage/wide.mp4", "in": 0, "out": 1.5,
         "focus": "face", "face_y": 0.2, "crop": {"x": 40, "y": 0, "w": 400, "h": 360}}
    )  # fmt: skip
    output = render_segment(seg, root=root, cache=Cache(root / "cache"), fmt=FORMAT)
    blob = _blob(_frame_at(output, 3))
    assert blob is not None
    assert blob[0] == pytest.approx(0.5, abs=0.1)  # the crop moved over to the face
    assert blob[1] == pytest.approx(0.2, abs=0.1)


def test_image_focus_face_centres_the_face_and_falls_back_without_data(
    tmp_path: Path, fake: FakeVisionEngine
) -> None:
    root = _project(tmp_path)
    _picture(root / "footage" / "a.png", (40, 60, 90), (230, 110, 60, 70), size=(432, 384))
    seg = ImageSegment.model_validate(
        {"type": "image", "source": "footage/a.png", "duration": 1, "focus": "face", "fit": "cover"}
    )
    output = render_segment(seg, root=root, cache=Cache(root / "cache"), fmt=FORMAT)
    blob = _blob(_frame_at(output, 2))
    assert blob is not None and blob[0] == pytest.approx(0.5, abs=0.1)
    assert blob[1] == pytest.approx(DEFAULT_FACE_Y, abs=0.1)

    # no vision environment: the plain centred crop, no failure
    with vision_client.use_engine(None):
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("ADANVIL_VISION", "off")
            other = Cache(root / "cache2")
            plain = render_segment(seg, root=root, cache=other, fmt=FORMAT)
    assert _frame_at(plain, 2).shape == (H, W, 3)
    # (the face data was stored by the first render, so a later render reuses it)
    assert load_vision(root, vision_relative(Cache.file_key(root / "footage" / "a.png")))


def test_reveal_aligns_both_faces_to_the_same_place_and_size(
    tmp_path: Path, fake: FakeVisionEngine
) -> None:
    root = _project(tmp_path)
    # the same woman, closer in the second picture (square pictures: the crop has room across)
    _picture(root / "footage" / "before.png", (60, 60, 60), (200, 190, 60, 76), size=(600, 600))
    _picture(root / "footage" / "after.png", (90, 90, 90), (330, 170, 90, 110), size=(600, 600))
    spec = {
        "type": "reveal",
        "duration": 2.0,
        "hold_before": 0.4,
        "sweep": 0.5,
        "divider": None,
        "before": {"source": "footage/before.png"},
        "after": {"source": "footage/after.png"},
    }
    aligned = render_segment(
        RevealSegment.model_validate(spec), root=root, cache=Cache(root / "cache"), fmt=FORMAT
    )
    first, last = _blob(_frame_at(aligned, 1)), _blob(_frame_at(aligned, 18))
    assert first is not None and last is not None
    for blob in (first, last):
        assert blob[0] == pytest.approx(0.5, abs=0.08)
        assert blob[1] == pytest.approx(DEFAULT_FACE_Y, abs=0.08)
    assert last[2] == pytest.approx(first[2], rel=0.15)  # same size on both sides

    loose = render_segment(
        RevealSegment.model_validate({**spec, "align_faces": False}),
        root=root,
        cache=Cache(root / "cache"),
        fmt=FORMAT,
    )
    a, b = _blob(_frame_at(loose, 1)), _blob(_frame_at(loose, 18))
    assert a is not None and b is not None and abs(a[2] - b[2]) > 0.04  # sizes differ


def test_reveal_is_left_alone_when_a_side_has_two_faces(
    tmp_path: Path, fake: FakeVisionEngine
) -> None:
    root = _project(tmp_path)
    _picture(root / "footage" / "one.png", (60, 60, 60), (30, 40, 40, 50), size=(432, 384))
    crowd = _picture(root / "footage" / "two.png", (90, 90, 90), (20, 40, 40, 50), size=(432, 384))
    with Image.open(crowd) as opened:
        drawing = opened.copy()
    drawing.paste(Image.new("RGB", (40, 50), FACE), (300, 200))
    drawing.save(crowd)
    seg = RevealSegment.model_validate(
        {
            "type": "reveal", "duration": 2.0, "hold_before": 0.4, "sweep": 0.5, "divider": None,
            "before": {"source": "footage/one.png"}, "after": {"source": "footage/two.png"},
        }
    )  # fmt: skip
    from adanvil.media.reveal import _aligned_framings

    assert _aligned_framings(seg, root, FORMAT, ["image", "image"]) == [None, None]


# ----------------------------------------------------------------------------------- compare_pair


def test_compare_pair_uses_the_face_detector_when_there_is_one(
    tmp_path: Path, fake: FakeVisionEngine
) -> None:
    root = _project(tmp_path)
    # no skin-coloured blob for the heuristic to find, but the fake detector sees a face
    first = _picture(root / "footage" / "x.png", (30, 30, 30), (30, 40, 40, 50), size=(200, 300))
    second = _picture(
        root / "footage" / "y.png", (200, 200, 200), (60, 80, 30, 40), size=(200, 300)
    )
    result, preview = compare_pair(root, first, second)
    assert result["face_source"] == "detector" and result["region"] == "face"
    assert result["faces"]["before"] and preview.is_file()
    finder = detector_face_finder(fake)
    with Image.open(first) as opened:
        found = finder(opened)
    assert found is not None and found["box"][0] == pytest.approx(30, abs=4)

    with vision_client.use_engine(None):
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("ADANVIL_VISION", "off")
            fallback, _ = compare_pair(root, first, second)
    assert fallback["face_source"] == "skin heuristic"


# ------------------------------------------------------------------------------------ real worker


def _real_environment() -> bool:
    return vision_client.configured_interpreter() is not None


@pytest.mark.skipif(not _real_environment(), reason="the vision environment is not installed")
def test_real_worker_embeds_text_and_images_and_finds_a_face(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ADANVIL_VISION", raising=False)
    engine = vision_client.get_engine()
    assert engine is not None
    pictures = [
        _picture(tmp_path / "red.png", (200, 20, 20)),
        _picture(tmp_path / "blue.png", (20, 30, 200)),
    ]
    images = engine.embed_images(pictures)
    texts = engine.embed_texts(["a red picture", "a blue picture", "une image bleue"])
    assert images.shape[0] == 2 and texts.shape[0] == 3 and images.shape[1] == texts.shape[1]
    similarity = images @ texts.T
    assert similarity[0, 0] > similarity[0, 1] and similarity[1, 1] > similarity[1, 0]
    assert similarity[1, 2] > similarity[0, 2]  # a French sentence works too
    assert engine.detect_faces(pictures)[0].faces == []
    assert os.environ.get("ADANVIL_VISION") is None or True
