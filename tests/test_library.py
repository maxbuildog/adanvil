from __future__ import annotations

import os
import shutil
import time
import unicodedata
from pathlib import Path

import pytest
from PIL import Image

from adanvil import ffmpeg
from adanvil.library import Annotation, annotate, entry_id, index_library, load_catalog, search
from adanvil.project import init_project, resolve_in_project


def _copy_fixture_media(media: dict[str, dict[str, object]], root: Path) -> list[Path]:
    footage = root / "footage"
    copied: list[Path] = []
    for name, metadata in media.items():
        source = Path(str(metadata["path"]))
        destination = footage / name
        shutil.copy2(source, destination)
        copied.append(destination)

    nested = footage / "sub folder"
    nested.mkdir()
    nfd_name = unicodedata.normalize("NFD", "café") + ".mp4"
    nfd_path = nested / nfd_name
    shutil.copy2(Path(str(media["sdr_portrait.mp4"]["path"])), nfd_path)
    copied.append(nfd_path)

    pipe_path = nested / "pipe|clip.mp4"
    try:
        shutil.copy2(Path(str(media["sdr_portrait.mp4"]["path"])), pipe_path)
    except OSError:
        pass
    else:
        copied.append(pipe_path)

    (footage / "broken.mp4").write_text("not media", encoding="utf-8")
    return copied


def _entry_by_name(root: Path, name: str):
    return next(
        entry for entry in load_catalog(root).entries.values() if Path(entry.path).name == name
    )


def _overlaps(left: tuple[int, int, int, int], right: list[int]) -> bool:
    return not (
        left[2] <= right[0]
        or right[2] <= left[0]
        or left[3] <= right[1]
        or right[3] <= left[1]
    )


def test_annotation_usable_accepts_agent_shapes_and_rejects_other_shapes() -> None:
    assert Annotation(usable=[[1, 2, "first"], [3, 4]]).usable == [
        (1.0, 2.0, "first"),
        (3.0, 4.0, ""),
    ]
    assert Annotation(usable=[{"start": 1, "end": 2, "note": "object"}]).usable == [
        (1.0, 2.0, "object")
    ]

    with pytest.raises(ValueError, match="accepted shapes"):
        Annotation(usable=True)
    with pytest.raises(ValueError, match="accepted shapes"):
        Annotation(usable=[[1, 2, "note", "extra"]])


def test_index_annotations_and_incremental_behaviour(
    tmp_path: Path, media: dict[str, dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    init_project(tmp_path, "Library")
    _copy_fixture_media(media, tmp_path)

    first_started = time.perf_counter()
    first = index_library(tmp_path)
    first_seconds = time.perf_counter() - first_started
    assert first.added
    assert not first.updated
    assert any(Path(error["path"]).name == "broken.mp4" for error in first.errors)

    catalog = load_catalog(tmp_path)
    ref = _entry_by_name(tmp_path, "ref_ad.mp4")
    assert [(shot.start, shot.end) for shot in ref.shots] == pytest.approx(
        [(0, 2), (2, 4), (4, 6)], abs=0.1
    )
    burnt = _entry_by_name(tmp_path, "burnt_text.mp4")
    manifest_box = media["burnt_text.mp4"]["text_bbox"]
    assert any(_overlaps(box, manifest_box) for box in burnt.text_boxes)
    assert _entry_by_name(tmp_path, "dark_portrait.mp4").yavg is not None
    assert _entry_by_name(tmp_path, "dark_portrait.mp4").yavg < 60
    rotated = _entry_by_name(tmp_path, "rotated.mp4")
    assert (rotated.width, rotated.height) == (1080, 1920)

    ref_sheet = tmp_path / ref.sheet
    with Image.open(ref_sheet) as image:
        assert image.size == (1080, 1440)
    product_sheet = tmp_path / _entry_by_name(tmp_path, "product.png").sheet
    with Image.open(product_sheet) as image:
        assert image.width == 540
        assert image.height == 675
    assert any("café" in entry.path for entry in catalog.entries.values())
    assert any("pipe|clip" in entry.path for entry in catalog.entries.values()) or not (
        tmp_path / "footage" / "sub folder" / "pipe|clip.mp4"
    ).exists()

    # Broken files are reported again only when they are actually probed; remove
    # the fixture before checking the no-ffmpeg unchanged fast path.
    (tmp_path / "footage" / "broken.mp4").unlink()
    calls = 0
    original_run = ffmpeg.run

    def counted_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_run(*args, **kwargs)

    monkeypatch.setattr(ffmpeg, "run", counted_run)
    unchanged_started = time.perf_counter()
    second = index_library(tmp_path)
    unchanged_seconds = time.perf_counter() - unchanged_started
    assert not calls
    assert not second.added
    assert not second.updated
    assert len(second.unchanged) == len(catalog.entries)
    print(f"LIBRARY_INDEX_TIMES first={first_seconds:.2f}s unchanged={unchanged_seconds:.2f}s")

    target_path = tmp_path / "footage" / "ref_ad.mp4"
    target = _entry_by_name(tmp_path, "ref_ad.mp4")
    annotated = annotate(
        tmp_path,
        target.id,
        description="Une démonstration café",
        tags=["hero", "café"],
        people=["Alice"],
        quality=5,
        product_visible=True,
        notes="warm opening",
    )
    assert annotated.annotation.updated
    old_updated = annotated.annotation.updated
    stat = target_path.stat()
    os.utime(target_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
    changed = index_library(tmp_path, text_scan=False)
    assert target.id in changed.updated
    assert _entry_by_name(tmp_path, "ref_ad.mp4").annotation.tags == ["hero", "café"]
    assert _entry_by_name(tmp_path, "ref_ad.mp4").annotation.updated == old_updated

    deleted_path = tmp_path / "footage" / "music.m4a"
    deleted_id = entry_id("footage/music.m4a")
    deleted_path.unlink()
    removed = index_library(tmp_path, text_scan=False)
    assert deleted_id in removed.removed
    assert deleted_id not in load_catalog(tmp_path).entries

    assert search(tmp_path, tags=["HERO"])[0].id == target.id
    assert search(tmp_path, text="DEMONSTRATION CAFE")[0].id == target.id
    assert search(tmp_path, min_quality=5)[0].id == target.id
    assert search(tmp_path, product_visible=True)[0].id == target.id
    annotate(tmp_path, target.id, exclude=True, exclude_reason="brand mismatch")
    assert not search(tmp_path, tags=["hero"])
    assert search(tmp_path, tags=["hero"], include_excluded=True)[0].id == target.id


def test_project_init_is_idempotent_and_paths_are_safe(tmp_path: Path) -> None:
    root = init_project(tmp_path / "project", "Demo")
    project_before = (root / "adanvil.json").read_text(encoding="utf-8")
    brand_before = (root / "brand.yaml").read_text(encoding="utf-8")
    init_project(root, "Changed name")
    assert (root / "adanvil.json").read_text(encoding="utf-8") == project_before
    assert (root / "brand.yaml").read_text(encoding="utf-8") == brand_before
    assert (root / "catalog.json").exists()
    assert resolve_in_project(root, "footage/a.mp4") == (root / "footage" / "a.mp4").resolve()
    with pytest.raises(ValueError):
        resolve_in_project(root, "../outside")
    with pytest.raises(ValueError):
        resolve_in_project(root, str(Path("C:/outside")))


def test_library_overview_tree_mosaic_and_symlink(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    from adanvil.library import library_overview

    init_project(tmp_path, "overview")
    footage = tmp_path / "footage"
    (footage / "B-ROLLS").mkdir()
    shutil.copy2(Path(str(media["ref_ad.mp4"]["path"])), footage / "a.mp4")
    shutil.copy2(Path(str(media["ref_ad.mp4"]["path"])), footage / "B-ROLLS" / "b.mp4")
    (footage / "cache").mkdir()
    shutil.copy2(Path(str(media["ref_ad.mp4"]["path"])), footage / "cache" / "skip.mp4")
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.copy2(Path(str(media["sdr_portrait.mp4"]["path"])), outside / "c.mp4")
    try:
        os.symlink(outside, footage / "linked", target_is_directory=True)
        os.symlink(footage, outside / "loop", target_is_directory=True)
        linked = True
    except (OSError, NotImplementedError):
        linked = False

    result = library_overview(tmp_path, "footage", depth=2, mosaic=True)

    tree = result["tree"]
    expected = 3 if linked else 2
    assert tree["videos"] == expected
    assert tree["files"] == expected
    assert tree["total_video_seconds"] > 10
    assert tree["first_files"] == ["a.mp4"]
    names = {folder["path"].rsplit("/", 1)[-1]: folder for folder in tree["folders"]}
    assert "cache" not in names
    assert names["B-ROLLS"]["videos"] == 1
    assert ("linked" in names) == linked
    mosaic = tmp_path / result["mosaic"]
    assert mosaic.is_file() and mosaic.parent == tmp_path / "cache" / "overview"
    with Image.open(mosaic) as image:
        assert image.width == 6 * 180
    assert (tmp_path / "cache" / "overview.json").is_file()

    flat = library_overview(tmp_path, "footage", depth=0, mosaic=False)
    assert "folders" not in flat["tree"] and "mosaic" not in flat


def test_library_overview_pages_big_mosaic(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    from adanvil.library import library_overview

    init_project(tmp_path, "overview-big")
    source = Path(str(media["sdr_portrait.mp4"]["path"]))
    for index in range(61):
        shutil.copy2(source, tmp_path / "footage" / f"clip{index:02d}.mp4")
    first = library_overview(tmp_path, "footage")
    assert first["tree"]["videos"] == 61
    assert (first["page"], first["pages"]) == (1, 2)
    assert len(first["mosaic_files"]) == 48
    assert "page=2" in first["note"]
    second = library_overview(tmp_path, "footage", page=2)
    assert len(second["mosaic_files"]) == 13
    assert second["mosaic"] != first["mosaic"]
    assert (tmp_path / second["mosaic"]).is_file()
    assert set(first["mosaic_files"]).isdisjoint(second["mosaic_files"])
