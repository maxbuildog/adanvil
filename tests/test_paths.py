from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from adanvil.brand import load_brand
from adanvil.library import index_library, load_catalog
from adanvil.media import Cache, ClipSegment, Format, render_segment
from adanvil.project import init_project, resolve_in_project
from adanvil.storyboard import validate_storyboard


def _link_directory(target: Path, link: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError) as symlink_error:
        if os.name != "nt":
            pytest.skip(f"directory symlinks are unavailable: {symlink_error}")

    result = subprocess.run(
        [os.environ.get("COMSPEC", "cmd.exe"), "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        pytest.skip(f"directory links are unavailable: {result.stderr or result.stdout}")


def _unlink_directory_link(link: Path) -> None:
    is_junction = getattr(link, "is_junction", lambda: False)
    if link.is_symlink() or is_junction():
        link.unlink()


@pytest.mark.parametrize("written", ["../x", "/abs", r"C:\x", "footage/../../x"])
def test_storyboard_path_escapes_are_reported(written: str, tmp_path: Path) -> None:
    root = init_project(tmp_path / "project", "Paths")
    report = validate_storyboard(
        {
            "name": "escape",
            "segments": [{"type": "clip", "source": written, "in": 0, "out": 1}],
        },
        root=root,
        brand=load_brand(root / "brand.yaml"),
    )

    assert not report.ok
    assert any(issue.code == "path_escape" for issue in report.errors)


def test_symlinked_footage_is_indexed_validated_and_rendered(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    root = init_project(tmp_path / "project", "Linked footage")
    external = tmp_path / "shared-rushes"
    source = external / "raw" / "x.mp4"
    source.parent.mkdir(parents=True)
    shutil.copy2(Path(str(media["sdr_portrait.mp4"]["path"])), source)

    shutil.rmtree(root / "footage")
    _link_directory(external, root / "footage")

    try:
        indexed = index_library(root, text_scan=False)
        assert indexed.errors == []
        entry = next(iter(load_catalog(root).entries.values()))
        assert entry.path == "footage/raw/x.mp4"

        storyboard = {
            "name": "linked_source",
            "segments": [
                {"type": "clip", "source": entry.path, "in": 0, "out": 1, "gamma": None}
            ],
        }
        report = validate_storyboard(
            storyboard,
            root=root,
            brand=load_brand(root / "brand.yaml"),
        )
        assert report.ok, report.errors

        segment = ClipSegment.model_validate(storyboard["segments"][0])
        rendered = render_segment(
            segment,
            root=root,
            cache=Cache(root / "cache"),
            fmt=Format(width=1080, height=1920, fps=30),
        )
        assert rendered.is_file()
    finally:
        _unlink_directory_link(root / "footage")


def test_resolve_in_project_is_lexical(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    link_target = tmp_path / "outside"
    link_target.mkdir()
    _link_directory(link_target, root / "footage")

    try:
        resolved = resolve_in_project(root, "footage/raw/x.mov")
        assert resolved == root / "footage" / "raw" / "x.mov"
    finally:
        _unlink_directory_link(root / "footage")
