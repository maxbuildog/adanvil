"""Security 0.3.1, part A: project confinement, jobs, export, MCP tool arguments.

Each test reproduces an attack an agent steered by a prompt injection (or a project folder
received from someone else) could try, and shows it is refused; the normal use is checked too.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client
from PIL import Image
from pydantic import ValidationError

from adanvil.batch import write_manifest
from adanvil.export import MANIFEST, _prepare_folder, _remove_file
from adanvil.jobs import is_job_id, start_job
from adanvil.library import index_library, load_catalog
from adanvil.library.catalog import EntrySpeech, save_catalog
from adanvil.library.index import _iter_media
from adanvil.library.overview import _walk
from adanvil.project import (
    check_managed_folders,
    check_project_dir,
    init_project,
    read_project_text,
    resolve_in_project,
)
from adanvil.server import _write_json, build_server
from adanvil.speech import store as speech_store
from adanvil.storyboard import Storyboard, validate_storyboard
from adanvil.storyboard.patch import rename_storyboard

# --------------------------------------------------------------------------------- helpers


def _call(tool: str, arguments: dict[str, Any]):
    async def run():
        async with Client(build_server()) as client:
            return await client.call_tool(tool, arguments)

    return anyio.run(run)


def _text(result) -> str:
    return " ".join(item.text for item in result.content if item.type == "text")


def _has_image(result) -> bool:
    return any(item.type == "image" for item in result.content)


def _file_link(target: Path, link: Path) -> None:
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")


def _folder_link(target: Path, link: Path) -> None:
    """A directory link: a symlink, or a junction on Windows without the privilege."""

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


def _unlink_folder(link: Path) -> None:
    is_junction = getattr(link, "is_junction", lambda: False)
    if link.is_symlink() or is_junction():
        link.unlink() if link.is_symlink() else os.rmdir(link)


def _fake_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / "users" / "max"
    home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


def _board(**changes: Any) -> dict[str, Any]:
    board: dict[str, Any] = {
        "name": "ad",
        "segments": [{"type": "clip", "source": "footage/a.mp4", "in": 0, "out": 1}],
    }
    board.update(changes)
    return board


# ----------------------------------------------------------------- A1 any folder is a project


def test_a1_home_and_its_parents_are_never_a_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = _fake_home(monkeypatch, tmp_path)
    secret = home / ".config" / "gh" / "hosts.yml"
    secret.parent.mkdir(parents=True)
    secret.write_text("oauth_token: ghp_secret", encoding="utf-8")

    # The reproduced attack: make the home folder a project, then read a token through it.
    made = _call("project_init", {"project_dir": str(home), "name": "x"})
    assert made.is_error
    assert "home folder" in _text(made)
    assert not (home / "adanvil.json").exists()
    read = _call("project_read_text", {"project_dir": str(home), "path": ".config/gh/hosts.yml"})
    assert read.is_error
    assert "ghp_secret" not in _text(read)

    for refused in (home, home.parent, tmp_path, Path(home.anchor)):
        with pytest.raises(ValueError):
            check_project_dir(str(refused))
        assert _call("project_init", {"project_dir": str(refused), "name": "x"}).is_error
    assert not (tmp_path / "adanvil.json").exists()


def test_a1_a_sub_folder_of_home_is_a_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = _fake_home(monkeypatch, tmp_path)
    project = home / "AdAnvil" / "my-ad"
    made = _call("project_init", {"project_dir": str(project), "name": "ok"})
    assert not made.is_error, _text(made)
    assert (project / "adanvil.json").is_file()
    assert check_project_dir(str(project)) == Path(os.path.realpath(project))


def test_a1_network_paths_are_refused_before_anything_touches_them() -> None:
    for written in ("//server/share/project", "//server/share", "\\\\server\\share\\project"):
        with pytest.raises(ValueError):
            check_project_dir(written)
    result = _call("project_init", {"project_dir": "//server/share/project", "name": "x"})
    assert result.is_error
    assert "network" in _text(result) or "absolute" in _text(result)


def test_a1_projects_dir_confines_every_tool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    projects = tmp_path / "projects"
    projects.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    monkeypatch.setenv("ADANVIL_PROJECTS_DIR", str(projects))

    refused = _call("project_init", {"project_dir": str(outside / "p"), "name": "x"})
    assert refused.is_error
    assert "ADANVIL_PROJECTS_DIR" in _text(refused)
    assert not (outside / "p").exists()
    # the configured folder itself is not a project: a project is strictly inside it
    assert _call("project_init", {"project_dir": str(projects), "name": "x"}).is_error
    assert not (projects / "adanvil.json").exists()

    known = str(outside / "known")
    init_project(outside / "known", "known")
    assert _call("project_read_text", {"project_dir": known, "path": "brand.yaml"}).is_error
    assert _call("job_status", {"job_id": "job-ab12", "project_dir": known}).is_error

    mine = str(projects / "mine")
    made = _call("project_init", {"project_dir": mine, "name": "mine"})
    assert not made.is_error, _text(made)
    assert not _call("project_read_text", {"project_dir": mine, "path": "adanvil.json"}).is_error


def test_a1_a_link_inside_the_projects_dir_cannot_lead_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    projects = tmp_path / "projects"
    projects.mkdir()
    outside = tmp_path / "elsewhere"
    init_project(outside / "stolen", "stolen")
    _folder_link(outside, projects / "door")
    monkeypatch.setenv("ADANVIL_PROJECTS_DIR", str(projects))
    try:
        stolen = str(projects / "door" / "stolen")
        result = _call("project_read_text", {"project_dir": stolen, "path": "adanvil.json"})
        assert result.is_error
        assert "ADANVIL_PROJECTS_DIR" in _text(result)
    finally:
        _unlink_folder(projects / "door")


@pytest.mark.parametrize("name", ["renders", "cache", "brand", "music", "reference", "exports"])
def test_a1_managed_folders_must_be_real_folders(name: str, tmp_path: Path) -> None:
    root = init_project(tmp_path / "project", "Links")
    outside = tmp_path / "outside"
    outside.mkdir()
    managed = root / name
    if managed.is_dir():
        shutil.rmtree(managed)
    _folder_link(outside, managed)
    try:
        with pytest.raises(ValueError, match="real folder"):
            check_managed_folders(root)
        result = _call("project_read_text", {"project_dir": str(root), "path": "adanvil.json"})
        assert result.is_error
        assert f"{name}/" in _text(result)
        assert _call("project_init", {"project_dir": str(root), "name": "again"}).is_error
    finally:
        _unlink_folder(managed)


def test_a1_footage_may_be_a_link_to_another_drive(tmp_path: Path) -> None:
    root = init_project(tmp_path / "project", "Linked footage")
    rushes = tmp_path / "rushes"
    rushes.mkdir()
    shutil.rmtree(root / "footage")
    _folder_link(rushes, root / "footage")
    try:
        check_managed_folders(root)
        result = _call("project_read_text", {"project_dir": str(root), "path": "adanvil.json"})
        assert not result.is_error, _text(result)
    finally:
        _unlink_folder(root / "footage")


# --------------------------------------------------------------------- A2 hidden names


@pytest.mark.parametrize(
    "written",
    [
        ".ssh/id_rsa",
        ".config/gh/hosts.yml",
        "brand/.env",
        "footage/.hidden/a.mp4",
        "a/./.b",
        ".env",
    ],
)
def test_a2_hidden_names_are_refused(written: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hidden"):
        resolve_in_project(tmp_path, written)


def test_a2_normal_names_still_resolve(tmp_path: Path) -> None:
    assert resolve_in_project(tmp_path, "footage/a.mp4") == tmp_path / "footage" / "a.mp4"
    assert resolve_in_project(tmp_path, "./footage//b..c.mp4") == tmp_path / "footage" / "b..c.mp4"
    assert resolve_in_project(tmp_path, ".") == tmp_path


def test_a2_the_library_walk_skips_hidden_files_and_folders(tmp_path: Path) -> None:
    for relative in ("a.mp4", ".hidden.mp4", ".secret/b.mp4", "sub/c.mp4", "sub/.d.mp4"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"not a video")
    assert [path.name for path in _iter_media(tmp_path)] == ["a.mp4", "c.mp4"]
    walked = {directory.name: [f.name for f in files] for directory, files in _walk(tmp_path)}
    assert sorted(name for names in walked.values() for name in names) == ["a.mp4", "c.mp4"]
    assert ".secret" not in walked


# ------------------------------------------------------------------------- A3 read text


def test_a3_only_adanvils_own_files_can_be_read(tmp_path: Path) -> None:
    root = init_project(tmp_path / "project", "Read")
    (root / "notes").mkdir()
    (root / "notes" / "a.txt").write_text("not adanvil's", encoding="utf-8")
    (root / "secret.txt").write_text("top level text", encoding="utf-8")
    (root / "music" / "LICENCES.txt").write_text("licensed", encoding="utf-8")
    for refused in ("secret.txt", "notes/a.txt", "../outside.txt", ".git/config", "src/app.json"):
        with pytest.raises(ValueError):
            read_project_text(root, refused)
    assert read_project_text(root, "music/LICENCES.txt")["content"] == "licensed"
    assert read_project_text(root, "adanvil.json")["content"].startswith("{")
    assert "name" in read_project_text(root, "adanvil.json")["content"]
    assert read_project_text(root, "brand.yaml")["path"] == "brand.yaml"
    assert read_project_text(root, "catalog.json")["path"] == "catalog.json"


def test_a3_max_bytes_is_capped_at_one_megabyte(tmp_path: Path) -> None:
    root = init_project(tmp_path / "project", "Big")
    (root / "music" / "big.txt").write_text("x" * 2_500_000, encoding="utf-8")
    result = read_project_text(root, "music/big.txt", 10_000_000)
    assert len(result["content"]) == 1_000_000
    assert result["truncated"] is True
    assert "limited to 1000000" in str(result["note"])
    small = read_project_text(root, "music/big.txt", 10)
    assert small["content"] == "x" * 10
    assert "limited" not in str(small["note"])


def test_a3_a_text_file_that_is_a_link_to_a_secret_is_not_read(tmp_path: Path) -> None:
    root = init_project(tmp_path / "project", "Linked text")
    secret = tmp_path / "id_rsa.txt"
    secret.write_text("PRIVATE KEY", encoding="utf-8")
    _file_link(secret, root / "brand" / "notes.txt")
    with pytest.raises(ValueError, match="link"):
        read_project_text(root, "brand/notes.txt")


# ------------------------------------------------------------------------ A4 output path


@pytest.mark.parametrize(
    "written",
    [
        "adanvil.json",
        "brand.yaml",
        "../victim.mp4",
        "renders/../adanvil.mp4",
        "/etc/cron.d/evil.mp4",
        "C:/Windows/evil.mp4",
        "\\\\server\\share\\evil.mp4",
        "renders/.hidden/a.mp4",
        "renders/a.json",
        "renders/a.mp4.exe",
        "renders/",
        "other/a.mp4",
        "renders/{name.__class__}.mp4",
        "renders/{0}.mp4",
        "renders/a:b.mp4",
    ],
)
def test_a4_output_path_must_be_a_render_under_renders(written: str) -> None:
    with pytest.raises(ValidationError, match="output.path"):
        Storyboard.model_validate(_board(output={"path": written}))


def test_a4_a_refused_output_path_never_reaches_the_renderer(tmp_path: Path) -> None:
    root = init_project(tmp_path / "project", "Output")
    before = (root / "adanvil.json").read_text(encoding="utf-8")
    board = _board(output={"path": "adanvil.json", "quality": "preview"})
    result = _call("render_start", {"project_dir": str(root), "storyboard": board})
    payload = json.loads(_text(result))
    assert payload["status"] == "rejected"
    assert any("output.path" in issue["message"] for issue in payload["errors"])
    assert (root / "adanvil.json").read_text(encoding="utf-8") == before
    assert not (root / "renders" / "jobs").exists()


@pytest.mark.parametrize(
    "written",
    [
        "renders/{name}.mp4",
        "renders\\{name}.mp4",
        "renders/batch/{name}_9x16.mp4",
        "renders/variants/_hooks/{name}.mp4",
        "renders/final/Spring.MP4",
    ],
)
def test_a4_the_paths_adanvil_builds_itself_stay_valid(written: str) -> None:
    Storyboard.model_validate(_board(output={"path": written}))


def test_a4_renames_variants_and_formats_keep_valid_output_paths() -> None:
    from adanvil.formats import reformat

    document = _board()
    renamed = rename_storyboard(dict(document), "ad_snapped")
    assert Storyboard.model_validate(renamed).output.path == "renders/ad_snapped.mp4"
    moved, _notes = reformat(document, "4:5", root=None, output_dir="renders/batch")
    assert Storyboard.model_validate(moved).output.path.startswith("renders/batch/")


def test_a4_a_storyboard_named_like_a_hidden_file_cannot_make_a_hidden_render() -> None:
    with pytest.raises(ValidationError, match="hidden"):
        Storyboard.model_validate(_board(name=".git"))


# --------------------------------------------------------------- A5 sizes and durations


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"format": {"width": 5000, "height": 1920, "fps": 30}}, "format.width"),
        ({"format": {"width": 1080, "height": 8000, "fps": 30}}, "format.height"),
        ({"format": {"width": 1080, "height": 1920, "fps": 240}}, "format.fps"),
        (
            {"segments": [{"type": "clip", "source": "footage/a.mp4", "in": 0, "out": 700}]},
            "at most 600",
        ),
        (
            {
                "segments": [
                    {
                        "type": "clip",
                        "source": "footage/a.mp4",
                        "in": 0,
                        "out": 2,
                        "freeze": {"hold": 700},
                    }
                ]
            },
            "freeze.hold",
        ),
        (
            {"segments": [{"type": "image", "source": "footage/a.jpg", "duration": 601}]},
            "(at most|less than or equal to) 600",
        ),
        (
            {
                "segments": [
                    {
                        "type": "reveal",
                        "duration": 700,
                        "before": {"source": "footage/a.jpg"},
                        "after": {"source": "footage/b.jpg"},
                    }
                ]
            },
            "(at most|less than or equal to) 600",
        ),
        ({"audio": {"fade_out": 5000}}, "fade_out"),
        ({"overlays": [{"type": "sfx", "source": "pop", "at": 100000}]}, "overlays[0].at"),
    ],
)
def test_a5_sizes_and_durations_have_upper_bounds(changes: dict[str, Any], message: str) -> None:
    # The segment models bound their own fields (``Field(le=...)``) and the storyboard checks
    # the rest: either message names the limit.
    pattern = message if message.startswith("(") else re.escape(message)
    with pytest.raises(ValidationError, match=pattern):
        Storyboard.model_validate(_board(**changes))


def test_a5_the_whole_ad_has_a_bound_too() -> None:
    clips = [{"type": "clip", "source": "footage/a.mp4", "in": 0, "out": 600}] * 7
    with pytest.raises(ValidationError, match="total duration"):
        Storyboard.model_validate(_board(segments=clips))


def test_a5_the_limits_themselves_and_normal_ads_are_valid() -> None:
    biggest = _board(
        format={"width": 4096, "height": 4096, "fps": 120},
        segments=[
            {"type": "clip", "source": "footage/a.mp4", "in": 0, "out": 600},
            {"type": "image", "source": "footage/a.jpg", "duration": 600},
        ],
    )
    assert Storyboard.model_validate(biggest).format.fps == 120
    assert Storyboard.model_validate(_board()).format.width == 1080


def test_a5_a_too_long_segment_is_reported_by_validation(tmp_path: Path) -> None:
    root = init_project(tmp_path / "project", "Bounds")
    from adanvil.brand import load_brand

    report = validate_storyboard(
        _board(format={"width": 9000, "height": 1920, "fps": 30}),
        root=root,
        brand=load_brand(root / "brand.yaml"),
    )
    assert not report.ok
    assert any("format.width" in f"{issue.where} {issue.message}" for issue in report.errors)


# ------------------------------------------------------------------- A6 export cleanup


def _planted_export(tmp_path: Path, files: list[object]) -> tuple[Path, Path, Path]:
    root = init_project(tmp_path / "project", "Export")
    folder = root / "exports" / "ad"
    folder.mkdir(parents=True)
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me", encoding="utf-8")
    (folder / MANIFEST).write_text(
        json.dumps({"adanvil_export": 2, "name": "ad", "mode": "absolute", "files": files}),
        encoding="utf-8",
    )
    return root, folder, victim


def test_a6_a_planted_manifest_cannot_delete_outside_the_export(tmp_path: Path) -> None:
    root, folder, victim = _planted_export(tmp_path, [])
    outside_in_project = root / "adanvil.json"
    (folder / MANIFEST).write_text(
        json.dumps(
            {
                "adanvil_export": 2,
                "name": "ad",
                "mode": "absolute",
                "files": [
                    "../../../victim.txt",  # the reproduced attack
                    "../../adanvil.json",
                    str(victim),
                    victim.as_posix(),
                    "C:/victim.txt",
                    "C:\\victim.txt",
                    "\\\\server\\share\\x",
                    "//server/share/x",
                    ".hidden",
                    "sub/../../../victim.txt",
                    "a:b",
                    "",
                    7,
                    None,
                    "ad.otio",
                ],
            }
        ),
        encoding="utf-8",
    )
    (folder / "ad.otio").write_text("old export", encoding="utf-8")
    _prepare_folder(folder, root, "absolute", False)
    assert victim.read_text(encoding="utf-8") == "keep me"
    assert outside_in_project.is_file()
    assert not (folder / "ad.otio").exists()  # what the manifest legitimately lists is cleaned


def test_a6_manifest_entries_that_are_not_a_list_delete_nothing(tmp_path: Path) -> None:
    root, folder, victim = _planted_export(tmp_path, [])
    planted = {"mode": "absolute", "files": "ad.otio"}
    (folder / MANIFEST).write_text(json.dumps(planted), encoding="utf-8")
    (folder / "a").write_text("single letter file", encoding="utf-8")
    _prepare_folder(folder, root, "absolute", False)
    assert (folder / "a").is_file()
    (folder / MANIFEST).write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    _prepare_folder(folder, root, "absolute", True)
    assert victim.is_file()


def test_a6_a_link_in_the_folder_is_removed_never_followed(tmp_path: Path) -> None:
    root, folder, victim = _planted_export(tmp_path, ["leak.txt"])
    _file_link(victim, folder / "leak.txt")
    _prepare_folder(folder, root, "absolute", False)
    assert not os.path.lexists(folder / "leak.txt")
    assert victim.read_text(encoding="utf-8") == "keep me"


@pytest.mark.parametrize("overwrite", [False, True])
def test_a6_a_sub_folder_that_leads_out_is_not_cleaned(tmp_path: Path, overwrite: bool) -> None:
    root, folder, victim = _planted_export(tmp_path, ["overlays/overlay_00_caption.png"])
    other = tmp_path / "pictures"
    other.mkdir()
    precious = other / "overlay_00_caption.png"
    precious.write_bytes(b"precious")
    _folder_link(other, folder / "overlays")
    try:
        if overwrite:
            (folder / MANIFEST).unlink()
            (folder / "stray.otio").write_text("x", encoding="utf-8")
        _prepare_folder(folder, root, "absolute", overwrite)
        assert precious.read_bytes() == b"precious"
        assert victim.is_file()
    finally:
        _unlink_folder(folder / "overlays")


def test_a6_remove_file_removes_a_listed_plain_file_and_nothing_else(tmp_path: Path) -> None:
    folder = tmp_path / "ad"
    (folder / "overlays").mkdir(parents=True)
    (folder / "overlays" / "a.png").write_bytes(b"png")
    (folder / "other.txt").write_text("kept", encoding="utf-8")
    _remove_file(folder, "overlays/a.png")
    assert not (folder / "overlays" / "a.png").exists()
    _remove_file(folder, "overlays")  # a folder name is not a file
    assert (folder / "overlays").is_dir()
    assert (folder / "other.txt").is_file()


def test_a6_an_export_folder_that_is_a_link_is_refused(tmp_path: Path) -> None:
    from adanvil.export import ExportError, export_timeline

    root = init_project(tmp_path / "project", "Export")
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "exports").mkdir()
    _folder_link(outside, root / "exports" / "ad")
    try:
        with pytest.raises(ExportError, match="link"):
            export_timeline(root, _board())
        assert list(outside.iterdir()) == []
    finally:
        _unlink_folder(root / "exports" / "ad")


# ------------------------------------------------------------- A7 vision_setup(python=)


def test_a7_vision_setup_takes_no_interpreter_from_the_agent() -> None:
    async def run():
        async with Client(build_server()) as client:
            return {tool.name: tool for tool in (await client.list_tools()).tools}

    tools = anyio.run(run)
    assert "python" not in tools["vision_setup"].input_schema.get("properties", {})
    assert "terminal" in tools["vision_setup"].description


# ----------------------------------------------------------------------- A8 job_status


def _job(root: Path, job_id: str, sheet: object) -> Path:
    folder = root / "renders" / "jobs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{job_id}.json"
    path.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "kind": "render",
                "project_dir": str(root),
                "job_file": str(path),
                "status": "done",
                "stage": "complete",
                "progress": {"done": 1, "total": 1, "unit": "segments"},
                "report": {"sheet": sheet},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "job_id",
    ["../../storyboards/x", "job-../x", "job-", "JOB-1", "job-xyz", "job-1/../../x", "x" * 300, ""],
)
def test_a8_a_job_id_is_a_job_id(job_id: str, tmp_path: Path) -> None:
    root = init_project(tmp_path / "project", "Jobs")
    assert not is_job_id(job_id)
    result = _call("job_status", {"job_id": job_id, "project_dir": str(root)})
    assert result.is_error
    assert "invalid job id" in _text(result)


def test_a8_a_planted_job_cannot_return_an_image_of_the_disk(tmp_path: Path) -> None:
    root = init_project(tmp_path / "project", "Jobs")
    outside = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), "red").save(outside)
    _job(root, "job-aaaa", str(outside))
    result = _call("job_status", {"job_id": "job-aaaa", "project_dir": str(root)})
    assert not result.is_error
    assert json.loads(_text(result))["status"] == "done"
    assert not _has_image(result)


def test_a8_the_reproduced_traversal_through_a_stored_storyboard_is_closed(tmp_path: Path) -> None:
    root = init_project(tmp_path / "project", "Jobs")
    outside = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), "red").save(outside)
    # storyboard_save stores any object: it becomes a "job file" for ../../storyboards/x
    saved = _call(
        "storyboard_save",
        {
            "project_dir": str(root),
            "name": "x",
            "storyboard": {
                "job_id": "job-x",
                "status": "done",
                "report": {"sheet": str(outside)},
            },
        },
    )
    assert not saved.is_error
    result = _call(
        "job_status", {"job_id": "../../storyboards/x", "project_dir": str(root)}
    )
    assert result.is_error
    assert not _has_image(result)


def test_a8_only_an_image_inside_the_project_that_owns_the_job_is_returned(tmp_path: Path) -> None:
    root = init_project(tmp_path / "project", "Jobs")
    sheet = root / "renders" / "ad_sheet.jpg"
    Image.new("RGB", (8, 8), "blue").save(sheet)
    (root / "renders" / "notes.txt").write_text("not a picture", encoding="utf-8")
    _job(root, "job-bbbb", str(sheet))
    assert _has_image(_call("job_status", {"job_id": "job-bbbb", "project_dir": str(root)}))
    _job(root, "job-cccc", str(root / "renders" / "notes.txt"))
    assert not _has_image(_call("job_status", {"job_id": "job-cccc", "project_dir": str(root)}))
    _job(root, "job-dddd", "//server/share/sheet.jpg")
    assert not _has_image(_call("job_status", {"job_id": "job-dddd", "project_dir": str(root)}))
    _job(root, "job-eeee", "renders/missing.jpg")
    assert not _has_image(_call("job_status", {"job_id": "job-eeee", "project_dir": str(root)}))
    _job(root, "job-ffff", "renders/ad_sheet.jpg")  # relative: read from the owning project
    assert _has_image(_call("job_status", {"job_id": "job-ffff", "project_dir": str(root)}))


def test_a8_a_sheet_that_is_a_link_to_another_image_is_not_returned(tmp_path: Path) -> None:
    root = init_project(tmp_path / "project", "Jobs")
    outside = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), "red").save(outside)
    _file_link(outside, root / "renders" / "sheet.jpg")
    _job(root, "job-1111", str(root / "renders" / "sheet.jpg"))
    assert not _has_image(_call("job_status", {"job_id": "job-1111", "project_dir": str(root)}))


def test_a8_the_working_folder_is_not_searched_for_jobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stray = tmp_path / "cwd"
    _job(stray, "job-2222", "x")
    monkeypatch.chdir(stray)
    result = _call("job_status", {"job_id": "job-2222"})
    assert result.is_error
    assert "not found" in _text(result)


def test_a8_a_catalog_sheet_must_be_an_image_inside_the_project(tmp_path: Path) -> None:
    root = init_project(tmp_path / "project", "Catalog")
    from adanvil.library.catalog import Catalog, Entry

    def entry(sheet: str) -> Entry:
        return Entry(
            id="clip-1",
            path="footage/a.mp4",
            kind="video",
            file_key={"path": "a", "size": 1, "mtime_ns": 1},
            sheet=sheet,
            indexed="2026-01-01T00:00:00+00:00",
        )

    Image.new("RGB", (8, 8), "red").save(root / "cache" / "sheet.jpg")
    save_catalog(root, Catalog(entries={"clip-1": entry("cache/sheet.jpg")}))
    assert _has_image(_call("library_view", {"project_dir": str(root), "entry_id": "clip-1"}))
    save_catalog(root, Catalog(entries={"clip-1": entry("brand.yaml")}))
    result = _call("library_view", {"project_dir": str(root), "entry_id": "clip-1"})
    assert result.is_error
    assert "image" in _text(result)


# ---------------------------------------------------------------------- A9 variants sheet


@pytest.mark.parametrize("bad_name", ["../../evil", "a/b", "x" * 200, "storyboards/..%2f"])
def test_a9_the_variants_sheet_never_uses_a_raw_base_name(bad_name: str, tmp_path: Path) -> None:
    root = init_project(tmp_path / "project", "Variants")
    result = _call(
        "variants_build",
        {
            "project_dir": str(root),
            "base": _board(name=bad_name),
            "variants": {"hook": [{"id": "a", "changes": []}]},
            "name_pattern": "v_{hook}",
        },
    )
    assert result.is_error
    assert "valid name" in _text(result)
    assert not (root / "renders" / "variants").exists()
    assert not list(tmp_path.glob("**/evil*"))


# ------------------------------------------------------------------ A10 planted catalog


def test_a10_a_planted_words_file_only_deletes_inside_cache_speech(tmp_path: Path) -> None:
    root = init_project(tmp_path / "project", "Speech")
    (root / "footage" / "keep.txt").write_text("footage", encoding="utf-8")
    (root / "cache" / "other.json").write_text("{}", encoding="utf-8")
    sidecar = root / "cache" / "speech" / "abc123.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("{}", encoding="utf-8")
    for planted in (
        "brand.yaml",
        "adanvil.json",
        "catalog.json",
        "footage/keep.txt",
        "cache/other.json",
        "cache/speech/sub/abc123.json",
        "cache/speech/abc123.txt",
        "../victim.txt",
        ".hidden",
        "C:/x.json",
    ):
        speech_store.remove_sidecar(root, planted)
    assert (root / "brand.yaml").is_file()
    assert (root / "adanvil.json").is_file()
    assert (root / "catalog.json").is_file()
    assert (root / "footage" / "keep.txt").is_file()
    assert (root / "cache" / "other.json").is_file()
    assert sidecar.is_file()
    speech_store.remove_sidecar(root, "cache/speech/abc123.json")  # a real sidecar still goes
    assert not sidecar.exists()


def test_a10_reindexing_a_changed_file_with_a_planted_catalog_deletes_nothing(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    root = init_project(tmp_path / "project", "Speech")
    clip = root / "footage" / "a.mp4"
    shutil.copy2(Path(str(media["sdr_portrait.mp4"]["path"])), clip)
    index_library(root, text_scan=False, transcribe=False, vision=False)
    catalog = load_catalog(root)
    identifier = next(iter(catalog.entries))
    for target in ("brand.yaml", "adanvil.json"):
        catalog.entries[identifier] = catalog.entries[identifier].model_copy(
            update={"speech": EntrySpeech(status="done", words_file=target)}
        )
        save_catalog(root, catalog)
        shutil.copy2(Path(str(media["ref_ad.mp4"]["path"])), clip)  # the file changed
        result = index_library(root, text_scan=False, transcribe=False, vision=False)
        assert identifier in result.updated
        assert (root / target).is_file(), f"{target} was deleted by a re-index"


# ------------------------------------------------------------------ A11 job runner


def test_a11_the_runner_is_started_without_the_working_folder_on_its_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import adanvil.jobs as jobs

    seen: list[list[str]] = []

    class FakeProcess:
        pid = 4242

    def fake_popen(command: list[str], **_kwargs: Any) -> FakeProcess:
        seen.append(command)
        return FakeProcess()

    monkeypatch.setattr(jobs.subprocess, "Popen", fake_popen)
    start_job(tmp_path / "project", "render", {"storyboard": {}, "quality": "preview"})
    assert seen[0][:4] == [sys.executable, "-P", "-m", "adanvil.jobs"]


def test_a11_a_python_file_planted_in_the_working_folder_is_not_imported(tmp_path: Path) -> None:
    folder = tmp_path / "started-here"
    folder.mkdir()
    marker = tmp_path / "planted-code-ran"
    (folder / "json.py").write_text(
        f"open({str(marker)!r}, 'w').close()\nraise ImportError('planted')\n", encoding="utf-8"
    )
    job = tmp_path / "job.json"
    env = dict(os.environ)
    command =[sys.executable, "-m", "adanvil.jobs", "run", str(job)]
    subprocess.run(command, cwd=folder, env=env, capture_output=True)
    assert marker.exists(), "the control run (no -P) should import the planted file"
    marker.unlink()
    subprocess.run([command[0], "-P", *command[1:]], cwd=folder, env=env, capture_output=True)
    assert not marker.exists()


# ------------------------------------------------------------ A12 whisper model names


@pytest.mark.parametrize(
    "model", ["Systran/faster-whisper-large-v3", "../../models/evil", "/tmp/evil", "", "huge"]
)
def test_a12_speech_refine_names_only_known_models(model: str, tmp_path: Path) -> None:
    root = init_project(tmp_path / "project", "Refine")
    result = _call(
        "speech_refine", {"project_dir": str(root), "sources": ["footage/a.mp4"], "model": model}
    )
    assert result.is_error
    assert "medium" in _text(result) or "model" in _text(result)


def test_a12_the_model_argument_is_an_enumeration() -> None:
    async def run():
        async with Client(build_server()) as client:
            return {tool.name: tool for tool in (await client.list_tools()).tools}

    schema = anyio.run(run)["speech_refine"].input_schema["properties"]["model"]
    assert set(schema["enum"]) == {
        "tiny",
        "base",
        "small",
        "medium",
        "large-v2",
        "large-v3",
        "large-v3-turbo",
    }
    assert schema["default"] == "medium"


def test_a12_a_known_model_gets_past_the_model_check(tmp_path: Path) -> None:
    root = init_project(tmp_path / "project", "Refine")
    result = _call(
        "speech_refine", {"project_dir": str(root), "sources": ["footage/a.mp4"], "model": "tiny"}
    )
    assert result.is_error
    assert "No clip to refine" in _text(result)  # not a model error: nothing is indexed


# ------------------------------------------------------------------- A13 formula injection


def test_a13_manifest_cells_that_would_be_formulas_are_quoted_in_the_csv_only(
    tmp_path: Path,
) -> None:
    root = init_project(tmp_path / "project", "Manifest")
    formulas = ["=HYPERLINK(\"http://evil\",\"x\")", "+1+1", "-2+3", "@SUM(A1)", "\t=1", "\r=1"]
    renders = [
        {
            "status": "done",
            "file": f"renders/b/ad_{number}.mp4",
            "ids": {"hook": f"h{number}"},
            "format": "9:16",
            "duration": 5.0,
            "score": 80,
            "hook_text": text,
            "cta_text": text,
            "music_licence": text,
            "quality": "final",
        }
        for number, text in enumerate(formulas)
    ]
    original = json.dumps(renders)
    path = write_manifest(root, "b", renders)
    assert json.dumps(renders) == original  # the JSON results keep the texts as they are
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    assert len(body) == len(formulas)
    for row, text in zip(body, formulas, strict=True):
        assert row[header.index("hook text")] == "'" + text
        assert row[header.index("CTA text")] == "'" + text
        assert row[header.index("music licence")] == "'" + text
        assert not any(cell.startswith(("=", "+", "-", "@", "\t", "\r")) for cell in row)
    # rewriting the manifest does not quote twice
    write_manifest(root, "b", [])
    with path.open(encoding="utf-8", newline="") as handle:
        assert list(csv.reader(handle))[1][header.index("hook text")] == "'" + formulas[0]


def test_a13_ordinary_texts_are_written_as_they_are(tmp_path: Path) -> None:
    root = init_project(tmp_path / "project", "Manifest")
    entry = {
        "status": "done",
        "file": "renders/b/ad.mp4",
        "ids": {"hook": "h1", "cta": "c1"},
        "format": "9:16",
        "duration": 12.5,
        "score": 91,
        "hook_text": "Stop scrolling, it's 2026",
        "cta_text": "Shop now",
        "music_licence": "CC0",
        "quality": "final",
    }
    path = write_manifest(root, "b", [entry])
    with path.open(encoding="utf-8", newline="") as handle:
        row = list(csv.DictReader(handle))[0]
    assert row["hook text"] == "Stop scrolling, it's 2026"
    assert row["duration"] == "12.50"
    assert row["variant ids"] == "hook=h1;cta=c1"


# ------------------------------------------------------------- A14 writes and planted links


def test_a14_write_json_replaces_a_planted_link_instead_of_following_it(tmp_path: Path) -> None:
    victim = tmp_path / "victim.json"
    victim.write_text("original", encoding="utf-8")
    link = tmp_path / "storyboards" / "ad.json"
    link.parent.mkdir()
    _file_link(victim, link)
    _write_json(link, {"name": "ad"})
    assert victim.read_text(encoding="utf-8") == "original"
    assert not link.is_symlink()
    assert json.loads(link.read_text(encoding="utf-8")) == {"name": "ad"}


def test_a14_write_json_still_writes_a_normal_file(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "er" / "ad.json"
    _write_json(target, {"name": "ad", "é": 1})
    assert target.read_text(encoding="utf-8") == '{\n  "name": "ad",\n  "é": 1\n}\n'


def test_a14_a_dangling_link_is_not_filled_in_by_project_init(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    target = tmp_path / "created-by-attacker-target.json"
    _file_link(target, root / "catalog.json")
    init_project(root, "Dangling")
    assert not target.exists()
    assert os.path.islink(root / "catalog.json")
    assert (root / "adanvil.json").is_file()


def test_a14_project_init_still_creates_every_file_of_a_new_project(tmp_path: Path) -> None:
    root = init_project(tmp_path / "new", "Fresh")
    for name in ("adanvil.json", "brand.yaml", "catalog.json"):
        assert (root / name).is_file()
    assert json.loads((root / "adanvil.json").read_text(encoding="utf-8"))["name"] == "Fresh"
    again = (root / "adanvil.json").read_text(encoding="utf-8")
    init_project(root, "Other name")  # safe on an existing folder: nothing is overwritten
    assert (root / "adanvil.json").read_text(encoding="utf-8") == again


def test_a14_the_batch_manifest_and_the_speech_sidecar_do_not_follow_links(tmp_path: Path) -> None:
    root = init_project(tmp_path / "project", "Links")
    victim = tmp_path / "victim.txt"
    victim.write_text("original", encoding="utf-8")
    (root / "renders" / "b").mkdir()
    _file_link(victim, root / "renders" / "b" / "manifest.csv")
    write_manifest(root, "b", [])
    assert victim.read_text(encoding="utf-8") == "original"
    from adanvil.speech.model import SpeechData

    relative = "cache/speech/0123456789abcdef.json"
    (root / "cache" / "speech").mkdir(parents=True)
    _file_link(victim, root / relative)
    speech_store.write_sidecar(
        root, relative, SpeechData(words=[], language="en", duration=1.0), source="x", model="m"
    )
    assert victim.read_text(encoding="utf-8") == "original"
    assert json.loads((root / relative).read_text(encoding="utf-8"))["model"] == "m"


def test_a14_an_export_does_not_write_through_a_planted_link(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    from adanvil.export import export_timeline

    root = init_project(tmp_path / "project", "Export")
    shutil.copy2(Path(str(media["sdr_portrait.mp4"]["path"])), root / "footage" / "a.mp4")
    victim = tmp_path / "victim.txt"
    victim.write_text("original", encoding="utf-8")
    folder = root / "exports" / "ad"
    board = _board(segments=[{"type": "clip", "source": "footage/a.mp4", "in": 0, "out": 1}])
    first = export_timeline(root, board, formats=["srt"])  # the normal use
    assert first["files"]["readme"] == "README.txt"
    assert (folder / MANIFEST).is_file()
    # an attacker replaces a file of the export by a link, then the export is made again
    (folder / "README.txt").unlink()
    _file_link(victim, folder / "README.txt")
    result = export_timeline(root, board, formats=["srt"])
    assert result["files"]["readme"] == "README.txt"
    assert not (folder / "README.txt").is_symlink()
    assert victim.read_text(encoding="utf-8") == "original"
    assert "Media mode" in (folder / "README.txt").read_text(encoding="utf-8")
