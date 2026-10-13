"""Usability fixes from the R7 field test: jobs, compact outputs, vocabularies, emoji."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client
from PIL import Image

import adanvil.server as server_module
from adanvil import ffmpeg
from adanvil._bin import ffmpeg_path
from adanvil._proc import NO_WINDOW
from adanvil.brand import default_brand, save_brand
from adanvil.jobs import jobs_pointer_dir, read_job, resolve_job_file, start_job
from adanvil.library import (
    annotate,
    entry_line,
    index_library,
    index_summary,
    load_catalog,
    pending_files,
    search,
)
from adanvil.library import index as index_module
from adanvil.library.index import _is_big, _video_shots
from adanvil.library.onepass import KeyframePass
from adanvil.onboard import build_brand, build_questionnaire
from adanvil.project import init_project
from adanvil.server import build_server, render_caption_preview
from adanvil.storyboard import validate_storyboard
from adanvil.text.emoji import DirectoryProvider, missing_emoji
from adanvil.textremoval import detect_text_boxes
from adanvil.voice import FakeEngine, VoiceSpec, generate_voice

FIRE = "\U0001f525"
SPARKLES = "✨"


def _media(media: dict[str, dict[str, object]], name: str) -> Path:
    return Path(str(media[name]["path"]))


def _json(result: Any) -> dict[str, Any]:
    return json.loads(result.content[0].text)


def _wait(job_file: Path, timeout: float = 180.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = read_job(job_file)
        if record.get("status") in {"done", "failed"}:
            return record
        time.sleep(0.2)
    return read_job(job_file)


@pytest.fixture(scope="module")
def uhd_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two seconds of 3840x2160 with a hard cut at 1 s and a keyframe every 10 frames."""

    path = tmp_path_factory.mktemp("uhd") / "uhd.mp4"
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
            "testsrc2=size=3840x2160:rate=15:duration=1",
            "-f",
            "lavfi",
            "-i",
            "smptebars=size=3840x2160:rate=15:duration=1",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "38",
            "-g",
            "10",
            "-keyint_min",
            "10",
            str(path),
        ],
        check=True,
        creationflags=NO_WINDOW,
    )
    return path


# --- item 1: job pointers --------------------------------------------------------------


def test_jobs_pointer_dir_follows_override_and_platform(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ADANVIL_JOBS_DIR", str(tmp_path / "override"))
    assert jobs_pointer_dir() == tmp_path / "override"
    monkeypatch.delenv("ADANVIL_JOBS_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    expected = tmp_path / ("local" if os.name == "nt" else "xdg")
    assert jobs_pointer_dir() == expected / "adanvil" / "jobs"


def test_start_job_writes_pointer_and_resolves_it(tmp_path: Path) -> None:
    record = start_job(
        tmp_path,
        "voice",
        {"name": "p", "spec": {"engine": "fake", "lines": ["one"]}},
        progress={"total": 1, "unit": "lines"},
    )
    pointer = jobs_pointer_dir() / f"{record['job_id']}.json"
    assert json.loads(pointer.read_text(encoding="utf-8")) == {"job_file": record["job_file"]}
    assert resolve_job_file(record["job_id"]) == Path(record["job_file"])
    assert resolve_job_file("job-0000000000000000") is None
    assert resolve_job_file("../etc/passwd") is None
    _wait(Path(record["job_file"]))


async def _status(server: Any, **arguments: Any) -> Any:
    async with Client(server) as client:
        return await client.call_tool("job_status", arguments)


def test_job_status_resolves_after_server_restart_and_with_project_dir(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The server also scans the working directory: keep it away from the project.
    monkeypatch.chdir(tmp_path_factory.mktemp("elsewhere"))
    project = tmp_path / "project"
    init_project(project, "restart")
    record = start_job(
        project,
        "voice",
        {"name": "r", "spec": {"engine": "fake", "lines": ["one", "two"]}},
        progress={"total": 2, "unit": "lines"},
    )
    job_id = record["job_id"]
    _wait(Path(record["job_file"]))

    # A fresh server process knows no id: the pointer file finds the job.
    monkeypatch.setattr(server_module, "_JOB_FILES", {})
    monkeypatch.setattr(server_module, "_PROJECT_ROOTS", set())
    result = anyio.run(lambda: _status(build_server(), job_id=job_id))
    assert not result.is_error
    assert _json(result)["status"] == "done"

    # Without the pointer only project_dir finds it; without it the error says what to do.
    monkeypatch.setattr(server_module, "_JOB_FILES", {})
    monkeypatch.setenv("ADANVIL_JOBS_DIR", str(tmp_path / "no-pointers"))
    missing = anyio.run(lambda: _status(build_server(), job_id=job_id))
    assert missing.is_error
    assert "job not found: pass project_dir" in missing.content[0].text
    found = anyio.run(lambda: _status(build_server(), job_id=job_id, project_dir=str(project)))
    assert not found.is_error
    assert _json(found)["status"] == "done"
    assert _json(found)["voice"]["lines"]

    unknown = anyio.run(
        lambda: _status(build_server(), job_id="job-ffffffffffffffff", project_dir=str(project))
    )
    assert unknown.is_error
    assert "job not found" in unknown.content[0].text


# --- item 2: library_index -------------------------------------------------------------


def test_index_single_file_source(tmp_path: Path, media: dict[str, dict[str, object]]) -> None:
    init_project(tmp_path, "single")
    shutil.copy2(_media(media, "sdr_portrait.mp4"), tmp_path / "footage" / "one.mp4")
    shutil.copy2(_media(media, "sdr_landscape.mp4"), tmp_path / "footage" / "two.mp4")

    files, pending = pending_files(tmp_path, "footage/one.mp4")
    assert [path.name for path in files] == ["one.mp4"]
    assert pending == 1
    result = index_library(tmp_path, "footage/one.mp4")
    assert len(result.added) == 1
    assert [entry.path for entry in load_catalog(tmp_path).entries.values()] == [
        "footage/one.mp4"
    ]
    assert pending_files(tmp_path, "footage/one.mp4")[1] == 0
    again = index_library(tmp_path, "footage/one.mp4")
    assert again.unchanged and not again.added and not again.removed


def test_4k_index_uses_keyframes_and_keeps_shots(
    tmp_path: Path, uhd_clip: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_project(tmp_path, "uhd")
    shutil.copy2(uhd_clip, tmp_path / "footage" / "uhd.mp4")
    info = ffmpeg.probe(uhd_clip)
    assert _is_big(info)
    cuts, yavg = ffmpeg.measure_video(uhd_clip)
    full_shots = _video_shots(info, cuts, yavg)

    decoded: list[int] = []
    original = index_module.keyframe_pass

    def spy(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        decoded.append(result.decoded)
        return result

    monkeypatch.setattr(index_module, "keyframe_pass", spy)
    result = index_library(tmp_path, "footage")

    assert decoded and decoded[0] >= 3, "the keyframe path must run for a source above 1920 px"
    assert len(result.added) == 1 and not result.errors
    entry = next(iter(load_catalog(tmp_path).entries.values()))
    assert abs(len(entry.shots) - len(full_shots)) <= 1
    assert len(entry.shots) == 2
    with Image.open(tmp_path / str(entry.sheet)) as sheet:
        assert sheet.width == 4 * 270


def test_4k_with_too_few_keyframes_falls_back_to_full_decode(
    tmp_path: Path, uhd_clip: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_project(tmp_path, "uhd-fallback")
    shutil.copy2(uhd_clip, tmp_path / "footage" / "uhd.mp4")
    full_calls: list[int] = []
    original = ffmpeg.measure_video

    def counted(*args: Any, **kwargs: Any) -> Any:
        full_calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(ffmpeg, "measure_video", counted)
    monkeypatch.setattr(
        index_module, "keyframe_pass", lambda *a, **k: KeyframePass([], [128.0], 2)
    )
    result = index_library(tmp_path, "footage")
    assert not result.errors and full_calls


def test_fast_text_scan_maps_boxes_back_to_source_pixels(
    media: dict[str, dict[str, object]],
) -> None:
    path = _media(media, "burnt_text.mp4")
    assert detect_text_boxes(path, duration=3.0, fps=2)
    fast = detect_text_boxes(path, duration=3.0, fps=2, fast=True)
    left, top, right, bottom = media["burnt_text.mp4"]["text_bbox"]
    assert any(
        not (box[2] <= left or right <= box[0] or box[3] <= top or bottom <= box[1])
        for box in fast
    )
    assert all(0 <= box[0] < box[2] <= 1080 and 0 <= box[1] < box[3] <= 1920 for box in fast)


def test_index_job_reports_progress_and_summary(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    init_project(tmp_path, "index job")
    for number in range(3):
        shutil.copy2(_media(media, "sdr_portrait.mp4"), tmp_path / "footage" / f"clip{number}.mp4")
    record = start_job(
        tmp_path, "index", {"source": "footage"}, progress={"total": 3, "unit": "files"}
    )
    assert record["progress"] == {"done": 0, "total": 3, "unit": "files"}

    result = _wait(Path(record["job_file"]))

    assert result["status"] == "done", result
    assert result["progress"] == {"done": 3, "total": 3, "unit": "files"}
    assert result["index"]["counts"]["added"] == 3
    assert len(result["index"]["entries"]) == 3
    assert all(" | footage/clip" in line for line in result["index"]["entries"])


async def _big_index(root: Path) -> None:
    async with Client(build_server()) as client:
        started = await client.call_tool("library_index", {"project_dir": str(root)})
        assert not started.is_error
        payload = _json(started)
        assert payload["status"] == "queued" and payload["job_id"].startswith("job-")
        assert "entries" not in payload
        done = await client.call_tool(
            "job_status", {"job_id": payload["job_id"], "wait_seconds": 170}
        )
        status = _json(done)
        assert status["status"] == "done", status
        assert status["progress"] == {"done": 9, "total": 9, "unit": "files"}
        assert status["index"]["counts"]["added"] == 9
        again = await client.call_tool("library_index", {"project_dir": str(root)})
        assert _json(again)["counts"]["unchanged"] == 9  # nothing new: synchronous again


def test_library_index_over_eight_new_files_is_a_detached_job(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    init_project(tmp_path, "nine files")
    for number in range(9):
        shutil.copy2(_media(media, "sdr_portrait.mp4"), tmp_path / "footage" / f"c{number}.mp4")
    anyio.run(_big_index, tmp_path)


def test_index_summary_is_compact(tmp_path: Path, media: dict[str, dict[str, object]]) -> None:
    init_project(tmp_path, "compact")
    shutil.copy2(_media(media, "sdr_portrait.mp4"), tmp_path / "footage" / "a.mp4")
    result = index_library(tmp_path, "footage")
    summary = index_summary(tmp_path, result)
    assert summary["counts"] == {
        "added": 1,
        "updated": 0,
        "unchanged": 0,
        "removed": 0,
        "errors": 0,
        "total": 1,
    }
    assert len(summary["entries"]) == 1
    assert summary["entries"][0].startswith("a-")
    assert "shots" not in json.dumps(summary)


# --- items 3, 5, 6: compact outputs, manifest message, emoji ---------------------------


def _voice_project(tmp_path: Path, media: dict[str, dict[str, object]]) -> tuple[Path, str]:
    init_project(tmp_path, "voice preview")
    shutil.copy2(_media(media, "ref_ad.mp4"), tmp_path / "footage" / "ref_ad.mp4")
    track = generate_voice(
        tmp_path,
        VoiceSpec(engine="fake", lines=["One two three four"]),
        "preview",
        engine=FakeEngine(),
    )
    return tmp_path, str(track.manifest)


def test_voice_captions_preview_lists_expanded_once(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root, manifest = _voice_project(tmp_path, media)
    payload, _ = render_caption_preview(
        root,
        "footage/ref_ad.mp4",
        1.0,
        {"type": "voice_captions", "manifest": manifest, "mode": "chunks", "max_words": 2},
    )
    assert payload["expanded"]
    assert "captions" not in payload


def test_voice_captions_without_manifest_explains_what_to_pass(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root, _ = _voice_project(tmp_path, media)
    with pytest.raises(Exception, match=r"pass manifest: the audio\.voice path of your storyboard"):
        render_caption_preview(
            root, "footage/ref_ad.mp4", 1.0, {"type": "voice_captions", "mode": "chunks"}
        )


def _emoji_folder(root: Path) -> Path:
    folder = root / "emoji"
    folder.mkdir()
    Image.new("RGBA", (72, 72), (220, 40, 40, 255)).save(folder / "1f525.png")  # fire
    return folder


def test_missing_emoji_helper(tmp_path: Path) -> None:
    provider = DirectoryProvider(_emoji_folder(tmp_path))
    assert missing_emoji([f"hot {FIRE} and {SPARKLES} and {SPARKLES}"], provider) == [SPARKLES]


def test_caption_preview_takes_emoji_folder_and_reports_missing(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    init_project(tmp_path, "emoji preview")
    shutil.copy2(_media(media, "ref_ad.mp4"), tmp_path / "footage" / "ref_ad.mp4")
    _emoji_folder(tmp_path)
    overlay = {
        "type": "caption",
        "style": "pill",
        "lines": [f"Hot {FIRE} {SPARKLES}"],
        "start": 0,
        "end": 2,
    }
    payload, _ = render_caption_preview(tmp_path, "footage/ref_ad.mp4", 1.0, overlay, emoji="emoji")
    assert payload["emoji_missing"] == [SPARKLES]
    clean, _ = render_caption_preview(tmp_path, "footage/ref_ad.mp4", 1.0, overlay, emoji="none")
    assert "emoji_missing" not in clean


def test_validate_warns_emoji_missing_for_project_folder(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    init_project(tmp_path, "emoji validate")
    shutil.copy2(_media(media, "sdr_portrait.mp4"), tmp_path / "footage" / "s.mp4")
    _emoji_folder(tmp_path)
    storyboard: dict[str, Any] = {
        "version": 1,
        "name": "emoji_check",
        "emoji": "emoji",
        "segments": [{"type": "clip", "source": "footage/s.mp4", "in": 0, "out": 2}],
        "overlays": [
            {
                "type": "caption",
                "style": "pill",
                "lines": [f"Hot {FIRE}", f"Sparkle {SPARKLES}"],
                "start": 0,
                "end": 2,
                "y": 330,
            }
        ],
    }
    report = validate_storyboard(copy.deepcopy(storyboard), root=tmp_path)
    warning = next(item for item in report.warnings if item.code == "emoji_missing")
    assert warning.message.startswith(f"emoji_missing: {SPARKLES} has no PNG")
    assert "Sparkle" in warning.message
    assert report.ok

    storyboard["emoji"] = "twemoji"
    clean = validate_storyboard(storyboard, root=tmp_path)
    assert not [item for item in clean.warnings if item.code == "emoji_missing"]


async def _decode_compact(root: Path) -> None:
    async with Client(build_server()) as client:
        decoded = await client.call_tool(
            "reference_decode", {"project_dir": str(root), "video": "footage/ref_ad.mp4"}
        )
        payload = _json(decoded)
        assert "transcript" not in payload and "rms_per_second" not in payload
        assert isinstance(payload["transcript_text"], str)
        assert payload["files"]["reference_json"] == "reference/ref-ad/reference.json"
        assert "reference.json" in payload["files"]["note"]
        assert all("spoken" in shot for shot in payload["shots"])
        vocab = payload["breakdown_vocabularies"]
        assert set(vocab) == {
            "type",
            "scale",
            "role",
            "motion",
            "emotion",
            "text_speaker",
            "text_device",
            "reading",
        }
        assert "character" in vocab["text_speaker"] and "quote" in vocab["text_device"]
        assert "a character is speaking" in vocab["reading"]
        assert payload["music_profile"]["feel"] in {"calm", "medium", "energetic"}
        assert "talking_head" in vocab["type"] and "closeup" in vocab["scale"]
        assert "hook" in vocab["role"] and "push_in" in vocab["motion"]
        saved = json.loads((root / "reference/ref-ad/reference.json").read_text(encoding="utf-8"))
        assert "rms_per_second" in saved and "transcript" in saved


def test_reference_decode_output_is_compact(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    init_project(tmp_path, "decode compact")
    shutil.copy2(_media(media, "ref_ad.mp4"), tmp_path / "footage" / "ref_ad.mp4")
    anyio.run(_decode_compact, tmp_path)


# --- item 7: product version -----------------------------------------------------------


def test_product_version_annotation_filter_and_compact_line(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    init_project(tmp_path, "versions")
    shutil.copy2(_media(media, "sdr_portrait.mp4"), tmp_path / "footage" / "new.mp4")
    shutil.copy2(_media(media, "sdr_landscape.mp4"), tmp_path / "footage" / "old.mp4")
    index_library(tmp_path, "footage")
    entries = {Path(entry.path).stem: entry for entry in load_catalog(tmp_path).entries.values()}
    annotate(
        tmp_path,
        entries["new"].id,
        product_visible=True,
        product_version="Current",
        people=["anna"],
    )
    annotate(tmp_path, entries["old"].id, product_visible=True, product_version="old")

    assert [e.id for e in search(tmp_path, product_version="current")] == [entries["new"].id]
    assert [e.id for e in search(tmp_path, product_version="OLD")] == [entries["old"].id]
    assert len(search(tmp_path, product_visible=True)) == 2
    assert search(tmp_path, product_version="v2") == []

    line = entry_line(search(tmp_path, product_version="current")[0])
    assert line.startswith(f"{entries['new'].id} | footage/new.mp4 | ")
    assert "people=anna" in line and "product_visible=true" in line and "version=Current" in line


async def _search_and_annotate(root: Path, entry_id: str) -> None:
    async with Client(build_server()) as client:
        annotated = await client.call_tool(
            "library_annotate",
            {"project_dir": str(root), "entry_id": entry_id, "product_version": "v2"},
        )
        assert not annotated.is_error
        assert _json(annotated)["annotation"]["product_version"] == "v2"
        compact = _json(
            await client.call_tool(
                "library_search", {"project_dir": str(root), "product_version": "v2"}
            )
        )
        assert compact["count"] == 1
        assert isinstance(compact["entries"][0], str) and "version=v2" in compact["entries"][0]
        full = _json(
            await client.call_tool(
                "library_search", {"project_dir": str(root), "product_version": "v2", "full": True}
            )
        )
        assert full["entries"][0]["annotation"]["product_version"] == "v2"
        none = _json(
            await client.call_tool(
                "library_search", {"project_dir": str(root), "product_version": "v1"}
            )
        )
        assert none["count"] == 0


def test_library_tools_expose_product_version_and_full(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    init_project(tmp_path, "tools")
    shutil.copy2(_media(media, "sdr_portrait.mp4"), tmp_path / "footage" / "a.mp4")
    index_library(tmp_path, "footage")
    entry_id = next(iter(load_catalog(tmp_path).entries))
    anyio.run(_search_and_annotate, tmp_path, entry_id)


def test_brand_product_versions_answer_is_stored_and_shown(tmp_path: Path) -> None:
    question = next(q for q in build_questionnaire(None) if q.id == "product_versions")
    assert not question.required and question.maps_to == "product_versions"

    answers = {"brand_name": "Acme", "product_versions": "Old box is white, current box is red."}
    brand, brief = build_brand(answers, project_root=tmp_path)
    assert brand.product_versions == "Old box is white, current box is red."
    assert "Product versions: Old box is white" in brief
    assert default_brand("Acme").product_versions is None

    root = tmp_path / "project"
    init_project(root, "brand versions")
    save_brand(brand, root / "brand.yaml")

    async def read() -> dict[str, Any]:
        async with Client(build_server()) as client:
            return _json(await client.call_tool("brand_get", {"project_dir": str(root)}))

    assert anyio.run(read)["brand"]["product_versions"].startswith("Old box")
