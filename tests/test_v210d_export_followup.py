"""V2.10d: the follow-up to the r22 check: QC crop names through the MCP server, stale export
files, portable EDL paths, one nominal frame rate rule, one numbering, sound effect lengths."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import anyio
import pytest
from mcp import Client

from adanvil.export import ExportError, _nominal_rate, export_timeline
from adanvil.ffmpeg import probe
from adanvil.project import init_project
from adanvil.render import compose
from adanvil.server import build_server
from adanvil.storyboard import Storyboard

from .test_compose import _project
from .test_v210b_export import FPS, _board


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_qc_run_names_crops_of_an_older_report_from_the_storyboard(media, tmp_path: Path) -> None:
    root = _project(media, tmp_path)
    init_project(root, "x")
    board = _board()
    board["audio"] = {}
    board["overlays"] = [
        {"type": "sfx", "source": "footage/music_short.m4a", "at": 0.2},
        *(
            {"type": "caption", "style": "outline", "lines": [t], "start": 0.0, "end": 2.0, "y": y}
            for t, y in (("Upper line", 1300), ("Lower line", 1400))
        ),
    ]
    (root / "storyboards").mkdir()
    (root / "storyboards" / "x.json").write_text(json.dumps(board), encoding="utf-8")
    render = compose(Storyboard.model_validate(board), root=root, quality="preview")
    document = json.loads(render.model_dump_json())
    for entry in document["overlays"]:  # a report written before entries carried these keys
        entry.pop("overlay", None)
        entry.pop("text", None)
    (root / "renders").mkdir(exist_ok=True)
    (root / "renders" / "x.report.json").write_text(json.dumps(document), encoding="utf-8")
    video = render.path.relative_to(root).as_posix()

    async def scenario(arguments: dict[str, Any]) -> list[str]:
        async with Client(build_server()) as client:
            result = await client.call_tool(
                "qc_run", {"project_dir": str(root), "video": video, **arguments}
            )
            assert not result.is_error, result.content[0].text
            return [Path(name).name for name in json.loads(result.content[0].text)["images"]]

    both = anyio.run(
        scenario, {"render_report": "renders/x.report.json", "storyboard": "storyboards/x.json"}
    )
    assert "overlay_1_upper-line.jpg" in both and "overlay_2_lower-line.jpg" in both
    only = anyio.run(scenario, {"storyboard": "storyboards/x.json"})
    assert "overlay_1_upper-line.jpg" in only and "overlay_2_lower-line.jpg" in only


def test_a_folder_of_an_older_version_is_refused_then_cleaned(media, tmp_path: Path) -> None:
    root = _project(media, tmp_path)
    out = tmp_path / "old"
    (out / "overlays").mkdir(parents=True)
    (out / "media").mkdir()
    (out / "overlays" / "001_caption.png").write_bytes(b"x")
    (out / "media" / "old.mov").write_bytes(b"x")
    (out / "export_demo.xml").write_text("old", encoding="utf-8")
    (out / "notes.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(ExportError, match="export_manifest"):
        export_timeline(root, _board(), out_dir=out)
    assert (out / "media" / "old.mov").exists()  # nothing was touched
    export_timeline(root, _board(), out_dir=out, overwrite=True)
    assert not (out / "overlays" / "001_caption.png").exists()
    assert not (out / "media").exists()
    assert (out / "notes.txt").read_text(encoding="utf-8") == "mine"
    manifest = _read(out / "export_manifest.json")
    assert manifest["mode"] == "absolute"
    assert "overlays/overlay_00_caption.png" in manifest["files"]
    assert "notes.txt" not in manifest["files"]
    # a second export cleans what the first wrote: a caption that is gone leaves no file
    board = _board()
    board["overlays"] = board["overlays"][:1]
    export_timeline(root, board, out_dir=out)
    names = sorted(p.name for p in (out / "overlays").iterdir())
    assert names == ["overlay_00_caption.png"]


def test_the_portable_edl_uses_relative_paths_and_relink_tells_the_truth(
    media, tmp_path: Path
) -> None:
    root = _project(media, tmp_path)
    result = export_timeline(root, _board(), copy_media=True)
    folder = Path(result["folder"])
    edl = (folder / "export_demo.edl").read_text(encoding="utf-8")
    sources = [x for x in edl.splitlines() if x.startswith("* SOURCE FILE:")]
    assert sources and all(x.split(": ", 1)[1].startswith("media/") for x in sources)
    assert str(root) not in edl
    relink = (folder / "relink.txt").read_text(encoding="utf-8")
    assert "EDL" in relink and "Adobe" in relink and "Link Media" in relink


def test_one_nominal_rate_rule_and_one_timecode_start(media, tmp_path: Path) -> None:
    assert _nominal_rate(29.99999, 30000 / 1001) == 30.0  # the catalog says 30
    assert _nominal_rate(29.97003, 30.0) == pytest.approx(30000 / 1001)
    assert _nominal_rate(59.9401, 60.0) == pytest.approx(60000 / 1001)
    assert _nominal_rate(29.8737, 30.0) == 30.0  # a variable-rate phone clip
    root = _project(media, tmp_path)
    out = tmp_path / "tc"
    export_timeline(root, _board(), out_dir=out)
    xml = ET.parse(out / "export_demo.xml").getroot()
    assert xml.findtext("sequence/timecode/string") == "01:00:00:00"
    assert xml.findtext("sequence/timecode/frame") == str(3600 * FPS)
    otio = _read(out / "export_demo.otio")
    assert otio["global_start_time"]["value"] == 3600 * FPS
    assert "01:00:00:00" in (out / "README.txt").read_text(encoding="utf-8")


def test_segments_and_sound_effects_are_numbered_the_same_way(media, tmp_path: Path) -> None:
    root = _project(media, tmp_path)
    board = _board()
    board["overlays"] = [
        {"type": "sfx", "source": "footage/music_short.m4a", "at": 0.5, "volume": 0.6},
        *board["overlays"],
    ]
    out = tmp_path / "numbers"
    result = export_timeline(root, board, out_dir=out)
    edl = (out / "export_demo.edl").read_text(encoding="utf-8")
    assert "* ADANVIL: segment 0 (counted from 0)" in edl
    assert "* ADANVIL: segment 2 (counted from 0)" in edl
    xml = ET.parse(out / "export_demo.xml").getroot()
    comments = [c.text for c in xml.iter("mastercomment1")]
    assert comments[:3] == [f"segment {n} (counted from 0)" for n in range(3)]
    readme = (out / "README.txt").read_text(encoding="utf-8")
    assert "segment 2: " in readme and "counted from 0" in readme
    names = [t["name"] for t in _read(out / "export_demo.otio")["tracks"]["children"]]
    assert any(name.startswith("A2 (sfx overlay 0: ") for name in names)
    assert result["sound_effects"] == 1


def test_a_sound_effect_never_runs_past_the_end_of_its_file(media, tmp_path: Path) -> None:
    root = _project(media, tmp_path)
    board = _board()
    board["audio"] = {}
    board["overlays"] = [{"type": "sfx", "source": "footage/music_short.m4a", "at": 0.1}]
    out = tmp_path / "sfx"
    export_timeline(root, board, out_dir=out, formats=["otio"])
    seconds = probe(root / "footage" / "music_short.m4a").duration or 0.0
    track = next(
        t for t in _read(out / "export_demo.otio")["tracks"]["children"] if "sfx" in t["name"]
    )
    clip = track["children"][-1]
    length = clip["source_range"]["duration"]["value"]
    assert length / FPS <= seconds + 1e-9
    assert length == min(math.floor(seconds * FPS + 1e-6), 99 - 3)
