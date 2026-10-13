"""V2.10e: a panel that runs out of source, real sound effect lengths, variable frame rate
sources and the audio licences in the export."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import anyio
from mcp import Client

from adanvil.export import export_timeline
from adanvil.ffmpeg import probe, run
from adanvil.project import init_project
from adanvil.server import build_server

from .test_compose import _project
from .test_v210b_export import FPS, _board


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _running_out_board() -> dict[str, Any]:
    board = _board()
    board["audio"] = {}
    board["overlays"] = []
    board["segments"] = [
        {
            "type": "split",
            "duration": 1.0,
            "panels": [
                {"source": "footage/hlg_portrait.mp4", "in": 1.5},  # the file lasts 2.0 s
                {"source": "footage/sdr_portrait.mp4", "in": 0.0},
            ],
        }
    ]
    return board


def test_a_panel_that_runs_out_of_source_is_clamped_and_held(media, tmp_path: Path) -> None:
    root = _project(media, tmp_path)
    init_project(root, "x")
    out = tmp_path / "out"
    result = export_timeline(root, _running_out_board(), out_dir=out)
    duration = probe(root / "footage" / "hlg_portrait.mp4").duration or 0.0
    assert any("holds the last frame" in w and "segment 0" in w for w in result["warnings"])
    xml = ET.parse(out / "export_demo.xml").getroot()
    items = xml.find("sequence/media/video/track").findall("clipitem")  # V1
    assert len(items) == 2
    main, held = items
    assert int(main.findtext("out")) <= int(main.findtext("duration"))
    assert (int(main.findtext("start")), int(main.findtext("end"))) == (0, 15)
    assert (int(held.findtext("start")), int(held.findtext("end"))) == (15, 30)
    assert held.findtext("filter/effect/name") == "Time Remap"
    assert int(held.findtext("in")) == int(duration * FPS) - 1  # the last frame
    otio = _read(out / "export_demo.otio")
    v1 = otio["tracks"]["children"][0]["children"]
    first = v1[0]
    begin = first["source_range"]["start_time"]
    start = begin["value"] / begin["rate"]
    assert start + 15 / FPS <= duration + 1e-9
    assert v1[1]["effects"][0]["effect_name"] == "FreezeFrame"
    readme = (out / "README.txt").read_text(encoding="utf-8")
    assert "segment 0: " in readme and "EDL event 001-002" in readme  # the hold is event 002

    async def validate() -> list[str]:
        async with Client(build_server()) as client:
            answer = await client.call_tool(
                "storyboard_validate",
                {"project_dir": str(root), "storyboard": _running_out_board()},
            )
            return [w["code"] for w in json.loads(answer.content[0].text)["warnings"]]

    assert "panel_source_short" in anyio.run(validate)


def test_sound_effect_files_are_declared_at_their_real_length(media, tmp_path: Path) -> None:
    root = _project(media, tmp_path)
    board = _board()
    board["audio"] = {}
    board["overlays"] = [{"type": "sfx", "source": "footage/music_short.m4a", "at": 0.1}]
    out = tmp_path / "sfx"
    result = export_timeline(root, board, out_dir=out)
    seconds = probe(root / "footage" / "music_short.m4a").duration or 0.0
    tracks = _read(out / "export_demo.otio")["tracks"]["children"]
    track = next(t for t in tracks if "sfx" in t["name"])
    clip = track["children"][-1]
    assert clip["media_reference"]["available_range"]["duration"]["value"] == seconds * FPS
    assert clip["source_range"]["duration"]["value"] <= seconds * FPS + 1e-9
    xml = ET.parse(out / "export_demo.xml").getroot()
    file = next(f for f in xml.iter("file") if f.findtext("name") == "music_short.m4a")
    assert int(file.findtext("duration")) >= seconds * FPS - 1e-6  # whole frames, rounded up
    item = next(c for t in xml.findall("sequence/media/audio/track") for c in t.findall("clipitem"))
    assert (int(item.findtext("end")) - int(item.findtext("start"))) <= seconds * FPS + 1e-6
    fractional = abs(seconds * FPS - round(seconds * FPS)) > 0.05
    assert fractional == any("sound effects are whole frames" in w for w in result["warnings"])


def test_variable_frame_rate_sources_are_named(media, tmp_path: Path) -> None:
    root = _project(media, tmp_path)
    vfr = root / "footage" / "vfr.mp4"
    run(
        [
            "-f", "lavfi", "-i", "testsrc=size=270x480:rate=30:duration=4",
            "-vf", "select='not(eq(mod(n,10),3))'", "-fps_mode", "vfr",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(vfr),
        ]
    )  # fmt: skip
    board = _board()
    board["audio"] = {}
    board["overlays"] = []
    board["segments"] = [{"type": "clip", "source": "footage/vfr.mp4", "in": 0.0, "out": 1.0}]
    result = export_timeline(root, board, out_dir=tmp_path / "vfr")
    text = " ".join(result["warnings"])
    assert "variable frame rate (VFR)" in text and "vfr.mp4" in text
    assert "constant frame rate" in text


def test_audio_licences_and_attribution_are_in_the_readme(media, tmp_path: Path) -> None:
    root = _project(media, tmp_path)
    (root / "music").mkdir(exist_ok=True)
    shutil.copy2(root / "footage" / "music_short.m4a", root / "music" / "fetched.m4a")
    (root / "music" / "LICENCES.txt").write_text(
        'fetched.m4a: "Upbeat" by Ann/Bob, CC BY-SA 3.0 '
        "(https://creativecommons.org/licenses/by-sa/3.0/), openverse 3b4580e1 "
        "(https://example.org/track), fetched 2026-10-09. "
        'Attribution required: "Upbeat by Ann/Bob, CC BY-SA 3.0".\n',
        encoding="utf-8",
    )
    board = _board()
    board["audio"] = {"music": "music/fetched.m4a"}
    out = tmp_path / "lic"
    result = export_timeline(root, board, out_dir=out)
    line = next(x for x in result["licences"] if x.startswith("music music/fetched.m4a"))
    assert "CC BY-SA 3.0" in line and "ATTRIBUTION REQUIRED" in line
    assert "Upbeat by Ann/Bob, CC BY-SA 3.0" in line
    readme = (out / "README.txt").read_text(encoding="utf-8")
    assert "Audio licences and attribution" in readme and line in readme
