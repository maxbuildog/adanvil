"""V2.10b: the timeline export (OTIO, Final Cut XML, EDL, SRT, overlay pictures) and the fixes
found by the r20 check (frame on screen, ledger details, QC images, variants sheet name)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import anyio
import pytest
from mcp import Client
from PIL import Image

from adanvil import music
from adanvil.export import ExportError, export_timeline
from adanvil.ffmpeg import extract_frame, frame_on_screen, run
from adanvil.project import init_project
from adanvil.qc import qc_report
from adanvil.render import compose
from adanvil.server import build_server
from adanvil.storyboard import Storyboard

from .test_compose import _project

FPS = 30


def _board(**extra: Any) -> dict[str, Any]:
    board: dict[str, Any] = {
        "name": "export_demo",
        "format": {"width": 1080, "height": 1920, "fps": FPS},
        "emoji": "none",
        "segments": [
            {"type": "clip", "source": "footage/sdr_portrait.mp4", "in": 0.2, "out": 1.7},
            {
                "type": "clip",
                "source": "footage/hlg_portrait.mp4",
                "in": 0.0,
                "out": 1.2,
                "speed": 1.5,
            },
            {"type": "image", "source": "footage/cta.png", "duration": 1.0},
        ],
        "overlays": [
            {
                "type": "caption",
                "style": "pill",
                "lines": ["First line"],
                "start": 0.0,
                "end": 1.5,
                "y": 330,
            },
            {
                "type": "caption",
                "style": "outline",
                "lines": ["Shop now"],
                "start": 1.5,
                "end": 2.3,
                "y": 1300,
            },
        ],
        "audio": {"music": "footage/music_short.m4a"},
    }
    board.update(extra)
    return board


def _export(media: Any, tmp_path: Path, **options: Any) -> tuple[Path, dict[str, Any], Path]:
    root = _project(media, tmp_path)
    out = tmp_path / "out"
    result = export_timeline(root, _board(), out_dir=out, **options)
    return root, result, out


def test_the_export_writes_all_four_formats_and_the_overlays(media, tmp_path: Path) -> None:
    _root, result, out = _export(media, tmp_path)
    assert set(result["files"]) == {"otio", "fcp7xml", "edl", "srt", "overlays", "readme"}
    for name in ("export_demo.otio", "export_demo.xml", "export_demo.edl", "export_demo.srt"):
        assert (out / name).is_file()
    listing = json.loads((out / "overlays.json").read_text(encoding="utf-8"))
    assert [item["type"] for item in listing["overlays"]] == ["caption", "caption"]
    for item in listing["overlays"]:
        with Image.open(out / item["file"]) as picture:
            assert picture.size == (1080, 1920) and picture.mode == "RGBA"
        assert item["width"] > 0 and item["end"] > item["start"]
    assert result["duration"] == pytest.approx((45 + 24 + 30) / FPS, abs=1e-6)


def test_otio_structure_timing_speed_stills_and_captions(media, tmp_path: Path) -> None:
    _root, _result, out = _export(media, tmp_path)
    document = json.loads((out / "export_demo.otio").read_text(encoding="utf-8"))
    assert document["OTIO_SCHEMA"] == "Timeline.1"
    tracks = {track["name"]: track for track in document["tracks"]["children"]}
    assert {"V1", "overlays", "A1 (music)"} <= set(tracks)
    assert "captions" not in tracks
    v1 = tracks["V1"]["children"]
    assert [c["source_range"]["duration"]["value"] for c in v1] == [45, 24, 30]
    first, second, third = v1
    assert first["source_range"]["start_time"]["value"] == 6  # in 0.2 s
    warp = second["effects"][0]
    assert warp["effect_name"] == "LinearTimeWarp" and warp["time_scalar"] == 1.5
    assert second["metadata"]["adanvil"]["speed"] == 1.5
    assert third["media_reference"]["target_url"].endswith("cta.png")
    assert first["media_reference"]["target_url"].startswith("file:///")
    markers = tracks["V1"]["markers"]
    assert [m["name"] for m in markers] == ["First line", "Shop now"]
    assert markers[1]["marked_range"]["start_time"]["value"] == round(1.5 * FPS)
    music_track = tracks["A1 (music)"]["children"][0]
    assert music_track["media_reference"]["target_url"].endswith("music_short.m4a")


def test_the_otio_loads_with_the_real_package(media, tmp_path: Path) -> None:
    otio = pytest.importorskip("opentimelineio")
    _root, _result, out = _export(media, tmp_path)
    timeline = otio.adapters.read_from_file(str(out / "export_demo.otio"))
    video = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
    assert video[0].name == "V1" and len(video[0]) == 3
    assert video[0].duration().value == 99
    assert video[0][1].effects[0].time_scalar == 1.5


def test_xmeml_structure(media, tmp_path: Path) -> None:
    _root, _result, out = _export(media, tmp_path)
    root = ET.parse(out / "export_demo.xml").getroot()
    assert root.tag == "xmeml" and root.get("version") == "4"
    sequence = root.find("sequence")
    assert sequence is not None
    assert sequence.findtext("rate/timebase") == "30"
    assert sequence.findtext("duration") == "99"
    assert sequence.findtext("media/video/format/samplecharacteristics/width") == "1080"
    assert sequence.findtext("media/video/format/samplecharacteristics/height") == "1920"
    first_track = sequence.find("media/video/track")
    assert first_track is not None
    items = first_track.findall("clipitem")
    assert [(i.findtext("start"), i.findtext("end")) for i in items] == [
        ("0", "45"),
        ("45", "69"),
        ("69", "99"),
    ]
    assert items[0].findtext("in") == "6" and items[0].findtext("out") == "51"
    assert items[1].findtext("filter/effect/parameter/value") == "150.0"
    described = [f for f in sequence.iter("file") if f.find("pathurl") is not None]
    assert len(described) == len({f.get("id") for f in described})
    assert all(f.findtext("pathurl").startswith("file://localhost") for f in described)
    assert [m.findtext("name") for m in sequence.findall("marker")] == ["First line", "Shop now"]
    assert len(sequence.findall("media/video/track")) >= 2
    assert sequence.find("media/audio/track/clipitem") is not None


def test_edl_lines(media, tmp_path: Path) -> None:
    _root, _result, out = _export(media, tmp_path)
    lines = (out / "export_demo.edl").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "TITLE: export_demo" and lines[1] == "FCM: NON-DROP FRAME"
    events = [line for line in lines if re.match(r"^\d{3}  ", line)]
    assert len(events) == 3
    pattern = re.compile(r"^(\d{3})  (\S{1,8})\s+V     C        (\S+) (\S+) (\S+) (\S+)$")
    parsed = [pattern.match(line) for line in events]
    assert all(parsed)
    assert parsed[0].group(3) == "00:00:00:06" and parsed[0].group(4) == "00:00:01:21"
    assert parsed[0].group(5) == "01:00:00:00" and parsed[0].group(6) == "01:00:01:15"
    assert parsed[2].group(6) == "01:00:03:09"
    assert any(line.startswith("M2   ") and "045.0" in line for line in lines)
    assert "* FROM CLIP NAME: sdr_portrait.mp4" in lines


def test_srt_times_equal_the_renders(media, tmp_path: Path) -> None:
    root, _result, out = _export(media, tmp_path, formats=["srt"])
    report = compose(Storyboard.model_validate(_board()), root=root, quality="preview")
    text = (out / "export_demo.srt").read_text(encoding="utf-8")
    blocks = [block for block in text.split("\n\n") if block.strip()]
    assert len(blocks) == 2

    def seconds(value: str) -> float:
        hours, minutes, rest = value.split(":")
        whole, millis = rest.split(",")
        return int(hours) * 3600 + int(minutes) * 60 + int(whole) + int(millis) / 1000

    captions = [o for o in report.overlays if o["type"] == "caption"]
    for block, shown in zip(blocks, captions, strict=True):
        _number, times, words = block.split("\n", 2)
        start, end = (seconds(part) for part in times.split(" --> "))
        assert abs(start - shown["start"]) <= 1 / FPS
        assert abs(end - shown["end"]) <= 1 / FPS
        assert words.strip() == shown["text"]


def test_relative_and_copied_media(media, tmp_path: Path) -> None:
    _root, _result, out = _export(media, tmp_path, relative=True)
    document = json.loads((out / "export_demo.otio").read_text(encoding="utf-8"))
    url = document["tracks"]["children"][0]["children"][0]["media_reference"]["target_url"]
    assert not Path(url).is_absolute() and url.endswith("sdr_portrait.mp4")
    root = _project(media, tmp_path / "again")
    copy = tmp_path / "copy"
    export_timeline(root, _board(), out_dir=copy, copy_media=True)
    assert (copy / "media" / "sdr_portrait.mp4").is_file()
    assert (copy / "media" / "music_short.m4a").is_file()
    assert "media/sdr_portrait.mp4" in (copy / "export_demo.otio").read_text(encoding="utf-8")


def test_a_bad_format_is_refused(media, tmp_path: Path) -> None:
    root = _project(media, tmp_path)
    with pytest.raises(ExportError):
        export_timeline(root, _board(), formats=["aaf"], out_dir=tmp_path / "x")


def _split_board() -> dict[str, Any]:
    board = _board()
    board["audio"] = {}
    board["overlays"] = []
    board["segments"] = [
        {
            "type": "split",
            "duration": 1.0,
            "panels": [
                {"source": "footage/sdr_portrait.mp4", "in": 0.0},
                {"source": "footage/hlg_portrait.mp4", "in": 0.0},
            ],
        },
        board["segments"][0],
    ]
    return board


def test_a_split_segment_puts_its_second_picture_on_v2(media, tmp_path: Path) -> None:
    root = _project(media, tmp_path)
    result = export_timeline(root, _split_board(), out_dir=tmp_path / "split")
    assert any("split" in warning for warning in result["warnings"])
    document = json.loads((tmp_path / "split" / "export_demo.otio").read_text(encoding="utf-8"))
    names = [track["name"] for track in document["tracks"]["children"]]
    assert "V2 (second pictures)" in names


# ------------------------------------------------------------- 2: the frame on screen at t


def _colour_video(path: Path) -> None:
    """Three red frames, then blue frames, at 30 fps."""

    run(
        [
            "-f", "lavfi", "-i", "color=c=red:s=64x64:r=30:d=0.1",
            "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=30:d=0.5",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", str(path),
        ]
    )  # fmt: skip


def _centre(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as picture:
        return picture.convert("RGB").getpixel((32, 32))


def test_the_frame_on_screen_is_the_floor_of_t_times_fps(tmp_path: Path) -> None:
    assert frame_on_screen(0.095, 30)[0] == 2
    assert frame_on_screen(2.87, 30)[0] == 86
    video = tmp_path / "colours.mp4"
    _colour_video(video)
    on_screen = tmp_path / "a.png"
    extract_frame(video, 0.095, on_screen, on_screen=True)  # frame 2: still red
    plain = tmp_path / "b.png"
    extract_frame(video, 0.095, plain)  # the old behaviour: frame 3, blue
    red, blue = _centre(on_screen), _centre(plain)
    assert red[0] > 150 > red[2] and blue[2] > 150 > blue[0]
    exact = tmp_path / "c.png"
    extract_frame(video, 0.1, exact, on_screen=True)  # frame 3: blue
    assert _centre(exact)[2] > 150


def test_frame_view_states_the_frame_time(tmp_path: Path) -> None:
    root = tmp_path / "project"
    init_project(root, "colours")
    _colour_video(root / "footage" / "colours.mp4")

    async def scenario() -> dict[str, Any]:
        async with Client(build_server()) as client:
            result = await client.call_tool(
                "frame_view",
                {"project_dir": str(root), "source": "footage/colours.mp4", "t": 0.095},
            )
            assert not result.is_error, result.content[0].text
            return json.loads(result.content[0].text)

    payload = anyio.run(scenario)
    assert payload["frame"] == 2 and payload["frame_t"] == pytest.approx(0.0667, abs=1e-4)


# ------------------------------------------------------------- 3: LICENCES.txt details


def test_the_ledger_line_gives_title_author_and_source(tmp_path: Path) -> None:
    (tmp_path / "music").mkdir()
    ledger = (
        "kulakovka.mp3 : TikTok (pop), Kulakovka, Pixabay 308512, licence Pixabay, "
        "telecharge 30/09/2026\n"
        'fetched.mp3: "Upbeat Corporate" by Soundrider/Dope, CC BY 3.0 '
        "(https://creativecommons.org/licenses/by/3.0/), openverse 3b4580e1 "
        "(https://www.jamendo.com/track/1670486), fetched 2026-10-09. "
        'Attribution required: "x".\n'
    )
    (tmp_path / "music" / "LICENCES.txt").write_text(ledger, encoding="utf-8")
    hand = music.music_licence(tmp_path, "music/kulakovka.mp3")
    assert (hand["title"], hand["author"], hand["source"]) == (
        "TikTok (pop)",
        "Kulakovka",
        "Pixabay 308512",
    )
    fetched = music.music_licence(tmp_path, "music/fetched.mp3")
    assert fetched["title"] == "Upbeat Corporate" and fetched["author"] == "Soundrider/Dope"
    assert fetched["source_url"] == "https://www.jamendo.com/track/1670486"
    assert fetched["source"] == "openverse 3b4580e1"


# ------------------------------------------------------------------- 4: the QC images


def test_the_caption_crop_is_named_after_its_storyboard_overlay(media, tmp_path: Path) -> None:
    root = _project(media, tmp_path)
    board = _board()
    board["audio"] = {}
    # a sound first: the report skips it, so the caption is report entry 0 but overlay 1
    board["overlays"] = [
        {"type": "sfx", "source": "footage/music_short.m4a", "at": 0.2},
        board["overlays"][0],
    ]
    render = compose(Storyboard.model_validate(board), root=root, quality="preview")
    assert render.overlays[0]["overlay"] == 1
    result = qc_report(
        render.path,
        render_report=render,
        storyboard=board,
        root=root,
        out_dir=tmp_path / "qc",
    )
    names = {path.name for path in result.images}
    assert any(name.startswith("overlay_1_first-line") for name in names)
    assert not any(name.startswith("overlay_0") for name in names)


def test_the_split_screen_image_is_returned(media, tmp_path: Path) -> None:
    root = _project(media, tmp_path)
    init_project(root, "split")
    render = compose(Storyboard.model_validate(_split_board()), root=root, quality="preview")
    (root / "renders").mkdir(exist_ok=True)
    (root / "renders" / "split.report.json").write_text(
        render.model_dump_json(), encoding="utf-8"
    )
    video = render.path.relative_to(root).as_posix()

    async def scenario() -> tuple[dict[str, Any], list[str]]:
        async with Client(build_server()) as client:
            result = await client.call_tool(
                "qc_run",
                {
                    "project_dir": str(root),
                    "video": video,
                    "render_report": "renders/split.report.json",
                },
            )
            return json.loads(result.content[0].text), [item.type for item in result.content]

    payload, kinds = anyio.run(scenario)
    listed = [name for name in payload["images"] if Path(name).name.startswith("split_")]
    assert listed
    assert kinds.count("image") >= 3  # sheet, end frame and the split picture
