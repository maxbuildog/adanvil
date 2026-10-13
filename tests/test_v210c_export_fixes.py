"""V2.10c: the fixes found by the r21 check of ``timeline_export``, ``frame_view`` and ``qc_run``.

One test per item: OTIO caption markers, speed conventions, real media descriptors, paths,
the ``not_exported`` list, one numbering, QC crops, frame_view by frame, the badge pop states.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import anyio
import pytest
from mcp import Client

from adanvil.export import ExportError, export_timeline
from adanvil.ffmpeg import frame_on_screen, run
from adanvil.project import init_project
from adanvil.qc.report import _crop_around, _crop_path, _slug, qc_report
from adanvil.render import compose
from adanvil.render.stickers import plan_sticker
from adanvil.server import build_server
from adanvil.storyboard import Storyboard
from adanvil.storyboard.schema import StickerOverlay

from .test_compose import _project
from .test_v210b_export import FPS, _board, _colour_video


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tracks(out: Path, name: str = "export_demo") -> dict[str, dict[str, Any]]:
    document = _read(out / f"{name}.otio")
    return {track["name"]: track for track in document["tracks"]["children"]}


def _uhd_like(path: Path, *, timecode: str | None) -> None:
    """A 25 fps portrait clip of another size than the sequence, with a start timecode."""

    args = [
        "-f", "lavfi", "-i", "testsrc=size=270x480:rate=25:duration=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25",
    ]  # fmt: skip
    if timecode:
        args += ["-timecode", timecode]
    run([*args, str(path)])


# ------------------------------------------------- 1: no offline caption clips in the OTIO


def test_item1_captions_are_markers_not_offline_clips(media, tmp_path: Path) -> None:
    root = _project(media, tmp_path)
    out = tmp_path / "out"
    export_timeline(root, _board(), out_dir=out)
    tracks = _tracks(out)
    assert "captions" not in tracks
    for track in tracks.values():
        for child in track["children"]:
            assert child["OTIO_SCHEMA"] != "Clip.1" or (
                child["media_reference"]["OTIO_SCHEMA"] == "ExternalReference.1"
            )
    markers = tracks["V1"]["markers"]
    assert [m["name"] for m in markers] == ["First line", "Shop now"]
    assert markers[0]["marked_range"]["duration"]["value"] == round(1.5 * FPS)
    assert tracks["overlays"]["children"]  # the overlay PNGs have real media and stay
    otio = pytest.importorskip("opentimelineio")
    timeline = otio.adapters.read_from_file(str(out / "export_demo.otio"))
    for track in timeline.tracks:
        if track.kind == otio.schema.TrackKind.Video:
            for clip in track.find_clips():
                assert not isinstance(clip.media_reference, otio.schema.MissingReference)


# ------------------------------------------------------------- 2: speed in XML and EDL


def _slow_board() -> dict[str, Any]:
    board = _board()
    board["audio"] = {}
    board["overlays"] = []
    board["segments"] = [
        {
            "type": "clip",
            "source": "footage/hlg_portrait.mp4",
            "in": 0.2,
            "out": 1.4,
            "speed": 0.8,
        },
        {"type": "clip", "source": "footage/sdr_portrait.mp4", "in": 0.0, "out": 1.0},
    ]
    return board


def test_item2_remapped_clips_follow_the_xmeml_and_cmx_conventions(media, tmp_path: Path) -> None:
    root = _project(media, tmp_path)
    out = tmp_path / "slow"
    export_timeline(root, _slow_board(), out_dir=out)
    xml = ET.parse(out / "export_demo.xml").getroot()
    first = xml.find("sequence/media/video/track/clipitem")
    assert first is not None
    start, end = int(first.findtext("start")), int(first.findtext("end"))
    first_in, first_out = int(first.findtext("in")), int(first.findtext("out"))
    assert (start, end) == (0, 45)  # 1.2 s at 0.8 plays 1.5 s
    assert first_out - first_in == end - start  # xmeml: out - in equals end - start
    speed = {
        p.findtext("parameterid"): p.findtext("value")
        for p in first.findall("filter/effect/parameter")
    }
    assert speed["speed"] == "80.0" and first.findtext("filter/effect/name") == "Time Remap"
    lines = (out / "export_demo.edl").read_text(encoding="utf-8").splitlines()
    event = next(line for line in lines if line.startswith("001  "))
    fields = event.split()
    source_in, source_out, record_in, record_out = fields[-4:]
    assert (source_in, source_out) == ("00:00:00:06", "00:00:01:21")  # in + 45 frames
    assert (record_in, record_out) == ("01:00:00:00", "01:00:01:15")
    motion = next(line for line in lines if line.startswith("M2   "))
    assert motion.split()[2] == "024.0"  # 0.8 x 30 fps of the source


# --------------------------------------------- 3: media descriptors, other fps, timecode


def test_item3_files_carry_their_real_rate_size_duration_and_timecode(
    media, tmp_path: Path
) -> None:
    root = _project(media, tmp_path)
    _uhd_like(root / "footage" / "tc25.mov", timecode="01:00:10:00")
    _uhd_like(root / "footage" / "plain25.mov", timecode=None)
    board = _board()
    board["audio"] = {}
    board["overlays"] = []
    board["segments"] = [
        {"type": "clip", "source": "footage/tc25.mov", "in": 1.0, "out": 3.0},
        {"type": "clip", "source": "footage/plain25.mov", "in": 0.0, "out": 1.0},
    ]
    out = tmp_path / "rates"
    result = export_timeline(root, board, out_dir=out)
    xml = ET.parse(out / "export_demo.xml").getroot()
    file = next(f for f in xml.iter("file") if f.findtext("name") == "tc25.mov")
    assert file.findtext("rate/timebase") == "25"
    assert file.findtext("duration") == "75"
    assert file.findtext("media/video/samplecharacteristics/width") == "270"
    assert file.findtext("media/video/samplecharacteristics/height") == "480"
    assert file.findtext("timecode/string") == "01:00:10:00"
    item = xml.find("sequence/media/video/track/clipitem")
    assert item is not None
    assert item.findtext("rate/timebase") == "25"
    assert (item.findtext("start"), item.findtext("end")) == ("0", "60")
    assert (item.findtext("in"), item.findtext("out")) == ("25", "75")  # frames of the 25 fps file
    clip = _tracks(out)["V1"]["children"][0]
    available = clip["media_reference"]["available_range"]
    assert available["duration"] == {"OTIO_SCHEMA": "RationalTime.1", "rate": 25.0, "value": 75.0}
    assert available["start_time"]["value"] == (3600 + 10) * 25
    assert clip["source_range"]["start_time"]["rate"] == 25.0
    assert clip["source_range"]["start_time"]["value"] == (3600 + 10) * 25 + 25
    assert clip["source_range"]["duration"]["value"] == 60  # sequence frames
    edl = (out / "export_demo.edl").read_text(encoding="utf-8").splitlines()
    first = next(line for line in edl if line.startswith("001  ")).split()
    assert first[-4:-2] == ["01:00:11:00", "01:00:13:00"]  # file timecode + in, at the EDL rate
    assert any("plain25.mov" in warning and "timecode" in warning for warning in result["warnings"])


# --------------------------------------------------------------------------- 4: paths


def test_item4_paths_urls_modes_and_collisions(media, tmp_path: Path) -> None:
    root = _project(media, tmp_path)
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "a").mkdir(parents=True)
    (elsewhere / "b").mkdir(parents=True)
    shutil.copy2(root / "footage" / "sdr_portrait.mp4", elsewhere / "a" / "clip.mp4")
    shutil.copy2(root / "footage" / "hlg_portrait.mp4", elsewhere / "b" / "clip.mp4")
    try:
        os.symlink(elsewhere / "a", root / "footage" / "linkA")
        os.symlink(elsewhere / "b", root / "footage" / "linkB")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available")
    board = _board()
    board["segments"] = [
        {"type": "clip", "source": "footage/linkA/clip.mp4", "in": 0.0, "out": 1.5},
        {"type": "clip", "source": "footage/linkB/clip.mp4", "in": 0.0, "out": 1.5},
    ]
    board["audio"] = {}
    absolute = export_timeline(root, board)
    folder = Path(absolute["folder"])
    assert folder == root / "exports" / "export_demo"
    reference = _tracks(folder)["V1"]["children"][0]["media_reference"]
    expected = (root / "footage" / "linkA" / "clip.mp4").as_posix()
    assert reference["target_url"] == "file://" + expected  # a file URL, the symlink kept
    assert "elsewhere" not in reference["target_url"]
    xml = (folder / "export_demo.xml").read_text(encoding="utf-8")
    assert "file://localhost" + expected in xml
    # a relative export has its own folder and cannot replace the absolute one
    before = (folder / "export_demo.xml").read_bytes()
    relative = export_timeline(root, board, relative=True)
    assert Path(relative["folder"]).name == "export_demo-relative"
    assert (folder / "export_demo.xml").read_bytes() == before
    assert (Path(relative["folder"]) / "relink.txt").is_file()
    relative_xml = (Path(relative["folder"]) / "export_demo.xml").read_text(encoding="utf-8")
    assert "file://localhost/RELINK_ROOT/../../footage/linkA/clip.mp4" in relative_xml
    with pytest.raises(ExportError, match="overwrite"):
        export_timeline(root, board, relative=True, out_dir=folder)
    export_timeline(root, board, relative=True, out_dir=folder, overwrite=True)
    # copied media: two files with one name stay two files, the overlay PNGs are not copied
    portable = export_timeline(root, board, copy_media=True)
    copied = sorted(p.name for p in (Path(portable["folder"]) / "media").iterdir())
    assert Path(portable["folder"]).name == "export_demo-portable"
    assert len(copied) == 2 and all(re.match(r"^[0-9a-f]{6}_clip\.mp4$", n) for n in copied)
    assert len(set(copied)) == 2
    assert not any(name.endswith(".png") for name in copied)
    # a media root writes absolute paths for a relative export
    rooted = export_timeline(
        root, board, relative=True, media_root="/Users/me/ad", out_dir=tmp_path / "r"
    )
    text = (Path(rooted["folder"]) / "export_demo.xml").read_text(encoding="utf-8")
    assert "file://localhost/Users/me/project/footage/linkA/clip.mp4" in text
    assert "file://localhost/Users/me/ad/overlays/" in text and "RELINK_ROOT" not in text


# ------------------------------------------------------------- 5: honest warnings


def test_item5_every_loss_is_listed_and_sound_effects_are_exported(media, tmp_path: Path) -> None:
    root = _project(media, tmp_path)
    board = _board()
    board["overlays"] = [
        {"type": "sfx", "source": "footage/music_short.m4a", "at": 0.5, "volume": 0.6},
        {
            "type": "caption",
            "style": "outline",
            "lines": ["Popping"],
            "start": 0.0,
            "end": 1.5,
            "y": 330,
            "animation": "pop",
        },
    ]
    board["segments"][0]["focus"] = [0.3, 0.4]
    board["segments"][2]["motion"] = "zoom_in"
    out = tmp_path / "loss"
    result = export_timeline(root, board, out_dir=out)
    listed = " | ".join(result["not_exported"]).lower()
    for word in (
        "gamma",
        "focus",
        "entry animations",
        "end-card or image zoom",
        "music volume and fades",
        "music loudness",
        "edl: video cuts only",
    ):
        assert word in listed, word
    readme = (out / "README.txt").read_text(encoding="utf-8")
    assert all(item in readme for item in result["not_exported"])
    assert result["sound_effects"] == 1
    tracks = _tracks(out)
    sfx_track = next(name for name in tracks if "sfx overlay 0" in name)
    sound = tracks[sfx_track]["children"][-1]
    assert sound["metadata"]["adanvil"]["level"] == 0.6
    assert tracks[sfx_track]["children"][0]["OTIO_SCHEMA"] == "Gap.1"  # starts at 0.5 s
    assert tracks[sfx_track]["children"][0]["source_range"]["duration"]["value"] == 15
    xml = ET.parse(out / "export_demo.xml").getroot()
    levels = [
        (c.findtext("name"), c.findtext("start"), c.findtext("filter/effect/parameter/value"))
        for t in xml.findall("sequence/media/audio/track")
        for c in t.findall("clipitem")
    ]
    assert ("music_short", "0", "0.8") in levels
    assert ("music_short", "15", "0.6") in levels
    music = next(
        c
        for c in xml.iter("clipitem")
        if c.findtext("name") == "music_short" and c.findtext("start") == "0"
    )
    keys = [(k.findtext("when"), k.findtext("value")) for k in music.iter("keyframe")]
    assert keys[0] == ("0", "0.0") and keys[-1][1] == "0.0"  # fade in and fade out


# --------------------------------------------------------------------- 6: one numbering


def test_item6_overlays_carry_their_storyboard_index_everywhere(media, tmp_path: Path) -> None:
    root = _project(media, tmp_path)
    board = _board()
    board["overlays"] = [
        {"type": "sfx", "source": "footage/music_short.m4a", "at": 0.2},
        *board["overlays"],
    ]
    out = tmp_path / "numbers"
    export_timeline(root, board, out_dir=out)
    names = sorted(p.name for p in (out / "overlays").iterdir())
    assert names == ["overlay_01_caption.png", "overlay_02_caption.png"]
    listing = _read(out / "overlays.json")
    assert [(o["storyboard_overlay"], o["file"]) for o in listing["overlays"]] == [
        (1, "overlays/overlay_01_caption.png"),
        (2, "overlays/overlay_02_caption.png"),
    ]
    xml = ET.parse(out / "export_demo.xml").getroot()
    clip_names = [c.findtext("name") for c in xml.findall("sequence/media/video/track/clipitem")]
    assert "overlay 1 caption" in clip_names and "overlay 2 caption" in clip_names
    assert [m.findtext("comment") for m in xml.findall("sequence/marker")] == [
        "overlay 1: First line",
        "overlay 2: Shop now",
    ]
    assert "overlay 2 caption" in " ".join(
        c["name"] for c in _tracks(out)["overlays"]["children"] if "name" in c
    )
    readme = (out / "README.txt").read_text(encoding="utf-8")
    assert "1: overlay 1: First line" in readme and "2: overlay 2: Shop now" in readme


# -------------------------------------------------------------------------------- 7: QC


def test_item7_crops_are_named_whole_and_the_end_card_is_saved_once(media, tmp_path: Path) -> None:
    # a neighbour that the margin would cut in half is left out, one that touches is kept whole
    own = (100.0, 400.0, 500.0, 460.0)
    below = (100.0, 500.0, 500.0, 560.0)
    left, top, right, bottom = _crop_around(own, [below])
    assert bottom < below[1] and top == own[1] - 80 and right == own[2] + 80
    touching = (100.0, 450.0, 500.0, 520.0)
    left, top, right, bottom = _crop_around(own, [touching])
    assert bottom >= touching[3] and top == own[1] - 80
    inside = (120.0, 470.0, 200.0, 480.0)
    assert _crop_around(own, [inside])[3] == own[3] + 80
    assert _slug("Clique sur « En savoir plus » ⬇️") == "clique-sur-en-savoir-plus"
    assert _crop_path(tmp_path, 10, [], "Clique sur « En savoir plus »").name == (
        "overlay_10_clique-sur-en-savoir-plus.jpg"
    )
    # a real render: two captions on screen together, the ad ends on an image
    root = _project(media, tmp_path)
    board = _board()
    board["audio"] = {}
    board["overlays"] = [
        {"type": "caption", "style": "outline", "lines": [text], "start": 0.0, "end": 2.0, "y": y}
        for text, y in (("Upper line", 1300), ("Lower line", 1400))
    ]
    render = compose(Storyboard.model_validate(board), root=root, quality="preview")
    result = qc_report(render.path, render_report=render, out_dir=tmp_path / "qc")
    names = [path.name for path in result.images]
    assert "overlay_0_upper-line.jpg" in names and "overlay_1_lower-line.jpg" in names
    assert "end.jpg" in names and "endcard.jpg" not in names
    visible = next(check for check in result.checks if check.id == "overlay_visible")
    assert "overlay 0" in visible.message
    assert "mid_pixel_difference_0_255" in visible.value and "meaning" in visible.value


# ---------------------------------------------------------------- 8: frame_view by frame


def test_item8_frame_view_takes_a_frame_number_and_a_time_near_a_boundary(
    tmp_path: Path,
) -> None:
    assert frame_on_screen(10.9333, 30)[0] == 328  # printed to four decimals, 0.03 ms early
    assert frame_on_screen(10.92, 30)[0] == 327
    assert frame_on_screen(10.9323, 30)[0] == 327  # 1.0 ms before is still frame 327
    assert frame_on_screen(10.934, 30)[0] == 328
    root = tmp_path / "project"
    init_project(root, "colours")
    _colour_video(root / "footage" / "colours.mp4")

    async def call(arguments: dict[str, Any]) -> tuple[bool, str]:
        async with Client(build_server()) as client:
            result = await client.call_tool(
                "frame_view",
                {"project_dir": str(root), "source": "footage/colours.mp4", **arguments},
            )
            return bool(result.is_error), result.content[0].text

    failed, text = anyio.run(call, {"frame": 3})
    assert not failed
    payload = json.loads(text)
    assert payload["frame"] == 3 and payload["frame_t"] == pytest.approx(0.1, abs=1e-4)
    failed, text = anyio.run(call, {"t": 0.0999})  # within 1 ms of the boundary: frame 3
    assert not failed and json.loads(text)["frame"] == 3
    failed, _text = anyio.run(call, {})
    assert failed
    failed, _text = anyio.run(call, {"t": 0.1, "frame": 3})
    assert failed


# ------------------------------------------------------------------ 9: badge pop states


def test_item9_a_badge_pop_draws_one_state_at_a_time() -> None:
    badge = StickerOverlay.model_validate(
        {"type": "sticker", "kind": "badge", "text": "-20%", "x": 184, "y": 248, "animation": "pop"}
    )
    for start, end in ((17.1675, 19.401), (2.0, 2.05), (0.0, 3.0)):
        states = plan_sticker(badge, start, end, 30)
        assert states[0].start == start and states[-1].end == end
        for before, after in zip(states, states[1:], strict=False):
            assert before.end <= after.start + 1e-9, (before, after)  # no overlap
            assert after.start == pytest.approx(before.end)  # and no gap
        assert states[-1].scale == 1.0 or end - start < 0.4
