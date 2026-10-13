"""V2.9b: the fixes found by the Opus check of V2.8e and V2.9 (cover art in fetched mp3, licence
read from LICENCES.txt, an honest feel guess, no silent storyboard overwrite, an honest draft,
captions on the frame of the cut)."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client
from PIL import Image

from adanvil import audio as audio_module
from adanvil import music
from adanvil._bin import ffmpeg_path, ffprobe_path
from adanvil.batch import MANIFEST_COLUMNS, plan_items, run_batch, write_manifest
from adanvil.ffmpeg import probe
from adanvil.project import init_project
from adanvil.render import compose
from adanvil.render.animated import Placement
from adanvil.render.compose import snap_layers, snap_time
from adanvil.server import build_server
from adanvil.storyboard import Storyboard, validate_storyboard

# --------------------------------------------------------------------------- 1: cover art


def _mp3_with_cover(path: Path, *, flagged: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cover = path.with_suffix(".cover.png")
    Image.new("RGB", (64, 64), (200, 30, 30)).save(cover)
    command = [
        ffmpeg_path(), "-nostdin", "-y", "-loglevel", "error",
        "-f", "lavfi", "-t", "4", "-i", "sine=frequency=330:sample_rate=44100",
        "-i", str(cover), "-map", "0:a", "-map", "1:v", "-c:a", "libmp3lame", "-c:v", "mjpeg",
        "-id3v2_version", "3",
    ]  # fmt: skip
    if flagged:
        command += ["-disposition:v", "attached_pic"]
    subprocess.run([*command, str(path)], check=True)
    cover.unlink()
    return path


def _stream_types(path: Path) -> list[str]:
    out = subprocess.run(
        [str(ffprobe_path()), "-v", "error", "-show_entries", "stream=codec_type", "-of",
         "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return [s["codec_type"] for s in json.loads(out)["streams"]]


def test_an_mp3_with_cover_art_is_audio_everywhere(tmp_path: Path) -> None:
    init_project(tmp_path, "kettles")
    track = _mp3_with_cover(tmp_path / "music" / "art.mp3")
    assert sorted(_stream_types(track)) == ["audio", "video"]
    info = probe(track)
    assert info.kind == "audio" and info.has_audio and info.width is None
    Image.new("RGB", (360, 640), (9, 9, 9)).save(tmp_path / "footage" / "a.png")
    raw = {
        "name": "ad", "format": {"width": 360, "height": 640, "fps": 10}, "emoji": "none",
        "segments": [{"type": "image", "source": "footage/a.png", "duration": 3.0}],
        "audio": {"music": "music/art.mp3"}, "output": {"path": "renders/ad.mp4"},
    }  # fmt: skip
    report = validate_storyboard(Storyboard.model_validate(raw), root=tmp_path)
    assert not [i for i in report.errors if "not audio" in i.message], report.errors
    profile = audio_module.music_profile(tmp_path, "music/art.mp3")["tracks"][0]
    assert "error" not in profile and profile["duration"]
    assert audio_module.beat_grid(track) is not None
    done = compose(Storyboard.model_validate(raw), root=tmp_path, quality="preview")
    assert done.path.is_file()


def test_a_single_still_frame_without_the_flag_is_cover_art_too(tmp_path: Path) -> None:
    track = _mp3_with_cover(tmp_path / "unflagged.mp3", flagged=False)
    assert probe(track).kind == "audio"


def test_a_real_video_is_still_a_video(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    subprocess.run(
        [ffmpeg_path(), "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi", "-t", "1",
         "-i", "color=c=red:s=64x64:r=10", "-f", "lavfi", "-t", "1", "-i", "sine", "-shortest",
         str(video)],
        check=True,
    )  # fmt: skip
    assert probe(video).kind == "video"


def test_strip_cover_art_keeps_the_audio_in_the_same_container(tmp_path: Path) -> None:
    track = _mp3_with_cover(tmp_path / "art.mp3")
    assert music.strip_cover_art(track) is True
    assert _stream_types(track) == ["audio"] and track.suffix == ".mp3"
    assert music.strip_cover_art(track) is False  # nothing left to strip


def test_music_fetch_stores_the_file_without_the_cover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_project(tmp_path, "kettles")
    item = {
        "id": "bbbbbbbb-0001", "title": "Arty", "foreign_landing_url": "https://e.org/1",
        "url": "https://cdn.e.org/1.mp3", "creator": "me", "license": "by",
        "license_version": "3.0", "tags": [], "filetype": "mp3", "duration": 4000,
    }  # fmt: skip
    monkeypatch.setattr(music, "_http_json", lambda url, headers=None: item)

    def download(url: str, destination: Path) -> int:
        _mp3_with_cover(destination)
        return destination.stat().st_size

    monkeypatch.setattr(music, "_http_download", download)
    result = music.music_fetch(tmp_path, "openverse", "bbbbbbbb-0001")
    stored = tmp_path / result["path"]
    assert _stream_types(stored) == ["audio"] and probe(stored).kind == "audio"
    assert result["licence"] == "CC BY 3.0"


# --------------------------------------------------------------- 2: licence from LICENCES.txt

LEDGER = (
    "night_street.mp3 : Night Street (Relaxed Vlog), Ashot, Pixabay 131746, licence Pixabay, "
    "telecharge 30/09/2026\n"
    'fetched.mp3: "Upbeat Corporate" by Soundrider/Dope, CC BY 3.0 '
    "(https://creativecommons.org/licenses/by/3.0/), openverse 3b4580e1 "
    "(https://www.jamendo.com/track/1670486), fetched 2026-10-09. Attribution required: "
    '""Upbeat Corporate" by Soundrider/Dope is licensed under CC BY 3.0."\n'
    "zero.mp3: Z, W, CC0 1.0 (https://creativecommons.org/publicdomain/zero/1.0/), "
    "openverse 1, fetched 2026-10-09. No attribution required.\n"
)


def test_the_licence_is_read_from_licences_txt_when_there_is_no_sidecar(tmp_path: Path) -> None:
    (tmp_path / "music").mkdir()
    (tmp_path / "music" / "LICENCES.txt").write_text(LEDGER, encoding="utf-8")
    pixabay = music.music_licence(tmp_path, "music/night_street.mp3")
    assert pixabay["status"] == "recorded" and pixabay["licence"] == "Pixabay"
    assert pixabay["attribution_required"] is False and pixabay["warnings"] == []
    assert "LICENCES.txt line 1" in pixabay["read_from"]
    fetched = music.music_licence(tmp_path, "music/fetched.mp3")
    assert fetched["licence"] == "CC BY 3.0" and fetched["attribution_required"] is True
    assert "Upbeat Corporate" in fetched["attribution_text"]
    assert fetched["warnings"][0].startswith("music_attribution_required")
    zero = music.music_licence(tmp_path, "music/zero.mp3")
    assert zero["licence"] == "CC0 1.0" and zero["attribution_required"] is False
    unknown = music.music_licence(tmp_path, "music/other.mp3")
    assert unknown["status"] == "unknown" and "LICENCES.txt" in unknown["warnings"][0]


def test_a_sidecar_wins_over_the_ledger(tmp_path: Path) -> None:
    (tmp_path / "music").mkdir()
    (tmp_path / "music" / "LICENCES.txt").write_text(LEDGER, encoding="utf-8")
    music.write_licence_record(
        tmp_path,
        "music/night_street.mp3",
        {
            "source": "x", "id": "1", "url": "u", "title": "T", "author": "A",
            "licence": "CC0 1.0", "licence_url": "https://x", "attribution_required": False,
            "attribution_text": "", "fetch_date": "2026-10-09",
        },
    )  # fmt: skip
    info = music.music_licence(tmp_path, "music/night_street.mp3")
    assert info["licence"] == "CC0 1.0" and info["read_from"].endswith(".licence.json")


# ------------------------------------------------------------------------ 3: the feel guess


def _track(title: str, tags: list[str]) -> dict[str, Any]:
    return {"title": title, "tags": tags, "duration": 60.0, "source_url": title}


def test_the_feel_guess_comes_from_the_track_only() -> None:
    assert music.guess_feel(_track("Dark Cathedral", ["gothic", "metal", "sad"])) == "unknown"
    assert music.guess_feel(_track("Happy Birthday Interlude", ["happy"])) == "unknown"
    assert music.guess_feel(_track("Evening piano", ["ambient"])) == "calm"
    assert music.guess_feel(_track("Upbeat Corporate", ["edm"])) == "energetic"
    assert music.guess_feel(_track("Sunday", ["indie", "pop"])) == "medium"
    assert music.guess_feel(_track("Untitled", [])) == "unknown"


def test_ranking_never_copies_the_requested_feel() -> None:
    tracks = [
        _track("Dark Cathedral", ["gothic", "metal", "sad"]),
        _track("Happy Birthday Interlude", ["happy"]),
        _track("Evening piano", ["ambient"]),
        _track("Upbeat Corporate", ["edm"]),
    ]
    for wanted in ("calm", "medium", "energetic"):
        ranked = music.rank_tracks(tracks, duration=None, feel=wanted)
        guesses = {t["title"]: t["fit"]["feel_guess"] for t in ranked}
        assert guesses == {
            "Dark Cathedral": "unknown", "Happy Birthday Interlude": "unknown",
            "Evening piano": "calm", "Upbeat Corporate": "energetic",
        }  # fmt: skip
    # the guess orders the list: a match first, the opposite last, unknown between
    energetic = [t["title"] for t in music.rank_tracks(tracks, duration=None, feel="energetic")]
    assert energetic[0] == "Upbeat Corporate" and energetic[-1] == "Evening piano"


# ------------------------------------------------------------------ 4: the overwrite guard


def _clip(path: Path, seconds: float, colour: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg_path(), "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi", "-t",
         str(seconds), "-i", f"color=c={colour}:s=360x640:r=10", "-pix_fmt", "yuv420p",
         "-c:v", "libx264", "-g", "5", str(path)],
        check=True,
    )  # fmt: skip


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    init_project(root, "Guard")
    _clip(root / "footage" / "hook_a.mp4", 2.0, "0x803020")
    _clip(root / "footage" / "hook_b.mp4", 2.0, "0x208030")
    _clip(root / "footage" / "body.mp4", 2.0, "0x305070")
    return root


def _base() -> dict[str, Any]:
    return {
        "name": "demo", "format": {"width": 1080, "height": 1920, "fps": 10}, "emoji": "none",
        "segments": [
            {"type": "clip", "source": "footage/hook_a.mp4", "in": 0, "out": 2, "label": "hook"},
            {"type": "clip", "source": "footage/body.mp4", "in": 0, "out": 2, "label": "body"},
        ],
        "overlays": [
            {"type": "caption", "style": "pill", "lines": ["Hook"], "start": 0.2, "end": 1.8},
        ],
    }  # fmt: skip


def _hook_b() -> dict[str, Any]:
    return {
        "replace_label": "hook",
        "segments": [
            {"type": "clip", "source": "footage/hook_b.mp4", "in": 0, "out": 2, "label": "hook"}
        ],
        "overlays": [],
    }


def test_variants_build_never_overwrites_a_storyboard_without_overwrite(tmp_path: Path) -> None:
    root = _project(tmp_path)
    variants = {"hook": [{"id": "h1"}, {"id": "h2", "changes": [_hook_b()]}]}

    async def call(**extra: Any) -> Any:
        async with Client(build_server()) as client:
            return await client.call_tool(
                "variants_build",
                {"project_dir": str(root), "base": _base(), "variants": variants,
                 "contact_sheet": False, **extra},
            )  # fmt: skip

    first = anyio.run(lambda: call())
    assert not first.is_error
    edited = root / "storyboards" / "demo_h2.json"
    edited.write_text(edited.read_text("utf-8") + "\n", encoding="utf-8")  # used by hand since
    marker = edited.read_text("utf-8")
    again = anyio.run(lambda: call())  # made by an earlier variants_build of the same base
    assert again.is_error
    text = again.content[0].text
    assert "storyboards/demo_h1.json" in text and "storyboards/demo_h2.json" in text
    assert "overwrite=true" in text
    assert edited.read_text("utf-8") == marker
    forced = anyio.run(lambda: call(overwrite=True))
    assert not forced.is_error and edited.read_text("utf-8") != marker


# ------------------------------------------------------------------------- 5: honest draft


def test_draft_reports_the_real_size_and_the_manifest_says_the_quality(tmp_path: Path) -> None:
    root = _project(tmp_path)
    items, _ = plan_items(root, [_base()], None, "drafts")
    result = run_batch(root, {"batch": "drafts", "items": items, "quality": "draft", "score": True})
    assert result["quality"] == "preview"  # draft is an alias of preview
    entry = result["renders"][0]
    assert entry["status"] == "done", entry
    real = probe(root / entry["file"])
    assert (entry["width"], entry["height"]) == (real.width, real.height) == (540, 960)
    assert entry["quality"] == "preview"
    with (root / "renders" / "drafts" / "manifest.csv").open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == MANIFEST_COLUMNS and rows[0][-1] == "quality"
    assert rows[1][-1] == "preview"


def test_an_older_manifest_gets_an_empty_quality(tmp_path: Path) -> None:
    folder = tmp_path / "renders" / "old"
    folder.mkdir(parents=True)
    columns = [c for c in MANIFEST_COLUMNS if c != "quality"]
    with (folder / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerow(["old.mp4", "hook=h1", "9:16", "5.00", "80", "h", "", "c", "unknown"])
    write_manifest(tmp_path, "old", [])
    with (folder / "manifest.csv").open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[1][0] == "old.mp4" and rows[1][-2] == "unknown" and rows[1][-1] == ""


# --------------------------------------------------------------- 6: captions on the cut frame


def test_snap_time_rounds_to_the_nearest_frame_and_to_the_cut() -> None:
    assert snap_time(7.9003, 30, []) == pytest.approx(7.9)
    assert snap_time(12.9006, 30, []) == pytest.approx(12.9)
    assert snap_time(0.2, 10, []) == pytest.approx(0.2)
    # within half a frame of a cut that is itself off the grid: the frame of the cut
    assert snap_time(7.91, 30, [7.9]) == pytest.approx(7.9)
    assert snap_time(2.6334, 30, [2.633]) == pytest.approx(79 / 30)


def test_snap_layers_keeps_one_frame_and_never_passes_the_end() -> None:
    layers = [Placement(Path("a.png"), 1.001, 1.002), Placement(Path("b.png"), 5.0, 9.0)]
    first, second = snap_layers(layers, 10, [], 6.0)  # type: ignore[misc]
    assert second.end == pytest.approx(6.0) and second.start == pytest.approx(5.0)
    assert first.end - first.start == pytest.approx(0.1)


def test_a_caption_timed_just_after_a_cut_shows_on_the_first_frame_of_the_shot(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    raw = _base()
    raw["overlays"] = [
        {"type": "caption", "style": "pill", "lines": ["Next shot"], "start": 2.04, "end": 3.96},
        {"type": "caption", "style": "pill", "lines": ["First shot"], "start": 0.1, "end": 1.96},
    ]
    raw["segments"][1]["source"] = "footage/hook_b.mp4"
    raw["output"] = {"path": "renders/cut.mp4"}
    report = compose(Storyboard.model_validate(raw), root=root, quality="preview")

    def has_caption(index: int) -> bool:
        target = root / f"f{index}.png"
        subprocess.run(
            [ffmpeg_path(), "-nostdin", "-y", "-loglevel", "error", "-i", str(report.path),
             "-vf", f"select=eq(n\\,{index})", "-frames:v", "1", str(target)],
            check=True,
        )  # fmt: skip
        image = Image.open(target).convert("RGB")
        _count, _colour = max(image.getcolors(maxcolors=image.width * image.height))
        return _count < 0.995 * image.width * image.height

    assert has_caption(19), "the caption of shot 1 is there until the cut"
    assert has_caption(20), "the caption of shot 2 is missing on its first frame"
    assert has_caption(39), "the caption of shot 2 lasts to the end of its shot"
