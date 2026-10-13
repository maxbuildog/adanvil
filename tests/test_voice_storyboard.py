from __future__ import annotations

import json
import shutil
from pathlib import Path

import anyio
import pytest
from mcp import Client
from PIL import Image

from adanvil.audio import audio_duration, rms_per_second
from adanvil.ffmpeg import extract_frame
from adanvil.media import ImageSegment
from adanvil.project import init_project
from adanvil.render import compose
from adanvil.server import build_server
from adanvil.storyboard import (
    AudioSpec,
    Storyboard,
    VoiceCaptionsOverlay,
    validate_storyboard,
)
from adanvil.voice import FakeEngine, VoiceSpec, generate_voice


def _fixture(media: dict[str, dict[str, object]], name: str) -> Path:
    return Path(str(media[name]["path"]))


def _project(tmp_path: Path, media: dict[str, dict[str, object]]) -> Path:
    init_project(tmp_path, "voice storyboard")
    (tmp_path / "footage").mkdir(exist_ok=True)
    (tmp_path / "music").mkdir(exist_ok=True)
    (tmp_path / "brand").mkdir(exist_ok=True)
    shutil.copy2(_fixture(media, "sdr_portrait.mp4"), tmp_path / "footage/sdr_portrait.mp4")
    shutil.copy2(_fixture(media, "hlg_portrait.mp4"), tmp_path / "footage/hlg_portrait.mp4")
    shutil.copy2(_fixture(media, "cta.png"), tmp_path / "brand/cta.png")
    shutil.copy2(_fixture(media, "music_short.m4a"), tmp_path / "music/music_short.m4a")
    return tmp_path


def _storyboard(root: Path, track: object) -> Storyboard:
    lines = track.lines  # type: ignore[attr-defined]
    return Storyboard(
        name="voice_storyboard",
        segments=[
            {
                "type": "clip",
                "source": "footage/sdr_portrait.mp4",
                "in": 0,
                "out": lines[0].duration,
                "label": "usage:1",
            },
            {
                "type": "clip",
                "source": "footage/hlg_portrait.mp4",
                "in": 0,
                "out": lines[1].duration,
                "label": "usage:2",
            },
            {
                "type": "image",
                "source": "brand/cta.png",
                "duration": lines[2].duration + 1.5,
                "motion": "zoom_in",
                "label": "end_card",
            },
        ],
        overlays=[VoiceCaptionsOverlay(manifest=track.manifest)],  # type: ignore[attr-defined]
        audio=AudioSpec(
            voice=track.manifest,  # type: ignore[attr-defined]
            music="music/music_short.m4a",
        ),
    )


def test_voice_storyboard_composes_voice_captions_and_tail(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    root = _project(tmp_path, media)
    track = generate_voice(
        root,
        VoiceSpec(engine="fake", lines=["First sentence", "Second sentence", "Final call"]),
        "demo",
        engine=FakeEngine(),
    )
    storyboard = _storyboard(root, track)

    report = validate_storyboard(storyboard, root=root)
    assert report.ok, report.errors
    rendered = compose(storyboard, root=root, quality="preview")
    assert rendered.duration == pytest.approx(storyboard.total_duration(), abs=0.08)
    assert audio_duration(rendered.path) == pytest.approx(rendered.duration, abs=0.08)
    assert all(level >= -45 for level in rms_per_second(rendered.path))
    assert rendered.voice == {
        "manifest": track.manifest,
        "lines": 3,
        "total_duration": track.total_duration,
        "suspect": [],
    }

    for index, line in enumerate(track.lines):
        frame = root / f"frame-{index}.png"
        extract_frame(rendered.path, line.start + 0.2, frame)
        with Image.open(frame) as image:
            crop = image.crop(
                (0, round(image.height * 0.57), image.width, round(image.height * 0.9))
            ).convert("RGB")
            bright = sum(1 for pixel in crop.getdata() if min(pixel) > 220)
            assert bright > 100, f"caption missing for voice line {index}"


def test_voice_caption_chunks_follow_word_timestamps(tmp_path: Path) -> None:
    track = generate_voice(
        tmp_path,
        VoiceSpec(engine="fake", lines=["one two three four five six"]),
        "chunks",
        engine=FakeEngine(),
    )
    overlay = VoiceCaptionsOverlay(
        manifest=track.manifest,
        mode="chunks",
        max_words=2,
    )
    captions = overlay.expand(track)
    assert len(captions) == 3
    assert [caption.start for caption in captions] == sorted(caption.start for caption in captions)
    assert captions[-1].end == pytest.approx(track.lines[0].start + track.lines[0].duration)


def test_voice_validation_reports_cut_tail_and_suspect(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    root = _project(tmp_path, media)
    track = generate_voice(
        root,
        VoiceSpec(engine="fake", lines=["A line", "Another line", "Last line"]),
        "validation",
        engine=FakeEngine(),
    )
    short = Storyboard(
        name="short_voice",
        segments=[ImageSegment(source="brand/cta.png", duration=track.total_duration - 0.1)],
        audio=AudioSpec(voice=track.manifest),
    )
    cut_report = validate_storyboard(short, root=root)
    assert any(issue.code == "voice_cut" for issue in cut_report.errors)

    tail = Storyboard(
        name="short_tail",
        segments=[ImageSegment(source="brand/cta.png", duration=track.total_duration + 0.5)],
        audio=AudioSpec(voice=track.manifest, music="music/music_short.m4a"),
    )
    tail_report = validate_storyboard(tail, root=root)
    assert any(issue.code == "voice_tail" for issue in tail_report.warnings)

    suspect = generate_voice(
        root,
        VoiceSpec(engine="fake", language="fr", lines=["Bonjour [mispronounce] monde"]),
        "suspect",
        engine=FakeEngine(),
    )
    suspect_storyboard = Storyboard(
        name="suspect_voice",
        segments=[ImageSegment(source="brand/cta.png", duration=suspect.total_duration + 1.5)],
        audio=AudioSpec(voice=suspect.manifest),
    )
    suspect_report = validate_storyboard(suspect_storyboard, root=root)
    assert any(issue.code == "voice_suspect" for issue in suspect_report.warnings)


async def _mcp_voice(root: Path) -> None:
    server = build_server()
    async with Client(server) as client:
        started = await client.call_tool(
            "voice_generate",
            {
                "project_dir": str(root),
                "name": "mcp_demo",
                "spec": {"engine": "fake", "lines": ["MCP first", "MCP second"]},
            },
        )
        payload = json.loads(started.content[0].text)
        result = await client.call_tool(
            "job_status", {"job_id": payload["job_id"], "wait_seconds": 20}
        )
        status = json.loads(result.content[0].text)
        assert status["status"] == "done", status
        assert len(status["voice"]["lines"]) == 2
        viewed = await client.call_tool(
            "voice_view", {"project_dir": str(root), "name": "mcp_demo", "offset": 0.5}
        )
        summary = json.loads(viewed.content[0].text)
        assert summary["timeline"][0]["start"] == pytest.approx(0.5)
        assert summary["timeline"][0]["end"] > summary["timeline"][0]["start"]


def test_mcp_voice_generate_and_view(tmp_path: Path) -> None:
    init_project(tmp_path, "mcp voice")
    anyio.run(_mcp_voice, tmp_path)


def test_script_words_keep_script_text_and_merge_fragments() -> None:
    from adanvil.storyboard.schema import _script_words

    aligned = [
        (0.0, 0.3, " Avec", 1.0),
        (0.3, 0.5, " l", 1.0),
        (0.5, 0.8, "'été,", 1.0),
        (0.9, 1.2, " porte", 1.0),
        (1.2, 1.6, "-clés", 1.0),
    ]
    words = _script_words("Avec l'été, porte-clés", aligned)
    assert [w[2] for w in words] == ["Avec", "l'été,", "porte-clés"]
    assert words[2][:2] == (0.9, 1.6)
    digits = [(0.0, 0.3, " de", 1.0), (0.3, 0.6, " 49", 1.0), (0.6, 0.9, " euros", 1.0)]
    assert [w[2] for w in _script_words("de quarante-neuf euros", digits)] == [
        "de",
        "quarante-neuf",
        "euros",
    ]


def test_script_words_attach_french_punctuation() -> None:
    from adanvil.storyboard.schema import _script_words

    aligned = [(0.0, 0.4, " prix", 1.0), (0.5, 0.7, " ce", 1.0), (0.7, 1.0, " sont", 1.0)]
    assert [w[2] for w in _script_words("prix : ce sont", aligned)] == ["prix :", "ce", "sont"]


def test_script_words_keep_french_quotes_with_their_words() -> None:
    from adanvil.storyboard.schema import _script_words

    aligned = [(0.0, 0.3, " Découvre", 1.0), (0.3, 0.5, " la", 1.0), (0.5, 0.8, " carte", 1.0),
               (0.8, 1.1, " du", 1.0), (1.1, 1.4, " jour", 1.0)]
    words = [w[2] for w in _script_words("Découvre la « carte du jour »", aligned)]
    assert words == ["Découvre", "la", "« carte", "du", "jour »"]
