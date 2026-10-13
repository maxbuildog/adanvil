from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image

from adanvil.ffmpeg import extract_frame, probe, run
from adanvil.media import Cache
from adanvil.render import compose
from adanvil.storyboard import Storyboard


def _project(media: dict[str, dict[str, object]], tmp_path: Path) -> Path:
    root = tmp_path / "project"
    footage = root / "footage"
    footage.mkdir(parents=True)
    for name in (
        "sdr_portrait.mp4",
        "hlg_portrait.mp4",
        "burnt_text.mp4",
        "cta.png",
        "music_short.m4a",
    ):
        shutil.copy2(Path(str(media[name]["path"])), footage / name)
    return root


def _storyboard() -> Storyboard:
    return Storyboard.model_validate(
        {
            "name": "compose_demo",
            "format": {"width": 1080, "height": 1920, "fps": 30},
            "emoji": "none",
            "segments": [
                {"type": "clip", "source": "footage/sdr_portrait.mp4", "in": 0.2, "out": 1.7},
                {
                    "type": "clip",
                    "source": "footage/hlg_portrait.mp4",
                    "in": 0.0,
                    "out": 1.2,
                    "zoom": {"from": 1.0, "to": 1.08},
                },
                {
                    "type": "clip",
                    "source": "footage/burnt_text.mp4",
                    "in": 0.5,
                    "out": 1.5,
                    "remove_text": {"method": "mask", "box": [316, 286, 771, 460]},
                },
                {
                    "type": "image",
                    "source": "footage/cta.png",
                    "duration": 1.5,
                    "motion": "zoom_in",
                },
            ],
            "overlays": [
                {
                    "type": "caption",
                    "style": "pill",
                    "lines": ["It's your sign"],
                    "start": 0,
                    "end": 1.5,
                    "y": 330,
                },
                {
                    "type": "caption",
                    "style": "outline",
                    "lines": ["Shop now"],
                    "start": 1.5,
                    "end": None,
                    "y": 1300,
                },
                {
                    "type": "comment",
                    "look": "facebook",
                    "text": "Does it really work?",
                    "reactions": 12,
                    "start": 0.2,
                    "end": 1.4,
                    "y": 900,
                },
            ],
            "audio": {"music": "footage/music_short.m4a"},
            "output": {"path": "renders/{name}.mp4", "quality": "final"},
        }
    )


def test_compose_end_to_end_and_cache(
    media: dict[str, dict[str, object]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(media, tmp_path)
    storyboard = _storyboard()
    report = compose(storyboard, root=root)

    info = probe(report.path)
    assert report.path.exists()
    assert (info.width, info.height, info.fps, info.pix_fmt) == (1080, 1920, 30, "yuv420p")
    assert (info.color_transfer, info.color_primaries, info.color_space) == (
        "bt709",
        "bt709",
        "bt709",
    )
    assert info.duration == pytest.approx(5.2, abs=0.05)
    assert info.audio_duration == pytest.approx(info.duration, abs=0.05)
    assert run(["-v", "error", "-i", str(report.path), "-f", "null", "-"]).stderr == ""
    expected_timeline = [
        (0.0, 1.5, "footage/sdr_portrait.mp4"),
        (1.5, 2.7, "footage/hlg_portrait.mp4"),
        (2.7, 3.7, "footage/burnt_text.mp4"),
        (3.7, 5.2, "footage/cta.png"),
    ]
    for item, (expected_start, expected_end, expected_source) in zip(
        report.timeline, expected_timeline, strict=True
    ):
        assert item["start"] == pytest.approx(expected_start, abs=0.05)
        assert item["end"] == pytest.approx(expected_end, abs=0.05)
        assert item["source"] == expected_source
    assert report.sheet.exists()
    assert Image.open(report.sheet).size[0] == 1920

    frame = tmp_path / "pill.png"
    extract_frame(report.path, 0.7, frame)
    with Image.open(frame).convert("RGB") as image:
        pill_region = image.crop((250, 330, 830, 420))
        assert sum(1 for pixel in pill_region.getdata() if min(pixel) > 245) > 500

    extract_frame(report.path, 2.0, frame)
    with Image.open(frame).convert("RGB") as image:
        pill_region = image.crop((250, 330, 830, 420))
        assert sum(1 for pixel in pill_region.getdata() if min(pixel) > 245) < 500

    extract_frame(report.path, 5.15, frame)
    with Image.open(frame).convert("RGB") as image:
        outline_region = image.crop((100, 1300, 980, 1450))
        pixels = list(outline_region.getdata())
        assert any(min(pixel) > 220 for pixel in pixels)
        assert any(max(pixel) < 55 for pixel in pixels)

    calls = 0
    import adanvil.media.segment as segment_module

    real_run = segment_module.run

    def count_segment_ffmpeg(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return real_run(*args, **kwargs)

    monkeypatch.setattr(segment_module, "run", count_segment_ffmpeg)
    cached = compose(storyboard, root=root, cache=Cache(root / "cache"))
    assert calls == 0
    assert cached.path.exists()

    preview = compose(storyboard, root=root, quality="preview")
    preview_info = probe(preview.path)
    assert (preview_info.width, preview_info.height) == (540, 960)


def test_compose_progress_reports_segments_and_encode(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    root = _project(media, tmp_path)
    events: list[dict[str, object]] = []

    compose(_storyboard(), root=root, quality="preview", progress=events.append)

    assert [event["stage"] for event in events if event["stage"] == "segments"] == [
        "segments",
        "segments",
        "segments",
        "segments",
    ]
    assert "encode" in [event["stage"] for event in events]
    assert events[-1]["stage"] == "complete"


def test_storyboard_validation_is_strict() -> None:
    image = {"type": "image", "source": "cta.png", "duration": 1}
    with pytest.raises(ValueError, match="unexpected field"):
        Storyboard(name="x", segments=[{**image, "typo": True}])
    with pytest.raises(ValueError, match="greater than"):
        Storyboard(
            name="x",
            segments=[image],
            overlays=[
                {
                    "type": "caption",
                    "style": "pill",
                    "lines": ["x"],
                    "start": 2,
                    "end": 1,
                }
            ],
        )
    with pytest.raises(ValueError, match="escapes"):
        Storyboard(name="x", emoji="../emoji", segments=[image])
    with pytest.raises(ValueError):
        Storyboard(name="x", segments=[])
    with pytest.raises(ValueError, match="mutually exclusive"):
        Storyboard(
            name="x",
            segments=[image],
            audio={"music": "music.m4a", "from_reference": "ref.mp4"},
        )
