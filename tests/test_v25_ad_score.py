"""V2.5: the creative score of an ad (``ad_score``), the choice of a before/after pair
(``compare_pair``, the ``crop`` of a reveal image side), the ``score_*`` pre-check of the
validator, and the docs that describe them."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import pytest
from mcp import Client
from PIL import Image, ImageDraw

from adanvil._bin import ffmpeg_path
from adanvil.media import RevealSegment, SplitSegment
from adanvil.media.segment import ImageSegment, _image_key
from adanvil.project import init_project
from adanvil.qc import PairError, ad_score, compare_pair
from adanvil.qc.pair import compare_images, find_face, verdict, visible_difference
from adanvil.qc.score import (
    THRESHOLDS,
    caption_contrast,
    contrast_ratio,
    detect_text,
    luminance,
    silence_intervals,
)
from adanvil.render import compose
from adanvil.safezones import (
    PLATFORMS,
    SAFE_ZONES,
    check_platforms,
    collisions,
    safe_zones,
    zones_table,
)
from adanvil.server import build_server
from adanvil.storyboard import CaptionOverlay, Storyboard, validate_storyboard
from adanvil.storyboard.craft import (
    CaptionEntry,
    reading_checks,
    reading_need,
    safe_zone_checks,
    story_checks,
)
from adanvil.voice import FakeEngine, VoiceSpec, generate_voice
from tests.test_v24c_reveal import BLUE, RED, _frames, _near, _render, _reveal
from tests.test_visual_grammar import FORMAT, FPS, H, W, _board, _project, _solid

REPO = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- helpers


def _ffmpeg(*args: str | Path) -> None:
    subprocess.run(
        [ffmpeg_path(), "-nostdin", "-y", "-loglevel", "error", *(str(arg) for arg in args)],
        check=True,
    )


def _tone(path: Path, seconds: float, *, frequency: int = 330) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg(
        "-f", "lavfi", "-i", f"sine=frequency={frequency}:sample_rate=48000:duration={seconds}",
        "-c:a", "aac", path,
    )  # fmt: skip
    return path


def _video(
    path: Path,
    video: str,
    seconds: float,
    audio: str | None = "sine=frequency=440",
    audio_filter: str | None = None,
) -> Path:
    """A synthetic 360x640 clip made by ffmpeg from a lavfi ``video`` source; ``audio_filter``
    is an ffmpeg audio filter chain applied to the ``audio`` source."""

    separator = ":" if "=" in video else "="
    args: list[str | Path] = [
        "-f", "lavfi", "-i", f"{video}{separator}s={W}x{H}:r={FPS}:d={seconds}"
    ]  # fmt: skip
    if audio is not None:
        args += ["-f", "lavfi", "-i", f"{audio}:duration={seconds}"]
    args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if audio is not None:
        args += ["-c:a", "aac", "-shortest"]
        if audio_filter:
            args += ["-af", audio_filter]
    path.parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg(*args, path)
    return path


def _caption(lines: list[str], start: float, end: float | None, **extra: Any) -> dict[str, Any]:
    return {
        "type": "caption", "style": "outline", "lines": lines, "start": start, "end": end,
        "y": 120, "size": 24, "max_width": 300, **extra,
    }  # fmt: skip


def _clip(source: str, start: float, end: float, **extra: Any) -> dict[str, Any]:
    return {"type": "clip", "source": f"footage/{source}", "in": start, "out": end, **extra}


def _storyboard(name: str, segments: list, overlays: list, **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "format": FORMAT,
        "emoji": "none",
        "segments": segments,
        "overlays": overlays,
        "audio": {"music": "music/bed.m4a", "music_volume": 0.35},
        "output": {"path": f"renders/{name}.mp4", "quality": "final"},
        **extra,
    }


BAD = _storyboard(
    "bad_ad",
    [
        _clip("grey.mp4", 0, 3, role="hook", shot_type="diagram", label="hook"),
        _clip("moving_long.mp4", 0, 1, role="problem", shot_type="product_hero"),
        _clip("moving_long.mp4", 1, 2, role="problem", shot_type="lifestyle"),
        _clip("moving_long.mp4", 2, 3, role="problem", shot_type="lifestyle"),
        _clip("long_red.mp4", 0, 5, role="problem", shot_type="lifestyle"),
        _clip("moving_long.mp4", 0, 1, role="problem", shot_type="lifestyle"),
        _clip("moving_long.mp4", 1, 2, role="problem", shot_type="lifestyle"),
        _clip("moving_long.mp4", 2, 3, role="demo", shot_type="product_in_use"),
    ],
    [
        _caption(["Eight words on one line stay too short"], 0.0, 1.0, size=14, max_width=340),
        _caption(["Low contrast"], 1.2, 3.0, y=300, fg="#808080", stroke="#808080"),
        _caption(["Big sale today"], 6.0, 9.0, y=330, x_center=240, style="pill"),
    ],
)
GOOD = _storyboard(
    "good_ad",
    [
        _clip("moving.mp4", 0, 1, role="hook", shot_type="reaction", emotion="worried"),
        _clip("moving_long.mp4", 0, 2, role="problem", shot_type="lifestyle"),
        _clip("moving_long.mp4", 2, 3.5, role="problem", shot_type="reaction", emotion="tired"),
        _clip("moving_long.mp4", 0.5, 2.5, role="demo", shot_type="product_in_use"),
        {"type": "image", "source": "footage/orange.png", "duration": 3.0, "role": "cta",
         "label": "end_card"},
    ],
    [
        _caption(["Mornings feel slow"], 0.0, 2.5),
        _caption(["Order now"], 6.0, 8.8, style="box", y=400, size=26),
    ],
)  # fmt: skip


@pytest.fixture(scope="module")
def ads(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = _project(tmp_path_factory.mktemp("v25"))
    _solid(root / "footage" / "long_red.mp4", "red", 6.0)
    _tone(root / "music" / "bed.m4a", 14.0)
    result: dict[str, Any] = {"root": root}
    for raw in (BAD, GOOD):
        board = Storyboard.model_validate(raw)
        report = compose(board, root=root)
        result[raw["name"]] = {
            "raw": raw,
            "video": Path(report.path),
            "score": ad_score(root, Path(report.path), storyboard=raw),
        }
    return result


def _ids(report) -> list[str]:
    return [finding["id"] for finding in report.findings]


def _finding(report, check: str, **detail: Any) -> dict[str, Any]:
    for finding in report.findings:
        if finding["id"] == check and all(
            finding.get("detail", {}).get(key) == value for key, value in detail.items()
        ):
            return finding
    raise AssertionError(f"no {check} {detail} among {_ids(report)}")


# ------------------------------------------------------------------------------ the score


def test_a_weak_ad_scores_low_with_findings_a_good_one_does_not(ads) -> None:
    bad, good = ads["bad_ad"]["score"], ads["good_ad"]["score"]
    assert good.score >= 80 and good.meets_target
    assert bad.score < good.score - 20 and not bad.meets_target
    assert not [f for f in good.findings if f["severity"] == "high"], good.findings
    for report in (bad, good):
        assert set(report.sub_scores) == {
            "hook", "pacing", "reading", "story", "safe_zones", "audio"
        }  # fmt: skip
        assert report.facts["coverage"] == 1.0 and report.facts["storyboard"] is True
        assert report.thresholds["target"] == 80 and report.safe_zones["tiktok"]
    impacts = [finding["impact"] for finding in bad.findings]
    assert impacts == sorted(impacts, reverse=True)
    for finding in bad.findings:
        assert {"id", "area", "severity", "time", "message", "fix", "impact"} <= set(finding)
        assert finding["severity"] in {"high", "medium", "low"} and finding["fix"]


def test_slow_hook_against_fast_hook(ads) -> None:
    bad, good = ads["bad_ad"]["score"], ads["good_ad"]["score"]
    slow = _finding(bad, "hook_static")
    assert slow["time"] == 0.0 and slow["severity"] == "high"
    assert "hook.motion_or_cut" in good.passed and "hook.motion_or_cut" not in bad.passed
    assert "hook_static" not in _ids(good)
    # the opening of the bad ad is a diagram, and the good one a reaction face
    assert "hook_no_subject" in _ids(bad) and "hook.face_or_product" in good.passed
    assert "hook.text_within_half_second" in good.passed and "hook.audio_at_once" in good.passed
    assert bad.sub_scores["hook"]["score"] < good.sub_scores["hook"]["score"]


def test_pacing_flags_a_long_shot_in_the_middle_of_a_fast_ad(ads) -> None:
    bad, good = ads["bad_ad"]["score"], ads["good_ad"]["score"]
    dip = _finding(bad, "pacing_dip")
    assert dip["time"] == pytest.approx(6.0) and dip["end"] == pytest.approx(11.0)
    pacing = bad.facts["pacing"]
    assert pacing["fast_ad"] and pacing["longest_shot"]["duration"] == pytest.approx(5.0)
    assert len(pacing["cuts_per_10s_by_quarter"]) == 4
    assert "pacing.rhythm" in good.passed and good.sub_scores["pacing"]["score"] == 100


def test_a_long_caption_on_screen_too_briefly_is_a_reading_finding(ads) -> None:
    bad, good = ads["bad_ad"]["score"], ads["good_ad"]["score"]
    short = _finding(bad, "reading_time", overlay="0")
    assert short["severity"] == "high" and short["time"] == 0.0
    assert "8 words need 2.6s" in short["message"]
    assert "reading_long_line" in _ids(bad)
    assert "reading.caption_time" in good.passed and "reading.line_length" in good.passed


def test_low_contrast_caption_is_found_on_the_real_frames(ads) -> None:
    bad, good = ads["bad_ad"]["score"], ads["good_ad"]["score"]
    low = _finding(bad, "reading_contrast", overlay="1")
    assert low["detail"]["ratio"] < 2.0 and low["severity"] == "high"
    assert "reading.contrast" in good.passed


def test_a_caption_in_the_tiktok_button_column_is_a_safe_zone_finding(ads) -> None:
    bad, good = ads["bad_ad"]["score"], ads["good_ad"]["score"]
    column = _finding(bad, "safe_zone", platform="tiktok", zone="right_buttons")
    assert column["time"] == pytest.approx(6.0) and "x=900" in column["fix"]
    assert "safe_zones.tiktok" in good.passed and "safe_zone" not in _ids(good)
    assert bad.sub_scores["safe_zones"]["score"] < 100 == good.sub_scores["safe_zones"]["score"]


def test_the_safe_zones_preview_and_finding_frames_are_written(ads) -> None:
    report = ads["bad_ad"]["score"]
    preview = report.images[0]
    assert preview.name == "safe_zones_preview.jpg" and preview.is_file()
    with Image.open(preview) as picture:
        assert picture.width > picture.height * 1.4  # three panels side by side
        pixels = np.asarray(picture.convert("RGB")).astype(int)
    red = (pixels[..., 0] > 200) & (pixels[..., 1] < 90) & (pixels[..., 2] < 90)
    assert red.sum() > 200  # the outline of the caption inside a zone is drawn red
    frames = [path.name for path in report.images[1:]]
    assert 1 <= len(frames) <= 3 and all(name.startswith("finding_") for name in frames)
    assert all(path.is_file() for path in report.images)
    assert ads["good_ad"]["score"].images[0].name == "safe_zones_preview.jpg"


def test_product_too_early_missing_cta_and_no_end_card(ads) -> None:
    bad, good = ads["bad_ad"]["score"], ads["good_ad"]["score"]
    early = _finding(bad, "product_too_early")
    assert early["time"] == pytest.approx(3.0) and early["detail"]["segment"] == 1
    cta = _finding(bad, "cta_missing")
    assert cta["severity"] == "high" and "call to action" in cta["message"]
    assert "cta_not_at_end" in _ids(bad)
    assert {"story.problem_before_product", "story.product_shown", "story.cta_wording",
            "story.cta_at_end"} <= set(good.passed)  # fmt: skip
    assert bad.sub_scores["story"]["score"] <= 60 and good.sub_scores["story"]["score"] == 100


def test_the_score_without_a_storyboard_lists_what_it_skipped(ads) -> None:
    root, video = ads["root"], ads["bad_ad"]["video"]
    report = ad_score(root, video)
    skipped = {item["check"] for item in report.skipped}
    assert {"reading.all", "story.problem_and_product", "story.cta_wording",
            "audio.voice_over_music"} <= skipped  # fmt: skip
    assert report.sub_scores["reading"]["score"] is None
    assert report.facts["storyboard"] is False and report.facts["coverage"] < 1.0
    assert "hook_static" in _ids(report)  # the opening is still static on the pixels
    assert 0 <= report.score <= 100


def test_a_storyboard_that_is_not_the_one_rendered_is_noted(ads) -> None:
    root = ads["root"]
    report = ad_score(root, ads["good_ad"]["video"], storyboard=BAD)
    assert any("rendered from" in note for note in report.notes)


def test_scoring_errors_are_clear(ads, tmp_path: Path) -> None:
    from adanvil.qc import ScoreError

    root = ads["root"]
    with pytest.raises(ScoreError, match="not valid"):
        ad_score(root, ads["good_ad"]["video"], storyboard={"name": "x"})
    with pytest.raises(ScoreError, match="not a video"):
        ad_score(root, root / "music" / "bed.m4a")
    with pytest.raises(ValueError, match="platforms"):
        ad_score(root, ads["good_ad"]["video"], platforms=["myspace"])


def test_platforms_can_be_chosen(ads) -> None:
    root, video = ads["root"], ads["bad_ad"]["video"]
    report = ad_score(root, video, storyboard=BAD, platforms=["stories"])
    assert set(report.safe_zones) == {"stories"}
    zones = {f["detail"]["platform"] for f in report.findings if f["id"] == "safe_zone"}
    assert zones <= {"stories"}


# ------------------------------------------------------------------- audio and the pixels


def test_a_silent_opening_and_a_long_silence_in_the_audio(tmp_path: Path) -> None:
    root = init_project(tmp_path, "Sunbrew kettles")
    late = _video(
        root / "renders" / "late.mp4", "testsrc2", 6.0,
        audio_filter="adelay=1500:all=1",
    )  # fmt: skip
    gap = _video(
        root / "renders" / "gap.mp4", "testsrc2", 8.0,
        audio_filter="volume=enable='between(t,4,5.5)':volume=0",
    )  # fmt: skip
    report = ad_score(root, late)
    hook = _finding(report, "hook_silent_start")
    assert hook["end"] == pytest.approx(1.5, abs=0.2) and hook["severity"] == "medium"
    assert "audio_silence" not in _ids(report)  # the silent opening is the hook's finding
    quiet = ad_score(root, gap)
    silence = _finding(quiet, "audio_silence")
    assert silence["time"] == pytest.approx(4.0, abs=0.2)
    assert silence["end"] == pytest.approx(5.5, abs=0.2)
    assert "hook.audio_at_once" in quiet.passed
    intervals = silence_intervals(gap, 8.0)
    assert any(abs(a - 4.0) < 0.2 and abs(b - 5.5) < 0.2 for a, b in intervals)


def test_no_audio_track_and_a_loudness_out_of_range(tmp_path: Path) -> None:
    root = init_project(tmp_path, "Sunbrew kettles")
    silent = _video(root / "renders" / "silent.mp4", "testsrc2", 4.0, audio=None)
    assert _finding(ad_score(root, silent), "audio_missing")["severity"] == "high"
    quiet = _video(root / "renders" / "quiet.mp4", "testsrc2", 4.0, audio_filter="volume=0.004")
    finding = _finding(ad_score(root, quiet), "loudness")
    assert finding["severity"] == "high" and "LUFS" in finding["message"]


def test_scene_cuts_are_detected_without_a_storyboard(tmp_path: Path) -> None:
    root = init_project(tmp_path, "Sunbrew kettles")
    path = root / "renders" / "cuts.mp4"
    inputs: list[str] = []
    for colour in ("white", "black", "white", "black"):
        inputs += ["-f", "lavfi", "-i", f"color=c={colour}:s={W}x{H}:r={FPS}:d=0.8"]
    _ffmpeg(
        *inputs, "-f", "lavfi", "-i", "sine=frequency=440:duration=3.2",
        "-filter_complex", "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0[v]",
        "-map", "[v]", "-map", "4:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-shortest", path,
    )  # fmt: skip
    report = ad_score(root, path)
    cuts = report.facts["pacing"]["cuts"]
    assert len(cuts) == 3 and cuts == pytest.approx([0.8, 1.6, 2.4], abs=0.1)
    assert "hook.motion_or_cut" in report.passed  # a cut at 0.8 s


def test_a_static_long_ad_is_slow(tmp_path: Path) -> None:
    root = init_project(tmp_path, "Sunbrew kettles")
    path = _video(root / "renders" / "still.mp4", "color=c=gray", 12.0)
    report = ad_score(root, path)
    slow = _finding(report, "pacing_slow")
    assert slow["severity"] == "medium" and "0 cut" in slow["message"]
    assert "long_shot" in _ids(report) and "hook_static" in _ids(report)


def test_voice_masked_by_loud_music_is_a_finding_a_quiet_bed_is_not(ads) -> None:
    root = ads["root"]
    generate_voice(
        root, VoiceSpec(engine="fake", lines=["Meet the kettle", "Boils fast"]), "narr",
        engine=FakeEngine(),
    )  # fmt: skip
    video = ads["good_ad"]["video"]

    def audio(volume: float) -> dict[str, Any]:
        raw = copy.deepcopy(GOOD)
        raw["audio"] = {
            "voice": "voice/narr/manifest.json", "music": "music/bed.m4a", "music_volume": volume
        }
        return ad_score(root, video, storyboard=raw)

    loud = audio(0.6)
    masked = _finding(loud, "voice_masked")
    assert masked["severity"] == "high" and masked["detail"]["margin_db"] < 3
    assert "audio.music_volume" in masked["fix"] or "music_volume" in masked["fix"]
    default = audio(0.35)  # the documented default sits about 5 dB under the voice and passes
    assert "audio.voice_over_music" in default.passed and "voice_masked" not in _ids(default)
    close = audio(0.42)  # a little louder: under the 4.5 dB bar but not masked
    near = _finding(close, "voice_masked")
    assert near["severity"] == "low" and 3 <= near["detail"]["margin_db"] < 4.5
    clear = audio(0.25)
    assert "audio.voice_over_music" in clear.passed and "voice_masked" not in _ids(clear)
    assert clear.facts["audio"]["voice_over_music_db"] >= 6


# --------------------------------------------------------------- units: contrast, text


def test_contrast_ratio_is_the_wcag_one() -> None:
    assert contrast_ratio(1.0, 0.0) == pytest.approx(21.0)
    assert contrast_ratio(0.5, 0.5) == 1.0
    white = luminance(np.array([255, 255, 255], dtype=np.uint8))
    black = luminance(np.array([0, 0, 0], dtype=np.uint8))
    assert float(white) == pytest.approx(1.0) and float(black) == 0.0
    assert THRESHOLDS["contrast_warn"] == 3.0


def _entry(**extra: Any) -> CaptionEntry:
    caption = CaptionOverlay.model_validate(
        {"style": "outline", "lines": ["Hello there"], "y": 250, "size": 48, **extra}
    )
    return CaptionEntry("0", caption, "written", 0.0, 2.0)


def test_caption_contrast_reads_the_pixels_around_the_letters(tmp_path: Path) -> None:
    grey = Image.new("RGB", (W, H), (128, 128, 128))
    flat = caption_contrast(
        grey, _entry(fg="#808080", stroke="#808080"), root=tmp_path, brand=None
    )
    assert flat is not None and flat["ratio"] < 1.2
    dark = Image.new("RGB", (W, H), (10, 10, 10))
    legible = caption_contrast(dark, _entry(fg="#FFFFFF"), root=tmp_path, brand=None)
    assert legible is not None and legible["ratio"] >= 10
    white = Image.new("RGB", (W, H), (250, 250, 250))
    washed = caption_contrast(white, _entry(fg="#FFFFFF"), root=tmp_path, brand=None)
    assert washed is not None and washed["ratio"] < 1.2


def test_detect_text_finds_outlined_lettering_not_a_plain_frame() -> None:
    frame = Image.new("RGB", (270, 480), (90, 120, 160))
    draw = ImageDraw.Draw(frame)
    for x in range(40, 230, 12):
        draw.rectangle((x, 300, x + 6, 330), fill=(0, 0, 0))
        draw.rectangle((x + 1, 302, x + 5, 328), fill=(255, 255, 255))
    boxes = detect_text(np.asarray(frame))
    assert boxes and 0.55 < boxes[0][1] < 0.72 and boxes[0][2] - boxes[0][0] > 0.4
    assert detect_text(np.asarray(Image.new("RGB", (270, 480), (90, 120, 160)))) == []


# ----------------------------------------------------------------- units: the craft checks


def _model(overlays: list, segments: list | None = None, **extra: Any) -> Storyboard:
    return Storyboard.model_validate(
        _storyboard(
            "unit",
            segments or [_clip("moving_long.mp4", 0, 3), _clip("moving_long.mp4", 0, 3)],
            overlays,
            **extra,
        )
    )


def _entries(model: Storyboard) -> list[CaptionEntry]:
    return [
        CaptionEntry(
            str(i), o, "written", o.start, model.total_duration() if o.end is None else o.end
        )
        for i, o in enumerate(model.overlays)
        if isinstance(o, CaptionOverlay)
    ]


def test_reading_time_needs_a_quarter_second_a_word_plus_six_tenths() -> None:
    assert reading_need(1) == 1.2 and reading_need(4) == pytest.approx(1.6)
    assert reading_need(8) == pytest.approx(2.6)
    long_short = _model([_caption(["one two three four five six seven eight"], 0.0, 1.4)])
    checks = reading_checks(_entries(long_short))
    finding = next(f for f in checks.findings if f.id == "reading_time")
    assert finding.severity == "medium" or finding.severity == "high"
    assert "8 words need 2.6s" in finding.message
    enough = _model([_caption(["one two three four five six seven eight"], 0.0, 2.7)])
    ok = reading_checks(_entries(enough))
    assert not [f for f in ok.findings if f.id == "reading_time"]
    assert "reading.caption_time" in ok.passed
    # exactly 7 words on a line is fine, 8 is not
    seven = reading_checks(_entries(_model([_caption(["a b c d e f g"], 0.0, 3.0)])))
    assert "reading.line_length" in seven.passed
    eight = reading_checks(_entries(_model([_caption(["a b c d e f g h"], 0.0, 3.0)])))
    assert [f.id for f in eight.findings] == ["reading_long_line"]
    split = reading_checks(_entries(_model([_caption(["a b c d", "e f g h"], 0.0, 3.0)])))
    assert not split.findings  # two lines of four words


def test_story_checks_cta_wording_end_card_and_order(tmp_path: Path) -> None:
    from adanvil.brand import Brand, Product

    def run_story(model: Storyboard, brand=None):
        timeline = model.timeline()
        return story_checks(
            model, brand, _entries(model), [], [t[0] for t in timeline], [t[1] for t in timeline]
        )

    plain = _model([_caption(["Hello"], 0.0, 2.0)])
    missing = run_story(plain)
    assert {"cta_missing", "cta_not_at_end"} <= {f.id for f in missing.findings}
    wording = _model([_caption(["Order now"], 4.0, 6.0)])
    found = run_story(wording)
    assert "story.cta_wording" in found.passed and "story.cta_at_end" in found.passed
    early_cta = _model([_caption(["Order now"], 0.0, 2.0)])
    assert "cta_not_at_end" in {f.id for f in run_story(early_cta).findings}
    branded = Brand(name="Sunbrew", product=Product(cta_text="Brew yours"))
    other = run_story(wording, branded)
    assert [f.id for f in other.findings if f.id.startswith("cta_w")] == ["cta_wording"]
    exact = run_story(_model([_caption(["Brew yours today"], 3.0, 6.0)]), branded)
    assert "story.cta_wording" in exact.passed
    card = _model(
        [],
        [_clip("moving_long.mp4", 0, 3), {"type": "image", "source": "footage/orange.png",
                                          "duration": 3, "label": "end_card"}],
    )  # fmt: skip
    assert not [f for f in run_story(card).findings if f.id.startswith("cta")]


def test_safe_zone_boxes_and_collisions() -> None:
    assert set(SAFE_ZONES) == set(PLATFORMS) | {"feed"}  # feed: the 4:5 and 1:1 placement (V2.8)
    assert PLATFORMS == ("tiktok", "reels", "stories")  # the default of ad_score
    assert set(SAFE_ZONES["tiktok"]) == {"top_bar", "right_buttons", "bottom_band"}
    assert "right_buttons" not in SAFE_ZONES["stories"]
    half = safe_zones("tiktok", (540, 960))
    assert half["right_buttons"] == (450.0, 380.0, 540.0, 740.0)
    hit = collisions((880, 900, 1000, 960), ["tiktok", "reels", "stories"])
    assert [(h["platform"], h["zone"]) for h in hit] == [("tiktok", "right_buttons")]
    assert collisions((100, 400, 700, 460), PLATFORMS) == []
    graze = collisions((100, 1470, 700, 1490), ["tiktok"])  # 10 px into the band: not counted
    assert graze == []
    assert check_platforms(None) == PLATFORMS and zones_table(["reels"]).keys() == {"reels"}
    with pytest.raises(ValueError):
        safe_zones("snapchat")
    inside = {
        "kind": "caption", "label": "x", "start": 0.0, "end": 2.0, "box": (100, 1600, 700, 1700)
    }  # fmt: skip
    checks, items = safe_zone_checks([inside], PLATFORMS)
    assert items[0]["inside"] and {f.detail["platform"] for f in checks.findings} == set(PLATFORMS)
    assert all(f.id == "safe_zone" and f.severity == "high" for f in checks.findings)


# ------------------------------------------------------------------- the score_ pre-check


def test_the_validator_adds_score_warnings_only_when_asked(ads) -> None:
    root = ads["root"]
    plain = validate_storyboard(BAD, root=root)
    assert plain.ok and not [i for i in plain.warnings if i.code.startswith("score_")]
    assert "score_precheck" not in plain.summary
    asked = validate_storyboard(BAD, root=root, score_check=True)
    codes = {issue.code for issue in asked.warnings if issue.code.startswith("score_")}
    assert {
        "score_reading_time", "score_reading_long_line", "score_product_too_early",
        "score_cta_missing", "score_safe_zone",
    } <= codes  # fmt: skip
    timing = next(i for i in asked.warnings if i.code == "score_reading_time")
    assert timing.where == "overlays[0]" and "Fix:" in timing.message
    assert asked.summary["score_precheck"]["passed"] is not None
    good = validate_storyboard(GOOD, root=root, score_check=True)
    assert not [i for i in good.warnings if i.code.startswith("score_")], good.warnings
    assert "reading.caption_time" in good.summary["score_precheck"]["passed"]


# --------------------------------------------------------------------------- compare_pair


def _face(path: Path, *, skin=(224, 172, 105), background=(70, 100, 150), spots=0,
          noise: int = 0, seed: int = 1, size=(540, 720)) -> Path:  # fmt: skip
    """A drawn portrait: a skin-coloured oval face on a background, with optional dark spots."""

    image = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(image)
    cx, cy = size[0] // 2, size[1] // 2
    draw.ellipse((cx - 150, cy - 200, cx + 150, cy + 200), fill=skin)
    draw.ellipse((cx - 85, cy - 70, cx - 45, cy - 40), fill=(40, 30, 30))
    draw.ellipse((cx + 45, cy - 70, cx + 85, cy - 40), fill=(40, 30, 30))
    rng = np.random.default_rng(seed)
    for _ in range(spots):
        x = int(rng.integers(cx - 110, cx + 110))
        y = int(rng.integers(cy - 150, cy + 150))
        draw.ellipse((x, y, x + 18, y + 18), fill=(120, 40, 40))
    if noise:
        array = np.asarray(image).astype(int)
        array += rng.integers(-noise, noise + 1, size=array.shape)
        image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def test_find_face_uses_the_skin_region(tmp_path: Path) -> None:
    with Image.open(_face(tmp_path / "a.png")) as picture:
        face = find_face(picture)
    assert face is not None
    left, top, right, bottom = face["box"]
    assert 90 < left < 140 and 90 < top < 180 and 400 < right < 450 and 540 < bottom < 640
    assert find_face(Image.new("RGB", (300, 300), (70, 100, 150))) is None


def test_near_identical_frames_score_low_and_a_different_pair_high(tmp_path: Path) -> None:
    root = init_project(tmp_path, "Sunbrew kettles")
    first = _face(root / "footage" / "a.png", noise=3, seed=1)
    second = _face(root / "footage" / "b.png", noise=3, seed=2)
    result, preview = compare_pair(root, first, second)
    assert result["region"] == "face" and result["visible_difference"] < 12
    assert result["verdict"] == "not_visible" and preview.is_file() and preview.suffix == ".jpg"

    different = _face(
        root / "footage" / "c.png", skin=(238, 205, 180), background=(200, 190, 60), spots=0
    )
    spotted = _face(root / "footage" / "d.png", skin=(190, 140, 110), spots=45, seed=7)
    clear = compare_pair(root, spotted, different)[0]
    assert clear["region"] == "face" and clear["visible_difference"] > 30
    assert clear["visible_difference"] > result["visible_difference"] + 25
    ranked = sorted(
        [("same", result), ("changed", clear)], key=lambda item: -item[1]["visible_difference"]
    )
    assert ranked[0][0] == "changed"


def test_two_rooms_are_compared_on_the_whole_frame_and_sizes_are_aligned(tmp_path: Path) -> None:
    root = init_project(tmp_path, "Sunbrew kettles")
    messy = Image.new("RGB", (600, 800), (190, 180, 160))
    draw = ImageDraw.Draw(messy)
    rng = np.random.default_rng(3)
    for _ in range(40):
        x, y = int(rng.integers(0, 560)), int(rng.integers(0, 760))
        draw.rectangle((x, y, x + 60, y + 40), fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    messy.save(root / "footage" / "messy.png")
    Image.new("RGB", (900, 1200), (225, 225, 220)).save(root / "footage" / "tidy.png")
    result, _ = compare_pair(root, root / "footage" / "messy.png", root / "footage" / "tidy.png")
    assert result["region"] == "frame" and result["visible_difference"] > 55
    assert result["verdict"] == "obvious"
    same = compare_pair(root, root / "footage" / "tidy.png", root / "footage" / "tidy.png")[0]
    assert same["visible_difference"] == 0 and same["verdict"] == "not_visible"
    shapes = compare_images(Image.new("RGB", (300, 400)), Image.new("RGB", (400, 300)))
    assert any("different shapes" in note for note in shapes["notes"])


def test_compare_pair_takes_frames_of_clips_and_crops_of_a_composite(tmp_path: Path) -> None:
    root = init_project(tmp_path, "Sunbrew kettles")
    clip = root / "footage" / "change.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", f"color=c=red:s={W}x{H}:r={FPS}:d=2",
        "-f", "lavfi", "-i", f"color=c=blue:s={W}x{H}:r={FPS}:d=2",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", clip,
    )  # fmt: skip
    across = compare_pair(root, clip, clip, before_t=0.5, after_t=3.0)[0]
    assert across["visible_difference"] > 55 and across["before"]["t"] == 0.5
    within = compare_pair(root, clip, clip, before_t=0.5, after_t=1.5)[0]
    assert within["visible_difference"] < 12
    composite = Image.new("RGB", (800, 600), (240, 240, 235))
    left = Image.open(_face(root / "footage" / "l.png", spots=40, seed=5)).resize((400, 600))
    right = Image.open(_face(root / "footage" / "r.png", skin=(238, 205, 180))).resize((400, 600))
    composite.paste(left, (0, 0))
    composite.paste(right, (400, 0))
    composite.save(root / "footage" / "composite.png")
    path = root / "footage" / "composite.png"
    halves = compare_pair(
        root, path, path,
        before_crop={"x": 0, "y": 0, "w": 400, "h": 600},
        after_crop={"x": 400, "y": 0, "w": 400, "h": 600},
    )[0]  # fmt: skip
    assert halves["visible_difference"] > 20 and halves["before"]["crop"]["w"] == 400
    whole = compare_pair(root, path, path)[0]
    assert whole["visible_difference"] == 0
    with pytest.raises(PairError, match="leaves"):
        compare_pair(root, path, path, after_crop={"x": 700, "y": 0, "w": 400, "h": 600})
    with pytest.raises(PairError, match="outside the clip"):
        compare_pair(root, clip, clip, before_t=99)
    with pytest.raises(PairError, match="audio"):
        compare_pair(root, root / "music" / "none.m4a", clip) if False else compare_pair(
            root, _tone(root / "footage" / "voice.m4a", 1.0), clip
        )


def test_visible_difference_bands() -> None:
    assert visible_difference(0.0, 0.0) == 0.0
    assert visible_difference(0.12, 0.30) == 100.0
    assert [verdict(v) for v in (0, 11.9, 12, 29.9, 30, 54.9, 55, 100)] == [
        "not_visible", "not_visible", "subtle", "subtle", "visible", "visible", "obvious",
        "obvious",
    ]  # fmt: skip


# ---------------------------------------------------------- reveal and split image crop


def _composite(root: Path) -> None:
    image = Image.new("RGB", (W * 2, H), (255, 0, 0))
    image.paste(Image.new("RGB", (W, H), (0, 0, 255)), (W, 0))
    image.save(root / "footage" / "composite_rb.png")


def test_a_reveal_side_can_be_cut_out_of_a_composite_with_crop(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _composite(root)
    raw = _reveal(
        before={"source": "footage/composite_rb.png", "crop": {"x": 0, "y": 0, "w": W, "h": H}},
        after={"source": "footage/composite_rb.png", "crop": {"x": W, "y": 0, "w": W, "h": H}},
    )
    frames = _frames(_render(root, raw))
    assert _near(frames[2], RED, 30).all() and _near(frames[-3], BLUE, 30).all()
    seg = RevealSegment.model_validate(raw)
    assert seg.before.crop.w == W and seg.after.crop.x == W
    # without crop the side is the whole composite, cover-cropped around its centre
    plain = _frames(_render(root, _reveal(before={"source": "footage/composite_rb.png"})))
    assert not _near(plain[2], RED, 30).all()


def test_a_split_panel_image_takes_crop_too(tmp_path: Path) -> None:
    from adanvil.media import Cache, Format, render_segment

    root = _project(tmp_path)
    _composite(root)
    seg = SplitSegment.model_validate(
        {
            "type": "split", "layout": "side_by_side", "duration": 1.0, "gap": 0,
            "panels": [
                {"source": "footage/composite_rb.png", "crop": {"x": 0, "y": 0, "w": W, "h": H}},
                {"source": "footage/composite_rb.png", "crop": {"x": W, "y": 0, "w": W, "h": H}},
            ],
        }
    )  # fmt: skip
    path = render_segment(
        seg, root=root, cache=Cache(root / "cache"), fmt=Format(width=W, height=H, fps=FPS)
    )
    frame = _frames(path)[5]
    assert _near(frame[300, 20], RED, 30).all() and _near(frame[300, W - 20], BLUE, 30).all()


def test_image_crop_is_validated_and_does_not_change_old_cache_keys(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _composite(root)
    segments = [
        {
            "type": "reveal", "duration": 3.0,
            "before": {"source": "footage/composite_rb.png",
                       "crop": {"x": 0, "y": 0, "w": 5000, "h": H}},
            "after": {"source": "footage/composite_rb.png",
                      "crop": {"x": W, "y": 0, "w": W, "h": H}},
        }
    ]  # fmt: skip
    report = validate_storyboard(_board(segments), root=root)
    assert "panel_crop" in {issue.code for issue in report.errors}
    ok = segments[0] | {
        "before": {"source": "footage/composite_rb.png", "crop": {"x": 0, "y": 0, "w": W, "h": H}}
    }
    assert validate_storyboard(_board([ok]), root=root).ok
    plain = {"type": "image", "source": "footage/composite_rb.png", "duration": 2,
             "crop": {"x": 0, "y": 0, "w": W, "h": H}}  # fmt: skip
    assert validate_storyboard(_board([plain]), root=root).ok
    too_wide = plain | {"crop": {"x": 0, "y": 0, "w": W * 3, "h": H}}
    report = validate_storyboard(_board([too_wide]), root=root)
    assert [issue.where for issue in report.errors if issue.code == "panel_crop"] == [
        "segments[0].crop"
    ]
    again = Storyboard.model_validate(
        Storyboard.model_validate(_board([plain]).model_dump(mode="json")).model_dump(
            mode="json", by_alias=True
        )
    )  # an image segment without crop keeps its round trip (crop is None in the dump)
    assert again.segments[0].crop.w == W
    with pytest.raises(ValueError, match="crop"):
        Storyboard.model_validate(
            {"name": "x", "segments": [{"type": "image", "source": "a.png", "duration": 1,
                                        "crop": {"x": 0, "y": 0, "w": 10, "h": 10, "z": 1}}]}
        )
    key = _image_key(Path(__file__), ImageSegment(source="cta.png", duration=2),
                     __import__("adanvil.media", fromlist=["Format"]).Format())  # fmt: skip
    assert "crop" not in key["segment"]


# -------------------------------------------------------------------------- tool contract


async def _tools(root: Path, video: Path, sb_path: str) -> dict[str, Any]:
    results: dict[str, Any] = {}
    async with Client(build_server()) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        results["tools"] = tools
        scored = await client.call_tool(
            "ad_score",
            {"project_dir": str(root), "video": video.relative_to(root).as_posix(),
             "storyboard": sb_path},
        )  # fmt: skip
        results["score"] = scored
        pair = await client.call_tool(
            "compare_pair",
            {"project_dir": str(root), "before": "footage/composite_rb.png",
             "after": "footage/composite_rb.png",
             "before_crop": {"x": 0, "y": 0, "w": W, "h": H},
             "after_crop": {"x": W, "y": 0, "w": W, "h": H}},
        )  # fmt: skip
        results["pair"] = pair
        results["bad_video"] = await client.call_tool(
            "ad_score", {"project_dir": str(root), "video": "footage/missing.mp4"}
        )
        results["bad_pair"] = await client.call_tool(
            "compare_pair",
            {"project_dir": str(root), "before": "footage/composite_rb.png",
             "after": "footage/composite_rb.png",
             "after_crop": {"x": 5, "y": 0, "w": 9999, "h": 5}},
        )  # fmt: skip
        results["validate"] = await client.call_tool(
            "storyboard_validate",
            {"project_dir": str(root), "storyboard": sb_path, "score_check": True},
        )
    return results


def test_the_tools_are_listed_and_work_through_mcp(ads) -> None:
    root = ads["root"]
    _composite(root)
    (root / "storyboards").mkdir(exist_ok=True)
    (root / "storyboards" / "bad_ad.json").write_text(json.dumps(BAD), encoding="utf-8")
    results = anyio.run(_tools, root, ads["bad_ad"]["video"], "storyboards/bad_ad.json")
    tools = results["tools"]
    assert {"ad_score", "compare_pair"} <= set(tools)
    properties = {name: set(tool.input_schema["properties"]) for name, tool in tools.items()}
    assert {"project_dir", "video", "storyboard", "platforms"} <= properties["ad_score"]
    assert {"before", "after", "before_t", "after_t", "before_crop", "after_crop"} <= properties[
        "compare_pair"
    ]
    assert "score_check" in properties["storyboard_validate"]
    for name in ("ad_score", "compare_pair"):
        assert tools[name].description and "LOOK" in tools[name].description
    description = tools["ad_score"].description
    assert "80" in description and "safe_zones_preview" in description

    score = results["score"]
    payload = json.loads([c.text for c in score.content if c.type == "text"][0])
    assert payload["score"] == ads["bad_ad"]["score"].score and payload["findings"]
    assert payload["images"][0].endswith("safe_zones_preview.jpg")
    assert sum(item.type == "image" for item in score.content) >= 2
    pair = results["pair"]
    pair_payload = json.loads([c.text for c in pair.content if c.type == "text"][0])
    assert pair_payload["visible_difference"] > 55 and pair_payload["preview"].endswith(".jpg")
    assert any(item.type == "image" for item in pair.content)
    assert results["bad_video"].is_error and results["bad_pair"].is_error
    validated = json.loads([c.text for c in results["validate"].content if c.type == "text"][0])
    assert any(issue["code"].startswith("score_") for issue in validated["warnings"])


# ------------------------------------------------------------------------------- the docs


def test_the_guides_describe_the_score_the_zones_and_the_pair_rule() -> None:
    agents = (REPO / "AGENTS.md").read_text(encoding="utf-8")
    skill = (REPO / "skill" / "SKILL.md").read_text(encoding="utf-8")
    guide = (REPO / "docs" / "EDITING_GUIDE.md").read_text(encoding="utf-8")
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for name, text in (("AGENTS.md", agents), ("SKILL.md", skill), ("guide", guide)):
        for word in (
            "ad_score", "compare_pair", "safe_zones_preview.jpg", "score_check",
            "visible_difference", "0.25 s per word", "7 words", "3:1", "4.5 dB", "0.7 s",
            "80 or more", "tiktok", "reels", "stories", "crop", "same pose",
        ):  # fmt: skip
            assert word in text, (name, word)
        plain = text.replace("`", "")
        for platform, zones in SAFE_ZONES.items():
            for zone, box in zones.items():
                assert f"{platform} {zone} {box}" in plain, (name, platform, zone)
    section = "### Before delivery: score the ad"
    assert section in agents and section in skill
    assert agents.split(section, 1)[1].split("###", 1)[0] == skill.split(section, 1)[1].split(
        "###", 1
    )[0]
    for tool in ("ad_score", "compare_pair"):
        assert f"`{tool}`" in readme
    assert "Tools (43)" in agents and "Tools (43)" in skill and "Tools (43)" in readme
    assert (REPO / "src" / "adanvil" / "assets" / "EDITING_GUIDE.md").read_bytes() == (
        REPO / "docs" / "EDITING_GUIDE.md"
    ).read_bytes()
