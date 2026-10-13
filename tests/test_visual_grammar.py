"""Visual devices (V2.4): split screens, video insets, stickers, progress bar, punch-in."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from adanvil._bin import ffmpeg_path
from adanvil.ffmpeg import probe, run
from adanvil.media import (
    Cache,
    ClipSegment,
    Format,
    SplitSegment,
    render_segment,
    segment_duration,
    split_geometry,
)
from adanvil.project import init_project
from adanvil.qc import qc_report
from adanvil.render import compose
from adanvil.render.compose import _with_inset_voices
from adanvil.render.stickers import draw_progress, plan_sticker, sticker_art, sticker_scales
from adanvil.storyboard import (
    StickerOverlay,
    Storyboard,
    storyboard_json_schema,
    validate_storyboard,
)

FPS = 30
W, H = 360, 640
FORMAT = {"width": W, "height": H, "fps": FPS}
GREEN = (0, 200, 0)


def _encode(path: Path, source: str, duration: float, size: str = f"{W}x{H}", audio: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg_path(), "-nostdin", "-y", "-loglevel", "error"]
    command += ["-t", str(duration), "-f", "lavfi", "-i", f"{source.format(size=size)}"]
    if audio:
        command += ["-t", str(duration), "-f", "lavfi", "-i", "sine=frequency=440"]
    command += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if audio:
        command += ["-c:a", "aac", "-shortest"]
    command.append(str(path))
    subprocess.run(command, check=True)
    return path


def _solid(path: Path, colour: str, duration: float = 3.0, **kw) -> Path:
    return _encode(path, f"color=c={colour}:s={{size}}:r={FPS}", duration, **kw)


def _moving(path: Path, duration: float = 3.0, size: str = f"{W}x{H}") -> Path:
    return _encode(path, f"testsrc2=s={{size}}:r={FPS}", duration, size=size)


def _project(tmp_path: Path) -> Path:
    init_project(tmp_path, "Sunbrew kettles")
    footage = tmp_path / "footage"
    _solid(footage / "grey.mp4", "0x808080", 4.0, audio=True)
    _solid(footage / "red.mp4", "red", 3.0)
    _solid(footage / "blue.mp4", "blue", 3.0)
    _solid(footage / "green.mp4", "0x00C800", 3.0, audio=True)
    _moving(footage / "moving.mp4", 1.0)
    _moving(footage / "moving_long.mp4", 4.0)
    _encode(
        footage / "rect.mp4",
        f"color=c=black:s={{size}}:r={FPS},drawbox=x=140:y=270:w=80:h=100:color=white:t=fill",
        3.0,
    )
    Image.new("RGB", (W, H), (200, 120, 40)).save(footage / "orange.png")
    return tmp_path


def _frames(video: Path, width: int = W, height: int = H) -> np.ndarray:
    raw = run(
        ["-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_stdout=True
    ).stdout
    return np.frombuffer(raw or b"", dtype=np.uint8).reshape(-1, height, width, 3)


def _near(frame: np.ndarray, colour: tuple[int, int, int], tolerance: int = 70) -> np.ndarray:
    return np.all(np.abs(frame.astype(int) - np.array(colour)) <= tolerance, axis=-1)


def _board(segments: list[dict], overlays: list[dict] | None = None, **extra) -> Storyboard:
    return Storyboard.model_validate(
        {
            "name": "visual",
            "format": FORMAT,
            "emoji": "none",
            "segments": segments,
            "overlays": overlays or [],
            **extra,
        }
    )


def _split(**kw) -> dict:
    return {
        "type": "split",
        "duration": 2.0,
        "panels": [{"source": "footage/red.mp4"}, {"source": "footage/blue.mp4"}],
        **kw,
    }


# ------------------------------------------------------------------------------ split


@pytest.mark.parametrize("layout", ["side_by_side", "top_bottom"])
@pytest.mark.parametrize("gap", [0, 7, 8, 21])
def test_split_geometry_fills_the_frame_with_even_panels(layout: str, gap: int) -> None:
    boxes, real = split_geometry(layout, 1080, 1920, gap)
    assert real >= gap and real - gap <= 3
    first, second = boxes
    assert all(value % 2 == 0 for box in boxes for value in box)
    if layout == "side_by_side":
        assert first[2] == second[2] and first[2] * 2 + real == 1080
        assert second[0] == first[2] + real and first[3] == 1920
    else:
        assert first[3] == second[3] and first[3] * 2 + real == 1920
        assert second[1] == first[3] + real and first[2] == 1080


def test_split_mezzanine_has_both_halves_gap_and_exact_frames(tmp_path: Path) -> None:
    root = _project(tmp_path)
    seg = SplitSegment.model_validate(_split(gap=8, gap_color="#00FF00"))
    assert segment_duration(seg) == 2.0
    path = render_segment(
        seg, root=root, cache=Cache(root / "cache"), fmt=Format(width=W, height=H, fps=FPS)
    )
    info = probe(path)
    assert (info.width, info.height, info.fps, info.pix_fmt) == (W, H, 30, "yuv420p")
    assert (info.color_space, info.color_primaries, info.color_transfer) == ("bt709",) * 3
    frames = _frames(path)
    assert len(frames) == 60
    frame = frames[30]
    assert _near(frame[300, 80], (255, 0, 0)).all()  # the left panel is red
    assert _near(frame[300, 280], (0, 0, 255)).all()  # the right panel is blue
    assert _near(frame[300, 180], (0, 255, 0), 30).all()  # the gap
    boxes, real = split_geometry("side_by_side", W, H, 8)
    gap_columns = [x for x in range(W) if _near(frame[300, x], (0, 255, 0), 30).all()]
    assert (min(gap_columns), max(gap_columns) + 1) == (boxes[0][2], boxes[1][0])


def test_split_top_bottom_and_image_panel(tmp_path: Path) -> None:
    root = _project(tmp_path)
    seg = SplitSegment.model_validate(
        _split(
            layout="top_bottom",
            panels=[{"source": "footage/orange.png"}, {"source": "footage/blue.mp4"}],
        )
    )
    path = render_segment(
        seg, root=root, cache=Cache(root / "cache"), fmt=Format(width=W, height=H, fps=FPS)
    )
    frame = _frames(path)[10]
    assert _near(frame[100, 180], (200, 120, 40), 30).all()
    assert _near(frame[500, 180], (0, 0, 255)).all()
    assert _near(frame[H // 2, 180], (0, 0, 0), 20).all()  # the black gap, 8 px thick


def test_split_labels_are_drawn_and_a_short_clip_freezes(tmp_path: Path) -> None:
    root = _project(tmp_path)
    fmt = Format(width=W, height=H, fps=FPS)
    panels = [{"source": "footage/moving.mp4"}, {"source": "footage/blue.mp4"}]
    plain = SplitSegment.model_validate(_split(panels=panels))
    labelled = SplitSegment.model_validate(_split(panels=panels, labels=["BEFORE", "AFTER"]))
    cache = Cache(root / "cache")
    plain_frames = _frames(render_segment(plain, root=root, cache=cache, fmt=fmt))
    label_frames = _frames(render_segment(labelled, root=root, cache=cache, fmt=fmt))
    assert len(plain_frames) == len(label_frames) == 60
    # the label card sits near the top of the (blue) right panel: white pixels appear there
    top_right = (slice(20, 120), slice(W // 2 + 4, W))
    assert (label_frames[30][top_right] > 235).all(axis=-1).sum() > 300
    assert (plain_frames[30][top_right] > 235).all(axis=-1).sum() == 0
    # the 1 s clip is over after 30 frames: its last frame is held for the second second
    left = (slice(0, H), slice(0, W // 2 - 4))
    held = np.abs(plain_frames[40][left].astype(int) - plain_frames[59][left].astype(int)).mean()
    moving = np.abs(plain_frames[5][left].astype(int) - plain_frames[25][left].astype(int)).mean()
    assert held < 2.0 < moving


def test_split_old_storyboards_are_unchanged() -> None:
    old = {
        "name": "old",
        "segments": [{"type": "clip", "source": "a.mp4", "in": 0, "out": 1}],
    }
    assert Storyboard.model_validate(old).segments[0].type == "clip"
    assert ClipSegment.model_validate(old["segments"][0]).punch == []
    schema = json.dumps(storyboard_json_schema())
    for word in ("SplitSegment", "VideoOverlay", "StickerOverlay", "ProgressOverlay", "Punch"):
        assert word in schema


def test_split_validation_messages(tmp_path: Path) -> None:
    root = _project(tmp_path)
    with pytest.raises(ValueError, match="two non-empty"):
        _board([_split(labels=["only one"])])
    with pytest.raises(ValueError, match="at least 0.1"):
        _board(
            [
                _split(
                    panels=[
                        {"source": "footage/moving.mp4", "freeze": {"hold": 1.95}},
                        {"source": "footage/blue.mp4"},
                    ]
                )
            ]
        )
    bad = {
        "name": "bad",
        "format": FORMAT,
        "segments": [
            _split(
                panels=[
                    {"source": "footage/moving.mp4", "in": 5.0},
                    {"source": "footage/blue.mp4", "crop": {"x": 300, "y": 0, "w": 200, "h": 100}},
                ]
            )
        ],
    }
    report = validate_storyboard(bad, root=root)
    codes = {issue.code for issue in report.errors}
    assert {"source_duration", "panel_crop"} <= codes
    wide = _board(
        [
            {
                "type": "split",
                "duration": 1.0,
                "panels": [
                    {"source": "footage/blue.mp4", "crop": {"x": 0, "y": 0, "w": 100, "h": 200}},
                    {"source": "footage/red.mp4"},
                ],
                "labels": ["A very long label that cannot fit in a panel", "B"],
            }
        ]
    )
    warnings = {issue.code for issue in validate_storyboard(wide, root=root).warnings}
    assert {"digital_zoom", "text_overflow"} <= warnings
    clean = validate_storyboard(_board([_split(labels=["BEFORE", "AFTER"])]), root=root)
    assert clean.ok, clean.errors


# --------------------------------------------------------------------------------- inset


def _inset(**kw) -> dict:
    return {
        "type": "video",
        "source": "footage/green.mp4",
        "in": 0.0,
        "start": 0.5,
        "end": 1.5,
        "x": 40,
        "y": 100,
        "width": 120,
        "height": 80,
        **kw,
    }


def _green_boxes(video: Path) -> tuple[list[int], tuple[int, int, int, int]]:
    frames = _frames(video)
    counts = [int(_near(frame, GREEN).sum()) for frame in frames]
    on = [index for index, count in enumerate(counts) if count > 200]
    mask = _near(frames[on[len(on) // 2]], GREEN)
    ys, xs = np.nonzero(mask)
    return on, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def test_inset_position_and_timing_within_a_frame(tmp_path: Path) -> None:
    root = _project(tmp_path)
    board = _board(
        [{"type": "clip", "source": "footage/grey.mp4", "in": 0, "out": 3}], [_inset()]
    )
    report = compose(board, root=root)
    on, box = _green_boxes(report.path)
    assert abs(on[0] - 15) <= 1 and abs(on[-1] - 44) <= 1 and on == list(range(on[0], on[-1] + 1))
    assert all(abs(a - b) <= 1 for a, b in zip(box, (40, 100, 160, 180), strict=True))
    entry = next(item for item in report.overlays if item["type"] == "video")
    assert entry["bbox"] == [40, 100, 160, 180]


def test_inset_circle_border_and_open_ended_span(tmp_path: Path) -> None:
    root = _project(tmp_path)
    board = _board(
        [{"type": "clip", "source": "footage/grey.mp4", "in": 0, "out": 4}],
        [
            _inset(
                end=None,
                start=1.0,
                x=100,
                y=300,
                width=100,
                height=None,
                shape="circle",
                border={"width": 6, "color": "#FFFFFF"},
            )
        ],
    )
    report = compose(board, root=root)
    frames = _frames(report.path)
    on = [i for i, f in enumerate(frames) if _near(f, GREEN).sum() > 200]
    # the green clip lasts 3 s from `in`, but the ad ends at 4 s: shown from 1.0 s to 4.0 s
    assert abs(on[0] - 30) <= 1 and on[-1] >= 118
    frame = frames[60]
    assert _near(frame[350, 150], GREEN).all()  # the middle is the clip
    assert not _near(frame[302, 102], GREEN).all()  # a corner of the square is outside the circle
    assert _near(frame[301, 150], (255, 255, 255), 30).all()  # the border ring, white


def test_inset_rounded_corner_and_plain_rect(tmp_path: Path) -> None:
    root = _project(tmp_path)
    rounded = compose(
        _board(
            [{"type": "clip", "source": "footage/grey.mp4", "in": 0, "out": 2}],
            [_inset(shape="rounded", radius=30)],
        ),
        root=root,
    )
    frame = _frames(rounded.path)[25]
    assert not _near(frame[101, 41], GREEN).all()  # the corner is cut
    assert _near(frame[140, 100], GREEN).all()


def test_progress_bar_fills_across_the_ad(tmp_path: Path) -> None:
    root = _project(tmp_path)
    board = _board(
        [{"type": "clip", "source": "footage/grey.mp4", "in": 0, "out": 3}],
        [{"type": "progress", "position": "bottom", "height": 8, "color": "#FF0000"}],
    )
    report = compose(board, root=root)
    frames = _frames(report.path)

    def filled(frame: np.ndarray) -> int:
        row = frame[H - 4]
        return int((row[:, 0] > 180).sum() * ((row[:, 1] < 80).sum() > 0))

    assert filled(frames[0]) <= 2
    span = 3.0 - 1.0 / FPS
    for index in (22, 45, 67):
        expected = W * (index / FPS) / span
        assert abs(filled(frames[index]) - expected) <= 3, index
    assert filled(frames[-1]) >= W - 2
    assert _near(frames[45][H - 12, 10], (128, 128, 128), 20).all()  # nothing above the bar
    top = compose(
        _board(
            [{"type": "clip", "source": "footage/grey.mp4", "in": 0, "out": 2}],
            [
                {
                    "type": "progress",
                    "position": "top",
                    "height": 6,
                    "color": "#FF0000",
                    "background": "#0000FF",
                    "start": 0.5,
                    "end": 1.5,
                }
            ],
        ),
        root=root,
    )
    frames = _frames(top.path)
    assert _near(frames[10][2, 300], (128, 128, 128), 20).all()  # before start: nothing
    assert _near(frames[20][2, 300], (0, 0, 255)).all()  # the track
    assert _near(frames[20][2, 5], (255, 0, 0)).all()  # the fill


# ------------------------------------------------------------------------------ stickers


def _sticker(**kw) -> StickerOverlay:
    return StickerOverlay.model_validate({"type": "sticker", **kw})


def test_draw_states_grow_and_cover_the_span() -> None:
    overlay = _sticker(kind="underline", x=40, y=300, width=200, animation="draw")
    states = plan_sticker(overlay, 1.0, 3.0, FPS)
    assert states[0].start == 1.0 and states[-1].end == 3.0
    assert all(a.end == b.start for a, b in zip(states, states[1:], strict=False))
    assert states[-1].progress == 1.0 and states[0].progress < 0.2
    # drawn over 300 ms: complete on the 9th frame
    complete = next(state for state in states if state.progress == 1.0)
    assert abs(complete.start - (1.0 + 0.3 - 1 / FPS)) < 1e-6
    widths = [
        sticker_art(overlay, progress=value).image.width for value in draw_progress(FPS)
    ]
    assert widths == sorted(widths) and widths[0] < widths[-1] * 0.4
    arrow = _sticker(kind="arrow", **{"from": [40, 300], "to": [240, 300]}, animation="draw")
    arrow_widths = [sticker_art(arrow, progress=v).image.width for v in draw_progress(FPS)]
    assert arrow_widths == sorted(arrow_widths) and arrow_widths[0] < arrow_widths[-1]


def test_pop_scales_overshoot_then_settle() -> None:
    scales = sticker_scales(FPS)
    assert scales[0] < 0.7 and max(scales) > 1.05 and abs(scales[-1] - 1.0) < 0.1
    overlay = _sticker(kind="badge", text="NEW", x=20, y=20, animation="pop")
    states = plan_sticker(overlay, 0.0, 2.0, FPS)
    assert states[0].scale < 0.7 and states[-1].scale == 1.0 and states[-1].end == 2.0


def test_sticker_geometry_of_every_kind(tmp_path: Path) -> None:
    arrow = sticker_art(_sticker(kind="arrow", x=200, y=300, direction="right", length=100))
    left, top, right, bottom = arrow.bbox
    assert 95 <= left <= 105 and 200 <= right <= 206 and top < 300 < bottom
    down = sticker_art(_sticker(kind="arrow", x=200, y=300, direction="down", length=100))
    assert down.bbox[3] >= 299 and down.bbox[1] <= 205
    circle = sticker_art(_sticker(kind="circle", x=180, y=320, width=100, height=60))
    assert 120 <= circle.bbox[0] <= 134 and 226 <= circle.bbox[2] <= 244
    badge = sticker_art(_sticker(kind="badge", text="-20%", x=30, y=40, size=30, color="#E5322D"))
    assert badge.image.getpixel((badge.image.width // 2, 6))[3] >= 0
    emoji = sticker_art(_sticker(kind="emoji", emoji="🔥", x=50, y=60, size=80))
    assert emoji.bbox == (50, 60, 130, 140)
    Image.new("RGBA", (40, 20), (255, 0, 0, 255)).save(tmp_path / "logo.png")
    image = sticker_art(
        _sticker(kind="image", source="logo.png", x=10, y=12, width=80), root=tmp_path
    )
    assert image.bbox == (10, 12, 90, 52)


def test_sticker_schema_errors() -> None:
    for payload, message in (
        ({"kind": "arrow"}, "needs from and to"),
        ({"kind": "circle", "x": 1, "y": 1}, "needs width"),
        ({"kind": "badge", "x": 1, "y": 1}, "needs text"),
        ({"kind": "badge", "x": 1, "y": 1, "text": "A", "animation": "draw"}, "animation draw"),
        ({"kind": "emoji", "x": 1, "y": 1}, "needs emoji"),
        ({"kind": "image", "x": 1, "y": 1}, "needs source"),
    ):
        with pytest.raises(ValueError, match=message):
            _sticker(**payload)


def test_stickers_render_in_the_final_video(tmp_path: Path) -> None:
    root = _project(tmp_path)
    board = _board(
        [{"type": "clip", "source": "footage/grey.mp4", "in": 0, "out": 3}],
        [
            {
                "type": "sticker",
                "kind": "arrow",
                "from": [40, 400],
                "to": [240, 400],
                "color": "#FF0000",
                "animation": "draw",
                "start": 0.5,
                "end": 2.5,
            },
            {
                "type": "sticker",
                "kind": "badge",
                "text": "NEW",
                "x": 20,
                "y": 30,
                "size": 28,
                "color": "#0000FF",
                "start": 1.0,
                "end": 2.0,
                "animation": "pop",
            },
        ],
    )
    report = compose(board, root=root)
    frames = _frames(report.path)
    red = [int(_near(f, (255, 0, 0), 60).sum()) for f in frames]
    assert red[14] == 0 and red[16] > 0  # appears at 0.5 s
    assert red[17] < red[25]  # drawn progressively
    assert abs(red[25] - red[70]) < red[25] * 0.03 and red[76] == 0  # complete, gone at 2.5 s
    blue = [int(_near(f, (0, 0, 255), 60).sum()) for f in frames]
    assert blue[29] == 0 and blue[31] > 0 and blue[61] == 0
    assert max(blue[30:40]) > blue[31]  # the pop overshoots then settles
    sticker = [item for item in report.overlays if item["type"] == "sticker"]
    assert len(sticker) == 2 and sticker[0]["states"][0]["start"] == pytest.approx(0.5)


def test_sticker_and_inset_validation(tmp_path: Path) -> None:
    root = _project(tmp_path)
    segments = [{"type": "clip", "source": "footage/grey.mp4", "in": 0, "out": 3}]
    board = _board(
        segments,
        [
            {"type": "sticker", "kind": "badge", "text": "OFF", "x": 340, "y": 600, "size": 40},
            _inset(x=300),
            _inset(y=-10, start=0, end=1),
        ],
    )
    report = validate_storyboard(board, root=root)
    codes = [issue.code for issue in report.errors]
    assert codes.count("inset_out_of_frame") == 2
    assert "sticker_out_of_frame" in codes
    ok = validate_storyboard(
        _board(
            segments,
            [
                {"type": "sticker", "kind": "circle", "x": 180, "y": 300, "width": 100},
                _inset(),
                {"type": "progress"},
            ],
        ),
        root=root,
    )
    assert ok.ok, ok.errors


def test_caption_hidden_by_an_inset_is_a_warning(tmp_path: Path) -> None:
    root = _project(tmp_path)
    board = _board(
        [{"type": "clip", "source": "footage/grey.mp4", "in": 0, "out": 3}],
        [
            {"type": "caption", "style": "pill", "lines": ["Hello"], "start": 0.0, "end": 2.0,
             "y": 110, "size": 30},
            _inset(x=80, y=100, width=200, height=100),
            {"type": "caption", "style": "pill", "lines": ["Elsewhere"], "start": 0.0,
             "end": 2.0, "y": 450, "size": 30},
        ],
    )  # fmt: skip
    warnings = [i for i in validate_storyboard(board, root=root).warnings
                if i.code == "caption_covered_by_inset"]  # fmt: skip
    assert len(warnings) == 1 and warnings[0].where == "overlays[0]"
    later = _board(
        [{"type": "clip", "source": "footage/grey.mp4", "in": 0, "out": 3}],
        [
            {"type": "caption", "style": "pill", "lines": ["Hello"], "start": 2.0, "end": 3.0,
             "y": 110, "size": 30},
            _inset(x=80, y=100, width=200, height=100),
        ],
    )  # fmt: skip
    assert not [i for i in validate_storyboard(later, root=root).warnings
                if i.code == "caption_covered_by_inset"]  # fmt: skip


# ---------------------------------------------------------------------------- inset audio


def test_inset_audio_rules(tmp_path: Path) -> None:
    root = _project(tmp_path)
    segments = [{"type": "clip", "source": "footage/grey.mp4", "in": 0, "out": 3}]
    voiced = _inset(audio=True)
    no_clips = validate_storyboard(_board(segments, [voiced]), root=root)
    assert [i.code for i in no_clips.errors] == ["inset_audio"]
    with_clips = validate_storyboard(
        _board(segments, [voiced], audio={"clips": True}), root=root
    )
    assert with_clips.ok
    assert "inset_audio_replaces" in {i.code for i in with_clips.warnings}
    two = validate_storyboard(
        _board(segments, [voiced, _inset(audio=True, start=1.0, end=2.0)], audio={"clips": True}),
        root=root,
    )
    assert [i.code for i in two.errors] == ["inset_audio"]
    silent = validate_storyboard(
        _board(segments, [_inset(audio=True, source="footage/red.mp4")], audio={"clips": True}),
        root=root,
    )
    assert "inset_audio_silent" in {i.code for i in silent.warnings}


def test_inset_voice_replaces_the_bed_under_it(tmp_path: Path) -> None:
    from adanvil.audio import AudioPiece

    root = _project(tmp_path)
    board = _board(
        [
            {"type": "clip", "source": "footage/grey.mp4", "in": 0, "out": 2},
            {"type": "image", "source": "footage/orange.png", "duration": 2},
        ],
        [_inset(audio=True, start=1.0, end=3.0, **{"in": 0.5})],
        audio={"clips": True},
    )
    grey = (root / "footage" / "grey.mp4").resolve()
    green = (root / "footage" / "green.mp4").resolve()
    base = [AudioPiece(grey, 0.0, 2.0, 2.0), AudioPiece(None, 0.0, 0.0, 2.0)]
    pieces = _with_inset_voices(board, root, base, [2.0, 2.0])
    assert [round(p.duration, 3) for p in pieces] == [1.0, 1.0, 1.0, 1.0]
    assert pieces[0].source == grey and pieces[0].start == 0.0 and pieces[0].length == 1.0
    assert pieces[1].source == green and pieces[1].start == pytest.approx(0.5)
    assert pieces[2].source == green and pieces[2].start == pytest.approx(1.5)
    assert pieces[3].source is None
    assert sum(p.duration for p in pieces) == 4.0
    untouched = _board(
        [{"type": "clip", "source": "footage/grey.mp4", "in": 0, "out": 2}],
        [_inset()],
        audio={"clips": True},
    )
    single = base[:1]
    assert _with_inset_voices(untouched, root, single, [2.0]) is single


def test_inset_audio_is_in_the_render(tmp_path: Path) -> None:
    root = _project(tmp_path)
    board = _board(
        [{"type": "image", "source": "footage/orange.png", "duration": 3}],
        [_inset(audio=True, start=1.0, end=2.0)],
        audio={"clips": True},
    )
    report = compose(board, root=root)
    levels = run(
        ["-i", str(report.path), "-vn", "-f", "f32le", "-ac", "1", "-ar", "8000", "-"],
        capture_stdout=True,
    ).stdout
    samples = np.frombuffer(levels or b"", dtype=np.float32)
    # ffmpeg 6.1 decodes the last AAC frame whole (a few ms past the 3 s the file declares),
    # newer versions cut it: only look at whole 0.1 s windows.
    samples = samples[: len(samples) // 800 * 800]
    loud = np.abs(samples.reshape(-1, 800)).mean(axis=1) > 0.01  # 0.1 s windows
    assert not loud[:8].any() and loud[12:18].all() and not loud[24:].any()


# --------------------------------------------------------------------------------- punch


def _rect_widths(video: Path) -> list[int]:
    widths = []
    for frame in _frames(video):
        row = frame[320]
        widths.append(int((row[:, 0] > 128).sum()))
    return widths


def test_punch_in_timing_and_scale(tmp_path: Path) -> None:
    root = _project(tmp_path)
    seg = ClipSegment.model_validate(
        {
            "type": "clip",
            "source": "footage/rect.mp4",
            "in": 0,
            "out": 3,
            "punch": [{"at": 1.0, "zoom": 1.5, "hold": 0.5}],
        }
    )
    path = render_segment(
        seg, root=root, cache=Cache(root / "cache"), fmt=Format(width=W, height=H, fps=FPS)
    )
    widths = _rect_widths(path)
    assert len(widths) == 90
    assert all(abs(w - 80) <= 2 for w in widths[:29])
    on = [i for i, w in enumerate(widths) if w > 110]
    assert on[0] in (29, 30, 31) and on[-1] in (43, 44, 45) and len(on) in (14, 15, 16)
    assert all(abs(widths[i] - 120) <= 3 for i in on)  # an instant 1.5x, no ramp
    assert all(abs(w - 80) <= 2 for w in widths[47:])


def test_punch_in_center_and_two_punches(tmp_path: Path) -> None:
    root = _project(tmp_path)
    seg = ClipSegment.model_validate(
        {
            "type": "clip",
            "source": "footage/rect.mp4",
            "in": 0,
            "out": 3,
            "punch": [
                {"at": 0.5, "zoom": 2.0, "hold": 0.3, "center": [100, 200]},
                {"at": 2.0, "zoom": 1.3, "hold": 0.4},
            ],
        }
    )
    path = render_segment(
        seg, root=root, cache=Cache(root / "cache"), fmt=Format(width=W, height=H, fps=FPS)
    )
    frames = _frames(path)
    # centred on (100, 200) at 2x: the rectangle (140..220, 270..370) goes to (260.., 460..)
    mask = frames[20][:, :, 0] > 128
    ys, xs = np.nonzero(mask)
    assert abs(int(xs.min()) - 260) <= 3 and abs(int(ys.min()) - 460) <= 3
    second = _rect_widths(path)[70]
    assert abs(second - 104) <= 3


def test_punch_validation_and_digital_zoom(tmp_path: Path) -> None:
    root = _project(tmp_path)
    with pytest.raises(ValueError, match="must not overlap"):
        ClipSegment.model_validate(
            {
                "type": "clip",
                "source": "a.mp4",
                "in": 0,
                "out": 3,
                "punch": [{"at": 1, "hold": 1}, {"at": 1.5, "hold": 0.2}],
            }
        )
    with pytest.raises(ValueError, match="ends after the segment"):
        ClipSegment.model_validate(
            {"type": "clip", "source": "a.mp4", "in": 0, "out": 1, "punch": [{"at": 0.8}]}
        )

    def codes(zoom: float) -> set[str]:
        board = _board(
            [
                {
                    "type": "clip",
                    "source": "footage/rect.mp4",
                    "in": 0,
                    "out": 2,
                    "punch": [{"at": 0.5, "zoom": zoom, "hold": 0.3}],
                }
            ]
        )
        return {i.code for i in validate_storyboard(board, root=root).warnings}

    assert "digital_zoom" not in codes(1.25) and "digital_zoom" in codes(1.4)
    with pytest.raises(ValueError, match="unexpected field"):
        Storyboard.model_validate(
            {
                "name": "x",
                "segments": [
                    {"type": "clip", "source": "a.mp4", "in": 0, "out": 2, "punch": [{"when": 1}]}
                ],
            }
        )


# ------------------------------------------------------------------------- qc and guides


def test_qc_crops_insets_and_split_frames(tmp_path: Path) -> None:
    root = _project(tmp_path)
    board = _board(
        [
            _split(labels=["A", "B"]),
            {"type": "clip", "source": "footage/grey.mp4", "in": 0, "out": 2},
        ],
        [
            {"type": "caption", "style": "pill", "lines": ["Hi"], "start": 0.0, "end": 1.0},
            _inset(start=2.5, end=3.5, y=200),
            {"type": "progress", "color": "#FF0000"},
        ],
    )  # fmt: skip
    render = compose(board, root=root)
    assert render.timeline[0]["type"] == "split" and "type" not in render.timeline[1]
    report = qc_report(
        render.path,
        render_report=render,
        expected_duration=render.duration,
        fmt=Format(width=W, height=H),
        out_dir=tmp_path / "qc",
        silent_ok=True,
    )
    names = {image.name for image in report.images}
    assert "split_0.jpg" in names and "overlay_1.jpg" in names
    assert not [c for c in report.checks if c.status == "fail"], report.checks
    with Image.open(tmp_path / "qc" / "split_0.jpg") as split_frame:
        assert split_frame.size[0] == 540 or split_frame.size == (W, H)


def test_the_guides_describe_the_visual_devices() -> None:
    repo = Path(__file__).resolve().parents[1]
    agents = (repo / "AGENTS.md").read_text(encoding="utf-8")
    skill = (repo / "skill" / "SKILL.md").read_text(encoding="utf-8")
    guide = (repo / "docs" / "EDITING_GUIDE.md").read_text(encoding="utf-8")
    for name, text in (("AGENTS.md", agents), ("SKILL.md", skill), ("guide", guide)):
        for word in (
            "Visual devices", "side_by_side", "gap_color", "label_style", "punch", "sticker",
            "progress", "inset_out_of_frame", "caption_covered_by_inset", "one or two",
        ):  # fmt: skip
            assert word in text, (name, word)
    assert agents.split("### Visual devices", 1)[1].split("###", 1)[0] == skill.split(
        "### Visual devices", 1
    )[1].split("###", 1)[0]
    assert (repo / "src" / "adanvil" / "assets" / "EDITING_GUIDE.md").read_bytes() == (
        repo / "docs" / "EDITING_GUIDE.md"
    ).read_bytes()
