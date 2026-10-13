"""V2.4c: the reveal segment (a divider sweeps a before picture away to show the after picture),
animated picture cards (image overlay with animation, frame and rotate) and the validation of
before/after pairs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from adanvil.media import (
    Cache,
    ClipSegment,
    Format,
    ImageSegment,
    RevealSegment,
    render_segment,
    segment_duration,
)
from adanvil.media.reveal import (
    boundary_expression,
    divider_margin,
    progress_expression,
    reveal_graph,
    reveal_legs,
    reveal_position,
    reveal_progress,
)
from adanvil.media.segment import _image_key
from adanvil.qc import qc_report
from adanvil.render import compose
from adanvil.render.cards import plan_card
from adanvil.storyboard import (
    ImageOverlay,
    Storyboard,
    storyboard_json_schema,
    validate_storyboard,
)
from tests.test_visual_grammar import (
    FORMAT,
    FPS,
    H,
    W,
    _board,
    _frames,
    _near,
    _project,
    _solid,
)

FMT = Format(width=W, height=H, fps=FPS)
RED, BLUE = (255, 0, 0), (0, 0, 255)


def _reveal(**kw) -> dict:
    return {
        "type": "reveal",
        "duration": 3.0,
        "before": {"source": "footage/red.mp4"},
        "after": {"source": "footage/blue.mp4"},
        **kw,
    }


def _render(root: Path, raw: dict) -> Path:
    seg = RevealSegment.model_validate(raw)
    return render_segment(seg, root=root, cache=Cache(root / "cache"), fmt=FMT)


def _white(frame: np.ndarray) -> np.ndarray:
    return (frame > 235).all(axis=-1)


def _divider_centre(frame: np.ndarray, direction: str) -> float | None:
    """Where the (white) divider is in a frame, to the pixel, or None when it is not on screen.
    It is read in the middle band of the frame, away from the labels at the top."""

    mask = (frame > 225).all(axis=-1)
    if direction == "top_to_bottom":
        rows = np.nonzero(mask[:, W // 4 : 3 * W // 4].mean(axis=1) > 0.9)[0]
        return None if rows.size == 0 else float(rows.mean() + 0.5)
    columns = np.nonzero(mask[H // 3 : 2 * H // 3, :].mean(axis=0) > 0.9)[0]
    return None if columns.size == 0 else float(columns.mean() + 0.5)


# ------------------------------------------------------------------ the formula of the sweep


def test_progress_is_zero_while_held_one_when_done_and_eased() -> None:
    assert reveal_progress(0.0, 0.8, 0.9, False) == 0.0
    assert reveal_progress(0.8, 0.8, 0.9, False) == 0.0
    assert reveal_progress(1.25, 0.8, 0.9, False) == pytest.approx(0.5)
    assert reveal_progress(1.7, 0.8, 0.9, False) == 1.0
    assert reveal_progress(9.0, 0.8, 0.9, False) == 1.0
    quarter = reveal_progress(0.8 + 0.225, 0.8, 0.9, False)
    assert quarter < 0.25  # ease in: slower than linear at the start
    values = [reveal_progress(0.8 + 0.9 * k / 30, 0.8, 0.9, False) for k in range(31)]
    assert values == sorted(values)


def test_back_and_forth_goes_across_back_to_the_middle_and_across_again() -> None:
    legs = reveal_legs(0.5, 2.0, True)
    assert [(a, b) for _s, _e, a, b in legs] == [(0.0, 1.0), (1.0, 0.5), (0.5, 1.0)]
    assert legs[0][0] == 0.5 and legs[-1][1] == pytest.approx(2.5)
    assert reveal_progress(0.5 + 0.8, 0.5, 2.0, True) == pytest.approx(1.0)
    assert reveal_progress(0.5 + 1.4, 0.5, 2.0, True) == pytest.approx(0.5)
    assert reveal_progress(2.5, 0.5, 2.0, True) == 1.0
    series = [reveal_progress(0.5 + 2.0 * k / 60, 0.5, 2.0, True) for k in range(61)]
    peak = series.index(max(series))
    assert series[peak] == pytest.approx(1.0, abs=1e-3) and min(series[peak:]) < 0.55
    assert series[-1] == 1.0


def _evaluate(expression: str, t: float) -> float:
    """The ffmpeg expression evaluated in Python (it only uses if, lt, clip, pow and round)."""

    names = {
        "_if": lambda condition, yes, no: yes if condition else no,
        "lt": lambda a, b: 1.0 if a < b else 0.0,
        "clip": lambda value, low, high: min(max(value, low), high),
        "round": lambda value: math.floor(value + 0.5),
        "t": t,
    }
    return eval(expression.replace("if(", "_if("), {"__builtins__": {}}, names)  # noqa: S307


@pytest.mark.parametrize("back_and_forth", [False, True])
def test_the_expression_is_the_formula(back_and_forth: bool) -> None:
    """What the filter graph evaluates per frame is exactly ``reveal_progress`` and
    ``reveal_position`` (so the pixel tests below test the formula, not a copy of it)."""

    expression = progress_expression(0.6, 1.7, back_and_forth)
    assert "clip((t-0.6)/" in expression
    for number in range(0, 130):
        t = number / 30
        assert _evaluate(expression, t) == pytest.approx(
            reveal_progress(t, 0.6, 1.7, back_and_forth), abs=1e-9
        )
    for direction in ("left_to_right", "right_to_left", "top_to_bottom"):
        seg = RevealSegment.model_validate(
            _reveal(duration=3.0, hold_before=0.6, sweep=1.7, direction=direction,
                    back_and_forth=back_and_forth)
        )  # fmt: skip
        boundary = boundary_expression(seg, W, H)
        for number in range(0, 90, 3):
            t = number / 30
            assert _evaluate(boundary, t) == reveal_position(t, seg, W, H)


# ---------------------------------------------------------------------------- the frames


@pytest.mark.parametrize("direction", ["left_to_right", "right_to_left", "top_to_bottom"])
def test_reveal_holds_the_before_sweeps_and_holds_the_after(tmp_path: Path, direction: str) -> None:
    root = _project(tmp_path)
    raw = _reveal(direction=direction)
    seg = RevealSegment.model_validate(raw)
    assert segment_duration(seg) == 3.0
    frames = _frames(_render(root, raw))
    assert len(frames) == 90
    # the before picture alone while held (no divider on screen)
    for n in (0, 10, 23):
        assert _near(frames[n], RED, 60).all()
        assert _divider_centre(frames[n], direction) is None
    # the after picture alone at the end, and until the end
    for n in (55, 70, 89):
        assert _near(frames[n], BLUE, 60).all()
        assert _divider_centre(frames[n], direction) is None
    # mid sweep: the divider where the formula puts it (two pixels), the after picture behind it
    for n in (28, 33, 38, 45):
        expected = reveal_position(n / FPS, seg, W, H)
        measured = _divider_centre(frames[n], direction)
        assert measured is not None and abs(measured - expected) <= 2, (n, measured, expected)
        frame = frames[n]
        if direction == "left_to_right":
            assert _near(frame[300, max(0, expected - 40)], BLUE, 60).all() or expected < 40
            assert _near(frame[300, min(W - 1, expected + 40)], RED, 60).all() or expected > W - 40
        elif direction == "right_to_left":
            assert _near(frame[300, min(W - 1, expected + 40)], BLUE, 60).all() or expected > W - 40
            assert _near(frame[300, max(0, expected - 40)], RED, 60).all() or expected < 40
        else:
            assert _near(frame[max(0, expected - 40), 180], BLUE, 60).all() or expected < 40
            assert _near(frame[min(H - 1, expected + 40), 180], RED, 60).all() or expected > H - 40


def test_the_divider_is_eased_not_linear(tmp_path: Path) -> None:
    root = _project(tmp_path)
    frames = _frames(_render(root, _reveal(hold_before=0.5, sweep=1.5)))
    centres = [
        (n, _divider_centre(frames[n], "left_to_right")) for n in range(0, len(frames))
    ]
    on_screen = [(n, c) for n, c in centres if c is not None]
    assert on_screen[0][1] < 60 and on_screen[-1][1] > W - 60
    speeds = [b[1] - a[1] for a, b in zip(on_screen, on_screen[1:], strict=False)]
    middle = max(speeds)
    assert speeds[0] < middle * 0.6 and speeds[-1] < middle * 0.6  # slow in, fast, slow out


@pytest.mark.parametrize("back_and_forth", [False, True])
def test_every_frame_of_the_sweep_is_where_the_formula_says(
    tmp_path: Path, back_and_forth: bool
) -> None:
    root = _project(tmp_path)
    raw = _reveal(duration=3.4, hold_before=0.4, sweep=2.4, back_and_forth=back_and_forth)
    seg = RevealSegment.model_validate(raw)
    frames = _frames(_render(root, raw))
    seen: list[float] = []
    for n, frame in enumerate(frames):
        expected = reveal_position(n / FPS, seg, W, H)
        measured = _divider_centre(frame, "left_to_right")
        if 4 <= expected <= W - 4 and measured is not None:  # the whole line is in the frame
            assert abs(measured - expected) <= 2, (n, measured, expected)
            seen.append(measured)
    assert len(seen) > 30
    steps = [(b > a) - (b < a) for a, b in zip(seen, seen[1:], strict=False) if b != a]
    runs = [step for index, step in enumerate(steps) if index == 0 or step != steps[index - 1]]
    if back_and_forth:
        assert runs == [1, -1, 1]  # across, back, across again
        top = next(i for i in range(len(seen) - 1) if seen[i + 1] < seen[i])
        assert seen[top] > W - 40 and 150 < min(seen[top:]) < 215  # back to about the middle
        assert seen[-1] > W - 40  # and across to the end
    else:
        assert runs == [1] and seen == sorted(seen)


def test_labels_follow_their_side(tmp_path: Path) -> None:
    root = _project(tmp_path)
    frames = _frames(_render(root, _reveal(labels=["OLD", "NEW"], label_style="label")))
    top = slice(20, 110)
    left, right = (slice(0, W // 2 - 20), slice(W // 2 + 20, W))

    def card(frame: np.ndarray, columns: slice) -> int:
        return int(_white(frame[top, columns]).sum())

    # held: the before label sits on the far end of the sweep (right), nothing on the left
    assert card(frames[10], right) > 300 and card(frames[10], left) == 0
    # swept: the after label has taken the left and the before label is gone
    assert card(frames[80], left) > 300 and card(frames[80], right) == 0
    # in between the before label goes when the divider passes it (right side, late)
    assert card(frames[34], right) > 300  # divider still left of the label at 0.72 W
    assert card(frames[48], right) == 0 or card(frames[48], right) < card(frames[34], right)
    plain = _frames(_render(root, _reveal()))
    assert card(plain[10], right) == 0 and card(plain[80], left) == 0


def test_a_vertical_sweep_swaps_the_label_in_place(tmp_path: Path) -> None:
    root = _project(tmp_path)
    frames = _frames(
        _render(root, _reveal(direction="top_to_bottom", labels=["OLD", "NEW"]))
    )
    top = (slice(20, 110), slice(W // 4, 3 * W // 4))
    assert _white(frames[10][top]).sum() > 300 and _white(frames[80][top]).sum() > 300
    # the two labels are not the same picture
    assert np.abs(frames[10][top].astype(int) - frames[80][top].astype(int)).mean() > 5


def test_clip_sides_play_in_sync_from_the_start_of_the_segment(tmp_path: Path) -> None:
    root = _project(tmp_path)
    raw = _reveal(
        duration=3.0,
        before={"source": "footage/moving_long.mp4", "in": 0.0},
        after={"source": "footage/moving_long.mp4", "in": 1.0},
        divider=None,
    )
    frames = _frames(_render(root, raw))
    plain = render_segment(
        ClipSegment(source="footage/moving_long.mp4", in_=1.0, out=4.0),
        root=root,
        cache=Cache(root / "cache"),
        fmt=FMT,
    )
    reference = _frames(plain)
    for n in (60, 75, 89):  # after the sweep: the after clip is where it would be on its own
        assert np.abs(frames[n].astype(int) - reference[n].astype(int)).mean() < 4
    # and it moved: it did not restart or freeze when the sweep ended
    assert np.abs(frames[60].astype(int) - frames[89].astype(int)).mean() > 4


def test_one_focus_for_both_images_lines_the_pictures_up(tmp_path: Path) -> None:
    root = _project(tmp_path)
    wide = Image.new("RGB", (900, 640), (255, 0, 0))
    wide.paste(Image.new("RGB", (450, 640), (0, 0, 255)), (450, 0))
    wide.save(root / "footage" / "halves.png")
    mid = 20  # inside the held before picture, divider off screen
    for focus, colour in (([0.0, 0.5], RED), ([1.0, 0.5], BLUE)):
        frames = _frames(
            _render(
                root,
                _reveal(
                    duration=2.0,
                    before={"source": "footage/halves.png", "fit": "cover", "focus": [0.5, 0.5]},
                    after={"source": "footage/halves.png", "fit": "cover"},
                    focus=focus,
                ),
            )
        )
        assert _near(frames[mid], colour, 30).all(), focus
    # the segment focus replaces the side focus, and without it the side's own is used
    seg = RevealSegment.model_validate(_reveal(focus=[0.2, 0.3]))
    assert seg.side_focus(seg.before) == (0.2, 0.3) == seg.side_focus(seg.after)
    own = RevealSegment.model_validate(_reveal(before={"source": "a.png", "focus": [0.1, 0.9]}))
    assert own.side_focus(own.before) == (0.1, 0.9)


def test_zoom_in_moves_both_sides_together(tmp_path: Path) -> None:
    root = _project(tmp_path)
    raw = _reveal(
        before={"source": "footage/rect.mp4"},
        after={"source": "footage/rect.mp4"},
        divider=None,
        motion="zoom_in",
        zoom_to=1.3,
    )
    frames = _frames(_render(root, raw))
    box = [int(_white(frame).sum()) for frame in frames]
    # the white box of the picture grows by 1.3 x 1.3 on both sides, before and after the sweep
    assert box[85] > box[2] * 1.45 and box[20] > box[2] * 1.1
    static = _frames(
        _render(
            root,
            _reveal(
                before={"source": "footage/rect.mp4"},
                after={"source": "footage/rect.mp4"},
                divider=None,
            ),
        )
    )
    assert int(_white(static[85]).sum()) == pytest.approx(int(_white(static[2]).sum()), rel=0.05)


def test_a_plain_wipe_has_no_divider_and_the_edge_is_one_pixel(tmp_path: Path) -> None:
    root = _project(tmp_path)
    raw = _reveal(divider=None)
    seg = RevealSegment.model_validate(raw)
    assert divider_margin(None) == 0 and seg.divider is None
    frames = _frames(_render(root, raw))
    n = 36
    expected = reveal_position(n / FPS, seg, W, H)
    row = frames[n][300]
    blue_columns = [x for x in range(W) if _near(row[x], BLUE, 60).all()]
    assert abs((max(blue_columns) + 1) - expected) <= 2
    assert not any(_white(row[None])[0])


def test_reveal_colour_tags_and_an_image_side_hold(tmp_path: Path) -> None:
    from adanvil.ffmpeg import probe

    root = _project(tmp_path)
    path = _render(
        root,
        _reveal(
            before={"source": "footage/orange.png"},
            after={"source": "footage/blue.mp4"},
            labels=["A", "B"],
            label_style="pill",
        ),
    )
    info = probe(path)
    assert (info.width, info.height, info.fps, info.pix_fmt) == (W, H, 30, "yuv420p")
    assert (info.color_space, info.color_primaries, info.color_transfer) == ("bt709",) * 3
    frames = _frames(path)
    assert len(frames) == 90
    assert _near(frames[5][300, 180], (200, 120, 40), 35).all()


def test_the_graph_has_no_label_inputs_without_labels() -> None:
    seg = RevealSegment.model_validate(_reveal(labels=["A", "B"]))
    with_labels = reveal_graph(seg, FMT, labels=True, divider=True)
    without = reveal_graph(seg, FMT, labels=False, divider=False)
    assert "[4:v]" in with_labels and "[4:v]" not in without and "[2:v]" not in without
    assert "alphamerge" in without


# ----------------------------------------------------------------------- schema and models


def test_reveal_model_validation() -> None:
    with pytest.raises(ValueError, match="hold_before"):
        RevealSegment.model_validate(_reveal(duration=1.5))  # 0.8 + 0.9 > 1.5
    with pytest.raises(ValueError, match="two non-empty"):
        RevealSegment.model_validate(_reveal(labels=["only one"]))
    with pytest.raises(ValueError, match="between 0 and 1"):
        RevealSegment.model_validate(_reveal(focus=[1.4, 0.5]))
    with pytest.raises(ValueError, match="invalid colour"):
        RevealSegment.model_validate(_reveal(divider={"color": "not a colour"}))
    with pytest.raises(ValueError, match="at least 0.1"):
        RevealSegment.model_validate(
            _reveal(before={"source": "footage/moving.mp4", "freeze": {"hold": 2.95}})
        )
    seg = RevealSegment.model_validate(_reveal(duration=2.5, role="reveal", label="before"))
    assert seg.direction == "left_to_right" and seg.hold_before == 0.8 and seg.sweep == 0.9
    assert seg.divider is not None and (seg.divider.width, seg.divider.color) == (6, "#FFFFFF")
    assert seg.divider.shadow is True and seg.motion == "static" and seg.labels is None
    assert seg.source == "footage/red.mp4"


def test_a_reveal_is_a_segment_of_the_storyboard() -> None:
    board = _board([_reveal(), {"type": "clip", "source": "footage/red.mp4", "in": 0, "out": 1}])
    assert board.segments[0].type == "reveal"
    assert board.timeline() == [(0.0, 3.0), (3.0, 4.0)]
    assert "RevealSegment" in json.dumps(storyboard_json_schema())
    bad = {"name": "x", "segments": [_reveal(colour="red")]}
    with pytest.raises(ValueError, match="unexpected field"):
        Storyboard.model_validate(bad)
    bad = {"name": "x", "segments": [_reveal(before={"source": "a.mp4", "oops": 1})]}
    with pytest.raises(ValueError, match="unexpected field"):
        Storyboard.model_validate(bad)
    bad = {"name": "x", "segments": [_reveal(divider={"width": 4, "oops": 1})]}
    with pytest.raises(ValueError, match="unexpected field"):
        Storyboard.model_validate(bad)


def test_old_storyboards_are_unchanged() -> None:
    old = {
        "name": "old",
        "segments": [
            {"type": "clip", "source": "a.mp4", "in": 0, "out": 1},
            {"type": "image", "source": "cta.png", "duration": 2},
            {"type": "split", "duration": 2, "panels": [{"source": "a.mp4"}, {"source": "b.mp4"}]},
        ],
        "overlays": [{"type": "image", "source": "badge.png", "x": 10, "y": 20, "width": 100}],
    }
    board = Storyboard.model_validate(old)
    assert [segment.type for segment in board.segments] == ["clip", "image", "split"]
    overlay = board.overlays[0]
    assert isinstance(overlay, ImageOverlay) and overlay.plain
    assert (overlay.animation, overlay.frame, overlay.rotate, overlay.exit) == (
        "none", None, 0.0, False,
    )  # fmt: skip
    # a centred image focus is not part of the cache key, so existing caches stay valid
    key = _image_key(Path(__file__), ImageSegment(source="cta.png", duration=2), FMT)
    assert "focus" not in key["segment"]
    moved = _image_key(
        Path(__file__), ImageSegment(source="cta.png", duration=2, fit="cover", focus=(0, 0)), FMT
    )
    assert moved["segment"]["focus"] == [0.0, 0.0]


# ------------------------------------------------------------------------------ warnings


def _codes(report) -> set[str]:
    return {issue.code for issue in report.warnings}


def test_split_same_source_warns_on_one_take_and_only_then(tmp_path: Path) -> None:
    root = _project(tmp_path)
    one_take = {
        "type": "split",
        "duration": 1.0,
        "panels": [{"source": "footage/moving_long.mp4"}, {"source": "footage/moving_long.mp4"}],
    }
    report = validate_storyboard(_board([one_take]), root=root)
    assert "split_same_source" in _codes(report)
    message = next(w.message for w in report.warnings if w.code == "split_same_source")
    assert "the difference will not be visible" in message and "real before and after" in message
    # the same file at moments that do not overlap shows two different things
    apart = {
        "type": "split",
        "duration": 1.0,
        "panels": [
            {"source": "footage/moving_long.mp4", "in": 0.0},
            {"source": "footage/moving_long.mp4", "in": 2.0},
        ],
    }
    assert "split_same_source" not in _codes(validate_storyboard(_board([apart]), root=root))
    different = {
        "type": "split",
        "duration": 1.0,
        "panels": [{"source": "footage/red.mp4"}, {"source": "footage/blue.mp4"}],
    }
    assert "split_same_source" not in _codes(validate_storyboard(_board([different]), root=root))
    # an image used twice is one take too, and the path may be written in either form
    twice = {
        "type": "split",
        "duration": 1.0,
        "panels": [{"source": "footage/orange.png"}, {"source": "./footage/orange.png"}],
    }
    assert "split_same_source" in _codes(validate_storyboard(_board([twice]), root=root))
    # an old split of two crops of one clip is only a warning: the storyboard still renders
    assert validate_storyboard(_board([one_take]), root=root).ok


def test_reveal_same_source_warns_too(tmp_path: Path) -> None:
    root = _project(tmp_path)
    same = _reveal(
        duration=3.0,
        before={"source": "footage/moving_long.mp4", "in": 0.0},
        after={"source": "footage/moving_long.mp4", "in": 0.5},
    )
    assert "split_same_source" in _codes(validate_storyboard(_board([same]), root=root))
    apart = _reveal(
        duration=3.0,
        before={"source": "footage/moving_long.mp4", "in": 0.0},
        after={"source": "footage/moving_long.mp4", "in": 3.5},
    )
    assert "split_same_source" not in _codes(validate_storyboard(_board([apart]), root=root))
    fine = validate_storyboard(_board([_reveal()]), root=root)
    assert fine.ok and not {"split_same_source", "reveal_orientation"} & _codes(fine)


def test_reveal_sides_of_different_orientation_warn(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _solid(root / "footage" / "wide.mp4", "green", 3.0, size="640x360")
    mixed = _reveal(after={"source": "footage/wide.mp4"})
    report = validate_storyboard(_board([mixed]), root=root)
    assert "reveal_orientation" in _codes(report)
    message = next(w.message for w in report.warnings if w.code == "reveal_orientation")
    assert "portrait" in message and "landscape" in message
    matching = validate_storyboard(_board([_reveal()]), root=root)
    assert "reveal_orientation" not in _codes(matching)


def test_reveal_source_errors_and_after_short_warning(tmp_path: Path) -> None:
    root = _project(tmp_path)
    bad = {
        "name": "bad",
        "format": FORMAT,
        "segments": [
            _reveal(
                before={"source": "footage/moving.mp4", "in": 5.0},
                after={
                    "source": "footage/blue.mp4",
                    "crop": {"x": 300, "y": 0, "w": 200, "h": 100},
                },
            )
        ],
    }
    errors = {issue.code for issue in validate_storyboard(bad, root=root).errors}
    assert {"source_duration", "panel_crop"} <= errors
    missing = {
        "name": "m",
        "format": FORMAT,
        "segments": [_reveal(after={"source": "footage/nope.mp4"})],
    }
    assert "missing_file" in {i.code for i in validate_storyboard(missing, root=root).errors}
    tight = _board([_reveal(duration=1.8)])  # 0.8 + 0.9 leaves 0.1 s of the after picture
    warnings = validate_storyboard(tight, root=root).warnings
    short = [w for w in warnings if w.code == "reveal_after_short"]
    assert short and "0.10s" in short[0].message
    assert "reveal_after_short" not in _codes(validate_storyboard(_board([_reveal()]), root=root))


def test_reveal_labels_are_checked_for_overflow_and_banned_words(tmp_path: Path) -> None:
    root = _project(tmp_path)
    long = _board(
        [_reveal(labels=["A very long label that cannot fit in the frame at all today", "B"])]
    )
    assert "text_overflow" in _codes(validate_storyboard(long, root=root))
    clean = validate_storyboard(_board([_reveal(labels=["OLD", "NEW"])]), root=root)
    assert clean.ok and "text_overflow" not in _codes(clean)
    from adanvil.brand import Brand, Claims

    brand = Brand(name="Sunbrew", claims=Claims(banned_words=["cure"]))
    banned = validate_storyboard(
        _board([_reveal(labels=["BEFORE", "THE CURE"])]), root=root, brand=brand
    )
    issue = next(issue for issue in banned.errors if issue.code == "banned_word")
    assert "cure" in issue.message and issue.where == "segments[0].labels[1]"


def test_a_reveal_is_composed_with_other_segments_and_qc_looks_at_it(tmp_path: Path) -> None:
    root = _project(tmp_path)
    board = _board(
        [
            {"type": "clip", "source": "footage/grey.mp4", "in": 0, "out": 1},
            _reveal(labels=["OLD", "NEW"], label="before"),
            {"type": "clip", "source": "footage/green.mp4", "in": 0, "out": 1},
        ]
    )
    report = compose(board, root=root)
    assert report.timeline[1]["type"] == "reveal" and report.timeline[1]["sweep"] == 0.9
    assert abs(report.duration - 5.0) < 0.1 and not report.warnings
    frames = _frames(report.path)
    assert len(frames) == 150
    middle = (slice(250, 600), slice(60, 300))  # away from the labels at the top
    assert _near(frames[45 + 10][middle], RED, 60).all()  # 0.33 s into the reveal: the before
    assert _near(frames[30 + 85][middle], BLUE, 60).all()  # near its end: the after
    qc = qc_report(
        report.path,
        render_report=report,
        expected_duration=report.duration,
        fmt=FMT,
        out_dir=tmp_path / "qc",
        silent_ok=True,
    )
    names = {image.name for image in qc.images}
    assert {"reveal_sweep_1.jpg", "reveal_after_1.jpg"} <= names
    assert not [check for check in qc.checks if check.status == "fail"], qc.checks


# --------------------------------------------------------------------- picture cards

ORANGE = (230, 120, 30)


def _card_picture(root: Path, colour=ORANGE, size=(200, 100)) -> str:
    Image.new("RGB", size, colour).save(root / "footage" / "card.png")
    return "footage/card.png"


def _card(root: Path, **kw) -> dict:
    return {
        "type": "image",
        "source": _card_picture(root),
        "start": 0.5,
        "end": 2.5,
        "x": 80,
        "y": 300,
        "width": 200,
        **kw,
    }


def _overlay(root: Path, **kw) -> ImageOverlay:
    return ImageOverlay.model_validate(_card(root, **kw))


def test_plan_card_has_one_state_per_frame_of_the_entry(tmp_path: Path) -> None:
    root = _project(tmp_path)
    states = plan_card(_overlay(root, animation="pop"), 0.5, 2.5, FPS, (W, H))
    assert len(states) == 9  # 8 frames of entry over 250 ms, then the settled card
    assert states[0].start == 0.5 and states[-1].end == 2.5
    for first, second in zip(states, states[1:], strict=False):
        assert first.end == pytest.approx(second.start)  # no gap and no overlap
    assert states[0].scale < 0.9 and states[0].alpha < 1.0
    assert max(state.scale for state in states) > 1.0  # the overshoot of a pop
    assert (states[-1].scale, states[-1].alpha, states[-1].dx, states[-1].dy) == (1.0, 1.0, 0, 0)
    assert states[7].end == pytest.approx(0.5 + 8 / FPS)
    assert (states[7].scale, states[7].alpha) == (1.0, 1.0)  # the entry ends settled


def test_plan_card_of_a_plain_overlay_and_of_each_animation(tmp_path: Path) -> None:
    root = _project(tmp_path)
    canvas = (W, H)
    plain = plan_card(_overlay(root), 0.5, 2.5, FPS, canvas)
    assert len(plain) == 1 and (plain[0].start, plain[0].end) == (0.5, 2.5)
    fade = plan_card(_overlay(root, animation="fade"), 0.5, 2.5, FPS, canvas)
    alphas = [state.alpha for state in fade[:8]]
    assert alphas == sorted(alphas) and 0 < alphas[0] < 0.6 and alphas[-1] == 1.0
    assert all(state.dx == state.dy == 0 and state.scale == 1.0 for state in fade)
    left = plan_card(_overlay(root, animation="slide_left"), 0.5, 2.5, FPS, canvas)
    shifts = [state.dx for state in left[:8]]
    assert left[0].dx > 0 and left[-1].dx == 0 and shifts == sorted(shifts, reverse=True)
    right = plan_card(_overlay(root, animation="slide_right"), 0.5, 2.5, FPS, canvas)
    assert right[0].dx < 0 and right[0].dx == -left[0].dx
    up = plan_card(_overlay(root, animation="slide_up"), 0.5, 2.5, FPS, canvas)
    assert up[0].dy > 0 and up[0].dx == 0 and up[-1].dy == 0


def test_plan_card_exit_mirrors_the_entry_and_short_overlays_are_cut(tmp_path: Path) -> None:
    root = _project(tmp_path)
    states = plan_card(_overlay(root, animation="slide_up", exit=True), 0.5, 2.5, FPS, (W, H))
    exits = [state for state in states if state.start >= 2.5 - 8 / FPS - 1e-9]
    assert len(exits) == 8 and exits[-1].end == pytest.approx(2.5)
    assert exits[0].alpha > exits[-1].alpha > 0 and exits[0].dy < exits[-1].dy
    for first, second in zip(states, states[1:], strict=False):
        assert first.end == pytest.approx(second.start)
    # an overlay shorter than the entry is cut, never longer than asked
    short = plan_card(_overlay(root, animation="pop"), 0.5, 0.6, FPS, (W, H))
    assert short[0].start == 0.5 and short[-1].end == pytest.approx(0.6)
    assert plan_card(_overlay(root, animation="pop"), 0.5, 0.5, FPS, (W, H)) == []


def test_image_overlay_validation() -> None:
    with pytest.raises(ValueError, match="exit needs an animation"):
        ImageOverlay.model_validate({"source": "a.png", "x": 0, "y": 0, "exit": True})
    with pytest.raises(ValueError):
        ImageOverlay.model_validate({"source": "a.png", "x": 0, "y": 0, "animation": "spin"})
    with pytest.raises(ValueError):
        ImageOverlay.model_validate({"source": "a.png", "x": 0, "y": 0, "rotate": 80})
    with pytest.raises(ValueError, match="invalid colour"):
        ImageOverlay.model_validate(
            {"source": "a.png", "x": 0, "y": 0, "frame": {"border_color": "nope"}}
        )
    with pytest.raises(ValueError):
        ImageOverlay.model_validate({"source": "a.png", "x": 0, "y": 0, "frame": {"glow": 3}})
    card = ImageOverlay.model_validate(
        {"source": "a.png", "x": 0, "y": 0, "frame": {"radius": 20}, "rotate": -4}
    )
    assert not card.plain and card.frame is not None and card.frame.border_width == 0
    assert card.frame.border_color == "#FFFFFF" and card.frame.shadow is False


def _overlay_run(root: Path, overlay: dict) -> tuple[np.ndarray, dict, np.ndarray]:
    """The frames of a grey clip with ``overlay`` on it, the report entry of the overlay, and
    the frames of the same clip without it (to tell the video from the card)."""

    segments = [{"type": "clip", "source": "footage/grey.mp4", "in": 0, "out": 3}]
    plain = _frames(compose(_board(segments), root=root).path)
    report = compose(_board(segments, [overlay]), root=root)
    entry = next(item for item in report.overlays if item["type"] == "image")
    return _frames(report.path), entry, plain


def _is_video(frames: np.ndarray, plain: np.ndarray, n: int, y: int, x: int, tol: int = 8) -> bool:
    """The pixel shows the video behind the card (it is what the clip alone shows)."""

    return bool(np.abs(frames[n][y, x].astype(int) - plain[n][y, x].astype(int)).max() <= tol)


def test_a_card_has_rounded_corners_a_border_and_a_shadow(tmp_path: Path) -> None:
    root = _project(tmp_path)
    frames, entry, plain = _overlay_run(
        root,
        _card(
            root,
            frame={"radius": 30, "border_width": 8, "border_color": "#FFFFFF", "shadow": True},
        ),
    )
    settled = frames[45]  # 1.5 s: the card is in
    assert _near(settled[350, 180], ORANGE, 30).all()  # the picture
    assert _near(settled[303, 180], (255, 255, 255), 25).all()  # the border, top edge
    assert _near(settled[350, 83], (255, 255, 255), 25).all()  # the border, left edge
    assert _is_video(frames, plain, 45, 301, 82)  # the rounded-off corner shows the video
    assert _near(settled[404, 180], (255, 255, 255), 25).all()  # the border, bottom edge
    below = settled[410:420, 180].astype(int)  # the shadow under the card (it ends at y 408)
    assert below.mean() < plain[45][410:420, 180].astype(int).mean() - 6
    assert _is_video(frames, plain, 45, 100, 100)  # far from the card: the video
    assert _is_video(frames, plain, 5, 350, 180) and _is_video(frames, plain, 5, 303, 180)
    assert _is_video(frames, plain, 80, 350, 180)  # gone after 2.5 s
    # the border is inside the footprint: 200 wide, 8 + 92 + 8 high (the picture keeps its shape)
    assert entry["bbox"] == [80, 300, 280, 408] and len(entry["states"]) == 1


def test_a_card_without_frame_keeps_its_square_corners(tmp_path: Path) -> None:
    root = _project(tmp_path)
    frames, _entry, plain = _overlay_run(root, _card(root, animation="fade"))
    assert _near(frames[60][350, 180], ORANGE, 25).all()
    assert _near(frames[60][302, 82], ORANGE, 25).all()  # square corner, no frame
    assert _is_video(frames, plain, 60, 298, 82)


def test_pop_comes_in_small_and_settles_to_the_full_card(tmp_path: Path) -> None:
    root = _project(tmp_path)
    frames, entry, plain = _overlay_run(root, _card(root, animation="pop"))

    def card_pixels(n: int) -> int:
        """Pixels the card changes (it is still translucent in the first states of a pop)."""

        changed = np.abs(frames[n].astype(int) - plain[n].astype(int)).max(axis=-1) > 20
        return int(changed[250:460, 40:320].sum())

    first, settled = card_pixels(15), card_pixels(45)
    assert 0.3 * settled < first < 0.85 * settled  # 0.5 s: the first state of a pop, still small
    assert settled == pytest.approx(200 * 100, rel=0.03)
    assert card_pixels(5) == 0  # nothing before the start
    assert len(entry["states"]) == 9 and entry["animation"] == "pop"
    assert entry["settled"][0] == pytest.approx(0.5 + 7 / FPS)  # the last frame of the entry


def test_fade_slide_and_exit_on_the_frame(tmp_path: Path) -> None:
    root = _project(tmp_path)
    frames, _entry, plain = _overlay_run(root, _card(root, animation="fade"))
    first = frames[15][350, 180].astype(int)  # the first state of the fade: part way to the card
    assert not _is_video(frames, plain, 15, 350, 180, 4) and not _near(first, ORANGE, 25)
    assert _near(frames[45][350, 180], ORANGE, 25).all()

    frames, entry, _plain = _overlay_run(root, _card(root, animation="slide_left"))
    columns = np.nonzero(_near(frames[15][350], ORANGE, 45))[0]
    settled = np.nonzero(_near(frames[45][350], ORANGE, 45))[0]
    # at its first state the card is far to the right of its place, and it ends in place
    assert columns.size == 0 or columns.min() > settled.min() + 60
    assert abs(settled.min() - 80) <= 2 and abs(settled.max() + 1 - 280) <= 2
    assert entry["states"][0]["offset"][0] > 80

    frames, _entry, _plain = _overlay_run(root, _card(root, animation="slide_up"))
    rows = np.nonzero(_near(frames[15][:, 180], ORANGE, 45))[0]
    assert rows.size == 0 or rows.min() > 300 + 40

    frames, _entry, plain = _overlay_run(root, _card(root, animation="fade", exit=True))
    assert _near(frames[45][350, 180], ORANGE, 25).all()
    assert not _is_video(frames, plain, 73, 350, 180, 4)  # fading out, still partly there
    assert _is_video(frames, plain, 80, 350, 180)  # gone after its end


def test_rotate_turns_the_card_around_its_centre(tmp_path: Path) -> None:
    root = _project(tmp_path)
    frames, entry, plain = _overlay_run(root, _card(root, rotate=20))
    x0, y0, x1, y1 = entry["bbox"]
    assert (x1 - x0) > 200 and (y1 - y0) > 100  # the rotated card needs a bigger box
    assert abs((x0 + x1) / 2 - 180) <= 1 and abs((y0 + y1) / 2 - 350) <= 1  # same centre
    assert _near(frames[45][350, 180], ORANGE, 25).all()
    assert _is_video(frames, plain, 45, 301, 82)  # a corner of the unrotated card is now empty
    # turned clockwise, its top-left corner rises above the unrotated card (y 300): the card
    # covers a pixel up there that the unrotated card would not
    assert _near(frames[45][275, 110], ORANGE, 25).all() and _is_video(frames, plain, 5, 275, 110)


def test_a_card_over_a_reveal_and_qc_do_not_fail_on_an_exit(tmp_path: Path) -> None:
    root = _project(tmp_path)
    board = _board(
        [_reveal(duration=3.0)],
        [_card(root, animation="slide_up", exit=True, start=0.3, end=2.6)],
    )
    report = compose(board, root=root)
    qc = qc_report(
        report.path,
        render_report=report,
        expected_duration=report.duration,
        fmt=FMT,
        out_dir=tmp_path / "qc",
        silent_ok=True,
    )
    visible = [check for check in qc.checks if check.id == "overlay_visible"]
    assert visible and all(check.status == "pass" for check in visible), visible


def test_a_plain_image_overlay_renders_as_before(tmp_path: Path) -> None:
    root = _project(tmp_path)
    frames, entry, plain = _overlay_run(root, _card(root))
    assert _near(frames[45][350, 180], ORANGE, 25).all()
    assert _is_video(frames, plain, 10, 350, 180) and _is_video(frames, plain, 80, 350, 180)
    assert "states" not in entry and entry["png"] is not None and "animation" not in entry


# ------------------------------------------------------------------------- the guides


def test_the_guides_describe_before_and_after() -> None:
    repo = Path(__file__).resolve().parents[1]
    agents = (repo / "AGENTS.md").read_text(encoding="utf-8")
    skill = (repo / "skill" / "SKILL.md").read_text(encoding="utf-8")
    guide = (repo / "docs" / "EDITING_GUIDE.md").read_text(encoding="utf-8")
    for name, text in (("AGENTS.md", agents), ("SKILL.md", skill), ("guide", guide)):
        for word in (
            "Before and after", "reveal", "hold_before", "sweep", "back_and_forth",
            "split_same_source", "reveal_orientation", "slide_left", "border_width",
            "rotate", "photo card", "unretouched",
        ):  # fmt: skip
            assert word in text, (name, word)
    section = "### Before and after"
    assert agents.split(section, 1)[1].split("###", 1)[0] == skill.split(section, 1)[1].split(
        "###", 1
    )[0]
    assert (repo / "src" / "adanvil" / "assets" / "EDITING_GUIDE.md").read_bytes() == (
        repo / "docs" / "EDITING_GUIDE.md"
    ).read_bytes()


def test_the_before_after_recipe_prefers_a_reveal() -> None:
    from adanvil.storyboard import get_recipe

    recipe = get_recipe("before_after")
    text = recipe.when_to_use
    assert "reveal" in text and text.index("reveal") < text.index("split")
    assert "unretouched" in text and "split_same_source" in text
    assert any("reveal" in item for item in recipe.checklist)
    slot = next(item for item in recipe.structure if item.slot == "before")
    assert "reveal" in slot.effects and "split" in slot.effects


def test_snap_never_shortens_a_reveal_below_its_sweep(tmp_path: Path) -> None:
    from adanvil.project import init_project
    from adanvil.storyboard.snap import snap_storyboard
    from tests.test_sound_design import _click_track

    root = tmp_path / "project"
    init_project(root, "snap")
    _click_track(root / "music" / "beat.wav", 60 / 0.45, seconds=30.0)  # a beat every 0.45 s
    raw = {
        "name": "snap",
        "emoji": "none",
        "segments": [
            {"type": "image", "source": "footage/cta.png", "duration": 0.9},
            _reveal(duration=1.5, hold_before=0.6, sweep=0.9),
            {"type": "image", "source": "footage/cta.png", "duration": 1.2},
        ],
        "audio": {"music": "music/beat.wav"},
    }
    result = snap_storyboard(root, raw)
    # the only beat near the second cut (2.4 s) is 2.25 s, which would leave 1.35 s for a
    # reveal that needs 0.6 + 0.9: the cut stays
    assert result["moves"] == []
    assert any("too short" in item["reason"] for item in result["kept"])
    snapped = Storyboard.model_validate(result["storyboard"])
    assert snapped.segments[1].duration == 1.5
