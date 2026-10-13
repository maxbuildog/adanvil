"""A slow zoom or move of a still (or of a clip) advances by fractions of a pixel.

A scale to an even integer size followed by an integer crop moves the picture by whole-pixel
steps: it stands still for a few frames, then jumps (a visible tremble on a still). Every test
here draws a fine grid of soft vertical lines, renders it through the real code and tracks the
centre of one line frame by frame: it must follow the exact curve of the zoom within half a
pixel and never stand still or go backwards.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from adanvil._bin import ffmpeg_path
from adanvil.ffmpeg import run
from adanvil.media import (
    Cache,
    ClipSegment,
    Format,
    ImageSegment,
    RevealSegment,
    render_segment,
)
from adanvil.media.filters import freeze_zoom, zoom
from adanvil.render.cards import card_placements, plan_card
from adanvil.storyboard import ImageOverlay

W, H, FPS = 540, 960, 30
FMT = Format(width=W, height=H, fps=FPS)
LINES = tuple(range(20, W, 80))  # soft vertical lines every 80 px
TRACKED = 500  # the line far from the centre, where the picture moves the most
ZOOM_TO = 1.06


def _grid(path: Path, width: int = W, height: int = H) -> Path:
    """A fine grid: soft (sigma 2.5 px) bright vertical lines on a dark background."""

    x = np.arange(width)[None, :]
    value = np.zeros((1, width))
    for line in LINES:
        value = np.maximum(value, np.exp(-((x - line) ** 2) / (2 * 2.5**2)))
    gray = (30 + 200 * np.repeat(value, height, axis=0)).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.stack([gray] * 3, axis=-1)).save(path)
    return path


def _centroid(row: np.ndarray, near: float, half: int = 14) -> float:
    low, high = int(max(0, near - half)), int(min(len(row), near + half + 1))
    part = row[low:high] - row[low:high].min()
    part = np.where(part > part.max() * 0.15, part, 0.0)
    return float((part * np.arange(low, high)).sum() / part.sum())


def _positions(video: Path, expected, width: int = W) -> tuple[np.ndarray, np.ndarray]:
    """The tracked line of every frame, and the line where the curve says it is."""

    raw = run(
        ["-i", str(video), "-vf", f"crop={width}:16:0:100"]
        + ["-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_stdout=True,
    ).stdout
    rows = np.frombuffer(raw, np.uint8).reshape(-1, 16, width).astype(float).mean(axis=1)
    wanted = np.array([expected(n) for n in range(len(rows))])
    found = np.array([_centroid(row, wanted[n]) for n, row in enumerate(rows)])
    return found, wanted


def _where(z_of_n, line: int = TRACKED, fx: float = 0.5):
    """Where ``line`` is at frame ``n`` when the picture is enlarged by ``z_of_n(n)`` around
    the focus ``fx`` (the continuous position of its centre, in pixel-centre coordinates)."""

    def expected(n: int) -> float:
        z = z_of_n(n)
        left = W * (1 - 1 / z) * fx
        return (line + 0.5 - left) * z - 0.5

    return expected


def _assert_smooth(video: Path, z_of_n, frames: int, from_frame: int = 0) -> None:
    found, wanted = _positions(video, _where(z_of_n))
    assert len(found) >= frames
    found, wanted = found[from_frame:frames], wanted[from_frame:frames]
    error = found - wanted
    error -= error[0]  # the tracker has a constant bias of its own, the curve is what counts
    assert np.abs(error).max() <= 0.5, np.round(error, 2)
    steps = np.diff(found)
    # the line moves 0.2 to 0.5 px per frame: it never stands still, goes back, or jumps
    assert steps.min() > 0.05, np.round(steps, 2)
    assert steps.max() < 0.9, np.round(steps, 2)


def _project(tmp_path: Path) -> Path:
    _grid(tmp_path / "footage" / "grid.png")
    return tmp_path


def _small_clip(root: Path) -> None:
    """The grid as a clip filmed at half the size of the frame (a low-resolution source)."""

    subprocess.run(
        [ffmpeg_path(), "-nostdin", "-y", "-loglevel", "error", "-loop", "1", "-framerate",
         str(FPS), "-t", "3", "-i", str(root / "footage" / "grid.png"), "-vf",
         f"scale={W // 2}:{H // 2}", "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p",
         str(root / "footage" / "small.mp4")],
        check=True,
    )


def test_a_zoom_in_image_moves_smoothly(tmp_path: Path) -> None:
    root = _project(tmp_path)
    seg = ImageSegment(source="footage/grid.png", duration=1.0, motion="zoom_in", zoom_to=ZOOM_TO)
    video = render_segment(seg, root=root, cache=Cache(root / "cache"), fmt=FMT)
    _assert_smooth(video, lambda n: 1 + (ZOOM_TO - 1) * min(n / FPS, 1), FPS)


def test_a_reveal_with_a_zoom_moves_smoothly(tmp_path: Path) -> None:
    root = _project(tmp_path)
    side = {"source": "footage/grid.png", "fit": "cover"}
    seg = RevealSegment.model_validate(
        {
            "type": "reveal",
            "duration": 1.0,
            "before": side,
            "after": side,
            "divider": None,
            "motion": "zoom_in",
            "zoom_to": ZOOM_TO,
            "hold_before": 0.1,
            "sweep": 0.5,
        }
    )
    video = render_segment(seg, root=root, cache=Cache(root / "cache"), fmt=FMT)
    # both sides carry the same grid: the picture is the same wherever the divider is
    _assert_smooth(video, lambda n: 1 + (ZOOM_TO - 1) * min(n / FPS, 1), FPS)


def test_a_clip_zoom_on_a_low_resolution_source_moves_smoothly(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _small_clip(root)
    seg = ClipSegment.model_validate(
        {"type": "clip", "source": "footage/small.mp4", "in": 0, "out": 1,
         "zoom": {"from": 1.0, "to": ZOOM_TO}}
    )
    video = render_segment(seg, root=root, cache=Cache(root / "cache"), fmt=FMT)
    _assert_smooth(video, lambda n: 1 + (ZOOM_TO - 1) * min(n / FPS, 1), FPS)


def test_a_freeze_zoom_is_smooth_keeps_its_frames_and_never_shows_a_border(
    tmp_path: Path,
) -> None:
    root = tmp_path
    (root / "footage").mkdir()
    subprocess.run(
        [ffmpeg_path(), "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         f"color=c=red:s={W}x{H}:r={FPS}:d=1", "-pix_fmt", "yuv420p",
         str(root / "footage" / "red.mp4")],
        check=True,
    )
    for centre in ([W - 10, H - 10], [10, 10]):
        seg = ClipSegment.model_validate(
            {"type": "clip", "source": "footage/red.mp4", "in": 0, "out": 1,
             "freeze": {"hold": 1.0, "zoom": 1.5, "center": centre}}
        )
        video = render_segment(seg, root=root, cache=Cache(root / "cache"), fmt=FMT)
        raw = run(
            ["-i", str(video), "-f", "rawvideo", "-pix_fmt", "gray", "-vf", "scale=54:96", "-"],
            capture_stdout=True,
        ).stdout
        frames = np.frombuffer(raw, np.uint8).reshape(-1, 96, 54)
        assert len(frames) == 2 * FPS
        # a red picture is the same grey everywhere: a black rim would show on the edges
        assert frames.min() >= frames.max() - 6


def test_the_freeze_zoom_of_a_grid_is_smooth(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _small_clip(root)
    seg = ClipSegment.model_validate(
        {"type": "clip", "source": "footage/small.mp4", "in": 0, "out": 1,
         "freeze": {"hold": 1.0, "zoom": 1.3, "center": [W // 2, H // 2]}}
    )
    video = render_segment(seg, root=root, cache=Cache(root / "cache"), fmt=FMT)
    ramp = 0.45 * FPS

    def z(n: int) -> float:
        return 1 + 0.3 * min(max((n - FPS) / ramp, 0), 1)

    found, wanted = _positions(video, _where(z, line=420))  # 420 stays on screen at 1.3
    window = slice(FPS, FPS + int(ramp))  # the ramp itself
    error = found[window] - wanted[window]
    assert np.abs(error - error[0]).max() <= 0.5
    steps = np.diff(found[window])
    # a linear ramp moves the same distance every frame (3.3 px here), to a fraction of a pixel
    assert steps.max() - steps.min() < 0.5, np.round(steps, 2)


def test_the_zoom_filters_use_the_frame_number_and_the_float_window() -> None:
    for expression in (zoom(1.0, 1.1, 2.0), freeze_zoom(0.75, 1.0, 1.35, 540, 960)):
        assert "perspective=" in expression and "interpolation=cubic" in expression
        assert "eval=frame" in expression
        assert "in/" in expression or "(in-" in expression
        assert "scale=" not in expression and "crop=" not in expression


# ------------------------------------------------------------------------------ the cards


def _card_overlay(root: Path, animation: str) -> ImageOverlay:
    """A 400 x 100 picture with one soft vertical line, as a card with no frame."""

    x = np.arange(400)[None, :]
    line = np.exp(-((x - 300) ** 2) / (2 * 2.5**2))
    gray = (40 + 200 * np.repeat(line, 100, axis=0)).astype(np.uint8)
    (root / "footage").mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.stack([gray] * 3, axis=-1)).save(root / "footage" / "card.png")
    return ImageOverlay.model_validate(
        {"type": "image", "source": "footage/card.png", "start": 0.5, "end": 3.0, "x": 60,
         "y": 300, "width": 400, "animation": animation}
    )


@pytest.mark.parametrize("animation", ["slide_left", "slide_right", "pop"])
def test_a_card_entry_is_where_the_animation_says_to_the_subpixel(
    tmp_path: Path, animation: str
) -> None:
    overlay = _card_overlay(tmp_path, animation)
    placements, entry = card_placements(
        overlay, tmp_path / "footage" / "card.png", 0.5, 3.0,
        canvas=(W, H), fps=FPS, cache=Cache(tmp_path / "cache"),
    )
    plan = plan_card(overlay, 0.5, 3.0, FPS, (W, H))
    assert len(entry["states"]) == len(placements) == len(plan) > 6
    # the card is 400 wide at x=60, so its centre is at 260 and its line at 100.5 right of it
    positions, wanted = [], []
    for state, planned in zip(entry["states"], plan, strict=True):
        with Image.open(state["png"]) as picture:
            rgba = np.asarray(picture.convert("RGBA")).astype(float)
        middle = rgba[rgba.shape[0] // 2, :, :3].mean(axis=-1)
        expected = 260 + planned.dx + 100.5 * planned.scale - 0.5
        local = expected - state["offset"][0]
        found = _centroid(middle, local, half=8)
        positions.append(state["offset"][0] + found)
        wanted.append(expected)
    error = np.array(positions) - np.array(wanted)
    assert np.abs(error).max() <= 0.5, np.round(error, 2)
    if animation != "pop":  # a slide goes one way only, one step per frame
        steps = np.diff(positions[:8])
        assert (steps < 0).all() if animation == "slide_left" else (steps > 0).all()


def test_the_card_plan_has_float_offsets_that_end_at_rest() -> None:
    overlay = ImageOverlay.model_validate(
        {"type": "image", "source": "x.png", "x": 0, "y": 0, "animation": "slide_up"}
    )
    states = plan_card(overlay, 0.0, 2.0, FPS, (W, H))
    offsets = [state.dy for state in states[:8]]
    assert offsets == sorted(offsets, reverse=True) and offsets[-1] == 0
    assert any(not float(value).is_integer() for value in offsets)
    assert math.isclose(states[0].dx, 0.0)
