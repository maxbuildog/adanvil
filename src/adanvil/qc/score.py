"""The creative score of a rendered ad: 0 to 100 with sub-scores and findings to act on.

Technical QC (``qc_run``) says the file is correct. ``ad_score`` says whether the ad is any
good at the things that can be measured: a first second that holds attention, a rhythm that
does not sag, captions that can be read, a story that shows the problem before the product and
ends on a call to action, a layout that stays out of the app interface, and a sound that is
audible and clear. Every finding has a time, a severity (``high``, ``medium``, ``low``) and a
fix, and the findings are sorted by impact (the points they cost on the global score).

It works on any rendered ad. With the storyboard it uses the exact timings, words, roles and
overlays; without it, it falls back on what the pixels and the sound show and lists the checks
it had to skip (``skipped``). Every threshold is a guideline, not a rule: a finding is a
question, and the images (``safe_zones_preview.jpg``, the frame of each top finding) are there
to be looked at.

How the score is built
======================

Six areas, each a sub-score from 100 down (a finding takes its ``penalty`` points off), mixed
by weight over the areas that could be evaluated::

    hook 20 · pacing 15 · reading 20 · story 20 · safe_zones 10 · audio 15

Aim for ``TARGET`` (80) or more.

Hook (first ``HOOK_SECONDS`` = 1.5 s)
    - a cut, or motion (mean frame change at 10 fps of at least ``MOTION_MIN`` = 0.8 % of the
      range), in the first second;
    - a face or the product visible: from the segment annotations when there are some
      (``shot_type`` ``talking_head``, ``face_closeup``, ``reaction``, ``product_hero``,
      ``product_in_use``, ``unboxing``, ``before_after``; subject ``person_*`` or ``product``),
      else the skin heuristic of ``compare_pair`` (a heuristic: a product cannot be seen);
    - a written text within ``TEXT_WITHIN`` (0.5 s): from the captions of the storyboard,
      else a detector of white-on-black lettering on the frames (a heuristic);
    - audio from the first half second (no silence of 0.5 s or more at the very start).

Pacing
    Cuts per 10 s in each quarter, the longest shot, the shots over ``DIP_SHOT`` (4 s) in the
    middle half of a fast ad (shots of 2.5 s or less on average), a shot over ``LONG_SHOT``
    (6 s), a slow ad (under 2 cuts per 10 s over 10 s or more). Cuts come from the storyboard
    timeline when given, else from a scene detector.

Reading (storyboard needed)
    Written captions on screen at least ``0.25 s per word + 0.6 s`` (minimum 1.2 s), at most 7
    words per line, and the contrast of the caption text against the pixels around it on the
    real frame: a WCAG-like luminance ratio, warned under ``CONTRAST_WARN`` (3:1), high under 2:1.

Story (storyboard needed, except the end check)
    Product after the problem (``product_too_early``), product shown at least once, call to
    action wording written or spoken, end card or call to action in the last 4 s. Without a
    storyboard only the last-4-seconds check runs, on the frames.

Safe zones (see ``adanvil.safezones`` for the boxes of ``tiktok``, ``reels`` and ``stories``)
    A caption, comment, badge, emoji or image sticker, or picture overlay inside the top bar,
    the right button column or the bottom band. Without a storyboard, the text-like regions
    detected on sampled frames. The picture ``safe_zones_preview.jpg`` draws the zones of each
    platform over a mid-ad frame.

Audio
    Integrated loudness inside ``LOUDNESS_OK`` (-20 to -10 LUFS; outside -24 to -9 is high);
    voice at least ``VOICE_OVER_MUSIC_DB`` (4.5 dB) above the music (storyboard with a voice or
    ``audio.clips`` and a music bed: computed from the mix levels); no silence of
    ``SILENCE_MAX`` (0.7 s) or more.
"""

from __future__ import annotations

import math
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageColor, ImageFilter

from adanvil.audio import MUSIC_REFERENCE_LUFS, integrated_loudness, music_gain
from adanvil.brand import Brand, load_brand
from adanvil.ffmpeg import extract_frame, probe, run
from adanvil.music import music_licence
from adanvil.safezones import PLATFORMS, check_platforms, draw_preview, zones_table
from adanvil.storyboard import Storyboard
from adanvil.storyboard.craft import (
    CaptionEntry,
    Checks,
    Finding,
    reading_checks,
    safe_zone_checks,
    story_checks,
    voice_lines,
)

from .pair import find_face

TARGET = 80
HOOK_SECONDS = 1.5
TEXT_WITHIN = 0.5
MOTION_MIN = 0.008
DIP_SHOT = 4.0
LONG_SHOT = 6.0
FAST_AD_SHOT = 2.5
SLOW_CUTS_PER_10S = 2.0
CONTRAST_WARN = 3.0
CONTRAST_HIGH = 2.0
LOUDNESS_OK = (-20.0, -10.0)
LOUDNESS_LIMITS = (-24.0, -9.0)
VOICE_OVER_MUSIC_DB = 4.5  # music_volume 0.35 (validated by ear) sits about 5 dB under
VOICE_MASKED_DB = 3.0
SILENCE_MAX = 0.7
SCENE_THRESHOLD = 0.3
WEIGHTS = {
    "hook": 20.0,
    "pacing": 15.0,
    "reading": 20.0,
    "story": 20.0,
    "safe_zones": 10.0,
    "audio": 15.0,
}
THRESHOLDS: dict[str, Any] = {
    "target": TARGET,
    "weights": WEIGHTS,
    "hook_seconds": HOOK_SECONDS,
    "text_within_s": TEXT_WITHIN,
    "motion_min": MOTION_MIN,
    "fast_ad_mean_shot_s": FAST_AD_SHOT,
    "dip_shot_s": DIP_SHOT,
    "long_shot_s": LONG_SHOT,
    "slow_cuts_per_10s": SLOW_CUTS_PER_10S,
    "contrast_warn": CONTRAST_WARN,
    "loudness_ok_lufs": list(LOUDNESS_OK),
    "loudness_limits_lufs": list(LOUDNESS_LIMITS),
    "voice_over_music_db": VOICE_OVER_MUSIC_DB,
    "silence_max_s": SILENCE_MAX,
}
_PEOPLE_TYPES = ("talking_head", "face_closeup", "reaction")
_SUBJECT_TYPES = _PEOPLE_TYPES + ("product_hero", "product_in_use", "unboxing", "before_after")
_PTS_TIME = re.compile(r"pts_time:([-+]?(?:\d+(?:\.\d*)?|\.\d+))")
_SILENCE_START = re.compile(r"silence_start:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))")
_SILENCE_END = re.compile(r"silence_end:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))")
_ANALYSIS_WIDTH = 270


class ScoreError(ValueError):
    """The ad cannot be scored (unreadable video, invalid storyboard)."""


@dataclass
class ScoreReport:
    """The result of :func:`ad_score`."""

    score: int
    meets_target: bool
    sub_scores: dict[str, dict[str, Any]]
    findings: list[dict[str, Any]]
    passed: list[str]
    skipped: list[dict[str, str]]
    facts: dict[str, Any]
    safe_zones: dict[str, dict[str, list[int]]]
    thresholds: dict[str, Any]
    images: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "target": TARGET,
            "meets_target": self.meets_target,
            "sub_scores": self.sub_scores,
            "findings": self.findings,
            "passed": self.passed,
            "skipped": self.skipped,
            "facts": self.facts,
            "safe_zones": self.safe_zones,
            "thresholds": self.thresholds,
            "notes": self.notes,
        }


# ----------------------------------------------------------------------- the media reads


def _raw_frames(
    video: Path,
    *,
    fps: float,
    width: int,
    height: int,
    start: float = 0.0,
    duration: float | None = None,
) -> np.ndarray:
    """Frames of ``video`` as an ``(n, height, width, 3)`` uint8 array, ``fps`` per second."""

    args: list[str] = []
    if start > 0:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", str(video)]
    if duration is not None:
        args += ["-t", f"{duration:.3f}"]
    args += [
        "-vf",
        f"fps={fps:g},scale={width}:{height}",
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    raw = run(args, capture_stdout=True).stdout
    data = np.frombuffer(raw or b"", dtype=np.uint8)
    size = width * height * 3
    return data[: (len(data) // size) * size].reshape(-1, height, width, 3)


def _scene_cuts(video: Path) -> list[float]:
    stderr = str(
        run(
            [
                "-loglevel",
                "info",
                "-i",
                str(video),
                "-vf",
                f"scale={_ANALYSIS_WIDTH}:-2,select='gt(scene,{SCENE_THRESHOLD})',showinfo",
                "-an",
                "-f",
                "null",
                "-",
            ]
        ).stderr
        or ""
    )
    return sorted({round(float(m.group(1)), 3) for m in _PTS_TIME.finditer(stderr)})


def silence_intervals(
    video: Path, duration: float, *, noise_db: int = -45
) -> list[tuple[float, float]]:
    """Stretches of 0.3 s or more where the audio is under ``noise_db`` dB: ``(start, end)``."""

    stderr = str(
        run(
            [
                "-loglevel",
                "info",
                "-i",
                str(video),
                "-map",
                "0:a:0",
                "-af",
                f"silencedetect=n={noise_db}dB:d=0.3",
                "-vn",
                "-f",
                "null",
                "-",
            ]
        ).stderr
        or ""
    )
    starts = [float(m.group(1)) for m in _SILENCE_START.finditer(stderr)]
    ends = [float(m.group(1)) for m in _SILENCE_END.finditer(stderr)]
    result: list[tuple[float, float]] = []
    for index, start in enumerate(starts):
        end = ends[index] if index < len(ends) else duration
        result.append((max(0.0, start), min(duration, end)))
    return result


# -------------------------------------------------------------------- text on the frames


def detect_text(rgb: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Boxes (as shares of the frame: left, top, right, bottom, 0 to 1) of lettering found in
    ``rgb``: bands of rows where near-white pixels touch near-black ones over a good part of
    the width, which is what outlined, boxed and pill captions look like. A heuristic: footage
    with white against black edges can fool it, and a low-contrast text cannot be seen by it."""

    gray = (
        0.299 * rgb[..., 0].astype(np.float32)
        + 0.587 * rgb[..., 1].astype(np.float32)
        + 0.114 * rgb[..., 2].astype(np.float32)
    )
    height, width = gray.shape
    light = Image.fromarray(((gray >= 225) * 255).astype(np.uint8))
    dark = Image.fromarray(((gray <= 40) * 255).astype(np.uint8))
    touching = (np.asarray(light) > 0) & (np.asarray(dark.filter(ImageFilter.MaxFilter(5))) > 0)
    rows = touching.sum(axis=1)
    boxes: list[tuple[float, float, float, float]] = []
    row = 0
    minimum_row = max(4, int(width * 0.02))
    while row < height:
        if rows[row] < minimum_row:
            row += 1
            continue
        top = row
        gap = 0
        while row < height and (rows[row] >= minimum_row or gap < 2):
            gap = 0 if rows[row] >= minimum_row else gap + 1
            row += 1
        bottom = row - gap
        band = touching[top:bottom]
        columns = np.nonzero(band.any(axis=0))[0]
        band_height = bottom - top
        if (
            band_height >= 5
            and band_height <= height * 0.25
            and band.sum() >= 60
            and columns.size
            and (columns[-1] - columns[0]) >= width * 0.15
        ):
            boxes.append(
                (
                    float(columns[0]) / width,
                    top / height,
                    float(columns[-1] + 1) / width,
                    bottom / height,
                )
            )
    return boxes


# ----------------------------------------------------------------------------- contrast


def _linear(channel: np.ndarray) -> np.ndarray:
    value = channel.astype(np.float32) / 255.0
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """WCAG relative luminance (0 to 1) of an ``(..., 3)`` uint8 colour array."""

    return (
        0.2126 * _linear(rgb[..., 0])
        + 0.7152 * _linear(rgb[..., 1])
        + 0.0722 * _linear(rgb[..., 2])
    )


def contrast_ratio(first: float, second: float) -> float:
    """WCAG contrast ratio (1 to 21) of two relative luminances."""

    high, low = max(first, second), min(first, second)
    return (high + 0.05) / (low + 0.05)


def caption_contrast(
    frame: Image.Image,
    entry: CaptionEntry,
    *,
    root: Path,
    brand: Brand | None,
) -> dict[str, Any] | None:
    """Contrast of the text of ``entry`` against the real pixels just around its letters.

    The caption is drawn on its own to find where its letters (fill colour) are; the text
    colour is the style's ``fg``; what is "behind" is the ring of 2 to 3 pixels around the
    letters on the real ``frame``, so an outline or a box that the render drew counts, and a
    plain text on busy video is judged against the video. The median luminance of the ring
    is used, against the luminance of the fill. Returns ``None`` when there is nothing to
    measure (no letters found, a comment bubble)."""

    from adanvil.storyboard.validate import _family_for_render
    from adanvil.text import render
    from adanvil.text.emoji import NoEmojiProvider
    from adanvil.text.styles import _DEFAULTS

    caption = entry.caption
    applied = brand.apply_to_caption(caption) if brand is not None else caption
    spec = applied.to_text_spec()
    if spec.style not in _DEFAULTS or spec.style == "comment":
        return None
    if spec.family is not None:
        family = _family_for_render(spec.family, root)
        if family is not None:
            spec = spec.model_copy(update={"family": family})
    fg_name = spec.fg or str(_DEFAULTS[spec.style]["fg"])
    fg = np.array(ImageColor.getcolor(fg_name, "RGBA")[:3], dtype=np.int32)
    canvas = frame.size
    try:
        result = render(spec, emoji=NoEmojiProvider(), canvas=canvas)
    except Exception:
        return None
    overlay = np.asarray(result.image, dtype=np.int32)
    result.image.close()
    ink = (overlay[..., 3] > 200) & (np.abs(overlay[..., :3] - fg).sum(axis=-1) <= 60)
    if int(ink.sum()) < 40:
        return None
    ys, xs = np.nonzero(ink)
    margin = 16
    top, bottom = max(0, int(ys.min()) - margin), min(canvas[1], int(ys.max()) + margin + 1)
    left, right = max(0, int(xs.min()) - margin), min(canvas[0], int(xs.max()) + margin + 1)
    window = ink[top:bottom, left:right]
    mask = Image.fromarray((window * 255).astype(np.uint8))
    near = np.asarray(mask.filter(ImageFilter.MaxFilter(3))) > 0
    wide = np.asarray(mask.filter(ImageFilter.MaxFilter(7))) > 0
    ring = wide & ~near
    if int(ring.sum()) < 20:
        return None
    pixels = np.asarray(frame.convert("RGB"))[top:bottom, left:right]
    behind = float(np.median(luminance(pixels[ring])))
    text = float(luminance(fg.astype(np.uint8)))
    return {
        "ratio": round(contrast_ratio(text, behind), 2),
        "text_luminance": round(text, 3),
        "behind_luminance": round(behind, 3),
        "box": [left, top, right, bottom],
    }


# -------------------------------------------------------------------------------- areas


@dataclass
class _Context:
    root: Path
    video: Path
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    model: Storyboard | None
    brand: Brand | None
    starts: list[float]
    ends: list[float]
    out_dir: Path
    notes: list[str] = field(default_factory=list)
    cuts: list[float] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)


def _label(segment: Any) -> str:
    """The shot type or subject a segment is annotated with, or an empty string."""

    return str(getattr(segment, "shot_type", None) or getattr(segment, "subject", None) or "")


def _severity(penalty: float) -> str:
    return "high" if penalty >= 22 else "medium" if penalty >= 9 else "low"


def _finding(
    area: str,
    check: str,
    penalty: float,
    message: str,
    fix: str,
    *,
    time: float | None = None,
    end: float | None = None,
    detail: dict[str, Any] | None = None,
    severity: str | None = None,
) -> Finding:
    return Finding(
        id=check,
        area=area,
        severity=severity or _severity(penalty),
        message=message,
        fix=fix,
        time=time,
        end=end,
        penalty=penalty,
        detail=detail,
    )


def _hook(
    ctx: _Context,
    first: np.ndarray,
    silences: list[tuple[float, float]],
    entries: list[CaptionEntry],
) -> Checks:
    checks = Checks()
    # Motion or a cut in the first second.
    cut_early = [cut for cut in ctx.cuts if cut <= 1.0]
    gray = first.astype(np.float32).mean(axis=-1)
    inside = gray[: max(2, int(round(1.0 * 10)) + 1)]
    changes = np.abs(np.diff(inside, axis=0)).mean(axis=(1, 2)) / 255.0 if len(inside) > 1 else []
    motion = float(np.mean(changes)) if len(changes) else 0.0
    ctx.facts["hook"] = {
        "first_cut": ctx.cuts[0] if ctx.cuts else None,
        "motion_first_second": round(motion, 4),
    }
    if cut_early or motion >= MOTION_MIN:
        checks.ok("hook", "motion_or_cut")
    else:
        checks.add(
            _finding(
                "hook",
                "hook_static",
                28.0,
                f"the first second is static: no cut and little motion (frame change "
                f"{motion:.2%}, minimum {MOTION_MIN:.1%})"
                + (f"; the first cut is at {ctx.cuts[0]:.1f}s" if ctx.cuts else ""),
                "open on movement or cut within the first second: start the first clip on the "
                "action, shorten the opening shot, or add a punch-in",
                time=0.0,
                end=1.0,
            )
        )

    # A face or the product in the first 1.5 s.
    verdict: bool | None = None
    source = "pixels"
    if ctx.model is not None:
        shots = [
            segment
            for segment, start in zip(ctx.model.segments, ctx.starts, strict=True)
            if start < HOOK_SECONDS
        ]
        annotated = [s for s in shots if _label(s)]
        if any(
            getattr(s, "shot_type", None) in _SUBJECT_TYPES
            or str(getattr(s, "subject", None) or "").startswith("person")
            or getattr(s, "subject", None) == "product"
            for s in shots
        ):
            verdict, source = True, "storyboard"
        elif annotated:
            verdict, source = False, "storyboard"
    if verdict is None:
        index = min(len(first) - 1, 5)
        face = find_face(Image.fromarray(first[index])) if len(first) else None
        if face is None and len(first) > 10:
            face = find_face(Image.fromarray(first[10]))
        verdict = True if face else None
    if verdict:
        checks.ok("hook", "face_or_product")
    elif verdict is False:
        checks.add(
            _finding(
                "hook",
                "hook_no_subject",
                20.0,
                "the opening shots show neither a person nor the product (annotated "
                f"{', '.join(sorted({_label(s) for s in shots if _label(s)}))})",
                "open on a face (reaction, talking head) or on the product in use",
                time=0.0,
                end=HOOK_SECONDS,
            )
        )
    else:
        checks.add(
            _finding(
                "hook",
                "hook_no_subject",
                8.0,
                f"no face could be found in the first {HOOK_SECONDS:g}s (skin heuristic; a "
                "product cannot be detected, so ignore this when it shows the product)",
                "open on a face or on the product in use",
                time=0.0,
                end=HOOK_SECONDS,
                detail={"source": source, "confidence": "low"},
            )
        )

    # A text within 0.5 s.
    if ctx.model is not None:
        has_text = any(
            entry.start <= TEXT_WITHIN + 1e-6 and entry.end > entry.start for entry in entries
        )
        confidence = 22.0
    else:
        has_text = any(len(detect_text(frame)) for frame in first[: int(TEXT_WITHIN * 10) + 1])
        confidence = 12.0
    if has_text:
        checks.ok("hook", "text_within_half_second")
    else:
        checks.add(
            _finding(
                "hook",
                "hook_no_text",
                confidence,
                f"no written text on screen within {TEXT_WITHIN:g}s"
                + ("" if ctx.model is not None else " (detected on the frames: a heuristic)"),
                "start the first caption at 0 to 0.3 s so the hook is readable with the sound off",
                time=0.0,
                end=TEXT_WITHIN,
            )
        )

    # Audio at once.
    if not ctx.has_audio:
        checks.skip("hook", "audio_at_once", "the video has no audio stream (see audio)")
    else:
        late = next((end for start, end in silences if start <= 0.05 and end - start >= 0.5), None)
        if late is None:
            checks.ok("hook", "audio_at_once")
        else:
            checks.add(
                _finding(
                    "hook",
                    "hook_silent_start",
                    20.0,
                    f"the audio is silent until {late:.1f}s: the ad opens in silence",
                    "start the music or the voice at 0 s (audio.start, voice_offset 0)",
                    time=0.0,
                    end=late,
                )
            )
    return checks


def _shots(ctx: _Context) -> list[tuple[float, float]]:
    edges = [0.0, *[cut for cut in ctx.cuts if 0.05 < cut < ctx.duration - 0.05], ctx.duration]
    return [(a, b) for a, b in zip(edges, edges[1:], strict=False) if b > a]


def _pacing(ctx: _Context) -> Checks:
    checks = Checks()
    shots = _shots(ctx)
    duration = ctx.duration
    if duration <= 0 or not shots:
        checks.skip("pacing", "all", "the video has no duration")
        return checks
    cuts = [b for _a, b in shots[:-1]]
    quarter = duration / 4.0
    per_quarter = [
        round(sum(1 for cut in cuts if i * quarter <= cut < (i + 1) * quarter) / quarter * 10.0, 1)
        for i in range(4)
    ]
    longest = max(shots, key=lambda shot: shot[1] - shot[0])
    mean_shot = duration / len(shots)
    fast = mean_shot <= FAST_AD_SHOT
    ctx.facts["pacing"] = {
        "cuts": [round(cut, 2) for cut in cuts],
        "shots": len(shots),
        "cuts_per_10s_by_quarter": per_quarter,
        "mean_shot_s": round(mean_shot, 2),
        "longest_shot": {
            "start": round(longest[0], 2),
            "end": round(longest[1], 2),
            "duration": round(longest[1] - longest[0], 2),
        },
        "fast_ad": fast,
    }
    problems = 0
    if fast:
        for start, end in shots:
            midpoint = (start + end) / 2.0
            if end - start > DIP_SHOT and duration * 0.25 <= midpoint <= duration * 0.75:
                problems += 1
                checks.add(
                    _finding(
                        "pacing",
                        "pacing_dip",
                        18.0,
                        f"a {end - start:.1f}s shot at {start:.1f}s to {end:.1f}s sits in the "
                        f"middle of a fast ad (shots average {mean_shot:.1f}s): the rhythm dips "
                        "where viewers leave",
                        f"cut it in two or shorten it under {DIP_SHOT:g}s, or add an insert shot",
                        time=start,
                        end=end,
                    )
                )
                break
    if longest[1] - longest[0] > LONG_SHOT and not problems:
        problems += 1
        checks.add(
            _finding(
                "pacing",
                "long_shot",
                8.0,
                f"the longest shot lasts {longest[1] - longest[0]:.1f}s "
                f"({longest[0]:.1f}s to {longest[1]:.1f}s)",
                f"cut away within {LONG_SHOT:g}s (a reaction, an insert, a punch-in)",
                time=longest[0],
                end=longest[1],
            )
        )
    overall = len(cuts) / duration * 10.0
    if duration >= 10.0 and overall < SLOW_CUTS_PER_10S:
        problems += 1
        checks.add(
            _finding(
                "pacing",
                "pacing_slow",
                15.0,
                f"a slow ad: {len(cuts)} cut(s) in {duration:.0f}s "
                f"({overall:.1f} per 10 s, under {SLOW_CUTS_PER_10S:g})",
                "add cuts: shorter shots and inserts keep attention on a paid feed",
                time=0.0,
                end=duration,
            )
        )
    if fast and not problems:
        middle = (per_quarter[1] + per_quarter[2]) / 2.0
        outer = max(per_quarter[0], per_quarter[3])
        if outer > 0 and middle < 0.5 * outer:
            problems += 1
            checks.add(
                _finding(
                    "pacing",
                    "pacing_sag",
                    6.0,
                    f"the cut rate sags in the middle: {per_quarter} cuts per 10 s by quarter",
                    "even the rhythm out: add a cut or an insert in the middle half",
                    time=duration * 0.25,
                    end=duration * 0.75,
                )
            )
    if not problems:
        checks.ok("pacing", "rhythm")
    return checks


def _reading(ctx: _Context, entries: list[CaptionEntry]) -> Checks:
    checks = Checks()
    if ctx.model is None:
        checks.skip("reading", "all", "needs the storyboard (words, timings, text styles)")
        return checks
    checks.extend(reading_checks(entries))
    if not entries:
        checks.skip("reading", "contrast", "no caption")
        return checks
    written = [entry for entry in entries if entry.kind == "written"]
    others = [entry for entry in entries if entry.kind != "written"]
    step = max(1, math.ceil(len(others) / 8)) if others else 1
    sample = (written + others[::step])[:16]
    worst: list[tuple[float, CaptionEntry, dict[str, Any]]] = []
    measured = 0
    for entry in sample:
        moment = min(max(0.0, (entry.start + entry.end) / 2.0), max(0.0, ctx.duration - 0.05))
        with tempfile.TemporaryDirectory(prefix="adanvil-score-", dir=ctx.out_dir) as temporary:
            path = Path(temporary) / "frame.png"
            extract_frame(ctx.video, moment, path)
            with Image.open(path) as opened:
                frame = opened.convert("RGB")
        frame = frame.resize(
            (ctx.model.format.width, ctx.model.format.height), Image.Resampling.LANCZOS
        )
        result = caption_contrast(frame, entry, root=ctx.root, brand=ctx.brand)
        frame.close()
        if result is None:
            continue
        measured += 1
        if result["ratio"] < CONTRAST_WARN:
            worst.append((result["ratio"], entry, {**result, "t": round(moment, 2)}))
    if not measured:
        checks.skip("reading", "contrast", "no caption letters could be measured")
    elif not worst:
        checks.ok("reading", "contrast")
    else:
        worst.sort(key=lambda item: item[0])
        for ratio, entry, result in worst[:4]:
            penalty = 12.0 if ratio < CONTRAST_HIGH else 7.0
            checks.add(
                _finding(
                    "reading",
                    "reading_contrast",
                    penalty,
                    f"caption {entry.key} ({entry.text[:40]!r}) has a contrast of {ratio:.1f}:1 "
                    f"against the pixels around it at {result['t']:.1f}s (warned under "
                    f"{CONTRAST_WARN:g}:1)",
                    "use the outline style with a dark stroke, a box or pill style, or a "
                    "text colour that opposes the shot",
                    time=result["t"],
                    end=entry.end,
                    detail={"overlay": entry.key, "ratio": ratio},
                    severity="high" if ratio < CONTRAST_HIGH else "medium",
                )
            )
    return checks


def _story(ctx: _Context, entries: list[CaptionEntry], last_frames: np.ndarray) -> Checks:
    if ctx.model is not None:
        return story_checks(
            ctx.model,
            ctx.brand,
            entries,
            voice_lines(ctx.model, ctx.root),
            ctx.starts,
            ctx.ends,
        )
    checks = Checks()
    checks.skip("story", "problem_and_product", "needs the storyboard (roles and annotations)")
    checks.skip("story", "cta_wording", "needs the storyboard (captions and voice)")
    if any(len(detect_text(frame)) for frame in last_frames):
        checks.ok("story", "cta_at_end")
    else:
        checks.add(
            _finding(
                "story",
                "cta_not_at_end",
                8.0,
                "no text or end card could be detected in the last 4 s (a heuristic on the frames)",
                "end on the brand's CTA image or an end card with the call to action",
                time=max(0.0, ctx.duration - 4.0),
                end=ctx.duration,
                detail={"confidence": "low"},
            )
        )
    return checks


def _zones(
    ctx: _Context,
    entries: list[CaptionEntry],
    platforms: tuple[str, ...],
    sampled: np.ndarray,
) -> tuple[Checks, list[dict[str, Any]]]:
    size = (ctx.width, ctx.height)
    if ctx.model is not None:
        from adanvil.storyboard.craft import overlay_items

        items = overlay_items(ctx.model, ctx.root, ctx.brand, entries)
        size = (ctx.model.format.width, ctx.model.format.height)
    else:
        items = []
        step = ctx.duration / max(len(sampled), 1)
        for number, frame in enumerate(sampled):
            for left, top, right, bottom in detect_text(frame):
                items.append(
                    {
                        "kind": "text",
                        "label": "text detected on the frame",
                        "start": number * step,
                        "end": (number + 1) * step,
                        "box": (
                            left * ctx.width,
                            top * ctx.height,
                            right * ctx.width,
                            bottom * ctx.height,
                        ),
                    }
                )
    checks, items = safe_zone_checks(items, platforms, size)
    for item in items:
        item["frame_size"] = size
    if ctx.model is not None:
        _card_content_findings(ctx, checks)
        _face_findings(ctx, items, checks)
    return checks, items


MAX_FACE_SAMPLES = 12
FACE_LOOKS = 4  # frames looked at per caption


def _face_findings(ctx: _Context, items: list[dict[str, Any]], checks: Checks) -> None:
    """``caption_on_face``: a caption or comment that covers a face on the RENDERED frame (the
    vision engine's detector, one frame in the middle of the caption), in every format."""

    from adanvil.captionfaces import caption_on_face, rendered_faces, sample_moments
    from adanvil.vision import get_engine

    texts = [item for item in items if item.get("kind") in {"caption", "comment"}]
    if not texts:
        return
    try:
        engine = get_engine()
    except Exception:
        engine = None
    if engine is None:
        checks.skip(
            "safe_zones",
            "caption_on_face",
            "no vision engine (install it with adanvil vision-setup)",
        )
        return
    if len(texts) > MAX_FACE_SAMPLES:
        step = len(texts) / MAX_FACE_SAMPLES
        texts = [texts[int(number * step)] for number in range(MAX_FACE_SAMPLES)]
    frame_w, frame_h = ctx.model.format.width, ctx.model.format.height
    found = False
    reported: set[tuple[str, int]] = set()
    for item in texts:
        begin = float(item["start"])
        stop = min(float(item["end"]), ctx.duration)
        hit = None
        for moment in sample_moments(
            begin, min(stop, ctx.duration - 0.05), ctx.duration, count=FACE_LOOKS
        ):
            try:
                faces, size = rendered_faces(ctx.video, moment, engine)
            except Exception:
                continue
            left, top, right, bottom = item["box"]
            box = (
                left * size[0] / frame_w,
                top * size[1] / frame_h,
                right * size[0] / frame_w,
                bottom * size[1] / frame_h,
            )
            if caption_on_face(box, faces):
                hit = moment
                break
        if hit is not None and (str(item.get("label")), round(hit)) not in reported:
            moment = hit
            reported.add((str(item.get("label")), round(moment)))
            found = True
            checks.add(
                _finding(
                    "safe_zones",
                    "caption_on_face",
                    12.0,
                    f"caption_on_face: the caption {item.get('label')!r} covers a face at "
                    f"{moment:.1f}s",
                    "move the caption above the head or below the chin (for one format: "
                    'formats: {"<ratio>": {"y": ...}} on the overlay)',
                    time=moment,
                    detail={"box": [round(value) for value in item["box"]]},
                )
            )
    if not found:
        checks.ok("safe_zones", "caption_on_face")


def _card_content_findings(ctx: _Context, checks: Checks) -> None:
    """``caption_over_card_content``: a caption laid over the lettering or button of an end
    card (or any still shown whole), found on the picture file (``adanvil.busy``)."""

    from adanvil.formats import card_content_clashes

    try:
        document = ctx.model.model_dump(mode="json", by_alias=True, exclude_none=True)
        clashes = card_content_clashes(document, ctx.root)
    except Exception:
        return
    for index, box, _busy, picture in clashes:
        start = float(document["overlays"][index].get("start") or 0.0)
        checks.add(
            _finding(
                "safe_zones",
                "caption_over_card_content",
                10.0,
                f"caption_over_card_content: overlay {index} covers the lettering or button "
                f"of {picture['source']} (y {round(box[1])} to {round(box[3])})",
                "move the caption into an empty band of the picture (or above or below it), "
                "or end it before the card",
                time=start,
                detail={"overlay": index, "source": picture["source"]},
            )
        )


def _audio(ctx: _Context, silences: list[tuple[float, float]]) -> Checks:
    checks = Checks()
    if not ctx.has_audio:
        checks.add(
            _finding(
                "audio",
                "audio_missing",
                40.0,
                "the video has no audio stream: the ad is silent",
                "add music (audio.music) or a voice-over; a silent paid ad loses the viewers "
                "who have the sound on",
                time=0.0,
            )
        )
        return checks
    loudness = integrated_loudness(ctx.video)
    ctx.facts["audio"] = {"loudness_lufs": None if loudness is None else round(loudness, 1)}
    if loudness is None:
        checks.add(
            _finding(
                "audio",
                "audio_silent",
                40.0,
                "the audio track is silent (no measurable loudness)",
                "check the music or voice path in the storyboard audio section",
                time=0.0,
            )
        )
    elif loudness < LOUDNESS_LIMITS[0] or loudness > LOUDNESS_LIMITS[1]:
        checks.add(
            _finding(
                "audio",
                "loudness",
                25.0,
                f"integrated loudness is {loudness:.1f} LUFS, outside {LOUDNESS_LIMITS[0]:g} to "
                f"{LOUDNESS_LIMITS[1]:g}",
                "set audio.loudness near -14 LUFS (or raise or lower the volumes)",
            )
        )
    elif not LOUDNESS_OK[0] <= loudness <= LOUDNESS_OK[1]:
        checks.add(
            _finding(
                "audio",
                "loudness",
                6.0,
                f"integrated loudness is {loudness:.1f} LUFS, outside the comfortable "
                f"{LOUDNESS_OK[0]:g} to {LOUDNESS_OK[1]:g}",
                "aim for about -14 LUFS (audio.loudness)",
            )
        )
    else:
        checks.ok("audio", "loudness")

    long_silences = [
        (start, end)
        for start, end in silences
        if end - start >= SILENCE_MAX and not (start <= 0.05 and end - start >= 0.5)
    ]
    for start, end in long_silences[:3]:
        checks.add(
            _finding(
                "audio",
                "audio_silence",
                15.0 if end - start >= 1.5 else 10.0,
                f"{end - start:.1f}s of silence at {start:.1f}s to {end:.1f}s",
                "fill the gap: extend the music, move the voice, or add a sound effect",
                time=start,
                end=end,
            )
        )
    if not long_silences:
        checks.ok("audio", "no_long_silence")

    checks.extend(_voice_over_music(ctx))
    return checks


def _music_licence(ctx: _Context) -> Checks:
    """The licence of ``audio.music``: ``facts.music_licence`` always; a finding when the track
    has no licence record (low penalty) or needs an attribution (no penalty: an action)."""

    checks = Checks()
    if ctx.model is None or not ctx.model.audio.music:
        return checks
    info = music_licence(ctx.root, ctx.model.audio.music)
    if info is None:
        return checks
    warnings = info.pop("warnings")
    ctx.facts.setdefault("audio", {})["music_licence"] = info
    if info["status"] == "unknown":
        checks.add(
            _finding(
                "audio",
                "music_licence_unknown",
                3.0,
                warnings[0],
                "fetch the track with music_fetch (it records the licence), or write the "
                "licence in music/<file>.licence.json after checking it allows commercial use",
                detail={"path": info["path"]},
                severity="low",
            )
        )
    elif info["attribution_required"]:
        checks.add(
            _finding(
                "audio",
                "music_attribution_required",
                0.0,
                warnings[0],
                "paste the attribution text in the ad's description (the post caption)",
                detail={"path": info["path"], "attribution_text": info["attribution_text"]},
                severity="low",
            )
        )
    else:
        checks.ok("audio", "music_licence")
    return checks


def _voice_over_music(ctx: _Context) -> Checks:
    checks = Checks()
    model = ctx.model
    if model is None:
        checks.skip("audio", "voice_over_music", "needs the storyboard (the mix levels)")
        return checks
    if not (model.audio.voice or model.audio.clips):
        checks.skip("audio", "voice_over_music", "no voice-over or clip sound under a music bed")
        return checks
    if model.audio.music is None:
        checks.skip("audio", "voice_over_music", "no music bed under the voice")
        return checks
    from adanvil.storyboard.validate import _relative_file

    music = _relative_file(ctx.root, model.audio.music)
    if music is None or not music.is_file():
        checks.skip("audio", "voice_over_music", "the music file was not found")
        return checks
    gain = music_gain(music, model.audio.music_volume)
    loudness = integrated_loudness(music)
    music_level = (
        MUSIC_REFERENCE_LUFS + 20 * math.log10(model.audio.music_volume)
        if loudness is None
        else loudness + 20 * math.log10(max(gain, 1e-6))
    )
    voice_level = float(model.audio.voice_loudness)
    margin = voice_level - music_level
    ctx.facts.setdefault("audio", {})["voice_over_music_db"] = round(margin, 1)
    if margin >= VOICE_OVER_MUSIC_DB:
        checks.ok("audio", "voice_over_music")
        return checks
    needed = model.audio.music_volume * 10 ** ((margin - VOICE_OVER_MUSIC_DB) / 20.0)
    floor = 0.2  # the lowest music_volume the schema accepts
    fix = (
        f"lower audio.music_volume to {max(needed, floor):.2f} or less"
        + (" (0.2 is the minimum: also pick a quieter track or raise voice_loudness)"
           if needed < floor else "")
    )
    penalty = 25.0 if margin < VOICE_MASKED_DB else 6.0
    checks.add(
        _finding(
            "audio",
            "voice_masked",
            penalty,
            f"the voice sits {margin:.1f} dB above the music (at least "
            f"{VOICE_OVER_MUSIC_DB:g} dB keeps it clear)",
            fix,
            detail={"margin_db": round(margin, 1), "music_volume": model.audio.music_volume},
        )
    )
    return checks


# ---------------------------------------------------------------------------- the score


def _area_scores(checks: Checks) -> dict[str, dict[str, Any]]:
    areas: dict[str, dict[str, Any]] = {}
    for area in WEIGHTS:
        found = [f for f in checks.findings if f.area == area]
        ran = any(p.startswith(f"{area}.") for p in checks.passed) or bool(found)
        if not ran:
            areas[area] = {"score": None, "weight": WEIGHTS[area], "findings": 0}
            continue
        score = max(0.0, 100.0 - sum(f.penalty for f in found))
        areas[area] = {"score": round(score), "weight": WEIGHTS[area], "findings": len(found)}
    return areas


def ad_score(
    root: Path,
    video: Path,
    *,
    storyboard: Storyboard | Mapping[str, Any] | None = None,
    platforms: list[str] | None = None,
    out_dir: Path | None = None,
) -> ScoreReport:
    """Score a rendered ad (see the module docstring). ``video`` is a path to the rendered
    file; ``storyboard`` is the storyboard it was rendered from, when known."""

    root = Path(root)
    video = Path(video)
    try:
        info = probe(video)
    except Exception as exc:
        raise ScoreError(f"cannot read the video {video.name}: {exc}") from exc
    if info.kind != "video" or not info.duration or not info.width or not info.height:
        raise ScoreError(f"{video.name} is not a video with a duration")
    chosen = check_platforms(platforms)
    brand: Brand | None = None
    brand_file = root / "brand.yaml"
    if brand_file.is_file():
        try:
            brand = load_brand(brand_file)
        except Exception:
            brand = None
    model: Storyboard | None = None
    if storyboard is not None:
        try:
            model = (
                storyboard
                if isinstance(storyboard, Storyboard)
                else Storyboard.model_validate(dict(storyboard))
            )
        except Exception as exc:
            raise ScoreError(
                f"the storyboard is not valid ({exc}): run storyboard_validate and fix it first"
            ) from exc
    output = Path(out_dir) if out_dir is not None else video.with_name(f"{video.stem}_score")
    output.mkdir(parents=True, exist_ok=True)
    duration = float(info.duration)
    ctx = _Context(
        root=root,
        video=video,
        duration=duration,
        width=info.width,
        height=info.height,
        fps=float(info.fps or 30.0),
        has_audio=info.has_audio,
        model=model,
        brand=brand,
        starts=[],
        ends=[],
        out_dir=output,
    )
    entries: list[CaptionEntry] = []
    if model is not None:
        timeline = model.timeline()
        ctx.starts = [item[0] for item in timeline]
        ctx.ends = [item[1] for item in timeline]
        ctx.cuts = [start for start in ctx.starts[1:] if start < duration]
        if abs(model.total_duration() - duration) > 0.3:
            ctx.notes.append(
                f"the storyboard lasts {model.total_duration():.1f}s but the video "
                f"{duration:.1f}s: is it the storyboard this video was rendered from?"
            )
        from adanvil.storyboard.craft import collect_captions

        entries, caption_notes = collect_captions(model, root, brand, ctx.starts, ctx.ends)
        ctx.notes.extend(caption_notes)
    else:
        ctx.cuts = _scene_cuts(video)

    height = max(2, round(_ANALYSIS_WIDTH * info.height / info.width / 2) * 2)
    first = _raw_frames(
        video, fps=10, width=_ANALYSIS_WIDTH, height=height, duration=HOOK_SECONDS + 0.1
    )
    last_start = max(0.0, duration - 4.0)
    last = _raw_frames(
        video, fps=5, width=_ANALYSIS_WIDTH, height=height, start=last_start, duration=4.0
    )
    sampled = _raw_frames(
        video,
        fps=max(0.2, 8.0 / duration),
        width=_ANALYSIS_WIDTH,
        height=height,
    )[:8]
    silences = silence_intervals(video, duration) if info.has_audio else []

    checks = Checks()
    checks.extend(_hook(ctx, first, silences, entries))
    checks.extend(_pacing(ctx))
    checks.extend(_reading(ctx, entries))
    checks.extend(_story(ctx, entries, last))
    zone_checks, items = _zones(ctx, entries, chosen, sampled)
    checks.extend(zone_checks)
    checks.extend(_audio(ctx, silences))
    checks.extend(_music_licence(ctx))

    areas = _area_scores(checks)
    evaluated = {name: data for name, data in areas.items() if data["score"] is not None}
    total_weight = sum(data["weight"] for data in evaluated.values())
    score = (
        sum(data["score"] * data["weight"] for data in evaluated.values()) / total_weight
        if total_weight
        else 0.0
    )
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    ranked: list[tuple[float, Finding]] = []
    for finding in checks.findings:
        weight = WEIGHTS[finding.area]
        impact = finding.penalty * weight / total_weight if total_weight else 0.0
        ranked.append((impact, finding))
    ranked.sort(key=lambda pair: (-pair[0], severity_rank[pair[1].severity]))
    findings = []
    for impact, finding in ranked:
        payload = finding.as_dict()
        payload["impact"] = round(impact, 1)
        findings.append(payload)

    images = _images(ctx, items, chosen, findings)
    ctx.facts.update(
        {
            "duration": round(duration, 2),
            "size": [info.width, info.height],
            "storyboard": model is not None,
            "coverage": round(total_weight / sum(WEIGHTS.values()), 2),
        }
    )
    return ScoreReport(
        score=round(score),
        meets_target=round(score) >= TARGET,
        sub_scores=areas,
        findings=findings,
        passed=checks.passed,
        skipped=checks.skipped,
        facts=ctx.facts,
        safe_zones=zones_table(chosen),
        thresholds=THRESHOLDS,
        images=images,
        notes=ctx.notes,
    )


def _images(
    ctx: _Context,
    items: list[dict[str, Any]],
    platforms: tuple[str, ...],
    findings: list[dict[str, Any]],
) -> list[Path]:
    """``safe_zones_preview.jpg`` and the frame of each of the top three findings."""

    images: list[Path] = []
    moment = ctx.duration / 2.0
    colliding = [item for item in items if item.get("inside")]
    if colliding and not any(item["start"] <= moment < item["end"] for item in colliding):
        worst = colliding[0]
        moment = (worst["start"] + worst["end"]) / 2.0
    moment = min(max(0.0, moment), max(0.0, ctx.duration - 0.05))
    with tempfile.TemporaryDirectory(prefix="adanvil-score-", dir=ctx.out_dir) as temporary:
        frame_path = Path(temporary) / "mid.png"
        extract_frame(ctx.video, moment, frame_path)
        with Image.open(frame_path) as opened:
            frame = opened.convert("RGB")
        size = items[0].get("frame_size") if items else None
        scale_x = frame.width / size[0] if size else 1.0
        scale_y = frame.height / size[1] if size else 1.0
        visible = []
        for item in items:
            if item["start"] <= moment < item["end"]:
                left, top, right, bottom = item["box"]
                visible.append(
                    {
                        "box": (left * scale_x, top * scale_y, right * scale_x, bottom * scale_y),
                        "inside": item.get("inside", False),
                    }
                )
        preview = draw_preview(frame, platforms, visible, ctx.out_dir / "safe_zones_preview.jpg")
        frame.close()
    images.append(preview)
    taken = 0
    for number, finding in enumerate(findings):
        if taken >= 3:
            break
        when = finding.get("time")
        if when is None:
            continue
        when = min(max(0.0, float(when)), max(0.0, ctx.duration - 0.05))
        target = ctx.out_dir / f"finding_{number + 1}_{finding['id']}.jpg"
        with tempfile.TemporaryDirectory(prefix="adanvil-score-", dir=ctx.out_dir) as temporary:
            raw = Path(temporary) / "frame.png"
            extract_frame(ctx.video, when, raw, width=540)
            with Image.open(raw) as opened:
                opened.convert("RGB").save(target, format="JPEG", quality=88)
        images.append(target)
        taken += 1
    return images


__all__ = [
    "PLATFORMS",
    "TARGET",
    "THRESHOLDS",
    "ScoreError",
    "ScoreReport",
    "ad_score",
    "caption_contrast",
    "contrast_ratio",
    "detect_text",
    "luminance",
    "silence_intervals",
]
