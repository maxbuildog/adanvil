"""Validation of the visual devices: split screens, video insets (picture-in-picture), stickers.

The rules and their codes (errors block rendering, warnings do not):

- ``split_layout`` (error): a panel would be smaller than 64 px.
- ``split_same_source`` (warning): the two panels of a split, or the two sides of a reveal,
  are the same file at overlapping moments: nothing to compare.
- ``reveal_orientation`` (warning): the two sides of a reveal are not the same orientation
  family (portrait, landscape, square).
- ``reveal_after_short`` (warning): a reveal leaves under 0.6 s on the after picture.
- ``source_duration`` (error): a panel or inset starts at or after the end of its clip.
- ``panel_crop`` (error): a panel ``crop`` leaves the source picture.
- ``split_crop`` (warning): the panel's cover crop keeps less than 40% of the source picture.
- ``digital_zoom`` (warning): a panel ``crop`` enlarges the source more than 1.25x.
- ``text_overflow`` (warning): a split label is wider than its panel after shrinking.
- ``inset_out_of_frame`` (error): an inset leaves the frame.
- ``inset_source_short`` (warning): the clip of an inset ends before its span (it holds its
  last frame).
- ``caption_covered_by_inset`` (warning): a caption is (partly) hidden under an inset.
- ``inset_audio`` (error) and ``inset_audio_replaces`` / ``inset_audio_silent`` (warnings): an
  inset with ``audio`` needs ``audio.clips`` and no other voice during its span.
- ``sticker_out_of_frame`` (error): a sticker leaves the frame.
- ``sticker_render`` (error): a sticker cannot be drawn.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from adanvil.media import ClipSegment, RevealSegment, SplitSegment, split_geometry
from adanvil.text.emoji import DirectoryProvider, NoEmojiProvider, missing_emoji

from .schema import CaptionOverlay, StickerOverlay, Storyboard, VideoOverlay

if TYPE_CHECKING:
    from .validate import Issue

_MIN_PANEL = 64
_MIN_VISIBLE = 0.4
_MAX_ENLARGEMENT = 1.25
_MIN_AFTER_HOLD = 0.6  # seconds the after picture of a reveal should stay on screen
_SLACK = {"arrow": 6, "circle": 6, "underline": 6, "badge": 14}  # px: halo or shadow


def _issue(code: str, message: str, where: str | None) -> Issue:
    from .validate import Issue

    return Issue(code=code, message=message, where=where)


def _plain_path(value: str) -> str:
    return value.replace("\\", "/").removeprefix("./")


def _played_range(panel: Any, duration: float) -> tuple[float, float]:
    """The part of its source a panel or side plays: ``in`` for ``duration`` seconds of the
    segment (a freeze holds the last frame, it adds nothing new to play)."""

    hold = panel.freeze.hold if panel.freeze is not None else 0.0
    return panel.in_, panel.in_ + max(0.0, duration - hold) * panel.speed


def _check_same_source(
    index: int, segment: SplitSegment | RevealSegment, warnings: list[Issue]
) -> None:
    """``split_same_source``: both pictures are the same file and play overlapping moments,
    so there is nothing to compare."""

    first, second = (
        segment.panels if isinstance(segment, SplitSegment) else (segment.before, segment.after)
    )
    if _plain_path(first.source) != _plain_path(second.source):
        return
    a_start, a_end = _played_range(first, segment.duration)
    b_start, b_end = _played_range(second, segment.duration)
    if a_start < b_end and b_start < a_end:
        warnings.append(
            _issue(
                "split_same_source",
                f"split_same_source: segments[{index}]: the two sides show the same take "
                f"('{first.source}'): the difference will not be visible; use real before "
                "and after pictures",
                f"segments[{index}]",
            )
        )


def _check_panel_source(
    who: str,
    where: str,
    panel: Any,
    info: Any,
    box: tuple[int, int, int, int],
    errors: list[Issue],
    warnings: list[Issue],
) -> tuple[int, int] | None:
    """The source checks of one panel (or reveal side) of the box ``box``: where it starts,
    its crop, how much of it the cover crop keeps. Returns the size of the picture the box is
    cut from (the crop, else the source) or ``None`` when it could not be measured."""

    if info is None or info.kind == "audio":
        return None
    if info.kind == "video" and info.duration is not None and panel.in_ >= info.duration - 0.05:
        errors.append(
            _issue(
                "source_duration",
                f"{who[0].upper()}{who[1:]} starts at {panel.in_:g}s, at or after the end of "
                f"'{panel.source}' ({info.duration:.3f}s).",
                f"{where}.in",
            )
        )
    if not info.width or not info.height:
        return None
    _x, _y, panel_w, panel_h = box
    source_w, source_h = info.width, info.height
    if panel.crop is not None:
        crop = panel.crop
        if (
            crop.w <= 0
            or crop.h <= 0
            or crop.x < 0
            or crop.y < 0
            or crop.x + crop.w > source_w
            or crop.y + crop.h > source_h
        ):
            errors.append(
                _issue(
                    "panel_crop",
                    f"panel_crop: the crop of {who} leaves the {source_w}x{source_h} source "
                    "picture.",
                    f"{where}.crop",
                )
            )
            return None
        plain = max(panel_w / source_w, panel_h / source_h)
        enlarged = max(panel_w / crop.w, panel_h / crop.h) / plain
        if enlarged > _MAX_ENLARGEMENT:
            warnings.append(
                _issue(
                    "digital_zoom",
                    f"digital_zoom: {who} enlarges the source {enlarged:.1f}x: a crop this "
                    "tight looks zoomed and soft; pick a shot filmed close instead",
                    where,
                )
            )
        source_w, source_h = crop.w, crop.h
    if panel.fit == "contain" and info.kind == "image":
        return source_w, source_h
    scale = max(panel_w / source_w, panel_h / source_h)
    visible = (panel_w / (source_w * scale)) * (panel_h / (source_h * scale))
    if visible < _MIN_VISIBLE:
        warnings.append(
            _issue(
                "split_crop",
                f"split_crop: {who} keeps only {visible:.0%} of '{panel.source}' once cropped "
                f"to {panel_w}x{panel_h}: set its focus on the subject, or pick a source "
                "closer to that shape (a portrait clip suits side by side, a landscape one "
                "top and bottom)",
                where,
            )
        )
    return source_w, source_h


def check_splits(
    model: Storyboard,
    infos: dict[tuple[int, int], Any],
    errors: list[Issue],
    warnings: list[Issue],
) -> None:
    """Layout, labels and panel sources of the split segments (``infos`` are the probed panel
    sources by ``(segment, panel)``, empty when the files are not checked)."""

    from adanvil.media.split import render_labels

    for index, segment in enumerate(model.segments):
        if not isinstance(segment, SplitSegment):
            continue
        boxes, _gap = split_geometry(
            segment.layout, model.format.width, model.format.height, segment.gap
        )
        if any(w < _MIN_PANEL or h < _MIN_PANEL for _x, _y, w, h in boxes):
            errors.append(
                _issue(
                    "split_layout",
                    f"split_layout: segments[{index}] gap {segment.gap} leaves panels of "
                    f"{boxes[0][2]}x{boxes[0][3]} px, under {_MIN_PANEL} px: use a smaller gap",
                    f"segments[{index}].gap",
                )
            )
            continue
        _check_same_source(index, segment, warnings)
        if segment.labels is not None:
            try:
                _picture, overflow = render_labels(
                    segment, model.format.width, model.format.height, NoEmojiProvider()
                )
                _picture.close()
            except Exception as exc:
                errors.append(
                    _issue(
                        "text_render",
                        f"Could not render the labels of segments[{index}]: {exc}",
                        f"segments[{index}].labels",
                    )
                )
                overflow = []
            for number, flag in enumerate(overflow):
                if flag:
                    warnings.append(
                        _issue(
                            "text_overflow",
                            f"Label {number} of segments[{index}] ({segment.labels[number]!r}) "
                            "is wider than its panel even after shrinking: shorten it.",
                            f"segments[{index}].labels[{number}]",
                        )
                    )
        for number, panel in enumerate(segment.panels):
            _check_panel_source(
                f"panel {number} of segments[{index}]",
                f"segments[{index}].panels[{number}]",
                panel,
                infos.get((index, number)),
                boxes[number],
                errors,
                warnings,
            )


def _orientation(width: int, height: int) -> str:
    ratio = width / max(height, 1)
    if ratio < 0.9:
        return "portrait"
    if ratio > 1.1:
        return "landscape"
    return "square"


def check_reveals(
    model: Storyboard,
    infos: dict[tuple[int, int], Any],
    errors: list[Issue],
    warnings: list[Issue],
) -> None:
    """Sources, labels, timing and orientation of the reveal segments (``infos`` are the probed
    sides by ``(segment, side)`` with 0 the before side and 1 the after side)."""

    from adanvil.media.reveal import render_labels

    frame = (0, 0, model.format.width, model.format.height)
    for index, segment in enumerate(model.segments):
        if not isinstance(segment, RevealSegment):
            continue
        _check_same_source(index, segment, warnings)
        settled = segment.duration - segment.hold_before - segment.sweep
        if settled < _MIN_AFTER_HOLD - 1e-9:
            warnings.append(
                _issue(
                    "reveal_after_short",
                    f"reveal_after_short: segments[{index}] stays on the after picture for only "
                    f"{settled:.2f}s: leave at least {_MIN_AFTER_HOLD:g}s so the result is seen "
                    "(lengthen duration or shorten hold_before or sweep)",
                    f"segments[{index}].duration",
                )
            )
        if segment.labels is not None:
            try:
                pictures, overflow = render_labels(
                    segment, model.format.width, model.format.height, NoEmojiProvider()
                )
                for picture in pictures:
                    picture.close()
            except Exception as exc:
                errors.append(
                    _issue(
                        "text_render",
                        f"Could not render the labels of segments[{index}]: {exc}",
                        f"segments[{index}].labels",
                    )
                )
                overflow = []
            for number, flag in enumerate(overflow):
                if flag:
                    warnings.append(
                        _issue(
                            "text_overflow",
                            f"Label {number} of segments[{index}] ({segment.labels[number]!r}) "
                            "is wider than its room even after shrinking: shorten it.",
                            f"segments[{index}].labels[{number}]",
                        )
                    )
        shapes: list[tuple[str, int, int]] = []
        for number, name in enumerate(("before", "after")):
            side = segment.before if number == 0 else segment.after
            size = _check_panel_source(
                f"the {name} side of segments[{index}]",
                f"segments[{index}].{name}",
                side,
                infos.get((index, number)),
                frame,
                errors,
                warnings,
            )
            if size is not None:
                shapes.append((name, *size))
        if len(shapes) == 2 and _orientation(*shapes[0][1:]) != _orientation(*shapes[1][1:]):
            (first, fw, fh), (second, sw, sh) = shapes
            warnings.append(
                _issue(
                    "reveal_orientation",
                    f"reveal_orientation: the {first} side of segments[{index}] is "
                    f"{_orientation(fw, fh)} ({fw}x{fh}) and the {second} side is "
                    f"{_orientation(sw, sh)} ({sw}x{sh}): both are cropped to the same frame, "
                    "so one loses most of its picture and the two will not line up; use two "
                    "pictures taken the same way",
                    f"segments[{index}]",
                )
            )


def inset_box(
    overlay: VideoOverlay, info: Any | None
) -> tuple[int, int, int, int | None]:
    """``(x0, y0, x1, y1)`` of an inset on the frame; ``y1`` is ``None`` when the height depends
    on a clip that was not probed."""

    from adanvil.render.inset import inset_footprint, inset_size

    if overlay.shape == "circle" or overlay.height is not None:
        height: int | None = inset_size(overlay, 1, 1)[1]
    elif info is not None and info.width and info.height:
        height = inset_size(overlay, info.width, info.height)[1]
    else:
        height = None
    x0, y0, x1, _ = inset_footprint(overlay, height or 0)
    return (x0, y0, x1, None if height is None else y0 + height)


def check_insets(
    model: Storyboard,
    infos: dict[int, Any],
    starts: list[float],
    ends: list[float],
    total: float,
    text_bboxes: dict[str, tuple[int, int, int, int]],
    caption_entries: list[tuple[str, CaptionOverlay]],
    errors: list[Issue],
    warnings: list[Issue],
) -> None:
    """Frame, source, sound and captions of the video insets."""

    from adanvil.render.inset import inset_span

    width, height = model.format.width, model.format.height
    spans: list[tuple[int, VideoOverlay, float, float]] = []
    for index, overlay in enumerate(model.overlays):
        if not isinstance(overlay, VideoOverlay):
            continue
        info = infos.get(index)
        begin, finish = inset_span(overlay, getattr(info, "duration", None), total)
        spans.append((index, overlay, begin, finish))
        where = f"overlays[{index}]"
        x0, y0, x1, y1 = inset_box(overlay, info)
        inside = x0 >= 0 and y0 >= 0 and x1 <= width and (y1 is None or y1 <= height)
        if not inside or (y1 is None and y0 >= height):
            shown = f"{x0},{y0} to {x1},{y1 if y1 is not None else '?'}"
            errors.append(
                _issue(
                    "inset_out_of_frame",
                    f"inset_out_of_frame: the video inset overlays[{index}] covers {shown}, "
                    f"outside the {width}x{height} frame: move it or make it smaller.",
                    where,
                )
            )
        if info is None:
            continue
        if info.duration is not None and overlay.in_ >= info.duration - 0.05:
            errors.append(
                _issue(
                    "source_duration",
                    f"The video inset overlays[{index}] starts at {overlay.in_:g}s, at or after "
                    f"the end of '{overlay.source}' ({info.duration:.3f}s).",
                    f"{where}.in",
                )
            )
        elif (
            overlay.end is not None
            and info.duration is not None
            and min(overlay.end, total) - overlay.start > info.duration - overlay.in_ + 0.05
        ):
            warnings.append(
                _issue(
                    "inset_source_short",
                    f"inset_source_short: '{overlay.source}' lasts "
                    f"{info.duration - overlay.in_:.2f}s from {overlay.in_:g}s but the inset "
                    f"shows it for {min(overlay.end, total) - overlay.start:.2f}s: it holds its "
                    "last frame for the rest",
                    where,
                )
            )

    for index, overlay, begin, finish in spans:
        for key, caption in caption_entries:
            box = text_bboxes.get(key)
            if box is None:
                continue
            caption_end = total if caption.end is None else min(caption.end, total)
            if min(caption_end, finish) - max(caption.start, begin) <= 1e-6:
                continue
            x0, y0, x1, y1 = inset_box(overlay, infos.get(index))
            y1 = y1 if y1 is not None else height
            if box[0] < x1 and box[2] > x0 and box[1] < y1 and box[3] > y0:
                warnings.append(
                    _issue(
                        "caption_covered_by_inset",
                        f"caption_covered_by_inset: the caption {key} "
                        f"({' / '.join(caption.lines)!r}) is partly hidden under the video "
                        f"inset overlays[{index}] while both are on screen: move one of them",
                        f"overlays[{key}]",
                    )
                )

    _check_inset_audio(model, infos, spans, starts, ends, errors, warnings)


def _check_inset_audio(
    model: Storyboard,
    infos: dict[int, Any],
    spans: list[tuple[int, VideoOverlay, float, float]],
    starts: list[float],
    ends: list[float],
    errors: list[Issue],
    warnings: list[Issue],
) -> None:
    voiced = [item for item in spans if item[1].audio]
    for number, (index, overlay, begin, finish) in enumerate(voiced):
        where = f"overlays[{index}].audio"
        if not model.audio.clips:
            errors.append(
                _issue(
                    "inset_audio",
                    f"inset_audio: the video inset overlays[{index}] has audio: true, which "
                    "needs audio.clips: true (its sound joins the sound of the clips)",
                    where,
                )
            )
            continue
        if model.audio.voice is not None:
            errors.append(
                _issue(
                    "inset_audio",
                    f"inset_audio: the video inset overlays[{index}] has audio: true while a "
                    "voice-over (audio.voice) is active: two voices at once. Use audio: false "
                    "or drop the voice-over",
                    where,
                )
            )
            continue
        for other_index, _other, other_begin, other_finish in voiced[number + 1 :]:
            if min(finish, other_finish) - max(begin, other_begin) > 1e-6:
                errors.append(
                    _issue(
                        "inset_audio",
                        f"inset_audio: the video insets overlays[{index}] and "
                        f"overlays[{other_index}] both have audio: true at the same time: two "
                        "voices at once",
                        where,
                    )
                )
        info = infos.get(index)
        if info is not None and not info.has_audio:
            warnings.append(
                _issue(
                    "inset_audio_silent",
                    f"inset_audio_silent: '{overlay.source}' has no audio stream: the "
                    "inset will be silent and the bed is silent under it",
                    where,
                )
            )
        for number_, segment in enumerate(model.segments):
            if min(finish, ends[number_]) - max(begin, starts[number_]) <= 1e-6:
                continue
            if isinstance(segment, ClipSegment) and segment.audio_from is not None:
                errors.append(
                    _issue(
                        "inset_audio",
                        f"inset_audio: segments[{number_}] plays the voice of "
                        f"'{segment.audio_from.source}' (audio_from) while the video inset "
                        f"overlays[{index}] speaks: two voices at once",
                        where,
                    )
                )
            elif isinstance(segment, ClipSegment):
                warnings.append(
                    _issue(
                        "inset_audio_replaces",
                        f"inset_audio_replaces: under the video inset overlays[{index}] the "
                        f"sound of segments[{number_}] is left out (the inset's sound replaces "
                        "it)",
                        where,
                    )
                )


def check_stickers(
    model: Storyboard,
    root: Path,
    source_exists: Callable[[str], bool],
    errors: list[Issue],
    warnings: list[Issue],
    emoji_dir: Path | None,
) -> None:
    """Every sticker is drawn (without fetching emoji pictures) and must stay inside the frame."""

    from adanvil.render.stickers import sticker_art

    width, height = model.format.width, model.format.height
    provider = DirectoryProvider(emoji_dir) if emoji_dir is not None else None
    for index, overlay in enumerate(model.overlays):
        if not isinstance(overlay, StickerOverlay):
            continue
        where = f"overlays[{index}]"
        if overlay.kind == "image" and not source_exists(overlay.source or ""):
            continue  # missing_file already says it
        if overlay.kind == "emoji" and provider is not None:
            absent = missing_emoji([overlay.emoji or ""], provider)
            if absent:
                warnings.append(
                    _issue(
                        "emoji_missing",
                        f"emoji_missing: {' '.join(absent)} has no PNG in the emoji folder "
                        f"'{model.emoji}' (sticker {where}); it cannot be drawn.",
                        where,
                    )
                )
        try:
            art = sticker_art(overlay, root=root)
        except Exception as exc:
            errors.append(
                _issue("sticker_render", f"Could not draw the sticker {where}: {exc}", where)
            )
            continue
        left, top, right, bottom = art.bbox
        art.image.close()
        slack = _SLACK.get(overlay.kind, 0)
        if left < -slack or top < -slack or right > width + slack or bottom > height + slack:
            errors.append(
                _issue(
                    "sticker_out_of_frame",
                    f"sticker_out_of_frame: the {overlay.kind} sticker {where} covers "
                    f"{left},{top} to {right},{bottom}, outside the {width}x{height} frame: "
                    "move it or make it smaller.",
                    where,
                )
            )


__all__ = ["check_insets", "check_reveals", "check_splits", "check_stickers", "inset_box"]
