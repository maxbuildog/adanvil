"""One storyboard in several aspect ratios: 9:16, 4:5 and 1:1.

A storyboard has one ``format`` (``width`` x ``height``) and every position in it (a caption's
``y``, a sticker's ``x`` and ``y``, an inset's place) is a pixel of that frame. ``reformat``
turns a storyboard into the same ad for another ratio:

* the picture follows by itself: segments are cover-cropped to the new frame (face-aware when
  ``focus: "face"``, else ``focus``), stills and end cards that do not fit are shown whole over
  their blurred copy (``contain`` + blur);
* what is laid over the picture is moved by bands. The part of the frame the platform leaves
  free (``safezones``: between the top bar and the bottom band) is stretched or squeezed onto
  the free part of the new frame, the part under the bars is scaled with the frame height, so a
  caption that was a third of the way down is still a third of the way down and nothing slides
  under the interface of the new placement;
* texts, stickers and images that would still leave the free part (a tall block on a short
  frame) are pushed back inside it, measured on the real boxes.

* a caption never sits on a face: with the face boxes of the footage under it (the stored
  vision faces, else the vision worker when it is installed, else the check is skipped with a
  note) carried to the new frame, a caption keeps its place when it covers no face core; else it
  moves the least, to the nearest free spot on the same side of the face (a hook at the top stays
  above the head), shrinking down to the readable minimum if needed, else the note
  ``caption_on_face`` says so; every move is noted with its distance;
* an overlay may carry ``formats: {"1:1": {"y": 900}, "4:5": {...}}`` (``x``, ``y``, ``size``,
  ``anchor``, ``hidden``): applied last, after all of the above, for that format only;
* when the storyboard already has the target frame (a 9:16 storyboard in 9:16) nothing is
  re-laid out: the output is the storyboard as it is (only the name, the output path and the
  per-format overrides change), so it scores like the plain render. The band remap applies to
  the other ratios only;

* text and graphics are scaled with the frame height (``text_scale``, formula below), and what
  is laid over a still shown whole is moved out of the picture's busy region (``busy``).

All three formats are 1080 px wide, so only the vertical positions change.

Text scale. ``text_scale = min(1, out_height / design_height * k)`` with the design frame being
the storyboard's own frame (1080x1920 by default) and ``k = (1 - max(0, aspect_out - 0.8)) /
(1 - max(0, aspect_design - 0.8))`` where ``aspect = width / height``. 9:16 keeps its size
(1), 4:5 shrinks with the frame height (0.703: the same share of the frame height as in 9:16)
and 1:1 shrinks to 0.8 of that share (0.45). A main text never goes under 38 px (at 1080 wide)
unless its own size was already smaller; lines are wrapped again by the renderer after the
scaling (``max_width`` is a width, so it is kept).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, NamedTuple

from adanvil.safezones import SAFE_ZONES


class FormatSpec(NamedTuple):
    ratio: str
    width: int
    height: int
    suffix: str
    placement: tuple[str, ...]  # platforms whose safe zones apply (see ``safezones``)


FORMATS: dict[str, FormatSpec] = {
    "9:16": FormatSpec("9:16", 1080, 1920, "9x16", ("tiktok", "reels", "stories")),
    "4:5": FormatSpec("4:5", 1080, 1350, "4x5", ("feed",)),
    "1:1": FormatSpec("1:1", 1080, 1080, "1x1", ("feed",)),
}

# The free band of each placement, in the pixels of a 1080x1920 frame: below the highest top
# bar, above the lowest bottom band (9:16 uses the Stories and Reels boxes, the usual caption
# area; TikTok's taller description band is a warning of ``ad_score``, not a layout rule).
_BANDS: dict[str, tuple[int, int]] = {
    "9:16": (
        max(SAFE_ZONES[name]["top_bar"][3] for name in ("reels", "stories")),
        min(SAFE_ZONES[name]["bottom_band"][1] for name in ("reels", "stories")),
    ),
    "4:5": (SAFE_ZONES["feed"]["top_bar"][3], SAFE_ZONES["feed"]["bottom_band"][1]),
    "1:1": (SAFE_ZONES["feed"]["top_bar"][3], SAFE_ZONES["feed"]["bottom_band"][1]),
}
_REFERENCE_HEIGHT = 1920
MIN_TEXT_SIZE = 38  # the smallest main text, in px at 1080 wide
_TEXT_TYPES = {"caption", "comment", "voice_captions", "speech_captions"}
_DEFAULT_TEXT_SIZE = {"voice_captions": 58, "speech_captions": 58}
_STICKER_SIZE = {"badge": 44, "emoji": 120}


def check_ratios(values: list[str] | None) -> list[str]:
    """``values`` validated: a non-empty list of known ratios without repeats."""

    if values is None:
        return []
    chosen = [str(value) for value in values]
    unknown = [value for value in chosen if value not in FORMATS]
    if unknown or not chosen or len(set(chosen)) != len(chosen):
        raise ValueError(
            f"formats must be a non-empty list of distinct ratios from {', '.join(FORMATS)} "
            f"(got {values})"
        )
    return chosen


def ratio_of(width: int, height: int) -> str | None:
    """The known ratio of a frame (``"4:5"``), or ``None`` for any other shape."""

    if width <= 0 or height <= 0:
        return None
    for name, spec in FORMATS.items():
        if abs(width / height - spec.width / spec.height) < 0.01:
            return name
    return None


def format_name(name: str, ratio: str) -> str:
    """The storyboard (and file) name of ``name`` in ``ratio``: ``<name>_9x16``."""

    return f"{name}_{FORMATS[ratio].suffix}"


def _band(ratio: str | None, height: int) -> tuple[float, float]:
    """The free band of a frame, in its own pixels (the whole frame when the ratio is unknown)."""

    if ratio is None:
        return 0.0, float(height)
    top, bottom = _BANDS[ratio]
    scale = height / _REFERENCE_HEIGHT
    return top * scale, bottom * scale


def _squeeze(width: float, height: float) -> float:
    """The ``k`` of the text scale for a frame: 1 up to 4:5, 0.8 for a square."""

    return 1.0 - max(0.0, width / height - 0.8)


def text_scale(
    out_size: tuple[float, float], design_size: tuple[float, float] = (1080, 1920)
) -> float:
    """The factor texts and graphics are scaled by in an ``out_size`` frame (see the module
    docstring): ``min(1, out_height / design_height * k)``."""

    out_width, out_height = out_size
    design_width, design_height = design_size
    ratio = out_height / design_height * _squeeze(out_width, out_height)
    return min(1.0, ratio / _squeeze(design_width, design_height))


class _Mapper:
    """Vertical positions of the source frame, expressed on the target frame."""

    def __init__(self, source: tuple[int, int], target: FormatSpec) -> None:
        width, height = source
        self.kx = target.width / width  # the horizontal and size scale (1.0 for 1080 wide)
        self.source_height = height * self.kx  # the source as if it were 1080 wide
        self.target = target
        self.s_top, self.s_bottom = _band(ratio_of(width, height), round(self.source_height))
        self.t_top, self.t_bottom = _band(target.ratio, target.height)
        if self.s_bottom <= self.s_top:
            self.s_top, self.s_bottom = 0.0, self.source_height
        self.text = text_scale((target.width, target.height), (target.width, self.source_height))

    def y(self, value: float) -> int:
        """A source y (source pixels) on the target frame."""

        y = value * self.kx
        if y <= self.s_top:
            scale = self.t_top / self.s_top if self.s_top else 1.0
            out = y * scale
        elif y >= self.s_bottom:
            below = self.source_height - self.s_bottom
            scale = (self.target.height - self.t_bottom) / below if below else 1.0
            out = self.target.height - (self.source_height - y) * scale
        else:
            middle = (y - self.s_top) / (self.s_bottom - self.s_top)
            out = self.t_top + middle * (self.t_bottom - self.t_top)
        return round(out)

    def x(self, value: float) -> int:
        return round(value * self.kx)

    def size(self, value: float) -> int:
        return max(1, round(value * self.kx))

    def frame_y(self, value: float) -> int:
        """A y inside the picture (a zoom centre): plain proportion, no band."""

        return round(value * self.kx * self.target.height / self.source_height)


# The ``y`` the schema gives a text overlay that sets none (it must follow the layout too).
_DEFAULT_Y = {"caption": 330, "comment": 330, "voice_captions": 1300, "speech_captions": 1300}


_Window = tuple[float, float, str, "tuple[int, int, int, int] | None"]


def _cover_is_whole(segment: dict[str, Any], document: dict[str, Any], root: Path) -> bool:
    """A ``cover`` still whose shape is the frame's own: it is shown whole all the same."""

    from PIL import Image

    from adanvil.project import resolve_in_project

    try:
        fmt = document.get("format") or {}
        with Image.open(resolve_in_project(root, str(segment.get("source")))) as picture:
            width, height = picture.size
        crop = segment.get("crop")
        if isinstance(crop, dict) and "w" in crop and "h" in crop:
            width, height = int(crop["w"]), int(crop["h"])
        frame = int(fmt.get("width", 1080)) / int(fmt.get("height", 1920))
        return abs(width / height - frame) < 0.02
    except Exception:
        return False


def _whole_picture_windows(
    document: dict[str, Any], root: Path | None = None
) -> list[_Window]:
    """The time spans of the ``contain`` stills (an end card): such a picture is scaled as a
    whole to the new frame, so what is laid on it keeps its place relative to the picture."""

    from adanvil.variants import VariantError, durations

    segments = list(document.get("segments") or [])
    try:
        spans = durations(segments)
    except VariantError:
        return []
    windows: list[_Window] = []
    start = 0.0
    for segment, span in zip(segments, spans, strict=True):
        if (
            isinstance(segment, dict)
            and segment.get("type") == "image"
            and (
                segment.get("fit", "contain") == "contain"
                or (root is not None and _cover_is_whole(segment, document, root))
            )
        ):
            crop = segment.get("crop")
            box = (
                (int(crop["x"]), int(crop["y"]), int(crop["w"]), int(crop["h"]))
                if isinstance(crop, dict) and all(k in crop for k in "xywh")
                else None
            )
            windows.append((start, start + span, str(segment.get("source", "")), box))
        start += span
    return windows


def _base_text_size(overlay: dict[str, Any], brand: Any) -> int:
    """The font size a text overlay really uses: its own, else the brand's, else the style's."""

    if isinstance(overlay.get("size"), (int, float)):
        return int(overlay["size"])
    kind = str(overlay.get("type"))
    if kind in _DEFAULT_TEXT_SIZE:
        return _DEFAULT_TEXT_SIZE[kind]
    style = "comment" if kind == "comment" else str(overlay.get("style", "pill"))
    try:
        override = brand.visual.styles.get(style) if brand is not None else None
        if override is not None and override.size is not None:
            return int(override.size)
    except AttributeError:
        pass
    from adanvil.text.styles import default_size

    try:
        return default_size(style)
    except KeyError:
        return 58


def _sticker_width(overlay: dict[str, Any]) -> int | None:
    """The drawn width of a badge, emoji, underline or image sticker placed by its left edge
    (``x`` a number); ``None`` for the kinds that are placed by a tip or a centre."""

    if overlay.get("kind") not in {"badge", "emoji", "underline", "image"}:
        return None
    if not isinstance(overlay.get("x"), (int, float)):
        return None
    if overlay.get("kind") in {"underline", "image"}:
        width = overlay.get("width")
        return int(width) if isinstance(width, (int, float)) else None
    try:
        from adanvil.render.stickers import sticker_art
        from adanvil.storyboard.schema import StickerOverlay

        probe = StickerOverlay.model_validate(
            {**{k: v for k, v in overlay.items() if not k.startswith("_")}, "x": 0, "y": 0}
        )
        art = sticker_art(probe)
        wide = art.image.width
        art.image.close()
        return int(wide)
    except Exception:
        return None


def _scale_graphics(overlay: dict[str, Any], mapper: _Mapper) -> None:
    """Text, stickers and cards shrink with the frame height (``text_scale``)."""

    factor = mapper.text
    kind = overlay.get("type")
    if factor >= 0.999:
        return
    if kind in _TEXT_TYPES:
        base = round(overlay["_base_size"] * mapper.kx)
        floor = min(base, round(MIN_TEXT_SIZE * mapper.kx))
        size = max(floor, round(base * factor))
        if size != base:
            overlay["size"] = size
    elif kind == "sticker":
        sticker = str(overlay.get("kind"))
        before = _sticker_width(overlay)
        if sticker in _STICKER_SIZE:
            base = int(overlay.get("size") or _STICKER_SIZE[sticker])
            overlay["size"] = max(1, round(base * factor))
        for key in ("width", "height", "length"):
            if isinstance(overlay.get(key), (int, float)):
                overlay[key] = max(1, round(overlay[key] * factor))
        if isinstance(overlay.get("stroke"), (int, float)):
            overlay["stroke"] = max(2, round(overlay["stroke"] * factor))
        after = _sticker_width(overlay)
        if before is not None and after is not None and isinstance(overlay.get("x"), (int, float)):
            # a sticker placed by its left edge shrinks around its centre, not its corner
            overlay["x"] = int(overlay["x"]) + round((before - after) / 2)
    elif kind == "image" and isinstance(overlay.get("width"), (int, float)):
        framed = any(overlay.get(key) for key in ("frame", "rotate")) or (
            overlay.get("animation") not in (None, "none")
        )
        if framed:  # a card; a plain picture is the product itself
            width = int(overlay["width"])
            narrower = max(1, round(width * factor))
            if isinstance(overlay.get("x"), (int, float)):
                overlay["x"] = int(overlay["x"]) + (width - narrower) // 2
            overlay["width"] = narrower


def _scale_overlay(
    overlay: dict[str, Any],
    mapper: _Mapper,
    *,
    proportional: bool = False,
    brand: Any = None,
) -> None:
    kind = overlay.get("type")
    kx = mapper.kx
    if kind in _TEXT_TYPES:
        overlay["_base_size"] = _base_text_size(overlay, brand)

    def move(key: str) -> None:
        value = overlay.get(key)
        if value is None and key == "y":
            value = _DEFAULT_Y.get(str(kind))  # the position the schema gives when none is set
        if isinstance(value, (int, float)):
            overlay[key] = mapper.frame_y(value) if proportional else mapper.y(value)

    def across(key: str) -> None:
        if kx != 1.0 and isinstance(overlay.get(key), (int, float)):
            overlay[key] = mapper.x(overlay[key])

    def sized(*keys: str) -> None:
        if kx != 1.0:
            for key in keys:
                if isinstance(overlay.get(key), (int, float)):
                    overlay[key] = mapper.size(overlay[key])

    if kind in {"caption", "comment", "voice_captions", "speech_captions"}:
        move("y")
        across("x_center")
        sized("size", "max_width")
    elif kind in {"image", "video"}:
        move("y")
        across("x")
        sized("width", "height")
    elif kind == "sticker":
        move("y")
        across("x")
        for key in ("from", "to"):
            point = overlay.get(key)
            if isinstance(point, (list, tuple)) and len(point) == 2:
                rise = mapper.frame_y(point[1]) if proportional else mapper.y(point[1])
                overlay[key] = [mapper.x(point[0]), rise]
        sized("length", "width", "height", "stroke", "size")
    _scale_graphics(overlay, mapper)
    overlay.pop("_base_size", None)


def _scale_segment(segment: dict[str, Any], mapper: _Mapper) -> None:
    freeze = segment.get("freeze")
    if isinstance(freeze, dict) and isinstance(freeze.get("center"), (list, tuple)):
        cx, cy = freeze["center"]
        freeze["center"] = [round(cx * mapper.kx), mapper.frame_y(cy)]
    for punch in segment.get("punch") or []:
        if isinstance(punch, dict) and isinstance(punch.get("center"), (list, tuple)):
            cx, cy = punch["center"]
            punch["center"] = [round(cx * mapper.kx), mapper.frame_y(cy)]


def reformat(
    document: dict[str, Any],
    ratio: str,
    *,
    root: Path | None = None,
    output_dir: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """A copy of the storyboard ``document`` laid out for ``ratio``, and the notes about it.

    The copy is named ``<name>_<suffix>`` (``..._4x5``) and renders to
    ``renders/<name>_<suffix>.mp4`` (or ``<output_dir>/...``). With ``root`` (the project) the
    texts, stickers and images are measured and pushed back inside the free band of the new
    placement when they would leave it. ``formats`` is dropped from the copy.
    """

    if ratio not in FORMATS:
        raise ValueError(f"unknown format {ratio!r}: use one of {', '.join(FORMATS)}")
    spec = FORMATS[ratio]
    result = copy.deepcopy(document)
    old = result.get("format") if isinstance(result.get("format"), dict) else {}
    width = int(old.get("width", 1080))
    height = int(old.get("height", 1920))
    fps = int(old.get("fps", 30))
    notes: list[str] = []
    result.pop("formats", None)
    result["format"] = {"width": spec.width, "height": spec.height, "fps": fps}
    name = str(result.get("name", "storyboard"))
    renamed = format_name(name, ratio)
    result["name"] = renamed
    output = dict(result["output"]) if isinstance(result.get("output"), dict) else {}
    folder = (output_dir or "renders").strip("/")
    output["path"] = f"{folder}/{renamed}.mp4"
    result["output"] = output

    if (width, height) != (spec.width, spec.height):
        mapper = _Mapper((width, height), spec)
        for number, segment in enumerate(result.get("segments") or []):
            if not isinstance(segment, dict):
                continue
            _scale_segment(segment, mapper)
            label = str(segment.get("label") or "").casefold()
            if (
                segment.get("type") == "image"
                and segment.get("fit") == "cover"
                and ("end_card" in label or "endcard" in label or label == "cta")
            ):
                # a call-to-action card is read whole: never cropped to fit another ratio
                segment["fit"] = "contain"
                notes.append(f"segment {number} ({label}) shown whole (contain + blur)")
        whole = _whole_picture_windows(result)
        brand = _load_brand(Path(root)) if root is not None else None
        for overlay in result.get("overlays") or []:
            if isinstance(overlay, dict):
                start = overlay.get("start")
                under = isinstance(start, (int, float)) and any(
                    a - 1e-6 <= start < b for a, b, _source, _crop in whole
                )
                _scale_overlay(overlay, mapper, proportional=under, brand=brand)
    resized = (width, height) != (spec.width, spec.height)
    protected = _override_y(result, ratio)
    if root is not None and resized:
        notes.extend(_keep_inside(result, spec, Path(root)))
        notes.extend(_avoid_card_content(result, spec, Path(root)))
        looks: dict[int, list[Any] | None] = {}
        notes.extend(_avoid_faces(result, spec, Path(root), protected, looks))
        notes.extend(_apply_overrides(result, ratio))
        notes.extend(final_face_notes(result, spec, Path(root), protected, looks))
        return result, notes
    notes.extend(_apply_overrides(result, ratio))
    return result, notes


def _override_y(document: dict[str, Any], ratio: str) -> set[int]:
    """The overlays that carry an explicit ``y`` for ``ratio`` (the automatic placement leaves
    them alone)."""

    found: set[int] = set()
    for index, overlay in enumerate(document.get("overlays") or []):
        mine = overlay.get("formats") if isinstance(overlay, dict) else None
        if isinstance(mine, dict) and isinstance(mine.get(ratio), dict) and (
            "y" in mine[ratio] or mine[ratio].get("hidden")
        ):
            found.add(index)
    return found


def _apply_overrides(document: dict[str, Any], ratio: str) -> list[str]:
    """Apply each overlay's ``formats[ratio]`` (x, y, size, anchor, hidden) and drop the
    ``formats`` key of every overlay: the copy is for ONE format."""

    notes: list[str] = []
    kept: list[Any] = []
    for index, overlay in enumerate(document.get("overlays") or []):
        if not isinstance(overlay, dict):
            kept.append(overlay)
            continue
        mine = overlay.pop("formats", None)
        override = mine.get(ratio) if isinstance(mine, dict) else None
        if not isinstance(override, dict) or not override:
            kept.append(overlay)
            continue
        if override.get("hidden"):
            notes.append(f"overlay {index} ({overlay.get('type')}) hidden in {ratio}")
            continue
        applied = []
        for key, value in override.items():
            if value is None or key == "hidden":
                continue
            if key == "x":
                kind = overlay.get("type")
                field = "x_center" if kind in _TEXT_TYPES else "x"
                overlay[field] = value
                applied.append(f"{field}={value}")
                if kind == "sticker" and value == "center":
                    overlay.pop("anchor", None)
            elif key in {"y", "size", "anchor"}:
                overlay[key] = value
                applied.append(f"{key}={value}")
        if applied:
            label = f"overlay {index} ({overlay.get('type')}) in {ratio}"
            notes.append(f"{label}: " + ", ".join(applied))
        kept.append(overlay)
    if "overlays" in document:
        document["overlays"] = kept
    return notes


# --------------------------------------------------------------------------- keep inside the band


def _load_brand(root: Path) -> Any:
    from adanvil.brand import load_brand

    brand_file = root / "brand.yaml"
    if brand_file.is_file():
        try:
            return load_brand(brand_file)
        except Exception:
            return None
    return None


def _boxes(model: Any, root: Path) -> dict[int, tuple[float, float, float, float]]:
    """The pixel box of every text, sticker and image overlay of ``model`` that can be measured."""

    from adanvil.brand import load_brand
    from adanvil.storyboard.schema import (
        CaptionOverlay,
        CommentOverlay,
        ImageOverlay,
        StickerOverlay,
    )
    from adanvil.storyboard.validate import _render_text_overlay

    brand = None
    brand_file = root / "brand.yaml"
    if brand_file.is_file():
        try:
            brand = load_brand(brand_file)
        except Exception:
            brand = None
    found: dict[int, tuple[float, float, float, float]] = {}
    for index, overlay in enumerate(model.overlays):
        box: tuple[float, float, float, float] | None = None
        try:
            if isinstance(overlay, (CaptionOverlay, CommentOverlay)):
                measured, _ = _render_text_overlay(
                    overlay,
                    root=root,
                    width=model.format.width,
                    height=model.format.height,
                    brand=brand,
                )
                if measured != (0, 0, 0, 0):
                    box = tuple(float(value) for value in measured)  # type: ignore[assignment]
            elif isinstance(overlay, StickerOverlay) and overlay.kind in {
                "badge",
                "emoji",
                "image",
            }:
                from adanvil.render.stickers import sticker_art

                art = sticker_art(overlay, root=root)
                box = tuple(float(value) for value in art.bbox)  # type: ignore[assignment]
                art.image.close()
            elif isinstance(overlay, ImageOverlay):
                from PIL import Image

                from adanvil.project import resolve_in_project

                with Image.open(resolve_in_project(root, overlay.source)) as picture:
                    across = overlay.width or picture.width
                    down = round(across * picture.height / max(picture.width, 1))
                box = (
                    float(overlay.x),
                    float(overlay.y),
                    float(overlay.x + across),
                    float(overlay.y + down),
                )
        except Exception:
            box = None
        if box is not None:
            found[index] = box
    return found


def _shift_overlay(overlay: dict[str, Any], moved: int) -> bool:
    """Move a text, sticker or image overlay down by ``moved`` px; False when it has no y."""

    if isinstance(overlay.get("y"), (int, float)):
        overlay["y"] = int(overlay["y"]) + moved
    elif isinstance(overlay.get("from"), list) and isinstance(overlay.get("to"), list):
        for key in ("from", "to"):
            overlay[key] = [overlay[key][0], overlay[key][1] + moved]
    else:
        return False
    return True


def card_content_clashes(
    document: dict[str, Any], root: Path
) -> list[tuple[int, tuple[float, float, float, float], list[Any], dict[str, Any]]]:
    """The captions and comments that cover the lettering or buttons of a still shown whole.

    Each item is ``(overlay index, overlay box, busy rectangles, picture)`` where ``picture``
    has the ``source`` and the ``rect`` the still takes on the frame. Empty when the storyboard
    has no such still or no overlay can be measured."""

    from adanvil.busy import overlaps, picture_busy
    from adanvil.project import resolve_in_project
    from adanvil.storyboard.schema import CaptionOverlay, CommentOverlay, Storyboard

    windows = _whole_picture_windows(document, root)
    if not windows:
        return []
    try:
        model = Storyboard.model_validate(document)
    except Exception:
        return []
    fmt = document.get("format") or {}
    frame = (int(fmt.get("width", 1080)), int(fmt.get("height", 1920)))
    found = []
    for index, box in _boxes(model, root).items():
        if not isinstance(model.overlays[index], (CaptionOverlay, CommentOverlay)):
            continue
        start = document["overlays"][index].get("start", 0.0)
        window = next((w for w in windows if w[0] - 1e-6 <= float(start) < w[1]), None)
        if window is None:
            continue
        try:
            rect, busy = picture_busy(resolve_in_project(root, window[2]), frame, crop=window[3])
        except Exception:
            continue
        if overlaps(box, busy):
            found.append((index, box, busy, {"source": window[2], "rect": rect}))
    return found


def _avoid_card_content(document: dict[str, Any], spec: FormatSpec, root: Path) -> list[str]:
    """Move each caption that covers a still's content into the largest free band; shrink it
    down to the readable minimum when no band is tall enough; warn when nothing fits."""

    from adanvil.busy import free_bands, overlaps
    from adanvil.storyboard.schema import Storyboard

    top, bottom = _band(spec.ratio, spec.height)
    notes: list[str] = []
    for index, box, busy, picture in card_content_clashes(document, root):
        overlay = document["overlays"][index]
        bands = free_bands(busy, top, bottom)
        brand = _load_brand(root)
        base = _base_text_size(overlay, brand)
        floor = min(base, MIN_TEXT_SIZE)
        placed = False
        size = base
        while True:
            height = box[3] - box[1]
            fit = next((band for band in bands if band[1] - band[0] >= height + 8), None)
            if fit is not None:
                shift = round((fit[0] + fit[1]) / 2 - (box[1] + box[3]) / 2)
                if size != base:
                    overlay["size"] = size
                _shift_overlay(overlay, shift)
                placed = True
                note = f"overlay {index} ({overlay.get('type')}) moved {shift:+d}px"
                note += f" out of the content of {picture['source']}"
                notes.append(note + (f", size {size}px" if size != base else ""))
                break
            if size <= floor:
                break
            size = max(floor, round(size * 0.9))
            trial = {**overlay, "size": size}
            try:
                measured = _boxes(
                    Storyboard.model_validate({**document, "overlays": [trial]}), root
                ).get(0)
            except Exception:
                measured = None
            if measured is None:
                break
            # keep the box where it was (centre), at the new size
            centre = (box[1] + box[3]) / 2
            box = (measured[0], centre - (measured[3] - measured[1]) / 2, measured[2],
                   centre + (measured[3] - measured[1]) / 2)
        if not placed:
            overlay["size"] = size if size != base else overlay.get("size")
            if overlay.get("size") is None:
                overlay.pop("size", None)
            widest = bands[0] if bands else (top, bottom)
            shift = round((widest[0] + widest[1]) / 2 - (box[1] + box[3]) / 2)
            _shift_overlay(overlay, shift)
            moved = Storyboard.model_validate(document)
            still = overlaps(_boxes(moved, root).get(index, box), busy)
            notes.append(
                f"caption_over_card_content: overlay {index} ({overlay.get('type')}) cannot "
                f"avoid the content of {picture['source']} in {spec.ratio}"
                + (" (still overlapping)" if still else "")
            )
    return notes


def _overlay_range(overlay: dict[str, Any], total: float) -> tuple[float, float]:
    start = overlay.get("start")
    start = float(start) if isinstance(start, (int, float)) else 0.0
    end = overlay.get("end")
    end = float(end) if isinstance(end, (int, float)) else total
    return start, max(start, min(end, total))


def caption_faces(
    document: dict[str, Any], root: Path, overlay: dict[str, Any], total: float
) -> list[Any] | None:
    """The face cores (pixels of the document's frame) under a caption at EVERY sampled moment
    of its time range (``adanvil.captionfaces.sample_moments``: every 0.2 s), as a list of
    ``(moment, cores)``; ``None`` when the footage under it has no face data and none can be
    made."""

    from adanvil.captionfaces import frame_faces, sample_moments

    start, end = _overlay_range(overlay, total)
    found: list[Any] = []
    for moment in sample_moments(start, end, total):
        faces = frame_faces(document, root, moment)
        if faces is None:
            return None
        found.append((moment, faces))
    return found


GAP = 6.0  # pixels kept between a caption and what it avoids
OTHER_SIDE = 1500.0  # extra cost (px) of a spot on the other side of the face than the caption's


def _still_busy(
    document: dict[str, Any], root: Path, overlay: dict[str, Any], windows: list[Any]
) -> list[Any]:
    """The busy rectangles (lettering, buttons, product shots) of the still shown whole under
    the start of ``overlay``; empty when it is not over one."""

    from adanvil.busy import picture_busy
    from adanvil.project import resolve_in_project

    start = overlay.get("start")
    start = float(start) if isinstance(start, (int, float)) else 0.0
    window = next((w for w in windows if w[0] - 1e-6 <= start < w[1]), None)
    if window is None:
        return []
    fmt = document.get("format") or {}
    frame = (int(fmt.get("width", 1080)), int(fmt.get("height", 1920)))
    try:
        return list(picture_busy(resolve_in_project(root, window[2]), frame, crop=window[3])[1])
    except Exception:
        return []


def _free_spots(
    box: tuple[float, float, float, float],
    obstacles: list[Any],
    top: float,
    bottom: float,
    reference: float,
) -> list[float]:
    """The new top edges of a caption ``box`` that cover none of the ``obstacles``, inside
    ``top``..``bottom``, the nearest first (moved up or down the least).

    A spot on the other side of ``reference`` (the middle of the faces) than the caption now is
    costs ``OTHER_SIDE`` more: a hook at the top stays at the top while a free spot exists
    there. The candidates are the places flush against an obstacle, the band limits and where
    the caption already is, which contain the nearest free spot of any layout."""

    left, upper, right, lower = box
    height = lower - upper
    lowest = bottom - height
    if lowest < top:
        return []
    across = [o for o in obstacles if o[0] < right and o[2] > left]
    candidates = {min(max(upper, top), lowest), top, lowest}
    for _l, o_top, _r, o_bottom in across:
        candidates.add(min(max(o_top - height - GAP, top), lowest))
        candidates.add(min(max(o_bottom + GAP, top), lowest))
    start_above = (upper + lower) / 2 < reference
    ranked: list[tuple[float, float]] = []
    for y in candidates:
        if any(y < o[3] and y + height > o[1] for o in across):
            continue
        cost = abs(y - upper)
        if ((y + height / 2) < reference) != start_above:
            cost += OTHER_SIDE
        ranked.append((cost, y))
    return [y for _cost, y in sorted(ranked)]


def _nearest_free_spot(
    box: tuple[float, float, float, float],
    obstacles: list[Any],
    top: float,
    bottom: float,
    reference: float,
) -> float | None:
    """The first of ``_free_spots``, ``None`` when no spot is free."""

    spots = _free_spots(box, obstacles, top, bottom, reference)
    return spots[0] if spots else None


def _measured_box(
    document: dict[str, Any], root: Path, overlay: dict[str, Any]
) -> tuple[float, float, float, float] | None:
    """The pixel box of ONE overlay as it is now (``overlay`` may differ from the document's)."""

    from adanvil.storyboard.schema import Storyboard

    try:
        return _boxes(Storyboard.model_validate({**document, "overlays": [overlay]}), root).get(0)
    except Exception:
        return None


MAX_SPOTS = 8  # free spots tried, nearest first, before the caption shrinks


def _avoid_faces(
    document: dict[str, Any],
    spec: FormatSpec,
    root: Path,
    protected: set[int],
    samples: dict[int, list[Any] | None] | None = None,
) -> list[str]:
    """Keep each caption or comment off the faces, moving it as little as possible.

    A caption stays where the re-layout put it when that spot is free: it does not cover a face
    core (``adanvil.captionfaces``) at any sampled moment of its time range. Otherwise the free
    spots are tried nearest first, on the same side of the face if there is one (a hook at the
    top stays above the head), where it covers no face, none of the content of a still shown
    whole and, if possible, none of the other overlays on screen at the same time, inside the
    free part of the frame. A spot is only taken after it is CHECKED: the caption is moved there,
    measured, and ``caption_on_face`` is run on that final position against the faces of every
    sampled moment; a spot that fails is dropped for the next one (other side, then a smaller
    size down to the readable minimum). A caption that fits nowhere stays where it is: the final
    check (``final_face_notes``) says so. A move is noted only once it passed the check."""

    from adanvil.captionfaces import caption_on_face
    from adanvil.storyboard.schema import Storyboard

    try:
        model = Storyboard.model_validate(document)
    except Exception:
        return []
    cache = samples if samples is not None else {}
    top, bottom = _band(spec.ratio, spec.height)
    total = model.total_duration()
    boxes = _boxes(model, root)
    brand = _load_brand(root)
    windows = _whole_picture_windows(document, root)
    notes: list[str] = []
    skipped = False

    def covered(where: tuple[float, float, float, float], looks: list[Any]) -> bool:
        return any(caption_on_face(where, found) for _moment, found in looks)

    for index in sorted(boxes):
        box = boxes[index]
        overlay = document["overlays"][index]
        if overlay.get("type") not in {"caption", "comment"} or index in protected:
            continue
        if index not in cache:
            cache[index] = caption_faces(document, root, overlay, total)
        looks = cache[index]
        if looks is None:
            skipped = True
            continue
        faces = [face for _moment, found in looks for face in found]
        if not faces or not covered(box, looks):
            continue  # nothing there, or the spot is free: it stays
        start, end = _overlay_range(overlay, total)
        hard = list(faces) + _still_busy(document, root, overlay, windows)
        soft = []
        for other, other_box in boxes.items():
            if other == index:
                continue
            other_start, other_end = _overlay_range(document["overlays"][other], total)
            if other_start < end - 1e-6 and other_end > start + 1e-6:
                a, b, c, d = other_box
                soft.append((a - GAP, b - GAP, c + GAP, d + GAP))
        reference = (min(f[1] for f in faces) + max(f[3] for f in faces)) / 2
        base = _base_text_size(overlay, brand)
        floor = min(base, MIN_TEXT_SIZE)
        size = base
        placed = False
        while True:
            centre = (box[1] + box[3]) / 2
            spots = _free_spots(box, hard + soft, top, bottom, reference)
            spots += [y for y in _free_spots(box, hard, top, bottom, reference) if y not in spots]
            for spot in spots[:MAX_SPOTS]:
                shift = round(spot - box[1])
                trial = copy.deepcopy(overlay)
                if size != base:
                    trial["size"] = size
                if not _shift_overlay(trial, shift):
                    continue
                measured = _measured_box(document, root, trial)
                if measured is None or covered(measured, looks):
                    continue  # the check on the final position rejects this spot
                overlay.clear()
                overlay.update(trial)
                placed = True
                same = ((centre + shift) < reference) == (centre < reference)
                boxes[index] = measured
                notes.append(
                    f"overlay {index} ({overlay.get('type')}) moved {shift:+d}px off a face"
                    f" ({'same side' if same else 'other side'} of it)"
                    + (f", size {size}px" if size != base else "")
                )
                break
            if placed or size <= floor:
                break
            size = max(floor, round(size * 0.9))
            measured = _measured_box(document, root, {**overlay, "size": size})
            if measured is None:
                break
            half = (measured[3] - measured[1]) / 2
            box = (measured[0], centre - half, measured[2], centre + half)
        if not placed:
            _least_covered(
                document, root, overlay, boxes[index], looks, (top, bottom), index, covered, notes
            )
    if skipped:
        notes.append(
            "caption_on_face check skipped for some captions: no face data for the footage "
            "under them (install the vision environment: adanvil vision-setup, or index the "
            "footage with vision)"
        )
    return notes


def _least_covered(
    document: dict[str, Any],
    root: Path,
    overlay: dict[str, Any],
    box: tuple[float, float, float, float],
    looks: list[Any],
    band: tuple[float, float],
    index: int,
    covered: Any,
    notes: list[str],
) -> None:
    """No spot is free of the face at every sampled moment (a head that moves across the whole
    frame while the caption stays): move the caption to the spot where it covers a face at the
    fewest sampled moments, if that is fewer than where it is, and say so. The final check
    (``final_face_notes``) still reports the moments that remain."""

    top, bottom = band
    height = box[3] - box[1]
    lowest = bottom - height
    if lowest < top:
        return
    candidates = {min(max(box[1], top), lowest), top, lowest}
    for _moment, found in looks:
        for _l, f_top, _r, f_bottom in found:
            candidates.add(min(max(f_top - height - GAP, top), lowest))
            candidates.add(min(max(f_bottom + GAP, top), lowest))

    def misses(y: float) -> int:
        shifted = (box[0], y, box[2], y + height)
        return sum(1 for look in looks if covered(shifted, [look]))

    best = min(candidates, key=lambda y: (misses(y), abs(y - box[1])))
    if misses(best) >= misses(box[1]) or abs(best - box[1]) < 1:
        return
    shift = round(best - box[1])
    trial = copy.deepcopy(overlay)
    if not _shift_overlay(trial, shift):
        return
    measured = _measured_box(document, root, trial)
    if measured is None or sum(1 for look in looks if covered(measured, [look])) >= misses(box[1]):
        return
    overlay.clear()
    overlay.update(trial)
    count = sum(1 for look in looks if covered(measured, [look]))
    notes.append(
        f"overlay {index} ({overlay.get('type')}) moved {shift:+d}px: no spot is free of the face "
        f"at every moment, it still covers one at {count} of {len(looks)} looks"
    )


def final_face_notes(
    document: dict[str, Any],
    spec: FormatSpec,
    root: Path,
    protected: set[int],
    samples: dict[int, list[Any] | None] | None = None,
) -> list[str]:
    """The final ``caption_on_face`` check, on the FINAL position of every caption and comment
    (after the layout and the per-format overrides), against the faces of every sampled moment of
    its time range. One note per caption that still covers a face, with the moment: this is the
    verdict that ``qc_run`` repeats on the rendered frames."""

    from adanvil.captionfaces import first_covered
    from adanvil.storyboard.schema import Storyboard

    try:
        model = Storyboard.model_validate(document)
    except Exception:
        return []
    cache = samples if samples is not None else {}
    total = model.total_duration()
    notes: list[str] = []
    for index, box in sorted(_boxes(model, root).items()):
        overlay = document["overlays"][index]
        if overlay.get("type") not in {"caption", "comment"}:
            continue
        if index not in cache:
            cache[index] = caption_faces(document, root, overlay, total)
        looks = cache[index]
        if looks is None:
            continue
        moment = first_covered(box, looks)
        if moment is None:
            continue
        what = f"overlay {index} ({overlay.get('type')})"
        if index in protected:
            notes.append(
                f"caption_on_face: {what} still covers a face at {moment:.1f}s in {spec.ratio} "
                f"after its formats[{spec.ratio!r}].y (source face data)"
            )
        else:
            notes.append(
                f"caption_on_face: {what} cannot avoid a face in {spec.ratio} (it covers one "
                f"at {moment:.1f}s in the source face data): set formats[{spec.ratio!r}].y on it"
            )
    return notes


def _keep_inside(document: dict[str, Any], spec: FormatSpec, root: Path) -> list[str]:
    from adanvil.storyboard.schema import Storyboard

    try:
        model = Storyboard.model_validate(document)
    except Exception:
        return []  # the storyboard has its own problems: validation reports them
    top, bottom = _band(spec.ratio, spec.height)
    notes: list[str] = []
    for index, (_left, upper, _right, lower) in _boxes(model, root).items():
        overlay = document["overlays"][index]
        shift = 0.0
        if lower - upper > bottom - top:
            continue  # taller than the band: nothing to do but say so in validation
        if upper < top:
            shift = top - upper
        elif lower > bottom:
            shift = bottom - lower
        if shift:
            moved = round(shift)
            if isinstance(overlay.get("y"), (int, float)):
                overlay["y"] = int(overlay["y"]) + moved
            elif isinstance(overlay.get("from"), list) and isinstance(overlay.get("to"), list):
                for key in ("from", "to"):
                    overlay[key] = [overlay[key][0], overlay[key][1] + moved]
            else:
                continue
            notes.append(
                f"overlay {index} ({overlay.get('type')}) moved {moved:+d}px to stay in "
                f"the {spec.ratio} free band ({round(top)} to {round(bottom)})"
            )
    return notes


def requested_formats(document: dict[str, Any], formats: list[str] | None) -> list[str]:
    """The ratios to render: ``formats`` if given, else the storyboard's own ``formats``."""

    if formats is not None:
        return check_ratios(formats)
    own = document.get("formats")
    return check_ratios(list(own)) if own else []


__all__ = [
    "FORMATS",
    "FormatSpec",
    "check_ratios",
    "format_name",
    "ratio_of",
    "reformat",
    "requested_formats",
]
