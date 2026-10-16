"""The variant matrix: one base storyboard, a few hooks x bodies x calls to action.

``expand_matrix`` turns a base storyboard and up to three axes of options into the cartesian
product of ready storyboards (at most 24). An option is a list of changes applied in order, each
either a ``storyboard_patch`` change (``{path, value}`` or ``{path, remove: true}``) or a **slot
replacement**:

    {"replace_label": "hook", "segments": [...], "overlays": [...]}

which swaps every segment labelled ``hook`` (they must be next to each other) for ``segments``,
and shifts everything after the slot by the difference of duration. Overlays that were timed
inside the slot go away; the new ``overlays`` are timed from the START OF THE SLOT (0 is the first
frame of the new segments). Overlays that start after the slot (and the ones that span it) follow
the shift, word times included. An overlay that only partly overlaps the slot is refused (split
it), and so is a change of duration while a recorded voice-over (``audio.voice``) is laid on the
timeline, because the voice would no longer match the picture.

Every overlap test uses the project tolerance (``1e-3`` s, the one of the storyboard checks), so
a range built from summed float durations still counts as inside the slot. Timed sounds follow
the slot like overlays: an ``sfx`` overlay after the slot shifts by the difference of duration;
an ``sfx`` inside the old slot goes away when the new slot brings its own sounds, else it keeps
its offset from the start of the slot (dropped when the new slot is too short for it). The
build lists what moved (``Variant.moved``). An option with no ``changes`` (or ``[]``) is the
control: the base unchanged.
"""

from __future__ import annotations

import copy
import io
import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from adanvil.media import segment_duration
from adanvil.safeio import replace_bytes
from adanvil.storyboard.patch import apply_changes, rename_storyboard, storyboard_name
from adanvil.storyboard.validate import _TOLERANCE as _EPS  # the shared 1e-3 s tolerance

MAX_VARIANTS = 24
MAX_AXES = 3
_ID = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")
_VOICE_TOLERANCE = 0.02
DEFAULT_PATTERN = "{base}_{hook}_{cta}"


class VariantError(ValueError):
    """A matrix that cannot be built (the message says which variant and why).

    ``errors`` lists every problem found as ``{"variant", "path", "message"}``; ``path`` is the
    place in the storyboard (or in the matrix) when it is known.
    """

    def __init__(
        self,
        message: str,
        *,
        path: str | None = None,
        errors: list[dict[str, str]] | None = None,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.errors = errors or []


@dataclass
class Variant:
    name: str
    ids: dict[str, str]
    document: dict[str, Any]
    duration: float
    warnings: list[str] = field(default_factory=list)
    moved: list[str] = field(default_factory=list)


def _segment_model():
    from adanvil.storyboard.schema import Segment

    return TypeAdapter(Segment)


def durations(segments: list[Any], where: str = "segments") -> list[float]:
    """The duration of each segment of a list of segment objects."""

    adapter = _segment_model()
    result: list[float] = []
    for number, segment in enumerate(segments):
        try:
            result.append(float(segment_duration(adapter.validate_python(segment))))
        except ValidationError as exc:
            first = exc.errors()[0]
            place = ".".join(str(part) for part in first["loc"])
            raise VariantError(
                f"{where}[{number}] is not a valid segment ({place}: {first['msg']})"
            ) from exc
    return result


def total_duration(document: dict[str, Any]) -> float:
    return sum(durations(list(document.get("segments") or [])))


def _voice_end(document: dict[str, Any], root: Path | None) -> float | None:
    """When the recorded voice of the storyboard ends on the timeline, ``None`` if unknown."""

    audio = document.get("audio") or {}
    manifest = audio.get("voice")
    if not manifest or root is None:
        return None
    try:
        from adanvil.project import resolve_in_project
        from adanvil.voice import load_track

        track = load_track(resolve_in_project(root, manifest))
    except Exception:
        return None
    return float(audio.get("voice_offset", 0.0)) + float(track.total_duration)


def _shift_times(overlay: dict[str, Any], delta: float) -> None:
    for key in ("start", "end"):
        if isinstance(overlay.get(key), (int, float)):
            overlay[key] = round(float(overlay[key]) + delta, 6)
    times = overlay.get("word_times")
    if isinstance(times, list):
        overlay["word_times"] = [[round(a + delta, 6), round(b + delta, 6)] for a, b in times]


def replace_slot(
    document: dict[str, Any],
    spec: dict[str, Any],
    root: Path | None = None,
    notes: list[str] | None = None,
) -> tuple[dict[str, Any], float]:
    """``document`` with a slot replaced (see the module docstring), and the duration change.

    ``notes`` (when given) receives one line for every sound that moved, stayed or went away."""

    unknown = sorted(set(spec) - {"replace_label", "segments", "overlays"})
    if unknown:
        raise VariantError(f"slot replacement has unexpected field(s): {', '.join(unknown)}")
    label = spec.get("replace_label")
    segments = spec.get("segments")
    if not isinstance(label, str) or not label:
        raise VariantError("replace_label must be the label of the segments to swap")
    if not isinstance(segments, list) or not segments:
        raise VariantError(f"replacing {label!r}: segments must be a non-empty list")
    new_overlays = spec.get("overlays") or []
    if not isinstance(new_overlays, list) or not all(isinstance(o, dict) for o in new_overlays):
        raise VariantError(f"replacing {label!r}: overlays must be a list of overlay objects")

    result = copy.deepcopy(document)
    old = list(result.get("segments") or [])
    positions = [
        i for i, seg in enumerate(old) if isinstance(seg, dict) and seg.get("label") == label
    ]
    if not positions:
        labels = sorted(
            {str(s.get("label")) for s in old if isinstance(s, dict) and s.get("label")}
        )
        found = ", ".join(labels) or "none"
        raise VariantError(f"no segment is labelled {label!r} (labels in the base: {found})")
    if positions != list(range(positions[0], positions[-1] + 1)):
        raise VariantError(
            f"the segments labelled {label!r} are not next to each other "
            f"(positions {positions}): a slot must be one run of segments"
        )
    spans = durations(old)
    first, last = positions[0], positions[-1]
    start = sum(spans[:first])
    old_length = sum(spans[first : last + 1])
    new_length = sum(durations(segments, f"replacing {label!r}: segments"))
    delta = new_length - old_length
    end = start + old_length

    if abs(delta) > _VOICE_TOLERANCE and (result.get("audio") or {}).get("voice"):
        voice_end = _voice_end(result, root)
        if voice_end is None or start < voice_end - _EPS:
            raise VariantError(
                f"replacing {label!r} changes the duration by {delta:+.2f}s, and the voice-over "
                f"(audio.voice) is laid on the timeline: it would no longer match the picture. "
                f"Give a replacement of {old_length:.2f}s (within {_VOICE_TOLERANCE}s), or "
                "generate a voice for this variant"
            )

    new_has_sfx = any(isinstance(o, dict) and o.get("type") == "sfx" for o in new_overlays)
    new_end = start + new_length
    log = notes if notes is not None else []
    kept: list[dict[str, Any]] = []
    removed_at: int | None = None
    for number, overlay in enumerate(result.get("overlays") or []):
        if not isinstance(overlay, dict):
            kept.append(overlay)
            continue
        if overlay.get("type") == "sfx":
            at = overlay.get("at")
            at = float(at) if isinstance(at, (int, float)) else 0.0
            name = f"overlay {number} (sfx {overlay.get('source')})"
            if at >= end - _EPS:
                if delta:
                    overlay["at"] = round(at + delta, 6)
                    log.append(f"{name} moved {at:g}s -> {overlay['at']:g}s")
                kept.append(overlay)
            elif at < start - _EPS:
                kept.append(overlay)
            elif new_has_sfx:
                if removed_at is None:
                    removed_at = len(kept)
                log.append(f"{name} at {at:g}s dropped (the new slot brings its own sounds)")
            elif at - start < new_length - _EPS:
                kept.append(overlay)  # same offset from the start of the slot
                log.append(f"{name} at {at:g}s kept inside the new slot")
            else:
                if removed_at is None:
                    removed_at = len(kept)
                log.append(f"{name} at {at:g}s dropped (the new slot ends at {new_end:g}s)")
            continue
        begin = overlay.get("start")
        begin = float(begin) if isinstance(begin, (int, float)) else 0.0
        stop = overlay.get("end")
        stop = float(stop) if isinstance(stop, (int, float)) else None
        kind = overlay.get("type")
        if stop is None:
            if begin >= end - _EPS and delta:
                _shift_times(overlay, delta)
            kept.append(overlay)
        elif stop <= start + _EPS:
            kept.append(overlay)
        elif begin >= end - _EPS:
            _shift_times(overlay, delta)
            kept.append(overlay)
        elif begin >= start - _EPS and stop <= end + _EPS:
            if removed_at is None:
                removed_at = len(kept)  # the new overlays take the place of the first removed
            continue
        elif begin < start - _EPS and stop >= end - _EPS:
            if delta:
                overlay["end"] = round(stop + delta, 6)
            kept.append(overlay)
        else:
            raise VariantError(
                f"overlay {number} ({kind}, {begin:g}s to {stop:g}s) only partly overlaps the "
                f"slot {label!r} ({start:g}s to {end:g}s): split it so each part is inside or "
                "outside the slot",
                path=f"overlays.{number}",
            )
    added = []
    for overlay in new_overlays:
        item = copy.deepcopy(overlay)
        if item.get("type") == "sfx":
            item["at"] = round(float(item.get("at", 0.0)) + start, 6)
        else:
            if "start" not in item:
                item["start"] = 0.0
            _shift_times(item, start)
        added.append(item)
    position = removed_at if removed_at is not None else len(kept)
    result["overlays"] = [*kept[:position], *added, *kept[position:]]
    result["segments"] = [*old[:first], *copy.deepcopy(segments), *old[last + 1 :]]
    return result, delta


def apply_option(
    document: dict[str, Any],
    changes: Any,
    root: Path | None = None,
    notes: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, bool]]:
    """One option applied to ``document``. Returns the result and what was checked about the
    voice (``replaced`` slots verified, ``patched`` anything else that could change timing)."""

    items = [changes] if isinstance(changes, dict) and "replace_label" in changes else changes
    if items is None:
        items = []
    if not isinstance(items, list):
        raise VariantError("changes must be a list of {path, value} or slot replacements")
    state = {"patched": False}
    if not items:  # no change: the control
        return copy.deepcopy(document), state
    current = document
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal current
        if pending:
            current = apply_changes(current, list(pending))
            pending.clear()
            state["patched"] = True

    for item in items:
        if isinstance(item, dict) and "replace_label" in item:
            flush()
            current, _ = replace_slot(current, item, root, notes)
        else:
            if not isinstance(item, dict):
                raise VariantError("each change must be an object")
            pending.append(item)
    flush()
    return current, state


def _combinations(variants: dict[str, Any]) -> list[tuple[list[str], list[list[dict[str, Any]]]]]:
    if not isinstance(variants, dict) or not variants:
        raise VariantError("variants must be an object {axis: [{id, changes}]} with 1 to 3 axes")
    if len(variants) > MAX_AXES:
        raise VariantError(f"variants has {len(variants)} axes: at most {MAX_AXES}")
    axes = list(variants)
    options: list[list[dict[str, Any]]] = []
    for axis in axes:
        if not _ID.match(axis):
            raise VariantError(f"axis name {axis!r}: use letters, digits, '_', '.' or '-'")
        entries = variants[axis]
        if not isinstance(entries, list) or not entries:
            raise VariantError(f"axis {axis!r} must be a non-empty list of {{id, changes}}")
        seen: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) - {"id", "changes"}:
                raise VariantError(f"axis {axis!r}: each option is {{id, changes}}")
            ident = entry.get("id")
            if not isinstance(ident, str) or not _ID.match(ident):
                raise VariantError(
                    f"axis {axis!r}: option id {ident!r} must be 1 to 40 letters, digits, "
                    "'_', '.' or '-'"
                )
            if ident in seen:
                raise VariantError(f"axis {axis!r}: the id {ident!r} appears twice")
            seen.add(ident)
        options.append(entries)
    count = 1
    for entries in options:
        count *= len(entries)
    if count > MAX_VARIANTS:
        sizes = " x ".join(str(len(entries)) for entries in options)
        raise VariantError(
            f"{sizes} = {count} variants: at most {MAX_VARIANTS}. Test fewer options "
            "(3 hooks x 1 body x 2 calls to action is a common start)"
        )
    return [(axes, list(combo)) for combo in itertools.product(*options)]


def _name(pattern: str, base: str, ids: dict[str, str]) -> str:
    keys = set(re.findall(r"{([A-Za-z0-9_.-]+)}", pattern))
    known = {"base", *ids}
    if keys - known:
        if pattern == DEFAULT_PATTERN:
            pattern = "_".join(["{base}", *(f"{{{axis}}}" for axis in ids)])
        else:
            missing = ", ".join(sorted(keys - known))
            raise VariantError(
                f"name_pattern uses {{{missing}}}, but the axes are {', '.join(ids)} (and {{base}})"
            )
    try:
        value = pattern.format(base=base, **ids)
    except (KeyError, IndexError, ValueError) as exc:
        raise VariantError(f"name_pattern {pattern!r} cannot be filled: {exc}") from exc
    try:
        return storyboard_name(value)
    except ValueError as exc:
        raise VariantError(f"name_pattern gives {value!r}: {exc}") from exc


def expand_matrix(
    base: dict[str, Any],
    variants: dict[str, Any],
    name_pattern: str = DEFAULT_PATTERN,
    *,
    root: Path | None = None,
) -> list[Variant]:
    """The cartesian product of the axes applied to ``base`` (see the module docstring).

    Every variant is checked against the schema; a matrix with refused combinations raises ONE
    ``VariantError`` naming all of them (``errors``: variant, path, message), and nothing is
    built.
    """

    if not isinstance(base, dict):
        raise VariantError("base must be a storyboard object")
    from adanvil.storyboard.schema import Storyboard

    base_name = str(base.get("name", "storyboard"))
    base_total = total_duration(base)
    built: list[Variant] = []
    taken: set[str] = set()
    errors: list[dict[str, str]] = []
    for axes, combo in _combinations(variants):
        ids = {axis: str(option["id"]) for axis, option in zip(axes, combo, strict=True)}
        label = ", ".join(f"{axis}={ident}" for axis, ident in ids.items())
        document = copy.deepcopy(base)
        patched = False
        moved: list[str] = []
        try:
            for option in combo:
                document, state = apply_option(document, option.get("changes"), root, moved)
                patched = patched or state["patched"]
            name = _name(name_pattern, base_name, ids)
            if name in taken:
                raise VariantError(
                    f"name_pattern gives {name!r} for two variants: put every axis in it",
                    path="name_pattern",
                )
            taken.add(name)
            total = total_duration(document)
            if (
                patched
                and abs(total - base_total) > _VOICE_TOLERANCE
                and (document.get("audio") or {}).get("voice")
            ):
                raise VariantError(
                    f"the changes alter the duration by {total - base_total:+.2f}s and the "
                    "voice-over (audio.voice) is laid on the timeline: it would no longer match",
                    path="audio.voice",
                )
            rename_storyboard(document, name)
            try:
                Storyboard.model_validate(document)
            except ValidationError as exc:
                found = exc.errors()
                for item in found[:5]:
                    place = ".".join(str(part) for part in item["loc"])
                    errors.append(
                        {
                            "variant": label,
                            "path": place,
                            "message": f"the storyboard is not valid ({place}: {item['msg']})",
                        }
                    )
                continue
        except (VariantError, ValueError) as exc:
            errors.append(
                {
                    "variant": label,
                    "path": str(getattr(exc, "path", None) or ""),
                    "message": str(exc),
                }
            )
            continue
        built.append(Variant(name=name, ids=ids, document=document, duration=total, moved=moved))
    if errors:
        raise VariantError(format_errors(errors), errors=errors)
    return built


def format_errors(errors: list[dict[str, str]]) -> str:
    """One message for the whole list: one line per error (variant, path, message)."""

    def line(item: dict[str, str]) -> str:
        where = f" [{item['path']}]" if item.get("path") else ""
        return f"variant {item['variant']}{where}: {item['message']}"

    if len(errors) == 1:
        return line(errors[0])
    body = chr(10).join(f"- {line(item)}" for item in errors)
    return f"{len(errors)} errors:" + chr(10) + body


# ---------------------------------------------------------------------------- the contact sheet


def hook_preview(document: dict[str, Any], seconds: float = 3.0) -> dict[str, Any]:
    """A copy of the storyboard cut after the first segment that reaches ``seconds`` (whole
    segments only: their cache entries are the ones the final render uses), rendering as a
    preview into ``renders/variants/_hooks``."""

    result = copy.deepcopy(document)
    spans = durations(list(result["segments"]))
    elapsed = 0.0
    keep = len(spans)
    for number, span in enumerate(spans):
        elapsed += span
        if elapsed >= seconds:
            keep = number + 1
            break
    result["segments"] = result["segments"][:keep]
    result["formats"] = []
    result["output"] = {
        "path": "renders/variants/_hooks/{name}.mp4",
        "quality": "preview",
    }
    return result


def tail_preview(document: dict[str, Any], seconds: float = 3.0) -> dict[str, Any]:
    """A copy of the storyboard that keeps the whole segments reaching the last ``seconds`` of
    the ad (the end: call to action, end card), the overlays shifted to the new start, rendering
    as a preview into ``renders/variants/_tails``. The sound is dropped (a preview of pictures:
    ``audio``, ``sfx`` and voice-driven captions go).

    When the ad ends on an image end card, the window is the ``seconds`` BEFORE the card (the
    call to action part, where two CTA variants differ) and the card itself is left out: a card
    shared by every variant would make the whole row look the same."""

    result = copy.deepcopy(document)
    segments = list(result["segments"])
    spans = durations(segments)
    if (
        len(segments) > 1
        and isinstance(segments[-1], dict)
        and segments[-1].get("type") == "image"
    ):
        segments, spans = segments[:-1], spans[:-1]
    total = sum(spans)
    elapsed = 0.0
    first = 0
    for number, span in enumerate(spans):
        if elapsed + span > total - seconds + _EPS:
            first = number
            break
        elapsed += span
    cut = sum(spans[:first])
    result["segments"] = segments[first:]
    length = total - cut
    kept: list[dict[str, Any]] = []
    for overlay in result.get("overlays") or []:
        if not isinstance(overlay, dict) or overlay.get("type") in {
            "sfx",
            "voice_captions",
            "speech_captions",
        }:
            continue
        stop = overlay.get("end")
        if isinstance(stop, (int, float)) and stop <= cut + _EPS:
            continue
        _shift_times(overlay, -cut)
        if isinstance(overlay.get("start"), (int, float)):
            if overlay["start"] < 0:
                overlay["start"] = 0.0
            if overlay["start"] >= length - _EPS:
                continue  # it starts after the preview ends (on the end card)
        kept.append(overlay)
    result["overlays"] = kept
    result.pop("audio", None)
    result["formats"] = []
    result["output"] = {
        "path": "renders/variants/_tails/{name}.mp4",
        "quality": "preview",
    }
    return result


def _strip(path: Path, moments: list[float], scratch: Path, name: str, thumb: int) -> list[Any]:
    from PIL import Image

    from adanvil.ffmpeg import extract_frame

    frames: list[Image.Image] = []
    for number, moment in enumerate(moments):
        frame_file = scratch / f"{name}_{number}.png"
        extract_frame(path, round(max(0.0, moment), 3), frame_file, width=thumb)
        with Image.open(frame_file) as opened:
            frames.append(opened.convert("RGB"))
        frame_file.unlink(missing_ok=True)
    return frames


def variants_sheet(
    root: Path,
    variants: list[Variant],
    out: Path,
    *,
    seconds: float = 3.0,
    cache: Any | None = None,
) -> tuple[Path, list[str]]:
    """A JPEG with one cell per variant, labelled with its name: two rows of three frames, the
    first ``seconds`` of the ad on top and its last ``seconds`` below (the call to action: the
    ``seconds`` before the end card when the ad ends on an image card).
    Returns the path and the notes (a variant whose preview could not be built gets an empty
    row and a note)."""

    from PIL import Image, ImageDraw

    from adanvil.ffmpeg import probe
    from adanvil.render import compose
    from adanvil.storyboard import load_storyboard

    notes: list[str] = []
    cells: list[tuple[str, list[Image.Image], list[Image.Image]]] = []
    scratch = Path(root) / "cache" / "variants"
    scratch.mkdir(parents=True, exist_ok=True)
    thumb = 150
    for variant in variants:
        head: list[Image.Image] = []
        tail: list[Image.Image] = []
        try:
            preview = load_storyboard(hook_preview(variant.document, seconds))
            report = compose(preview, root=Path(root), cache=cache, quality="preview")
            length = float(probe(report.path).duration or seconds)
            top = min(seconds, length) - 0.05
            head = _strip(
                report.path,
                [top * fraction for fraction in (0.12, 0.5, 0.95)],
                scratch,
                f"{variant.name}_head",
                thumb,
            )
        except Exception as exc:  # one broken preview must not lose the sheet
            notes.append(f"{variant.name}: no start preview ({str(exc)[:160]})")
        try:
            ending = load_storyboard(tail_preview(variant.document, seconds))
            report = compose(ending, root=Path(root), cache=cache, quality="preview")
            length = float(probe(report.path).duration or seconds)
            span = min(seconds, length)
            tail = _strip(
                report.path,
                [length - span * fraction - 0.05 for fraction in (0.95, 0.5, 0.12)],
                scratch,
                f"{variant.name}_tail",
                thumb,
            )
        except Exception as exc:
            notes.append(f"{variant.name}: no end preview ({str(exc)[:160]})")
        cells.append((variant.name, head, tail))

    height = max(
        (f.height for _, head, tail in cells for f in [*head, *tail]), default=int(thumb * 16 / 9)
    )
    label_height = 22
    gap = 6
    cell_width = 3 * thumb + 2 * 2
    cell_height = label_height + 2 * height + 2
    per_row = min(3, max(1, len(cells)))
    rows = -(-len(cells) // per_row)
    sheet = Image.new(
        "RGB",
        (per_row * cell_width + (per_row + 1) * gap, rows * (cell_height + gap) + gap),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(sheet)
    for number, (name, head, tail) in enumerate(cells):
        row, column = divmod(number, per_row)
        left = gap + column * (cell_width + gap)
        top = gap + row * (cell_height + gap)
        draw.text((left + 2, top + 5), f"{name[:36]}  start / end", fill=(235, 235, 235))
        for line, frames in enumerate((head, tail)):
            for index, frame in enumerate(frames):
                sheet.paste(
                    frame,
                    (left + index * (thumb + 2), top + label_height + line * (height + 2)),
                )
    out = Path(out)
    encoded = io.BytesIO()
    sheet.save(encoded, format="JPEG", quality=85)
    replace_bytes(out, encoded.getvalue())  # a link planted at ``out`` is replaced, not followed
    return out, notes


__all__ = [
    "DEFAULT_PATTERN",
    "MAX_AXES",
    "MAX_VARIANTS",
    "Variant",
    "VariantError",
    "apply_option",
    "durations",
    "expand_matrix",
    "format_errors",
    "hook_preview",
    "replace_slot",
    "tail_preview",
    "total_duration",
    "variants_sheet",
]
