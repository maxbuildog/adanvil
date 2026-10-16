"""Pre-render storyboard validation for agents and humans."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from adanvil.audio import beat_grid
from adanvil.brand import Brand, find_banned
from adanvil.ffmpeg import MAX_SECONDS, probe
from adanvil.media import (
    ClipSegment,
    ImageSegment,
    RevealSegment,
    SplitSegment,
    segment_duration,
)
from adanvil.project import resolve_in_project
from adanvil.sfx import resolve_sfx
from adanvil.speech.analyze import brand_filler_lists, lists_from_brand, normalize_word
from adanvil.speech.captions import (
    PlayedRange,
    expand_speech_captions,
    played_ranges,
    spoken_words,
)
from adanvil.speech.library import SpeechLibrary, normalise_path
from adanvil.text import render
from adanvil.text.emoji import DirectoryProvider, NoEmojiProvider, missing_emoji
from adanvil.text.fonts import unusable_family_message, usable_family
from adanvil.vocab import normalize_shot_type
from adanvil.voice import VoiceTrack, load_track

from .craft import CaptionEntry, storyboard_checks
from .devices import check_insets, check_reveals, check_splits, check_stickers
from .recipes import Recipe, get_recipe
from .schema import (
    CaptionOverlay,
    CommentOverlay,
    ImageOverlay,
    SfxOverlay,
    SpeechCaptionsOverlay,
    StickerOverlay,
    Storyboard,
    VideoOverlay,
    VoiceCaptionsOverlay,
)
from .sfx import all_sfx

_CUT_MARGIN = 0.02  # seconds: a cut this close to a word edge is not "inside" the word


@dataclass
class Issue:
    code: str
    message: str
    where: str | None = None


@dataclass
class ValidationReport:
    ok: bool
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    info: list[Issue] = field(default_factory=list)
    reference_comparison: dict[str, Any] | None = None


def _format_path(loc: tuple[Any, ...] | list[Any]) -> str | None:
    if not loc:
        return None
    result = ""
    for part in loc:
        if isinstance(part, int):
            result += f"[{part}]"
            continue
        name = "in" if part == "in_" else str(part)
        if result.endswith("]") and name in {"clip", "image", "caption", "comment"}:
            continue
        result += ("." if result else "") + name
    return result


def _plain_error(message: str) -> str:
    if message == "Field required":
        return "field is required"
    return re.sub(r"^(?:Value error|Input should be),?\s*", "", message).strip()


def _schema_issues(exc: ValidationError) -> list[Issue]:
    issues: list[Issue] = []
    for error in exc.errors():
        message = str(error.get("msg", "Invalid value"))
        loc = tuple(error.get("loc", ()))
        where = _format_path(loc)
        unknown = re.search(r"unexpected field\(s\):\s*(.+)$", message)
        if unknown:
            fields = [item.strip() for item in unknown.group(1).split(",") if item.strip()]
            base = re.sub(r"\s+has unexpected field\(s\):.*$", "", message)
            base = re.sub(r"^Value error,\s*", "", base)
            base = base.removeprefix("storyboard.")
            for name in fields:
                issue_where = f"{base}.{name}" if base and base != "storyboard" else name
                issues.append(
                    Issue(
                        code="schema",
                        message=f"Unknown field '{name}' in the storyboard.",
                        where=issue_where,
                    )
                )
            continue

        plain = _plain_error(message)
        code = "speech_corrections" if plain.startswith("speech_corrections:") else "schema"
        if (
            where
            and where.startswith("overlays[")
            and any(
                word in plain.lower() for word in ("overlay.start", "overlay.end", "greater than")
            )
        ):
            code = "overlay_timing"
        issues.append(Issue(code=code, message=f"Invalid storyboard value: {plain}", where=where))
    return issues


def _summary(raw: Mapping[str, Any] | None, recipe_id: str | None) -> dict[str, Any]:
    segments = raw.get("segments", []) if raw else []
    overlays = raw.get("overlays", []) if raw else []
    return {
        "total_duration": None,
        "segment_count": len(segments) if isinstance(segments, list) else 0,
        "overlay_count": len(overlays) if isinstance(overlays, list) else 0,
        "recipe_id": recipe_id,
        "checklist": [],
    }


def _relative_file(root: Path, written: str) -> Path | None:
    try:
        return resolve_in_project(root, written)
    except (OSError, TypeError, ValueError):
        return None


def _family_for_render(family: str, root: Path) -> str | None:
    """The family a text may be drawn with: a bundled name, or the absolute path of a font file
    of the project; ``None`` for anything else (the default family is used instead)."""

    return usable_family(family, root)


def _checked_font(overlay: Any, root: Path, index: int, warnings: list[Issue]) -> Any:
    """``overlay`` with a ``family`` it may be drawn with: a font that is neither bundled nor a
    file of the project is replaced by the default family, with a warning."""

    family = getattr(overlay, "family", None)
    if family is None:
        return overlay
    usable = _family_for_render(family, root)
    if usable is None:
        warnings.append(
            Issue(
                code="font_not_found",
                message=f"font_not_found: {unusable_family_message(family)}",
                where=f"overlays[{index}].family",
            )
        )
    return overlay if usable == family else overlay.model_copy(update={"family": usable})


def _pad_for_text(spec: Any, width: int, height: int) -> int:
    size = int(spec.size or 64)
    estimated_height = max(120, round(size * 1.8 * max(1, len(spec.lines)) + 180))
    estimated_width = min(max(120, spec.max_width + 120), 2400)
    center = spec.x_center if spec.x_center is not None else width // 2
    return max(
        256,
        -spec.y + 256,
        spec.y + estimated_height - height + 256,
        -center + estimated_width // 2 + 256,
        center + estimated_width // 2 - width + 256,
    )


def _render_text_overlay(
    overlay: CaptionOverlay | CommentOverlay,
    *,
    root: Path,
    width: int,
    height: int,
    brand: Brand | None,
) -> tuple[tuple[int, int, int, int], bool]:
    applied = (
        brand.apply_to_caption(overlay) if brand is not None else overlay.model_copy(deep=True)
    )
    spec = applied.to_text_spec()
    if spec.family is not None:
        spec = spec.model_copy(update={"family": _family_for_render(spec.family, root)})

    pad = _pad_for_text(spec, width, height)
    center = spec.x_center if spec.x_center is not None else width // 2
    shifted = spec.model_copy(update={"x_center": center + pad, "y": spec.y + pad})
    try:
        result = render(
            shifted,
            emoji=NoEmojiProvider(),
            canvas=(width + 2 * pad, height + 2 * pad),
        )
    except Exception as exc:
        # Pillow refuses pathological agent input before the renderer can
        # return its normal overflow flag.  It is still the same actionable
        # condition: the line cannot fit, and should be reported as a warning.
        if "image size" in str(exc).lower() or "decompression bomb" in str(exc).lower():
            return (0, 0, 0, 0), True
        raise
    raw_bbox = result.bbox
    bbox = tuple(value - pad for value in raw_bbox)
    result.image.close()
    return bbox, result.overflow


def _fmt(value: Any) -> str:
    """Round floats for messages: 0.8000000000000007 reads as 0.8."""
    return f"{value:g}" if isinstance(value, float) else str(value)


def _rule_issue(
    rule: Any,
    *,
    what: str,
    value: Any,
    limit: Any,
    where: str | None,
    errors: list[Issue],
    warnings: list[Issue],
) -> None:
    issue = Issue(
        code=rule.id,
        message=f"{rule.id}: {what} ({_fmt(value)} vs {_fmt(limit)})",
        where=where,
    )
    (warnings if rule.severity == "warn" else errors).append(issue)


def _check_recipe_rules(
    sb: Storyboard,
    recipe: Recipe,
    *,
    total: float,
    caption_entries: list[tuple[str, CaptionOverlay]],
    text_bboxes: dict[str, tuple[int, int, int, int]],
    errors: list[Issue],
    warnings: list[Issue],
) -> None:
    captions = caption_entries
    # Captions generated from a voice track (keys "i.j") follow the speech by design:
    # timing and count rules target the captions the agent wrote, not each word chunk.
    written = [(key, overlay) for key, overlay in caption_entries if "." not in key]

    for rule in recipe.rules:
        params = rule.params
        if rule.check == "total_duration":
            minimum, maximum = params.get("min"), params.get("max")
            if minimum is not None and total < float(minimum) - _TOLERANCE:
                _rule_issue(
                    rule,
                    what="total duration is below the minimum",
                    value=total,
                    limit=minimum,
                    where="segments",
                    errors=errors,
                    warnings=warnings,
                )
            elif maximum is not None and total > float(maximum) + _TOLERANCE:
                _rule_issue(
                    rule,
                    what="total duration is above the maximum",
                    value=total,
                    limit=maximum,
                    where="segments",
                    errors=errors,
                    warnings=warnings,
                )

        elif rule.check == "shot_duration":
            prefix = str(params.get("label_prefix", ""))
            for index, segment in enumerate(sb.segments):
                label = segment.label or ""
                if not label.startswith(prefix):
                    continue
                value = segment_duration(segment)
                minimum, maximum = params.get("min"), params.get("max")
                if minimum is not None and value < float(minimum) - _TOLERANCE:
                    _rule_issue(
                        rule,
                        what=f"segment '{label}' is shorter than required",
                        value=value,
                        limit=minimum,
                        where=f"segments[{index}]",
                        errors=errors,
                        warnings=warnings,
                    )
                if maximum is not None and value > float(maximum) + _TOLERANCE:
                    _rule_issue(
                        rule,
                        what=f"segment '{label}' is longer than allowed",
                        value=value,
                        limit=maximum,
                        where=f"segments[{index}]",
                        errors=errors,
                        warnings=warnings,
                    )

        elif rule.check == "caption_min_on_screen":
            minimum = float(params.get("min", 0))
            for index, overlay in written:
                end = total if overlay.end is None else overlay.end
                value = end - overlay.start
                if value < minimum - _TOLERANCE:
                    _rule_issue(
                        rule,
                        what=f"caption overlay {index} is on screen too briefly",
                        value=value,
                        limit=minimum,
                        where=f"overlays[{index}]",
                        errors=errors,
                        warnings=warnings,
                    )

        elif rule.check == "caption_safe_zone":
            top, bottom = float(params.get("top", 0)), float(params.get("bottom", 1920))
            for index, _overlay in captions:
                bbox = text_bboxes.get(index)
                if bbox is None:
                    continue
                if bbox[1] < top:
                    _rule_issue(
                        rule,
                        what=f"caption overlay {index} starts above the safe zone",
                        value=bbox[1],
                        limit=top,
                        where=f"overlays[{index}]",
                        errors=errors,
                        warnings=warnings,
                    )
                if bbox[3] > bottom:
                    _rule_issue(
                        rule,
                        what=f"caption overlay {index} ends below the safe zone",
                        value=bbox[3],
                        limit=bottom,
                        where=f"overlays[{index}]",
                        errors=errors,
                        warnings=warnings,
                    )

        elif rule.check == "end_card_required":
            prefix = str(params.get("label_prefix", ""))
            label = (sb.segments[-1].label or "") if sb.segments else ""
            if not label.startswith(prefix):
                _rule_issue(
                    rule,
                    what="the last segment is not the required end card",
                    value=label or "<missing>",
                    limit=prefix,
                    where=f"segments[{len(sb.segments) - 1}]" if sb.segments else "segments",
                    errors=errors,
                    warnings=warnings,
                )

        elif rule.check == "max_caption_lines":
            maximum = int(params.get("max", 0))
            for index, overlay in captions:
                value = len(overlay.lines)
                if value > maximum:
                    _rule_issue(
                        rule,
                        what=f"caption overlay {index} has too many lines",
                        value=value,
                        limit=maximum,
                        where=f"overlays[{index}].lines",
                        errors=errors,
                        warnings=warnings,
                    )

        elif rule.check == "caption_count":
            value = len(written)
            minimum, maximum = params.get("min"), params.get("max")
            if minimum is not None and value < int(minimum):
                _rule_issue(
                    rule,
                    what="caption count is below the minimum",
                    value=value,
                    limit=minimum,
                    where="overlays",
                    errors=errors,
                    warnings=warnings,
                )
            elif maximum is not None and value > int(maximum):
                _rule_issue(
                    rule,
                    what="caption count is above the maximum",
                    value=value,
                    limit=maximum,
                    where="overlays",
                    errors=errors,
                    warnings=warnings,
                )

        elif rule.check == "banned_words":
            # The brand pass reports the precise word and overlay.  Keeping
            # this rule as a no-op avoids duplicate errors for the same text.
            continue


_CLOSEUP_SCALES = ("extreme_closeup", "closeup")
# A caption that reads as someone speaking: an optional emoji or name, a colon, an opening quote.
_QUOTE_CAPTION = re.compile(
    r"^\s*(?:[^\w\s:«\"“]{1,8}|[^\W\d_][\w'’-]*(?:\s+[\w'’-]+){0,2})?\s*:\s*[«\"“]"
)
_MAX_ENLARGEMENT = 1.25


def source_enlargement(segment: ClipSegment, width: int, height: int, fmt: Any) -> float:
    """How much more than a plain cover-fit the segment magnifies its source.

    ``width`` and ``height`` are the source display size. A crop of the source and the end
    of a slow zoom enlarge the picture; 1.0 means a plain cover-fit. A freeze zoom is a
    deliberate push-in on a held frame, so it does not count here.
    """
    magnification = 1.0
    if segment.crop is not None and segment.crop.w > 0 and segment.crop.h > 0:
        plain = max(fmt.width / width, fmt.height / height)
        cropped = max(fmt.width / segment.crop.w, fmt.height / segment.crop.h)
        magnification = cropped / plain
    extra = 1.0
    if segment.zoom is not None:
        extra = max(extra, segment.zoom.from_, segment.zoom.to)
    # A punch-in is a deliberate emphasis: it only counts when it is a really tight crop.
    for punch in segment.punch:
        if punch.zoom > _MAX_ENLARGEMENT:
            extra = max(extra, punch.zoom)
    return magnification * extra


def _load_breakdown(root: Path, written: str) -> tuple[dict[str, Any], Path] | str:
    """Return the breakdown payload and its folder, or the reason it cannot be used."""
    path = _relative_file(root, written)
    if path is None:
        return f"reference '{written}' is absolute or leaves the project folder"
    if path.is_dir():
        path = path / "breakdown.json"
    if not path.is_file():
        return f"reference breakdown '{written}' does not exist; run reference_breakdown first"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"could not read reference breakdown '{written}': {exc}"
    summary = payload.get("summary") if isinstance(payload, dict) else None
    if not isinstance(summary, dict) or not isinstance(summary.get("type_demand"), dict):
        return f"'{written}' is not a reference breakdown (no summary.type_demand)"
    return payload, path.parent


def _current_type_demand(type_demand: Mapping[str, Any]) -> dict[str, Any]:
    """Type demand of a stored breakdown under the current shot type names.

    A breakdown saved before the rename still counts ``skin_texture`` and
    ``anatomy_diagram``: those entries are folded into ``texture_macro`` and ``diagram``.
    """

    merged: dict[str, Any] = {}
    for name, demand in type_demand.items():
        key = normalize_shot_type(name)
        previous = merged.get(key)
        if isinstance(previous, dict) and isinstance(demand, Mapping):
            merged[key] = {
                "shots": int(previous.get("shots", 0)) + int(demand.get("shots", 0)),
                "seconds": float(previous.get("seconds", 0.0)) + float(demand.get("seconds", 0.0)),
                "share": float(previous.get("share", 0.0)) + float(demand.get("share", 0.0)),
            }
        else:
            merged[key] = dict(demand) if isinstance(demand, Mapping) else demand
    return merged


def _reference_shot_lengths(folder: Path) -> list[float]:
    """Decoded shot lengths of the reference, when its decode sits next to the breakdown."""
    try:
        decode = json.loads((folder / "reference.json").read_text(encoding="utf-8"))
        return [float(shot["duration"]) for shot in decode["shots"]]
    except (OSError, ValueError, KeyError, TypeError):
        return []


# Durations come from subtractions (3.4 - 1.8 = 1.5999...): compare with 1 ms of slack,
# far below one frame, so a 1.6 s shot meets a 1.6 s minimum.
_TOLERANCE = 1e-3

_COMPLAINT_ROLES = ("problem", "myth", "agitation")
_SMILING = ("happy", "satisfied", "relaxed")
_REACTIONS = ("shocked", "skeptical", "frustrated")
_FLAT = ("neutral", "relaxed", "happy")
# The story turns from the problem to the solution at the first of these roles.
_SOLUTION_ROLES = ("reveal", "mechanism", "demo", "proof", "payoff", "offer", "cta")
_PRODUCT_TYPES = ("product_hero", "product_in_use", "unboxing")


def _check_emotions(model: Storyboard, warnings: list[Issue]) -> None:
    """A face that smiles under a complaint says the opposite of its text."""

    for index, segment in enumerate(model.segments):
        if segment.role in _COMPLAINT_ROLES and segment.emotion in _SMILING:
            warnings.append(
                Issue(
                    code="emotion_mismatch",
                    message=(
                        f"emotion_mismatch: segments[{index}] has role {segment.role} but "
                        f"emotion {segment.emotion}: the text complains while the face "
                        "smiles; pick a sad, worried, tired, closed or frustrated face"
                    ),
                    where=f"segments[{index}]",
                )
            )


def _check_story_order(model: Storyboard, warnings: list[Issue]) -> None:
    """The problem part shows people living the problem; the product comes with the solution."""

    roles = [segment.role for segment in model.segments]
    turn = next((i for i, role in enumerate(roles) if role in _SOLUTION_ROLES), None)
    if turn is None:
        return
    for index, segment in enumerate(model.segments[:turn]):
        if segment.shot_type in _PRODUCT_TYPES or segment.subject == "product":
            warnings.append(
                Issue(
                    code="product_too_early",
                    message=(
                        f"product_too_early: segments[{index}] shows the product before the "
                        f"story turns to the solution (segments[{turn}], role {roles[turn]}): "
                        "open on people living the problem (a worried face, a messy kitchen, "
                        "a runner out of breath, no product), then show that a solution exists"
                    ),
                    where=f"segments[{index}]",
                )
            )


def _check_reference_emotions(
    model: Storyboard, shots: list[Any], warnings: list[Issue]
) -> None:
    """The reference answers a punch or a question with a reaction face: so should the edit."""

    felt = [
        shot.get("emotion")
        for shot in shots
        if isinstance(shot, dict)
        and (shot.get("role") == "punch" or shot.get("text_device") == "question")
        and shot.get("emotion") not in (None, "none")
    ]
    if not felt or sum(1 for item in felt if item in _REACTIONS) * 2 <= len(felt):
        return
    for index, segment in enumerate(model.segments):
        if segment.role == "punch" and (segment.emotion is None or segment.emotion in _FLAT):
            shown = segment.emotion or "no emotion"
            warnings.append(
                Issue(
                    code="emotion_mismatch",
                    message=(
                        f"emotion_mismatch: segments[{index}] is a punch with {shown}, but "
                        "the reference answers with a reaction face (shocked, skeptical or "
                        "frustrated): pick a shot with a visible reaction"
                    ),
                    where=f"segments[{index}]",
                )
            )


def _check_reference(
    model: Storyboard,
    root: Path,
    *,
    written_captions: int,
    warnings: list[Issue],
) -> dict[str, Any] | None:
    """Compare the storyboard with the reference breakdown (all findings are warnings)."""
    assert model.reference is not None
    loaded = _load_breakdown(root, model.reference)
    if isinstance(loaded, str):
        warnings.append(
            Issue(code="ref_missing", message=f"ref_missing: {loaded}", where="reference")
        )
        return None
    payload, folder = loaded
    summary = payload["summary"]

    segments = model.segments
    total = model.total_duration()
    durations = [segment_duration(segment) for segment in segments]

    # Pacing: the end card and long holds are not part of the editing rhythm on either side.
    own = [
        duration
        for segment, duration in zip(segments, durations, strict=True)
        if not (segment.label or "").startswith("end_card") and duration <= 6.0
    ]
    own_mean = sum(own) / len(own) if own else None
    lengths = [length for length in _reference_shot_lengths(folder) if length <= 6.0]
    pacing = summary.get("pacing") if isinstance(summary.get("pacing"), dict) else {}
    ref_mean = (sum(lengths) / len(lengths)) if lengths else float(pacing.get("mean") or 0.0)
    if own_mean is not None and ref_mean > 0 and own_mean > 1.35 * ref_mean:
        warnings.append(
            Issue(
                code="ref_pacing",
                message=(
                    f"ref_pacing: your mean shot is {own_mean:.1f} s, the reference "
                    f"{ref_mean:.1f} s: cut faster or split shots"
                ),
                where="segments",
            )
        )

    # Shot types the reference leans on must be covered or explained.
    used_types = {segment.shot_type for segment in segments if segment.shot_type}
    gap_types = {gap.type for gap in model.gaps}
    needed = {
        name: demand
        for name, demand in _current_type_demand(summary["type_demand"]).items()
        if isinstance(demand, dict)
        and (int(demand.get("shots", 0)) >= 2 or float(demand.get("share", 0.0)) >= 0.15)
    }
    missing = [name for name in needed if name not in used_types and name not in gap_types]
    if missing:
        listed = ", ".join(
            f"{name} ({needed[name].get('shots', 0)} shots, "
            f"{_fmt(round(float(needed[name].get('seconds', 0.0)), 1))} s)"
            for name in missing
        )
        warnings.append(
            Issue(
                code="ref_types",
                message=(
                    f"ref_types: the reference uses shot types you do not cover: {listed}; "
                    "add them as segment shot_type or list them in gaps with a reason"
                ),
                where="segments",
            )
        )

    # Close-ups: segments without a scale count as not close.
    scale_mix = summary.get("scale_mix") if isinstance(summary.get("scale_mix"), dict) else {}
    ref_close = sum(float(scale_mix.get(name, 0.0)) for name in _CLOSEUP_SCALES)
    close_seconds = sum(
        duration
        for segment, duration in zip(segments, durations, strict=True)
        if segment.scale in _CLOSEUP_SCALES
    )
    own_close = close_seconds / total if total > 0 else 0.0
    if ref_close >= 0.30 and own_close < ref_close / 2:
        warnings.append(
            Issue(
                code="ref_closeups",
                message=(
                    f"ref_closeups: the reference spends {ref_close * 100:.0f}% of its duration "
                    f"in close-ups, you have {own_close * 100:.0f}%: use tighter shots "
                    "(set scale on each segment); use shots filmed close, "
                    "never crop to fake a close-up"
                ),
                where="segments",
            )
        )

    # People.
    ref_people = [person for person in summary.get("people", []) if isinstance(person, str)]
    own_people = sorted(
        {s.subject for s in segments if s.subject and s.subject.startswith("person_")}
    )
    if len(ref_people) >= 3 and len(own_people) < 2 and "talking_head" not in gap_types:
        warnings.append(
            Issue(
                code="ref_people",
                message=(
                    f"ref_people: the reference shows {len(ref_people)} distinct people, "
                    f"you show {len(own_people)}: use at least 2 (subject person_*) "
                    "or explain in gaps (talking_head)"
                ),
                where="segments",
            )
        )

    # Text rhythm: the written captions, not the ones generated from the voice.
    patterns = summary.get("repeated_patterns")
    patterns = patterns if isinstance(patterns, dict) else {}
    repeated = any(
        key.replace(" ", "") == "myth>punch" or int(count) >= 2 for key, count in patterns.items()
    )
    text_shots = int(summary.get("text_shots") or 0)
    if repeated and written_captions < text_shots / 2:
        warnings.append(
            Issue(
                code="ref_text_rhythm",
                message=(
                    f"ref_text_rhythm: the reference changes its on-screen text {text_shots} "
                    f"times; you have {written_captions} captions"
                ),
                where="overlays",
            )
        )

    # Who speaks in the text: a reference whose texts are people talking keeps that device.
    devices = summary.get("text_devices") if isinstance(summary.get("text_devices"), dict) else {}
    ref_quotes = int(devices.get("quote", 0) or 0)
    own_quotes = sum(
        1
        for overlay in model.overlays
        if isinstance(overlay, CaptionOverlay)
        and _QUOTE_CAPTION.match(" ".join(overlay.lines).strip())
    )
    if summary.get("dialogue") and own_quotes < ref_quotes / 2:
        warnings.append(
            Issue(
                code="ref_text_device",
                message=(
                    "ref_text_device: in the reference the texts are people speaking "
                    f"({ref_quotes} quotes: an emoji or a name, a colon, then a quote); "
                    f"only {own_quotes} of your captions look like that. Keep the device: "
                    "write them as <emoji> : « ... »"
                ),
                where="overlays",
            )
        )

    _check_reference_emotions(model, payload.get("shots") or [], warnings)
    people_segments = [
        segment for segment in segments if segment.subject and segment.subject.startswith("person_")
    ]
    felt_people = sum(1 for segment in people_segments if segment.emotion)
    if people_segments and felt_people / len(people_segments) < 0.7:
        warnings.append(
            Issue(
                code="ref_unannotated_emotion",
                message=(
                    f"ref_unannotated_emotion: only {felt_people} of {len(people_segments)} "
                    "person segments have an emotion, so the emotion checks are approximate"
                ),
                where="segments",
            )
        )

    annotated = sum(1 for segment in segments if segment.shot_type)
    share = annotated / len(segments)
    if share < 0.7:
        warnings.append(
            Issue(
                code="ref_unannotated",
                message=(
                    f"ref_unannotated: only {annotated} of {len(segments)} segments have a "
                    "shot_type, so the reference checks are approximate"
                ),
                where="segments",
            )
        )

    return {
        "reference": model.reference,
        "mean_shot": {
            "storyboard": None if own_mean is None else round(own_mean, 2),
            "reference": round(ref_mean, 2),
        },
        "type_coverage": {
            "required": sorted(needed),
            "covered": sorted(name for name in needed if name in used_types),
            "gaps": sorted(name for name in needed if name in gap_types),
            "missing": sorted(missing),
        },
        "closeup_share": {"storyboard": round(own_close, 3), "reference": round(ref_close, 3)},
        "people": {"storyboard": len(own_people), "reference": len(ref_people)},
        "captions": {"storyboard": written_captions, "reference_text_changes": text_shots},
        "annotated_share": round(share, 2),
    }


def _continues(first: PlayedRange | None, second: PlayedRange | None, frame: float) -> bool:
    """Whether the sound of two consecutive segments is one continuous stretch of a source.

    True when the second segment plays the same file starting where the first one stopped
    (within a frame), whether through ``audio_from`` or its own clip: the voice goes on across
    the cut, so a cut of the picture inside a word does not cut that word.
    """

    if first is None or second is None or second.segment != first.segment + 1:
        return False
    return (
        normalise_path(first.source) == normalise_path(second.source)
        and abs(second.src_in - first.src_out) <= frame + 1e-6
    )


def _check_spoken_audio(
    model: Storyboard,
    library: SpeechLibrary,
    banned: list[str],
    starts: list[float],
    ends: list[float],
    warnings: list[Issue],
) -> None:
    """With ``audio.clips``: cuts inside spoken words, banned words said, missing transcripts."""

    reported_missing: set[str] = set()
    ranges = played_ranges(model, starts, ends)
    frame = 1.0 / max(1, model.format.fps)
    for position, played in enumerate(ranges):
        previous = ranges[position - 1] if position > 0 else None
        following = ranges[position + 1] if position + 1 < len(ranges) else None
        data = library.data(played.source)
        segment_where = f"segments[{played.segment}]"
        if data is None:
            entry = library.entry(played.source)
            if (entry is None or entry.has_audio) and played.source not in reported_missing:
                reported_missing.add(played.source)
                warnings.append(
                    Issue(
                        code="speech_missing",
                        message=(
                            f"speech_missing: '{played.source}' has no speech index, so cuts "
                            "inside words and spoken banned words cannot be checked (and "
                            "speech_captions cannot show it): run library_index with "
                            "transcribe=true"
                        ),
                        where=segment_where,
                    )
                )
            continue
        edges = (
            ("starts", played.src_in, f"{segment_where}.audio_from.in"
             if played.audio_from else f"{segment_where}.in",
             _continues(previous, played, frame)),
            ("ends", played.src_out, segment_where
             if played.audio_from else f"{segment_where}.out",
             _continues(played, following, frame)),
        )  # fmt: skip
        for label, edge, where, continuous in edges:
            if continuous:
                continue  # the sound goes on across this cut: the word is not cut
            inside = next(
                (
                    word
                    for word in data.words
                    if word.start + _CUT_MARGIN < edge < word.end - _CUT_MARGIN
                ),
                None,
            )
            if inside is not None:
                warnings.append(
                    Issue(
                        code="cut_mid_word",
                        message=(
                            f"cut_mid_word: segments[{played.segment}] sound {label} at "
                            f"{edge:.2f}s of '{played.source}', inside the word "
                            f"'{inside.text}' ({inside.start:.2f}-{inside.end:.2f}s): move the "
                            "cut into the pause before or after the word (speech_view "
                            "clean_segments give safe cuts)"
                        ),
                        where=where,
                    )
                )
        if not banned:
            continue
        kept = spoken_words([played], library)
        for entry in find_banned(" ".join(word.text for word in kept), banned):
            first = normalize_word(entry.split()[0]) if entry.split() else ""
            heard = next((word for word in kept if normalize_word(word.text) == first), None)
            at = (
                data.words[heard.index].start
                if heard is not None
                else played.src_in
            )
            warnings.append(
                Issue(
                    code="banned_word_spoken",
                    message=(
                        f"banned_word_spoken: segments[{played.segment}] plays the creator "
                        f"saying '{entry}' ('{played.source}' at {at:.2f}s): the brand may "
                        "not claim it; cut it out or pick another take"
                    ),
                    where=segment_where,
                )
            )


_END_CARD_OVERLAP = 0.2  # seconds a caption may run onto the end card image
_QUOTES = re.compile("[\u2018\u2019\u201a\u201b\u201c\u201d\u201e\u00ab\u00bb\"'`]")


def _same_file(left: str, right: str) -> bool:
    def plain(value: str) -> str:
        return value.replace("\\", "/").removeprefix("./")

    return plain(left) == plain(right)


def _check_emphasis(
    caption_entries: list[tuple[str, CaptionOverlay]], warnings: list[Issue]
) -> None:
    """Warn for each ``emphasis`` entry of a written caption that matches none of its words.

    Captions expanded from a voice or speech track share one list of entries over many chunks,
    so only the captions written in the storyboard are checked.
    """

    for key, caption in caption_entries:
        if "." in key:
            continue
        for entry in caption.unmatched_emphasis():
            warnings.append(
                Issue(
                    code="emphasis_unmatched",
                    message=(
                        f"emphasis_unmatched: emphasis {entry!r} matches no word of caption "
                        f"{key} ({' / '.join(caption.lines)!r}): nothing is highlighted. Write "
                        "the entry as it is shown (a price as '24,90 €' or '24,90', a "
                        "phrase as consecutive words), or mark it with *...* in the text"
                    ),
                    where=f"overlays[{key}].emphasis",
                )
            )


def _check_end_card_captions(
    model: Storyboard,
    brand: Brand | None,
    caption_entries: list[tuple[str, CaptionOverlay]],
    starts: list[float],
    ends: list[float],
    total: float,
    warnings: list[Issue],
) -> None:
    """Warn when a caption (written, voice or speech) runs onto the end card image.

    The end card image is the last segment labelled ``end_card`` when it is an image, and the
    brand's CTA image wherever it is used: they carry the call to action themselves and, by
    the doctrine, no subtitle or caption. An ``end_card`` that is a video clip (a product shot
    with its own box text) is not concerned.
    """

    cards: list[int] = []
    last_card = next(
        (
            index
            for index in range(len(model.segments) - 1, -1, -1)
            if (model.segments[index].label or "").startswith("end_card")
        ),
        None,
    )
    if last_card is not None and isinstance(model.segments[last_card], ImageSegment):
        cards.append(last_card)
    if brand is not None and brand.cta_image:
        cards.extend(
            index
            for index, segment in enumerate(model.segments)
            if _same_file(segment.source, brand.cta_image) and index not in cards
        )
    for card in cards:
        for key, caption in caption_entries:
            caption_end = total if caption.end is None else min(caption.end, total)
            overlap = min(caption_end, ends[card]) - max(caption.start, starts[card])
            if overlap > _END_CARD_OVERLAP:
                warnings.append(
                    Issue(
                        code="caption_on_end_card",
                        message=(
                            f"caption_on_end_card: caption {key} ({' / '.join(caption.lines)!r}) "
                            f"stays {overlap:.1f}s on the end card image of segments[{card}] "
                            f"({starts[card]:.1f}s to {ends[card]:.1f}s): the end card carries "
                            "the call to action, so end the caption before it (a speech caption "
                            "that runs over comes from the last spoken words: cut or trim them)"
                        ),
                        where=f"overlays[{key}]",
                    )
                )


def _check_card_content(model: Storyboard, root: Path, warnings: list[Issue]) -> None:
    """Warn when a caption or comment covers the lettering or a button of a still shown whole
    (an end card): the busy region of the picture file is computed (see ``adanvil.busy``)."""

    from adanvil.formats import card_content_clashes

    try:
        document = model.model_dump(mode="json", by_alias=True, exclude_none=True)
        clashes = card_content_clashes(document, root)
    except Exception:
        return
    for index, box, _busy, picture in clashes:
        warnings.append(
            Issue(
                code="caption_over_card_content",
                message=(
                    f"caption_over_card_content: overlays[{index}] covers the content "
                    f"(lettering or button) of {picture['source']} at y {round(box[1])} to "
                    f"{round(box[3])}: move it into an empty band of the picture or above or "
                    "below it, or end it before the card"
                ),
                where=f"overlays[{index}]",
            )
        )


def _fold_cta(text: str) -> str:
    """Case, accents, quotes and punctuation removed: the form a CTA is looked for in."""

    folded = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(char for char in folded if not unicodedata.combining(char))
    plain = _QUOTES.sub("", plain)
    return " ".join(re.sub(r"[^\w]+", " ", plain, flags=re.UNICODE).split())


def _check_cta_text(
    model: Storyboard,
    brand: Brand | None,
    caption_entries: list[tuple[str, CaptionOverlay]],
    warnings: list[Issue],
) -> None:
    """Warn when the brand's CTA wording is in no caption (written, voice or speech)."""

    cta = brand.product.cta_text if brand is not None and brand.product is not None else None
    wanted = _fold_cta(cta) if cta else ""
    if not wanted or not model.segments:
        return
    # Even a brand CTA image may not show the exact wording: warn and let the agent judge.
    ordered = sorted(caption_entries, key=lambda item: item[1].start)
    shown = f" {_fold_cta(' '.join(' '.join(caption.lines) for _key, caption in ordered))} "
    if f" {wanted} " in shown:
        return
    warnings.append(
        Issue(
            code="cta_text_missing",
            message=(
                f"cta_text_missing: the brand's call to action wording ('{cta}') is in no "
                "caption: add it as a caption on the end, unless your end card image already "
                "shows it"
            ),
            where="overlays",
        )
    )


def _check_sfx(
    model: Storyboard,
    root: Path,
    starts: list[float],
    total: float,
    errors: list[Issue],
    warnings: list[Issue],
    summary: dict[str, Any],
    check_files: bool,
) -> None:
    """Sound effects: the files exist, none starts after the end, and the automatic ones are
    listed (``summary.sfx``) so the agent sees where they land."""

    auto = model.audio.sfx_auto
    entries = all_sfx(model, starts, total, root)
    if entries:
        summary["sfx"] = entries
    if auto is not None and auto.cuts is None and auto.captions is None:
        warnings.append(
            Issue(
                code="sfx_auto_empty",
                message=(
                    "sfx_auto_empty: audio.sfx_auto names no sound: set cuts and/or captions "
                    "(for example cuts: whoosh_short, captions: pop), or remove it"
                ),
                where="audio.sfx_auto",
            )
        )
    for index, overlay in enumerate(model.overlays):
        if isinstance(overlay, SfxOverlay) and overlay.at >= total:
            errors.append(
                Issue(
                    code="sfx_after_end",
                    message=(
                        f"sfx at {overlay.at:g}s starts at or after the end of the video "
                        f"({total:g}s): it would never be heard."
                    ),
                    where=f"overlays[{index}].at",
                )
            )
    if not check_files:
        return
    wanted: list[tuple[str, str]] = []
    for index, overlay in enumerate(model.overlays):
        if isinstance(overlay, SfxOverlay):
            wanted.append((overlay.source, f"overlays[{index}].source"))
    if auto is not None:
        if auto.cuts is not None:
            wanted.append((auto.cuts, "audio.sfx_auto.cuts"))
        if auto.captions is not None:
            wanted.append((auto.captions, "audio.sfx_auto.captions"))
    checked: dict[str, bool] = {}
    for written, where in wanted:
        try:
            path = resolve_sfx(root, written)
        except (TypeError, ValueError):
            continue
        if str(path) in checked:
            continue
        if not path.is_file():
            checked[str(path)] = False
            errors.append(
                Issue(
                    code="sfx_missing",
                    message=(
                        f"sfx_missing: the sound '{written}' is not in the project. Run "
                        "`adanvil sfx-fetch <project_dir>` (or sfx_list with fetch=true) to "
                        "build the pack in sfx/, or drop your own file there"
                    ),
                    where=where,
                )
            )
            continue
        try:
            checked[str(path)] = bool(probe(path).has_audio)
        except Exception:
            checked[str(path)] = False
        if not checked[str(path)]:
            errors.append(
                Issue(
                    code="sfx_not_audio",
                    message=f"sfx_not_audio: '{written}' has no audio stream to play.",
                    where=where,
                )
            )


def _check_beat_alignment(
    model: Storyboard,
    root: Path,
    starts: list[float],
    total: float,
    info: list[Issue],
    summary: dict[str, Any],
) -> None:
    """Info ``beat_alignment``: the share of cuts that land on a beat of the music, at one
    frame, so an agent making a music-led ad can see whether ``storyboard_snap`` would help."""

    written = model.audio.music or model.audio.from_reference
    cuts = starts[1:]
    if written is None or not cuts:
        return
    path = _relative_file(root, written)
    if path is None or not path.is_file():
        return
    try:
        grid = beat_grid(path, start=model.audio.start, duration=max(total + 1.0, 20.0))
    except Exception:
        return
    if not grid.beats:
        info.append(
            Issue(
                code="beat_alignment",
                message=(
                    "beat_alignment: the music has no steady beat (a pad or a piano), so the "
                    "cuts cannot be placed on it"
                ),
                where="audio",
            )
        )
        return
    frame = 1.0 / model.format.fps
    on_beat = 0
    for cut in cuts:
        beat = grid.nearest(cut)
        if beat is not None and abs(cut - beat) <= frame + 1e-6:
            on_beat += 1
    summary["beat_alignment"] = {
        "cuts": len(cuts),
        "on_beat": on_beat,
        "share": round(on_beat / len(cuts), 2),
        "tempo_bpm": grid.tempo,
    }
    hint = (
        ""
        if on_beat == len(cuts)
        else '; storyboard_snap(to="beats") moves the others onto the nearest beat'
    )
    info.append(
        Issue(
            code="beat_alignment",
            message=(
                f"beat_alignment: {on_beat} of {len(cuts)} cuts land on a beat of the music "
                f"({grid.tempo:g} BPM, within one frame){hint}"
            ),
            where="segments",
        )
    )


def _score_precheck(
    model: Storyboard,
    root: Path,
    brand: Brand | None,
    caption_entries: list[tuple[str, CaptionOverlay]],
    speech_keys: set[str],
    text_bboxes: dict[str, tuple[int, int, int, int]],
    starts: list[float],
    ends: list[float],
    warnings: list[Issue],
    summary: dict[str, Any],
) -> None:
    """The ``score_*`` warnings: the pixel-free checks of ``ad_score`` run on the storyboard."""

    total = model.total_duration()
    entries: list[CaptionEntry] = []
    for key, caption in caption_entries:
        kind = "written" if "." not in key else ("speech" if key in speech_keys else "voice")
        end = total if caption.end is None else min(float(caption.end), total)
        entries.append(
            CaptionEntry(key, caption, kind, float(caption.start), max(float(caption.start), end))
        )
    checks, _entries, _items = storyboard_checks(
        model, root, brand, entries=entries, text_bboxes=text_bboxes
    )
    for finding in checks.findings:
        detail = finding.detail or {}
        where = None
        if "overlay" in detail:
            where = f"overlays[{detail['overlay']}]"
        elif "segment" in detail:
            where = f"segments[{detail['segment']}]"
        when = f" (at {finding.time:.1f}s)" if finding.time is not None else ""
        warnings.append(
            Issue(
                code=f"score_{finding.id}",
                message=f"score_{finding.id}: {finding.message}{when}. Fix: {finding.fix}",
                where=where,
            )
        )
    summary["score_precheck"] = {
        "passed": list(checks.passed),
        "skipped": list(checks.skipped),
    }


def validate_storyboard(
    sb: Storyboard | dict[str, Any],
    *,
    root: Path,
    brand: Brand | None = None,
    recipe: Recipe | None = None,
    check_files: bool = True,
    score_check: bool = False,
) -> ValidationReport:
    """Validate a storyboard without raising on agent-authored input errors.

    With ``score_check`` the quick pre-check of the creative score (``ad_score``) is added as
    warnings with the ``score_`` prefix: the checks that need no pixels (reading time and line
    length, story order and call to action, platform safe zones of the overlays). They are
    guidelines to review, like every warning; ``summary["score_precheck"]`` lists what passed
    and what could not be checked."""

    raw = sb if isinstance(sb, Mapping) else None
    requested_recipe_id = (
        recipe.id
        if recipe is not None
        else (str(raw.get("recipe")) if raw and raw.get("recipe") is not None else None)
    )
    report_summary = _summary(raw, requested_recipe_id)
    errors: list[Issue] = []
    warnings: list[Issue] = []
    notes: list[Issue] = []

    if isinstance(sb, Storyboard):
        model = sb
    else:
        try:
            model = Storyboard.model_validate(sb)
        except ValidationError as exc:
            errors.extend(_schema_issues(exc))
            return ValidationReport(
                ok=False, errors=errors, warnings=warnings, summary=report_summary
            )
        except (TypeError, ValueError) as exc:
            errors.append(Issue(code="schema", message=f"Invalid storyboard: {exc}"))
            return ValidationReport(
                ok=False, errors=errors, warnings=warnings, summary=report_summary
            )

    total = model.total_duration()
    if not total <= MAX_SECONDS:
        errors.append(
            Issue(
                code="too_long",
                message=(
                    f"too_long: the storyboard lasts {total:g}s; AdAnvil renders at most "
                    f"{MAX_SECONDS:g}s. Remove segments or shorten them."
                ),
                where="segments",
            )
        )
    report_summary.update(
        {
            "total_duration": total,
            "segment_count": len(model.segments),
            "overlay_count": len(model.overlays),
        }
    )
    project_root = Path(root)

    voice_tracks: dict[str, VoiceTrack] = {}
    timeline = model.timeline()
    starts = [item[0] for item in timeline]
    ends = [item[1] for item in timeline]
    speech_library = SpeechLibrary(
        project_root,
        lists_from_brand(brand) if brand is not None else brand_filler_lists(project_root),
    )
    speech_keys: set[str] = set()  # expanded speech captions: shown text, not claims

    for index, segment in enumerate(model.segments):
        if (
            isinstance(segment, ClipSegment)
            and segment.audio_from is not None
            and not model.audio.clips
        ):
            errors.append(
                Issue(
                    code="audio_from",
                    message=(
                        "audio_from only works with audio.clips: true (it replaces the sound "
                        "this clip contributes to the clip-sound bed)"
                    ),
                    where=f"segments[{index}].audio_from",
                )
            )

    def load_voice_manifest(written: str | None, where: str) -> VoiceTrack | None:
        if not written:
            errors.append(
                Issue(
                    code="voice_manifest",
                    message="voice_manifest: a voice manifest is required",
                    where=where,
                )
            )
            return None
        resolved = _relative_file(project_root, written)
        if resolved is None or not resolved.is_file():
            errors.append(
                Issue(
                    code="voice_manifest",
                    message=f"voice_manifest: manifest '{written}' does not exist",
                    where=where,
                )
            )
            return None
        key = str(resolved)
        if key in voice_tracks:
            return voice_tracks[key]
        try:
            track = load_track(resolved)
        except (OSError, ValueError) as exc:
            errors.append(
                Issue(
                    code="voice_manifest",
                    message=f"voice_manifest: could not parse '{written}': {exc}",
                    where=where,
                )
            )
            return None
        voice_tracks[key] = track
        return track

    audio_track: VoiceTrack | None = None
    if model.audio.voice is not None:
        audio_track = load_voice_manifest(model.audio.voice, "audio.voice")
        if audio_track is not None:
            voice_end = model.audio.voice_offset + audio_track.total_duration
            if total < voice_end - 0.05:
                errors.append(
                    Issue(
                        code="voice_cut",
                        message="the video ends before the voice",
                        where="audio.voice",
                    )
                )
            if (
                model.audio.music is not None or model.audio.from_reference is not None
            ) and total - voice_end < 1.0:
                warnings.append(
                    Issue(
                        code="voice_tail",
                        message="keep about 1.5 s of music after the last sentence",
                        where="audio.voice",
                    )
                )
            for line in audio_track.lines:
                if line.status == "suspect":
                    unmatched = (
                        f" (unmatched words: {', '.join(line.unmatched_words)})"
                        if line.unmatched_words
                        else ""
                    )
                    warnings.append(
                        Issue(
                            code="voice_suspect",
                            message=(
                                f"voice line {line.index} is suspect: {line.transcript}"
                                f"{unmatched}"
                            ),
                            where=f"audio.voice.lines[{line.index}]",
                        )
                    )

    panel_infos: dict[tuple[int, int], Any] = {}
    inset_infos: dict[int, Any] = {}
    if check_files:
        seen_paths: set[tuple[str, str]] = set()
        probed: dict[Path, Any] = {}

        def check_reference(written: str, *, kind: str, where: str) -> Any | None:
            key = (written, kind)
            if key in seen_paths:
                resolved = _relative_file(project_root, written)
                return probed.get(resolved) if resolved is not None else None
            seen_paths.add(key)
            resolved = _relative_file(project_root, written)
            if resolved is None:
                errors.append(
                    Issue(
                        code="path_escape",
                        message=(
                            f"Path '{written}' is absolute or leaves the project folder; "
                            "use a path relative to the project root."
                        ),
                        where=where,
                    )
                )
                return None
            if not resolved.is_file():
                errors.append(
                    Issue(
                        code="missing_file",
                        message=(
                            f"Referenced file '{written}' does not exist under the project root."
                        ),
                        where=where,
                    )
                )
                return None
            if resolved not in probed:
                try:
                    probed[resolved] = probe(resolved)
                except Exception as exc:  # ffprobe errors are user-facing file issues here.
                    errors.append(
                        Issue(
                            code="file_probe",
                            message=f"Could not probe referenced file '{written}': {exc}",
                            where=where,
                        )
                    )
                    return None
            info = probed[resolved]
            if kind == "clip" and info.kind != "video":
                errors.append(
                    Issue(
                        code="source_type",
                        message=(
                            f"Segment source '{written}' is {info.kind}, but a video is required."
                        ),
                        where=where,
                    )
                )
            elif kind == "image" and info.kind != "image":
                errors.append(
                    Issue(
                        code="source_type",
                        message=f"Image source '{written}' is {info.kind}, not an image.",
                        where=where,
                    )
                )
            elif kind == "visual" and info.kind == "audio":
                errors.append(
                    Issue(
                        code="source_type",
                        message=f"Panel source '{written}' is audio, but a video or an image "
                        "is required.",
                        where=where,
                    )
                )
            elif kind == "audio" and info.kind != "audio":
                errors.append(
                    Issue(
                        code="source_type",
                        message=f"Music source '{written}' is {info.kind}, not audio.",
                        where=where,
                    )
                )
            return info

        for index, segment in enumerate(model.segments):
            if isinstance(segment, SplitSegment):
                for number, panel in enumerate(segment.panels):
                    panel_infos[(index, number)] = check_reference(
                        panel.source,
                        kind="visual",
                        where=f"segments[{index}].panels[{number}].source",
                    )
                continue
            if isinstance(segment, RevealSegment):
                for number, name in enumerate(("before", "after")):
                    panel_infos[(index, number)] = check_reference(
                        (segment.before if number == 0 else segment.after).source,
                        kind="visual",
                        where=f"segments[{index}].{name}.source",
                    )
                continue
            where = f"segments[{index}].source"
            info = check_reference(
                segment.source,
                kind="clip" if isinstance(segment, ClipSegment) else "image",
                where=where,
            )
            if (
                isinstance(segment, ImageSegment)
                and segment.crop is not None
                and info is not None
                and info.width
                and info.height
            ):
                crop = segment.crop
                if (
                    crop.w <= 0
                    or crop.h <= 0
                    or crop.x < 0
                    or crop.y < 0
                    or crop.x + crop.w > info.width
                    or crop.y + crop.h > info.height
                ):
                    errors.append(
                        Issue(
                            code="panel_crop",
                            message=(
                                f"panel_crop: the crop of segments[{index}] leaves the "
                                f"{info.width}x{info.height} source picture."
                            ),
                            where=f"segments[{index}].crop",
                        )
                    )
            if isinstance(segment, ClipSegment) and (
                info is not None
                and info.duration is not None
                and segment.out > info.duration + 0.05
            ):
                errors.append(
                    Issue(
                        code="source_duration",
                        message=(
                            f"Clip segment {index} out point {segment.out} exceeds source "
                            f"'{segment.source}' duration {info.duration:.3f}."
                        ),
                        where=f"segments[{index}].out",
                    )
                )
            if (
                isinstance(segment, ClipSegment)
                and info is not None
                and info.width
                and info.height
                and (segment.crop is not None or segment.zoom is not None or segment.punch)
            ):
                enlarged = source_enlargement(segment, info.width, info.height, model.format)
                if enlarged > _MAX_ENLARGEMENT:
                    warnings.append(
                        Issue(
                            code="digital_zoom",
                            message=(
                                f"digital_zoom: segment {index} enlarges the source "
                                f"{enlarged:.1f}x: a crop this tight looks zoomed and soft; "
                                "pick a shot filmed close instead"
                            ),
                            where=f"segments[{index}]",
                        )
                    )

        for index, segment in enumerate(model.segments):
            if not (isinstance(segment, ClipSegment) and segment.audio_from is not None):
                continue
            where = f"segments[{index}].audio_from.source"
            source_info = check_reference(segment.audio_from.source, kind="audio_from", where=where)
            if source_info is None:
                continue
            if source_info.kind == "image" or not source_info.has_audio:
                errors.append(
                    Issue(
                        code="audio_from",
                        message=(
                            f"audio_from source '{segment.audio_from.source}' has no audio "
                            "stream to play"
                        ),
                        where=where,
                    )
                )
            elif (
                source_info.audio_duration is not None
                and segment.audio_from.in_ + (ends[index] - starts[index])
                > source_info.audio_duration + 0.05
            ):
                errors.append(
                    Issue(
                        code="audio_from",
                        message=(
                            f"audio_from needs {ends[index] - starts[index]:.2f}s of "
                            f"'{segment.audio_from.source}' from {segment.audio_from.in_:g}s, "
                            f"but its audio lasts {source_info.audio_duration:.2f}s"
                        ),
                        where=f"segments[{index}].audio_from.in",
                    )
                )

        if model.audio.music is not None:
            check_reference(model.audio.music, kind="audio", where="audio.music")
        if model.audio.from_reference is not None:
            check_reference(
                model.audio.from_reference,
                kind="reference",
                where="audio.from_reference",
            )

        for index, overlay in enumerate(model.overlays):
            if isinstance(overlay, ImageOverlay):
                check_reference(overlay.source, kind="image", where=f"overlays[{index}].source")
            elif isinstance(overlay, VideoOverlay):
                inset_infos[index] = check_reference(
                    overlay.source, kind="clip", where=f"overlays[{index}].source"
                )
            elif isinstance(overlay, StickerOverlay) and overlay.kind == "image":
                check_reference(
                    overlay.source or "", kind="image", where=f"overlays[{index}].source"
                )

        for field_name, value in (("storyboard.emoji", model.emoji),):
            if value not in {"twemoji", "none"}:
                emoji_path = _relative_file(project_root, value)
                if emoji_path is None or not emoji_path.is_dir():
                    errors.append(
                        Issue(
                            code="missing_file",
                            message=(
                                f"Emoji folder '{value}' does not exist under the project root."
                            ),
                            where=field_name,
                        )
                    )
        if brand is not None and brand.visual is not None:
            emoji = brand.visual.emoji
            if emoji not in {None, "twemoji", "none"}:
                emoji_path = _relative_file(project_root, str(emoji))
                if emoji_path is None or not emoji_path.is_dir():
                    errors.append(
                        Issue(
                            code="missing_file",
                            message=(
                                f"Brand emoji folder '{emoji}' does not exist under the "
                                "project root."
                            ),
                            where="brand.visual.emoji",
                        )
                    )

    for index, segment in enumerate(model.segments):
        if isinstance(segment, ClipSegment) and segment.in_ < 0:
            errors.append(
                Issue(
                    code="segment_in",
                    message=(f"Clip segment {index} has a negative in point ({segment.in_})."),
                    where=f"segments[{index}].in",
                )
            )

    total_for_timeline = total
    text_bboxes: dict[str, tuple[int, int, int, int]] = {}
    caption_entries: list[tuple[str, CaptionOverlay]] = []
    render_entries: list[tuple[str, Any]] = []
    for index, overlay in enumerate(model.overlays):
        key = str(index)
        if isinstance(overlay, SfxOverlay):
            continue  # a sound, not a picture: checked with the audio below
        overlay = _checked_font(overlay, project_root, index, warnings)
        if isinstance(overlay, SpeechCaptionsOverlay):
            if not model.audio.clips:
                errors.append(
                    Issue(
                        code="speech_captions",
                        message=(
                            "speech_captions: the captions follow the speech of the clips, "
                            "so audio.clips must be true (the sound of the clips is kept)"
                        ),
                        where=f"overlays[{index}]",
                    )
                )
                continue
            corrections_used: dict[str, int] = {}
            try:
                expanded = expand_speech_captions(
                    overlay,
                    model,
                    speech_library,
                    starts,
                    ends,
                    corrections_used=corrections_used,
                )
            except (TypeError, ValueError, KeyError) as exc:
                errors.append(
                    Issue(
                        code="speech_captions",
                        message=f"speech_captions: could not build the captions: {exc}",
                        where=f"overlays[{index}]",
                    )
                )
                continue
            if overlay.corrections:
                listing = ", ".join(
                    f"'{heard}' -> '{shown}' ({corrections_used.get(heard, 0)}x)"
                    for heard, shown in overlay.corrections.items()
                )
                warnings.append(
                    Issue(
                        code="speech_corrections",
                        message=(
                            "speech_corrections: the captions show words that differ from the "
                            f"transcript: {listing}. A human should check each one against the "
                            "audio: a correction fixes a spelling, it must never put words in "
                            "the speaker's mouth (a word never heard is listed as 0x). When the "
                            "transcript is often wrong, run speech_refine on the clip instead"
                        ),
                        where=f"overlays[{index}].corrections",
                    )
                )
            if not expanded:
                warnings.append(
                    Issue(
                        code="speech_missing",
                        message=(
                            "speech_missing: no spoken word was found for the speech_captions "
                            "overlay: index the clips with speech (library_index, "
                            "transcribe=true) or check the played ranges"
                        ),
                        where=f"overlays[{index}]",
                    )
                )
            for expanded_index, caption in enumerate(expanded):
                expanded_key = f"{index}.{expanded_index}"
                caption_entries.append((expanded_key, caption))
                render_entries.append((expanded_key, caption))
                speech_keys.add(expanded_key)
            continue
        if isinstance(overlay, VoiceCaptionsOverlay):
            written_manifest = overlay.manifest or model.audio.voice
            track = load_voice_manifest(written_manifest, f"overlays[{index}].manifest")
            if track is None:
                continue
            try:
                expanded = overlay.expand(
                    track,
                    model.audio.voice_offset if overlay.offset is None else overlay.offset,
                )
            except (TypeError, ValueError, KeyError) as exc:
                errors.append(
                    Issue(
                        code="voice_manifest",
                        message=f"voice_manifest: could not expand captions: {exc}",
                        where=f"overlays[{index}]",
                    )
                )
                continue
            for expanded_index, caption in enumerate(expanded):
                expanded_key = f"{index}.{expanded_index}"
                caption_entries.append((expanded_key, caption))
                render_entries.append((expanded_key, caption))
            continue

        if isinstance(overlay, CaptionOverlay):
            caption_entries.append((key, overlay))
            chunks = overlay.expand_words(total)
            if len(chunks) > 1:  # a plain "word" caption is shown one or two words at a time
                render_entries.extend((f"{key}.{n}", chunk) for n, chunk in enumerate(chunks))
                continue
        render_entries.append((key, overlay))

    for key, overlay in render_entries:
        if overlay.start >= total_for_timeline:
            errors.append(
                Issue(
                    code="overlay_after_end",
                    message=(
                        f"Overlay {key} starts at {overlay.start}, after the video ends at "
                        f"{total_for_timeline}."
                    ),
                    where=f"overlays[{key}].start",
                )
            )
        if overlay.end is not None and overlay.end > total_for_timeline + 0.05:
            warnings.append(
                Issue(
                    code="overlay_clamped",
                    message=(
                        f"Overlay {key} ends at {overlay.end}, after the video ends at "
                        f"{total_for_timeline}; it will be clamped."
                    ),
                    where=f"overlays[{key}].end",
                )
            )
        if overlay.end is not None and overlay.end <= overlay.start:
            errors.append(
                Issue(
                    code="overlay_timing",
                    message=(
                        f"Overlay {key} end {overlay.end} must be greater than "
                        f"start {overlay.start}."
                    ),
                    where=f"overlays[{key}]",
                )
            )

        if isinstance(overlay, (CaptionOverlay, CommentOverlay)):
            try:
                bbox, overflow = _render_text_overlay(
                    overlay,
                    root=project_root,
                    width=model.format.width,
                    height=model.format.height,
                    brand=brand,
                )
                text_bboxes[key] = bbox
                if overflow:
                    warnings.append(
                        Issue(
                            code="text_overflow",
                            message=(
                                f"Text in overlay {key} still overflows its maximum width "
                                "after fitting."
                            ),
                            where=f"overlays[{key}]",
                        )
                    )
                if (bbox == (0, 0, 0, 0) and not overflow) or (
                    bbox[0] < 0
                    or bbox[1] < 0
                    or bbox[2] > model.format.width
                    or bbox[3] > model.format.height
                ):
                    errors.append(
                        Issue(
                            code="bbox_out_of_canvas",
                            message=(
                                f"Rendered text bbox {bbox} for overlay {key} is outside the "
                                f"{model.format.width}x{model.format.height} canvas."
                            ),
                            where=f"overlays[{key}]",
                        )
                    )
            except Exception as exc:
                code = (
                    "font_not_found"
                    if "font" in str(exc).lower() or "family" in str(exc).lower()
                    else "text_render"
                )
                errors.append(
                    Issue(
                        code=code,
                        message=f"Could not render text in overlay {key}: {exc}",
                        where=f"overlays[{key}]",
                    )
                )

    if model.emoji not in {"twemoji", "none"}:
        emoji_dir = _relative_file(project_root, model.emoji)
        if emoji_dir is not None and emoji_dir.is_dir():
            emoji_provider = DirectoryProvider(emoji_dir)
            for key, overlay in render_entries:
                if isinstance(overlay, CaptionOverlay):
                    texts = list(overlay.lines)
                elif isinstance(overlay, CommentOverlay):
                    texts = [overlay.text]
                else:
                    continue
                absent = missing_emoji(texts, emoji_provider)
                if absent:
                    warnings.append(
                        Issue(
                            code="emoji_missing",
                            message=(
                                f"emoji_missing: {' '.join(absent)} has no PNG in the emoji "
                                f"folder '{model.emoji}' (overlay {key}: "
                                f"{' / '.join(texts)!r}); it would render without it."
                            ),
                            where=f"overlays[{key}]",
                        )
                    )

    if brand is not None:
        if brand.visual is not None and brand.visual.caption_family is not None:
            if _family_for_render(brand.visual.caption_family, project_root) is None:
                warnings.append(
                    Issue(
                        code="font_not_found",
                        message=(
                            "font_not_found: brand.visual.caption_family: "
                            + unusable_family_message(brand.visual.caption_family)
                        ),
                        where="brand.visual.caption_family",
                    )
                )
        banned = brand.claims.banned_words if brand.claims is not None else []
        for key, overlay in render_entries:
            if key in speech_keys:
                continue  # the spoken-word check reports these (banned_word_spoken)
            if isinstance(overlay, CaptionOverlay):
                texts = overlay.lines
            elif isinstance(overlay, CommentOverlay):
                texts = [overlay.text]
            elif isinstance(overlay, StickerOverlay) and overlay.text:
                texts = [overlay.text]
            else:
                texts = []
            for line_number, text in enumerate(texts):
                for word in find_banned(text, banned):
                    errors.append(
                        Issue(
                            code="banned_word",
                            message=f"Banned word '{word}' found in overlay {key}.",
                            where=(
                                f"overlays[{key}].lines[{line_number}]"
                                if isinstance(overlay, CaptionOverlay)
                                else f"overlays[{key}].text"
                            ),
                        )
                    )
        for index, segment in enumerate(model.segments):
            for number, text in enumerate(
                segment.labels or [] if isinstance(segment, SplitSegment | RevealSegment) else []
            ):
                for word in find_banned(text, banned):
                    errors.append(
                        Issue(
                            code="banned_word",
                            message=(
                                f"Banned word '{word}' found in the label of "
                                f"segments[{index}]."
                            ),
                            where=f"segments[{index}].labels[{number}]",
                        )
                    )

    if brand is not None and brand.cta_image:
        last = model.segments[-1]
        wanted = brand.cta_image.replace("\\", "/").removeprefix("./")
        written = last.source.replace("\\", "/").removeprefix("./")
        explained = (last.label or "").startswith("end_card") and any(
            gap.type == "other" for gap in model.gaps
        )
        if written != wanted and not explained:
            warnings.append(
                Issue(
                    code="cta_image_unused",
                    message=(
                        f"cta_image_unused: brand.yaml has a ready-made CTA image "
                        f"({brand.cta_image}) but the last segment is {last.source}: end on "
                        "the CTA image, or label the last segment end_card and give the "
                        "reason in gaps (type other)"
                    ),
                    where=f"segments[{len(model.segments) - 1}]",
                )
            )

    caption_renders = [
        (key, overlay) for key, overlay in render_entries if isinstance(overlay, CaptionOverlay)
    ]
    check_splits(model, panel_infos, errors, warnings)
    check_reveals(model, panel_infos, errors, warnings)
    check_insets(
        model, inset_infos, starts, ends, total, text_bboxes, caption_renders, errors, warnings
    )
    emoji_folder = (
        _relative_file(project_root, model.emoji)
        if model.emoji not in {"twemoji", "none"}
        else None
    )

    def sticker_source_exists(written: str) -> bool:
        resolved = _relative_file(project_root, written)
        return check_files and resolved is not None and resolved.is_file()

    check_stickers(
        model,
        project_root,
        sticker_source_exists,
        errors,
        warnings,
        emoji_folder if emoji_folder is not None and emoji_folder.is_dir() else None,
    )

    _check_emphasis(caption_entries, warnings)
    _check_end_card_captions(model, brand, caption_entries, starts, ends, total, warnings)
    _check_cta_text(model, brand, caption_entries, warnings)
    if check_files and not errors:
        _check_card_content(model, project_root, warnings)

    _check_sfx(model, project_root, starts, total, errors, warnings, report_summary, check_files)
    if check_files and not errors:
        _check_beat_alignment(model, project_root, starts, total, notes, report_summary)

    if model.audio.clips:
        banned_spoken = (
            list(brand.claims.banned_words) if brand is not None and brand.claims else []
        )
        _check_spoken_audio(model, speech_library, banned_spoken, starts, ends, warnings)

    selected_recipe = recipe
    if selected_recipe is None and model.recipe is not None:
        try:
            selected_recipe = get_recipe(model.recipe)
        except KeyError as exc:
            errors.append(Issue(code="recipe_not_found", message=str(exc), where="recipe"))
    if selected_recipe is not None:
        report_summary["recipe_id"] = selected_recipe.id
        report_summary["checklist"] = list(selected_recipe.checklist)
        _check_recipe_rules(
            model,
            selected_recipe,
            total=total,
            caption_entries=caption_entries,
            text_bboxes=text_bboxes,
            errors=errors,
            warnings=warnings,
        )

    _check_emotions(model, warnings)
    _check_story_order(model, warnings)

    if score_check and not errors:
        _score_precheck(
            model,
            project_root,
            brand,
            caption_entries,
            speech_keys,
            text_bboxes,
            starts,
            ends,
            warnings,
            report_summary,
        )

    reference_comparison = None
    if model.reference is not None:
        reference_comparison = _check_reference(
            model,
            project_root,
            written_captions=sum(1 for key, _ in caption_entries if "." not in key),
            warnings=warnings,
        )

    return ValidationReport(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        summary=report_summary,
        info=notes,
        reference_comparison=reference_comparison,
    )


__all__ = ["Issue", "ValidationReport", "validate_storyboard"]
