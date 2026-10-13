"""Creative checks of an ad that need no pixels: what a storyboard already tells.

They feed two things: the pixel-free part of ``ad_score`` (``adanvil.qc.score``), and the
optional ``score_*`` warnings of ``storyboard_validate`` (``score_check=true``), so the problems
that cost points are visible before a render. Every threshold below is a guideline, not a rule.

Thresholds
==========

Reading (``reading_time``, ``reading_long_line``)
    A written caption needs ``READING_PER_WORD`` (0.25 s) per word plus ``READING_BASE``
    (0.6 s), and never less than ``READING_MIN`` (1.2 s): 4 words need 1.6 s, 8 words 2.6 s.
    A caption line has at most ``MAX_WORDS_PER_LINE`` (7) words.

Story (``product_too_early``, ``product_never_shown``, ``cta_missing``, ``cta_wording``,
``cta_not_at_end``)
    The product (``shot_type`` ``product_hero``, ``product_in_use`` or ``unboxing``, or subject
    ``product``) must not appear before the first solution role (reveal, mechanism, demo,
    proof, payoff, offer, cta), and must appear at least once when the segments are annotated.
    The call to action must be written or spoken: the brand's ``product.cta_text`` when it has
    one, else any wording of ``CTA_PHRASES``; ending on the brand's CTA image, or on a segment
    labelled ``end_card``, counts as showing it. An end card, or a call to action, must be on
    screen in the last ``LAST_SECONDS`` (4 s).

Platform safe zones (``safe_zone``): see ``adanvil.safezones`` for the boxes.

A finding costs points of its area's sub-score (``penalty``): ``high`` findings cost 18 to 30,
``medium`` ones 6 to 12 and ``low`` ones 2 to 6. A sub-score starts at 100.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adanvil.brand import Brand
from adanvil.safezones import PLATFORMS, SAFE_ZONES, collisions

from .schema import (
    CaptionOverlay,
    CommentOverlay,
    ImageOverlay,
    SpeechCaptionsOverlay,
    StickerOverlay,
    Storyboard,
    VoiceCaptionsOverlay,
)

READING_PER_WORD = 0.25
READING_BASE = 0.6
READING_MIN = 1.2
MAX_WORDS_PER_LINE = 7
LAST_SECONDS = 4.0
PRODUCT_TYPES = ("product_hero", "product_in_use", "unboxing")
SOLUTION_ROLES = ("reveal", "mechanism", "demo", "proof", "payoff", "offer", "cta")

# Wordings of a call to action when the brand gives none (folded: lower case, no accents).
CTA_PHRASES = (
    "shop now", "shop the", "learn more", "order now", "order today", "order yours",
    "get yours", "get started", "get it", "buy now", "try it", "try now", "sign up",
    "download", "book now", "book your", "claim yours", "claim your", "swipe up", "tap the",
    "tap below",
    "link in bio", "use code", "free trial", "start now", "start today", "visit",
    "commande", "commandez", "achete", "achetez", "decouvre", "decouvrez", "profite",
    "profitez", "essaie", "essayez", "reserve", "reservez", "telecharge", "telechargez",
    "clique", "cliquez", "inscris", "inscrivez", "lien en bio", "code promo",
)  # fmt: skip

_QUOTES = re.compile("[‘’‚‛“”„«»\"'`]")
_MARKS = re.compile(r"\*([^*]+)\*")


def fold_text(text: str) -> str:
    """Case, accents, quotes and punctuation removed: the form a wording is looked for in."""

    folded = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(char for char in folded if not unicodedata.combining(char))
    plain = _QUOTES.sub("", plain)
    return " ".join(re.sub(r"[^\w]+", " ", plain, flags=re.UNICODE).split())


def plain_words(text: str) -> list[str]:
    """The words of ``text`` without the ``*emphasis*`` stars."""

    return _MARKS.sub(r"\1", text).split()


# ------------------------------------------------------------------------------- results


@dataclass
class Finding:
    """One problem with a time, a severity and a fix.

    ``penalty`` is the number of points taken off the sub-score of ``area``; the global impact
    is that penalty weighted by the area's share of the total score.
    """

    id: str
    area: str
    severity: str
    message: str
    fix: str
    time: float | None = None
    end: float | None = None
    penalty: float = 0.0
    detail: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "area": self.area,
            "severity": self.severity,
            "time": None if self.time is None else round(self.time, 2),
            "message": self.message,
            "fix": self.fix,
        }
        if self.end is not None:
            payload["end"] = round(self.end, 2)
        if self.detail:
            payload["detail"] = self.detail
        return payload


@dataclass
class Checks:
    """What a group of checks found: the findings, the checks that passed, those skipped."""

    findings: list[Finding] = field(default_factory=list)
    passed: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def ok(self, area: str, check: str) -> None:
        self.passed.append(f"{area}.{check}")

    def skip(self, area: str, check: str, why: str) -> None:
        self.skipped.append({"check": f"{area}.{check}", "why": why})

    def extend(self, other: Checks) -> None:
        self.findings.extend(other.findings)
        self.passed.extend(other.passed)
        self.skipped.extend(other.skipped)


@dataclass
class CaptionEntry:
    """A caption of the ad with its place on the timeline. ``kind`` is ``written`` (a caption
    overlay of the storyboard), ``voice`` or ``speech`` (a chunk expanded from a voice track or
    from the speech of the clips)."""

    key: str
    caption: CaptionOverlay
    kind: str
    start: float
    end: float

    @property
    def text(self) -> str:
        return " ".join(self.caption.lines)

    @property
    def words(self) -> int:
        return len(plain_words(self.text))


def _clip(text: str, limit: int = 40) -> str:
    plain = " ".join(_MARKS.sub(r"\1", text).split())
    return plain if len(plain) <= limit else plain[: limit - 1].rstrip() + "..."


# --------------------------------------------------------------------------- the captions


def collect_captions(
    model: Storyboard,
    root: Path,
    brand: Brand | None,
    starts: list[float],
    ends: list[float],
) -> tuple[list[CaptionEntry], list[str]]:
    """Every caption of the storyboard on the timeline: the written ones, and the chunks of the
    voice and speech captions that can be built (a voice manifest or a transcript that is
    missing is reported in the notes, not raised)."""

    from adanvil.speech.analyze import brand_filler_lists, lists_from_brand
    from adanvil.speech.captions import expand_speech_captions
    from adanvil.speech.library import SpeechLibrary
    from adanvil.voice import load_track

    from .validate import _relative_file

    total = model.total_duration()
    notes: list[str] = []
    entries: list[CaptionEntry] = []
    library: SpeechLibrary | None = None

    def clamp(caption: CaptionOverlay) -> tuple[float, float]:
        end = total if caption.end is None else min(float(caption.end), total)
        return float(caption.start), max(float(caption.start), end)

    for index, overlay in enumerate(model.overlays):
        key = str(index)
        if isinstance(overlay, CaptionOverlay):
            start, end = clamp(overlay)
            entries.append(CaptionEntry(key, overlay, "written", start, end))
        elif isinstance(overlay, VoiceCaptionsOverlay):
            manifest = overlay.manifest or model.audio.voice
            path = _relative_file(root, manifest) if manifest else None
            if path is None or not path.is_file():
                notes.append(f"overlays[{index}]: voice manifest not found, captions not counted")
                continue
            try:
                track = load_track(path)
                offset = model.audio.voice_offset if overlay.offset is None else overlay.offset
                chunks = overlay.expand(track, offset)
            except (OSError, TypeError, ValueError, KeyError) as exc:
                notes.append(f"overlays[{index}]: voice captions could not be built ({exc})")
                continue
            for number, chunk in enumerate(chunks):
                start, end = clamp(chunk)
                entries.append(CaptionEntry(f"{index}.{number}", chunk, "voice", start, end))
        elif isinstance(overlay, SpeechCaptionsOverlay):
            if not model.audio.clips:
                continue
            if library is None:
                library = SpeechLibrary(
                    root,
                    lists_from_brand(brand) if brand is not None else brand_filler_lists(root),
                )
            try:
                chunks = expand_speech_captions(overlay, model, library, starts, ends)
            except (OSError, TypeError, ValueError, KeyError) as exc:
                notes.append(f"overlays[{index}]: speech captions could not be built ({exc})")
                continue
            for number, chunk in enumerate(chunks):
                start, end = clamp(chunk)
                entries.append(CaptionEntry(f"{index}.{number}", chunk, "speech", start, end))
    return entries, notes


def voice_lines(model: Storyboard, root: Path) -> list[tuple[float, str]]:
    """``(start on the timeline, shown or spoken text)`` of each line of the voice-over."""

    from adanvil.voice import load_track

    from .validate import _relative_file

    if model.audio.voice is None:
        return []
    path = _relative_file(root, model.audio.voice)
    if path is None or not path.is_file():
        return []
    try:
        track = load_track(path)
    except (OSError, ValueError):
        return []
    return [
        (model.audio.voice_offset + line.start, line.display or line.text) for line in track.lines
    ]


# ---------------------------------------------------------------------------- reading


def reading_need(words: int) -> float:
    """Seconds a written caption of ``words`` words must stay on screen."""

    return max(READING_MIN, READING_PER_WORD * words + READING_BASE)


def reading_checks(entries: Iterable[CaptionEntry]) -> Checks:
    """Reading time of the written captions and the length of every caption line."""

    checks = Checks()
    written = [entry for entry in entries if entry.kind == "written"]
    entries = list(entries)
    if not entries:
        checks.skip("reading", "caption_time", "no caption in the storyboard")
        return checks
    short: list[Finding] = []
    for entry in written:
        need = reading_need(entry.words)
        shown = entry.end - entry.start
        if shown >= need - 0.05:
            continue
        severity = "high" if shown < 0.6 * need else "medium"
        short.append(
            Finding(
                id="reading_time",
                area="reading",
                severity=severity,
                time=entry.start,
                end=entry.end,
                penalty=12.0 if severity == "high" else 6.0,
                message=(
                    f"caption {entry.key} ({_clip(entry.text)!r}) stays {shown:.1f}s on screen "
                    f"but its {entry.words} words need {need:.1f}s "
                    f"({READING_PER_WORD:g}s per word + {READING_BASE:g}s, at least "
                    f"{READING_MIN:g}s)"
                ),
                fix=(
                    f"keep it on screen until {entry.start + need:.1f}s (set end)"
                    + (
                        f", or cut it to {int((shown - READING_BASE) / READING_PER_WORD)} words"
                        if shown >= READING_MIN
                        else ""
                    )
                ),
                detail={"overlay": entry.key, "words": entry.words, "shown": round(shown, 2)},
            )
        )
    short.sort(key=lambda item: -item.penalty)
    for finding in short[:6]:
        checks.add(finding)
    if len(short) > 6:
        checks.add(
            Finding(
                id="reading_time",
                area="reading",
                severity="medium",
                penalty=min(10.0, 2.0 * (len(short) - 6)),
                time=short[6].time,
                message=f"{len(short) - 6} more captions are on screen for too short a time",
                fix="lengthen them or shorten their text",
            )
        )
    if written and not short:
        checks.ok("reading", "caption_time")
    elif not written:
        checks.skip("reading", "caption_time", "no written caption (voice or speech chunks only)")

    long_lines = [
        (entry, max(len(plain_words(line)) for line in entry.caption.lines))
        for entry in entries
        if max(len(plain_words(line)) for line in entry.caption.lines) > MAX_WORDS_PER_LINE
    ]
    if long_lines:
        first, words = long_lines[0]
        checks.add(
            Finding(
                id="reading_long_line",
                area="reading",
                severity="medium",
                time=first.start,
                penalty=min(14.0, 6.0 + 2.0 * (len(long_lines) - 1)),
                message=(
                    f"{len(long_lines)} caption(s) have a line of more than "
                    f"{MAX_WORDS_PER_LINE} words (caption {first.key}, {words} words: "
                    f"{_clip(first.text)!r})"
                ),
                fix=f"break the text into chunks of {MAX_WORDS_PER_LINE} words or fewer per line",
                detail={"overlays": [entry.key for entry, _ in long_lines][:10]},
            )
        )
    else:
        checks.ok("reading", "line_length")
    return checks


# ------------------------------------------------------------------------------ story


def _shows_product(segment: Any) -> bool:
    return getattr(segment, "shot_type", None) in PRODUCT_TYPES or (
        getattr(segment, "subject", None) == "product"
    )


def story_checks(
    model: Storyboard,
    brand: Brand | None,
    entries: Iterable[CaptionEntry],
    spoken: Iterable[tuple[float, str]],
    starts: list[float],
    ends: list[float],
) -> Checks:
    """Problem before product, product shown, call to action present and near the end."""

    checks = Checks()
    total = model.total_duration()
    roles = [getattr(segment, "role", None) for segment in model.segments]
    turn = next((i for i, role in enumerate(roles) if role in SOLUTION_ROLES), None)
    if turn is None:
        checks.skip("story", "problem_before_product", "no segment has a solution role")
    else:
        early = [i for i in range(turn) if _shows_product(model.segments[i])]
        for index in early[:3]:
            checks.add(
                Finding(
                    id="product_too_early",
                    area="story",
                    severity="medium",
                    time=starts[index],
                    end=ends[index],
                    penalty=10.0,
                    message=(
                        f"segments[{index}] shows the product at {starts[index]:.1f}s, before "
                        f"the story turns to the solution (segments[{turn}], role "
                        f"{roles[turn]}, {starts[turn]:.1f}s)"
                    ),
                    fix=(
                        "open on people living the problem (a worried face, a messy kitchen, no "
                        "product), then show that a solution exists"
                    ),
                    detail={"segment": index},
                )
            )
        if not early:
            checks.ok("story", "problem_before_product")

    annotated = [
        s for s in model.segments if getattr(s, "shot_type", None) or getattr(s, "subject", None)
    ]
    if not annotated:
        checks.skip(
            "story",
            "product_shown",
            "no segment has a shot_type or subject: annotate the segments to check it",
        )
    elif any(_shows_product(segment) for segment in model.segments):
        checks.ok("story", "product_shown")
    else:
        checks.add(
            Finding(
                id="product_never_shown",
                area="story",
                severity="high",
                time=None,
                penalty=25.0,
                message="no segment shows the product (shot_type product_hero, product_in_use, "
                "unboxing, or subject product)",
                fix="add at least one shot of the product as sold, after the problem part",
            )
        )

    entries = list(entries)
    cta = brand.product.cta_text if brand is not None and brand.product is not None else None
    last = model.segments[-1] if model.segments else None
    last_label = (getattr(last, "label", None) or "") if last is not None else ""
    brand_card = bool(
        last is not None
        and brand is not None
        and brand.cta_image
        and getattr(last, "source", "").replace("\\", "/").removeprefix("./")
        == brand.cta_image.replace("\\", "/").removeprefix("./")
    )
    end_card = last_label.startswith("end_card") or brand_card

    texts = [(entry.start, entry.end, entry.text) for entry in entries]
    texts += [(start, total, text) for start, text in spoken]
    joined = f" {fold_text(' '.join(text for _s, _e, text in sorted(texts)))} "
    wanted = fold_text(cta) if cta else ""
    exact = bool(wanted) and f" {wanted} " in joined
    generic = _generic_cta(joined)
    if exact or (not wanted and generic):
        checks.ok("story", "cta_wording")
    elif end_card:
        checks.ok("story", "cta_wording")  # the end card image carries it
    elif wanted and generic:
        checks.add(
            Finding(
                id="cta_wording",
                area="story",
                severity="low",
                penalty=5.0,
                message=f"the brand's call to action ({cta!r}) is in no caption or voice line; "
                "another call to action is present",
                fix="use the brand's exact wording in a caption near the end",
            )
        )
    else:
        checks.add(
            Finding(
                id="cta_missing",
                area="story",
                severity="high",
                penalty=25.0,
                message=(
                    "no call to action wording in the captions or the voice"
                    + (f" (the brand's is {cta!r})" if cta else "")
                ),
                fix="say or write the call to action (shop, order, learn more) in the last "
                "seconds, or end on the brand's CTA image",
            )
        )

    window = max(0.0, total - LAST_SECONDS)
    if end_card:
        late = True
    else:
        late = any(
            end > window and start < total and _has_cta(text, wanted)
            for start, end, text in texts
        )
    if late:
        checks.ok("story", "cta_at_end")
    else:
        checks.add(
            Finding(
                id="cta_not_at_end",
                area="story",
                severity="medium",
                time=window,
                end=total,
                penalty=12.0,
                message=f"no end card and no call to action in the last {LAST_SECONDS:g}s "
                f"({window:.1f}s to {total:.1f}s)",
                fix="end on the brand's CTA image or an end card (label the last segment "
                "end_card), or put the call to action in a caption over the last seconds",
            )
        )
    return checks


def _generic_cta(folded: str) -> bool:
    """``folded`` (already folded, padded with spaces) holds a wording of ``CTA_PHRASES``."""

    return any(f" {phrase} " in folded or f" {phrase}" in folded for phrase in CTA_PHRASES)


def _has_cta(text: str, wanted: str) -> bool:
    folded = f" {fold_text(text)} "
    if wanted and f" {wanted} " in folded:
        return True
    return _generic_cta(folded)


# ------------------------------------------------------------------------ safe zones

_FIX = {
    "top_bar": "move it below y={edge}",
    "right_buttons": "keep it left of x={edge} (narrower text, or shift it left)",
    "bottom_band": "move it above y={edge}",
}
_ZONE_NAME = {
    "top_bar": "top bar",
    "right_buttons": "right-side button column",
    "bottom_band": "bottom caption band",
}


def overlay_items(
    model: Storyboard,
    root: Path,
    brand: Brand | None,
    entries: Iterable[CaptionEntry],
    text_bboxes: Mapping[str, tuple[int, int, int, int]] | None = None,
) -> list[dict[str, Any]]:
    """The boxes that must stay out of the platform interface: captions, comments, badge,
    emoji and image stickers, picture overlays. ``text_bboxes`` are the already rendered text
    boxes by overlay key (the validator has them); the others are rendered here."""

    from .validate import _render_text_overlay

    width, height = model.format.width, model.format.height
    total = model.total_duration()
    items: list[dict[str, Any]] = []

    def text_box(key: str, overlay: Any) -> tuple[int, int, int, int] | None:
        if text_bboxes is not None and key in text_bboxes:
            return text_bboxes[key]
        try:
            box, _ = _render_text_overlay(
                overlay, root=root, width=width, height=height, brand=brand
            )
        except Exception:
            return None
        return box

    for entry in entries:
        caption = entry.caption
        pieces: list[tuple[str, CaptionOverlay]] = [(entry.key, caption)]
        if entry.kind == "written" and caption.animation == "word":
            chunks = caption.expand_words(total)
            if len(chunks) > 1:
                pieces = [(f"{entry.key}.{n}", chunk) for n, chunk in enumerate(chunks)]
        for key, piece in pieces:
            box = text_box(key, piece)
            if box is None or box == (0, 0, 0, 0):
                continue
            end = entry.end if piece is caption else (
                total if piece.end is None else min(float(piece.end), total)
            )
            items.append(
                {
                    "kind": "caption",
                    "label": _clip(" ".join(piece.lines), 30),
                    "start": float(piece.start),
                    "end": end,
                    "box": tuple(float(v) for v in box),
                }
            )
    for index, overlay in enumerate(model.overlays):
        end = total if getattr(overlay, "end", None) is None else min(float(overlay.end), total)
        if isinstance(overlay, CommentOverlay):
            box = text_box(str(index), overlay)
            if box is not None and box != (0, 0, 0, 0):
                items.append(
                    {
                        "kind": "comment",
                        "label": _clip(overlay.text, 30),
                        "start": float(overlay.start),
                        "end": end,
                        "box": tuple(float(v) for v in box),
                    }
                )
        elif isinstance(overlay, StickerOverlay) and overlay.kind in {"badge", "emoji", "image"}:
            try:
                from adanvil.render.stickers import sticker_art

                art = sticker_art(overlay, root=root)
                box = art.bbox
                art.image.close()
            except Exception:
                continue
            label = overlay.text or overlay.emoji or overlay.source or overlay.kind
            items.append(
                {
                    "kind": f"sticker:{overlay.kind}",
                    "label": _clip(str(label), 30),
                    "start": float(overlay.start),
                    "end": end,
                    "box": tuple(float(v) for v in box),
                }
            )
        elif isinstance(overlay, ImageOverlay):
            size = _image_size(root, overlay.source)
            if size is None:
                continue
            across = overlay.width or size[0]
            down = round(across * size[1] / max(size[0], 1))
            items.append(
                {
                    "kind": "image",
                    "label": _clip(overlay.source, 30),
                    "start": float(overlay.start),
                    "end": end,
                    "box": (
                        float(overlay.x),
                        float(overlay.y),
                        float(overlay.x + across),
                        float(overlay.y + down),
                    ),
                }
            )
    return items


def _image_size(root: Path, source: str) -> tuple[int, int] | None:
    from PIL import Image

    from .validate import _relative_file

    path = _relative_file(root, source)
    if path is None or not path.is_file():
        return None
    try:
        with Image.open(path) as picture:
            return picture.size
    except (OSError, ValueError):
        return None


def safe_zone_checks(
    items: list[dict[str, Any]],
    platforms: Iterable[str] = PLATFORMS,
    size: tuple[int, int] = (1080, 1920),
) -> tuple[Checks, list[dict[str, Any]]]:
    """Findings for the items that fall inside a platform zone (one finding per platform and
    zone, with the first item and the count), and the items marked ``inside`` for the preview."""

    checks = Checks()
    chosen = tuple(platforms)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in items:
        item["inside"] = False
        for hit in collisions(item["box"], chosen, size):
            item["inside"] = True
            groups.setdefault((hit["platform"], hit["zone"]), []).append({**item, **hit})
    if not items:
        checks.skip("safe_zones", "overlays", "no caption or sticker to place")
        return checks, items
    clear = [p for p in chosen if not any(key[0] == p for key in groups)]
    for platform in clear:
        checks.ok("safe_zones", platform)
    for (platform, zone), hits in sorted(groups.items()):
        hits.sort(key=lambda hit: hit["start"])
        worst = max(hit["share"] for hit in hits)
        severity = "high" if worst >= 0.5 else "medium" if worst >= 0.15 else "low"
        edge = SAFE_ZONES[platform][zone]
        edge_value = {"top_bar": edge[3], "right_buttons": edge[0], "bottom_band": edge[1]}[zone]
        first = hits[0]
        penalty = {"high": 18.0, "medium": 10.0, "low": 4.0}[severity] + min(
            8.0, 2.0 * (len(hits) - 1)
        )
        checks.add(
            Finding(
                id="safe_zone",
                area="safe_zones",
                severity=severity,
                time=first["start"],
                end=max(hit["end"] for hit in hits),
                penalty=penalty,
                message=(
                    f"{len(hits)} overlay(s) fall inside the {platform} {_ZONE_NAME[zone]} "
                    f"{tuple(int(v) for v in edge)}: first {first['kind']} "
                    f"{first['label']!r} at {first['start']:.1f}s "
                    f"({first['share']:.0%} of it inside)"
                ),
                fix=f"{_FIX[zone].format(edge=edge_value)} (1080x1920 pixels)",
                detail={
                    "platform": platform,
                    "zone": zone,
                    "items": [
                        {"label": hit["label"], "start": round(hit["start"], 2)}
                        for hit in hits[:8]
                    ],
                },
            )
        )
    return checks, items


def storyboard_checks(
    model: Storyboard,
    root: Path,
    brand: Brand | None,
    *,
    platforms: Iterable[str] = PLATFORMS,
    entries: list[CaptionEntry] | None = None,
    text_bboxes: Mapping[str, tuple[int, int, int, int]] | None = None,
) -> tuple[Checks, list[CaptionEntry], list[dict[str, Any]]]:
    """The pixel-free checks (reading, story, safe zones) of a storyboard. Returns the checks,
    the captions and the safe-zone items (for the preview)."""

    timeline = model.timeline()
    starts = [item[0] for item in timeline]
    ends = [item[1] for item in timeline]
    notes: list[str] = []
    if entries is None:
        entries, notes = collect_captions(model, root, brand, starts, ends)
    checks = Checks()
    checks.extend(reading_checks(entries))
    checks.extend(story_checks(model, brand, entries, voice_lines(model, root), starts, ends))
    items = overlay_items(model, root, brand, entries, text_bboxes)
    zone_checks, items = safe_zone_checks(
        items, platforms, (model.format.width, model.format.height)
    )
    checks.extend(zone_checks)
    for note in notes:
        checks.skip("reading", "voice_or_speech_captions", note)
    return checks, entries, items


__all__ = [
    "CTA_PHRASES",
    "CaptionEntry",
    "Checks",
    "Finding",
    "collect_captions",
    "fold_text",
    "overlay_items",
    "reading_checks",
    "reading_need",
    "safe_zone_checks",
    "story_checks",
    "storyboard_checks",
    "voice_lines",
]
