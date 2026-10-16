"""Brand profiles used to keep storyboard decisions consistent."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from adanvil.safeio import replace_text


class _BrandModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Tone(_BrandModel):
    address: str | None = None
    voice: str | None = None


class Product(_BrandModel):
    name: str | None = None
    offer: str | None = None
    guarantee: str | None = None
    cta_text: str | None = None


class Claims(_BrandModel):
    banned_words: list[str] = Field(default_factory=list)
    guidance: str | None = None


class StyleOverride(_BrandModel):
    size: int | None = None
    family: str | None = None
    weight: int | None = None
    fg: str | None = None
    stroke: str | None = None
    box: str | None = None
    accent: str | None = None


class Visual(_BrandModel):
    primary_color: str | None = None
    accent_color: str | None = None
    # Colour of the spoken word in animated captions; unset means the social-video yellow.
    highlight_color: str | None = None
    caption_family: str | None = None
    caption_weight: int | None = None
    emoji: str | None = None
    styles: dict[str, StyleOverride] = Field(default_factory=dict)


class Person(_BrandModel):
    id: str
    notes: str | None = None
    consent: bool | None = None


class Rhythm(_BrandModel):
    ugc_shot: list[float] | None = None
    product_shot: list[float] | None = None
    min_shot: float | None = None
    caption_on_screen: float | None = None


class EndCard(_BrandModel):
    source: str | None = None
    duration: float | None = None


class Speech(_BrandModel):
    """How the speech of a creator's footage is read (see the ``speech_view`` tool).

    ``fillers`` are the hesitation sounds flagged wherever they appear (default: euh, heu,
    hum, bah, ben, um, uh, er); ``isolated_fillers`` are ordinary words that count as
    fillers only when said alone between two pauses (default: genre, du coup, like). A list
    given here replaces the default one.

    ``vocabulary`` lists extra words the transcriber should spell right (names, product or
    ingredient terms, jargon). They are added to the brand name and the product name, which
    are always given to the transcriber as its initial prompt.
    """

    fillers: list[str] | None = None
    isolated_fillers: list[str] | None = None
    vocabulary: list[str] | None = None


class ProductVersion(_BrandModel):
    """One version of the product, told apart by photos (not by words).

    ``photos`` are project paths of pictures of that version (the pack, the device, the label);
    the library compares every shot with them to guess which version it shows (``library_index``
    with the vision environment). ``notes`` is free text for the human.
    """

    name: str
    photos: list[str] = Field(default_factory=list)
    notes: str | None = None


class Brand(_BrandModel):
    """A project brand profile.

    The nested models deliberately use optional values: a partially onboarded
    brand is still useful to an agent, and the renderer can fall back to its
    own defaults for fields that are absent.
    """

    version: int = 1
    name: str
    website: str | None = None
    language: str | None = None
    markets: list[str] = Field(default_factory=list)
    niche: str | None = None
    tone: Tone | None = None
    product: Product | None = None
    claims: Claims | None = None
    visual: Visual | None = None
    footage_rules: list[str] = Field(default_factory=list)
    # Either a few words on how to tell the versions apart (the onboarding answer), or the
    # versions with their reference photos: [{name: current, photos: [brand/current-1.jpg]}].
    product_versions: str | list[ProductVersion] | None = None
    people: list[Person] = Field(default_factory=list)
    naming: str = "{name}"
    rhythm: Rhythm | None = None
    end_card: EndCard | None = None
    cta_image: str | None = None
    speech: Speech | None = None

    def version_references(self) -> list[ProductVersion]:
        """The product versions that come with reference photos (empty for a text answer)."""

        if isinstance(self.product_versions, list):
            return list(self.product_versions)
        return []

    def version_text(self) -> str | None:
        """How the versions are told apart, as one line (the text answer, or the list)."""

        value = self.product_versions
        if isinstance(value, list):
            parts = [
                f"{item.name} ({len(item.photos)} photo{'s' if len(item.photos) != 1 else ''})"
                for item in value
            ]
            return ", ".join(parts) or None
        return value or None

    def apply_to_caption(self, overlay: Any) -> Any:
        """Return *overlay* with unset caption defaults filled from the brand."""

        visual = self.visual
        if visual is None or not hasattr(overlay, "style"):
            return overlay.model_copy(deep=True)

        style = str(overlay.style)
        style_override = visual.styles.get(style)
        updates: dict[str, Any] = {}

        # The brand caption family is intentionally limited to the styles for
        # which the onboarding contract promises a shared caption treatment.
        if style in {"pill", "meme", "box", "label"}:
            if getattr(overlay, "family", None) is None and visual.caption_family is not None:
                updates["family"] = visual.caption_family
            if getattr(overlay, "weight", None) is None and visual.caption_weight is not None:
                updates["weight"] = visual.caption_weight

        if style_override is not None:
            for field in ("size", "family", "weight", "fg", "stroke", "box", "accent"):
                value = getattr(style_override, field)
                if value is not None and getattr(overlay, field, None) is None:
                    updates[field] = value

        if style == "label" and getattr(overlay, "accent", None) is None:
            if visual.accent_color is not None:
                updates["accent"] = visual.accent_color

        return overlay.model_copy(update=updates, deep=True)


def _normalise(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in folded if not unicodedata.combining(char))


def find_banned(text: str, banned: list[str]) -> list[str]:
    """Return banned entries occurring as whole words or phrases in *text*."""

    normalised_text = _normalise(text)
    found: list[str] = []
    for word in banned:
        normalised_word = _normalise(word).strip()
        if not normalised_word:
            continue
        pattern = rf"\b{re.escape(normalised_word)}\b"
        if re.search(pattern, normalised_text, flags=re.UNICODE):
            found.append(word)
    return found


def load_brand(path: str | Path) -> Brand:
    """Load a UTF-8 YAML brand profile."""

    source = Path(path)
    values = yaml.safe_load(source.read_text(encoding="utf-8"))
    return Brand.model_validate(values)


def brand_accent(root: str | Path) -> str | None:
    """The highlight colour of animated captions set in the brand profile, or ``None``.

    Only ``visual.highlight_color`` counts: a brand accent (often a dark or saturated
    colour) is rarely readable as the lit word, so the default stays yellow.
    """

    path = Path(root) / "brand.yaml"
    try:
        brand = load_brand(path)
    except Exception:  # no profile, or one that does not parse: no accent
        return None
    visual = brand.visual
    return visual.highlight_color if visual is not None else None


def brand_accent_color(root: str | Path) -> str | None:
    """The brand accent colour (``visual.accent_color``) of the project, or ``None``."""

    try:
        brand = load_brand(Path(root) / "brand.yaml")
    except Exception:  # no profile, or one that does not parse: no accent
        return None
    visual = brand.visual
    return visual.accent_color if visual is not None else None


def save_brand(brand: Brand, path: str | Path) -> None:
    """Save a brand profile as ordered, human-readable UTF-8 YAML."""

    values = brand.model_dump(mode="python")
    replace_text(path, yaml.safe_dump(values, allow_unicode=True, sort_keys=False))


def default_brand(name: str) -> Brand:
    """Return sensible defaults for a newly created project."""

    return Brand(
        name=name,
        visual=Visual(
            primary_color="#2563EB",
            accent_color="#2563EB",
            caption_family="plus-jakarta-sans",
            caption_weight=700,
            emoji="twemoji",
        ),
        naming="{name}",
        rhythm=Rhythm(
            ugc_shot=[2.0, 2.5],
            product_shot=[1.6, 2.0],
            min_shot=1.0,
            caption_on_screen=3.0,
        ),
        end_card=EndCard(source="brand/end_card.png", duration=3.0),
    )


__all__ = [
    "Brand",
    "Claims",
    "EndCard",
    "Person",
    "Product",
    "ProductVersion",
    "Rhythm",
    "Speech",
    "StyleOverride",
    "Tone",
    "Visual",
    "brand_accent",
    "default_brand",
    "find_banned",
    "load_brand",
    "save_brand",
]
