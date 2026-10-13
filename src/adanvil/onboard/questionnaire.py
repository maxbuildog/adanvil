"""The short, deterministic questionnaire used after site discovery."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel

from .site import SiteFacts

QuestionKind = Literal["text", "list", "choice", "bool"]


class Question(BaseModel):
    id: str
    text: str
    why: str
    kind: QuestionKind
    choices: list[str] | None = None
    prefill: Any | None = None
    required: bool
    maps_to: str


BUNDLED_FAMILIES = ["plus-jakarta-sans", "montserrat", "tiktok-sans", "roboto", "poppins"]
EMOJI_CHOICES = ["twemoji", "none", "project-relative folder path"]


def _site_name(facts: SiteFacts | None) -> str | None:
    if facts is None or not facts.title:
        return None
    name = facts.title
    for separator in (" | ", " - ", " – ", ": "):
        if separator in name:
            name = name.split(separator, 1)[0]
            break
    return name.strip() or None


def _offer_prefill(facts: SiteFacts | None) -> str | None:
    if facts is None or not facts.products:
        return None
    product = facts.products[0]
    if product.price is None:
        return None
    if product.compare_at_price and product.compare_at_price != product.price:
        return f"{product.price} {product.currency or ''} (was {product.compare_at_price})".strip()
    return f"{product.price} {product.currency or ''}".strip()


_NON_HERO_MARKERS = (
    "coffret",
    "bundle",
    "pack",
    "kit",
    "gift",
    "cadeau",
    "emballage",
    "box set",
    "duo",
    "trio",
)


def _hero_product(facts: SiteFacts | None):
    if facts is None or not facts.products:
        return None
    candidates = [
        product
        for product in facts.products
        if not any(marker in product.title.casefold() for marker in _NON_HERO_MARKERS)
    ]
    if not candidates:
        return None
    heading_text = " ".join(facts.headings).casefold()
    for product in candidates:
        words = re.findall(r"[\w]+", product.title.casefold())
        if words and all(word in heading_text for word in words):
            return product
    return candidates[0]


def _family_prefill(facts: SiteFacts | None) -> str:
    if facts is not None:
        aliases = {
            "plusjakartasans": "plus-jakarta-sans",
            "montserrat": "montserrat",
            "tiktoksans": "tiktok-sans",
            "roboto": "roboto",
            "poppins": "poppins",
        }
        for family in facts.fonts:
            candidate = family.casefold().replace(" ", "").replace("-", "")
            if candidate in aliases:
                return aliases[candidate]
    return "plus-jakarta-sans"


def build_questionnaire(facts: SiteFacts | None) -> list[Question]:
    """Return the onboarding questions in the stable product order."""

    website = facts.final_url if facts is not None and facts.final_url else None
    hero = _hero_product(facts)
    hero_product = hero.title if hero is not None else None
    guarantee = (
        facts.guarantee_phrases[0] if facts is not None and facts.guarantee_phrases else None
    )
    colours = [colour.hex for colour in facts.colors] if facts is not None else []
    primary = colours[0] if colours else None
    accent = colours[1] if len(colours) > 1 else primary
    return [
        Question(
            id="brand_name",
            text="What is the brand name?",
            why="Every generated file and ad needs one consistent name.",
            kind="text",
            prefill=_site_name(facts),
            required=True,
            maps_to="name",
        ),
        Question(
            id="website",
            text="What is the main website URL?",
            why="The site is the source of truth for the offer and claims.",
            kind="text",
            prefill=website,
            required=True,
            maps_to="website",
        ),
        Question(
            id="niche",
            text="What category or niche is the brand in?",
            why="The category changes safe claims, visuals and vocabulary.",
            kind="text",
            required=True,
            maps_to="niche",
        ),
        Question(
            id="hero_product",
            text="Which product should lead the first ads?",
            why="A clear hero product keeps the first creatives focused.",
            kind="text",
            prefill=hero_product,
            required=True,
            maps_to="product.name",
        ),
        Question(
            id="target_customer",
            text="Who is the target customer: age range and main pains?",
            why="The customer description guides casting, hooks and language.",
            kind="text",
            required=True,
            maps_to="brief.customer",
        ),
        Question(
            id="angles_worked",
            text="Which angles or messages have already worked?",
            why="Existing winners are safer starting points than guesses.",
            kind="list",
            required=False,
            maps_to="brief.angles_worked",
        ),
        Question(
            id="angles_failed",
            text="Which angles or messages have failed?",
            why="Known losers prevent repeating expensive tests.",
            kind="list",
            required=False,
            maps_to="brief.angles_failed",
        ),
        Question(
            id="current_offer",
            text="What is the current offer, including price or discount?",
            why="Ads must match the live landing page.",
            kind="text",
            prefill=_offer_prefill(facts),
            required=True,
            maps_to="product.offer",
        ),
        Question(
            id="guarantee",
            text="What guarantee or reassurance may we state?",
            why="Guarantee wording must use the brand's real terms.",
            kind="text",
            prefill=guarantee,
            required=True,
            maps_to="product.guarantee",
        ),
        Question(
            id="cta_text",
            text="What exact CTA wording should end the ad?",
            why="A consistent CTA makes the action clear and measurable.",
            kind="text",
            required=True,
            maps_to="product.cta_text",
        ),
        Question(
            id="markets",
            text="Which countries or markets are active?",
            why="Market and currency rules affect copy and compliance.",
            kind="list",
            required=True,
            maps_to="markets",
        ),
        Question(
            id="caption_language",
            text="Which language should captions use?",
            why="Captions should not mix languages by accident.",
            kind="text",
            prefill=facts.language if facts is not None else None,
            required=True,
            maps_to="language",
        ),
        Question(
            id="viewer_address",
            text="How should the ad address the viewer?",
            why="You, tu, first person and brand voice create different relationships.",
            kind="text",
            required=True,
            maps_to="tone.address",
        ),
        Question(
            id="brand_voice",
            text="How would you describe the brand voice in a few words?",
            why="The voice keeps hooks and CTAs recognisable across ads.",
            kind="text",
            required=True,
            maps_to="tone.voice",
        ),
        Question(
            id="banned_words",
            text="Which words or claims must never appear?",
            why=(
                "Category rules matter; for example, supplements or cosmetics "
                "cannot claim medical effects."
            ),
            kind="list",
            required=True,
            maps_to="claims.banned_words",
        ),
        Question(
            id="claims_guidance",
            text="What claims guidance should every later tool follow?",
            why="A short rule prevents unsupported promises and risky rewrites.",
            kind="text",
            required=True,
            maps_to="claims.guidance",
        ),
        Question(
            id="people",
            text="Who may appear in footage, and did each person consent to paid ads?",
            why="Consent is required before using creators, customers or staff in paid media.",
            kind="list",
            required=True,
            maps_to="people",
        ),
        Question(
            id="footage_rules",
            text="What footage rules apply, such as only the current product version?",
            why="The editor needs explicit rules for product versions and people.",
            kind="list",
            required=True,
            maps_to="footage_rules",
        ),
        Question(
            id="product_versions",
            text=(
                "If the product exists in several versions, how can they be told apart "
                "on camera (packaging, colour, label, shape)?"
            ),
            why=(
                "Footage annotations carry a product_version; this tells the editor which "
                "version is current."
            ),
            kind="text",
            required=False,
            maps_to="product_versions",
        ),
        Question(
            id="primary_color",
            text="What is the primary brand colour?",
            why="A known colour makes captions and end cards feel like the brand.",
            kind="text",
            prefill=primary,
            required=False,
            maps_to="visual.primary_color",
        ),
        Question(
            id="accent_color",
            text="What is the accent colour?",
            why="The accent is used for labels, emphasis and CTA details.",
            kind="text",
            prefill=accent,
            required=False,
            maps_to="visual.accent_color",
        ),
        Question(
            id="caption_font",
            text="Which caption font should be used?",
            why="A bundled font keeps rendering reproducible on every machine.",
            kind="choice",
            choices=BUNDLED_FAMILIES,
            prefill=_family_prefill(facts),
            required=False,
            maps_to="visual.caption_family",
        ),
        Question(
            id="emoji",
            text=(
                "Which emoji option should captions use? Choose twemoji, none, "
                "or a project-relative folder containing PNG emoji assets."
            ),
            why="One provider avoids inconsistent emoji shapes and missing glyphs.",
            kind="choice",
            choices=EMOJI_CHOICES,
            prefill="twemoji",
            required=False,
            maps_to="visual.emoji",
        ),
        Question(
            id="end_card_image",
            text="What image path should the end card use?",
            why="The final frame needs a known brand asset.",
            kind="text",
            prefill="brand/end_card.png",
            required=False,
            maps_to="end_card.source",
        ),
        Question(
            id="cta_image",
            text=(
                "Does the brand already have a ready-made call-to-action image "
                "(for example in a cta/ folder or brand/)? Give its project path."
            ),
            why=(
                "A ready-made CTA image is the last segment; an end card is composed "
                "only when there is none."
            ),
            kind="text",
            required=False,
            maps_to="cta_image",
        ),
        Question(
            id="naming",
            text="What output naming template should be used?",
            why="Stable names make batches easy to find and replace.",
            kind="text",
            prefill="{name}",
            required=False,
            maps_to="naming",
        ),
    ]


__all__ = ["BUNDLED_FAMILIES", "EMOJI_CHOICES", "Question", "build_questionnaire"]
