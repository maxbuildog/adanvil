"""Controlled vocabularies shared by the breakdown, the catalog and the storyboard."""

from __future__ import annotations

from typing import Any

ROLE_VALUES = (
    "hook",
    "problem",
    "myth",
    "punch",
    "agitation",
    "reveal",
    "mechanism",
    "demo",
    "proof",
    "payoff",
    "offer",
    "cta",
    "transition",
    "other",
)
EMOTION_VALUES = (
    "none",
    "neutral",
    "worried",
    "frustrated",
    "sad",
    "tired",
    "closed",
    "skeptical",
    "shocked",
    "curious",
    "relaxed",
    "happy",
    "satisfied",
)

TYPE_VALUES = (
    "talking_head",
    "face_closeup",
    "eye_or_feature_macro",
    "texture_macro",
    "diagram",
    "animation_or_3d",
    "product_hero",
    "product_in_use",
    "unboxing",
    "before_after",
    "lifestyle",
    "reaction",
    "text_card",
    "screen_or_ui",
    "other",
)
SCALE_VALUES = ("extreme_closeup", "closeup", "medium", "wide", "insert")

# Shot types renamed to neutral names. Breakdowns written before the rename still say
# ``skin_texture`` and ``anatomy_diagram``: the old names stay valid on input and are
# normalised to the new ones.
SHOT_TYPE_ALIASES = {
    "skin_texture": "texture_macro",
    "anatomy_diagram": "diagram",
}


def normalize_shot_type(value: Any) -> Any:
    """Map an old shot type name to its current name; anything else comes back unchanged."""

    if isinstance(value, str):
        return SHOT_TYPE_ALIASES.get(value, value)
    return value


def check_emotions(values: Any) -> list[str]:
    """One emotion or a list of them as a list, each in ``EMOTION_VALUES``; else a ``ValueError``.

    The one check behind a library annotation, a library search filter, a segment and a
    breakdown shot, so all of them accept and reject the same values.
    """

    items = [values] if isinstance(values, str) else values
    if not isinstance(items, (list, tuple)):
        raise ValueError(f"emotion must be one of: {', '.join(EMOTION_VALUES)}")
    for item in items:
        if item not in EMOTION_VALUES:
            raise ValueError(f"emotion must be one of: {', '.join(EMOTION_VALUES)} (got {item!r})")
    return list(items)


def check_vocabulary(value: Any, name: str, allowed: tuple[str, ...]) -> Any:
    """``value`` when it is ``None`` or allowed; else an error that lists the allowed values."""

    if value is not None and value not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(allowed)} (got {value!r})")
    return value


__all__ = [
    "EMOTION_VALUES",
    "ROLE_VALUES",
    "SCALE_VALUES",
    "SHOT_TYPE_ALIASES",
    "TYPE_VALUES",
    "check_emotions",
    "check_vocabulary",
    "normalize_shot_type",
]
