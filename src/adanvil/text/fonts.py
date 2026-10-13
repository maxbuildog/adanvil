from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from PIL import ImageFont

_FONT_DIR = files("adanvil") / "assets" / "fonts"

# Values are deliberately kept as asset names rather than resolved Paths.  This
# makes the registry useful to callers while keeping package-resource lookup in
# one place.
FAMILIES: dict[str, dict[int, str] | str] = {
    "plus-jakarta-sans": "PlusJakartaSans-VF.ttf",
    "montserrat": "Montserrat-VF.ttf",
    "tiktok-sans": "TikTokSans-VF.ttf",
    "roboto": "Roboto-VF.ttf",
    "poppins": {600: "Poppins-SemiBold.ttf", 700: "Poppins-Bold.ttf"},
}


def _axis_info(font: ImageFont.FreeTypeFont) -> list[dict[str, Any]]:
    try:
        return list(font.get_variation_axes())
    except (AttributeError, OSError):
        return []


def _axis_name(axis: dict[str, Any]) -> str:
    name = axis.get("name", "")
    if isinstance(name, bytes):
        return name.decode("ascii", errors="ignore").lower()
    return str(name).lower()


def _font_path(family: str, weight: int) -> tuple[Path, int | None]:
    path = Path(family)
    if path.is_absolute() and path.suffix.lower() in {".ttf", ".otf", ".ttc"}:
        return path, weight

    key = family.lower()
    if key not in FAMILIES:
        known = ", ".join(sorted(FAMILIES))
        raise ValueError(f"Unknown font family {family!r}; known families: {known}")
    entry = FAMILIES[key]
    if isinstance(entry, str):
        return Path(str(_FONT_DIR / entry)), weight
    available = sorted(entry)
    nearest = min(available, key=lambda candidate: (abs(candidate - weight), candidate))
    return Path(str(_FONT_DIR / entry[nearest])), None


@lru_cache(maxsize=256)
def load_font(family: str, weight: int, size: int) -> ImageFont.FreeTypeFont:
    """Load a bundled or absolute-path font and select its requested weight."""

    path, variable_weight = _font_path(family, int(weight))
    # One layout engine on every machine: Raqm (in the Linux wheels, often missing on Windows)
    # and the basic engine measure the same line a few pixels apart (kerning, hinting), so a
    # text fitted to its box on one machine would overflow it on another.
    font = ImageFont.truetype(str(path), int(size), layout_engine=ImageFont.Layout.BASIC)
    if variable_weight is None:
        return font

    axes = _axis_info(font)
    if not axes:
        return font

    values: list[float | int] = []
    weight_index: int | None = None
    for index, axis in enumerate(axes):
        name = _axis_name(axis)
        if "weight" in name or name in {"wght", "weight"}:
            weight_index = index
        values.append(axis.get("default", axis.get("minimum", 0)))
    if weight_index is None:
        # The bundled fonts all expose a named weight axis.  This fallback is
        # useful for a brand font whose metadata uses an unusual axis name.
        weight_index = min(1, len(values) - 1)
    axis = axes[weight_index]
    minimum = float(axis.get("minimum", variable_weight))
    maximum = float(axis.get("maximum", variable_weight))
    selected = max(minimum, min(maximum, float(variable_weight)))
    values[weight_index] = selected
    try:
        font.set_variation_by_axes(values)
    except (AttributeError, OSError, ValueError):
        # A static or malformed brand font should still be usable.
        pass
    return font
