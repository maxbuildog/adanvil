from __future__ import annotations

import os
from functools import lru_cache
from importlib.resources import files
from pathlib import Path, PureWindowsPath
from typing import Any

from PIL import ImageFont

from adanvil.project import resolve_in_project

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


FONT_SUFFIXES = (".ttf", ".otf", ".ttc")


def _is_network_path(value: str) -> bool:
    """Whether ``value`` names a UNC, device or network-drive path: opening one makes Windows
    send the user's credentials to that server, so it is never touched."""

    if value.startswith(("\\\\", "//")):
        return True
    drive = PureWindowsPath(value).drive
    if not drive or os.name != "nt":
        return False
    if drive.startswith("\\\\"):
        return True
    try:
        import ctypes

        drive_type = ctypes.windll.kernel32.GetDriveTypeW(drive + "\\")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return False
    return drive_type == 4  # DRIVE_REMOTE


def is_font_reference(family: str) -> bool:
    """Whether ``family`` is written as a font file (a path or a ``.ttf``, ``.otf``, ``.ttc``
    name) rather than as the name of a bundled family."""

    if family.lower() in FAMILIES:
        return False
    return any(mark in family for mark in "/\\") or family.lower().endswith(FONT_SUFFIXES)


def usable_family(family: str | None, root: Path | None) -> str | None:
    """``family`` when a text may be drawn with it, else ``None`` (the default family is used).

    A bundled family name is returned as written. A font file (``.ttf``, ``.otf``, ``.ttc``)
    is returned as its absolute path when it is a regular file of the project at ``root``,
    written relative to it or as an absolute path inside it, and its real path stays inside the
    project (a link out of it is refused). Anything else is ``None``: a UNC or network path
    (never touched), a file outside the project, a missing file, an unknown name.
    """

    if not isinstance(family, str) or not family.strip() or "\x00" in family:
        return None
    if family.lower() in FAMILIES:
        return family
    if root is None or _is_network_path(family):
        return None
    try:
        if Path(family).is_absolute() or PureWindowsPath(family).is_absolute():
            candidate = os.path.abspath(family)
            project = os.path.abspath(root)
            if os.path.normcase(os.path.commonpath((project, candidate))) != os.path.normcase(
                project
            ):
                return None
        else:
            candidate = os.path.abspath(resolve_in_project(Path(root), family))
        if not candidate.lower().endswith(FONT_SUFFIXES):
            return None
        real_root = os.path.normcase(os.path.realpath(root))
        real = os.path.normcase(os.path.realpath(candidate))
        if os.path.commonpath((real_root, real)) != real_root or not os.path.isfile(real):
            return None
    except (OSError, TypeError, ValueError):
        return None
    return candidate


def unusable_family_message(family: str) -> str:
    """Why ``family`` is not used and what is: the warning shown when it is replaced."""

    known = ", ".join(sorted(FAMILIES))
    return (
        f"font {family!r} is neither a bundled family ({known}) nor a font file "
        "(.ttf, .otf, .ttc) inside the project; the default family is used"
    )


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
    if path.is_absolute() and path.suffix.lower() in FONT_SUFFIXES:
        # Second line of defence (the callers send a project font through ``usable_family``):
        # never open a network path, nor anything that is not an existing regular file.
        if _is_network_path(family):
            raise ValueError(
                f"Font {family!r} is a network path: use a bundled family or a font file "
                "inside the project"
            )
        if not path.is_file():
            raise ValueError(f"Font file not found: {family!r} is not an existing regular file")
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
