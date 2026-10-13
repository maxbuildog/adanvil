"""Which version of the product a shot shows, from reference photos (not from words).

The brand profile lists the versions as ``product_versions: [{name, photos: [paths]}]``. Each
photo is embedded; a frame shows the product when its best similarity to any reference photo
reaches ``PRESENT_MIN``, and the version is the one whose photos it resembles most. The
confidence is 0.5 when two versions tie and 1.0 when the best leads the other by ``LEAD_FULL``
or more (with a single version it only says how close the frame is to its photos).
A human ``product_version`` annotation always wins over this guess.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from adanvil.brand import Brand, ProductVersion, load_brand
from adanvil.media.cache import Cache
from adanvil.project import resolve_in_project

PRESENT_MIN = 0.70  # similarity to a reference photo from which a frame shows the product
LEAD_FULL = 0.08  # lead over the other versions that makes a match fully confident
SINGLE_SPAN = 0.2  # with one version: similarity above PRESENT_MIN that gives confidence 1.0


@dataclass(frozen=True)
class ProductReference:
    """One version of the product: its name and the embeddings of its photos."""

    name: str
    embeddings: np.ndarray  # (photos, D)


@dataclass(frozen=True)
class FrameMatch:
    """The closest version for a frame that shows the product."""

    version: str
    confidence: float
    similarity: float


def brand_versions(root: Path) -> list[ProductVersion]:
    """The versions with photos listed in ``brand.yaml`` (none without the file or the list)."""

    path = Path(root) / "brand.yaml"
    if not path.is_file():
        return []
    try:
        brand: Brand = load_brand(path)
    except Exception:
        return []
    return [version for version in brand.version_references() if version.photos]


def version_photos(root: Path, versions: list[ProductVersion]) -> dict[str, list[Path]]:
    """The photo files of each version that exist in the project (missing ones are skipped)."""

    found: dict[str, list[Path]] = {}
    for version in versions:
        files: list[Path] = []
        for photo in version.photos:
            try:
                candidate = resolve_in_project(Path(root), photo)
            except (TypeError, ValueError):
                continue
            if candidate.is_file():
                files.append(candidate)
        if files:
            found[version.name] = files
    return found


def references_signature(photos: dict[str, list[Path]]) -> str:
    """Changes when a version, a photo or the content of a photo changes."""

    parts = []
    for name in sorted(photos):
        for photo in photos[name]:
            try:
                key = Cache.file_key(photo)
                parts.append(f"{name}|{photo.name}|{key['size']}|{key['mtime_ns']}")
            except OSError:
                parts.append(f"{name}|{photo.name}|missing")
    if not parts:
        return ""
    parts.append(f"present>={PRESENT_MIN}|lead={LEAD_FULL}")
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:10]


def embed_references(engine, photos: dict[str, list[Path]]) -> list[ProductReference]:
    """Embed every reference photo (one call) and group the rows by version."""

    order = [(name, photo) for name in sorted(photos) for photo in photos[name]]
    if not order:
        return []
    vectors = engine.embed_images([photo for _name, photo in order])
    references: list[ProductReference] = []
    for name in sorted(photos):
        rows = [index for index, (owner, _photo) in enumerate(order) if owner == name]
        references.append(ProductReference(name, vectors[rows]))
    return references


def match_frames(
    frames: np.ndarray,
    references: list[ProductReference],
    *,
    present_min: float = PRESENT_MIN,
) -> list[FrameMatch | None]:
    """For each frame embedding: the closest version, or ``None`` when no product is shown."""

    if frames.size == 0 or not references:
        return [None] * len(frames)
    best = np.stack([(frames @ ref.embeddings.T).max(axis=1) for ref in references], axis=1)
    matches: list[FrameMatch | None] = []
    for row in best:
        order = np.argsort(row)[::-1]
        top = float(row[order[0]])
        if top < present_min:
            matches.append(None)
            continue
        if len(references) > 1:
            lead = top - float(row[order[1]])
            confidence = 0.5 + 0.5 * min(1.0, lead / LEAD_FULL)
        else:
            confidence = 0.5 + 0.5 * min(1.0, (top - present_min) / SINGLE_SPAN)
        matches.append(
            FrameMatch(references[int(order[0])].name, round(float(confidence), 3), round(top, 4))
        )
    return matches


__all__ = [
    "LEAD_FULL",
    "PRESENT_MIN",
    "FrameMatch",
    "ProductReference",
    "brand_versions",
    "embed_references",
    "match_frames",
    "references_signature",
    "version_photos",
]
