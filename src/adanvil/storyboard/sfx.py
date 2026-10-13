"""``audio.sfx_auto``: the sound effects placed at the cuts and at the captions of a storyboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adanvil.sfx import sfx_source_path

from .schema import CaptionOverlay, SfxOverlay, Storyboard

_EPSILON = 1e-9


def explicit_sfx(sb: Storyboard) -> list[SfxOverlay]:
    """The ``sfx`` overlays the storyboard carries."""

    return [overlay for overlay in sb.overlays if isinstance(overlay, SfxOverlay)]


def expand_sfx_auto(
    sb: Storyboard, starts: list[float], total: float, root: Path | None = None
) -> list[dict[str, Any]]:
    """The explicit sound effects ``audio.sfx_auto`` stands for, sorted by time.

    ``starts`` is the start of each segment on the timeline (the first is 0: the cuts are the
    others). Cut sounds are placed first, one cut in ``every``, then caption sounds at the
    start of each written caption (``caption`` overlays only: the captions generated from a
    voice or from the speech are not written ones). A sound that would fall within ``min_gap``
    seconds of one already placed is skipped, and none is placed at or after ``total``.
    Each entry is ``{"type": "sfx", "source", "at", "volume", "auto": "cuts" | "captions"}``;
    ``source`` is the project path of the file when ``root`` is given.
    """

    auto = sb.audio.sfx_auto
    if auto is None:
        return []

    def source(name: str) -> str:
        if root is None:
            return name if "/" in name else f"sfx/{name}.wav"
        return sfx_source_path(root, name)

    placed: list[dict[str, Any]] = []

    def free(moment: float) -> bool:
        return all(abs(moment - item["at"]) >= auto.min_gap - _EPSILON for item in placed)

    def place(moment: float, name: str, kind: str) -> None:
        if moment < 0 or moment >= total - _EPSILON or not free(moment):
            return
        placed.append(
            {
                "type": "sfx",
                "source": source(name),
                "at": round(moment, 4),
                "volume": auto.volume,
                "auto": kind,
            }
        )

    if auto.cuts is not None:
        for moment in list(starts[1:])[:: auto.every]:
            place(float(moment), auto.cuts, "cuts")
    if auto.captions is not None:
        seen: set[float] = set()
        for overlay in sb.overlays:
            if isinstance(overlay, CaptionOverlay) and overlay.start not in seen:
                seen.add(overlay.start)
                place(float(overlay.start), auto.captions, "captions")
    return sorted(placed, key=lambda item: (item["at"], item["auto"]))


def all_sfx(
    sb: Storyboard, starts: list[float], total: float, root: Path | None = None
) -> list[dict[str, Any]]:
    """Every sound effect of the storyboard (explicit overlays, then the automatic ones) as
    ``{"type", "source", "at", "volume", "auto"}`` dicts, sorted by time; ``auto`` is ``None``
    for an explicit overlay."""

    entries: list[dict[str, Any]] = []
    for overlay in explicit_sfx(sb):
        path = overlay.source
        if root is not None:
            path = sfx_source_path(root, overlay.source)
        entries.append(
            {
                "type": "sfx",
                "source": path,
                "at": overlay.at,
                "volume": overlay.volume,
                "auto": None,
            }
        )
    entries.extend(expand_sfx_auto(sb, starts, total, root))
    return sorted(entries, key=lambda item: (item["at"], item["auto"] or ""))
