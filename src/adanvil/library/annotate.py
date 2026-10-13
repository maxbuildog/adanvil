"""Agent annotations and catalog search."""

from __future__ import annotations

import difflib
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

from adanvil.vocab import check_emotions

from .catalog import Annotation, Catalog, Entry, load_catalog, update_catalog


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def annotate(root: Path, entry_id: str, **fields: object) -> Entry:
    """Merge annotation fields, replace list values, and persist the catalog."""

    updated_entry: Entry | None = None

    def change(catalog: Catalog) -> None:
        nonlocal updated_entry
        entry = catalog.entries.get(entry_id)
        if entry is None:
            close = difflib.get_close_matches(entry_id, catalog.entries, n=3, cutoff=0.35)
            hint = ", ".join(close) if close else "none"
            raise KeyError(f"unknown catalog entry {entry_id!r}; close matches: {hint}")

        values = entry.annotation.model_dump(mode="python")
        values.update(fields)
        values["updated"] = datetime.now(UTC).isoformat()
        annotation = Annotation.model_validate(values)
        updated_entry = entry.model_copy(update={"annotation": annotation}, deep=True)
        catalog.entries[entry_id] = updated_entry

    update_catalog(root, change)
    assert updated_entry is not None
    return updated_entry


def _matches_text(entry: Entry, query: str) -> bool:
    annotation = entry.annotation
    values = [
        entry.path,
        annotation.description or "",
        annotation.notes or "",
        entry.speech.text if entry.speech is not None else "",  # what the clip says
        *annotation.tags,
    ]
    return any(query in _fold(value) for value in values)


def version_matches(entry: Entry, requested: str) -> bool:
    """Whether the entry shows the product version ``requested``.

    A human ``product_version`` annotation always wins; without one, the guess made from the
    brand's reference photos (``auto_product``) is used.
    """

    human = entry.annotation.product_version
    if human is not None:
        return _fold(human) == _fold(requested)
    wanted = _fold(requested)
    guess = entry.auto_product
    if guess is not None and _fold(guess.version) == wanted:
        return True
    return any(
        shot.product_version is not None and _fold(shot.product_version) == wanted
        for shot in entry.shots
    )


def _matches_auto(entry: Entry, requested: list[str]) -> bool:
    from adanvil.vision.labels import resolve_tag

    available = {_fold(tag) for tag in entry.auto_tags}
    return all(_fold(resolve_tag(word) or word) in available for word in requested)


def _matches_list(values: list[str], requested: list[str]) -> bool:
    available = {_fold(value) for value in values}
    return all(_fold(value) in available for value in requested)


def search(
    root: Path,
    *,
    text: str | None = None,
    tags: list[str] | None = None,
    people: list[str] | None = None,
    kind: str | None = None,
    min_quality: int | None = None,
    product_visible: bool | None = None,
    product_version: str | None = None,
    emotion: list[str] | None = None,
    include_excluded: bool = False,
    min_duration: float | None = None,
    auto_tags: list[str] | None = None,
) -> list[Entry]:
    """Search annotations and technical metadata, sorted for agent selection."""

    query = _fold(text) if text is not None else None
    if emotion:  # an unknown emotion would silently match nothing: say so, as annotate does
        emotion = check_emotions(emotion)
    matches: list[Entry] = []
    for entry in load_catalog(root).entries.values():
        annotation = entry.annotation
        if not include_excluded and annotation.exclude:
            continue
        if query and not _matches_text(entry, query):
            continue
        if tags and not _matches_list(annotation.tags, tags):
            continue
        if people and not _matches_list(annotation.people, people):
            continue
        if kind is not None and entry.kind != kind:
            continue
        if product_version is not None and not version_matches(entry, product_version):
            continue
        if auto_tags and not _matches_auto(entry, auto_tags):
            continue
        if emotion and not _matches_list(annotation.emotion, emotion):
            continue
        if min_quality is not None and (
            annotation.quality is None or annotation.quality < min_quality
        ):
            continue
        if product_visible is not None and annotation.product_visible != product_visible:
            continue
        if min_duration is not None and (
            entry.duration is None or entry.duration < min_duration
        ):
            continue
        matches.append(entry)

    matches.sort(
        key=lambda entry: (
            -(entry.annotation.quality or 0),
            _fold(entry.path),
            entry.path,
        )
    )
    return matches


def entry_line(entry: Entry) -> str:
    """One compact line for an entry: id, path, duration, people, product visibility, flags."""

    annotation = entry.annotation
    duration = f"{entry.duration:.1f}s" if entry.duration is not None else "-"
    people = ",".join(annotation.people) if annotation.people else "-"
    visible = "-" if annotation.product_visible is None else str(annotation.product_visible).lower()
    parts = [
        entry.id,
        entry.path,
        duration,
        f"people={people}",
        f"product_visible={visible}",
    ]
    if annotation.product_version:
        parts.append(f"version={annotation.product_version}")
    elif entry.auto_product is not None:
        guess = entry.auto_product
        parts.append(f"version~{guess.version}({guess.confidence:.2f})")
    if entry.auto_tags:
        best = sorted(entry.auto_tags, key=lambda tag: -entry.auto_tags[tag])[:3]
        parts.append(f"auto={','.join(best)}")
    if annotation.emotion:
        parts.append(f"emotion={','.join(annotation.emotion)}")
    if annotation.quality is not None:
        parts.append(f"quality={annotation.quality}")
    if entry.kind == "audio":
        parts.append("audio")
    if entry.speech is not None and entry.speech.has_speech:
        parts.append("speech")
    if entry.speech is not None and entry.speech.status == "silent_audio":
        parts.append("silent_audio")
    if entry.speech is not None and entry.speech.separate_audio_candidates:
        parts.append(f"voice_files={len(entry.speech.separate_audio_candidates)}")
    if annotation.exclude:
        parts.append("excluded")
    return " | ".join(parts)


__all__ = ["annotate", "entry_line", "search", "version_matches"]
