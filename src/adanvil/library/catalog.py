"""Persistent, JSON-compatible models for the footage library."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from adanvil.jobs import _ProjectLock, _retry_sharing
from adanvil.project import CATALOG_FILE
from adanvil.vocab import check_emotions

CATALOG_LOCK_FILE = f"{CATALOG_FILE}.lock"
_LOCK_TIMEOUT = 60.0


class Shot(BaseModel):
    start: float
    end: float
    yavg: float | None = None
    # The product version this shot seems to show, guessed from the brand's reference photos
    # (``adanvil vision-setup``); a human ``product_version`` annotation of the entry wins.
    product_version: str | None = None
    product_confidence: float | None = None

    @model_validator(mode="after")
    def valid_range(self) -> Shot:
        if self.start < 0 or self.end < self.start:
            raise ValueError("shot must have a non-negative range")
        return self

    @model_serializer(mode="wrap")
    def _omit_unset_product(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        for key in ("product_version", "product_confidence"):
            if data.get(key) is None:
                data.pop(key, None)  # keeps catalog.json short for footage without versions
        return data


class VisionInfo(BaseModel):
    """What the catalogue keeps of the vision pass of an entry; the frames are in ``file``.

    ``status`` is ``done`` or why there is no vision data (``unavailable: ...``,
    ``failed: ...``). ``tags_sig`` and ``versions_sig`` say which label list and which
    reference photos the automatic tags and product guesses were computed with.
    """

    model_config = ConfigDict(extra="forbid")

    status: str
    model: str | None = None
    file: str | None = None  # project-relative: cache/vision/<file key>.npz
    frames: int = 0
    faces_max: int = 0  # most faces seen in one frame
    face_frames: int = 0  # frames with at least one face
    tags_sig: str | None = None
    versions_sig: str | None = None
    indexed: str | None = None

    @field_validator("file")
    @classmethod
    def normalise_file(cls, value: str | None) -> str | None:
        return None if value is None else _normalise_relative(value)


class AutoProduct(BaseModel):
    """The product version most of the entry's product frames resemble."""

    model_config = ConfigDict(extra="forbid")

    version: str
    confidence: float
    frames: int  # frames that show the product


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    product_visible: bool | None = None
    product_version: str | None = None
    emotion: list[str] = Field(default_factory=list)
    quality: int | None = Field(default=None, ge=1, le=5)
    usable: list[tuple[float, float, str]] = Field(default_factory=list)
    notes: str | None = None
    exclude: bool = False
    exclude_reason: str | None = None
    updated: str | None = None

    @field_validator("emotion", mode="before")
    @classmethod
    def valid_emotion(cls, value: Any) -> Any:
        """Accept one emotion or a list; each must be in the shared vocabulary."""

        return check_emotions(value)

    @field_validator("usable", mode="before")
    @classmethod
    def normalise_usable(cls, value: Any) -> list[tuple[float, float, str]]:
        """Accept the compact JSON annotation shapes used by agents."""

        accepted = (
            "usable must be a list of [start, end, note] or [start, end] entries, "
            "or {start: ..., end: ..., note: ...} entries; accepted shapes: "
            "[[start, end, note]], [[start, end]], "
            "or [{\"start\": ..., \"end\": ..., \"note\": ...}]"
        )
        if not isinstance(value, list):
            raise ValueError(accepted)

        normalised: list[tuple[float, float, str]] = []
        for item in value:
            if isinstance(item, dict):
                if set(item) != {"start", "end", "note"}:
                    raise ValueError(accepted)
                start_value, end_value, note_value = (
                    item["start"],
                    item["end"],
                    item["note"],
                )
            elif isinstance(item, list | tuple) and len(item) in {2, 3}:
                start_value, end_value = item[:2]
                note_value = item[2] if len(item) == 3 else ""
            else:
                raise ValueError(accepted)

            if isinstance(start_value, bool) or isinstance(end_value, bool):
                raise ValueError(accepted)
            if not isinstance(note_value, str):
                raise ValueError(accepted)
            try:
                start = float(start_value)
                end = float(end_value)
            except (TypeError, ValueError):
                raise ValueError(accepted) from None
            normalised.append((start, end, note_value))
        return normalised


class EntrySpeech(BaseModel):
    """What the catalogue keeps of the speech of a clip; the timed words are in a sidecar.

    ``status`` is ``done``, ``silent_audio`` (the audio stream carries no sound: nothing to
    transcribe) or why there is no transcript (``not attempted: ...``, ``unavailable: ...``,
    ``failed: ...``). A video without sound lists in ``separate_audio_candidates`` the audio
    files nearby whose names share words with it: the voice may have been recorded apart.
    """

    model_config = ConfigDict(extra="forbid")

    status: str  # "done", or why there is no transcript ("not attempted: ...", "unavailable: ...")
    language: str | None = None
    text: str = ""
    words_file: str | None = None  # project-relative sidecar: cache/speech/<file key>.json
    has_speech: bool = False
    separate_audio_candidates: list[str] = Field(default_factory=list)  # project paths

    @field_validator("words_file")
    @classmethod
    def normalise_words_file(cls, value: str | None) -> str | None:
        return None if value is None else _normalise_relative(value)

    @model_serializer(mode="wrap")
    def _omit_empty_candidates(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        if not data.get("separate_audio_candidates"):
            data.pop("separate_audio_candidates", None)  # keeps catalog.json short
        return data


def _normalise_relative(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("catalog paths must be strings")
    normalised = unicodedata.normalize("NFC", value).replace("\\", "/")
    posix = PurePosixPath(normalised)
    windows = PureWindowsPath(normalised)
    if (
        not normalised
        or "\x00" in normalised
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or not posix.parts
        or any(part in {"", ".."} for part in posix.parts)
    ):
        raise ValueError("entry path must be relative and must not escape the project")
    return "/".join(posix.parts)


class Entry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    path: str
    kind: str
    file_key: dict[str, Any]
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    rotation: int = 0
    hdr: bool = False
    has_audio: bool = False
    yavg: float | None = None
    shots: list[Shot] = Field(default_factory=list)
    text_boxes: list[tuple[int, int, int, int]] = Field(default_factory=list)
    sheet: str | None = None
    annotation: Annotation = Field(default_factory=Annotation)
    speech: EntrySpeech | None = None
    # Vision pass (optional environment): automatic tags {label: best score}, never mixed
    # with the human ``annotation.tags``, the file of frame data, the product version guess.
    auto_tags: dict[str, float] = Field(default_factory=dict)
    vision: VisionInfo | None = None
    auto_product: AutoProduct | None = None
    indexed: str

    @model_serializer(mode="wrap")
    def _omit_empty_vision(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        for key in ("auto_tags", "vision", "auto_product"):
            if not data.get(key):
                data.pop(key, None)  # keeps catalog.json short for footage without vision
        return data

    @field_validator("path")
    @classmethod
    def normalise_path(cls, value: str) -> str:
        return _normalise_relative(value)

    @field_validator("sheet")
    @classmethod
    def normalise_sheet(cls, value: str | None) -> str | None:
        return None if value is None else _normalise_relative(value)


class Catalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    entries: dict[str, Entry] = Field(default_factory=dict)


def entry_id(rel_path: str) -> str:
    """Return a readable stable id for a relative path."""

    normalised = _normalise_relative(rel_path)
    stem = PurePosixPath(normalised).stem
    ascii_stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_stem).strip("-").lower()[:40].rstrip("-")
    if not slug:
        slug = "file"
    digest = hashlib.sha1(normalised.encode("utf-8")).hexdigest()[:6]
    return f"{slug}-{digest}"


def load_catalog(root: Path) -> Catalog:
    """Load a project catalog, returning an empty v1 catalog if absent."""

    path = Path(root) / CATALOG_FILE
    if not path.exists():
        return Catalog()
    text = _retry_sharing(lambda: path.read_text(encoding="utf-8"))
    return Catalog.model_validate(json.loads(text))


def save_catalog(root: Path, catalog: Catalog) -> None:
    """Atomically replace the project's catalog JSON."""

    destination = Path(root) / CATALOG_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.close(fd)
        temporary.write_text(
            json.dumps(
                catalog.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=False,
            )
            + "\n",
            encoding="utf-8",
        )
        _retry_sharing(lambda: os.replace(temporary, destination))
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def catalog_lock(root: Path) -> Iterator[None]:
    """Hold the exclusive catalog lock; every catalog write goes through it."""

    lock = _ProjectLock(Path(root) / CATALOG_LOCK_FILE, timeout=_LOCK_TIMEOUT, poll=0.05)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def _timestamp(value: str | None) -> float:
    if not value:
        return float("-inf")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return float("-inf")
    return parsed.timestamp() if parsed.tzinfo else parsed.replace(tzinfo=UTC).timestamp()


def merge_entries(stored: Entry, incoming: Entry) -> Entry:
    """Merge two versions of one entry: newest measurement, newest annotation win.

    Ties keep the stored version, so an annotation saved while an index was running is
    never overwritten by the index's older copy of it.
    """

    entry = incoming if _timestamp(incoming.indexed) > _timestamp(stored.indexed) else stored
    newest = (
        incoming.annotation
        if _timestamp(incoming.annotation.updated) > _timestamp(stored.annotation.updated)
        else stored.annotation
    )
    if newest is entry.annotation:
        return entry
    return entry.model_copy(update={"annotation": newest}, deep=True)


def update_catalog(root: Path, change: Callable[[Catalog], None]) -> Catalog:
    """Read, change and write the catalog as one step under the catalog lock."""

    with catalog_lock(root):
        catalog = load_catalog(root)
        change(catalog)
        save_catalog(root, catalog)
    return catalog


def merge_into_catalog(
    root: Path, entries: Iterable[Entry] = (), removed: Iterable[str] = ()
) -> Catalog:
    """Merge entries (by id) into the stored catalog and drop ``removed`` ids."""

    incoming = list(entries)
    gone = list(removed)

    def change(catalog: Catalog) -> None:
        for entry in incoming:
            stored = catalog.entries.get(entry.id)
            catalog.entries[entry.id] = entry if stored is None else merge_entries(stored, entry)
        for identifier in gone:
            catalog.entries.pop(identifier, None)

    return update_catalog(root, change)


__all__ = [
    "Annotation",
    "AutoProduct",
    "Catalog",
    "Entry",
    "EntrySpeech",
    "Shot",
    "VisionInfo",
    "catalog_lock",
    "entry_id",
    "load_catalog",
    "merge_entries",
    "merge_into_catalog",
    "save_catalog",
    "update_catalog",
]
