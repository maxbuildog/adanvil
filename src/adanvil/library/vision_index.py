"""The vision pass of indexing: frame embeddings, faces, automatic tags, product versions.

It runs after the files are measured (like the speech pass) and only when the optional vision
environment is installed (``adanvil vision-setup``); otherwise each entry gets the status
``unavailable: ...`` and indexing still succeeds. Per entry, the frames of its contact sheet
are embedded and searched for faces once (``cache/vision/<file key>.npz``); automatic tags
and product versions are then derived from those stored embeddings, so a new label list or
new reference photos re-tag the library without decoding any video again.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from adanvil.media.cache import Cache
from adanvil.project import resolve_in_project
from adanvil.vision import client as vision_client
from adanvil.vision.analyse import BIG_SOURCE_SIDE, analyse_media
from adanvil.vision.engine import VisionEngine, probabilities
from adanvil.vision.labels import (
    AUTO_TAG_LIMIT,
    AUTO_TAG_MIN,
    label_prompts,
    labels_signature,
)
from adanvil.vision.store import VisionData, load_vision, save_vision, vision_relative
from adanvil.vision.versions import (
    FrameMatch,
    ProductReference,
    brand_versions,
    embed_references,
    match_frames,
    references_signature,
    version_photos,
)

from .catalog import AutoProduct, Catalog, Entry, Shot, VisionInfo, load_catalog, update_catalog

VisionRequest = bool | Literal["auto"]
DONE = "done"


def vision_plan(vision: VisionRequest) -> VisionRequest | None:
    """``True`` (redo everything), ``"auto"``, or ``None`` when no vision work is wanted."""

    if vision not in {True, False, "auto"}:
        raise ValueError("vision must be true, false, or 'auto'")
    return None if vision is False else vision


def _now() -> str:
    return datetime.now(UTC).isoformat()


def sheet_times(entry: Entry) -> list[float]:
    """The times (seconds) of the frames the vision pass looks at: those of the contact sheet."""

    if entry.kind == "image":
        return [0.0]
    from .index import _sheet_timestamps

    return list(_sheet_timestamps(float(entry.duration or 0.0), list(entry.shots))[0])


def _is_big(entry: Entry) -> bool:
    return max(entry.width or 0, entry.height or 0) > BIG_SOURCE_SIDE


def _wanted(entry: Entry) -> bool:
    return entry.kind in {"video", "image"}


def _stale(
    project_root: Path,
    entry: Entry,
    mode: VisionRequest,
    tags_sig: str,
    versions_sig: str,
    model: str | None,
) -> str | None:
    """Why the entry needs the vision pass: ``analyse`` (frames), ``derive`` (tags, versions)
    or ``None`` when it is up to date."""

    info = entry.vision
    if mode is True or info is None or info.status != DONE:
        return "analyse"
    if info.file is None or not resolve_in_project(project_root, info.file).is_file():
        return "analyse"
    if model is not None and info.model != model:
        return "analyse"
    if info.tags_sig != tags_sig or (info.versions_sig or "") != versions_sig:
        return "derive"
    return None


def _signatures(project_root: Path) -> tuple[str, str]:
    photos = version_photos(project_root, brand_versions(project_root))
    return labels_signature(), references_signature(photos)


def pending_vision(root: Path, entries: list[Entry], vision: VisionRequest) -> int:
    """How many of ``entries`` a vision pass would work on (0 without an engine)."""

    mode = vision_plan(vision)
    if mode is None or not vision_client.available():
        return 0
    project_root = Path(root).resolve()
    tags_sig, versions_sig = _signatures(project_root)
    expected = vision_client.expected_model()
    return sum(
        1
        for entry in entries
        if _wanted(entry) and _stale(project_root, entry, mode, tags_sig, versions_sig, expected)
    )


def derive(
    data: VisionData,
    engine: VisionEngine,
    labels: np.ndarray,
    references: list[ProductReference],
    entry: Entry,
) -> tuple[dict[str, float], list[Shot], AutoProduct | None]:
    """Automatic tags, the shots with their product guess, and the entry's product guess."""

    tags: dict[str, float] = {}
    if data.has_embeddings and labels.size:
        scores = probabilities(engine, data.embeddings, labels).max(axis=0)
        ranked = sorted(
            ((tag, float(score)) for tag, score in zip(_tag_names(), scores, strict=True)),
            key=lambda item: (-item[1], item[0]),
        )
        tags = {
            tag: round(score, 3) for tag, score in ranked[:AUTO_TAG_LIMIT] if score >= AUTO_TAG_MIN
        }
    matches: list[FrameMatch | None] = (
        match_frames(data.embeddings, references)
        if data.has_embeddings and references
        else [None] * data.frames
    )
    shots: list[Shot] = []
    for position, shot in enumerate(entry.shots):
        last = position == len(entry.shots) - 1
        inside = [
            match
            for index, match in enumerate(matches)
            if match is not None and _in_shot(float(data.times[index]), shot, last)
        ]
        best = max(inside, key=lambda item: item.similarity, default=None)
        shots.append(
            shot.model_copy(
                update={
                    "product_version": best.version if best else None,
                    "product_confidence": best.confidence if best else None,
                }
            )
        )
    found = [match for match in matches if match is not None]
    product: AutoProduct | None = None
    if found:
        counts: dict[str, list[float]] = {}
        for match in found:
            counts.setdefault(match.version, []).append(match.confidence)
        version = max(counts, key=lambda name: (len(counts[name]), sum(counts[name])))
        product = AutoProduct(
            version=version,
            confidence=round(sum(counts[version]) / len(counts[version]), 3),
            frames=len(counts[version]),
        )
    return tags, shots, product


def _in_shot(moment: float, shot: Shot, last: bool) -> bool:
    return shot.start <= moment < shot.end or (last and moment == shot.end)


def _tag_names() -> list[str]:
    from adanvil.vision.labels import LABELS

    return [label.tag for label in LABELS]


def _info(
    entry: Entry, data: VisionData, relative: str, tags_sig: str, versions_sig: str
) -> VisionInfo:
    counts = data.face_counts()
    return VisionInfo(
        status=DONE,
        model=data.model,
        file=relative,
        frames=data.frames,
        faces_max=max(counts, default=0),
        face_frames=sum(1 for count in counts if count),
        tags_sig=tags_sig,
        versions_sig=versions_sig,
        indexed=_now(),
    )


def _store(project_root: Path, updates: dict[str, tuple[dict[str, Any], dict[str, Any]]]) -> None:
    """Write the vision fields onto the entries that still describe the same file."""

    def change(catalog: Catalog) -> None:
        for identifier, (file_key, fields) in updates.items():
            stored = catalog.entries.get(identifier)
            if stored is not None and stored.file_key == file_key:
                catalog.entries[identifier] = stored.model_copy(update=fields)

    if updates:
        update_catalog(project_root, change)


def vision_phase(
    project_root: Path,
    identifiers: set[str],
    mode: VisionRequest,
    result: Any,
    *,
    engine: VisionEngine | None = None,
    progress: Callable[[str], None] | None = None,
) -> None:
    """Run the vision pass on the entries of this run that need it (``result.vision`` gets
    one status per entry)."""

    catalog = load_catalog(project_root)
    entries = [
        catalog.entries[identifier]
        for identifier in sorted(identifiers)
        if identifier in catalog.entries and _wanted(catalog.entries[identifier])
    ]
    if not entries:
        return
    active = engine if engine is not None else vision_client.get_engine()
    tags_sig, versions_sig = _signatures(project_root)
    expected = (active.model if active is not None else "") or vision_client.expected_model()
    todo = [
        (entry, reason)
        for entry in entries
        if (reason := _stale(project_root, entry, mode, tags_sig, versions_sig, expected))
    ]
    for entry in entries:
        if all(entry is not item[0] for item in todo):
            result.vision[entry.id] = DONE
    if not todo:
        return
    if active is None:
        status = vision_client.status()
        updates = {}
        for entry, _reason in todo:
            result.vision[entry.id] = status
            if entry.vision is None or entry.vision.status != status:
                updates[entry.id] = (entry.file_key, {"vision": VisionInfo(status=status)})
        _store(project_root, updates)
        return
    try:
        _run(project_root, todo, active, mode, tags_sig, versions_sig, result, progress)
    except vision_client.VisionUnavailableError as exc:
        status = f"failed: {exc}"[:300]
        for entry, _reason in todo:
            if entry.id not in result.vision or result.vision[entry.id] != DONE:
                result.vision[entry.id] = status
        result.errors.append({"path": "(vision)", "message": status})


def _run(
    project_root: Path,
    todo: list[tuple[Entry, str]],
    engine: VisionEngine,
    mode: VisionRequest,
    tags_sig: str,
    versions_sig: str,
    result: Any,
    progress: Callable[[str], None] | None,
) -> None:
    total = len(todo)
    if progress:
        progress(f"vision 0/{total} preparing the labels")
    labels = engine.embed_texts(label_prompts())
    photos = version_photos(project_root, brand_versions(project_root))
    references = embed_references(engine, photos) if photos else []
    for number, (entry, reason) in enumerate(todo, start=1):
        if progress:
            progress(f"vision {number - 1}/{total} {entry.path}")
        try:
            relative = vision_relative(entry.file_key)
            data = None if reason == "analyse" else load_vision(project_root, relative)
            if data is None or not data.has_embeddings or data.model != engine.model:
                source = resolve_in_project(project_root, entry.path)
                if Cache.file_key(source) != entry.file_key:
                    raise OSError("the file changed during the vision pass")
                data = analyse_media(
                    engine,
                    source,
                    kind=entry.kind,
                    times=sheet_times(entry),
                    big=_is_big(entry),
                    scratch=resolve_in_project(project_root, "cache"),
                )
                save_vision(project_root, relative, data)
            tags, shots, product = derive(data, engine, labels, references, entry)
            fields = {
                "auto_tags": tags,
                "shots": shots,
                "auto_product": product,
                "vision": _info(entry, data, relative, tags_sig, versions_sig),
            }
            _store(project_root, {entry.id: (entry.file_key, fields)})
            result.vision[entry.id] = DONE
        except vision_client.VisionUnavailableError:
            raise
        except Exception as exc:  # one bad file must not stop the pass
            status = f"failed: {exc}"[:300]
            result.vision[entry.id] = status
            result.errors.append({"path": entry.path, "message": f"vision {status}"})
            _store(
                project_root,
                {entry.id: (entry.file_key, {"vision": VisionInfo(status=status)})},
            )
    if progress:
        progress(f"vision {total}/{total} done")


__all__ = [
    "DONE",
    "VisionRequest",
    "derive",
    "pending_vision",
    "sheet_times",
    "vision_phase",
    "vision_plan",
]
