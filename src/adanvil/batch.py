"""Batch rendering: many storyboards (variants, formats) in one detached job.

``plan_items`` turns storyboards and a list of formats into the renders to make, ``run_batch``
renders them in order with ONE cache, so a segment shared by several variants (the body, the end
card) is built once, and ``write_manifest`` writes ``renders/<batch>/manifest.csv``, a table ready
for an ads manager:

``file, variant ids, format, duration, score, hook text, CTA id, CTA text, music licence,
quality``

``quality`` is the quality rendered AND scored (``preview`` or ``final``; ``draft`` is an alias of
``preview``): a score read on a preview is lower than the final's, so a media buyer compares
only rows of the same quality. ``width`` and ``height`` of a render are those of the file.

``CTA id`` is the id of the variant's ``cta`` axis (empty without one). When the batch has several
CTA variants, ``CTA text`` lists only the texts that differ between them, each with its role
(``caption: 65,90 € au lieu de 89,90 € | badge: Satisfaite ou remboursée 60 jours``; a phrase
shared by two CTAs but in another role still tells them apart), so two calls to action are told
apart at a glance; with no CTA axis, or when nothing differs, it is the whole text of the call
to action.

Naming: a variant is ``<base>_<hook>_<cta>`` (``name_pattern`` of ``variants_build``), a format
adds ``_9x16``, ``_4x5`` or ``_1x1``: ``<base>_<hook>_<cta>_4x5.mp4``. Every file of a batch is
in ``renders/<batch>/``, next to its ``.report.json`` and ``_sheet.jpg``.
"""

from __future__ import annotations

import copy
import csv
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from adanvil.ffmpeg import probe
from adanvil.formats import FORMATS, format_name, ratio_of, reformat, requested_formats
from adanvil.music import licence_label
from adanvil.safezones import platforms_for_size
from adanvil.storyboard.patch import storyboard_name
from adanvil.storyboard.schema import normalise_quality

MANIFEST_COLUMNS = [
    "file",
    "variant ids",
    "format",
    "duration",
    "score",
    "hook text",
    "CTA id",
    "CTA text",
    "music licence",
    "quality",
]
# Older manifests: before V2.8d (no CTA id), before V2.9 (no music licence), before V2.9b (no
# quality). A row of an older header is read by column name; the new columns stay empty.
_LEGACY_COLUMNS = [c for c in MANIFEST_COLUMNS if c not in ("CTA id", "music licence", "quality")]
_V28D_COLUMNS = [c for c in MANIFEST_COLUMNS if c not in ("music licence", "quality")]
_V29_COLUMNS = [c for c in MANIFEST_COLUMNS if c != "quality"]
CTA_AXIS = "cta"
CTA_SEPARATOR = " | "
HOOK_SECONDS = 3.0
_BATCH_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


def batch_name(name: str) -> str:
    """A safe folder name for ``renders/<batch>/``."""

    if not _BATCH_NAME.match(name) or name.startswith("."):
        raise ValueError(
            f"batch name must be 1 to 80 letters, digits, '_', '.' or '-' (got {name!r})"
        )
    return name


def variant_ids(root: Path, name: str) -> dict[str, str]:
    """The ids ``variants_build`` stored for the storyboard ``name`` (empty if it is not one)."""

    try:
        side = Path(root) / "storyboards" / f"{storyboard_name(name)}.variant.json"
        data = json.loads(side.read_text(encoding="utf-8"))
        ids = data.get("ids")
        return {str(k): str(v) for k, v in ids.items()} if isinstance(ids, dict) else {}
    except (OSError, ValueError, AttributeError):
        return {}


def existing_files(root: Path, paths: list[str]) -> list[str]:
    """The files among ``paths`` (relative to the project) that already exist. A storyboard is
    never replaced silently, whoever made it: renders may be replaced, storyboards may not."""

    return [relative for relative in paths if (Path(root) / relative).is_file()]


def plan_items(
    root: Path,
    documents: list[dict[str, Any]],
    formats: list[str] | None,
    batch: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """The renders of a batch: one item per storyboard and format, each with the storyboard
    object to render (its output going to ``renders/<batch>/``). Returns the items and notes."""

    batch = batch_name(batch)
    folder = f"renders/{batch}"
    items: list[dict[str, Any]] = []
    notes: list[str] = []
    seen: set[str] = set()
    for document in documents:
        source = str(document.get("name", "storyboard"))
        ids = variant_ids(root, source)
        ratios = requested_formats(document, formats)
        pieces: list[tuple[str | None, dict[str, Any]]] = []
        if not ratios:
            native = copy.deepcopy(document)
            native.pop("formats", None)
            output = dict(native["output"]) if isinstance(native.get("output"), dict) else {}
            output["path"] = f"{folder}/{source}.mp4"
            native["output"] = output
            fmt = native.get("format") or {}
            width, height = int(fmt.get("width", 1080)), int(fmt.get("height", 1920))
            pieces.append((ratio_of(width, height), native))
        else:
            for ratio in ratios:
                moved, moved_notes = reformat(document, ratio, root=root, output_dir=folder)
                notes.extend(f"{moved['name']}: {note}" for note in moved_notes)
                pieces.append((ratio, moved))
        for ratio, piece in pieces:
            name = str(piece["name"])
            if name in seen:
                raise ValueError(f"two renders of the batch are named {name!r}")
            seen.add(name)
            items.append(
                {"name": name, "source": source, "ratio": ratio, "ids": ids, "storyboard": piece}
            )
    return items, notes


def _slot_windows(model: Any, labels: tuple[str, ...]) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    for segment, (start, end) in zip(model.segments, model.timeline(), strict=True):
        label = str(getattr(segment, "label", "") or "").casefold()
        if any(label == wanted or label.startswith(f"{wanted}_") for wanted in labels):
            windows.append((start, end))
    return windows


def _cta_window(model: Any) -> tuple[float, float] | None:
    """The last labelled CTA part (``cta``, ``end_card``, ``endcard``: the last run of such
    segments), else the segments after the last labelled one; ``None`` when there is neither."""

    windows = _slot_windows(model, ("cta", "end_card", "endcard"))
    timeline = model.timeline()
    if windows:
        end = windows[-1][1]
        begin = windows[-1][0]
        for first, last in reversed(windows[:-1]):
            if abs(last - begin) < 1e-6:  # next to the one after it: the same part
                begin = first
            else:
                break
        return begin, end
    labelled = [
        number
        for number, segment in enumerate(model.segments)
        if str(getattr(segment, "label", "") or "").strip()
    ]
    if labelled and labelled[-1] < len(timeline) - 1:
        return timeline[labelled[-1] + 1][0], timeline[-1][1]
    return None


_ROLE_RANK = {"caption": 0, "comment": 1, "badge": 2}


def _timed_texts(model: Any, entries: list[Any]) -> list[tuple[float, float, str, str, float]]:
    """Every text on screen with its time span, its role (``caption``, ``comment`` or ``badge``)
    and its prominence (smaller is more prominent): captions, comments and badge stickers."""

    items: list[tuple[float, float, str, str, float]] = []
    for entry in entries:
        size = getattr(entry.caption, "size", None)
        bigger = float(size) / 1000.0 if isinstance(size, (int, float)) else 0.0
        rank = _ROLE_RANK["caption"] - bigger
        items.append((entry.start, entry.end, entry.text, "caption", rank))
    total = model.total_duration()
    for overlay in model.overlays:
        role = None
        if getattr(overlay, "type", None) == "comment":
            role = "comment"
        elif getattr(overlay, "type", None) == "sticker" and overlay.kind == "badge":
            role = "badge"
        if role and overlay.text:
            end = overlay.end if overlay.end is not None else total
            span = (float(overlay.start), float(end))
            items.append((*span, str(overlay.text), role, _ROLE_RANK[role]))
    return sorted(items, key=lambda item: item[0])


def _cta_text(model: Any, entries: list[Any], total: float) -> str:
    """The texts shown during the call to action part, joined (see ``_cta_pieces``)."""

    return CTA_SEPARATOR.join(_cta_pieces(model, entries, total))


def _cta_pieces(model: Any, entries: list[Any], total: float) -> list[str]:
    """The texts shown during the call to action part, one by one, each with its role
    (``caption: 65,90 € au lieu de 89,90 €``, ``badge: Satisfaite ou remboursée 60 jours``), most
    prominent first (captions, largest first, then comments, then badges): what tells two CTAs
    apart, even when they share phrases in other roles.

    An end card with no text of its own widens the part backwards, one segment at a time (not
    before 40% of the ad), until some text is found."""

    window = _cta_window(model)
    texts = _timed_texts(model, entries)
    if window is not None:
        begin, end = window
        timeline = model.timeline()
        while True:
            seen: list[tuple[float, float, str]] = []
            for start, stop, text, role, rank in texts:
                piece = f"{role}: {text}"
                if (
                    start < end - 1e-6
                    and stop > begin + 1e-6
                    and piece not in [p for _r, _s, p in seen]
                ):
                    seen.append((rank, start, piece))
            if seen:
                return [piece for _rank, _start, piece in sorted(seen)]
            earlier = [a for a, _ in timeline if a < begin - 1e-6 and a >= 0.4 * total]
            if not earlier:
                break
            begin = max(earlier)
    late = [f"{role}: {text}" for start, _stop, text, role, _rank in texts if start >= 0.6 * total]
    return [late[-1]] if late else []


def copy_texts(root: Path, document: dict[str, Any]) -> tuple[str, str]:
    """The hook text and the whole call to action text of a storyboard (see ``copy_parts``)."""

    hook, pieces = copy_parts(root, document)
    return hook, CTA_SEPARATOR.join(pieces)


def copy_parts(root: Path, document: dict[str, Any]) -> tuple[str, list[str]]:
    """The hook text (the first caption of the first 3 s, or of the ``hook`` slot) and the call
    to action text (the texts of the last CTA part, each with its role, joined with `` | ``: the
    ``cta`` or ``end_card`` segments, else the segments after the last labelled one, else the
    last caption of the last 40% of the ad) of a storyboard, for the manifest. Empty when there is
    no caption there. The call to action comes as the list of its texts."""

    try:
        from adanvil.brand import load_brand
        from adanvil.storyboard.craft import collect_captions
        from adanvil.storyboard.schema import Storyboard

        model = Storyboard.model_validate(document)
        brand = None
        brand_file = Path(root) / "brand.yaml"
        if brand_file.is_file():
            try:
                brand = load_brand(brand_file)
            except Exception:
                brand = None
        timeline = model.timeline()
        entries, _ = collect_captions(
            model, Path(root), brand, [a for a, _ in timeline], [b for _, b in timeline]
        )
    except Exception:
        return "", []
    entries = sorted(entries, key=lambda entry: entry.start)
    total = model.total_duration()
    hook_windows = _slot_windows(model, ("hook",))
    hook_end = max((end for _, end in hook_windows), default=HOOK_SECONDS)
    hook = next((entry.text for entry in entries if entry.start < hook_end - 1e-6), "")
    return hook, _cta_pieces(model, entries, total)


def _cta_texts(renders: list[dict[str, Any]]) -> dict[str, str]:
    """The ``CTA text`` of each CTA variant of the batch: only the texts that are not in every
    CTA variant (what tells them apart), in the order they appear. The whole text when there is
    one CTA only, or when the variants share all their texts."""

    pieces: dict[str, list[str]] = {}
    for entry in renders:
        if entry.get("status") != "done":
            continue
        ident = str((entry.get("ids") or {}).get(CTA_AXIS, ""))
        if ident and ident not in pieces:
            pieces[ident] = list(entry.get("cta_pieces") or [])
    shared = (
        set.intersection(*(set(texts) for texts in pieces.values())) if len(pieces) > 1 else set()
    )
    found: dict[str, str] = {}
    for ident, texts in pieces.items():
        own = [text for text in texts if text not in shared]
        found[ident] = CTA_SEPARATOR.join(own or texts)
    return found


def _row(entry: dict[str, Any], texts: dict[str, str] | None = None) -> list[str]:
    ids = entry.get("ids") or {}
    cta_id = str(ids.get(CTA_AXIS, ""))
    cta_text = (texts or {}).get(cta_id) if cta_id else None
    if cta_text is None:
        cta_text = entry.get("cta_text", "")
    return [
        Path(str(entry["file"])).name,
        ";".join(f"{axis}={ident}" for axis, ident in ids.items()),
        entry.get("format") or "",
        f"{float(entry['duration']):.2f}",
        "" if entry.get("score") is None else str(entry["score"]),
        entry.get("hook_text", ""),
        cta_id,
        cta_text,
        entry.get("music_licence", ""),
        entry.get("quality") or "",
    ]


def write_manifest(root: Path, batch: str, renders: list[dict[str, Any]]) -> Path:
    """``renders/<batch>/manifest.csv`` (UTF-8, one row per successful render).

    A manifest that already exists is MERGED by file name: a row of the same file is replaced
    (in place), the other rows are kept, new files go at the end. Rendering one more format
    into a batch folder therefore never loses the earlier rows."""

    path = Path(root) / "renders" / batch_name(batch) / "manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list[str]] = []
    if path.is_file():
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                found = list(csv.reader(handle))
            if found and found[0] == MANIFEST_COLUMNS:
                rows = [row for row in found[1:] if len(row) == len(MANIFEST_COLUMNS)]
            elif found and found[0] in (_LEGACY_COLUMNS, _V28D_COLUMNS, _V29_COLUMNS):
                header = found[0]
                for row in found[1:]:
                    if len(row) != len(header):
                        continue
                    values = dict(zip(header, row, strict=True))
                    if "CTA id" not in values:  # before V2.8d: take the CTA id from the ids
                        ids = dict(
                            part.split("=", 1) for part in values["variant ids"].split(";")
                            if "=" in part
                        )
                        values["CTA id"] = ids.get(CTA_AXIS, "")
                    rows.append([values.get(column, "") for column in MANIFEST_COLUMNS])
        except (OSError, csv.Error, UnicodeDecodeError):
            rows = []
    position = {row[0]: number for number, row in enumerate(rows)}
    texts = _cta_texts(renders)
    for entry in renders:
        if entry.get("status") != "done":
            continue
        row = _row(entry, texts)
        if row[0] in position:
            rows[position[row[0]]] = row
        else:
            position[row[0]] = len(rows)
            rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_COLUMNS)
        writer.writerows(rows)
    return path


def _score(root: Path, video: Path, document: dict[str, Any], ratio: str | None) -> int | None:
    from adanvil.qc import ad_score

    platforms = list(FORMATS[ratio].placement) if ratio else None
    if platforms is None:
        fmt = document.get("format") or {}
        found = platforms_for_size(int(fmt.get("width", 1080)), int(fmt.get("height", 1920)))
        platforms = list(found) if found else None
    report = ad_score(root, video, storyboard=document, platforms=platforms)
    return int(report.score)


def run_batch(
    root: Path,
    inputs: dict[str, Any],
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
    publish: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Render the items of a batch in order with one shared cache and write the manifest.

    ``progress`` gets one event per render; ``publish`` gets the partial result after each
    render (so ``job_status`` lists the outputs as they appear). A render that fails is listed
    with its error and the batch goes on.
    """

    from adanvil.media import Cache
    from adanvil.render import compose
    from adanvil.storyboard import load_storyboard

    root = Path(root).resolve()
    batch = batch_name(str(inputs["batch"]))
    quality = str(normalise_quality(inputs.get("quality", "final")))
    with_score = bool(inputs.get("score", True))
    items = list(inputs["items"])
    cache = Cache(root / "cache")
    started = time.perf_counter()
    renders: list[dict[str, Any]] = []

    def snapshot(done: bool) -> dict[str, Any]:
        return {
            "batch": batch,
            "quality": quality,
            "done": done,
            "count": len(items),
            "total_seconds": round(time.perf_counter() - started, 2),
            "cache_builds": cache.builds,
            "manifest": f"renders/{batch}/manifest.csv" if done else None,
            "renders": list(renders),
        }

    for number, item in enumerate(items):
        name = str(item["name"])
        document = item["storyboard"]
        if progress is not None:
            progress(
                {
                    "stage": "rendering",
                    "done": number,
                    "total": len(items),
                    "unit": "renders",
                    "message": f"rendering {number + 1}/{len(items)}: {name}",
                }
            )
        begun = time.perf_counter()
        builds_before = cache.builds
        entry: dict[str, Any] = {
            "name": name,
            "ids": item.get("ids") or {},
            "format": item.get("ratio"),
        }
        try:
            model = load_storyboard(document)
            report = compose(model, root=root, cache=cache, quality=quality)
            report_path = Path(report.path).with_suffix(".report.json")
            report_path.write_text(
                json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8"
            )
            hook, cta_pieces = copy_parts(root, document)
            # The size of the file written (a preview is half the storyboard's format).
            produced = probe(report.path)
            real_width = produced.width or model.format.width
            real_height = produced.height or model.format.height
            entry.update(
                status="done",
                file=report.path.resolve().relative_to(root).as_posix(),
                sheet=report.sheet.resolve().relative_to(root).as_posix(),
                report=report_path.resolve().relative_to(root).as_posix(),
                duration=round(float(report.duration), 3),
                width=real_width,
                height=real_height,
                quality=report.quality,
                warnings=list(report.warnings),
                hook_text=hook,
                cta_text=CTA_SEPARATOR.join(cta_pieces),
                music=report.music,
                music_licence=licence_label(report.music),
                cta_pieces=cta_pieces,
                score=None,
            )
            if with_score:  # a preview render is scored on the preview file
                try:
                    entry["score"] = _score(root, report.path, document, item.get("ratio"))
                except Exception as exc:  # a score that cannot be read must not lose the render
                    entry["score_error"] = str(exc)[:200]
        except Exception as exc:
            entry.update(status="failed", error=str(exc))
        entry["seconds"] = round(time.perf_counter() - begun, 2)
        entry["cache_builds"] = cache.builds - builds_before
        renders.append(entry)
        if publish is not None:
            publish(snapshot(False))
    write_manifest(root, batch, renders)
    result = snapshot(True)
    if progress is not None:
        progress(
            {
                "stage": "complete",
                "done": len(items),
                "total": len(items),
                "unit": "renders",
                "message": "batch complete",
            }
        )
    return result


__all__ = [
    "MANIFEST_COLUMNS",
    "batch_name",
    "copy_parts",
    "copy_texts",
    "format_name",
    "plan_items",
    "run_batch",
    "existing_files",
    "variant_ids",
    "write_manifest",
]
