"""Search the footage by sentence: shots ranked by text-image similarity.

``search_shots`` embeds the sentence and compares it with the stored frame embeddings of the
entries it is given (the ones that passed the usual filters), so it works over the whole
library without opening a single contact sheet. A shot is as good as its best frame. The
answer is a list of ``ShotHit`` (entry, time range, best frame) and, with ``search_mosaic``,
one picture with the best frames side by side.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from adanvil.ffmpeg import extract_frame
from adanvil.project import resolve_in_project
from adanvil.vision import client as vision_client
from adanvil.vision.engine import VisionEngine, probabilities
from adanvil.vision.store import VisionData, load_vision

from .annotate import _fold
from .catalog import Entry, Shot

DEFAULT_LIMIT = 12
MAX_LIMIT = 48
MOSAIC_COLUMNS = 4
MOSAIC_WIDTH = 270


@dataclass
class ShotHit:
    entry: Entry
    start: float
    end: float
    time: float  # the best frame of the shot
    similarity: float  # cosine between the sentence and that frame
    score: float  # match probability (0 to 1)
    shot: int | None = None  # 1-based number of the shot, None for a still image

    def line(self, rank: int) -> str:
        span = (
            "image"
            if self.entry.kind == "image"
            else f"{self.start:.1f}-{self.end:.1f}s t={self.time:.1f}s"
        )
        parts = [f"#{rank}", self.entry.id, self.entry.path, span, f"score={self.score:.2f}"]
        if self.shot is not None:
            parts.insert(3, f"S{self.shot}")
        return " | ".join(parts)

    def as_dict(self, rank: int) -> dict[str, object]:
        return {
            "rank": rank,
            "id": self.entry.id,
            "path": self.entry.path,
            "kind": self.entry.kind,
            "shot": self.shot,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "time": round(self.time, 3),
            "score": round(self.score, 3),
            "similarity": round(self.similarity, 4),
        }


def _shot_ranges(
    entry: Entry, data: VisionData
) -> list[tuple[int | None, float, float, Shot | None]]:
    if entry.kind == "image" or not entry.shots:
        end = float(entry.duration or (float(data.times.max()) if data.frames else 0.0))
        return [(None, 0.0, end, None)]
    return [
        (number, shot.start, shot.end, shot) for number, shot in enumerate(entry.shots, start=1)
    ]


def search_shots(
    root: Path,
    query: str,
    entries: list[Entry],
    *,
    engine: VisionEngine | None = None,
    limit: int = DEFAULT_LIMIT,
    product_version: str | None = None,
) -> tuple[list[ShotHit], int]:
    """The best shots of ``entries`` for ``query`` and how many entries have no vision data.

    With ``product_version``, an entry without a human ``product_version`` annotation only
    offers the shots that the reference photos matched to that version.
    """

    sentence = query.strip()
    if not sentence:
        raise ValueError("query must not be empty")
    limit = max(1, min(int(limit), MAX_LIMIT))
    active = engine if engine is not None else vision_client.get_engine()
    if active is None:
        raise RuntimeError(
            f"semantic search needs the vision environment: {vision_client.status()}"
        )
    project_root = Path(root).resolve()
    text = active.embed_texts([sentence])
    hits: list[ShotHit] = []
    missing = 0
    wanted = _fold(product_version) if product_version is not None else None
    for entry in entries:
        if entry.kind not in {"video", "image"}:
            continue
        data = (
            load_vision(project_root, entry.vision.file)
            if entry.vision is not None and entry.vision.status == "done"
            else None
        )
        if data is None or not data.has_embeddings:
            missing += 1
            continue
        similarity = (data.embeddings @ text[0]).astype(np.float32)
        scores = probabilities(active, data.embeddings, text)[:, 0]
        human = entry.annotation.product_version
        for number, start, end, shot in _shot_ranges(entry, data):
            if wanted is not None and human is None:
                if shot is None:
                    if entry.auto_product is None or _fold(entry.auto_product.version) != wanted:
                        continue
                elif shot.product_version is None or _fold(shot.product_version) != wanted:
                    continue
            last = number is None or number == len(entry.shots)
            frames = [
                index
                for index in range(data.frames)
                if start <= float(data.times[index]) < end
                or (last and float(data.times[index]) == end)
            ]
            if not frames:
                continue
            best = max(frames, key=lambda index: float(similarity[index]))
            hits.append(
                ShotHit(
                    entry,
                    start,
                    end,
                    float(data.times[best]),
                    float(similarity[best]),
                    float(scores[best]),
                    number,
                )
            )
    hits.sort(key=lambda hit: (-hit.similarity, hit.entry.path, hit.start))
    return hits[:limit], missing


def search_mosaic(root: Path, hits: list[ShotHit], query: str) -> Path | None:
    """One picture with the best frame of each hit, numbered like the result list."""

    if not hits:
        return None
    project_root = Path(root).resolve()
    folder = resolve_in_project(project_root, "cache/vision/search")
    folder.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(
        json.dumps([query, [(hit.entry.id, round(hit.time, 3)) for hit in hits]]).encode("utf-8")
    ).hexdigest()[:12]
    output = folder / f"search-{key}.jpg"
    tiles: list[Image.Image] = []
    for hit in hits:
        tile = _frame(project_root, hit, folder)
        tiles.append(tile)
    cell = max(tile.height for tile in tiles)
    label_height = 34
    rows = -(-len(tiles) // MOSAIC_COLUMNS)
    canvas = Image.new(
        "RGB",
        (MOSAIC_COLUMNS * MOSAIC_WIDTH, rows * (cell + label_height)),
        (32, 32, 32),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for number, (hit, tile) in enumerate(zip(hits, tiles, strict=True)):
        x = (number % MOSAIC_COLUMNS) * MOSAIC_WIDTH
        y = (number // MOSAIC_COLUMNS) * (cell + label_height)
        canvas.paste(tile, (x, y))
        tile.close()
        when = "img" if hit.entry.kind == "image" else f"{hit.time:.1f}s"
        draw.text(
            (x + 4, y + cell + 4),
            f"#{number + 1} {hit.entry.id[:28]}",
            font=font,
            fill=(245, 245, 245),
        )
        draw.text(
            (x + 4, y + cell + 18),
            f"{when} score {hit.score:.2f}",
            font=font,
            fill=(255, 220, 90),
        )
    canvas.save(output, format="JPEG", quality=86)
    return output


def _frame(root: Path, hit: ShotHit, scratch: Path) -> Image.Image:
    source = resolve_in_project(root, hit.entry.path)
    if hit.entry.kind == "image":
        with Image.open(source) as opened:
            picture = opened.convert("RGB")
    else:
        temporary = scratch / f".frame-{uuid.uuid4().hex}.png"
        try:
            extract_frame(source, hit.time, temporary, width=MOSAIC_WIDTH)
            with Image.open(temporary) as opened:
                picture = opened.convert("RGB")
        except Exception:
            return Image.new("RGB", (MOSAIC_WIDTH, MOSAIC_WIDTH * 16 // 9), (80, 80, 80))
        finally:
            temporary.unlink(missing_ok=True)
    height = max(1, round(picture.height * MOSAIC_WIDTH / picture.width))
    return picture.resize((MOSAIC_WIDTH, height), Image.Resampling.LANCZOS)


__all__ = ["DEFAULT_LIMIT", "MAX_LIMIT", "ShotHit", "search_mosaic", "search_shots"]
