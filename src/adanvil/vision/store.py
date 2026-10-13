"""Vision data of one file: ``cache/vision/<file key>.npz``.

Per frame that was looked at (the frames of the contact sheet): its time, the size of the
picture, its embedding, and the faces found in it (normalised boxes). The catalogue keeps only
a short summary; automatic tags, product versions and search are all derived from this file,
so they can be recomputed without decoding the footage again.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from adanvil.media.cache import Cache
from adanvil.project import resolve_in_project

from .engine import Face

STORE_VERSION = 1


def vision_relative(file_key: dict[str, Any]) -> str:
    """Project-relative path of the vision file of a file with this ``Cache.file_key``."""

    return Cache(Path("cache")).path_for("vision", file_key, suffix=".npz").as_posix()


@dataclass
class VisionData:
    """The frames of one file: times, sizes, embeddings (maybe none) and faces."""

    model: str
    times: np.ndarray  # (F,) seconds
    sizes: np.ndarray  # (F, 2) pixels of the analysed picture (width, height)
    embeddings: np.ndarray  # (F, D); shape (F, 0) when only the faces were computed
    face_frame: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    face_box: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    face_score: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    @property
    def frames(self) -> int:
        return int(len(self.times))

    @property
    def has_embeddings(self) -> bool:
        return self.embeddings.ndim == 2 and self.embeddings.shape[1] > 0

    def faces(self, frame: int) -> list[Face]:
        """The faces of one frame, the biggest first."""

        found = [
            Face(tuple(float(value) for value in self.face_box[row]), float(self.face_score[row]))
            for row in np.flatnonzero(self.face_frame == frame)
        ]
        return sorted(found, key=lambda face: face.area, reverse=True)

    def face_counts(self) -> list[int]:
        return [int((self.face_frame == frame).sum()) for frame in range(self.frames)]

    @classmethod
    def build(
        cls,
        model: str,
        times: list[float],
        sizes: list[tuple[int, int]],
        embeddings: np.ndarray | None,
        faces: list[list[Face]],
    ) -> VisionData:
        frame_index: list[int] = []
        boxes: list[tuple[float, float, float, float]] = []
        scores: list[float] = []
        for index, found in enumerate(faces):
            for face in found:
                frame_index.append(index)
                boxes.append(face.box)
                scores.append(face.score)
        return cls(
            model=model,
            times=np.asarray(times, dtype=np.float32),
            sizes=np.asarray(sizes, dtype=np.int32).reshape(-1, 2),
            embeddings=(
                np.asarray(embeddings, dtype=np.float32)
                if embeddings is not None
                else np.zeros((len(times), 0), dtype=np.float32)
            ),
            face_frame=np.asarray(frame_index, dtype=np.int32),
            face_box=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            face_score=np.asarray(scores, dtype=np.float32),
        )


def save_vision(root: Path, relative: str, data: VisionData) -> None:
    """Atomically write the vision file."""

    destination = resolve_in_project(Path(root), relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    meta = json.dumps({"version": STORE_VERSION, "model": data.model})
    temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.npz")
    try:
        np.savez_compressed(
            temporary,
            meta=np.asarray(meta),
            times=data.times,
            sizes=data.sizes,
            embeddings=data.embeddings.astype(np.float16),
            face_frame=data.face_frame,
            face_box=data.face_box,
            face_score=data.face_score,
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_vision(root: Path, relative: str | None) -> VisionData | None:
    """The vision file, or ``None`` when it is missing, unreadable or from another version."""

    if not relative:
        return None
    try:
        path = resolve_in_project(Path(root), relative)
        with np.load(path, allow_pickle=False) as stored:
            meta = json.loads(str(stored["meta"]))
            if meta.get("version") != STORE_VERSION:
                return None
            return VisionData(
                model=str(meta.get("model", "")),
                times=stored["times"].astype(np.float32),
                sizes=stored["sizes"].astype(np.int32),
                embeddings=stored["embeddings"].astype(np.float32),
                face_frame=stored["face_frame"].astype(np.int32),
                face_box=stored["face_box"].astype(np.float32),
                face_score=stored["face_score"].astype(np.float32),
            )
    except (OSError, ValueError, KeyError, TypeError):
        return None


__all__ = ["STORE_VERSION", "VisionData", "load_vision", "save_vision", "vision_relative"]
