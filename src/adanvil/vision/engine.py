"""What the vision features need from an engine, and a deterministic fake for tests.

An engine turns pictures (file paths) and texts into unit-length vectors of one shared space,
and finds faces. The real one is the worker of the optional vision environment
(``adanvil vision-setup``); ``FakeVisionEngine`` gives the same answers for synthetic pictures
without any model, so storage, ranking, tags, version matching and face framing are testable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class Face:
    """A face box in 0..1 of the picture (left, top, right, bottom) and its detector score."""

    box: tuple[float, float, float, float]
    score: float = 1.0

    @property
    def centre(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class FacePicture:
    """The faces found in one picture and the size (pixels) of the picture they were found in."""

    size: tuple[int, int]
    faces: list[Face] = field(default_factory=list)


class VisionEngine(Protocol):
    """Embeddings and faces for the library (see the module docstring)."""

    model: str
    logit_scale: float
    logit_bias: float

    def embed_images(self, paths: Sequence[Path]) -> np.ndarray: ...

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray: ...

    def detect_faces(self, paths: Sequence[Path]) -> list[FacePicture]: ...

    def close(self) -> None: ...


def probabilities(engine: VisionEngine, images: np.ndarray, texts: np.ndarray) -> np.ndarray:
    """Match probability of every image row with every text row (SigLIP's sigmoid score)."""

    if images.size == 0 or texts.size == 0:
        return np.zeros((len(images), len(texts)), dtype=np.float32)
    logits = images @ texts.T * float(engine.logit_scale) + float(engine.logit_bias)
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))).astype(np.float32)


# ------------------------------------------------------------------------------- the fake

FAKE_FACE_RGB = (224, 172, 140)  # the colour the fake engine takes for a face
_FAKE_DIM = 8
_FAKE_WORDS = {
    0: {"red", "rouge"},
    1: {"green", "vert"},
    2: {"blue", "bleu"},
    3: {"face", "person", "woman", "man", "visage", "femme", "homme", "sad", "triste"},
    4: {"dark", "sombre", "night"},
    5: {"bright", "light", "clair", "white"},
}


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return (vector / norm).astype(np.float32) if norm > 0 else vector.astype(np.float32)


class FakeVisionEngine:
    """Deterministic stand-in for tests: colours and a flat-coloured "face" decide everything.

    A picture embeds as its dominant colour (red, green, blue), its share of face-coloured
    pixels, and how dark or bright it is; a text embeds by the colour, face and light words it
    holds (English or French), anything else lands on two filler axes. A face is any blob of
    ``FAKE_FACE_RGB``; blobs separated by empty columns are different faces.
    """

    model = "fake-vision-1"
    logit_scale = 20.0
    logit_bias = -10.0

    def __init__(self) -> None:
        self.calls: list[str] = []

    @staticmethod
    def _open(path: Path) -> Image.Image:
        with Image.open(path) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB")

    def _face_mask(self, picture: Image.Image) -> np.ndarray:
        small = picture.copy()
        small.thumbnail((128, 128))
        pixels = np.asarray(small, dtype=np.int16)
        return np.abs(pixels - np.array(FAKE_FACE_RGB, dtype=np.int16)).max(axis=-1) <= 14

    def embed_images(self, paths: Sequence[Path]) -> np.ndarray:
        self.calls.append("embed_images")
        rows = []
        for path in paths:
            picture = self._open(Path(path))
            small = picture.copy()
            small.thumbnail((64, 64))
            pixels = np.asarray(small, dtype=np.float32)
            red, green, blue = (float(pixels[..., index].mean()) for index in range(3))
            luma = (red + green + blue) / 3 / 255
            skin = float(self._face_mask(picture).mean())
            vector = np.array(
                [
                    max(red - (green + blue) / 2, 0.0) / 255,
                    max(green - (red + blue) / 2, 0.0) / 255,
                    max(blue - (red + green) / 2, 0.0) / 255,
                    1.0 if skin >= 0.004 else 0.0,
                    max(0.0, 0.3 - luma) * 2.0,
                    max(0.0, luma - 0.6) * 2.0,
                    0.05,
                    0.05,
                ],
                dtype=np.float32,
            )
            rows.append(_unit(vector))
        return np.stack(rows) if rows else np.zeros((0, _FAKE_DIM), dtype=np.float32)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append("embed_texts")
        rows = []
        for text in texts:
            words = {word.strip(".,;:!?'\"") for word in text.casefold().split()}
            vector = np.zeros(_FAKE_DIM, dtype=np.float32)
            for axis, vocabulary in _FAKE_WORDS.items():
                vector[axis] = float(len(words & vocabulary))
            if not vector.any():
                digest = hashlib.sha1(text.encode("utf-8")).digest()
                vector[6 + digest[0] % 2] = 1.0
            rows.append(_unit(vector))
        return np.stack(rows) if rows else np.zeros((0, _FAKE_DIM), dtype=np.float32)

    def detect_faces(self, paths: Sequence[Path]) -> list[FacePicture]:
        self.calls.append("detect_faces")
        results: list[FacePicture] = []
        for path in paths:
            picture = self._open(Path(path))
            mask = self._face_mask(picture)
            height, width = mask.shape
            faces: list[Face] = []
            columns = mask.any(axis=0)
            start: int | None = None
            for column in range(width + 1):
                inside = column < width and bool(columns[column])
                if inside and start is None:
                    start = column
                elif not inside and start is not None:
                    rows = np.flatnonzero(mask[:, start:column].any(axis=1))
                    if mask[:, start:column].sum() >= 6:
                        faces.append(
                            Face(
                                (
                                    start / width,
                                    float(rows[0]) / height,
                                    column / width,
                                    (float(rows[-1]) + 1) / height,
                                ),
                                0.99,
                            )
                        )
                    start = None
            results.append(FacePicture(picture.size, faces))
        return results

    def close(self) -> None:
        return None


__all__ = [
    "FAKE_FACE_RGB",
    "Face",
    "FacePicture",
    "FakeVisionEngine",
    "VisionEngine",
    "probabilities",
]
