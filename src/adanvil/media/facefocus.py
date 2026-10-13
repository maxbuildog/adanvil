"""``focus: "face"``: the segment fields, and the step that turns them into a crop.

Clips, images and the sides of splits and reveals accept ``focus: "face"`` and ``face_y``
(where the face sits, 0 top to 1 bottom, default 0.38). The text is kept as ``face_focus`` and
the ``focus`` pair keeps its default; before a segment is rendered ``with_framing`` finds the
main face (boxes stored by the vision pass, else a detection made on the spot) and attaches the
``Framing`` the crop follows. When there is no face data and no vision environment the segment
is rendered with the plain centred crop, and validation says so.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr, model_serializer, model_validator

from adanvil.ffmpeg import probe
from adanvil.media.cache import Cache
from adanvil.vision import client as vision_client
from adanvil.vision.analyse import analyse_media
from adanvil.vision.framing import DEFAULT_FACE_Y, FaceTrack, Framing, face_track
from adanvil.vision.store import VisionData, load_vision, save_vision, vision_relative

FACE_WORD = "face"
ON_DEMAND_FRAMES = 12  # frames looked at when a clip has no stored face data yet


class FaceFocusModel(BaseModel):
    """Shared fields of everything that can be framed on a face (see the module docstring)."""

    face_y: float = Field(
        default=DEFAULT_FACE_Y,
        ge=0.05,
        le=0.95,
        description='With focus "face": how far down the frame the face sits (0 top, 1 bottom).',
    )
    face_focus: bool = Field(default=False, exclude=True, repr=False)
    _framing: Framing | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _face_word(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("focus"), str):
            if data["focus"].strip().casefold() != FACE_WORD:
                raise ValueError('focus must be two numbers between 0 and 1, or "face"')
            rest = {key: value for key, value in data.items() if key != "focus"}
            return {**rest, "face_focus": True}
        return data

    @model_serializer(mode="wrap")
    def _dump_face(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        if self.face_focus:
            data["focus"] = FACE_WORD
        elif data.get("face_y") == DEFAULT_FACE_Y:
            data.pop("face_y", None)  # nothing to say about a face that is not used
        return data

    @property
    def framing(self) -> Framing | None:
        return self._framing

    def with_framing_value(self, framing: Framing | None) -> Any:
        """A copy that carries ``framing`` (the face the crop follows)."""

        clone = self.model_copy()
        clone._framing = framing
        return clone


# ---------------------------------------------------------------------------- face data


def face_data(
    root: Path, source: Path, kind: str, *, engine: Any | None = None
) -> VisionData | None:
    """The stored frames of ``source``, else a detection made now (and stored), else ``None``."""

    root = Path(root).resolve()
    relative = vision_relative(Cache.file_key(source))
    stored = load_vision(root, relative)
    if stored is not None:
        return stored
    active = engine if engine is not None else vision_client.get_engine()
    if active is None:
        return None
    info = probe(source)
    duration = float(info.duration or 0.0)
    if kind == "image" or duration <= 0:
        times = [0.0]
    else:
        count = ON_DEMAND_FRAMES
        times = [round((index + 0.5) * duration / count, 3) for index in range(count)]
    data = analyse_media(
        active,
        source,
        kind="image" if kind == "image" else "video",
        times=times,
        big=max(info.width or 0, info.height or 0) > 1920,
        embeddings=False,
        scratch=root / "cache" if (root / "cache").is_dir() else None,
    )
    save_vision(root, relative, data)
    return data


def segment_track(seg: Any, root: Path, source: Path, kind: str) -> FaceTrack | None:
    """The face track of a clip or image segment (its window, its crop), or ``None``."""

    data = face_data(root, source, kind)
    if data is None:
        return None
    info = probe(source)
    size = (float(info.width or 0), float(info.height or 0))
    crop = None
    if getattr(seg, "crop", None) is not None:
        crop = (float(seg.crop.x), float(seg.crop.y), float(seg.crop.w), float(seg.crop.h))
    if kind == "image":
        start, end = 0.0, 0.0
    else:
        start, end = float(seg.in_), float(seg.out)
    return face_track(
        data,
        start=start,
        end=end,
        crop=crop,
        source_size=size if size[0] > 0 and size[1] > 0 else None,
    )


def with_framing(seg: Any, root: Path, source: Path) -> Any:
    """``seg`` carrying the framing of its face when it asks for ``focus: "face"``.

    Without face data the segment is returned unchanged (the plain centred crop is used).
    """

    if not getattr(seg, "face_focus", False) or seg.framing is not None:
        return seg
    if seg.type == "image" and seg.fit == "contain":
        return seg  # nothing is cropped
    kind = "image" if seg.type == "image" else "video"
    track = segment_track(seg, root, source, kind)
    if track is None:
        return seg
    return seg.with_framing_value(Framing(track, seg.face_y))


__all__ = [
    "FACE_WORD",
    "FaceFocusModel",
    "face_data",
    "segment_track",
    "with_framing",
]
