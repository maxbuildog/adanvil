"""Speech sidecar files: ``cache/speech/<file key>.json``, one per indexed clip.

The catalogue keeps only a short summary of the speech of an entry; the timed words and
the audio level live in the sidecar, so ``catalog.json`` stays small and readable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from adanvil.media.cache import Cache
from adanvil.project import resolve_in_project
from adanvil.safeio import replace_text

from .model import SpeechData, SpeechWord

SIDECAR_VERSION = 1
SIDECAR_DIR = "cache/speech"


def sidecar_relative(file_key: dict[str, Any]) -> str:
    """Project-relative path of the sidecar of a file with this ``Cache.file_key``."""

    path = Cache(Path("cache")).path_for("speech", file_key, suffix=".json")
    return path.as_posix()


def write_sidecar(root: Path, relative: str, data: SpeechData, *, source: str, model: str) -> None:
    """Atomically write the sidecar of ``source`` (project-relative path)."""

    destination = resolve_in_project(Path(root), relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "version": SIDECAR_VERSION,
        "source": source,
        "model": model,
        "language": data.language,
        "duration": None if data.duration is None else round(float(data.duration), 3),
        "words": [
            {
                "start": round(float(word.start), 3),
                "end": round(float(word.end), 3),
                "text": word.text,
                "probability": round(float(word.probability), 3),
            }
            for word in data.words
        ],
        "levels": {
            "step": data.level_step,
            "db": [round(float(value), 1) for value in data.levels_db],
        },
    }
    # a new file renamed into place: a link planted at the destination is replaced, not followed
    replace_text(
        destination, json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


def read_sidecar(root: Path, relative: str | None) -> SpeechData | None:
    """Load a sidecar, or ``None`` when it is missing or unreadable."""

    if not relative:
        return None
    try:
        path = resolve_in_project(Path(root), relative)
        document = json.loads(path.read_text(encoding="utf-8"))
        words = [
            SpeechWord(
                start=float(item["start"]),
                end=float(item["end"]),
                text=str(item["text"]),
                probability=float(item.get("probability", 1.0)),
            )
            for item in document.get("words", [])
        ]
        levels = document.get("levels") or {}
        duration = document.get("duration")
        return SpeechData(
            words=words,
            language=document.get("language"),
            duration=None if duration is None else float(duration),
            level_step=float(levels.get("step", 0.1)) or 0.1,
            levels_db=[float(value) for value in levels.get("db", [])],
            model=document.get("model") if isinstance(document.get("model"), str) else None,
        )
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _sidecar_file(root: Path, relative: str) -> Path | None:
    """The file a catalogue entry names as its sidecar, only when it is a ``.json`` directly in
    ``cache/speech/``: catalog.json is a project file anyone can plant, and what it names must
    never make AdAnvil delete a file of the project."""

    path = resolve_in_project(Path(root), relative)
    folder = resolve_in_project(Path(root), SIDECAR_DIR)
    if os.path.normcase(path.parent) != os.path.normcase(folder) or path.suffix != ".json":
        return None
    return path


def remove_sidecar(root: Path, relative: str | None) -> None:
    """Delete a sidecar that no entry points to any more (best effort). Only a file of
    ``cache/speech/`` is ever deleted; any other path is ignored."""

    if not relative:
        return
    try:
        path = _sidecar_file(root, relative)
        if path is not None:
            path.unlink(missing_ok=True)
    except (OSError, ValueError, TypeError):
        pass


__all__ = [
    "SIDECAR_DIR",
    "SIDECAR_VERSION",
    "read_sidecar",
    "remove_sidecar",
    "sidecar_relative",
    "write_sidecar",
]
