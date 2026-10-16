"""Writes that never go through a symbolic link planted in a project.

``Path.write_text`` opens the existing file, so a project received from someone else could
hold ``brand/site_facts.json`` as a link to any file of the user and have it overwritten.
These helpers write a new file next to the target (exclusive create, which refuses an
existing name or link) and move it over the target with ``os.replace``: the rename replaces
a link itself, never the file it points to.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

_SHARING_RETRIES = 50
_SHARING_DELAY = 0.02


def _replace(source: Path, target: Path) -> None:
    # On Windows a file that another process is reading cannot be replaced for a few
    # milliseconds (sharing violation).
    for attempt in range(_SHARING_RETRIES):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if os.name != "nt" or attempt == _SHARING_RETRIES - 1:
                raise
            time.sleep(_SHARING_DELAY)


def replace_bytes(path: str | os.PathLike[str], data: bytes) -> Path:
    """Write ``data`` to ``path`` through a new file and a rename; return the path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "xb") as handle:
            handle.write(data)
        _replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def replace_text(
    path: str | os.PathLike[str],
    text: str,
    *,
    encoding: str = "utf-8",
    newline: str | None = None,
) -> Path:
    """Write ``text`` to ``path`` through a new file and a rename; return the path.

    ``newline`` has the meaning of :func:`open`: ``None`` translates ``\\n`` to the
    platform line ending, ``""`` or ``"\\n"`` keeps it as written.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "x", encoding=encoding, newline=newline) as handle:
            handle.write(text)
        _replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


__all__ = ["replace_bytes", "replace_text"]
