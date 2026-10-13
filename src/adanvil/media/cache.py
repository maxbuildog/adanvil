"""Small content-addressed cache for rendered media stages."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_BUILD_LOCKS: dict[str, threading.Lock] = {}
_BUILD_LOCKS_GUARD = threading.Lock()


class Cache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        # How many files this object had to build (not find): a batch reads it to show that
        # the segments shared by its variants were built once.
        self.builds = 0
        self._count = threading.Lock()

    def path_for(self, kind: str, key: dict[str, Any], suffix: str = ".mp4") -> Path:
        canonical = json.dumps(
            key,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()[:16]
        return self.root / kind / f"{digest}{suffix}"

    @staticmethod
    @contextmanager
    def building(final: Path) -> Iterator[None]:
        """Serialise threads building the same cache file (parallel segments can share one)."""

        with _BUILD_LOCKS_GUARD:
            lock = _BUILD_LOCKS.setdefault(str(final), threading.Lock())
        with lock:
            yield

    def get_or_build(
        self,
        kind: str,
        key: dict[str, Any],
        builder: Callable[[Path], None],
    ) -> Path:
        final = self.path_for(kind, key)
        if final.is_file():
            return final

        with self.building(final):
            if final.is_file():
                return final
            final.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(f"{final}.tmp{final.suffix}")
            temporary.unlink(missing_ok=True)
            try:
                builder(temporary)
                os.replace(temporary, final)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            with self._count:
                self.builds += 1
        return final

    @staticmethod
    def file_key(path: Path) -> dict[str, Any]:
        resolved = Path(path).resolve()
        stat = resolved.stat()
        return {
            "path": str(resolved),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
