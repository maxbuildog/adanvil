"""Read access to the speech index of a project: catalogue entries and their sidecar words."""

from __future__ import annotations

import unicodedata
from pathlib import Path

from adanvil.library.catalog import Entry, load_catalog

from . import store
from .analyze import FillerLists, find_fillers
from .model import SpeechData


def normalise_path(value: str) -> str:
    """The form catalogue paths are stored in (NFC, forward slashes, no ``./``)."""

    return unicodedata.normalize("NFC", value).replace("\\", "/").removeprefix("./")


class SpeechLibrary:
    """The speech of the clips of a project, loaded once and looked up by project path."""

    def __init__(self, root: Path, lists: FillerLists | None = None) -> None:
        self.root = Path(root)
        self.lists = lists
        self._entries: dict[str, Entry] | None = None
        self._data: dict[str, SpeechData | None] = {}
        self._fillers: dict[str, set[int]] = {}

    def _by_path(self) -> dict[str, Entry]:
        if self._entries is None:
            try:
                catalog = load_catalog(self.root)
            except (OSError, ValueError):
                self._entries = {}
            else:
                self._entries = {entry.path: entry for entry in catalog.entries.values()}
        return self._entries

    def entry(self, path: str) -> Entry | None:
        """The catalogue entry of a project path, or None when the file is not indexed."""

        return self._by_path().get(normalise_path(path))

    def data(self, path: str) -> SpeechData | None:
        """The timed words of a clip, or None (not indexed, no transcript, no sidecar)."""

        key = normalise_path(path)
        if key not in self._data:
            entry = self._by_path().get(key)
            speech = entry.speech if entry is not None else None
            usable = speech is not None and speech.status == "done"
            self._data[key] = store.read_sidecar(self.root, speech.words_file) if usable else None
        return self._data[key]

    def fillers(self, path: str) -> set[int]:
        """Indexes of the filler words of a clip (empty when it has no transcript)."""

        key = normalise_path(path)
        if key not in self._fillers:
            data = self.data(key)
            self._fillers[key] = find_fillers(data.words, self.lists) if data else set()
        return self._fillers[key]


__all__ = ["SpeechLibrary", "normalise_path"]
