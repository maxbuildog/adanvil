"""Find the separate voice recordings of a video whose own sound is silent or missing.

Phones and recorders often keep the voice in its own file (``HOOK1.mov`` filmed with the
sound off and ``VOIX OFF/HOOK1.1.m4a`` recorded next to it). The catalogue lists, for such a
video, the audio files nearby whose names share words with it, so the agent can lay one of
them with ``audio_from``. Syncing is not done here: the agent checks it on a preview.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path, PurePosixPath

from .catalog import Catalog, Entry, load_catalog, update_catalog

NO_SOUND_STATUSES = ("silent_audio", "not attempted: no audio")
MAX_CANDIDATES = 5

# Words of a file name that say what kind of file it is, not which take it belongs to.
_GENERIC = frozenset(
    {
        "audio", "voice", "voix", "off", "son", "sound", "track", "rec", "recording", "copy",
        "copie", "final", "new", "clip", "video", "mov", "mp4", "m4a", "wav", "mp3", "aac",
    }
)  # fmt: skip


def _tokens(name: str) -> list[str]:
    """The lower-case ASCII words of a file name without its extension."""

    stem = PurePosixPath(name.replace("\\", "/")).stem
    folded = unicodedata.normalize("NFKD", stem.casefold())
    plain = "".join(char for char in folded if not unicodedata.combining(char))
    return [token for token in re.split(r"[^a-z0-9]+", plain) if token]


def _nearby(video: str, audio: str) -> bool:
    """Whether the audio file is in the video's folder, a sibling folder, the parent folder or a
    folder below the video's folder."""

    video_dir = PurePosixPath(video).parent
    audio_dir = PurePosixPath(audio).parent
    return (
        audio_dir == video_dir
        or audio_dir.parent == video_dir.parent
        or audio_dir == video_dir.parent
        or video_dir in audio_dir.parents
    )


def _score(video_tokens: list[str], audio_tokens: list[str]) -> int:
    """How much a name looks like the video's (0 when it does not at all)."""

    if not video_tokens or not audio_tokens:
        return 0
    shared = set(video_tokens) & set(audio_tokens)
    meaningful = {token for token in shared if token not in _GENERIC and not token.isdigit()}
    prefix = (
        audio_tokens[: len(video_tokens)] == video_tokens
        or video_tokens[: len(audio_tokens)] == audio_tokens
    )
    if not meaningful and not prefix and "".join(video_tokens) != "".join(audio_tokens):
        return 0
    return len(shared) + (2 if prefix else 0)


def separate_audio_candidates(video: str, audio_paths: list[str]) -> list[str]:
    """The audio files (project paths) that may be the voice of ``video``, best first.

    A candidate is in the same, a sibling, the parent or a child folder and shares a word with
    the video's name (``HOOK1.mov`` and ``HOOK1.1.m4a``; a bare number or ``voice`` does not
    count). At most five, ranked by shared words, then same folder, then path.
    """

    video_tokens = _tokens(video)
    ranked: list[tuple[int, int, str]] = []
    for audio in audio_paths:
        if not _nearby(video, audio):
            continue
        score = _score(video_tokens, _tokens(audio))
        if score:
            same_folder = PurePosixPath(audio).parent == PurePosixPath(video).parent
            ranked.append((-score, 0 if same_folder else 1, audio))
    ranked.sort()
    return [audio for _score_value, _folder, audio in ranked[:MAX_CANDIDATES]]


def _wanted(catalog: Catalog) -> dict[str, list[str]]:
    """The candidates each video without sound should have, by entry id (changed ones only)."""

    audio = [entry.path for entry in catalog.entries.values() if entry.kind == "audio"]
    changed: dict[str, list[str]] = {}
    for identifier, entry in catalog.entries.items():
        speech = entry.speech
        if entry.kind != "video" or speech is None or speech.status not in NO_SOUND_STATUSES:
            continue
        found = separate_audio_candidates(entry.path, audio)
        if found != speech.separate_audio_candidates:
            changed[identifier] = found
    return changed


def refresh_separate_audio(root: Path) -> int:
    """Recompute the candidates of every video without sound; returns how many changed.

    Run after indexing: audio files indexed later (another folder, another run) reach the videos
    that were indexed before them.
    """

    if not _wanted(load_catalog(root)):
        return 0
    count = 0

    def change(catalog: Catalog) -> None:
        nonlocal count
        for identifier, found in _wanted(catalog).items():
            entry: Entry = catalog.entries[identifier]
            assert entry.speech is not None
            speech = entry.speech.model_copy(update={"separate_audio_candidates": found})
            catalog.entries[identifier] = entry.model_copy(update={"speech": speech})
            count += 1

    update_catalog(root, change)
    return count


__all__ = ["MAX_CANDIDATES", "refresh_separate_audio", "separate_audio_candidates"]
