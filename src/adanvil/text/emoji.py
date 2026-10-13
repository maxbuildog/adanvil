from __future__ import annotations

import os
import urllib.request
import warnings
from pathlib import Path
from typing import Protocol

from PIL import Image


def _is_base(codepoint: int, following: str | None = None) -> bool:
    if 0x1F000 <= codepoint <= 0x1FAFF:
        return True
    if 0x2600 <= codepoint <= 0x27BF:
        return True
    if 0x2B00 <= codepoint <= 0x2BFF:
        return True
    if 0x2300 <= codepoint <= 0x23FF:
        return True
    if 0x2190 <= codepoint <= 0x21FF:
        return following == "\ufe0f"
    if codepoint in {0x00A9, 0x00AE}:
        return following == "\ufe0f"
    return False


def _consume_emoji(chars: list[str], index: int) -> int:
    codepoint = ord(chars[index])
    if 0x1F1E6 <= codepoint <= 0x1F1FF:
        if index + 1 < len(chars) and 0x1F1E6 <= ord(chars[index + 1]) <= 0x1F1FF:
            return index + 2
        return index
    if chars[index] in "0123456789#*" and index + 2 < len(chars):
        if chars[index + 1] == "\ufe0f" and chars[index + 2] == "\u20e3":
            return index + 3
    if not _is_base(codepoint, chars[index + 1] if index + 1 < len(chars) else None):
        return index

    cursor = index + 1
    while cursor < len(chars):
        if chars[cursor] == "\ufe0f":
            cursor += 1
        elif 0x1F3FB <= ord(chars[cursor]) <= 0x1F3FF:
            cursor += 1
        elif chars[cursor] == "\u200d" and cursor + 1 < len(chars):
            # A ZWJ continuation is part of the same cluster.  Continue
            # consuming variation selectors/modifiers after that base too.
            cursor += 2
        else:
            break
    return cursor


def split_runs(line: str) -> list[tuple[bool, str]]:
    """Split *line* into ordinary text and extended emoji grapheme clusters."""

    chars = list(line)
    runs: list[tuple[bool, str]] = []
    text_start = 0
    index = 0
    while index < len(chars):
        end = _consume_emoji(chars, index)
        if end == index:
            index += 1
            continue
        if text_start < index:
            runs.append((False, "".join(chars[text_start:index])))
        runs.append((True, "".join(chars[index:end])))
        index = end
        text_start = index
    if text_start < len(chars):
        runs.append((False, "".join(chars[text_start:])))
    return runs


def cluster_codepoints(cluster: str, strip_fe0f: bool = False) -> str:
    codepoints = [f"{ord(char):x}" for char in cluster if not (strip_fe0f and char == "\ufe0f")]
    return "-".join(codepoints)


class EmojiProvider(Protocol):
    def get(self, cluster: str, size: int) -> Image.Image | None: ...


def _variants(cluster: str) -> list[str]:
    preferred_strip = "\u200d" not in cluster
    preferred = cluster_codepoints(cluster, strip_fe0f=preferred_strip)
    alternate = cluster_codepoints(cluster, strip_fe0f=not preferred_strip)
    with_fe0f = cluster_codepoints(cluster if "\ufe0f" in cluster else f"{cluster}\ufe0f")
    return list(dict.fromkeys((preferred, alternate, with_fe0f)))


class TwemojiProvider:
    def __init__(self, cache_dir: Path | None = None, version: str = "17.0.3") -> None:
        default_cache = Path.home() / ".cache" / "adanvil"
        root = cache_dir or Path(os.environ.get("ADANVIL_CACHE", str(default_cache)))
        self.cache_dir = Path(root) / "emoji" / f"twemoji-{version}"
        self.version = version
        self._warned: set[str] = set()

    def _warn_missing(self, cluster: str) -> None:
        if cluster not in self._warned:
            warnings.warn(
                f"Could not load Twemoji cluster {cluster_codepoints(cluster)}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned.add(cluster)

    def get(self, cluster: str, size: int) -> Image.Image | None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for codepoints in _variants(cluster):
            target = self.cache_dir / f"{codepoints}.png"
            if not target.is_file():
                url = (
                    f"https://cdn.jsdelivr.net/gh/jdecked/twemoji@{self.version}/assets/72x72/"
                    f"{codepoints}.png"
                )
                try:
                    urllib.request.urlretrieve(url, str(target))
                except Exception:
                    continue
            try:
                with Image.open(target) as source:
                    return source.convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
            except (OSError, ValueError):
                continue
        self._warn_missing(cluster)
        return None


class DirectoryProvider:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def get(self, cluster: str, size: int) -> Image.Image | None:
        for codepoints in _variants(cluster):
            candidate = self.path / f"{codepoints}.png"
            try:
                with Image.open(candidate) as source:
                    return source.convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
            except (FileNotFoundError, OSError, ValueError):
                continue
        return None


class NoEmojiProvider:
    def get(self, cluster: str, size: int) -> Image.Image | None:
        return None


def directory_emoji(path: Path) -> list[str]:
    """The emoji characters of a directory provider: PNGs named by codepoints, in name order."""

    found: list[str] = []
    for file in sorted(Path(path).glob("*.png")):
        try:
            cluster = "".join(chr(int(part, 16)) for part in file.stem.split("-"))
        except (ValueError, OverflowError):
            continue
        base = cluster.replace("️", "")
        if base and base not in found:
            found.append(base)
    return found


def missing_emoji(lines: list[str], provider: DirectoryProvider) -> list[str]:
    """Emoji clusters in ``lines`` that have no PNG in a directory provider, in order."""

    missing: list[str] = []
    for line in lines:
        for is_emoji, cluster in split_runs(line):
            if is_emoji and cluster not in missing and provider.get(cluster, 8) is None:
                missing.append(cluster)
    return missing


def get_provider(name: str, **kw: object) -> EmojiProvider:
    lowered = name.lower()
    if lowered == "twemoji":
        cache_value = kw.get("cache_dir")
        return TwemojiProvider(
            cache_dir=Path(str(cache_value)) if cache_value is not None else None,
            version=str(kw.get("version", "17.0.3")),
        )
    if lowered == "directory":
        path = kw.get("path") or kw.get("directory")
        if path is None:
            raise ValueError("directory emoji provider requires path=...")
        return DirectoryProvider(Path(str(path)))
    if lowered == "none":
        return NoEmojiProvider()
    raise ValueError("Unknown emoji provider; expected twemoji, directory, or none")
