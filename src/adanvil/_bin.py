from __future__ import annotations

import os
from pathlib import Path


def _same_folder(first: str, second: str) -> bool:
    try:
        return os.path.normcase(os.path.realpath(first)) == os.path.normcase(
            os.path.realpath(second)
        )
    except OSError:
        return False


def _candidates(folder: str, name: str) -> list[str]:
    if os.name != "nt":
        return [os.path.join(folder, name)]
    extensions = [
        extension
        for extension in os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(os.pathsep)
        if extension
    ]
    return [os.path.join(folder, name + extension) for extension in extensions]


def _which(name: str) -> str | None:
    """Find ``name`` on ``PATH`` without ever looking in the current folder.

    ``shutil.which`` searches the current folder first on Windows, so an ``ffmpeg.exe`` dropped
    in the folder where the server starts (a downloaded project, say) would be the one run.
    The entries of ``PATH`` are walked in order; an empty entry, a relative entry and the
    current folder are skipped, and the first absolute path that is an executable regular file
    is returned.
    """

    current = os.getcwd()
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        entry = entry.strip().strip('"')
        if not entry or not Path(entry).is_absolute() or _same_folder(entry, current):
            continue
        for candidate in _candidates(entry, name):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return None


def _configured(variable: str, tool: str) -> str | None:
    configured = os.environ.get(variable)
    if not configured:
        return None
    if Path(configured).is_absolute() and Path(configured).is_file():
        return configured
    raise FileNotFoundError(
        f"{variable} must be the absolute path of an existing file, got {configured!r}. "
        f"Install {tool} or set {variable} to its executable path."
    )


def ffmpeg_path() -> str:
    configured = _configured("ADANVIL_FFMPEG", "ffmpeg")
    if configured:
        return configured
    resolved = _which("ffmpeg")
    if resolved:
        return resolved
    raise FileNotFoundError(
        "ffmpeg was not found on PATH. Install ffmpeg >= 6.1 with libzimg, "
        "or set ADANVIL_FFMPEG to its executable path."
    )


def ffprobe_path() -> str:
    configured = _configured("ADANVIL_FFPROBE", "ffprobe")
    if configured:
        return configured
    resolved = _which("ffprobe")
    if resolved:
        return resolved
    raise FileNotFoundError(
        "ffprobe was not found on PATH. Install ffmpeg >= 6.1, "
        "or set ADANVIL_FFPROBE to its executable path."
    )
