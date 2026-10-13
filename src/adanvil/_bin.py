import os
import shutil
from pathlib import Path


def ffmpeg_path() -> str:
    configured = os.environ.get("ADANVIL_FFMPEG")
    if configured:
        if Path(configured).is_file():
            return configured
        raise FileNotFoundError(
            f"ADANVIL_FFMPEG points to a missing file: {configured!r}. "
            "Install ffmpeg or set ADANVIL_FFMPEG to its executable path."
        )
    resolved = shutil.which("ffmpeg")
    if resolved:
        return resolved
    raise FileNotFoundError(
        "ffmpeg was not found on PATH. Install ffmpeg >= 6.1 with libzimg, "
        "or set ADANVIL_FFMPEG to its executable path."
    )


def ffprobe_path() -> str:
    configured = os.environ.get("ADANVIL_FFPROBE")
    if configured:
        if Path(configured).is_file():
            return configured
        raise FileNotFoundError(
            f"ADANVIL_FFPROBE points to a missing file: {configured!r}. "
            "Install ffprobe or set ADANVIL_FFPROBE to its executable path."
        )
    resolved = shutil.which("ffprobe")
    if resolved:
        return resolved
    raise FileNotFoundError(
        "ffprobe was not found on PATH. Install ffmpeg >= 6.1, "
        "or set ADANVIL_FFPROBE to its executable path."
    )
