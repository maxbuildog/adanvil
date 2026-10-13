"""Entry point of the AdAnvil Claude Desktop extension.

Checks that ffmpeg and ffprobe are usable, then runs the AdAnvil MCP server over stdio.
Standard library only before the import of ``adanvil``.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path

_INSTALL = {
    "Darwin": "brew install ffmpeg",
    "Windows": "winget install Gyan.FFmpeg   (or: choco install ffmpeg)",
    "Linux": "sudo apt install ffmpeg   (Debian/Ubuntu 24.04+; Fedora: sudo dnf install ffmpeg)",
}


def _find(tool: str, env: str) -> str | None:
    configured = os.environ.get(env)
    if configured:
        return configured if Path(configured).is_file() else None
    return shutil.which(tool)


def check_ffmpeg() -> str | None:
    """Return an error message when ffmpeg or ffprobe cannot be found, else None."""

    missing = [
        tool
        for tool, env in (("ffmpeg", "ADANVIL_FFMPEG"), ("ffprobe", "ADANVIL_FFPROBE"))
        if _find(tool, env) is None
    ]
    if not missing:
        return None
    command = _INSTALL.get(platform.system(), "install ffmpeg 6.1 or newer from ffmpeg.org")
    return (
        f"AdAnvil needs ffmpeg 6.1 or newer, but {' and '.join(missing)} was not found.\n"
        f"Install it with:  {command}\n"
        "Then restart Claude Desktop. If ffmpeg is installed in an unusual place, set the "
        "environment variables ADANVIL_FFMPEG and ADANVIL_FFPROBE to the full paths."
    )


def main() -> int:
    problem = check_ffmpeg()
    if problem:
        print(problem, file=sys.stderr)
        return 1
    folder = os.environ.get("ADANVIL_PROJECTS_DIR", "").strip()
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        os.chdir(folder)
    from adanvil.server import main as serve

    serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
