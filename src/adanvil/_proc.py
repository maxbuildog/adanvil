"""Subprocess flags shared by every launcher.

On Windows a console program started from a process without a console opens its
own window: every ffmpeg call would flash a terminal. CREATE_NO_WINDOW gives the
child a hidden console that its own children inherit, so nothing pops up.
"""

from __future__ import annotations

import os
import subprocess

NO_WINDOW: int = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def detached_flags() -> int:
    """Flags for a background runner that must outlive its parent, without a window."""
    if os.name != "nt":
        return 0
    return NO_WINDOW | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
