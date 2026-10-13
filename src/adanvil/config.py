"""Concurrency settings shared by every parallel stage (indexing, mosaics, renders).

``ADANVIL_WORKERS`` sets how many files, thumbnails or segments are processed at
once; ``ADANVIL_FFMPEG_THREADS`` sets the threads of each ffmpeg decoder/encoder.
Unset, the worker count is ``max(1, min(4, cpu_count // 2))``, capped at 2 when the
machine has less than 12 GB of RAM (or when its RAM cannot be read), and the
threads per ffmpeg are ``max(1, cpu_count // workers)``.
"""

from __future__ import annotations

import os

LOW_RAM_BYTES = 12 * 1024**3
LOW_RAM_WORKERS = 2


def total_ram_bytes() -> int | None:
    """Physical memory of the machine, or ``None`` when it cannot be read."""

    if os.name == "nt":
        try:
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [  # noqa: RUF012
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return int(status.ullTotalPhys)
        except Exception:
            return None
        return None
    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        return None


def _positive_int(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        number = int(value)
    except ValueError:
        return None
    return number if number >= 1 else None


def workers() -> int:
    """Parallel files/segments/thumbnails: ``ADANVIL_WORKERS`` or a RAM-safe default."""

    override = _positive_int("ADANVIL_WORKERS")
    if override is not None:
        return override
    count = max(1, min(4, (os.cpu_count() or 2) // 2))
    ram = total_ram_bytes()
    if ram is None or ram < LOW_RAM_BYTES:
        count = min(count, LOW_RAM_WORKERS)
    return count


def ffmpeg_threads(parallel: int | None = None) -> int:
    """Threads per ffmpeg: ``ADANVIL_FFMPEG_THREADS`` or the cores shared by the workers."""

    override = _positive_int("ADANVIL_FFMPEG_THREADS")
    if override is not None:
        return override
    return max(1, (os.cpu_count() or 1) // max(1, parallel or workers()))


def thread_args() -> list[str]:
    """``-threads N`` for the next ffmpeg input (decoder) or output (encoder)."""

    return ["-threads", str(ffmpeg_threads())]


__all__ = ["ffmpeg_threads", "thread_args", "total_ram_bytes", "workers"]
