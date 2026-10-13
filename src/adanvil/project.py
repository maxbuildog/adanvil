"""Project folder layout and safe path helpers."""

from __future__ import annotations

import json
import os
import unicodedata
from datetime import date
from pathlib import Path, PurePosixPath, PureWindowsPath

FOOTAGE_DIR = "footage"
BRAND_DIR = "brand"
MUSIC_DIR = "music"
REFERENCE_DIR = "reference"
RENDERS_DIR = "renders"
CACHE_DIR = "cache"

PROJECT_FILE = "adanvil.json"
BRAND_FILE = "brand.yaml"
CATALOG_FILE = "catalog.json"

PROJECT_DIRS = (
    FOOTAGE_DIR,
    BRAND_DIR,
    MUSIC_DIR,
    REFERENCE_DIR,
    RENDERS_DIR,
    CACHE_DIR,
)

TEXT_EXTENSIONS = {".txt", ".md", ".json", ".yaml", ".yml", ".csv"}


def _write_if_missing(path: Path, value: str) -> None:
    if not path.exists():
        path.write_text(value, encoding="utf-8")


def init_project(root: Path, name: str) -> Path:
    """Create a project skeleton without overwriting existing files."""

    project_root = Path(root)
    project_root.mkdir(parents=True, exist_ok=True)
    for directory in PROJECT_DIRS:
        (project_root / directory).mkdir(parents=True, exist_ok=True)

    project_path = project_root / PROJECT_FILE
    _write_if_missing(
        project_path,
        json.dumps(
            {"version": 1, "name": name, "created": date.today().isoformat()},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )

    brand_path = project_root / BRAND_FILE
    if not brand_path.exists():
        # The brand module is optional while the project and library steps land in parallel.
        try:
            from adanvil.brand import default_brand, save_brand
        except ModuleNotFoundError as exc:
            if exc.name != "adanvil.brand":
                raise
        else:
            save_brand(default_brand(name), brand_path)

    # Keeping an empty catalog in a new project makes the layout self-describing.
    _write_if_missing(
        project_root / CATALOG_FILE,
        json.dumps({"version": 1, "entries": {}}, indent=2) + "\n",
    )
    return project_root


def resolve_in_project(root: Path, rel: str) -> Path:
    """Return a safe project-relative path without resolving symlinks.

    Project paths are agent-authored names, so the boundary check is lexical:
    symlinks inside the project are intentionally allowed to point elsewhere.
    """

    if not isinstance(rel, str):
        raise TypeError("project path must be a string")
    normalised = unicodedata.normalize("NFC", rel)
    if not normalised or "\x00" in normalised:
        raise ValueError("project path must be a non-empty relative path")

    # Check both syntaxes so the safety property is stable if a catalog is moved
    # between Windows and a Unix worker.
    posix = PurePosixPath(normalised.replace("\\", "/"))
    windows = PureWindowsPath(normalised)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError(f"project path must be relative: {rel!r}")
    if any(part == ".." for part in posix.parts):
        raise ValueError(f"project path escapes the project: {rel!r}")

    # Use forward slashes as the portable spelling, then let the host OS
    # normalise repeated separators and ``.`` segments.  Do not call
    # Path.resolve(): that would follow a project-local symlink.
    normalised = normalised.replace("\\", "/")
    project_root = Path(root)
    root_name = os.path.normpath(os.fspath(project_root))
    candidate_name = os.path.normpath(os.path.join(root_name, normalised))
    # ``abspath`` is lexical (unlike ``realpath``/``Path.resolve``), and makes
    # the common-path comparison work when callers pass ``Path('.')``.
    check_root = os.path.abspath(root_name)
    check_candidate = os.path.abspath(candidate_name)
    try:
        common = os.path.commonpath((check_root, check_candidate))
    except ValueError as exc:
        raise ValueError(f"project path escapes the project: {rel!r}") from exc
    if os.path.normcase(common) != os.path.normcase(check_root):
        raise ValueError(f"project path escapes the project: {rel!r}")
    if not os.path.lexists(candidate_name):
        candidate_name = _match_unicode_names(root_name, candidate_name)
    return Path(candidate_name)


def _match_unicode_names(root_name: str, candidate_name: str) -> str:
    """Find the on-disk spelling of a path whose names differ only by Unicode form.

    macOS stores names decomposed (NFD) while agents and the catalog use NFC.  Each
    component that does not exist as written is matched against the directory entries
    by NFC form.  Unmatched components are kept as written.
    """

    relative = os.path.relpath(candidate_name, root_name)
    if relative == ".":
        return candidate_name
    current = root_name
    parts = relative.split(os.sep)
    for index, part in enumerate(parts):
        step = os.path.join(current, part)
        if os.path.lexists(step):
            current = step
            continue
        wanted = unicodedata.normalize("NFC", part)
        try:
            names = sorted(os.listdir(current))
        except OSError:
            return os.path.join(current, *parts[index:])
        match = next(
            (name for name in names if unicodedata.normalize("NFC", name) == wanted), None
        )
        if match is None:
            return os.path.join(current, *parts[index:])
        current = os.path.join(current, match)
    return current


def read_project_text(root: Path, rel: str, max_bytes: int = 65536) -> dict[str, object]:
    """Read one small, UTF-8 project text file without leaving the project."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    path = resolve_in_project(root, rel)
    suffix = path.suffix.casefold()
    if suffix not in TEXT_EXTENSIONS:
        allowed = ", ".join(sorted(TEXT_EXTENSIONS))
        raise ValueError(f"text file extension must be one of: {allowed}")
    if not path.is_file():
        raise FileNotFoundError(path)
    data = path.read_bytes()
    truncated = len(data) > max_bytes
    content = data[:max_bytes].decode("utf-8", errors="replace")
    result: dict[str, object] = {
        "path": path.relative_to(Path(root)).as_posix(),
        "content": content,
        "truncated": truncated,
    }
    if truncated:
        result["note"] = f"Content truncated after {max_bytes} bytes."
    return result


__all__ = [
    "BRAND_DIR",
    "BRAND_FILE",
    "CACHE_DIR",
    "CATALOG_FILE",
    "FOOTAGE_DIR",
    "MUSIC_DIR",
    "PROJECT_DIRS",
    "PROJECT_FILE",
    "REFERENCE_DIR",
    "RENDERS_DIR",
    "init_project",
    "resolve_in_project",
    "read_project_text",
]
