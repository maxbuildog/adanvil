"""Project folder layout and safe path helpers."""

from __future__ import annotations

import json
import os
import unicodedata
from datetime import date
from pathlib import Path, PurePosixPath, PureWindowsPath

from adanvil.safeio import replace_text

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

# The server only opens projects inside this folder when it is set (the Claude Desktop
# extension and the Docker image set it).
PROJECTS_DIR_ENV = "ADANVIL_PROJECTS_DIR"
# Folders AdAnvil writes into: each must be a real folder of the project, never a link to
# somewhere else (``footage/`` may be a link: rushes often live on another drive).
MANAGED_DIRS = (
    RENDERS_DIR,
    CACHE_DIR,
    BRAND_DIR,
    MUSIC_DIR,
    REFERENCE_DIR,
    "storyboards",
    "exports",
    "voice",
    "sfx",
)
# What ``read_project_text`` may read: these top-level files and anything under these folders.
READABLE_FILES = (PROJECT_FILE, BRAND_FILE, CATALOG_FILE, "brand_brief.md")
READABLE_DIRS = (*MANAGED_DIRS, FOOTAGE_DIR)
MAX_READ_BYTES = 1_000_000


def _write_if_missing(path: Path, value: str) -> None:
    # ``lexists``: a dangling link counts as present, so its target is never created.
    if not os.path.lexists(path):
        replace_text(path, value)


def _normal(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.normpath(os.fspath(path)))


def _is_inside(path: str, folder: str) -> bool:
    """Whether ``path`` is ``folder`` or below it (both already ``_normal``)."""

    try:
        return os.path.commonpath((path, folder)) == folder
    except ValueError:  # another drive
        return False


def is_unc_path(text: str) -> bool:
    """A UNC or network path (``\\\\server\\share``, ``//server/share``); a local path written
    with the extended prefix (``\\\\?\\C:\\...``) is not."""

    flipped = text.replace("/", "\\")
    if flipped.startswith(("\\\\?\\", "\\\\.\\")):
        rest = flipped[4:]
        return not (len(rest) >= 2 and rest[0].isalpha() and rest[1] == ":")
    return flipped.startswith("\\\\")


def _is_filesystem_root(path: str) -> bool:
    rest = os.path.splitdrive(path)[1]
    return rest in {"", os.sep, os.altsep or os.sep} or os.path.dirname(path) == path


def _home() -> Path | None:
    try:
        return Path.home()
    except (RuntimeError, OSError):
        return None


def _refuse_unsafe_folder(path: str, written: str, label: str) -> None:
    """Raise for a drive or filesystem root, the user's home folder or one of its ancestors."""

    if _is_filesystem_root(path):
        raise ValueError(
            f"{label} must be a project folder, not a drive or filesystem root: {written}"
        )
    home = _home()
    if home is not None:
        for candidate in {_normal(home), _normal(os.path.realpath(home))}:
            if _is_inside(candidate, path):
                raise ValueError(
                    f"{label} must be a project folder, not your home folder or one of its "
                    f"parents: {written}. Use a sub-folder such as {home / 'AdAnvil' / 'my-ad'}"
                )


def configured_projects_dir() -> str | None:
    """The real path of ``ADANVIL_PROJECTS_DIR`` when it is set and not empty."""

    configured = os.environ.get(PROJECTS_DIR_ENV, "").strip()
    return os.path.realpath(os.path.expanduser(configured)) if configured else None


def check_project_dir(value: object, label: str = "project_dir") -> Path:
    """The real path of a project folder an agent asked for, or ``ValueError``.

    Every path is judged before anything touches the file system (a UNC path would already
    send the user's Windows credentials to its server): it must be absolute and not a network
    path, a drive or filesystem root, the user's home folder or one of its parents. When
    ``ADANVIL_PROJECTS_DIR`` is set the real path must also be strictly inside that folder.
    """

    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be an absolute path")
    if not Path(value).is_absolute():
        raise ValueError(f"{label} must be an absolute path: {value!r}")
    configured = os.environ.get(PROJECTS_DIR_ENV, "").strip()
    wanted = os.path.expanduser(configured) if configured else None
    if is_unc_path(value) and not (wanted and _is_inside(_normal(value), _normal(wanted))):
        raise ValueError(
            f"{label} must not be a network (UNC) path: {value!r}. Use a folder on a local drive"
            + (f" inside {PROJECTS_DIR_ENV} ({configured})" if configured else "")
        )
    lexical = _normal(value)
    _refuse_unsafe_folder(lexical, value, label)
    real = os.path.realpath(value)
    if is_unc_path(real) and not (wanted and _is_inside(_normal(real), _normal(wanted))):
        raise ValueError(f"{label} resolves to a network (UNC) path: {value!r}")
    _refuse_unsafe_folder(_normal(real), value, label)
    inside = configured_projects_dir()
    if inside is not None:
        folder = _normal(inside)
        if _normal(real) == folder or not _is_inside(_normal(real), folder):
            raise ValueError(
                f"{label} must be inside {PROJECTS_DIR_ENV} ({inside}): {value!r}. "
                "Use a sub-folder of it for the project"
            )
    return Path(real)


def check_managed_folders(root: Path) -> None:
    """Refuse a project whose managed folders (``renders``, ``cache``, ``brand``...) are links.

    AdAnvil writes renders, caches and brand files there: each one that exists must be a real
    folder of the project, not a symbolic link or junction, and its real path must stay inside
    the project. ``footage/`` is not checked: it may be a link to rushes on another drive.
    """

    real_root = _normal(os.path.realpath(root))
    junction = getattr(os.path, "isjunction", None)
    for name in MANAGED_DIRS:
        folder = Path(root) / name
        if not os.path.lexists(folder):
            continue
        linked = os.path.islink(folder) or bool(junction and junction(folder))
        if linked or not _is_inside(_normal(os.path.realpath(folder)), real_root):
            raise ValueError(
                f"{name}/ of the project is a link or points outside it ({root}): AdAnvil writes "
                f"into {name}/, so it must be a real folder of the project. Replace it with a "
                "real folder (only footage/ may be a link)"
            )


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
    if not os.path.lexists(brand_path):
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
    # A name that starts with a dot is a hidden file or folder (.ssh, .config, .env, .git):
    # never part of an ad project, and where credentials live.
    if any(part.startswith(".") or not part.strip(" .") for part in posix.parts):
        raise ValueError(
            f"project path must not contain a hidden name (starting with '.'): {rel!r}"
        )

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


def _readable_top(root: Path, rel: str) -> Path:
    """Where the project file ``rel`` must stay: the project, or the real folder it is under."""

    parts = PurePosixPath(unicodedata.normalize("NFC", rel).replace("\\", "/")).parts
    if len(parts) == 1 and parts[0] in READABLE_FILES:
        return Path(root)
    if len(parts) > 1 and parts[0] in READABLE_DIRS:
        return Path(root) / parts[0]
    allowed = ", ".join(READABLE_FILES)
    folders = ", ".join(f"{name}/" for name in READABLE_DIRS)
    raise ValueError(
        f"{rel!r} is not a file AdAnvil reads here: only {allowed} at the top of the project "
        f"and files under {folders}"
    )


def read_project_text(root: Path, rel: str, max_bytes: int = 65536) -> dict[str, object]:
    """Read one small, UTF-8 project text file of AdAnvil's own, without leaving the project.

    Only the top-level files ``adanvil.json``, ``brand.yaml``, ``catalog.json`` and
    ``brand_brief.md`` and the files under ``brand``, ``reference``, ``renders``, ``music``,
    ``storyboards``, ``exports``, ``cache``, ``voice``, ``sfx`` and ``footage`` can be read;
    ``max_bytes`` is capped at 1 MB.
    """

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    clamped = max_bytes > MAX_READ_BYTES
    max_bytes = min(max_bytes, MAX_READ_BYTES)
    path = resolve_in_project(root, rel)
    top = _readable_top(root, path.relative_to(Path(root)).as_posix())
    suffix = path.suffix.casefold()
    if suffix not in TEXT_EXTENSIONS:
        allowed = ", ".join(sorted(TEXT_EXTENSIONS))
        raise ValueError(f"text file extension must be one of: {allowed}")
    if not path.is_file():
        raise FileNotFoundError(path)
    # A file that is a link must still lead to a file of the same folder (a link planted in a
    # project received from someone else must not read what is elsewhere on the disk).
    if not _is_inside(_normal(os.path.realpath(path)), _normal(os.path.realpath(top))):
        raise ValueError(f"{rel!r} is a link to a file outside the project folder it is in")
    with open(path, "rb") as handle:
        data = handle.read(max_bytes + 1)
    truncated = len(data) > max_bytes
    content = data[:max_bytes].decode("utf-8", errors="replace")
    result: dict[str, object] = {
        "path": path.relative_to(Path(root)).as_posix(),
        "content": content,
        "truncated": truncated,
    }
    notes = []
    if clamped:
        notes.append(f"max_bytes was limited to {MAX_READ_BYTES} bytes.")
    if truncated:
        notes.append(f"Content truncated after {max_bytes} bytes.")
    if notes:
        result["note"] = " ".join(notes)
    return result


__all__ = [
    "BRAND_DIR",
    "BRAND_FILE",
    "CACHE_DIR",
    "CATALOG_FILE",
    "FOOTAGE_DIR",
    "MANAGED_DIRS",
    "MAX_READ_BYTES",
    "MUSIC_DIR",
    "PROJECTS_DIR_ENV",
    "PROJECT_DIRS",
    "PROJECT_FILE",
    "READABLE_DIRS",
    "READABLE_FILES",
    "REFERENCE_DIR",
    "RENDERS_DIR",
    "check_managed_folders",
    "check_project_dir",
    "configured_projects_dir",
    "init_project",
    "is_unc_path",
    "resolve_in_project",
    "read_project_text",
]
