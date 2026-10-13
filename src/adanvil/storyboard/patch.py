"""Small edits of a saved storyboard: a list of ``{path, value}`` changes.

A path walks the JSON with dots and indexes: ``segments.3.out``, ``overlays.0.lines.1`` or
``audio.music`` (``segments[3].out`` works too). Setting a missing key of an object adds it;
setting the index just past the end of a list, or ``-``, appends. ``{"path": ..., "remove":
true}`` deletes a key or a list item. All the changes are applied to a copy, in order, and a
bad one (unknown key on the way, index out of range) is an error that names it and changes
nothing.
"""

from __future__ import annotations

import copy
import re
from typing import Any

_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


def storyboard_name(name: str) -> str:
    """A safe file stem for ``storyboards/<name>.json`` (a trailing ``.json`` is accepted)."""

    stem = name.removeprefix("storyboards/").removesuffix(".json")
    if not _NAME.match(stem) or stem.startswith(".") or stem in {"", ".", ".."}:
        raise ValueError(
            "storyboard name must be 1 to 80 letters, digits, '_', '.' or '-' "
            f"(a file name, not a path), got {name!r}"
        )
    return stem


_NAME_LIMIT = 80


def snapped_name(name: object) -> str:
    """``name`` with the ``_snapped`` suffix (cut short so the result is still a valid name)."""

    base = name if isinstance(name, str) and name else "storyboard"
    suffix = "_snapped"
    return base[: _NAME_LIMIT - len(suffix)] + suffix


def rename_storyboard(document: dict[str, Any], new_name: str) -> dict[str, Any]:
    """``document`` renamed to ``new_name``, its render going to ``renders/<new_name>.mp4``.

    The quality of ``output`` is kept. ``document`` is changed in place and returned.
    """

    document["name"] = new_name
    output = document.get("output")
    output = dict(output) if isinstance(output, dict) else {}
    output["path"] = f"renders/{new_name}.mp4"
    document["output"] = output
    return document


def follow_rename(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """After a patch that changed ``name``, move a default ``output.path`` to the new name.

    The path is only touched when it was the default of the old name (``renders/<old>.mp4`` or
    the template ``renders/{name}.mp4``); a path chosen by hand stays. Returns whether it moved.
    """

    old, new = before.get("name"), after.get("name")
    output = after.get("output")
    if not isinstance(old, str) or not isinstance(new, str) or old == new:
        return False
    if not isinstance(output, dict) or "path" not in output:
        return False
    if output["path"] not in {f"renders/{old}.mp4", "renders/{name}.mp4"}:
        return False
    output["path"] = f"renders/{new}.mp4"
    return True


def _steps(path: str) -> list[str]:
    cleaned = re.sub(r"\[(-?\w*)\]", r".\1", path).strip(".")
    steps = [step for step in cleaned.split(".") if step != ""]
    if not steps:
        raise ValueError(f"empty path {path!r}")
    return steps


def _index(step: str, size: int, path: str, *, append_ok: bool) -> int:
    if step == "-" and append_ok:
        return size
    try:
        position = int(step)
    except ValueError as exc:
        raise ValueError(f"{path}: {step!r} is not a list index") from exc
    limit = size if append_ok else size - 1
    if not 0 <= position <= limit:
        raise ValueError(f"{path}: index {position} is out of range (the list has {size} items)")
    return position


def apply_changes(document: dict[str, Any], changes: list[dict[str, Any]]) -> dict[str, Any]:
    """A copy of ``document`` with ``changes`` applied (the original is left alone)."""

    if not isinstance(changes, list) or not changes:
        raise ValueError("changes must be a non-empty list of {path, value}")
    result = copy.deepcopy(document)
    for number, change in enumerate(changes):
        if not isinstance(change, dict) or "path" not in change:
            raise ValueError(f"change {number} must be an object with a path")
        unknown = sorted(set(change) - {"path", "value", "remove"})
        if unknown:
            raise ValueError(f"change {number} has unexpected field(s): {', '.join(unknown)}")
        path = str(change["path"])
        remove = bool(change.get("remove"))
        if not remove and "value" not in change:
            raise ValueError(f"{path}: a change needs a value (or remove: true)")
        steps = _steps(path)
        node: Any = result
        for step in steps[:-1]:
            if isinstance(node, dict):
                if step not in node:
                    raise ValueError(f"{path}: no key {step!r}")
                node = node[step]
            elif isinstance(node, list):
                node = node[_index(step, len(node), path, append_ok=False)]
            else:
                raise ValueError(f"{path}: cannot go into {step!r}, the value is not a container")
        last = steps[-1]
        if isinstance(node, dict):
            if remove:
                if last not in node:
                    raise ValueError(f"{path}: no key {last!r} to remove")
                del node[last]
            else:
                node[last] = copy.deepcopy(change["value"])
        elif isinstance(node, list):
            position = _index(last, len(node), path, append_ok=not remove)
            if remove:
                del node[position]
            elif position == len(node):
                node.append(copy.deepcopy(change["value"]))
            else:
                node[position] = copy.deepcopy(change["value"])
        else:
            raise ValueError(f"{path}: cannot set {last!r}, the value is not a container")
    return result


__all__ = [
    "apply_changes",
    "follow_rename",
    "rename_storyboard",
    "snapped_name",
    "storyboard_name",
]
