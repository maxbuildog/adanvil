from __future__ import annotations

import os
from pathlib import Path

import pytest

from adanvil.safeio import replace_bytes, replace_text


def _link(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available here")


def test_replace_text_writes_and_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b.json"
    replace_text(path, "one", newline="")
    replace_text(path, "two", newline="")
    assert path.read_text(encoding="utf-8") == "two"
    assert [p.name for p in path.parent.iterdir()] == ["b.json"]  # no temporary left


def test_replace_bytes_writes(tmp_path: Path) -> None:
    path = tmp_path / "x.bin"
    replace_bytes(path, b"\x00\x01")
    assert path.read_bytes() == b"\x00\x01"


def test_a_planted_link_is_replaced_not_followed(tmp_path: Path) -> None:
    victim = tmp_path / "outside.txt"
    victim.write_text("precious", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    link = project / "site_facts.json"
    _link(link, victim)
    replace_text(link, "{}", newline="")
    assert victim.read_text(encoding="utf-8") == "precious"
    assert not link.is_symlink() and link.read_text(encoding="utf-8") == "{}"
