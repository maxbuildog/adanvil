from __future__ import annotations

import hashlib
import io
import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from adanvil._bin import ffprobe_path
from adanvil.brand import load_brand
from examples.demo import fetch_demo

PACK_PATH = Path(__file__).parents[1] / "examples" / "demo" / "pack.json"


def test_pack_manifest_is_five_verified_commons_files() -> None:
    manifest = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    entries = manifest["files"]

    assert len(entries) == 5
    assert all(entry["url"].startswith("https://upload.wikimedia.org/") for entry in entries)
    assert all(
        len(entry["sha1"]) == 40
        and all(c in "0123456789abcdef" for c in entry["sha1"])
        for entry in entries
    )
    assert {entry["license"] for entry in entries} <= {"CC0", "Public domain"}


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def test_fetch_builds_project_without_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "out" / "sdr_portrait.mp4"
    payload = fixture.read_bytes()
    digest = hashlib.sha1(payload).hexdigest()
    monkeypatch.setattr(
        fetch_demo,
        "EXPECTED_SHA1",
        {entry["target"]: digest for entry in fetch_demo.PACK},
    )
    monkeypatch.setattr(
        fetch_demo.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(payload),
    )

    assert fetch_demo.main([str(tmp_path)]) == 0
    assert (tmp_path / "adanvil.json").is_file()
    assert (tmp_path / "brand.yaml").is_file()
    assert (tmp_path / "footage" / "hands_toast.webm").is_file()
    assert (tmp_path / "music" / "demo_loop.m4a").is_file()
    assert not (tmp_path / "footage" / "hands_toast.webm.part").exists()

    brand = load_brand(tmp_path / "brand.yaml")
    assert brand.claims is not None
    assert "detox" in brand.claims.banned_words
    with Image.open(tmp_path / "brand" / "end_card.png") as end_card:
        assert end_card.size == (1080, 1920)

    result = subprocess.run(
        [
            ffprobe_path(),
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(tmp_path / "music" / "demo_loop.m4a"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert abs(float(result.stdout.strip()) - 24.0) <= 0.05


def test_sha1_mismatch_discards_partial_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"fixture bytes"
    entry = fetch_demo.PACK[0]
    target = tmp_path / entry["target"]
    monkeypatch.setattr(
        fetch_demo.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(payload),
    )
    monkeypatch.setattr(fetch_demo, "EXPECTED_SHA1", {entry["target"]: "0" * 40})

    with pytest.raises(fetch_demo.DemoPackError, match="SHA-1 mismatch"):
        fetch_demo.download_file(entry, target)

    assert not target.exists()
    assert not target.with_name(target.name + ".part").exists()
