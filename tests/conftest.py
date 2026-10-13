from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.fixtures.make import build


@pytest.fixture(scope="session")
def media() -> dict[str, dict[str, object]]:
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "tests" / "fixtures" / "out"
    manifest = build(out_dir)
    for name, properties in manifest.items():
        properties["path"] = str((out_dir / name).resolve())
    return manifest


@pytest.fixture(scope="session", autouse=True)
def _isolated_job_pointers(tmp_path_factory: pytest.TempPathFactory):
    """Keep job pointer files out of the real user cache (detached runners inherit it)."""

    previous = os.environ.get("ADANVIL_JOBS_DIR")
    os.environ["ADANVIL_JOBS_DIR"] = str(tmp_path_factory.mktemp("job-pointers"))
    yield
    if previous is None:
        os.environ.pop("ADANVIL_JOBS_DIR", None)
    else:
        os.environ["ADANVIL_JOBS_DIR"] = previous


@pytest.fixture(scope="session", autouse=True)
def _vision_off_by_default():
    """The suite never starts a real vision worker, even where the environment is installed
    (detached runners inherit this); tests of the vision features inject a fake engine."""

    previous = os.environ.get("ADANVIL_VISION")
    os.environ["ADANVIL_VISION"] = "off"
    yield
    if previous is None:
        os.environ.pop("ADANVIL_VISION", None)
    else:
        os.environ["ADANVIL_VISION"] = previous


@pytest.fixture(scope="session", autouse=True)
def _speech_off_by_default():
    """``transcribe="auto"`` never starts a real transcriber in the suite (detached runners
    inherit this); tests of the speech features inject an engine or clear the variable."""

    previous = os.environ.get("ADANVIL_SPEECH")
    os.environ["ADANVIL_SPEECH"] = "off"
    yield
    if previous is None:
        os.environ.pop("ADANVIL_SPEECH", None)
    else:
        os.environ["ADANVIL_SPEECH"] = previous
