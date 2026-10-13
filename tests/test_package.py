from pathlib import Path

import pytest

import adanvil
from adanvil._bin import ffmpeg_path


def test_version_is_string() -> None:
    assert isinstance(adanvil.__version__, str)


def test_ffmpeg_path_is_existing_file() -> None:
    assert Path(ffmpeg_path()).is_file()


def test_missing_ffmpeg_override_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    missing = Path(__file__).resolve().parent / "does-not-exist" / "ffmpeg"
    monkeypatch.setenv("ADANVIL_FFMPEG", str(missing))
    with pytest.raises(FileNotFoundError, match="ADANVIL_FFMPEG"):
        ffmpeg_path()
