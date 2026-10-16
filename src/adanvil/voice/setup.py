"""Install the optional voice stack in an isolated environment."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from adanvil.safeio import replace_text


def cache_dir() -> Path:
    return Path(os.environ.get("ADANVIL_CACHE", Path.home() / ".cache" / "adanvil"))


def voice_config_path() -> Path:
    return cache_dir() / "voice_config.json"


def _interpreter(env_dir: Path) -> Path:
    return env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


_TORCH = ["torch==2.6.0", "torchaudio==2.6.0"]  # the versions chatterbox-tts pins
_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
# The versions that load models are pinned (they were run together on 15/10/2026); the audio
# libraries only get a lower bound, because their wheels differ by platform.
CHATTERBOX_PACKAGE = "chatterbox-tts==0.1.7"
WHISPER_PACKAGE = "faster-whisper==1.2.1"
KOKORO_PACKAGE = "kokoro==0.9.4"
_SPACY_ENGLISH = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
    # SHA-256 of the release wheel (12 806 118 bytes), checked by pip and uv at install.
    "#sha256=1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
)


ENGINES = ("chatterbox", "kokoro")


def _commands(
    env_dir: Path, python: str | None, engine: str = "chatterbox"
) -> tuple[list[str], list[list[str]], Path]:
    uv = shutil.which("uv")
    if uv:
        create = [uv, "venv", str(env_dir)]
        if python:
            create.extend(["--python", python])
        interpreter = _interpreter(env_dir)
        pip = [uv, "pip", "install", "--python", str(interpreter)]
    else:
        base = python or sys.executable
        create = [base, "-m", "venv", str(env_dir)]
        interpreter = _interpreter(env_dir)
        pip = [str(interpreter), "-m", "pip", "install"]
    installs = []
    if sys.platform.startswith("linux"):
        # The default Linux torch wheel pulls ~3 GB of CUDA packages; the voice runs on CPU.
        installs.append([*pip, "--index-url", _CPU_INDEX, *_TORCH])
    if engine == "kokoro":
        # Kokoro (Apache-2.0) speaks with fixed voices; faster-whisper still aligns and checks
        # (librosa decodes the audio for it, as in the chatterbox stack).
        # Two steps: asked together, the resolver once picked a tokenizers that has no wheel.
        installs.append([*pip, WHISPER_PACKAGE])
        installs.append([*pip, KOKORO_PACKAGE, "soundfile", "librosa"])
        # English phonemes need this spaCy model; kokoro would otherwise try to download it at
        # run time with a pip that a uv environment does not have.
        installs.append([*pip, _SPACY_ENGLISH])
    else:
        installs.append([*pip, CHATTERBOX_PACKAGE, WHISPER_PACKAGE])
    return create, installs, interpreter


def voice_setup(
    target: Path | None = None,
    *,
    python: str | None = None,
    plan: bool = False,
    engine: str = "chatterbox",
) -> Path | list[list[str]]:
    """Create the voice venv (or reuse it) and remember its interpreter.

    ``engine`` picks what is installed: ``chatterbox`` (cloned voice, the default) or
    ``kokoro`` (fast, fixed voices). Run it twice to have both in the same environment.
    ``plan=True`` is side-effect free and returns the command lists that would be
    executed, which keeps setup unit-testable without network access.
    """
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {', '.join(ENGINES)}")
    env_dir = Path(target) if target is not None else cache_dir() / "voice-env"
    create, installs, interpreter = _commands(env_dir, python, engine)
    if plan:
        return [create, *installs]

    env_dir.parent.mkdir(parents=True, exist_ok=True)
    if interpreter.is_file():
        print(f"Reusing the voice environment in {env_dir}")
    else:
        print(f"Creating isolated voice environment in {env_dir}")
        subprocess.run(create, check=True)
    for install in installs:
        print("Running: " + " ".join(install))
        subprocess.run(install, check=True)
    config = voice_config_path()
    config.parent.mkdir(parents=True, exist_ok=True)
    replace_text(config, json.dumps({"python": str(interpreter)}, indent=2) + "\n")
    print(f"Saved voice interpreter path to {config}")
    return interpreter


__all__ = ["ENGINES", "cache_dir", "voice_config_path", "voice_setup"]
