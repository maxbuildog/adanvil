"""Install the optional vision stack in an isolated environment."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def cache_dir() -> Path:
    return Path(os.environ.get("ADANVIL_CACHE", Path.home() / ".cache" / "adanvil"))


def vision_config_path() -> Path:
    return cache_dir() / "vision_config.json"


def models_dir() -> Path:
    return Path(os.environ.get("ADANVIL_VISION_DIR", cache_dir() / "vision-models"))


def worker_path() -> Path:
    return Path(__file__).with_name("worker.py")


def _interpreter(env_dir: Path) -> Path:
    return env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
# transformers loads SigLIP 2; OpenCV runs the YuNet face detector; Pillow and numpy read
# the pictures. Nothing here is a dependency of the base package.
_PACKAGES = ["transformers", "numpy", "pillow", "opencv-python-headless", "huggingface_hub"]


def _commands(
    env_dir: Path, python: str | None
) -> tuple[list[str], list[list[str]], list[str], Path]:
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
    installs: list[list[str]] = []
    if sys.platform.startswith("linux"):
        # The default Linux torch wheel pulls ~3 GB of CUDA packages; the vision runs on CPU.
        installs.append([*pip, "--index-url", _CPU_INDEX, "torch"])
    else:
        installs.append([*pip, "torch"])
    installs.append([*pip, *_PACKAGES])
    prefetch = [str(interpreter), str(worker_path()), "--prefetch"]
    return create, installs, prefetch, interpreter


def vision_setup(
    target: Path | None = None,
    *,
    python: str | None = None,
    plan: bool = False,
) -> Path | list[list[str]]:
    """Create the vision venv (or reuse it), download the models and remember its interpreter.

    The models are SigLIP 2 (image and text embeddings, Apache-2.0) and the YuNet face
    detector (MIT); see ``THIRD_PARTY.md``. ``plan=True`` is side-effect free and returns the
    command lists that would be executed, which keeps setup unit-testable without network.
    """

    env_dir = Path(target) if target is not None else cache_dir() / "vision-env"
    create, installs, prefetch, interpreter = _commands(env_dir, python)
    if plan:
        return [create, *installs, prefetch]

    env_dir.parent.mkdir(parents=True, exist_ok=True)
    if interpreter.is_file():
        print(f"Reusing the vision environment in {env_dir}")
    else:
        print(f"Creating isolated vision environment in {env_dir}")
        subprocess.run(create, check=True)
    for install in installs:
        print("Running: " + " ".join(install))
        subprocess.run(install, check=True)
    folder = models_dir()
    print(f"Downloading the models into {folder}")
    subprocess.run(prefetch, check=True, env={**os.environ, "ADANVIL_VISION_DIR": str(folder)})
    config = vision_config_path()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        json.dumps({"python": str(interpreter), "models": str(folder)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved vision interpreter path to {config}")
    return interpreter


__all__ = [
    "cache_dir",
    "models_dir",
    "vision_config_path",
    "vision_setup",
    "worker_path",
]
