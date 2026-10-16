"""Optional local music generation with ACE-Step v1 (code Apache-2.0, weights Apache-2.0).

An isolated environment (``adanvil music-setup``) holds torch and ACE-Step; AdAnvil talks to it
through ``musicgen_worker.py``, as it does for the voice. The licence of the model was checked
(see THIRD_PARTY.md section 1.8): code and weights allow commercial use of the output, so the
track gets a licence record like any other and needs no attribution. CPU speed measured on an
8-core server: see THIRD_PARTY.md; a 30 s track takes minutes, so it runs as a job.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from adanvil.music import MUSIC_DIR, MusicError, _slug, write_licence_record
from adanvil.safeio import replace_bytes, replace_text
from adanvil.voice.setup import cache_dir

MODEL_ID = "ACE-Step/ACE-Step-v1-3.5B"
MODEL_PAGE = "https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B"
ACE_STEP_REF = "1bee4c9f5b43e30995f8d4d33b3919197ce1bd68"  # main, 2026-02-15
LICENCE_NAME = "Apache-2.0 (ACE-Step v1: code and weights)"
LICENCE_URL = "https://www.apache.org/licenses/LICENSE-2.0"
MIN_DURATION = 10.0
MAX_DURATION = 120.0
DEFAULT_STEPS = 27
_TORCH = ["torch==2.6.0", "torchaudio==2.6.0", "torchvision==0.21.0"]
_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
# The requirements of ACE-Step minus the GUI, training and notebook extras.
_REQUIREMENTS = [
    "diffusers>=0.33.0", "librosa==0.11.0", "loguru==0.7.3", "numpy", "pypinyin==0.53.0",
    "pytorch_lightning==2.5.1", "soundfile==0.13.1", "tqdm", "transformers==4.50.0",
    "py3langid==0.3.0", "hangul-romanize==0.1.0", "num2words==0.5.14", "spacy==3.8.4",
    "accelerate==1.6.0", "cutlet", "fugashi[unidic-lite]", "click", "peft", "datasets==3.4.1",
    "matplotlib", "huggingface_hub",
]  # fmt: skip


class MusicUnavailableError(RuntimeError):
    """The optional music environment is missing or failed."""


def music_config_path() -> Path:
    return cache_dir() / "music_config.json"


def _interpreter(env_dir: Path) -> Path:
    return env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def music_setup(
    target: Path | None = None, *, python: str | None = None, plan: bool = False
) -> Path | list[list[str]]:
    """Create the music venv (or reuse it) and remember its interpreter.

    ``plan=True`` returns the command lists without running them. The model weights (about
    8 GB) are downloaded from Hugging Face by the first generation.
    """

    env_dir = Path(target) if target is not None else cache_dir() / "music-env"
    interpreter = _interpreter(env_dir)
    uv = shutil.which("uv")
    if uv:
        create = [uv, "venv", str(env_dir)]
        if python:
            create.extend(["--python", python])
        pip = [uv, "pip", "install", "--python", str(interpreter)]
    else:
        create = [python or sys.executable, "-m", "venv", str(env_dir)]
        pip = [str(interpreter), "-m", "pip", "install"]
    installs = []
    if sys.platform.startswith("linux"):
        installs.append([*pip, "--index-url", _CPU_INDEX, *_TORCH])  # no CUDA packages
    else:
        installs.append([*pip, *_TORCH])
    installs.append([*pip, *_REQUIREMENTS])
    installs.append(
        [*pip, "--no-deps", f"git+https://github.com/ace-step/ACE-Step.git@{ACE_STEP_REF}"]
    )
    if plan:
        return [create, *installs]
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    if not interpreter.is_file():
        print(f"Creating the isolated music environment in {env_dir}")
        subprocess.run(create, check=True)
    for install in installs:
        print("Running: " + " ".join(install))
        subprocess.run(install, check=True)
    config = music_config_path()
    config.parent.mkdir(parents=True, exist_ok=True)
    replace_text(config, json.dumps({"python": str(interpreter)}, indent=2) + "\n")
    print(f"Saved the music interpreter path to {config}")
    return interpreter


def configured_interpreter() -> Path | None:
    try:
        data = json.loads(music_config_path().read_text(encoding="utf-8"))
        path = Path(str(data["python"]))
    except (OSError, ValueError, KeyError):
        return None
    return path if path.is_file() else None


WORKER = Path(__file__).with_name("musicgen_worker.py")
_PROGRESS = re.compile(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)")
_RESULT = "ADANVIL_RESULT "


def available() -> bool:
    return configured_interpreter() is not None


# Measured peak of a 30 s track on CPU: 18 GB. A generation that runs out of memory can take
# down every other process of the machine, so it is refused when too little memory is free.
MIN_FREE_GB = 20.0


def _check_memory() -> None:
    """Refuse a generation when the free memory is under ``ADANVIL_MUSIC_MIN_FREE_GB``
    (default 20). Only checked where ``/proc/meminfo`` exists (Linux)."""

    try:
        need = float(os.environ.get("ADANVIL_MUSIC_MIN_FREE_GB", MIN_FREE_GB))
    except ValueError:
        need = MIN_FREE_GB
    try:
        text = Path("/proc/meminfo").read_text(encoding="ascii")
    except OSError:
        return
    match = re.search(r"^MemAvailable:\s+(\d+) kB", text, re.MULTILINE)
    if match is None:
        return
    free_gb = int(match.group(1)) / 1024 / 1024
    if free_gb < need:
        raise MusicUnavailableError(
            f"music generation needs about {need:g} GB of free memory (peak measured: 18 GB "
            f"for 30 s of audio) and only {free_gb:.1f} GB is free; set "
            "ADANVIL_MUSIC_MIN_FREE_GB to change the limit, or use music_search instead"
        )


def _own_file(value: object, folder: Path) -> Path | None:
    """The file ``value`` names, when it is a real file inside ``folder`` (links resolved), else
    None. The path comes from a line of the worker's output, which merges its stderr and can
    echo text of the prompt: it is never trusted to point anywhere else."""

    if not isinstance(value, str) or not value:
        return None
    try:
        path = Path(value).resolve(strict=True)
        inside = path.is_relative_to(folder.resolve(strict=True))
    except (OSError, ValueError, RuntimeError):
        return None
    return path if inside and path.is_file() else None


def _run_worker(
    interpreter: Path,
    work: Path,
    name: str,
    prompt: str,
    duration: float,
    steps: int,
    seed: int,
    progress: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    """Run the worker on one request inside ``work`` and return its result line (a dict with
    ``ok``); a worker that failed raises ``MusicUnavailableError``."""

    request = work / "request.json"
    replace_text(
        request,
        json.dumps(
            {
                "prompt": prompt,
                "duration": duration,
                "steps": steps,
                "seed": seed,
                "out": str(work / (Path(name).stem + ".wav")),
                "checkpoint_dir": str(cache_dir() / "music-models"),
            }
        ),
    )
    timeout = float(os.environ.get("ADANVIL_MUSIC_TIMEOUT", "3600"))
    if progress is not None:
        progress({"stage": "generating", "done": 0, "total": steps, "unit": "steps",
                  "message": "loading the model and generating"})  # fmt: skip
    process = subprocess.Popen(
        [str(interpreter), str(WORKER), str(request)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    tail = ""
    result: dict[str, Any] | None = None
    try:
        assert process.stdout is not None
        timer = threading.Timer(timeout, process.kill)
        timer.start()
        try:
            while True:
                chunk = process.stdout.read1(4096) if hasattr(process.stdout, "read1") else b""
                if not chunk:
                    break
                tail = (tail + chunk.decode("utf-8", "replace"))[-6000:]
                last = None
                for found in _PROGRESS.finditer(tail[-400:]):
                    last = found
                if last is not None and progress is not None:
                    progress({"stage": "generating", "done": int(last.group(2)),
                              "total": int(last.group(3)), "unit": "steps",
                              "message": f"step {last.group(2)}/{last.group(3)}"})  # fmt: skip
        finally:
            timer.cancel()
        process.wait()
    except BaseException:
        process.kill()
        raise
    for line in tail.splitlines():
        if line.startswith(_RESULT):
            try:
                found_result = json.loads(line[len(_RESULT) :])
            except ValueError:
                continue  # a line that only looks like the result (echoed text)
            if isinstance(found_result, dict):
                result = found_result
    if result is None or not result.get("ok"):
        reason = (result or {}).get("error") or f"the worker stopped (exit {process.returncode})"
        raise MusicUnavailableError(f"music generation failed: {reason}\n{tail[-600:]}")
    return result


def generate_music(
    root: Path,
    prompt: str,
    duration: float = 30.0,
    *,
    steps: int = DEFAULT_STEPS,
    seed: int | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Generate an instrumental track of ``duration`` seconds into ``music/`` with its licence
    record, then profile it. Raises ``MusicUnavailableError`` without the music environment."""

    root = Path(root)
    prompt = (prompt or "").strip()
    if not prompt:
        raise MusicError("prompt must not be empty")
    if not MIN_DURATION <= float(duration) <= MAX_DURATION:
        raise MusicError(f"duration must be between {MIN_DURATION:g} and {MAX_DURATION:g} seconds")
    if not 8 <= int(steps) <= 60:
        raise MusicError("steps must be between 8 and 60")
    interpreter = configured_interpreter()
    if interpreter is None:
        raise MusicUnavailableError(
            "the music environment is not installed: run 'adanvil music-setup' "
            "(about 3 GB of packages, then about 8 GB of model weights at the first generation)"
        )
    _check_memory()
    seed = secrets.randbelow(2**31) if seed is None else int(seed)
    folder = root / MUSIC_DIR
    folder.mkdir(parents=True, exist_ok=True)
    name = f"gen-{_slug(prompt)[:28].strip('-')}-{seed}.flac"
    target = folder / name
    scratch = root / "cache" / "music-gen"
    scratch.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="gen-", dir=scratch))  # this request's own folder
    try:
        result = _run_worker(
            interpreter, work, name, prompt, float(duration), int(steps), seed, progress
        )
        produced = _own_file(result.get("out"), work)
        if produced is None:
            raise MusicUnavailableError(
                "music generation failed: the worker did not report a file of its own output "
                "folder, so nothing was read or deleted"
            )
        # The model writes 32-bit float WAV (11 MB for 30 s): keep a 16-bit FLAC (lossless
        # enough). ffmpeg writes inside the request folder; the track is then moved over the
        # target by a rename, which never follows a link planted at its name.
        from adanvil.ffmpeg import run as ffmpeg_run

        encoded = work / name
        ffmpeg_run(
            ["-y", "-i", str(produced), "-sample_fmt", "s16", "-c:a", "flac", str(encoded)]
        )
        replace_bytes(target, encoded.read_bytes())
    finally:
        shutil.rmtree(work, ignore_errors=True)
    relative = f"{MUSIC_DIR}/{name}"
    write_licence_record(
        root,
        relative,
        {
            "source": "generated",
            "id": f"acestep-v1-seed{seed}",
            "url": MODEL_PAGE,
            "title": prompt[:80],
            "author": "ACE-Step v1 (generated on this machine by AdAnvil)",
            "licence": LICENCE_NAME,
            "licence_code": "apache-2.0",
            "licence_url": LICENCE_URL,
            "attribution_required": False,
            "attribution_text": "",
            "fetch_date": date.today().isoformat(),
            "file": relative,
            "model": MODEL_ID,
            "prompt": prompt,
            "duration_requested": float(duration),
            "steps": int(steps),
            "seed": seed,
            "generation_seconds": result.get("seconds"),
        },
    )
    from adanvil.audio import music_profile

    profile = music_profile(root, relative)["tracks"][0]
    return {
        "path": relative,
        "licence": LICENCE_NAME,
        "attribution_required": False,
        "prompt": prompt,
        "seed": seed,
        "generation_seconds": result.get("seconds"),
        "duration": profile.get("duration"),
        "tempo_bpm": profile.get("tempo_bpm"),
        "feel": profile.get("feel"),
        "loudness_lufs": profile.get("loudness_lufs"),
        "profile": profile,
    }


__all__ = [
    "MusicUnavailableError",
    "available",
    "generate_music",
    "music_config_path",
    "music_setup",
]
