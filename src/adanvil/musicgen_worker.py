"""Standalone worker of the optional music environment (ACE-Step v1, Apache-2.0).

Run by the interpreter of the music environment, never imported by AdAnvil. It imports neither
``adanvil`` nor any project dependency. Usage: ``python musicgen_worker.py request.json``; the
last line printed is ``ADANVIL_RESULT {json}``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

MODEL = "ACE-Step/ACE-Step-v1-3.5B"


def _threads() -> int:
    try:
        override = int(os.environ.get("ADANVIL_MUSIC_THREADS", ""))
    except ValueError:
        override = 0
    if override >= 1:
        return override
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def main(argv: list[str]) -> int:
    request = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    threads = _threads()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, str(threads))
    # float32 on CPU: bfloat16 is slower without hardware support.
    os.environ.setdefault("ACE_PIPELINE_DTYPE", "float32")
    try:
        import torch
        from acestep.pipeline_ace_step import ACEStepPipeline
    except Exception as exc:  # the environment is not set up
        print(
            "ADANVIL_RESULT "
            + json.dumps({"ok": False, "error": f"ACE-Step is unavailable: {exc}"})
        )
        return 1
    torch.set_num_threads(threads)
    started = time.time()
    out = Path(request["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        pipeline = ACEStepPipeline(
            checkpoint_dir=request["checkpoint_dir"], dtype="float32", overlapped_decode=True
        )
        pipeline(
            format="wav",
            audio_duration=float(request["duration"]),
            prompt=str(request["prompt"]),
            lyrics="[instrumental]",
            infer_step=int(request["steps"]),
            guidance_scale=15.0,
            scheduler_type="euler",
            cfg_type="apg",
            omega_scale=10.0,
            manual_seeds=[int(request["seed"])],
            guidance_interval=0.5,
            guidance_interval_decay=0.0,
            min_guidance_scale=3.0,
            use_erg_tag=True,
            use_erg_lyric=False,
            use_erg_diffusion=True,
            save_path=str(out),
        )
    except Exception as exc:
        print("ADANVIL_RESULT " + json.dumps({"ok": False, "error": f"generation failed: {exc}"}))
        return 1
    if not out.is_file():
        print("ADANVIL_RESULT " + json.dumps({"ok": False, "error": "no audio file was written"}))
        return 1
    print(
        "ADANVIL_RESULT "
        + json.dumps({"ok": True, "out": str(out), "seconds": round(time.time() - started, 1)})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
