from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from adanvil.jobs import mark_dead_runner, read_job, start_job
from adanvil.project import init_project
from adanvil.storyboard import Storyboard


def _wait(job_file: Path, timeout: float = 120.0) -> dict:  # runners start a new Python process
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = read_job(job_file)
        if record.get("status") in {"done", "failed"}:
            return record
        time.sleep(0.1)
    return read_job(job_file)


def _voice_inputs(name: str) -> dict:
    return {"name": name, "spec": {"engine": "fake", "lines": ["one", "two"]}}


def test_voice_runner_completes_and_reaches_total(tmp_path: Path) -> None:
    record = start_job(
        tmp_path,
        "voice",
        _voice_inputs("direct"),
        progress={"total": 2, "unit": "lines"},
    )

    result = _wait(Path(record["job_file"]))

    assert result["status"] == "done", result
    assert result["progress"] == {"done": 2, "total": 2, "unit": "lines"}
    assert result["voice"]["lines"]


def test_job_survives_parent_process_exit(tmp_path: Path) -> None:
    code = (
        "from pathlib import Path; "
        "from adanvil.jobs import start_job; "
        f"r=start_job(Path({str(tmp_path)!r}), 'voice', {_voice_inputs('child')!r}, "
        "progress={'total': 2, 'unit': 'lines'}); "
        "print(r['job_file'], flush=True)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    job_file = Path(completed.stdout.strip())
    result = _wait(job_file)

    assert result["status"] == "done", result


def test_small_render_runner_completes(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    init_project(tmp_path, "detached render")
    footage = tmp_path / "footage" / "ref_ad.mp4"
    footage.write_bytes(Path(str(media["ref_ad.mp4"]["path"])).read_bytes())
    storyboard = Storyboard.model_validate(
        {
            "name": "detached_render",
            "segments": [{"type": "clip", "source": "footage/ref_ad.mp4", "in": 0, "out": 1}],
            "audio": {},
            "output": {"path": "renders/{name}.mp4", "quality": "preview"},
        }
    )
    record = start_job(
        tmp_path,
        "render",
        {"storyboard": storyboard.model_dump(mode="json", by_alias=True), "quality": "preview"},
        progress={"total": 1, "unit": "segments"},
    )

    result = _wait(Path(record["job_file"]), timeout=180)

    assert result["status"] == "done", result
    assert result["progress"] == {"done": 1, "total": 1, "unit": "segments"}


def test_jobs_for_one_project_run_sequentially(tmp_path: Path) -> None:
    first = start_job(
        tmp_path,
        "voice",
        _voice_inputs("first"),
        progress={"total": 2, "unit": "lines"},
    )
    second = start_job(
        tmp_path,
        "voice",
        _voice_inputs("second"),
        progress={"total": 2, "unit": "lines"},
    )

    first_result = _wait(Path(first["job_file"]))
    second_result = _wait(Path(second["job_file"]))

    assert first_result["status"] == second_result["status"] == "done"
    assert first_result["finished_at"] <= second_result["started_at"]


def test_dead_runner_is_detected(tmp_path: Path) -> None:
    job_file = tmp_path / "renders" / "jobs" / "job-dead.json"
    job_file.parent.mkdir(parents=True)
    job_file.write_text(
        json.dumps(
            {
                "job_id": "job-dead",
                "status": "running",
                "pid": 2_147_483_647,
                "stage": "tts",
                "progress": {"done": 1, "total": 2, "unit": "lines"},
            }
        ),
        encoding="utf-8",
    )

    result = mark_dead_runner(job_file, read_job(job_file))

    assert result["status"] == "failed"
    assert result["message"] == "runner process died"
