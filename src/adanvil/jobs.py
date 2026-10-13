"""Durable, detached jobs for rendering, voice generation, library indexing and speech refining.

The MCP server only creates a job document and starts this module in a
separate process.  The runner owns all long-running work, so closing the MCP
client cannot terminate an ffmpeg or TTS operation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from adanvil._proc import detached_flags

JobKind = Literal["render", "voice", "index", "refine", "batch", "music"]
# They only take the short catalog lock when saving (a generated track is a new file).
_UNLOCKED_KINDS = ("index", "refine", "music")
ProgressEvent = dict[str, Any]

# On Windows a file that another process is reading cannot be replaced or
# opened for a few milliseconds (sharing violation).  Runners and status
# readers touch the same job files, so retry briefly instead of failing a job.
_SHARING_RETRIES = 50
_SHARING_DELAY = 0.04


def _retry_sharing(action: Callable[[], Any]) -> Any:
    for attempt in range(_SHARING_RETRIES):
        try:
            return action()
        except PermissionError:
            if os.name != "nt" or attempt == _SHARING_RETRIES - 1:
                raise
            time.sleep(_SHARING_DELAY)
    raise AssertionError("unreachable")


def jobs_pointer_dir() -> Path:
    """Folder of ``<job_id>.json`` pointers that outlive any one server process.

    ``ADANVIL_JOBS_DIR`` overrides it; otherwise it is ``adanvil/jobs`` under the user
    cache (``%LOCALAPPDATA%`` on Windows, ``$XDG_CACHE_HOME`` or ``~/.cache`` elsewhere).
    """

    override = os.environ.get("ADANVIL_JOBS_DIR")
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "adanvil" / "jobs"


def resolve_job_file(job_id: str, project_dir: str | Path | None = None) -> Path | None:
    """Find a job document by id: in the project when given, else through the pointer."""

    if not re.fullmatch(r"job-[0-9a-f]{1,64}", job_id):
        return None
    if project_dir is not None:
        candidate = Path(project_dir) / "renders" / "jobs" / f"{job_id}.json"
        if candidate.is_file():
            return candidate
    pointer = jobs_pointer_dir() / f"{job_id}.json"
    try:
        target = Path(str(read_job(pointer)["job_file"]))
    except (KeyError, ValueError):
        return None
    return target if target.is_file() else None


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    try:
        _retry_sharing(lambda: os.replace(temporary, path))
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_job(job_file: str | Path) -> dict[str, Any]:
    """Read one job document, returning a useful failure for malformed JSON."""

    path = Path(job_file)
    try:
        value = json.loads(_retry_sharing(lambda: path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read job file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Job file must contain an object: {path}")
    return value


def update_job(job_file: str | Path, **updates: Any) -> dict[str, Any]:
    """Atomically merge updates into a job document and return the new record."""

    path = Path(job_file)
    record = read_job(path)
    record.update(updates)
    _atomic_write(path, record)
    return record


def _pid_is_alive(pid: object) -> bool | None:
    """Return whether *pid* exists, or ``None`` when the check is inconclusive."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return None
        return True

    # Avoid a dependency on psutil.  PROCESS_QUERY_LIMITED_INFORMATION is
    # enough to distinguish an exited process from one that is still running.
    try:
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            process_query_limited_information, False, pid
        )
        if not handle:
            error = ctypes.windll.kernel32.GetLastError()  # type: ignore[attr-defined]
            # Access denied means the process may still exist; all other
            # failures are treated as an inconclusive check.
            return True if error == 5 else False if error == 87 else None
        exit_code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(  # type: ignore[attr-defined]
            handle, ctypes.byref(exit_code)
        )
        ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        if not ok:
            return None
        return int(exit_code.value) == 259  # STILL_ACTIVE
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def mark_dead_runner(job_file: str | Path, record: dict[str, Any]) -> dict[str, Any]:
    """Mark an orphaned running job, if its process can be proven dead."""

    if record.get("status") != "running":
        return record
    alive = _pid_is_alive(record.get("pid"))
    if alive is False:
        return update_job(
            job_file,
            status="failed",
            stage="failed",
            finished_at=time.time(),
            message="runner process died",
            error="runner process died",
        )
    return record


class _ProjectLock:
    """A cross-platform advisory lock held for the whole runner lifetime.

    The lock is the first byte of the file, whatever the file holds. On Windows
    ``msvcrt.locking`` locks from the *OS* file position, so that position is set with
    ``os.lseek`` on an unbuffered file: a buffered ``seek`` inside Python's read buffer
    leaves the OS position where the last read stopped, which made the lock land on a
    different byte (and so exclude nobody) as soon as the lock file held more than one byte.
    """

    def __init__(self, path: Path, *, timeout: float | None = None, poll: float = 0.2) -> None:
        self.path = path
        self.timeout = timeout
        self.poll = poll
        self._file: Any = None

    def _wait(self, deadline: float | None) -> None:
        if deadline is not None and time.monotonic() >= deadline:
            assert self._file is not None
            self._file.close()
            self._file = None
            raise TimeoutError(f"timed out after {self.timeout}s waiting for {self.path}")
        time.sleep(self.poll)

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a+b", buffering=0)
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        if os.name == "nt":
            import msvcrt

            # Nothing is written to the file: a byte range may be locked beyond the end of
            # a file, and creating the byte raced between processes (two of them each
            # appended one, leaving a longer file).
            while True:
                try:
                    os.lseek(self._file.fileno(), 0, os.SEEK_SET)
                    msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    self._wait(deadline)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    self._wait(deadline)

    def release(self) -> None:
        if self._file is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(self._file.fileno(), 0, os.SEEK_SET)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None

    def __enter__(self) -> _ProjectLock:
        self.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


def _job_sort_key(record: Mapping[str, Any], path: Path) -> tuple[float, str]:
    created = record.get("created_at")
    return (float(created) if isinstance(created, (int, float)) else float("inf"), path.name)


def _has_earlier_pending_job(job_file: Path, record: Mapping[str, Any]) -> bool:
    """Keep lock acquisition FIFO for jobs in this project's jobs directory."""

    current_key = _job_sort_key(record, job_file)
    for candidate in job_file.parent.glob("job-*.json"):
        if candidate == job_file:
            continue
        try:
            other = read_job(candidate)
        except ValueError:
            continue
        if other.get("status") not in {"queued", "running"}:
            continue
        if other.get("kind") in _UNLOCKED_KINDS:
            continue  # index and refine jobs never hold the project lock
        if _job_sort_key(other, candidate) < current_key:
            if other.get("status") == "running":
                other = mark_dead_runner(candidate, other)
                if other.get("status") not in {"queued", "running"}:
                    continue
            return True
    return False


def _acquire_project_lock(job_file: Path, record: dict[str, Any]) -> _ProjectLock:
    lock = _ProjectLock(job_file.parent / ".lock")
    while True:
        current = read_job(job_file)
        if _has_earlier_pending_job(job_file, current):
            update_job(
                job_file,
                stage="waiting_for_project_lock",
                message="waiting for an earlier job in this project",
            )
            time.sleep(0.2)
            continue
        lock.acquire()
        # A job older than this one may have become queued between the check
        # and the lock acquisition.  Release and retry in that case.
        current = read_job(job_file)
        if _has_earlier_pending_job(job_file, current):
            lock.release()
            time.sleep(0.2)
            continue
        return lock


def _emit_progress(progress: Callable[[ProgressEvent], None] | None, event: ProgressEvent) -> None:
    if progress is not None:
        progress(event)


def _voice_summary(root: Path, track: Any, replaced: bool = False) -> dict[str, Any]:
    return {
        "replaced": replaced,
        "manifest": Path(track.manifest).as_posix(),
        "lines": [
            {
                "index": line.index,
                "text": line.text,
                "display": line.display,
                "start": line.start,
                "duration": line.duration,
                "status": line.status,
                "transcript": line.transcript,
                "unmatched_words": line.unmatched_words,
            }
            for line in track.lines
        ],
        "total_duration": track.total_duration,
        "total": track.total_duration,
        "warnings": list(track.warnings),
    }


def _report_json(report: Any) -> dict[str, Any]:
    return report.model_dump(mode="json")


def _run_record(job_file: Path) -> None:
    record = read_job(job_file)
    if record.get("status") in {"done", "failed"}:
        return
    root = Path(str(record["project_dir"])).resolve()
    kind = str(record.get("kind"))
    # Indexing and refining only touch the catalog (locked on save), the sheet cache and the
    # speech sidecars, so they never wait behind a render or a voice job.
    lock = None if kind in _UNLOCKED_KINDS else _acquire_project_lock(job_file, record)
    started = time.time()
    try:
        update_job(
            job_file,
            pid=os.getpid(),
            status="running",
            stage="starting",
            started_at=started,
            message=f"{kind} running",
        )

        def progress(event: ProgressEvent) -> None:
            update_job(
                job_file,
                stage=str(event.get("stage", "running")),
                progress={
                    "done": int(event.get("done", 0)),
                    "total": int(event.get("total", 0)),
                    "unit": str(event.get("unit", "segments")),
                },
                message=str(event.get("message", "")),
            )

        inputs = record.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError("job inputs must be an object")
        if kind == "voice":
            from adanvil.voice import VoiceSpec, generate_voice

            name = str(inputs["name"])
            spec = VoiceSpec.model_validate(inputs["spec"])
            replaced = (root / "voice" / name / "manifest.json").is_file()
            track = generate_voice(root, spec, name, progress=progress)
            update_job(
                job_file,
                status="done",
                stage="complete",
                progress={"done": len(spec.lines), "total": len(spec.lines), "unit": "lines"},
                finished_at=time.time(),
                message="voice generation complete",
                voice=_voice_summary(root, track, replaced),
            )
        elif kind == "music":
            from adanvil.musicgen import generate_music

            result = generate_music(
                root,
                str(inputs["prompt"]),
                float(inputs.get("duration", 30.0)),
                steps=int(inputs.get("steps", 27)),
                seed=inputs.get("seed"),
                progress=progress,
            )
            update_job(
                job_file,
                status="done",
                stage="complete",
                progress={"done": 1, "total": 1, "unit": "tracks"},
                finished_at=time.time(),
                message="music generation complete",
                music=result,
            )
        elif kind == "render":
            from adanvil.render import compose
            from adanvil.storyboard import load_storyboard

            storyboard = load_storyboard(inputs["storyboard"])
            quality = str(inputs.get("quality", "final"))
            report = compose(storyboard, root=root, quality=quality, progress=progress)
            report_json = _report_json(report)
            report_path = Path(report.path).with_suffix(".report.json")
            _atomic_write(report_path, report_json)
            update_job(
                job_file,
                status="done",
                stage="complete",
                progress={
                    "done": len(storyboard.segments),
                    "total": len(storyboard.segments),
                    "unit": "segments",
                },
                finished_at=time.time(),
                message="render complete",
                report=report_json,
                report_path=str(report_path),
            )
        elif kind == "batch":
            from adanvil.batch import run_batch

            total = len(inputs["items"])

            def publish(partial: dict[str, Any]) -> None:
                update_job(job_file, batch=partial)

            result = run_batch(root, inputs, progress=progress, publish=publish)
            failed = [item["name"] for item in result["renders"] if item["status"] != "done"]
            update_job(
                job_file,
                status="done",
                stage="complete",
                progress={"done": total, "total": total, "unit": "renders"},
                finished_at=time.time(),
                message=(
                    f"batch complete in {result['total_seconds']:.1f}s"
                    + (f", {len(failed)} failed: {', '.join(failed)}" if failed else "")
                ),
                batch=result,
            )
        elif kind == "index":
            from adanvil.library import index_library, index_summary

            source = str(inputs.get("source", "footage"))
            total = int(record.get("progress", {}).get("total", 0))
            seen = 0

            def index_progress(message: str) -> None:
                nonlocal seen
                speech = re.match(r"speech (\d+)/(\d+)", message)
                if speech:
                    # Transcription runs in this job after the files are measured: its own
                    # counter (clips transcribed) replaces the file counter while it runs.
                    progress(
                        {
                            "stage": "transcribing",
                            "done": int(speech.group(1)),
                            "total": int(speech.group(2)),
                            "unit": "clips",
                            "message": message,
                        }
                    )
                    return
                looked = re.match(r"vision (\d+)/(\d+)", message)
                if looked:
                    # The vision pass runs after the files are measured: its own counter.
                    progress(
                        {
                            "stage": "vision",
                            "done": int(looked.group(1)),
                            "total": int(looked.group(2)),
                            "unit": "entries",
                            "message": message,
                        }
                    )
                    return
                if not message.startswith(("indexed ", "unchanged ", "error ", "removed ")):
                    return
                seen += 1
                progress(
                    {
                        "stage": "indexing",
                        "done": min(seen, total) if total else seen,
                        "total": total,
                        "unit": "files",
                        "message": message,
                    }
                )

            result = index_library(
                root,
                source,
                progress=index_progress,
                transcribe=inputs.get("transcribe", "auto"),
                vision=inputs.get("vision", "auto"),
            )
            update_job(
                job_file,
                status="done",
                stage="complete",
                progress={"done": total, "total": total, "unit": "files"},
                finished_at=time.time(),
                message="index complete",
                index=index_summary(root, result),
            )
        elif kind == "refine":
            from adanvil.library import refine_speech, refine_targets

            entries, skipped = refine_targets(root, [str(item) for item in inputs["entry_ids"]])
            total = len(entries)

            def refine_progress(message: str) -> None:
                speech = re.match(r"speech (\d+)/(\d+)", message)
                if speech:
                    progress(
                        {
                            "stage": "transcribing",
                            "done": int(speech.group(1)),
                            "total": int(speech.group(2)),
                            "unit": "clips",
                            "message": message,
                        }
                    )

            outcome = refine_speech(
                root,
                [entry.id for entry in entries],
                model=inputs.get("model"),
                language=inputs.get("language"),
                progress=refine_progress,
            )
            update_job(
                job_file,
                status="done",
                stage="complete",
                progress={"done": total, "total": total, "unit": "clips"},
                finished_at=time.time(),
                message="speech refine complete",
                refine={
                    "model": outcome.model,
                    "refined": outcome.refined,
                    "skipped": [*skipped, *outcome.skipped],
                    "errors": outcome.errors,
                },
            )
        else:
            raise ValueError(f"unknown job kind: {kind!r}")
    except Exception as exc:
        error = str(exc)
        if exc.__class__.__name__ == "VoiceUnavailableError":
            error = f"{error} Run 'adanvil voice-setup' to install the voice environment."
        update_job(
            job_file,
            status="failed",
            stage="failed",
            finished_at=time.time(),
            message=f"{kind} failed",
            error=error,
        )
    finally:
        if lock is not None:
            lock.release()


def run_job(job_file: str | Path) -> int:
    """Run one job document in the detached child process."""

    _run_record(Path(job_file).resolve())
    return 0


def start_job(
    project_dir: str | Path,
    kind: JobKind,
    inputs: Mapping[str, Any],
    *,
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create and launch a detached job, returning its initial job record."""

    if kind not in {"render", "voice", "index", "refine", "batch", "music"}:
        raise ValueError("kind must be 'render', 'voice', 'index', 'refine', 'batch' or 'music'")
    root = Path(project_dir).resolve()
    jobs_dir = root / "renders" / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_id = f"job-{uuid.uuid4().hex[:16]}"
    job_file = jobs_dir / f"{job_id}.json"
    total = int((progress or {}).get("total", 0))
    default_unit = {
        "voice": "lines",
        "index": "files",
        "refine": "clips",
        "batch": "renders",
        "music": "steps",
    }.get(kind, "segments")
    unit = str((progress or {}).get("unit", default_unit))
    record: dict[str, Any] = {
        "job_id": job_id,
        "kind": kind,
        "project_dir": str(root),
        "job_file": str(job_file),
        "inputs": dict(inputs),
        "status": "queued",
        "stage": "queued",
        "progress": {"done": 0, "total": total, "unit": unit},
        "message": f"{kind} queued",
        "created_at": time.time(),
        "queued_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "pid": None,
        "report": None,
        "voice": None,
        "index": None,
        "refine": None,
        "batch": None,
        "music": None,
        "report_path": None,
        "error": None,
    }
    _atomic_write(job_file, record)
    try:
        _atomic_write(jobs_pointer_dir() / f"{job_id}.json", {"job_file": str(job_file)})
    except OSError:
        pass  # job_status can still find the job through project_dir
    log_file = job_file.with_suffix(".log")
    log = log_file.open("ab")
    try:
        kwargs: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": log,
            "stderr": log,
            "close_fds": True,
        }
        if os.name == "nt":
            kwargs["creationflags"] = detached_flags()
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen(
            [sys.executable, "-m", "adanvil.jobs", "run", str(job_file)],
            **kwargs,
        )
    finally:
        log.close()
    record = update_job(job_file, pid=process.pid)
    return record


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m adanvil.jobs")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run a durable job file")
    run_parser.add_argument("job_file", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_job(args.job_file)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "JobKind",
    "ProgressEvent",
    "mark_dead_runner",
    "jobs_pointer_dir",
    "read_job",
    "resolve_job_file",
    "run_job",
    "start_job",
    "update_job",
]
