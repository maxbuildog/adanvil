"""Talk to the vision worker, and hand out the engine the rest of AdAnvil uses.

``open_engine()`` yields the engine to use or ``None``: the worker of the vision environment
(one persistent process, models loaded once, stopped after ten idle minutes), an engine
injected by a test (``use_engine``), or nothing when the environment is not installed or
``ADANVIL_VISION=off``.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import queue
import subprocess
import tempfile
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import IO, Any

import numpy as np

from adanvil._proc import NO_WINDOW

from .engine import Face, FacePicture, VisionEngine
from .labels import TAG_CENTER, TAG_SCALE
from .setup import models_dir, vision_config_path, worker_path

UNAVAILABLE = "unavailable: run adanvil vision-setup to install the vision environment"
_START_TIMEOUT = 600.0  # the first call loads the models
_CALL_TIMEOUT = 900.0
_IDLE_SECONDS = 600.0


class VisionUnavailableError(RuntimeError):
    """The vision environment is missing or its worker failed."""


class VisionOperationError(RuntimeError):
    """The worker is fine but one operation failed (an unreadable picture, for example)."""


def vision_disabled() -> bool:
    return os.environ.get("ADANVIL_VISION", "").strip().casefold() in {"off", "0", "false", "no"}


def configured_interpreter() -> str | None:
    """The Python of the vision environment, when ``adanvil vision-setup`` was run."""

    path = vision_config_path()
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("python")
    except (OSError, ValueError, AttributeError):
        return None
    return str(value) if value and Path(str(value)).is_file() else None


_override: VisionEngine | None = None


@contextlib.contextmanager
def use_engine(engine: VisionEngine | None) -> Iterator[None]:
    """Make ``open_engine`` hand out ``engine`` (tests inject a fake one here)."""

    global _override
    previous = _override
    _override = engine
    try:
        yield
    finally:
        _override = previous


def available() -> bool:
    """Whether ``open_engine`` would give an engine (without starting any process)."""

    if _override is not None:
        return True
    return not vision_disabled() and configured_interpreter() is not None


def expected_model() -> str | None:
    """The model name stored with frame data made by the engine that ``get_engine`` would give,
    or ``None`` when there is none: it tells which stored data is out of date."""

    if _override is not None:
        return _override.model
    if vision_disabled() or configured_interpreter() is None:
        return None
    from .worker import MODEL, MODEL_REVISION

    return f"{MODEL}@{MODEL_REVISION[:8]}"


def status() -> str:
    """``ready`` or the reason there is no engine."""

    if _override is not None:
        return "ready"
    if vision_disabled():
        return "unavailable: switched off (ADANVIL_VISION=off)"
    return "ready" if configured_interpreter() is not None else UNAVAILABLE


class WorkerVisionEngine:
    """One ``worker.py --serve`` process, shared by every caller of this Python process."""

    def __init__(self, interpreter: str) -> None:
        self.interpreter = interpreter
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._replies: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._log: IO[str] | None = None
        self._log_path: Path | None = None
        self._next = 0
        self._timer: threading.Timer | None = None
        self.model = ""
        self.logit_scale = 0.0
        self.logit_bias = 0.0

    # -- process --------------------------------------------------------------------------

    def _alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _reader(self, process: subprocess.Popen[str]) -> None:
        try:
            for line in process.stdout if process.stdout is not None else ():
                try:
                    document = json.loads(line)
                except ValueError:
                    continue  # a stray line is not a protocol message
                if isinstance(document, dict):
                    self._replies.put(document)
        except (OSError, ValueError):
            pass
        finally:
            self._replies.put(None)  # the worker is gone

    def _log_tail(self) -> str:
        if self._log_path is None:
            return ""
        try:
            return self._log_path.read_text(encoding="utf-8", errors="replace")[-1500:].strip()
        except OSError:
            return ""

    def _start(self) -> None:
        handle, name = tempfile.mkstemp(prefix="adanvil-vision-", suffix=".log")
        os.close(handle)
        self._log_path = Path(name)
        self._log = open(name, "w", encoding="utf-8", errors="replace")  # noqa: SIM115
        environment = {**os.environ, "ADANVIL_VISION_DIR": str(models_dir())}
        try:
            self._process = subprocess.Popen(
                [self.interpreter, str(worker_path()), "--serve"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=environment,
                creationflags=NO_WINDOW,
            )
        except OSError as exc:
            self._stop()
            raise VisionUnavailableError(f"could not start the vision worker: {exc}") from exc
        self._replies = queue.Queue()
        threading.Thread(target=self._reader, args=(self._process,), daemon=True).start()
        try:
            hello = self._replies.get(timeout=_START_TIMEOUT)
        except queue.Empty:
            hello = None
        if not hello or not hello.get("ready"):
            detail = self._log_tail() or "no worker output"
            self._stop()
            raise VisionUnavailableError(f"the vision worker did not start: {detail}")

    def _stop(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                with contextlib.suppress(OSError, ValueError):
                    if process.stdin is not None:
                        process.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                        process.stdin.flush()
                        process.stdin.close()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    with contextlib.suppress(Exception):
                        process.wait(timeout=10)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    with contextlib.suppress(OSError, ValueError):
                        stream.close()
        if self._log is not None:
            self._log.close()
            self._log = None
        if self._log_path is not None:
            with contextlib.suppress(OSError):
                self._log_path.unlink(missing_ok=True)
            self._log_path = None

    def close(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._stop()

    def _arm_idle(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(_IDLE_SECONDS, self.close)
        self._timer.daemon = True
        self._timer.start()

    # -- calls ----------------------------------------------------------------------------

    def call(self, op: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if not self._alive():
                self._stop()
                self._start()
            assert self._process is not None and self._process.stdin is not None
            self._next += 1
            try:
                self._process.stdin.write(
                    json.dumps({"id": self._next, "op": op}, ensure_ascii=False) + "\n"
                )
                self._process.stdin.flush()
            except (OSError, ValueError) as exc:
                self._stop()
                raise VisionUnavailableError(f"the vision worker is not reachable: {exc}") from exc
            while True:
                try:
                    reply = self._replies.get(timeout=_CALL_TIMEOUT)
                except queue.Empty:
                    detail = self._log_tail() or "no worker output"
                    self._stop()
                    raise VisionUnavailableError(f"the vision worker timed out: {detail}") from None
                if reply is None:
                    detail = self._log_tail() or "no worker output"
                    self._stop()
                    raise VisionUnavailableError(f"the vision worker stopped: {detail}")
                if reply.get("id") == self._next:
                    break
            self._arm_idle()
        result = reply.get("result")
        if not isinstance(result, dict) or not result.get("ok"):
            error = result.get("error") if isinstance(result, dict) else None
            message = str(error or "the vision worker returned an error")
            if "dependencies are unavailable" in message:
                raise VisionUnavailableError(message)
            raise VisionOperationError(message)
        return result

    def _ensure_info(self) -> None:
        if not self.model:
            info = self.call({"op": "info"})
            self.model = str(info["model"])
            # The model's own scale and bias are tuned for caption pairs: see labels.TAG_CENTER.
            self.logit_scale = TAG_SCALE
            self.logit_bias = -TAG_SCALE * TAG_CENTER

    def embed_images(self, paths: Sequence[Path]) -> np.ndarray:
        self._ensure_info()
        if not paths:
            return np.zeros((0, 0), dtype=np.float32)
        result = self.call({"op": "embed_images", "paths": [str(path) for path in paths]})
        return np.asarray(result["embeddings"], dtype=np.float32)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_info()
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        result = self.call({"op": "embed_texts", "texts": list(texts)})
        return np.asarray(result["embeddings"], dtype=np.float32)

    def detect_faces(self, paths: Sequence[Path]) -> list[FacePicture]:
        if not paths:
            return []
        result = self.call({"op": "detect_faces", "paths": [str(path) for path in paths]})
        return [
            FacePicture(
                (int(item["size"][0]), int(item["size"][1])),
                [Face((row[0], row[1], row[2], row[3]), float(row[4])) for row in item["faces"]],
            )
            for item in result["pictures"]
        ]


_shared: WorkerVisionEngine | None = None
_shared_lock = threading.Lock()


def _shared_engine(interpreter: str) -> WorkerVisionEngine:
    global _shared
    with _shared_lock:
        if _shared is None or _shared.interpreter != interpreter:
            if _shared is not None:
                _shared.close()
            _shared = WorkerVisionEngine(interpreter)
        return _shared


def close_shared_engine() -> None:
    global _shared
    with _shared_lock:
        engine, _shared = _shared, None
    if engine is not None:
        engine.close()


atexit.register(close_shared_engine)


def get_engine() -> VisionEngine | None:
    """The engine to use now, or ``None`` when there is none (see the module docstring)."""

    if _override is not None:
        return _override
    if vision_disabled():
        return None
    interpreter = configured_interpreter()
    return _shared_engine(interpreter) if interpreter is not None else None


__all__ = [
    "UNAVAILABLE",
    "VisionOperationError",
    "VisionUnavailableError",
    "WorkerVisionEngine",
    "available",
    "close_shared_engine",
    "configured_interpreter",
    "expected_model",
    "get_engine",
    "status",
    "use_engine",
    "vision_disabled",
]
