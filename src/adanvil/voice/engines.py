"""Voice engine implementations and the optional worker protocol."""

from __future__ import annotations

import contextlib
import json
import math
import os
import queue
import re
import shutil
import struct
import subprocess
import threading
import time
import uuid
import wave
from array import array
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Protocol

from adanvil._proc import NO_WINDOW
from adanvil.ffmpeg import probe, run
from adanvil.safeio import replace_text

# A Kokoro voice is a name like ``af_heart``: Kokoro treats a value ending in ``.pt`` as a file
# to load and a comma as a blend, so nothing else reaches the worker (which checks it again).
KOKORO_VOICE = re.compile(r"[a-z]{2}_[a-z0-9]+")


class VoiceUnavailableError(RuntimeError):
    """Raised when the optional voice environment cannot be used."""


@dataclass
class LineJob:
    index: int
    text: str
    language: str = "en"
    reference: Path | None = None
    out: Path | None = None
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.7
    seed: int | None = None
    voice: str | None = (
        None  # a fixed voice of the engine (kokoro); chatterbox clones ``reference``
    )


@dataclass
class LineResult:
    index: int
    path: Path
    sample_rate: int = 48_000


@dataclass
class Alignment:
    path: Path
    words: list[tuple[float, float, str, float]] = field(default_factory=list)
    text: str = ""
    language: str | None = None  # the language the transcriber detected, when it says
    error: str | None = None  # why this file failed, when a batch ran with ``strict=False``


class VoiceEngine(Protocol):
    def synthesize(
        self,
        lines: list[LineJob],
        *,
        progress: Callable[[int], None] | None = None,
    ) -> list[LineResult]: ...

    def align(self, paths: list[Path], language: str, model: str) -> list[Alignment]: ...


def _cache_root(project_dir: Path | None, paths: list[Path]) -> Path:
    if project_dir is not None:
        return Path(project_dir).resolve() / "cache"
    for path in paths:
        resolved = Path(path).resolve()
        for parent in resolved.parents:
            if parent.name == "voice":
                return parent.parent / "cache"
    return Path(os.environ.get("ADANVIL_CACHE", Path.home() / ".cache" / "adanvil"))


def _interpreter_from_config() -> str | None:
    config = Path(os.environ.get("ADANVIL_CACHE", Path.home() / ".cache" / "adanvil"))
    config_path = config / "voice_config.json"
    if not config_path.is_file():
        return None
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
        return str(value.get("python")) if value.get("python") else None
    except (OSError, ValueError, TypeError):
        return None


def _resolve_executable(value: str | None) -> str | None:
    if not value:
        return None
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate)
    return shutil.which(value)


class SessionFailedError(RuntimeError):
    """The persistent worker could not start or died; ``results`` holds what it finished."""

    def __init__(self, message: str, results: list[dict] | None = None) -> None:
        super().__init__(message)
        self.results = results or []


class _Session:
    """One ``worker.py --serve`` process: models load once and serve every operation."""

    def __init__(
        self,
        interpreter: str,
        worker: Path,
        log_path: Path,
    ) -> None:
        self.log_path = log_path
        self.replies: queue.Queue[dict | None] = queue.Queue()
        self._log = open(log_path, "w", encoding="utf-8", errors="replace")  # noqa: SIM115
        try:
            self.process = subprocess.Popen(
                [interpreter, str(worker), "--serve"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log,
                shell=False,
                text=True,
                encoding="utf-8",
                bufsize=1,
                creationflags=NO_WINDOW,
            )
        except OSError:
            self._log.close()
            raise
        self._next = 0
        self.ready = False  # the worker said hello (it reads requests now)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        stdout = self.process.stdout
        try:
            for line in stdout if stdout is not None else ():
                try:
                    document = json.loads(line)
                except ValueError:
                    continue  # a stray line is not a protocol message
                if isinstance(document, dict):
                    self.replies.put(document)
        except (OSError, ValueError):
            pass
        finally:
            self.replies.put(None)  # the worker is gone

    def send(self, op: dict) -> int:
        self._next += 1
        stdin: IO[str] | None = self.process.stdin
        if stdin is None:
            raise SessionFailedError("persistent voice worker has no input pipe")
        try:
            stdin.write(json.dumps({"id": self._next, "op": op}, ensure_ascii=False) + "\n")
            stdin.flush()
        except (OSError, ValueError) as exc:
            raise SessionFailedError(f"persistent voice worker is not reachable: {exc}") from exc
        return self._next

    def alive(self) -> bool:
        return self.process.poll() is None

    def output(self) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")[-2000:].strip()
        except OSError:
            return ""

    def kill(self) -> None:
        with contextlib.suppress(OSError):
            self.process.kill()
        with contextlib.suppress(Exception):
            self.process.wait(timeout=10)
        self._cleanup()

    def close(self) -> None:
        stdin = self.process.stdin
        if self.alive() and stdin is not None:
            with contextlib.suppress(OSError, ValueError):
                stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                stdin.flush()
                stdin.close()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.kill()
                return
        self._cleanup()

    def _cleanup(self) -> None:
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                with contextlib.suppress(OSError, ValueError):
                    stream.close()
        self._log.close()
        with contextlib.suppress(OSError):  # a scanner may hold it for a moment (Windows)
            self.log_path.unlink(missing_ok=True)


class WorkerEngine:
    """Run all heavy TTS/transcription work in a separate Python process."""

    def __init__(
        self,
        python: str | None = None,
        *,
        project_dir: Path | None = None,
        timeout: float | None = None,
        persistent: bool | None = None,
        engine: str = "chatterbox",
    ) -> None:
        self.engine = engine  # the synthesis engine of the worker: chatterbox or kokoro
        self.python = python
        self.project_dir = Path(project_dir).resolve() if project_dir else None
        self.timeout = timeout
        self.worker = Path(__file__).with_name("worker.py")
        # One long-lived worker per job (models load once). ``ADANVIL_VOICE_PERSISTENT=0``
        # or ``persistent=False`` goes back to one process per operation batch.
        if persistent is None:
            persistent = os.environ.get("ADANVIL_VOICE_PERSISTENT", "1").strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
        self.persistent = persistent
        self.fallbacks: list[str] = []  # why the persistent mode was given up, if it was
        self._session: _Session | None = None
        self._session_depth = 0

    def _timeout(self, operations: int) -> float:
        """Model loading plus a generous per-operation budget (CPU TTS on a busy box)."""
        if self.timeout is not None:
            return self.timeout
        per_op = float(os.environ.get("ADANVIL_VOICE_TIMEOUT", "1200"))
        return 600.0 + per_op * max(1, operations)

    def _python(self, paths: list[Path]) -> str:
        selected = _resolve_executable(self.python)
        if selected is None:
            selected = _resolve_executable(os.environ.get("ADANVIL_VOICE_PYTHON"))
        if selected is None:
            selected = _resolve_executable(_interpreter_from_config())
        if selected is None:
            raise VoiceUnavailableError(
                "Voice TTS is unavailable: no voice interpreter was found. "
                "Run adanvil.voice.setup.voice_setup() (or set ADANVIL_VOICE_PYTHON)."
            )
        return selected

    @contextlib.contextmanager
    def session(self) -> Iterator[WorkerEngine]:
        """Keep one worker process (and its loaded models) for every call in the block."""

        self._session_depth += 1
        try:
            yield self
        finally:
            self._session_depth -= 1
            if self._session_depth == 0:
                self._close_session()

    def _close_session(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            session.close()

    def _cache_dir(self, paths: list[Path]) -> Path:
        cache_dir = _cache_root(self.project_dir, paths) / "voice-worker"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _run(
        self,
        ops: list[dict],
        paths: list[Path],
        *,
        progress: Callable[[int], None] | None = None,
        strict: bool = True,
        on_partial: Callable[[list[dict]], None] | None = None,
    ) -> list[dict]:
        interpreter = self._python(paths)
        if not ops:
            return []
        results: list[dict] = []
        if self.persistent:
            try:
                results = self._run_persistent(
                    interpreter, ops, paths, progress=progress, on_partial=on_partial
                )
            except SessionFailedError as failure:
                # Give up on the persistent mode for this engine and finish the remaining
                # operations the old way (one process per batch).
                self.persistent = False
                self.fallbacks.append(str(failure))
                results = list(failure.results)
        if len(results) < len(ops):
            done = results[:]

            def shifted_progress(count: int) -> None:
                if progress is not None:
                    progress(len(done) + count)

            def shifted_partial(values: list[dict]) -> None:
                if on_partial is not None:
                    on_partial([*done, *values])

            results = [
                *done,
                *self._run_files(
                    interpreter,
                    ops[len(done) :],
                    paths,
                    progress=shifted_progress if done else progress,
                    on_partial=shifted_partial if done else on_partial,
                ),
            ]
        if strict:
            for result in results:
                if not result.get("ok"):
                    raise RuntimeError(str(result.get("error", "Voice worker operation failed")))
        return results

    def _run_persistent(
        self,
        interpreter: str,
        ops: list[dict],
        paths: list[Path],
        *,
        progress: Callable[[int], None] | None = None,
        on_partial: Callable[[list[dict]], None] | None = None,
    ) -> list[dict]:
        """Send ``ops`` to the long-lived worker; results come back one by one."""

        if self._session is None or not self._session.alive():
            self._close_session()
            log_path = self._cache_dir(paths) / f".worker-{uuid.uuid4().hex}.log"
            try:
                self._session = _Session(interpreter, self.worker, log_path)
            except OSError as exc:
                raise SessionFailedError(f"could not start the voice worker: {exc}") from exc
        session = self._session
        one_shot = self._session_depth == 0
        timeout = self._timeout(len(ops))
        deadline = time.monotonic() + timeout
        results: list[dict] = []
        sent = 0
        in_flight = 0
        try:
            while len(results) < len(ops):
                while session.ready and sent < len(ops) and in_flight < 2:
                    session.send(ops[sent])
                    sent += 1
                    in_flight += 1
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    session.kill()
                    self._session = None
                    raise RuntimeError(
                        f"Voice worker timed out after {timeout:.0f}s with "
                        f"{len(results)}/{len(ops)} operations done. "
                        "CPU text-to-speech is slow on a busy machine: set "
                        "ADANVIL_VOICE_TIMEOUT (seconds per operation) higher, or retry "
                        "when the machine is idle."
                    )
                try:
                    reply = session.replies.get(timeout=min(0.2, remaining))
                except queue.Empty:
                    continue
                if reply is None:
                    detail = session.output() or "no worker output"
                    session.kill()
                    self._session = None
                    raise SessionFailedError(
                        f"persistent voice worker stopped after {len(results)}/{len(ops)} "
                        f"operations: {detail}",
                        results,
                    )
                if reply.get("ready"):
                    session.ready = True
                    continue
                result = reply.get("result")
                if not isinstance(result, dict):
                    continue
                in_flight -= 1
                results.append(result)
                if on_partial is not None:
                    try:
                        on_partial(list(results))
                    except BaseException:  # the caller failed: never leave the worker running
                        session.kill()
                        self._session = None
                        raise
                if progress is not None:
                    progress(len(results))
        except SessionFailedError:
            session.kill()
            self._session = None
            raise
        finally:
            if one_shot and self._session is session:
                self._close_session()
        return results

    def _run_files(
        self,
        interpreter: str,
        ops: list[dict],
        paths: list[Path],
        *,
        progress: Callable[[int], None] | None = None,
        on_partial: Callable[[list[dict]], None] | None = None,
    ) -> list[dict]:
        """One process for this batch, files as the protocol (the original mode)."""

        cache_dir = self._cache_dir(paths)
        token = uuid.uuid4().hex
        jobs_path = cache_dir / f".jobs-{token}.json"
        results_path = cache_dir / f".results-{token}.json"
        try:
            replace_text(jobs_path, json.dumps({"ops": ops}))
            try:
                log_path = cache_dir / f".worker-{token}.log"
                worker_log = open(log_path, "w", encoding="utf-8", errors="replace")  # noqa: SIM115
                process = subprocess.Popen(
                    [interpreter, str(self.worker), str(jobs_path), str(results_path)],
                    stdout=worker_log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    shell=False,
                    creationflags=NO_WINDOW,
                )
            except OSError as exc:
                raise VoiceUnavailableError(f"Could not start voice worker: {exc}") from exc
            deadline = time.monotonic() + self._timeout(len(ops))
            reported = -1
            while process.poll() is None:
                done = 0
                values: list = []
                try:
                    partial = json.loads(results_path.read_text(encoding="utf-8"))
                    values = partial.get("results", [])
                    values = values if isinstance(values, list) else []
                    done = len(values)
                except (OSError, ValueError):
                    pass
                if done != reported:
                    reported = done
                    if on_partial is not None and values:
                        try:
                            on_partial(values)
                        except BaseException:  # the caller failed: never leave the worker running
                            process.kill()
                            process.wait()
                            worker_log.close()
                            raise
                    if progress is not None:
                        progress(done)
                if time.monotonic() >= deadline:
                    process.kill()
                    process.wait()
                    worker_log.close()
                    raise RuntimeError(
                        f"Voice worker timed out after {self._timeout(len(ops)):.0f}s with "
                        f"{reported if reported >= 0 else 0}/{len(ops)} operations done. "
                        "CPU text-to-speech is slow on a busy machine: set "
                        "ADANVIL_VOICE_TIMEOUT (seconds per operation) higher, or retry "
                        "when the machine is idle."
                    )
                time.sleep(0.2)
            process.wait()
            worker_log.close()
            try:
                output = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            except OSError:
                output = ""
            completed_returncode = process.returncode
            if progress is not None and reported < len(ops):
                progress(len(ops))
            if completed_returncode != 0 and not results_path.is_file():
                detail = output.strip() or "no worker output"
                raise RuntimeError(
                    f"Voice worker failed with code {completed_returncode}: {detail}"
                )
            try:
                document = json.loads(results_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                detail = output.strip() or "no worker output"
                raise RuntimeError(f"Voice worker returned invalid results: {detail}") from exc
        finally:
            jobs_path.unlink(missing_ok=True)
            results_path.unlink(missing_ok=True)
            results_path.with_name(f".{results_path.name}.tmp").unlink(missing_ok=True)
            (cache_dir / f".worker-{token}.log").unlink(missing_ok=True)
        results = document.get("results")
        if not isinstance(results, list) or len(results) != len(ops):
            raise RuntimeError("Voice worker returned an invalid results list")
        return results

    def synthesize(
        self,
        lines: list[LineJob],
        *,
        progress: Callable[[int], None] | None = None,
    ) -> list[LineResult]:
        operations = []
        for line in lines:
            if line.out is None or (self.engine != "kokoro" and line.reference is None):
                raise ValueError("Chatterbox lines require reference and out paths")
            if self.engine == "kokoro":
                if line.voice is not None and not KOKORO_VOICE.fullmatch(line.voice):
                    raise ValueError(
                        f"{line.voice!r} is not a Kokoro voice name such as af_heart or ff_siwis"
                    )
                operations.append(
                    {
                        "op": "tts",
                        "engine": "kokoro",
                        "text": line.text,
                        "language": line.language,
                        "voice": line.voice,
                        "out": str(line.out),
                    }
                )
                continue
            operations.append(
                {
                    "op": "tts",
                    "engine": "chatterbox",
                    "text": line.text,
                    "language": line.language,
                    "reference": str(line.reference),
                    "out": str(line.out),
                    "exaggeration": line.exaggeration,
                    "cfg_weight": line.cfg_weight,
                    "temperature": line.temperature,
                    "seed": line.seed,
                }
            )
        out_paths = [line.out for line in lines if line.out is not None]
        results = self._run(operations, out_paths, progress=progress)
        return [
            LineResult(line.index, Path(result.get("out", line.out)))
            for line, result in zip(lines, results, strict=True)
        ]

    def align(
        self,
        paths: list[Path],
        language: str | None,
        model: str,
        *,
        vad: bool = False,
        prompt: str | None = None,
        strict: bool = True,
        progress: Callable[[int], None] | None = None,
        on_result: Callable[[int, Alignment], None] | None = None,
    ) -> list[Alignment]:
        """Transcribe ``paths`` in one worker run (the model is loaded once).

        ``vad`` filters out the non-speech parts first (rushes with room noise); ``prompt``
        is a text that biases the transcriber (for example towards keeping hesitations, or
        spelling the names of a brand). With ``strict=False`` a file that fails comes back
        with ``error`` set instead of raising, so one bad file does not lose the others.
        ``progress`` receives the number of files done. ``on_result(index, alignment)`` is
        called for each file as soon as the worker has finished it (while the next file is
        running), so a caller can store a file's words without waiting for the whole batch;
        every file is reported once, the rest right after the run.
        """

        operations = []
        for path in paths:
            operation: dict = {
                "op": "align",
                "audio": str(path),
                "language": language,
                "model": model,
            }
            if vad:
                operation["vad"] = True
            if prompt:
                operation["prompt"] = prompt
            operations.append(operation)

        def alignment_of(path: Path, result: dict) -> Alignment:
            return Alignment(
                path=path,
                words=[tuple(word) for word in result.get("words", [])],
                text=str(result.get("text", "")),
                language=result.get("language") or None,
                error=None if result.get("ok") else str(result.get("error", "failed")),
            )

        streamed = 0

        def partial(values: list[dict]) -> None:
            nonlocal streamed
            while streamed < min(len(values), len(paths)):
                if on_result is not None and isinstance(values[streamed], dict):
                    on_result(streamed, alignment_of(paths[streamed], values[streamed]))
                streamed += 1

        options: dict = {"progress": progress, "strict": strict}
        if on_result is not None:
            options["on_partial"] = partial
        results = self._run(operations, paths, **options)
        alignments = [
            alignment_of(path, result) for path, result in zip(paths, results, strict=True)
        ]
        if on_result is not None:
            for index in range(streamed, len(alignments)):
                on_result(index, alignments[index])
        return alignments


class FilesEngine:
    """Use user-recorded line files, converting them to the voice contract."""

    def __init__(self, mapping: dict[int, str], *, worker: WorkerEngine | None = None) -> None:
        self.mapping = {int(index): Path(path) for index, path in mapping.items()}
        self.worker = worker or WorkerEngine()
        self.warnings: list[str] = []

    def session(self) -> contextlib.AbstractContextManager:
        return self.worker.session()

    def synthesize(
        self,
        lines: list[LineJob],
        *,
        progress: Callable[[int], None] | None = None,
    ) -> list[LineResult]:
        results = []
        for line in lines:
            source = self.mapping.get(line.index)
            if source is None:
                raise ValueError(f"No recorded voice file for line {line.index}")
            if line.out is None:
                raise ValueError("Recorded voice lines require an out path")
            line.out.parent.mkdir(parents=True, exist_ok=True)
            run(
                [
                    "-i",
                    str(source),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-ar",
                    "48000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(line.out),
                ]
            )
            results.append(LineResult(line.index, line.out))
            if progress is not None:
                progress(len(results))
        return results

    def align(self, paths: list[Path], language: str, model: str) -> list[Alignment]:
        try:
            return self.worker.align(paths, language, model)
        except (VoiceUnavailableError, RuntimeError, OSError) as exc:
            self.warnings.append(
                f"Voice alignment unavailable ({exc}); recorded lines are kept whole "
                "without trimming."
            )
            return [Alignment(path=path) for path in paths]


class FakeEngine:
    """Deterministic speech-shaped audio for tests and dry runs."""

    def __init__(self) -> None:
        self.synthesis_calls = 0
        self.synthesized_lines = 0
        self.align_calls = 0
        self.aligned_paths = 0
        self.align_models: list[str] = []  # the model of every align call, in order
        self.confidence = 1.0  # the word probability reported by ``align``
        self._text: dict[Path, str] = {}
        self._timing: dict[Path, tuple[float, float]] = {}

    def synthesize(
        self,
        lines: list[LineJob],
        *,
        progress: Callable[[int], None] | None = None,
    ) -> list[LineResult]:
        self.synthesis_calls += 1
        self.synthesized_lines += len(lines)
        results = []
        sample_rate = 48_000
        for line in lines:
            if line.out is None:
                raise ValueError("Fake voice lines require an out path")
            line.out.parent.mkdir(parents=True, exist_ok=True)
            tone_duration = 0.06 * len(line.text)
            total = 0.3 + tone_duration + 0.4
            samples = int(round(total * sample_rate))
            tone_start = int(round(0.3 * sample_rate))
            tone_end = tone_start + int(round(tone_duration * sample_rate))
            frames = []
            for sample in range(samples):
                if tone_start <= sample < tone_end:
                    t = sample / sample_rate
                    modulation = 0.65 + 0.35 * math.sin(2 * math.pi * 4.0 * t)
                    value = 0.18 * modulation * math.sin(2 * math.pi * 180.0 * t)
                else:
                    value = 0.0005 * math.sin(2 * math.pi * 37.0 * sample / sample_rate)
                frames.append(struct.pack("<h", int(max(-1.0, min(1.0, value)) * 32767)))
            with wave.open(str(line.out), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(sample_rate)
                output.writeframes(b"".join(frames))
            self._text[line.out.resolve()] = line.text
            self._timing[line.out.resolve()] = (tone_start / sample_rate, tone_end / sample_rate)
            results.append(LineResult(line.index, line.out))
            if progress is not None:
                progress(len(results))
        return results

    def register_audio(self, path: Path, text: str) -> None:
        self._text[Path(path).resolve()] = text

    def align(self, paths: list[Path], language: str, model: str) -> list[Alignment]:
        self.align_calls += 1
        self.aligned_paths += len(paths)
        self.align_models.append(model)
        results = []
        for path in paths:
            resolved = Path(path).resolve()
            text = self._text.get(resolved, "")
            try:
                info = probe(path)
                duration = float(info.audio_duration or 0.0)
            except (FileNotFoundError, RuntimeError, ValueError):
                duration = 0.0
            if resolved in self._timing:
                start, end = self._timing[resolved]
            else:
                # Files get renamed after synthesis: find the speech in the samples,
                # like a real aligner would, instead of relying on the original path.
                start, end = _speech_span(path, duration)
            words = text.split()
            if "[mispronounce]" in text and words:
                words = words[:1] + words[2:] if len(words) > 1 else []
            span = max(0.0, end - start)
            count = len(words)
            aligned = [
                (
                    start + span * index / count,
                    start + span * (index + 1) / count,
                    word,
                    self.confidence,
                )
                for index, word in enumerate(words)
            ]
            transcript = " ".join(words)
            results.append(Alignment(path=path, words=aligned, text=transcript))
        return results


def get_engine(name: str, **kw: object) -> VoiceEngine:
    lowered = name.lower()
    if lowered == "chatterbox":
        return WorkerEngine(
            python=kw.get("python") if isinstance(kw.get("python"), str) else None,
            project_dir=kw.get("project_dir") if isinstance(kw.get("project_dir"), Path) else None,
        )
    if lowered == "kokoro":
        return WorkerEngine(
            python=kw.get("python") if isinstance(kw.get("python"), str) else None,
            project_dir=kw.get("project_dir") if isinstance(kw.get("project_dir"), Path) else None,
            engine="kokoro",
        )
    if lowered == "files":
        mapping = kw.get("mapping")
        if not isinstance(mapping, dict):
            raise ValueError("The files voice engine requires files={index: path}")
        worker = kw.get("worker")
        return FilesEngine(mapping, worker=worker if isinstance(worker, WorkerEngine) else None)
    if lowered == "fake":
        return FakeEngine()
    raise ValueError("Unknown voice engine; expected chatterbox, kokoro, files, or fake")


__all__ = [
    "Alignment",
    "FakeEngine",
    "FilesEngine",
    "LineJob",
    "LineResult",
    "VoiceEngine",
    "SessionFailedError",
    "VoiceUnavailableError",
    "WorkerEngine",
    "get_engine",
]


def _speech_span(path: Path, duration: float, threshold: int = 1500) -> tuple[float, float]:
    """Return the first and last time the fake tone exceeds ``threshold`` (16-bit PCM)."""
    try:
        with wave.open(str(path), "rb") as source:
            rate = source.getframerate()
            width = source.getsampwidth()
            channels = source.getnchannels()
            data = source.readframes(source.getnframes())
    except (wave.Error, OSError, EOFError):
        return 0.0, duration
    if width != 2 or rate <= 0:
        return 0.0, duration
    samples = array("h", data)
    step = max(1, channels)
    loud = [i // step for i, v in enumerate(samples) if abs(v) > threshold]
    if not loud:
        return 0.0, duration
    return loud[0] / rate, (loud[-1] + 1) / rate
