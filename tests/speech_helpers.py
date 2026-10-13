"""Shared helpers of the speech tests: scripted transcriber and synthetic talking clips."""

from __future__ import annotations

import subprocess
from pathlib import Path

from adanvil._bin import ffmpeg_path
from adanvil.speech.model import SpeechWord
from adanvil.voice.engines import Alignment


def words(*items: tuple[float, float, str]) -> list[SpeechWord]:
    return [SpeechWord(start, end, text) for start, end, text in items]


def spoken(text: str, start: float = 0.0, *, rate: float = 0.3, gap: float = 0.05):
    """Timed words for ``text``: ``rate`` seconds a word, ``gap`` between words.

    A token like ``"|0.5"`` inserts a pause of that many seconds.
    """

    timed: list[tuple[float, float, str]] = []
    cursor = start
    for token in text.split():
        if token.startswith("|"):
            cursor += float(token[1:])
            continue
        timed.append((round(cursor, 3), round(cursor + rate, 3), token))
        cursor += rate + gap
    return timed


class ScriptedEngine:
    """A deterministic transcriber: the words of each clip are given, keyed by file stem.

    It has the ``align(paths, language, model)`` shape of the voice engines, like
    ``FakeEngine`` has for text to speech, and records its calls (``calls``) and the options
    it received (``options``, one dict per call: ``prompt``...). ``by_model`` gives other
    scripts for some models (a stronger model hears better): ``{"medium": {stem: words}}``.
    """

    def __init__(
        self,
        scripts: dict[str, list[tuple[float, float, str]]],
        language: str = "fr",
        by_model: dict[str, dict[str, list[tuple[float, float, str]]]] | None = None,
    ):
        self.scripts = scripts
        self.language = language
        self.by_model = by_model or {}
        self.calls: list[tuple[list[str], str | None, str]] = []
        self.options: list[dict] = []

    def align(self, paths, language, model, **options):
        self.calls.append(([Path(path).stem for path in paths], language, model))
        self.options.append(options)
        scripts = self.by_model.get(model, self.scripts)
        results = []
        for path in paths:
            script = scripts.get(Path(path).stem, [])
            results.append(
                Alignment(
                    path=Path(path),
                    words=[(start, end, text, 0.95) for start, end, text in script],
                    text=" ".join(text for _start, _end, text in script),
                    language=self.language,
                )
            )
        return results


def make_clip(
    path: Path,
    duration: float,
    bursts: list[tuple[float, float, float]] | None = None,
    *,
    audio: bool = True,
    size: str = "320x568",
    fps: int = 30,
) -> Path:
    """A small test clip with a speech-like sound: tone ``bursts`` of (start, end, Hz)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path(), "-nostdin", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={fps}:duration={duration}",
    ]  # fmt: skip
    if audio:
        terms = "+".join(
            f"between(t,{start},{end})*sin(2*PI*{freq}*t)" for start, end, freq in bursts or []
        )
        expression = f"0.4*({terms or '0'})"
        command += ["-f", "lavfi", "-i", f"aevalsrc='{expression}':s=48000:d={duration}"]
    command += ["-map", "0:v:0"]
    if audio:
        command += ["-map", "1:a:0", "-c:a", "aac", "-ar", "48000", "-ac", "2"]
    else:
        command += ["-an"]
    command += ["-t", str(duration), "-c:v", "libx264", "-preset", "ultrafast"]
    command += ["-pix_fmt", "yuv420p"]
    command.append(str(path))
    subprocess.run(command, check=True)
    return path


def make_audio(
    path: Path,
    duration: float,
    bursts: list[tuple[float, float, float]] | None = None,
) -> Path:
    """An audio-only file (the format follows the extension) with tone ``bursts`` like make_clip."""

    path.parent.mkdir(parents=True, exist_ok=True)
    terms = "+".join(
        f"between(t,{start},{end})*sin(2*PI*{freq}*t)" for start, end, freq in bursts or []
    )
    codec = {
        ".wav": "pcm_s16le",
        ".mp3": "libmp3lame",
        ".m4a": "aac",
        ".aac": "aac",
    }[path.suffix.casefold()]
    command = [
        ffmpeg_path(), "-nostdin", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"aevalsrc='0.4*({terms or '0'})':s=44100:d={duration}",
        "-c:a", codec, "-ar", "44100", "-ac", "1", str(path),
    ]  # fmt: skip
    subprocess.run(command, check=True)
    return path
