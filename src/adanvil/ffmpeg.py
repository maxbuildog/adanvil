"""The single subprocess gateway for ffmpeg and ffprobe."""

from __future__ import annotations

import json
import math
import re
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from adanvil._bin import ffmpeg_path, ffprobe_path
from adanvil._proc import NO_WINDOW

_TAIL_LINES = 20
# The largest picture, frame rate and duration AdAnvil renders: whatever the inputs compute to,
# no ffmpeg command is started for more than this.
MAX_SIDE = 4096  # pixels, width and height
MAX_FPS = 120
MAX_SECONDS = 600.0
_IMAGE_CODECS = {"png", "mjpeg", "webp", "bmp", "tiff"}
_YAVG_RE = re.compile(r"YAVG=([-+]?(?:\d+(?:\.\d*)?|\.\d+))")
_PTS_TIME_RE = re.compile(r"pts_time:([-+]?(?:\d+(?:\.\d*)?|\.\d+))")


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _command_text(cmd: list[str]) -> str:
    return " ".join(f'"{arg}"' if any(char.isspace() for char in arg) else arg for arg in cmd)


class FFmpegError(RuntimeError):
    """Raised when an ffmpeg or ffprobe subprocess fails."""

    def __init__(self, cmd: list[str], returncode: int, stderr: str) -> None:
        self.cmd = list(cmd)
        self.returncode = returncode
        lines = [line.strip() for line in stderr.splitlines() if line.strip()]
        self.stderr_tail = "\n".join(lines[-_TAIL_LINES:])
        super().__init__(self._message())

    def _message(self) -> str:
        binary = Path(self.cmd[0]).name if self.cmd else "ffmpeg"
        tail = self.stderr_tail.replace("\n", " | ") or "(no stderr output)"
        if self.returncode in {3221225794, 3221225477}:
            return (
                "Windows could not start ffmpeg/ffprobe (0xC0000142). This usually means "
                "the machine is out of memory or desktop heap: close other applications "
                f"and retry. stderr tail: {tail}; command: {_command_text(self.cmd)}"
            )
        return (
            f"{binary} failed with return code {self.returncode}; "
            f"stderr tail: {tail}; command: {_command_text(self.cmd)}"
        )


def _run_binary(
    binary: str,
    args: list[str],
    *,
    timeout: float | None = None,
    capture_stdout: bool,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[bytes | None]:
    command = [binary, *args]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if capture_stdout else None,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            shell=False,
            creationflags=NO_WINDOW,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(command, -1, _decode(exc.stderr)) from exc

    stderr = _decode(completed.stderr)
    completed.stderr = stderr
    if completed.returncode != 0:
        raise FFmpegError(command, completed.returncode, stderr)
    return completed


_INPUT_PROTOCOLS = ["-protocol_whitelist", "file"]


# A media file AdAnvil hands to ffmpeg must be a media file. A playlist or session file named
# like a video (DASH, HLS, SMIL, PLS, SDP, an ffconcat list) is sniffed by ffmpeg and opened as
# such; some builds (a 7.0 static build, measured) then fetch its URLs even under
# ``-protocol_whitelist file``. So the first bytes of every input are read first.
_PLAYLIST_STARTS = (b"#extm3u", b"ffconcat", b"[playlist]", b"v=0", b"#ext-x-")
_PLAYLIST_TAGS = (b"<mpd", b"<smil", b"#ext-x-", b"<playlist")


def _looks_like_playlist(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            head = handle.read(2048)
    except OSError:
        return False  # not a readable file: ffmpeg reports it
    text = head.lstrip(bytes([0xEF, 0xBB, 0xBF, 0x20, 0x09, 0x0D, 0x0A])).lower()
    return text.startswith(_PLAYLIST_STARTS) or any(tag in text for tag in _PLAYLIST_TAGS)


def _refuse_playlist(value: str, cwd: Path | None) -> None:
    path = Path(value)
    if not path.is_absolute() and cwd is not None:
        path = Path(cwd) / path
    if path.is_file() and _looks_like_playlist(path):
        raise ValueError(
            f"{value} is a playlist or stream description, not a media file: refused "
            "(AdAnvil only reads local media files)"
        )


def _local_inputs_only(args: list[str], cwd: Path | None = None) -> list[str]:
    """``args`` with ``-protocol_whitelist file`` before every ``-i``: an input can only be a
    local file, so a crafted path or a playlist disguised as a video (HLS, concat) cannot make
    ffmpeg reach the network or another protocol."""

    safe: list[str] = []
    position = 0
    forced: str | None = None  # the ``-f`` given for the next input, if any
    while position < len(args):
        if args[position] == "-f" and position + 1 < len(args):
            forced = args[position + 1]
        if args[position] == "-i" and position + 1 < len(args):
            # the value after ``-i`` is the input itself, even when it looks like an option
            if forced not in {"concat", "lavfi"}:  # AdAnvil's own lists and generated sources
                _refuse_playlist(args[position + 1], cwd)
            safe.extend([*_INPUT_PROTOCOLS, "-i", args[position + 1]])
            position += 2
            forced = None
            continue
        safe.append(args[position])
        position += 1
    return safe


def run(
    args: list[str],
    *,
    timeout: float | None = None,
    capture_stdout: bool = False,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[bytes | None]:
    """Run ffmpeg with the project-wide safe defaults and error handling.

    ``cwd`` is the working directory of the process: with it, input paths can be given
    relative to it, which keeps a command with hundreds of inputs under the Windows limit
    of 32767 characters. Every input is limited to the ``file`` protocol.
    """
    command_args = ["-nostdin", "-y"]
    if "-loglevel" not in args:
        command_args.extend(["-loglevel", "error"])
    command_args.extend(_local_inputs_only(args, cwd))
    return _run_binary(
        ffmpeg_path(),
        command_args,
        timeout=timeout,
        capture_stdout=capture_stdout,
        cwd=cwd,
    )


_FILTER_SCRIPT_OPTION: dict[str, str] = {}


def filter_script_option() -> str:
    """The ffmpeg option that reads a complex filter graph from a file.

    ``-/filter_complex`` from ffmpeg 7.0 on (``-filter_complex_script`` was removed), and
    ``-filter_complex_script`` before. A build whose version cannot be read (a git build) is
    taken for a recent one.
    """

    binary = ffmpeg_path()
    if binary not in _FILTER_SCRIPT_OPTION:
        try:
            banner = _decode(_run_binary(binary, ["-version"], capture_stdout=True).stdout)
        except (FFmpegError, OSError):
            banner = ""
        found = re.search(r"version\s+n?(\d+)\.", banner)
        old = found is not None and int(found.group(1)) < 7
        _FILTER_SCRIPT_OPTION[binary] = "-filter_complex_script" if old else "-/filter_complex"
    return _FILTER_SCRIPT_OPTION[binary]


def run_probe(args: list[str]) -> str:
    """Run ffprobe and return its decoded stdout, raising on probe failure. The input is
    limited to the ``file`` protocol."""
    for value in args:
        if not value.startswith("-"):
            _refuse_playlist(value, None)
    completed = _run_binary(
        ffprobe_path(), ["-v", "error", *_INPUT_PROTOCOLS, *args], capture_stdout=True
    )
    stdout = completed.stdout
    if isinstance(stdout, bytes):
        return stdout.decode("utf-8", errors="replace")
    return stdout or ""


class MediaInfo(BaseModel):
    """Normalised stream and media metadata returned by :func:`probe`."""

    model_config = ConfigDict(frozen=True)

    path: Path
    kind: Literal["video", "image", "audio"]
    width: int | None
    height: int | None
    stored_width: int | None
    stored_height: int | None
    rotation: int
    duration: float | None
    fps: float | None
    nb_frames: int | None
    pix_fmt: str | None
    color_transfer: str | None
    color_primaries: str | None
    color_space: str | None
    has_audio: bool
    audio_duration: float | None
    audio_sample_rate: int | None
    audio_channels: int | None

    @property
    def is_hdr(self) -> bool:
        """Whether the video transfer characteristic is HLG or PQ."""
        return self.color_transfer in {"arib-std-b67", "smpte2084"}

    @property
    def is_portrait(self) -> bool:
        """Whether display dimensions are taller than they are wide."""
        return self.width is not None and self.height is not None and self.height > self.width

    @property
    def aspect(self) -> float | None:
        """Return the display aspect ratio, or ``None`` without video dimensions."""
        if self.width is None or self.height in (None, 0):
            return None
        return self.width / self.height


def _number(value: Any) -> float | None:
    if value in (None, "", "N/A", "NA"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _fraction(value: Any) -> float | None:
    if value in (None, "", "N/A", "NA", "0/0"):
        return None
    try:
        result = float(Fraction(str(value)))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return result


def _int(value: Any) -> int | None:
    if value in (None, "", "N/A", "NA"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _stream_duration(stream: dict[str, Any], format_data: dict[str, Any]) -> float | None:
    return _number(stream.get("duration")) or _number(format_data.get("duration"))


def _rotation(stream: dict[str, Any]) -> int:
    value: Any = None
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            value = side_data["rotation"]
            break
    if value is None:
        value = stream.get("tags", {}).get("rotate")
    parsed = _number(value)
    if parsed is None:
        return 0
    return int(round(parsed)) % 360


_COVER_SUFFIXES = {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus"}


def _is_cover_art(stream: dict[str, Any], audio_streams: list[dict[str, Any]], path: Path) -> bool:
    """Whether a video stream is the picture embedded in an audio file (a cover), not a video.

    True for a stream flagged ``attached_pic``, and for the single still mjpeg/png frame of a
    file that has audio and an audio extension (some encoders do not set the flag).
    """

    if _int((stream.get("disposition") or {}).get("attached_pic")) == 1:
        return True
    if not audio_streams or path.suffix.lower() not in _COVER_SUFFIXES:
        return False
    frames = _int(stream.get("nb_frames"))
    return stream.get("codec_name") in {"mjpeg", "png"} and (frames is None or frames <= 1)


def probe(path: str | Path) -> MediaInfo:
    """Probe one media file and return normalised metadata."""
    media_path = Path(path)
    if not media_path.exists():
        raise FileNotFoundError(media_path)

    data = json.loads(run_probe(["-show_streams", "-show_format", "-of", "json", str(media_path)]))
    streams = data.get("streams", [])
    format_data = data.get("format", {})
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    video_streams = [
        stream
        for stream in streams
        if stream.get("codec_type") == "video"
        and not _is_cover_art(stream, audio_streams, media_path)
    ]
    video = video_streams[0] if video_streams else {}
    audio = audio_streams[0] if audio_streams else {}

    stored_width = _int(video.get("width")) if video else None
    stored_height = _int(video.get("height")) if video else None
    rotation = _rotation(video) if video else 0
    if rotation in {90, 270}:
        width, height = stored_height, stored_width
    else:
        width, height = stored_width, stored_height

    format_names = set(str(format_data.get("format_name", "")).split(","))
    image_stream = (
        len(video_streams) == 1
        and video.get("codec_name") in _IMAGE_CODECS
        and (
            video.get("duration") is None
            or any(name.startswith("image2") for name in format_names)
        )
    )
    if not video_streams:
        kind: Literal["video", "image", "audio"] = "audio"
    elif image_stream:
        kind = "image"
    else:
        kind = "video"

    video_duration = _stream_duration(video, format_data) if video else None
    duration = None if kind == "image" else (video_duration or _number(format_data.get("duration")))
    fps = None
    if video:
        fps = _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate"))
    audio_duration = _stream_duration(audio, format_data) if audio else None

    return MediaInfo(
        path=media_path,
        kind=kind,
        width=width,
        height=height,
        stored_width=stored_width,
        stored_height=stored_height,
        rotation=rotation,
        duration=duration,
        fps=fps,
        nb_frames=_int(video.get("nb_frames")) if video else None,
        pix_fmt=video.get("pix_fmt") if video else None,
        color_transfer=video.get("color_transfer") if video else None,
        color_primaries=video.get("color_primaries") if video else None,
        color_space=video.get("color_space") if video else None,
        has_audio=bool(audio_streams),
        audio_duration=audio_duration,
        audio_sample_rate=_int(audio.get("sample_rate")) if audio else None,
        audio_channels=_int(audio.get("channels")) if audio else None,
    )


def measure_yavg(path: str | Path, *, start: float = 0.0, duration: float | None = None) -> float:
    """Measure mean luma over a downscaled four-frame-per-second pass."""
    args = [
        "-loglevel",
        "info",
        "-ss",
        str(start),
        "-i",
        str(path),
    ]
    if duration is not None:
        args.extend(["-t", str(duration)])
    args.extend(
        [
            "-vf",
            "fps=4,scale=360:-2,format=yuv420p,signalstats,metadata=print:key=lavfi.signalstats.YAVG",
            "-an",
            "-f",
            "null",
            "-",
        ]
    )
    result = run(args)
    values = [float(match.group(1)) for match in _YAVG_RE.finditer(result.stderr or "")]
    return sum(values) / len(values) if values else 128.0


def measure_video(
    path: str | Path, *, threshold: float = 0.3, fps: int = 4, width: int = 360
) -> tuple[list[float], list[float]]:
    """Detect cuts and collect per-frame luma in one downscaled decode pass.

    The scene branch and luma branch share the same decoded, downscaled
    stream. Callers can assign the uniform samples to any shot without
    starting another full-resolution ffmpeg process.
    """

    cuts, yavg, _ = _measure_video(path, threshold=threshold, fps=fps, width=width)
    return cuts, yavg


def measure_video_keyframes(
    path: str | Path, *, threshold: float = 0.3, fps: int = 4, width: int = 360
) -> tuple[list[float], list[float], int]:
    """Like :func:`measure_video`, but decode only the keyframes (``-skip_frame nokey``).

    Meant for sources above 1920 px, where decoding every frame dominates the cost.
    Cuts are seen at the first keyframe after them. The third value is the number of
    keyframes decoded: callers fall back to :func:`measure_video` when it is below 3.
    """

    return _measure_video(path, threshold=threshold, fps=fps, width=width, keyframes_only=True)


def _measure_video(
    path: str | Path,
    *,
    threshold: float,
    fps: int,
    width: int,
    keyframes_only: bool = False,
) -> tuple[list[float], list[float], int]:
    if fps <= 0 or width <= 0:
        raise ValueError("fps and width must be positive")
    stream = f"fps={fps},scale={width}:-2,format=yuv420p"
    head = "[0:v]"
    scene_tag = "showinfo"
    if keyframes_only:
        # Count the real decoded frames before ``fps`` duplicates them to a uniform rate.
        head = "[0:v]showinfo@kf,"
        scene_tag = "showinfo@cut"
    args = ["-loglevel", "info"]
    if keyframes_only:
        args.extend(["-skip_frame", "nokey"])
    args.extend(
        [
            "-i",
            str(path),
            "-filter_complex",
            (
                f"{head}{stream},split=2[scene][luma];"
                f"[scene]select='gt(scene,{threshold})',{scene_tag},nullsink;"
                "[luma]signalstats,metadata=print:key=lavfi.signalstats.YAVG[lumas]"
            ),
            "-map",
            "[lumas]",
            "-an",
            "-f",
            "null",
            "-",
        ]
    )
    result = run(args)
    stderr = result.stderr or ""
    cuts = sorted(
        {
            round(float(match.group(1)), 3)
            for line in stderr.splitlines()
            if scene_tag in line
            for match in _PTS_TIME_RE.finditer(line)
        }
    )
    yavg = [float(match.group(1)) for match in _YAVG_RE.finditer(stderr)]
    decoded = sum(1 for line in stderr.splitlines() if "showinfo@kf" in line and "pts_time" in line)
    return cuts, yavg, decoded


def detect_cuts(path: str | Path, *, threshold: float = 0.3) -> list[float]:
    """Return scene-change timestamps detected by ffmpeg's scene filter."""
    result = run(
        [
            "-loglevel",
            "info",
            "-i",
            str(path),
            "-vf",
            f"select='gt(scene,{threshold})',showinfo",
            "-an",
            "-f",
            "null",
            "-",
        ]
    )
    timestamps = [float(match.group(1)) for match in _PTS_TIME_RE.finditer(result.stderr or "")]
    return sorted({round(timestamp, 3) for timestamp in timestamps})


FRAME_TOLERANCE = 0.001  # seconds


def frame_on_screen(t: float, fps: float | None) -> tuple[int, float]:
    """The frame displayed at time ``t`` of a video at ``fps``: its number ``floor(t * fps)``
    (a ``t`` within 1 ms before a frame boundary counts as that frame) and its exact start time.
    Without a frame rate, ``(round(t), t)`` is not guessed: the number is ``-1`` and ``t`` is
    returned unchanged."""

    if not fps or fps <= 0:
        return -1, t
    # a time within 1 ms before a frame boundary is that frame: 10.9333 s (printed to four
    # decimals) at 30 fps is frame 328, not 327
    number = max(0, math.floor(t * fps + FRAME_TOLERANCE * fps + 1e-6))
    return number, number / fps


def extract_frame(
    path: str | Path,
    t: float,
    out: Path,
    *,
    width: int | None = None,
    keyframe: bool = False,
    threads: int | None = None,
    on_screen: bool = False,
) -> Path:
    """Extract one PNG frame, seeking before input and optionally scaling its width.

    ``keyframe`` decodes only keyframes and takes the one at or before ``t`` (cheap on
    large sources, but the frame can lie earlier than ``t``); ``threads`` sets the
    decoder's thread count. ``on_screen`` returns the frame displayed at ``t``
    (``floor(t * fps)``), where a plain seek returns the first frame at or after ``t``,
    which is the next shot when ``t`` falls inside the last frame before a cut.
    """
    if on_screen and not keyframe:
        try:
            fps = probe(path).fps
        except Exception:
            fps = None
        number, _start = frame_on_screen(t, fps)
        if number >= 0 and fps:
            # a quarter frame before the frame start: the first frame at or after that is it
            t = max(0.0, (number - 0.25) / fps)
    args: list[str] = []
    if keyframe:
        args.extend(["-skip_frame", "nokey", "-noaccurate_seek"])
    if threads is not None:
        args.extend(["-threads", str(threads)])
    args.extend(["-ss", str(t), "-i", str(path)])
    if width is not None:
        args.extend(["-vf", f"scale={width}:-2"])
    args.extend(["-frames:v", "1", "-an", "-c:v", "png", str(out)])
    # ffmpeg opens the output name as it is: drop a link planted there instead of writing
    # through it
    Path(out).unlink(missing_ok=True)
    run(args)
    return out
