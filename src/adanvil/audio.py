"""Audio building blocks for storyboard renders and audio quality control."""

from __future__ import annotations

import math
import os
import re
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from adanvil.ffmpeg import probe, run

_LUFS_RE = re.compile(r"\bI:\s*(-?(?:\d+(?:\.\d+)?|inf))\s+LUFS", re.IGNORECASE)

# Music beds are brought to the level of a typical mastered track before
# ``music_volume`` applies, so 0.35 sits about 5 dB under a -15 LUFS voice
# whether the track is a loud beat or a soft piano.
MUSIC_REFERENCE_LUFS = -11.0
_MUSIC_GAIN_RANGE_DB = (-12.0, 15.0)

_RMS_RE = re.compile(
    r"lavfi\.astats\.Overall\.RMS_level=(?P<value>[^\r\n]+)"
)


def _positive_duration(value: float, name: str = "duration") -> float:
    """Validate and return a duration accepted by ffmpeg filters."""
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero")
    return value


def _audio_or_error(src: Path) -> tuple[float, int]:
    """Return source audio duration and sample rate, or explain the failure."""
    info = probe(src)
    if not info.has_audio:
        raise ValueError(f"No audio stream found in {src}")
    if info.audio_duration is None or info.audio_duration <= 0:
        raise ValueError(f"Audio stream in {src} has no usable duration")
    return info.audio_duration, info.audio_sample_rate or 48_000


def music_bed(
    src: Path,
    duration: float,
    out: Path,
    *,
    start: float = 0.0,
    volume: float = 0.8,
    fade_in: float = 0.2,
    fade_out: float = 1.0,
    crossfade: float = 1.5,
) -> Path:
    """Build an exact-length stereo AAC bed from the first audio stream of ``src``.

    Short sources are repeated as explicit filter inputs and joined with
    ``acrossfade``.  The output is always normalised to the project's audio
    contract (48 kHz, stereo, AAC at 160 kbit/s).
    """
    src = Path(src)
    out = Path(out)
    duration = _positive_duration(duration)
    start = float(start)
    if not math.isfinite(start) or start < 0:
        raise ValueError("start must be a finite number greater than or equal to zero")
    volume = float(volume)
    if not math.isfinite(volume) or volume < 0:
        raise ValueError("volume must be a finite number greater than or equal to zero")

    source_duration, _ = _audio_or_error(src)
    piece_duration = source_duration - start
    if piece_duration <= 0:
        raise ValueError(f"No audio remains in {src} at start={start}")

    if piece_duration >= duration:
        copies = 1
        join_duration = 0.0
    else:
        # acrossfade consumes part of each join.  For a short source, keeping
        # the overlap below 40% avoids a repeated piece swallowing itself.
        join_duration = min(max(float(crossfade), 0.0), piece_duration * 0.4)
        progress_per_copy = piece_duration - join_duration
        # Leave one extra piece for codec priming/rounding: the probed
        # duration can be a few AAC samples longer than decoded audio.
        copies = max(2, math.ceil((duration - piece_duration) / progress_per_copy) + 2)

    fade_in = min(max(float(fade_in), 0.0), duration)
    fade_out = min(max(float(fade_out), 0.0), duration)

    args: list[str] = []
    for _ in range(copies):
        # Seeking before each input makes every repeated piece start at the
        # requested source position, including when src is a video.
        args.extend(["-ss", str(start), "-i", str(src)])

    filters: list[str] = []
    for index in range(copies):
        filters.append(
            f"[{index}:a:0]aformat=sample_rates=48000:channel_layouts=stereo,"
            f"asetpts=PTS-STARTPTS[a{index}]"
        )

    current = "a0"
    for index in range(1, copies):
        next_label = f"joined{index}"
        if join_duration:
            filters.append(
                f"[{current}][a{index}]acrossfade=d={join_duration}:c1=tri:c2=tri"
                f"[{next_label}]"
            )
        else:
            filters.append(f"[{current}][a{index}]concat=n=2:v=0:a=1[{next_label}]")
        current = next_label

    fade_start = max(0.0, duration - fade_out)
    filters.append(
        f"[{current}]atrim=start=0:end={duration},asetpts=PTS-STARTPTS,"
        f"afade=t=in:st=0:d={fade_in},"
        f"afade=t=out:st={fade_start}:d={fade_out},volume={volume}[aout]"
    )

    args.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[aout]",
            "-vn",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-t",
            str(duration),
            str(out),
        ]
    )
    run(args)
    return out


def reference_bed(video: Path, duration: float, out: Path, **kw: object) -> Path:
    """Keep the soundtrack of a reference ad as an exact-length audio bed."""
    return music_bed(video, duration, out, **kw)


def silence(duration: float, out: Path) -> Path:
    """Create an exact-length silent stereo AAC track."""
    duration = _positive_duration(duration)
    out = Path(out)
    run(
        [
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-af",
            f"atrim=start=0:end={duration},asetpts=PTS-STARTPTS",
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-t",
            str(duration),
            str(out),
        ]
    )
    return out


def integrated_loudness(src: Path) -> float | None:
    """Return the integrated loudness of the first audio stream in LUFS, or None if silent."""
    stderr = str(
        run(
            [
                "-loglevel",
                "info",
                "-i",
                str(src),
                "-map",
                "0:a:0",
                "-af",
                "ebur128",
                "-vn",
                "-f",
                "null",
                "-",
            ]
        ).stderr
        or ""
    )
    values = _LUFS_RE.findall(stderr)
    if not values or values[-1].lower() in {"inf", "-inf"}:
        return None
    value = float(values[-1])
    return value if value > -70.0 else None


def music_gain(src: Path, music_volume: float) -> float:
    """Linear gain that brings ``src`` to the music reference level, times ``music_volume``."""
    loudness = integrated_loudness(src)
    if loudness is None:
        return float(music_volume)
    low, high = _MUSIC_GAIN_RANGE_DB
    gain_db = min(max(MUSIC_REFERENCE_LUFS - loudness, low), high)
    return float(music_volume) * 10 ** (gain_db / 20)


def loudness_normalize(
    src: Path, out: Path, *, i: float = -15.0, tp: float = -1.5
) -> Path:
    """Run a standalone single-pass ``loudnorm`` operation on an audio stream."""
    src = Path(src)
    out = Path(out)
    _audio_or_error(src)
    run(
        [
            "-i",
            str(src),
            "-map",
            "0:a:0",
            "-vn",
            "-af",
            f"loudnorm=I={float(i)}:TP={float(tp)}:print_format=none",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(out),
        ]
    )
    return out


def voice_music_mix(
    voice: Path,
    duration: float,
    out: Path,
    *,
    music: Path | None = None,
    music_start: float = 0.0,
    music_volume: float = 0.35,
    voice_loudness: float = -15.0,
    music_fade_in: float = 0.8,
    music_fade_out: float = 1.5,
    voice_offset: float = 0.0,
    normalize_music: bool = True,
) -> Path:
    """Mix a normalised voice track over an exact-length music bed.

    Voice normalisation is deliberately a separate ffmpeg pass.  Applying
    ``loudnorm`` in the final graph can consume the last seconds of speech.
    With ``normalize_music`` the bed is first brought to
    ``MUSIC_REFERENCE_LUFS`` by a plain gain (no dynamics change), so
    ``music_volume`` means the same level for every track.
    """
    voice = Path(voice)
    out = Path(out)
    duration = _positive_duration(duration)
    voice_offset = float(voice_offset)
    if not math.isfinite(voice_offset) or voice_offset < 0:
        raise ValueError("voice_offset must be a finite number greater than or equal to zero")
    music_volume = float(music_volume)
    if not math.isfinite(music_volume) or music_volume < 0:
        raise ValueError("music_volume must be a finite number greater than or equal to zero")
    voice_duration, _ = _audio_or_error(voice)
    if duration < voice_offset + voice_duration - 0.05:
        raise ValueError(
            f"duration {duration:g}s would cut the voice at "
            f"{voice_offset + voice_duration:g}s"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    normalized = out.with_name(f".{out.stem}.{token}.voice.m4a")
    bed = out.with_name(f".{out.stem}.{token}.music.m4a") if music is not None else None
    temporary_output = out.with_name(f".{out.stem}.{token}.mix.m4a")
    try:
        loudness_normalize(voice, normalized, i=voice_loudness, tp=-1.5)
        if bed is not None:
            music_bed(
                Path(music),
                duration,
                bed,
                start=music_start,
                volume=1.0,
                fade_in=music_fade_in,
                fade_out=music_fade_out,
            )
            if normalize_music:
                music_volume = music_gain(bed, music_volume)

        delay_ms = max(0, round(voice_offset * 1000))
        voice_chain = (
            f"[0:a]aformat=sample_rates=48000:channel_layouts=stereo,"
            f"adelay=delays={delay_ms}:all=1,apad=whole_dur={duration:.9f}[voice]"
        )
        if bed is None:
            graph = (
                f"{voice_chain};[voice]atrim=start=0:end={duration:.9f},"
                "asetpts=PTS-STARTPTS[aout]"
            )
            inputs = ["-i", str(normalized)]
        else:
            graph = (
                f"{voice_chain};[1:a]volume={music_volume:.9f}[music];"
                "[voice][music]amix=inputs=2:duration=longest:normalize=0,"
                f"atrim=start=0:end={duration:.9f},asetpts=PTS-STARTPTS[aout]"
            )
            inputs = ["-i", str(normalized), "-i", str(bed)]

        run(
            [
                *inputs,
                "-filter_complex",
                graph,
                "-map",
                "[aout]",
                "-vn",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-t",
                str(duration),
                str(temporary_output),
            ]
        )
        os.replace(temporary_output, out)
    finally:
        for temporary in (normalized, bed, temporary_output):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return out


CLIP_CROSSFADE = 0.010  # seconds of crossfade at each cut of the clip-sound bed


@dataclass(frozen=True)
class AudioPiece:
    """One stretch of the clip-sound bed: ``length`` seconds of ``source`` from ``start``.

    ``duration`` is the time the piece fills on the timeline (the segment's duration): a
    ``speed`` other than 1 stretches the sound to fit, and a source that is shorter, absent
    (``None``) or without audio is completed with silence. ``length`` is read in source time.
    """

    source: Path | None
    start: float
    length: float
    duration: float
    speed: float = 1.0


def _tempo(speed: float) -> str:
    """atempo chain for ``speed`` (one atempo takes 0.5 to 100)."""

    if abs(speed - 1.0) < 1e-9:
        return ""
    steps: list[float] = []
    remaining = speed
    while remaining < 0.5:
        steps.append(0.5)
        remaining /= 0.5
    steps.append(remaining)
    return ",".join(f"atempo={step:.9f}" for step in steps)


def clips_bed(pieces: list[AudioPiece], out: Path, *, crossfade: float = CLIP_CROSSFADE) -> Path:
    """Join the audio of clip segments into one stereo WAV exactly as long as the video.

    Every piece is as long as its segment, so the cuts of the sound fall on the cuts of the
    picture. Each cut is a short crossfade: a piece reads ``crossfade`` seconds past its end
    (the true continuation of its sound) and that tail fades out while the next piece fades
    in, so a speech that goes on across the cut stays whole and nothing clicks. Total length
    is the sum of the piece durations, to the sample.
    """

    out = Path(out)
    if not pieces:
        raise ValueError("clips_bed needs at least one piece")
    crossfade = float(crossfade)
    inputs: list[str] = []
    graph: list[str] = []
    input_count = 0
    total = 0.0
    for number, piece in enumerate(pieces):
        duration = _positive_duration(piece.duration, "piece duration")
        total += duration
        extra = crossfade if number < len(pieces) - 1 else 0.0
        wanted = duration + extra
        usable = False
        if piece.source is not None:
            info = probe(piece.source)
            usable = bool(
                info.has_audio
                and info.audio_duration is not None
                and piece.start < info.audio_duration - 0.001
            )
        if usable:
            read = piece.length + extra * piece.speed  # source time, with the crossfade tail
            inputs.extend(
                ["-ss", f"{piece.start:.6f}", "-t", f"{read:.6f}", "-i", str(piece.source)]
            )
            chain = [
                f"[{input_count}:a:0]aformat=sample_rates=48000:channel_layouts=stereo",
                "asetpts=PTS-STARTPTS",
            ]
            tempo = _tempo(piece.speed)
            if tempo:
                chain.append(tempo)
            chain.append(f"apad=whole_dur={wanted:.9f}")
            chain.append(f"atrim=end={wanted:.9f}")
            chain.append("asetpts=PTS-STARTPTS")
            graph.append(",".join(chain) + f"[p{number}]")
            input_count += 1
        else:
            graph.append(
                f"anullsrc=r=48000:cl=stereo,atrim=end={wanted:.9f},asetpts=PTS-STARTPTS"
                f"[p{number}]"
            )

    current = "p0"
    for number in range(1, len(pieces)):
        label = f"j{number}"
        graph.append(
            f"[{current}][p{number}]acrossfade=d={crossfade:.6f}:c1=tri:c2=tri[{label}]"
        )
        current = label
    graph.append(f"[{current}]atrim=end={total:.9f},asetpts=PTS-STARTPTS[aout]")

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.stem}.{uuid.uuid4().hex}{out.suffix}")
    try:
        run(
            [
                *inputs,
                "-filter_complex",
                ";".join(graph),
                "-map",
                "[aout]",
                "-c:a",
                "pcm_s16le",
                "-ar",
                "48000",
                "-ac",
                "2",
                str(temporary),
            ]
        )
        os.replace(temporary, out)
    finally:
        temporary.unlink(missing_ok=True)
    return out


def rms_per_second(path: Path) -> list[float]:
    """Return one RMS level in dBFS for each one-second audio window."""
    path = Path(path)
    info = probe(path)
    if not info.has_audio or info.audio_duration is None:
        return []
    sample_rate = info.audio_sample_rate or 48_000
    result = run(
        [
            "-loglevel",
            "info",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-af",
            (
                f"asetnsamples=n={sample_rate}:p=0,"
                f"astats=metadata=1:reset={sample_rate},"
                "ametadata=print:key=lavfi.astats.Overall.RMS_level"
            ),
            "-vn",
            "-f",
            "null",
            "-",
        ]
    )
    values: list[float] = []
    for match in _RMS_RE.finditer(result.stderr or ""):
        raw = match.group("value").strip()
        if raw.lower() in {"-inf", "-infinity", "nan"}:
            values.append(-120.0)
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def audio_duration(path: Path) -> float | None:
    """Return the first audio stream duration, if the file has one."""
    info = probe(Path(path))
    return info.audio_duration if info.has_audio else None


# ---------------------------------------------------------------------------
# Music profile: how energetic is a track, to match it with a reference.
# ---------------------------------------------------------------------------

AUDIO_SUFFIXES = (".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus")

# Thresholds of the one-word feel, documented in the editing guide.  ``onset_rate`` is the
# number of onsets (drum hits, plucks, chord changes) per second.
FEEL_THRESHOLDS = {
    "calm": "pulse < 2.0 (no clear beat: piano, pad; tempo is then null)",
    "energetic": "pulse >= 2.2 and tempo >= 110 BPM",
    "medium": "everything else (a clear beat under 110 BPM, relaxed vlog or lo-fi)",
}

_PROFILE_RATE = 22_050
_PROFILE_MAX_SECONDS = 120.0
_PROFILE_FFT = 1024
_PROFILE_HOP = 256
_ONSET_FLOOR = 1.0
_LRA_RE = re.compile(r"\bLRA:\s*(-?\d+(?:\.\d+)?)\s+LU", re.IGNORECASE)


def _onset_envelope(samples):
    """Positive spectral flux of a mono signal, one value per hop."""
    if len(samples) < _PROFILE_FFT * 2:
        return np.zeros(0)
    window = np.hanning(_PROFILE_FFT).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(samples, _PROFILE_FFT)[::_PROFILE_HOP]
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) / (_PROFILE_FFT / 4)
    compressed = np.log1p(100.0 * spectrum)
    flux = np.maximum(np.diff(compressed, axis=0), 0.0).sum(axis=1)
    return np.concatenate([[0.0], flux])


def _estimate_tempo(envelope, frames_per_second: float) -> float | None:
    """Tempo in BPM from the autocorrelation of the onset envelope (60 to 200 BPM)."""
    if len(envelope) < frames_per_second * 4 or float(envelope.max()) < _ONSET_FLOOR:
        return None
    centred = envelope - envelope.mean()
    size = 1 << (2 * len(centred) - 1).bit_length()
    spectrum = np.fft.rfft(centred, size)
    autocorrelation = np.fft.irfft(spectrum * np.conj(spectrum))[: len(centred)]
    if autocorrelation[0] <= 0:
        return None
    autocorrelation = autocorrelation / autocorrelation[0]
    low = int(frames_per_second * 60.0 / 200.0)
    high = min(int(frames_per_second * 60.0 / 60.0), len(autocorrelation) - 2)
    if high <= low + 1:
        return None
    lags = np.arange(low, high + 1)
    bpm = 60.0 * frames_per_second / lags
    # Tempo prior: music clusters around 120 BPM, which settles octave ambiguities.
    prior = np.exp(-0.5 * np.log2(bpm / 120.0) ** 2)
    best = int(lags[int(np.argmax(autocorrelation[low : high + 1] * prior))])
    left, centre, right = autocorrelation[best - 1 : best + 2]
    curvature = left - 2 * centre + right
    shift = 0.0 if curvature == 0 else 0.5 * (left - right) / curvature
    return float(60.0 * frames_per_second / (best + max(-1.0, min(1.0, shift))))


def _onset_rate(envelope, frames_per_second: float, duration: float) -> float:
    """Onsets per second: local maxima of the envelope that stand out and clear the floor."""
    if len(envelope) < 3 or duration <= 0:
        return 0.0
    threshold = max(_ONSET_FLOOR, float(envelope.mean() + envelope.std()))
    radius = max(1, int(0.1 * frames_per_second))
    count = 0
    last = -radius - 1
    for index in np.flatnonzero(envelope > threshold):
        window = envelope[max(0, index - radius) : index + radius + 1]
        if envelope[index] >= window.max() and index - last > radius:
            count += 1
            last = int(index)
    return count / duration


def _pulse(envelope) -> float:
    """How much the attacks stand out: 95th percentile of the onset envelope over its median.

    Measured on real tracks: a piano or a pad stays under 2, a track with a drum beat
    is between 3 and 7. Onset counts alone cannot tell them apart (a piano has many notes).
    """
    if len(envelope) < 3:
        return 0.0
    peak = float(np.percentile(envelope, 95))
    if peak <= _ONSET_FLOOR:
        # Sparse attacks over silence (under 5 % of the frames) are the clearest beat
        # there is; nothing above the floor at all is a steady tone or silence.
        return 20.0 if float(envelope.max()) > _ONSET_FLOOR * 10 else 0.0
    # Clicks over silence have a near-zero median: cap the ratio instead of dividing by ~0.
    return min(20.0, peak / max(float(np.median(envelope)), _ONSET_FLOOR))


_CALM_PULSE = 2.0
_ENERGETIC_PULSE = 2.2
_ENERGETIC_BPM = 110.0


def _feel(tempo: float | None, pulse: float) -> str:
    """calm: no clear beat; energetic: a clear beat at 110 BPM or more; medium otherwise."""
    if pulse < _CALM_PULSE:
        return "calm"
    if (tempo or 0.0) >= _ENERGETIC_BPM and pulse >= _ENERGETIC_PULSE:
        return "energetic"
    return "medium"


def profile_audio(src: Path) -> dict[str, object]:
    """Duration, loudness, tempo and energy of the first audio stream of ``src``.

    ``src`` may be a video: the profile is then the one of its soundtrack (voice included).
    The tempo and the onset rate come from the first two minutes.
    """
    src = Path(src)
    duration, _ = _audio_or_error(src)
    decoded = run(
        [
            "-t",
            str(_PROFILE_MAX_SECONDS),
            "-i",
            str(src),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(_PROFILE_RATE),
            "-f",
            "f32le",
            "-",
        ],
        capture_stdout=True,
    )
    samples = np.frombuffer(decoded.stdout or b"", dtype="<f4")
    analysed = len(samples) / _PROFILE_RATE
    frames_per_second = _PROFILE_RATE / _PROFILE_HOP
    envelope = _onset_envelope(samples)
    tempo = _estimate_tempo(envelope, frames_per_second)
    onset_rate = _onset_rate(envelope, frames_per_second, analysed)
    pulse = _pulse(envelope)
    if pulse < _CALM_PULSE:
        tempo = None  # no steady beat: an autocorrelation peak would be noise
    stats = str(
        run(
            [
                "-loglevel",
                "info",
                "-i",
                str(src),
                "-map",
                "0:a:0",
                "-af",
                "ebur128",
                "-vn",
                "-f",
                "null",
                "-",
            ]
        ).stderr
        or ""
    )
    integrated = _LUFS_RE.findall(stats)
    loudness = None
    if integrated and integrated[-1].lower() not in {"inf", "-inf"}:
        loudness = float(integrated[-1])
        loudness = loudness if loudness > -70.0 else None
    ranges = _LRA_RE.findall(stats)
    return {
        "path": src.as_posix(),
        "duration": round(duration, 2),
        "loudness_lufs": None if loudness is None else round(loudness, 1),
        "tempo_bpm": None if tempo is None else round(tempo, 1),
        "energy": {
            "onset_rate_per_s": round(onset_rate, 2),
            "pulse": round(pulse, 2),
            "loudness_range_lu": round(float(ranges[-1]), 1) if ranges else None,
        },
        "feel": _feel(tempo, pulse),
    }


def music_profile(root: Path, path: str | None = None) -> dict[str, object]:
    """Profile every audio file of ``music/``, or one file (a video's soundtrack works too)."""
    from adanvil.project import resolve_in_project

    root = Path(root)
    if path is not None:
        target = resolve_in_project(root, path)
        if not target.is_file():
            raise FileNotFoundError(f"file does not exist: {path}")
        files = [target]
    else:
        folder = root / "music"
        files = (
            sorted(
                item
                for item in folder.rglob("*")
                if item.is_file() and item.suffix.lower() in AUDIO_SUFFIXES
            )
            if folder.is_dir()
            else []
        )
    tracks: list[dict[str, object]] = []
    for item in files:
        relative = item.relative_to(root).as_posix()
        try:
            profile = profile_audio(item)
        except Exception as exc:  # one unreadable file must not hide the others
            tracks.append({"path": relative, "error": str(exc)})
            continue
        profile["path"] = relative
        tracks.append(profile)
    return {"tracks": tracks, "feel_thresholds": FEEL_THRESHOLDS}


# ---------------------------------------------------------------------------
# Sound effects over the bed, and the peak of a file.
# ---------------------------------------------------------------------------

SFX_CEILING_DB = -1.5  # the sample peak of a mix with sounds; the AAC encode stays under -1 dB
_PEAK_RE = re.compile(r"\bPeak:\s*(-?(?:\d+(?:\.\d+)?|inf))\s+dBFS", re.IGNORECASE)


def true_peak(src: Path) -> float | None:
    """True peak of the first audio stream in dBFS (``None`` for digital silence)."""
    stderr = str(
        run(
            [
                "-loglevel",
                "info",
                "-i",
                str(src),
                "-map",
                "0:a:0",
                "-af",
                "ebur128=peak=true",
                "-vn",
                "-f",
                "null",
                "-",
            ]
        ).stderr
        or ""
    )
    values = _PEAK_RE.findall(stderr)
    if not values or values[-1].lower() in {"inf", "-inf"}:
        return None
    return float(values[-1])


@dataclass(frozen=True)
class SfxHit:
    """One sound effect on the final timeline: ``path`` starts at ``at`` seconds, at ``volume``."""

    path: Path
    at: float
    volume: float = 1.0


_LIMIT_BLOCK = 48  # samples (1 ms): the limiter works on the minimum gain of each block
_LIMIT_HOLD = 10  # blocks of look-ahead and hold around a peak (10 ms each side)


def peak_limit(samples: np.ndarray, ceiling: float) -> np.ndarray:
    """Bring the sample peak of ``samples`` (frames x channels) under ``ceiling`` (linear).

    A look-ahead limiter with no latency: the gain needed at every sample is reduced to what
    the loudest sample within 10 ms around it requires, then smoothed over 10 ms, so it eases
    in before a peak and out after it. Where the signal is under the ceiling the gain is 1 and
    the samples are returned untouched.
    """
    peak = np.max(np.abs(samples), axis=1) if samples.ndim == 2 else np.abs(samples)
    if peak.size == 0 or float(peak.max()) <= ceiling:
        return samples
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-12))
    blocks = -(-len(need) // _LIMIT_BLOCK)
    padded = np.concatenate([need, np.ones(blocks * _LIMIT_BLOCK - len(need))])
    per_block = padded.reshape(blocks, _LIMIT_BLOCK).min(axis=1)
    held = np.pad(per_block, _LIMIT_HOLD, constant_values=1.0)
    window = np.lib.stride_tricks.sliding_window_view(held, 2 * _LIMIT_HOLD + 1)
    floor = window.min(axis=1)
    kernel = np.hanning(_LIMIT_HOLD * 2 + 3)[1:-1]
    smooth = np.convolve(np.pad(floor, _LIMIT_HOLD, mode="edge"), kernel / kernel.sum(), "valid")
    centres = (np.arange(blocks) + 0.5) * _LIMIT_BLOCK
    gain = np.interp(np.arange(len(need)), centres, smooth[:blocks])
    gain = np.minimum(gain, need)  # never above what a sample needs, whatever the smoothing
    return samples * (gain[:, None] if samples.ndim == 2 else gain)


def _write_pcm16(path: Path, samples: np.ndarray) -> None:
    """A stereo 16-bit 48 kHz WAV from float samples (frames x 2)."""
    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        handle.writeframes(pcm.tobytes())


def _encoded_true_peak(wav: Path) -> float | None:
    """True peak of ``wav`` after the AAC encode of the final video (same encoder settings)."""
    checked = wav.with_name(f".{wav.stem}.{uuid.uuid4().hex}.check.m4a")
    try:
        run(
            ["-i", str(wav), "-map", "0:a:0", "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
             "-ac", "2", str(checked)]
        )  # fmt: skip
        return true_peak(checked)
    finally:
        checked.unlink(missing_ok=True)


def mix_sfx(
    bed: Path,
    hits: list[SfxHit],
    duration: float,
    out: Path,
    *,
    ceiling_db: float = SFX_CEILING_DB,
    true_peak_db: float | None = -1.0,
) -> Path:
    """Lay sound effects over a finished bed: a stereo WAV exactly ``duration`` long.

    The bed is not touched (its loudness was set before); each sound is added at its time and
    volume, a sound that would run past the end is faded out over the last 20 ms, and a limiter
    keeps the sample peak under ``ceiling_db`` (it does nothing while the mix stays under it).
    The AAC encode of a sharp sound can overshoot the samples by a couple of dB, so with
    ``true_peak_db`` the mix is encoded as the final video will be, and when its true peak is
    above that value the ceiling is lowered and the mix limited again (up to three times).
    """

    bed = Path(bed)
    out = Path(out)
    duration = _positive_duration(duration)
    if not hits:
        raise ValueError("mix_sfx needs at least one sound")
    frames = round(duration * 48000)
    args: list[str] = ["-i", str(bed)]
    graph = ["[0:a:0]aformat=sample_rates=48000:channel_layouts=stereo[bed]"]
    lengths: dict[Path, float] = {}
    for number, hit in enumerate(hits, start=1):
        at = float(hit.at)
        volume = float(hit.volume)
        if not math.isfinite(at) or at < 0:
            raise ValueError(f"sfx time must be finite and not negative, got {hit.at!r}")
        if not math.isfinite(volume) or volume < 0:
            raise ValueError(f"sfx volume must be finite and not negative, got {hit.volume!r}")
        if hit.path not in lengths:
            lengths[hit.path], _ = _audio_or_error(hit.path)
        args.extend(["-i", str(hit.path)])
        steps = [
            f"[{number}:a:0]aformat=sample_rates=48000:channel_layouts=stereo",
            f"volume={volume:.6f}",
        ]
        remaining = duration - at
        if lengths[hit.path] > remaining:  # it would run past the end: stop it softly
            fade = min(0.02, max(remaining, 0.001))
            steps.append(f"afade=t=out:st={max(remaining - fade, 0.0):.6f}:d={fade:.6f}")
        steps.append(f"adelay=delays={round(at * 48000)}S:all=1")
        graph.append(",".join(steps) + f"[s{number}]")
    labels = "".join(f"[s{number}]" for number in range(1, len(hits) + 1))
    graph.append(
        f"[bed]{labels}amix=inputs={len(hits) + 1}:duration=first:dropout_transition=0:"
        "normalize=0[aout]"
    )
    mixed = run(
        [
            *args,
            "-filter_complex",
            ";".join(graph),
            "-map",
            "[aout]",
            "-f",
            "f32le",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-",
        ],
        capture_stdout=True,
    )
    samples = np.frombuffer(mixed.stdout or b"", dtype="<f4").reshape(-1, 2).astype(np.float64)
    # ffmpeg's frame sizes can leave the mix a few samples long or short: exact length here.
    samples = np.concatenate([samples, np.zeros((max(0, frames - len(samples)), 2))])[:frames]

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.stem}.{uuid.uuid4().hex}{out.suffix}")
    ceiling_now = float(ceiling_db)
    try:
        for _ in range(4):
            _write_pcm16(temporary, peak_limit(samples, 10 ** (ceiling_now / 20)))
            if true_peak_db is None:
                break
            peak = _encoded_true_peak(temporary)
            if peak is None or peak <= true_peak_db:
                break
            ceiling_now -= (peak - true_peak_db) + 0.2
        os.replace(temporary, out)
    finally:
        temporary.unlink(missing_ok=True)
    return out


# ---------------------------------------------------------------------------
# Beat grid: where the beats of a track fall on the timeline of the ad.
# ---------------------------------------------------------------------------

_BEAT_FFT = 512
_BEAT_HOP = 128  # 5.8 ms at 22.05 kHz: finer than a video frame
_BEAT_MAX_SECONDS = 120.0
_BEAT_MAX_COUNT = 600
_BEAT_LOW_EMPHASIS = 250.0  # Hz: above this the weight of a frequency falls off
_BEAT_FASTEST_SHARE = 0.8  # a faster tempo wins an octave if its grid scores this much
# Seconds taken off the grid so that beats fall on the attacks themselves (the flux of an
# attack peaks a little after the window centre or before it: measured on generated click,
# kick and hi-hat tracks).
_BEAT_LAG = -0.003


@dataclass(frozen=True)
class BeatGrid:
    """Beats of a track as it plays: seconds from the start of the played music."""

    tempo: float | None
    beats: list[float]
    analysed: float
    strength: float = 0.0

    def nearest(self, t: float) -> float | None:
        """The beat closest to ``t`` (``None`` without a beat)."""
        if not self.beats:
            return None
        return min(self.beats, key=lambda beat: abs(beat - t))


def _fine_envelope(samples) -> np.ndarray:
    """Spectral flux on short windows, one value every ``_BEAT_HOP`` samples.

    The low frequencies weigh more than the high ones, so the kick of a track outweighs its
    hi-hats and the grid lands on the beat, not on the off-beat.
    """
    if len(samples) < _BEAT_FFT * 4:
        return np.zeros(0)
    window = np.hanning(_BEAT_FFT).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(samples, _BEAT_FFT)[::_BEAT_HOP]
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) / (_BEAT_FFT / 4)
    compressed = np.log1p(100.0 * spectrum)
    frequencies = np.fft.rfftfreq(_BEAT_FFT, 1.0 / _PROFILE_RATE)
    weights = 1.0 / np.sqrt(1.0 + (frequencies / _BEAT_LOW_EMPHASIS) ** 2)
    weights[0] = 0.0
    flux = (np.maximum(np.diff(compressed, axis=0), 0.0) * weights).sum(axis=1)
    return np.concatenate([[0.0], flux])


def _comb_scores(smooth, fps: float, tempo: float, phases):
    """Mean envelope on the grid of ``tempo`` for each phase (in frames)."""
    size = len(smooth)
    period = fps * 60.0 / tempo
    count = int((size - 2) // period) + 1
    positions = phases[:, None] + period * np.arange(count)[None, :]
    inside = positions < size - 1
    clipped = np.clip(positions, 0, size - 2)
    low = np.floor(clipped).astype(int)
    frac = clipped - low
    values = np.where(inside, smooth[low] * (1 - frac) + smooth[low + 1] * frac, 0.0)
    return values.sum(axis=1) / np.maximum(inside.sum(axis=1), 1)


def _best_grid(smooth, fps: float, tempo: float) -> tuple[float, float, float]:
    """Tempo (searched within 2.5% of ``tempo``), phase in frames and score of the best grid."""
    best = (-1.0, tempo, 0.0)
    for candidate in tempo * (1.0 + np.linspace(-0.025, 0.025, 251)):
        phases = np.arange(0.0, fps * 60.0 / candidate, 1.0)
        scores = _comb_scores(smooth, fps, float(candidate), phases)
        index = int(np.argmax(scores))
        if scores[index] > best[0]:
            best = (float(scores[index]), float(candidate), float(phases[index]))
    return best[1], best[2], best[0]


def _grid_from_envelope(envelope, fps: float, tempo: float) -> tuple[float, float, float]:
    """Refine ``tempo`` and find the phase that puts a regular grid on the strongest attacks.

    The tempo estimate may be an octave off (a snare on 2 and 4 reads as half the tempo, hi-hats
    as double), so the half and double are tried too and the fastest grid that is almost as
    strong as the best one wins. Returns the tempo (BPM), the position of the first beat in
    seconds on the envelope's clock and how much stronger the envelope is on the grid than on
    average.
    """
    kernel = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2)
    smooth = np.convolve(envelope, kernel / kernel.sum(), mode="same")
    grids = [
        _best_grid(smooth, fps, candidate)
        for candidate in (tempo / 2, tempo, tempo * 2)
        if 60.0 <= candidate <= 200.0
    ]
    top = max(score for _, _, score in grids)
    found, phase, score = max(
        (grid for grid in grids if grid[2] >= _BEAT_FASTEST_SHARE * top), key=lambda g: g[0]
    )
    around = _comb_scores(smooth, fps, found, np.array([phase - 1.0, phase, phase + 1.0]))
    curvature = around[0] - 2 * around[1] + around[2]
    if curvature < 0:
        phase += max(-1.0, min(1.0, 0.5 * (around[0] - around[2]) / curvature))
    strength = score / max(float(np.mean(smooth)), 1e-9)
    return found, phase / fps, strength


def beat_grid(src: Path, *, start: float = 0.0, duration: float | None = None) -> BeatGrid:
    """Beat times of the music in ``src`` as it plays from ``start`` (seconds, from 0).

    The tempo comes from the onset envelope; the beats are a regular grid whose tempo and phase
    are tuned to the strongest onsets, so every beat falls on an attack of the music. A track
    without a steady beat (a pad, a piano) has no beats. ``src`` may be a video.
    """
    src = Path(src)
    start = float(start)
    if not math.isfinite(start) or start < 0:
        raise ValueError("start must be a finite number greater than or equal to zero")
    wanted = _BEAT_MAX_SECONDS
    if duration is not None:
        wanted = min(_positive_duration(duration), _BEAT_MAX_SECONDS)
    source_duration, _ = _audio_or_error(src)
    if start >= source_duration:
        raise ValueError(f"No audio remains in {src} at start={start}")
    decoded = run(
        [
            "-ss",
            str(start),
            "-t",
            str(wanted),
            "-i",
            str(src),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(_PROFILE_RATE),
            "-f",
            "f32le",
            "-",
        ],
        capture_stdout=True,
    )
    samples = np.frombuffer(decoded.stdout or b"", dtype="<f4")
    analysed = len(samples) / _PROFILE_RATE
    coarse = _onset_envelope(samples)
    tempo = _estimate_tempo(coarse, _PROFILE_RATE / _PROFILE_HOP)
    if tempo is None or _pulse(coarse) < _CALM_PULSE:
        return BeatGrid(None, [], analysed)
    envelope = _fine_envelope(samples)
    if len(envelope) == 0:
        return BeatGrid(None, [], analysed)
    found, first, strength = _grid_from_envelope(envelope, _PROFILE_RATE / _BEAT_HOP, tempo)
    # Frame k covers samples k*hop to k*hop+fft: read its onset from the window centre.
    first = first + (_BEAT_FFT / 2) / _PROFILE_RATE - _BEAT_LAG
    period = 60.0 / found
    # A beat a few milliseconds before 0 is the beat at 0 (the music starts on it).
    first -= period * math.floor((first + 0.02) / period)
    beats: list[float] = []
    index = 0
    while len(beats) < _BEAT_MAX_COUNT and first + index * period < analysed:
        beats.append(round(max(first + index * period, 0.0), 4))
        index += 1
    return BeatGrid(round(found, 2), beats, analysed, round(strength, 2))
