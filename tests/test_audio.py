from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest

from adanvil.audio import (
    MUSIC_REFERENCE_LUFS,
    audio_duration,
    integrated_loudness,
    loudness_normalize,
    music_bed,
    music_gain,
    music_profile,
    profile_audio,
    reference_bed,
    rms_per_second,
    silence,
)
from adanvil.ffmpeg import probe, run


def _fixture(media: dict[str, dict[str, object]], name: str) -> Path:
    return Path(str(media[name]["path"]))


def test_music_bed_is_exact_stereo_aac_and_fades(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    out = music_bed(_fixture(media, "music.m4a"), 5.0, tmp_path / "bed.m4a")
    info = probe(out)
    levels = rms_per_second(out)

    assert audio_duration(out) == pytest.approx(5.0, abs=0.03)
    assert info.audio_sample_rate == 48_000
    assert info.audio_channels == 2
    assert levels[-1] < levels[2]
    assert levels[0] < levels[1]


def test_short_music_loops_without_silent_gap(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    out = music_bed(_fixture(media, "music_short.m4a"), 9.0, tmp_path / "loop.m4a")
    levels = rms_per_second(out)

    assert audio_duration(out) == pytest.approx(9.0, abs=0.03)
    assert all(level > -40 for level in levels[:-1])


def test_reference_bed_uses_video_audio(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    out = reference_bed(_fixture(media, "ref_ad.mp4"), 4.0, tmp_path / "ref.m4a", start=1.0)
    assert audio_duration(out) == pytest.approx(4.0, abs=0.03)


def test_audio_missing_raises(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="No audio stream"):
        music_bed(_fixture(media, "sdr_portrait.mp4"), 2.0, tmp_path / "missing.m4a")


def test_silence_is_silent(media: dict[str, dict[str, object]], tmp_path: Path) -> None:
    out = silence(2.0, tmp_path / "silence.m4a")
    assert audio_duration(out) == pytest.approx(2.0, abs=0.03)
    assert all(level <= -90 for level in rms_per_second(out))


def test_loudness_normalize_hits_target(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    source = _fixture(media, "music.m4a")
    out = loudness_normalize(source, tmp_path / "normalized.m4a")
    measured = run(
        [
            "-loglevel",
            "info",
            "-i",
            str(out),
            "-af",
            "loudnorm=I=-15:TP=-1.5:print_format=json",
            "-f",
            "null",
            "-",
        ]
    )
    report_start = measured.stderr.rfind("{")
    report_end = measured.stderr.find("}", report_start) + 1
    report = json.loads(measured.stderr[report_start:report_end])

    assert audio_duration(out) == pytest.approx(audio_duration(source), abs=0.03)
    assert float(report["input_i"]) == pytest.approx(-15.0, abs=1.5)


def test_music_gain_levels_loud_and_soft_tracks(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    source = _fixture(media, "music.m4a")
    loud = tmp_path / "loud.m4a"
    soft = tmp_path / "soft.m4a"
    # A mastered-like loud track, and the same track 12 dB softer.
    loudness_normalize(source, loud, i=-12.0)
    run(["-i", str(loud), "-af", "volume=0.25", "-c:a", "aac", "-b:a", "160k", str(soft)])

    loud_level = integrated_loudness(loud)
    soft_level = integrated_loudness(soft)
    assert loud_level is not None and soft_level is not None
    assert loud_level - soft_level == pytest.approx(12.0, abs=1.0)

    def level_after(path: Path) -> float:
        gain = music_gain(path, 0.35)
        assert integrated_loudness(path) is not None
        return integrated_loudness(path) + 20 * math.log10(gain)  # type: ignore[operator]

    expected = MUSIC_REFERENCE_LUFS + 20 * math.log10(0.35)
    assert level_after(loud) == pytest.approx(expected, abs=0.5)
    assert level_after(soft) == pytest.approx(expected, abs=0.5)


def _write_wav(path: Path, samples: np.ndarray, rate: int = 44_100) -> Path:
    data = (np.clip(samples, -1, 1) * 32_000).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.tobytes())
    return path


def _click_track(path: Path, bpm: float, seconds: float = 24.0, rate: int = 44_100) -> Path:
    samples = np.zeros(int(seconds * rate))
    burst = np.arange(int(0.03 * rate)) / rate
    click = np.sin(2 * np.pi * 1_800 * burst) * np.exp(-burst * 120)
    for beat in np.arange(0.2, seconds - 0.1, 60.0 / bpm):
        start = int(beat * rate)
        samples[start : start + len(click)] += click * 0.8
    return _write_wav(path, samples, rate)


@pytest.mark.parametrize("bpm", [90, 128])
def test_profile_estimates_the_tempo_of_a_click_track(tmp_path: Path, bpm: int) -> None:
    profile = profile_audio(_click_track(tmp_path / f"click{bpm}.wav", bpm))
    assert profile["tempo_bpm"] == pytest.approx(bpm, rel=0.05)
    assert profile["energy"]["onset_rate_per_s"] == pytest.approx(bpm / 60.0, rel=0.15)
    assert profile["duration"] == pytest.approx(24.0, abs=0.1)
    assert profile["loudness_lufs"] is not None


def test_profile_feel_separates_a_fast_track_from_a_sine_pad(tmp_path: Path) -> None:
    rate = 44_100
    time = np.arange(24 * rate) / rate
    pad = 0.3 * (np.sin(2 * np.pi * 220 * time) + np.sin(2 * np.pi * 330 * time)) / 2
    calm = profile_audio(_write_wav(tmp_path / "pad.wav", pad, rate))
    assert calm["feel"] == "calm"
    assert calm["energy"]["onset_rate_per_s"] < 0.5
    assert calm["tempo_bpm"] is None
    fast = profile_audio(_click_track(tmp_path / "fast.wav", 128))
    assert fast["feel"] == "energetic"


def test_music_profile_lists_the_music_folder_or_one_file(tmp_path: Path) -> None:
    (tmp_path / "music").mkdir()
    _click_track(tmp_path / "music" / "b.wav", 90, seconds=12)
    _click_track(tmp_path / "music" / "a.wav", 128, seconds=12)
    (tmp_path / "music" / "LICENCES.txt").write_text("x", encoding="utf-8")
    everything = music_profile(tmp_path)
    assert [track["path"] for track in everything["tracks"]] == ["music/a.wav", "music/b.wav"]
    assert "energetic" in everything["feel_thresholds"]
    single = music_profile(tmp_path, "music/b.wav")
    assert [track["path"] for track in single["tracks"]] == ["music/b.wav"]
    with pytest.raises(FileNotFoundError):
        music_profile(tmp_path, "music/missing.wav")


def test_profile_reads_an_encoded_music_file(media: dict[str, dict[str, object]]) -> None:
    profile = profile_audio(_fixture(media, "music.m4a"))
    assert profile["feel"] in {"calm", "medium", "energetic"}
    assert profile["duration"] > 0
    assert profile["energy"]["loudness_range_lu"] is not None


def test_profile_without_audio_raises(media: dict[str, dict[str, object]]) -> None:
    with pytest.raises(ValueError, match="No audio stream"):
        profile_audio(_fixture(media, "sdr_portrait.mp4"))
