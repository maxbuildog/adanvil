"""Talking-head editing: clip sound bed, audio_from, speech captions and their checks."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from adanvil.audio import audio_duration
from adanvil.brand import Brand, Claims
from adanvil.ffmpeg import probe, run
from adanvil.library import index_library
from adanvil.project import init_project
from adanvil.render import compose
from adanvil.server import render_caption_preview
from adanvil.speech.captions import expand_speech_captions, played_ranges
from adanvil.speech.library import SpeechLibrary
from adanvil.storyboard import AudioSpec, Storyboard, validate_storyboard
from tests.speech_helpers import ScriptedEngine, make_clip

SAMPLE_RATE = 48_000


def _pcm(path: Path) -> np.ndarray:
    raw = run(
        ["-i", str(path), "-map", "0:a:0", "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        capture_stdout=True,
    ).stdout
    return np.frombuffer(raw or b"", dtype="<f4")


def _peak_hz(samples: np.ndarray, start: float, end: float) -> float:
    window = samples[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
    if window.size == 0 or float(np.sqrt(np.mean(window**2))) < 1e-4:
        return 0.0
    spectrum = np.abs(np.fft.rfft(window * np.hanning(window.size)))
    return float(np.argmax(spectrum) * SAMPLE_RATE / window.size)


def _rms(samples: np.ndarray, start: float, end: float) -> float:
    window = samples[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
    return float(np.sqrt(np.mean(window**2))) if window.size else 0.0


def _silences(path: Path, noise: str = "-45dB", minimum: float = 0.12) -> list[float]:
    result = run(
        [
            "-loglevel", "info", "-i", str(path), "-map", "0:a:0",
            "-af", f"silencedetect=noise={noise}:d={minimum}", "-f", "null", "-",
        ]
    )  # fmt: skip
    return [float(item) for item in re.findall(r"silence_duration: ([0-9.]+)", result.stderr or "")]


def _project(tmp_path: Path, media: dict[str, dict[str, object]]) -> Path:
    init_project(tmp_path, "Ember & Oak")
    footage = tmp_path / "footage"
    make_clip(footage / "tone440.mp4", 4.0, [(0.0, 4.0, 440)])
    make_clip(footage / "tone880.mp4", 4.0, [(0.0, 4.0, 880)])
    make_clip(footage / "mute.mp4", 3.0, audio=False)
    shutil.copy2(Path(str(media["cta.png"]["path"])), tmp_path / "brand" / "cta.png")
    shutil.copy2(Path(str(media["music_short.m4a"]["path"])), tmp_path / "music" / "bed.m4a")
    return tmp_path


def _board(segments: list[dict], name: str = "talk", **audio: object) -> Storyboard:
    return Storyboard.model_validate(
        {"name": name, "segments": segments, "audio": {"clips": True, **audio}}
    )


# ----------------------------------------------------------------------------- audio.clips


def test_clip_sound_bed_is_as_long_as_the_video_and_cuts_align(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _project(tmp_path, media)
    board = _board(
        [
            {"type": "clip", "source": "footage/tone440.mp4", "in": 0.5, "out": 1.7},
            {"type": "clip", "source": "footage/tone880.mp4", "in": 0.0, "out": 1.0},
            {"type": "image", "source": "brand/cta.png", "duration": 0.8},
            {"type": "clip", "source": "footage/mute.mp4", "in": 0.0, "out": 0.7},
            {"type": "clip", "source": "footage/tone440.mp4", "in": 2.0, "out": 3.0},
        ]
    )
    assert validate_storyboard(board, root=root).ok

    report = compose(board, root=root, quality="preview")

    video = probe(report.path)
    assert video.duration == pytest.approx(board.total_duration(), abs=0.05)
    assert audio_duration(report.path) == pytest.approx(video.duration, abs=0.05)
    sound = _pcm(report.path)
    # Each segment plays its own clip: 440 Hz, 880 Hz, silence (image), silence (no audio), 440.
    assert _peak_hz(sound, 0.15, 1.05) == pytest.approx(440, abs=20)
    assert _peak_hz(sound, 1.35, 2.05) == pytest.approx(880, abs=20)
    assert _peak_hz(sound, 2.35, 2.9) == 0.0
    assert _peak_hz(sound, 3.05, 3.55) == 0.0
    assert _peak_hz(sound, 3.65, 4.5) == pytest.approx(440, abs=20)
    # The change of sound happens on the cut of the picture (1.2 s), 10 ms crossfade included.
    steps = np.arange(1.05, 1.35, 0.02)
    first_high = next(t for t in steps if _peak_hz(sound, t, t + 0.02) > 660)
    assert first_high == pytest.approx(1.2, abs=0.04)
    # The bed is brought to the voice level (-15 LUFS) on its own.
    assert _rms(sound, 0.15, 1.05) > 0.05


def test_music_plays_under_the_clip_sound(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _project(tmp_path, media)
    segments = [
        {"type": "clip", "source": "footage/tone440.mp4", "in": 0.0, "out": 1.2},
        {"type": "clip", "source": "footage/mute.mp4", "in": 0.0, "out": 1.2},
    ]
    bare = compose(_board(segments, "bare"), root=root, quality="preview")
    mixed = compose(
        _board(segments, "mixed", music="music/bed.m4a", music_volume=0.4),
        root=root,
        quality="preview",
    )
    plain, with_music = _pcm(bare.path), _pcm(mixed.path)
    assert audio_duration(mixed.path) == pytest.approx(probe(mixed.path).duration, abs=0.05)
    # Under the silent clip the music is heard; under the creator it stays well below.
    assert _rms(plain, 1.4, 2.2) < 1e-4
    assert _rms(with_music, 1.4, 2.2) > 0.003
    assert _rms(with_music, 0.2, 1.0) > 5 * _rms(with_music, 1.4, 2.2)


def test_the_speed_of_a_clip_keeps_the_sound_aligned_with_the_picture(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _project(tmp_path, media)
    board = _board(
        [
            {"type": "clip", "source": "footage/tone440.mp4", "in": 0.0, "out": 2.0, "speed": 2.0},
            {"type": "clip", "source": "footage/tone880.mp4", "in": 0.0, "out": 0.8,
             "freeze": {"hold": 0.5, "zoom": 1.2, "center": [540, 960]}},
        ]
    )  # fmt: skip
    report = compose(board, root=root, quality="preview")
    sound = _pcm(report.path)
    assert audio_duration(report.path) == pytest.approx(probe(report.path).duration, abs=0.05)
    assert _peak_hz(sound, 0.1, 0.9) == pytest.approx(440, abs=20)  # pitch kept at 2x speed
    assert _peak_hz(sound, 1.15, 1.7) == pytest.approx(880, abs=20)
    assert _peak_hz(sound, 1.9, 2.25) == 0.0  # the frozen hold is silent


def test_clips_excludes_voice_and_reference_audio() -> None:
    assert AudioSpec().clips is False
    with pytest.raises(ValidationError, match="audio.clips and audio.voice"):
        AudioSpec(clips=True, voice="voice/demo/manifest.json")
    with pytest.raises(ValidationError, match="audio.clips and audio.from_reference"):
        AudioSpec(clips=True, from_reference="footage/ref.mp4")
    assert AudioSpec(clips=True, music="music/bed.m4a").music == "music/bed.m4a"


def test_validation_reports_the_exclusions(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _project(tmp_path, media)
    report = validate_storyboard(
        {
            "name": "talk",
            "segments": [{"type": "clip", "source": "footage/tone440.mp4", "in": 0, "out": 1}],
            "audio": {"clips": True, "from_reference": "footage/tone880.mp4"},
        },
        root=root,
    )
    assert not report.ok
    assert any("audio.clips and audio.from_reference" in e.message for e in report.errors)


# ------------------------------------------------------------------------------ audio_from


def test_audio_from_plays_another_clip_under_the_picture(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _project(tmp_path, media)
    board = _board(
        [
            {"type": "clip", "source": "footage/tone880.mp4", "in": 0.0, "out": 1.2},
            # The picture of the 880 Hz clip, the sound of the 440 Hz clip from 2.0 s on.
            {
                "type": "clip",
                "source": "footage/tone880.mp4",
                "in": 1.0,
                "out": 2.5,
                "audio_from": {"source": "footage/tone440.mp4", "in": 2.0},
            },
            {"type": "clip", "source": "footage/tone880.mp4", "in": 2.0, "out": 3.0},
        ]
    )
    assert validate_storyboard(board, root=root).ok
    report = compose(board, root=root, quality="preview")
    sound = _pcm(report.path)
    assert audio_duration(report.path) == pytest.approx(probe(report.path).duration, abs=0.05)
    assert _peak_hz(sound, 0.1, 1.1) == pytest.approx(880, abs=20)
    assert _peak_hz(sound, 1.35, 2.65) == pytest.approx(440, abs=20)  # the other clip's sound
    assert _peak_hz(sound, 2.8, 3.6) == pytest.approx(880, abs=20)
    # The model round-trips with its alias, as render jobs store storyboards.
    dumped = board.model_dump(mode="json", by_alias=True)
    assert dumped["segments"][1]["audio_from"] == {"source": "footage/tone440.mp4", "in": 2.0}
    assert Storyboard.model_validate(dumped).segments[1].audio_from.in_ == 2.0


def test_audio_from_needs_clips_and_a_source_with_enough_sound(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _project(tmp_path, media)

    def report(audio: dict, source: str = "footage/tone440.mp4", at: float = 0.0):
        return validate_storyboard(
            {
                "name": "talk",
                "segments": [
                    {
                        "type": "clip",
                        "source": "footage/tone880.mp4",
                        "in": 0,
                        "out": 1.5,
                        "audio_from": {"source": source, "in": at},
                    }
                ],
                "audio": audio,
            },
            root=root,
        )

    without_clips = report({})
    assert not without_clips.ok
    assert [e.code for e in without_clips.errors] == ["audio_from"]
    assert "audio.clips" in without_clips.errors[0].message
    assert report({"clips": True}).ok

    assert any(e.code == "audio_from" for e in report({"clips": True}, "footage/mute.mp4").errors)
    too_far = report({"clips": True}, at=3.5)  # 3.5 + 1.5 s is beyond the 4 s of the clip
    assert any(e.code == "audio_from" and "lasts" in e.message for e in too_far.errors)
    missing = report({"clips": True}, "footage/none.mp4")
    assert any(e.code == "missing_file" for e in missing.errors)
    assert (
        not Storyboard.model_validate(
            {
                "name": "talk",
                "segments": [{"type": "clip", "source": "a.mp4", "in": 0, "out": 1}],
            }
        )
        .segments[0]
        .audio_from
    )


def test_audio_from_rejects_unknown_fields_and_unsafe_paths() -> None:
    base = {"type": "clip", "source": "footage/a.mp4", "in": 0, "out": 1}
    for bad in (
        {"source": "footage/b.mp4", "in": 0, "extra": 1},
        {"source": "../outside.mp4", "in": 0},
        {"source": "footage/b.mp4", "in": -1},
    ):
        with pytest.raises((ValidationError, ValueError)):
            Storyboard.model_validate(
                {"name": "x", "segments": [{**base, "audio_from": bad}], "audio": {"clips": True}}
            )


# ------------------------------------------------------------------------- speech captions

TALK = [
    (0.5, 0.8, "Bonjour"),
    (0.85, 1.1, "tout"),
    (1.15, 1.5, "le"),
    (1.55, 1.9, "monde."),
    (3.0, 3.3, "Voici"),
    (3.35, 3.6, "Brio"),
    (3.65, 3.9, "euh"),
    (3.95, 4.3, "pour"),
    (4.35, 4.7, "vous."),
]


def _talk_project(tmp_path: Path, media: dict[str, dict[str, object]]) -> Path:
    root = _project(tmp_path, media)
    make_clip(root / "footage" / "talk.mov", 6.0, [(0.5, 4.7, 300)])
    index_library(root, "footage", engine=ScriptedEngine({"talk": TALK}), transcribe=True)
    return root


def _jump_cut_board() -> Storyboard:
    return Storyboard.model_validate(
        {
            "name": "talk",
            "segments": [
                {"type": "clip", "source": "footage/talk.mov", "in": 0.44, "out": 2.0},
                {"type": "clip", "source": "footage/talk.mov", "in": 2.9, "out": 4.8, "speed": 2.0},
                {
                    "type": "clip",
                    "source": "footage/mute.mp4",
                    "in": 0.0,
                    "out": 1.0,
                    "audio_from": {"source": "footage/talk.mov", "in": 3.0},
                },
            ],
            "overlays": [{"type": "speech_captions", "mode": "chunks", "max_words": 2}],
            "audio": {"clips": True},
        }
    )


def _expand(root: Path, board: Storyboard, overlay: dict | None = None):
    from adanvil.storyboard import SpeechCaptionsOverlay

    spec = overlay or board.overlays[0].model_dump()
    timeline = board.timeline()
    return expand_speech_captions(
        SpeechCaptionsOverlay.model_validate(spec),
        board,
        SpeechLibrary(root),
        [item[0] for item in timeline],
        [item[1] for item in timeline],
    )


def test_played_ranges_follow_speed_and_audio_from(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    board = _jump_cut_board()
    timeline = board.timeline()
    ranges = played_ranges(board, [t[0] for t in timeline], [t[1] for t in timeline])
    assert [(r.segment, r.source, r.speed, r.audio_from) for r in ranges] == [
        (0, "footage/talk.mov", 1.0, False),
        (1, "footage/talk.mov", 2.0, False),
        (2, "footage/talk.mov", 1.0, True),
    ]
    assert (ranges[1].src_in, ranges[1].src_out) == (2.9, 4.8)
    assert (ranges[2].src_in, ranges[2].src_out) == (3.0, pytest.approx(4.0))
    assert ranges[1].start == pytest.approx(1.56) and ranges[1].end == pytest.approx(2.51)


def test_speech_captions_land_on_the_timeline_after_jump_cuts_and_speed(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _talk_project(tmp_path, media)
    captions = _expand(root, _jump_cut_board())

    shown = [(" ".join(c.lines), round(c.start, 3), round(c.end, 3)) for c in captions]
    # Segment 0 plays 0.44-2.0 at normal speed: the words move 0.44 s earlier.
    assert shown[0] == ("Bonjour tout", 0.06, 0.71)
    assert shown[1][0] == "le monde." and shown[1][1] == 0.71
    # It would linger 0.2 s after "monde." (1.46 s) but the next sentence starts at 1.61 s.
    assert shown[1][2] == pytest.approx(1.61)
    # Segment 1 plays 2.9-4.8 at 2x: 0.1 s into the source is 0.05 s on the timeline.
    assert shown[2][0] == "Voici Brio" and shown[2][1] == pytest.approx(1.61)
    assert shown[3][0] == "pour vous." and shown[3][1] == pytest.approx(2.085)
    assert shown[3][2] == pytest.approx(2.51)  # up to the next sentence
    # Segment 2 shows B-roll over the creator's voice (audio_from 3.0 s): "euh" is dropped
    # by default and "pour" (middle at 4.125 s) is past the 1 s that segment plays.
    assert shown[4][0] == "Voici Brio" and shown[4][1] == pytest.approx(2.51)
    assert len(shown) == 5
    assert all(c.start < c.end for c in captions)
    assert [c.start for c in captions] == sorted(c.start for c in captions)

    with_fillers = _expand(
        root, _jump_cut_board(), {"type": "speech_captions", "mode": "chunks", "max_words": 2,
                                  "keep_fillers": True}
    )  # fmt: skip
    assert any("euh" in " ".join(c.lines) for c in with_fillers)
    assert not any("euh" in " ".join(c.lines) for c in captions)


def test_captions_never_outlast_the_ad(tmp_path: Path, media: dict[str, dict[str, object]]) -> None:
    root = _talk_project(tmp_path, media)
    board = Storyboard.model_validate(
        {
            "name": "talk",
            "segments": [{"type": "clip", "source": "footage/talk.mov", "in": 2.9, "out": 4.75}],
            "overlays": [{"type": "speech_captions"}],
            "audio": {"clips": True},
        }
    )
    captions = _expand(root, board)
    # "vous." ends at 1.80 s of the ad, which lasts 1.85 s: the 0.2 s linger is cut at the end.
    assert captions[-1].end == pytest.approx(board.total_duration())
    report = validate_storyboard(board, root=root)
    assert not [w for w in report.warnings if w.code == "overlay_clamped"]


def test_sentence_mode_and_case(tmp_path: Path, media: dict[str, dict[str, object]]) -> None:
    root = _talk_project(tmp_path, media)
    board = _jump_cut_board()
    sentence_mode = {"type": "speech_captions", "mode": "sentence", "case": "upper"}
    captions = _expand(root, board, sentence_mode)
    assert [" ".join(c.lines) for c in captions[:2]] == [
        "BONJOUR TOUT LE MONDE.",
        "VOICI BRIO POUR VOUS.",
    ]
    assert captions[0].start == pytest.approx(0.06)


def test_caption_preview_builds_speech_captions_with_a_chunk_index(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _talk_project(tmp_path, media)
    # Without a storyboard the preview shows the captions of the source clip itself.
    payload, image = render_caption_preview(
        root,
        "footage/talk.mov",
        1.0,
        {"type": "speech_captions", "mode": "chunks", "max_words": 2, "chunk_index": 1},
    )
    assert image.is_file() and payload["bbox"]
    assert payload["lines"] == ["le monde."]
    assert [item["text"] for item in payload["expanded"]] == [
        "Bonjour tout",
        "le monde.",
        "Voici Brio",
        "pour vous.",
    ]
    # With a storyboard the timeline is the one of the ad.
    storyboard = _jump_cut_board().model_dump(mode="json", by_alias=True)
    payload, _ = render_caption_preview(
        root,
        "footage/talk.mov",
        1.0,
        {
            "type": "speech_captions",
            "mode": "chunks",
            "max_words": 2,
            "chunk_index": 4,
            "storyboard": storyboard,
        },
    )
    assert payload["lines"] == ["Voici Brio"]
    assert payload["expanded"][4]["start"] == pytest.approx(2.51)
    with pytest.raises(Exception, match="out of range"):
        render_caption_preview(
            root, "footage/talk.mov", 1.0, {"type": "speech_captions", "chunk_index": 99}
        )


# ------------------------------------------------------------------------------ validation


def _valid_ad(overrides: dict | None = None) -> dict:
    board = {
        "name": "talk",
        "segments": [
            {"type": "clip", "source": "footage/talk.mov", "in": 0.44, "out": 2.0, "label": "hook"},
            {"type": "clip", "source": "footage/talk.mov", "in": 2.9, "out": 4.8, "label": "demo"},
        ],
        "overlays": [{"type": "speech_captions", "mode": "chunks", "max_words": 3}],
        "audio": {"clips": True},
    }
    board.update(overrides or {})
    return board


def test_cut_inside_a_spoken_word_is_a_warning(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _talk_project(tmp_path, media)
    clean = validate_storyboard(_valid_ad(), root=root)
    assert clean.ok and not [w for w in clean.warnings if w.code == "cut_mid_word"]

    cut = _valid_ad()
    cut["segments"][0]["out"] = 1.7  # inside "monde." (1.55 to 1.90)
    report = validate_storyboard(cut, root=root)
    assert report.ok
    mid = [w for w in report.warnings if w.code == "cut_mid_word"]
    assert len(mid) == 1 and mid[0].where == "segments[0].out"
    assert "monde." in mid[0].message and "1.70" in mid[0].message

    starts = _valid_ad()
    starts["segments"][1]["in"] = 3.1  # inside "Voici" (3.00 to 3.30)
    warned = validate_storyboard(starts, root=root).warnings
    assert [w.where for w in warned if w.code == "cut_mid_word"] == ["segments[1].in"]

    # Without audio.clips the sound is not kept: nothing to say about the cuts.
    muted = _valid_ad({"overlays": [], "audio": {}})
    muted["segments"][0]["out"] = 1.7
    warned = validate_storyboard(muted, root=root).warnings
    assert not [w for w in warned if w.code == "cut_mid_word"]

    # An audio_from cut is checked on the clip whose sound plays.
    over = _valid_ad({"overlays": []})
    over["segments"][0]["audio_from"] = {"source": "footage/talk.mov", "in": 1.7}
    found = [w for w in validate_storyboard(over, root=root).warnings if w.code == "cut_mid_word"]
    # It starts inside "monde." and, 1.56 s later (3.26 s), ends inside "Voici".
    assert [w.where for w in found] == ["segments[0].audio_from.in", "segments[0]"]


def test_banned_word_said_by_the_creator_is_a_warning_not_an_error(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _talk_project(tmp_path, media)
    brand = Brand(name="Ember & Oak", claims=Claims(banned_words=["Brio", "guéri"]))
    report = validate_storyboard(_valid_ad(), root=root, brand=brand)
    assert report.ok, report.errors  # the shown words are the spoken ones: no banned_word error
    said = [w for w in report.warnings if w.code == "banned_word_spoken"]
    assert len(said) == 1 and said[0].where == "segments[1]"
    assert "'Brio'" in said[0].message and "3.35" in said[0].message

    # A range that leaves the word out does not warn.
    short = _valid_ad()
    short["segments"][1]["out"] = 3.32
    assert not [
        w
        for w in validate_storyboard(short, root=root, brand=brand).warnings
        if w.code == "banned_word_spoken"
    ]


def test_speech_captions_without_audio_clips_is_an_error(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _talk_project(tmp_path, media)
    report = validate_storyboard(_valid_ad({"audio": {}}), root=root)
    assert not report.ok
    error = next(e for e in report.errors if e.code == "speech_captions")
    assert "audio.clips" in error.message and error.where == "overlays[0]"
    with pytest.raises(ValueError, match="requires audio.clips"):
        compose(Storyboard.model_validate(_valid_ad({"audio": {}})), root=root, quality="preview")


def test_clips_without_a_transcript_are_reported(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _talk_project(tmp_path, media)
    make_clip(root / "footage" / "new.mp4", 2.0, [(0.0, 2.0, 300)])  # never indexed
    board = {
        "name": "talk",
        "segments": [{"type": "clip", "source": "footage/new.mp4", "in": 0, "out": 1}],
        "overlays": [{"type": "speech_captions"}],
        "audio": {"clips": True},
    }
    report = validate_storyboard(board, root=root)
    assert report.ok
    missing = [w for w in report.warnings if w.code == "speech_missing"]
    # The clip has no speech index, and the overlay then finds no word to show.
    by_place = {w.where: w for w in missing}
    assert sorted(by_place) == ["overlays[0]", "segments[0]"]
    assert "library_index" in by_place["segments[0]"].message
    assert "new.mp4" in by_place["segments[0]"].message

    # B-roll without sound needs no transcript.
    board["segments"] = [{"type": "clip", "source": "footage/mute.mp4", "in": 0, "out": 1}]
    board["overlays"] = []
    assert not [
        w for w in validate_storyboard(board, root=root).warnings if w.code == "speech_missing"
    ]


def test_speech_captions_are_checked_like_any_caption(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _talk_project(tmp_path, media)
    ad = _valid_ad({"overlays": [{"type": "speech_captions", "y": 5000}]})
    report = validate_storyboard(ad, root=root)
    assert any(e.code == "bbox_out_of_canvas" for e in report.errors)
    assert validate_storyboard(_valid_ad(), root=root).summary["overlay_count"] == 1


# ------------------------------------------------------------------------ the whole ad


def test_talking_head_ad_end_to_end(tmp_path: Path, media: dict[str, dict[str, object]]) -> None:
    from adanvil.speech.takes import ClipSpeech, rank_takes

    root = _project(tmp_path, media)
    footage = root / "footage"
    # Two takes of one hook: the first has a hesitation, the second is clean.
    hook1 = [(0.4, 0.8, "Mon"), (0.85, 1.3, "matin"), (1.9, 2.2, "euh"), (2.3, 2.8, "change.")]
    hook2 = [(0.3, 0.7, "Mon"), (0.75, 1.2, "matin"), (1.25, 1.8, "change."), (2.0, 2.4, "Brio.")]

    def bursts(items):
        return [(start, end, 300) for start, end, _ in items]

    make_clip(footage / "HOOK1.mov", 3.2, bursts(hook1))
    make_clip(footage / "HOOK1.1.mov", 3.2, bursts(hook2))
    make_clip(footage / "product.mp4", 3.0, [(0.0, 3.0, 880)])
    engine = ScriptedEngine({"HOOK1": hook1, "HOOK1.1": hook2})
    index_library(root, "footage", engine=engine, transcribe=True)

    library = SpeechLibrary(root)
    clips = []
    for name in ("HOOK1.mov", "HOOK1.1.mov"):
        entry = library.entry(f"footage/{name}")
        clips.append(ClipSpeech(entry.id, entry.path, entry.duration, library.data(entry.path)))
    best = rank_takes(clips)[0]["best"]
    assert best["source"] == "footage/HOOK1.1.mov"

    segments = [{"type": "clip", **segment, "label": "hook"} for segment in best["clean_segments"]]
    segments[0] = {**segments[0], "in": segments[0]["in"]}
    segments.append(
        {
            "type": "clip",
            "source": "footage/product.mp4",
            "in": 0.0,
            "out": 0.6,
            "audio_from": {"source": "footage/HOOK1.1.mov", "in": 1.9},
            "label": "demo",
        }
    )
    board = {
        "name": "talk",
        "segments": segments,
        "overlays": [
            {"type": "speech_captions", "mode": "chunks", "max_words": 2, "case": "upper"}
        ],
        "audio": {"clips": True, "music": "music/bed.m4a", "music_volume": 0.3},
    }
    report = validate_storyboard(board, root=root)
    assert report.ok, report.errors
    assert not [w for w in report.warnings if w.code == "cut_mid_word"]

    rendered = compose(Storyboard.model_validate(board), root=root, quality="preview")
    assert rendered.duration == pytest.approx(
        sum(float(s["out"]) - float(s["in"]) for s in segments), abs=0.1
    )
    assert audio_duration(rendered.path) == pytest.approx(rendered.duration, abs=0.06)
    assert any(item["type"] == "caption" for item in rendered.overlays)
    assert len(rendered.overlays) >= 2
    # The creator's pauses are gone: no stretch of silence is longer than the cut margins.
    quiet = _silences(rendered.path, noise="-35dB", minimum=0.2)
    assert max(quiet, default=0.0) < 0.3
    # The hook is the clean take: its filler-free words come first.
    assert rendered.overlays[0]["type"] == "caption" and rendered.overlays[0]["start"] < 0.3


# --------------------------------------------------------------------------------- recipe


def _imaginary_index(root: Path, files: dict[str, tuple[float, list[tuple[float, float, str]]]]):
    """Write catalogue entries and speech sidecars for clips that do not exist on disk."""

    from adanvil.library.catalog import Entry, EntrySpeech, entry_id, merge_into_catalog
    from adanvil.speech import store
    from adanvil.speech.model import SpeechData, SpeechWord

    entries = []
    for path, (duration, script) in files.items():
        key = {"path": str(root / path), "size": 1, "mtime_ns": 1}
        relative = store.sidecar_relative(key)
        store.write_sidecar(
            root,
            relative,
            SpeechData(
                words=[SpeechWord(start, end, text, 0.95) for start, end, text in script],
                language="en",
                duration=duration,
                levels_db=[-22.0] * int(duration * 10 + 1),
            ),
            source=path,
            model="small",
        )
        entries.append(
            Entry(
                id=entry_id(path),
                path=path,
                kind="video",
                file_key=key,
                duration=duration,
                has_audio=True,
                speech=EntrySpeech(
                    status="done",
                    language="en",
                    text=" ".join(text for _, _, text in script),
                    words_file=relative,
                    has_speech=True,
                ),
                indexed="2026-01-01T00:00:00+00:00",
            )
        )
    merge_into_catalog(root, entries)


def test_ugc_talking_recipe_loads_and_its_example_fits_an_imaginary_speech_index(
    tmp_path: Path,
) -> None:
    from adanvil.speech.analyze import read_speech
    from adanvil.storyboard import get_recipe, list_recipes
    from tests.speech_helpers import spoken, words

    recipe = get_recipe("ugc_talking")
    assert recipe in list_recipes()
    assert [slot.slot for slot in recipe.structure] == [
        "hook", "problem", "demo", "result", "cta", "end_card",
    ]  # fmt: skip
    allowed = {
        "total_duration", "shot_duration", "caption_min_on_screen", "caption_safe_zone",
        "end_card_required", "max_caption_lines", "caption_count", "banned_words",
    }  # fmt: skip
    assert all(rule.check in allowed for rule in recipe.rules)
    assert recipe.example["recipe"] == "ugc_talking" and recipe.checklist

    # The imaginary index described in the example's comments.
    hook = spoken("Honestly, I almost gave up on my mornings.", 0.5)
    main = [
        *spoken("Every morning was the same rush, and I was always late.", 0.4),
        *spoken("Then a friend showed me this.", 4.75),
        *spoken("It takes two minutes, no setup.", 7.3),
        *spoken("And the difference is crazy.", 9.85),
        *spoken("Now I actually enjoy my mornings.", 12.05),
        *spoken("Tap the link and try it for yourself.", 14.6),
    ]
    init_project(tmp_path, "Ember & Oak")
    _imaginary_index(
        tmp_path,
        {
            "footage/creator_a_hook_2.mov": (5.0, hook),
            "footage/creator_a_main.mov": (18.0, main),
        },
    )

    example = recipe.example
    segments = example["segments"]
    face = {s["label"]: s for s in segments if s.get("source") == "footage/creator_a_main.mov"}
    clean = read_speech(words(*main), "footage/creator_a_main.mov").segments
    # Every cut of the example is a clean segment of the index: nowhere inside a word.
    assert [(s["in"], s["out"]) for s in [segments[0]]] == [(0.44, 3.35)]
    assert read_speech(words(*hook), "x").segments[0]["in"] == 0.44
    assert read_speech(words(*hook), "x").segments[0]["out"] == 3.35
    expected = [(item["in"], item["out"]) for item in clean]
    assert [(s["in"], s["out"]) for s in face.values()] == [
        expected[0], expected[1], expected[4], expected[5],
    ]  # fmt: skip
    over = [s["audio_from"]["in"] for s in segments if "audio_from" in s]
    assert over == [expected[2][0], expected[3][0]]
    assert sum(s["out"] - s["in"] for s in segments if "audio_from" in s) == pytest.approx(
        (expected[2][1] - expected[2][0]) + (expected[3][1] - expected[3][0]), abs=0.01
    )

    report = validate_storyboard(example, root=tmp_path, recipe=recipe, check_files=False)
    assert report.ok, report.errors
    assert [w for w in report.warnings if w.code in {"cut_mid_word", "banned_word_spoken"}] == []
    assert report.summary["total_duration"] == pytest.approx(21.27, abs=0.02)

    # The captions come from the speech, in chunks of three upper-case words.
    board = Storyboard.model_validate(example)
    timeline = board.timeline()
    captions = expand_speech_captions(
        board.overlays[0],
        board,
        SpeechLibrary(tmp_path),
        [item[0] for item in timeline],
        [item[1] for item in timeline],
    )
    texts = [" ".join(caption.lines) for caption in captions]
    assert texts[0] == "HONESTLY, I ALMOST"
    assert "TAP THE LINK" in texts and all(len(text.split()) <= 3 for text in texts)
    assert captions[-1].end <= board.total_duration()


def test_speech_caption_words_glue_elisions_and_punctuation() -> None:
    from adanvil.speech.captions import SpokenWord, _merge_fragments

    def word(start: float, text: str, segment: int = 0) -> SpokenWord:
        return SpokenWord(start, start + 0.2, text, "footage/a.mp4", segment, int(start * 10))

    words = [
        word(0.0, "J"),
        word(0.2, "'ai"),
        word(0.4, "craqué"),
        word(0.6, "."),
        word(0.8, "l'"),
        word(1.0, "appareil"),
        word(1.2, "?"),
        word(1.4, "peut"),
        word(1.6, "-être"),
    ]
    assert [item.text for item in _merge_fragments(words)] == [
        "J'ai",
        "craqué.",
        "l'appareil ?",
        "peut-être",
    ]
    # Pieces from two different segments (a jump cut) are never glued together.
    split = [word(0.0, "J", segment=0), word(0.2, "'ai", segment=1)]
    assert [item.text for item in _merge_fragments(split)] == ["J", "'ai"]
