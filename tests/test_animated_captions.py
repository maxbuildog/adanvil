"""Animated captions (karaoke, word, pop), keyword emphasis, punctuation-aware chunks, and the
storyboards saved by name on the project."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import anyio
import numpy as np
import pytest
from mcp import Client
from PIL import Image

from adanvil._bin import ffmpeg_path
from adanvil.ffmpeg import run
from adanvil.project import init_project
from adanvil.render import compose
from adanvil.server import build_server, render_caption_preview
from adanvil.storyboard import (
    CaptionOverlay,
    SpeechCaptionsOverlay,
    Storyboard,
    VoiceCaptionsOverlay,
    validate_storyboard,
)
from adanvil.storyboard.patch import apply_changes, storyboard_name
from adanvil.text import TextSpec, get_provider, render, render_states, spoken_words
from adanvil.text.animate import plan_states, pop_scales, strip_markers
from adanvil.text.chunking import chunk_ranges
from adanvil.voice import FakeEngine, VoiceSpec, generate_voice
from tests.test_speech_edit import _expand, _jump_cut_board, _talk_project

GREEN = "#00FF00"
FPS = 30
NONE = get_provider("none")


def _grey_clip(path: Path, duration: float, size: str = "360x640") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg_path(), "-nostdin", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=0x808080:s={size}:r={FPS}:d={duration}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )  # fmt: skip
    return path


def _project(tmp_path: Path, duration: float = 6.0) -> Path:
    init_project(tmp_path, "Sunbrew kettles")
    _grey_clip(tmp_path / "footage" / "grey.mp4", duration)
    return tmp_path


def _board(duration: float, overlays: list[dict], name: str = "anim") -> Storyboard:
    return Storyboard.model_validate(
        {
            "name": name,
            "format": {"width": 360, "height": 640, "fps": FPS},
            "segments": [
                {"type": "clip", "source": "footage/grey.mp4", "in": 0.0, "out": duration}
            ],
            "overlays": overlays,
        }
    )


def _frames(video: Path, width: int = 360, height: int = 640) -> np.ndarray:
    raw = run(
        ["-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_stdout=True
    ).stdout
    data = np.frombuffer(raw or b"", dtype=np.uint8)
    return data.reshape(-1, height, width, 3)


def _is_green(frame: np.ndarray) -> np.ndarray:
    red, green, blue = (frame[..., i].astype(int) for i in range(3))
    return (green > 170) & (red < 100) & (blue < 100)


def _green_x(frame: np.ndarray) -> float | None:
    columns = np.nonzero(_is_green(frame).any(axis=0))[0]
    return float(columns.mean()) if columns.size else None


def _ink_width(frame: np.ndarray) -> int:
    ink = (np.abs(frame.astype(int) - 128).max(axis=2) > 70).any(axis=0)
    columns = np.nonzero(ink)[0]
    return int(columns[-1] - columns[0] + 1) if columns.size else 0


# ----------------------------------------------------------------------------- chunking


def test_chunks_never_cross_a_sentence_end() -> None:
    words = ["Hello,", "my", "friend.", "How", "are", "you?"]
    assert chunk_ranges(words, 4) == [(0, 3), (3, 6)]
    # A one-word sentence is its own chunk, and the next sentence starts a new one.
    assert chunk_ranges(["Yes.", "Really", "now"], 3) == [(0, 1), (1, 3)]


def test_chunks_stop_at_a_comma_but_never_leave_a_lone_first_word() -> None:
    assert chunk_ranges(["So", "now,", "we", "go", "home", "fast."], 4) == [(0, 2), (2, 6)]
    # "Honestly," alone would be a one-word chunk: it stays with the words after it.
    assert chunk_ranges(["Honestly,", "it", "works"], 3) == [(0, 3)]
    # Plain text keeps the greedy cut (three, three, then the rest).
    assert chunk_ranges([f"w{i}" for i in range(7)], 3) == [(0, 3), (3, 6), (6, 7)]
    assert chunk_ranges([], 3) == []


def test_chunks_never_start_with_punctuation() -> None:
    # A detached mark rides with the word before it, an opening one with the word after it.
    assert chunk_ranges(["Great", "!", "Now"], 1) == [(0, 2), (2, 3)]
    assert chunk_ranges(["Wait", "«", "now", "»", "ok"], 1) == [(0, 1), (1, 4), (4, 5)]
    for first, _ in chunk_ranges(["a", "—", "b", "…", "c", "d"], 2):
        assert first not in (1, 3)


def test_the_word_animation_pairs_short_words() -> None:
    words = ["I", "love", "this", "kettle"]
    assert chunk_ranges(words, 4, pair_short=True) == [(0, 2), (2, 3), (3, 4)]


def test_voice_chunks_respect_punctuation(tmp_path: Path) -> None:
    track = generate_voice(
        tmp_path,
        VoiceSpec(engine="fake", lines=["Honestly this works, we love it. Try it now"]),
        "punct",
        engine=FakeEngine(),
    )
    texts = [
        " ".join(c.lines)
        for c in VoiceCaptionsOverlay(manifest=track.manifest, mode="chunks", max_words=4).expand(
            track
        )
    ]
    assert texts == ["Honestly this works,", "we love it.", "Try it now"]


def test_speech_chunks_respect_punctuation(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _talk_project(tmp_path, media)
    board = _jump_cut_board()
    captions = _expand(root, board, {"type": "speech_captions", "mode": "chunks", "max_words": 3})
    assert [" ".join(c.lines) for c in captions[:2]] == ["Bonjour tout le", "monde."]


# ------------------------------------------------------------------- markers and emphasis


def test_strip_markers_reads_stars_around_words() -> None:
    assert strip_markers("so *fast* today") == ("so fast today", {1})
    assert strip_markers("*very fast*, really") == ("very fast, really", {0, 1})
    assert strip_markers("*one* two *three*!") == ("one two three!", {0, 2})
    # A footnote star, or a star that never closes, is plain text.
    assert strip_markers("49€* only *terms") == ("49€* only *terms", set())
    assert strip_markers("plain text") == ("plain text", set())


def test_caption_markers_become_emphasis_words() -> None:
    caption = CaptionOverlay(style="outline", lines=["Only *Café* today", "no *rush*"])
    assert caption.lines == ["Only Café today", "no rush"]
    assert caption.emphasis == ["Café", "rush"]
    assert spoken_words(caption.lines) == ["Only", "Café", "today", "no", "rush"]
    # Old captions are untouched.
    plain = CaptionOverlay(style="pill", lines=["Free shipping*", "*terms apply"])
    assert plain.lines == ["Free shipping*", "*terms apply"] and plain.emphasis == []


def _green_columns(spec: TextSpec) -> tuple[int, int] | None:
    image = np.asarray(render(spec, NONE, canvas=(1080, 1920)).image)
    mask = (
        (image[..., 1] > 200) & (image[..., 0] < 60) & (image[..., 2] < 60) & (image[..., 3] > 250)
    )
    columns = np.nonzero(mask.any(axis=0))[0]
    return (int(columns[0]), int(columns[-1])) if columns.size else None


def test_emphasis_draws_listed_words_in_the_highlight_colour() -> None:
    lines = ["Free shipping on every Éclat order"]
    base = {"style": "outline", "lines": lines, "y": 500, "highlight": GREEN}
    assert _green_columns(TextSpec(**base)) is None
    # Case and accents do not matter, and only the listed word is lit.
    lit = _green_columns(TextSpec(**base, emphasis=["ECLAT"]))
    only_free = _green_columns(TextSpec(**base, emphasis=["free"]))
    assert lit is not None and only_free is not None
    assert only_free[1] < lit[0]  # "Free" is the first word, "Éclat" comes later
    both = _green_columns(TextSpec(**base, emphasis=["free", "eclat"]))
    assert both == (only_free[0], lit[1])


def test_emphasis_in_a_voice_line_show_text(tmp_path: Path) -> None:
    track = generate_voice(
        tmp_path,
        VoiceSpec(
            engine="fake",
            lines=[
                {"say": "forty nine dollars today", "show": "only *$49* today"},
                "pour *vite* maintenant",
            ],
        ),
        "mark",
        engine=FakeEngine(),
    )
    assert track.lines[0].display == "only *$49* today"
    assert track.lines[1].text == "pour vite maintenant"  # the voice never says the stars
    captions = VoiceCaptionsOverlay(manifest=track.manifest, mode="chunks", max_words=2).expand(
        track
    )
    assert [" ".join(c.lines) for c in captions] == ["only $49", "today", "pour vite", "maintenant"]
    assert [c.emphasis for c in captions] == [["$49"], [], ["vite"], []]
    # The same without any animation, in sentence mode.
    sentences = VoiceCaptionsOverlay(manifest=track.manifest).expand(track)
    assert sentences[0].lines == ["only $49 today"] and sentences[0].emphasis == ["$49"]


# --------------------------------------------------------------------- states and timing


def test_pop_scales_go_from_085_to_one_in_120_ms() -> None:
    scales = pop_scales(FPS)
    assert scales[0] == 0.85 and len(scales) == 4  # 120 ms at 30 fps, then 1.0
    assert scales == sorted(scales) and scales[-1] < 1.0
    assert scales[1] - scales[0] > scales[3] - scales[2]  # ease-out: fast first, then slows
    bounce = pop_scales(FPS, bounce=True)
    assert bounce[0] == 0.85 and max(bounce) == pytest.approx(1.08, abs=0.01)
    assert bounce[-1] < max(bounce) and bounce[-1] == pytest.approx(1.0, abs=0.02)


def test_state_plan_lights_each_word_until_the_next_one_starts() -> None:
    plans = plan_states("karaoke", 1.0, 2.0, FPS, [(1.0, 1.3), (1.3, 1.6), (1.62, 1.9)])
    assert [(p.active, p.start, p.end) for p in plans] == [
        (0, 1.0, 1.3),
        (1, 1.3, 1.62),
        (2, 1.62, 2.0),
    ]
    # A word shorter than half a frame can never be seen: the word before it runs on.
    short = plan_states("karaoke", 0.0, 1.0, FPS, [(0.0, 0.3), (0.3, 0.31), (0.31, 0.9)])
    assert [p.active for p in short] == [0, 2] and short[0].end == 0.31
    pop = plan_states("pop", 1.0, 1.1, FPS, [])
    assert [p.scale for p in pop] == pop_scales(FPS)[:3]  # cut at the end of the chunk
    assert plan_states("none", 1.0, 2.0, FPS, []) == plan_states("none", 1.0, 2.0, FPS, [])


def _states(style: str = "outline", **extra: object):
    spec = TextSpec(
        style=style, lines=["Gentle reminder that you", "deserve a slower morning 🔥"], y=900,
        highlight=GREEN, size=60, **extra,
    )  # fmt: skip
    return spec, render_states(spec, get_provider("twemoji"), [0, 1, 2, 4, 5, 6, 7])


def _lit_columns(state) -> tuple[int, int] | None:
    image = np.asarray(state.image)
    mask = (
        (image[..., 1] > 200) & (image[..., 0] < 60) & (image[..., 2] < 60) & (image[..., 3] > 250)
    )
    columns = np.nonzero(mask.any(axis=0))[0]
    return (state.x + int(columns[0]), state.x + int(columns[-1])) if columns.size else None


def test_each_state_lights_the_right_word_and_the_layout_never_moves() -> None:
    _, states = _states()
    assert states.states[0].image.size == states.states[3].image.size
    assert len({(s.image.size, s.x, s.y, s.bbox) for s in states.states}) == 1
    lit = [_lit_columns(s) for s in states.states]
    assert all(lit)
    # Words 0, 1, 2 are on the first line, left to right; 4 and 5 on the second.
    assert lit[0][1] < lit[1][0] < lit[1][1] < lit[2][0]
    assert lit[3][0] < lit[4][0]
    # Only the lit word changes from a state to the next: the rest of the ink is identical.
    first, second = (np.asarray(states.states[i].image) for i in (0, 1))
    assert first.shape == second.shape
    changed = np.nonzero((first != second).any(axis=(0, 2)))[0] + states.states[0].x
    assert changed.min() >= lit[0][0] - 20 and changed.max() <= lit[1][1] + 20  # shadow blur
    rows = np.nonzero((first != second).any(axis=(1, 2)))[0]
    assert rows.max() < first.shape[0] // 2 + 10  # the second line did not change


@pytest.mark.parametrize("style", ["pill", "box", "outline_soft", "label"])
def test_scale_and_box_keep_the_layout_stable_in_every_style(style: str) -> None:
    spec = TextSpec(
        style=style, lines=["one fine day"], y=700, highlight=GREEN, highlight_scale=1.2,
        highlight_box="#FFD400", size=56,
    )  # fmt: skip
    states = render_states(spec, NONE, [0, 1, 2])
    assert len({(s.image.size, s.x, s.y) for s in states.states}) == 1
    assert states.overflow is False


def test_highlight_box_text_is_readable_when_no_colour_is_given() -> None:
    spec = TextSpec(style="pill", lines=["one fine day"], y=700, highlight_box="#FFD400", size=56)
    state = render_states(spec, NONE, [1]).states[0]
    image = np.asarray(state.image)
    yellow = (image[..., 0] > 240) & (image[..., 1] > 190) & (image[..., 2] < 40)
    assert yellow.sum() > 500  # the box
    assert (image[..., 3] > 250).sum() > yellow.sum()


def test_states_are_cropped_to_even_offsets() -> None:
    for y in (701, 700):
        spec = TextSpec(style="box", lines=["odd or even"], y=y, highlight=GREEN, size=50)
        for state in render_states(spec, NONE, [0, 1]).states:
            assert state.x % 2 == 0 and state.y % 2 == 0


# ------------------------------------------------------------------ rendered clips


def _lit_word_per_frame(frames: np.ndarray, centres: list[float]) -> list[int | None]:
    result: list[int | None] = []
    for frame in frames:
        x = _green_x(frame)
        result.append(None if x is None else int(np.argmin([abs(x - c) for c in centres])))
    return result


def test_karaoke_lights_each_word_within_a_frame_of_its_start(tmp_path: Path) -> None:
    root = _project(tmp_path, 4.0)
    times = [(0.5, 1.1), (1.1, 1.65), (1.8, 2.4), (2.4, 3.3)]
    overlay = {
        "type": "caption", "style": "outline", "lines": ["alpha bravo charlie delta"],
        "start": 0.5, "end": 3.5, "y": 250, "size": 34, "max_width": 330,
        "animation": "karaoke", "highlight": GREEN, "word_times": times,
    }  # fmt: skip
    board = _board(4.0, [overlay])
    assert validate_storyboard(board, root=root).ok
    report = compose(board, root=root, quality="final")

    spec = board.overlays[0].to_text_spec()  # the same picture as the render, to find the words
    centres = []
    for state in render_states(spec, NONE, [0, 1, 2, 3], canvas=(360, 640)).states:
        columns = _lit_columns(state)
        assert columns
        centres.append(sum(columns) / 2)
    frames = _frames(report.path)
    lit = _lit_word_per_frame(frames, centres)

    expected_start = [0.5, 1.1, 1.8, 2.4]  # a word is lit from its own start
    for word, start in enumerate(expected_start):
        first = next(index for index, value in enumerate(lit) if value == word)
        assert abs(first - round(start * FPS)) <= 1, (word, first, start * FPS)
    # Nothing is lit before the chunk and after it; the lit word is always the one speaking.
    assert all(value is None for value in lit[: round(0.5 * FPS) - 1])
    assert all(value is None for value in lit[round(3.5 * FPS) + 1 :])
    for index in range(round(0.6 * FPS), round(3.4 * FPS)):
        clock = index / FPS
        speaking = max(i for i, start in enumerate(expected_start) if start <= clock + 1e-9)
        assert lit[index] is not None
        assert abs(lit[index] - speaking) <= 1 and (
            lit[index] == speaking or min(abs(clock - s) for s in expected_start) < 2 / FPS
        )
    # The chunk never blinks off while words change (the 1.65 to 1.8 s pause stays lit).
    assert all(value is not None for value in lit[round(0.6 * FPS) : round(3.4 * FPS)])


def test_the_highlight_uses_the_brand_highlight_colour_in_the_render(tmp_path: Path) -> None:
    from adanvil.brand import load_brand, save_brand

    root = _project(tmp_path, 2.5)
    brand = load_brand(root / "brand.yaml")
    brand.visual.highlight_color = GREEN
    save_brand(brand, root / "brand.yaml")
    overlay = {
        "type": "caption", "style": "outline", "lines": ["accent colour please"], "start": 0.2,
        "end": 2.2, "y": 250, "size": 34, "animation": "karaoke", "emphasis": ["please"],
    }  # fmt: skip
    report = compose(_board(2.5, [overlay]), root=root, quality="final")
    frames = _frames(report.path)
    assert _green_x(frames[round(0.5 * FPS)]) is not None  # a word is lit
    # With no brand highlight colour the default is yellow, never green, even with a
    # green accent: the accent is a brand colour, not a readable highlight.
    brand.visual.highlight_color = None
    brand.visual.accent_color = GREEN
    save_brand(brand, root / "brand.yaml")
    report = compose(_board(2.5, [overlay], "anim_yellow"), root=root, quality="final")
    assert _green_x(_frames(report.path)[round(0.5 * FPS)]) is None


def test_pop_scales_the_chunk_in_over_a_few_frames(tmp_path: Path) -> None:
    root = _project(tmp_path, 3.0)
    overlay = {
        "type": "caption", "style": "outline", "lines": ["pop in now"], "start": 0.5, "end": 2.5,
        "y": 250, "size": 40, "animation": "pop",
    }  # fmt: skip
    board = _board(3.0, [overlay])
    assert validate_storyboard(board, root=root).ok
    report = compose(board, root=root, quality="final")
    widths = [_ink_width(frame) for frame in _frames(report.path)]
    first = round(0.5 * FPS)
    assert widths[first - 1] == 0 and widths[first] > 0
    settled = widths[first + 8]
    assert widths[first + 12] == settled  # steady after the scale-in
    assert widths[first] == pytest.approx(0.85 * settled, rel=0.06)
    ramp = widths[first : first + 4]
    assert ramp == sorted(ramp) and ramp[0] < ramp[-1] <= settled
    assert widths[round(2.5 * FPS) + 1] == 0  # gone at the end

    bounce = {**overlay, "bounce": True}
    report = compose(_board(3.0, [bounce], "anim_bounce"), root=root, quality="final")
    widths = [_ink_width(frame) for frame in _frames(report.path)]
    assert max(widths[first : first + 8]) > widths[first + 14]  # overshoots, then settles


def test_word_animation_shows_one_or_two_big_words_at_a_time(tmp_path: Path) -> None:
    root = _project(tmp_path, 4.0)
    overlay = {
        "type": "caption", "style": "outline", "lines": ["go fast with Sunbrew today"],
        "start": 0.5, "end": 3.5, "y": 250, "size": 30, "animation": "word", "highlight": GREEN,
    }  # fmt: skip
    board = _board(4.0, [overlay])
    chunks = board.overlays[0].expand_words()
    assert [c.lines for c in chunks] == [["go fast"], ["with"], ["Sunbrew"], ["today"]]
    assert chunks[0].start == pytest.approx(0.5) and chunks[-1].end == pytest.approx(3.5)
    report = compose(board, root=root, quality="final")
    frames = _frames(report.path)
    widths = [_ink_width(f) for f in frames]
    assert widths[round(0.4 * FPS)] == 0 and widths[round(3.6 * FPS)] == 0
    # "with" (4 letters) is drawn alone and bigger than the 30 px size of the other captions.
    assert max(widths[round(1.4 * FPS) : round(2.0 * FPS)]) > 0
    # The lit word is green all the time (a word alone is the active word).
    assert _green_x(frames[round(2.2 * FPS)]) is not None


def test_many_animated_states_render_in_one_command(tmp_path: Path) -> None:
    """Hundreds of pictures overflow a Windows command line: the graph goes in a file."""

    root = _project(tmp_path, 14.0)
    words = " ".join(f"word{i % 9}x" for i in range(220))
    overlay = {
        "type": "caption", "style": "outline", "lines": [words], "start": 0.2, "end": 13.5,
        "y": 250, "size": 30, "animation": "word", "highlight": GREEN,
    }  # fmt: skip
    board = _board(14.0, [overlay])
    report = compose(board, root=root, quality="final")
    assert report.path.is_file() and len(report.overlays) == 220
    assert not list(report.path.parent.glob("*.filters.txt"))


def test_animated_caption_report_lists_its_states_for_qc(tmp_path: Path) -> None:
    from adanvil.qc import qc_report

    root = _project(tmp_path, 3.0)
    overlay = {
        "type": "caption", "style": "outline", "lines": ["one two three"], "start": 0.4,
        "end": 2.6, "y": 250, "size": 34, "animation": "karaoke", "highlight": GREEN,
    }  # fmt: skip
    board = _board(3.0, [overlay])
    report = compose(board, root=root, quality="final")
    entry = report.overlays[0]
    assert [s["active"] for s in entry["states"]] == [0, 1, 2]
    assert entry["offset"] == entry["states"][0]["offset"] and entry["canvas"] == [360, 640]
    assert json.dumps(report.model_dump(mode="json"))  # serialisable for the job report
    result = qc_report(report.path, render_report=report, out_dir=tmp_path / "qc")
    visible = [c for c in result.checks if c.id == "overlay_visible"]
    assert visible and all(c.status == "pass" for c in visible), visible


def test_a_caption_without_animation_renders_as_before(tmp_path: Path) -> None:
    root = _project(tmp_path, 3.0)
    overlay = {
        "type": "caption", "style": "pill", "lines": ["Free shipping"], "start": 0.2, "end": 2.0,
        "y": 300, "size": 30,
    }  # fmt: skip
    model = CaptionOverlay.model_validate(overlay)
    assert model.animation == "none" and model.emphasis == [] and model.word_times is None
    spec = model.to_text_spec()
    assert spec.styled is False and spec.size == 30
    board = _board(3.0, [overlay])
    report = compose(board, root=root, quality="final")
    entry = report.overlays[0]
    assert "states" not in entry
    png = np.asarray(Image.open(entry["png"]))
    direct = np.asarray(render(spec, get_provider("twemoji"), canvas=(360, 640)).image)
    assert png.shape == direct.shape and np.array_equal(png, direct)


# ----------------------------------------------------------- generated captions and previews


def test_voice_karaoke_captions_carry_the_time_of_their_words(tmp_path: Path) -> None:
    track = generate_voice(
        tmp_path,
        VoiceSpec(engine="fake", lines=["one two three four five six"]),
        "kara",
        engine=FakeEngine(),
    )
    overlay = VoiceCaptionsOverlay(
        manifest=track.manifest, mode="chunks", max_words=3, animation="karaoke", highlight=GREEN
    )
    chunks = overlay.expand(track)
    assert [len(c.word_times or []) for c in chunks] == [3, 3]
    for chunk in chunks:
        assert chunk.animation == "karaoke" and chunk.highlight == GREEN
        assert chunk.word_times[0][0] == pytest.approx(chunk.start)
        assert chunk.word_times == sorted(chunk.word_times)
    # Without the animation there is no word timing to carry.
    still = VoiceCaptionsOverlay(manifest=track.manifest, mode="chunks", max_words=3).expand(track)
    assert all(c.word_times is None and c.animation == "none" for c in still)
    # The word animation always cuts one or two words, and its size is not applied twice.
    word = VoiceCaptionsOverlay(
        manifest=track.manifest, mode="chunks", max_words=4, animation="word", size=40
    ).expand(track)
    assert [" ".join(c.lines) for c in word] == ["one two", "three", "four", "five", "six"]
    assert word[0].size == 40 and word[0].to_text_spec().size == 52


def test_speech_captions_carry_the_time_of_their_words(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _talk_project(tmp_path, media)
    captions = _expand(
        root,
        _jump_cut_board(),
        {
            "type": "speech_captions", "mode": "chunks", "max_words": 2, "animation": "karaoke",
            "emphasis": ["Brio"],
        },
    )  # fmt: skip
    assert [" ".join(c.lines) for c in captions[:2]] == ["Bonjour tout", "le monde."]
    assert [len(c.word_times) for c in captions[:2]] == [2, 2]
    assert captions[0].word_times[0][0] == pytest.approx(captions[0].start)
    assert captions[2].emphasis == ["Brio"]


def test_caption_preview_draws_the_requested_state(tmp_path: Path) -> None:
    root = _project(tmp_path, 4.0)
    track = generate_voice(
        root,
        VoiceSpec(engine="fake", lines=["one two three four five six"]),
        "prev",
        engine=FakeEngine(),
    )
    overlay = {
        "type": "voice_captions", "manifest": track.manifest, "mode": "chunks", "max_words": 3,
        "animation": "karaoke", "highlight": GREEN, "style": "outline", "size": 50,
    }  # fmt: skip

    def lit_x(payload: dict, image: Path) -> float:
        frame = np.asarray(Image.open(image).convert("RGB"))
        mask = _is_green(frame)
        assert mask.any(), payload
        return float(np.nonzero(mask.any(axis=0))[0].mean())

    xs = []
    for word in range(3):
        payload, image = render_caption_preview(
            root, "footage/grey.mp4", 1.0, {**overlay, "chunk_index": 1, "word_index": word}
        )
        assert payload["word_index"] == word and payload["chunk_index"] == 1
        assert (
            payload["words"] == 3 and payload["states"] == 3 and payload["animation"] == "karaoke"
        )
        assert payload["lines"] == ["four five six"]
        xs.append(lit_x(payload, image))
    assert xs[0] < xs[1] < xs[2]
    # Without indexes the state on screen at timeline time t is drawn.
    words = track.lines[0].words
    for word in (0, 2, 4):
        middle = (words[word][0] + words[word][1]) / 2 + track.lines[0].start
        payload, _ = render_caption_preview(root, "footage/grey.mp4", middle, overlay)
        assert (payload["chunk_index"], payload["word_index"]) == (word // 3, word % 3)
        assert payload["window"][0] <= middle < payload["window"][1]
    with pytest.raises(Exception, match="word_index 3 is out of range"):
        render_caption_preview(
            root, "footage/grey.mp4", 1.0, {**overlay, "chunk_index": 0, "word_index": 3}
        )
    with pytest.raises(Exception, match="needs an animated caption"):
        render_caption_preview(
            root,
            "footage/grey.mp4",
            1.0,
            {"type": "caption", "style": "pill", "lines": ["hi"], "word_index": 0},
        )


def test_caption_preview_of_pop_and_word_captions(tmp_path: Path) -> None:
    root = _project(tmp_path, 4.0)
    pop = {"type": "caption", "style": "box", "lines": ["pop in"], "start": 1, "end": 3,
           "animation": "pop"}  # fmt: skip
    first, _ = render_caption_preview(root, "footage/grey.mp4", 1.0, pop)
    settled, _ = render_caption_preview(root, "footage/grey.mp4", 2.0, pop)
    assert first["animation"] == "pop" and first["window"][0] == pytest.approx(1.0)
    assert settled["window"][0] > 1.1
    word = {"type": "caption", "style": "outline", "lines": ["go fast with Sunbrew"], "start": 0,
            "end": 4, "animation": "word", "highlight": GREEN}  # fmt: skip
    payload, _ = render_caption_preview(root, "footage/grey.mp4", 3.5, word)
    assert payload["lines"] == ["Sunbrew"] and payload["chunk_index"] == 2
    assert [e["text"] for e in payload["expanded"]] == ["go fast", "with", "Sunbrew"]


def test_emphasis_preview_uses_the_brand_highlight_colour(tmp_path: Path) -> None:
    from adanvil.brand import load_brand, save_brand

    root = _project(tmp_path, 3.0)
    brand = load_brand(root / "brand.yaml")
    brand.visual.highlight_color = GREEN
    save_brand(brand, root / "brand.yaml")
    payload, image = render_caption_preview(
        root, "footage/grey.mp4", 1.0,
        {"type": "caption", "style": "outline", "lines": ["only *today* please"], "size": 50},
    )  # fmt: skip
    assert payload["lines"] == ["only today please"]
    assert _is_green(np.asarray(Image.open(image).convert("RGB"))).any()


# ------------------------------------------------------------ vocabulary errors, all at once


def test_every_vocabulary_error_of_every_segment_is_reported_together(tmp_path: Path) -> None:
    root = _project(tmp_path)
    clip = {"type": "clip", "source": "footage/grey.mp4", "in": 0, "out": 1}
    board = {
        "name": "bad",
        "segments": [
            {**clip, "shot_type": "closeup", "scale": "huge"},
            {**clip, "emotion": "smug", "role": "villain"},
            {**clip, "shot_type": "zoom_thing"},
            {"type": "image", "source": "brand/cta.png", "duration": 1, "role": "ending"},
        ],
    }
    report = validate_storyboard(board, root=root)
    assert not report.ok
    where = {issue.where: issue.message for issue in report.errors}
    assert set(where) == {
        "segments[0].shot_type", "segments[0].scale", "segments[1].emotion",
        "segments[1].role", "segments[2].shot_type", "segments[3].role",
    }  # fmt: skip
    # Each one lists the values it accepts.
    assert "product_hero" in where["segments[0].shot_type"]
    assert "extreme_closeup" in where["segments[0].scale"]
    assert "worried" in where["segments[1].emotion"]
    assert "payoff" in where["segments[1].role"]
    assert "smug" in where["segments[1].emotion"]
    with pytest.raises(ValueError, match="must be one of"):
        Storyboard.model_validate(board)


# --------------------------------------------------------- storyboards saved by name


def test_apply_changes_walks_dots_and_indexes() -> None:
    document = {
        "name": "ad",
        "segments": [{"in": 0, "out": 1}, {"in": 2, "out": 3}],
        "overlays": [{"lines": ["a", "b"]}],
    }
    patched = apply_changes(
        document,
        [
            {"path": "segments.1.out", "value": 4.5},
            {"path": "segments[0].label", "value": "hook"},
            {"path": "overlays.0.lines.1", "value": "c"},
            {"path": "segments.-", "value": {"in": 5, "out": 6}},
            {"path": "overlays.0.lines.0", "remove": True},
        ],
    )
    assert patched["segments"][1]["out"] == 4.5 and patched["segments"][0]["label"] == "hook"
    assert patched["overlays"][0]["lines"] == ["c"] and len(patched["segments"]) == 3
    assert document["segments"][1]["out"] == 3  # the original is left alone
    for bad, text in [
        ([{"path": "segments.5.out", "value": 1}], "out of range"),
        ([{"path": "nothing.here", "value": 1}], "no key 'nothing'"),
        ([{"path": "name"}], "needs a value"),
        ([{"path": "segments.x.out", "value": 1}], "not a list index"),
        ([{"path": "name", "value": 1, "extra": 2}], "unexpected"),
        ([], "non-empty"),
    ]:
        with pytest.raises(ValueError, match=text):
            apply_changes(document, bad)
    assert storyboard_name("storyboards/hook_v2.json") == "hook_v2"
    for name in ("../x", "a/b", ".hidden", "", "x" * 81):
        with pytest.raises(ValueError):
            storyboard_name(name)


def _text(result) -> dict:
    return json.loads([item.text for item in result.content if item.type == "text"][0])


async def _by_name(root: Path) -> None:
    storyboard = {
        "version": 1,
        "name": "by_name",
        "segments": [
            {"type": "clip", "source": "footage/grey.mp4", "in": 0, "out": 1.0},
            {"type": "clip", "source": "footage/grey.mp4", "in": 1.0, "out": 3.0},
        ],
        "overlays": [
            {"type": "caption", "style": "pill", "lines": ["Hello"], "start": 0, "end": 1}
        ],
    }
    async with Client(build_server()) as client:
        saved = _text(
            await client.call_tool(
                "storyboard_save",
                {"project_dir": str(root), "storyboard": storyboard, "name": "by_name"},
            )
        )
        assert saved["path"] == "storyboards/by_name.json" and saved["segments"] == 2
        assert json.loads((root / saved["path"]).read_text("utf-8")) == storyboard

        bad_name = await client.call_tool(
            "storyboard_save",
            {"project_dir": str(root), "storyboard": storyboard, "name": "../escape"},
        )
        assert bad_name.is_error

        ok = _text(
            await client.call_tool(
                "storyboard_validate", {"project_dir": str(root), "storyboard": saved["path"]}
            )
        )
        assert ok["ok"] is True

        patched = _text(
            await client.call_tool(
                "storyboard_patch",
                {
                    "project_dir": str(root),
                    "name": "by_name",
                    "changes": [
                        {"path": "segments.1.out", "value": 2.0},
                        {"path": "overlays.0.lines.0", "value": "Hello there"},
                    ],
                },
            )
        )
        assert patched["problems"] == [] and patched["applied"] == [
            "segments.1.out",
            "overlays.0.lines.0",
        ]
        on_disk = json.loads((root / saved["path"]).read_text("utf-8"))
        assert on_disk["segments"][1]["out"] == 2.0 and on_disk["overlays"][0]["lines"] == [
            "Hello there"
        ]

        refused = await client.call_tool(
            "storyboard_patch",
            {"project_dir": str(root), "name": "by_name",
             "changes": [{"path": "segments.9.out", "value": 1}]},
        )  # fmt: skip
        assert refused.is_error
        assert json.loads((root / saved["path"]).read_text("utf-8")) == on_disk
        missing = await client.call_tool(
            "storyboard_patch",
            {"project_dir": str(root), "name": "nope", "changes": [{"path": "name", "value": "x"}]},
        )
        assert missing.is_error

        broken = _text(
            await client.call_tool(
                "storyboard_patch",
                {"project_dir": str(root), "name": "by_name",
                 "changes": [{"path": "segments.0.shot_type", "value": "nope"},
                             {"path": "segments.1.role", "value": "nope"}]},
            )
        )  # fmt: skip
        assert [p["where"] for p in broken["problems"]] == [
            "segments[0].shot_type",
            "segments[1].role",
        ]
        rejected = _text(
            await client.call_tool(
                "render_start", {"project_dir": str(root), "storyboard": saved["path"]}
            )
        )
        assert rejected["status"] == "rejected" and len(rejected["errors"]) == 2

        await client.call_tool(
            "storyboard_patch",
            {"project_dir": str(root), "name": "by_name",
             "changes": [{"path": "segments.0.shot_type", "remove": True},
                         {"path": "segments.1.role", "remove": True}]},
        )  # fmt: skip
        started = _text(
            await client.call_tool(
                "render_start",
                {"project_dir": str(root), "storyboard": "storyboards/by_name.json",
                 "quality": "preview"},
            )
        )  # fmt: skip
        assert started["status"] in {"queued", "running"}
        status = _text(
            await client.call_tool("job_status", {"job_id": started["job_id"], "wait_seconds": 180})
        )
        assert status["status"] == "done", status

        preview = await client.call_tool(
            "caption_preview",
            {
                "project_dir": str(root),
                "source": "footage/grey.mp4",
                "t": 0.5,
                "overlay": {"type": "caption", "style": "pill", "lines": ["Hi"]},
                "storyboard": "storyboards/by_name.json",
            },
        )
        assert not preview.is_error and any(item.type == "image" for item in preview.content)


def test_storyboards_saved_by_name_work_in_validate_render_and_patch(tmp_path: Path) -> None:
    anyio.run(_by_name, _project(tmp_path, 4.0))


def test_caption_preview_takes_the_storyboard_by_path_for_speech_captions(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    root = _talk_project(tmp_path, media)
    path = root / "storyboards" / "talk.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(_jump_cut_board().model_dump(mode="json", by_alias=True)), encoding="utf-8"
    )
    overlay = {"type": "speech_captions", "mode": "chunks", "max_words": 2, "chunk_index": 4}
    payload, _ = render_caption_preview(
        root, "footage/talk.mov", 1.0, overlay, storyboard="storyboards/talk.json"
    )
    assert payload["lines"] == ["Voici Brio"]
    # The object itself in the overlay keeps working.
    inner = {**overlay, "storyboard": "storyboards/talk.json"}
    payload, _ = render_caption_preview(root, "footage/talk.mov", 1.0, inner)
    assert payload["lines"] == ["Voici Brio"]


def test_the_tool_list_has_the_storyboard_tools() -> None:
    async def run_it() -> None:
        async with Client(build_server()) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        assert {"storyboard_save", "storyboard_patch"} <= set(tools)
        assert "storyboard" in tools["caption_preview"].input_schema["properties"]
        assert "word_index" in tools["caption_preview"].description

    anyio.run(run_it)


def test_animated_overlays_are_in_the_schema() -> None:
    from adanvil.storyboard import storyboard_json_schema

    defs = storyboard_json_schema()["$defs"]
    for name in ("CaptionOverlay", "VoiceCaptionsOverlay", "SpeechCaptionsOverlay"):
        properties = set(defs[name]["properties"])
        assert {"animation", "highlight", "highlight_scale", "highlight_box"} <= properties
        assert {"bounce", "emphasis"} <= properties
    animation = defs["CaptionOverlay"]["properties"]["animation"]
    assert set(animation["enum"]) == {"none", "karaoke", "pop", "word"}
    assert animation["default"] == "none"
    with pytest.raises(ValueError, match="karaoke"):
        SpeechCaptionsOverlay.model_validate({"animation": "wobble"})
    with pytest.raises(ValueError, match="invalid colour"):
        CaptionOverlay.model_validate({"style": "pill", "lines": ["a"], "highlight": "bluish"})


def test_validation_checks_the_chunks_of_a_word_caption(tmp_path: Path) -> None:
    root = _project(tmp_path)
    board = _board(
        4.0,
        [
            {
                "type": "caption",
                "style": "outline",
                "lines": ["go fast with Sunbrew"],
                "start": 0,
                "end": 3,
                "size": 40,
                "animation": "word",
            }
        ],
    ).model_dump(mode="json", by_alias=True)
    report = validate_storyboard(board, root=root)
    assert report.ok, report.errors
    assert not [w for w in report.warnings if w.code == "text_overflow"]
