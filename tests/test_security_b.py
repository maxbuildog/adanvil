"""Security 0.3.1, agent B: ffmpeg lookup and inputs, ffconcat and filter-graph escaping, fonts
given as paths, writes through planted links, bounds on computed durations.

Every test reproduces an attack and shows it is refused, or shows that normal use still works.
"""

from __future__ import annotations

import importlib
import os
import shutil
import socket
import subprocess
import threading
from pathlib import Path

import pytest
from PIL import Image, ImageFont
from pydantic import ValidationError

from adanvil import ffmpeg as ffmpeg_module
from adanvil._bin import ffmpeg_path, ffprobe_path
from adanvil.audio import silence
from adanvil.endcard import parse_spec, render_end_card, write_end_card
from adanvil.ffmpeg import FFmpegError, extract_frame, probe, run, run_probe
from adanvil.media import Cache, ClipSegment, RevealSegment, SplitSegment, render_segment
from adanvil.media.segment import Format, ImageSegment
from adanvil.project import init_project
from adanvil.render.compose import _concat_mezzanines, _drawable_font, _ffconcat_quote, compose
from adanvil.storyboard import Storyboard
from adanvil.storyboard.schema import CaptionOverlay
from adanvil.storyboard.validate import _render_text_overlay, validate_storyboard
from adanvil.text import fonts
from adanvil.text.fonts import load_font, usable_family
from adanvil.text.styles import save_image
from adanvil.textremoval import RemoveText, ffmpeg_filter_path

compose_module = importlib.import_module("adanvil.render.compose")
UNC_FONT = r"\\attacker\share\a.ttf"
AWKWARD = "pro'j, ;[x] y"  # a folder name with a quote, a comma, a semicolon, brackets, a space


def _link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("symbolic links are not available here")


def _plant(link: Path, target: Path) -> None:
    """Make ``link`` another name for ``target``: a symbolic link, or where those are not
    allowed (Windows without the privilege) a hard link, which a write through the name
    modifies the same way."""

    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(target, link)
        return
    except (OSError, NotImplementedError, AttributeError):
        pass
    try:
        os.link(target, link)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("neither symbolic nor hard links are available here")


def _victim(tmp_path: Path) -> Path:
    victim = tmp_path / "victim.txt"
    victim.write_text("KEEP", encoding="utf-8")
    return victim


def _bundled_font() -> Path:
    return Path(str(fonts._FONT_DIR / "Roboto-VF.ttf"))


# ---------------------------------------------------------------------------------------------
# B1. ffmpeg found in the working folder
# ---------------------------------------------------------------------------------------------


def _fake_binary(folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (name + (".exe" if os.name == "nt" else ""))
    path.write_bytes(b"not a real program")
    path.chmod(0o755)
    return path


@pytest.mark.parametrize(("name", "lookup"), [("ffmpeg", ffmpeg_path), ("ffprobe", ffprobe_path)])
def test_binary_in_the_working_folder_is_never_picked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, lookup
) -> None:
    work = tmp_path / "work"
    _fake_binary(work, name)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.delenv("ADANVIL_FFMPEG", raising=False)
    monkeypatch.delenv("ADANVIL_FFPROBE", raising=False)
    # most MCP clients do not set it: with it unset, Windows (and shutil.which) search the
    # current folder first
    monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)
    monkeypatch.setenv("PATHEXT", ".EXE;.CMD")
    # an empty entry, the current folder written two ways, a relative entry, a real empty folder
    monkeypatch.setenv(
        "PATH", os.pathsep.join(["", ".", str(work), "..\\work" if os.name == "nt" else "../work",
                                 str(empty)])
    )
    with pytest.raises(FileNotFoundError):
        lookup()


@pytest.mark.parametrize(("name", "lookup"), [("ffmpeg", ffmpeg_path), ("ffprobe", ffprobe_path)])
def test_binary_is_found_on_an_absolute_path_entry_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, lookup
) -> None:
    work = tmp_path / "work"
    _fake_binary(work, name)
    first = _fake_binary(tmp_path / "first", name)
    second = _fake_binary(tmp_path / "second", name)
    monkeypatch.chdir(work)
    monkeypatch.delenv("ADANVIL_FFMPEG", raising=False)
    monkeypatch.delenv("ADANVIL_FFPROBE", raising=False)
    # most MCP clients do not set it: with it unset, Windows (and shutil.which) search the
    # current folder first
    monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)
    monkeypatch.setenv("PATHEXT", ".EXE;.CMD")
    monkeypatch.setenv(
        "PATH", os.pathsep.join([".", str(work), str(first.parent), str(second.parent)])
    )
    assert Path(lookup()) == first


def test_configured_binary_must_be_an_absolute_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = _fake_binary(tmp_path / "bin", "ffmpeg")
    monkeypatch.setenv("ADANVIL_FFMPEG", str(real))
    assert Path(ffmpeg_path()) == real
    monkeypatch.chdir(real.parent)
    monkeypatch.setenv("ADANVIL_FFMPEG", real.name)  # relative: refused
    with pytest.raises(FileNotFoundError, match="ADANVIL_FFMPEG"):
        ffmpeg_path()
    monkeypatch.setenv("ADANVIL_FFPROBE", str(tmp_path / "missing" / "ffprobe"))
    with pytest.raises(FileNotFoundError, match="ADANVIL_FFPROBE"):
        ffprobe_path()


def test_real_ffmpeg_is_still_found_on_path() -> None:
    assert Path(ffmpeg_path()).is_absolute() and Path(ffmpeg_path()).is_file()
    assert Path(ffprobe_path()).is_absolute() and Path(ffprobe_path()).is_file()


# ---------------------------------------------------------------------------------------------
# B2. stdin and protocols
# ---------------------------------------------------------------------------------------------


def _capture_subprocess(monkeypatch: pytest.MonkeyPatch) -> dict:
    captured: dict = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout=b"{}", stderr=b"")

    monkeypatch.setattr(ffmpeg_module.subprocess, "run", fake_run)
    return captured


def test_ffmpeg_child_does_not_inherit_the_mcp_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_subprocess(monkeypatch)
    run(["-f", "lavfi", "-i", "color=c=red", "-t", "1", "out.mp4"])
    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL
    run_probe(["-show_streams", "in.mp4"])
    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL


def test_every_input_is_limited_to_the_file_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_subprocess(monkeypatch)
    run(
        [
            "-ss", "1", "-i", "a.mp4",
            "-f", "lavfi", "-i", "color=c=red",
            "-loop", "1", "-framerate", "30", "-i", "b.png",
            "out.mp4",
        ]
    )
    command = captured["command"]
    inputs = [index for index, token in enumerate(command) if token == "-i"]
    assert len(inputs) == 3
    for index in inputs:
        assert command[index - 2 : index] == ["-protocol_whitelist", "file"]
    assert command.count("-protocol_whitelist") == 3

    run_probe(["-show_streams", "-of", "json", "in.mp4"])
    probe_command = captured["command"]
    assert probe_command[probe_command.index("-protocol_whitelist") + 1] == "file"
    assert probe_command.index("-protocol_whitelist") < probe_command.index("in.mp4")


def test_a_value_that_looks_like_an_option_after_dash_i_is_not_doubled() -> None:
    assert ffmpeg_module._local_inputs_only(["-i", "-i", "x"]) == [
        "-protocol_whitelist", "file", "-i", "-i", "x"
    ]


class _Listener:
    """A local TCP server that counts the connections it receives."""

    def __init__(self) -> None:
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(5)
        self.server.settimeout(0.2)
        self.port = self.server.getsockname()[1]
        self.connections = 0
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop:
            try:
                connection, _ = self.server.accept()
            except OSError:
                continue
            self.connections += 1
            connection.close()

    def __enter__(self) -> _Listener:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop = True
        self._thread.join(timeout=2)
        self.server.close()


def test_a_url_input_never_reaches_the_network(tmp_path: Path) -> None:
    with _Listener() as listener:
        url = f"http://127.0.0.1:{listener.port}/x.mp4"
        # control: without the option ffmpeg does connect (otherwise this test proves nothing)
        subprocess.run(
            [ffmpeg_path(), "-nostdin", "-v", "error", "-i", url, "-f", "null", "-"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if listener.connections == 0:
            pytest.skip("this ffmpeg build does not open http inputs")
        listener.connections = 0

        with pytest.raises(FFmpegError) as refused:
            run(["-i", url, "-f", "null", "-"], timeout=60)
        assert "not on whitelist" in str(refused.value)
        with pytest.raises(FFmpegError):
            run_probe(["-show_streams", url])
        assert listener.connections == 0


def test_a_playlist_disguised_as_a_video_cannot_reach_the_network(tmp_path: Path) -> None:
    with _Listener() as listener:
        manifest = f"""<?xml version="1.0"?>
<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" type="static" mediaPresentationDuration="PT1S"
 minBufferTime="PT1S" profiles="urn:mpeg:dash:profile:isoff-on-demand:2011">
<Period><AdaptationSet mimeType="video/mp4"><Representation id="1" bandwidth="1000"
 width="64" height="64" codecs="avc1.42c00d">
<BaseURL>http://127.0.0.1:{listener.port}/v.mp4</BaseURL>
<SegmentBase indexRange="0-100"><Initialization range="0-50"/></SegmentBase>
</Representation></AdaptationSet></Period></MPD>
"""
        disguised = tmp_path / "clip.mp4"
        disguised.write_text(manifest, encoding="utf-8")
        # A 7.0 static ffmpeg fetches the URLs of a DASH manifest even under
        # ``-protocol_whitelist file``: AdAnvil refuses the file before ffmpeg opens it.
        with pytest.raises(ValueError, match="playlist"):
            run(["-i", str(disguised), "-f", "null", "-"], timeout=60)
        assert listener.connections == 0
        with pytest.raises(Exception):  # noqa: B017 - a probe failure, whatever its type
            probe(disguised)
        assert listener.connections == 0


def test_local_inputs_lavfi_concat_and_image_sequences_still_work(tmp_path: Path) -> None:
    clip = tmp_path / "a.mp4"
    run(
        [
            "-f", "lavfi", "-i", "color=c=red:s=64x64:r=10:d=1",
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
            "-t", "1", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip),
        ]
    )
    assert probe(clip).duration == pytest.approx(1.0, abs=0.2)

    listing = tmp_path / "list.ffconcat"
    entry = f"file '{clip.as_posix()}'\n"
    listing.write_text("ffconcat version 1.0\n" + entry + entry, encoding="utf-8")
    joined = tmp_path / "b.mp4"
    run(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(joined)])
    assert probe(joined).duration == pytest.approx(2.0, abs=0.3)

    run(["-i", str(clip), "-frames:v", "3", str(tmp_path / "f%03d.png")])
    sequence = tmp_path / "c.mp4"
    pattern = str(tmp_path / "f%03d.png")
    run(["-framerate", "10", "-i", pattern, "-pix_fmt", "yuv420p", str(sequence)])
    assert probe(sequence).kind == "video"


# ---------------------------------------------------------------------------------------------
# B3. ffconcat list
# ---------------------------------------------------------------------------------------------


def _mezzanines(folder: Path, count: int = 2) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = folder / f"m{index}.mp4"
        run(
            [
                "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=10:d=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
            ]
        )
        paths.append(path)
    return paths


def test_ffconcat_quote_closes_and_reopens_the_quote() -> None:
    quoted = _ffconcat_quote(Path("/tmp/it's here/a.mp4"))
    assert quoted.startswith("file '") and quoted.endswith("'")
    assert "it'\\''s here" in quoted
    assert "\\'s" not in quoted.replace("'\\''", "")


def test_ffconcat_quote_refuses_a_line_break_in_a_path() -> None:
    for name in ("a\nfile '/etc/passwd'", "a\rb"):
        with pytest.raises(ValueError, match="line break"):
            _ffconcat_quote(Path("/tmp") / name)


def test_concat_works_in_a_folder_with_awkward_characters(tmp_path: Path) -> None:
    root = tmp_path / AWKWARD
    paths = _mezzanines(root / "cache" / "mezz")
    result = _concat_mezzanines(paths, Cache(root / "cache"))
    assert probe(result).duration == pytest.approx(2.0, abs=0.3)


def test_a_planted_concat_list_is_never_reused(tmp_path: Path) -> None:
    root = tmp_path / "project"
    cache = Cache(root / "cache")
    paths = _mezzanines(root / "cache" / "mezz")
    key = {"paths": [str(path.resolve()) for path in paths]}
    listing = cache.path_for("concat", key, suffix=".ffconcat")
    listing.parent.mkdir(parents=True, exist_ok=True)
    planted = "ffconcat version 1.0\nfile '/definitely/not/here/secret.mp4'\n"
    listing.write_text(planted, encoding="utf-8")

    result = _concat_mezzanines(paths, cache)

    assert probe(result).duration == pytest.approx(2.0, abs=0.3)
    written = listing.read_text(encoding="utf-8")
    assert "secret.mp4" not in written
    assert "m0.mp4" in written and "m1.mp4" in written


def test_a_concat_list_planted_as_a_link_is_replaced_not_followed(tmp_path: Path) -> None:
    root = tmp_path / "project"
    cache = Cache(root / "cache")
    paths = _mezzanines(root / "cache" / "mezz")
    key = {"paths": [str(path.resolve()) for path in paths]}
    listing = cache.path_for("concat", key, suffix=".ffconcat")
    victim = _victim(tmp_path)
    _plant(listing, victim)

    _concat_mezzanines(paths, cache)

    assert victim.read_text(encoding="utf-8") == "KEEP"
    assert not listing.is_symlink() and "m0.mp4" in listing.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# B4. filter graph escaping for the text removal mask
# ---------------------------------------------------------------------------------------------


def test_filter_path_with_quotes_and_separators_is_escaped_for_both_levels(
    tmp_path: Path,
) -> None:
    assert ffmpeg_filter_path("/tmp/it's.png") == "'/tmp/it\\'\\''s.png'"
    directory = tmp_path / AWKWARD
    directory.mkdir()
    mask = directory / "m.png"
    Image.new("L", (64, 64), 255).save(mask)
    output = directory / "out.mp4"
    run(
        [
            "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=1",
            "-loop", "1", "-i", str(mask),
            "-filter_complex", f"[0:v]format=yuv420p,removelogo=f={ffmpeg_filter_path(mask)}[v]",
            "-map", "[v]", "-frames:v", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(output),
        ]
    )
    assert output.is_file()


def test_mask_removal_works_in_an_awkward_project_folder_without_the_path_in_the_graph(
    media: dict[str, dict[str, object]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import adanvil.media.segment as segment_module

    source = Path(str(media["burnt_text.mp4"]["path"]))
    box = tuple(int(value) for value in media["burnt_text.mp4"]["text_bbox"])
    calls: list[tuple[list[str], Path | None]] = []
    original = segment_module.run

    def recording(args, **kwargs):
        calls.append((list(args), kwargs.get("cwd")))
        return original(args, **kwargs)

    monkeypatch.setattr(segment_module, "run", recording)
    cache = Cache(tmp_path / AWKWARD / "cache")
    result = render_segment(
        ClipSegment(
            source="burnt_text.mp4",
            **{"in": 0.5, "out": 1.0, "gamma": None},
            remove_text=RemoveText(method="mask", box=box),
        ),
        root=source.parent,
        cache=cache,
    )
    assert result.is_file() and probe(result).kind == "video"
    graphs = [
        args[args.index("-filter_complex") + 1] for args, _ in calls if "-filter_complex" in args
    ]
    assert graphs and all("pro'j" not in graph for graph in graphs)
    masked = [cwd for args, cwd in calls if any("removelogo" in item for item in args)]
    assert masked and masked[0] is not None and masked[0].name == "mask"


# ---------------------------------------------------------------------------------------------
# B5. fonts given as a path
# ---------------------------------------------------------------------------------------------


def _project_with_font(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project"
    font = root / "brand" / "fonts" / "mine.ttf"
    font.parent.mkdir(parents=True)
    shutil.copy2(_bundled_font(), font)
    return root, font


def test_usable_family_accepts_bundled_names_and_project_font_files(tmp_path: Path) -> None:
    root, font = _project_with_font(tmp_path)
    assert usable_family("poppins", root) == "poppins"
    assert usable_family("Roboto", root) == "Roboto"
    assert usable_family("brand/fonts/mine.ttf", root) == str(font)
    assert usable_family("brand\\fonts\\mine.ttf", root) == str(font)
    assert usable_family(str(font), root) == str(font)  # absolute, but inside the project


def test_usable_family_refuses_everything_else(tmp_path: Path) -> None:
    root, font = _project_with_font(tmp_path)
    outside = tmp_path / "outside" / "x.ttf"
    outside.parent.mkdir()
    shutil.copy2(font, outside)
    (root / "brand" / "fonts" / "not-a-font.png").write_bytes(b"x")
    (root / "brand" / "fonts" / "folder.ttf").mkdir()
    for family in (
        UNC_FONT,
        "//attacker/share/a.ttf",
        "\\\\?\\UNC\\attacker\\share\\a.ttf",
        str(outside),
        "../outside/x.ttf",
        "brand/fonts/missing.ttf",
        "brand/fonts/not-a-font.png",
        "brand/fonts/folder.ttf",
        "no-such-family",
        "",
        "brand/fonts/mine\x00.ttf",
    ):
        assert usable_family(family, root) is None, family
    assert usable_family(None, root) is None
    assert usable_family("brand/fonts/mine.ttf", None) is None


def test_a_font_linked_out_of_the_project_is_refused(tmp_path: Path) -> None:
    root, _ = _project_with_font(tmp_path)
    outside = tmp_path / "outside.ttf"
    shutil.copy2(_bundled_font(), outside)
    _link(root / "brand" / "fonts" / "linked.ttf", outside)
    assert usable_family("brand/fonts/linked.ttf", root) is None


def test_a_network_font_path_is_never_touched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    touched: list[str] = []
    real_stat, real_lstat, real_realpath = os.stat, os.lstat, os.path.realpath

    def spy(real):
        def wrapper(path, *args, **kwargs):
            if isinstance(path, str) and path.startswith(("\\\\", "//")):
                touched.append(path)
            return real(path, *args, **kwargs)

        return wrapper

    monkeypatch.setattr(os, "stat", spy(real_stat))
    monkeypatch.setattr(os, "lstat", spy(real_lstat))
    monkeypatch.setattr(os.path, "realpath", spy(real_realpath))
    load_font.cache_clear()
    root, _ = _project_with_font(tmp_path)
    assert usable_family(UNC_FONT, root) is None
    with pytest.raises(ValueError):
        load_font(UNC_FONT, 400, 20)
    with pytest.raises(ValueError):
        load_font("//attacker/share/a.ttf", 400, 20)
    assert parse_spec({"product_image": "a.png", "title": "Hi", "family": UNC_FONT}).family
    assert touched == []


def test_font_path_second_line_of_defence(tmp_path: Path) -> None:
    load_font.cache_clear()
    with pytest.raises(ValueError, match="not an existing regular file"):
        load_font(str(tmp_path / "missing.ttf"), 400, 20)
    folder = tmp_path / "folder.ttf"
    folder.mkdir()
    with pytest.raises(ValueError, match="not an existing regular file"):
        load_font(str(folder), 400, 20)
    # an operator-chosen absolute font file that exists still loads
    assert load_font(str(_bundled_font()), 400, 20) is not None
    assert load_font("poppins", 700, 30) is not None


def test_a_storyboard_font_that_is_a_network_path_renders_with_the_default_and_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    real_truetype = ImageFont.truetype

    def spy(font, *args, **kwargs):
        opened.append(str(font))
        return real_truetype(font, *args, **kwargs)

    monkeypatch.setattr(ImageFont, "truetype", spy)
    load_font.cache_clear()
    root = tmp_path / "project"
    init_project(root, "Demo")
    overlay = CaptionOverlay(style="pill", lines=["Hello"], start=0, end=1, family=UNC_FONT)
    bbox, _ = _render_text_overlay(overlay, root=root, width=1080, height=1920, brand=None)
    assert bbox != (0, 0, 0, 0)
    assert opened and not any(item.startswith("\\\\") for item in opened)

    storyboard = {
        "name": "fonts",
        "format": {"width": 1080, "height": 1920, "fps": 30},
        "emoji": "none",
        "segments": [{"type": "image", "source": "footage/a.png", "duration": 2}],
        "overlays": [
            {
                "type": "caption", "style": "pill", "lines": ["Hello"], "start": 0, "end": 1,
                "family": UNC_FONT,
            }
        ],
    }
    report = validate_storyboard(storyboard, root=root, check_files=False)
    font_errors = [i for i in report.errors if i.code in {"font_not_found", "text_render"}]
    assert font_errors == []
    warned = [issue for issue in report.warnings if issue.code == "font_not_found"]
    assert warned and "default family" in warned[0].message
    assert not any(item.startswith("\\\\") for item in opened)


def test_a_project_font_is_used_by_the_validation(tmp_path: Path) -> None:
    root, _ = _project_with_font(tmp_path)
    storyboard = {
        "name": "fonts",
        "format": {"width": 1080, "height": 1920, "fps": 30},
        "emoji": "none",
        "segments": [{"type": "image", "source": "footage/a.png", "duration": 2}],
        "overlays": [
            {
                "type": "caption", "style": "pill", "lines": ["Hello"], "start": 0, "end": 1,
                "family": "brand/fonts/mine.ttf",
            }
        ],
    }
    report = validate_storyboard(storyboard, root=root, check_files=False)
    assert not [issue for issue in report.warnings if issue.code == "font_not_found"]
    assert not [issue for issue in report.errors if issue.code in {"font_not_found", "text_render"}]


def test_drawable_font_replaces_a_refused_family_before_the_render(tmp_path: Path) -> None:
    root, font = _project_with_font(tmp_path)
    warnings: list[str] = []
    overlay = CaptionOverlay(style="pill", lines=["Hello"], family=UNC_FONT)
    cleaned = _drawable_font(overlay, root, warnings)
    assert cleaned.family is None and len(warnings) == 1 and "default family" in warnings[0]
    _drawable_font(overlay, root, warnings)
    assert len(warnings) == 1  # the same message is not repeated
    written = CaptionOverlay(style="pill", lines=["x"], family="brand/fonts/mine.ttf")
    project = _drawable_font(written, root, warnings)
    assert project.family == str(font)
    bundled = CaptionOverlay(style="pill", lines=["x"], family="montserrat")
    assert _drawable_font(bundled, root, warnings) is bundled


def _end_card_project(tmp_path: Path, media: dict[str, dict[str, object]]) -> Path:
    root = tmp_path / "project"
    init_project(root, "Ember & Oak")
    shutil.copy2(str(media["product.png"]["path"]), root / "footage" / "product.png")
    return root


def _card(**changes) -> dict:
    spec = {
        "product_image": "footage/product.png",
        "title": "Your first box is free",
        "button": {"text": "Order now"},
        "emoji": "none",
    }
    spec.update(changes)
    return spec


def test_end_card_font_must_be_bundled_or_a_project_file(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    root = _end_card_project(tmp_path, media)
    refused = render_end_card(root, _card(family=UNC_FONT))
    try:
        assert refused.layout["warnings"] and "default family" in refused.layout["warnings"][0]
    finally:
        refused.image.close()

    shutil.copy2(_bundled_font(), root / "brand" / "mine.ttf")
    fine = render_end_card(root, _card(family="brand/mine.ttf"))
    try:
        assert "warnings" not in fine.layout
    finally:
        fine.image.close()

    plain = render_end_card(root, _card())
    try:
        assert "warnings" not in plain.layout
    finally:
        plain.image.close()
    with pytest.raises(ValueError, match="family"):
        parse_spec(_card(family="comic-sans"))


# ---------------------------------------------------------------------------------------------
# B6. writes through planted links
# ---------------------------------------------------------------------------------------------


def test_save_image_replaces_a_planted_link(tmp_path: Path) -> None:
    victim = _victim(tmp_path)
    target = tmp_path / "out" / "sheet.jpg"
    _plant(target, victim)
    save_image(Image.new("RGB", (8, 8), "red"), target, format="JPEG")
    assert victim.read_text(encoding="utf-8") == "KEEP"
    assert not target.is_symlink()
    with Image.open(target) as saved:
        assert saved.size == (8, 8)
    # the format comes from the extension when it is not given
    other = save_image(Image.new("RGB", (4, 4), "blue"), tmp_path / "x.png")
    with Image.open(other) as saved:
        assert saved.format == "PNG"


def test_extract_frame_does_not_write_through_a_link(tmp_path: Path) -> None:
    clip = _mezzanines(tmp_path / "clips", count=1)[0]
    victim = _victim(tmp_path)
    out = tmp_path / "frames" / "frame.png"
    _plant(out, victim)
    extract_frame(clip, 0.0, out)
    assert victim.read_text(encoding="utf-8") == "KEEP"
    with Image.open(out) as saved:
        assert saved.format == "PNG"


def test_end_card_png_is_not_written_through_a_link(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    root = _end_card_project(tmp_path, media)
    victim = _victim(tmp_path)
    _plant(root / "brand" / "endcards" / "card.png", victim)
    output, _ = write_end_card(root, _card(), "card")
    assert victim.read_text(encoding="utf-8") == "KEEP"
    assert not output.is_symlink()
    with Image.open(output) as saved:
        assert saved.format == "PNG"


def test_final_encode_filter_file_is_not_written_through_a_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storyboard = Storyboard.model_validate(
        {
            "name": "x",
            "format": {"width": 1080, "height": 1920, "fps": 30},
            "emoji": "none",
            "segments": [{"type": "image", "source": "footage/a.png", "duration": 1}],
        }
    )
    output = tmp_path / "renders" / "x.mp4"
    victim = _victim(tmp_path)
    graph = Path(f"{output}.tmp.filters.txt")
    _plant(graph, victim)
    seen: dict[str, object] = {}

    def fake_run(args, **kwargs):
        seen["is_link"] = graph.is_symlink()
        seen["text"] = graph.read_text(encoding="utf-8")
        Path(f"{output}.tmp.mp4").write_bytes(b"video")

    monkeypatch.setattr(compose_module, "run", fake_run)
    compose_module._render_final(
        storyboard, tmp_path / "concat.mp4", [], tmp_path / "bed.m4a", output, 1.0, "final"
    )
    assert victim.read_text(encoding="utf-8") == "KEEP"
    assert seen["is_link"] is False and "[vout]" in str(seen["text"])
    assert not graph.exists()


def test_reveal_filter_file_is_not_written_through_a_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import adanvil.media.reveal as reveal_module

    segment = RevealSegment.model_validate(
        {
            "type": "reveal",
            "duration": 2,
            "before": {"source": "footage/a.png"},
            "after": {"source": "footage/b.png"},
        }
    )
    output = tmp_path / "reveal.mp4"
    graph = Path(f"{output}.filters.txt")
    victim = _victim(tmp_path)
    _plant(graph, victim)
    seen: dict[str, object] = {}

    def fake_run(args, **kwargs):
        seen["is_link"] = graph.is_symlink()
        seen["text"] = graph.read_text(encoding="utf-8")

    monkeypatch.setattr(reveal_module, "run", fake_run)
    reveal_module._build_reveal(
        [tmp_path / "a.png", tmp_path / "b.png"], [], None, segment, Format(), output
    )
    assert victim.read_text(encoding="utf-8") == "KEEP"
    assert seen["is_link"] is False and "[v]" in str(seen["text"])


def test_breakdown_json_is_not_written_through_a_link(tmp_path: Path) -> None:
    from adanvil.reference import breakdown_reference
    from tests.test_reference import _decoded_with_shots, _entry

    _decoded_with_shots(tmp_path, 1)
    victim = _victim(tmp_path)
    saved = tmp_path / "reference" / "x" / "breakdown.json"
    _plant(saved, victim)
    breakdown_reference(tmp_path, "reference/x", [_entry([1], "hook")])
    assert victim.read_text(encoding="utf-8") == "KEEP"
    assert not saved.is_symlink() and '"shots"' in saved.read_text(encoding="utf-8")


def test_reference_json_is_not_written_through_a_link(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    from adanvil.reference import decode_reference

    destination = tmp_path / "reference_ads" / "ref_ad.mp4"
    destination.parent.mkdir(parents=True)
    shutil.copy2(Path(str(media["ref_ad.mp4"]["path"])), destination)
    victim = _victim(tmp_path)
    saved = tmp_path / "reference" / "planted" / "reference.json"
    _plant(saved, victim)
    decode_reference(tmp_path, "reference_ads/ref_ad.mp4", out="reference/planted")
    assert victim.read_text(encoding="utf-8") == "KEEP"
    assert not saved.is_symlink() and '"shots"' in saved.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# B7. bounds that computed values cannot bypass
# ---------------------------------------------------------------------------------------------


def test_format_is_bounded() -> None:
    assert Format(width=4096, height=4096, fps=120).fps == 120
    for changes in ({"width": 4097}, {"height": 5000}, {"fps": 121}):
        with pytest.raises(ValidationError):
            Format(**changes)


def test_a_slow_speed_cannot_stretch_a_clip_past_the_limit() -> None:
    ok = ClipSegment(source="a.mp4", **{"in": 0, "out": 100, "speed": 0.25})  # 400 s
    assert ok.speed == 0.25
    with pytest.raises(ValidationError, match="at most 600"):
        ClipSegment(source="a.mp4", **{"in": 0, "out": 200, "speed": 0.25})  # 800 s
    with pytest.raises(ValidationError, match="at most 600"):
        ClipSegment(source="a.mp4", **{"in": 0, "out": 590, "freeze": {"hold": 30}})
    with pytest.raises(ValidationError):
        ClipSegment(source="a.mp4", **{"in": 0, "out": 1, "freeze": {"hold": 601}})
    with pytest.raises(ValidationError):
        ClipSegment(source="a.mp4", **{"in": 0, "out": float("inf")})


def test_other_segment_durations_are_bounded() -> None:
    assert ImageSegment(source="a.png", duration=600).duration == 600
    with pytest.raises(ValidationError):
        ImageSegment(source="a.png", duration=601)
    panels = [{"source": "a.mp4"}, {"source": "b.mp4"}]
    with pytest.raises(ValidationError):
        SplitSegment.model_validate({"type": "split", "duration": 601, "panels": panels})
    with pytest.raises(ValidationError):
        RevealSegment.model_validate(
            {"type": "reveal", "duration": 601, "before": panels[0], "after": panels[1]}
        )
    with pytest.raises(ValidationError):
        SplitSegment.model_validate(
            {
                "type": "split",
                "duration": 5,
                "panels": [{**panels[0], "freeze": {"hold": 700}}, panels[1]],
            }
        )


def _long_storyboard() -> Storyboard:
    return Storyboard.model_validate(
        {
            "name": "long",
            "format": {"width": 1080, "height": 1920, "fps": 30},
            "emoji": "none",
            "segments": [
                {"type": "image", "source": "footage/a.png", "duration": 400},
                {"type": "image", "source": "footage/a.png", "duration": 400},
            ],
        }
    )


def test_compose_refuses_an_ad_longer_than_the_limit_before_building_anything(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="at most 600"):
        compose(_long_storyboard(), root=tmp_path)
    assert not (tmp_path / "cache").exists()


def test_validation_reports_a_storyboard_longer_than_the_limit(tmp_path: Path) -> None:
    report = validate_storyboard(_long_storyboard(), root=tmp_path, check_files=False)
    assert [issue.code for issue in report.errors if issue.code == "too_long"] == ["too_long"]


def test_audio_durations_are_bounded(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at most 600"):
        silence(601, tmp_path / "silence.m4a")


def test_reference_longer_than_the_limit_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from adanvil.reference import decode as decode_module
    from adanvil.reference import decode_reference

    video = tmp_path / "reference_ads" / "long.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    fake = ffmpeg_module.MediaInfo(
        path=video, kind="video", width=1080, height=1920, stored_width=1080,
        stored_height=1920, rotation=0, duration=700.0, fps=30.0, nb_frames=None,
        pix_fmt="yuv420p", color_transfer=None, color_primaries=None, color_space=None,
        has_audio=False, audio_duration=None, audio_sample_rate=None, audio_channels=None,
    )
    monkeypatch.setattr(decode_module.ffmpeg, "probe", lambda path: fake)
    with pytest.raises(ValueError, match="at most 600"):
        decode_reference(tmp_path, "reference_ads/long.mp4")


# ---------------------------------------------------------------------------------------------
# End to end: an awkward project folder, a network font, a planted filter file
# ---------------------------------------------------------------------------------------------


def test_compose_in_an_awkward_folder_with_a_network_font_and_a_planted_link(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    root = tmp_path / AWKWARD
    footage = root / "footage"
    footage.mkdir(parents=True)
    (root / "renders").mkdir()
    for name in ("burnt_text.mp4", "cta.png"):
        shutil.copy2(Path(str(media[name]["path"])), footage / name)
    victim = _victim(tmp_path)
    resolved = root.resolve()
    _plant(Path(f"{resolved / 'renders' / 'awkward.mp4'}.tmp.filters.txt"), victim)
    storyboard = Storyboard.model_validate(
        {
            "name": "awkward",
            "format": {"width": 1080, "height": 1920, "fps": 30},
            "emoji": "none",
            "segments": [
                {
                    "type": "clip",
                    "source": "footage/burnt_text.mp4",
                    "in": 0.5,
                    "out": 1.2,
                    "remove_text": {"method": "mask", "box": [316, 286, 771, 460]},
                },
                {"type": "image", "source": "footage/cta.png", "duration": 0.8},
            ],
            "overlays": [
                {
                    "type": "caption", "style": "pill", "lines": ["Hello"], "start": 0, "end": 1,
                    "y": 330, "family": UNC_FONT,
                }
            ],
            "output": {"path": "renders/{name}.mp4", "quality": "final"},
        }
    )
    report = compose(storyboard, root=root)
    assert probe(report.path).duration == pytest.approx(1.5, abs=0.15)
    assert any("default family" in warning for warning in report.warnings)
    assert victim.read_text(encoding="utf-8") == "KEEP"

