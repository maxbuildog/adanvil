"""Compose a storyboard into a final video and contact sheet."""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

from PIL import Image
from pydantic import BaseModel, ConfigDict

from adanvil import config
from adanvil.audio import (
    MUSIC_REFERENCE_LUFS,
    AudioPiece,
    SfxHit,
    clips_bed,
    loudness_normalize,
    mix_sfx,
    music_bed,
    reference_bed,
    silence,
    voice_music_mix,
)
from adanvil.brand import brand_accent, brand_accent_color
from adanvil.ffmpeg import MAX_SECONDS, filter_script_option, probe, run
from adanvil.media import (
    Cache,
    ClipSegment,
    RevealSegment,
    SplitSegment,
    contact_sheet,
    render_segment,
    segment_duration,
    tonemap,
)
from adanvil.music import music_licence
from adanvil.project import resolve_in_project
from adanvil.safeio import replace_text
from adanvil.sfx import resolve_sfx
from adanvil.speech.analyze import brand_filler_lists
from adanvil.speech.captions import expand_speech_captions
from adanvil.speech.library import SpeechLibrary
from adanvil.storyboard import (
    CaptionOverlay,
    CommentOverlay,
    ImageOverlay,
    ProgressOverlay,
    SfxOverlay,
    SpeechCaptionsOverlay,
    StickerOverlay,
    Storyboard,
    VideoOverlay,
    VoiceCaptionsOverlay,
)
from adanvil.storyboard.schema import normalise_quality
from adanvil.storyboard.sfx import all_sfx
from adanvil.text import get_provider
from adanvil.text import render as render_text
from adanvil.text.fonts import unusable_family_message, usable_family
from adanvil.voice import VoiceTrack, load_track

from .animated import Placement, animated_placements
from .cards import card_placements
from .inset import (
    InsetLayer,
    inner_size,
    inset_decor,
    inset_footprint,
    inset_mask,
    inset_size,
    inset_span,
    progress_pictures,
    progress_x_expression,
)
from .stickers import sticker_placements


class RenderReport(BaseModel):
    """JSON-serialisable summary of one composition."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    sheet: Path
    duration: float
    timeline: list[dict[str, Any]]
    overlays: list[dict[str, Any]]
    warnings: list[str]
    quality: Literal["preview", "final"]
    voice: dict[str, Any] | None = None
    sfx: list[dict[str, Any]] | None = None
    # Licence of ``audio.music``: status, licence, attribution (see ``adanvil.music``).
    music: dict[str, Any] | None = None


def _resolve_project_path(root: Path, value: str, field_name: str) -> Path:
    try:
        return resolve_in_project(root, value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name}: {exc}") from exc


def _cache_build(
    cache: Cache,
    kind: str,
    key: dict[str, Any],
    suffix: str,
    builder: Callable[[Path], None],
) -> Path:
    final = cache.path_for(kind, key, suffix=suffix)
    if final.is_file():
        return final
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{final}.tmp{final.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        builder(temporary)
        os.replace(temporary, final)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return final


def _number(value: float) -> str:
    return format(float(value), ".9g")


def _ffconcat_quote(path: Path) -> str:
    # ffconcat accepts forward slashes on Windows. Inside single quotes everything is literal,
    # so a quote is written as '\'' (close the quote, an escaped quote, reopen the quote).
    value = path.resolve().as_posix()
    if "\n" in value or "\r" in value:
        raise ValueError(
            "a project folder name cannot contain a line break (the ffconcat list reads one "
            f"directive per line): {value!r}"
        )
    escaped = value.replace("'", "'\\''")
    return f"file '{escaped}'"


def _concat_mezzanines(paths: list[Path], cache: Cache) -> Path:
    key = {"paths": [str(path.resolve()) for path in paths]}
    list_path = cache.path_for("concat", key, suffix=".ffconcat")
    # Written anew for each build, never reused as found: a list planted in ``cache/concat/``
    # would otherwise be read with ``-safe 0``.
    text = "ffconcat version 1.0\n" + "\n".join(_ffconcat_quote(path) for path in paths) + "\n"

    def build(output: Path) -> None:
        replace_text(list_path, text, newline="\n")
        run(["-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(output)])

    return cache.get_or_build("concat-video", key, build)


def _drawable_font(overlay: Any, root: Path, warnings: list[str]) -> Any:
    """``overlay`` with a ``family`` it may be drawn with: a font that is neither bundled nor a
    file of the project (a UNC or absolute path, a missing file) becomes the default family."""

    family = getattr(overlay, "family", None)
    if family is None:
        return overlay
    usable = usable_family(family, root)
    if usable is None:
        message = unusable_family_message(family)
        if message not in warnings:
            warnings.append(message)
    return overlay if usable == family else overlay.model_copy(update={"family": usable})


def _emoji_provider(storyboard: Storyboard, root: Path):
    if storyboard.emoji in {"twemoji", "none"}:
        return get_provider(storyboard.emoji)
    directory = _resolve_project_path(root, storyboard.emoji, "storyboard.emoji")
    return get_provider("directory", path=directory)


def _caption_png(
    overlay: CaptionOverlay | CommentOverlay,
    storyboard: Storyboard,
    provider: object,
    cache: Cache,
    warnings: list[str],
    highlight: str | None = None,
) -> Path:
    spec = (
        overlay.to_text_spec(highlight)
        if isinstance(overlay, CaptionOverlay)
        else overlay.to_text_spec()
    )
    key = {
        "spec": spec.model_dump(mode="json"),
        "overlay": overlay.model_dump(mode="json"),
        "emoji": storyboard.emoji,
        "canvas": [storyboard.format.width, storyboard.format.height],
    }
    output = cache.path_for("overlays", key, suffix=".png")
    result = render_text(
        spec,
        provider,
        canvas=(storyboard.format.width, storyboard.format.height),
    )
    if result.overflow:
        warnings.append(
            f"{overlay.type} overlay at {overlay.start:g}s overflowed its minimum text size"
        )
    if not output.is_file():
        result.save(output)
    return output


def _image_overlay_png(
    overlay: ImageOverlay,
    source: Path,
    storyboard: Storyboard,
    cache: Cache,
) -> Path:
    key = {
        "overlay": overlay.model_dump(mode="json"),
        "source": Cache.file_key(source),
        "canvas": [storyboard.format.width, storyboard.format.height],
    }
    output = cache.path_for("overlays", key, suffix=".png")
    if output.is_file():
        return output

    with Image.open(source) as source_image:
        image = source_image.convert("RGBA")
    if overlay.width is not None and image.width != overlay.width:
        height = max(1, round(image.height * overlay.width / image.width))
        image = image.resize((overlay.width, height), Image.Resampling.LANCZOS)
    canvas = Image.new(
        "RGBA",
        (storyboard.format.width, storyboard.format.height),
        (0, 0, 0, 0),
    )
    canvas.paste(image, (overlay.x, overlay.y), image)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{output}.tmp{output.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        canvas.save(temporary, format="PNG")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def _clip_pieces(storyboard: Storyboard, root: Path, durations: list[float]) -> list[AudioPiece]:
    """The sound each segment brings to the clip-sound bed, one piece per segment.

    A clip plays its own audio from ``in`` to ``out`` (stretched by its speed); with
    ``audio_from`` it plays the other clip instead, at normal speed, for the whole segment.
    An image, a clip without audio or the frozen hold at the end of a clip is silence.
    """

    pieces: list[AudioPiece] = []
    for segment, duration in zip(storyboard.segments, durations, strict=True):
        if not isinstance(segment, ClipSegment):
            pieces.append(AudioPiece(None, 0.0, 0.0, duration))
        elif segment.audio_from is not None:
            source = _resolve_project_path(root, segment.audio_from.source, "audio_from.source")
            pieces.append(AudioPiece(source, segment.audio_from.in_, duration, duration))
        else:
            source = _resolve_project_path(root, segment.source, "segment.source")
            pieces.append(
                AudioPiece(
                    source, segment.in_, segment.out - segment.in_, duration, segment.speed
                )
            )
    return _with_inset_voices(storyboard, root, pieces, durations)


_SNAP = 0.05  # an inset voice starting or ending this close to a cut is moved onto the cut


def _inset_voices(
    storyboard: Storyboard, root: Path, total: float
) -> list[tuple[float, float, Path, float]]:
    """The ``(start, end, source, in)`` of every video inset that brings its own sound."""

    voices: list[tuple[float, float, Path, float]] = []
    for overlay in storyboard.overlays:
        if isinstance(overlay, VideoOverlay) and overlay.audio:
            source = _resolve_project_path(root, overlay.source, "overlay.source")
            info = probe(source)
            start, end = inset_span(overlay, info.duration, total)
            if end > start and info.has_audio:
                voices.append((start, end, source, overlay.in_))
    return voices


def _with_inset_voices(
    storyboard: Storyboard, root: Path, pieces: list[AudioPiece], durations: list[float]
) -> list[AudioPiece]:
    """The pieces of the clip-sound bed with the sound of the audio insets laid in.

    Under an inset that brings its sound the bed is that sound (the segments' own sound is
    left out for that stretch); everywhere else it is the segments' sound, cut at the same
    places as before. Without an audio inset the pieces come back untouched.
    """

    total = sum(durations)
    voices = _inset_voices(storyboard, root, total)
    if not voices:
        return pieces
    starts = [0.0]
    for duration in durations:
        starts.append(starts[-1] + duration)

    def snapped(moment: float) -> float:
        for boundary in starts:
            if abs(moment - boundary) < _SNAP:
                return boundary
        return moment

    spans = [
        (snapped(a), snapped(b), source, in_ + (snapped(a) - a)) for a, b, source, in_ in voices
    ]
    cuts = sorted({*starts, *(point for a, b, _s, _i in spans for point in (a, b))})
    result: list[AudioPiece] = []
    for left, right in zip(cuts, cuts[1:], strict=False):
        if right - left < 1e-6:
            continue
        voice = next((v for v in spans if v[0] <= left + 1e-6 and right <= v[1] + 1e-6), None)
        if voice is not None:
            offset = voice[3] + (left - voice[0])
            result.append(AudioPiece(voice[2], offset, right - left, right - left))
            continue
        index = max(i for i, begin in enumerate(starts[:-1]) if begin <= left + 1e-6)
        piece = pieces[index]
        into = (left - starts[index]) * piece.speed
        length = max(0.0, min(piece.length - into, (right - left) * piece.speed))
        if piece.source is None or length <= 1e-6:
            result.append(AudioPiece(None, 0.0, 0.0, right - left))
        else:
            result.append(
                AudioPiece(piece.source, piece.start + into, length, right - left, piece.speed)
            )
    return result


def _clips_audio_bed(
    storyboard: Storyboard, root: Path, cache: Cache, duration: float, durations: list[float]
) -> Path:
    """``audio.clips``: the original sound of the segments, normalised, with music under it."""

    audio = storyboard.audio
    pieces = _clip_pieces(storyboard, root, durations)
    piece_key = [
        {
            "source": None if piece.source is None else Cache.file_key(piece.source),
            "start": piece.start,
            "length": piece.length,
            "duration": piece.duration,
            "speed": piece.speed,
        }
        for piece in pieces
    ]
    sound = _cache_build(
        cache,
        "audio-clips",
        {"pieces": piece_key},
        ".wav",
        lambda output: clips_bed(pieces, output),
    )
    music: Path | None = None
    music_key: dict[str, Any] | None = None
    if audio.music is not None:
        music = _resolve_project_path(root, audio.music, "audio.music")
        music_key = {"kind": "music", "source": Cache.file_key(music)}
    key = {
        "sound": Cache.file_key(sound),
        "music": music_key,
        "duration": duration,
        "music_start": audio.start,
        "music_volume": audio.music_volume,
        "music_reference_lufs": MUSIC_REFERENCE_LUFS,
        "voice_loudness": audio.voice_loudness,
    }
    return _cache_build(
        cache,
        "audio-clips-mix",
        key,
        ".m4a",
        lambda output: voice_music_mix(
            sound,
            duration,
            output,
            music=music,
            music_start=audio.start,
            music_volume=audio.music_volume,
            voice_loudness=audio.voice_loudness,
        ),
    )


def _audio_bed(
    storyboard: Storyboard,
    root: Path,
    cache: Cache,
    duration: float,
    durations: list[float] | None = None,
) -> Path:
    audio = storyboard.audio
    if audio.clips:
        return _clips_audio_bed(
            storyboard,
            root,
            cache,
            duration,
            durations or [segment_duration(segment) for segment in storyboard.segments],
        )
    if audio.voice is not None:
        manifest = _resolve_project_path(root, audio.voice, "audio.voice")
        track = load_track(manifest)
        voice = manifest.parent / "voice_full.wav"
        if not voice.is_file():
            raise ValueError(f"Voice audio is missing next to manifest: {voice}")
        music: Path | None = None
        music_key: dict[str, Any] | None = None
        if audio.music is not None:
            music = _resolve_project_path(root, audio.music, "audio.music")
            music_key = {"kind": "music", "source": Cache.file_key(music)}
        elif audio.from_reference is not None:
            music = _resolve_project_path(root, audio.from_reference, "audio.from_reference")
            music_key = {"kind": "reference", "source": Cache.file_key(music)}
        key = {
            "manifest": Cache.file_key(manifest),
            "voice_full": Cache.file_key(voice),
            "music": music_key,
            "duration": duration,
            "music_start": audio.start,
            "music_volume": audio.music_volume,
            "music_reference_lufs": MUSIC_REFERENCE_LUFS,
            "voice_loudness": audio.voice_loudness,
            "voice_offset": audio.voice_offset,
            "track_total": track.total_duration,
        }
        return _cache_build(
            cache,
            "audio-voice-mix",
            key,
            ".m4a",
            lambda output: voice_music_mix(
                voice,
                duration,
                output,
                music=music,
                music_start=audio.start,
                music_volume=audio.music_volume,
                voice_loudness=audio.voice_loudness,
                voice_offset=audio.voice_offset,
            ),
        )
    bed = _raw_audio_bed(storyboard, root, cache, duration)
    if audio.loudness is None or (audio.music is None and audio.from_reference is None):
        return bed
    # Separate pass on a standalone file: loudnorm inside a trimmed graph eats the tail.
    key = {"bed": Cache.file_key(bed), "loudness": audio.loudness}
    return _cache_build(
        cache,
        "audio-norm",
        key,
        ".m4a",
        lambda output: loudness_normalize(bed, output, i=audio.loudness, tp=-1.5),
    )


def _raw_audio_bed(storyboard: Storyboard, root: Path, cache: Cache, duration: float) -> Path:
    audio = storyboard.audio
    params = {
        "duration": duration,
        "start": audio.start,
        "volume": audio.volume,
        "fade_in": audio.fade_in,
        "fade_out": audio.fade_out,
    }
    if audio.music is not None:
        source = _resolve_project_path(root, audio.music, "audio.music")
        key = {"source": Cache.file_key(source), "kind": "music", **params}
        return _cache_build(
            cache,
            "audio",
            key,
            ".m4a",
            lambda output: music_bed(
                source,
                duration,
                output,
                start=audio.start,
                volume=audio.volume,
                fade_in=audio.fade_in,
                fade_out=audio.fade_out,
            ),
        )
    if audio.from_reference is not None:
        source = _resolve_project_path(root, audio.from_reference, "audio.from_reference")
        key = {"source": Cache.file_key(source), "kind": "reference", **params}
        return _cache_build(
            cache,
            "audio",
            key,
            ".m4a",
            lambda output: reference_bed(
                source,
                duration,
                output,
                start=audio.start,
                volume=audio.volume,
                fade_in=audio.fade_in,
                fade_out=audio.fade_out,
            ),
        )

    key = {"source": None, "kind": "silence", **params}
    return _cache_build(cache, "audio", key, ".m4a", lambda output: silence(duration, output))


def _with_sfx(
    storyboard: Storyboard,
    root: Path,
    cache: Cache,
    bed: Path,
    starts: list[float],
    total: float,
    warnings: list[str],
) -> tuple[Path, list[dict[str, Any]]]:
    """The bed with the sound effects (``sfx`` overlays and ``audio.sfx_auto``) laid over it.

    This is the last audio step, after the bed was normalised, so a sound keeps its level.
    Returns the new bed (the bed itself when there is no sound) and the report entries.
    """

    entries = all_sfx(storyboard, starts, total, root)
    hits: list[SfxHit] = []
    report: list[dict[str, Any]] = []
    for entry in entries:
        if entry["at"] >= total:
            warnings.append(f"sfx at {entry['at']:g}s starts at or after the video end: skipped")
            continue
        path = resolve_sfx(root, entry["source"])
        if not path.is_file():
            raise ValueError(
                f"sfx source '{entry['source']}' does not exist: run adanvil sfx-fetch or "
                "drop the file in sfx/"
            )
        hits.append(SfxHit(path, float(entry["at"]), float(entry["volume"])))
        report.append({key: entry[key] for key in ("source", "at", "volume", "auto")})
    if not hits:
        return bed, report
    key = {
        "bed": Cache.file_key(bed),
        "duration": total,
        "hits": [
            {"source": Cache.file_key(hit.path), "at": hit.at, "volume": hit.volume}
            for hit in hits
        ],
    }
    mixed = _cache_build(
        cache, "audio-sfx", key, ".wav", lambda output: mix_sfx(bed, hits, total, output)
    )
    return mixed, report


def _even_width(overlay: VideoOverlay) -> int:
    return max(2, overlay.width - overlay.width % 2)


def _shared_directory(paths: list[Path]) -> Path | None:
    """The deepest directory holding all of ``paths`` (``None`` across drives)."""

    try:
        return Path(os.path.commonpath([str(path.resolve().parent) for path in paths]))
    except ValueError:
        return None


def snap_time(value: float, fps: float, cuts: list[float]) -> float:
    """``value`` (seconds) on the frame grid: the nearest frame, and the frame of a cut when
    ``value`` is within half a frame of it. A caption timed 7.9003 s over a cut that lands on
    frame 237 (7.9 s) starts ON that frame, not one frame after it."""

    for cut in cuts:
        if abs(value - cut) <= 0.5 / fps + 1e-9:
            return round(cut * fps) / fps
    return round(value * fps) / fps


def snap_layers(
    layers: list[Placement | InsetLayer], fps: float, cuts: list[float], total: float
) -> list[Placement | InsetLayer]:
    """The overlay pictures and insets with their start and end on the frame grid of the cuts
    (see ``snap_time``); a span keeps at least one frame and never goes past the end."""

    ceiling = round(total * fps) / fps
    snapped: list[Placement | InsetLayer] = []
    for layer in layers:
        start = snap_time(layer.start, fps, cuts)
        end = min(snap_time(layer.end, fps, cuts), max(ceiling, start + 1.0 / fps))
        end = max(end, start + 1.0 / fps)
        snapped.append(replace(layer, start=start, end=end))
    return snapped


def _render_final(
    storyboard: Storyboard,
    concat: Path,
    overlay_paths: list[Placement | InsetLayer],
    audio: Path,
    output: Path,
    total: float,
    quality: Literal["preview", "final"],
) -> None:
    # Hundreds of overlay pictures (animated captions) would not fit in a Windows command line:
    # the inputs are given relative to the directory they share and the filter graph goes in a
    # file read by ffmpeg.
    files = [concat, audio]
    for item in overlay_paths:
        if isinstance(item, InsetLayer):
            files.append(item.source)
            if item.mask is not None:
                files.append(item.mask)
        else:
            files.append(item.png)
    base = _shared_directory(files)

    def given(path: Path) -> str:
        if base is None:
            return str(path)
        return Path(os.path.relpath(path.resolve(), base)).as_posix()

    args = ["-i", given(concat)]
    # Each overlay image is only decoded while it is on screen (``-t`` = its own
    # duration, shifted to its start with setpts): looping every image over the
    # whole ad made renders with many word-by-word captions several times slower.
    fps = storyboard.format.fps
    half_frame = 0.5 / fps  # a frame at t is on when t is inside [start, end) with this margin
    slots: list[tuple[Placement | InsetLayer, int, int | None]] = []
    next_input = 1
    for placement in overlay_paths:
        if isinstance(placement, InsetLayer):
            span = max(placement.end - placement.start, 1.0 / fps)
            args.extend(
                ["-ss", _number(placement.in_), "-t", _number(span), "-i", given(placement.source)]
            )
            video_index = next_input
            next_input += 1
            mask_index = None
            if placement.mask is not None:
                args.extend(
                    [
                        "-loop",
                        "1",
                        "-framerate",
                        str(fps),
                        "-t",
                        _number(span + 1.0),
                        "-i",
                        given(placement.mask),
                    ]
                )
                mask_index = next_input
                next_input += 1
            slots.append((placement, video_index, mask_index))
            continue
        # +0.5 s margin: ffmpeg 6.1 ends a looped image one frame early, and ``enable`` below
        # still cuts the caption at its exact end. The states of an animated caption are short
        # and many, and follow each other, so they only need a few frames of margin.
        margin = 3.0 / fps if placement.exact else 0.5
        args.extend(
            [
                "-loop",
                "1",
                "-framerate",
                str(fps),
                "-t",
                _number(max(placement.end - placement.start, 1.0 / fps) + margin),
                "-i",
                given(placement.png),
            ]
        )
        slots.append((placement, next_input, None))
        next_input += 1
    audio_index = next_input
    args.extend(["-i", given(audio)])

    filter_steps: list[str] = []
    previous = "[0:v]"
    for index, (placement, source_index, mask_index) in enumerate(slots, start=1):
        start, end = placement.start, placement.end
        output_label = f"[v{index}]"
        if isinstance(placement, InsetLayer):
            span = max(end - start, 1.0 / fps)
            chain = [f"[{source_index}:v]setpts=PTS-STARTPTS"]
            if placement.hdr_filter:
                chain.append(placement.hdr_filter)
            chain.extend(
                [
                    f"fps={fps}",
                    f"scale={placement.width}:{placement.height}:"
                    "force_original_aspect_ratio=increase",
                    f"crop={placement.width}:{placement.height}",
                    "setsar=1",
                    f"tpad=stop_mode=clone:stop_duration={_number(span)}",
                    f"trim=duration={_number(span)}",
                    "setpts=PTS-STARTPTS",
                    "format=yuv420p",
                ]
            )
            if mask_index is None:
                filter_steps.append(",".join(chain) + f",setpts=PTS+{_number(start)}/TB[o{index}]")
            else:
                filter_steps.append(",".join(chain) + f"[iv{index}]")
                filter_steps.append(
                    f"[iv{index}][{mask_index}:v]alphamerge,"
                    f"setpts=PTS+{_number(start)}/TB[o{index}]"
                )
        else:
            filter_steps.append(f"[{source_index}:v]setpts=PTS+{_number(start)}/TB[o{index}]")
        if placement.exact if isinstance(placement, Placement) else True:
            # A state is on screen on [start, end) and the next one starts at that very number,
            # so no frame is left without a picture between two states.
            enable = f"gte(t,{_number(start - half_frame)})*lt(t,{_number(end - half_frame)})"
        else:
            enable = f"gte(t,{_number(start - half_frame)})*lt(t,{_number(end - half_frame)})"
        x_value = (
            f"'{placement.x_expr}'"
            if isinstance(placement, Placement) and placement.x_expr
            else str(placement.x)
        )
        filter_steps.append(
            f"{previous}[o{index}]overlay={x_value}:{placement.y}:eof_action=pass:"
            f"enable='{enable}'[v{index}]"
        )
        previous = output_label
    if quality == "preview":
        tail = f"{previous}scale=trunc(iw/4)*2:trunc(ih/4)*2,format=yuv420p[vout]"
    else:
        tail = f"{previous}format=yuv420p[vout]"
    filter_steps.append(tail)
    chain = ";".join(filter_steps)

    temporary = Path(f"{output}.tmp.mp4")
    temporary.unlink(missing_ok=True)
    graph = Path(f"{output}.tmp.filters.txt")
    replace_text(graph, chain, encoding="utf-8")
    args.extend(
        [
            filter_script_option(),
            str(graph),
            "-map",
            "[vout]",
            "-map",
            f"{audio_index}:a:0",
            "-t",
            _number(total),
            "-r",
            str(storyboard.format.fps),
            "-c:v",
            "libx264",
            "-preset",
            "medium" if quality == "final" else "veryfast",
            "-crf",
            "19" if quality == "final" else "26",
            "-pix_fmt",
            "yuv420p",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-x264-params",
            "colorprim=bt709:transfer=bt709:colormatrix=bt709",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
    )
    try:
        run(args, cwd=base)
    finally:
        graph.unlink(missing_ok=True)
    os.replace(temporary, output)


def build_overlays(
    sb: Storyboard,
    *,
    root: Path,
    cache: Cache,
    timeline: list[dict[str, Any]],
    total: float,
    warnings: list[str],
) -> tuple[list[Placement | InsetLayer], list[dict[str, Any]], VoiceTrack | None]:
    """Every overlay of the storyboard as the final encode draws it: the pictures with their
    spans and places, the report entries (each tagged with the index of its storyboard overlay)
    and the voice track. Voice and speech captions are expanded to their final times."""

    provider = _emoji_provider(sb, root)
    report_overlays: list[dict[str, Any]] = []
    final_overlays: list[Placement | InsetLayer] = []
    expanded_overlays: list[
        CaptionOverlay
        | CommentOverlay
        | ImageOverlay
        | VideoOverlay
        | StickerOverlay
        | ProgressOverlay
    ] = []
    voice_track: VoiceTrack | None = None
    if sb.audio.voice is not None:
        voice_manifest = _resolve_project_path(root, sb.audio.voice, "audio.voice")
        voice_track = load_track(voice_manifest)
    speech_library: SpeechLibrary | None = None
    # ``source_of[k]`` is where the k-th kept storyboard overlay starts in the expanded list,
    # ``source_indexes[k]`` its index in the storyboard: the report names each entry by it.
    source_of: list[int] = []
    source_indexes: list[int] = []
    for source_index, overlay in enumerate(sb.overlays):
        if isinstance(overlay, SfxOverlay):
            continue  # a sound: mixed into the audio bed below
        overlay = _drawable_font(overlay, root, warnings)
        source_of.append(len(expanded_overlays))
        source_indexes.append(source_index)
        if isinstance(overlay, SpeechCaptionsOverlay):
            if not sb.audio.clips:
                raise ValueError("speech_captions overlay requires audio.clips: true")
            if speech_library is None:
                speech_library = SpeechLibrary(root, brand_filler_lists(root))
            expanded_overlays.extend(
                expand_speech_captions(
                    overlay,
                    sb,
                    speech_library,
                    [item["start"] for item in timeline],
                    [item["end"] for item in timeline],
                )
            )
        elif isinstance(overlay, VoiceCaptionsOverlay):
            manifest_value = overlay.manifest or sb.audio.voice
            if manifest_value is None:
                raise ValueError("voice_captions overlay requires audio.voice or manifest")
            manifest_path = _resolve_project_path(root, manifest_value, "voice_captions.manifest")
            track = voice_track if manifest_value == sb.audio.voice else load_track(manifest_path)
            expanded_overlays.extend(
                overlay.expand(
                    track,
                    sb.audio.voice_offset if overlay.offset is None else overlay.offset,
                )
            )
        else:
            expanded_overlays.append(overlay)

    accent = brand_accent(root)
    badge_color = brand_accent_color(root)
    tagged_from = 0
    tagged_source = 0
    for position, overlay in enumerate(expanded_overlays):
        owner = max(
            (k for k, begin in enumerate(source_of) if begin <= position), default=0
        )
        for entry in report_overlays[tagged_from:]:
            entry.setdefault("overlay", tagged_source)
        tagged_from = len(report_overlays)
        tagged_source = source_indexes[owner] if source_indexes else 0
        if overlay.start >= total:
            warnings.append(
                f"{overlay.type} overlay at {overlay.start:g}s starts at or after video end"
            )
            continue
        end = total if overlay.end is None else min(overlay.end, total)
        if end <= overlay.start:
            warnings.append(f"{overlay.type} overlay has no visible time after clamping")
            continue
        if isinstance(overlay, CaptionOverlay) and overlay.animated:
            for chunk in overlay.expand_words(end):
                chunk_end = end if chunk.end is None else min(chunk.end, end)
                if chunk_end <= chunk.start:
                    continue
                placements, entry = animated_placements(
                    chunk,
                    chunk.start,
                    chunk_end,
                    storyboard=sb,
                    provider=provider,
                    cache=cache,
                    warnings=warnings,
                    highlight=accent,
                )
                if placements:
                    entry.setdefault("text", " ".join(chunk.lines))
                    final_overlays.extend(placements)
                    report_overlays.append(entry)
            continue
        if isinstance(overlay, VideoOverlay):
            if overlay.audio and not sb.audio.clips:
                raise ValueError("a video overlay with audio needs audio.clips: true")
            source = _resolve_project_path(root, overlay.source, "overlay.source")
            info = probe(source)
            if info.kind != "video" or not info.width or not info.height:
                raise ValueError(f"video overlay source '{overlay.source}' is not a video")
            inset_start, inset_end = inset_span(overlay, info.duration, total)
            if inset_end <= inset_start:
                warnings.append("video overlay has no visible time after clamping")
                continue
            width, height = inset_size(overlay, info.width, info.height)
            x0, y0, _x1, _y1 = inset_footprint(overlay, height)
            border = overlay.border.width if overlay.border is not None else 0
            decor = inset_decor(overlay, height, cache)
            if decor is not None:
                decor_png, (dx, dy) = decor
                final_overlays.append(
                    Placement(decor_png, inset_start, inset_end, x0 + dx, y0 + dy, exact=True)
                )
            inner_width, inner_height = inner_size(overlay, height)
            final_overlays.append(
                InsetLayer(
                    source,
                    overlay.in_,
                    inset_start,
                    inset_end,
                    x0 + border,
                    y0 + border,
                    inner_width,
                    inner_height,
                    inset_mask(overlay, height, cache),
                    tonemap(info.color_transfer or "") if info.is_hdr else "",
                )
            )
            report_overlays.append(
                {
                    "type": "video",
                    "start": inset_start,
                    "end": inset_end,
                    "png": None,
                    "bbox": [x0, y0, x0 + _even_width(overlay), y0 + height],
                    "shape": overlay.shape,
                    "canvas": [sb.format.width, sb.format.height],
                }
            )
            continue
        if isinstance(overlay, StickerOverlay):
            sticker_list, sticker_entry = sticker_placements(
                overlay,
                overlay.start,
                end,
                canvas=(sb.format.width, sb.format.height),
                fps=sb.format.fps,
                emoji_name=sb.emoji,
                provider=provider,
                cache=cache,
                root=root,
                accent=accent,
                badge_color=badge_color,
            )
            if sticker_list:
                final_overlays.extend(sticker_list)
                report_overlays.append(sticker_entry)
            continue
        if isinstance(overlay, ProgressOverlay):
            fill, track = progress_pictures(
                overlay, sb.format.width, overlay.color or accent or "#FFFFFF", cache
            )
            bar_y = 0 if overlay.position == "top" else sb.format.height - overlay.height
            bar_y -= bar_y % 2
            if track is not None:
                final_overlays.append(Placement(track, overlay.start, end, 0, bar_y, exact=True))
            final_overlays.append(
                Placement(
                    fill,
                    overlay.start,
                    end,
                    0,
                    bar_y,
                    exact=True,
                    x_expr=progress_x_expression(overlay.start, end, sb.format.fps),
                )
            )
            report_overlays.append(
                {
                    "type": "progress",
                    "start": overlay.start,
                    "end": end,
                    "png": None,
                    "bbox": [0, bar_y, sb.format.width, bar_y + overlay.height],
                    "canvas": [sb.format.width, sb.format.height],
                }
            )
            continue
        if isinstance(overlay, ImageOverlay):
            source = _resolve_project_path(root, overlay.source, "overlay.source")
            if not overlay.plain:
                card_list, card_entry = card_placements(
                    overlay,
                    source,
                    overlay.start,
                    end,
                    canvas=(sb.format.width, sb.format.height),
                    fps=sb.format.fps,
                    cache=cache,
                )
                if card_list:
                    final_overlays.extend(card_list)
                    report_overlays.append(card_entry)
                continue
            png = _image_overlay_png(overlay, source, sb, cache)
        else:
            png = _caption_png(overlay, sb, provider, cache, warnings, accent)
        final_overlays.append(Placement(png, overlay.start, end))
        written: dict[str, Any] = {
            "type": overlay.type,
            "start": overlay.start,
            "end": end,
            "png": png,
        }
        if isinstance(overlay, CaptionOverlay):
            written["text"] = " ".join(overlay.lines)
        elif isinstance(overlay, CommentOverlay):
            written["text"] = overlay.text
        report_overlays.append(written)

    for entry in report_overlays[tagged_from:]:
        entry.setdefault("overlay", tagged_source)

    return final_overlays, report_overlays, voice_track


def compose(
    sb: Storyboard,
    *,
    root: Path,
    cache: Cache | None = None,
    quality: str | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> RenderReport:
    """Render all storyboard stages and return a report for the final ad.

    ``progress`` receives JSON-shaped events after each segment and when the
    final encode begins and ends.
    """

    root = Path(root).resolve()
    expected_total = sum(segment_duration(segment) for segment in sb.segments)
    if not expected_total <= MAX_SECONDS:
        raise ValueError(
            f"The ad would last {expected_total:g}s: AdAnvil renders at most {MAX_SECONDS:g}s. "
            "Remove segments or shorten them."
        )
    cache = cache or Cache(root / "cache")
    selected_quality = normalise_quality(quality) or sb.output.quality
    if selected_quality not in {"preview", "final"}:
        raise ValueError("quality must be 'preview' or 'final'")
    selected_quality_value = cast(Literal["preview", "final"], selected_quality)

    warnings: list[str] = []
    mezzanines: list[Path] = []
    timeline: list[dict[str, Any]] = []
    real_durations: list[float] = []
    start = 0.0
    frame_tolerance = 1.0 / sb.format.fps
    # Mezzanines are built by a pool; they are collected in storyboard order, so the
    # timeline, warnings and progress events read exactly as in a sequential render.
    # The mezzanine format does not depend on the quality, so a final render after a
    # preview (or the reverse) finds every segment in the cache.
    with ThreadPoolExecutor(max_workers=max(1, min(config.workers(), len(sb.segments)))) as pool:
        futures = [
            pool.submit(render_segment, segment, root=root, cache=cache, fmt=sb.format)
            for segment in sb.segments
        ]
        try:
            for index, segment in enumerate(sb.segments):
                mezzanine = futures[index].result()
                mezzanines.append(mezzanine)
                expected = segment_duration(segment)
                info = probe(mezzanine)
                actual = info.duration or expected
                if abs(actual - expected) > frame_tolerance:
                    warnings.append(
                        f"segment {index} duration {actual:.3f}s differs from "
                        f"expected {expected:.3f}s by more than one frame"
                    )
                end = start + actual
                entry = {
                    "label": getattr(segment, "label", None) or Path(segment.source).stem,
                    "start": start,
                    "end": end,
                    "source": segment.source,
                }
                if isinstance(segment, SplitSegment):
                    entry["type"] = "split"
                elif isinstance(segment, RevealSegment):
                    entry["type"] = "reveal"
                    entry["direction"] = segment.direction
                    entry["hold_before"] = segment.hold_before
                    entry["sweep"] = segment.sweep
                timeline.append(entry)
                real_durations.append(actual)
                start = end
                if progress is not None:
                    progress(
                        {
                            "stage": "segments",
                            "done": index + 1,
                            "total": len(sb.segments),
                            "unit": "segments",
                            "message": f"rendered segment {index + 1}/{len(sb.segments)}",
                        }
                    )
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    total = sum(real_durations)
    concat = _concat_mezzanines(mezzanines, cache)
    final_overlays, report_overlays, voice_track = build_overlays(
        sb, root=root, cache=cache, timeline=timeline, total=total, warnings=warnings
    )

    bed = _audio_bed(sb, root, cache, total, real_durations)
    bed, sfx_report = _with_sfx(
        sb, root, cache, bed, [item["start"] for item in timeline], total, warnings
    )
    output_value = sb.output.path.format(name=sb.name)
    output = _resolve_project_path(root, output_value, "output.path")
    if selected_quality_value == "preview":
        # A preview must never overwrite the delivered final render.
        output = output.with_name(f"{output.stem}.preview{output.suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # ``renders/`` is checked when the project opens; a folder inside it could still be a link
    # planted in a received project, and the render, its report and its sheet go there.
    real_root = os.path.normcase(os.path.realpath(root))
    real_parent = os.path.normcase(os.path.realpath(output.parent))
    try:
        inside = os.path.commonpath((real_root, real_parent)) == real_root
    except ValueError:  # another drive
        inside = False
    if not inside:
        raise ValueError(
            f"output.path: the folder {output.parent} leads outside the project through a link"
        )
    if progress is not None:
        progress(
            {
                "stage": "encode",
                "done": len(sb.segments),
                "total": len(sb.segments),
                "unit": "segments",
                "message": "encoding final video",
            }
        )
    _render_final(
        sb,
        concat,
        snap_layers(
            final_overlays, sb.format.fps, [item["start"] for item in timeline[1:]], total
        ),
        bed,
        output,
        total,
        selected_quality_value,
    )
    output_info = probe(output)
    sheet = output.with_name(f"{output.stem}_sheet.jpg")
    contact_sheet(output, sheet, n=16, cols=8, thumb_width=240, labels=True)
    if progress is not None:
        progress(
            {
                "stage": "complete",
                "done": len(sb.segments),
                "total": len(sb.segments),
                "unit": "segments",
                "message": "render complete",
            }
        )
    voice_report = None
    if voice_track is not None:
        voice_report = {
            "manifest": sb.audio.voice,
            "lines": len(voice_track.lines),
            "total_duration": voice_track.total_duration,
            "suspect": [line.index for line in voice_track.lines if line.status == "suspect"],
        }
    music_report = music_licence(root, sb.audio.music)
    if music_report is not None:
        warnings.extend(music_report.pop("warnings"))
    return RenderReport(
        path=output,
        sheet=sheet,
        duration=output_info.duration or total,
        timeline=timeline,
        overlays=report_overlays,
        warnings=warnings,
        quality=selected_quality_value,
        voice=voice_report,
        sfx=sfx_report or None,
        music=music_report,
    )


__all__ = ["RenderReport", "compose"]
