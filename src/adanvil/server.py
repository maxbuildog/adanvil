"""MCP server exposing AdAnvil's deterministic editing library."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal

from mcp.server.mcpserver import Context, Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from PIL import Image as PILImage
from pydantic import BaseModel, ValidationError

import adanvil.audio as audio_module
import adanvil.qc.report as qc_module
from adanvil import __version__
from adanvil.batch import existing_files, plan_items, variant_ids
from adanvil.brand import load_brand
from adanvil.endcard import emoji_provider, write_end_card
from adanvil.ffmpeg import extract_frame, probe
from adanvil.formats import FORMATS, ratio_of
from adanvil.jobs import mark_dead_runner, read_job, resolve_job_file, start_job
from adanvil.library import (
    annotate,
    entry_line,
    index_library,
    index_summary,
    load_catalog,
    pending_files,
    pending_speech,
    pending_vision,
    refine_targets,
    search,
    search_mosaic,
    search_shots,
)
from adanvil.library import library_overview as _library_overview
from adanvil.media import Cache, Format
from adanvil.music import MusicError
from adanvil.music import music_fetch as _music_fetch
from adanvil.music import music_search as _music_search
from adanvil.onboard import facts_summary, onboard
from adanvil.project import init_project, read_project_text, resolve_in_project
from adanvil.qc import PairError, qc_report
from adanvil.qc import ad_score as _ad_score
from adanvil.qc import compare_pair as _compare_pair
from adanvil.reference import breakdown_reference, decode_reference
from adanvil.reference.breakdown import (
    EMOTION_VALUES,
    MOTION_VALUES,
    ROLE_VALUES,
    SCALE_VALUES,
    TEXT_DEVICE_VALUES,
    TEXT_SPEAKER_VALUES,
    TYPE_VALUES,
)
from adanvil.safezones import platforms_for_size
from adanvil.sfx import SfxError, list_sfx
from adanvil.speech.analyze import (
    brand_filler_lists,
    read_speech,
    reading_payload,
)
from adanvil.speech.captions import expand_speech_captions
from adanvil.speech.library import SpeechLibrary, normalise_path
from adanvil.speech.takes import ClipSpeech, rank_takes
from adanvil.storyboard import (
    CaptionOverlay,
    CommentOverlay,
    SpeechCaptionsOverlay,
    Storyboard,
    VoiceCaptionsOverlay,
    get_recipe,
    list_recipes,
    load_storyboard,
    storyboard_json_schema,
    validate_storyboard,
)
from adanvil.storyboard.patch import (
    apply_changes,
    follow_rename,
    rename_storyboard,
    snapped_name,
    storyboard_name,
)
from adanvil.storyboard.schema import normalise_quality
from adanvil.storyboard.sfx import all_sfx
from adanvil.storyboard.snap import snap_storyboard
from adanvil.storyboard.validate import _schema_issues
from adanvil.text import render as render_text
from adanvil.text.animate import StatePlan
from adanvil.text.emoji import DirectoryProvider, directory_emoji, missing_emoji
from adanvil.variants import expand_matrix, format_errors, variants_sheet
from adanvil.vision import available as vision_available
from adanvil.vision import status as vision_status
from adanvil.vision import vision_setup as _vision_setup
from adanvil.voice import VoiceSpec, load_track

_INSTRUCTIONS = """Workflow for an AdAnvil editing agent:
1. Onboard the brand and keep the open questions visible.
2. Index the footage, then LOOK at the contact sheets before choosing clips.
3. Annotate the useful entries and search the catalogue instead of rescanning folders.
4. Decode the reference ad, then LOOK at its contact sheet and key moments.
5. Pick a recipe and read its checklist before writing the storyboard.
6. Write a storyboard using project-relative paths and the recipe structure.
7. Preview captions on real frames and check the returned bbox and overflow.
8. Validate the storyboard and fix errors before rendering.
9. If the ad needs narration, queue voice_generate, read its manifest, and
   align one or more shots to each line before adding voice_captions.
   If the ad is a creator talking to camera, index the footage with speech
   (library_index transcribes clips with sound), choose the takes with
   speech_takes, cut with the clean_segments of speech_view, keep the creator's
   own sound with audio.clips, cover cuts with audio_from, and caption with
   speech_captions. Indexing transcribes fast (a small model) to make the
   speech searchable: run speech_refine on the clips you choose before
   captioning, and fix a last misheard word with speech_captions.corrections.
10. Render a preview or final job, then run QC and LOOK at the returned images.
11. Iterate on the storyboard when the images or report reveal a problem.
12. Before delivery, run ad_score on the render (with its storyboard), LOOK at the safe zones
    preview and the frame of each top finding, fix the findings by impact, render again and
    aim for 80 or more. storyboard_validate(score_check=true) shows the cheap part earlier.
Before and after: rank every candidate pair (photos, a composite split with crop, frames of
before/after clips) with compare_pair, LOOK at the previews and use the pair whose difference
is obvious at a glance; a still pair of the same pose beats moving clips for a reveal.
Music: music_search finds tracks whose licence allows an ad (CC0, PDM, CC BY, CC BY-SA),
music_fetch downloads one into music/ with its licence; the render warns music_licence_unknown
or music_attribution_required (put the given text in the ad's description).
Sound design (optional): sfx_list(fetch=true) builds the CC0 sound pack in sfx/;
add {type: sfx, source, at, volume} overlays or audio.sfx_auto (cuts, captions);
for a music-led ad, music_beats shows the beats and storyboard_snap moves the
cuts onto them (validation reports beat_alignment).
Visual devices (optional, one or two per ad): a reveal segment (a divider sweeps a before
picture away to show the after picture) or a split segment (two panels, labels) for honest
before/after, a video overlay (picture-in-picture), sticker overlays (arrow, circle, underline,
badge, emoji, image), an image overlay as a card (animation, frame, rotate: a before/after photo
or a review screenshot coming in over the video), a progress overlay and punch on a clip segment
(instant zoom).
Voice lines marked suspect cannot be listened to by an agent: regenerate with
slightly different wording, or accept the line when the transcript differs
only by punctuation. A line also turns suspect when 2 or more of its
`unmatched_words` (words of 4+ letters not heard in the transcript) are
missing: reword those words and check the transcript.
The rules in docs/EDITING_GUIDE.md apply; retrieve them with editing_guide.
"""

_INDEX_SYNC_LIMIT = 8  # more new files than this and library_index runs as a job
_SPEECH_SYNC_LIMIT = 2  # more clips to transcribe than this: also a job (CPU speech is slow)
_VISION_SYNC_LIMIT = 4  # more entries for the vision pass than this: also a job
_JOB_FILES: dict[str, Path] = {}
_PROJECT_ROOTS: set[Path] = set()


def _tool_error(message: str) -> ToolError:
    return ToolError(message)


def _require_absolute(path_value: str, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise _tool_error(f"{label} must be an absolute path")
    path = Path(path_value)
    if not path.is_absolute():
        raise _tool_error(f"{label} must be an absolute path: {path_value!r}")
    return path.resolve()


def _project(project_dir: str) -> Path:
    root = _require_absolute(project_dir, "project_dir")
    if not root.exists():
        raise _tool_error(f"Project directory does not exist: {root}")
    if not root.is_dir():
        raise _tool_error(f"Project path is not a directory: {root}")
    if not (root / "adanvil.json").is_file():
        raise _tool_error(f"Not an AdAnvil project (adanvil.json is missing): {root}")
    _PROJECT_ROOTS.add(root)
    return root


def _project_for_init(project_dir: str) -> Path:
    root = _require_absolute(project_dir, "project_dir")
    if root.exists() and not root.is_dir():
        raise _tool_error(f"Project path is not a directory: {root}")
    _PROJECT_ROOTS.add(root)
    return root


def _safe_project_path(root: Path, value: str, label: str = "path") -> Path:
    try:
        return resolve_in_project(root, value)
    except (TypeError, ValueError, OSError) as exc:
        raise _tool_error(f"Invalid {label}: {exc}") from exc


def _relative_path(root: Path, path: Path) -> str:
    try:
        return Path(path).relative_to(Path(root)).as_posix()
    except ValueError:
        return str(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _source_entries(root: Path, source: str) -> list[Any]:
    """The catalogue entries under a project-relative folder or file."""

    prefix = source.rstrip("/")
    return [
        entry
        for entry in load_catalog(root).entries.values()
        if entry.path == prefix or entry.path.startswith(f"{prefix}/")
    ]


def _image(path: Path) -> Image:
    """Return a bounded JPEG image for an MCP response."""

    try:
        with PILImage.open(path) as source:
            image = source.convert("RGB")
    except (OSError, ValueError) as exc:
        raise _tool_error(f"Could not open image {path}: {exc}") from exc
    longest = max(image.size)
    if longest > 1280:
        scale = 1280 / longest
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            PILImage.Resampling.LANCZOS,
        )
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=82, optimize=True)
    image.close()
    return Image(data=output.getvalue(), format="jpeg")


def _load_brand(root: Path):
    path = root / "brand.yaml"
    if not path.is_file():
        raise _tool_error(f"Brand profile is missing: {_relative_path(root, path)}")
    try:
        return load_brand(path)
    except Exception as exc:
        raise _tool_error(f"Could not load brand.yaml: {exc}") from exc


def _emoji_available(root: Path, brand: Any) -> list[str] | str:
    """The emoji the brand's emoji setting offers: "all" for twemoji, else the folder's."""

    name = brand.visual.emoji if brand.visual and brand.visual.emoji else "twemoji"
    if name == "twemoji":
        return "all"
    if name == "none":
        return []
    try:
        folder = resolve_in_project(root, name)
    except (TypeError, ValueError):
        return []
    return directory_emoji(folder) if folder.is_dir() else []


def _load_storyboard_raw(root: Path, storyboard: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(storyboard, str):
        path = _safe_project_path(root, storyboard, "storyboard path")
        if not path.is_file():
            raise _tool_error(f"Storyboard file does not exist: {_relative_path(root, path)}")
        try:
            source = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise _tool_error(f"Could not read storyboard {storyboard!r}: {exc}") from exc
    elif isinstance(storyboard, Mapping):
        source = storyboard
    else:
        raise _tool_error("storyboard must be an object or a relative JSON file path")
    if not isinstance(source, dict):
        raise _tool_error("storyboard JSON must contain an object")
    return source


def _storyboard_file(root: Path, name: str) -> Path:
    """``storyboards/<name>.json`` of the project, for a safe ``name``."""

    try:
        return root / "storyboards" / f"{storyboard_name(name)}.json"
    except ValueError as exc:
        raise _tool_error(str(exc)) from exc


def _validation_json(report: Any) -> dict[str, Any]:
    payload = _jsonable(report)
    if payload.get("reference_comparison") is None:
        payload.pop("reference_comparison", None)
    if not payload.get("info"):
        payload.pop("info", None)
    return payload


def _validate(
    root: Path,
    storyboard: dict[str, Any] | str,
    recipe_id: str | None,
    score_check: bool = False,
):
    raw = _load_storyboard_raw(root, storyboard)
    brand = _load_brand(root)
    recipe = None
    if recipe_id is not None:
        try:
            recipe = get_recipe(recipe_id)
        except KeyError as exc:
            raise _tool_error(str(exc)) from exc
    report = validate_storyboard(
        raw, root=root, brand=brand, recipe=recipe, score_check=score_check
    )
    if not report.ok:
        return None, report
    try:
        return load_storyboard(raw), report
    except Exception as exc:
        raise _tool_error(f"Invalid storyboard: {exc}") from exc


def _expanded_sfx(root: Path, storyboard: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """The sound effects of a storyboard (``sfx`` overlays and what ``audio.sfx_auto`` expands
    to), so the agent sees them; ``None`` when there are none or the storyboard is not valid
    yet (``storyboard_validate`` says what is wrong)."""

    audio = storyboard.get("audio")
    wanted = isinstance(audio, Mapping) and audio.get("sfx_auto") is not None
    wanted = wanted or any(
        isinstance(item, Mapping) and item.get("type") == "sfx"
        for item in storyboard.get("overlays") or []
    )
    if not wanted:
        return None
    try:
        model = Storyboard.model_validate(storyboard)
        starts = [begin for begin, _ in model.timeline()]
        return all_sfx(model, starts, model.total_duration(), root)
    except Exception:
        return None


def _speech_entry(root: Path, entry_id: str | None, source: str | None):
    """The catalogue entry named by ``entry_id`` or by its project path ``source``."""

    catalog = load_catalog(root)
    if entry_id is not None:
        entry = catalog.entries.get(entry_id)
        if entry is None:
            raise _tool_error(f"Unknown catalog entry: {entry_id}")
        return entry
    if source is None:
        raise _tool_error("pass entry_id or source")
    _safe_project_path(root, source, "source")
    wanted = normalise_path(source)
    for entry in catalog.entries.values():
        if entry.path == wanted:
            return entry
    raise _tool_error(f"{source!r} is not indexed: run library_index first")


def _frame_path(root: Path, source_path: Path, t: float, suffix: str = ".png") -> Path:
    try:
        stat = source_path.stat()
        stamp = f"{source_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{t:g}"
    except OSError:
        stamp = f"{source_path.resolve()}:{t:g}"
    digest = hashlib.sha1(stamp.encode("utf-8")).hexdigest()[:16]
    return root / "cache" / "frames" / f"{source_path.stem}-{digest}{suffix}"


def _extract_or_open_frame(root: Path, source: str, t: float) -> tuple[Path, bool]:
    if t < 0:
        raise _tool_error("t must be non-negative")
    source_path = _safe_project_path(root, source, "source")
    if not source_path.is_file():
        raise _tool_error(f"Source file does not exist: {source}")
    try:
        info = probe(source_path)
    except Exception as exc:
        raise _tool_error(f"Could not probe source {source!r}: {exc}") from exc
    frame_path = _frame_path(root, source_path, t)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    if info.kind == "video":
        if not frame_path.is_file():
            try:
                extract_frame(source_path, t, frame_path)
            except Exception as exc:
                raise _tool_error(f"Could not extract frame from {source!r}: {exc}") from exc
        return frame_path, True
    if info.kind == "image":
        if not frame_path.is_file():
            try:
                with PILImage.open(source_path) as image:
                    image.convert("RGB").save(frame_path, format="PNG")
            except (OSError, ValueError) as exc:
                raise _tool_error(f"Could not read image source {source!r}: {exc}") from exc
        return frame_path, False
    raise _tool_error(f"Source must be a video or image, not {info.kind}: {source}")


def _cover_crop(image: PILImage.Image, width: int = 1080, height: int = 1920) -> PILImage.Image:
    image = image.convert("RGB")
    target_ratio = width / height
    source_ratio = image.width / image.height
    if source_ratio > target_ratio:
        crop_width = max(1, round(image.height * target_ratio))
        left = (image.width - crop_width) // 2
        box = (left, 0, left + crop_width, image.height)
    else:
        crop_height = max(1, round(image.width / target_ratio))
        top = (image.height - crop_height) // 2
        box = (0, top, image.width, top + crop_height)
    return image.crop(box).resize((width, height), PILImage.Resampling.LANCZOS)


def _brand_emoji_provider(root: Path, brand: Any, emoji: str | None = None):
    """The emoji provider: ``emoji`` (a storyboard ``emoji`` value) or else the brand's."""

    try:
        return emoji_provider(root, brand, emoji)
    except (TypeError, ValueError, OSError) as exc:
        raise _tool_error(f"Invalid brand emoji path: {exc}") from exc


def _index_argument(values: dict[str, Any], name: str, overlay_type: str) -> int | None:
    """Pop an optional non-negative integer such as ``chunk_index`` off an overlay object."""

    value = values.pop(name, None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{overlay_type}.{name} must be a non-negative integer")
    return value


def _chunk_at(chunks: list[CaptionOverlay], index: int | None, t: float) -> int:
    """The chunk to show: ``index``, else the one on screen at timeline time ``t``."""

    if index is not None:
        return index
    for position, chunk in enumerate(chunks):
        end = float("inf") if chunk.end is None else chunk.end
        if chunk.start <= t < end:
            return position
    started = [position for position, chunk in enumerate(chunks) if chunk.start <= t]
    return started[-1] if started else 0


def _expanded_payload(captions: list[CaptionOverlay]) -> list[dict[str, Any]]:
    return [
        {
            "text": " ".join(caption.lines),
            "lines": caption.lines,
            "start": caption.start,
            "end": caption.end,
        }
        for caption in captions
    ]


def _animated_state(
    chunk: CaptionOverlay,
    t: float,
    word_index: int | None,
    provider: Any,
    canvas: tuple[int, int],
    highlight: str | None,
    fps: int,
    explicit_chunk: bool,
):
    """The picture of the state of an animated ``chunk``: its lit word, or the one at ``t``.

    Returns the state, the render of all its states (for size and overflow) and a dict of what
    was chosen. The picture is placed like the final render places it.
    """

    from adanvil.render.animated import plan_caption, state_images

    end = chunk.end if chunk.end is not None else chunk.start + 3.0
    plans, times = plan_caption(chunk, chunk.start, end, fps)
    if not plans:
        raise ValueError("this caption has no visible time to preview")
    if chunk.animation == "pop":
        if word_index not in (None, 0):
            raise ValueError("a pop caption has no lit word: leave word_index out")
        pick = plans[-1]
        if not explicit_chunk:
            pick = next((p for p in plans if p.start <= t < p.end), plans[-1])
    elif word_index is not None:
        if word_index >= len(times):
            raise ValueError(
                f"word_index {word_index} is out of range for {len(times)} words in the chunk"
            )
        pick = next((p for p in plans if p.active == word_index), None)
        if pick is None:  # the word is too short to be seen in the render: show it anyway
            at = times[word_index][0]
            pick = StatePlan(at, max(at, times[word_index][1]), word_index, 1.0)
            plans = [*plans, pick]
    elif explicit_chunk:
        pick = plans[0]
    else:
        pick = next((p for p in plans if p.start <= t < p.end), plans[-1 if t >= end else 0])
    images, _spec, rendered = state_images(chunk, plans, provider, canvas, highlight)
    state = images[(pick.active, pick.scale)]
    info = {
        "animation": chunk.animation,
        "word_index": pick.active,
        "words": len(times),
        "states": len(plans),
        "window": [round(pick.start, 4), round(pick.end, 4)],
    }
    return state, rendered, info


def render_caption_preview(
    root: Path,
    source: str,
    t: float,
    overlay: dict[str, Any],
    output: Path | None = None,
    emoji: str | None = None,
    storyboard: dict[str, Any] | str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Render a caption/comment over a real source frame for both wrappers.

    ``emoji`` takes the same values as the storyboard ``emoji`` field (twemoji, none or a
    project folder); without it the brand's emoji setting is used. ``storyboard`` (a project
    path or the object) gives the timeline of a ``speech_captions`` overlay.

    An animated caption (``animation`` ``karaoke``, ``word`` or ``pop``) is drawn in one of its
    states: the lit word ``word_index`` of the chunk ``chunk_index`` (both part of the overlay
    object), or, without them, the state on screen at timeline time ``t``.
    """

    frame_path, _ = _extract_or_open_frame(root, source, t)
    if not isinstance(overlay, Mapping):
        raise _tool_error(
            "overlay must be a storyboard caption, comment, voice_captions or "
            "speech_captions object"
        )
    overlay_type = overlay.get("type")
    expanded_payload: list[dict[str, Any]] | None = None
    animated_info: dict[str, Any] | None = None
    unmatched_emphasis: list[str] = []
    state = None
    animated_render = None
    try:
        values = dict(overlay)
        chunk_index = _index_argument(values, "chunk_index", str(overlay_type))
        word_index = _index_argument(values, "word_index", str(overlay_type))
        chosen: int | None = chunk_index
        chunks: list[CaptionOverlay] = []
        if overlay_type == "caption":
            model = CaptionOverlay.model_validate(values)
            unmatched_emphasis = model.unmatched_emphasis()
            chunks = model.expand_words(model.end if model.end is not None else model.start + 3.0)
            if len(chunks) > 1:
                expanded_payload = _expanded_payload(chunks)
            position = _chunk_at(chunks, chunk_index, t)
            if position >= len(chunks):
                raise ValueError(
                    f"caption.chunk_index {position} is out of range for {len(chunks)} chunks"
                )
            model = chunks[position]
            chosen = position
        elif overlay_type == "comment":
            model = CommentOverlay.model_validate(values)
        elif overlay_type == "voice_captions":
            voice_model = VoiceCaptionsOverlay.model_validate(values)
            if voice_model.manifest is None:
                raise ValueError(
                    "voice_captions overlay requires manifest: pass manifest: the audio.voice "
                    "path of your storyboard (voice/<name>/manifest.json)"
                )
            manifest_path = _safe_project_path(
                root, voice_model.manifest, "voice_captions.manifest"
            )
            if not manifest_path.is_file():
                raise ValueError(f"Voice manifest does not exist: {voice_model.manifest}")
            track = load_track(manifest_path)
            offset = 0.0 if voice_model.offset is None else voice_model.offset
            chunks = voice_model.expand(track, offset=offset)
            # Without an index a plain caption shows chunk 0; an animated one follows ``t``.
            animated_voice = voice_model.animation != "none"
            position = _chunk_at(chunks, chunk_index, t) if animated_voice else (chunk_index or 0)
            if position >= len(chunks):
                raise ValueError(
                    f"voice_captions.chunk_index {position} is out of range "
                    f"for {len(chunks)} expanded captions"
                )
            expanded_payload = _expanded_payload(chunks)
            model = chunks[position]
            chosen = position
        elif overlay_type == "speech_captions":
            written_storyboard = values.pop("storyboard", None)
            if storyboard is not None:
                written_storyboard = storyboard
            speech_model = SpeechCaptionsOverlay.model_validate(values)
            if written_storyboard is not None:
                board = load_storyboard(_load_storyboard_raw(root, written_storyboard))
            else:
                # No storyboard given: the captions of the whole ``source`` clip.
                clip_info = probe(_safe_project_path(root, source, "source"))
                if clip_info.kind != "video" or not clip_info.duration:
                    raise ValueError(
                        "speech_captions preview without a storyboard needs a video source"
                    )
                board = Storyboard.model_validate(
                    {
                        "name": "speech_preview",
                        "segments": [
                            {"type": "clip", "source": source, "in": 0, "out": clip_info.duration}
                        ],
                        "audio": {"clips": True},
                    }
                )
            spans = board.timeline()
            chunks = expand_speech_captions(
                speech_model,
                board,
                SpeechLibrary(root, brand_filler_lists(root)),
                [span[0] for span in spans],
                [span[1] for span in spans],
            )
            if not chunks:
                raise ValueError(
                    "speech_captions found no spoken word: index the clips with speech "
                    "(library_index, transcribe=true) and check the segments play their sound"
                )
            animated_speech = speech_model.animation != "none"
            position = _chunk_at(chunks, chunk_index, t) if animated_speech else (chunk_index or 0)
            if position >= len(chunks):
                raise ValueError(
                    f"speech_captions.chunk_index {position} is out of range "
                    f"for {len(chunks)} expanded captions"
                )
            expanded_payload = _expanded_payload(chunks)
            model = chunks[position]
            chosen = position
        else:
            raise ValueError(
                "overlay.type must be 'caption', 'comment', 'voice_captions' or 'speech_captions'"
            )
        brand = _load_brand(root)
        applied = brand.apply_to_caption(model)
        provider = _brand_emoji_provider(root, brand, emoji)
        emoji_lines = (
            list(applied.lines)
            if hasattr(applied, "lines")
            else [str(getattr(applied, "text", ""))]
        )
        absent = (
            missing_emoji(emoji_lines, provider) if isinstance(provider, DirectoryProvider) else []
        )
        with PILImage.open(frame_path) as source_image:
            background = _cover_crop(source_image)
        accent = brand.visual.highlight_color if brand.visual is not None else None
        if isinstance(applied, CaptionOverlay) and applied.animated:
            state, animated_render, animated_info = _animated_state(
                applied,
                t,
                word_index,
                provider,
                background.size,
                accent,
                30,
                chunk_index is not None or word_index is not None,
            )
            background = background.convert("RGBA")
            background.alpha_composite(state.image, (state.x, state.y))
            animated_info["chunk_index"] = chosen
        else:
            if word_index is not None:
                raise ValueError("word_index needs an animated caption (set animation)")
            spec = (
                applied.to_text_spec(accent)
                if isinstance(applied, CaptionOverlay)
                else applied.to_text_spec()
            )
            rendered = render_text(spec, provider, canvas=background.size)
            background = background.convert("RGBA")
            background.alpha_composite(rendered.image)
    except ToolError:
        raise
    except Exception as exc:
        raise _tool_error(f"Could not render caption preview: {exc}") from exc

    if output is None:
        encoded = json.dumps(
            {
                "overlay": _jsonable(model.model_dump(mode="json")),
                "expanded": expanded_payload,
                "state": animated_info,
            },
            sort_keys=True,
        )
        digest = hashlib.sha1(f"{source}:{t:g}:{encoded}".encode()).hexdigest()[:16]
        output = root / "cache" / "previews" / f"caption-{digest}.jpg"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_format = "PNG" if output.suffix.casefold() == ".png" else "JPEG"
    if output_format == "PNG":
        background.convert("RGB").save(output, format=output_format, optimize=True)
    else:
        background.convert("RGB").save(output, format=output_format, quality=92, optimize=True)
    if animated_render is not None and state is not None:
        payload = {
            "path": _relative_path(root, output),
            "source": source,
            "t": t,
            "bbox": list(state.bbox),
            "size_used": animated_render.size_used,
            "overflow": animated_render.overflow,
            "lines": animated_render.lines,
        }
    else:
        payload = {
            "path": _relative_path(root, output),
            "source": source,
            "t": t,
            "bbox": list(rendered.bbox),
            "size_used": rendered.size_used,
            "overflow": rendered.overflow,
            "lines": rendered.lines,
        }
        rendered.image.close()
    if animated_info is not None:
        payload.update(animated_info)
    if absent:
        payload["emoji_missing"] = absent
    if unmatched_emphasis:
        payload["emphasis_unmatched"] = unmatched_emphasis
        payload["warnings"] = [
            f"emphasis_unmatched: {entry!r} matches no word of the caption: nothing is "
            "highlighted (write it as shown, e.g. '65,90 €', or mark it with *...*)"
            for entry in unmatched_emphasis
        ]
    if expanded_payload is not None:
        payload["expanded"] = expanded_payload
    background.close()
    return payload, output


def _render_report_json(report: Any) -> dict[str, Any]:
    return _jsonable(report.model_dump(mode="json"))


def _voice_summary(root: Path, track: Any) -> dict[str, Any]:
    return {
        "manifest": _relative_path(root, root / track.manifest),
        "lines": [
            {
                "index": line.index,
                "text": line.text,
                "display": line.display,
                "start": line.start,
                "duration": line.duration,
                "status": line.status,
                "transcript": line.transcript,
                "unmatched_words": line.unmatched_words,
            }
            for line in track.lines
        ],
        "total_duration": track.total_duration,
        "total": track.total_duration,
        "warnings": list(track.warnings),
    }


class _DirectTemporaryDirectory:
    """Windows-safe replacement for QC's ffmpeg scratch directory."""

    def __init__(self, prefix: str = "tmp-", dir: str | Path | None = None) -> None:
        base = Path(dir) if dir is not None else Path.cwd()
        self.path = base / f"{prefix}{uuid.uuid4().hex}"
        self.path.mkdir(parents=True, exist_ok=False)
        self.name = str(self.path)

    def __enter__(self) -> str:
        return self.name

    def __exit__(self, *_args: Any) -> bool:
        shutil.rmtree(self.path, ignore_errors=True)
        return False


_QC_LOCK = threading.Lock()


class JobNotFound(ValueError):
    """No job document could be located for an id."""


class _RenderJobs:
    """Locate durable jobs and wait for detached runners."""

    def submit(
        self, root: Path, storyboard: Any, quality: Literal["preview", "final", "draft"]
    ) -> str:
        record = start_job(
            root,
            "render",
            {
                "storyboard": storyboard.model_dump(mode="json", by_alias=True),
                "quality": quality,
            },
            progress={"total": len(storyboard.segments), "unit": "segments"},
        )
        job_id = str(record["job_id"])
        _JOB_FILES[job_id] = Path(str(record["job_file"]))
        return job_id

    def submit_batch(
        self,
        root: Path,
        batch: str,
        items: list[dict[str, Any]],
        quality: str,
        score: bool,
    ) -> str:
        record = start_job(
            root,
            "batch",
            {"batch": batch, "items": items, "quality": quality, "score": score},
            progress={"total": len(items), "unit": "renders"},
        )
        job_id = str(record["job_id"])
        _JOB_FILES[job_id] = Path(str(record["job_file"]))
        return job_id

    def submit_voice(self, root: Path, name: str, spec: VoiceSpec) -> str:
        record = start_job(
            root,
            "voice",
            {"name": name, "spec": spec.model_dump(mode="json")},
            progress={"total": len(spec.lines), "unit": "lines"},
        )
        job_id = str(record["job_id"])
        _JOB_FILES[job_id] = Path(str(record["job_file"]))
        return job_id

    def submit_music(
        self, root: Path, prompt: str, duration: float, steps: int, seed: int | None
    ) -> str:
        record = start_job(
            root,
            "music",
            {"prompt": prompt, "duration": duration, "steps": steps, "seed": seed},
            progress={"total": steps, "unit": "steps"},
        )
        job_id = str(record["job_id"])
        _JOB_FILES[job_id] = Path(str(record["job_file"]))
        return job_id

    def submit_index(
        self,
        root: Path,
        source: str,
        total: int,
        transcribe: bool | str = "auto",
        vision: bool | str = "auto",
    ) -> str:
        record = start_job(
            root,
            "index",
            {"source": source, "transcribe": transcribe, "vision": vision},
            progress={"total": total, "unit": "files"},
        )
        job_id = str(record["job_id"])
        _JOB_FILES[job_id] = Path(str(record["job_file"]))
        return job_id

    def submit_refine(
        self,
        root: Path,
        entry_ids: list[str],
        model: str,
        language: str | None,
    ) -> str:
        record = start_job(
            root,
            "refine",
            {"entry_ids": entry_ids, "model": model, "language": language},
            progress={"total": len(entry_ids), "unit": "clips"},
        )
        job_id = str(record["job_id"])
        _JOB_FILES[job_id] = Path(str(record["job_file"]))
        return job_id

    def _find_file(self, job_id: str, project_dir: Path | None = None) -> Path | None:
        known = _JOB_FILES.get(job_id)
        if known is not None and known.is_file():
            return known
        # A restarted server has an empty map: the pointer file or the project finds it.
        found = resolve_job_file(job_id, project_dir)
        if found is not None:
            _JOB_FILES[job_id] = found
            return found
        candidates = [root / "renders" / "jobs" / f"{job_id}.json" for root in _PROJECT_ROOTS]
        try:
            candidates.extend(
                path
                for path in Path.cwd().rglob(f"{job_id}.json")
                if path.parent.name == "jobs" and path.parent.parent.name == "renders"
            )
        except (OSError, PermissionError):
            pass
        for candidate in candidates:
            if candidate.is_file():
                _JOB_FILES[job_id] = candidate
                return candidate
        return None

    def get(self, job_id: str, project_dir: Path | None = None) -> dict[str, Any]:
        job_file = self._find_file(job_id, project_dir)
        if job_file is None:
            raise JobNotFound(f"job not found: pass project_dir ({job_id})")
        try:
            record = read_job(job_file)
        except ValueError as exc:
            return {
                "job_id": job_id,
                "status": "failed",
                "stage": "failed",
                "progress": {"done": 0, "total": 0, "unit": "segments"},
                "message": f"could not read job status: {exc}",
                "report": None,
                "error": str(exc),
            }
        record = mark_dead_runner(job_file, record)
        return record

    @staticmethod
    def _status_payload(record: dict[str, Any]) -> dict[str, Any]:
        started = record.get("started_at")
        finished = record.get("finished_at")
        now = time.time()
        elapsed = 0.0
        if isinstance(started, (int, float)):
            end = finished if isinstance(finished, (int, float)) else now
            elapsed = max(0.0, float(end) - float(started))
        progress = record.get("progress")
        if not isinstance(progress, dict):
            progress = {"done": 0, "total": 0, "unit": "segments"}
        eta: float | None = None
        done = progress.get("done")
        total = progress.get("total")
        if (
            isinstance(done, (int, float))
            and isinstance(total, (int, float))
            and total > 0
            and done >= total
            and record.get("status") == "done"
        ):
            eta = 0.0
        elif (
            isinstance(done, (int, float))
            and isinstance(total, (int, float))
            and done > 0
            and total > done
            and elapsed > 0
        ):
            eta = round(elapsed * (total - done) / done, 3)
        return {
            "job_id": record.get("job_id"),
            "status": record.get("status", "failed"),
            "stage": record.get("stage"),
            "progress": progress,
            "elapsed": round(elapsed, 3),
            "eta_seconds": eta,
            "message": record.get("message"),
            "report": record.get("report"),
            "voice": record.get("voice"),
            "index": record.get("index"),
            "refine": record.get("refine"),
            "batch": record.get("batch"),
            "music": record.get("music"),
            "error": record.get("error"),
            "sheet": (record.get("report") or {}).get("sheet")
            if record.get("status") == "done"
            else None,
        }

    def status(
        self, job_id: str, wait_seconds: float = 0, project_dir: Path | None = None
    ) -> dict[str, Any]:
        if wait_seconds < 0 or wait_seconds > 600:
            raise ValueError("wait_seconds must be between 0 and 600")
        deadline = time.monotonic() + wait_seconds
        while True:
            record = self.get(job_id, project_dir)
            if record.get("status") not in {"queued", "running"}:
                return self._status_payload(record)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._status_payload(record)
            time.sleep(min(2.0, remaining))


def _report_from_job(manager: _RenderJobs, job_id: str) -> Any:
    try:
        record = manager.get(job_id)
    except JobNotFound as exc:
        raise _tool_error(str(exc)) from exc
    if record.get("status") != "done" or not record.get("report"):
        raise _tool_error(f"Render job is not done: {job_id}")
    try:
        from adanvil.render.compose import RenderReport

        return RenderReport.model_validate(record["report"])
    except Exception as exc:
        raise _tool_error(f"Saved render report is invalid for job {job_id}: {exc}") from exc


def _report_from_argument(root: Path, manager: _RenderJobs, value: str | None) -> Any | None:
    if value is None:
        return None
    if value.startswith("job-"):
        return _report_from_job(manager, value)
    path = _safe_project_path(root, value, "render report path")
    if not path.is_file():
        raise _tool_error(f"Render report file does not exist: {value}")
    try:
        from adanvil.render.compose import RenderReport

        return RenderReport.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise _tool_error(f"Could not load render report {value!r}: {exc}") from exc


def build_server() -> MCPServer:
    """Build an in-process MCP server with exactly AdAnvil's public tools."""

    server = MCPServer(
        name="adanvil",
        version=__version__,
        instructions=_INSTRUCTIONS,
    )
    jobs = _RenderJobs()

    @server.tool(structured_output=False)
    def project_init(project_dir: str, name: str) -> list[dict[str, Any]]:
        """Create an AdAnvil project skeleton.

        Use this first for a new absolute project directory, then inspect the
        returned folders before adding media.
        """

        root = _project_for_init(project_dir)
        try:
            init_project(root, name)
        except Exception as exc:
            raise _tool_error(f"Could not initialise project {root}: {exc}") from exc
        return [{"project_dir": str(root), "name": name, "created": True}]

    @server.tool(structured_output=False)
    def project_read_text(
        project_dir: str, path: str, max_bytes: int = 65536
    ) -> list[dict[str, object]]:
        """Read a UTF-8 project text file using the project's safe path rules."""

        root = _project(project_dir)
        try:
            return [read_project_text(root, path, max_bytes)]
        except Exception as exc:
            raise _tool_error(f"Could not read project text {path!r}: {exc}") from exc

    @server.tool(structured_output=False)
    def brand_onboard(
        project_dir: str,
        url: str | None = None,
        answers: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run brand onboarding step one or two.

        Use without answers to discover site facts and questions, then call
        with answers to write brand.yaml and brand_brief.md. Use open_questions
        to decide what the human must answer. ``facts`` is a compact summary of
        the site (title, colours, fonts, headings, prices, guarantee phrases,
        a few products); the complete facts are saved in ``facts_file``
        (brand/site_facts.json): read it with project_read_text only when you
        need more. ``more`` counts what the summary left out.
        """

        root = _project(project_dir)
        try:
            result = onboard(root, url=url, answers=answers)
        except Exception as exc:
            raise _tool_error(f"Brand onboarding failed: {exc}") from exc
        facts_path = root / "brand" / "site_facts.json"
        payload = {
            "facts": (
                facts_summary(
                    result.facts,
                    facts_file=_relative_path(root, facts_path) if facts_path.is_file() else None,
                )
                if result.facts is not None
                else None
            ),
            "questionnaire": [
                question.model_dump(mode="json") for question in result.questionnaire
            ],
            "brand_path": _relative_path(root, result.brand_path) if result.brand_path else None,
            "brief_path": _relative_path(root, result.brief_path) if result.brief_path else None,
            "open_questions": result.open_questions,
        }
        return [payload]

    @server.tool(structured_output=False)
    def brand_get(project_dir: str) -> list[dict[str, Any]]:
        """Read the current brand profile and brief.

        Use this before writing captions or storyboards so copy follows the
        brand defaults and constraints. ``brand.product_versions`` says how to
        tell the product versions apart (match it with the ``product_version``
        annotation of library entries). ``brand.cta_image`` is the brand's
        ready-made call-to-action image, to use as the last segment.
        ``emoji_available`` lists the emoji characters of the brand's emoji
        folder, or is "all" for twemoji (and empty for none).
        """

        root = _project(project_dir)
        brand = _load_brand(root)
        brief_path = root / "brand_brief.md"
        return [
            {
                "brand": brand.model_dump(mode="json"),
                "brief": brief_path.read_text(encoding="utf-8") if brief_path.is_file() else None,
                "emoji_available": _emoji_available(root, brand),
                "brand_path": _relative_path(root, root / "brand.yaml"),
                "brief_path": _relative_path(root, brief_path) if brief_path.is_file() else None,
            }
        ]

    @server.tool(structured_output=False)
    async def library_index(
        ctx: Context,
        project_dir: str,
        source: str = "footage",
        transcribe: bool | Literal["auto"] = "auto",
        vision: bool | Literal["auto"] = "auto",
    ) -> list[dict[str, Any]]:
        """Index or refresh media under a project-relative folder or file.

        Use it before selection. ``source`` is a folder or a single media file.
        Up to 8 new or changed files are indexed in this call; more than 8 run
        as a detached job and the call returns at once with ``job_id`` (wait
        with job_status(job_id, wait_seconds=90), progress is per file). The
        result is compact: counts, one line per new or updated entry (id, path,
        duration, people, product_visible, ``speech`` when the clip speaks) and
        errors. Contact sheets are not sent inline: use library_view, or
        library_search with full=true for entries in full.

        ``transcribe`` controls the speech of the clips: "auto" (default)
        transcribes, with word timestamps, the clips that have an audio stream
        that is not near-silent; true transcribes every clip with an audio
        stream; false does no speech work. It uses faster-whisper when
        installed, else the voice environment (adanvil voice-setup); with
        neither, entries get a clear ``speech`` status ("unavailable: ...") and
        indexing still succeeds. The model is ADANVIL_SPEECH_MODEL (default
        base: fast, good enough to search) and the language is the brand
        language, else auto-detected; the brand name, the product name and
        ``speech.vocabulary`` of brand.yaml are given to the transcriber. The
        speech of each file is saved as soon as that file is transcribed, so
        library_search(text=...) works while the job runs (stage
        "transcribing"). Words are right about nine times in ten: run
        speech_refine on the clips you will use before captioning. Audio files
        (.m4a, .wav, .mp3, .aac) are indexed as ``audio`` entries, with their
        speech, and can be used as ``audio_from`` sources. A video whose sound
        is silent gets the speech status ``silent_audio`` and, when audio files
        with a similar name are nearby, ``separate_audio_candidates``. The
        result has a ``speech`` count by status.

        ``vision`` controls the vision pass: with the vision environment
        installed (``adanvil vision-setup`` or the tool ``vision_setup``) each
        image and video is also looked at ("auto", default): the frames of its
        contact sheet are embedded (so ``library_search(query=...)`` can find
        shots by sentence) and searched for faces, the entry gets ``auto_tags``
        (zero-shot labels such as person, bathroom, smiling, never mixed with
        your ``tags``) and, when ``brand.product_versions`` has reference
        photos, a guess of the product version per shot. ``true`` redoes every
        entry; ``false`` skips the pass. Without the environment the result
        says ``vision: {"unavailable": n}`` and indexing still succeeds.
        """

        root = _project(project_dir)
        _safe_project_path(root, source, "source")
        if transcribe not in {True, False, "auto"}:
            raise _tool_error("transcribe must be true, false, or 'auto'")
        if vision not in {True, False, "auto"}:
            raise _tool_error("vision must be true, false, or 'auto'")
        current_files, pending = await asyncio.to_thread(pending_files, root, source)
        speech_pending = await asyncio.to_thread(pending_speech, root, source, transcribe)
        vision_pending = 0
        if vision is not False and vision_available():
            vision_pending = pending + await asyncio.to_thread(
                pending_vision, root, _source_entries(root, source), vision
            )
        if (
            pending > _INDEX_SYNC_LIMIT
            or speech_pending > _SPEECH_SYNC_LIMIT
            or vision_pending > _VISION_SYNC_LIMIT
        ):
            source_prefix = source.rstrip("/")
            previous_files = sum(
                1
                for entry in load_catalog(root).entries.values()
                if entry.path == source_prefix or entry.path.startswith(f"{source_prefix}/")
            )
            total = max(len(current_files), previous_files)
            job_id = jobs.submit_index(root, source, total, transcribe, vision)
            return [
                {
                    "job_id": job_id,
                    "status": "queued",
                    "files": total,
                    "to_index": pending,
                    "to_transcribe": speech_pending,
                    "to_look_at": vision_pending,
                    "message": (
                        f"{pending} new or changed files, {speech_pending} clips to "
                        f"transcribe and {vision_pending} entries for the vision pass: "
                        "indexing runs as a job. Wait with "
                        "job_status(job_id, wait_seconds=90); call it again if the "
                        "connection drops."
                    ),
                }
            ]
        total = max(len(current_files), 1)
        loop = asyncio.get_running_loop()
        done = 0

        def progress(message: str) -> None:
            nonlocal done
            speech = re.match(r"speech (\d+)/(\d+)", message)
            if speech:
                # Transcription comes after the files: it extends the same progress bar.
                current, out_of = total + int(speech.group(1)), total + int(speech.group(2))
            elif message.startswith("vision "):
                # The vision pass comes after the files (and the speech): same progress bar.
                looked = re.match(r"vision (\d+)/(\d+)", message)
                if not looked:
                    return
                current, out_of = total + int(looked.group(1)), total + int(looked.group(2))
            elif message.startswith(("indexed ", "unchanged ", "error ", "removed ")):
                done += 1
                current, out_of = (min(done, total) if total else done), total
            else:
                return
            print(f"[library_index] {current}/{out_of} {message}", file=sys.stderr, flush=True)

            def send() -> None:
                task = asyncio.create_task(ctx.report_progress(current, out_of, message))

                def consume(completed: asyncio.Task[Any]) -> None:
                    if not completed.cancelled():
                        completed.exception()

                task.add_done_callback(consume)

            loop.call_soon_threadsafe(send)

        try:
            result = await asyncio.to_thread(
                index_library,
                root,
                source,
                progress=progress,
                transcribe=transcribe,
                vision=vision,
            )
        except Exception as exc:
            raise _tool_error(f"Library indexing failed: {exc}") from exc
        return [index_summary(root, result)]

    @server.tool(structured_output=False)
    def library_view(project_dir: str, entry_id: str) -> list[Any]:
        """Return one catalogue entry and its contact sheet.

        Use it when a summary identifies a candidate; LOOK at the image before
        choosing a shot. An audio entry has no sheet: only its data is returned.
        """

        root = _project(project_dir)
        entry = load_catalog(root).entries.get(entry_id)
        if entry is None:
            raise _tool_error(f"Unknown catalog entry: {entry_id}")
        if entry.kind == "audio":
            return [entry.model_dump(mode="json")]  # nothing to look at: no contact sheet
        if not entry.sheet:
            raise _tool_error(f"Catalog entry has no contact sheet: {entry_id}")
        sheet_path = _safe_project_path(root, entry.sheet, "sheet path")
        if not sheet_path.is_file():
            raise _tool_error(f"Contact sheet does not exist: {entry.sheet}")
        payload = entry.model_dump(mode="json")
        payload["sheet_path"] = entry.sheet
        return [payload, _image(sheet_path)]

    @server.tool(structured_output=False)
    def library_annotate(
        project_dir: str,
        entry_id: str,
        description: str | None = None,
        tags: list[str] | None = None,
        people: list[str] | None = None,
        product_visible: bool | None = None,
        product_version: str | None = None,
        emotion: list[str] | None = None,
        quality: int | None = None,
        usable: Any | None = None,
        notes: str | None = None,
        exclude: bool | None = None,
        exclude_reason: str | None = None,
    ) -> list[dict[str, Any]]:
        """Merge annotations into one catalogue entry.

        Parameters: ``project_dir`` (str, absolute project path), ``entry_id``
        (str, catalog entry id), ``description`` (str or null), ``tags``
        (list[str] or null), ``people`` (list[str] or null), ``product_visible``
        (bool or null), ``product_version`` (short free string such as
        ``current``, ``old`` or ``v2``, or null), ``emotion`` (list of what the
        faces express, a clip can show several: none, neutral, worried,
        frustrated, sad, tired, closed, skeptical, shocked, curious, relaxed,
        happy, satisfied; the same words as the breakdown, the segments and the
        search filter, anything else is refused; or null), ``quality`` (int 1-5 or null),
        ``usable`` (list of ``[start, end, note]``, ``[start, end]``, or objects with ``start``,
        ``end`` and ``note``; or null), ``notes`` (str or null), ``exclude``
        (bool or null), and ``exclude_reason`` (str or null).

        Example: ``library_annotate('/abs/project', 'clip-a1b2c3',
        tags=['hero'], quality=5, usable=[[1.2, 3.4, 'clear demo']])``.
        Use after looking at the contact sheet; search can then filter by tags,
        people, product version, emotion, quality and exclusion.
        """

        root = _project(project_dir)
        fields = {
            key: value
            for key, value in {
                "description": description,
                "tags": tags,
                "people": people,
                "product_visible": product_visible,
                "product_version": product_version,
                "emotion": emotion,
                "quality": quality,
                "usable": usable,
                "notes": notes,
                "exclude": exclude,
                "exclude_reason": exclude_reason,
            }.items()
            if value is not None
        }
        try:
            entry = annotate(root, entry_id, **fields)
        except Exception as exc:
            raise _tool_error(f"Could not annotate {entry_id!r}: {exc}") from exc
        return [entry.model_dump(mode="json")]

    @server.tool(structured_output=False)
    def library_search(
        project_dir: str,
        text: str | None = None,
        tags: list[str] | None = None,
        people: list[str] | None = None,
        kind: str | None = None,
        min_quality: int | None = None,
        product_visible: bool | None = None,
        product_version: str | None = None,
        emotion: list[str] | None = None,
        include_excluded: bool = False,
        min_duration: float | None = None,
        full: bool = False,
        query: str | None = None,
        limit: int = 12,
        auto_tags: list[str] | None = None,
    ) -> list[Any]:
        """Search indexed footage using annotations and technical metadata.

        Use the returned entries to choose clips without rescanning the
        filesystem. Each result is one compact line (id, path, duration,
        people, product_visible, version, emotion, quality); ``full=true`` returns
        the entries in full. ``product_version`` and ``emotion`` (a list: every
        listed emotion must be on the clip) filter on what was set with
        library_annotate; without a human ``product_version`` annotation the
        guess made from the brand's reference photos is used (a human annotation
        always wins). ``auto_tags`` filters on the automatic tags of the vision
        pass (English labels such as ``bathroom`` or ``smiling``; common French,
        Spanish and German words are understood).

        ``query`` searches by SENTENCE across the whole library, with no sheet
        to open: ``query="a woman with a sad face touching her jaw"`` (any
        language) ranks SHOTS by text-image similarity and returns the best
        ``limit`` (default 12, at most 48) as ``results`` (entry, shot, start,
        end, time of the best frame, score) plus ONE mosaic image of those frames,
        numbered like the list. All the other filters apply first. It needs the
        vision environment (``vision_setup``) and entries indexed with it
        (``library_index``); ``missing_vision`` counts the entries that have no
        vision data yet. Look at the mosaic, then library_view the entry to confirm
        before choosing. The score is a centred match probability: 0.5 and more is a
        clear match, under 0.1 means nothing in the library looks like it.
        """

        root = _project(project_dir)
        try:
            entries = search(
                root,
                text=text,
                tags=tags,
                people=people,
                kind=kind,
                min_quality=min_quality,
                product_visible=product_visible,
                product_version=product_version,
                emotion=emotion,
                include_excluded=include_excluded,
                min_duration=min_duration,
                auto_tags=auto_tags,
            )
        except Exception as exc:
            raise _tool_error(f"Library search failed: {exc}") from exc
        if query is not None and query.strip():
            try:
                hits, missing = search_shots(
                    root, query, entries, limit=limit, product_version=product_version
                )
                mosaic = search_mosaic(root, hits, query)
            except Exception as exc:
                raise _tool_error(f"Library search by sentence failed: {exc}") from exc
            answer: dict[str, Any] = {
                "query": query,
                "count": len(hits),
                "candidates": len(entries),
                "missing_vision": missing,
                "lines": [hit.line(rank) for rank, hit in enumerate(hits, start=1)],
                "results": [hit.as_dict(rank) for rank, hit in enumerate(hits, start=1)],
                "mosaic": _relative_path(root, mosaic) if mosaic else None,
            }
            if missing:
                answer["note"] = (
                    f"{missing} of {len(entries)} entries have no vision data: run "
                    "library_index (with the vision environment installed) to include them"
                )
            return [answer, *([_image(mosaic)] if mosaic else [])]
        return [
            {
                "count": len(entries),
                "entries": (
                    [entry.model_dump(mode="json") for entry in entries]
                    if full
                    else [entry_line(entry) for entry in entries]
                ),
            }
        ]

    @server.tool(structured_output=False)
    async def vision_setup(python: str | None = None) -> list[dict[str, Any]]:
        """Install the optional vision environment (once, several minutes, about 2.5 GB).

        It makes its own Python environment in the AdAnvil cache folder with a
        small multilingual image and text embedding model (SigLIP 2, Apache-2.0)
        and a face detector (YuNet, MIT), both free for commercial use, and
        downloads their weights. It runs on the CPU. Afterwards ``library_index``
        also looks at every image and video (shot search by sentence, automatic
        tags, product versions, faces for ``focus: "face"``). Without it
        everything else works as before. ``python`` picks the interpreter the
        environment is made from. The same as ``adanvil vision-setup``.
        """

        if vision_available() and python is None:
            return [{"status": "ready", "message": "the vision environment is already installed"}]
        try:
            interpreter = await asyncio.to_thread(_vision_setup, python=python)
        except Exception as exc:
            raise _tool_error(f"Vision setup failed: {exc}") from exc
        return [{"status": vision_status(), "python": str(interpreter)}]

    @server.tool(structured_output=False)
    def speech_view(
        project_dir: str,
        entry_id: str | None = None,
        source: str | None = None,
        max_pause: float = 0.35,
        drop_restarts: bool = False,
        words: bool = False,
        max_phrases: int = 60,
        first_phrase: int = 0,
    ) -> dict[str, Any]:
        """Read the speech of one indexed clip and get safe cuts for it.

        Pass ``entry_id`` (from library_index or library_search) or ``source`` (the
        project-relative path of a video or audio file). Needs the clip to be indexed with
        speech (library_index transcribes clips with sound). The output is compact:
        ``phrases`` (split on pauses longer than ``max_pause`` seconds and on sentence
        punctuation), each with ``start``, ``end``, ``text`` and ``flags`` (``restart``: the
        next phrase repeats it, a retake; ``filler``: it holds a hesitation; ``uncertain``:
        the transcriber doubted a word, so it may be misheard); ``fillers`` (euh, heu, hum,
        bah, ben, um, uh, er, and genre, du coup, like when said alone between pauses; the
        lists can be replaced in brand.yaml under ``speech.fillers`` and
        ``speech.isolated_fillers``); ``restarts`` (a phrase that repeats the start of the
        next one: a retake inside the file); and ``clean_segments``: clip segments
        ``{source, in, out}`` ready to paste into a storyboard, which keep the speech and drop
        pauses over ``max_pause`` and the fillers (0.06 s kept before a word, 0.10 s after,
        never cutting inside a word). Per-word data (start, end, text and ``filler`` for each
        word) only comes with ``words=true``. At most ``max_phrases`` phrases are returned
        (default 60) from ``first_phrase``: when there are more, ``note`` and
        ``next_first_phrase`` tell how to get the next ones, and the other lists are those of
        the phrases shown. With ``drop_restarts`` true the restart phrases are cut out of the
        clean segments too (they are only flagged by default, because a deliberate repetition
        looks the same). ``model`` says which transcription model made the words: if a clip
        is used in the ad, run speech_refine on it first. A clip whose sound is silent says
        ``speech_status: silent_audio`` and lists ``separate_audio_candidates``.
        """

        root = _project(project_dir)
        entry = _speech_entry(root, entry_id, source)
        if max_pause <= 0:
            raise _tool_error("max_pause must be greater than zero")
        if max_phrases < 1:
            raise _tool_error("max_phrases must be at least 1")
        if first_phrase < 0:
            raise _tool_error("first_phrase must not be negative")
        library = SpeechLibrary(root, brand_filler_lists(root))
        data = library.data(entry.path)
        speech = entry.speech
        payload: dict[str, Any] = {
            "entry_id": entry.id,
            "source": entry.path,
            "kind": entry.kind,
            "duration": entry.duration,
            "speech_status": speech.status if speech is not None else "not transcribed",
            "language": speech.language if speech is not None else None,
            "has_speech": bool(speech and speech.has_speech),
            "max_pause": max_pause,
        }
        if data is not None and data.model:
            payload["model"] = data.model
        if speech is not None and speech.separate_audio_candidates:
            payload["separate_audio_candidates"] = list(speech.separate_audio_candidates)
            payload["hint"] = (
                "this clip has no sound of its own: the voice may be one of "
                "separate_audio_candidates. Lay it on a segment with audio_from and check the "
                "sync on a preview, AdAnvil does not sync separate sound"
            )
        if data is None:
            payload.update(
                {
                    "phrase_count": 0,
                    "phrases": [],
                    "fillers": [],
                    "restarts": [],
                    "clean_segments": [],
                }
            )
            if "hint" not in payload:
                payload["hint"] = (
                    "no transcript for this clip: run library_index on it with transcribe=true "
                    "(it needs faster-whisper or the voice environment, adanvil voice-setup)"
                )
            return payload
        reading = read_speech(
            data.words,
            entry.path,
            max_pause=max_pause,
            lists=library.lists,
            drop_restarts=drop_restarts,
            duration=entry.duration,
        )
        payload.update(
            reading_payload(
                reading,
                source=entry.path,
                words=words,
                first_phrase=first_phrase,
                max_phrases=max_phrases,
            )
        )
        payload.pop("source", None)
        payload["clean_duration"] = round(
            sum(float(item["out"]) - float(item["in"]) for item in reading.segments),
            3,
        )
        return payload

    @server.tool(structured_output=False)
    def speech_refine(
        project_dir: str,
        sources: list[str],
        model: str = "medium",
        language: str | None = None,
    ) -> dict[str, Any]:
        """Transcribe chosen clips again with a stronger model and replace their words.

        Indexing transcribes fast with a small model, so about one key word per line can be
        misheard (names, numbers, rare words). Call this on the clips you will use in the ad
        (before cutting and captioning): ``sources`` lists project paths of indexed clips (a
        file, or a folder for every clip below it) or catalogue entry ids. It starts a
        detached job (wait with job_status(job_id, wait_seconds=90); progress counts clips)
        that transcribes them with ``model`` (default medium: slower, much more accurate; a
        medium model needs about 2 GB of memory) and the brand vocabulary (brand name,
        product name and ``speech.vocabulary`` of brand.yaml), then replaces the words of
        each clip in place (``speech_view`` reports the ``model``). A clip that fails keeps
        its previous words. The language is the brand's, else detected, unless ``language``
        is given. The result of the finished job is ``refine`` in job_status (refined ids,
        skipped clips with the reason, errors). A word that is still wrong afterwards can be
        shown correctly with ``speech_captions.corrections`` (close spellings only).
        """

        root = _project(project_dir)
        if not isinstance(sources, list) or not sources:
            raise _tool_error("sources must be a non-empty list of clip paths or entry ids")
        if not model.strip():
            raise _tool_error("model must not be empty")
        known_ids = set(load_catalog(root).entries)
        for written in sources:
            if not isinstance(written, str):
                raise _tool_error("sources must be strings")
            if written not in known_ids:
                _safe_project_path(root, written, "source")
        entries, skipped = refine_targets(root, sources)
        if not entries:
            reasons = "; ".join(f"{item['path']}: {item['reason']}" for item in skipped)
            raise _tool_error(f"No clip to refine ({reasons or 'nothing matched'})")
        job_id = jobs.submit_refine(root, [entry.id for entry in entries], model.strip(), language)
        return {
            "job_id": job_id,
            "status": "queued",
            "clips": len(entries),
            "model": model.strip(),
            "skipped": skipped,
            "message": (
                f"{len(entries)} clip(s) to transcribe again with {model.strip()}: wait with "
                "job_status(job_id, wait_seconds=90); call it again if the connection drops."
            ),
        }

    @server.tool(structured_output=False)
    def speech_takes(
        project_dir: str,
        source: str | None = None,
        entry_ids: list[str] | None = None,
        max_pause: float = 0.35,
    ) -> dict[str, Any]:
        """Pick the best take of each line across several recordings of a script.

        Pass ``source`` (a project folder such as footage/hooks, or one file) and/or
        ``entry_ids``: the clips to compare, for example HOOK1.mov, HOOK1.1.mov and
        HOOK1.2.mov. Phrases that say the same line (normalised text similarity of at
        least 0.6, inside one file or across files) form a line; each take is ranked on
        fluency (no filler, no restart), a complete ending (last word not cut), a speech
        rate in the natural band (about 1.9 to 4.2 words per second) and the audio level.
        Returns ``lines``: for each, the ``best`` take (entry_id, source, start, end,
        text, score, ``why`` and ready-to-paste ``clean_segments``) and the ``others``
        with the ``issues`` that put them behind. ``skipped`` lists clips without a
        transcript (run library_index with transcribe=true on them).
        """

        root = _project(project_dir)
        if max_pause <= 0:
            raise _tool_error("max_pause must be greater than zero")
        if source is None and not entry_ids:
            raise _tool_error("pass source (a folder or file) or entry_ids")
        catalog = load_catalog(root)
        chosen: dict[str, Any] = {}
        for identifier in entry_ids or []:
            found = catalog.entries.get(identifier)
            if found is None:
                raise _tool_error(f"Unknown catalog entry: {identifier}")
            chosen[found.id] = found
        if source is not None:
            _safe_project_path(root, source, "source")
            prefix = normalise_path(source).rstrip("/")
            for found in catalog.entries.values():
                if found.kind in {"video", "audio"} and (
                    found.path == prefix or found.path.startswith(f"{prefix}/")
                ):
                    chosen[found.id] = found
            if not chosen:
                raise _tool_error(f"No indexed clip under {source!r}: run library_index first")
        library = SpeechLibrary(root, brand_filler_lists(root))
        clips: list[ClipSpeech] = []
        skipped: list[dict[str, Any]] = []
        for found in sorted(chosen.values(), key=lambda item: item.path):
            data = library.data(found.path)
            if data is None:
                skipped.append(
                    {
                        "entry_id": found.id,
                        "source": found.path,
                        "speech_status": found.speech.status
                        if found.speech is not None
                        else "not transcribed",
                    }
                )
                continue
            clips.append(ClipSpeech(found.id, found.path, found.duration, data))
        lines = rank_takes(clips, max_pause=max_pause, lists=library.lists)
        return {"files": len(clips), "lines": lines, "skipped": skipped}

    @server.tool(structured_output=False)
    def reference_decode(
        project_dir: str,
        video: str,
        transcribe: bool | Literal["auto"] = "auto",
        language: str | None = None,
    ) -> list[Any]:
        """Decode a reference video into shots, pacing, text, speech and sheets.

        ``transcribe`` is true, false or "auto" (default: transcribe when the
        reference has audible sound and a transcriber is available). Returns
        the JSON, the overview sheet, then up to 6 shot sheets (one row per
        shot, three frames each). LOOK at every shot sheet and read the
        on-screen text, then call reference_breakdown with the filled
        breakdown_template before choosing footage. The JSON is compact: the
        per-word transcript and rms_per_second stay in reference.json (see
        ``files``); each shot has its ``spoken`` text and ``transcript_text``
        holds the whole transcript. ``breakdown_vocabularies`` lists the
        allowed type, scale, role, motion, emotion, text_speaker and text_device
        values and how to read who speaks in the on-screen text; ``music_profile`` is
        the tempo and energy of the reference soundtrack (voice included).
        """

        root = _project(project_dir)
        _safe_project_path(root, video, "video")
        try:
            result = decode_reference(root, video, transcribe=transcribe, language=language)
        except Exception as exc:
            raise _tool_error(f"Reference decode failed: {exc}") from exc
        sheet_path = _safe_project_path(root, result.sheet, "reference sheet")
        payload = result.model_dump(mode="json")
        payload["sheet_path"] = result.sheet
        words = payload.pop("transcript", None)
        payload.pop("rms_per_second", None)
        payload["transcript_text"] = (
            " ".join(str(word["text"]).strip() for word in words) if words else ""
        )
        payload["files"] = {
            "reference_json": f"{Path(result.sheet).parent.as_posix()}/reference.json",
            "note": "per-word transcript and rms_per_second are kept in reference.json only",
        }
        payload["breakdown_vocabularies"] = {
            "type": list(TYPE_VALUES),
            "scale": list(SCALE_VALUES),
            "role": list(ROLE_VALUES),
            "motion": list(MOTION_VALUES),
            "emotion": list(EMOTION_VALUES),
            "text_speaker": list(TEXT_SPEAKER_VALUES),
            "text_device": list(TEXT_DEVICE_VALUES),
            "reading": (
                "an emoji or a name followed by ':' and quotes means a character is "
                "speaking (text_speaker character, text_device quote); the remake must "
                "keep the same device (who says what), not just the words; "
                "emotion is what the person feels, action (free text) what they do"
            ),
        }
        inline = result.shot_sheets[:6]
        payload["shot_sheets_inline"] = inline
        payload["shot_sheets_not_inline"] = result.shot_sheets[6:]
        images = [_image(sheet_path)]
        for shot_sheet in inline:
            images.append(_image(_safe_project_path(root, shot_sheet, "shot sheet")))
        return [payload, *images]

    @server.tool(structured_output=False)
    def reference_breakdown(
        project_dir: str,
        reference: str,
        shots: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Save the agent's shot-by-shot breakdown of a decoded reference.

        ``reference`` is the decode folder (reference/<slug>) or its
        reference.json. ``shots`` follows breakdown_template: one entry per
        shot, or one entry for consecutive shots (``"shots": [4, 5]``); every
        decoded shot must be covered exactly once. Controlled values: type,
        scale, role, motion, emotion, text_speaker,
        text_device (errors list the allowed ones); ``action`` is free short
        text: what the person does (for example "wipes the counter and sighs"). For every
        shot note what the person feels (``emotion``) and does, then pick a
        library shot with the same emotion and the same kind of action.
        text_speaker and text_device say who speaks in the on-screen text (an
        emoji or name, ":" and quotes is a character talking); subject is a
        short id (person_a, product, diagram), on_screen_text is read verbatim,
        replacement is a library entry id or "missing: ...". Returns type
        demand, scale mix, role sequence, repeated patterns, people, text
        shots, text_devices, dialogue, emotion_by_role (the emotions used for each
        role), pacing and missing shots. Use it before library_overview.
        """

        root = _project(project_dir)
        try:
            summary = breakdown_reference(root, reference, shots)
        except Exception as exc:
            raise _tool_error(f"Reference breakdown failed: {exc}") from exc
        return [summary]

    @server.tool(structured_output=False)
    def library_overview(
        project_dir: str,
        path: str = "footage",
        depth: int = 2,
        mosaic: bool = True,
        page: int = 1,
    ) -> list[Any]:
        """Overview of the footage folders without indexing, plus a mosaic.

        Lists folders up to ``depth`` with file, video and image counts, total
        video seconds and first file names. With ``mosaic``, returns a labelled
        thumbnail mosaic of the media under ``path`` (48 per page, 6 columns);
        ``pages`` tells how many, call again with ``page`` to see the next
        ones. Run it on the whole footage
        tree (B-roll folders included) to find each shot type the reference
        breakdown needs, before choosing shots.
        """

        root = _project(project_dir)
        try:
            payload = _library_overview(root, path, depth=depth, mosaic=mosaic, page=page)
        except Exception as exc:
            raise _tool_error(f"Library overview failed: {exc}") from exc
        result: list[Any] = [payload]
        if payload.get("mosaic"):
            result.append(_image(_safe_project_path(root, payload["mosaic"], "mosaic")))
        return result

    @server.tool(structured_output=False)
    def frame_view(project_dir: str, source: str, t: float) -> list[Any]:
        """Extract one full-resolution frame from a project video or image.

        Use it to inspect the exact moment behind a timecode; the saved frame
        is under cache/frames/.
        """

        root = _project(project_dir)
        frame_path, _ = _extract_or_open_frame(root, source, t)
        return [
            {"path": _relative_path(root, frame_path), "source": source, "t": t},
            _image(frame_path),
        ]

    @server.tool(structured_output=False)
    def recipe_list() -> list[dict[str, Any]]:
        """List available ad recipes with purpose and duration.

        Use this to shortlist a shape, then read its checklist with recipe_get
        before writing a storyboard.
        """

        return [
            {
                "id": recipe.id,
                "name": recipe.name,
                "summary": recipe.summary,
                "duration": recipe.duration.model_dump(mode="json"),
            }
            for recipe in list_recipes()
        ]

    @server.tool(structured_output=False)
    def recipe_get(recipe_id: str) -> list[dict[str, Any]]:
        """Return one complete recipe, including rules and example.

        Use its checklist as the storyboard acceptance criteria.
        """

        try:
            return [get_recipe(recipe_id).model_dump(mode="json")]
        except KeyError as exc:
            raise _tool_error(str(exc)) from exc

    @server.tool(structured_output=False)
    def editing_guide() -> str:
        """Return the editing doctrine for footage, copy, rendering and QC.

        Read it before the first storyboard and whenever a rule is unclear.
        """

        path = Path(__file__).parent / "assets" / "EDITING_GUIDE.md"
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise _tool_error(f"Packaged editing guide is unavailable: {exc}") from exc

    @server.tool(structured_output=False)
    def storyboard_schema() -> dict[str, Any]:
        """Return the JSON Schema for Storyboard v1.

        Use it to construct agent-authored storyboards before validation.
        """

        return storyboard_json_schema()

    @server.tool(structured_output=False)
    def storyboard_save(
        project_dir: str, storyboard: dict[str, Any], name: str
    ) -> dict[str, Any]:
        """Save a storyboard in the project as ``storyboards/<name>.json`` and return its path.

        Send the storyboard object once, then pass the returned path
        (``"storyboards/<name>.json"``) as ``storyboard`` to ``storyboard_validate``,
        ``render_start`` and ``caption_preview`` instead of repeating the whole object
        (useful on a remote server). Saving again under the same name replaces the file.
        The object is stored as sent, not validated: ``storyboard_validate`` checks it.
        """

        root = _project(project_dir)
        path = _storyboard_file(root, name)
        if not isinstance(storyboard, Mapping):
            raise _tool_error("storyboard must be an object")
        _write_json(path, dict(storyboard))
        saved: dict[str, Any] = {
            "path": _relative_path(root, path),
            "bytes": path.stat().st_size,
            "segments": len(storyboard.get("segments") or []),
            "overlays": len(storyboard.get("overlays") or []),
        }
        sfx = _expanded_sfx(root, storyboard)
        if sfx is not None:
            saved["sfx"] = sfx
        return saved

    @server.tool(structured_output=False)
    def storyboard_patch(
        project_dir: str, name: str, changes: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Edit a saved storyboard in place: ``changes`` is a list of ``{path, value}``.

        A path walks the JSON with dots and indexes: ``segments.3.out`` (``segments[3].out``
        works too). Setting a missing key of an object adds it; the index just past the end of
        a list, or ``-``, appends; ``{"path": "overlays.1", "remove": true}`` deletes. The
        changes apply in order to a copy: if one is wrong, nothing is saved and the error
        names it. When the changes rename the storyboard (``name``), its ``output.path`` follows
        if it was the default of the old name (a path set by hand stays). The file
        ``storyboards/<name>.json`` is rewritten and the result lists the
        schema problems left (``problems``, empty when the storyboard is valid; run
        ``storyboard_validate`` for the full check).
        """

        root = _project(project_dir)
        path = _storyboard_file(root, name)
        if not path.is_file():
            raise _tool_error(
                f"Storyboard {_relative_path(root, path)} does not exist: save it first "
                "with storyboard_save"
            )
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            patched = apply_changes(document, changes)
            follow_rename(document, patched)
        except (OSError, ValueError) as exc:
            raise _tool_error(f"Could not patch {_relative_path(root, path)}: {exc}") from exc
        _write_json(path, patched)
        problems: list[dict[str, Any]] = []
        try:
            Storyboard.model_validate(patched)
        except ValidationError as exc:
            problems = [
                {"where": issue.where, "message": issue.message} for issue in _schema_issues(exc)
            ]
        return {
            "path": _relative_path(root, path),
            "applied": [str(change["path"]) for change in changes],
            "problems": problems,
        }

    @server.tool(structured_output=False)
    def caption_preview(
        project_dir: str,
        source: str,
        t: float,
        overlay: dict[str, Any],
        emoji: str | None = None,
        storyboard: dict[str, Any] | str | None = None,
    ) -> list[Any]:
        """Render a caption or comment over a real 9:16 frame.

        Use it before rendering to inspect placement; the JSON reports bbox,
        fitted size, overflow and final lines. ``emoji`` takes the storyboard
        ``emoji`` values (twemoji, none or a project folder) so the preview
        matches the render; ``emoji_missing`` lists emoji absent from a folder.
        For a ``voice_captions`` overlay pass ``manifest``: the ``audio.voice``
        path of your storyboard (voice/<name>/manifest.json); ``expanded``
        lists the resulting captions once. A ``speech_captions`` overlay is
        built from the speech of the clips: pass ``chunk_index`` to pick the
        caption to draw and ``storyboard`` (a project path such as
        ``storyboards/<name>.json`` or the object) so the timeline is the one of
        your ad; without it the captions are those of ``source`` played whole.
        ``expanded`` lists them all. An animated caption (``animation``
        ``karaoke``, ``word`` or ``pop``) is drawn in one of its states, to check
        where the highlight lands: put ``chunk_index`` and ``word_index`` in the
        overlay object to pick the chunk and the lit word, or leave them out to
        get the state on screen at timeline time ``t``. The result names the
        ``word_index``, the ``window`` of the state and how many ``states`` the
        chunk has.
        """

        root = _project(project_dir)
        payload, image_path = render_caption_preview(
            root, source, t, overlay, emoji=emoji, storyboard=storyboard
        )
        return [payload, _image(image_path)]

    @server.tool(structured_output=False)
    def end_card(project_dir: str, spec: dict[str, Any], name: str) -> list[Any]:
        """Compose a clean end card image from a real product photo.

        Use it only when the brand has no ready-made call-to-action image: when
        it has one (``brand.cta_image``, a ``cta/`` folder or ``brand/``),
        use that image as the last segment (``image``, ``zoom_in``, about 3 s)
        instead. It writes
        ``brand/endcards/<name>.png`` (1080x1920 unless ``spec.format`` says
        otherwise) and returns the path, the layout boxes and the image: LOOK at
        it. ``spec``: ``product_image`` (project path of a photo or a video, with
        ``t`` for the frame, optional ``crop`` {x,y,w,h} in the source's display
        pixels, before scaling: on a 1920x1080 frame ``{"x": 600, "y": 100,
        "w": 700, "h": 700}`` keeps a 700 px square; the result returns
        ``source_size`` [w, h] to measure it), ``pill``, ``title``
        (a string or runs ``{text, accent}``), ``subline``, ``badges`` (up to 3,
        ``{text, icon: check|star|none}``), ``button`` (``{text, arrow:
        down|right|none}``), colours ``accent``, ``text``, ``muted``,
        ``background`` (two colours), ``family``, ``emoji``. Colours default to
        the brand's. Banned words of the brand on any text are an error. Use the
        PNG as the last segment: ``type: image``, ``motion: zoom_in``, about 3 s,
        label ``end_card``, no subtitles on it.
        """

        root = _project(project_dir)
        try:
            output, payload = write_end_card(root, spec, name)
        except ToolError:
            raise
        except Exception as exc:
            raise _tool_error(f"Could not compose the end card: {exc}") from exc
        return [payload, _image(output)]

    @server.tool(structured_output=False)
    def storyboard_validate(
        project_dir: str,
        storyboard: dict[str, Any] | str,
        recipe_id: str | None = None,
        score_check: bool = False,
    ) -> dict[str, Any]:
        """Validate a storyboard object or project-relative JSON file.

        ``storyboard`` is the object, or the path saved by ``storyboard_save``
        (``"storyboards/<name>.json"``). Every schema error is reported at once.

        The check covers Storyboard v1, its files, brand and recipe. Fix every
        error path before render_start; warnings are questions to review. With
        ``reference`` set, ``ref_*`` warnings and a ``reference_comparison``
        block compare the edit with the reference breakdown. With
        ``score_check=true`` it also runs the quick pre-check of ``ad_score``
        that needs no pixels, as warnings prefixed ``score_`` (reading time and
        line length, product too early, call to action, platform safe zones of
        the overlays); ``summary.score_precheck`` lists what passed and what
        could not be checked.
        """

        root = _project(project_dir)
        _, report = _validate(root, storyboard, recipe_id, score_check)
        return _validation_json(report)

    @server.tool(structured_output=False)
    def render_start(
        project_dir: str,
        storyboard: dict[str, Any] | str,
        quality: Literal["final", "preview", "draft"] = "final",
    ) -> dict[str, Any]:
        """Validate and queue a background render.

        ``storyboard`` is the object or a saved path (``"storyboards/<name>.json"``).

        Use preview while iterating and final after QC. A rejected storyboard
        returns validation errors and creates no job.
        """

        root = _project(project_dir)
        quality = normalise_quality(quality)
        if quality not in {"preview", "final"}:
            raise _tool_error("quality must be 'preview', 'final' or 'draft' (= preview)")
        model, report = _validate(root, storyboard, None)
        if not report.ok:
            payload = _validation_json(report)
            payload["status"] = "rejected"
            return payload
        storyboard_path = root / "renders" / f"{model.name}.storyboard.json"
        _write_json(storyboard_path, model.model_dump(mode="json", by_alias=True))
        job_id = jobs.submit(root, model, quality)
        return {"job_id": job_id, "status": "queued"}

    @server.tool(structured_output=False)
    def variants_build(
        project_dir: str,
        base: dict[str, Any] | str,
        variants: dict[str, Any],
        name_pattern: str = "{base}_{hook}_{cta}",
        contact_sheet: bool = True,
        overwrite: bool = False,
    ) -> list[Any]:
        """Build the matrix of variants of one storyboard: hooks x bodies x calls to action.

        ``base`` is a storyboard (object or ``"storyboards/<name>.json"``). ``variants`` is
        ``{axis: [{id, changes}]}`` for up to 3 axes (for example ``hook``, ``body``, ``cta``).
        ``changes`` is a list applied in order, each a ``storyboard_patch`` change (``{path,
        value}`` or ``{path, remove: true}``) or a SLOT REPLACEMENT ``{replace_label: "hook",
        segments: [...], overlays: [...]}``: it swaps every segment of that label (they must be
        next to each other) and shifts everything after it by the difference of duration
        (overlays, sounds and word times follow). The new ``overlays`` are timed from the start
        of the slot; the overlays that were on the replaced segments go away. A voice-over
        (``audio.voice``) laid on the timeline must still match: a change of duration while one
        is set is refused with the reason, as is an overlay that only partly overlaps the slot.
        An option with no ``changes`` (or ``[]``) is the control: the base unchanged. Timed
        sounds follow the slot: an ``sfx`` overlay after it shifts by the difference of duration,
        one inside it goes away when the new slot brings its own sounds (else it keeps its
        offset); each variant lists what moved under ``moved``. All the errors of all the
        variants come back at once (variant, path, message).

        Makes the cartesian product (at most 24, each validated), saves each as
        ``storyboards/<name>.json`` (``name_pattern`` uses ``{base}`` and the axis names; the
        default falls back to the axis ids in order when an axis is not hook or cta) and returns
        the list with the path, ids, duration and warnings of each, plus a contact sheet: the
        first 3 s (top row) and the last 3 s (bottom row, the call to action) of every variant
        (LOOK at it, a hook that does not read at that size will not read in the feed). A base
        that is itself a variant keeps its ids in the new ones (the manifest keeps
        ``hook=h2;cta=ca``). Then ``render_batch`` renders them with a shared cache.
        Vary the hook first; keep everything else identical.

        A storyboard file that already exists is NEVER overwritten, whoever made it (an earlier
        ``variants_build`` included, or a file edited by hand since): the error lists every file
        that would be replaced. ``overwrite=true`` replaces them all. Renders may be replaced;
        storyboards may not.
        """

        root = _project(project_dir)
        document = _load_storyboard_raw(root, base)
        try:
            built = expand_matrix(document, variants, name_pattern, root=root)
        except ValueError as exc:
            raise _tool_error(f"Could not build the variants: {exc}") from exc
        listed: list[dict[str, Any]] = []
        problems: list[dict[str, str]] = []
        for variant in built:
            _model, report = _validate(root, variant.document, None)
            if not report.ok:
                label = ", ".join(f"{axis}={ident}" for axis, ident in variant.ids.items())
                for issue in report.errors:
                    problems.append(
                        {
                            "variant": label,
                            "path": str(issue.where or ""),
                            "message": f"{variant.name} does not validate: {issue.message}",
                        }
                    )
                continue
            variant.warnings = [
                f"{issue.where + ': ' if issue.where else ''}{issue.message}"
                for issue in report.warnings
            ]
        if problems:
            raise _tool_error(f"Could not build the variants: {format_errors(problems)}")
        parent_ids = variant_ids(root, str(document.get("name", "")))
        if not overwrite:
            kept = existing_files(
                root, [_relative_path(root, _storyboard_file(root, v.name)) for v in built]
            )
            if kept:
                raise _tool_error(
                    f"Refusing to overwrite {len(kept)} existing storyboard file(s): "
                    f"{', '.join(kept)}. A storyboard is never replaced silently, even one "
                    "an earlier variants_build made. Change name_pattern, rename the files, "
                    "or pass overwrite=true."
                )
        for variant in built:
            path = _storyboard_file(root, variant.name)
            _write_json(path, variant.document)
            _write_json(
                path.with_name(f"{variant.name}.variant.json"),
                {
                    "base": str(document.get("name", "")),
                    "ids": {**parent_ids, **variant.ids},
                },
            )
            item: dict[str, Any] = {
                "name": variant.name,
                "path": _relative_path(root, path),
                "ids": {**parent_ids, **variant.ids},
                "duration": round(variant.duration, 3),
                "warnings": variant.warnings,
            }
            if variant.moved:
                item["moved"] = variant.moved
            listed.append(item)
        payload: dict[str, Any] = {
            "base": str(document.get("name", "")),
            "count": len(listed),
            "variants": listed,
        }
        images: list[Any] = []
        if contact_sheet:
            sheet = root / "renders" / "variants" / f"{payload['base']}_variants_sheet.jpg"
            try:
                made, notes = variants_sheet(root, built, sheet, cache=Cache(root / "cache"))
                payload["contact_sheet"] = _relative_path(root, made)
                if notes:
                    payload["notes"] = notes
                images.append(_image(made))
            except Exception as exc:
                payload["notes"] = [f"contact sheet failed: {exc}"]
        return [payload, *images]

    def _batch_documents(
        root: Path, storyboards: list[dict[str, Any] | str]
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        if not isinstance(storyboards, list) or not storyboards:
            raise _tool_error("storyboards must be a non-empty list of objects or saved paths")
        documents: list[dict[str, Any]] = []
        for number, entry in enumerate(storyboards):
            _model, report = _validate(root, entry, None)
            if not report.ok:
                payload = _validation_json(report)
                payload["status"] = "rejected"
                payload["storyboard_index"] = number
                return [], payload
            documents.append(_load_storyboard_raw(root, entry))
        return documents, None

    def _submit_batch(
        root: Path,
        documents: list[dict[str, Any]],
        formats: list[str] | None,
        batch: str | None,
        quality: str,
        score: bool,
    ) -> dict[str, Any]:
        quality = normalise_quality(quality)
        if quality not in {"preview", "final"}:
            raise _tool_error("quality must be 'preview', 'final' or 'draft' (= preview)")
        name = batch or f"{documents[0].get('name', 'batch')}_batch"
        try:
            items, notes = plan_items(root, documents, formats, name)
        except ValueError as exc:
            raise _tool_error(str(exc)) from exc
        for item in items:
            _model, report = _validate(root, item["storyboard"], None)
            if not report.ok:
                payload = _validation_json(report)
                payload["status"] = "rejected"
                payload["storyboard"] = item["name"]
                return payload
            _write_json(
                root / "renders" / name / "_storyboards" / f"{item['name']}.json",
                item["storyboard"],
            )
        job_id = jobs.submit_batch(root, name, items, quality, score)
        result: dict[str, Any] = {
            "job_id": job_id,
            "status": "queued",
            "batch": name,
            "renders": [{"name": item["name"], "format": item["ratio"]} for item in items],
            "manifest": f"renders/{name}/manifest.csv",
        }
        if notes:
            result["notes"] = notes
        return result

    @server.tool(structured_output=False)
    def render_batch(
        project_dir: str,
        storyboards: list[dict[str, Any] | str],
        quality: Literal["final", "preview", "draft"] = "final",
        batch: str | None = None,
        formats: list[str] | None = None,
        score: bool = True,
    ) -> dict[str, Any]:
        """Queue ONE background job that renders several storyboards in order.

        ``quality``: ``final``, ``preview`` or ``draft`` (an alias of ``preview``: half-size
        ``*.preview.mp4`` files; the report and the manifest say ``preview``, and the manifest
        has a ``quality`` column because a preview scores lower than its final).

        ``storyboards``: objects or saved paths (typically the variants of
        ``variants_build``). The segment cache is shared, so a segment used by several variants
        (the body, the end card) is built once: the first render is the slow one. ``formats``
        (any of ``"9:16"``, ``"4:5"``, ``"1:1"``) renders each storyboard in those ratios; when
        omitted each storyboard's own ``formats`` field is used, else it renders as it is. Files
        go to ``renders/<batch>/`` (``batch`` defaults to ``<first name>_batch``) named
        ``<name>[_9x16|_4x5|_1x1].mp4``, and ``renders/<batch>/manifest.csv`` lists them for an
        ads manager (file, variant ids, format, duration, score, hook text, CTA text).
        ``score`` runs ``ad_score`` on each final render for the manifest (with the safe zones of
        its placement). ``job_status`` shows the progress per render, each output as it
        finishes, and at the end ``total_seconds``, ``cache_builds`` and the manifest path. A
        storyboard that does not validate rejects the whole batch with its errors and no job.
        """

        root = _project(project_dir)
        documents, rejected = _batch_documents(root, storyboards)
        if rejected is not None:
            return rejected
        return _submit_batch(root, documents, formats, batch, quality, score)

    @server.tool(structured_output=False)
    def render_formats(
        project_dir: str,
        storyboard: dict[str, Any] | str,
        formats: list[str] | None = None,
        quality: Literal["final", "preview", "draft"] = "final",
        batch: str | None = None,
        score: bool = True,
    ) -> dict[str, Any]:
        """Render one storyboard in several formats: ``9:16``, ``4:5``, ``1:1``.

        ``formats`` defaults to the storyboard's own ``formats`` field, else all three. One
        detached job (see ``render_batch``), one file per format: ``<name>_9x16.mp4``,
        ``<name>_4x5.mp4``, ``<name>_1x1.mp4`` in ``renders/<batch>/`` (``batch`` defaults to
        ``<name>_formats``). Each is the same ad laid out for its placement: the footage is
        re-cropped (face-aware with ``focus: "face"``, else by ``focus``); a still or an end card
        that does not fit is shown whole over its blurred copy; captions, stickers and images
        keep their place by bands (the part the platform leaves free is mapped onto the free part
        of the new frame, 4:5 and 1:1 being the Feed) and are pushed back inside when a block
        would leave it. ``ad_score`` uses the safe zones of the matching placement (``feed`` for
        4:5 and 1:1). LOOK at each format's contact sheet before delivery.
        """

        root = _project(project_dir)
        documents, rejected = _batch_documents(root, [storyboard])
        if rejected is not None:
            return rejected
        wanted = formats if formats is not None else (documents[0].get("formats") or list(FORMATS))
        name = batch or f"{documents[0].get('name', 'batch')}_formats"
        return _submit_batch(root, documents, list(wanted), name, quality, score)


    @server.tool(structured_output=False)
    def voice_generate(project_dir: str, name: str, spec: dict[str, Any]) -> dict[str, Any]:
        """Queue a checked voice-over generation job.

        Poll job_status until it is done, then use each manifest line's
        ``start`` and ``duration`` to place shots. If a line is ``suspect``,
        regenerate it with slightly different wording or accept it when the
        transcript differs only by punctuation; agents cannot listen to audio.
        ``unmatched_words`` lists the key words (4+ letters) missing from the
        transcript: two or more make the line suspect, so reword them.
        A line is a string (said and shown) or ``{"say": ..., "show": ...}``
        when captions must display something other than the spoken words
        (say "forty-nine dollars ninety", show "$49.90"). Calling again with an
        existing ``name`` replaces that voice (unchanged lines come from the
        cache); the job summary then has ``replaced: true``. ``language`` in the
        spec is optional: without it the ``language`` of brand.yaml is used,
        else ``en``.
        """

        root = _project(project_dir)
        try:
            voice_spec = VoiceSpec.model_validate(spec)
            if not name or Path(name).name != name or name in {".", ".."}:
                raise ValueError("voice name must be a single safe directory name")
        except Exception as exc:
            raise _tool_error(f"Invalid voice specification: {exc}") from exc
        replaced = (root / "voice" / name / "manifest.json").is_file()
        job_id = jobs.submit_voice(root, name, voice_spec)
        return {"job_id": job_id, "status": "queued", "replaced": replaced}

    @server.tool(structured_output=False)
    def sfx_list(
        project_dir: str, fetch: bool = False, offline: bool = False
    ) -> dict[str, Any]:
        """List the sound effects of the project's ``sfx/`` folder, and build the pack.

        ``fetch=true`` first builds the missing sounds of the pack (about 18: whoosh short and
        long, swoosh, swipe, pop, click, tap, ding, chime, notification, cash register,
        camera shutter, record scratch, typing, riser 1 s and 2 s, impact, boom). Each is
        downloaded from a pinned CC0 source (checked by SHA-1) or synthesised in code, trimmed
        and set to one level, with the source and licence in ``sfx/LICENCES.txt``. A download
        that fails is synthesised instead (``offline=true`` synthesises all of them). The
        files you drop in ``sfx/`` yourself are listed too. Each sound returns its ``name``,
        ``path`` (the ``source`` of an ``sfx`` overlay), ``duration`` and ``origin``. Use a
        sound in an ``sfx`` overlay (``{"type": "sfx", "source": "sfx/pop.wav", "at": 3.2,
        "volume": 0.8}``) or by name in ``audio.sfx_auto``. A riser is placed so that its end
        meets the moment it builds towards: ``at`` = that moment minus its duration.
        """

        root = _project(project_dir)
        try:
            return list_sfx(root, fetch=fetch, offline=offline)
        except (SfxError, OSError, ValueError) as exc:
            raise _tool_error(f"Sound list failed: {exc}") from exc

    @server.tool(structured_output=False)
    def music_beats(
        project_dir: str,
        path: str,
        start: float = 0.0,
        duration: float | None = None,
    ) -> dict[str, Any]:
        """Beat times of a music track, as it will play in the ad.

        ``path`` is a project file (audio, or a video whose soundtrack is the music). ``start``
        is the storyboard's ``audio.start``: the times are in seconds from the start of the
        ad, so beat 0 of the result is the first beat heard. They come from the onset envelope
        and the estimated tempo, on a regular grid whose phase is aligned to the strongest
        onsets (within a few milliseconds on a steady beat). ``duration`` limits the analysis
        (default the first 120 s). A track without a steady beat (a piano, a pad) returns no
        beats. Cut on these times for a music-led ad, or let ``storyboard_snap`` do it.
        """

        root = _project(project_dir)
        target = _safe_project_path(root, path, "path")
        if not target.is_file():
            raise _tool_error(f"File does not exist: {path}")
        try:
            grid = audio_module.beat_grid(target, start=start, duration=duration)
        except Exception as exc:
            raise _tool_error(f"Beat analysis failed: {exc}") from exc
        payload: dict[str, Any] = {
            "path": _relative_path(root, target),
            "start": start,
            "analysed_seconds": round(grid.analysed, 2),
            "tempo_bpm": grid.tempo,
            "count": len(grid.beats),
            "beats": grid.beats,
        }
        if not grid.beats:
            payload["note"] = "no steady beat: a pad or a piano has nothing to cut on"
        return payload

    @server.tool(structured_output=False)
    def storyboard_snap(
        project_dir: str,
        storyboard: dict[str, Any] | str,
        to: Literal["beats"] = "beats",
        max_shift: float = 0.2,
        save_as: str | None = None,
    ) -> dict[str, Any]:
        """Move the cuts of a storyboard onto the beats of its music.

        ``storyboard`` is the object or a saved path. Each cut within ``max_shift`` seconds
        (default 0.2) of a beat of ``audio.music`` (from ``audio.start``) moves to it: the
        ``out`` of the earlier clip (or the ``duration`` of an image) changes, ``in`` points and
        segment order never do, no segment ends up under 0.5 s (or shorter than it was), the
        other cuts and the total duration stay, and the captions, images and sounds that
        started on a moved cut move with it. With ``audio.clips`` the cuts of talking segments
        never move. The result lists ``moves``, the cuts ``kept`` with the reason, and how many
        cuts are on a beat before and after. It returns the new ``storyboard``, or with
        ``save_as`` (a name) writes ``storyboards/<name>.json`` and returns its path instead.
        The snapped storyboard has its own identity so a render never overwrites the original:
        with ``save_as`` its ``name`` is ``save_as``, without it the old name plus ``_snapped``,
        and ``output.path`` is ``renders/<that name>.mp4`` (the quality is kept).
        Run ``storyboard_validate`` afterwards: its ``beat_alignment`` info shows the share of
        cuts on a beat.
        """

        root = _project(project_dir)
        raw = _load_storyboard_raw(root, storyboard)
        try:
            result = snap_storyboard(root, raw, to=to, max_shift=max_shift)
        except ValidationError as exc:
            problems = "; ".join(
                f"{issue.where}: {issue.message}" for issue in _schema_issues(exc)
            )
            raise _tool_error(f"Invalid storyboard: {problems}") from exc
        except (ValueError, OSError) as exc:
            raise _tool_error(f"Snap failed: {exc}") from exc
        # The snapped version is a new storyboard: its name and its render path are its own, so
        # rendering it never overwrites the video of the original.
        if save_as is not None:
            path = _storyboard_file(root, save_as)
            rename_storyboard(result["storyboard"], path.stem)
            _write_json(path, result.pop("storyboard"))
            result["saved"] = _relative_path(root, path)
        else:
            rename_storyboard(
                result["storyboard"], snapped_name(result["storyboard"].get("name"))
            )
        return result

    @server.tool(structured_output=False)
    def music_profile(project_dir: str, path: str | None = None) -> dict[str, Any]:
        """Tempo, loudness and energy of the music tracks, to match a reference.

        Without ``path``, profiles every audio file of ``music/``; with ``path``
        (project-relative), one file, which may be a video such as the reference
        (its soundtrack mixes voice and music). Each track returns its duration,
        integrated loudness (LUFS), estimated tempo (``tempo_bpm``), ``energy``
        (onsets per second and loudness range in LU) and ``feel``: calm, medium or
        energetic (thresholds in ``feel_thresholds``). Pick the library track whose
        profile is closest to the reference's: a calm track under a fast reference
        feels flat.
        """

        root = _project(project_dir)
        try:
            return audio_module.music_profile(root, path)
        except Exception as exc:
            raise _tool_error(f"Music profile failed: {exc}") from exc

    @server.tool(structured_output=False)
    def music_search(
        project_dir: str,
        query: str,
        mood: str | None = None,
        min_duration: float | None = None,
        max_duration: float | None = None,
        source: str = "openverse",
        limit: int = 12,
        fit_to: str | None = None,
        storyboard: str | None = None,
        feel: str | None = None,
    ) -> dict[str, Any]:
        """Find music whose licence lets an ad use it (commercial use and modification).

        Searches Openverse (no key; ``source="openverse"``), Freesound (``"freesound"``, only with
        the ``FREESOUND_API_KEY`` environment variable) or both (``"all"``; without the key the
        answer says so and still lists Openverse). Kept licences: CC0, PDM, CC BY, CC BY-SA;
        NC and ND are dropped. Each track returns ``source``, ``id``, ``title``, ``author``,
        ``licence``, ``duration`` (s), ``source_url``, ``preview_url`` (listen before fetching)
        and ``attribution_required`` (with the ``attribution_text`` to put in the ad's
        description). ``mood`` adds words to the query ("calm", "upbeat"). ``min_duration`` and
        ``max_duration`` filter in seconds. ``fit_to="storyboard"`` with ``storyboard`` (a
        project path such as ``storyboards/summer.json``) ranks for that ad: tracks at least as
        long as the ad first, then the wanted feel (``audio.music_feel``, or the ``feel`` argument:
        calm, medium, energetic) against ``fit.feel_guess``, a guess from the track's OWN title,
        tags and genres (``unknown`` when they say nothing; never copied from the wanted feel).
        The real feel, tempo and loudness come from ``music_fetch``: trust that, not the guess.
        Fetch the best ones, listen, and choose.
        """

        root = _project(project_dir)
        try:
            return _music_search(
                root,
                query,
                mood=mood,
                min_duration=min_duration,
                max_duration=max_duration,
                source=source,
                limit=limit,
                fit_to=fit_to,
                storyboard=storyboard,
                feel=feel,
            )
        except MusicError as exc:
            raise _tool_error(f"Music search failed: {exc}") from exc

    @server.tool(structured_output=False)
    def music_fetch(project_dir: str, source: str, id: str) -> dict[str, Any]:
        """Download a track from ``music_search`` into ``music/``, with its licence.

        ``source`` and ``id`` come from the search result. Writes ``music/<file>``, the record
        ``music/<file>.licence.json`` (source, id, URL, title, author, licence, licence URL,
        attribution text, fetch date) and a line in ``music/LICENCES.txt``, then profiles the
        track: the answer has ``path`` (the ``audio.music`` value), ``duration``, ``tempo_bpm``,
        ``feel`` and ``loudness_lufs``. A licence that is not CC0, PDM, CC BY or CC BY-SA is
        refused. An embedded cover picture is removed from the stored file. The render and
        ``ad_score`` read the record (the sidecar, else the track's line in ``LICENCES.txt``):
        ``music_licence_unknown`` when a track has neither,
        ``music_attribution_required`` (with the exact text) when it needs one.
        Freesound serves its high-quality MP3 preview (the original needs a login).
        """

        root = _project(project_dir)
        try:
            return _music_fetch(root, source, id)
        except (MusicError, OSError) as exc:
            raise _tool_error(f"Music fetch failed: {exc}") from exc

    @server.tool(structured_output=False)
    def music_generate(
        project_dir: str,
        prompt: str,
        duration: float = 30.0,
        steps: int = 27,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Make an instrumental track locally (optional ACE-Step environment), as a job.

        Needs ``adanvil music-setup`` (about 3 GB of packages, then 8 GB of weights at the first
        run). ``prompt`` is a list of tags ("upbeat acoustic guitar, warm, 110 bpm,
        instrumental"); ``duration`` 10 to 120 s. On a CPU a 30 s track takes several minutes:
        poll ``job_status``; its ``music`` field has the ``path`` (the ``audio.music`` value),
        tempo, feel and loudness. The model and its weights are Apache-2.0: no attribution; the
        licence record is written next to the file like for a fetched track.
        """

        root = _project(project_dir)
        from adanvil.musicgen import MAX_DURATION, MIN_DURATION, available

        if not isinstance(prompt, str) or not prompt.strip():
            raise _tool_error("prompt must not be empty")
        if not MIN_DURATION <= duration <= MAX_DURATION:
            raise _tool_error(
                f"duration must be between {MIN_DURATION:g} and {MAX_DURATION:g} seconds"
            )
        if not 8 <= steps <= 60:
            raise _tool_error("steps must be between 8 and 60")
        if not available():
            raise _tool_error(
                "The music environment is not installed: run 'adanvil music-setup' "
                "(optional; music_search finds licensed tracks without it)."
            )
        job_id = jobs.submit_music(root, prompt.strip(), float(duration), int(steps), seed)
        return {"job_id": job_id, "status": "queued"}

    @server.tool(structured_output=False)
    def voice_view(project_dir: str, name: str, offset: float = 0.0) -> dict[str, Any]:
        """Read a voice manifest and suggest final-timeline shot boundaries.

        ``offset`` is the intended voice start on the final video timeline.
        Suspect lines should be regenerated with slightly different wording
        or accepted only when the transcript difference is punctuation.
        Each line's ``unmatched_words`` lists key words the transcript lacks.
        """

        root = _project(project_dir)
        if offset < 0:
            raise _tool_error("offset must be non-negative")
        manifest = _safe_project_path(root, f"voice/{name}/manifest.json", "voice manifest")
        if not manifest.is_file():
            raise _tool_error(f"Voice manifest does not exist: {_relative_path(root, manifest)}")
        try:
            track = load_track(manifest)
        except (OSError, ValueError) as exc:
            raise _tool_error(str(exc)) from exc
        summary = _voice_summary(root, track)
        summary["timeline"] = [
            {
                "line": line.index,
                "start": line.start + offset,
                "end": line.start + line.duration + offset,
            }
            for line in track.lines
        ]
        return summary

    @server.tool(structured_output=False)
    async def job_status(
        job_id: str, wait_seconds: float = 0, project_dir: str | None = None
    ) -> list[Any]:
        """Read a queued, running, completed or failed render, batch, voice, index or refine job.

        With ``wait_seconds`` greater than zero, wait in this call (up to 600
        seconds) instead of polling from the client; if the connection drops,
        just call again, a restarted server still finds the job. Pass
        ``project_dir`` when it says "job not found". Completed index jobs
        carry the compact index summary; completed speech_refine jobs carry
        ``refine`` (refined clips, skipped clips with the reason, errors). Completed render jobs
        include the persisted RenderReport JSON; batch jobs (``render_batch``,
        ``render_formats``) carry ``batch``: one entry per render (status, file, format,
        duration, score, seconds, error) as they finish, then ``total_seconds``,
        ``cache_builds`` and the ``manifest`` path; completed voice jobs include
        each manifest line and its warnings. If a line is ``suspect``, an
        agent cannot listen to it: regenerate with slightly different wording,
        or accept it when the transcript differs only in punctuation.
        """

        project = _project(project_dir) if project_dir is not None else None
        try:
            status = await asyncio.to_thread(jobs.status, job_id, wait_seconds, project)
        except ValueError as exc:
            raise _tool_error(str(exc)) from exc
        images: list[Any] = []
        if status["status"] == "done" and status.get("sheet"):
            sheet_path = Path(str(status["sheet"]))
            if sheet_path.is_file():
                images.append(_image(sheet_path))
        return [status, *images]

    @server.tool(structured_output=False)
    def qc_run(
        project_dir: str,
        video: str,
        render_report: str | None = None,
        storyboard: dict[str, Any] | str | None = None,
    ) -> list[Any]:
        """Run technical QC on a rendered project-relative video.

        Pass a job id or saved report path when available. Inspect the returned
        sheet, the end card frame (``endcard.jpg``, per format) and up to four overlay
        crops before delivery. ``storyboard`` is the EFFECTIVE storyboard of that render (the
        one of its format after the re-layout: ``renders/<batch>/_storyboards/<name>.json``,
        found by itself next to a batch render): the overlay crops are then taken where its
        overlays really are (a caption moved by ``render_formats`` is not cut in half), and
        ``caption_on_face`` checks the rendered frames for a caption on a face (needs the
        vision environment; every format, 9:16 too).
        """

        root = _project(project_dir)
        video_path = _safe_project_path(root, video, "video")
        if not video_path.is_file():
            raise _tool_error(f"Video file does not exist: {video}")
        report = _report_from_argument(root, jobs, render_report)
        effective: dict[str, Any] | None = None
        if storyboard is not None:
            effective = _load_storyboard_raw(root, storyboard)
        else:
            sidecar = video_path.parent / "_storyboards" / f"{video_path.stem}.json"
            if sidecar.is_file():
                try:
                    effective = json.loads(sidecar.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    effective = None
        try:
            fmt = (
                Format(width=540, height=960)
                if report and report.quality == "preview"
                else Format()
            )
            seen = probe(video_path)
            if seen.width and seen.height and ratio_of(seen.width, seen.height) in {"4:5", "1:1"}:
                # a 4:5 or 1:1 render of render_formats: its own size is the expected one
                fmt = Format(width=seen.width, height=seen.height, fps=fmt.fps)
            with _QC_LOCK:
                original_tempdir = qc_module.tempfile.TemporaryDirectory
                if os.name == "nt":
                    qc_module.tempfile.TemporaryDirectory = _DirectTemporaryDirectory
                try:
                    result = qc_report(
                        video_path, render_report=report, fmt=fmt, storyboard=effective, root=root
                    )
                finally:
                    qc_module.tempfile.TemporaryDirectory = original_tempdir
        except Exception as exc:
            raise _tool_error(f"QC failed: {exc}") from exc
        payload = _jsonable(result)
        if not isinstance(payload, dict):  # pragma: no cover
            raise _tool_error("QC returned an invalid report")
        image_paths = [Path(str(path)) for path in result.images]
        payload["images"] = [_relative_path(root, path) for path in image_paths]
        inline: list[Any] = []
        named = {path.name: path for path in image_paths}
        for name in ("sheet.jpg", "end.jpg", "endcard.jpg"):
            path = named.get(name)
            if path is not None:
                inline.append(_image(path))
        crops = [path for path in image_paths if path.name.startswith("overlay_")][:4]
        inline.extend(_image(path) for path in crops)
        return [payload, *inline]

    @server.tool(structured_output=False)
    def ad_score(
        project_dir: str,
        video: str,
        storyboard: dict[str, Any] | str | None = None,
        platforms: list[str] | None = None,
    ) -> list[Any]:
        """Score a rendered ad out of 100 for creative quality, with findings to fix.

        ``qc_run`` says the file is technically correct; this says whether the ad
        is any good: a first second that holds attention (motion or a cut, a face
        or the product, a text within 0.5 s, audio at once), a pacing without a
        dip in the middle, captions that can be read (0.25 s per word + 0.6 s,
        7 words a line at most, contrast of 3:1 or more on the real frames), the
        problem before the product, a call to action and an end card in the last
        4 s, overlays out of the platform interface (``platforms``: any of
        ``tiktok``, ``reels``, ``stories``, ``feed``; the first three by default, ``feed``
        for a 4:5 or 1:1 video) and a clear
        loudness, a voice 6 dB over the music and no silence over 0.7 s.
        Pass the ``storyboard`` it was rendered from (object or saved path) for
        exact timings, words, roles and overlays; without it the checks that
        need them are listed in ``skipped``. The result has ``score``,
        ``sub_scores`` (hook, pacing, reading, story, safe_zones, audio),
        ``findings`` sorted by ``impact`` (each with ``time``, ``severity``,
        ``message`` and ``fix``), ``passed``, ``skipped``, the pixel boxes of the
        ``safe_zones`` and the ``thresholds`` (guidelines, not rules). Images:
        ``safe_zones_preview.jpg`` (the zones of each platform over a mid-ad
        frame, red where an overlay falls inside) and the frame of each top
        finding: LOOK at them. Run it before delivery, fix the top findings,
        render again, and aim for 80 or more.
        """

        root = _project(project_dir)
        video_path = _safe_project_path(root, video, "video")
        if not video_path.is_file():
            raise _tool_error(f"Video file does not exist: {video}")
        raw = _load_storyboard_raw(root, storyboard) if storyboard is not None else None
        if platforms is None:
            # a 4:5 or 1:1 render is judged against the Feed, not Stories and Reels
            seen = probe(video_path)
            found = platforms_for_size(seen.width or 0, seen.height or 0)
            if found is not None and found != ("tiktok", "reels", "stories"):
                platforms = list(found)
        try:
            with _QC_LOCK:
                original_tempdir = qc_module.tempfile.TemporaryDirectory
                if os.name == "nt":
                    qc_module.tempfile.TemporaryDirectory = _DirectTemporaryDirectory
                try:
                    report = _ad_score(root, video_path, storyboard=raw, platforms=platforms)
                finally:
                    qc_module.tempfile.TemporaryDirectory = original_tempdir
        except Exception as exc:
            raise _tool_error(f"Scoring failed: {exc}") from exc
        payload = report.as_dict()
        payload["images"] = [_relative_path(root, path) for path in report.images]
        return [payload, *(_image(path) for path in report.images)]

    @server.tool(structured_output=False)
    def compare_pair(
        project_dir: str,
        before: str,
        after: str,
        before_t: float | None = None,
        after_t: float | None = None,
        before_crop: dict[str, int] | None = None,
        after_crop: dict[str, int] | None = None,
    ) -> list[Any]:
        """Measure how different a before and an after look, to rank candidate pairs.

        ``before`` and ``after`` are project files: images, or clips (the frame
        at ``before_t`` / ``after_t`` seconds, the middle of the clip by default).
        ``before_crop`` / ``after_crop`` (``{x, y, w, h}`` in source pixels) cut
        one half out of a side-by-side composite, the same crop a ``reveal`` side
        takes. The comparison is made on the face when a skin-region heuristic
        finds one in both pictures (so a different distance still lines up), else
        on the whole frame after aligning sizes. ``visible_difference`` is 0 to
        100 (under 12 not visible, under 30 subtle, under 55 visible, above that
        obvious at a glance); ``region`` says what was compared. The preview image
        shows the two pictures side by side: LOOK at it. Compare every available
        pair (photos, composites split with ``crop``, frames of before/after
        clips) and use the one whose difference is obvious; a still pair of the
        same pose beats moving clips for a reveal.
        """

        root = _project(project_dir)
        first = _safe_project_path(root, before, "before")
        second = _safe_project_path(root, after, "after")
        for label, path in (("before", first), ("after", second)):
            if not path.is_file():
                raise _tool_error(f"The {label} file does not exist: {path.name}")
        try:
            payload, preview = _compare_pair(
                root,
                first,
                second,
                before_t=before_t,
                after_t=after_t,
                before_crop=before_crop,
                after_crop=after_crop,
            )
        except PairError as exc:
            raise _tool_error(str(exc)) from exc
        except Exception as exc:
            raise _tool_error(f"Comparison failed: {exc}") from exc
        payload["before"]["path"] = before
        payload["after"]["path"] = after
        payload["preview"] = _relative_path(root, preview)
        return [payload, _image(preview)]

    return server


def main() -> None:
    """Run the AdAnvil MCP server over stdio."""

    build_server().run("stdio")


__all__ = ["build_server", "main", "render_caption_preview"]
