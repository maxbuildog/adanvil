"""V2.8: the variant matrix, batch rendering with a shared cache, formats and the manifest."""

from __future__ import annotations

import copy
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from adanvil import batch as batch_module
from adanvil._bin import ffmpeg_path
from adanvil.batch import plan_items, run_batch
from adanvil.ffmpeg import probe, run
from adanvil.formats import FORMATS, check_ratios, ratio_of, reformat
from adanvil.jobs import read_job, start_job
from adanvil.media import Cache, ClipSegment, Format, ImageSegment, render_segment
from adanvil.media import segment as segment_module
from adanvil.project import init_project
from adanvil.safezones import collisions, platforms_for_size
from adanvil.storyboard import Storyboard
from adanvil.storyboard.craft import overlay_items
from adanvil.variants import VariantError, expand_matrix, hook_preview, replace_slot
from adanvil.vision import client as vision_client
from adanvil.vision.engine import FAKE_FACE_RGB, FakeVisionEngine

FACE = FAKE_FACE_RGB


def _make_clip(path: Path, seconds: float = 3.0, colour: str = "0x305070") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg_path(),
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-t",
            str(seconds),
            "-i",
            f"color=c={colour}:s=360x640:r=10",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-g",
            "5",
            str(path),
        ],
        check=True,
    )


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    init_project(root, "Variants")
    for name, colour in (("hook_a", "0x803020"), ("hook_b", "0x208030"), ("hook_c", "0x202080")):
        _make_clip(root / "footage" / f"{name}.mp4", 2.0, colour)
    _make_clip(root / "footage" / "body.mp4", 3.0)
    Image.new("RGB", (216, 384), (230, 230, 230)).save(root / "footage" / "card.png")
    return root


def _base(voice: bool = False) -> dict[str, Any]:
    document: dict[str, Any] = {
        "name": "demo",
        "format": {"width": 1080, "height": 1920, "fps": 10},
        "emoji": "none",
        "segments": [
            {"type": "clip", "source": "footage/hook_a.mp4", "in": 0, "out": 2, "label": "hook"},
            {"type": "clip", "source": "footage/body.mp4", "in": 0, "out": 3, "label": "body"},
            {"type": "image", "source": "footage/card.png", "duration": 1.5, "label": "end_card"},
        ],
        "overlays": [
            {"type": "caption", "style": "pill", "lines": ["Old hook"], "start": 0.2, "end": 1.8},
            {"type": "caption", "style": "outline", "lines": ["Body line"], "start": 2.5, "end": 4},
            {
                "type": "caption",
                "style": "outline",
                "lines": ["two words"],
                "start": 3.0,
                "end": 4.0,
                "animation": "word",
                "word_times": [[3.0, 3.5], [3.5, 4.0]],
            },
            {"type": "caption", "style": "pill", "lines": ["Shop now"], "start": 5.0, "end": 6.4},
            {"type": "progress", "start": 0.0},
        ],
    }
    if voice:
        document["audio"] = {"voice": "voice/lines/manifest.json"}
    return document


def _hook(seconds: list[float], text: str) -> dict[str, Any]:
    return {
        "replace_label": "hook",
        "segments": [
            {
                "type": "clip",
                "source": f"footage/hook_{'abc'[i]}.mp4",
                "in": 0,
                "out": length,
                "label": "hook",
            }
            for i, length in enumerate(seconds)
        ],
        "overlays": [
            {"type": "caption", "style": "pill", "lines": [text], "start": 0.3, "end": 1.0}
        ],
    }


# --------------------------------------------------------------------------- the matrix


def test_matrix_makes_the_product_and_shifts_what_follows() -> None:
    variants = {
        "hook": [
            {"id": "long", "changes": [_hook([1.5, 1.0], "Long hook")]},
            {"id": "short", "changes": [_hook([1.5], "Short hook")]},
        ],
        "cta": [
            {"id": "shop", "changes": [{"path": "overlays.3.lines.0", "value": "Shop today"}]},
            {"id": "learn", "changes": [{"path": "overlays.3.lines.0", "value": "Learn more"}]},
        ],
    }
    built = expand_matrix(_base(), variants)
    assert [v.name for v in built] == [
        "demo_long_shop",
        "demo_long_learn",
        "demo_short_shop",
        "demo_short_learn",
    ]
    assert [round(v.duration, 3) for v in built] == [7.0, 7.0, 6.0, 6.0]
    long, short = built[0].document, built[2].document
    assert long["name"] == "demo_long_shop"
    assert long["output"]["path"] == "renders/demo_long_shop.mp4"
    for variant in built:
        Storyboard.model_validate(variant.document)  # every variant is a valid storyboard
    # the old hook caption went away, the new one is timed from the slot start (0)
    texts = [o.get("lines") for o in long["overlays"] if o["type"] == "caption"]
    assert ["Old hook"] not in texts and ["Long hook"] in texts
    new = next(o for o in long["overlays"] if o.get("lines") == ["Long hook"])
    assert new["start"] == pytest.approx(0.3) and new["end"] == pytest.approx(1.0)
    # +0.5 s: later overlays and their word times follow, a whole-ad overlay stays
    body = next(o for o in long["overlays"] if o.get("lines") == ["Body line"])
    assert body["start"] == pytest.approx(3.0) and body["end"] == pytest.approx(4.5)
    words = next(o for o in long["overlays"] if o.get("lines") == ["two words"])
    assert words["word_times"] == [[3.5, 4.0], [4.0, 4.5]]
    progress = next(o for o in long["overlays"] if o["type"] == "progress")
    assert progress["start"] == 0.0 and "end" not in progress
    # -0.5 s the other way
    body_short = next(o for o in short["overlays"] if o.get("lines") == ["Body line"])
    assert body_short["start"] == pytest.approx(2.0)
    cta = next(o for o in short["overlays"] if o.get("lines") == ["Shop today"])
    assert cta["start"] == pytest.approx(4.5)
    # the other axis changed nothing else
    assert built[1].document["segments"] == built[0].document["segments"]


def test_a_patch_only_matrix_and_the_default_name_pattern() -> None:
    variants = {
        "body": [
            {"id": f"b{i}", "changes": [{"path": "overlays.1.lines.0", "value": f"Line {i}"}]}
            for i in range(3)
        ]
    }
    built = expand_matrix(_base(), variants)  # no {hook} axis: the default pattern adapts
    assert [v.name for v in built] == ["demo_b0", "demo_b1", "demo_b2"]


def test_refusals_say_why() -> None:
    base = _base()

    def refused(variants: dict[str, Any], match: str, document: Any = base, **kw: Any) -> None:
        with pytest.raises(VariantError, match=match):
            expand_matrix(document, variants, **kw)

    many = [{"id": f"h{i}", "changes": [{"path": "name", "value": "x"}]} for i in range(5)]
    refused({"hook": many, "cta": many}, "25 variants: at most 24")
    refused({"a": many[:1], "b": many[:1], "c": many[:1], "d": many[:1]}, "at most 3")
    refused({"hook": [many[0], many[0]]}, "appears twice")
    refused(
        {"hook": [{"id": "x", "changes": [{"replace_label": "nope", "segments": [1]}]}]},
        "no segment is labelled 'nope'",
    )
    refused(
        {"hook": [{"id": "x", "changes": [{"path": "segments.9.out", "value": 1}]}]},
        "variant hook=x",
    )
    refused(
        {
            "hook": [
                {"id": "x", "changes": [{"path": "name", "value": "same"}]},
                {"id": "y", "changes": [{"path": "name", "value": "same"}]},
            ]
        },
        "for two variants",
        name_pattern="fixed",
    )
    # a label that is split in two runs
    split = copy.deepcopy(base)
    split["segments"].insert(1, {**split["segments"][0], "label": "other"})
    split["segments"].append({**split["segments"][0]})
    refused(
        {"hook": [{"id": "x", "changes": [_hook([1.0], "t")]}]}, "not next to each other", split
    )
    # an overlay half in the slot
    partial = copy.deepcopy(base)
    partial["overlays"][0]["end"] = 2.6
    refused(
        {"hook": [{"id": "x", "changes": [_hook([1.0], "t")]}]}, "only partly overlaps", partial
    )
    # an invalid replacement segment
    bad = {
        "replace_label": "hook",
        "segments": [{"type": "clip", "source": "a.mp4", "in": 2, "out": 1}],
    }
    refused({"hook": [{"id": "x", "changes": [bad]}]}, "not a valid segment")


def test_a_voice_over_must_still_match() -> None:
    voiced = _base(voice=True)
    with pytest.raises(VariantError, match="voice-over"):
        expand_matrix(voiced, {"hook": [{"id": "x", "changes": [_hook([1.0], "t")]}]})
    with pytest.raises(VariantError, match="voice-over"):  # a patch that moves the timeline too
        expand_matrix(
            voiced, {"hook": [{"id": "x", "changes": [{"path": "segments.0.out", "value": 1.0}]}]}
        )
    # the same duration keeps the voice valid: the picture changes, nothing shifts
    same = expand_matrix(voiced, {"hook": [{"id": "x", "changes": [_hook([1.2, 0.8], "t")]}]})
    assert same[0].duration == pytest.approx(6.5)
    # a change that is not about time is fine with a voice
    fine = expand_matrix(
        voiced, {"cta": [{"id": "x", "changes": [{"path": "overlays.3.lines.0", "value": "Go"}]}]}
    )
    assert fine[0].document["overlays"][3]["lines"] == ["Go"]


def test_replace_slot_leaves_the_base_alone_and_keeps_spanning_overlays() -> None:
    base = _base()
    before = copy.deepcopy(base)
    base["overlays"].append(
        {"type": "caption", "style": "pill", "lines": ["spans"], "start": 1.0, "end": 6.0}
    )
    spanning = copy.deepcopy(base)
    body = {
        "replace_label": "body",
        "segments": [{"type": "clip", "source": "footage/body.mp4", "in": 0, "out": 2}],
    }
    result, delta = replace_slot(base, body)  # the body slot is 2 s to 5 s
    assert base == spanning  # a pure function
    assert delta == pytest.approx(-1.0)
    span = next(o for o in result["overlays"] if o.get("lines") == ["spans"])
    assert span["start"] == pytest.approx(1.0) and span["end"] == pytest.approx(5.0)
    gone = [o.get("lines") for o in result["overlays"]]
    assert ["Body line"] not in gone and ["two words"] not in gone  # timed on the old body
    assert before["segments"][0]["label"] == "hook"


def test_hook_preview_keeps_whole_segments_so_the_cache_is_shared() -> None:
    preview = hook_preview(_base(), 1.0)
    assert len(preview["segments"]) == 1 and preview["segments"][0] == _base()["segments"][0]
    assert preview["output"]["quality"] == "preview"
    assert len(hook_preview(_base(), 3.0)["segments"]) == 2


# --------------------------------------------------------------------------- formats


def test_formats_field_is_validated_and_old_storyboards_are_unchanged() -> None:
    plain = Storyboard.model_validate(_base())
    assert plain.formats == [] and "formats" not in plain.model_dump(mode="json")
    both = Storyboard.model_validate({**_base(), "formats": ["9:16", "4:5"]})
    assert both.model_dump(mode="json")["formats"] == ["9:16", "4:5"]
    with pytest.raises(ValueError):
        Storyboard.model_validate({**_base(), "formats": ["16:9"]})
    with pytest.raises(ValueError, match="repeat"):
        Storyboard.model_validate({**_base(), "formats": ["4:5", "4:5"]})
    assert check_ratios(["1:1"]) == ["1:1"]
    with pytest.raises(ValueError):
        check_ratios(["2:1"])
    assert ratio_of(1080, 1350) == "4:5" and ratio_of(1920, 1080) is None
    assert platforms_for_size(1080, 1080) == ("feed",)
    assert platforms_for_size(1080, 1920) == ("tiktok", "reels", "stories")


def _layout_document(root: Path) -> dict[str, Any]:
    document = _base()
    document["overlays"] = [
        {
            "type": "caption",
            "style": "pill",
            "lines": ["A top hook that is long", "on two lines"],
            "start": 0,
            "end": 2,
            "y": 330,
        },
        {
            "type": "caption",
            "style": "outline",
            "lines": ["Bottom line up", "two lines"],
            "start": 1,
            "end": 4,
            "y": 1500,
            "anchor": "bottom",
        },
        {
            "type": "caption",
            "style": "pill",
            "lines": ["Low block", "three", "lines here"],
            "start": 2,
            "end": 5,
            "y": 1450,
        },
        {
            "type": "sticker",
            "kind": "badge",
            "text": "NEW",
            "x": 540,
            "y": 1000,
            "start": 1,
            "end": 3,
        },
        {
            "type": "caption",
            "style": "pill",
            "lines": ["Shop now"],
            "start": 5,
            "end": 6.4,
            "y": 1380,
        },
    ]
    return document


@pytest.mark.parametrize("ratio", ["4:5", "1:1"])
def test_reformat_keeps_captions_inside_the_frame_and_the_feed_safe_zones(
    tmp_path: Path, ratio: str
) -> None:
    root = _project(tmp_path)
    document = _layout_document(root)
    moved, notes = reformat(document, ratio, root=root)
    spec = FORMATS[ratio]
    assert moved["format"] == {"width": 1080, "height": spec.height, "fps": 10}
    assert moved["name"] == f"demo_{spec.suffix}"
    assert moved["output"]["path"] == f"renders/demo_{spec.suffix}.mp4"
    assert document == _layout_document(root)  # the source is not touched
    model = Storyboard.model_validate(moved)
    from adanvil.storyboard.craft import collect_captions

    timeline = model.timeline()
    entries, _ = collect_captions(
        model, root, None, [a for a, _ in timeline], [b for _, b in timeline]
    )
    items = overlay_items(model, root, None, entries)
    assert len(items) == 5
    for item in items:
        left, top, right, bottom = item["box"]
        assert 0 <= left < right <= spec.width, item
        assert 0 <= top < bottom <= spec.height, item
        assert not collisions(item["box"], spec.placement, (spec.width, spec.height)), item
    # the y of the first caption follows its band: higher in the shorter frame
    assert moved["overlays"][0]["y"] < document["overlays"][0]["y"]
    assert all(isinstance(note, str) for note in notes)


def test_reformat_to_the_same_ratio_changes_only_the_name() -> None:
    document = _base()
    same, notes = reformat(document, "9:16")
    assert same["overlays"] == document["overlays"] and same["segments"] == document["segments"]
    assert same["name"] == "demo_9x16" and notes == []
    # from another ratio back to 9:16 the bands map the other way
    feed, _ = reformat({**document, "overlays": [{**document["overlays"][0], "y": 330}]}, "4:5")
    back, _ = reformat(feed, "9:16")
    assert back["format"]["height"] == 1920
    assert abs(back["overlays"][0]["y"] - 330) <= 3


def test_end_card_is_shown_whole_when_the_ratio_changes() -> None:
    document = _base()
    document["segments"][2]["fit"] = "cover"
    moved, notes = reformat(document, "1:1")
    assert moved["segments"][2]["fit"] == "contain" and any("shown whole" in n for n in notes)
    assert reformat(document, "9:16")[0]["segments"][2]["fit"] == "cover"


def _frames(video: Path, size: tuple[int, int]) -> np.ndarray:
    raw = run(
        ["-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_stdout=True
    ).stdout
    return np.frombuffer(raw or b"", dtype=np.uint8).reshape(-1, size[1], size[0], 3)


def _blob(frame: np.ndarray) -> tuple[float, float] | None:
    mask = np.all(np.abs(frame.astype(int) - np.array(FACE)) <= 45, axis=-1)
    if mask.sum() < 10:
        return None
    rows, columns = np.nonzero(mask)
    return (
        (columns.min() + columns.max() + 1) / 2 / frame.shape[1],
        (rows.min() + rows.max() + 1) / 2 / frame.shape[0],
    )


@pytest.mark.parametrize(("width", "height"), [(216, 270), (216, 216)])
def test_face_stays_in_frame_in_4x5_and_1x1(tmp_path: Path, width: int, height: int) -> None:
    root = tmp_path / "project"
    init_project(root, "Face")
    source = root / "footage" / "walk.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    graph = (
        "color=c=0x203040:s=640x360:r=10,format=yuv420p[bg];"
        f"color=c=0x{FACE[0]:02X}{FACE[1]:02X}{FACE[2]:02X}:s=60x60:r=10,format=yuv420p[face];"
        "[bg][face]overlay=x='140+t*80':y=107:shortest=1[v]"
    )
    subprocess.run(
        [
            ffmpeg_path(),
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-t",
            "3",
            "-filter_complex",
            graph,
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-g",
            "5",
            str(source),
        ],
        check=True,
    )
    segment = ClipSegment.model_validate(
        {"type": "clip", "source": "footage/walk.mp4", "in": 0, "out": 2.5, "focus": "face"}
    )
    with vision_client.use_engine(FakeVisionEngine()):
        out = render_segment(
            segment,
            root=root,
            cache=Cache(root / "cache"),
            fmt=Format(width=width, height=height, fps=10),
        )
    frames = _frames(out, (width, height))
    assert len(frames) >= 20
    for index in (1, 12, 22):
        blob = _blob(frames[index])
        assert blob is not None, index
        assert 0.15 < blob[0] < 0.85 and 0.1 < blob[1] < 0.9, (index, blob)
        assert blob[0] == pytest.approx(0.5, abs=0.12)  # centred while it walks


def test_a_still_that_does_not_fit_is_shown_whole_over_its_blur(tmp_path: Path) -> None:
    root = tmp_path / "project"
    init_project(root, "Still")
    picture = Image.new("RGB", (216, 384), (120, 120, 120))
    for x in range(216):
        for y in range(10):
            picture.putpixel((x, y), (250, 20, 20))  # red band on top
            picture.putpixel((x, 383 - y), (20, 20, 250))  # blue band at the bottom
    (root / "footage").mkdir(exist_ok=True)
    picture.save(root / "footage" / "card.png")
    segment = ImageSegment(source="footage/card.png", duration=1.0)
    out = render_segment(
        segment, root=root, cache=Cache(root / "cache"), fmt=Format(width=216, height=216, fps=10)
    )
    frame = _frames(out, (216, 216))[3]
    assert frame[3, 108, 0] > 180 and frame[3, 108, 2] < 90  # the red band is on screen
    assert frame[212, 108, 2] > 180 and frame[212, 108, 0] < 90  # and the blue one
    assert frame[100, 5].tolist() != frame[100, 108].tolist()  # the sides are the blur


# --------------------------------------------------------------------------- batch and cache


def _built(root: Path, hooks: int = 3, ctas: int = 2) -> list[Any]:
    variants = {
        "hook": [
            {"id": f"h{i}", "changes": [_hook([1.0 + i * 0.5], f"Hook {i}")]} for i in range(hooks)
        ],
        "cta": [
            {"id": f"c{i}", "changes": [{"path": "overlays.3.lines.0", "value": f"Go {i}"}]}
            for i in range(ctas)
        ],
    }
    return expand_matrix(_base(), variants, root=root)


def test_batch_shares_the_segment_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path)
    built = _built(root)
    assert len(built) == 6
    calls: list[str] = []
    original = segment_module._build_clip_base

    def counting(source: Path, seg: Any, *args: Any, **kwargs: Any) -> None:
        calls.append(f"{source.name}:{seg.in_}-{seg.out}")
        return original(source, seg, *args, **kwargs)

    monkeypatch.setattr(segment_module, "_build_clip_base", counting)
    items, _ = plan_items(root, [v.document for v in built], None, "test_batch")
    assert [i["name"] for i in items] == [v.name for v in built]
    seen: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    result = run_batch(
        root,
        {"batch": "test_batch", "items": items, "quality": "preview", "score": False},
        progress=events.append,
        publish=seen.append,
    )
    assert all(r["status"] == "done" for r in result["renders"]), result["renders"]
    # 3 distinct hooks + 1 shared body, built once for six renders
    assert sorted(calls) == sorted(set(calls))
    assert len(calls) == 4, calls
    assert result["cache_builds"] >= 4 and result["total_seconds"] > 0
    assert len(seen) == 6 and seen[0]["done"] is False and seen[0]["count"] == 6
    assert [e["done"] for e in events if e["stage"] == "rendering"] == list(range(6))
    # a second batch finds everything in the cache: no clip is built at all
    calls.clear()
    again = run_batch(
        root, {"batch": "second", "items": items, "quality": "preview", "score": False}
    )
    assert calls == [] and again["cache_builds"] < result["cache_builds"]
    # per-render files live in the batch folder
    first = result["renders"][0]
    assert first["file"] == "renders/test_batch/demo_h0_c0.preview.mp4"
    assert (root / first["file"]).is_file() and (root / first["report"]).is_file()


def test_manifest_lists_every_render_for_an_ads_manager(tmp_path: Path) -> None:
    root = _project(tmp_path)
    built = _built(root, hooks=2, ctas=2)
    for variant in built:
        side = root / "storyboards" / f"{variant.name}.variant.json"
        side.parent.mkdir(exist_ok=True)
        side.write_text(json.dumps({"base": "demo", "ids": variant.ids}), encoding="utf-8")
    items, _ = plan_items(root, [v.document for v in built[:3]], ["9:16", "4:5"], "manifest_demo")
    assert [i["name"] for i in items][:2] == ["demo_h0_c0_9x16", "demo_h0_c0_4x5"]
    result = run_batch(
        root, {"batch": "manifest_demo", "items": items, "quality": "preview", "score": False}
    )
    path = root / result["manifest"]
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    assert rows[0] == [
        "file",
        "variant ids",
        "format",
        "duration",
        "score",
        "hook text",
        "CTA id",
        "CTA text",
        "music licence",
        "quality",
    ]
    assert len(rows) == 1 + 6
    by_file = {row[0]: row for row in rows[1:]}
    row = by_file["demo_h0_c0_4x5.preview.mp4"]
    assert row[1] == "hook=h0;cta=c0" and row[2] == "4:5"
    assert float(row[3]) == pytest.approx(5.5, abs=0.2)
    assert row[4] == ""  # score=False: not scored
    assert row[5] == "Hook 0" and row[6] == "c0" and row[7] == "caption: Go 0"
    assert by_file["demo_h1_c0_9x16.preview.mp4"][5] == "Hook 1"
    sizes = {(r["width"], r["height"]) for r in result["renders"]}
    assert sizes == {(540, 960), (540, 674)}  # the size of the preview files, not the storyboard's
    for entry in result["renders"]:
        info = probe(root / entry["file"])
        assert (info.width, info.height) == (entry["width"], entry["height"])
        assert entry["quality"] == "preview"


def test_a_batch_survives_one_failing_render(tmp_path: Path) -> None:
    root = _project(tmp_path)
    built = _built(root, hooks=1, ctas=2)
    items, _ = plan_items(root, [v.document for v in built], None, "partial")
    items[0]["storyboard"]["segments"][1]["source"] = "footage/missing.mp4"
    result = run_batch(
        root, {"batch": "partial", "items": items, "quality": "preview", "score": False}
    )
    assert [r["status"] for r in result["renders"]] == ["failed", "done"]
    assert "missing" in result["renders"][0]["error"]
    rows = (root / result["manifest"]).read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2


def test_plan_items_refuses_duplicate_names_and_bad_batches(tmp_path: Path) -> None:
    root = _project(tmp_path)
    with pytest.raises(ValueError, match="named"):
        plan_items(root, [_base(), _base()], None, "dup")
    with pytest.raises(ValueError, match="batch name"):
        plan_items(root, [_base()], None, "../escape")
    with pytest.raises(ValueError, match="formats"):
        plan_items(root, [_base()], ["2:1"], "bad")
    own, _ = plan_items(root, [{**_base(), "formats": ["1:1"]}], None, "own")
    assert [i["name"] for i in own] == ["demo_1x1"]


def test_batch_job_runs_detached_and_reports_each_render(tmp_path: Path) -> None:
    root = _project(tmp_path)
    built = _built(root, hooks=1, ctas=2)
    items, _ = plan_items(root, [v.document for v in built], None, "job_batch")
    record = start_job(
        root,
        "batch",
        {"batch": "job_batch", "items": items, "quality": "preview", "score": False},
        progress={"total": len(items), "unit": "renders"},
    )
    job_file = Path(record["job_file"])
    deadline = time.monotonic() + 240
    result = read_job(job_file)
    while result.get("status") not in {"done", "failed"} and time.monotonic() < deadline:
        time.sleep(0.3)
        result = read_job(job_file)
    assert result["status"] == "done", result
    assert result["progress"] == {"done": 2, "total": 2, "unit": "renders"}
    summary = result["batch"]
    assert summary["done"] is True and summary["count"] == 2
    assert [r["name"] for r in summary["renders"]] == ["demo_h0_c0", "demo_h0_c1"]
    assert (root / "renders" / "job_batch" / "manifest.csv").is_file()


@pytest.mark.parametrize("ratio", ["4:5", "1:1"])
def test_formats_render_score_and_qc_use_the_feed_zones(tmp_path: Path, ratio: str) -> None:
    from adanvil.qc import ad_score, qc_report

    root = _project(tmp_path)
    document = _layout_document(root)
    items, _ = plan_items(root, [document], [ratio], "formats_demo")
    result = run_batch(
        root, {"batch": "formats_demo", "items": items, "quality": "final", "score": True}
    )
    entry = result["renders"][0]
    assert entry["status"] == "done", entry
    spec = FORMATS[ratio]
    info = probe(root / entry["file"])
    assert (info.width, info.height) == (spec.width, spec.height)
    assert entry["file"] == f"renders/formats_demo/demo_{spec.suffix}.mp4"
    assert isinstance(entry["score"], int) and 0 <= entry["score"] <= 100
    report = ad_score(
        root, root / entry["file"], storyboard=items[0]["storyboard"], platforms=["feed"]
    )
    assert report.safe_zones and "feed" in json.dumps(report.safe_zones)
    assert not [f for f in report.findings if f["id"] == "safe_zone"]
    qc = qc_report(root / entry["file"], fmt=Format(width=spec.width, height=spec.height, fps=10))
    assert next(c for c in qc.checks if c.id == "resolution").status == "pass"


def test_tools_build_variants_and_render_formats_through_the_server(tmp_path: Path) -> None:
    import anyio
    from mcp import Client

    from adanvil.server import build_server

    root = _project(tmp_path)
    variants = {
        "hook": [
            {"id": "a", "changes": [_hook([1.0], "Hook a")]},
            {"id": "b", "changes": [_hook([1.5], "Hook b")]},
        ],
        "cta": [{"id": "go", "changes": [{"path": "overlays.3.lines.0", "value": "Go"}]}],
    }

    async def scenario() -> tuple[dict, dict, dict, dict]:
        async with Client(build_server()) as client:
            made = await client.call_tool(
                "variants_build",
                {"project_dir": str(root), "base": _base(), "variants": variants},
            )
            payload = json.loads(next(i.text for i in made.content if i.type == "text"))
            refused = await client.call_tool(
                "variants_build",
                {"project_dir": str(root), "base": _base(voice=True), "variants": variants},
            )
            error = {"is_error": refused.is_error, "text": refused.content[0].text}
            started = await client.call_tool(
                "render_formats",
                {
                    "project_dir": str(root),
                    "storyboard": payload["variants"][0]["path"],
                    "formats": ["4:5", "1:1"],
                    "quality": "preview",
                    "score": False,
                },
            )
            queued = json.loads(started.content[0].text)
            done = await client.call_tool(
                "job_status",
                {"job_id": queued["job_id"], "wait_seconds": 240, "project_dir": str(root)},
            )
            final = json.loads(done.content[0].text)
            return payload, error, queued, final

    payload, error, queued, final = anyio.run(scenario)
    assert payload["count"] == 2 and payload["contact_sheet"].endswith("_variants_sheet.jpg")
    assert [v["name"] for v in payload["variants"]] == ["demo_a_go", "demo_b_go"]
    assert (root / "storyboards" / "demo_a_go.json").is_file()
    assert (root / "storyboards" / "demo_a_go.variant.json").is_file()
    assert (root / payload["contact_sheet"]).is_file()
    assert error["is_error"] and "voice-over" in error["text"]
    assert [r["name"] for r in queued["renders"]] == ["demo_a_go_4x5", "demo_a_go_1x1"]
    assert final["status"] == "done", final
    assert final["batch"]["count"] == 2 and final["batch"]["manifest"].endswith("manifest.csv")
    assert all(r["status"] == "done" for r in final["batch"]["renders"])


def test_batch_module_exports() -> None:
    assert batch_module.MANIFEST_COLUMNS[0] == "file"


def test_overlays_on_an_end_card_follow_the_card_not_the_bands() -> None:
    document = _base()  # "Shop now" starts with the end card (a contained still)
    document["overlays"][3]["y"] = 1380
    moved, _ = reformat(document, "4:5")
    on_card = next(o for o in moved["overlays"] if o.get("lines") == ["Shop now"])
    assert on_card["y"] == round(1380 * 1350 / 1920)  # scaled with the picture
    body = next(o for o in moved["overlays"] if o.get("lines") == ["Body line"])
    # a caption on footage uses the bands, and a default y (330) follows the layout too
    assert body["y"] != round(330 * 1350 / 1920) and "y" in body
