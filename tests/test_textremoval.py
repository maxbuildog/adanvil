from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import adanvil.textremoval.core as textremoval_core
from adanvil.ffmpeg import extract_frame, run
from adanvil.media import Cache, ClipSegment, render_segment
from adanvil.textremoval import (
    RemoveText,
    build_mask,
    detect_text_boxes,
    ffmpeg_filter_path,
)


def _source(media: dict[str, dict[str, object]], name: str = "burnt_text.mp4") -> Path:
    return Path(str(media[name]["path"]))


def _box_union(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area_left = (left[2] - left[0]) * (left[3] - left[1])
    area_right = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (area_left + area_right - intersection)


def test_filter_path_works_with_windows_and_space_paths(tmp_path: Path) -> None:
    assert ffmpeg_filter_path(r"C:\Users\x\a b\m.png") == "'C\\:\\\\Users\\\\x\\\\a\\ b\\\\m.png'"
    assert ffmpeg_filter_path("/tmp/m.png") == "'/tmp/m.png'"

    directory = tmp_path / "dir with space"
    directory.mkdir()
    mask = directory / "m.png"
    Image.new("L", (64, 64), 255).save(mask)
    output = directory / "out.mp4"
    run(
        [
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:r=1",
            "-loop",
            "1",
            "-i",
            str(mask),
            "-filter_complex",
            f"[0:v]format=yuv420p,removelogo=f={ffmpeg_filter_path(mask)}[v]",
            "-map",
            "[v]",
            "-frames:v",
            "1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
    )
    assert output.is_file()


def test_detect_text_boxes_matches_manifest(media: dict[str, dict[str, object]]) -> None:
    source = _source(media)
    expected = tuple(int(value) for value in media["burnt_text.mp4"]["text_bbox"])
    boxes = detect_text_boxes(source)
    assert 1 <= len(boxes) <= 2
    assert _iou(_box_union(boxes), expected) >= 0.6
    assert all(box[1] < 900 for box in boxes)


def test_detect_text_boxes_ignores_tall_bright_ellipse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = np.zeros((4, 400, 400, 3), dtype=np.uint8)
    ellipse = Image.new("L", (400, 400), 0)
    from PIL import ImageDraw

    ImageDraw.Draw(ellipse).ellipse((140, 70, 260, 330), fill=255)
    bright = np.asarray(ellipse)
    frames[:, :, :, :] = bright[:, :, None]
    monkeypatch.setattr(textremoval_core, "sample_frames", lambda *args, **kwargs: frames)

    assert detect_text_boxes("synthetic-ellipse.mp4") == []


def test_build_mask_is_non_empty_and_rejects_empty_box(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    source = _source(media)
    box = tuple(int(value) for value in media["burnt_text.mp4"]["text_bbox"])
    rt = RemoveText(method="mask", box=box)
    path = build_mask(source, 0, 3, rt, tmp_path / "mask.png")
    mask = np.asarray(Image.open(path)) > 0
    assert mask.any()
    x0, y0, x1, y1 = box
    grown = mask[max(0, y0 - rt.dilate) : y1 + rt.dilate, max(0, x0 - rt.dilate) : x1 + rt.dilate]
    assert grown.sum() / mask.sum() >= 0.9

    with pytest.raises(ValueError, match="mask is empty"):
        build_mask(
            source,
            0,
            3,
            RemoveText(method="mask", box=(0, 1500, 100, 1600)),
            tmp_path / "empty.png",
        )


@pytest.mark.parametrize("method", ["delogo", "mask", "crop_out"])
def test_render_methods_remove_burnt_text(
    media: dict[str, dict[str, object]], tmp_path: Path, method: str
) -> None:
    source = _source(media)
    box = tuple(int(value) for value in media["burnt_text.mp4"]["text_bbox"])
    plain = render_segment(
        ClipSegment(source="burnt_text.mp4", **{"in": 0.5, "out": 1.5, "gamma": None}),
        root=source.parent,
        cache=Cache(tmp_path / "plain-cache"),
    )
    removed = render_segment(
        ClipSegment(
            source="burnt_text.mp4",
            **{"in": 0.5, "out": 1.5, "gamma": None},
            remove_text=RemoveText(method=method, box=box),
        ),
        root=source.parent,
        cache=Cache(tmp_path / "removed-cache"),
    )
    plain_frame = tmp_path / f"plain-{method}.png"
    removed_frame = tmp_path / f"removed-{method}.png"
    extract_frame(plain, 0.5, plain_frame)
    extract_frame(removed, 0.5, removed_frame)
    plain_pixels = np.asarray(Image.open(plain_frame).convert("RGB"))
    removed_pixels = np.asarray(Image.open(removed_frame).convert("RGB"))

    if method == "crop_out":
        boxes = detect_text_boxes(removed)
        assert all(box_item[1] >= 1920 * 0.3 for box_item in boxes)
        return

    x0, y0, x1, y1 = box
    plain_energy = np.mean(plain_pixels[y0:y1, x0:x1].min(axis=2) > 230)
    removed_energy = np.mean(removed_pixels[y0:y1, x0:x1].min(axis=2) > 230)
    ratio = removed_energy / max(plain_energy, 1e-9)
    assert ratio < (0.4 if method == "delogo" else 0.25)

    outside = np.ones(plain_pixels.shape[:2], dtype=bool)
    outside[max(0, y0 - 12) : min(1920, y1 + 12), max(0, x0 - 12) : min(1080, x1 + 12)] = False
    difference = np.abs(removed_pixels.astype(np.int16) - plain_pixels.astype(np.int16))
    assert difference[outside].mean() < 6


def test_mask_is_cached(
    media: dict[str, dict[str, object]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import adanvil.media.segment as segment_module

    source = _source(media)
    box = tuple(int(value) for value in media["burnt_text.mp4"]["text_bbox"])
    segment = ClipSegment(
        source="burnt_text.mp4",
        **{"in": 0.5, "out": 1.5, "gamma": None},
        remove_text=RemoveText(method="mask", box=box),
    )
    calls = 0
    original = segment_module.build_mask

    def counted(*args: object, **kwargs: object) -> Path:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(segment_module, "build_mask", counted)
    cache = Cache(tmp_path / "cache")
    render_segment(segment, root=source.parent, cache=cache)
    render_segment(segment, root=source.parent, cache=cache)
    assert calls == 1
