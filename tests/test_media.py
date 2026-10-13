from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageStat

from adanvil.ffmpeg import measure_yavg, probe
from adanvil.media import (
    Cache,
    ClipSegment,
    Freeze,
    ImageSegment,
    Zoom,
    auto_gamma,
    freeze_zoom,
    measure_motion,
    render_segment,
    segment_duration,
    tonemap,
    zoom,
)
from adanvil.media.filters import gamma, source_crop


def _root(media: dict[str, dict[str, object]]) -> Path:
    return Path(str(media["sdr_portrait.mp4"]["path"])).parent


def _render(
    media: dict[str, dict[str, object]], segment: ClipSegment | ImageSegment, tmp_path: Path
) -> Path:
    return render_segment(segment, root=_root(media), cache=Cache(tmp_path / "cache"))


def _frames(path: Path) -> int:
    info = probe(path)
    assert info.nb_frames is not None
    return info.nb_frames


def _mean(path: Path, box: tuple[int, int, int, int] | None = None) -> tuple[float, ...]:
    with Image.open(path).convert("RGB") as image:
        if box:
            image = image.crop(box)
        return tuple(ImageStat.Stat(image).mean)


def test_filter_snapshots_and_gamma() -> None:
    hlg = tonemap("arib-std-b67")
    pq = tonemap("smpte2084")
    assert "tin=arib-std-b67" in hlg
    assert "tin=smpte2084" in pq
    assert "tonemap=hable:desat=0" in hlg
    assert tonemap("bt709") == ""
    assert source_crop(1, 2, 3, 4) == "crop=3:4:1:2"
    assert zoom(1.0, 1.1, 2.0).count("iw") == 0
    assert zoom(1.0, 1.1, 2.0).count("ih") == 0
    assert freeze_zoom(0.75, 1.0, 1.35, 540, 960).count("iw") == 0
    assert freeze_zoom(0.75, 1.0, 1.35, 540, 960).count("ih") == 0
    assert auto_gamma(200) == 1.0
    assert 1.3 < auto_gamma(60) <= 1.6
    assert gamma(1.0) == ""


def test_cache_is_atomic_and_content_addressed(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "cache")
    calls = 0

    def builder(path: Path) -> None:
        nonlocal calls
        calls += 1
        path.write_bytes(b"ok")

    key = {"b": 2, "a": [1, 2]}
    first = cache.get_or_build("clip", key, builder)
    second = cache.get_or_build("clip", key, builder)
    assert first == second
    assert first.read_bytes() == b"ok"
    assert calls == 1

    with pytest.raises(RuntimeError):
        def broken(_: Path) -> None:
            raise RuntimeError("stop")

        cache.get_or_build("clip", {"broken": True}, broken)
    assert not list((tmp_path / "cache" / "clip").glob("*.tmp.mp4"))

    file_key = cache.file_key(first)
    assert file_key["path"] == str(first.resolve())
    assert file_key["size"] == 2


def test_clip_hlg_is_tonemapped(media: dict[str, dict[str, object]], tmp_path: Path) -> None:
    seg = ClipSegment(source="hlg_portrait.mp4", **{"in": 0, "out": 1, "gamma": None})
    output = _render(media, seg, tmp_path)
    info = probe(output)
    assert (info.width, info.height) == (1080, 1920)
    assert (info.pix_fmt, info.color_transfer, info.color_primaries, info.color_space) == (
        "yuv420p",
        "bt709",
        "bt709",
        "bt709",
    )
    source_frame = tmp_path / "source.png"
    output_frame = tmp_path / "output.png"
    from adanvil.ffmpeg import extract_frame

    extract_frame(Path(str(media["hlg_portrait.mp4"]["path"])), 0.5, source_frame)
    extract_frame(output, 0.5, output_frame)
    source_mean = _mean(source_frame)
    output_mean = _mean(output_frame)
    assert output_mean[0] < output_mean[1] * 1.4
    assert source_mean[0] >= source_mean[1] * 1.4 or output_mean != pytest.approx(source_mean)


def test_clip_focus_speed_zoom_and_freeze(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    left = ClipSegment(
        source="sdr_landscape.mp4", **{"in": 0, "out": 1, "gamma": None}, focus=(0.0, 0.5)
    )
    right = left.model_copy(update={"focus": (1.0, 0.5)})
    left_out = _render(media, left, tmp_path / "left")
    right_out = _render(media, right, tmp_path / "right")
    assert left_out.read_bytes() != right_out.read_bytes()

    fast = ClipSegment(source="sdr_portrait.mp4", **{"in": 0, "out": 2, "gamma": None}, speed=2)
    fast_out = _render(media, fast, tmp_path / "fast")
    assert _frames(fast_out) == pytest.approx(30, abs=1)

    moving = ClipSegment(
        source="sdr_portrait.mp4",
        **{"in": 0, "out": 1, "gamma": None},
        zoom=Zoom(**{"from": 1.0, "to": 1.1}),
    )
    moving_out = _render(media, moving, tmp_path / "moving")
    motion = measure_motion(moving_out)
    assert _frames(moving_out) == pytest.approx(30, abs=1)
    assert motion and max(motion) <= 4 * sorted(motion)[len(motion) // 2] + 1

    frozen = ClipSegment(
        source="sdr_portrait.mp4",
        **{"in": 0, "out": 0.75, "gamma": None},
        freeze=Freeze(hold=1.0),
    )
    frozen_out = _render(media, frozen, tmp_path / "frozen")
    assert _frames(frozen_out) == pytest.approx(52.5, abs=1)
    frozen_motion = measure_motion(frozen_out)
    assert frozen_motion[-10:] and max(frozen_motion[-10:]) > 0


def test_clip_gamma_and_orientation(media: dict[str, dict[str, object]], tmp_path: Path) -> None:
    source = Path(str(media["dark_portrait.mp4"]["path"]))
    bright = ClipSegment(source="dark_portrait.mp4", **{"in": 0, "out": 1})
    neutral = bright.model_copy(update={"gamma": None})
    bright_out = _render(media, bright, tmp_path / "bright")
    neutral_out = _render(media, neutral, tmp_path / "neutral")
    assert measure_yavg(bright_out) > measure_yavg(source) + 15
    assert measure_yavg(neutral_out) == pytest.approx(measure_yavg(source), abs=8)

    rotated = ClipSegment(source="rotated.mp4", **{"in": 0, "out": 1, "gamma": None})
    rotated_out = _render(media, rotated, tmp_path / "rotated")
    assert (probe(rotated_out).width, probe(rotated_out).height, probe(rotated_out).rotation) == (
        1080,
        1920,
        0,
    )


def test_images_and_segment_models(media: dict[str, dict[str, object]], tmp_path: Path) -> None:
    static = ImageSegment(source="cta.png", duration=2)
    static_out = _render(media, static, tmp_path / "static")
    assert _frames(static_out) == pytest.approx(60, abs=1)
    assert max(measure_motion(static_out)) < 0.5

    zoomed = static.model_copy(update={"duration": 3, "motion": "zoom_in"})
    zoomed_out = _render(media, zoomed, tmp_path / "zoomed")
    assert _frames(zoomed_out) == pytest.approx(90, abs=1)
    motion = measure_motion(zoomed_out)
    assert max(motion) <= 4 * sorted(motion)[len(motion) // 2] + 1

    product = ImageSegment(source="product.png", duration=1, fit="contain")
    product_out = _render(media, product, tmp_path / "product")
    frame = tmp_path / "product.png"
    from adanvil.ffmpeg import extract_frame

    extract_frame(product_out, 0, frame)
    corners = [_mean(frame, box) for box in [(0, 0, 40, 40), (1040, 0, 1080, 40)]]
    assert all(any(channel > 2 for channel in mean) for mean in corners)


def test_segment_validation_and_paths(media: dict[str, dict[str, object]], tmp_path: Path) -> None:
    segment = ClipSegment(
        source="sdr_portrait.mp4", **{"in": 0, "out": 1}, freeze=Freeze(hold=1)
    )
    assert segment_duration(segment) == 2
    assert segment_duration(ImageSegment(source="cta.png", duration=2)) == 2
    with pytest.raises(ValueError):
        ClipSegment(source="sdr_portrait.mp4", **{"in": 1, "out": 1})
    with pytest.raises(ValueError):
        ClipSegment(source="sdr_portrait.mp4", **{"in": 0, "out": 1}, speed=5)
    with pytest.raises(ValueError):
        render_segment(
            ImageSegment(source="../outside.mp4", duration=1),
            root=_root(media),
            cache=Cache(tmp_path / "cache"),
        )
    with pytest.raises(ValueError):
        render_segment(
            ImageSegment(source=str(Path(str(media["cta.png"]["path"])).resolve()), duration=1),
            root=_root(media),
            cache=Cache(tmp_path / "cache2"),
        )


def test_cache_key_is_json_stable(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    left = cache.path_for("x", {"a": 1, "b": [2, 3]})
    right = cache.path_for("x", json.loads('{"b":[2,3],"a":1}'))
    assert left == right
