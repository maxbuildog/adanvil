from __future__ import annotations

from pathlib import Path

from adanvil.ffmpeg import run
from adanvil.qc import QCReport, qc_report
from adanvil.render import RenderReport, compose

from .test_compose import _project, _storyboard


def _encode_video_filter(source: Path, output: Path, video_filter: str) -> None:
    run(
        [
            "-i",
            str(source),
            "-vf",
            video_filter,
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "19",
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
            "copy",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )


def _failed_ids(report: QCReport) -> list[str]:
    return [check.id for check in report.checks if check.status == "fail"]


def test_qc_good_render_is_clean_and_writes_control_images(media, tmp_path: Path) -> None:
    root = _project(media, tmp_path)
    render = compose(_storyboard(), root=root)

    report = qc_report(
        render.path,
        render_report=render,
        expected_duration=render.duration,
        out_dir=tmp_path / "good_qc",
    )

    assert report.ok
    assert all(check.status == "pass" for check in report.checks)
    assert [path.name for path in report.images] == [
        "sheet.jpg",
        "overlay_0.jpg",
        "overlay_1.jpg",
        "overlay_2.jpg",
        "endcard.jpg",  # the render ends on a still: its frame is saved too (V2.8b)
        "end.jpg",
    ]
    assert report.summary["fail"] == 0
    assert {check.id for check in report.checks} >= {
        "decode",
        "resolution",
        "fps",
        "pixel_format",
        "color_tags",
        "duration",
        "audio_present",
        "audio_duration",
        "audio_tail",
        "audio_gaps",
        "black_frames",
        "frozen_frames",
        "overlay_visible",
        "loudness",
    }


def test_qc_preview_scales_storyboard_overlays_to_video_size(media, tmp_path: Path) -> None:
    root = _project(media, tmp_path)
    render = compose(_storyboard(), root=root, quality="preview")

    report = qc_report(
        render.path,
        render_report=render,
        out_dir=tmp_path / "preview_qc",
    )

    assert next(check for check in report.checks if check.id == "overlay_visible").status == "pass"
    assert (tmp_path / "preview_qc" / "end.jpg").is_file()


def test_qc_detects_hdr_tags_and_audio_tail(media, tmp_path: Path) -> None:
    root = _project(media, tmp_path)
    render = compose(_storyboard(), root=root)

    hdr = tmp_path / "hdr.mp4"
    run(
        [
            "-i",
            str(render.path),
            "-map",
            "0",
            "-c",
            "copy",
            "-bsf:v",
            "h264_metadata=transfer_characteristics=18",
            "-color_trc",
            "arib-std-b67",
            str(hdr),
        ]
    )
    hdr_report = qc_report(hdr, expected_duration=render.duration, out_dir=tmp_path / "hdr_qc")
    assert _failed_ids(hdr_report) == ["color_tags"]

    audio_cut = tmp_path / "audio-cut.mp4"
    run(
        [
            "-i",
            str(render.path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-af",
            f"atrim=duration={render.duration - 1.5},asetpts=PTS-STARTPTS",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-t",
            str(render.duration),
            str(audio_cut),
        ]
    )
    audio_report = qc_report(
        audio_cut, expected_duration=render.duration, out_dir=tmp_path / "audio_qc"
    )
    assert _failed_ids(audio_report) == ["audio_duration"]
    assert next(check for check in audio_report.checks if check.id == "audio_tail").status == "warn"


def test_qc_detects_early_caption_black_frames_and_wrong_size(media, tmp_path: Path) -> None:
    root = _project(media, tmp_path)
    render = compose(_storyboard(), root=root)
    overlay = Path(str(render.overlays[0]["png"]))

    disappearing = tmp_path / "disappearing.mp4"
    run(
        [
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=1080x1920:rate=30:duration={render.duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=48000:duration={render.duration}",
            "-i",
            str(overlay),
            "-filter_complex",
            "[0:v][2:v]overlay=0:0:repeatlast=0[v]",
            "-map",
            "[v]",
            "-map",
            "1:a:0",
            "-t",
            str(render.duration),
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
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
            "-ar",
            "48000",
            "-ac",
            "2",
            str(disappearing),
        ]
    )
    fake_report = RenderReport(
        path=disappearing,
        sheet=tmp_path / "unused-sheet.jpg",
        duration=render.duration,
        timeline=render.timeline,
        overlays=[{"type": "caption", "start": 0.0, "end": render.duration, "png": overlay}],
        warnings=[],
        quality="final",
    )
    overlay_report = qc_report(
        disappearing,
        render_report=fake_report,
        expected_duration=render.duration,
        out_dir=tmp_path / "overlay_qc",
    )
    assert _failed_ids(overlay_report) == ["overlay_visible"]

    black = tmp_path / "black.mp4"
    _encode_video_filter(
        render.path,
        black,
        "drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='between(t,2,3)'",
    )
    black_report = qc_report(
        black, expected_duration=render.duration, out_dir=tmp_path / "black_qc"
    )
    assert _failed_ids(black_report) == []
    assert (
        next(check for check in black_report.checks if check.id == "black_frames").status
        == "warn"
    )

    wrong_size = tmp_path / "wrong-size.mp4"
    _encode_video_filter(render.path, wrong_size, "scale=720:1280")
    wrong_size_report = qc_report(
        wrong_size, expected_duration=render.duration, out_dir=tmp_path / "size_qc"
    )
    assert _failed_ids(wrong_size_report) == ["resolution"]
