from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from adanvil.ffmpeg import probe
from adanvil.reference import ReferenceDecode, decode_reference
from adanvil.reference import decode as decode_module


def _copy_media(root: Path, media: dict[str, dict[str, object]], name: str) -> None:
    destination = root / "reference_ads" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(str(media[name]["path"])), destination)


def _iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    intersection = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1])
    )
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def test_decode_reference_writes_manifest_and_assets(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    _copy_media(tmp_path, media, "ref_ad.mp4")

    result = decode_reference(tmp_path, "reference_ads/ref_ad.mp4")

    assert [(shot.start, shot.end) for shot in result.shots] == [
        pytest.approx((0.0, 2.0), abs=0.1),
        pytest.approx((2.0, 4.0), abs=0.1),
        pytest.approx((4.0, 6.0), abs=0.1),
    ]
    assert any("burnt-in text" in note for note in result.notes)
    for shot in result.shots:
        keyframe = tmp_path / shot.keyframe
        assert keyframe.exists()
        with Image.open(keyframe) as image:
            assert image.size == (1080, 1920)
        manifest_box = tuple(media["ref_ad.mp4"]["text_bbox"])
        assert any(_iou(box, manifest_box) >= 0.5 for box in shot.text_boxes)

    sheet = tmp_path / result.sheet
    assert sheet.exists()
    with Image.open(sheet) as image:
        assert image.width == 6 * 180
        assert image.height == 3 * 320
    assert result.soundtrack is not None
    assert probe(tmp_path / result.soundtrack).duration == pytest.approx(6.0, abs=0.05)
    assert result.music_profile is not None
    assert result.music_profile["feel"] in {"calm", "medium", "energetic"}
    assert result.music_profile["path"] == result.soundtrack
    assert "voice" in str(result.music_profile["note"])
    assert result.breakdown_template[0]["text_speaker"] is None
    assert result.breakdown_template[0]["text_device"] is None

    manifest = json.loads((tmp_path / "reference" / "ref-ad" / "reference.json").read_text())
    assert ReferenceDecode.model_validate(manifest) == result


def test_decode_reference_transcription_status_and_word_flattening(
    media: dict[str, dict[str, object]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_media(tmp_path, media, "ref_ad.mp4")
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    # No voice environment either, whatever is installed on this machine.
    monkeypatch.setenv("ADANVIL_CACHE", str(tmp_path / "no-voice-environment"))
    monkeypatch.delenv("ADANVIL_VOICE_PYTHON", raising=False)

    missing = decode_reference(
        tmp_path,
        "reference_ads/ref_ad.mp4",
        out="reference/missing-transcriber",
        transcribe=True,
    )
    assert missing.transcript is None
    assert "adanvil[transcribe]" in missing.transcript_status

    class FakeModel:
        def __init__(self, model: str, *, device: str, compute_type: str) -> None:
            assert (model, device, compute_type) == ("small", "cpu", "int8")

        def transcribe(self, source: str, **kwargs: object):
            assert source.endswith("reference_ads\\ref_ad.mp4") or source.endswith(
                "reference_ads/ref_ad.mp4"
            )
            assert kwargs == {
                "word_timestamps": True,
                "language": "en",
                "vad_filter": True,
            }
            words = [
                SimpleNamespace(start=0.1, end=0.4, word="Hello", probability=0.91),
                SimpleNamespace(start=0.5, end=0.8, word="world", probability=0.87),
            ]
            return [SimpleNamespace(start=0.1, end=0.8, words=words)], None

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeModel))
    decoded = decode_reference(
        tmp_path,
        "reference_ads/ref_ad.mp4",
        out="reference/fake-transcriber",
        transcribe=True,
        language="en",
    )
    assert decoded.transcript_status == "done"
    assert decoded.transcript is not None
    assert [word.text for word in decoded.transcript] == ["Hello", "world"]
    assert [word.probability for word in decoded.transcript] == [0.91, 0.87]


def test_decode_reference_uses_voice_environment_when_extra_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeEngine:
        def __init__(self, *, project_dir: Path) -> None:
            assert project_dir == tmp_path.resolve()

        def align(self, paths: list[Path], language: str | None, model: str):
            assert paths[0].parent == tmp_path / "reference" / "voice-fallback"
            assert language is None
            assert model == "medium"
            return [SimpleNamespace(words=[(0.2, 0.5, "Hello", 0.91), (0.6, 0.9, "world", 0.87)])]

    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setattr(decode_module, "WorkerEngine", FakeEngine)
    monkeypatch.setattr(decode_module.ffmpeg, "run", lambda _args: None)
    output_dir = tmp_path / "reference" / "voice-fallback"
    output_dir.mkdir(parents=True)

    words, status = decode_module._transcribe(
        tmp_path / "reference.mp4",
        requested=True,
        language=None,
        model="medium",
        output_dir=output_dir,
        project_root=tmp_path,
    )

    assert status == "done (voice environment)"
    assert words is not None
    assert [word.text for word in words] == ["Hello", "world"]
    assert [word.probability for word in words] == [0.91, 0.87]


def test_decode_reference_without_audio_has_no_soundtrack(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    _copy_media(tmp_path, media, "sdr_portrait.mp4")

    result = decode_reference(tmp_path, "reference_ads/sdr_portrait.mp4")

    assert result.soundtrack is None
    assert "no audio" in result.notes


def _fake_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def __init__(self, model: str, *, device: str, compute_type: str) -> None:
            pass

        def transcribe(self, source: str, **kwargs: object):
            words = [
                SimpleNamespace(start=0.2, end=0.6, word="Are", probability=0.9),
                SimpleNamespace(start=0.7, end=1.1, word="you", probability=0.9),
                SimpleNamespace(start=2.5, end=3.0, word="No", probability=0.9),
                SimpleNamespace(start=3.2, end=3.6, word="way", probability=0.9),
            ]
            return [SimpleNamespace(start=0.2, end=3.6, words=words)], None

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeModel))


def test_decode_v2_pacing_spoken_sheets_and_template(
    media: dict[str, dict[str, object]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_media(tmp_path, media, "ref_ad.mp4")
    _fake_whisper(monkeypatch)

    result = decode_reference(tmp_path, "reference_ads/ref_ad.mp4")

    assert result.transcript_status == "done"
    assert [shot.spoken for shot in result.shots] == ["Are you", "No way", ""]
    assert all(shot.text_regions for shot in result.shots if shot.text_boxes)
    assert {region["band"] for shot in result.shots for region in shot.text_regions} <= {
        "top",
        "middle",
        "bottom",
    }

    pacing = result.pacing
    assert pacing["shot_count"] == 3
    assert pacing["mean"] == pytest.approx(2.0, abs=0.1)
    assert pacing["median"] == pytest.approx(2.0, abs=0.1)
    assert pacing["min"] == pytest.approx(2.0, abs=0.1)
    assert pacing["max"] == pytest.approx(2.0, abs=0.1)
    assert pacing["cuts_per_10s"] == pytest.approx(2 * 10 / 6.0, abs=0.2)
    assert len(pacing["by_quarter"]) == 4

    assert result.shot_sheets == ["reference/ref-ad/shots-01.jpg"]
    with Image.open(tmp_path / result.shot_sheets[0]) as sheet:
        assert sheet.width >= 3 * 240
        assert sheet.height > 0

    assert [entry["shots"] for entry in result.breakdown_template] == [[1], [2], [3]]
    assert result.breakdown_template[0]["spoken"] == "Are you"
    assert result.breakdown_template[0]["type"] is None
    assert result.breakdown_template[0]["start"] == result.shots[0].start


def test_decode_v2_auto_without_transcriber_is_not_an_error(
    media: dict[str, dict[str, object]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_media(tmp_path, media, "ref_ad.mp4")
    monkeypatch.setitem(sys.modules, "faster_whisper", None)

    class NoVoice:
        def __init__(self, *, project_dir: Path) -> None:
            pass

        def _python(self, _args: list[str]):
            raise decode_module.VoiceUnavailableError("none")

    monkeypatch.setattr(decode_module, "WorkerEngine", NoVoice)
    result = decode_reference(tmp_path, "reference_ads/ref_ad.mp4", transcribe="auto")
    assert result.transcript is None
    assert "adanvil[transcribe]" in result.transcript_status

    off = decode_reference(
        tmp_path, "reference_ads/ref_ad.mp4", out="reference/off", transcribe=False
    )
    assert off.transcript_status == "not requested"


def test_shot_sheet_pagination(media: dict[str, dict[str, object]], tmp_path: Path) -> None:
    from adanvil.media.sheet import reference_shot_sheets

    source = Path(str(media["ref_ad.mp4"]["path"]))
    shots = [
        {"index": index, "start": index * 0.5, "end": index * 0.5 + 0.5} for index in range(1, 9)
    ]
    sheets = reference_shot_sheets(source, shots, tmp_path / "sheets", shots_per_sheet=6)
    assert [sheet.name for sheet in sheets] == ["shots-01.jpg", "shots-02.jpg"]
    with Image.open(sheets[0]) as first, Image.open(sheets[1]) as second:
        assert first.height > second.height


def _decoded_with_shots(tmp_path: Path, count: int) -> None:
    shots = [
        decode_module.RefShot(
            index=index,
            start=float(index - 1),
            end=float(index),
            duration=1.0,
            keyframe="reference/x/keyframes/k.png",
            yavg=100.0,
        )
        for index in range(1, count + 1)
    ]
    decoded = ReferenceDecode(
        source="reference_ads/x.mp4",
        duration=float(count),
        width=1080,
        height=1920,
        fps=30.0,
        has_audio=False,
        shots=shots,
        sheet="reference/x/sheet.jpg",
        soundtrack=None,
        transcript=None,
        transcript_status="not requested",
        pacing={"shot_count": count},
    )
    folder = tmp_path / "reference" / "x"
    folder.mkdir(parents=True)
    (folder / "reference.json").write_text(decoded.model_dump_json(), encoding="utf-8")


def _entry(shots, role, subject="person_a", text="", **extra):
    return {
        "shots": shots,
        "type": "face_closeup",
        "scale": "closeup",
        "subject": subject,
        "on_screen_text": text,
        "role": role,
        **extra,
    }


def test_breakdown_summary_and_validation(tmp_path: Path) -> None:
    from adanvil.reference import breakdown_reference

    _decoded_with_shots(tmp_path, 6)
    entries = [
        _entry([1], "myth", text="Are you sure?"),
        _entry([2], "punch", subject="person_b", text="No you wouldn't!"),
        _entry([3], "myth", text="Really?"),
        _entry([4], "punch", subject="person_c", replacement="missing: pour macro"),
        _entry([5, 6], "proof", subject="diagram", type="diagram", scale="insert"),
    ]
    summary = breakdown_reference(tmp_path, "reference/x", entries)

    assert summary["role_sequence"] == "myth > punch > myth > punch > proof"
    assert summary["repeated_patterns"] == {"myth > punch": 2}
    assert summary["people"] == ["person_a", "person_b", "person_c"]
    assert summary["text_shots"] == 3
    assert summary["type_demand"]["diagram"]["shots"] == 2
    assert summary["type_demand"]["face_closeup"]["share"] == pytest.approx(4 / 6)
    assert summary["scale_mix"]["insert"] == pytest.approx(2 / 6)
    assert len(summary["missing"]) == 1
    assert summary["pacing"] == {"shot_count": 6}
    saved = json.loads((tmp_path / "reference" / "x" / "breakdown.json").read_text("utf-8"))
    assert len(saved["shots"]) == 5
    breakdown_reference(tmp_path, "reference/x/reference.json", entries)

    with pytest.raises(ValueError, match=r"missing shots: \[6\]"):
        breakdown_reference(tmp_path, "reference/x", [*entries[:4], _entry([5], "proof")])
    with pytest.raises(ValueError, match=r"duplicated shots: \[2\]"):
        breakdown_reference(tmp_path, "reference/x", [*entries, _entry([2], "punch")])
    with pytest.raises(ValueError, match="talking_head"):
        breakdown_reference(tmp_path, "reference/x", [_entry([1], "myth", type="selfie")])
    with pytest.raises(ValueError, match="hook"):
        breakdown_reference(tmp_path, "reference/x", [_entry([1], "villain")])
    with pytest.raises(ValueError, match="static"):
        breakdown_reference(tmp_path, "reference/x", [_entry([1], "hook", motion="zoomy")])


def test_breakdown_accepts_old_shot_type_names_as_aliases(tmp_path: Path) -> None:
    from adanvil.reference import breakdown_reference
    from adanvil.reference.breakdown import TYPE_VALUES

    assert "texture_macro" in TYPE_VALUES and "diagram" in TYPE_VALUES
    assert "skin_texture" not in TYPE_VALUES and "anatomy_diagram" not in TYPE_VALUES

    _decoded_with_shots(tmp_path, 3)
    entries = [
        _entry([1], "hook", type="skin_texture"),
        _entry([2], "mechanism", type="anatomy_diagram", subject="diagram"),
        _entry([3], "proof", type="texture_macro"),
    ]
    summary = breakdown_reference(tmp_path, "reference/x", entries)

    assert summary["type_demand"]["texture_macro"]["shots"] == 2
    assert summary["type_demand"]["diagram"]["shots"] == 1
    assert "skin_texture" not in summary["type_demand"]
    assert "anatomy_diagram" not in summary["type_demand"]
    saved = json.loads((tmp_path / "reference" / "x" / "breakdown.json").read_text("utf-8"))
    assert [entry["type"] for entry in saved["shots"]] == [
        "texture_macro",
        "diagram",
        "texture_macro",
    ]


def test_breakdown_counts_text_changes_not_text_shots(tmp_path: Path) -> None:
    from adanvil.reference import breakdown_reference

    _decoded_with_shots(tmp_path, 4)
    entries = [
        _entry([1], "myth", text="Mornings drag on and on"),
        _entry([2], "myth", text="Mornings  drag on and on"),  # same text held over a cut
        _entry([3], "punch", text="Are you sure?"),
        _entry([4], "punch", text="Are you sure?"),
    ]
    assert breakdown_reference(tmp_path, "reference/x", entries)["text_shots"] == 2


def test_breakdown_text_speaker_device_and_dialogue(tmp_path: Path) -> None:
    from adanvil.reference import breakdown_reference

    _decoded_with_shots(tmp_path, 4)
    speaking = {"text_speaker": "character", "text_device": "quote"}
    entries = [
        _entry([1], "myth", text="X : « a »", **speaking),
        _entry([2], "punch", text="Y : « b »", **speaking),
        _entry([3], "offer", text="-20 %", text_speaker="brand", text_device="price_or_offer"),
        _entry([4], "cta", text="", text_speaker="none", text_device="none"),
    ]
    summary = breakdown_reference(tmp_path, "reference/x", entries)
    assert summary["text_devices"] == {"quote": 2, "price_or_offer": 1}
    assert summary["dialogue"] is True
    saved = json.loads((tmp_path / "reference" / "x" / "breakdown.json").read_text("utf-8"))
    assert saved["shots"][0]["text_speaker"] == "character"

    one = [_entry([1, 2, 3, 4], "myth", text="X : « a »", **speaking)]
    assert breakdown_reference(tmp_path, "reference/x", one)["dialogue"] is False
    plain = [_entry([1, 2, 3, 4], "myth", text="hello")]
    summary = breakdown_reference(tmp_path, "reference/x", plain)
    assert summary["text_devices"] == {}
    assert summary["dialogue"] is False

    with pytest.raises(ValueError, match="text_speaker must be one of"):
        breakdown_reference(tmp_path, "reference/x", [_entry([1], "hook", text_speaker="me")])
    with pytest.raises(ValueError, match="text_device must be one of"):
        breakdown_reference(tmp_path, "reference/x", [_entry([1], "hook", text_device="meme")])
