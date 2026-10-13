from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from adanvil.server import _validation_json
from adanvil.storyboard import Storyboard, validate_storyboard


def _breakdown(**summary: Any) -> dict[str, Any]:
    base = {
        "type_demand": {},
        "scale_mix": {},
        "role_sequence": "",
        "repeated_patterns": {},
        "people": [],
        "text_shots": 0,
        "pacing": {"mean": 1.0},
        "missing": [],
    }
    return {"reference": "reference/x", "shots": [], "summary": {**base, **summary}}


def _write(root: Path, payload: dict[str, Any]) -> None:
    folder = root / "reference" / "x"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "breakdown.json").write_text(json.dumps(payload), encoding="utf-8")


def _segments(count: int, duration: float = 1.0, **extra: Any) -> list[dict[str, Any]]:
    return [
        {
            "type": "clip",
            "source": f"footage/s{index}.mp4",
            "in": 0,
            "out": duration,
            "label": f"usage:{index}",
            **extra,
        }
        for index in range(count)
    ]


def _storyboard(**kwargs: Any) -> dict[str, Any]:
    return {
        "version": 1,
        "name": "remake",
        "reference": "reference/x",
        "segments": kwargs.pop("segments", _segments(4)),
        **kwargs,
    }


def _run(root: Path, storyboard: dict[str, Any]):
    return validate_storyboard(copy.deepcopy(storyboard), root=root, check_files=False)


def _codes(report) -> list[str]:
    return [issue.code for issue in report.warnings if issue.code.startswith("ref_")]


def _annotated(count: int, **fields: Any) -> list[dict[str, Any]]:
    return _segments(count, shot_type="product_hero", **fields)


def test_old_storyboard_without_reference_is_untouched(tmp_path: Path) -> None:
    report = _run(tmp_path, {k: v for k, v in _storyboard().items() if k != "reference"})
    assert report.ok
    assert _codes(report) == []
    assert report.reference_comparison is None
    assert "reference_comparison" not in _validation_json(report)


def test_schema_round_trip_and_vocabularies() -> None:
    raw = _storyboard(
        gaps=[{"type": "talking_head", "reason": "not in the library"}],
        segments=_segments(1, shot_type="face_closeup", scale="closeup", subject="person_a")
        + [
            {
                "type": "image",
                "source": "brand/card.png",
                "duration": 2,
                "label": "end_card",
                "shot_type": "text_card",
                "scale": "insert",
            }
        ],
    )
    model = Storyboard.model_validate(raw)
    again = Storyboard.model_validate(model.model_dump(mode="json", by_alias=True))
    assert again == model
    assert again.segments[0].subject == "person_a"
    assert again.gaps[0].type == "talking_head"
    for bad in (
        {"segments": _segments(1, shot_type="selfie")},
        {"segments": _segments(1, scale="huge")},
        {"gaps": [{"type": "selfie", "reason": "x"}]},
        {"gaps": [{"type": "lifestyle", "reason": ""}]},
    ):
        report = _run(Path("."), _storyboard(**bad))
        assert not report.ok
        assert report.errors[0].code == "schema"


def test_missing_or_invalid_reference_is_one_warning(tmp_path: Path) -> None:
    report = _run(tmp_path, _storyboard())
    assert _codes(report) == ["ref_missing"]
    assert report.ok and report.reference_comparison is None

    folder = tmp_path / "reference" / "x"
    folder.mkdir(parents=True)
    (folder / "breakdown.json").write_text("{not json", encoding="utf-8")
    assert _codes(_run(tmp_path, _storyboard())) == ["ref_missing"]
    (folder / "breakdown.json").write_text(json.dumps({"summary": 3}), encoding="utf-8")
    assert _codes(_run(tmp_path, _storyboard())) == ["ref_missing"]


def test_reference_accepts_folder_or_file(tmp_path: Path) -> None:
    _write(tmp_path, _breakdown())
    for written in ("reference/x", "reference/x/breakdown.json"):
        report = _run(tmp_path, _storyboard(reference=written, segments=_annotated(4)))
        assert _codes(report) == []
        assert report.reference_comparison["reference"] == written


def test_ref_pacing(tmp_path: Path) -> None:
    _write(tmp_path, _breakdown(pacing={"mean": 1.0}))
    slow = _run(tmp_path, _storyboard(segments=_annotated(3, duration=2.8)))
    issue = next(i for i in slow.warnings if i.code == "ref_pacing")
    assert issue.message == (
        "ref_pacing: your mean shot is 2.8 s, the reference 1.0 s: cut faster or split shots"
    )
    assert slow.reference_comparison["mean_shot"] == {"storyboard": 2.8, "reference": 1.0}
    assert "ref_pacing" not in _codes(
        _run(tmp_path, _storyboard(segments=_annotated(3, duration=1.3)))
    )
    # The end card and holds longer than 6 s are left out of the storyboard mean.
    held = _annotated(3, duration=1.0) + [
        {
            "type": "image",
            "source": "brand/c.png",
            "duration": 3,
            "label": "end_card",
            "shot_type": "text_card",
        },
        {"type": "clip", "source": "footage/long.mp4", "in": 0, "out": 8, "shot_type": "lifestyle"},
    ]
    assert "ref_pacing" not in _codes(_run(tmp_path, _storyboard(segments=held)))


def test_ref_pacing_uses_decode_without_long_holds(tmp_path: Path) -> None:
    _write(tmp_path, _breakdown(pacing={"mean": 2.0}))
    decode = {"shots": [{"duration": 1.0}, {"duration": 1.0}, {"duration": 9.0}]}
    (tmp_path / "reference" / "x" / "reference.json").write_text(json.dumps(decode), "utf-8")
    report = _run(tmp_path, _storyboard(segments=_annotated(3, duration=2.0)))
    assert report.reference_comparison["mean_shot"]["reference"] == 1.0
    assert "ref_pacing" in _codes(report)


TYPES = {
    "face_closeup": {"shots": 3, "seconds": 3.0, "share": 0.3},
    "diagram": {"shots": 1, "seconds": 2.0, "share": 0.2},
    "lifestyle": {"shots": 1, "seconds": 0.5, "share": 0.05},
}


def test_ref_types_warns_gaps_silence_and_minor_types_are_ignored(tmp_path: Path) -> None:
    _write(tmp_path, _breakdown(type_demand=TYPES))
    report = _run(tmp_path, _storyboard(segments=_annotated(4)))
    message = next(i.message for i in report.warnings if i.code == "ref_types")
    assert "face_closeup (3 shots, 3 s)" in message
    assert "diagram (1 shots, 2 s)" in message
    assert "lifestyle" not in message  # 1 shot and 5 % of the duration

    covered = _annotated(2) + _segments(2, shot_type="face_closeup")
    partly = _run(tmp_path, _storyboard(segments=covered))
    assert "diagram" in next(i.message for i in partly.warnings if i.code == "ref_types")

    gaps = [{"type": "diagram", "reason": "none in the library after library_overview"}]
    silenced = _run(tmp_path, _storyboard(segments=covered, gaps=gaps))
    assert "ref_types" not in _codes(silenced)
    assert silenced.reference_comparison["type_coverage"]["gaps"] == ["diagram"]


def test_old_shot_type_names_are_aliases_of_the_new_ones(tmp_path: Path) -> None:
    # A breakdown saved before the rename keeps validating: its old names count as the new.
    legacy = {
        "face_closeup": {"shots": 3, "seconds": 3.0, "share": 0.3},
        "anatomy_diagram": {"shots": 1, "seconds": 2.0, "share": 0.2},
        "skin_texture": {"shots": 1, "seconds": 1.0, "share": 0.1},
    }
    _write(tmp_path, _breakdown(type_demand=legacy))
    uncovered = _run(tmp_path, _storyboard(segments=_annotated(4)))
    message = next(i.message for i in uncovered.warnings if i.code == "ref_types")
    assert "diagram (1 shots, 2 s)" in message
    assert "texture_macro (1 shots, 1 s)" not in message  # one shot, 10 % of the duration
    assert "anatomy_diagram" not in message

    # Old names are accepted on segments and in gaps, and stored under the new ones.
    old_segments = _segments(2, shot_type="face_closeup") + _segments(
        2, shot_type="anatomy_diagram"
    )
    model = Storyboard.model_validate(_storyboard(segments=old_segments))
    assert [s.shot_type for s in model.segments] == [
        "face_closeup",
        "face_closeup",
        "diagram",
        "diagram",
    ]
    covered = _run(tmp_path, _storyboard(segments=old_segments))
    assert "ref_types" not in _codes(covered)

    gaps = [{"type": "anatomy_diagram", "reason": "none in the library after library_overview"}]
    partly = _annotated(2) + _segments(2, shot_type="face_closeup")
    silenced = _run(tmp_path, _storyboard(segments=partly, gaps=gaps))
    assert "ref_types" not in _codes(silenced)
    assert silenced.reference_comparison["type_coverage"]["gaps"] == ["diagram"]

    # Two old entries that fold into one new name add up.
    folded = {
        "diagram": {"shots": 1, "seconds": 1.0, "share": 0.1},
        "anatomy_diagram": {"shots": 1, "seconds": 1.0, "share": 0.1},
    }
    _write(tmp_path, _breakdown(type_demand=folded))
    merged = _run(tmp_path, _storyboard(segments=_annotated(4)))
    message = next(i.message for i in merged.warnings if i.code == "ref_types")
    assert "diagram (2 shots, 2 s)" in message


def test_ref_closeups(tmp_path: Path) -> None:
    _write(tmp_path, _breakdown(scale_mix={"extreme_closeup": 0.2, "closeup": 0.2, "wide": 0.6}))
    wide = _run(tmp_path, _storyboard(segments=_annotated(4, scale="wide")))
    assert "ref_closeups" in _codes(wide)
    assert "40%" in next(i.message for i in wide.warnings if i.code == "ref_closeups")
    assert "never crop to fake a close-up" in next(
        i.message for i in wide.warnings if i.code == "ref_closeups"
    )
    unscaled = _run(tmp_path, _storyboard(segments=_annotated(4)))
    assert "ref_closeups" in _codes(unscaled)
    close = _annotated(2, scale="closeup") + _annotated(2, scale="wide")
    assert "ref_closeups" not in _codes(_run(tmp_path, _storyboard(segments=close)))
    # Below 30 % in the reference the check does not apply.
    _write(tmp_path, _breakdown(scale_mix={"closeup": 0.29}))
    assert "ref_closeups" not in _codes(_run(tmp_path, _storyboard(segments=_annotated(4))))


def test_ref_people(tmp_path: Path) -> None:
    _write(tmp_path, _breakdown(people=["person_a", "person_b", "person_c"]))
    alone = _run(tmp_path, _storyboard(segments=_annotated(4, subject="person_a")))
    assert "ref_people" in _codes(alone)
    two = _annotated(2, subject="person_a") + _annotated(2, subject="person_b")
    assert "ref_people" not in _codes(_run(tmp_path, _storyboard(segments=two)))
    gaps = [{"type": "talking_head", "reason": "only one creator cleared for ads"}]
    gapped = _run(tmp_path, _storyboard(segments=_annotated(4), gaps=gaps))
    assert "ref_people" not in _codes(gapped)
    _write(tmp_path, _breakdown(people=["person_a", "person_b"]))
    assert "ref_people" not in _codes(_run(tmp_path, _storyboard(segments=_annotated(4))))


def _caption(index: int) -> dict[str, Any]:
    return {
        "type": "caption",
        "style": "pill",
        "lines": [f"Line {index}"],
        "start": index * 0.5,
        "end": index * 0.5 + 0.5,
    }


def test_ref_text_rhythm(tmp_path: Path) -> None:
    _write(
        tmp_path,
        _breakdown(repeated_patterns={"myth > punch": 3}, text_shots=8),
    )
    base = {"segments": _annotated(4)}
    few = _run(tmp_path, _storyboard(overlays=[_caption(0), _caption(1)], **base))
    issue = next(i for i in few.warnings if i.code == "ref_text_rhythm")
    assert issue.message == (
        "ref_text_rhythm: the reference changes its on-screen text 8 times; you have 2 captions"
    )
    enough = _run(tmp_path, _storyboard(overlays=[_caption(i) for i in range(4)], **base))
    assert "ref_text_rhythm" not in _codes(enough)
    assert enough.reference_comparison["captions"] == {
        "storyboard": 4,
        "reference_text_changes": 8,
    }
    # Without a repeated pattern the check does not apply.
    _write(tmp_path, _breakdown(repeated_patterns={}, text_shots=8))
    assert "ref_text_rhythm" not in _codes(_run(tmp_path, _storyboard(**base)))
    # A bigram other than myth > punch counts too.
    _write(tmp_path, _breakdown(repeated_patterns={"hook > proof": 2}, text_shots=8))
    assert "ref_text_rhythm" in _codes(_run(tmp_path, _storyboard(**base)))


def test_voice_captions_do_not_count_as_written_captions(tmp_path: Path) -> None:
    from adanvil.storyboard.validate import _check_reference

    _write(tmp_path, _breakdown(repeated_patterns={"myth > punch": 2}, text_shots=6))
    model = Storyboard.model_validate(_storyboard(segments=_annotated(4)))
    warnings: list = []
    _check_reference(model, tmp_path, written_captions=0, warnings=warnings)
    assert [w.code for w in warnings] == ["ref_text_rhythm"]


def _text_caption(text: str) -> dict[str, Any]:
    return {"type": "caption", "style": "outline", "lines": [text], "start": 0, "end": 3}


def test_ref_text_device(tmp_path: Path) -> None:
    dialogue = {"text_devices": {"quote": 4}, "dialogue": True}
    _write(tmp_path, _breakdown(**dialogue))
    flat = [_text_caption("Le lundi sans café"), _text_caption("Le déclic qui change tout")]
    report = _run(tmp_path, _storyboard(overlays=flat))
    message = next(i.message for i in report.warnings if i.code == "ref_text_device")
    assert "only 0 of your captions" in message
    assert "ref_text_device" in _codes(_run(tmp_path, _storyboard()))
    quotes = [
        _text_caption("😩 : « Mes baskets sont usées »"),
        _text_caption("Sam : « Plus jamais »"),
    ]
    assert "ref_text_device" in _codes(_run(tmp_path, _storyboard(overlays=quotes[:1])))
    assert "ref_text_device" not in _codes(_run(tmp_path, _storyboard(overlays=quotes)))
    straight = [_text_caption('😍 : "Waouh"'), _text_caption("Moi : “Enfin”")]
    assert "ref_text_device" not in _codes(_run(tmp_path, _storyboard(overlays=straight)))
    # No dialogue in the reference: nothing to say.
    _write(tmp_path, _breakdown(text_devices={"quote": 1}, dialogue=False))
    assert "ref_text_device" not in _codes(_run(tmp_path, _storyboard(overlays=flat)))
    _write(tmp_path, _breakdown())
    assert "ref_text_device" not in _codes(_run(tmp_path, _storyboard(overlays=flat)))


def test_ref_unannotated(tmp_path: Path) -> None:
    _write(tmp_path, _breakdown())
    mixed = _annotated(2) + _segments(2)
    report = _run(tmp_path, _storyboard(segments=mixed))
    assert "ref_unannotated" in _codes(report)
    assert report.reference_comparison["annotated_share"] == 0.5
    mostly = _annotated(3) + _segments(1)
    assert "ref_unannotated" not in _codes(_run(tmp_path, _storyboard(segments=mostly)))


def test_reference_checks_never_block(tmp_path: Path) -> None:
    _write(tmp_path, _breakdown(type_demand=TYPES, people=["a", "b", "c"], pacing={"mean": 0.5}))
    report = _run(tmp_path, _storyboard(segments=_segments(3, duration=3.0)))
    assert report.ok
    assert set(_codes(report)) >= {"ref_pacing", "ref_types", "ref_people", "ref_unannotated"}
    payload = _validation_json(report)
    assert payload["reference_comparison"]["people"] == {"storyboard": 0, "reference": 3}


@pytest.mark.parametrize("written", ["/abs/path", "../outside"])
def test_unsafe_reference_path_is_rejected(tmp_path: Path, written: str) -> None:
    report = _run(tmp_path, _storyboard(reference=written))
    assert not report.ok or _codes(report) == ["ref_missing"]
