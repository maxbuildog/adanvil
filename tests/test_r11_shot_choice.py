"""Shot choice (R11): emotion, the brand's own CTA image, small usability fixes."""

from __future__ import annotations

import copy
import inspect
import json
import shutil
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client
from PIL import Image

from adanvil.brand import default_brand, load_brand, save_brand
from adanvil.endcard import render_end_card
from adanvil.library import Annotation, annotate, entry_line, index_library, load_catalog, search
from adanvil.onboard import build_brand, build_questionnaire
from adanvil.project import init_project
from adanvil.reference import breakdown_reference, decode
from adanvil.reference.breakdown import EMOTION_VALUES
from adanvil.server import build_server
from adanvil.storyboard import Storyboard, validate_storyboard
from adanvil.text.emoji import directory_emoji

from .test_reference import _decoded_with_shots, _entry
from .test_reference_checks import _breakdown, _segments, _storyboard, _write

FIRE = "\U0001f525"
SPARKLES = "✨"


def _json(result: Any) -> dict[str, Any]:
    return json.loads(result.content[0].text)


def _run(root: Path, storyboard: dict[str, Any], brand=None):
    return validate_storyboard(copy.deepcopy(storyboard), root=root, brand=brand, check_files=False)


def _mismatches(report) -> list[str]:
    return [issue.message for issue in report.warnings if issue.code == "emotion_mismatch"]


def _no_reference(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in _storyboard(**kwargs).items() if k != "reference"}


# --- item 1: emotion in the breakdown ---------------------------------------------------


def test_breakdown_emotion_action_and_summary(tmp_path: Path) -> None:
    _decoded_with_shots(tmp_path, 4)
    entries = [
        _entry([1], "myth", emotion="worried", action="wipes the counter and sighs"),
        _entry([2], "punch", emotion="shocked"),
        _entry([3], "punch", emotion="skeptical"),
        _entry([4], "proof", emotion="none"),
    ]
    summary = breakdown_reference(tmp_path, "reference/x", entries)
    assert summary["emotion_by_role"] == {"myth": ["worried"], "punch": ["shocked", "skeptical"]}
    saved = json.loads((tmp_path / "reference" / "x" / "breakdown.json").read_text("utf-8"))
    assert saved["shots"][0]["action"] == "wipes the counter and sighs"
    with pytest.raises(ValueError, match="emotion must be one of: none, neutral, worried"):
        breakdown_reference(tmp_path, "reference/x", [_entry([1], "myth", emotion="smug")])
    # An old breakdown without emotion stays valid.
    old = breakdown_reference(tmp_path, "reference/x", [_entry([1, 2, 3, 4], "hook")])
    assert old["emotion_by_role"] == {}


def test_decode_template_and_vocabulary_carry_emotion() -> None:
    assert EMOTION_VALUES == (
        "none",
        "neutral",
        "worried",
        "frustrated",
        "sad",
        "tired",
        "closed",
        "skeptical",
        "shocked",
        "curious",
        "relaxed",
        "happy",
        "satisfied",
    )
    source = inspect.getsource(decode)
    assert '"emotion": None' in source and '"action": None' in source


# --- item 2: emotion in the library -----------------------------------------------------


def test_library_emotion_annotate_search_and_line(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    init_project(tmp_path, "emo")
    for name in ("a", "b"):
        shutil.copy2(str(media["sdr_portrait.mp4"]["path"]), tmp_path / "footage" / f"{name}.mp4")
    index_library(tmp_path, "footage")
    entries = {Path(e.path).stem: e for e in load_catalog(tmp_path).entries.values()}
    annotate(tmp_path, entries["a"].id, emotion=["worried", "skeptical"])
    annotate(tmp_path, entries["b"].id, emotion=["happy"])

    assert [e.id for e in search(tmp_path, emotion=["skeptical"])] == [entries["a"].id]
    assert [e.id for e in search(tmp_path, emotion=["worried", "skeptical"])] == [entries["a"].id]
    assert search(tmp_path, emotion=["worried", "happy"]) == []
    assert len(search(tmp_path)) == 2
    assert "emotion=worried,skeptical" in entry_line(search(tmp_path, emotion=["worried"])[0])
    with pytest.raises(ValueError, match="emotion must be one of"):
        annotate(tmp_path, entries["a"].id, emotion=["smug"])
    assert Annotation.model_validate({"emotion": "happy"}).emotion == ["happy"]
    assert Annotation().emotion == []

    async def run() -> None:
        async with Client(build_server()) as client:
            done = await client.call_tool(
                "library_annotate",
                {"project_dir": str(tmp_path), "entry_id": entries["b"].id, "emotion": ["relaxed"]},
            )
            assert _json(done)["annotation"]["emotion"] == ["relaxed"]
            found = _json(
                await client.call_tool(
                    "library_search", {"project_dir": str(tmp_path), "emotion": ["relaxed"]}
                )
            )
            assert found["count"] == 1 and "emotion=relaxed" in found["entries"][0]

    anyio.run(run)


# --- item 3: emotion on segments + checks ------------------------------------------------


def test_segments_accept_emotion_and_role_and_reject_unknown() -> None:
    model = Storyboard.model_validate(
        _storyboard(segments=_segments(1, emotion="worried", role="problem"))
    )
    assert model.segments[0].emotion == "worried" and model.segments[0].role == "problem"
    image = {"type": "image", "source": "a.png", "duration": 2, "emotion": "happy", "role": "cta"}
    Storyboard.model_validate(_storyboard(segments=[image]))
    with pytest.raises(ValueError, match="emotion must be one of"):
        Storyboard.model_validate(_storyboard(segments=_segments(1, emotion="smug")))
    with pytest.raises(ValueError, match="role must be one of"):
        Storyboard.model_validate(_storyboard(segments=_segments(1, role="villain")))


def test_emotion_mismatch_smiling_under_a_complaint(tmp_path: Path) -> None:
    segments = _segments(3)
    segments[0].update(role="problem", emotion="happy")
    segments[1].update(role="myth", emotion="relaxed")
    segments[2].update(role="agitation", emotion="worried")
    messages = _mismatches(_run(tmp_path, _no_reference(segments=segments)))
    assert len(messages) == 2
    assert "segments[0]" in messages[0] and "role problem" in messages[0]
    assert "the text complains while the face smiles" in messages[0]
    # Another role, or no emotion: nothing to say.
    segments[0].update(role="reveal")
    segments[1].pop("emotion")
    assert _mismatches(_run(tmp_path, _no_reference(segments=segments))) == []


def _punch_breakdown(emotions: list[str]) -> dict[str, Any]:
    payload = _breakdown()
    payload["shots"] = [
        {"shots": [index + 1], "role": "punch", "emotion": emotion}
        for index, emotion in enumerate(emotions)
    ]
    return payload


def test_emotion_mismatch_punch_without_reaction(tmp_path: Path) -> None:
    segments = _segments(3)
    segments[0].update(role="punch", emotion="happy")
    segments[1].update(role="punch")
    segments[2].update(role="punch", emotion="shocked")

    _write(tmp_path, _punch_breakdown(["shocked", "skeptical", "neutral"]))
    report = _run(tmp_path, _storyboard(segments=segments))
    messages = _mismatches(report)
    assert len(messages) == 2
    assert "segments[0] is a punch with happy" in messages[0]
    assert "segments[1] is a punch with no emotion" in messages[1]
    assert "the reference answers with a reaction face" in messages[0]
    assert report.ok

    # Question-device shots count as well.
    payload = _breakdown()
    payload["shots"] = [
        {"shots": [1], "role": "myth", "text_device": "question", "emotion": "skeptical"}
    ]
    _write(tmp_path, payload)
    assert len(_mismatches(_run(tmp_path, _storyboard(segments=segments)))) == 2
    # A reference whose punches are not reaction faces says nothing; neither does an old one.
    _write(tmp_path, _punch_breakdown(["neutral", "happy", "shocked"]))
    assert _mismatches(_run(tmp_path, _storyboard(segments=segments))) == []
    _write(tmp_path, _breakdown())
    assert _mismatches(_run(tmp_path, _storyboard(segments=segments))) == []


def test_ref_unannotated_emotion(tmp_path: Path) -> None:
    _write(tmp_path, _breakdown())

    def flagged(segments: list[dict[str, Any]], **kwargs: Any) -> bool:
        report = _run(tmp_path, _storyboard(segments=segments, **kwargs))
        return "ref_unannotated_emotion" in [issue.code for issue in report.warnings]

    people = _segments(10, subject="person_a")
    assert flagged(people)
    for index in range(6):
        people[index]["emotion"] = "neutral"
    assert flagged(people)  # 6 of 10
    people[6]["emotion"] = "none"
    assert not flagged(people)  # 7 of 10 reaches 70 %
    assert not flagged(_segments(3, subject="product"))  # no person segment
    raw = _no_reference(segments=_segments(3, subject="person_a"))
    assert "ref_unannotated_emotion" not in [i.code for i in _run(tmp_path, raw).warnings]


# --- item 5: the brand's own CTA image --------------------------------------------------


def test_cta_image_question_is_stored_and_checked(tmp_path: Path) -> None:
    questions = build_questionnaire(None)
    assert next(q for q in questions if q.id == "cta_image").required is False
    (tmp_path / "cta").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "cta" / "cta.png")
    answers = {"brand_name": "Acme", "cta_image": "cta/cta.png"}
    brand, brief = build_brand(answers, project_root=tmp_path)
    assert brand.cta_image == "cta/cta.png" and "cta/cta.png" in brief
    with pytest.raises(ValueError, match="cta_image file does not exist"):
        build_brand({**answers, "cta_image": "cta/none.png"}, project_root=tmp_path)
    assert default_brand("x").cta_image is None
    save_brand(brand, tmp_path / "brand.yaml")
    assert load_brand(tmp_path / "brand.yaml").cta_image == "cta/cta.png"


def test_cta_image_unused_warning(tmp_path: Path) -> None:
    brand = default_brand("Acme")
    brand.cta_image = "cta/cta.png"

    def codes(segments: list[dict[str, Any]], brand=brand, **extra: Any) -> list[str]:
        report = _run(tmp_path, _no_reference(segments=segments, **extra), brand=brand)
        return [issue.code for issue in report.warnings]

    image = {"type": "image", "source": "cta/cta.png", "duration": 3, "label": "end_card"}
    composed = {**image, "source": "brand/endcards/x.png"}
    assert "cta_image_unused" in codes(_segments(2))
    assert "cta_image_unused" not in codes([*_segments(2), image])
    assert "cta_image_unused" in codes([*_segments(2), composed])
    gaps = [{"type": "other", "reason": "the CTA image shows an old product"}]
    assert "cta_image_unused" not in codes([*_segments(2), composed], gaps=gaps)
    # The gap alone is not enough: the last segment must be labelled end_card.
    assert "cta_image_unused" in codes(_segments(2), gaps=gaps)
    # No cta_image in the brand: nothing to say.
    assert "cta_image_unused" not in codes(_segments(2), brand=default_brand("Acme"))


# --- item 6: emoji set and end card crop ------------------------------------------------


def test_directory_emoji_lists_characters(tmp_path: Path) -> None:
    for stem in ("1f525", "2728", "1f468-200d-1f4bb", "2764-fe0f", "notes", "2728-fe0f"):
        Image.new("RGBA", (4, 4)).save(tmp_path / f"{stem}.png")
    (tmp_path / "readme.txt").write_text("x", encoding="utf-8")
    listed = directory_emoji(tmp_path)
    assert FIRE in listed and SPARKLES in listed and "❤" in listed
    assert "\U0001f468‍\U0001f4bb" in listed
    assert listed.count(SPARKLES) == 1 and len(listed) == 4


def test_brand_get_returns_emoji_available(tmp_path: Path) -> None:
    init_project(tmp_path, "emoji")
    brand = default_brand("Emoji")
    save_brand(brand, tmp_path / "brand.yaml")
    folder = tmp_path / "emoji_set"
    folder.mkdir()
    for stem in ("1f525", "2728"):
        Image.new("RGBA", (4, 4)).save(folder / f"{stem}.png")

    async def run() -> None:
        async with Client(build_server()) as client:
            first = _json(await client.call_tool("brand_get", {"project_dir": str(tmp_path)}))
            assert first["emoji_available"] == "all"
            assert first["brand"]["cta_image"] is None
            assert brand.visual is not None
            brand.visual.emoji = "emoji_set"
            save_brand(brand, tmp_path / "brand.yaml")
            second = _json(await client.call_tool("brand_get", {"project_dir": str(tmp_path)}))
            assert second["emoji_available"] == [FIRE, SPARKLES]
            brand.visual.emoji = "none"
            save_brand(brand, tmp_path / "brand.yaml")
            third = _json(await client.call_tool("brand_get", {"project_dir": str(tmp_path)}))
            assert third["emoji_available"] == []

    anyio.run(run)


def test_end_card_returns_source_size(tmp_path: Path, media: dict[str, dict[str, object]]) -> None:
    init_project(tmp_path, "crop")
    shutil.copy2(str(media["product.png"]["path"]), tmp_path / "footage" / "product.png")
    with Image.open(tmp_path / "footage" / "product.png") as source:
        size = list(source.size)
    spec = {
        "product_image": "footage/product.png",
        "title": "Offre",
        "emoji": "none",
        "crop": {"x": 0, "y": 0, "w": 100, "h": 100},
    }
    assert render_end_card(tmp_path, spec).layout["source_size"] == size


def test_product_too_early_before_the_solution(tmp_path: Path) -> None:
    def too_early(segments: list[dict[str, Any]]) -> list[str]:
        report = _run(tmp_path, _no_reference(segments=segments))
        return [issue.message for issue in report.warnings if issue.code == "product_too_early"]

    segments = _segments(4)
    segments[0].update(role="hook", emotion="sad", subject="person_a")
    segments[1].update(role="myth", shot_type="product_in_use", subject="person_b")
    segments[2].update(role="reveal", shot_type="product_hero", subject="product")
    segments[3].update(role="offer", subject="product")
    messages = too_early(segments)
    assert len(messages) == 1 and "segments[1]" in messages[0]
    assert "segments[2], role reveal" in messages[0]
    # Without any solution role the story turn is unknown: nothing to say.
    for segment in segments:
        segment.update(role="other")
    assert too_early(segments) == []
