from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pytest

from adanvil.brand import Brand, Claims, default_brand, find_banned, load_brand, save_brand
from adanvil.ffmpeg import probe
from adanvil.storyboard import get_recipe, list_recipes, validate_storyboard


def test_brand_round_trip_and_defaults(tmp_path: Path) -> None:
    brand = default_brand("Acme Coffee")
    brand.claims = Claims(banned_words=["miracle", "cure"], guidance="Describe the product.")
    path = tmp_path / "brand.yaml"
    save_brand(brand, path)
    loaded = load_brand(path)
    assert loaded == brand
    assert loaded.visual is not None
    assert loaded.visual.caption_family == "plus-jakarta-sans"


def test_find_banned_is_case_accent_and_boundary_aware() -> None:
    banned = ["eco-friendly", "cure"]
    assert find_banned("ÉCO-friendly and a CURE", banned) == banned
    assert find_banned("secure curing", ["cure"]) == []
    assert find_banned("This is a fat-free recipe", ["fat-free"]) == ["fat-free"]
    assert find_banned("A money back guarantee", ["money back"]) == ["money back"]


def test_recipes_load_and_examples_validate() -> None:
    recipes = list_recipes()
    assert len(recipes) == 9
    assert "ugc_talking" in {recipe.id for recipe in recipes}
    assert [recipe.id for recipe in recipes] == sorted({recipe.id for recipe in recipes})
    allowed = {
        "total_duration",
        "shot_duration",
        "caption_min_on_screen",
        "caption_safe_zone",
        "end_card_required",
        "max_caption_lines",
        "caption_count",
        "banned_words",
    }
    for recipe in recipes:
        assert recipe.example["recipe"] == recipe.id
        assert all(rule.check in allowed for rule in recipe.rules)


@pytest.fixture
def storyboard_project(tmp_path: Path, media: dict[str, dict[str, object]]) -> tuple[Path, dict]:
    root = tmp_path
    (root / "footage").mkdir()
    (root / "brand").mkdir()
    (root / "music").mkdir()
    shutil.copy2(media["sdr_portrait.mp4"]["path"], root / "footage" / "source.mp4")
    shutil.copy2(media["cta.png"]["path"], root / "brand" / "end_card.png")
    shutil.copy2(media["music.m4a"]["path"], root / "music" / "track.m4a")
    storyboard = {
        "version": 1,
        "name": "valid_rules",
        "recipe": "before_after",
        "format": {"width": 1080, "height": 1920, "fps": 30},
        "segments": [
            {
                "type": "clip",
                "source": "footage/source.mp4",
                "in": 0,
                "out": 2,
                "label": "before",
            },
            {
                "type": "image",
                "source": "brand/end_card.png",
                "duration": 3,
                "label": "end_card",
            },
        ],
        "overlays": [
            {
                "type": "caption",
                "style": "pill",
                "lines": ["A calm start"],
                "start": 0,
                "end": 2,
                "y": 330,
            },
            {
                "type": "caption",
                "style": "pill",
                "lines": ["A clear finish"],
                "start": 2,
                "end": 4,
                "y": 330,
            },
        ],
        "audio": {"music": "music/track.m4a"},
    }
    return root, storyboard


def _report(root: Path, storyboard: dict, **kwargs):
    return validate_storyboard(
        copy.deepcopy(storyboard),
        root=root,
        recipe=get_recipe("before_after"),
        **kwargs,
    )


def test_valid_storyboard_has_no_errors(storyboard_project) -> None:
    root, storyboard = storyboard_project
    report = _report(root, storyboard)
    assert report.ok
    assert report.errors == []
    assert report.summary["segment_count"] == 2
    assert report.summary["checklist"]


def test_missing_file_is_reported(storyboard_project) -> None:
    root, storyboard = storyboard_project
    storyboard["segments"][0]["source"] = "footage/missing.mp4"
    report = _report(root, storyboard)
    assert not report.ok
    issue = next(issue for issue in report.errors if issue.code == "missing_file")
    assert "missing.mp4" in issue.message


def test_clip_out_beyond_source_is_reported(storyboard_project) -> None:
    root, storyboard = storyboard_project
    storyboard["segments"][0]["out"] = 3.2
    report = _report(root, storyboard)
    issue = next(issue for issue in report.errors if issue.code == "source_duration")
    assert issue.where == "segments[0].out"


def test_overlay_after_end_is_reported(storyboard_project) -> None:
    root, storyboard = storyboard_project
    storyboard["overlays"].append(
        {"type": "caption", "style": "pill", "lines": ["Late"], "start": 6, "end": 7}
    )
    report = _report(root, storyboard)
    issue = next(issue for issue in report.errors if issue.code == "overlay_after_end")
    assert "overlay 2" in issue.message.lower()


def test_overlay_end_before_start_is_reported(storyboard_project) -> None:
    root, storyboard = storyboard_project
    storyboard["overlays"][0]["end"] = 0
    report = _report(root, storyboard)
    issue = next(issue for issue in report.errors if issue.code == "overlay_timing")
    assert "overlay" in issue.message.lower()


def test_banned_word_is_reported(storyboard_project) -> None:
    root, storyboard = storyboard_project
    storyboard["overlays"][0]["lines"] = ["A cure"]
    brand = Brand(name="Acme", claims=Claims(banned_words=["cure"]))
    report = _report(root, storyboard, brand=brand)
    issue = next(issue for issue in report.errors if issue.code == "banned_word")
    assert "cure" in issue.message and "overlay 0" in issue.message


def test_overflow_is_a_warning(storyboard_project) -> None:
    root, storyboard = storyboard_project
    storyboard["overlays"][0]["lines"] = ["very long " * 100]
    storyboard["overlays"][0]["size"] = 1000
    report = _report(root, storyboard)
    issue = next(issue for issue in report.warnings if issue.code == "text_overflow")
    assert "overlay 0" in issue.message


def test_safe_zone_violation_is_reported(storyboard_project) -> None:
    root, storyboard = storyboard_project
    storyboard["overlays"][0]["y"] = 1800
    report = _report(root, storyboard)
    issue = next(issue for issue in report.warnings if issue.code == "safe_zone")
    assert "overlay 0" in issue.message


def test_product_shot_floor_is_reported(storyboard_project) -> None:
    root, storyboard = storyboard_project
    storyboard["segments"].insert(
        1,
        {
            "type": "clip",
            "source": "footage/source.mp4",
            "in": 0,
            "out": 0.8,
            "label": "product:1",
        },
    )
    report = _report(root, storyboard)
    issue = next(issue for issue in report.errors if issue.code == "product_shot_floor")
    assert "product:1" in issue.message


def test_end_card_required_is_reported(storyboard_project) -> None:
    root, storyboard = storyboard_project
    storyboard["segments"][-1]["label"] = "finish"
    report = _report(root, storyboard)
    issue = next(issue for issue in report.errors if issue.code == "end_card")
    assert "finish" in issue.message


def test_schema_typo_has_a_json_path(storyboard_project) -> None:
    root, storyboard = storyboard_project
    storyboard["segmets"] = storyboard.pop("segments")
    report = _report(root, storyboard)
    issue = next(issue for issue in report.errors if issue.code == "schema")
    assert issue.where == "segmets"
    assert "segmets" in issue.message


def test_negative_clip_in_is_reported(storyboard_project) -> None:
    root, storyboard = storyboard_project
    storyboard["segments"][0]["in"] = -1
    report = _report(root, storyboard)
    issue = next(issue for issue in report.errors if issue.code == "segment_in")
    assert issue.where == "segments[0].in"


def _digital_zoom(report) -> list:
    return [issue for issue in report.warnings if issue.code == "digital_zoom"]


def test_digital_zoom_warns_on_a_tight_crop(storyboard_project) -> None:
    root, storyboard = storyboard_project
    info = probe(root / "footage" / "source.mp4")
    assert _digital_zoom(_report(root, storyboard)) == []
    segment = storyboard["segments"][0]
    segment["crop"] = {"x": 0, "y": 0, "w": info.width // 2, "h": info.height // 2}
    issues = _digital_zoom(_report(root, storyboard))
    assert len(issues) == 1
    assert "segment 0 enlarges the source 2.0x" in issues[0].message
    assert "pick a shot filmed close instead" in issues[0].message
    # A light crop that only removes the edges stays quiet.
    segment["crop"] = {"x": 0, "y": 0, "w": int(info.width * 0.9), "h": int(info.height * 0.9)}
    assert _digital_zoom(_report(root, storyboard)) == []


def test_digital_zoom_counts_zoom_not_freeze(storyboard_project) -> None:
    root, storyboard = storyboard_project
    segment = storyboard["segments"][0]
    segment["zoom"] = {"from": 1.0, "to": 1.04}
    assert _digital_zoom(_report(root, storyboard)) == []
    segment["zoom"] = {"from": 1.0, "to": 1.5}
    assert "1.5x" in _digital_zoom(_report(root, storyboard))[0].message
    segment.pop("zoom")
    # A freeze zoom is a deliberate push-in on a held frame, not a fake close-up.
    segment["freeze"] = {"hold": 1.0, "zoom": 1.35}
    assert _digital_zoom(_report(root, storyboard)) == []


def test_shot_duration_bound_tolerates_float_subtraction() -> None:
    from adanvil.storyboard.recipes import get_recipe
    from adanvil.storyboard.schema import Storyboard
    from adanvil.storyboard.validate import _check_recipe_rules

    recipe = get_recipe("voiceover_ad")
    storyboard = Storyboard.model_validate(
        {
            "version": 1,
            "name": "float_bound",
            # 3.4 - 1.8 is 1.5999999999999999 in floating point.
            "segments": [
                {"type": "clip", "source": "footage/a.mp4", "in": 1.8, "out": 3.4, "label": "hook"}
            ],
        }
    )
    rule = next(rule for rule in recipe.rules if rule.id == "shot_floor")
    floor = rule.model_copy(update={"params": {"min": 1.6}})
    errors: list = []
    warnings: list = []
    _check_recipe_rules(
        storyboard,
        recipe.model_copy(update={"rules": [floor]}),
        total=storyboard.total_duration(),
        caption_entries=[],
        text_bboxes={},
        errors=errors,
        warnings=warnings,
    )
    assert errors == [] and warnings == []
