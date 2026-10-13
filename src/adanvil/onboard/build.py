"""Turn onboarding answers and site facts into the shared brand artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from adanvil.brand import Brand, default_brand, save_brand
from adanvil.project import resolve_in_project

from .questionnaire import BUNDLED_FAMILIES, Question, build_questionnaire
from .site import SiteFacts, analyze_site


@dataclass
class OnboardResult:
    facts: SiteFacts | None = None
    questionnaire: list[Question] = field(default_factory=list)
    brand_path: Path | None = None
    brief_path: Path | None = None
    open_questions: list[str] = field(default_factory=list)


_ALIASES = {
    "language": "caption_language",
    "address": "viewer_address",
    "voice": "brand_voice",
    "caption_family": "caption_font",
    "end_card_source": "end_card_image",
}
_COLOUR = re.compile(r"^#[0-9a-f]{3}(?:[0-9a-f]{3})?$", re.IGNORECASE)


def _clean(value: object) -> str:
    return str(value).replace("\u2014", "-").replace("\u2013", "-").strip()


def _blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or value == []


def _normalise_colour(value: str) -> str:
    if not _COLOUR.fullmatch(value.strip()):
        raise ValueError("must be a hex colour such as #AABBCC")
    value = value.strip().upper()
    if len(value) == 4:
        value = "#" + "".join(char * 2 for char in value[1:])
    return value


def _as_list(value: object, question_id: str) -> list[str] | list[dict[str, Any]]:
    values: list[object]
    if isinstance(value, str):
        values = [item.strip() for item in re.split(r"[,\n]", value) if item.strip()]
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError("must be a list or a comma-separated string")
    result: list[str] | list[dict[str, Any]] = []
    for item in values:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
        elif question_id == "people" and isinstance(item, dict):
            result.append(dict(item))
        else:
            raise ValueError("every list item must be a non-empty string")
    return result


def _normalise_answer(
    question: Question, value: object, *, project_root: Path | None = None
) -> object:
    if _blank(value):
        return None
    if question.kind == "text":
        if not isinstance(value, str):
            raise ValueError("must be text")
        return _clean(value)
    if question.kind == "list":
        return _as_list(value, question.id)
    if question.kind == "choice":
        if not isinstance(value, str):
            raise ValueError("must be one of the listed choices")
        if question.id == "emoji":
            for choice in ("twemoji", "none"):
                if value.casefold() == choice:
                    return choice
            if project_root is None:
                raise ValueError(
                    "emoji folder must be a project-relative path and onboarding "
                    "needs a project root to check it"
                )
            try:
                path = resolve_in_project(project_root, value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"emoji folder path is invalid: {exc}") from exc
            if not path.is_dir():
                raise ValueError(f"emoji folder does not exist: {value}")
            return path.relative_to(project_root).as_posix()
        choices = question.choices or []
        for choice in choices:
            if value.casefold() == choice.casefold():
                return choice
        raise ValueError(f"must be one of: {', '.join(choices)}")
    if question.kind == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.casefold() in {"true", "yes", "y", "1"}:
            return True
        if isinstance(value, str) and value.casefold() in {"false", "no", "n", "0"}:
            return False
        raise ValueError("must be a boolean")
    raise ValueError(f"unsupported question kind {question.kind}")


def _cta_image(value: str, project_root: Path | None) -> str:
    if project_root is None:
        return value
    try:
        path = resolve_in_project(project_root, value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cta_image path is invalid: {exc}") from exc
    if not path.is_file():
        raise ValueError(f"cta_image file does not exist: {value}")
    return path.relative_to(project_root).as_posix()


def _people(value: list[str] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            result.append({"id": item})
            continue
        person = dict(item)
        person_id = person.get("id")
        if not isinstance(person_id, str) or not person_id.strip():
            raise ValueError("each person needs a non-empty id")
        person["id"] = person_id.strip()
        if "consent" in person and not isinstance(person["consent"], bool):
            consent = person["consent"]
            if isinstance(consent, str) and consent.casefold() in {"yes", "true", "y", "1"}:
                person["consent"] = True
            elif isinstance(consent, str) and consent.casefold() in {"no", "false", "n", "0"}:
                person["consent"] = False
            else:
                raise ValueError("person consent must be boolean")
        result.append(person)
    return result


def _set_path(data: dict[str, Any], path: str, value: object) -> None:
    parts = path.split(".")
    current = data
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _get_path(data: Mapping[str, Any], path: str) -> object:
    current: object = data
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _inferred_name(facts: SiteFacts | None) -> str:
    if facts is not None and facts.title:
        name = facts.title
        for separator in (" | ", " - ", " – ", ": "):
            if separator in name:
                name = name.split(separator, 1)[0]
                break
        if name.strip():
            return _clean(name)
    if facts is not None and facts.final_url:
        hostname = re.sub(r"^www\.", "", facts.final_url.split("//", 1)[-1].split("/", 1)[0])
        if hostname:
            return hostname.split(":", 1)[0]
    return "Unnamed brand"


def _format_list(value: object) -> str:
    if isinstance(value, list):
        formatted: list[str] = []
        for item in value:
            if not isinstance(item, dict):
                formatted.append(_clean(item))
                continue
            label = _clean(item.get("id", ""))
            if item.get("consent") is not None:
                label += f" (consent: {'yes' if item['consent'] else 'no'})"
            formatted.append(label)
        return ", ".join(formatted)
    return _clean(value) if value is not None else "Not provided"


def _render_brief(
    brand: Brand,
    facts: SiteFacts | None,
    brief_answers: Mapping[str, object],
    open_questions: list[Question],
) -> str:
    product = brand.product
    tone = brand.tone
    claims = brand.claims
    visual = brand.visual
    end_card = brand.end_card
    name = _clean(brand.name)
    niche = _clean(brand.niche) if brand.niche else "the stated category"
    customer = _format_list(brief_answers.get("customer"))
    worked = _format_list(brief_answers.get("angles_worked"))
    failed = _format_list(brief_answers.get("angles_failed"))
    people = _format_list(brand.people)
    rules = _format_list(brand.footage_rules)
    versions = _clean(brand.version_text()) if brand.version_text() else "Not provided"
    banned = _format_list(claims.banned_words if claims else [])
    guidance = _clean(claims.guidance) if claims and claims.guidance else "Not provided"
    offer = _clean(product.offer) if product and product.offer else "Not provided"
    guarantee = _clean(product.guarantee) if product and product.guarantee else "Not provided"
    cta = _clean(product.cta_text) if product and product.cta_text else "Not provided"
    primary_colour = visual.primary_color if visual and visual.primary_color else "Not provided"
    accent_colour = visual.accent_color if visual and visual.accent_color else "Not provided"
    caption_font = visual.caption_family if visual and visual.caption_family else "Not provided"
    logo_candidates = (
        ", ".join(facts.logo_candidates) if facts and facts.logo_candidates else "None found"
    )

    paragraph = f"{name} is a {niche} brand"
    if brand.website:
        paragraph += f" at {brand.website}"
    if product and product.name:
        paragraph += f", led by {product.name}"
    if tone and tone.voice:
        paragraph += f" with a {tone.voice} voice"
    paragraph += "."

    lines = [
        "# Brand brief",
        "",
        "## Brand in one paragraph",
        "",
        _clean(paragraph),
        "",
        "## Customer",
        "",
        f"- Target customer: {customer}",
        f"- Angles that worked: {worked}",
        f"- Angles that failed: {failed}",
        "",
        "## What to show",
        "",
        f"- Hero product: {_clean(product.name) if product and product.name else 'Not provided'}",
        f"- People allowed: {people}",
        f"- Footage rules: {rules}",
        f"- Product versions: {versions}",
        "",
        "## What to avoid",
        "",
        f"- Failed angles: {failed}",
        f"- Banned words or claims: {banned}",
        f"- Footage restrictions: {rules}",
        "",
        "## Claims",
        "",
        f"- Banned words or claims: {banned}",
        f"- Guidance: {guidance}",
        "",
        "## Offer and CTA",
        "",
        f"- Offer: {offer}",
        f"- Guarantee: {guarantee}",
        f"- CTA: {cta}",
        "",
        "## Visual identity",
        "",
        f"- Primary colour: {primary_colour}",
        f"- Accent colour: {accent_colour}",
        f"- Caption font: {caption_font}",
        f"- Logo candidates: {logo_candidates}",
        f"- Emoji: {visual.emoji if visual and visual.emoji else 'Not provided'}",
        f"- End card: {end_card.source if end_card and end_card.source else 'Not provided'}",
        f"- CTA image: {brand.cta_image or 'Not provided'}",
        "",
        "## Site facts",
        "",
        f"- Title: {_clean(facts.title) if facts and facts.title else 'Not found'}",
    ]
    if facts and facts.products:
        lines.append("- Products:")
        for item in facts.products:
            price = item.price or "price not found"
            if item.currency:
                price = f"{price} {item.currency}"
            if item.compare_at_price:
                price += f" (was {item.compare_at_price})"
            lines.append(f"  - {_clean(item.title)}: {price}")
    else:
        lines.append("- Products: None found")
    if facts and facts.guarantee_phrases:
        lines.append("- Guarantee phrases found:")
        lines.extend(f"  - {_clean(item)}" for item in facts.guarantee_phrases)
    else:
        lines.append("- Guarantee phrases found: None")
    if facts and facts.review_snippets:
        lines.append("- Review snippets:")
        lines.extend(f"  - {_clean(item)}" for item in facts.review_snippets)
    else:
        lines.append("- Review snippets: None")
    if facts and facts.warnings:
        lines.append("- Site warnings:")
        lines.extend(f"  - {_clean(item)}" for item in facts.warnings)
    lines.extend(["", "## Open questions", ""])
    if open_questions:
        lines.extend(f"- {question.id}: {_clean(question.text)}" for question in open_questions)
    else:
        lines.append("None.")
    return "\n".join(lines).replace("\u2014", "-") + "\n"


def build_brand(
    answers: dict[str, Any],
    facts: SiteFacts | None = None,
    base: Brand | None = None,
    project_root: Path | None = None,
) -> tuple[Brand, str]:
    """Validate answers, apply them to a brand profile and render its brief."""

    if not isinstance(answers, dict):
        raise ValueError("Invalid answers:\n- answers: must be a dictionary")
    questions = build_questionnaire(facts)
    by_id = {question.id: question for question in questions}
    canonical_answers: dict[str, Any] = {}
    invalid: list[str] = []
    for raw_id, raw_value in answers.items():
        question_id = _ALIASES.get(raw_id, raw_id)
        question = by_id.get(question_id)
        if question is None:
            invalid.append(f"- {raw_id}: unknown question id")
            continue
        try:
            value = _normalise_answer(question, raw_value, project_root=project_root)
            if value is not None:
                if question_id in {"primary_color", "accent_color"}:
                    value = _normalise_colour(value)  # type: ignore[arg-type]
                elif question_id == "caption_font":
                    if value not in BUNDLED_FAMILIES:
                        raise ValueError(f"unknown bundled family {value!r}")
                elif question_id == "people":
                    value = _people(value)  # type: ignore[arg-type]
                elif question_id == "cta_image":
                    value = _cta_image(value, project_root)  # type: ignore[arg-type]
                canonical_answers[question_id] = value
        except ValueError as exc:
            invalid.append(f"- {question_id}: {exc}")

    name_answer = canonical_answers.get("brand_name")
    if base is None:
        name = (
            name_answer if isinstance(name_answer, str) and name_answer else _inferred_name(facts)
        )
        brand = default_brand(name)
    else:
        brand = base.model_copy(deep=True)
    data = brand.model_dump(mode="python")
    brief_answers: dict[str, object] = {}
    for question in questions:
        if question.id not in canonical_answers:
            continue
        value = canonical_answers[question.id]
        if question.maps_to.startswith("brief."):
            brief_answers[question.maps_to.removeprefix("brief.")] = value
        else:
            _set_path(data, question.maps_to, value)

    if invalid:
        raise ValueError("Invalid answers:\n" + "\n".join(invalid))
    try:
        brand = Brand.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(
            ".".join(str(part) for part in error["loc"]) + ": " + error["msg"]
            for error in exc.errors()
        )
        raise ValueError(f"Invalid answers:\n- brand: {details}") from exc

    stored = brand.model_dump(mode="python")
    open_questions: list[Question] = []
    for question in questions:
        if not question.required or question.id in canonical_answers:
            continue
        existing = (
            _get_path(stored, question.maps_to)
            if not question.maps_to.startswith("brief.")
            else None
        )
        if _blank(existing):
            open_questions.append(question)
    return brand, _render_brief(brand, facts, brief_answers, open_questions)


def _load_facts(path: Path) -> SiteFacts | None:
    if not path.is_file():
        return None
    try:
        return SiteFacts.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def onboard(root: Path, *, url: str | None, answers: dict[str, Any] | None) -> OnboardResult:
    """Run either the discovery step or the answer-to-artifact step."""

    root = Path(root)
    facts_path = root / "brand" / "site_facts.json"
    if answers is None:
        if not url:
            raise ValueError("A website URL is required for the discovery step")
        facts = analyze_site(url)
        facts_path.parent.mkdir(parents=True, exist_ok=True)
        facts_path.write_text(facts.model_dump_json(indent=2), encoding="utf-8")
        return OnboardResult(facts=facts, questionnaire=build_questionnaire(facts))

    if url:
        facts = analyze_site(url)
        facts_path.parent.mkdir(parents=True, exist_ok=True)
        facts_path.write_text(facts.model_dump_json(indent=2), encoding="utf-8")
    else:
        facts = _load_facts(facts_path)
    brand, brief = build_brand(answers, facts=facts, project_root=root)
    brand_path = root / "brand.yaml"
    brief_path = root / "brand_brief.md"
    save_brand(brand, brand_path)
    brief_path.write_text(brief, encoding="utf-8")
    questions = build_questionnaire(facts)
    answered_ids = {_ALIASES.get(key, key) for key, value in answers.items() if not _blank(value)}
    open_questions = [
        question.id
        for question in questions
        if question.required and question.id not in answered_ids
    ]
    return OnboardResult(
        facts=facts,
        questionnaire=questions,
        brand_path=brand_path,
        brief_path=brief_path,
        open_questions=open_questions,
    )


__all__ = ["OnboardResult", "build_brand", "onboard"]
