from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import pytest

from adanvil.brand import load_brand
from adanvil.onboard import analyze_site, build_brand, build_questionnaire, onboard

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "site"


class FixtureHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        files = {
            "/shopify/": (FIXTURE_ROOT / "shopify" / "index.html", "text/html; charset=utf-8"),
            "/shopify/styles.css": (
                FIXTURE_ROOT / "shopify" / "styles.css",
                "text/css; charset=utf-8",
            ),
            "/shopify-regressions/": (
                FIXTURE_ROOT / "shopify-regressions" / "index.html",
                "text/html; charset=utf-8",
            ),
            "/shopify-regressions/styles.css": (
                FIXTURE_ROOT / "shopify-regressions" / "styles.css",
                "text/css; charset=utf-8",
            ),
            "/products.json": (FIXTURE_ROOT / "shopify" / "products.json", "application/json"),
            "/plain/": (FIXTURE_ROOT / "plain" / "index.html", "text/html; charset=utf-8"),
            "/plain/products/x": (
                FIXTURE_ROOT / "plain" / "products" / "x.html",
                "text/html; charset=utf-8",
            ),
        }
        entry = files.get(path)
        if entry is None:
            self.send_error(404)
            return
        file_path, content_type = entry
        payload = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture()
def site_server(monkeypatch: pytest.MonkeyPatch) -> str:
    # The fixture site is on the loopback address, which the download guard refuses unless the
    # developer opts in (see adanvil.netguard); tests/test_security_c.py covers the refusal.
    monkeypatch.setenv("ADANVIL_ALLOW_PRIVATE_URLS", "1")
    server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _answers(url: str) -> dict[str, object]:
    return {
        "brand_name": "Fernwood Coffee",
        "website": url,
        "niche": "coffee",
        "hero_product": "Morning Blend",
        "target_customer": "Adults 25-45 with a busy morning and a taste for good coffee.",
        "current_offer": "14.90 EUR, reduced from 18.90 EUR",
        "guarantee": "Retours et remboursement sous 30 jours",
        "cta_text": "Discover Morning Blend",
        "markets": ["France"],
        "caption_language": "fr",
        "viewer_address": "vous",
        "brand_voice": "warm and reassuring",
        "banned_words": ["cure", "medical"],
        "claims_guidance": "Describe taste and craft only.",
        "people": [{"id": "Léa", "consent": True}],
        "footage_rules": ["Use only the current product version."],
        "primary_color": "#D94A3A",
        "accent_color": "#F2A65A",
        "caption_font": "poppins",
    }


def _regression_fetch(url: str) -> tuple[int, str, str]:
    path = urlparse(url).path
    files = {
        "/shopify-regressions/": (
            FIXTURE_ROOT / "shopify-regressions" / "index.html",
            "text/html; charset=utf-8",
        ),
        "/shopify-regressions/styles.css": (
            FIXTURE_ROOT / "shopify-regressions" / "styles.css",
            "text/css; charset=utf-8",
        ),
        "/products.json": (
            FIXTURE_ROOT / "shopify-regressions" / "products.json",
            "application/json",
        ),
    }
    file_path, content_type = files[path]
    return 200, content_type, file_path.read_text(encoding="utf-8")


def test_shopify_site_facts_are_extracted(site_server: str) -> None:
    facts = analyze_site(f"{site_server}/shopify/")

    assert facts.title == "Fernwood Coffee | Small-batch roasters"
    assert facts.language == "fr"
    assert facts.is_shopify is True
    assert "#D94A3A" in [colour.hex for colour in facts.colors]
    assert "Poppins" in facts.fonts
    assert f"{site_server}/assets/fernwood-logo.svg" in facts.logo_candidates
    assert "14,90 €" in facts.prices
    assert facts.guarantee_phrases == ["Retours et remboursement sous 30 jours."]
    assert any("Avis clients" in snippet or "★" in snippet for snippet in facts.review_snippets)
    assert len(facts.products) == 2
    assert facts.products[0].price == "14.90"
    assert facts.products[0].compare_at_price == "18.90"
    assert facts.products[0].currency == "EUR"


def test_title_ignores_svg_payment_titles(site_server: str) -> None:
    facts = analyze_site(f"{site_server}/shopify-regressions/", fetch=_regression_fetch)

    assert facts.title == "Brand – Tagline"


def test_css_variables_and_system_fonts_are_resolved(site_server: str) -> None:
    facts = analyze_site(f"{site_server}/shopify-regressions/", fetch=_regression_fetch)

    assert facts.fonts == ["Assistant", "Playfair Display"]


def test_shopify_currency_drives_products_and_offer_prefill(site_server: str) -> None:
    facts = analyze_site(f"{site_server}/shopify-regressions/", fetch=_regression_fetch)
    questions = build_questionnaire(facts)

    assert facts.products[0].currency == "EUR"
    assert (
        next(question for question in questions if question.id == "current_offer").prefill
        == "74.00 EUR (was 110.00)"
    )


def test_cart_and_localization_headings_are_filtered(site_server: str) -> None:
    facts = analyze_site(f"{site_server}/shopify-regressions/", fetch=_regression_fetch)

    assert facts.headings == ["Light running shoes"]


def test_non_shopify_follow_up_page_is_scanned(site_server: str) -> None:
    facts = analyze_site(f"{site_server}/plain/", max_pages=2)

    assert facts.is_shopify is False
    assert "Travel Mug" in facts.headings
    assert "29.00 $" in facts.prices


def test_network_error_returns_partial_facts() -> None:
    def broken_fetch(url: str) -> tuple[int, str, str]:
        raise OSError("offline")

    facts = analyze_site("https://example.invalid", fetch=broken_fetch)

    assert facts.title == ""
    assert facts.warnings
    assert "offline" in facts.warnings[0]


def test_questionnaire_has_prefills_and_stable_shape(site_server: str) -> None:
    facts = analyze_site(f"{site_server}/shopify/")
    questions = build_questionnaire(facts)
    ids = [question.id for question in questions]

    assert 18 <= len(questions) <= 26
    assert len(ids) == len(set(ids))
    assert (
        next(question for question in questions if question.id == "hero_product").prefill
        == "Morning Blend"
    )
    assert (
        next(question for question in questions if question.id == "primary_color").prefill
        == "#D94A3A"
    )
    font_question = next(question for question in questions if question.id == "caption_font")
    assert font_question.prefill == "poppins"
    assert font_question.choices == [
        "plus-jakarta-sans",
        "montserrat",
        "tiktok-sans",
        "roboto",
        "poppins",
    ]
    emoji_question = next(question for question in questions if question.id == "emoji")
    assert emoji_question.choices == ["twemoji", "none", "project-relative folder path"]


def test_build_brand_maps_answers_and_reports_invalid_question_ids(site_server: str) -> None:
    facts = analyze_site(f"{site_server}/shopify/")
    answers = _answers(facts.final_url)
    brand, brief = build_brand(answers, facts=facts)

    assert brand.claims is not None
    assert brand.claims.banned_words == ["cure", "medical"]
    assert brand.visual is not None
    assert brand.visual.primary_color == "#D94A3A"
    assert brand.visual.caption_family == "poppins"
    assert "Morning Blend" in brief
    assert "## Open questions" in brief

    with pytest.raises(ValueError, match="primary_color"):
        build_brand({**answers, "primary_color": "not-a-colour"}, facts=facts)
    with pytest.raises(ValueError, match="caption_font"):
        build_brand({**answers, "caption_font": "comic-sans"}, facts=facts)


def test_emoji_answer_accepts_existing_project_folder_and_names_question(
    site_server: str, tmp_path: Path
) -> None:
    facts = analyze_site(f"{site_server}/shopify/")
    emoji_folder = tmp_path / "emoji_apple"
    emoji_folder.mkdir()
    answers = {**_answers(facts.final_url), "emoji": "emoji_apple"}

    brand, _ = build_brand(answers, facts=facts, project_root=tmp_path)
    assert brand.visual is not None
    assert brand.visual.emoji == "emoji_apple"

    with pytest.raises(ValueError, match="emoji.*does not exist"):
        build_brand({**answers, "emoji": "missing-emojis"}, facts=facts, project_root=tmp_path)


def test_onboard_two_step_flow_writes_facts_and_brand_files(site_server: str) -> None:
    root = FIXTURE_ROOT
    facts_path = root / "brand" / "site_facts.json"
    brand_path = root / "brand.yaml"
    brief_path = root / "brand_brief.md"
    try:
        first = onboard(root, url=f"{site_server}/shopify/", answers=None)
        assert first.facts is not None
        assert facts_path.is_file()
        assert json.loads(facts_path.read_text(encoding="utf-8"))["is_shopify"] is True

        second = onboard(root, url=None, answers=_answers(f"{site_server}/shopify/"))
        assert second.brand_path == brand_path
        assert second.brief_path == brief_path
        assert brand_path.is_file()
        assert brief_path.is_file()
        assert load_brand(second.brand_path).product is not None
        assert "Morning Blend" in brief_path.read_text(encoding="utf-8")
    finally:
        for path in (facts_path, brand_path, brief_path):
            if path.is_file():
                path.unlink()
        if facts_path.parent.is_dir():
            facts_path.parent.rmdir()
