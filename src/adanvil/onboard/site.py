"""Extract the useful parts of a brand site without third-party HTML tools."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from html import unescape
from html.parser import HTMLParser
from urllib.parse import unquote_plus, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

from adanvil import __version__


class ColorFact(BaseModel):
    """A non-neutral colour and the number of times it appeared in the site."""

    hex: str
    count: int


class ProductFact(BaseModel):
    """A small, provider-neutral product snapshot."""

    title: str
    handle: str
    url: str
    price: str | None = None
    compare_at_price: str | None = None
    currency: str | None = None
    images: list[str] = Field(default_factory=list)
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class SiteFacts(BaseModel):
    """Facts found during the first pass over a brand website."""

    url: str
    final_url: str
    title: str = ""
    description: str = ""
    language: str | None = None
    og_image: str | None = None
    logo_candidates: list[str] = Field(default_factory=list)
    colors: list[ColorFact] = Field(default_factory=list)
    fonts: list[str] = Field(default_factory=list)
    headings: list[str] = Field(default_factory=list)
    prices: list[str] = Field(default_factory=list)
    guarantee_phrases: list[str] = Field(default_factory=list)
    review_snippets: list[str] = Field(default_factory=list)
    is_shopify: bool = False
    products: list[ProductFact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


Fetch = Callable[[str], tuple[int, str, str]]

_GENERIC_FONTS = {
    "cursive",
    "fantasy",
    "inherit",
    "initial",
    "math",
    "monospace",
    "sans-serif",
    "serif",
    "system-ui",
    "ui-monospace",
    "ui-rounded",
    "ui-sans-serif",
    "ui-serif",
    "unset",
}
_SYSTEM_FONTS = {
    "-apple-system",
    "android emoji",
    "apple color emoji",
    "arial",
    "blinkmacsystemfont",
    "emoji",
    "emojione color",
    "helvetica neue",
    "noto color emoji",
    "noto emoji",
    "noto sans",
    "segoe ui",
    "segoe ui emoji",
    "segoe ui symbol",
    "liberation sans",
    "dejavu sans",
    "helvetica",
    "roboto",
    "sans-serif",
    "serif",
    "system-ui",
    "system_ui",
    "twemoji",
}
_FONT_DECLARATION = re.compile(r"font-family\s*:\s*([^;}{]+)", re.IGNORECASE)
_CUSTOM_PROPERTY = re.compile(r"(?:^|[;{])\s*(--[\w-]+)\s*:\s*([^;}{]+)", re.IGNORECASE)
_CSS_VARIABLE = re.compile(
    r"var\(\s*(--[\w-]+)(?:\s*,\s*([^)]*))?\s*\)",
    re.IGNORECASE,
)
_HEX = re.compile(r"(?<![\w-])#([0-9a-f]{8}|[0-9a-f]{6}|[0-9a-f]{4}|[0-9a-f]{3})(?![0-9a-z])", re.I)
_PRICE = re.compile(
    r"(?<![\w])(?:(?:AUD|CAD|CHF|EUR|GBP|JPY|NOK|PLN|SEK|USD|€|\$|£)\s*"
    r"\d+(?:[ .]\d{3})*(?:[,.]\d{1,2})?"
    r"|\d+(?:[ .]\d{3})*(?:[,.]\d{1,2})?\s*"
    r"(?:AUD|CAD|CHF|EUR|GBP|JPY|NOK|PLN|SEK|USD|€|\$|£))(?![\w])",
    re.IGNORECASE,
)
_CURRENCY_SYMBOLS = {"€": "EUR", "$": "USD", "£": "GBP"}
_CURRENCY_CODES = {
    "AUD",
    "CAD",
    "CHF",
    "EUR",
    "GBP",
    "JPY",
    "NOK",
    "PLN",
    "SEK",
    "USD",
}
_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
_GUARANTEE_TERMS = ("money back", "guarantee", "garantie", "rembours", "satisfait")
_FOLLOW_PATH_TERMS = ("product", "shop", "collections")


def _collapse(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(value)).strip()


def _normalise_currency(value: str) -> str | None:
    candidate = value.strip().upper()
    return candidate if candidate in _CURRENCY_CODES else None


def _valid_heading(value: str) -> bool:
    if not value:
        return False
    without_currency = re.sub(
        rf"\b(?:{'|'.join(_CURRENCY_CODES)})\b|[€$£]",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    letters = re.findall(r"[^\W\d_]", without_currency, flags=re.UNICODE)
    return len(letters) >= 3


def _origin(value: str) -> str:
    parsed = urlparse(value)
    return urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))


def _same_origin(left: str, right: str) -> bool:
    a = urlparse(left)
    b = urlparse(right)
    return (a.scheme.lower(), a.netloc.lower()) == (b.scheme.lower(), b.netloc.lower())


def _normalise_hex(value: str) -> str | None:
    digits = value.lstrip("#")
    if len(digits) == 3:
        digits = "".join(char * 2 for char in digits)
    elif len(digits) == 4:
        digits = digits[:3]
        digits = "".join(char * 2 for char in digits)
    elif len(digits) == 8:
        digits = digits[:6]
    if len(digits) != 6 or not re.fullmatch(r"[0-9a-f]{6}", digits, flags=re.IGNORECASE):
        return None
    digits = digits.upper()
    if digits[0:2] == digits[2:4] == digits[4:6]:
        return None
    return f"#{digits}"


def _colour_counts(chunks: list[str]) -> list[ColorFact]:
    counts: Counter[str] = Counter()
    for chunk in chunks:
        for match in _HEX.finditer(chunk):
            colour = _normalise_hex(match.group(0))
            if colour is not None:
                counts[colour] += 1
    return [
        ColorFact(hex=colour, count=count)
        for colour, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]
    ]


_ICON_FONT = re.compile(r"(icon|icons|star|stars|symbol|emoji|awesome|glyph|fontello)", re.I)


def _clean_font(value: str) -> str | None:
    # Leftovers of split var() fallbacks carry stray quotes and parentheses.
    value = value.strip().strip("'\"()").strip().strip("'\"").strip()
    value = re.sub(r"\s*!important\s*$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value)
    if not value or value.casefold() in _GENERIC_FONTS or value.casefold() in _SYSTEM_FONTS:
        return None
    if _ICON_FONT.search(value):  # review widgets and icon sets, not brand typography
        return None
    return value


def _custom_properties(chunks: list[str]) -> dict[str, str]:
    properties: dict[str, str] = {}
    for chunk in chunks:
        for name, value in _CUSTOM_PROPERTY.findall(chunk):
            properties[name.casefold()] = value.strip()
    return properties


def _resolve_css_value(value: str, properties: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name, fallback = match.group(1), match.group(2)
        return properties.get(name.casefold(), fallback or match.group(0))

    return _CSS_VARIABLE.sub(replace, value)


def _font_families(declaration: str, properties: dict[str, str]) -> list[str]:
    # Themes chain variables (an override pointing at the base family): resolve a few
    # levels, then fall back to var() defaults so no fragment is split on its commas.
    resolved = declaration
    for _ in range(4):
        following = _resolve_css_value(resolved, properties)
        if following == resolved:
            break
        resolved = following
    resolved = _CSS_VARIABLE.sub(lambda match: match.group(2) or "", resolved)
    result: list[str] = []
    in_system_stack = False
    for value in resolved.split(","):
        cleaned = _clean_font(value)
        if cleaned is None:
            if value.strip().strip("'\"").casefold() == "-apple-system":
                in_system_stack = True
            continue
        if cleaned.casefold() == "roboto" and in_system_stack:
            continue
        result.append(cleaned)
    return result


def _fonts_from_css(chunks: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    properties = _custom_properties(chunks)
    for chunk in chunks:
        for declaration in _FONT_DECLARATION.findall(chunk):
            for family in _font_families(declaration, properties):
                if "var(" in family.casefold() or family.casefold() in seen:
                    continue
                result.append(family)
                seen.add(family.casefold())
    return result


def _google_fonts(hrefs: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for href in hrefs:
        parsed = urlparse(href)
        if "fonts.googleapis.com" not in parsed.netloc.lower():
            continue
        for encoded in re.findall(r"(?:^|&)family=([^&]+)", parsed.query, flags=re.IGNORECASE):
            family = unquote_plus(encoded).split(":", 1)[0].replace("|", ",")
            for item in family.split(","):
                cleaned = _clean_font(item)
                if cleaned is not None and cleaned.casefold() not in seen:
                    result.append(cleaned)
                    seen.add(cleaned.casefold())
    return result


def _prices(text: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for match in _PRICE.finditer(text):
        value = _collapse(match.group(0))
        if value.casefold() not in seen:
            result.append(value)
            seen.add(value.casefold())
    return result


def _currency_from_price(value: str) -> str | None:
    for symbol, currency in _CURRENCY_SYMBOLS.items():
        if symbol in value:
            return currency
    match = re.search(r"\b([A-Za-z]{3})\b", value)
    code = match.group(1).upper() if match else None
    return code if code in _CURRENCY_CODES else None


def _currency_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for match in _PRICE.finditer(text):
        currency = _currency_from_price(match.group(0))
        if currency is not None:
            counts[currency] += 1
    return counts


def _sentences(text: str) -> list[str]:
    return [part for part in (_collapse(item) for item in _SENTENCE_SPLIT.split(text)) if part]


def _guarantees(text: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for sentence in _sentences(text):
        lowered = sentence.casefold()
        if any(term in lowered for term in _GUARANTEE_TERMS) and lowered not in seen:
            result.append(sentence)
            seen.add(lowered)
    return result


def _reviews(text: str) -> list[str]:
    sentences = _sentences(text)
    marked = [
        index
        for index, sentence in enumerate(sentences)
        if "★" in sentence
        or "☆" in sentence
        or "review" in sentence.casefold()
        or "avis" in sentence.casefold()
    ]
    result: list[str] = []
    seen: set[str] = set()
    for index in marked:
        for candidate_index in (index - 1, index, index + 1):
            if not 0 <= candidate_index < len(sentences):
                continue
            candidate = sentences[candidate_index]
            key = candidate.casefold()
            if key not in seen:
                result.append(candidate)
                seen.add(key)
            if len(result) == 10:
                return result
    return result


class _PageParser(HTMLParser):
    """Collect just the HTML signals needed by the onboarding pass."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title_parts: list[str] = []
        self.description = ""
        self.language: str | None = None
        self.og_image: str | None = None
        self.logo_candidates: list[str] = []
        self.css_chunks: list[str] = []
        self.theme_colors: list[str] = []
        self.google_font_hrefs: list[str] = []
        self.stylesheet_hrefs: list[str] = []
        self.links: list[str] = []
        self.headings: list[str] = []
        self.visible_parts: list[str] = []
        self.currency_parts: list[str] = []
        self.currency_meta: list[str] = []
        self.inline_scripts: list[str] = []
        self._head_depth = 0
        self._svg_depth = 0
        self._title_depth = 0
        self._page_title_seen = False
        self._style_depth = 0
        self._script_depth = 0
        self._script_parts: list[str] | None = None
        self._noscript_depth = 0
        self._currency_excluded_depth = 0
        self._currency_exclusion_stack: list[bool] = []
        self._chrome_depth = 0
        self._chrome_stack: list[bool] = []
        self._heading: list[str] | None = None
        self._heading_excluded = False

    @staticmethod
    def _attrs(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
        return {key.lower(): value or "" for key, value in attrs}

    def _add_logo(self, href: str) -> None:
        absolute = urljoin(self.base_url, href)
        if absolute and absolute not in self.logo_candidates:
            self.logo_candidates.append(absolute)

    @staticmethod
    def _is_currency_excluded(values: dict[str, str]) -> bool:
        marker = f"{values.get('id', '')} {values.get('class', '')}".casefold()
        return any(term in marker for term in ("country", "currency", "localization"))

    @staticmethod
    def _is_chrome_container(tag: str, values: dict[str, str]) -> bool:
        if tag == "footer":
            return True
        marker = f"{values.get('id', '')} {values.get('class', '')}".casefold()
        return any(
            term in marker
            for term in ("cart", "drawer", "modal", "localization", "country", "header__menu")
        )

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        values = self._attrs(attrs)
        if tag in {
            "article",
            "blockquote",
            "div",
            "footer",
            "h1",
            "h2",
            "h3",
            "li",
            "main",
            "p",
            "section",
        }:
            self.visible_parts.append("\n")
        if tag == "head":
            self._head_depth += 1
        elif tag == "svg":
            self._svg_depth += 1
        currency_excluded = tag == "select" or self._is_currency_excluded(values)
        chrome_container = self._is_chrome_container(tag, values)
        if tag not in _VOID_TAGS:
            self._currency_exclusion_stack.append(currency_excluded)
            if currency_excluded:
                self._currency_excluded_depth += 1
            self._chrome_stack.append(chrome_container)
            if chrome_container:
                self._chrome_depth += 1
        if tag == "html" and values.get("lang"):
            self.language = values["lang"].strip()
        if (
            tag == "title"
            and self._head_depth
            and not self._svg_depth
            and not self._page_title_seen
        ):
            self._title_depth = 1
            self._page_title_seen = True
        elif tag == "style":
            self._style_depth += 1
        elif tag == "script":
            self._script_depth += 1
            self._script_parts = [] if not values.get("src") else None
        elif tag == "noscript":
            self._noscript_depth += 1
        elif tag in {"h1", "h2", "h3"} and self._heading is None:
            self._heading = []
            self._heading_excluded = self._chrome_depth > 0
        elif tag == "meta":
            key = values.get("name", "").casefold()
            property_name = values.get("property", "").casefold()
            content = values.get("content", "")
            if key == "description":
                self.description = _collapse(content)
            elif key == "theme-color":
                self.theme_colors.append(content)
            elif property_name == "og:image" and content:
                self.og_image = urljoin(self.base_url, content)
            elif property_name in {"og:price:currency", "product:price:currency"}:
                currency = _normalise_currency(content)
                if currency is not None:
                    self.currency_meta.append(currency)
        elif tag == "img":
            src = values.get("src", "")
            marker = " ".join(values.get(key, "") for key in ("src", "alt", "class", "id"))
            if src and "logo" in marker.casefold():
                self._add_logo(src)
        elif tag == "link":
            href = values.get("href", "")
            rel = set(values.get("rel", "").casefold().split())
            if href and (
                "icon" in rel or "apple-touch-icon" in rel or "apple-touch-icon-precomposed" in rel
            ):
                self._add_logo(href)
            if href and "stylesheet" in rel:
                self.stylesheet_hrefs.append(urljoin(self.base_url, href))
            if href:
                self.google_font_hrefs.append(urljoin(self.base_url, href))
        elif tag == "a" and values.get("href"):
            self.links.append(urljoin(self.base_url, values["href"]))

        if values.get("style"):
            self.css_chunks.append(values["style"])

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title" and self._title_depth:
            self._title_depth = 0
        elif tag == "style" and self._style_depth:
            self._style_depth -= 1
        elif tag == "script" and self._script_depth:
            if self._script_parts is not None:
                self.inline_scripts.append("".join(self._script_parts))
            self._script_depth -= 1
            self._script_parts = None
        elif tag == "noscript" and self._noscript_depth:
            self._noscript_depth -= 1
        elif tag in {"h1", "h2", "h3"} and self._heading is not None:
            heading = _collapse(" ".join(self._heading))
            if (
                not self._heading_excluded
                and _valid_heading(heading)
                and len(self.headings) < 20
            ):
                self.headings.append(heading)
            self._heading = None
            self._heading_excluded = False
        if tag == "head" and self._head_depth:
            self._head_depth -= 1
        elif tag == "svg" and self._svg_depth:
            self._svg_depth -= 1
        if tag in {
            "article",
            "blockquote",
            "div",
            "footer",
            "h1",
            "h2",
            "h3",
            "li",
            "main",
            "p",
            "section",
        }:
            self.visible_parts.append("\n")
        if tag not in _VOID_TAGS:
            currency_excluded = (
                self._currency_exclusion_stack.pop() if self._currency_exclusion_stack else False
            )
            if currency_excluded:
                self._currency_excluded_depth -= 1
            chrome_container = self._chrome_stack.pop() if self._chrome_stack else False
            if chrome_container:
                self._chrome_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._style_depth:
            self.css_chunks.append(data)
            return
        if self._script_depth:
            if self._script_parts is not None:
                self._script_parts.append(data)
            return
        if self._noscript_depth:
            return
        if self._title_depth:
            self.title_parts.append(data)
            return
        if self._heading is not None:
            self._heading.append(data)
        if data.strip():
            self.visible_parts.append(data)
            if not self._currency_excluded_depth:
                self.currency_parts.append(data)


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _strip_html(value: str) -> str:
    parser = _TextParser()
    try:
        parser.feed(value)
        parser.close()
    except Exception:
        return _collapse(re.sub(r"<[^>]+>", " ", value))[:600]
    return _collapse(" ".join(parser.parts))[:600]


def _extract_charset(content_type: str) -> str | None:
    match = re.search(r"charset\s*=\s*[\"']?([^;\s\"']+)", content_type, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _default_fetch(url: str, timeout: float) -> tuple[int, str, str, str]:
    request = Request(
        url,
        headers={"User-Agent": f"AdAnvil/{__version__} (+https://github.com/maxbuildog/adanvil)"},
    )
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        raw = response.read()
        charset = (
            response.headers.get_content_charset() or _extract_charset(content_type) or "utf-8"
        )
        try:
            text = raw.decode(charset, errors="replace")
        except LookupError:
            text = raw.decode("utf-8", errors="replace")
        return response.status, content_type, text, response.geturl()


def _fetch_one(
    target: str,
    *,
    timeout: float,
    fetch: Fetch | None,
) -> tuple[int, str, str, str]:
    if fetch is None:
        return _default_fetch(target, timeout)
    status, content_type, text = fetch(target)
    return status, content_type, text, target


def _warning(target: str, error: Exception | str) -> str:
    return f"Could not fetch {target}: {error}"


def _shopify_currency(scripts: list[str]) -> str | None:
    active_pattern = re.compile(
        r"Shopify\s*\.\s*currency\s*=\s*\{.*?(?:['\"]active['\"]|\bactive\b)\s*:\s*['\"]([A-Za-z]{3})['\"]",
        re.IGNORECASE | re.DOTALL,
    )
    for script in scripts:
        match = active_pattern.search(script)
        if match:
            currency = _normalise_currency(match.group(1))
            if currency is not None:
                return currency
    return None


def _site_currency(parser: _PageParser) -> str | None:
    if parser.currency_meta:
        return parser.currency_meta[0]
    shopify_currency = _shopify_currency(parser.inline_scripts)
    if shopify_currency is not None:
        return shopify_currency
    counts = _currency_counts(" ".join(parser.currency_parts))
    return counts.most_common(1)[0][0] if counts else None


def _as_price(value: object) -> tuple[Decimal, str] | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return Decimal(raw.replace(",", ".")), raw
    except InvalidOperation:
        return None


def _product_facts(payload: object, origin: str, site_currency: str | None) -> list[ProductFact]:
    if isinstance(payload, dict):
        entries = payload.get("products", [])
    else:
        entries = payload
    if not isinstance(entries, list):
        return []

    products: list[ProductFact] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        handle = str(item.get("handle", "")).strip()
        if not title and not handle:
            continue
        variants = item.get("variants", [])
        variants = variants if isinstance(variants, list) else []
        priced: list[tuple[Decimal, str, dict[str, object]]] = []
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            price = _as_price(variant.get("price"))
            if price is not None:
                priced.append((price[0], price[1], variant))
        selected: dict[str, object] = priced[0][2] if priced else {}
        if priced:
            _, selected_price, selected = min(priced, key=lambda entry: entry[0])
        else:
            fallback = _as_price(item.get("price"))
            selected_price = fallback[1] if fallback is not None else None
        compare = selected.get("compare_at_price")
        compare_price = (
            str(compare).strip() if compare is not None and str(compare).strip() else None
        )
        images: list[str] = []
        raw_images = item.get("images", [])
        if isinstance(raw_images, list):
            for raw_image in raw_images:
                image_url = raw_image.get("src") if isinstance(raw_image, dict) else raw_image
                if image_url and len(images) < 5:
                    absolute = urljoin(origin + "/", str(image_url))
                    if absolute not in images:
                        images.append(absolute)
        raw_tags = item.get("tags", [])
        if isinstance(raw_tags, str):
            tags = [part.strip() for part in raw_tags.split(",") if part.strip()]
        elif isinstance(raw_tags, list):
            tags = [str(tag) for tag in raw_tags]
        else:
            tags = []
        currency = site_currency or item.get("currency")
        products.append(
            ProductFact(
                title=title,
                handle=handle,
                url=urljoin(origin + "/", f"products/{handle}"),
                price=selected_price,
                compare_at_price=compare_price,
                currency=str(currency) if currency else None,
                images=images,
                description=_strip_html(str(item.get("body_html", item.get("description", "")))),
                tags=tags,
            )
        )
    return products


def analyze_site(
    url: str,
    *,
    timeout: float = 15,
    max_pages: int = 4,
    fetch: Fetch | None = None,
) -> SiteFacts:
    """Fetch and extract brand facts from *url*.

    ``fetch`` is deliberately tiny so tests can provide a local, deterministic
    transport. The default transport follows redirects through urllib and
    decodes the response using its declared charset.
    """

    facts = SiteFacts(url=url, final_url=url)
    try:
        status, content_type, html, final_url = _fetch_one(url, timeout=timeout, fetch=fetch)
    except Exception as exc:
        facts.warnings.append(_warning(url, exc))
        return facts

    facts.final_url = final_url
    if status < 200 or status >= 400:
        facts.warnings.append(f"HTTP {status} while fetching {url}")
    if (
        content_type
        and "html" not in content_type.casefold()
        and "text" not in content_type.casefold()
    ):
        facts.warnings.append(f"Unexpected content type for {url}: {content_type}")

    parser = _PageParser(final_url)
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:
        facts.warnings.append(f"Could not parse {final_url}: {exc}")

    visible_text_raw = " ".join(parser.visible_parts)
    visible_text = _collapse(visible_text_raw)
    facts.title = _collapse(" ".join(parser.title_parts))
    facts.description = parser.description
    facts.language = parser.language
    facts.og_image = parser.og_image
    facts.logo_candidates = list(parser.logo_candidates)
    facts.headings = list(parser.headings)
    facts.prices = _prices(visible_text)
    facts.guarantee_phrases = _guarantees(visible_text_raw)
    facts.review_snippets = _reviews(visible_text_raw)
    html_lower = html.casefold()
    facts.is_shopify = "cdn.shopify.com" in html_lower or "shopify.shop" in html_lower

    stylesheet_urls = [href for href in parser.stylesheet_hrefs if _same_origin(href, final_url)][
        :2
    ]
    style_chunks = list(parser.css_chunks) + list(parser.theme_colors)
    fonts = _google_fonts(parser.google_font_hrefs)
    for stylesheet_url in stylesheet_urls:
        try:
            style_status, _, stylesheet, _ = _fetch_one(
                stylesheet_url,
                timeout=timeout,
                fetch=fetch,
            )
            if style_status < 200 or style_status >= 400:
                facts.warnings.append(f"HTTP {style_status} while fetching {stylesheet_url}")
            style_chunks.append(stylesheet)
        except Exception as exc:
            facts.warnings.append(_warning(stylesheet_url, exc))
    for family in _fonts_from_css(style_chunks):
        if family.casefold() not in {font.casefold() for font in fonts}:
            fonts.append(family)
    facts.fonts = fonts
    facts.colors = _colour_counts(style_chunks)

    site_currency = _site_currency(parser)
    if facts.is_shopify:
        products_url = _origin(final_url) + "/products.json?limit=50"
        try:
            product_status, _, product_text, _ = _fetch_one(
                products_url,
                timeout=timeout,
                fetch=fetch,
            )
            if product_status < 200 or product_status >= 400:
                facts.warnings.append(f"HTTP {product_status} while fetching {products_url}")
            else:
                try:
                    payload = json.loads(product_text)
                    facts.products = _product_facts(payload, _origin(final_url), site_currency)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    facts.warnings.append(
                        f"Could not parse Shopify products from {products_url}: {exc}"
                    )
        except Exception as exc:
            facts.warnings.append(_warning(products_url, exc))
    else:
        seen_pages = {final_url}
        follow_limit = max(0, int(max_pages) - 1)
        follow_urls: list[str] = []
        for candidate in parser.links:
            parsed = urlparse(candidate)
            if not _same_origin(candidate, final_url) or not any(
                term in parsed.path.casefold() for term in _FOLLOW_PATH_TERMS
            ):
                continue
            clean = urlunparse(parsed._replace(fragment=""))
            if clean not in seen_pages and clean not in follow_urls:
                follow_urls.append(clean)
        for page_url in follow_urls[:follow_limit]:
            seen_pages.add(page_url)
            try:
                page_status, _, page_html, page_final = _fetch_one(
                    page_url,
                    timeout=timeout,
                    fetch=fetch,
                )
                if page_status < 200 or page_status >= 400:
                    facts.warnings.append(f"HTTP {page_status} while fetching {page_url}")
                    continue
                page_parser = _PageParser(page_final)
                page_parser.feed(page_html)
                page_parser.close()
                page_text = _collapse(" ".join(page_parser.visible_parts))
                for heading in page_parser.headings:
                    if len(facts.headings) >= 20:
                        break
                    if heading not in facts.headings:
                        facts.headings.append(heading)
                for price in _prices(page_text):
                    if price not in facts.prices:
                        facts.prices.append(price)
            except Exception as exc:
                facts.warnings.append(_warning(page_url, exc))

    return facts


def _short(value: object, limit: int) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def facts_summary(facts: SiteFacts, *, facts_file: str | None = None) -> dict[str, object]:
    """A compact view of the site facts for an agent: the essentials, not the page dump.

    Long lists are cut (``more`` says how many items were left out), texts are shortened and
    products keep only a title, price and link. The full facts stay in ``facts_file``.
    """

    more: dict[str, int] = {}

    def cut(name: str, items: list, limit: int) -> list:
        if len(items) > limit:
            more[name] = len(items) - limit
        return items[:limit]

    products = [
        {
            key: value
            for key, value in {
                "title": _short(product.title, 80),
                "price": product.price,
                "currency": product.currency,
                "compare_at_price": product.compare_at_price,
                "url": product.url,
            }.items()
            if value
        }
        for product in cut("products", facts.products, 8)
    ]
    summary: dict[str, object] = {
        "url": facts.url,
        "final_url": facts.final_url,
        "title": _short(facts.title, 120),
        "description": _short(facts.description, 240),
        "language": facts.language,
        "is_shopify": facts.is_shopify,
        "logo_candidates": cut("logo_candidates", facts.logo_candidates, 3),
        "colors": [colour.hex for colour in cut("colors", facts.colors, 5)],
        "fonts": cut("fonts", facts.fonts, 5),
        "headings": [_short(item, 100) for item in cut("headings", facts.headings, 8)],
        "prices": cut("prices", facts.prices, 6),
        "guarantee_phrases": [
            _short(item, 160) for item in cut("guarantee_phrases", facts.guarantee_phrases, 4)
        ],
        "review_snippets": [
            _short(item, 160) for item in cut("review_snippets", facts.review_snippets, 3)
        ],
        "product_count": len(facts.products),
        "products": products,
        "warnings": [_short(item, 200) for item in cut("warnings", facts.warnings, 6)],
    }
    if more:
        summary["more"] = more
    if facts_file:
        summary["facts_file"] = facts_file
    return summary


__all__ = [
    "ColorFact",
    "Fetch",
    "ProductFact",
    "SiteFacts",
    "analyze_site",
    "facts_summary",
]
