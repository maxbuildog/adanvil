"""Compose a clean call-to-action end card from a real product photo.

A brand without a ready-made call-to-action image (the brand's own comes first) gets one here:
soft gradient, a small pill, a big title with an accent word, the product photo in a white card,
a subline, badges and a button.
Text uses the shared ``adanvil.text`` helpers (fonts, emoji, optical centring).
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFilter, ImageOps
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from adanvil.brand import Brand, find_banned, load_brand
from adanvil.ffmpeg import extract_frame, probe
from adanvil.media import Format
from adanvil.project import resolve_in_project
from adanvil.text import get_provider, load_font
from adanvil.text.emoji import EmojiProvider, split_runs
from adanvil.text.fonts import is_font_reference, unusable_family_message, usable_family
from adanvil.text.layout import fit_size, measure_line, wrap
from adanvil.text.styles import _cap_height, _draw_runs, _line_width, _text_band, save_image

ENDCARDS_DIR = "brand/endcards"
DEFAULT_FAMILY = "poppins"
DEFAULT_ACCENT = "#2563EB"
DEFAULT_TEXT = "#18181B"
DEFAULT_MUTED = "#6E6E76"
DEFAULT_BACKGROUND = ("#FBF7F3", "#F3E9E2")
MAX_UPSCALE = 1.25
_SS = 3  # supersampling factor for shapes and icons
_STAR = (245, 180, 20)
_CHECK = (34, 160, 90)
_BADGE_BORDER = (230, 224, 220)
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Crop(_Model):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)


class TitleRun(_Model):
    text: str = Field(min_length=1)
    accent: bool = False


class Badge(_Model):
    text: str = Field(min_length=1, max_length=60)
    icon: Literal["check", "star", "none"] = "check"


class Button(_Model):
    text: str = Field(min_length=1, max_length=40)
    arrow: Literal["down", "right", "none"] = "down"


def _colour(value: str) -> str:
    try:
        ImageColor.getrgb(value)
    except ValueError as exc:
        raise ValueError(f"invalid colour {value!r}") from exc
    return value


class EndCardSpec(_Model):
    product_image: str = Field(min_length=1)
    t: float = Field(default=0.0, ge=0)
    # Source display pixels (after rotation), before scaling: the tool returns the source
    # size. Example: a 1920x1080 frame, crop {"x": 600, "y": 100, "w": 700, "h": 700}.
    crop: Crop | None = None
    pill: str | None = Field(default=None, max_length=40)
    title: list[TitleRun] = Field(min_length=1)
    subline: str | None = Field(default=None, max_length=80)
    badges: list[Badge] = Field(default_factory=list, max_length=3)
    button: Button | None = None
    accent: str | None = None
    text: str | None = None
    muted: str | None = None
    background: list[str] | None = Field(default=None, min_length=2, max_length=2)
    family: str = DEFAULT_FAMILY
    emoji: str | None = None
    format: Format = Field(default_factory=Format)

    @field_validator("title", mode="before")
    @classmethod
    def _title_runs(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [{"text": value}]
        if isinstance(value, list):
            return [{"text": item} if isinstance(item, str) else item for item in value]
        return value

    @field_validator("badges", mode="before")
    @classmethod
    def _badge_items(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [{"text": item} if isinstance(item, str) else item for item in value]
        return value

    @field_validator("button", mode="before")
    @classmethod
    def _button_text(cls, value: Any) -> Any:
        return {"text": value} if isinstance(value, str) else value

    @field_validator("accent", "text", "muted")
    @classmethod
    def _valid_colour(cls, value: str | None) -> str | None:
        return None if value is None else _colour(value)

    @field_validator("background")
    @classmethod
    def _valid_background(cls, value: list[str] | None) -> list[str] | None:
        if value is not None:
            for item in value:
                _colour(item)
        return value

    @field_validator("family")
    @classmethod
    def _known_family(cls, value: str) -> str:
        if is_font_reference(value):
            # A font file is checked against the project when the card is drawn; it is not
            # opened here (a network path would be contacted).
            return value
        try:
            load_font(value, 600, 20)
        except (ValueError, OSError) as exc:
            raise ValueError(str(exc)) from exc
        return value

    @model_validator(mode="after")
    def _check(self) -> EndCardSpec:
        if sum(len(run.text) for run in self.title) > 90:
            raise ValueError("title is limited to 90 characters")
        if not "".join(run.text for run in self.title).strip():
            raise ValueError("title must not be empty")
        if not 200 <= self.format.width <= 4096 or not 200 <= self.format.height <= 4096:
            raise ValueError("format width and height must be between 200 and 4096")
        return self

    def texts(self) -> list[str]:
        values = [self.pill, "".join(run.text for run in self.title), self.subline]
        values.extend(badge.text for badge in self.badges)
        values.append(self.button.text if self.button else None)
        return [value for value in values if value]


def parse_spec(value: Any) -> EndCardSpec:
    """Validate an end card spec and explain every problem with its field path."""

    if isinstance(value, EndCardSpec):
        return value
    try:
        return EndCardSpec.model_validate(value)
    except ValidationError as exc:
        problems = []
        for error in exc.errors():
            where = ".".join(str(part) for part in error["loc"]) or "spec"
            message = str(error["msg"]).removeprefix("Value error, ")
            problems.append(f"{where}: {message}")
        raise ValueError("Invalid end card spec: " + "; ".join(problems)) from exc


@dataclass
class EndCardRender:
    image: Image.Image
    layout: dict[str, Any]

    def save(self, path: str | Path) -> Path:
        return save_image(self.image.convert("RGB"), path, format="PNG", optimize=True)


def emoji_provider(root: Path, brand: Brand | None, emoji: str | None = None) -> EmojiProvider:
    """The emoji provider: ``emoji`` (twemoji, none or a project folder) or else the brand's."""

    name = emoji or (
        brand.visual.emoji if brand and brand.visual and brand.visual.emoji else "twemoji"
    )
    if name in {"twemoji", "none"}:
        return get_provider(name, cache_dir=Path(root) / "cache")
    return get_provider("directory", path=resolve_in_project(Path(root), str(name)))


def _load_brand(root: Path) -> Brand | None:
    path = Path(root) / "brand.yaml"
    if not path.is_file():
        return None
    try:
        return load_brand(path)
    except Exception as exc:
        raise ValueError(f"Could not load brand.yaml: {exc}") from exc


def _check_banned(spec: EndCardSpec, brand: Brand | None) -> None:
    banned = brand.claims.banned_words if brand and brand.claims else []
    problems = []
    for text in spec.texts():
        found = find_banned(text, banned)
        if found:
            problems.append(f"{text!r} contains {', '.join(repr(word) for word in found)}")
    if problems:
        raise ValueError("Banned words of the brand on the end card: " + "; ".join(problems))


def _load_product(root: Path, spec: EndCardSpec) -> tuple[Image.Image, tuple[int, int]]:
    source = resolve_in_project(Path(root), spec.product_image)
    if not source.is_file():
        raise ValueError(f"product_image does not exist: {spec.product_image}")
    try:
        kind = probe(source).kind
    except Exception as exc:
        raise ValueError(f"Could not read product_image {spec.product_image!r}: {exc}") from exc
    if kind == "video":
        stamp = f"{source.resolve()}:{source.stat().st_mtime_ns}:{spec.t:g}"
        digest = hashlib.sha1(stamp.encode("utf-8")).hexdigest()[:12]
        frame = Path(root) / "cache" / "frames" / f"endcard-{source.stem}-{digest}.png"
        frame.parent.mkdir(parents=True, exist_ok=True)
        if not frame.is_file():
            extract_frame(source, spec.t, frame)
        source = frame
    elif kind != "image":
        raise ValueError(f"product_image must be a photo or a video, not {kind}")
    with Image.open(source) as opened:
        photo = ImageOps.exif_transpose(opened).convert("RGBA")
    source_size = (photo.width, photo.height)
    if spec.crop is not None:
        crop = spec.crop
        if crop.x + crop.w > photo.width or crop.y + crop.h > photo.height:
            raise ValueError(
                f"crop {crop.x},{crop.y},{crop.w},{crop.h} exceeds the image "
                f"({photo.width}x{photo.height})"
            )
        photo = photo.crop((crop.x, crop.y, crop.x + crop.w, crop.y + crop.h))
    return photo, source_size


def _rgb(value: str) -> tuple[int, int, int]:
    return ImageColor.getrgb(value)[:3]


def _on_colour(background: tuple[int, int, int], dark: str) -> tuple[int, int, int]:
    """White text on a dark fill, the dark text colour on a light one."""

    def linear(channel: int) -> float:
        value = channel / 255
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    brightness = sum(
        w * linear(c) for w, c in zip((0.2126, 0.7152, 0.0722), background, strict=True)
    )
    return _rgb(dark) if brightness > 0.55 else (255, 255, 255)


def _gradient(size: tuple[int, int], top: str, bottom: str) -> Image.Image:
    mask = Image.linear_gradient("L").resize(size, Image.Resampling.BILINEAR)
    return Image.composite(
        Image.new("RGB", size, _rgb(bottom)), Image.new("RGB", size, _rgb(top)), mask
    ).convert("RGBA")


def _aa(img: Image.Image, box: tuple[float, float, float, float], paint) -> None:
    """Draw anti-aliased shapes: ``paint(draw, t)`` draws at 3x, ``t`` maps canvas coordinates."""

    x0, y0 = max(0, math.floor(box[0]) - 2), max(0, math.floor(box[1]) - 2)
    x1, y1 = min(img.width, math.ceil(box[2]) + 2), min(img.height, math.ceil(box[3]) + 2)
    layer = Image.new("RGBA", ((x1 - x0) * _SS, (y1 - y0) * _SS), (0, 0, 0, 0))

    def t(*values: float) -> list[float]:
        return [(v - (x0 if i % 2 == 0 else y0)) * _SS for i, v in enumerate(values)]

    paint(ImageDraw.Draw(layer), t)
    img.alpha_composite(layer.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS), (x0, y0))


def _round_rect(img, box, radius, fill, outline=None, width=0) -> None:
    def paint(d, t):
        d.rounded_rectangle(
            t(*box), radius * _SS, fill=fill, outline=outline, width=round(width * _SS)
        )

    _aa(img, box, paint)


def _icon(img: Image.Image, kind: str, cx: float, cy: float, r: float) -> None:
    box = (cx - r, cy - r, cx + r, cy + r)
    if kind == "check":

        def paint(d, t):
            d.ellipse(t(*box), fill=_CHECK)
            points = [
                (cx - 0.45 * r, cy),
                (cx - 0.1 * r, cy + 0.4 * r),
                (cx + 0.5 * r, cy - 0.4 * r),
            ]
            flat = [value for point in points for value in t(*point)]
            width = round(0.32 * r * _SS)
            d.line(flat, fill="white", width=width, joint="curve")
            for point in (points[0], points[2]):
                px, py = t(*point)
                d.ellipse((px - width / 2, py - width / 2, px + width / 2, py + width / 2), "white")

    else:

        def paint(d, t):
            points = []
            for index in range(10):
                radius = r if index % 2 == 0 else r * 0.47
                angle = math.radians(-90 + index * 36)
                points.extend(
                    t(cx + radius * math.cos(angle), cy + radius * math.sin(angle) + 0.06 * r)
                )
            d.polygon(points, fill=_STAR)
            d.line(
                [*points, points[0], points[1]],
                fill=_STAR,
                width=round(0.12 * r * _SS),
                joint="curve",
            )

    _aa(img, box, paint)


def _arrow(img: Image.Image, direction: str, cx: float, cy: float, size: float, fill) -> None:
    half = size / 2

    def paint(d, t):
        shaft, head = 0.16 * size, 0.5 * size
        if direction == "down":
            d.rectangle(
                t(cx - shaft / 2 * 1.2, cy - half, cx + shaft / 2 * 1.2, cy + 0.05 * size), fill
            )
            d.polygon(
                t(cx - head, cy - 0.02 * size, cx + head, cy - 0.02 * size, cx, cy + half), fill
            )
        else:
            d.rectangle(
                t(cx - half, cy - shaft / 2 * 1.2, cx + 0.05 * size, cy + shaft / 2 * 1.2), fill
            )
            d.polygon(
                t(cx - 0.02 * size, cy - head, cx - 0.02 * size, cy + head, cx + half, cy), fill
            )

    _aa(img, (cx - half, cy - half, cx + half, cy + half), paint)


def _draw_text(img, text, font, size, emoji, x, baseline, cap, fill) -> None:
    """Draw one line on a shared baseline (emoji centred on the capital band)."""

    draw = ImageDraw.Draw(img)
    emoji_size = round(size * 1.05)
    for is_emoji, value in split_runs(text):
        if is_emoji:
            icon = emoji.get(value, emoji_size)
            if icon is not None:
                img.alpha_composite(icon, (round(x), round(baseline - cap / 2 - emoji_size / 2)))
            x += emoji_size + 6
        else:
            draw.text((round(x), round(baseline)), value, font=font, fill=fill, anchor="ls")
            x += font.getlength(value)


def _words(runs: list[TitleRun]) -> list[list[tuple[str, bool]]]:
    """Title words, each a list of (text, accent) pieces so a word can change colour."""

    words: list[list[tuple[str, bool]]] = []
    current: list[tuple[str, bool]] = []
    for run in runs:
        for char in run.text:
            if char.isspace():
                if current:
                    words.append(current)
                    current = []
            elif current and current[-1][1] == run.accent:
                current[-1] = (current[-1][0] + char, run.accent)
            else:
                current.append((char, run.accent))
    if current:
        words.append(current)
    return words


def _piece_width(text: str, font, size: int) -> float:
    return measure_line(split_runs(text), font, round(size * 1.05), 6)


def _title_lines(words, font, size):
    """Word widths and the best 1-line and 2-line breaks at one size: [(width, spans)]."""

    space = font.getlength(" ")
    widths = [sum(_piece_width(text, font, size) for text, _ in word) for word in words]
    ends_emoji = [bool(split_runs(word[-1][0])[-1][0]) for word in words]

    def span_width(span: tuple[int, int]) -> float:
        start, end = span
        total = sum(widths[start:end]) + space * (end - start - 1)
        return total - 6 if ends_emoji[end - 1] else total  # the gap after an emoji is not ink

    count = len(words)
    breaks = [[(0, count)]]
    if count > 1:
        cut = min(
            range(1, count),
            key=lambda at: (max(span_width((0, at)), span_width((at, count))), at),
        )
        breaks.append([(0, cut), (cut, count)])
    return widths, [([span_width(span) for span in spans], spans) for spans in breaks]


def _fit_title(words, family, weight, size, max_width):
    """Largest size at which the title fits in 1 or 2 lines inside ``max_width``."""

    floor = max(28, round(size * 0.55))
    current = size
    while True:
        font = load_font(family, weight, current)
        widths, options = _title_lines(words, font, current)
        for row_widths, spans in options:
            if max(row_widths) <= max_width:
                return current, spans, widths, row_widths, False
        if current <= floor:
            return current, options[-1][1], widths, options[-1][0], True
        current = max(floor, current - 2)


@dataclass
class _Block:
    kind: str
    height: float
    gap: float  # space above this block
    data: Any = None


def _plan(spec, photo, W, H, k):
    """Measure every block at scale ``k``; ``None`` when even the smallest photo cannot fit."""

    family = spec.family
    side = round(W * 60 / 1080)
    max_width = W - 2 * side
    blocks: list[_Block] = []

    if spec.pill:
        size = fit_size([spec.pill], family, 600, round(30 * k), max_width - round(56 * k))[0]
        font = load_font(family, 600, size)
        width = _line_width(split_runs(spec.pill), font, size) + round(56 * k)
        blocks.append(_Block("pill", round(56 * k), 0, (size, width)))

    words = _words(spec.title)
    size, spans, widths, row_widths, overflow = _fit_title(
        words, family, 800, round(74 * k), max_width
    )
    font = load_font(family, 800, size)
    cap = _cap_height(font)
    line_height = round(size * 1.18)
    height = cap + (len(spans) - 1) * line_height + round(0.14 * size)
    blocks.append(
        _Block(
            "title",
            height,
            round(48 * k),
            (size, spans, widths, row_widths, words, cap, line_height, overflow),
        )
    )

    blocks.append(_Block("card", 0, round(56 * k)))

    if spec.subline:
        size = fit_size([spec.subline], family, 600, round(44 * k), max_width, round(32 * k))[0]
        font = load_font(family, 600, size)
        lines = wrap(spec.subline, font, max_width)
        line_height = round(size * 1.3)
        top = min(_text_band(font, split_runs(line))[0] for line in lines)
        bottom = max(_text_band(font, split_runs(line))[1] for line in lines)
        height = (bottom - top) + (len(lines) - 1) * line_height
        blocks.append(_Block("subline", height, round(44 * k), (size, lines, line_height, top)))

    badge_pad, icon_size, icon_gap = round(34 * k), round(40 * k), round(18 * k)
    for index, badge in enumerate(spec.badges):
        extra = 2 * badge_pad + (icon_size + icon_gap if badge.icon != "none" else 0)
        size = fit_size([badge.text], family, 600, round(38 * k), max_width - extra, round(24 * k))[
            0
        ]
        font = load_font(family, 600, size)
        width = _line_width(split_runs(badge.text), font, size) + extra
        gap = round((50 if index == 0 else 20) * k)
        blocks.append(
            _Block(
                "badge", round(78 * k), gap, (badge, size, width, icon_size, icon_gap, badge_pad)
            )
        )

    if spec.button:
        pad, arrow = round(56 * k), round(34 * k)
        arrow_space = arrow + round(18 * k) if spec.button.arrow != "none" else 0
        size = fit_size(
            [spec.button.text],
            family,
            700,
            round(44 * k),
            max_width - 2 * pad - arrow_space,
            round(28 * k),
        )[0]
        font = load_font(family, 700, size)
        text_width = _line_width(split_runs(spec.button.text), font, size)
        width = text_width + arrow_space + 2 * pad
        gap = round((48 if spec.badges else 50) * k)
        blocks.append(
            _Block("button", round(100 * k), gap, (size, text_width, arrow, arrow_space, width))
        )

    margin = max(round(40 * k), min(round(150 * k), round(H * 0.078)))
    blocks[0].gap = 0  # nothing above the first block
    fixed = sum(b.height + b.gap for b in blocks)
    pad = round(30 * k)
    available = H - 2 * margin - fixed
    max_photo_h = available - 2 * pad
    if max_photo_h < 200 * k:
        return None
    scale = min(MAX_UPSCALE, (round(880 * k) - 2 * pad) / photo.width, max_photo_h / photo.height)
    photo_size = (max(1, round(photo.width * scale)), max(1, round(photo.height * scale)))
    card_height = photo_size[1] + 2 * pad
    for block in blocks:
        if block.kind == "card":
            block.height = card_height
            block.data = (photo_size, pad, scale)
    return blocks, margin


def render_end_card(root: Path, spec: EndCardSpec | dict[str, Any]) -> EndCardRender:
    """Render the end card image and report where each element sits."""

    spec = parse_spec(spec)
    root = Path(root)
    warnings: list[str] = []
    family = usable_family(spec.family, root)
    if family is None:
        warnings.append(unusable_family_message(spec.family))
        family = DEFAULT_FAMILY
    spec = spec.model_copy(update={"family": family})
    brand = _load_brand(root)
    _check_banned(spec, brand)
    photo, source_size = _load_product(root, spec)
    provider = emoji_provider(root, brand, spec.emoji)

    accent = (
        spec.accent
        or (
            (brand.visual.accent_color or brand.visual.primary_color)
            if brand and brand.visual
            else None
        )
        or DEFAULT_ACCENT
    )
    ink = spec.text or DEFAULT_TEXT
    muted = spec.muted or DEFAULT_MUTED
    top_bg, bottom_bg = spec.background or DEFAULT_BACKGROUND
    accent_rgb, ink_rgb, muted_rgb = _rgb(accent), _rgb(ink), _rgb(muted)
    on_accent = _on_colour(accent_rgb, ink)

    W, H = spec.format.width, spec.format.height
    k = min(W / 1080, H / 1920 * 1.2)
    plan = None
    for _ in range(14):
        plan = _plan(spec, photo, W, H, k)
        if plan is not None:
            break
        k *= 0.93
    if plan is None:
        raise ValueError("The end card does not fit this format: shorten the texts or drop a badge")
    blocks, margin = plan

    # Spread the free height over the gaps (up to +50 %) and centre what is left.
    total = sum(b.height + b.gap for b in blocks)
    free = H - 2 * margin - total
    gaps = sum(b.gap for b in blocks)
    grow = min(free, 0.5 * gaps) / gaps if gaps else 0
    y = margin + (free - grow * gaps) / 2
    family = spec.family
    layout: dict[str, Any] = {"width": W, "height": H, "source_size": list(source_size)}
    if warnings:
        layout["warnings"] = warnings

    img = _gradient((W, H), top_bg, bottom_bg)
    for block in blocks:
        y += block.gap * (1 + grow)
        top = y
        bottom = y + block.height
        if block.kind == "pill":
            size, width = block.data
            x0 = (W - width) / 2
            _round_rect(img, (x0, top, x0 + width, bottom), block.height / 2, accent_rgb)
            font = load_font(family, 600, size)
            runs = split_runs(spec.pill)
            _draw_runs(
                img,
                runs,
                font,
                provider,
                x0 + round(28 * k),
                round(top),
                round(block.height),
                (*on_accent, 255),
                emoji_size=round(size * 1.05),
                optical=True,
            )
            layout["pill"] = [round(x0), round(top), round(x0 + width), round(bottom)]
        elif block.kind == "title":
            size, spans, widths, row_widths, words, cap, line_height, overflow = block.data
            font = load_font(family, 800, size)
            space = font.getlength(" ")
            ink_left, ink_right = W, 0
            for row, (start, end) in enumerate(spans):
                line_w = row_widths[row]
                x = (W - line_w) / 2
                ink_left, ink_right = min(ink_left, x), max(ink_right, x + line_w)
                baseline = top + cap + row * line_height
                for index in range(start, end):
                    for text, is_accent in words[index]:
                        fill = (*(accent_rgb if is_accent else ink_rgb), 255)
                        _draw_text(img, text, font, size, provider, x, baseline, cap, fill)
                        x += _piece_width(text, font, size)
                    x += space
            layout["title"] = [round(ink_left), round(top), round(ink_right), round(bottom)]
            layout["title_lines"] = [
                " ".join("".join(t for t, _ in words[i]) for i in range(start, end))
                for start, end in spans
            ]
            layout["title_size"] = size
            layout["title_overflow"] = overflow
        elif block.kind == "card":
            (pw, ph), pad, scale = block.data
            cw = pw + 2 * pad
            x0 = round((W - cw) / 2)
            radius = round(48 * k)
            shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ImageDraw.Draw(shadow).rounded_rectangle(
                (x0, top + 14 * k, x0 + cw, bottom + 14 * k), radius, fill=(0, 0, 0, 70)
            )
            blur_box = (
                max(0, x0 - 80),
                max(0, round(top) - 40),
                min(W, x0 + cw + 80),
                min(H, round(bottom + 14 * k) + 80),
            )
            region = shadow.crop(blur_box).filter(ImageFilter.GaussianBlur(22 * k))
            img.alpha_composite(region, blur_box[:2])
            _round_rect(img, (x0, top, x0 + cw, bottom), radius, (255, 255, 255))
            fitted = photo.resize((pw, ph), Image.Resampling.LANCZOS)
            mask = Image.new("L", (pw * _SS, ph * _SS), 0)
            ImageDraw.Draw(mask).rounded_rectangle(
                (0, 0, pw * _SS - 1, ph * _SS - 1), round(36 * k) * _SS, fill=255
            )
            mask = mask.resize((pw, ph), Image.Resampling.LANCZOS)
            fitted.putalpha(ImageChops.multiply(fitted.getchannel("A"), mask))
            img.alpha_composite(fitted, (x0 + pad, round(top) + pad))
            layout["photo_card"] = [x0, round(top), x0 + cw, round(bottom)]
            layout["photo"] = [x0 + pad, round(top) + pad, x0 + pad + pw, round(top) + pad + ph]
            layout["photo_scale"] = round(scale, 3)
        elif block.kind == "subline":
            size, lines, line_height, band_top = block.data
            font = load_font(family, 600, size)
            cap = _cap_height(font)
            left, right = W, 0
            for row, line in enumerate(lines):
                width = _line_width(split_runs(line), font, size)
                x = (W - width) / 2
                left, right = min(left, x), max(right, x + width)
                _draw_text(
                    img,
                    line,
                    font,
                    size,
                    provider,
                    x,
                    top - band_top + row * line_height,
                    cap,
                    (*muted_rgb, 255),
                )
            layout["subline"] = [round(left), round(top), round(right), round(bottom)]
        elif block.kind == "badge":
            badge, size, width, icon_size, icon_gap, pad = block.data
            x0 = (W - width) / 2
            _round_rect(
                img,
                (x0, top, x0 + width, bottom),
                block.height / 2,
                (255, 255, 255),
                outline=_BADGE_BORDER,
                width=max(1, round(2 * k)),
            )
            x = x0 + pad
            if badge.icon != "none":
                _icon(img, badge.icon, x + icon_size / 2, (top + bottom) / 2, icon_size / 2)
                x += icon_size + icon_gap
            font = load_font(family, 600, size)
            _draw_runs(
                img,
                split_runs(badge.text),
                font,
                provider,
                x,
                round(top),
                round(block.height),
                (*ink_rgb, 255),
                emoji_size=round(size * 1.05),
                optical=True,
            )
            layout.setdefault("badges", []).append(
                [round(x0), round(top), round(x0 + width), round(bottom)]
            )
        elif block.kind == "button":
            size, text_width, arrow, arrow_space, width = block.data
            x0 = (W - width) / 2
            _round_rect(img, (x0, top, x0 + width, bottom), block.height / 2, accent_rgb)
            x = x0 + round(56 * k)
            font = load_font(family, 700, size)
            _draw_runs(
                img,
                split_runs(spec.button.text),
                font,
                provider,
                x,
                round(top),
                round(block.height),
                (*on_accent, 255),
                emoji_size=round(size * 1.05),
                optical=True,
            )
            if arrow_space:
                _arrow(
                    img,
                    spec.button.arrow,
                    x + text_width + round(18 * k) + arrow / 2,
                    (top + bottom) / 2,
                    arrow * 0.9,
                    on_accent,
                )
            layout["button"] = [round(x0), round(top), round(x0 + width), round(bottom)]
        y = bottom
    return EndCardRender(img, layout)


def compose_end_card(root: Path, spec: EndCardSpec | dict[str, Any], out: Path) -> Path:
    """Render ``spec`` to a PNG at ``out`` (any format size) and return the path."""

    rendered = render_end_card(root, spec)
    try:
        return rendered.save(out)
    finally:
        rendered.image.close()


def write_end_card(root: Path, spec: EndCardSpec | dict[str, Any], name: str) -> tuple[Path, dict]:
    """Write ``brand/endcards/<name>.png`` and return its path and a JSON-ready layout report."""

    if not isinstance(name, str) or not _NAME.match(name):
        raise ValueError(
            f"name must be a short file name of letters, digits, - and _ (got {name!r})"
        )
    rendered = render_end_card(root, spec)
    try:
        output = rendered.save(Path(root) / ENDCARDS_DIR / f"{name}.png")
    finally:
        rendered.image.close()
    payload = {"path": f"{ENDCARDS_DIR}/{name}.png", **rendered.layout}
    return output, payload


__all__ = [
    "EndCardRender",
    "EndCardSpec",
    "compose_end_card",
    "emoji_provider",
    "parse_spec",
    "render_end_card",
    "write_end_card",
]
