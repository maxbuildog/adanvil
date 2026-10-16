from __future__ import annotations

import io
import re
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageColor, ImageDraw, ImageFilter
from pydantic import BaseModel, Field, field_validator, model_validator

from adanvil.safeio import replace_bytes

from .chunking import is_mark, is_opening_mark, match_emphasis
from .emoji import EmojiProvider, NoEmojiProvider, split_runs
from .fonts import load_font
from .layout import balance_lines, fit_size, measure_line, wrap

StyleName = Literal[
    "pill",
    "meme",
    "outline",
    "outline_soft",
    "box",
    "label",
    "comment",
]


class TextSpec(BaseModel):
    style: StyleName
    lines: list[str] = Field(min_length=1)
    y: int = 330
    anchor: Literal["top", "bottom"] = "top"
    x_center: int | None = None
    size: int | None = None
    family: str | None = None
    weight: int | None = None
    fg: str | None = None
    stroke: str | None = None
    box: str | None = None
    accent: str | None = None
    max_width: int = Field(default=960, gt=0)
    uniform_width: bool = False
    look: Literal["facebook", "tiktok"] = "facebook"
    header: str | None = None
    meta: str = "2w   Like   Reply"
    reactions: int | None = None
    # Word styling (animated captions and keyword emphasis). ``active`` is the index of the
    # spoken word drawn as the active one; words of ``emphasis`` are always in the highlight
    # colour. Both are off by default, and then the text renders exactly as before.
    highlight: str | None = None
    highlight_scale: float = Field(default=1.0, gt=0)
    highlight_box: str | None = None
    active: int | None = None
    emphasis: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _style_y_default(cls, values: object) -> object:
        if isinstance(values, dict) and "y" not in values and values.get("style") == "meme":
            values = dict(values)
            values["y"] = 1290
        return values

    @field_validator("lines", mode="before")
    @classmethod
    def _split_pipe(cls, value: object) -> object:
        if isinstance(value, str):
            return value.split("|")
        if isinstance(value, list):
            result: list[str] = []
            for line in value:
                result.extend(line.split("|") if isinstance(line, str) else [line])
            return result
        return value

    @field_validator("fg", "stroke", "box", "accent", "highlight", "highlight_box")
    @classmethod
    def _valid_color(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                ImageColor.getcolor(value, "RGBA")
            except ValueError as exc:
                raise ValueError(f"invalid colour {value!r}") from exc
        return value


    @property
    def styled(self) -> bool:
        """True when single words are drawn differently from the rest of the text."""

        return self.style != "comment" and (
            self.active is not None
            or bool(self.emphasis)
            or self.highlight_scale != 1
            or self.highlight_box is not None
        )


DEFAULT_HIGHLIGHT = "#FFD400"


def save_image(image: Image.Image, path: str | Path, **options: Any) -> Path:
    """Save ``image`` at ``path`` through a new file and a rename, so a link planted at ``path``
    is replaced, never followed. The format is ``options["format"]``, else the extension's."""

    destination = Path(path)
    image_format = options.pop("format", None)
    if image_format is None:
        image_format = Image.registered_extensions().get(destination.suffix.lower(), "PNG")
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, **options)
    replace_bytes(destination, buffer.getvalue())
    return destination


@dataclass
class RenderResult:
    image: Image.Image
    bbox: tuple[int, int, int, int]
    size_used: int
    overflow: bool
    lines: list[str]

    def save(self, path: str | Path) -> None:
        save_image(self.image, path)


_DEFAULTS: dict[str, dict[str, object]] = {
    "pill": {
        "family": "plus-jakarta-sans",
        "weight": 700,
        "size": 60,
        "fg": "#000000",
        "box": "#FFFFFF",
    },
    "meme": {
        "family": "plus-jakarta-sans",
        "weight": 700,
        "size": 52,
        "fg": "#000000",
        "box": "#FFFFFF",
    },
    "outline": {
        "family": "tiktok-sans",
        "weight": 800,
        "size": 62,
        "fg": "#FFFFFF",
        "stroke": "#000000",
    },
    "outline_soft": {
        "family": "tiktok-sans",
        "weight": 600,
        "size": 62,
        "fg": "#FFFFFF",
        "stroke": "#000000C8",
    },
    "box": {
        "family": "tiktok-sans",
        "weight": 700,
        "size": 56,
        "fg": "#FFFFFF",
        "box": "#000000E0",
    },
    "label": {
        "family": "poppins",
        "weight": 600,
        "size": 40,
        "fg": "#18181B",
        "box": "#FFFFFF",
    },
    "comment": {
        "family": "roboto",
        "weight": 400,
        "size": 44,
        "fg": "#161823",
    },
}


def default_size(style: str) -> int:
    """The font size a style uses when the caption sets none."""

    return int(_DEFAULTS[style]["size"])  # type: ignore[call-overload]


def _rgba(value: str | None, default: str) -> tuple[int, int, int, int]:
    return ImageColor.getcolor(value or default, "RGBA")


def _glyph_height(font) -> int:
    bbox = font.getbbox("Ag")
    return max(1, bbox[3] - bbox[1])


def _line_width(runs: list[tuple[bool, str]], font, size: int) -> float:
    width = measure_line(runs, font, round(size * 1.05), 6)
    if runs and runs[-1][0]:
        width -= 6  # the gap after an emoji is not ink: do not count it at the end of a line
    return width


def _cap_height(font) -> int:
    """Height of a capital letter, from its top to the baseline."""
    return max(1, -font.getbbox("H", anchor="ls")[1])


def _text_band(font, runs: list[tuple[bool, str]]) -> tuple[int, int]:
    """Top and bottom of the visual band of a line, relative to the baseline (y grows down).

    The band runs from the capital top to the baseline and only grows below the baseline
    when the line really has descenders, so mixed-case text sits at its visual centre
    whether or not it has a "g" or a "p".
    """
    top = font.getbbox("H", anchor="ls")[1]
    bottom = 0
    text = "".join(value for is_emoji, value in runs if not is_emoji).strip()
    if text:
        ink = font.getbbox(text, anchor="ls")
        top = min(top, ink[1])
        bottom = max(0, ink[3])
    return top, bottom


def _draw_runs(
    image: Image.Image,
    runs: list[tuple[bool, str]],
    font,
    emoji: EmojiProvider,
    x: float,
    top: int,
    line_height: int,
    fill: tuple[int, int, int, int],
    *,
    emoji_size: int,
    stroke_width: int = 0,
    stroke_fill: tuple[int, int, int, int] | None = None,
    shadow: bool = False,
    optical: bool = False,
) -> None:
    """Draw one line of text and emoji inside the slot ``top`` to ``top + line_height``.

    With ``optical`` the line is centred on its visual band (see ``_text_band``) and the
    emoji on the same centre, which is what boxes, pills and cards need to look even.
    """
    draw = ImageDraw.Draw(image)
    if optical:
        band_top, band_bottom = _text_band(font, runs)
        centre = top + line_height / 2
        text_y = centre - (band_top + band_bottom) / 2
        emoji_y = centre - emoji_size / 2
        anchor = "ls"
    else:
        glyph_box = font.getbbox("Ag")
        glyph_height = max(1, glyph_box[3] - glyph_box[1])
        text_y = top + max(0, (line_height - glyph_height) // 2) - glyph_box[1]
        emoji_y = top + (line_height - emoji_size) / 2
        anchor = "la"
    for is_emoji, value in runs:
        if is_emoji:
            icon = emoji.get(value, emoji_size)
            if icon is not None:
                if shadow:
                    alpha = icon.getchannel("A")
                    icon = Image.new("RGBA", icon.size, (0, 0, 0, fill[3]))
                    icon.putalpha(alpha.point(lambda pixel: pixel * fill[3] // 255))
                image.alpha_composite(icon, (round(x), round(emoji_y)))
            x += emoji_size + 6
        else:
            draw.text(
                (round(x), round(text_y)),
                value,
                font=font,
                fill=fill,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill or fill,
                anchor=anchor,
            )
            x += font.getlength(value)


_EMOJI_GAP = 6  # pixels after an emoji, as in the plain layout


@dataclass
class _Token:
    is_emoji: bool
    text: str
    unit: int  # the spoken word the token belongs to
    space_before: bool
    x: float = 0.0
    width: float = 0.0  # the slot reserved for the token


def _tokenize(lines: list[str]) -> list[list[_Token]]:
    """The words and emoji of each line, numbered by spoken word ("unit") across the lines.

    A token with no letter or digit (punctuation, a symbol) belongs to the word before it, an
    opening mark to the word after it, and an emoji to the word before it: they are lit together.
    """

    result: list[list[_Token]] = []
    unit = -1
    pending_open = False
    for line in lines:
        tokens: list[_Token] = []
        space = False
        for is_emoji, value in split_runs(line):
            if is_emoji:
                tokens.append(_Token(True, value, max(unit, 0), space))
                space = False
                continue
            for part in re.finditer(r"\s+|\S+", value):
                text = part.group()
                if text.isspace():
                    space = True
                    continue
                if is_mark(text):
                    if is_opening_mark(text):
                        unit += 1
                        pending_open = True
                        tokens.append(_Token(False, text, unit, space))
                    else:
                        tokens.append(_Token(False, text, max(unit, 0), space))
                else:
                    if pending_open:
                        pending_open = False
                    else:
                        unit += 1
                    tokens.append(_Token(False, text, unit, space))
                space = False
        result.append(tokens)
    return result


def spoken_words(lines: list[str]) -> list[str]:
    """The text of each spoken word of ``lines`` (a caption's lines), in order."""

    names: dict[int, list[str]] = {}
    for tokens in _tokenize(lines):
        for token in tokens:
            if not token.is_emoji:
                names.setdefault(token.unit, []).append(token.text)
    return [" ".join(names[unit]) for unit in sorted(names)]


class _Styled:
    """Layout and drawing of lines whose words are not all drawn alike.

    The geometry never depends on which word is active: every word gets the slot of its widest
    variant (the highlight scale), and a highlight box reserves its padding between the words, so
    the text does not move from one state of an animated caption to the next.
    """

    def __init__(self, lines: list[str], font, scaled_font, size: int, spec: TextSpec) -> None:
        self.lines = lines
        self.tokens = _tokenize(lines)
        self.font = font
        self.scaled_font = scaled_font
        self.size = size
        self.active = spec.active
        self.scale = spec.highlight_scale
        self.box = _rgba(spec.highlight_box, "#FFD400") if spec.highlight_box else None
        if spec.highlight:
            self.highlight_fill = _rgba(spec.highlight, DEFAULT_HIGHLIGHT)
        elif self.box is not None:
            luminance = 0.299 * self.box[0] + 0.587 * self.box[1] + 0.114 * self.box[2]
            self.highlight_fill = (0, 0, 0, 255) if luminance > 150 else (255, 255, 255, 255)
        else:
            self.highlight_fill = _rgba(DEFAULT_HIGHLIGHT, DEFAULT_HIGHLIGHT)
        self.marked: set[int] = (
            match_emphasis(spoken_words(lines), spec.emphasis)[0] if spec.emphasis else set()
        )
        self.pad_x = round(0.16 * size) if self.box is not None else 0
        self.emoji_size = round(size * 1.05)
        space = float(font.getlength(" "))
        self.widths: list[float] = []
        for tokens in self.tokens:
            x = 0.0
            previous: _Token | None = None
            for token in tokens:
                gap = 0.0
                if previous is not None:
                    if token.space_before:
                        gap += space + 2 * self.pad_x
                    if previous.is_emoji:
                        gap += _EMOJI_GAP
                token.x = x + gap
                if token.is_emoji:
                    token.width = float(self.emoji_size)
                else:
                    token.width = float(font.getlength(token.text))
                    if self.scale > 1:
                        token.width = max(token.width, float(scaled_font.getlength(token.text)))
                x = token.x + token.width
                previous = token
            self.widths.append(x)

    def line_width(self, index: int) -> float:
        return self.widths[index]

    def draw_line(
        self,
        image: Image.Image,
        index: int,
        emoji: EmojiProvider,
        x: float,
        top: int,
        line_height: int,
        fill: tuple[int, int, int, int],
        *,
        optical: bool = False,
        stroke_width: int = 0,
        stroke_fill: tuple[int, int, int, int] | None = None,
        shadow: bool = False,
        colored: bool = True,
    ) -> None:
        font = self.font
        tokens = self.tokens[index]
        if optical:
            band_top, band_bottom = _text_band(font, split_runs(self.lines[index]))
            centre = top + line_height / 2
            baseline = centre - (band_top + band_bottom) / 2
            emoji_y = centre - self.emoji_size / 2
        else:
            glyph_box = font.getbbox("Ag")
            glyph_height = max(1, glyph_box[3] - glyph_box[1])
            text_y = top + max(0, (line_height - glyph_height) // 2) - glyph_box[1]
            baseline = text_y + font.getmetrics()[0]
            emoji_y = top + (line_height - self.emoji_size) / 2
        draw = ImageDraw.Draw(image)
        cap = _cap_height(font)
        grown = max(1.0, self.scale)
        active = [token for token in tokens if token.unit == self.active and not token.is_emoji]
        if colored and self.box is not None and active:
            left = x + min(token.x for token in active) - self.pad_x
            right = x + max(token.x + token.width for token in active) + self.pad_x
            descent = max(0, font.getbbox("p", anchor="ls")[3])
            pad_y = round(0.12 * self.size)
            shift = cap * (self.scale - 1) / 2
            draw.rounded_rectangle(
                (
                    round(left),
                    round(baseline + shift - cap * grown - pad_y),
                    round(right),
                    round(baseline + shift + descent * grown + pad_y),
                ),
                radius=round(0.28 * self.size),
                fill=self.box,
            )
        for token in tokens:
            if token.is_emoji:
                icon = emoji.get(token.text, self.emoji_size)
                if icon is not None:
                    if shadow:
                        alpha = icon.getchannel("A")
                        icon = Image.new("RGBA", icon.size, (0, 0, 0, fill[3]))
                        icon.putalpha(alpha.point(lambda pixel: pixel * fill[3] // 255))
                    image.alpha_composite(icon, (round(x + token.x), round(emoji_y)))
                continue
            lit = token.unit == self.active
            colour = self.highlight_fill if colored and (lit or token.unit in self.marked) else fill
            use = self.scaled_font if lit and self.scale != 1 else font
            # Every word is centred in its slot, so the extra room of a highlight scale is
            # shared evenly and the spacing looks the same whichever word is lit.
            left = x + token.x + (token.width - float(use.getlength(token.text))) / 2
            line_baseline = baseline
            if use is not font:
                line_baseline += cap * (self.scale - 1) / 2
            draw.text(
                (round(left), round(line_baseline)),
                token.text,
                font=use,
                fill=colour,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill or colour,
                anchor="ls",
            )


def _width(styled: _Styled | None, index: int, line: str, font, size: int) -> float:
    """Width of line ``index``: plain, or with the slots of a styled layout."""

    if styled is not None:
        return styled.line_width(index)
    return _line_width(split_runs(line), font, size)


def _paint(
    styled: _Styled | None,
    image: Image.Image,
    index: int,
    line: str,
    font,
    emoji: EmojiProvider,
    x: float,
    top: int,
    line_height: int,
    fill: tuple[int, int, int, int],
    **options,
) -> None:
    """Draw line ``index`` plain, or word by word when the layout is styled."""

    if styled is not None:
        styled.draw_line(image, index, emoji, x, top, line_height, fill, **options)
        return
    options.pop("colored", None)
    _draw_runs(
        image,
        split_runs(line),
        font,
        emoji,
        x,
        top,
        line_height,
        fill,
        emoji_size=round(font.size * 1.05),
        **options,
    )


def _alpha_crop(image: Image.Image) -> Image.Image:
    bbox = image.getchannel("A").getbbox()
    if bbox is None:
        return Image.new("RGBA", (1, 1))
    return image.crop(bbox)


_TEXT_ONLY_STYLES = {"outline", "outline_soft"}
# Tall accents and descenders: a band that fits them holds every line of the same layout.
_BAND_REFERENCE = "ÂÉÀHgjpqy"
_BAND_CACHE: dict[tuple, tuple[int, int]] = {}


def _reference_band(spec: TextSpec, line_count: int, used: int) -> tuple[int, int] | None:
    """Vertical ink extent of reference text laid out exactly like ``spec``.

    Text-only styles are cropped to their ink, so a chunk with an accent or a descender
    would sit a few pixels higher or lower than the next one. Cropping every chunk to
    the band of the same reference text keeps the baseline still from chunk to chunk.
    """

    if spec.style not in _TEXT_ONLY_STYLES or line_count < 1:
        return None
    key = (spec.style, spec.family, spec.weight, used, line_count, spec.stroke, spec.max_width)
    if key not in _BAND_CACHE:
        reference = spec.model_copy(
            update={
                "lines": [_BAND_REFERENCE] * line_count,
                "size": used,
                "active": None,
                "emphasis": [],
            }
        )
        block, *_ = _build_block(reference, NoEmojiProvider())
        box = block.getchannel("A").getbbox()
        block.close()
        if box is None:
            return None
        _BAND_CACHE[key] = (box[1], box[3])
    return _BAND_CACHE[key]


def _band_crop(block: Image.Image, band: tuple[int, int] | None) -> Image.Image:
    """Crop to the ink horizontally and, when given, to ``band`` (widened by the ink) vertically."""

    box = block.getchannel("A").getbbox()
    if box is None:
        return Image.new("RGBA", (1, 1))
    if band is None:
        return block.crop(box)
    top = min(band[0], box[1])
    bottom = max(band[1], box[3])
    return block.crop((box[0], top, box[2], bottom))


def _finish(
    canvas: Image.Image,
    block: Image.Image,
    x_center: int,
    y: int,
    anchor: str,
    band: tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    cropped = _band_crop(block, band)
    x = round(x_center - cropped.width / 2)
    top = y if anchor == "top" else y - cropped.height
    left = max(0, x)
    upper = max(0, top)
    right = min(canvas.width, x + cropped.width)
    lower = min(canvas.height, top + cropped.height)
    if right > left and lower > upper:
        source = cropped.crop((left - x, upper - top, right - x, lower - top))
        canvas.alpha_composite(source, (left, upper))
    bbox = canvas.getchannel("A").getbbox()
    return bbox or (0, 0, 0, 0)


def _fit_for_style(
    lines: list[str],
    family: str,
    weight: int,
    size: int,
    style: str,
    max_width: int,
    look: str = "facebook",
) -> tuple[int, bool]:
    padding = {"pill": 48, "meme": 48, "box": 56, "label": 60}.get(style, 48)
    if style == "comment" and look == "facebook":
        padding = 186
    return fit_size(lines, family, weight, size, max(1, max_width - padding))


def _prepare_lines(spec: TextSpec, family: str, weight: int, size: int) -> list[str]:
    if spec.style == "meme" and len(spec.lines) == 1:
        font = load_font(family, weight, size)
        current = spec.lines[0]
        inner = max(1, spec.max_width - 48)
        if _line_width(split_runs(current), font, size) > inner:
            for count in range(2, 6):
                candidate = balance_lines(current, count, font)
                if all(_line_width(split_runs(line), font, size) <= inner for line in candidate):
                    return candidate
            return balance_lines(current, 5, font)
    if spec.style == "label":
        font = load_font(family, weight, size)
        width = max(1, spec.max_width - 60)
        result: list[str] = []
        for line in spec.lines:
            result.extend(wrap(line, font, width))
        return result or [""]
    if spec.style == "comment":
        font = load_font(family, weight, size)
        width = max(40, spec.max_width - (186 if spec.look == "facebook" else 48))
        return wrap(" ".join(spec.lines), font, width)
    return list(spec.lines)


def _build_pill(
    lines: list[str],
    font,
    size: int,
    emoji: EmojiProvider,
    fg: tuple[int, int, int, int],
    box: tuple[int, int, int, int],
    uniform_width: bool,
    styled: _Styled | None = None,
) -> Image.Image:
    line_height = int(size * 1.4)
    widths = [
        ceil(_width(styled, index, line, font, size)) + 48 for index, line in enumerate(lines)
    ]
    widest = max(widths)
    actual_widths = [widest if uniform_width else width for width in widths]
    height = line_height + max(0, len(lines) - 1) * (line_height - 2)
    block = Image.new("RGBA", (widest + 8, height + 8), (0, 0, 0, 0))
    draw = ImageDraw.Draw(block)
    for index, (line, width) in enumerate(zip(lines, actual_widths, strict=True)):
        top = index * (line_height - 2)
        left = (block.width - width) // 2
        draw.rounded_rectangle((left, top, left + width, top + line_height), radius=20, fill=box)
        text_width = _width(styled, index, line, font, size)
        _paint(
            styled,
            block,
            index,
            line,
            font,
            emoji,
            left + (width - text_width) / 2,
            top,
            line_height,
            fg,
            optical=True,
        )
    return block


def _build_outline(
    lines: list[str],
    font,
    size: int,
    emoji: EmojiProvider,
    fg: tuple[int, int, int, int],
    stroke: tuple[int, int, int, int],
    soft: bool,
    styled: _Styled | None = None,
) -> Image.Image:
    line_height = int(size * 1.28) if soft else int(sum(font.getmetrics()) * 1.08)
    widths = [ceil(_width(styled, index, line, font, size)) for index, line in enumerate(lines)]
    widest = max(widths, default=1)
    pad = 22
    block = Image.new(
        "RGBA",
        (widest + pad * 2 + 16, line_height * len(lines) + pad * 2 + 16),
        (0, 0, 0, 0),
    )
    shadow_layer = Image.new("RGBA", block.size, (0, 0, 0, 0))
    shadow_fill = (0, 0, 0, 242 if soft else 150)
    for index, (line, width) in enumerate(zip(lines, widths, strict=True)):
        x = pad + (widest - width) / 2
        y = pad + index * line_height
        _paint(
            styled,
            shadow_layer,
            index,
            line,
            font,
            emoji,
            x,
            y,
            line_height,
            shadow_fill,
            stroke_width=0 if soft else 6,
            stroke_fill=shadow_fill,
            shadow=True,
            colored=False,
        )
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(6))
    block.alpha_composite(shadow_layer, (4, 4))
    for index, (line, width) in enumerate(zip(lines, widths, strict=True)):
        x = pad + (widest - width) / 2
        y = pad + index * line_height
        _paint(
            styled,
            block,
            index,
            line,
            font,
            emoji,
            x,
            y,
            line_height,
            fg,
            stroke_width=4,
            stroke_fill=stroke,
        )
    return block


def _slot_height(font, size: int, lines: list[str], pad_v: float) -> int:
    """Height of one text slot: the capital band, half a descender, and the padding around.

    A line that holds an emoji (drawn about 1.05 em tall) gets the room it needs, so the
    slot height is the same for every line of a box.
    """
    allowance = max(0, font.getbbox("p", anchor="ls")[3]) // 2
    height = _cap_height(font) + allowance + round(2 * pad_v * size)
    if any(is_emoji for line in lines for is_emoji, _ in split_runs(line)):
        height = max(height, round(size * 1.05) + round(0.2 * size))
    return height


def _build_box(
    lines: list[str],
    font,
    size: int,
    emoji: EmojiProvider,
    fg: tuple[int, int, int, int],
    box: tuple[int, int, int, int],
    uniform_width: bool,
    styled: _Styled | None = None,
) -> Image.Image:
    pad_x = round(0.5 * size)
    gap = round(0.12 * size)
    radius = round(0.3 * size)
    line_height = _slot_height(font, size, lines, 0.28)
    widths = [
        ceil(_width(styled, index, line, font, size)) + 2 * pad_x
        for index, line in enumerate(lines)
    ]
    widest = max(widths, default=2 * pad_x)
    actual_widths = [widest if uniform_width else width for width in widths]
    height = len(lines) * line_height + max(0, len(lines) - 1) * gap
    block = Image.new("RGBA", (widest + 8, height + 8), (0, 0, 0, 0))
    draw = ImageDraw.Draw(block)
    top = 0
    for index, (line, width) in enumerate(zip(lines, actual_widths, strict=True)):
        left = (block.width - width) // 2
        draw.rounded_rectangle(
            (left, top, left + width, top + line_height), radius=radius, fill=box
        )
        text_width = _width(styled, index, line, font, size)
        _paint(
            styled,
            block,
            index,
            line,
            font,
            emoji,
            left + (width - text_width) / 2,
            top,
            line_height,
            fg,
            optical=True,
        )
        top += line_height + gap
    return block


def _build_label(
    lines: list[str],
    font,
    size: int,
    emoji: EmojiProvider,
    fg: tuple[int, int, int, int],
    box: tuple[int, int, int, int],
    styled: _Styled | None = None,
) -> Image.Image:
    """A small clean tag: one white rounded card, dark text, soft shadow, no accent."""
    pad_x = round(0.55 * size)
    radius = round(0.4 * size)
    line_height = _slot_height(font, size, lines, 0.35)
    widths = [ceil(_width(styled, index, line, font, size)) for index, line in enumerate(lines)]
    card_width = max(widths, default=1) + 2 * pad_x
    card_height = len(lines) * line_height
    margin = 24
    block = Image.new("RGBA", (card_width + 2 * margin, card_height + 2 * margin), (0, 0, 0, 0))
    shadow = Image.new("RGBA", block.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        (margin, margin + 5, margin + card_width, margin + 5 + card_height),
        radius=radius,
        fill=(0, 0, 0, 70),
    )
    block.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(9)))
    ImageDraw.Draw(block).rounded_rectangle(
        (margin, margin, margin + card_width, margin + card_height), radius=radius, fill=box
    )
    for index, line in enumerate(lines):
        _paint(
            styled,
            block,
            index,
            line,
            font,
            emoji,
            margin + (card_width - widths[index]) / 2,
            margin + index * line_height,
            line_height,
            fg,
            optical=True,
        )
    return block


def _avatar(size: int = 88) -> Image.Image:
    avatar = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(avatar)
    for y in range(size):
        ratio = y / max(1, size - 1)
        colour = (round(121 - ratio * 38), round(181 - ratio * 47), round(205 - ratio * 34), 255)
        draw.line((0, y, size, y), fill=colour)
    draw.ellipse((22, 14, 66, 58), fill=(221, 181, 151, 150))
    draw.ellipse((10, 49, 78, 112), fill=(76, 105, 139, 150))
    avatar = avatar.filter(ImageFilter.GaussianBlur(5))
    mask = Image.new("L", (size * 4, size * 4), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size * 4 - 1, size * 4 - 1), fill=255)
    mask = mask.resize((size, size), Image.Resampling.LANCZOS)
    avatar.putalpha(mask)
    return avatar


def _build_facebook_comment(
    lines: list[str],
    font,
    size: int,
    emoji: EmojiProvider,
    fg: tuple[int, int, int, int],
    max_width: int,
    meta: str,
    reactions: int | None,
) -> Image.Image:
    available_bubble = max(140, max_width - 138)
    inner_width = max(40, min(690, available_bubble - 48))
    comment_lines = wrap(" ".join(lines), font, inner_width)
    text_widths = [ceil(_line_width(split_runs(line), font, size)) for line in comment_lines]
    bubble_width = max(160, min(available_bubble, max(text_widths, default=1) + 48))
    name_font = load_font("roboto", 700, size)
    meta_font = load_font("roboto", 600, 32)
    name_height = max(32, int(size * 0.9))
    line_height = int(size * 1.15)
    bubble_height = 24 + name_height + 8 + len(comment_lines) * line_height + 18
    if reactions is not None:
        bubble_height += 38
    outer_width = 18 + 88 + 14 + bubble_width + 18
    card_left, card_top = 8, 8
    meta_y = card_top + 18 + bubble_height + 8
    meta_bbox = meta_font.getbbox(meta)
    meta_bottom = meta_y + meta_bbox[3]
    outer_height = max(88 + 36, meta_bottom - card_top + 20)
    block = Image.new("RGBA", (outer_width + 30, outer_height + 30), (0, 0, 0, 0))
    shadow = Image.new("RGBA", block.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        (card_left, card_top, card_left + outer_width, card_top + outer_height),
        radius=34,
        fill=(0, 0, 0, 100),
    )
    block.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(12)), (8, 8))
    draw = ImageDraw.Draw(block)
    draw.rounded_rectangle(
        (card_left, card_top, card_left + outer_width, card_top + outer_height),
        radius=34,
        fill=(255, 255, 255, 255),
    )
    avatar = _avatar()
    avatar_x = card_left + 18
    avatar_y = card_top + 18
    block.alpha_composite(avatar, (avatar_x, avatar_y))
    bubble_x = avatar_x + 88 + 14
    bubble_y = card_top + 18
    draw.rounded_rectangle(
        (bubble_x, bubble_y, bubble_x + bubble_width, bubble_y + bubble_height),
        radius=36,
        fill=(240, 242, 245, 255),
    )
    name_layer = Image.new("RGBA", block.size, (0, 0, 0, 0))
    name_draw = ImageDraw.Draw(name_layer)
    name_draw.text(
        (bubble_x + 24, bubble_y + 16),
        "Customer name",
        font=name_font,
        fill=(101, 103, 107, 230),
    )
    block.alpha_composite(name_layer.filter(ImageFilter.GaussianBlur(7)))
    comment_top = bubble_y + 24 + name_height + 8
    for index, line in enumerate(comment_lines):
        _draw_runs(
            block,
            split_runs(line),
            font,
            emoji,
            bubble_x + 24,
            comment_top + index * line_height,
            line_height,
            fg,
            emoji_size=round(size * 1.05),
        )
    draw.text((bubble_x, meta_y), meta, font=meta_font, fill=(101, 103, 107, 255))
    if reactions is not None:
        pill_w = 100
        pill_h = 36
        pill_x = bubble_x + bubble_width - pill_w - 14
        pill_y = bubble_y + bubble_height - pill_h - 10
        draw.rounded_rectangle(
            (pill_x, pill_y, pill_x + pill_w, pill_y + pill_h), radius=18, fill="white"
        )
        blue_left, blue_top = pill_x + 8, pill_y + 7
        blue_right, blue_bottom = pill_x + 30, pill_y + 29
        draw.ellipse((blue_left, blue_top, blue_right, blue_bottom), fill="#1877F2")
        draw.rounded_rectangle(
            (pill_x + 12, pill_y + 15, pill_x + 19, pill_y + 23), radius=2, fill="white"
        )
        draw.rounded_rectangle(
            (pill_x + 18, pill_y + 11, pill_x + 25, pill_y + 23), radius=2, fill="white"
        )
        red_outline = (pill_x + 23, pill_y + 4, pill_x + 49, pill_y + 30)
        draw.ellipse(red_outline, fill="white")
        red_left, red_top = pill_x + 26, pill_y + 7
        red_right, red_bottom = pill_x + 46, pill_y + 27
        draw.ellipse((red_left, red_top, red_right, red_bottom), fill="#F33E58")
        draw.ellipse((pill_x + 29, pill_y + 11, pill_x + 36, pill_y + 18), fill="white")
        draw.ellipse((pill_x + 36, pill_y + 11, pill_x + 43, pill_y + 18), fill="white")
        draw.polygon(
            ((pill_x + 28, pill_y + 15), (pill_x + 44, pill_y + 15), (pill_x + 36, pill_y + 24)),
            fill="white",
        )
        draw.text((pill_x + 58, pill_y + 7), str(reactions), font=meta_font, fill="#65676B")
    return block


def _build_tiktok_comment(
    lines: list[str],
    font,
    size: int,
    emoji: EmojiProvider,
    fg: tuple[int, int, int, int],
    max_width: int,
    header: str,
) -> Image.Image:
    body_width = max(40, max_width - 48)
    comment_lines = wrap(" ".join(lines), font, body_width)
    header_font = load_font("roboto", 600, max(12, round(size * 0.7)))
    header_width = ceil(header_font.getlength(header)) if header else 0
    widths = [ceil(_line_width(split_runs(line), font, size)) for line in comment_lines]
    content_width = max([header_width, *widths, 1])
    bubble_width = min(max_width, content_width + 48)
    header_height = max(20, round(size * 0.7 * 1.2))
    line_height = int(size * 1.15)
    bubble_height = 20 + header_height + 6 + len(comment_lines) * line_height + 22
    block = Image.new("RGBA", (bubble_width + 20, bubble_height + 22), (0, 0, 0, 0))
    draw = ImageDraw.Draw(block)
    draw.rounded_rectangle((8, 8, 8 + bubble_width, 8 + bubble_height), radius=28, fill="white")
    draw.polygon(
        ((26, 8 + bubble_height - 2), (14, 8 + bubble_height + 14), (43, 8 + bubble_height - 2)),
        fill="white",
    )
    draw.ellipse((24, 20, 56, 52), fill="#FE2C55")
    if header:
        draw.text((72, 20), header, font=header_font, fill="#8A8B91")
    body_top = 20 + header_height + 6
    for index, line in enumerate(comment_lines):
        width = widths[index]
        _draw_runs(
            block,
            split_runs(line),
            font,
            emoji,
            8 + (bubble_width - width) / 2,
            body_top + index * line_height,
            line_height,
            fg,
            emoji_size=round(size * 1.05),
        )
    return block


def _styled_layout(
    spec: TextSpec, lines: list[str], family: str, weight: int, size: int
) -> tuple[_Styled, int, bool]:
    """The word-by-word layout of ``lines`` and the font size it fits at (like ``fit_size``)."""

    padding = {"pill": 48, "meme": 48, "box": 56, "label": 60}.get(spec.style, 48)
    limit = max(1, spec.max_width - padding)
    floor = min(max(28, round(size * 0.6)), size)
    candidate = size
    while True:
        font = load_font(family, weight, candidate)
        scaled = (
            load_font(family, weight, max(1, round(candidate * spec.highlight_scale)))
            if spec.highlight_scale != 1
            else font
        )
        styled = _Styled(lines, font, scaled, candidate, spec)
        if all(width <= limit for width in styled.widths):
            return styled, candidate, False
        if candidate <= floor:
            return styled, candidate, True
        candidate = max(floor, candidate - 2)


def _build_block(
    spec: TextSpec, emoji: EmojiProvider
) -> tuple[Image.Image, list[str], int, bool]:
    """The text block of ``spec`` (not yet placed on a canvas), its lines, size and overflow."""

    defaults = _DEFAULTS[spec.style]
    family = spec.family or str(defaults["family"])
    weight = int(spec.weight if spec.weight is not None else defaults["weight"])
    size = int(spec.size if spec.size is not None else defaults["size"])
    lines = _prepare_lines(spec, family, weight, size)

    styled: _Styled | None = None
    if spec.styled:
        styled, used, overflow = _styled_layout(spec, lines, family, weight, size)
    else:
        used, overflow = _fit_for_style(
            lines, family, weight, size, spec.style, spec.max_width, spec.look
        )
    font = load_font(family, weight, used)
    fg = _rgba(spec.fg, str(defaults.get("fg", "#FFFFFF")))
    stroke = _rgba(spec.stroke, str(defaults.get("stroke", "#000000")))
    box = _rgba(spec.box, str(defaults.get("box", "#FFFFFF")))

    if spec.style in {"pill", "meme"}:
        block = _build_pill(lines, font, used, emoji, fg, box, spec.uniform_width, styled)
    elif spec.style == "outline":
        block = _build_outline(lines, font, used, emoji, fg, stroke, False, styled)
    elif spec.style == "outline_soft":
        block = _build_outline(lines, font, used, emoji, fg, stroke, True, styled)
    elif spec.style == "box":
        block = _build_box(lines, font, used, emoji, fg, box, spec.uniform_width, styled)
    elif spec.style == "label":
        block = _build_label(lines, font, used, emoji, fg, box, styled)
    elif spec.look == "facebook":
        block = _build_facebook_comment(
            lines, font, used, emoji, fg, spec.max_width, spec.meta, spec.reactions
        )
    else:
        block = _build_tiktok_comment(
            lines,
            font,
            used,
            emoji,
            fg,
            spec.max_width,
            spec.header or "",
        )
    return block, lines, used, overflow


def render(
    spec: TextSpec,
    emoji: EmojiProvider,
    canvas: tuple[int, int] = (1080, 1920),
) -> RenderResult:
    """Render a caption spec as a full-canvas transparent RGBA image."""

    if emoji is None:
        emoji = NoEmojiProvider()
    block, lines, used, overflow = _build_block(spec, emoji)
    image = Image.new("RGBA", canvas, (0, 0, 0, 0))
    center = spec.x_center if spec.x_center is not None else canvas[0] // 2
    band = _reference_band(spec, len(lines), used)
    bbox = _finish(image, block, center, spec.y, spec.anchor, band)
    return RenderResult(image=image, bbox=bbox, size_used=used, overflow=overflow, lines=lines)


@dataclass
class StateImage:
    """One state of an animated caption: the ink of the block, cropped, and where it goes."""

    image: Image.Image  # RGBA, the same size for every state of a caption
    x: int  # left of the image on the canvas (even, so chroma planes line up)
    y: int  # top of the image on the canvas (even)
    bbox: tuple[int, int, int, int]  # ink box of the whole caption on the canvas
    active: int | None


@dataclass
class StateSet:
    states: list[StateImage]
    size_used: int
    overflow: bool
    lines: list[str]


def _even_place(
    image: Image.Image, x: int, y: int, canvas: tuple[int, int]
) -> tuple[Image.Image, int, int]:
    """Clip ``image`` placed at ``(x, y)`` to the canvas, with an even left and top."""

    left, top = max(0, x), max(0, y)
    right, bottom = min(canvas[0], x + image.width), min(canvas[1], y + image.height)
    if right <= left or bottom <= top:
        return Image.new("RGBA", (2, 2)), 0, 0
    clipped = image.crop((left - x, top - y, right - x, bottom - y))
    pad_x, pad_y = left % 2, top % 2
    if pad_x or pad_y:
        padded = Image.new("RGBA", (clipped.width + pad_x, clipped.height + pad_y), (0, 0, 0, 0))
        padded.paste(clipped, (pad_x, pad_y))
        clipped, left, top = padded, left - pad_x, top - pad_y
    return clipped, left, top


def scale_state(state: StateImage, scale: float, canvas: tuple[int, int]) -> StateImage:
    """``state`` scaled about its centre (a frame of the pop-in of a caption)."""

    if abs(scale - 1.0) < 0.005:  # too close to 1 to see, and resampling would only blur it
        return state
    width, height = state.image.size
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    resized = state.image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    x = round(state.x + width / 2 - new_width / 2)
    y = round(state.y + height / 2 - new_height / 2)
    image, left, top = _even_place(resized, x, y, canvas)
    return StateImage(image, left, top, state.bbox, state.active)


def place_block(
    block: Image.Image, spec: TextSpec, canvas: tuple[int, int]
) -> tuple[Image.Image, int, int]:
    """Place an already cropped block like ``render`` does: centred on ``x_center``, at ``y``."""

    center = spec.x_center if spec.x_center is not None else canvas[0] // 2
    x = round(center - block.width / 2)
    top = spec.y if spec.anchor == "top" else spec.y - block.height
    return _even_place(block, x, top, canvas)


def render_states(
    spec: TextSpec,
    emoji: EmojiProvider,
    actives: list[int | None],
    canvas: tuple[int, int] = (1080, 1920),
) -> StateSet:
    """Render the states of an animated caption, one per entry of ``actives`` (the lit word).

    The layout does not depend on the lit word, and every state is cropped to the union of the
    ink of all of them, so the text stays exactly where it is from one state to the next.
    """

    if emoji is None:
        emoji = NoEmojiProvider()
    blocks: list[Image.Image] = []
    meta: tuple[list[str], int, bool] = ([], 0, False)
    for active in actives:
        state_spec = spec.model_copy(update={"active": active})
        block, lines, used, overflow = _build_block(state_spec, emoji)
        blocks.append(block)
        meta = (lines, used, overflow)
    boxes = [block.getchannel("A").getbbox() or (0, 0, 1, 1) for block in blocks]
    union = (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )
    # Same vertical band as every other chunk of the caption (see ``_reference_band``).
    band = _reference_band(spec, len(meta[0]), meta[1])
    if band is not None:
        union = (union[0], min(union[1], band[0]), union[2], max(union[3], band[1]))
    states: list[StateImage] = []
    for active, block in zip(actives, blocks, strict=True):
        image, x, y = place_block(block.crop(union), spec, canvas)
        states.append(StateImage(image, x, y, (0, 0, 0, 0), active))
        block.close()
    ink = [
        (x + box[0], y + box[1], x + box[2], y + box[3])
        for state in states
        for x, y, box in [(state.x, state.y, state.image.getchannel("A").getbbox() or (0, 0, 0, 0))]
    ]
    whole = (
        min(box[0] for box in ink),
        min(box[1] for box in ink),
        max(box[2] for box in ink),
        max(box[3] for box in ink),
    )
    for state in states:
        state.bbox = whole
    return StateSet(states, meta[1], meta[2], meta[0])
