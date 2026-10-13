"""Storyboard v1 models and loading helpers."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from PIL import ImageColor
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from adanvil.media import (
    ClipSegment,
    Format,
    ImageSegment,
    RevealSegment,
    SplitSegment,
    segment_duration,
)
from adanvil.project import resolve_in_project
from adanvil.speech.corrections import check_corrections
from adanvil.text import TextSpec
from adanvil.text.animate import WORD_SIZE, marked_entries, spread_times, strip_markers
from adanvil.text.chunking import chunk_ranges, match_emphasis
from adanvil.text.fonts import load_font
from adanvil.text.layout import balance_lines, wrap
from adanvil.text.styles import default_size, spoken_words
from adanvil.vocab import TYPE_VALUES, normalize_shot_type

if TYPE_CHECKING:
    from adanvil.voice import VoiceTrack

_CAPTION_STYLES = Literal["pill", "meme", "outline", "outline_soft", "box", "label"]
_NAME_PATTERN = r"^[A-Za-z0-9_.-]{1,80}$"

Segment = Annotated[
    ClipSegment | ImageSegment | SplitSegment | RevealSegment, Field(discriminator="type")
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


FormatRatio = Literal["9:16", "4:5", "1:1"]


class FormatOverride(_StrictModel):
    """What one overlay changes in ONE output format of ``render_formats``/``render_batch``.

    Applied after the automatic re-layout of that format. ``x`` is the horizontal position
    (``x_center`` for a caption, a comment or generated captions, the top-left corner of a
    sticker, an image or an inset; ``"center"`` for a sticker), ``y``, ``size`` and ``anchor``
    are the overlay's own fields, ``hidden`` removes the overlay from that format.
    """

    x: int | Literal["center"] | None = None
    y: int | None = None
    size: int | None = Field(default=None, gt=0)
    anchor: Literal["top", "bottom", "center"] | None = None
    hidden: bool | None = None


class _OverlayBase(_StrictModel):
    """What every placed overlay shares: the optional per-format overrides (``formats``)."""

    formats: dict[FormatRatio, FormatOverride] | None = None

    @model_serializer(mode="wrap")
    def _dump_format_overrides(self, handler: Any) -> Any:
        data = handler(self)
        if isinstance(data, dict) and data.get("formats") is None:
            data.pop("formats", None)  # nothing to say: dumps of older storyboards are unchanged
        return data


Animation = Literal["none", "karaoke", "pop", "word"]


class _AnimatedCaption(_OverlayBase):
    """The look shared by the captions that can be animated or have emphasised words.

    ``animation``: ``none`` (static, the default), ``karaoke`` (the whole chunk shown, the word
    being spoken lit), ``word`` (one or two big words at a time, the spoken one lit) or ``pop``
    (the chunk scales in over 120 ms). The lit word uses ``highlight`` (a colour; default the
    brand's ``visual.highlight_color``, else ``#FFD400``), may be drawn ``highlight_scale``
    times larger and sit on a rounded ``highlight_box`` colour. ``bounce`` gives the ``pop``
    an overshoot. Words of ``emphasis`` (case and accent insensitive) are always drawn in the
    highlight colour.
    """

    animation: Animation = "none"
    highlight: str | None = None
    highlight_scale: float = Field(default=1.0, ge=0.5, le=2.0)
    highlight_box: str | None = None
    bounce: bool = False
    emphasis: list[str] = Field(default_factory=list)

    @field_validator("highlight", "highlight_box")
    @classmethod
    def _valid_colour(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                ImageColor.getcolor(value, "RGBA")
            except ValueError as exc:
                raise ValueError(f"invalid colour {value!r}") from exc
        return value

    def _animation_fields(self) -> dict[str, Any]:
        return {
            "animation": self.animation,
            "highlight": self.highlight,
            "highlight_scale": self.highlight_scale,
            "highlight_box": self.highlight_box,
            "bounce": self.bounce,
            "emphasis": list(self.emphasis),
        }


def word_size(size: int | None, style: str, animation: str) -> int | None:
    """``size`` as drawn: ``word`` captions are 30% larger than the plain size of the style."""

    if animation != "word":
        return size
    return round((size if size is not None else default_size(style)) * WORD_SIZE)


class CaptionOverlay(_AnimatedCaption):
    """One of the six non-comment text styles.

    With ``animation`` the chunk is animated (see ``_AnimatedCaption``); ``karaoke`` and ``word``
    time its words evenly over ``start`` to ``end`` unless ``word_times`` gives each spoken
    word's ``[start, end]`` on the timeline (the voice and speech captions fill it in).
    ``*word*`` in ``lines`` marks a word as emphasised.
    """

    type: Literal["caption"] = "caption"
    style: _CAPTION_STYLES
    lines: list[str] = Field(min_length=1)
    start: float = 0.0
    end: float | None = None
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
    word_times: list[tuple[float, float]] | None = None

    @model_validator(mode="after")
    def validate_timing(self) -> CaptionOverlay:
        _validate_overlay_timing(self.start, self.end)
        return self

    @model_validator(mode="after")
    def read_emphasis_markers(self) -> CaptionOverlay:
        """Turn ``*word*`` markers of the lines into ``emphasis`` words."""

        if not any("*" in line for line in self.lines):
            return self
        text, marked = strip_markers(" ".join(self.lines))
        if not marked:
            return self
        tokens = text.split()
        self.emphasis = [*self.emphasis, *marked_entries(tokens, marked)]
        # Take the markers off line by line, token by token, keeping the line breaks.
        cursor = 0
        lines: list[str] = []
        for line in self.lines:
            count = len(line.split())
            lines.append(" ".join(tokens[cursor : cursor + count]))
            cursor += count
        self.lines = lines
        return self

    @property
    def animated(self) -> bool:
        return self.animation != "none"

    def unmatched_emphasis(self) -> list[str]:
        """The ``emphasis`` entries that match no word of the lines (nothing would be lit)."""

        return match_emphasis(spoken_words(self.lines), self.emphasis)[1]

    def expand_words(self, end: float | None = None) -> list[CaptionOverlay]:
        """The chunks of a plain ``word`` caption: one or two words each, timed evenly.

        ``end`` is the end of the caption when it has none. Any other caption, and a caption
        that already carries ``word_times`` (a generated chunk), comes back as it is.
        """

        last = self.end if self.end is not None else end
        if self.animation != "word" or self.word_times is not None:
            return [self]
        if last is None or last <= self.start:
            return [self]
        words = spoken_words(self.lines)
        if not words:
            return [self]
        times = spread_times(words, self.start, last)
        chunks: list[CaptionOverlay] = []
        for first, stop in chunk_ranges(words, 2, pair_short=True):
            finish = times[stop][0] if stop < len(words) else last
            chunks.append(
                self.model_copy(
                    update={
                        "lines": [" ".join(words[first:stop])],
                        "start": times[first][0],
                        "end": finish,
                        "word_times": times[first:stop],
                    }
                )
            )
        return chunks

    def to_text_spec(self, highlight: str | None = None, active: int | None = None) -> TextSpec:
        """The text spec of this caption, with ``active`` the lit word (``highlight`` is the
        colour to use when the overlay has none: the brand accent)."""

        lit = self.animation in {"karaoke", "word"}
        colour = self.highlight or (None if self.highlight_box and lit else highlight)
        return TextSpec(
            style=self.style,
            lines=self.lines,
            y=self.y,
            anchor=self.anchor,
            x_center=self.x_center,
            size=word_size(self.size, self.style, self.animation),
            family=self.family,
            weight=self.weight,
            fg=self.fg,
            stroke=self.stroke,
            box=self.box,
            accent=self.accent,
            max_width=self.max_width,
            uniform_width=self.uniform_width,
            highlight=colour,
            highlight_scale=self.highlight_scale if lit else 1.0,
            highlight_box=self.highlight_box if lit else None,
            active=active if lit else None,
            emphasis=list(self.emphasis),
        )


class CommentOverlay(_OverlayBase):
    type: Literal["comment"] = "comment"
    start: float = 0.0
    end: float | None = None
    look: Literal["facebook", "tiktok"] = "facebook"
    text: str
    header: str | None = None
    meta: str = "2w   Like   Reply"
    reactions: int | None = None
    y: int = 330
    x_center: int | None = None
    size: int | None = None

    @model_validator(mode="after")
    def validate_timing(self) -> CommentOverlay:
        _validate_overlay_timing(self.start, self.end)
        return self

    def to_text_spec(self) -> TextSpec:
        return TextSpec(
            style="comment",
            lines=[self.text],
            look=self.look,
            header=self.header,
            meta=self.meta,
            reactions=self.reactions,
            y=self.y,
            x_center=self.x_center,
            size=self.size,
        )


IMAGE_ANIMATIONS = Literal["none", "pop", "slide_left", "slide_right", "slide_up", "fade"]


class ImageFrame(_StrictModel):
    """The look of a picture card: rounded corners (``radius`` pixels), a ``border_width`` line
    of ``border_color`` drawn inside the card's footprint, and a soft ``shadow``."""

    radius: int = Field(default=0, ge=0, le=400)
    border_width: int = Field(default=0, ge=0, le=60)
    border_color: str = "#FFFFFF"
    shadow: bool = False

    @field_validator("border_color")
    @classmethod
    def _colour(cls, value: str) -> str:
        try:
            ImageColor.getcolor(value, "RGBA")
        except ValueError as exc:
            raise ValueError(f"invalid colour {value!r}") from exc
        return value


class ImageOverlay(_OverlayBase):
    """A picture laid over the video: a badge, an inset, a before/after photo, a review.

    ``x``, ``y`` (top-left) and ``width`` place it. Left out, ``animation``, ``frame`` and
    ``rotate`` leave it a plain picture. With them it is a card:

    - ``animation`` is how it comes in, over 250 ms: ``pop`` (scales in with a small overshoot),
      ``slide_left`` (comes in from the right, moving left), ``slide_right`` (from the left),
      ``slide_up`` (from below) or ``fade``. ``exit`` plays the same movement backwards over
      the last 250 ms.
    - ``frame`` rounds the corners, adds a border and a shadow. ``x``, ``y`` and ``width`` then
      describe the whole card, border included.
    - ``rotate`` turns the card by a small angle in degrees (positive is clockwise) around its
      centre, for a photo-card look.
    """

    type: Literal["image"] = "image"
    source: str
    start: float = 0.0
    end: float | None = None
    x: int
    y: int
    width: int | None = Field(default=None, gt=0)
    animation: IMAGE_ANIMATIONS = "none"
    exit: bool = False
    frame: ImageFrame | None = None
    rotate: float = Field(default=0.0, ge=-30.0, le=30.0)

    @model_validator(mode="after")
    def validate_timing(self) -> ImageOverlay:
        _validate_overlay_timing(self.start, self.end)
        if self.exit and self.animation == "none":
            raise ValueError(
                "exit needs an animation (pop, slide_left, slide_right, slide_up, fade)"
            )
        return self

    @property
    def plain(self) -> bool:
        """A plain picture: no animation, no frame, no rotation."""

        return self.animation == "none" and self.frame is None and self.rotate == 0.0


def _valid_colour(value: str | None) -> str | None:
    if value is not None:
        try:
            ImageColor.getcolor(value, "RGBA")
        except ValueError as exc:
            raise ValueError(f"invalid colour {value!r}") from exc
    return value


class InsetBorder(_StrictModel):
    """A border drawn inside the footprint of a video inset."""

    width: int = Field(default=4, ge=0, le=40)
    color: str = "#FFFFFF"

    @field_validator("color")
    @classmethod
    def _colour(cls, value: str) -> str:
        return _valid_colour(value) or value


class VideoOverlay(_OverlayBase):
    """A clip played inside the frame on top of the segments (picture-in-picture).

    ``source`` plays from ``in`` during ``start`` to ``end`` of the final timeline (without
    ``end``, for as long as the clip lasts from ``in``, up to the end of the ad; a clip that is
    shorter than the span holds its last frame). ``x`` and ``y`` are the top-left corner and
    ``width`` the width of the whole inset, border included; ``height`` is optional and, left
    out, follows the clip's aspect (a ``circle`` is always square). ``shape`` is ``rect``,
    ``rounded`` (corner ``radius``, default 12% of the short side) or ``circle``. ``border``
    and ``shadow`` frame it. The inset is silent unless ``audio`` is true, which needs
    ``audio.clips`` and replaces the sound of the bed under it (validation refuses it when a
    voice-over or another voice is active).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: Literal["video"] = "video"
    source: str
    in_: float = Field(default=0.0, alias="in", ge=0)
    start: float = 0.0
    end: float | None = None
    x: int
    y: int
    width: int = Field(gt=0)
    height: int | None = Field(default=None, gt=0)
    shape: Literal["rect", "rounded", "circle"] = "rect"
    radius: int | None = Field(default=None, ge=0)
    border: InsetBorder | None = None
    shadow: bool = False
    audio: bool = False

    @model_validator(mode="after")
    def validate_timing(self) -> VideoOverlay:
        _validate_overlay_timing(self.start, self.end)
        if self.shape == "circle" and self.height not in (None, self.width):
            raise ValueError("a circle inset is square: leave height out or make it equal to width")
        return self


STICKER_KINDS = Literal["arrow", "circle", "underline", "badge", "emoji", "image"]
_ARROW_DIRECTIONS = Literal[
    "up", "down", "left", "right", "up_left", "up_right", "down_left", "down_right"
]


class StickerOverlay(_OverlayBase):
    """A vector arrow, a hand-drawn circle or underline, a badge, an emoji or an image sticker.

    Positions are on the final frame, in pixels:

    - ``arrow``: from ``from`` to ``to`` (the tip), or the tip at ``x``, ``y`` pointing
      ``direction`` with ``length`` (default 220); ``stroke`` is the line thickness.
    - ``circle``: centred on ``x``, ``y``, ``width`` by ``height`` (default 0.62 x width).
    - ``underline``: starts at ``x``, ``y`` and runs ``width`` to the right.
    - ``badge``: a rounded rectangle with ``text`` (``size`` 44 by default, ``fg`` auto) whose
      top-left corner is ``x``, ``y``.

    ``x`` may be ``"center"`` (or give ``anchor: "center"``, same thing): the sticker is
    centred across the frame, whatever its width (any kind except ``image`` without ``width``;
    for an arrow, a circle: the tip or the centre is put at the middle of the frame).
    - ``emoji``: the character ``emoji`` at ``size`` (default 120), top-left at ``x``, ``y``.
    - ``image``: the project file ``source`` at ``x``, ``y`` (top-left), ``width`` optional.

    ``color`` defaults to the brand highlight colour (else yellow); a ``badge`` defaults to the
    brand's ``visual.accent_color`` (else a neutral dark ``#111827``) with white or dark text.
    ``animation``: ``none``; ``pop`` (a quick overshooting scale-in, any kind); ``draw`` (the
    arrow, circle or underline is drawn over 300 ms).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: Literal["sticker"] = "sticker"
    kind: STICKER_KINDS
    start: float = 0.0
    end: float | None = None
    animation: Literal["none", "pop", "draw"] = "none"
    color: str | None = None
    x: int | Literal["center"] | None = None
    anchor: Literal["center"] | None = None
    y: int | None = None
    from_: tuple[int, int] | None = Field(default=None, alias="from")
    to: tuple[int, int] | None = None
    direction: _ARROW_DIRECTIONS | None = None
    length: int = Field(default=220, gt=0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    stroke: int = Field(default=10, ge=2, le=60)
    text: str | None = None
    size: int | None = Field(default=None, gt=0)
    fg: str | None = None
    emoji: str | None = None
    source: str | None = None

    @field_validator("color", "fg")
    @classmethod
    def _colour(cls, value: str | None) -> str | None:
        return _valid_colour(value)

    @model_validator(mode="after")
    def validate_kind(self) -> StickerOverlay:
        _validate_overlay_timing(self.start, self.end)
        kind = self.kind
        if self.anchor == "center" and self.x is None:
            self.x = "center"
        if self.anchor == "center" and self.x != "center":
            raise ValueError('anchor "center" centres the sticker: leave x out or use "center"')
        placed = self.x is not None and self.y is not None
        if kind == "arrow":
            by_points = self.from_ is not None and self.to is not None
            by_direction = placed and self.direction is not None
            if not by_points and not by_direction:
                raise ValueError(
                    "an arrow sticker needs from and to, or x, y (the tip) and direction"
                )
            if by_points and self.from_ == self.to:
                raise ValueError("an arrow sticker needs two different points (from and to)")
        elif not placed:
            raise ValueError(f"a {kind} sticker needs x and y")
        if kind in {"circle", "underline"} and self.width is None:
            raise ValueError(f"a {kind} sticker needs width")
        if kind == "badge" and not (self.text or "").strip():
            raise ValueError("a badge sticker needs text")
        if kind == "emoji" and not (self.emoji or "").strip():
            raise ValueError("an emoji sticker needs emoji")
        if kind == "image" and not self.source:
            raise ValueError("an image sticker needs source")
        if self.x == "center" and kind == "image" and self.width is None:
            raise ValueError('a centred image sticker needs width (x: "center")')
        if self.animation == "draw" and kind not in {"arrow", "circle", "underline"}:
            raise ValueError("animation draw only works for an arrow, a circle or an underline")
        return self


class ProgressOverlay(_StrictModel):
    """A bar filling from left to right across the ad (or from ``start`` to ``end``).

    ``position`` is ``top`` or ``bottom`` edge of the frame, ``height`` its thickness in pixels,
    ``color`` the fill (default the brand accent, else white) and ``background`` an optional
    track drawn under it for the same span. The bar is full on the last frame of the span.
    """

    type: Literal["progress"] = "progress"
    position: Literal["top", "bottom"] = "bottom"
    height: int = Field(default=8, ge=1, le=80)
    color: str | None = None
    background: str | None = None
    start: float = 0.0
    end: float | None = None

    @field_validator("color", "background")
    @classmethod
    def _colour(cls, value: str | None) -> str | None:
        return _valid_colour(value)

    @model_validator(mode="after")
    def validate_timing(self) -> ProgressOverlay:
        _validate_overlay_timing(self.start, self.end)
        return self


_VOICE_CAPTION_DEFAULTS = {
    "pill": ("plus-jakarta-sans", 700),
    "meme": ("plus-jakarta-sans", 700),
    "outline": ("tiktok-sans", 800),
    "outline_soft": ("tiktok-sans", 600),
    "box": ("tiktok-sans", 700),
    "label": ("poppins", 600),
}


def _caption_lines(
    text: str, style: str, family: str | None, weight: int | None, size: int, max_width: int
) -> list[str]:
    """Wrap ``text`` for a generated caption: one line, or two balanced lines at most."""

    default_family, default_weight = _VOICE_CAPTION_DEFAULTS[style]
    font = load_font(family or default_family, weight or default_weight, size)
    wrapped = wrap(text, font, max_width)
    if len(wrapped) <= 1:
        return wrapped
    return balance_lines(text, min(2, len(wrapped)), font)


class _GeneratedCaptions(_AnimatedCaption):
    """What ``voice_captions`` and ``speech_captions`` share: how a chunk of words is cut, styled
    and turned into an ordinary caption overlay (with its word timings when animated)."""

    mode: Literal["sentence", "chunks"] = "sentence"
    max_words: int = Field(default=4, gt=0)
    style: _CAPTION_STYLES = "outline"
    y: int = 1300
    anchor: Literal["top", "bottom"] = "top"
    size: int = Field(default=58, gt=0)
    family: str | None = None
    weight: int | None = None
    fg: str | None = None
    stroke: str | None = None
    box: str | None = None
    max_width: int = Field(default=900, gt=0)
    case: Literal["as_is", "upper"] = "as_is"  # "upper" for punchy word-by-word captions

    @property
    def lit_words(self) -> bool:
        """True when the captions need the timing of each word (``karaoke`` and ``word``)."""

        return self.animation in {"karaoke", "word"}

    @property
    def chunked(self) -> bool:
        """The ``word`` animation always shows one or two words at a time."""

        return self.mode == "chunks" or self.animation == "word"

    def chunk_ranges(self, words: list[str]) -> list[tuple[int, int]]:
        """Index ranges cutting ``words`` into chunks (see ``text.chunking``)."""

        if self.animation == "word":
            return chunk_ranges(words, min(self.max_words, 2), pair_short=True)
        return chunk_ranges(words, self.max_words)

    def _caption(
        self,
        text: str,
        start: float,
        end: float,
        word_times: list[tuple[float, float]] | None = None,
        emphasis: list[str] | None = None,
    ) -> CaptionOverlay:
        if self.case == "upper":
            text = text.upper()
        lines = _caption_lines(
            text,
            self.style,
            self.family,
            self.weight,
            word_size(self.size, self.style, self.animation) or self.size,
            self.max_width,
        )
        fields = self._animation_fields()
        fields["emphasis"] = [*self.emphasis, *(emphasis or [])]
        return CaptionOverlay(
            style=self.style,
            lines=lines,
            start=start,
            end=end,
            y=self.y,
            anchor=self.anchor,
            size=self.size,
            family=self.family,
            weight=self.weight,
            fg=self.fg,
            stroke=self.stroke,
            box=self.box,
            max_width=self.max_width,
            word_times=word_times if self.lit_words else None,
            **fields,
        )


class VoiceCaptionsOverlay(_GeneratedCaptions):
    """Generate ordinary caption overlays from a voice-track manifest.

    ``suspect`` lines are still usable. An agent cannot listen to a generated
    line, so it should regenerate with slightly different wording or accept it
    when the transcript only differs by punctuation. A ``*word*`` marker in the shown
    text of a line emphasises that word.
    """

    type: Literal["voice_captions"] = "voice_captions"
    manifest: str | None = None
    lines: list[int] | None = None
    offset: float | None = Field(default=None, ge=0)

    def expand(self, track: VoiceTrack, offset: float = 0.0) -> list[CaptionOverlay]:
        """Expand the selected voice lines into timeline caption overlays."""

        offset = float(offset)
        if not math.isfinite(offset) or offset < 0:
            raise ValueError("voice caption offset must be finite and non-negative")
        selected = set(self.lines) if self.lines is not None else None
        expanded: list[CaptionOverlay] = []
        for line in track.lines:
            if selected is not None and line.index not in selected:
                continue
            line_start = float(line.start) + offset
            line_end = line_start + float(line.duration)
            shown, marked_tokens = strip_markers(line.display or line.text)
            tokens = shown.split()
            if not self.chunked:
                times: list[tuple[float, float]] | None = None
                if self.lit_words:
                    words = _script_words(shown, list(line.words))
                    times = [(line_start + word[0], line_start + word[1]) for word in words]
                expanded.append(
                    self._caption(
                        shown,
                        line_start,
                        line_end,
                        times,
                        marked_entries(tokens, marked_tokens),
                    )
                )
                continue

            words, marked = _script_words_marked(shown, list(line.words), marked_tokens)
            if not words:
                expanded.append(self._caption(shown, line_start, line_end))
                continue
            for first, last in self.chunk_ranges([word[2] for word in words]):
                group = words[first:last]
                start = line_start + float(group[0][0])
                next_start = float(words[last][0]) if last < len(words) else float(line.duration)
                end = line_start + min(float(line.duration), next_start)
                if end <= start:
                    continue
                expanded.append(
                    self._caption(
                        " ".join(word[2] for word in group),
                        start,
                        end,
                        [(line_start + word[0], line_start + word[1]) for word in group],
                        [words[index][2] for index in sorted(marked) if first <= index < last],
                    )
                )
        return expanded


_JOINERS = ("-", "'", "’")


def _script_words(text: str, aligned: list) -> list[tuple[float, float, str]]:
    """Time the words of the script (what the agent wrote) with the aligner's timings.

    The transcript is only used for timing: it may spell numbers as digits or pick a
    homophone, and it splits elisions and compounds ("l'", "-clés"), which would
    otherwise start a caption chunk with a stray fragment.
    """

    return _script_words_marked(text, aligned, set())[0]


def _script_words_marked(
    text: str, aligned: list, marked: set[int]
) -> tuple[list[tuple[float, float, str]], set[int]]:
    """``_script_words`` and the indexes of the words that hold a marked token of ``text``."""

    merged: list[list] = []
    for start, end, word, *_ in aligned:
        token = str(word).strip()
        if not token:
            continue
        if merged and (token.startswith(_JOINERS) or merged[-1][2].endswith(_JOINERS)):
            merged[-1][1] = float(end)
            merged[-1][2] += token
        else:
            merged.append([float(start), float(end), token])
    script: list[str] = []
    marked_words: set[int] = set()
    pending_open = ""
    for position, token in enumerate(text.split()):
        punctuation_only = all(not ch.isalnum() for ch in token)
        # Opening marks (« “ ( ¿ ¡) belong to the next word, closing ones (: ; ! ? » …)
        # to the previous one, so no caption chunk starts or ends on a lone mark.
        if punctuation_only and token[-1] in "«“‘(¿¡[":
            pending_open += token + " "
            if position in marked:
                marked_words.add(len(script))
        elif punctuation_only and script:
            script[-1] += " " + token
            if position in marked:
                marked_words.add(len(script) - 1)
        else:
            script.append(pending_open + token)
            pending_open = ""
            if position in marked:
                marked_words.add(len(script) - 1)
    if pending_open and script:
        script[-1] += " " + pending_open.strip()
    if not merged or not script:
        return [(m[0], m[1], m[2]) for m in merged], set()
    if len(script) == len(merged):
        return [(m[0], m[1], word) for m, word in zip(merged, script, strict=True)], marked_words
    # Different word counts (e.g. "49" vs "quarante-neuf"): spread the script words
    # over the spoken span in proportion to their length.
    timed = [
        (begin, finish, word)
        for (begin, finish), word in zip(
            spread_times(script, merged[0][0], merged[-1][1]), script, strict=True
        )
    ]
    return timed, marked_words


class SpeechCaptionsOverlay(_GeneratedCaptions):
    """Captions built from the speech of the clips on the timeline (needs ``audio.clips``).

    For every clip segment (and every ``audio_from`` audio) the words whose middle falls
    inside the played range are taken, placed on the timeline (speed aware) and shown as
    ``sentence`` or ``chunks`` captions, like ``voice_captions``. Hesitations (euh, um...)
    are left out unless ``keep_fillers`` is true. The words come from the speech index of
    the library (``library_index``), so every source must have been indexed with speech.

    ``corrections`` fixes a misheard word: ``{"heard": "shown"}`` is applied word by word,
    ignoring case, keeping the punctuation. Each correction must be close to the heard word
    (character similarity of at least 0.5), else it is an error that asks for a new
    transcription (``speech_refine``); ``storyboard_validate`` lists every correction as a
    ``speech_corrections`` warning for a human to review.
    """

    type: Literal["speech_captions"] = "speech_captions"
    keep_fillers: bool = False
    corrections: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_corrections(self) -> SpeechCaptionsOverlay:
        problems = check_corrections(self.corrections)
        if problems:
            raise ValueError("speech_corrections: " + "; ".join(problems))
        return self

    def caption(
        self,
        text: str,
        start: float,
        end: float,
        word_times: list[tuple[float, float]] | None = None,
    ) -> CaptionOverlay:
        """One ordinary caption overlay for ``text`` shown from ``start`` to ``end``."""

        return self._caption(text, start, end, word_times)


class SfxOverlay(_StrictModel):
    """A sound effect laid over the audio bed at ``at`` seconds of the final timeline.

    ``source`` is a file of the project (``sfx/whoosh_short.wav``) or a bare name of the
    ``sfx/`` folder (``whoosh_short``). ``volume`` scales the sound, which the pack brings to a
    fixed level (about 6 dB under a -15 LUFS voice at 1.0, so 0.5 sits under speech). It is
    mixed after the bed is normalised, never clips and never runs past the end of the ad.
    """

    type: Literal["sfx"] = "sfx"
    source: str
    at: float = Field(ge=0)
    volume: float = Field(default=0.8, ge=0, le=2.0)

    @field_validator("source")
    @classmethod
    def _valid_source(cls, value: str) -> str:
        return _sfx_name(value, "sfx.source")

    @field_validator("at")
    @classmethod
    def _finite_at(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("sfx.at must be finite")
        return value


class SfxAuto(_StrictModel):
    """Sound effects placed for you: ``cuts`` at each cut between segments, ``captions`` at the
    start of each written caption (a sound name of ``sfx/``, or ``None`` for none).

    ``every`` keeps one cut in N (1 is all of them, 2 every second cut), ``min_gap`` skips a
    sound that would fall within that many seconds of the one before (cut sounds first, then
    caption sounds). The sounds are expanded to explicit ``sfx`` overlays, listed in the
    validation summary (``summary.sfx``) and in the ``storyboard_save`` result.
    """

    cuts: str | None = None
    captions: str | None = None
    every: int = Field(default=1, ge=1)
    min_gap: float = Field(default=0.6, ge=0)
    volume: float = Field(default=0.5, ge=0, le=2.0)

    @field_validator("cuts", "captions")
    @classmethod
    def _valid_name(cls, value: str | None) -> str | None:
        return None if value is None else _sfx_name(value, "sfx_auto")

    @field_validator("min_gap")
    @classmethod
    def _finite_gap(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("sfx_auto.min_gap must be finite")
        return value


def _sfx_name(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must name a sound (a file of sfx/, for example whoosh_short)"
        )
    try:
        resolve_in_project(Path.cwd(), value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name}: {exc}") from exc
    return value


Overlay = Annotated[
    CaptionOverlay
    | CommentOverlay
    | ImageOverlay
    | VoiceCaptionsOverlay
    | SpeechCaptionsOverlay
    | SfxOverlay
    | VideoOverlay
    | StickerOverlay
    | ProgressOverlay,
    Field(discriminator="type"),
]


class AudioSpec(_StrictModel):
    music: str | None = None
    # The feel wanted of the music (``music_search(fit_to="storyboard")`` reads it).
    music_feel: Literal["calm", "medium", "energetic"] | None = None
    from_reference: str | None = None
    start: float = Field(default=0.0, ge=0)
    volume: float = Field(default=0.8, ge=0)
    fade_in: float = Field(default=0.2, ge=0)
    fade_out: float = Field(default=1.0, ge=0)
    # Integrated loudness target of the final bed (LUFS), applied in a separate pass;
    # None keeps the source level. Feeds normalise around -14 LUFS. It does not
    # apply to the voice mix when ``voice`` is set.
    loudness: float | None = Field(default=-14.0, ge=-40, le=-6)
    voice: str | None = None
    voice_offset: float = Field(default=0.0, ge=0)
    music_volume: float = Field(default=0.35, ge=0.2, le=0.6)
    voice_loudness: float = -15.0
    # True: the bed is the original sound of the clip segments (see ``audio_from``), brought
    # to ``voice_loudness`` on its own, with ``music`` under it at ``music_volume``.
    clips: bool = False
    # Sounds placed at the cuts and at the captions (see ``SfxAuto``).
    sfx_auto: SfxAuto | None = None

    @model_validator(mode="after")
    def validate_source_choice(self) -> AudioSpec:
        if self.music is not None and self.from_reference is not None:
            raise ValueError("audio.music and audio.from_reference are mutually exclusive")
        if self.clips and self.voice is not None:
            raise ValueError("audio.clips and audio.voice are mutually exclusive")
        if self.clips and self.from_reference is not None:
            raise ValueError("audio.clips and audio.from_reference are mutually exclusive")
        if self.voice is not None:
            _validate_relative_path(self.voice, "audio.voice")
        return self


def normalise_quality(value: Any) -> Any:
    """``draft`` is an alias of ``preview`` wherever a render quality is taken."""

    return "preview" if isinstance(value, str) and value.strip().casefold() == "draft" else value


class OutputSpec(_StrictModel):
    path: str = "renders/{name}.mp4"
    quality: Literal["preview", "final"] = "final"

    @field_validator("quality", mode="before")
    @classmethod
    def _quality_alias(cls, value: Any) -> Any:
        return normalise_quality(value)


def _validate_overlay_timing(start: float, end: float | None) -> None:
    if not math.isfinite(start) or start < 0:
        raise ValueError("overlay.start must be finite and non-negative")
    if end is not None and (not math.isfinite(end) or end <= start):
        raise ValueError("overlay.end must be greater than overlay.start")


def _reject_unknown(mapping: object, allowed: set[str], where: str) -> None:
    if isinstance(mapping, Mapping):
        unknown = sorted(set(mapping) - allowed)
        if unknown:
            raise ValueError(f"{where} has unexpected field(s): {', '.join(unknown)}")


def _validate_nested_fields(values: object) -> None:
    if not isinstance(values, Mapping):
        return
    _reject_unknown(
        values,
        {
            "version",
            "name",
            "recipe",
            "reference",
            "gaps",
            "format",
            "formats",
            "emoji",
            "segments",
            "overlays",
            "audio",
            "output",
        },
        "storyboard",
    )
    _reject_unknown(values.get("format"), {"width", "height", "fps"}, "storyboard.format")
    _reject_unknown(
        values.get("audio"),
        {
            "music",
            "music_feel",
            "from_reference",
            "start",
            "volume",
            "fade_in",
            "fade_out",
            "loudness",
            "voice",
            "voice_offset",
            "music_volume",
            "voice_loudness",
            "clips",
            "sfx_auto",
        },
        "storyboard.audio",
    )
    _reject_unknown(
        (values.get("audio") or {}).get("sfx_auto") if isinstance(values.get("audio"), Mapping)
        else None,
        {"cuts", "captions", "every", "min_gap", "volume"},
        "storyboard.audio.sfx_auto",
    )
    _reject_unknown(values.get("output"), {"path", "quality"}, "storyboard.output")

    for index, segment in enumerate(values.get("segments") or []):
        if not isinstance(segment, Mapping):
            continue
        kind = segment.get("type")
        if kind == "clip":
            _reject_unknown(
                segment,
                {
                    "type",
                    "source",
                    "in",
                    "out",
                    "crop",
                    "focus",
                    "face_y",
                    "speed",
                    "gamma",
                    "zoom",
                    "freeze",
                    "remove_text",
                    "label",
                    "shot_type",
                    "scale",
                    "subject",
                    "emotion",
                    "role",
                    "audio_from",
                    "punch",
                },
                f"storyboard.segments[{index}]",
            )
            for punch in segment.get("punch") or []:
                _reject_unknown(punch, {"at", "zoom", "hold", "center"}, "segment.punch")
            _reject_unknown(segment.get("audio_from"), {"source", "in"}, "segment.audio_from")
            _reject_unknown(segment.get("crop"), {"x", "y", "w", "h"}, "segment.crop")
            _reject_unknown(segment.get("zoom"), {"from", "to", "focus"}, "segment.zoom")
            _reject_unknown(segment.get("freeze"), {"hold", "zoom", "center"}, "segment.freeze")
            _reject_unknown(
                segment.get("remove_text"),
                {"method", "box", "extra_boxes", "threshold", "dilate", "blur"},
                "segment.remove_text",
            )
        elif kind == "split":
            _reject_unknown(
                segment,
                {
                    "type",
                    "layout",
                    "duration",
                    "panels",
                    "gap",
                    "gap_color",
                    "labels",
                    "label_style",
                    "label",
                    "shot_type",
                    "scale",
                    "subject",
                    "emotion",
                    "role",
                },
                f"storyboard.segments[{index}]",
            )
            for number, panel in enumerate(segment.get("panels") or []):
                where = f"storyboard.segments[{index}].panels[{number}]"
                _reject_unknown(
                    panel,
                    {"source", "in", "focus", "face_y", "crop", "speed", "gamma", "freeze", "fit"},
                    where,
                )
                if isinstance(panel, Mapping):
                    _reject_unknown(panel.get("crop"), {"x", "y", "w", "h"}, f"{where}.crop")
                    _reject_unknown(
                        panel.get("freeze"), {"hold", "zoom", "center"}, f"{where}.freeze"
                    )
        elif kind == "reveal":
            _reject_unknown(
                segment,
                {
                    "type",
                    "duration",
                    "before",
                    "after",
                    "direction",
                    "hold_before",
                    "sweep",
                    "back_and_forth",
                    "focus",
                    "face_y",
                    "align_faces",
                    "labels",
                    "label_style",
                    "divider",
                    "motion",
                    "zoom_to",
                    "label",
                    "shot_type",
                    "scale",
                    "subject",
                    "emotion",
                    "role",
                },
                f"storyboard.segments[{index}]",
            )
            _reject_unknown(
                segment.get("divider"),
                {"width", "color", "shadow"},
                f"storyboard.segments[{index}].divider",
            )
            for side_name in ("before", "after"):
                side = segment.get(side_name)
                where = f"storyboard.segments[{index}].{side_name}"
                _reject_unknown(
                    side,
                    {"source", "in", "focus", "face_y", "crop", "speed", "gamma", "freeze", "fit"},
                    where,
                )
                if isinstance(side, Mapping):
                    _reject_unknown(side.get("crop"), {"x", "y", "w", "h"}, f"{where}.crop")
                    _reject_unknown(
                        side.get("freeze"), {"hold", "zoom", "center"}, f"{where}.freeze"
                    )
        elif kind == "image":
            _reject_unknown(
                segment,
                {
                    "type",
                    "source",
                    "duration",
                    "motion",
                    "zoom_to",
                    "background",
                    "fit",
                    "focus",
                    "face_y",
                    "crop",
                    "label",
                    "shot_type",
                    "scale",
                    "subject",
                    "emotion",
                    "role",
                },
                f"storyboard.segments[{index}]",
            )
            _reject_unknown(segment.get("crop"), {"x", "y", "w", "h"}, "segment.crop")


class Gap(_StrictModel):
    """A shot type of the reference the edit could not cover, and why."""

    type: str
    reason: str = Field(min_length=1)

    @field_validator("type", mode="before")
    @classmethod
    def rename_old_type(cls, value: Any) -> Any:
        return normalize_shot_type(value)

    @model_validator(mode="after")
    def validate_type(self) -> Gap:
        if self.type not in TYPE_VALUES:
            raise ValueError(f"gaps.type must be one of: {', '.join(TYPE_VALUES)}")
        return self


class Storyboard(_StrictModel):
    version: Literal[1] = 1
    name: str = Field(pattern=_NAME_PATTERN)
    recipe: str | None = None
    reference: str | None = None
    gaps: list[Gap] = Field(default_factory=list)
    format: Format = Field(default_factory=Format)
    # Aspect ratios ``render_formats`` / ``render_batch`` produce from this one storyboard
    # (``9:16`` Stories, Reels and TikTok, ``4:5`` and ``1:1`` Feed). ``render_start`` ignores it.
    formats: list[FormatRatio] = Field(default_factory=list)
    emoji: str = "twemoji"
    segments: list[Segment] = Field(min_length=1)
    overlays: list[Overlay] = Field(default_factory=list)
    audio: AudioSpec = Field(default_factory=AudioSpec)
    output: OutputSpec = Field(default_factory=OutputSpec)

    @model_validator(mode="before")
    @classmethod
    def validate_nested_extras(cls, values: object) -> object:
        _validate_nested_fields(values)
        return values

    @field_validator("formats")
    @classmethod
    def formats_are_distinct(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("formats must not repeat a ratio")
        return value

    @model_serializer(mode="wrap")
    def _dump_formats(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        if not data.get("formats"):
            data.pop("formats", None)  # nothing to say: dumps of older storyboards are unchanged
        return data

    @model_validator(mode="after")
    def validate_reference_annotations(self) -> Storyboard:
        if self.reference is not None:
            _validate_relative_path(self.reference, "storyboard.reference")
        # shot_type, scale, emotion and role are checked by the segment models, so pydantic
        # reports every bad value of every segment at once.
        for index, segment in enumerate(self.segments):
            audio_from = getattr(segment, "audio_from", None)
            if audio_from is not None:
                _validate_relative_path(audio_from.source, f"segments[{index}].audio_from.source")
        return self

    @model_validator(mode="after")
    def centre_stickers(self) -> Storyboard:
        """A sticker with ``x: "center"`` gets the x that centres it on this storyboard's frame."""

        for overlay in self.overlays:
            if isinstance(overlay, StickerOverlay) and overlay.x == "center":
                from adanvil.render.stickers import centered_x

                overlay.x = centered_x(overlay, self.format.width)
        return self

    @model_validator(mode="after")
    def validate_emoji_path(self) -> Storyboard:
        if self.emoji in {"twemoji", "none"}:
            return self
        _validate_relative_path(self.emoji, "storyboard.emoji")
        return self

    def total_duration(self) -> float:
        return sum(segment_duration(segment) for segment in self.segments)

    def timeline(self) -> list[tuple[float, float]]:
        timeline: list[tuple[float, float]] = []
        start = 0.0
        for segment in self.segments:
            end = start + segment_duration(segment)
            timeline.append((start, end))
            start = end
        return timeline


def _validate_relative_path(value: str, field_name: str) -> None:
    try:
        resolve_in_project(Path.cwd(), value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name}: {exc}") from exc


def load_storyboard(path_or_dict: str | Path | Mapping[str, Any]) -> Storyboard:
    if isinstance(path_or_dict, Mapping):
        return Storyboard.model_validate(path_or_dict)
    path = Path(path_or_dict)
    return Storyboard.model_validate(json.loads(path.read_text(encoding="utf-8")))


def storyboard_json_schema() -> dict[str, Any]:
    return Storyboard.model_json_schema(by_alias=True)


__all__ = [
    "AudioSpec",
    "CaptionOverlay",
    "CommentOverlay",
    "Gap",
    "ImageFrame",
    "ImageOverlay",
    "InsetBorder",
    "Overlay",
    "OutputSpec",
    "ProgressOverlay",
    "Segment",
    "SfxAuto",
    "SfxOverlay",
    "SpeechCaptionsOverlay",
    "StickerOverlay",
    "Storyboard",
    "VideoOverlay",
    "VoiceCaptionsOverlay",
    "load_storyboard",
    "storyboard_json_schema",
]
