"""Put the speech of the clips on the storyboard timeline: captions and spoken-word checks.

``played_ranges`` says which part of which source file each segment plays as audio when
``audio.clips`` is on (the segment's own clip, or the ``audio_from`` clip of an L-cut).
``spoken_words`` maps the words of those ranges to timeline time, speed aware, and
``expand_speech_captions`` turns them into ordinary caption overlays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from adanvil.media import ClipSegment
from adanvil.text.chunking import ends_sentence

from .corrections import apply_corrections
from .library import SpeechLibrary

if TYPE_CHECKING:  # the storyboard package imports this one: no runtime import back
    from adanvil.storyboard.schema import CaptionOverlay, SpeechCaptionsOverlay, Storyboard

SENTENCE_GAP = 0.6  # a silence this long (timeline seconds) ends a caption sentence
HOLD = 0.2  # seconds a caption stays after its last word, never into the next one


@dataclass(frozen=True)
class PlayedRange:
    """The audio one segment plays: ``source`` from ``src_in`` to ``src_out``."""

    segment: int
    source: str
    start: float  # timeline start of the segment
    end: float  # timeline end of the played audio
    src_in: float
    src_out: float
    speed: float
    audio_from: bool


@dataclass(frozen=True)
class SpokenWord:
    """A word of the speech that survives in the final audio, on the timeline."""

    start: float
    end: float
    text: str
    source: str
    segment: int
    index: int  # position of the word in its source's word list


def played_ranges(
    storyboard: Storyboard, starts: list[float], ends: list[float]
) -> list[PlayedRange]:
    """The audio each clip segment plays on the timeline (``starts`` and ``ends`` per segment).

    A segment with ``audio_from`` plays that clip at normal speed for its whole duration;
    a plain clip segment plays its own source from ``in`` to ``out`` at its ``speed`` (the
    frozen hold after it is silent). Image segments play nothing.
    """

    ranges: list[PlayedRange] = []
    for index, segment in enumerate(storyboard.segments):
        if not isinstance(segment, ClipSegment):
            continue
        start, end = starts[index], ends[index]
        if segment.audio_from is not None:
            ranges.append(
                PlayedRange(
                    index,
                    segment.audio_from.source,
                    start,
                    end,
                    segment.audio_from.in_,
                    segment.audio_from.in_ + (end - start),
                    1.0,
                    True,
                )
            )
        else:
            moving = (segment.out - segment.in_) / segment.speed
            ranges.append(
                PlayedRange(
                    index,
                    segment.source,
                    start,
                    min(end, start + moving),
                    segment.in_,
                    segment.out,
                    segment.speed,
                    False,
                )
            )
    return ranges


def spoken_words(
    ranges: list[PlayedRange], library: SpeechLibrary, *, keep_fillers: bool = True
) -> list[SpokenWord]:
    """The words whose middle falls in a played range, placed on the timeline in order.

    A word cut by the edge of its range is clamped to it. With ``keep_fillers`` false the
    hesitation words of each source are dropped.
    """

    spoken: list[SpokenWord] = []
    for played in ranges:
        data = library.data(played.source)
        if data is None:
            continue
        fillers = set() if keep_fillers else library.fillers(played.source)
        for index, word in enumerate(data.words):
            if index in fillers:
                continue
            middle = (word.start + word.end) / 2
            if not played.src_in <= middle < played.src_out:
                continue
            begin = played.start + (word.start - played.src_in) / played.speed
            finish = played.start + (word.end - played.src_in) / played.speed
            begin = min(max(begin, played.start), played.end)
            finish = min(max(finish, begin), played.end)
            spoken.append(
                SpokenWord(begin, finish, word.text, played.source, played.segment, index)
            )
    spoken.sort(key=lambda item: (item.start, item.end))
    return spoken


_LEADING_JOIN = ("'", "’", "-")  # "'ai", "-t-il": glued to the word before
_TRAILING_JOIN = ("'", "’")  # "l'", "qu'": glued to the word after
_CLOSING = (",", ".", "!", "?", ";", ":", "»", ")", "…")


def _merge_fragments(words: list[SpokenWord]) -> list[SpokenWord]:
    """Glue the pieces a transcriber splits (elisions, hyphens, detached punctuation).

    Whisper often returns "J" + "'ai" or "m" + "'en"; shown with a space they read "J 'ai".
    A fragment is merged only with a neighbour from the same source and segment.
    """

    merged: list[SpokenWord] = []
    for word in words:
        text = word.text.strip()
        if not text:
            continue
        if merged:
            last = merged[-1]
            same_take = last.source == word.source and last.segment == word.segment
            if same_take and (
                text.startswith(_LEADING_JOIN)
                or last.text.endswith(_TRAILING_JOIN)
                or (text.startswith(_CLOSING) and not text.startswith("»»"))
            ):
                joiner = " " if text in ("?", "!", ":", ";", "»") else ""
                merged[-1] = SpokenWord(
                    last.start,
                    max(last.end, word.end),
                    last.text + joiner + text,
                    last.source,
                    last.segment,
                    last.index,
                )
                continue
        merged.append(SpokenWord(word.start, word.end, text, word.source, word.segment, word.index))
    return merged


def _sentences(words: list[SpokenWord]) -> list[list[SpokenWord]]:
    sentences: list[list[SpokenWord]] = []
    current: list[SpokenWord] = []
    for position, word in enumerate(words):
        current.append(word)
        last = position == len(words) - 1
        if last:
            break
        closes_sentence = ends_sentence(word.text)
        if closes_sentence or words[position + 1].start - word.end > SENTENCE_GAP:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def expand_speech_captions(
    overlay: SpeechCaptionsOverlay,
    storyboard: Storyboard,
    library: SpeechLibrary,
    starts: list[float],
    ends: list[float],
    corrections_used: dict[str, int] | None = None,
) -> list[CaptionOverlay]:
    """Ordinary caption overlays for the speech of the segments of ``storyboard``.

    Sentences end at sentence punctuation or after a silence of 0.6 s; ``sentence`` mode shows
    one caption per sentence, ``chunks`` mode groups of ``max_words`` words that never run across
    a sentence end and stop at a comma (see ``text.chunking``); the ``word`` animation shows one
    or two words at a time. Each caption carries the timeline time of its words. A caption lasts
    until the next one starts, or 0.2 s after its last word, and never past the end of the ad.
    The ``corrections`` of the overlay are applied to the words first; when ``corrections_used``
    is given it is filled with how many times each one was applied.
    """

    words = spoken_words(
        played_ranges(storyboard, starts, ends),
        library,
        keep_fillers=overlay.keep_fillers,
    )
    words, used = apply_corrections(words, overlay.corrections)
    if corrections_used is not None:
        corrections_used.update(used)
    # text, start, last word end, next limit, timeline (start, end) of each word
    pieces: list[tuple[str, float, float, float, list[tuple[float, float]]]] = []
    sentences = _sentences(_merge_fragments(words))
    for number, sentence in enumerate(sentences):
        limit = sentences[number + 1][0].start if number + 1 < len(sentences) else float("inf")
        if overlay.chunked:
            groups = [
                sentence[first:last]
                for first, last in overlay.chunk_ranges([word.text for word in sentence])
            ]
        else:
            groups = [sentence]
        for position, group in enumerate(groups):
            following = groups[position + 1][0].start if position + 1 < len(groups) else limit
            pieces.append(
                (
                    " ".join(word.text for word in group),
                    group[0].start,
                    group[-1].end,
                    following,
                    [(word.start, word.end) for word in group],
                )
            )
    captions: list[CaptionOverlay] = []
    timeline_end = ends[-1] if ends else float("inf")
    for text, start, last_end, following, timed in pieces:
        end = min(following, last_end + HOLD, timeline_end)
        if end > start:
            captions.append(overlay.caption(text, start, end, timed))
    return captions


__all__ = [
    "HOLD",
    "SENTENCE_GAP",
    "PlayedRange",
    "SpokenWord",
    "expand_speech_captions",
    "played_ranges",
    "spoken_words",
]
