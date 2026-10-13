"""Read the speech of a clip: phrases, fillers, restarts and clean cut segments."""

from __future__ import annotations

import difflib
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .model import SpeechWord

DEFAULT_MAX_PAUSE = 0.35  # seconds of silence that end a phrase and are cut out
LEAD = 0.06  # seconds kept before the first word of a clean segment
TAIL = 0.10  # seconds kept after the last word of a clean segment
MERGE_GAP = 0.12  # clean segments closer than this (seconds) become one
ISOLATION_GAP = 0.2  # a pause this long (seconds) next to a word makes it "alone"

# Hesitation sounds: flagged wherever they appear.
DEFAULT_FILLERS = (
    "euh", "heu", "hum", "hmm", "hm", "mmh", "bah", "ben", "um", "umm", "uh", "uhm", "er", "erm",
)  # fmt: skip
# Ordinary words that are fillers only when said alone, between two pauses or commas.
DEFAULT_ISOLATED_FILLERS = ("genre", "du coup", "like")

_SENTENCE_END = (".", "!", "?", "…")
_BREAK_MARKS = ",;:.!?…—–-"


@dataclass(frozen=True)
class FillerLists:
    """The filler words of a language mix: ``always`` and ``isolated`` (see module docs)."""

    always: tuple[str, ...] = DEFAULT_FILLERS
    isolated: tuple[str, ...] = DEFAULT_ISOLATED_FILLERS


def normalize_word(text: str) -> str:
    """Lower case, accents and punctuation removed: the form words are compared in."""

    folded = unicodedata.normalize("NFKD", str(text).casefold())
    stripped = "".join(char for char in folded if not unicodedata.combining(char))
    return re.sub(r"[^\w]+", "", stripped, flags=re.UNICODE)


def _collapse(word: str) -> str:
    """"euhhh" and "ummm" read as "euh" and "um"."""

    return re.sub(r"(.)\1+", r"\1", word)


def _entry_tokens(entry: str) -> tuple[str, ...]:
    return tuple(
        token for token in (_collapse(normalize_word(part)) for part in entry.split()) if token
    )


def find_fillers(words: list[SpeechWord], lists: FillerLists | None = None) -> set[int]:
    """Indexes of the words that are hesitations (see :class:`FillerLists`)."""

    lists = lists or FillerLists()
    always = {tokens[0] for tokens in map(_entry_tokens, lists.always) if len(tokens) == 1}
    always_phrases = [tokens for tokens in map(_entry_tokens, lists.always) if len(tokens) > 1]
    isolated = [tokens for tokens in map(_entry_tokens, lists.isolated) if tokens]
    keys = [_collapse(normalize_word(word.text)) for word in words]

    def alone(first: int, last: int) -> bool:
        before = (
            first == 0
            or words[first].start - words[first - 1].end >= ISOLATION_GAP
            or words[first - 1].text.rstrip().endswith(tuple(_BREAK_MARKS))
        )
        after = (
            last == len(words) - 1
            or words[last + 1].start - words[last].end >= ISOLATION_GAP
            or words[last].text.rstrip().endswith(tuple(_BREAK_MARKS))
        )
        return before and after

    flagged: set[int] = set()
    for index, key in enumerate(keys):
        if key and key in always:
            flagged.add(index)
    for tokens in always_phrases:
        size = len(tokens)
        for first in range(len(words) - size + 1):
            if tuple(keys[first : first + size]) == tokens:
                flagged.update(range(first, first + size))
    for tokens in isolated:
        size = len(tokens)
        for first in range(len(words) - size + 1):
            if tuple(keys[first : first + size]) == tokens and alone(first, first + size - 1):
                flagged.update(range(first, first + size))
    return flagged


@dataclass
class Phrase:
    """A run of words said without a long pause, or up to a sentence end."""

    index: int
    first: int  # index of its first word in the clip's word list
    last: int  # index of its last word (inclusive)
    start: float
    end: float
    text: str

    def indexes(self) -> range:
        return range(self.first, self.last + 1)


def split_phrases(words: list[SpeechWord], max_pause: float = DEFAULT_MAX_PAUSE) -> list[Phrase]:
    """Split on pauses longer than ``max_pause`` and after sentence punctuation."""

    phrases: list[Phrase] = []
    first = 0
    for index, word in enumerate(words):
        last_word = index == len(words) - 1
        if not last_word:
            gap = words[index + 1].start - word.end
            ends_sentence = word.text.rstrip().endswith(_SENTENCE_END)
            if gap <= max_pause and not ends_sentence:
                continue
        phrases.append(
            Phrase(
                index=len(phrases),
                first=first,
                last=index,
                start=words[first].start,
                end=word.end,
                text=" ".join(item.text for item in words[first : index + 1]),
            )
        )
        first = index + 1
    return phrases


def _same_word(left: str, right: str) -> bool:
    return left == right or difflib.SequenceMatcher(None, left, right).ratio() >= 0.8


def content_words(
    words: list[SpeechWord], phrase: Phrase, fillers: set[int]
) -> list[str]:
    """Normalised words of a phrase without its fillers (what it actually says)."""

    return [
        key
        for index in phrase.indexes()
        if index not in fillers and (key := normalize_word(words[index].text))
    ]


def find_restarts(
    words: list[SpeechWord], phrases: list[Phrase], fillers: set[int]
) -> list[int]:
    """Indexes of phrases that repeat the start of the next one (a retake inside one file).

    A phrase is a restart when all of it (two words or more) is found at the start of the
    next phrase, or when its first three words are.
    """

    restarts: list[int] = []
    for index in range(len(phrases) - 1):
        said = content_words(words, phrases[index], fillers)
        later = content_words(words, phrases[index + 1], fillers)
        if not said or not later:
            continue
        shared = 0
        for left, right in zip(said, later, strict=False):
            if not _same_word(left, right):
                break
            shared += 1
        if (shared == len(said) and len(said) >= 2) or shared >= 3:
            restarts.append(index)
    return restarts


def _floor(value: float) -> float:
    return math.floor(value * 1000 + 1e-6) / 1000


def _ceil(value: float) -> float:
    return math.ceil(value * 1000 - 1e-6) / 1000


def clean_segments(
    words: list[SpeechWord],
    source: str,
    *,
    max_pause: float = DEFAULT_MAX_PAUSE,
    remove: set[int] | None = None,
    selected: range | None = None,
    duration: float | None = None,
) -> list[dict[str, Any]]:
    """Clip segments (``source``, ``in``, ``out``) that keep the speech and drop the rest.

    Pauses longer than ``max_pause`` and the words in ``remove`` (fillers) are cut out. Each
    segment keeps 0.06 s before its first word and 0.10 s after its last one, never reaching
    into a neighbouring word, so no cut falls inside a word. Segments separated by a pause
    shorter than 0.12 s are merged; a removed word is always a cut. ``selected`` restricts the
    work to a range of words (one phrase).
    """

    remove = remove or set()
    pool = selected if selected is not None else range(len(words))
    kept = [index for index in pool if index not in remove]
    runs: list[list[int]] = []
    for index in kept:
        if (
            runs
            and index == runs[-1][-1] + 1
            and words[index].start - words[index - 1].end <= max_pause
        ):
            runs[-1].append(index)
        else:
            runs.append([index])

    pieces: list[tuple[float, float, int, int]] = []  # in, out, first word, last word
    for run in runs:
        first, last = run[0], run[-1]
        lead = words[first].start - LEAD
        if first > 0:
            lead = max(lead, words[first - 1].end)
        lead = max(0.0, min(lead, words[first].start))
        tail = words[last].end + TAIL
        if last + 1 < len(words):
            tail = min(tail, words[last + 1].start)
        elif duration is not None:
            tail = min(tail, duration)
        tail = max(tail, words[last].end)
        pieces.append((lead, tail, first, last))

    merged: list[tuple[float, float, int, int]] = []
    for piece in pieces:
        if merged:
            previous = merged[-1]
            # Only a plain pause is merged: a removed word between two runs stays a cut.
            if piece[2] == previous[3] + 1 and piece[0] - previous[1] < MERGE_GAP:
                merged[-1] = (previous[0], max(previous[1], piece[1]), previous[2], piece[3])
                continue
        merged.append(piece)

    return [
        {"source": source, "in": _floor(start), "out": _ceil(end)}
        for start, end, _first, _last in merged
        if end > start
    ]


@dataclass
class SpeechReading:
    """Everything ``speech_view`` reports about one clip."""

    words: list[SpeechWord]
    phrases: list[Phrase]
    fillers: set[int]
    restarts: list[int]
    segments: list[dict[str, Any]] = field(default_factory=list)


def read_speech(
    words: list[SpeechWord],
    source: str,
    *,
    max_pause: float = DEFAULT_MAX_PAUSE,
    lists: FillerLists | None = None,
    drop_restarts: bool = False,
    duration: float | None = None,
) -> SpeechReading:
    """Phrases, fillers, restarts and clean segments of one clip."""

    fillers = find_fillers(words, lists)
    phrases = split_phrases(words, max_pause)
    restarts = find_restarts(words, phrases, fillers)
    remove = set(fillers)
    if drop_restarts:
        for index in restarts:
            remove.update(phrases[index].indexes())
    segments = clean_segments(
        words, source, max_pause=max_pause, remove=remove, duration=duration
    )
    return SpeechReading(words, phrases, fillers, restarts, segments)


def word_payload(word: SpeechWord, *, filler: bool = False) -> dict[str, Any]:
    item: dict[str, Any] = {
        "start": round(word.start, 3),
        "end": round(word.end, 3),
        "text": word.text,
    }
    if filler:
        item["filler"] = True
    return item


UNCERTAIN_BELOW = 0.5  # a word the transcriber gave under this probability is worth checking
DEFAULT_MAX_PHRASES = 60


def phrase_flags(reading: SpeechReading, phrase: Phrase) -> list[str]:
    """What an agent should notice about a phrase without reading its words.

    ``restart``: the next phrase repeats it (a retake); ``filler``: it holds a hesitation word;
    ``uncertain``: the transcriber doubted a word of it, so it may be misheard.
    """

    flags: list[str] = []
    if phrase.index in reading.restarts:
        flags.append("restart")
    if any(index in reading.fillers for index in phrase.indexes()):
        flags.append("filler")
    if any(reading.words[index].probability < UNCERTAIN_BELOW for index in phrase.indexes()):
        flags.append("uncertain")
    return flags


def reading_payload(
    reading: SpeechReading,
    *,
    source: str,
    words: bool = False,
    first_phrase: int = 0,
    max_phrases: int | None = DEFAULT_MAX_PHRASES,
) -> dict[str, object]:
    """The compact JSON of a reading: phrases (start, end, text, flags), fillers, restarts, cuts.

    Per-word data is only included with ``words``. At most ``max_phrases`` phrases are returned,
    from ``first_phrase``; when the clip has more, ``note`` and ``next_first_phrase`` say how to
    get the rest, and ``fillers``, ``restarts`` and ``clean_segments`` cover the phrases shown
    (``clean_segments`` are the ones that overlap them).
    """

    all_words = reading.words
    phrase_of = {
        index: phrase.index for phrase in reading.phrases for index in phrase.indexes()
    }
    total = len(reading.phrases)
    first = max(0, min(first_phrase, total))
    last = total if max_phrases is None else min(total, first + max(1, max_phrases))
    shown = reading.phrases[first:last]
    shown_indexes = {phrase.index for phrase in shown}

    def phrase_item(phrase: Phrase) -> dict[str, object]:
        item: dict[str, object] = {
            "index": phrase.index,
            "start": round(phrase.start, 3),
            "end": round(phrase.end, 3),
            "text": phrase.text,
            "flags": phrase_flags(reading, phrase),
        }
        if words:
            item["words"] = [
                word_payload(all_words[index], filler=index in reading.fillers)
                for index in phrase.indexes()
            ]
        return item

    if shown:
        window_start, window_end = shown[0].start, shown[-1].end
        segments = [
            segment
            for segment in reading.segments
            if float(segment["out"]) > window_start and float(segment["in"]) < window_end
        ]
    else:
        segments = []
    payload: dict[str, object] = {
        "phrase_count": total,
        "phrases": [phrase_item(phrase) for phrase in shown],
        "fillers": [
            {
                "phrase": phrase_of[index],
                "start": round(all_words[index].start, 3),
                "end": round(all_words[index].end, 3),
                "text": all_words[index].text,
            }
            for index in sorted(reading.fillers)
            if phrase_of.get(index) in shown_indexes
        ],
        "restarts": [
            {
                "phrase": index,
                "repeated_by": index + 1,
                "start": round(reading.phrases[index].start, 3),
                "end": round(reading.phrases[index].end, 3),
                "text": reading.phrases[index].text,
            }
            for index in reading.restarts
            if index in shown_indexes
        ],
        "clean_segments": segments,
        "source": source,
    }
    if first > 0 or last < total:
        if shown:
            note = (
                f"phrases {shown[0].index + 1} to {shown[-1].index + 1} of {total} shown "
                f"(max_phrases={max_phrases})"
            )
        else:
            note = f"no phrase from first_phrase={first_phrase} ({total} phrases in all)"
        if last < total:
            note += f"; call again with first_phrase={last} for the next ones"
            payload["next_first_phrase"] = last
        note += "; fillers, restarts and clean_segments are those of the phrases shown"
        payload["note"] = note
    return payload


def lists_from_brand(brand: Any) -> FillerLists:
    """The filler lists of a brand profile: its ``speech`` section, else the defaults."""

    speech = getattr(brand, "speech", None)
    if speech is None:
        return FillerLists()
    return FillerLists(
        always=tuple(speech.fillers) if speech.fillers is not None else DEFAULT_FILLERS,
        isolated=(
            tuple(speech.isolated_fillers)
            if speech.isolated_fillers is not None
            else DEFAULT_ISOLATED_FILLERS
        ),
    )


def brand_filler_lists(root: Path) -> FillerLists:
    """The filler lists of the project: ``speech`` of brand.yaml, else the defaults."""

    from adanvil.brand import load_brand
    from adanvil.project import BRAND_FILE

    path = Path(root) / BRAND_FILE
    if not path.is_file():
        return FillerLists()
    try:
        return lists_from_brand(load_brand(path))
    except Exception:  # an unreadable brand profile must not stop the speech tools
        return FillerLists()


__all__ = [
    "DEFAULT_FILLERS",
    "DEFAULT_ISOLATED_FILLERS",
    "DEFAULT_MAX_PAUSE",
    "DEFAULT_MAX_PHRASES",
    "UNCERTAIN_BELOW",
    "FillerLists",
    "Phrase",
    "SpeechReading",
    "brand_filler_lists",
    "clean_segments",
    "content_words",
    "find_fillers",
    "find_restarts",
    "lists_from_brand",
    "normalize_word",
    "phrase_flags",
    "read_speech",
    "reading_payload",
    "split_phrases",
    "word_payload",
]
