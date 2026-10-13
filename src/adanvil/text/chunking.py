"""Group the words of a caption into chunks that respect the punctuation of the sentence.

The rules, applied to words in speaking order (a word keeps its own punctuation, so ``fast,``
and ``today.`` end with it):

* a chunk never runs across the end of a sentence (``.``, ``!``, ``?``, ``...``);
* a chunk stops after a comma, semicolon or colon once it holds two words, so a clause is not
  cut in the middle; a lone first word is never split off by a comma (it stays with the next
  words, which would otherwise leave it as a one-word chunk);
* a chunk never starts with punctuation: a token with no letter or digit stays glued to the word
  before it (to the word after it for an opening mark such as ``(`` or ``«``).
"""

from __future__ import annotations

import re
import unicodedata

SENTENCE_END = (".", "!", "?", "…")
CLAUSE_END = (",", ";", ":", "،", "、")
_CLOSERS = "\"'»”’)]}"
_OPENERS = "«“‘([¿¡{"


def _bare(word: str) -> str:
    return word.rstrip().rstrip(_CLOSERS)


def ends_sentence(word: str) -> bool:
    """True when ``word`` ends a sentence (closing quotes and brackets are looked through)."""

    return _bare(word).endswith(SENTENCE_END)


def ends_clause(word: str) -> bool:
    """True when ``word`` ends with a comma, semicolon or colon."""

    return _bare(word).endswith(CLAUSE_END)


def is_mark(word: str) -> bool:
    """True for a token with no letter or digit: punctuation, a symbol or an emoji."""

    return not any(char.isalnum() for char in word)


def is_opening_mark(word: str) -> bool:
    return is_mark(word) and word.rstrip()[-1:] in _OPENERS


def letters(word: str) -> int:
    return sum(1 for char in word if char.isalnum())


def fold(word: str) -> str:
    """Lower-case, accent-free form of ``word`` with the punctuation around it removed."""

    plain = "".join(
        char
        for char in unicodedata.normalize("NFKD", word.casefold())
        if not unicodedata.combining(char)
    )
    return plain.strip(" \t\r\n.,;:!?…\"'«»“”‘’()[]{}*_–—-")


_EMPHASIS_EDGE = ".,;:!?…\"'«»“”‘’()[]{}*_–—"
_SPLIT_PARTS = re.compile(r"(?<=\w)['’\-](?=\w)")  # inside a word: l'éclat, anti-âge


def _plain(word: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", word.casefold())
        if not unicodedata.combining(char)
    )


def emphasis_key(text: str) -> str:
    """Comparable form of an emphasis word or a spoken word, numbers and prices included.

    Case, accents, spaces and the punctuation around the text are dropped; a decimal comma or
    dot between digits is one dot, so ``65,90 €``, ``65.90€`` and ``65,90€.`` give the same key;
    a sign before a number or a currency is kept (``-20%``).
    """

    plain = re.sub(r"[–−](?=[\d$€£])", "-", _plain(text))
    plain = "".join(plain.split())
    plain = re.sub(r"(?<=\d)[,.](?=\d)", ".", plain).strip(_EMPHASIS_EDGE)
    plain = plain.rstrip("-")
    if plain.startswith("-") and not (
        len(plain) > 1 and (plain[1].isdigit() or unicodedata.category(plain[1]) == "Sc")
    ):
        plain = plain.lstrip("-")
    return plain


def _is_symbol(char: str) -> bool:
    return char == "%" or unicodedata.category(char) == "Sc"


def _bare_number(key: str) -> str:
    """``key`` of a price or a rate without its currency signs, percent sign and sign."""

    return "".join(char for char in key if not _is_symbol(char)).lstrip("+-")


def _entry_units(entry: str) -> list[str]:
    """The spoken words of an emphasis entry: a mark with no letter stays with its word."""

    units: list[str] = []
    pending = ""
    for token in entry.split():
        if is_mark(token):
            if is_opening_mark(token):
                pending += token + " "
            elif units:
                units[-1] += " " + token
            else:
                units.append(token)
            continue
        units.append(pending + token)
        pending = ""
    if pending.strip():
        if units:
            units[-1] += " " + pending.strip()
        else:
            units.append(pending.strip())
    return units


def match_emphasis(words: list[str], entries: list[str]) -> tuple[set[int], list[str]]:
    """The indexes of the spoken ``words`` that ``entries`` emphasise, and the unmatched entries.

    An entry is one or several consecutive words (``65,90 €``, ``livraison offerte``). It matches
    a word with the same key (see ``emphasis_key``) or, for a single word, one of the parts of a
    hyphenated or elided word. An entry that is a plain number (``65,90``) also matches that
    number with a currency or percent sign (``65,90 €``). An entry with no letter or digit at
    all (an empty string, a lone punctuation mark) is ignored.
    """

    keyed = [
        (
            emphasis_key(word),
            {emphasis_key(part) for part in _SPLIT_PARTS.split(word)} - {""},
        )
        for word in words
    ]
    matched: set[int] = set()
    unmatched: list[str] = []
    for entry in entries:
        units = [emphasis_key(unit) for unit in _entry_units(entry)]
        if not units or not all(units):
            continue
        found = False
        for first in range(len(words) - len(units) + 1):
            if all(
                _same_word(unit, keyed[first + offset])
                for offset, unit in enumerate(units)
            ):
                matched.update(range(first, first + len(units)))
                found = True
        if not found:
            unmatched.append(entry)
    return matched, unmatched


def _same_word(unit: str, word: tuple[str, set[str]]) -> bool:
    key, parts = word
    if unit == key or unit in parts:
        return True
    return not any(_is_symbol(char) for char in unit) and _bare_number(key) == unit


def chunk_ranges(
    words: list[str], max_words: int, *, pair_short: bool = False
) -> list[tuple[int, int]]:
    """Index ranges ``(first, end)`` that cut ``words`` into chunks of at most ``max_words``.

    With ``pair_short`` (the ``word`` animation) a chunk is one word, or two when the first has
    three letters or fewer (a short word reads better with the next one).
    """

    count = len(words)
    if count == 0:
        return []
    limit = max(1, int(max_words))
    # A mark that opens (« or () belongs to the word after it, any other mark to the word before
    # it, so a chunk never starts with a closing mark.
    glued_to_previous = [
        index > 0 and is_mark(word) and not is_opening_mark(word)
        for index, word in enumerate(words)
    ]
    ranges: list[tuple[int, int]] = []
    first = 0
    while first < count:
        end = first + 1
        target = limit
        if pair_short:
            target = 2 if letters(words[first]) <= 3 and limit >= 2 else 1
        while end < count and (end - first) < target:
            previous = words[end - 1]
            if ends_sentence(previous):
                break
            if ends_clause(previous) and end - first >= 2:
                break
            end += 1
        # An opening mark takes the word after it, and the marks that belong to the last word
        # of the chunk ride along (``... ?``, a lone emoji).
        while end < count and (glued_to_previous[end] or is_opening_mark(words[end - 1])):
            end += 1
        ranges.append((first, end))
        first = end
    return ranges


__all__ = [
    "CLAUSE_END",
    "SENTENCE_END",
    "chunk_ranges",
    "ends_clause",
    "ends_sentence",
    "fold",
    "is_mark",
    "is_opening_mark",
    "emphasis_key",
    "letters",
    "match_emphasis",
]
