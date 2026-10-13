"""Fix a misheard word in the speech captions without putting words in a speaker's mouth.

``speech_captions.corrections`` maps a word as it was heard to the word to show:
``{"rites": "rides"}``. It is applied word by word, ignoring case, and keeps the punctuation
around the word. A correction must be close to the heard word (the character ratio of the two
spellings is at least 0.5): changing a word into a very different one is not a spelling fix, it
is a different sentence, so the clip must be transcribed again instead (``speech_refine``).
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping
from dataclasses import replace
from typing import TypeVar

from .analyze import normalize_word

MIN_RATIO = 0.5  # how close a corrected word must be to the word that was heard

_EDGES = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)
_Word = TypeVar("_Word")


def closeness(heard: str, shown: str) -> float:
    """Character similarity (0 to 1) of two words, ignoring case, accents and punctuation."""

    left, right = normalize_word(heard), normalize_word(shown)
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def correction_problem(heard: str, shown: str) -> str | None:
    """Why ``{heard: shown}`` is not an acceptable correction, or ``None`` when it is."""

    if not heard.strip() or not shown.strip():
        return "a correction needs a heard word and a word to show"
    if len(heard.split()) != 1:
        return (
            f"'{heard}' is not one word: a correction replaces one heard word. For a name heard "
            "as several words, run speech_refine on the clip (put the name in brand.yaml "
            "under speech.vocabulary)"
        )
    if "\n" in shown or "\r" in shown:
        return f"the word to show for '{heard}' must be on one line"
    ratio = closeness(heard, shown)
    if ratio < MIN_RATIO:
        return (
            f"'{heard}' -> '{shown}' changes the word too much (similarity {ratio:.2f}, at least "
            f"{MIN_RATIO} needed): a correction only fixes a spelling. Transcribe the clip again "
            "with speech_refine instead (a stronger model, with the brand vocabulary)"
        )
    return None


def check_corrections(corrections: Mapping[str, str]) -> list[str]:
    """The problems of a whole ``corrections`` mapping (empty when it is fine)."""

    problems = [
        problem
        for heard, shown in corrections.items()
        if (problem := correction_problem(heard, shown)) is not None
    ]
    seen: dict[str, str] = {}
    for heard in corrections:
        key = heard.strip().casefold()
        if key in seen:
            problems.append(f"'{heard}' and '{seen[key]}' are the same heard word")
        seen[key] = heard
    return problems


def _carry_case(heard: str, shown: str) -> str:
    """A lower-case replacement takes the capitals of the heard word it replaces.

    ``Rites`` (a sentence start) becomes ``Rides`` and ``RITES`` becomes ``RIDES``; a replacement
    written with capitals of its own (a brand name) is shown as written.
    """

    if not shown.islower():
        return shown
    if len(heard) > 1 and heard.isupper():
        return shown.upper()
    if heard[:1].isupper():
        return shown[:1].upper() + shown[1:]
    return shown


def correct_text(text: str, corrections: Mapping[str, str]) -> tuple[str, str | None]:
    """``text`` (one word with its punctuation) corrected, and the heard word that matched.

    The match ignores case; leading and trailing punctuation is kept as it is.
    """

    match = _EDGES.match(text)
    if match is None or not match.group(2):
        return text, None
    before, core, after = match.groups()
    wanted = core.casefold()
    for heard, shown in corrections.items():
        if heard.strip().casefold() == wanted:
            return f"{before}{_carry_case(core, shown.strip())}{after}", heard
    return text, None


def apply_corrections(
    words: list[_Word], corrections: Mapping[str, str]
) -> tuple[list[_Word], dict[str, int]]:
    """The words (anything with a ``text`` field) corrected, and how often each correction hit.

    Every heard word of ``corrections`` appears in the count, with 0 when it was never heard.
    """

    used = {heard: 0 for heard in corrections}
    if not corrections:
        return list(words), used
    result: list[_Word] = []
    for word in words:
        text, heard = correct_text(word.text, corrections)  # type: ignore[attr-defined]
        if heard is not None:
            used[heard] += 1
            word = replace(word, text=text)  # type: ignore[type-var]
        result.append(word)
    return result, used


__all__ = [
    "MIN_RATIO",
    "apply_corrections",
    "check_corrections",
    "closeness",
    "correct_text",
    "correction_problem",
]
