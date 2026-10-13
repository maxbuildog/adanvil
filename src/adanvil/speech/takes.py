"""Choose the best take of each line across several recordings of the same script.

A creator films a line several times (``HOOK1.mov``, ``HOOK1.1.mov``, ``HOOK1.2.mov``) and
may retake it inside one file too. Phrases from every file that say the same thing (text
similarity of at least 0.6) form a *line*; each take of a line is scored on fluency (no
filler, no restart), a complete ending (last word not cut), a natural speech rate and the
audio level, and the best one is returned with its clean segments and the reasons.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from .analyze import (
    DEFAULT_MAX_PAUSE,
    FillerLists,
    Phrase,
    clean_segments,
    content_words,
    find_fillers,
    find_restarts,
    split_phrases,
)
from .model import SpeechData

SIMILARITY = 0.6  # normalised text similarity that makes two phrases the same line

# Speech rate (words per second) that sounds natural to a viewer: outside it, a take
# loses points in proportion to the distance.
RATE_BAND = (1.9, 4.2)
_RATE_FALLOFF = 1.0  # words per second beyond the band at which the rate score reaches 0
_CUT_END_SECONDS = 0.12  # the last word ends this close to the end of the file: it was cut
_LOW_CONFIDENCE = 0.35  # a last word the transcriber is this unsure of is probably cut

# Points per criterion (they add up to 100).
_FLUENCY, _ENDING, _RATE, _LEVEL = 40.0, 30.0, 15.0, 15.0


@dataclass
class ClipSpeech:
    """The speech of one library clip, ready for take comparison."""

    entry_id: str
    path: str
    duration: float | None
    data: SpeechData


@dataclass
class Take:
    clip: ClipSpeech
    phrase: Phrase
    words: list[str]  # normalised content words (fillers removed)
    fillers: int = 0
    restart: bool = False
    cut_end: bool = False
    rate: float | None = None
    level: float | None = None
    score: float = 0.0
    why: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(self.words)


def _similar(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    matcher = difflib.SequenceMatcher(None, left, right)
    if matcher.real_quick_ratio() < SIMILARITY or matcher.quick_ratio() < SIMILARITY:
        return 0.0
    return matcher.ratio()


def _collect(clips: list[ClipSpeech], max_pause: float, lists: FillerLists | None) -> list[Take]:
    takes: list[Take] = []
    for clip in clips:
        words = clip.data.words
        if not words:
            continue
        fillers = find_fillers(words, lists)
        phrases = split_phrases(words, max_pause)
        restarts = set(find_restarts(words, phrases, fillers))
        for phrase in phrases:
            content = content_words(words, phrase, fillers)
            if not content:
                continue  # a phrase made only of hesitations is not a line
            take = Take(
                clip=clip,
                phrase=phrase,
                words=content,
                fillers=sum(1 for index in phrase.indexes() if index in fillers),
                restart=phrase.index in restarts,
            )
            span = phrase.end - phrase.start
            if len(content) >= 3 and span > 0:
                take.rate = len(content) / span
            take.level = clip.data.level(phrase.start, phrase.end)
            last = words[phrase.last]
            take.cut_end = (
                (clip.duration is not None and clip.duration - last.end < _CUT_END_SECONDS)
                or last.probability < _LOW_CONFIDENCE
            )
            takes.append(take)
    return takes


def _group(takes: list[Take], similarity: float) -> list[list[Take]]:
    """Greedy grouping: a take joins the line it resembles most, else starts a new line."""

    groups: list[list[Take]] = []
    for take in takes:
        best, best_score = None, 0.0
        for group in groups:
            score = max(_similar(take.text, member.text) for member in group)
            if score >= similarity and score > best_score:
                best, best_score = group, score
        if best is None:
            groups.append([take])
        else:
            best.append(take)
    return groups


def _mark_truncated(group: list[Take]) -> None:
    """A take that is only the start of a longer take of the same line was cut short.

    Its last word may itself be cut ("produi" for "produit"): it only has to start the
    other take's word.
    """

    for take in group:
        size = len(take.words)
        for other in group:
            if (
                other is not take
                and len(other.words) > size
                and other.words[: size - 1] == take.words[:-1]
                and other.words[size - 1].startswith(take.words[-1])
            ):
                take.cut_end = True
                break


def _score(take: Take) -> None:
    fluency = _FLUENCY
    take.why.clear()
    take.issues.clear()
    if take.fillers:
        fluency -= min(_FLUENCY, 15.0 * take.fillers)
        take.issues.append(f"{take.fillers} filler{'s' if take.fillers > 1 else ''}")
    else:
        take.why.append("no filler")
    if take.restart:
        fluency = max(0.0, fluency - _FLUENCY)
        take.issues.append("restart (repeated by the next phrase)")
    else:
        take.why.append("no restart")
    ending = _ENDING
    if take.cut_end:
        ending = 0.0
        take.issues.append("ending cut (last word probably truncated)")
    else:
        take.why.append("complete ending")
    rate = _RATE
    if take.rate is not None:
        low, high = RATE_BAND
        distance = max(low - take.rate, take.rate - high, 0.0)
        rate = _RATE * max(0.0, 1.0 - distance / _RATE_FALLOFF)
        label = f"speech rate {take.rate:.1f} words/s"
        (take.why if distance == 0 else take.issues).append(
            label if distance == 0 else f"{label} (natural band {low:g}-{high:g})"
        )
    level = _LEVEL
    if take.level is not None:
        level = _LEVEL * min(1.0, max(0.0, (take.level + 40.0) / 20.0))  # -20 dBFS and up: full
        (take.why if level >= _LEVEL * 0.75 else take.issues).append(
            f"level {take.level:.0f} dBFS"
        )
    take.score = round(fluency + ending + rate + level, 2)


def _payload(take: Take, *, segments: bool, lists: FillerLists | None, max_pause: float) -> dict:
    words = take.clip.data.words
    item: dict = {
        "entry_id": take.clip.entry_id,
        "source": take.clip.path,
        "start": round(take.phrase.start, 3),
        "end": round(take.phrase.end, 3),
        "text": take.phrase.text,
        "score": take.score,
    }
    if segments:
        fillers = find_fillers(words, lists)
        item["clean_segments"] = clean_segments(
            words,
            take.clip.path,
            max_pause=max_pause,
            remove=fillers,
            selected=take.phrase.indexes(),
            duration=take.clip.duration,
        )
        item["why"] = list(take.why)
    else:
        item["issues"] = list(take.issues)
    return item


def rank_takes(
    clips: list[ClipSpeech],
    *,
    max_pause: float = DEFAULT_MAX_PAUSE,
    similarity: float = SIMILARITY,
    lists: FillerLists | None = None,
) -> list[dict]:
    """Group the phrases of ``clips`` into lines and rank the takes of each line.

    Each result has the ``line`` number, its ``text`` (the best take's), the number of
    ``takes``, the ``best`` take (source, start, end, text, score, why, clean_segments) and
    the ``others`` with the issues that put them behind.
    """

    takes = _collect(clips, max_pause, lists)
    lines: list[dict] = []
    for number, group in enumerate(_group(takes, similarity), start=1):
        _mark_truncated(group)
        for take in group:
            _score(take)
        ranked = sorted(
            group,
            key=lambda take: (
                -take.score,
                -(take.level if take.level is not None else -120.0),
                take.clip.path,
                take.phrase.start,
            ),
        )
        best, others = ranked[0], ranked[1:]
        lines.append(
            {
                "line": number,
                "text": best.phrase.text,
                "takes": len(group),
                "best": _payload(best, segments=True, lists=lists, max_pause=max_pause),
                "others": [
                    _payload(take, segments=False, lists=lists, max_pause=max_pause)
                    for take in others
                ],
            }
        )
    return lines


__all__ = ["RATE_BAND", "SIMILARITY", "ClipSpeech", "Take", "rank_takes"]
