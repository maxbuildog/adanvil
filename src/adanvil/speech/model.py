"""Plain data shapes for the speech of a clip: timed words and what we know of the audio."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpeechWord:
    """One spoken word with its time range in the source clip (seconds)."""

    start: float
    end: float
    text: str
    probability: float = 1.0


@dataclass
class SpeechData:
    """The words of one clip and the level of its audio, as stored in the sidecar file."""

    words: list[SpeechWord] = field(default_factory=list)
    language: str | None = None
    duration: float | None = None
    level_step: float = 0.1  # seconds per entry of ``levels_db``
    levels_db: list[float] = field(default_factory=list)  # RMS in dBFS per step
    model: str | None = None  # the transcription model that made the words

    def level(self, start: float, end: float) -> float | None:
        """Mean level (dBFS, power average) over ``start`` to ``end``, or None if unknown."""

        if not self.levels_db or self.level_step <= 0 or end <= start:
            return None
        first = max(0, int(start / self.level_step))
        last = min(len(self.levels_db), max(first + 1, math.ceil(end / self.level_step)))
        values = self.levels_db[first:last]
        if not values:
            return None
        power = sum(10 ** (value / 10) for value in values) / len(values)
        return 10 * math.log10(power) if power > 0 else -120.0


__all__ = ["SpeechData", "SpeechWord"]
