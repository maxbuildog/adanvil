from __future__ import annotations

from functools import cache

from PIL import ImageFont

from .emoji import split_runs
from .fonts import load_font


def measure_line(
    runs: list[tuple[bool, str]],
    font: ImageFont.FreeTypeFont,
    emoji_size: int,
    emoji_gap: int,
) -> float:
    width = 0.0
    for is_emoji, value in runs:
        if is_emoji:
            width += emoji_size + emoji_gap
        else:
            width += float(font.getlength(value))
    return width


def fit_size(
    lines: list[str],
    family: str,
    weight: int,
    size: int,
    max_width: int = 960,
    min_size: int | None = None,
    step: int = 2,
) -> tuple[int, bool]:
    if step <= 0:
        raise ValueError("step must be positive")
    floor = max(28, round(size * 0.6)) if min_size is None else int(min_size)
    floor = min(floor, int(size))
    candidate = int(size)
    while True:
        font = load_font(family, weight, candidate)
        emoji_size = round(candidate * 1.05)
        fits = all(
            measure_line(split_runs(line), font, emoji_size, 6) <= max_width for line in lines
        )
        if fits:
            return candidate, False
        if candidate <= floor:
            return candidate, True
        candidate = max(floor, candidate - step)


def _line_width(words: tuple[str, ...], font: ImageFont.FreeTypeFont) -> float:
    return float(font.getlength(" ".join(words)))


def balance_lines(text: str, n_lines: int, font: ImageFont.FreeTypeFont) -> list[str]:
    words = text.split()
    if not words:
        return [""] * max(1, n_lines)
    n_lines = max(1, int(n_lines))
    if n_lines >= len(words):
        return [*words, *([""] * (n_lines - len(words)))]

    @cache
    def solve(start: int, groups: int) -> tuple[float, tuple[str, ...]]:
        if groups == 1:
            tail = tuple(words[start:])
            return _line_width(tail, font), (" ".join(tail),)
        best: tuple[float, tuple[str, ...]] | None = None
        # Leave at least one word per remaining group.
        for end in range(start + 1, len(words) - groups + 2):
            head = tuple(words[start:end])
            head_width = _line_width(head, font)
            tail_width, tail_lines = solve(end, groups - 1)
            score = max(head_width, tail_width)
            candidate = (score, (" ".join(head), *tail_lines))
            if best is None or (score, candidate[1]) < (best[0], best[1]):
                best = candidate
        assert best is not None
        return best

    return list(solve(0, n_lines)[1])


def _width_for_wrap(line: str, font: ImageFont.FreeTypeFont) -> float:
    return measure_line(split_runs(line), font, round(font.size * 1.05), 6)


def wrap(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join((*current, word))
        if current and _width_for_wrap(candidate, font) > max_width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines
