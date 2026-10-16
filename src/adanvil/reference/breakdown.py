"""Validate and summarize an agent-authored reference shot breakdown."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from adanvil.project import resolve_in_project
from adanvil.safeio import replace_text
from adanvil.vocab import (
    EMOTION_VALUES,
    ROLE_VALUES,
    SCALE_VALUES,
    TYPE_VALUES,
    normalize_shot_type,
)

from .decode import ReferenceDecode

MOTION_VALUES = (
    "static",
    "handheld",
    "push_in",
    "pull_out",
    "pan",
    "cut_on_action",
    "speed_ramp",
)
TEXT_SPEAKER_VALUES = ("none", "character", "narrator", "viewer", "brand")
TEXT_DEVICE_VALUES = (
    "none",
    "quote",
    "punchline",
    "question",
    "label",
    "subtitle_of_speech",
    "price_or_offer",
    "cta",
)


class BreakdownEntry(BaseModel):
    """One or more consecutive decoded shots described by the agent."""

    model_config = ConfigDict(extra="forbid")

    shots: list[int] = Field(min_length=1)
    start: float | None = None
    end: float | None = None
    type: str | None = None
    scale: str | None = None
    subject: str | None = None
    on_screen_text: str | None = None
    spoken: str = ""
    role: str | None = None
    motion: str | None = None
    replacement: str | None = None
    text_speaker: str | None = None
    text_device: str | None = None
    emotion: str | None = None
    action: str | None = None

    @field_validator("shots")
    @classmethod
    def valid_shot_numbers(cls, value: list[int]) -> list[int]:
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise ValueError("shots must contain integer indexes")
        return value

    @field_validator("type", mode="before")
    @classmethod
    def valid_type(cls, value: str | None) -> str | None:
        # Old names (skin_texture, anatomy_diagram) are accepted and renamed.
        return _controlled(normalize_shot_type(value), "type", TYPE_VALUES)

    @field_validator("scale")
    @classmethod
    def valid_scale(cls, value: str | None) -> str | None:
        return _controlled(value, "scale", SCALE_VALUES)

    @field_validator("role")
    @classmethod
    def valid_role(cls, value: str | None) -> str | None:
        return _controlled(value, "role", ROLE_VALUES)

    @field_validator("motion")
    @classmethod
    def valid_motion(cls, value: str | None) -> str | None:
        return _controlled(value, "motion", MOTION_VALUES)

    @field_validator("emotion")
    @classmethod
    def valid_emotion(cls, value: str | None) -> str | None:
        return _controlled(value, "emotion", EMOTION_VALUES)

    @field_validator("text_speaker")
    @classmethod
    def valid_text_speaker(cls, value: str | None) -> str | None:
        return _controlled(value, "text_speaker", TEXT_SPEAKER_VALUES)

    @field_validator("text_device")
    @classmethod
    def valid_text_device(cls, value: str | None) -> str | None:
        return _controlled(value, "text_device", TEXT_DEVICE_VALUES)


def _controlled(value: str | None, name: str, allowed: tuple[str, ...]) -> str | None:
    if value is None:
        return None
    if value not in allowed:
        values = ", ".join(allowed)
        raise ValueError(f"{name} must be one of: {values}")
    return value


def _load_decode(root: Path, reference: str) -> tuple[ReferenceDecode, Path]:
    path = resolve_in_project(root, reference)
    manifest = path / "reference.json" if path.is_dir() else path
    if not manifest.is_file():
        raise FileNotFoundError(f"reference decode does not exist: {reference}")
    try:
        decoded = ReferenceDecode.model_validate(json.loads(manifest.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read reference decode {reference!r}: {exc}") from exc
    return decoded, manifest.parent


def _validate_coverage(entries: list[BreakdownEntry], shot_count: int) -> None:
    seen: dict[int, int] = {}
    for entry_index, entry in enumerate(entries):
        if entry.shots != list(range(entry.shots[0], entry.shots[-1] + 1)):
            raise ValueError(
                f"breakdown entry {entry_index} must cover consecutive shots; got {entry.shots}"
            )
        for shot_index in entry.shots:
            if shot_index < 1 or shot_index > shot_count:
                raise ValueError(
                    f"breakdown shot {shot_index} is outside decoded shots 1-{shot_count}"
                )
            seen[shot_index] = seen.get(shot_index, 0) + 1
    missing = [index for index in range(1, shot_count + 1) if seen.get(index, 0) == 0]
    duplicated = [index for index, count in seen.items() if count > 1]
    if missing or duplicated:
        parts = []
        if missing:
            parts.append(f"missing shots: {missing}")
        if duplicated:
            parts.append(f"duplicated shots: {sorted(duplicated)}")
        raise ValueError("reference breakdown coverage error; " + "; ".join(parts))


def _summary(decoded: ReferenceDecode, entries: list[BreakdownEntry]) -> dict[str, Any]:
    shot_by_index = {shot.index: shot for shot in decoded.shots}
    total_duration = decoded.duration
    type_seconds: dict[str, float] = {}
    type_shots: dict[str, int] = {}
    scale_seconds: dict[str, float] = {}
    roles: list[str] = []
    people: list[str] = []
    repeated_text_shots = 0  # on-screen text changes: the same text held over shots counts once
    previous_text = ""
    text_devices: Counter[str] = Counter()
    quotes_by_characters = 0
    emotions: dict[str, list[str]] = {}
    missing: list[dict[str, Any]] = []

    for entry in entries:
        duration = sum(shot_by_index[index].duration for index in entry.shots)
        if entry.type is not None:
            type_seconds[entry.type] = type_seconds.get(entry.type, 0.0) + duration
            type_shots[entry.type] = type_shots.get(entry.type, 0) + len(entry.shots)
        if entry.scale is not None:
            scale_seconds[entry.scale] = scale_seconds.get(entry.scale, 0.0) + duration
        if entry.role is not None and (not roles or roles[-1] != entry.role):
            roles.append(entry.role)
        if entry.subject is not None and entry.subject.startswith("person_"):
            if entry.subject not in people:
                people.append(entry.subject)
        text = " ".join((entry.on_screen_text or "").split()).casefold()
        if text and text != previous_text:
            repeated_text_shots += 1
        previous_text = text
        if entry.text_device not in (None, "none"):
            text_devices[entry.text_device] += 1
        if entry.role is not None and entry.emotion not in (None, "none"):
            used = emotions.setdefault(entry.role, [])
            if entry.emotion not in used:
                used.append(entry.emotion)
        if entry.text_speaker == "character" and entry.text_device == "quote":
            quotes_by_characters += 1
        if entry.replacement and entry.replacement.casefold().startswith("missing"):
            missing.append(entry.model_dump(mode="json"))

    bigrams = Counter(f"{left} > {right}" for left, right in zip(roles, roles[1:], strict=False))
    type_demand = {
        name: {
            "shots": type_shots[name],
            "seconds": type_seconds[name],
            "share": type_seconds[name] / total_duration if total_duration else 0.0,
        }
        for name in type_seconds
    }
    scale_mix = {
        name: seconds / total_duration if total_duration else 0.0
        for name, seconds in scale_seconds.items()
    }
    return {
        "type_demand": type_demand,
        "scale_mix": scale_mix,
        "role_sequence": " > ".join(roles),
        "repeated_patterns": {pattern: count for pattern, count in bigrams.items() if count >= 2},
        "people": people,
        "text_shots": repeated_text_shots,
        "text_devices": dict(text_devices),
        "dialogue": quotes_by_characters >= 2,
        "emotion_by_role": emotions,
        "pacing": decoded.pacing,
        "missing": missing,
    }


def breakdown_reference(
    root: Path,
    reference: str,
    shots: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate, persist and summarize a decoded reference breakdown."""

    project_root = Path(root).resolve()
    decoded, output_dir = _load_decode(project_root, reference)
    try:
        entries = [BreakdownEntry.model_validate(entry) for entry in shots]
    except ValueError as exc:
        raise ValueError(f"invalid reference breakdown: {exc}") from exc
    _validate_coverage(entries, len(decoded.shots))
    summary = _summary(decoded, entries)
    payload = {
        "reference": reference,
        "shots": [entry.model_dump(mode="json") for entry in entries],
        "summary": summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    replace_text(
        output_dir / "breakdown.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    saved = (output_dir / "breakdown.json").relative_to(project_root).as_posix()
    return {**summary, "breakdown": saved}


__all__ = [
    "BreakdownEntry",
    "EMOTION_VALUES",
    "MOTION_VALUES",
    "ROLE_VALUES",
    "SCALE_VALUES",
    "TEXT_DEVICE_VALUES",
    "TEXT_SPEAKER_VALUES",
    "TYPE_VALUES",
    "breakdown_reference",
]
