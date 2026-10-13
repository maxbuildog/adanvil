"""Typed storyboard recipe templates loaded from packaged YAML."""

from __future__ import annotations

from importlib import resources
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from adanvil.vocab import ROLE_VALUES

from .schema import Storyboard


class _RecipeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Duration(_RecipeModel):
    min: float
    max: float


class Audio(_RecipeModel):
    source: str
    notes: str


class Captions(_RecipeModel):
    style: str
    y: int
    size: int
    notes: str


class Slot(_RecipeModel):
    """One part of the structure of an ad.

    ``slot`` is the name used in segment labels (``slot`` or ``slot:n``). ``role`` is the value
    to set as the ``role`` of the segments of this slot (one of the shared role vocabulary:
    ``hook``, ``problem``, ``payoff``, ``cta``...): a slot named like a role gets that role by
    default, any other slot says which one it plays (the ``result`` slot is the ``payoff``).
    ``purpose`` says in words what the slot is for.
    """

    slot: str
    role: str
    purpose: str
    count: int | list[int]
    shot_duration: list[float] = Field(min_length=2, max_length=2)
    footage: str
    caption: str
    effects: str | None = None

    @model_validator(mode="before")
    @classmethod
    def default_role(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        if values.get("role") is None and values.get("slot") in ROLE_VALUES:
            return {**values, "role": values["slot"]}
        return values

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        if value not in ROLE_VALUES:
            raise ValueError(f"slot role must be one of: {', '.join(ROLE_VALUES)}")
        return value

    @field_validator("count")
    @classmethod
    def validate_count(cls, value: int | list[int]) -> int | list[int]:
        if isinstance(value, list) and len(value) != 2:
            raise ValueError("count ranges must contain exactly two integers")
        return value

    @field_validator("shot_duration")
    @classmethod
    def validate_shot_duration(cls, value: list[float]) -> list[float]:
        if value[0] > value[1]:
            raise ValueError("shot_duration minimum must not exceed maximum")
        return value


RecipeCheck = Literal[
    "total_duration",
    "shot_duration",
    "caption_min_on_screen",
    "caption_safe_zone",
    "end_card_required",
    "max_caption_lines",
    "caption_count",
    "banned_words",
]


class Rule(_RecipeModel):
    id: str
    check: RecipeCheck
    params: dict[str, Any] = Field(default_factory=dict)
    severity: Literal["warn", "fail"]


class Recipe(_RecipeModel):
    id: str
    name: str
    summary: str
    when_to_use: str
    duration: Duration
    audio: Audio
    captions: Captions
    structure: list[Slot]
    rules: list[Rule]
    checklist: list[str]
    example: dict[str, Any]

    @model_validator(mode="after")
    def validate_example(self) -> Recipe:
        Storyboard.model_validate(self.example)
        return self


def _recipe_resources() -> list[Any]:
    directory = resources.files("adanvil.storyboard").joinpath("recipes")
    return sorted(
        (entry for entry in directory.iterdir() if entry.name.endswith(".yaml")),
        key=lambda entry: entry.name,
    )


def _load_resource(resource: Any) -> Recipe:
    values = yaml.safe_load(resource.read_text(encoding="utf-8"))
    return Recipe.model_validate(values)


def list_recipes() -> list[Recipe]:
    """Load all packaged recipes in stable id order."""

    return sorted(
        (_load_resource(resource) for resource in _recipe_resources()),
        key=lambda r: r.id,
    )


def get_recipe(recipe_id: str) -> Recipe:
    """Return one recipe, listing valid ids when the name is unknown."""

    recipes = list_recipes()
    for recipe in recipes:
        if recipe.id == recipe_id:
            return recipe
    valid = ", ".join(recipe.id for recipe in recipes)
    raise KeyError(f"Unknown recipe {recipe_id!r}; valid ids: {valid}")


__all__ = [
    "Audio",
    "Captions",
    "Duration",
    "Recipe",
    "RecipeCheck",
    "Rule",
    "Slot",
    "get_recipe",
    "list_recipes",
]
