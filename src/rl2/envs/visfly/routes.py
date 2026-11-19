"""Serializable flight routes, difficulty labels, and sampling configuration."""

from math import dist
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass

type Point = tuple[float, float, float]
type Category = Literal["long_traversal", "obstacle_avoidance", "substantial_detour", "constrained_passage"]
type Difficulty = Literal["easy", "medium", "hard"]
CATEGORIES: tuple[Category, ...] = ("long_traversal", "obstacle_avoidance", "substantial_detour", "constrained_passage")
DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")


@dataclass(frozen=True, config=ConfigDict(extra="forbid", allow_inf_nan=False))
class RouteConfig:
    spacing: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.75
    probe_step: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.2
    max_nodes: Annotated[int, Field(ge=8)] = 32768
    neighbors: Annotated[int, Field(gt=0)] = 10
    sources: Annotated[int, Field(gt=0)] = 24
    goals_per_source: Annotated[int, Field(ge=3)] = 12
    min_distance: Annotated[float, Field(gt=0.5, allow_inf_nan=False)] = 2.0
    long_fraction: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] = 0.6
    detour_ratio: Annotated[float, Field(gt=1, allow_inf_nan=False)] = 1.25
    passage_margin: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.3
    weights: tuple[float, float, float] = (0.2, 0.4, 0.4)  # easy, medium, hard
    jitter: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.2

    @model_validator(mode="after")
    def validate_weights(self) -> Self:
        if any(not 0 <= weight < float("inf") for weight in self.weights) or sum(self.weights) <= 0:
            raise ValueError("route weights must be nonnegative with a positive sum")
        return self


@dataclass(frozen=True, config=ConfigDict(extra="forbid", allow_inf_nan=False))
class Route:
    points: Annotated[tuple[Point, ...], Field(min_length=2)]  # (P, 3) collision-checked polyline
    direct_blocked: bool
    constrained: bool
    long: bool
    detour_threshold: float

    @property
    def initial_distance(self) -> float:
        return dist(self.points[0], self.points[-1])

    @property
    def length(self) -> float:
        return sum(dist(a, b) for a, b in zip(self.points, self.points[1:]))

    @property
    def categories(self) -> tuple[Category, ...]:
        flags = (
            self.long,
            self.direct_blocked,
            self.direct_blocked and self.length >= self.detour_threshold * self.initial_distance,
            self.constrained,
        )
        return tuple(category for category, flag in zip(CATEGORIES, flags) if flag)

    @property
    def difficulty(self) -> Difficulty:
        if self.direct_blocked:
            return "hard"
        return "medium" if self.long else "easy"
