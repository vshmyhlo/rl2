"""Serializable evaluation scenarios and deterministic scene coverage selection."""

import json
from collections.abc import Sequence
from hashlib import sha256
from math import atan2, ceil, cos, sin
from pathlib import Path
from typing import Annotated, Self

from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass

from rl2.envs.visfly.routes import CATEGORIES, Category, Point, Route, RouteConfig


@dataclass(frozen=True, config=ConfigDict(extra="forbid", allow_inf_nan=False))
class EvaluationConfig:
    manifest_path: str
    routes: RouteConfig = RouteConfig()
    reference_speed: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1.0  # m/s
    maneuver_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 10.0
    min_episode_steps: Annotated[int, Field(gt=0)] = 500


@dataclass(frozen=True, config=ConfigDict(extra="forbid", allow_inf_nan=False))
class Scenario:
    scene: str  # Relative to scene_kwargs.path.
    category: Category
    route: Route
    orientation: tuple[float, float, float, float]  # (4,) wxyz quaternion
    velocity: Point  # (3,) m/s
    angular_velocity: Point  # (3,) rad/s
    max_episode_steps: Annotated[int, Field(gt=0)]
    seed: Annotated[int, Field(ge=0, lt=2**32)]
    probe_step: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.2

    @model_validator(mode="after")
    def validate_scene(self) -> Self:
        if Path(self.scene).is_absolute() or ".." in Path(self.scene).parts:
            raise ValueError("scenario scene must be relative to the scene root")
        if abs(sum(x * x for x in self.orientation) - 1) > 1e-5:
            raise ValueError("scenario orientation must be a unit quaternion")
        if self.category not in self.route.categories:
            raise ValueError("scenario category does not match its route")
        return self


@dataclass(frozen=True, config=ConfigDict(extra="forbid", allow_inf_nan=False))
class SceneCoverage:
    scene: str
    aliases: tuple[str, ...]
    unavailable: tuple[Category, ...]
    reason: str | None = None


@dataclass(frozen=True, config=ConfigDict(extra="forbid", allow_inf_nan=False))
class Manifest:
    scenarios: tuple[Scenario, ...]
    coverage: tuple[SceneCoverage, ...]
    seed: int
    generation: EvaluationConfig
    control_dt: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    version: int = 1

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        if self.version != 1:
            raise ValueError("Unsupported VisFly manifest version")
        if not self.scenarios:
            raise ValueError("No evaluation routes found; inspect scene geometry and route settings")
        inventory = {item.scene for item in self.coverage}
        if len(inventory) != len(self.coverage):
            raise ValueError("Manifest contains duplicate scenes")
        pairs = [(scenario.scene, scenario.category) for scenario in self.scenarios]
        if len(set(pairs)) != len(pairs) or any(scene not in inventory for scene, _ in pairs):
            raise ValueError("Manifest scenarios must have unique scene/category pairs in the coverage inventory")
        for item in self.coverage:
            present = {category for scene, category in pairs if scene == item.scene}
            if set(item.unavailable) != set(CATEGORIES) - present:
                raise ValueError("Manifest coverage does not match its scenarios")
        return self


def unique_scenes(paths: Sequence[str], root: Path) -> list[tuple[str, tuple[str, ...]]]:
    """Deduplicate canonical scene JSON, retaining different object placements and asset references."""
    groups: dict[str, list[str]] = {}
    for path in sorted(paths):
        content = json.loads(Path(path).read_text())
        digest = sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        groups.setdefault(digest, []).append(Path(path).relative_to(root).as_posix())
    return [(names[0], tuple(names[1:])) for names in groups.values()]


def select_scenarios(
    scene: str,
    routes: list[Route],
    config: EvaluationConfig,
    seed: int,
    control_dt: float,
) -> tuple[Scenario, ...]:
    """Choose one route per supported category, favoring endpoint and direction diversity."""
    selected: list[Scenario] = []
    for category in CATEGORIES:
        candidates = [route for route in routes if category in route.categories]
        if not candidates:
            continue

        def rank(route: Route) -> tuple[bool, float]:
            unused = all(route.points != prior.route.points for prior in selected)
            # Separation includes direction, so reversed routes remain useful variants.
            diversity = min(
                (
                    sum(
                        (x - y) ** 2
                        for a, b in zip(
                            (route.points[0], route.points[-1]), (prior.route.points[0], prior.route.points[-1])
                        )
                        for x, y in zip(a, b)
                    )
                    for prior in selected
                ),
                default=0.0,
            )
            return unused, diversity + route.length

        route = max(candidates, key=rank)
        start, goal = route.points[0], route.points[-1]
        # Alternate headings without changing the route geometry.
        yaw = atan2(goal[1] - start[1], goal[0] - start[0]) + (len(selected) % 2) * 1.5707963267948966
        selected.append(
            Scenario(
                scene=scene,
                category=category,
                route=route,
                orientation=(cos(yaw / 2), 0.0, 0.0, sin(yaw / 2)),
                velocity=(0.0, 0.0, 0.0),
                angular_velocity=(0.0, 0.0, 0.0),
                max_episode_steps=max(
                    config.min_episode_steps,
                    ceil((route.length / config.reference_speed + config.maneuver_seconds) / control_dt),
                ),
                seed=(seed + len(selected)) % 2**32,
                probe_step=config.routes.probe_step,
            )
        )
    return tuple(selected)
