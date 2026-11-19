"""Navigation rewards and dynamics with RGB and optional aligned depth observations."""

import logging
from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack, closing
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from random import Random
from typing import Any, Literal, NamedTuple

import numpy as np
import torch
from habitat_sim import SensorType
from numpy.typing import NDArray
from pydantic import TypeAdapter
from VisFly.envs.NavigationEnv import NavigationEnv
from VisFly.utils.common import habitat_to_std
from VisFly.utils.maths import Quaternion
from VisFly.utils.type import TensorDict

from rl2.envs.visfly.collision import collision_points
from rl2.envs.visfly.dynamics import CompiledEuler
from rl2.envs.visfly.planning import build_roadmap, build_routes, sample_route, segments_clear
from rl2.envs.visfly.routes import Route, RouteConfig
from rl2.envs.visfly.sampling import sample_positions
from rl2.envs.visfly.scenarios import Scenario


class SamplingBounds(NamedTuple):
    lower: torch.Tensor  # (3,)
    upper: torch.Tensor  # (3,)


def sampling_bounds(bounds: Any, clearance: float, height: tuple[float, float]) -> SamplingBounds:
    """Return (3,) bounds within dynamics limits; height is relative to the scene AABB floor."""
    corners, _ = habitat_to_std(np.array([bounds.min, bounds.max]))  # (2, 3)
    lower = corners.amin(dim=0) + clearance  # (3,)
    upper = corners.amax(dim=0) - clearance  # (3,)
    floor = corners[:, 2].min()  # ()
    lower[2] = torch.maximum(lower[2], floor + height[0])
    upper[2] = torch.minimum(upper[2], floor + height[1])
    # Match Dynamics._ugly_fix in pinned VisFly; starts and goals must survive its position clamp.
    lower = torch.maximum(lower, lower.new_tensor([-100.0, -100.0, 0.0]))  # (3,)
    upper = torch.minimum(upper, upper.new_tensor([100.0, 100.0, 20.0]))  # (3,)
    return SamplingBounds(lower=lower, upper=upper)


class SampledState(NamedTuple):
    position: torch.Tensor  # (N, 3)
    orientation: torch.Tensor  # (N, 4)
    velocity: torch.Tensor  # (N, 3)
    angular_velocity: torch.Tensor  # (N, 3)


@dataclass
class RouteCatalogue:
    scene: Any
    config: RouteConfig
    routes: list[Route]
    long_threshold: float


@dataclass
class RouteStatistics:
    count: int = 0
    initial_distance: float = 0.0
    reference_length: float = 0.0
    blocked: int = 0
    difficulties: Counter[str] = field(default_factory=Counter)

    def add(self, route: Route) -> None:
        self.count += 1
        self.initial_distance += route.initial_distance
        self.reference_length += route.length
        self.blocked += route.direct_blocked
        self.difficulties[route.difficulty] += 1

    def metrics(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {
            "rollout/sampled_episode_count": self.count,
            "rollout/initial_goal_distance_meters": self.initial_distance / self.count,
            "rollout/reference_route_length_meters": self.reference_length / self.count,
            "rollout/direct_path_blocked_fraction": self.blocked / self.count,
            **{
                f"rollout/difficulty_fraction/{level}": self.difficulties[level] / self.count
                for level in ("easy", "medium", "hard")
            },
        }


@dataclass
class SceneStateRandomizer:
    base: Any
    manager: Any
    scene_id: int
    clearance: float
    height: tuple[float, float]
    route_sampler: Callable[[int], Route] | None = None
    scenario: Scenario | None = None
    route: Route | None = None

    def safe_generate(self, num: int = 1, **kwargs: Any) -> SampledState:
        """Return (N, 3) position/velocities and (N, 4) quaternion orientation."""
        if self.scenario is not None:
            scenario = self.scenario
            bounds = sampling_bounds(self.manager._scene_bounds[self.scene_id], self.clearance, self.height)
            points = torch.tensor(scenario.route.points)  # (P, 3)

            def collision(positions: NDArray[np.float32], radius: float) -> NDArray[np.bool_]:  # (N, 3) -> (N,)
                return collision_points(self.manager.scenes[self.scene_id], positions, radius)

            if not bool(((points >= bounds.lower) & (points <= bounds.upper)).all()) or not bool(
                segments_clear(points[:-1], points[1:], collision, self.clearance, scenario.probe_step).all()
            ):
                raise ValueError(
                    f"Saved evaluation route is outside current flight bounds or obstructed: {scenario.scene}"
                )
            self.route = scenario.route
            return SampledState(
                position=torch.tensor(scenario.route.points[0], device=self.base.device)
                .expand(num, 3)
                .clone(),  # (N, 3)
                orientation=torch.tensor(scenario.orientation, device=self.base.device)
                .expand(num, 4)
                .clone(),  # (N, 4)
                velocity=torch.tensor(scenario.velocity, device=self.base.device).expand(num, 3).clone(),  # (N, 3)
                angular_velocity=torch.tensor(scenario.angular_velocity, device=self.base.device)
                .expand(num, 3)
                .clone(),  # (N, 3)
            )
        bounds = sampling_bounds(self.manager._scene_bounds[self.scene_id], self.clearance, self.height)
        if self.route_sampler is not None:
            self.route = self.route_sampler(self.scene_id)
            position = torch.tensor(self.route.points[0]).expand(num, 3).clone()  # (N, 3)
        else:
            position = sample_positions(
                lower=bounds.lower,
                upper=bounds.upper,
                count=num,
                collision=partial(
                    self.manager.get_point_is_collision, scene_id=self.scene_id, uav_radius=self.clearance
                ),
            )  # (N, 3)
        _, angles, velocity, angular_velocity = self.base.generate(num, **kwargs)  # each (N, 3)
        orientation = Quaternion.from_euler(*angles.T).toTensor().T  # (N, 4)
        return SampledState(
            position=position.to(self.base.device),  # (N, 3)
            orientation=orientation.to(self.base.device),  # (N, 4)
            velocity=velocity.to(self.base.device),  # (N, 3)
            angular_velocity=angular_velocity.to(self.base.device),  # (N, 3)
        )


class RGBNavigationEnv(NavigationEnv):
    """Expose RGB, optional depth, and quaternion state, including pre-autoreset observations."""

    def __init__(
        self,
        *,
        visual: bool = True,
        sensor_kwargs: list[dict[str, Any]] | None = None,
        scene_randomization: bool = False,
        scene_split: Literal["train", "test"] = "train",
        spawn_clearance: float = 0.3,
        flight_height: tuple[float, float] = (0.5, 2.5),
        goal_distance: tuple[float, float] = (2.0, 8.0),
        failure_penalty: float = 0.0,
        route_sampling: dict[str, Any] | None = None,
        scenario: dict[str, Any] | None = None,
        scene_path: str | None = None,
        compiled_dynamics: bool = True,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("dynamics_kwargs", {}).get("ori_output_type", "quaternion") != "quaternion":
            raise ValueError("RGBNavigationEnv requires ori_output_type='quaternion' for the upstream reward")
        if not 0 <= failure_penalty < float("inf"):
            raise ValueError("failure_penalty must be nonnegative and finite")
        if spawn_clearance <= 0 or not 0 <= flight_height[0] < flight_height[1]:
            raise ValueError(
                "spawn_clearance must be positive and flight_height must be an increasing nonnegative range"
            )
        if not 0.5 < goal_distance[0] < goal_distance[1]:
            raise ValueError("goal_distance must be an increasing range above the 0.5 m success radius")
        self.route_config = (
            TypeAdapter(RouteConfig).validate_python(route_sampling) if route_sampling is not None else None
        )
        self.scenario = TypeAdapter(Scenario).validate_python(scenario) if scenario is not None else None
        if (self.route_config is not None or self.scenario is not None) and not scene_randomization:
            raise ValueError("Route sampling and fixed scenarios require scene_randomization=True")
        self._route_catalogues: dict[int, RouteCatalogue] = {}
        self._initial_scene_objects: dict[int, tuple[Any, list[Any]]] = {}
        self._wind_settings = kwargs.get("dynamics_kwargs", {}).get("wind_settings", [0, 0, 0])
        self.route_statistics = RouteStatistics()
        self.scene_randomization = scene_randomization
        self.spawn_clearance = spawn_clearance
        self.flight_height = flight_height
        self.goal_distance = goal_distance
        self.failure_penalty = failure_penalty
        if not visual:
            raise ValueError("RGBNavigationEnv requires visual=True")
        if sensor_kwargs is None:
            sensor_kwargs = [{"sensor_type": "COLOR", "uuid": "color", "resolution": [64, 64]}]
        sensors = {sensor.get("uuid"): sensor for sensor in sensor_kwargs}
        allowed_types = {
            "color": ("COLOR", "color", SensorType.COLOR),
            "depth": ("DEPTH", "depth", SensorType.DEPTH),
        }
        if (
            len(sensors) != len(sensor_kwargs)
            or "color" not in sensors
            or any(
                uuid not in allowed_types or sensor.get("sensor_type") not in allowed_types[uuid]
                for uuid, sensor in sensors.items()
            )
        ):
            raise ValueError(
                "RGBNavigationEnv requires one COLOR sensor with uuid='color' and optional DEPTH with uuid='depth'"
            )
        self.has_depth = "depth" in sensors
        if self.has_depth:
            for key, default in (
                ("resolution", [128, 128]),
                ("position", [0, 0, -0.2]),
                ("orientation", [0, 0, 0]),
            ):
                if tuple(sensors["color"].get(key, default)) != tuple(sensors["depth"].get(key, default)):
                    raise ValueError(f"RGBNavigationEnv requires matching RGB/depth {key}")
        # Flight collision/route queries use mesh geometry, not Habitat's walking navmesh.
        kwargs["scene_kwargs"] = {"build_navmesh": False, **kwargs.get("scene_kwargs", {})}
        super().__init__(visual=True, sensor_kwargs=sensor_kwargs, **kwargs)
        with ExitStack() as resources:
            resources.enter_context(closing(self))
            if (
                compiled_dynamics
                and self.envs.dynamics._integrator == "euler"
                and torch.device(self.device).type == "cpu"
            ):
                self.envs.dynamics.step = CompiledEuler(self.envs.dynamics).step
            if spawn_clearance < self.envs.uav_radius:
                raise ValueError(
                    f"spawn_clearance must be at least the drone collision radius ({self.envs.uav_radius} m)"
                )
            self.observation_space.spaces["rgb"] = self.observation_space.spaces.pop("color")
            manager = self.envs.sceneManager
            loader = manager._scene_loader
            root = Path(manager.root_path)
            loader.paths = [
                path
                for path in loader.paths
                if any(
                    part == "test" or part.endswith("_test")
                    for part in (root.name, *Path(path).relative_to(root).parts[:-1])
                )
                == (scene_split == "test")
            ]
            if not loader.paths:
                raise ValueError(f"No scenes in the {scene_split} split under {root}")
            self._scene_paths = frozenset(loader.paths)
            if self.scenario is not None:
                scene_path = str(root / self.scenario.scene)
            self._pinned_scene = scene_path
            if scene_path is not None:
                if scene_path not in loader.paths:
                    raise ValueError(f"Requested scene is not in the {scene_split} split: {scene_path}")
                loader.paths = [scene_path]
            # Filter before the first reset, then cycle through the selected inventory.
            loader.paths = sorted(loader.paths)
            if loader.shuffle:
                Random(kwargs.get("seed", 42)).shuffle(loader.paths)
            loader._select_index = list(range(len(loader.paths)))
            if scene_randomization:
                self._rejected_scenes: dict[str, str] = {}
                self._load_unchecked_scene = manager._load_scene
                manager._load_scene = self._load_valid_scene
                self.envs.stateGenerators = [
                    SceneStateRandomizer(
                        base=base,
                        manager=manager,
                        scene_id=index // self.num_agent_per_scene,
                        clearance=spawn_clearance,
                        height=flight_height,
                        route_sampler=self._sample_route if self.route_config is not None else None,
                        scenario=self.scenario,
                    )
                    for index, base in enumerate(self.envs.stateGenerators)
                ]
            resources.pop_all()

    def configure_scenario(self, scenario: Scenario) -> None:
        """Select deterministic conditions, retaining loaded geometry within a scene."""
        manager = self.envs.sceneManager
        scene_path = str(Path(manager.root_path) / scenario.scene)
        if scene_path not in self._scene_paths:
            raise ValueError(f"Requested scene is not in the evaluation split: {scene_path}")
        changed = scene_path != self._pinned_scene or bool(manager.obj_settings)
        self.scenario = scenario
        self.max_episode_steps = scenario.max_episode_steps
        self._pinned_scene = scene_path
        loader = manager._scene_loader
        loader.paths = [scene_path]
        loader._select_index = [0]
        self.envs._scene_iter = False
        self.envs.set_seed(scenario.seed)
        self.envs.dynamics._t.zero_()
        self.envs.dynamics._create_wind(self._wind_settings)
        for generator in self.envs.stateGenerators:
            generator.scenario = scenario
            generator.route = None
        if changed:
            self._route_catalogues.clear()
            manager.load_scenes()
        for scene in manager.scenes:
            if scene is not None:
                scene.seed(scenario.seed)

    def _load_valid_scene(self, scene_path: str) -> Any:
        """Load a scene with nonempty spawn bounds, trying at most one inventory pass."""
        loader = self.envs.sceneManager._scene_loader
        for attempt in range(len(loader.paths) + 1):
            if scene_path not in self._rejected_scenes:
                with ExitStack() as resources:
                    scene = resources.enter_context(closing(self._load_unchecked_scene(scene_path)))
                    bounds = sampling_bounds(
                        scene.get_active_scene_graph().get_root_node().cumulative_bb,
                        self.spawn_clearance,
                        self.flight_height,
                    )
                    if torch.all(bounds.upper > bounds.lower):
                        resources.pop_all()
                        return scene
                    reason = (
                        f"{scene_path}: no sampling volume; lower={bounds.lower.tolist()}, "
                        f"upper={bounds.upper.tolist()}, clearance={self.spawn_clearance}, "
                        f"flight_height={self.flight_height}"
                    )
                    self._rejected_scenes[scene_path] = reason
                    if self._pinned_scene is not None:
                        raise ValueError(reason)
                    logging.getLogger(__name__).warning("Skipping VisFly scene %s", reason)
            if attempt < len(loader.paths):
                scene_path = loader.next(1)[0]
        raise ValueError("No compatible VisFly scenes remain. " + "; ".join(self._rejected_scenes.values()))

    def _route_collision(
        self, positions: NDArray[np.float32], radius: float, *, scene_id: int
    ) -> NDArray[np.bool_]:  # (N, 3) -> (N,)
        return collision_points(self.envs.sceneManager.scenes[scene_id], positions, radius)

    def route_catalogue(self, scene_id: int, config: RouteConfig) -> list[Route]:
        manager = self.envs.sceneManager
        scene = manager.scenes[scene_id]
        cached = self._route_catalogues.get(scene_id)
        if cached is not None and cached.scene is scene and cached.config == config:
            return cached.routes
        bounds = sampling_bounds(manager._scene_bounds[scene_id], self.spawn_clearance, self.flight_height)
        collision = partial(self._route_collision, scene_id=scene_id)
        graph = build_roadmap(bounds.lower, bounds.upper, collision, self.spawn_clearance, config)
        routes = build_routes(graph, collision, self.spawn_clearance, config)
        self._route_catalogues[scene_id] = RouteCatalogue(
            scene=scene, config=config, routes=routes, long_threshold=config.long_fraction * graph.extent
        )
        counts = {level: sum(route.difficulty == level for route in routes) for level in ("easy", "medium", "hard")}
        logging.getLogger(__name__).info("VisFly scene %s route catalogue: %s", scene_id, counts)
        return routes

    def _sample_route(self, scene_id: int) -> Route:
        config = self.route_config
        assert config is not None
        bounds = sampling_bounds(
            self.envs.sceneManager._scene_bounds[scene_id], self.spawn_clearance, self.flight_height
        )
        route = sample_route(
            self.route_catalogue(scene_id, config),
            partial(self._route_collision, scene_id=scene_id),
            self.spawn_clearance,
            config,
            bounds.lower,
            bounds.upper,
            long_threshold=self._route_catalogues[scene_id].long_threshold,
        )
        self.route_statistics.add(route)
        return route

    def sampling_metrics(self) -> dict[str, float]:
        metrics = self.route_statistics.metrics()
        self.route_statistics = RouteStatistics()
        return metrics

    def get_reward(
        self,
        predicted_obs: dict[str, torch.Tensor] | None = None,  # leaves: (B, *O)
    ) -> torch.Tensor:  # (B,)
        """Return (B,) navigation rewards with an optional one-time terminal failure cost.

        Zero preserves upstream rewards. Success overrides failure; timeouts and
        scene rotations alone incur no penalty. VisFly records rewards before autoreset.
        """
        reward = super().get_reward(predicted_obs=predicted_obs)  # (B,)
        if self.failure_penalty == 0:
            return reward  # (B,)
        # End flags are updated after get_reward; _episode_done describes the previous step.
        failure = (
            ~self._episode_done
            & ~self._success
            & (self._failure | self.is_out_bounds | (self.is_collision & self.is_collision_reset))
        )  # (B,)
        return reward - self.failure_penalty * failure  # (B,)

    def _randomize_targets(self, indices: torch.Tensor) -> None:  # (N,) agent indices
        manager = self.envs.sceneManager
        if self.scenario is not None or self.route_config is not None:
            targets = self.target.clone()  # (B, 3)
            for index in indices.tolist():
                route = self.envs.stateGenerators[index].route
                targets[index] = targets.new_tensor(route.points[-1])  # (3,)
            self.target = targets  # (B, 3)
            return
        # Preserve targets referenced by terminal observations before autoreset.
        targets = self.target.clone()  # (B, 3)
        scene_ids = indices // self.num_agent_per_scene  # (N,)
        for scene_id in scene_ids.unique().tolist():
            agents = indices[scene_ids == scene_id]  # (M,)
            bounds = sampling_bounds(manager._scene_bounds[scene_id], self.spawn_clearance, self.flight_height)
            targets[agents] = sample_positions(
                lower=bounds.lower,
                upper=bounds.upper,
                count=len(agents),
                collision=partial(manager.get_point_is_collision, scene_id=scene_id, uav_radius=self.spawn_clearance),
                reference=self.position[agents].cpu(),  # (M, 3)
                distance=self.goal_distance,
            ).to(targets.device)  # (M, 3)
        self.target = targets  # (B, 3)

    def reset(self, state: Any = None, **kwargs: Any) -> TensorDict:
        """Return batched observations after scene loading and start/goal randomization."""
        manager = self.envs.sceneManager
        # Fresh scenes start visual drone objects at their template poses. Restore
        # those poses before sensing a reused scene, rather than showing the last flight.
        for index, (scene, transforms) in self._initial_scene_objects.items():
            if manager.scenes[index] is scene:
                for drone, transform in zip(manager._drones[index], transforms, strict=True):
                    drone.root_scene_node.transformation = transform
        observations = super().reset(state=state, **kwargs)  # leaves: (B, *O)
        if manager.render_settings is not None or manager.is_multi_drone:
            for index, scene in enumerate(manager.scenes):
                cached = self._initial_scene_objects.get(index)
                if cached is None or cached[0] is not scene:
                    self._initial_scene_objects[index] = (
                        scene,
                        [drone.root_scene_node.transformation for drone in manager._drones[index]],
                    )
        if self.scene_randomization:
            self._randomize_targets(torch.arange(self.num_envs))
            observations["target"] = self.target.to(self.device)  # (B, 3); retain the existing camera snapshots
        return observations  # leaves: (B, *O)

    def reset_agent_by_id(
        self,
        agent_indices: torch.Tensor | None = None,  # (N,)
        state: Any = None,
        reset_obs: Any = None,
    ) -> TensorDict:
        """Return batched observations after randomizing the resetting agents' starts/goals."""
        observations = super().reset_agent_by_id(agent_indices=agent_indices, state=state, reset_obs=reset_obs)
        if self.scene_randomization:
            indices = torch.arange(self.num_envs) if agent_indices is None else agent_indices.cpu()  # (N,)
            self._randomize_targets(indices)
            observations["target"] = self.target.to(self.device)  # (B, 3)
        return observations  # leaves: (B, *O)

    def get_observation(
        self,
        indices: torch.Tensor | None = None,  # (N,) selected agents; upstream returns the full batch
        predicted_obs: dict[str, torch.Tensor] | None = None,  # leaves: (B, *O); unused by NavigationEnv
    ) -> TensorDict:
        """Return (B, S) state, (B, 3) target, (B, 3, H, W) RGB, and optional (B, 1, H, W) depth.

        Depth is float32 metres; upstream maps zero background readings to 20 m.
        Copy camera buffers because partial autoresets overwrite them in place.
        """
        observation = TensorDict(
            {
                "state": self.state.to(self.device),  # (B, S)
                "target": self.target.to(self.device),  # (B, 3)
                "rgb": torch.tensor(self.sensor_obs["color"], device=self.device),  # (B, 3, H, W)
            }
        )
        if self.has_depth:
            observation["depth"] = torch.tensor(self.sensor_obs["depth"], device=self.device)  # (B, 1, H, W)
        return observation

    def close(self) -> None:
        """Release dynamics and loaded scenes, including before the first successful reset."""
        self.envs.dynamics.close()
        # The first scene owns the shared GL context; release its dependents first.
        scenes = self.envs.sceneManager.scenes
        for index in reversed(range(len(scenes))):
            scene = scenes[index]
            if scene is not None:
                scene.close()
                scenes[index] = None
