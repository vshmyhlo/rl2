"""Navigation rewards and dynamics with RGB and optional aligned depth observations."""

import logging
from collections.abc import Callable
from contextlib import ExitStack, closing
from copy import copy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from random import Random
from typing import Any, Literal, NamedTuple

import numpy as np
import torch
from habitat_sim import SensorType
from pydantic import TypeAdapter
from VisFly.envs.NavigationEnv import NavigationEnv
from VisFly.utils.maths import Quaternion
from VisFly.utils.type import TensorDict

from rl2.envs.visfly.catalogues import CataloguePlanner, PlanningTask, planning_seed
from rl2.envs.visfly.collision import SceneCollision
from rl2.envs.visfly.dynamics import CompiledEuler
from rl2.envs.visfly.errors import NoSamplingVolumeError
from rl2.envs.visfly.planning import build_roadmap, build_routes, sample_route, segments_clear
from rl2.envs.visfly.planning_worker import plan_scene
from rl2.envs.visfly.power import NativePowerMeter, PowerConfig, PowerUsage
from rl2.envs.visfly.routes import Route, RouteConfig
from rl2.envs.visfly.sampling import sample_positions, sampling_bounds
from rl2.envs.visfly.scenarios import Scenario
from rl2.envs.visfly.sensors import SharedRgbdRenderer


class SampledState(NamedTuple):
    position: torch.Tensor  # (N, 3)
    orientation: torch.Tensor  # (N, 4)
    velocity: torch.Tensor  # (N, 3)
    angular_velocity: torch.Tensor  # (N, 3)


@dataclass
class RouteCatalogue:
    scene: Any
    config: RouteConfig
    training: bool
    routes: list[Route]
    long_threshold: float


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

            collision = SceneCollision(scene=self.manager.scenes[self.scene_id])
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
        if self.route_sampler is not None:
            self.route = self.route_sampler(self.scene_id)
            position = torch.tensor(self.route.points[0]).expand(num, 3).clone()  # (N, 3)
        else:
            bounds = sampling_bounds(self.manager._scene_bounds[self.scene_id], self.clearance, self.height)
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
        power: dict[str, Any] | None = None,
        route_sampling: dict[str, Any] | None = None,
        scenario: dict[str, Any] | None = None,
        scene_path: str | None = None,
        compiled_dynamics: bool = True,
        shared_rgbd: bool = True,
        planning_workers: int = 0,
        planning_prefetch: bool = True,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("dynamics_kwargs", {}).get("ori_output_type", "quaternion") != "quaternion":
            raise ValueError("RGBNavigationEnv requires ori_output_type='quaternion' for the upstream reward")
        if not 0 <= failure_penalty < float("inf"):
            raise ValueError("failure_penalty must be nonnegative and finite")
        if not 0 < spawn_clearance < float("inf"):
            raise ValueError("spawn_clearance must be positive and finite")
        if not 0 <= flight_height[0] < flight_height[1]:
            raise ValueError("flight_height must be an increasing nonnegative range")
        if not 0.5 < goal_distance[0] < goal_distance[1]:
            raise ValueError("goal_distance must be an increasing range above the 0.5 m success radius")
        self.route_config = (
            TypeAdapter(RouteConfig).validate_python(route_sampling) if route_sampling is not None else None
        )
        self.scenario = TypeAdapter(Scenario).validate_python(scenario) if scenario is not None else None
        if (self.route_config is not None or self.scenario is not None) and not scene_randomization:
            raise ValueError("Route sampling and fixed scenarios require scene_randomization=True")
        if planning_workers < 0:
            raise ValueError("planning_workers must be nonnegative")
        self._planner: CataloguePlanner | None = None
        self._planning_generation = 0
        self._planning_prefetch = planning_prefetch
        self._route_catalogues: dict[int, RouteCatalogue] = {}
        self._initial_scene_objects: dict[int, tuple[Any, list[Any]]] = {}
        self._wind_settings = kwargs.get("dynamics_kwargs", {}).get("wind_settings", [0, 0, 0])
        self.sampled_routes: list[Route] = []
        self.scene_randomization = scene_randomization
        self.spawn_clearance = spawn_clearance
        self.flight_height = flight_height
        self.goal_distance = goal_distance
        self.failure_penalty = failure_penalty
        self.power_config = TypeAdapter(PowerConfig).validate_python(power) if power is not None else None
        self._power_meter: CompiledEuler | NativePowerMeter | None = None
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
        if self.scenario is not None:
            kwargs["seed"] = self.scenario.seed
            kwargs["max_episode_steps"] = self.scenario.max_episode_steps
        super().__init__(visual=True, sensor_kwargs=sensor_kwargs, **kwargs)
        with ExitStack() as resources:
            resources.enter_context(closing(self))
            if (
                compiled_dynamics
                and not kwargs.get("requires_grad", False)
                and self.envs.dynamics._integrator == "euler"
                and torch.device(self.device).type == "cpu"
            ):
                integrator = CompiledEuler(self.envs.dynamics, power=self.power_config)
                self.envs.dynamics.step = integrator.step
                if self.power_config is not None:
                    self._power_meter = integrator
            elif self.power_config is not None:
                self._power_meter = NativePowerMeter(self.envs.dynamics, self.power_config)
                self.envs.dynamics.step = self._power_meter.step
            if spawn_clearance < self.envs.uav_radius:
                raise ValueError(
                    f"spawn_clearance must be at least the drone collision radius ({self.envs.uav_radius} m)"
                )
            self.observation_space.spaces["rgb"] = self.observation_space.spaces.pop("color")
            manager = self.envs.sceneManager
            if shared_rgbd and self.has_depth:
                SharedRgbdRenderer(manager)
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
            if planning_workers and self.route_config is not None and self.scenario is None:
                if manager.obj_settings:
                    raise ValueError("Parallel route planning requires static scene geometry")
                self._planner = CataloguePlanner(workers=planning_workers, worker=plan_scene)
                self._load_unplanned_scenes = manager.load_scenes
                manager.load_scenes = self._load_planned_scenes
            resources.pop_all()

    def _planning_task(self, scene_id: int, path: str) -> PlanningTask:
        manager = self.envs.sceneManager
        assert self.route_config is not None
        return PlanningTask(
            scene_path=path,
            dataset_path=manager._datasets_path,
            seed=planning_seed(manager.seed, self._planning_generation, scene_id),
            clearance=self.spawn_clearance,
            height=tuple(self.flight_height),
            config=self.route_config,
            scene_seed=manager.seed,
        )

    def _submit_next_scenes(self) -> None:
        assert self._planner is not None
        manager = self.envs.sceneManager
        # Peek without advancing the loader, including at inventory wraparound.
        paths = copy(manager._scene_loader).next(manager.num_scene)
        self._planner.submit(
            [
                self._planning_task(scene_id, path)
                for scene_id, path in enumerate(paths)
                if path not in self._rejected_scenes
            ]
        )

    def _load_planned_scenes(self, indices: Any = None) -> None:
        """Overlap fresh planning with scene loading; prefetch one future rotation."""
        assert self._planner is not None and self.route_config is not None
        manager = self.envs.sceneManager
        if indices is None:
            self._submit_next_scenes()
        self._load_unplanned_scenes(indices=indices)
        selected = range(manager.num_scene) if indices is None else np.atleast_1d(indices).tolist()
        tasks = {
            scene_id: self._planning_task(scene_id, manager.scenes[scene_id].config.sim_cfg.scene_id)
            for scene_id in selected
        }
        # A rejected scene may have advanced the loader beyond the prefetched set.
        self._planner.submit(list(tasks.values()))
        for scene_id, task in tasks.items():
            result = self._planner.take(task)
            self._route_catalogues[scene_id] = RouteCatalogue(
                scene=manager.scenes[scene_id],
                config=self.route_config,
                training=True,
                routes=result.routes,
                long_threshold=result.long_threshold,
            )
        self._planner.discard()
        self._planning_generation += 1
        if self._planning_prefetch and self.envs._scene_iter:
            self._submit_next_scenes()

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
                        raise NoSamplingVolumeError(reason)
                    logging.getLogger(__name__).warning("Skipping VisFly scene %s", reason)
            if attempt < len(loader.paths):
                scene_path = loader.next(1)[0]
        raise NoSamplingVolumeError("No compatible VisFly scenes remain. " + "; ".join(self._rejected_scenes.values()))

    def route_catalogue(self, scene_id: int, config: RouteConfig, *, training: bool = False) -> list[Route]:
        """Cache scene routes; training omits passage labels and path shortcutting."""
        manager = self.envs.sceneManager
        scene = manager.scenes[scene_id]
        cached = self._route_catalogues.get(scene_id)
        if cached is not None and cached.scene is scene and cached.config == config and cached.training == training:
            return cached.routes
        bounds = sampling_bounds(manager._scene_bounds[scene_id], self.spawn_clearance, self.flight_height)
        collision = SceneCollision(scene=scene)
        graph = build_roadmap(
            bounds.lower, bounds.upper, collision, self.spawn_clearance, config, classify_passages=not training
        )
        routes = build_routes(graph, collision, self.spawn_clearance, config, shortcut=not training)
        self._route_catalogues[scene_id] = RouteCatalogue(
            scene=scene,
            config=config,
            training=training,
            routes=routes,
            long_threshold=config.long_fraction * graph.extent,
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
            self.route_catalogue(scene_id, config, training=True),
            SceneCollision(scene=self.envs.sceneManager.scenes[scene_id]),
            self.spawn_clearance,
            config,
            bounds.lower,
            bounds.upper,
            long_threshold=self._route_catalogues[scene_id].long_threshold,
        )
        self.sampled_routes.append(route)
        return route

    @property
    def power_usage(self) -> PowerUsage[torch.Tensor] | None:
        """Return (B,) consumption leaves for the last action, including terminal substeps."""
        return self._power_meter.power_usage if self._power_meter is not None else None

    def get_reward(
        self,
        predicted_obs: dict[str, torch.Tensor] | None = None,  # leaves: (B, *O)
    ) -> torch.Tensor:  # (B,)
        """Return (B,) navigation rewards minus optional consumption and terminal failure costs.

        Consumption includes terminal actions. Success overrides the failure cost;
        timeouts and scene rotations alone incur no failure cost. Zero weights
        preserve upstream rewards. VisFly records rewards before autoreset.
        """
        reward = super().get_reward(predicted_obs=predicted_obs)  # (B,)
        usage = self.power_usage
        if self.power_config is not None and usage is not None:
            reward = (
                reward
                - self.power_config.energy_weight * usage.energy
                - self.power_config.excess_power_weight * usage.excess
            )  # (B,)
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
        # Preserve targets referenced by terminal observations before autoreset.
        targets = self.target.clone()  # (B, 3)
        if self.scenario is not None or self.route_config is not None:
            for index in indices.tolist():
                route = self.envs.stateGenerators[index].route
                targets[index] = targets.new_tensor(route.points[-1])  # (3,)
            self.target = targets  # (B, 3)
            return
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
        if self._power_meter is not None:
            self._power_meter.power_usage = None
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
        if self._planner is not None:
            self._planner.close()
            self._planner = None
        self.envs.dynamics.close()
        # The first scene owns the shared GL context; release its dependents first.
        scenes = self.envs.sceneManager.scenes
        for index in reversed(range(len(scenes))):
            scene = scenes[index]
            if scene is not None:
                scene.close()
                scenes[index] = None
