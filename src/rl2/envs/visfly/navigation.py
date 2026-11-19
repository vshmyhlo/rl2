"""Navigation rewards and dynamics with RGB and optional aligned depth observations."""

import logging
from contextlib import ExitStack, closing
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from random import Random
from typing import Any, Literal, NamedTuple

import numpy as np
import torch
from habitat_sim import SensorType
from VisFly.envs.NavigationEnv import NavigationEnv
from VisFly.utils.common import habitat_to_std
from VisFly.utils.maths import Quaternion
from VisFly.utils.type import TensorDict

from rl2.envs.visfly.sampling import sample_positions


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
class SceneStateRandomizer:
    base: Any
    manager: Any
    scene_id: int
    clearance: float
    height: tuple[float, float]

    def safe_generate(self, num: int = 1, **kwargs: Any) -> SampledState:
        """Return (N, 3) position/velocities and (N, 4) quaternion orientation."""
        bounds = sampling_bounds(self.manager._scene_bounds[self.scene_id], self.clearance, self.height)
        position = sample_positions(
            lower=bounds.lower,
            upper=bounds.upper,
            count=num,
            collision=partial(self.manager.get_point_is_collision, scene_id=self.scene_id, uav_radius=self.clearance),
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
        super().__init__(visual=True, sensor_kwargs=sensor_kwargs, **kwargs)
        with ExitStack() as resources:
            resources.enter_context(closing(self))
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
                    )
                    for index, base in enumerate(self.envs.stateGenerators)
                ]
            resources.pop_all()

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
                    logging.getLogger(__name__).warning("Skipping VisFly scene %s", reason)
            if attempt < len(loader.paths):
                scene_path = loader.next(1)[0]
        raise ValueError("No compatible VisFly scenes remain. " + "; ".join(self._rejected_scenes.values()))

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
        observations = super().reset(state=state, **kwargs)  # leaves: (B, *O)
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
