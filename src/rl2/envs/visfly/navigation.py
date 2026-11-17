"""Navigation rewards and dynamics with an RGB-only camera observation."""

from dataclasses import dataclass
from functools import partial
from random import Random
from typing import Any, NamedTuple

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


def sampling_bounds(manager: Any, scene_id: int, clearance: float, height: tuple[float, float]) -> SamplingBounds:
    """Return (3,) lower/upper bounds, with height measured above the scene AABB floor."""
    bounds = manager._scene_bounds[scene_id]
    corners, _ = habitat_to_std(np.array([bounds.min, bounds.max]))  # (2, 3)
    lower = corners.amin(dim=0) + clearance  # (3,)
    upper = corners.amax(dim=0) - clearance  # (3,)
    floor = corners[:, 2].min()  # ()
    lower[2] = torch.maximum(lower[2], floor + height[0])
    upper[2] = torch.minimum(upper[2], floor + height[1])
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
        bounds = sampling_bounds(self.manager, self.scene_id, self.clearance, self.height)
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
    """Expose one color sensor as RGB, including pre-autoreset observations."""

    def __init__(
        self,
        *,
        visual: bool = True,
        sensor_kwargs: list[dict[str, Any]] | None = None,
        scene_randomization: bool = False,
        spawn_clearance: float = 0.3,
        flight_height: tuple[float, float] = (0.5, 2.5),
        goal_distance: tuple[float, float] = (2.0, 8.0),
        **kwargs: Any,
    ) -> None:
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
        if not visual:
            raise ValueError("RGBNavigationEnv requires visual=True")
        if sensor_kwargs is None:
            sensor_kwargs = [{"sensor_type": "COLOR", "uuid": "color", "resolution": [64, 64]}]
        if (
            len(sensor_kwargs) != 1
            or sensor_kwargs[0].get("sensor_type") not in ("COLOR", "color", SensorType.COLOR)
            or sensor_kwargs[0].get("uuid") != "color"
        ):
            raise ValueError("RGBNavigationEnv requires one COLOR sensor with uuid='color'")
        super().__init__(visual=True, sensor_kwargs=sensor_kwargs, **kwargs)
        self.observation_space.spaces["rgb"] = self.observation_space.spaces.pop("color")
        if scene_randomization:
            manager = self.envs.sceneManager
            # Cycle through a reproducible permutation of the complete recursive scene inventory.
            loader = manager._scene_loader
            loader.paths = sorted(loader.paths)
            if loader.shuffle:
                Random(kwargs.get("seed", 42)).shuffle(loader.paths)
            loader._select_index = list(range(len(loader.paths)))
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

    def _randomize_targets(self, indices: torch.Tensor) -> None:  # (N,) agent indices
        manager = self.envs.sceneManager
        # Preserve targets referenced by terminal observations before autoreset.
        targets = self.target.clone()  # (B, 3)
        scene_ids = indices // self.num_agent_per_scene  # (N,)
        for scene_id in scene_ids.unique().tolist():
            agents = indices[scene_ids == scene_id]  # (M,)
            bounds = sampling_bounds(manager, scene_id, self.spawn_clearance, self.flight_height)
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
            observations = self.get_full_observation()  # leaves: (B, *O)
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
            observations = self.get_full_observation()  # leaves: (B, *O)
        return observations  # leaves: (B, *O)

    def get_observation(
        self,
        indices: torch.Tensor | None = None,  # (N,) selected agents; upstream returns the full batch
        predicted_obs: dict[str, torch.Tensor] | None = None,  # leaves: (B, *O); unused by NavigationEnv
    ) -> TensorDict:
        """Return (B, S) state, (B, 3) target, and (B, 3, H, W) uint8 RGB."""
        return TensorDict(
            {
                "state": self.state.to(self.device),  # (B, S)
                "target": self.target.to(self.device),  # (B, 3)
                # Partial autoresets overwrite the NumPy sensor buffer in place.
                "rgb": torch.tensor(self.sensor_obs["color"], device=self.device),  # (B, 3, H, W)
            }
        )
