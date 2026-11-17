"""Host-stepped Gymnasium vector environments with same-step autoreset."""

from contextlib import ExitStack, closing
from copy import deepcopy
from dataclasses import field
from functools import partial
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from numpy.typing import NDArray
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from rl2.envs.gym_style_env import (
    GymStyleEnv,
    GymStyleStep,
    Observation,
    convert_observation,
    host_array,
    host_observation,
    replace_observation,
)


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class GymConfig:
    id: Literal["Gymnasium"] = "Gymnasium"
    env_id: str = "CartPole-v1"
    kwargs: dict[str, Any] = field(default_factory=dict, hash=False)
    channels_first: bool = False


class GymEnv(GymStyleEnv):
    """Own a Gymnasium vector environment."""

    def __init__(self, config: GymConfig, num_envs: int, seed: int, *, record: bool = False) -> None:
        self.config = config
        self.num_envs = num_envs
        self.seed = seed
        kwargs = deepcopy(config.kwargs)
        if record:
            kwargs["render_mode"] = "rgb_array"
        with ExitStack() as resources:
            # Register each instance immediately so partial construction also cleans up.
            instances = [resources.enter_context(closing(gym.make(config.env_id, **kwargs))) for _ in range(num_envs)]
            self.backend = SyncVectorEnv(
                [partial(self._instance, env) for env in instances], autoreset_mode=AutoresetMode.SAME_STEP
            )
            self.action_space = self.backend.single_action_space
            resources.pop_all()

    @staticmethod
    def _instance(env: gym.Env) -> gym.Env:
        return env

    def reset(self) -> Observation[NDArray[Any]]:
        """Return (B, *O) initial observation leaves seeded on each reset."""
        observations, _ = self.backend.reset(seed=self.seed)  # leaves: (B, *O)
        return convert_observation(observations, self.config.channels_first)  # leaves: (B, *O)

    def step(self, actions: NDArray[Any]) -> GymStyleStep:  # (B, *action_shape)
        host_actions = np.array(actions, copy=True)  # (B, *action_shape)
        observations, rewards, terminated, truncated, infos = self.backend.step(host_actions)  # obs leaves: (B, *O)
        final = host_observation(observations)  # leaves: (B, *O)
        for index in np.flatnonzero(terminated | truncated):
            terminal = infos["final_obs"][index]  # leaves: (*O,)
            replace_observation(final, terminal, index=int(index))
        return GymStyleStep(
            observations=convert_observation(observations, self.config.channels_first),
            rewards=rewards.astype(np.float32, copy=False),  # (B,)
            terminated=terminated,  # (B,)
            truncated=truncated,  # (B,)
            final_observations=convert_observation(final, self.config.channels_first),
        )

    def render(self) -> NDArray[np.uint8]:
        """Return (H, W, 3) RGB pixels for a single evaluation environment."""
        frame = self.backend.envs[0].render()  # (H, W, C)
        frame = host_array(frame)  # (H, W, C) or (1, H, W, C)
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]  # (H, W, C)
        if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
            raise ValueError("Rendering requires an RGB array")
        return frame[..., :3].astype(np.uint8)  # (H, W, 3)

    def close(self) -> None:
        self.backend.close()
