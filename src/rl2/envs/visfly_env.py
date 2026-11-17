"""Host-stepped VisFly batches with tensor snapshots and timeout bootstrapping."""

from collections.abc import Mapping
from contextlib import ExitStack, closing
from copy import deepcopy
from dataclasses import field, replace
from typing import Annotated, Any, Literal

import numpy as np
from gymnasium.envs.registration import load_env_creator
from numpy.typing import NDArray
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.envs.gym_style_env import (
    GymStyleEnv,
    GymStyleStep,
    Observation,
    replace_observation,
)


def host_array(value: Any) -> NDArray[Any]:
    """Return (*S,) owned host data from a VisFly tensor or NumPy value."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()  # (*S,)
    return np.array(value, copy=True)  # (*S,)


def host_observation(value: Any) -> Observation[NDArray[Any]]:
    """Return (*S, *O) copied leaves before VisFly reuses its buffers."""
    if isinstance(value, Mapping):
        return {key: host_observation(leaf) for key, leaf in value.items()}
    if isinstance(value, tuple):
        return tuple(host_observation(leaf) for leaf in value)
    return host_array(value)  # (*S, *O)


def convert_observation(
    value: Observation[NDArray[Any]],  # leaves: (B, *O)
    channels_first: bool,
) -> Observation[NDArray[Any]]:
    """Return (B, *O) NumPy leaves, moving CHW images to HWC when requested."""
    if isinstance(value, dict):
        return {key: convert_observation(leaf, channels_first) for key, leaf in value.items()}
    if isinstance(value, tuple):
        return tuple(convert_observation(leaf, channels_first) for leaf in value)
    if channels_first and value.ndim == 4:
        return np.moveaxis(value, 1, -1)  # (B, H, W, C)
    return value  # (B, *O)


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class VisFlyConfig:
    id: Literal["VisFly"] = "VisFly"
    entry_point: str = "VisFly.envs.NavigationEnv:NavigationEnv"
    num_scenes: Annotated[int, Field(gt=0)] = 1
    kwargs: dict[str, Any] = field(default_factory=dict, hash=False)
    channels_first: bool = True


class VisFlyEnv(GymStyleEnv):
    """Own a VisFly autoresetting batch, seeded at construction."""

    def __init__(self, config: VisFlyConfig, num_envs: int, seed: int, *, evaluation: bool = False) -> None:
        if evaluation:
            config = replace(config, num_scenes=1)
        self.config = config
        self.num_envs = num_envs
        if num_envs % config.num_scenes:
            raise ValueError("num_envs must be divisible by env.num_scenes")
        reserved = {"num_scene", "num_agent_per_scene", "seed", "requires_grad"} & config.kwargs.keys()
        if reserved:
            raise ValueError(f"VisFly kwargs are managed by the trainer: {sorted(reserved)}")
        creator = load_env_creator(config.entry_point)
        with ExitStack() as resources:
            self.backend = resources.enter_context(
                closing(
                    creator(
                        num_scene=config.num_scenes,
                        num_agent_per_scene=num_envs // config.num_scenes,
                        seed=seed,
                        requires_grad=False,
                        **deepcopy(config.kwargs),
                    )
                )
            )
            self.action_space = self.backend.action_space
            if self.backend.num_envs != num_envs:
                raise ValueError("VisFly environment count does not match num_envs")
            resources.pop_all()

    def reset(self) -> Observation[NDArray[Any]]:
        """Return (B, *O) initial observation leaves."""
        observations = self.backend.reset()  # leaves: (B, *O)
        return convert_observation(host_observation(observations), self.config.channels_first)  # leaves: (B, *O)

    def step(self, actions: NDArray[Any]) -> GymStyleStep:  # (B, *action_shape)
        host_actions = np.array(actions, copy=True)  # (B, *action_shape)
        observations, rewards, done, infos = self.backend.step(host_actions)
        done = host_array(done).astype(bool)  # (B,)
        truncated = np.array(
            [done[i] and info.get("TimeLimit.truncated", False) for i, info in enumerate(infos)]
        )  # (B,)
        # VisFly can terminate on the same step as its horizon; termination wins.
        terminated = np.array(
            [done[i] and info.get("episode_done", not truncated[i]) for i, info in enumerate(infos)]
        )  # (B,)
        observations = host_observation(observations)  # leaves: (B, *O)
        final = host_observation(observations)  # leaves: (B, *O)
        for index in np.flatnonzero(done):
            if truncated[index] and not terminated[index] and "terminal_observation" not in infos[index]:
                raise ValueError("VisFly timeouts require terminal_observation")
            if "terminal_observation" in infos[index]:
                terminal = host_observation(infos[index]["terminal_observation"])  # leaves: (*O,)
                replace_observation(final, terminal, index=int(index))
        return GymStyleStep(
            observations=convert_observation(observations, self.config.channels_first),
            rewards=host_array(rewards).astype(np.float32, copy=False),  # (B,)
            terminated=host_array(terminated).astype(bool, copy=False),  # (B,)
            truncated=host_array(truncated).astype(bool, copy=False),  # (B,)
            final_observations=convert_observation(final, self.config.channels_first),
        )

    def render(self) -> NDArray[np.uint8]:  # (B, H, W, 3)
        """Return (B, H, W, 3) uint8 RGB pixels; agents share their scene's camera."""
        frames = host_array(self.backend.render())  # (S, H, W, C) or (H, W, C); S: scenes
        if frames.ndim == 3:
            frames = frames[None]  # (1, H, W, C)
        if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
            raise ValueError("Rendering requires RGB arrays; configure VisFly render settings")
        if frames.shape[0] != self.config.num_scenes:
            raise ValueError("VisFly rendering requires one RGB array per scene")
        frames = frames[..., :3].astype(np.uint8)  # (S, H, W, 3)
        return np.repeat(frames, self.num_envs // self.config.num_scenes, axis=0)  # (B, H, W, 3)

    def close(self) -> None:
        self.backend.close()
