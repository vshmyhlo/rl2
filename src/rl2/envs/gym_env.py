"""Host-stepped Gymnasium and VisFly environments.

Reset/step consume and return NumPy arrays. B is the environment count;
observation leaves retain their space shapes, with image channels moved last.
"""

from collections.abc import Mapping
from contextlib import ExitStack, closing
from copy import deepcopy
from dataclasses import field
from functools import partial
from typing import Annotated, Any, Literal, cast

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import load_env_creator
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from numpy.typing import NDArray
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.envs.gym_style_env import GymStyleEnv, GymStyleStep, Observation


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class GymConfig:
    id: Literal["Gymnasium"] = "Gymnasium"
    env_id: str = "CartPole-v1"
    kwargs: dict[str, Any] = field(default_factory=dict, hash=False)
    channels_first: bool = False


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class VisFlyConfig:
    id: Literal["VisFly"] = "VisFly"
    entry_point: str = "VisFly.envs.NavigationEnv:NavigationEnv"
    num_scenes: Annotated[int, Field(gt=0)] = 1
    kwargs: dict[str, Any] = field(default_factory=dict, hash=False)
    channels_first: bool = True


type EnvConfig = Annotated[GymConfig | VisFlyConfig, Field(discriminator="id")]


def host_array(value: Any) -> NDArray[Any]:
    """Return (*S,) owned host data from a NumPy value or simulator tensor."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)  # (*S,)


def host_observation(value: Any) -> Observation:
    """Return (*S, *O) copied leaves before a simulator reuses its buffers."""
    if isinstance(value, Mapping):
        return {key: host_observation(leaf) for key, leaf in value.items()}
    if isinstance(value, tuple):
        return tuple(host_observation(leaf) for leaf in value)
    return host_array(value)  # (*S, *O)


def convert_observation(value: Observation, channels_first: bool) -> Observation:
    """Return (B, *O) NumPy leaves, moving CHW images to HWC when requested."""
    if isinstance(value, dict):
        return {key: convert_observation(leaf, channels_first) for key, leaf in value.items()}
    if isinstance(value, tuple):
        return tuple(convert_observation(leaf, channels_first) for leaf in value)
    if channels_first and value.ndim == 4:
        return np.moveaxis(value, 1, -1)  # (B, H, W, C)
    return value  # (B, *O)


def replace_observation(
    batch: Observation,  # leaves: (B, *O)
    observation: Observation,  # leaves: (*O,)
    *,
    index: int,
) -> None:
    """Replace one batch entry in place, preserving dict and tuple structure."""
    if isinstance(batch, dict):
        observation = cast(dict[str, Observation], observation)
        for key in batch:
            replace_observation(batch[key], observation[key], index=index)
    elif isinstance(batch, tuple):
        observation = cast(tuple[Observation, ...], observation)
        for leaf, replacement in zip(batch, observation, strict=True):
            replace_observation(leaf, replacement, index=index)
    else:
        batch[index] = cast(NDArray[Any], observation)


class GymEnv(GymStyleEnv):
    """Own a Gymnasium vector environment or a VisFly autoresetting batch."""

    def __init__(self, config: EnvConfig, num_envs: int, seed: int, *, record: bool = False) -> None:
        self.config = config
        self.num_envs = num_envs
        self.seed = seed
        # Constructors can fail after acquiring renderer or environment resources.
        with ExitStack() as resources:
            if isinstance(config, VisFlyConfig):
                if num_envs % config.num_scenes:
                    raise ValueError("num_envs must be divisible by env.num_scenes")
                reserved = {"num_scene", "num_agent_per_scene", "seed", "requires_grad"} & config.kwargs.keys()
                if reserved:
                    raise ValueError(f"VisFly kwargs are managed by the trainer: {sorted(reserved)}")
                creator = load_env_creator(config.entry_point)
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
                space = self.backend.action_space
                if self.backend.num_envs != num_envs:
                    raise ValueError("VisFly environment count does not match num_envs")
            else:
                kwargs = deepcopy(config.kwargs)
                if record:
                    kwargs["render_mode"] = "rgb_array"
                # Register each instance immediately so partial construction also cleans up.
                instances = [
                    resources.enter_context(closing(gym.make(config.env_id, **kwargs))) for _ in range(num_envs)
                ]
                self.backend = SyncVectorEnv(
                    [partial(self._instance, env) for env in instances], autoreset_mode=AutoresetMode.SAME_STEP
                )
                space = self.backend.single_action_space
            self.action_space = space
            resources.pop_all()

    @staticmethod
    def _instance(env: gym.Env) -> gym.Env:
        return env

    def reset(self) -> Observation:
        """Return (B, *O) initial observations; VisFly is seeded at construction."""
        if isinstance(self.config, VisFlyConfig):
            observations = self.backend.reset()
        else:
            observations, _ = self.backend.reset(seed=self.seed)
        return convert_observation(host_observation(observations), self.config.channels_first)

    def step(self, actions: NDArray[Any]) -> GymStyleStep:  # (B, *action_shape)
        host_actions = np.array(actions, copy=True)  # (B, *action_shape)
        if isinstance(self.config, VisFlyConfig):
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
        else:
            observations, rewards, terminated, truncated, infos = self.backend.step(host_actions)
            observations = host_observation(observations)  # leaves: (B, *O)
            final = host_observation(observations)  # leaves: (B, *O)
            for index in np.flatnonzero(terminated | truncated):
                terminal = host_observation(infos["final_obs"][index])  # leaves: (*O,)
                replace_observation(final, terminal, index=int(index))
        return GymStyleStep(
            observations=convert_observation(observations, self.config.channels_first),
            rewards=host_array(rewards).astype(np.float32, copy=False),  # (B,)
            terminated=host_array(terminated).astype(bool, copy=False),  # (B,)
            truncated=host_array(truncated).astype(bool, copy=False),  # (B,)
            final_observations=convert_observation(final, self.config.channels_first),
        )

    def render(self) -> NDArray[np.uint8]:
        """Return (H, W, 3) RGB pixels for a single evaluation environment."""
        frame = self.backend.render() if isinstance(self.config, VisFlyConfig) else self.backend.envs[0].render()
        frame = host_array(frame)  # (H, W, C) or (1, H, W, C)
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]  # (H, W, C)
        if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
            raise ValueError("Rendering requires an RGB array; configure VisFly render settings")
        return frame[..., :3].astype(np.uint8)  # (H, W, 3)

    def close(self) -> None:
        self.backend.close()
