"""Host-stepped Gymnasium and VisFly environments for nonnative PPO.

Reset/step run in Python, outside JAX compilation. B is the environment count;
observation leaves retain their space shapes, with image channels moved last.
"""

from collections.abc import Mapping
from contextlib import ExitStack, closing
from copy import deepcopy
from dataclasses import field
from functools import partial
from typing import Annotated, Any, Literal

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.envs.registration import load_env_creator
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from numpy.typing import NDArray
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.actions import ActionSpace, ContinuousSpace, DiscreteSpace
from rl2.envs.nonnative import BaseNonnativeEnv, Observation
from rl2.envs.nonnative import NonnativeStep as GymStep


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
    render_key: str | None = None


type EnvConfig = Annotated[GymConfig | VisFlyConfig, Field(discriminator="id")]


def host_array(value: Any) -> NDArray[Any]:
    """Return an owned host array from a NumPy value or simulator tensor."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)  # (*S,)


def host_observation(value: Any) -> Any:
    """Copy observation leaves before a stateful simulator reuses its buffers."""
    if isinstance(value, Mapping):
        return {key: host_observation(leaf) for key, leaf in value.items()}
    if isinstance(value, tuple):
        return tuple(host_observation(leaf) for leaf in value)
    return host_array(value)  # (*S, *O)


def convert_observation(value: Any, channels_first: bool) -> Observation:
    """Convert (B, *O) leaves to JAX, including CHW images to HWC."""

    def convert(leaf: NDArray[Any]) -> jax.Array:  # (B, *O)
        array = jnp.asarray(leaf)  # (B, *O)
        if channels_first and array.ndim == 4:
            array = jnp.moveaxis(array, 1, -1)  # (B, H, W, C)
        return array  # (B, *O)

    return jax.tree.map(convert, value)  # leaves: (B, *O)


def replace_observation(
    batch: NDArray[Any],  # (B, *O)
    leaf: NDArray[Any],  # (*O,)
    *,
    index: int,
) -> NDArray[Any]:
    batch[index] = leaf
    return batch  # (B, *O)


def action_space(space: gym.Space) -> ActionSpace:
    if isinstance(space, gym.spaces.Discrete):
        if space.start != 0:
            raise ValueError("Discrete actions must start at zero")
        return DiscreteSpace(n=int(space.n))
    if isinstance(space, gym.spaces.Box) and np.issubdtype(space.dtype, np.floating):
        return ContinuousSpace(
            shape=space.shape,
            low=tuple(float(x) for x in space.low.ravel()),
            high=tuple(float(x) for x in space.high.ravel()),
        )
    raise ValueError("Supported action spaces are zero-based Discrete and bounded floating-point Box")


class GymEnv(BaseNonnativeEnv):
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
            self.action_space = action_space(space)
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

    def step(self, actions: jax.Array) -> GymStep:  # (B, *action_shape)
        host_actions = np.array(jax.device_get(actions), copy=True)  # (B, *action_shape)
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
            final = jax.tree.map(np.copy, observations)  # leaves: (B, *O)
            for index in np.flatnonzero(done):
                if truncated[index] and not terminated[index] and "terminal_observation" not in infos[index]:
                    raise ValueError("VisFly timeouts require terminal_observation")
                if "terminal_observation" in infos[index]:
                    terminal = host_observation(infos[index]["terminal_observation"])  # leaves: (*O,)

                    final = jax.tree.map(partial(replace_observation, index=int(index)), final, terminal)
        else:
            observations, rewards, terminated, truncated, infos = self.backend.step(host_actions)
            observations = host_observation(observations)  # leaves: (B, *O)
            final = jax.tree.map(np.copy, observations)  # leaves: (B, *O)
            for index in np.flatnonzero(terminated | truncated):
                terminal = host_observation(infos["final_obs"][index])  # leaves: (*O,)

                final = jax.tree.map(partial(replace_observation, index=int(index)), final, terminal)
        return GymStep(
            observations=convert_observation(observations, self.config.channels_first),
            rewards=jnp.asarray(host_array(rewards), dtype=jnp.float32),  # (B,)
            terminated=jnp.asarray(terminated, dtype=bool),  # (B,)
            truncated=jnp.asarray(truncated, dtype=bool),  # (B,)
            final_observations=convert_observation(final, self.config.channels_first),
        )

    def render(self) -> NDArray[np.uint8]:
        """Return (H, W, 3) RGB pixels for a single evaluation environment."""
        frame = self.backend.render() if isinstance(self.config, VisFlyConfig) else self.backend.envs[0].render()
        if isinstance(self.config, VisFlyConfig) and self.config.render_key is not None:
            frame = frame[self.config.render_key]
        frame = host_array(frame)  # (H, W, C) or (1, H, W, C)
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]  # (H, W, C)
        if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
            raise ValueError("Rendering requires an RGB array; configure VisFly render settings/render_key")
        return frame[..., :3].astype(np.uint8)  # (H, W, 3)

    def close(self) -> None:
        self.backend.close()
