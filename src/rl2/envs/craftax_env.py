"""Single-environment Craftax and Craftax-Classic pixel environments for PPO."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Annotated, Literal

import chex
import jax
import jax.numpy as jnp
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.envs.gymnax_env import EnvStep

if TYPE_CHECKING:
    from craftax.craftax.craftax_state import EnvParams
    from craftax.craftax_classic.envs.craftax_state import EnvParams as ClassicEnvParams

    type CraftaxParams = EnvParams | ClassicEnvParams


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class CraftaxConfig:
    env_id: Literal["Craftax-Pixels-v1", "Craftax-Classic-Pixels-v1"] = "Craftax-Pixels-v1"
    max_steps: Annotated[int, Field(gt=0, lt=2**31, strict=True)] | None = None


def encode_pixels(images: jax.Array) -> jax.Array:  # (..., H, W, 3) -> (..., 3, H, W)
    pixels = jnp.rint(jnp.clip(images, 0, 1) * 255).astype(jnp.uint8)  # (..., H, W, 3)
    return jnp.moveaxis(pixels, -1, -3)  # (..., 3, H, W)


class CraftaxEnv:
    def __init__(self, config: CraftaxConfig) -> None:
        try:
            from craftax.craftax_env import make_craftax_env_from_name
        except ModuleNotFoundError as error:
            if error.name == "craftax":
                raise ImportError("Craftax requires the optional dependency: uv sync --extra craftax") from error
            raise

        self.env = make_craftax_env_from_name(config.env_id, auto_reset=False)
        self.params = self.env.default_params
        if config.max_steps is not None:
            self.params = self.params.replace(max_timesteps=config.max_steps)
        self.num_actions = self.env.num_actions

    @partial(jax.jit, static_argnames="self")
    def reset(self, key: jax.Array, params: CraftaxParams) -> tuple[jax.Array, chex.ArrayTree]:  # key: ()
        """Return (C, H, W) uint8 observation and (...) state leaves."""
        observation, state = self.env.reset(key, params)  # (H, W, 3); state leaves: (...)
        return encode_pixels(observation), state  # (C, H, W); state leaves: (...)

    @partial(jax.jit, static_argnames="self")
    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: chex.ArrayTree,  # leaves: (...)
        action: jax.Array,  # ()
        params: CraftaxParams,
    ) -> EnvStep[chex.ArrayTree]:
        """Advance and autoreset; info retains (C, H, W) final_observation."""
        step_key, reset_key = jax.random.split(key)  # each () typed PRNG key
        # Separate time limits from termination for PPO timeout bootstrapping.
        observation, state, reward, terminated, _ = self.env.step_env(
            step_key, state, action, params.replace(max_timesteps=2**31 - 1)
        )  # (H, W, 3); state leaves: (...); reward/terminated: ()
        final_observation = encode_pixels(observation)  # (C, H, W)
        truncated = state.timestep >= params.max_timesteps  # ()
        done = terminated | truncated  # ()

        def reset_done(_: None) -> tuple[jax.Array, chex.ArrayTree]:
            return self.reset(reset_key, params)  # (C, H, W); state leaves: (...)

        def continue_episode(_: None) -> tuple[jax.Array, chex.ArrayTree]:
            return final_observation, state  # (C, H, W); state leaves: (...)

        observation, state = jax.lax.cond(done, reset_done, continue_episode, None)  # (C,H,W); leaves: (...)
        return (
            observation,  # (C, H, W)
            state,  # leaves: (...)
            reward,  # ()
            done,  # ()
            {
                "terminated": terminated,  # ()
                "truncated": truncated,  # ()
                "final_observation": final_observation,  # (C, H, W)
            },
        )

    @staticmethod
    def render_observation(observation: jax.Array) -> jax.Array:  # (3, H, W) -> (H, W, 3)
        return jnp.moveaxis(observation, 0, -1)  # (H, W, 3)
