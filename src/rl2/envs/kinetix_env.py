"""Single-environment Kinetix pixel observations and discrete actions for PPO."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Annotated, Literal

import jax
import jax.numpy as jnp
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.envs.base import BaseEnv, EnvStep

if TYPE_CHECKING:
    from kinetix.environment.env_state import EnvParams, EnvState


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class KinetixConfig:
    size: Literal["s", "m", "l"] = "s"
    # None samples procedural levels; otherwise sample uniformly from this list.
    levels: Annotated[tuple[Annotated[str, Field(min_length=1)], ...], Field(min_length=1, strict=False)] | None = None
    max_steps: Annotated[int, Field(gt=0, lt=2**31, strict=True)] = 256
    frame_skip: Annotated[int, Field(gt=0, strict=True)] = 2
    downscale: Literal[1, 2, 4] = 4
    dense_reward_scale: Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)] = 0.1


def encode_pixels(images: jax.Array) -> jax.Array:  # (..., W, H, 3) -> (..., 3, H, W)
    """Convert normalized, y-up Kinetix pixels to channel-first uint8 images."""
    pixels = jnp.rint(jnp.clip(images, 0, 1) * 255).astype(jnp.uint8)  # (..., W, H, 3)
    return jnp.moveaxis(pixels, (-1, -2, -3), (-3, -2, -1))[..., ::-1, :]  # (..., 3, H, W)


class KinetixEnv(BaseEnv["EnvState", "EnvParams"]):
    """Explicit transitions preserve final observations for timeout bootstrapping."""

    def __init__(self, config: KinetixConfig) -> None:
        try:
            from kinetix.environment import ActionType, EnvParams, ObservationType, make_kinetix_env
            from kinetix.environment.ued.ued import (
                make_reset_fn_list_of_levels,
                make_reset_fn_sample_kinetix_level,
            )
            from kinetix.environment.utils import static_env_params_from_size
        except ModuleNotFoundError as error:
            if error.name == "kinetix":
                raise ImportError("Kinetix requires the optional dependency: uv sync --extra kinetix") from error
            raise

        self.params = EnvParams(max_timesteps=config.max_steps, dense_reward_scale=config.dense_reward_scale)
        static_params = static_env_params_from_size(config.size).replace(
            frame_skip=config.frame_skip, downscale=config.downscale
        )
        if config.levels is None:
            reset_fn = make_reset_fn_sample_kinetix_level(self.params, static_params)
        else:
            reset_fn = make_reset_fn_list_of_levels(list(config.levels), static_params)
        self.env = make_kinetix_env(
            action_type=ActionType.DISCRETE,
            observation_type=ObservationType.PIXELS,
            reset_fn=reset_fn,
            env_params=self.params,
            static_env_params=static_params,
            auto_reset=False,
        )
        self.num_actions = self.env.action_space(self.params).n

    @partial(jax.jit, static_argnames="self")
    def reset(self, key: jax.Array, params: EnvParams) -> tuple[jax.Array, EnvState]:  # key: ()
        """Return (C, H, W) uint8 observation and (...) state leaves."""
        observation, state = self.env.reset(key, params)  # image: (W, H, 3); state leaves: (...)
        return encode_pixels(observation.image), state  # (C, H, W); state leaves: (...)

    @partial(jax.jit, static_argnames="self")
    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: EnvState,  # leaves: (...)
        action: jax.Array,  # ()
        params: EnvParams,
    ) -> EnvStep[EnvState]:
        """Advance and autoreset; info retains (C, H, W) final_observation."""
        step_key, reset_key = jax.random.split(key)  # each () typed PRNG key
        # Separate time limits from termination for PPO timeout bootstrapping.
        observation, state, reward, terminated, _ = self.env.step_env(
            step_key, state, action, params.replace(max_timesteps=2**31 - 1)
        )  # image: (W, H, 3); state leaves: (...); reward/terminated: ()
        final_observation = encode_pixels(observation.image)  # (C, H, W)
        truncated = state.timestep >= params.max_timesteps  # ()
        done = terminated | truncated  # ()
        final_state = state  # leaves: (...), before autoreset

        def reset_done(_: None) -> tuple[jax.Array, EnvState]:
            return self.reset(reset_key, params)  # (C, H, W); state leaves: (...)

        def continue_episode(_: None) -> tuple[jax.Array, EnvState]:
            return final_observation, state  # (C, H, W); state leaves: (...)

        observation, state = jax.lax.cond(done, reset_done, continue_episode, None)  # (C,H,W); leaves: (...)
        return EnvStep(
            observation,  # (C, H, W)
            state,  # leaves: (...)
            reward,  # ()
            done,  # ()
            {
                "terminated": terminated,  # ()
                "truncated": truncated,  # ()
                "final_observation": final_observation,  # (C, H, W)
                "final_state": final_state,  # leaves: (...)
            },
        )

    @staticmethod
    def render_observation(observation: jax.Array) -> jax.Array:  # (3, H, W) -> (H, W, 3)
        return jnp.moveaxis(observation, 0, -1)  # (H, W, 3)

    @partial(jax.jit, static_argnames="self")
    def render(self, state: EnvState) -> jax.Array:  # state leaves: (...) -> (H, W, 3)
        observation = encode_pixels(self.env.get_obs(state).image)  # (3, H, W)
        return self.render_observation(observation)  # (H, W, 3)
