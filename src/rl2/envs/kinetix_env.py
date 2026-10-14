"""Single-environment Kinetix symbolic observations and discrete or continuous actions for PPO."""

from functools import partial
from typing import Annotated, Literal

import jax
import jax.numpy as jnp
from kinetix.environment import ActionType, ObservationType, make_kinetix_env
from kinetix.environment.env_state import EnvParams, EnvState
from kinetix.environment.ued.ued import make_reset_fn_list_of_levels, make_reset_fn_sample_kinetix_level
from kinetix.environment.utils import static_env_params_from_size
from kinetix.render.renderer_pixels import make_render_pixels
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.actions import ContinuousSpace, DiscreteSpace
from rl2.envs.base import BaseEnv, EnvStep


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class KinetixConfig:
    id: Literal["Kinetix"] = "Kinetix"
    action_type: Literal["discrete", "continuous"] = "discrete"
    size: Literal["s", "m", "l"] = "s"
    # None samples procedural levels; otherwise sample uniformly from this list.
    levels: Annotated[tuple[Annotated[str, Field(min_length=1)], ...], Field(min_length=1, strict=False)] | None = None
    max_steps: Annotated[int, Field(gt=0, lt=2**31, strict=True)] = 256
    frame_skip: Annotated[int, Field(gt=0, strict=True)] = 2
    downscale: Literal[1, 2, 4] = 4
    dense_reward_scale: Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)] = 0.1


class KinetixEnv(BaseEnv[EnvState, EnvParams]):
    """Explicit transitions preserve final observations for timeout bootstrapping."""

    def __init__(self, config: KinetixConfig) -> None:
        self.params = EnvParams(max_timesteps=config.max_steps, dense_reward_scale=config.dense_reward_scale)
        static_params = static_env_params_from_size(config.size).replace(
            frame_skip=config.frame_skip, downscale=config.downscale
        )
        if config.levels is None:
            reset_fn = make_reset_fn_sample_kinetix_level(self.params, static_params)
        else:
            reset_fn = make_reset_fn_list_of_levels(list(config.levels), static_params)
        self.env = make_kinetix_env(
            action_type=ActionType.DISCRETE if config.action_type == "discrete" else ActionType.CONTINUOUS,
            observation_type=ObservationType.SYMBOLIC_FLAT,
            reset_fn=reset_fn,
            env_params=self.params,
            static_env_params=static_params,
            auto_reset=False,
        )
        self.render_pixels = make_render_pixels(self.params, static_params)
        space = self.env.action_space(self.params)
        if config.action_type == "discrete":
            self.action_space = DiscreteSpace(space.n)
        else:
            self.action_space = ContinuousSpace(
                space.shape, tuple(map(float, space.low.ravel())), tuple(map(float, space.high.ravel()))
            )

    @partial(jax.jit, static_argnames="self")
    def reset(self, key: jax.Array, params: EnvParams) -> tuple[jax.Array, EnvState]:  # key: ()
        """Return (F,) float32 symbolic observation and (...) state leaves."""
        observation, state = self.env.reset(key, params)  # observation: (F,); state leaves: (...)
        return observation, state  # (F,); state leaves: (...)

    @partial(jax.jit, static_argnames="self")
    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: EnvState,  # leaves: (...)
        action: jax.Array,  # () integer ID or (D,) continuous action
        params: EnvParams,
    ) -> EnvStep[EnvState]:
        """Advance and autoreset; info retains (F,) final_observation."""
        step_key, reset_key = jax.random.split(key)  # each () typed PRNG key
        # Separate time limits from termination for PPO timeout bootstrapping.
        final_observation, final_state, reward, terminated, _ = self.env.step_env(
            step_key, state, action, params.replace(max_timesteps=2**31 - 1)
        )  # observation: (F,); state leaves: (...); reward/terminated: ()
        truncated = final_state.timestep >= params.max_timesteps  # ()
        done = terminated | truncated  # ()

        def reset_done(_: None) -> tuple[jax.Array, EnvState]:
            return self.reset(reset_key, params)  # (F,); state leaves: (...)

        def continue_episode(_: None) -> tuple[jax.Array, EnvState]:
            return final_observation, final_state  # (F,); state leaves: (...)

        observation, state = jax.lax.cond(done, reset_done, continue_episode, None)  # (F,); leaves: (...)
        return EnvStep(
            observation,  # (F,)
            state,  # leaves: (...)
            reward,  # ()
            done,  # ()
            {
                "terminated": terminated,  # ()
                "truncated": truncated,  # ()
                "final_observation": final_observation,  # (F,)
                "final_state": final_state,  # leaves: (...)
            },
        )

    @partial(jax.jit, static_argnames="self")
    def render(self, state: EnvState) -> jax.Array:  # state leaves: (...) -> (H, W, 3)
        pixels = self.render_pixels(state)  # (W, H, 3), y-up in [0, 255]
        return jnp.rint(jnp.clip(pixels, 0, 255)).astype(jnp.uint8).swapaxes(0, 1)[::-1]  # (H, W, 3)
