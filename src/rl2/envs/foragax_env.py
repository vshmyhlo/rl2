"""Foragax square-wave foraging with local grid observations and PPO time limits."""

from functools import partial
from typing import Annotated, Literal

import jax
import jax.numpy as jnp
from foragax.env import EnvParams, EnvState
from foragax.registry import make
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.actions import DiscreteSpace
from rl2.envs.base import BaseEnv, EnvStep


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class ForagaxConfig:
    id: Literal["Foragax"] = "Foragax"
    env_id: Literal["ForagaxSquareWaveTwoBiome-v11"] = "ForagaxSquareWaveTwoBiome-v11"
    observation_type: Literal["color", "object", "rgb"] = "color"
    aperture_size: Annotated[int, Field(gt=0, strict=True)] = 9
    period: Annotated[int, Field(gt=1, lt=2**31, strict=True)] = 1024
    reward_delay: Annotated[int, Field(ge=0, strict=True)] = 0
    max_steps: Annotated[int, Field(gt=0, lt=2**31, strict=True)] = 4096


class ForagaxEnv(BaseEnv[EnvState, EnvParams]):
    def __init__(self, config: ForagaxConfig) -> None:
        self.env = make(
            config.env_id,
            observation_type=config.observation_type,
            aperture_size=config.aperture_size,
            period=config.period,
            reward_delay=config.reward_delay,
        )
        self.params = self.env.default_params.replace(max_steps_in_episode=config.max_steps)
        self.action_space = DiscreteSpace(n=self.env.num_actions)

    @partial(jax.jit, static_argnames="self")
    def reset(self, key: jax.Array, params: EnvParams) -> tuple[jax.Array, EnvState]:  # key: ()
        """Return (H, W, C) float32 observation and (...) state leaves."""
        return self.env.reset(key, params)  # (H, W, C); state leaves: (...)

    @partial(jax.jit, static_argnames="self")
    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: EnvState,  # leaves: (...)
        action: jax.Array,  # () int32 action ID
        params: EnvParams,
    ) -> EnvStep[EnvState]:
        """Advance and autoreset; retain (H, W, C) final_observation for bootstrapping."""
        step_key, reset_key = jax.random.split(key)  # each ()
        final_observation, final_state, reward, _, _ = self.env.step_env(
            step_key, state, action, params
        )  # (H, W, C); state leaves: (...); reward: ()
        # Native Foragax never terminates and ignores max_steps_in_episode.
        truncated = final_state.time >= params.max_steps_in_episode  # ()
        observation, next_state = jax.lax.cond(
            truncated, lambda: self.reset(reset_key, params), lambda: (final_observation, final_state)
        )  # (H, W, C); state leaves: (...)
        return EnvStep(
            observation=observation,  # (H, W, C)
            state=next_state,  # leaves: (...)
            reward=reward,  # ()
            done=truncated,  # ()
            info={
                "terminated": jnp.asarray(False),  # ()
                "truncated": truncated,  # ()
                "final_observation": final_observation,  # (H, W, C)
                "final_state": final_state,  # leaves: (...)
            },
        )

    @partial(jax.jit, static_argnames="self")
    def render(self, state: EnvState) -> jax.Array:  # state leaves: (...) -> (135, 216, 3)
        return self.env.render(state, self.params, render_mode="world").astype(jnp.uint8)  # (135, 216, 3)
