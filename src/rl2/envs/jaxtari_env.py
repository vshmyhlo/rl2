"""JAXtari Breakout with native RGB frames and explicit episode boundaries."""

from functools import partial
from typing import Annotated, Literal, cast

import jax
import jax.numpy as jnp
import jaxtari
from flax import struct
from jaxtari.games.jax_breakout import BreakoutState, JaxBreakout
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.actions import DiscreteSpace
from rl2.envs.base import BaseEnv, EnvStep


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class JAXtariConfig:
    id: Literal["JAXtari"] = "JAXtari"
    game: Literal["breakout"] = "breakout"
    max_steps: Annotated[int, Field(gt=0, lt=2**31, strict=True)] = 108_000


@struct.dataclass
class JAXtariParams:
    max_steps: int = 108_000


class JAXtariEnv(BaseEnv[BreakoutState, JAXtariParams]):
    def __init__(self, config: JAXtariConfig) -> None:
        self.env = cast(JaxBreakout, jaxtari.make(config.game))
        self.params = JAXtariParams(max_steps=config.max_steps)
        self.action_space = DiscreteSpace(n=self.env.action_space().n)

    @partial(jax.jit, static_argnames="self")
    def reset(self, key: jax.Array, params: JAXtariParams) -> tuple[jax.Array, BreakoutState]:  # key: ()
        """Return (210, 160, 3) uint8 RGB and (...) native game state leaves."""
        _, state = self.env.reset(key)  # state leaves: (...)
        return self.render(state), state  # (210, 160, 3); state leaves: (...)

    @partial(jax.jit, static_argnames="self")
    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: BreakoutState,  # leaves: (...)
        action: jax.Array,  # () int32 action ID
        params: JAXtariParams,
    ) -> EnvStep[BreakoutState]:
        """Advance one frame; retain (210, 160, 3) final_observation before autoreset."""
        _, final_state, reward, terminated, _ = self.env.step(state, action)  # leaves: (...); scalars: ()
        final_observation = self.render(final_state)  # (210, 160, 3)
        truncated = final_state.step_counter >= params.max_steps  # ()
        done = terminated | truncated  # ()

        def reset_done(_: None) -> tuple[jax.Array, BreakoutState]:
            return self.reset(key, params)  # (210, 160, 3); state leaves: (...)

        def continue_episode(_: None) -> tuple[jax.Array, BreakoutState]:
            return final_observation, final_state  # (210, 160, 3); state leaves: (...)

        observation, next_state = jax.lax.cond(done, reset_done, continue_episode, None)  # (210,160,3); leaves: (...)
        return EnvStep(
            observation=observation,  # (210, 160, 3)
            state=next_state,  # leaves: (...)
            reward=reward.astype(jnp.float32),  # ()
            done=done,  # ()
            info={
                "terminated": terminated,  # ()
                "truncated": truncated,  # ()
                "final_observation": final_observation,  # (210, 160, 3)
                "final_state": final_state,  # leaves: (...)
            },
        )

    @partial(jax.jit, static_argnames="self")
    def render(self, state: BreakoutState) -> jax.Array:  # leaves: (...) -> (210, 160, 3)
        return self.env.render(state)  # (210, 160, 3) uint8
