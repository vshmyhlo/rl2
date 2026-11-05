"""JAXtari Breakout and Space Invaders with native RGB frames and episode boundaries."""

from typing import Annotated, Literal, cast

import jax
import jax.numpy as jnp
import jaxtari
from flax import struct
from jaxtari.games.jax_breakout import BreakoutState, JaxBreakout
from jaxtari.games.jax_spaceinvaders import JaxSpaceInvaders, SpaceInvadersState
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.actions import DiscreteSpace
from rl2.envs.base import BaseEnv, EnvStep


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class JAXtariConfig:
    id: Literal["JAXtari"] = "JAXtari"
    game: Literal["breakout", "spaceinvaders"] = "breakout"
    max_steps: Annotated[int, Field(gt=0, lt=2**31, strict=True)] = 108_000


@struct.dataclass
class JAXtariParams:
    max_steps: int = 108_000


@struct.dataclass
class JAXtariState:
    game_state: BreakoutState | SpaceInvadersState  # leaves: (...)
    elapsed_steps: jax.Array  # () int32; native counters may reset after life loss


class JAXtariEnv(BaseEnv[JAXtariState, JAXtariParams]):
    def __init__(self, config: JAXtariConfig) -> None:
        self.env = cast(JaxBreakout | JaxSpaceInvaders, jaxtari.make(config.game))
        self.params = JAXtariParams(max_steps=config.max_steps)
        self.action_space = DiscreteSpace(n=self.env.action_space().n)

    def reset(self, key: jax.Array, params: JAXtariParams) -> tuple[jax.Array, JAXtariState]:  # key: ()
        """Return (210, 160, 3) uint8 RGB and (...) native game state leaves."""
        _, game_state = self.env.reset(key)  # state leaves: (...)
        state = JAXtariState(game_state=game_state, elapsed_steps=jnp.array(0, jnp.int32))  # leaves: (...)
        return self.render(state), state  # (210, 160, 3); state leaves: (...)

    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: JAXtariState,  # leaves: (...)
        action: jax.Array,  # () int32 action ID
        params: JAXtariParams,
    ) -> EnvStep[JAXtariState]:
        """Advance one frame; retain (210, 160, 3) final_observation before autoreset."""
        _, game_state, reward, terminated, _ = self.env.step(state.game_state, action)  # leaves: (...); scalars: ()
        final_state = JAXtariState(
            game_state=game_state,  # leaves: (...)
            elapsed_steps=state.elapsed_steps + 1,  # ()
        )
        final_observation = self.render(final_state)  # (210, 160, 3)
        truncated = final_state.elapsed_steps >= params.max_steps  # ()
        done = terminated | truncated  # ()

        def reset_done(_: None) -> tuple[jax.Array, JAXtariState]:
            return self.reset(key, params)  # (210, 160, 3); state leaves: (...)

        def continue_episode(_: None) -> tuple[jax.Array, JAXtariState]:
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

    def render(self, state: JAXtariState) -> jax.Array:  # leaves: (...) -> (210, 160, 3)
        return self.env.render(state.game_state)  # (210, 160, 3) uint8
