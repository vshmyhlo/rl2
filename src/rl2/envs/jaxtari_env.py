"""JAXtari Breakout and Space Invaders with native RGB frames and episode boundaries."""

from functools import partial
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
from rl2.envs.autoreset import reset_batch
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
        # Native constants must also work inside shard-local reset computation.
        with jax.set_mesh(None):
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
        *,
        auto_reset: bool = True,
    ) -> EnvStep[JAXtariState]:
        """Advance one frame; retain (210, 160, 3) final_observation before autoreset.

        auto_reset is static; step_batch disables it before applying vmap.
        """
        _, game_state, reward, terminated, _ = self.env.step(state.game_state, action)  # leaves: (...); scalars: ()
        final_state = JAXtariState(
            game_state=game_state,  # leaves: (...)
            elapsed_steps=state.elapsed_steps + 1,  # ()
        )
        final_observation = self.render(final_state)  # (210, 160, 3)
        truncated = final_state.elapsed_steps >= params.max_steps  # ()
        done = terminated | truncated  # ()

        if auto_reset:
            observation, next_state = jax.lax.cond(
                done, lambda: self.reset(key, params), lambda: (final_observation, final_state)
            )  # (210, 160, 3); state leaves: (...)
        else:
            observation, next_state = final_observation, final_state  # (210, 160, 3); state leaves: (...)
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

    def step_batch(
        self,
        keys: jax.Array,  # (B,) typed PRNG keys
        state: JAXtariState,  # leaves: (B, ...)
        actions: jax.Array,  # (B,) int32 action IDs
        params: JAXtariParams,
    ) -> EnvStep[JAXtariState]:
        """Skip reset rendering on shards without completed episodes."""
        result = jax.vmap(partial(self.step, auto_reset=False), in_axes=(0, 0, 0, None))(
            keys, state, actions, params
        )  # observations: (B,210,160,3); state leaves: (B,...); rewards/flags: (B,)
        observation, next_state = reset_batch(
            partial(self.reset, params=params), keys, result.done, (result.observation, result.state)
        )  # (B,210,160,3); state leaves: (B,...)
        return result._replace(observation=observation, state=next_state)

    def render(self, state: JAXtariState) -> jax.Array:  # leaves: (...) -> (210, 160, 3)
        return self.env.render(state.game_state)  # (210, 160, 3) uint8
