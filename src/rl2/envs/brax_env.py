"""Brax Pusher with native continuous actions and time-limit bootstrapping."""

from functools import partial
from typing import Annotated, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from brax.base import State as PipelineState
from brax.envs.base import State
from brax.envs.pusher import Pusher
from flax import struct
from numpy.typing import NDArray
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.actions import ContinuousSpace
from rl2.envs.base import BaseEnv, EnvStep


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class BraxConfig:
    id: Literal["Brax"] = "Brax"
    task: Literal["pusher"] = "pusher"
    backend: Literal["generalized"] = "generalized"
    max_steps: Annotated[int, Field(gt=0, lt=2**31, strict=True)] = 1000


@struct.dataclass
class BraxParams:
    max_steps: int = 1000


@struct.dataclass
class BraxState:
    physics: State  # leaves: (...)
    step_count: jax.Array  # () int32


class BraxEnv(BaseEnv[BraxState, BraxParams]):
    def __init__(self, config: BraxConfig) -> None:
        self.env = Pusher(backend=config.backend)
        self.params = BraxParams(max_steps=config.max_steps)
        size = self.env.action_size
        self.action_space = ContinuousSpace(shape=(size,), low=(-1.0,) * size, high=(1.0,) * size)

    @partial(jax.jit, static_argnames="self")
    def reset(self, key: jax.Array, params: BraxParams) -> tuple[jax.Array, BraxState]:  # key: ()
        """Return (23,) float32 observation and (...) physics state leaves."""
        physics = self.env.reset(key)  # leaves: (...)
        state = BraxState(physics=physics, step_count=jnp.array(0, jnp.int32))
        return cast(jax.Array, physics.obs), state  # (23,); state leaves: (...)

    @partial(jax.jit, static_argnames="self")
    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: BraxState,  # leaves: (...)
        action: jax.Array,  # (7,) continuous controls
        params: BraxParams,
    ) -> EnvStep[BraxState]:
        """Advance physics and autoreset, retaining (23,) final_observation for bootstrapping."""
        physics = self.env.step(state.physics, action)  # leaves: (...)
        final_state = state.replace(physics=physics, step_count=state.step_count + 1)
        final_observation = cast(jax.Array, physics.obs)  # (23,)
        terminated = physics.done.astype(jnp.bool_)  # ()
        truncated = final_state.step_count >= params.max_steps  # ()
        done = terminated | truncated  # ()

        def reset_done(_: None) -> tuple[jax.Array, BraxState]:
            return self.reset(key, params)  # (23,); state leaves: (...)

        def continue_episode(_: None) -> tuple[jax.Array, BraxState]:
            return final_observation, final_state  # (23,); state leaves: (...)

        observation, next_state = jax.lax.cond(done, reset_done, continue_episode, None)  # (23,); leaves: (...)
        return EnvStep(
            observation=observation,  # (23,)
            state=next_state,  # leaves: (...)
            reward=physics.reward,  # ()
            done=done,  # ()
            info={
                "terminated": terminated,  # ()
                "truncated": truncated,  # ()
                "final_observation": final_observation,  # (23,)
                "final_state": final_state,  # leaves: (...)
            },
        )

    def render(self, state: BraxState) -> NDArray[np.uint8]:  # leaves: (...) -> (240, 320, 3)
        return self.env.render([cast(PipelineState, state.physics.pipeline_state)], height=240, width=320)[
            0
        ]  # (240, 320, 3)
