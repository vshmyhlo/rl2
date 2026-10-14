"""Single-environment Gymnax Breakout adapter with native observations."""

from functools import partial
from typing import Literal

import gymnax
import jax
import jax.numpy as jnp
from gymnax.environments.minatar.breakout import EnvParams, EnvState
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from rl2.actions import DiscreteSpace
from rl2.envs.base import BaseEnv, EnvStep


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class GymnaxConfig:
    id: Literal["Breakout-MinAtar"] = "Breakout-MinAtar"


class GymnaxEnv(BaseEnv[EnvState, EnvParams]):
    """Breakout-MinAtar with explicit state, independent RNGs, and same-step reset."""

    @staticmethod
    def render_observation(observation: jax.Array) -> jax.Array:  # (H, W, 4) -> (H*16, W*16, 3)
        return render_observation(observation)  # (H*16, W*16, 3)

    @partial(jax.jit, static_argnames="self")
    def render(self, state: EnvState) -> jax.Array:  # state leaves: (...) -> (H*16, W*16, 3)
        observation = self.env.get_obs(state, self.params)  # (H, W, 4)
        return render_observation(observation)  # (H*16, W*16, 3)

    def __init__(self, env_id: str, *, max_steps_in_episode: int | None = None) -> None:
        if env_id != "Breakout-MinAtar":
            raise ValueError("PPO currently supports only Gymnax Breakout-MinAtar")
        self.env, self.params = gymnax.make(env_id)
        if max_steps_in_episode is not None:
            if max_steps_in_episode <= 0:
                raise ValueError("max_steps_in_episode must be positive")
            self.params = self.params.replace(max_steps_in_episode=max_steps_in_episode)
        self.action_space = DiscreteSpace(self.env.num_actions)

    @partial(jax.jit, static_argnames="self")
    def reset(self, key: jax.Array, params: EnvParams) -> tuple[jax.Array, EnvState]:  # key: ()
        """Return (H, W, C) float32 observation and (...) state leaves."""
        observation, state = self.env.reset(key, params)  # (H, W, C); state leaves: (...)
        return observation, state  # (H, W, C); state leaves: (...)

    @partial(jax.jit, static_argnames="self")
    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: EnvState,  # leaves: (...)
        action: jax.Array,  # ()
        params: EnvParams,
    ) -> EnvStep[EnvState]:
        """Advance and autoreset; info retains (H, W, C) final_observation."""
        step_key, reset_key = jax.random.split(key)  # each () typed PRNG key
        # Separate time limits from termination for PPO timeout bootstrapping.
        observation, state, reward, terminated, _ = self.env.step_env(
            step_key, state, action, params.replace(max_steps_in_episode=2**31 - 1)
        )  # (H, W, C); state leaves: (...); reward/terminated: ()
        final_observation = observation  # (H, W, C)
        truncated = state.time >= params.max_steps_in_episode  # ()
        done = terminated | truncated  # ()
        final_state = state  # leaves: (...), before autoreset

        def reset_done(_: None) -> tuple[jax.Array, EnvState]:
            return self.reset(reset_key, params)  # (H, W, C); state leaves: (...)

        def continue_episode(_: None) -> tuple[jax.Array, EnvState]:
            return final_observation, state  # (H, W, C); state leaves: (...)

        observation, state = jax.lax.cond(done, reset_done, continue_episode, None)  # (H,W,C); leaves: (...)
        return EnvStep(
            observation=observation,  # (H, W, C)
            state=state,  # leaves: (...)
            reward=reward,  # ()
            done=done,  # ()
            info={
                "terminated": terminated,  # ()
                "truncated": truncated,  # ()
                "final_observation": final_observation,  # (H, W, C)
                "final_state": final_state,  # leaves: (...)
            },
        )


def render_observation(observation: jax.Array) -> jax.Array:  # (H, W, C=4) -> (H*16, W*16, 3)
    """Color the paddle, ball, trail, and bricks for TensorBoard playback."""
    palette = jnp.array([[255, 255, 255], [255, 80, 80], [100, 100, 100], [80, 160, 255]], dtype=jnp.uint8)  # (C,3)
    pixels = jnp.max(jnp.where(observation[..., None] > 0, palette, 0), axis=2)  # (H,W,3)
    return pixels.repeat(16, axis=0).repeat(16, axis=1)  # (H*16, W*16, 3)
