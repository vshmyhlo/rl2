"""Gymnax MinAtar and FourRooms with native visual observations."""

from typing import Annotated, Literal

import gymnax
import jax
import jax.numpy as jnp
from gymnax.environments.minatar import breakout, space_invaders
from gymnax.environments.misc import rooms
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.actions import DiscreteSpace
from rl2.envs.base import BaseEnv, EnvStep

type EnvParams = breakout.EnvParams | space_invaders.EnvParams | rooms.EnvParams
type EnvState = breakout.EnvState | space_invaders.EnvState | rooms.EnvState


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class GymnaxConfig:
    id: Literal["Breakout-MinAtar", "SpaceInvaders-MinAtar", "FourRooms-misc"] = "Breakout-MinAtar"
    # Reserve int32 max for disabling the native timeout during step.
    max_steps: Annotated[int, Field(gt=0, lt=2**31 - 1, strict=True)] | None = None


class GymnaxEnv(BaseEnv[EnvState, EnvParams]):
    """Visual environments with explicit state, independent RNGs, and same-step reset."""

    def __init__(self, config: GymnaxConfig) -> None:
        kwargs = {"use_visual_obs": True} if config.id == "FourRooms-misc" else {}
        self.env, self.params = gymnax.make(config.id, **kwargs)
        if config.max_steps is not None:
            self.params = self.params.replace(max_steps_in_episode=config.max_steps)
        self.action_space = DiscreteSpace(self.env.num_actions)

    def reset(self, key: jax.Array, params: EnvParams) -> tuple[jax.Array, EnvState]:  # key: ()
        """Return (H, W, C) float32 observation and (...) state leaves."""
        observation, state = self.env.reset(key, params)  # (H, W, C); state leaves: (...)
        return observation, state  # (H, W, C); state leaves: (...)

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
            reward=reward.astype(jnp.float32),  # () FourRooms returns a boolean reward
            done=done,  # ()
            info={
                "terminated": terminated,  # ()
                "truncated": truncated,  # ()
                "final_observation": final_observation,  # (H, W, C)
                "final_state": final_state,  # leaves: (...)
            },
        )

    @staticmethod
    def render_observation(observation: jax.Array) -> jax.Array:  # (H, W, C) -> (H*16, W*16, 3)
        return render_observation(observation)  # (H*16, W*16, 3)

    def render(self, state: EnvState) -> jax.Array:  # state leaves: (...) -> (H*16, W*16, 3)
        if isinstance(state, rooms.EnvState):
            pixels = jnp.where(self.env.occupied_map[..., None], 80, 255)  # (H, W, 1)
            pixels = jnp.repeat(pixels, 3, axis=-1).astype(jnp.uint8)  # (H, W, 3)
            pixels = pixels.at[state.goal[0], state.goal[1]].set(jnp.array([80, 200, 80], jnp.uint8))  # (H, W, 3)
            pixels = pixels.at[state.pos[0], state.pos[1]].set(jnp.array([80, 160, 255], jnp.uint8))  # (H, W, 3)
            return pixels.repeat(16, axis=0).repeat(16, axis=1)  # (H*16, W*16, 3)
        observation = self.env.get_obs(state, self.params)  # (H, W, C)
        return render_observation(observation)  # (H*16, W*16, 3)


def render_observation(observation: jax.Array) -> jax.Array:  # (H, W, C) -> (H*16, W*16, 3)
    """Color (H, W, C) visual channels; C=2 for FourRooms, 4 or 6 for MinAtar."""
    palette = jnp.array(
        [[255, 255, 255], [255, 80, 80], [100, 100, 100], [80, 160, 255], [80, 255, 80], [255, 200, 80]],
        dtype=jnp.uint8,
    )[: observation.shape[-1]]  # (C, 3)
    pixels = jnp.max(jnp.where(observation[..., None] > 0, palette, 0), axis=2)  # (H,W,3)
    return pixels.repeat(16, axis=0).repeat(16, axis=1)  # (H*16, W*16, 3)
