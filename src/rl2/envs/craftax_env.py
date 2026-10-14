"""Craftax and Craftax-Classic symbolic/pixel observations for PPO.

Symbolic observations contain a spatial grid and non-spatial features. Pixels are (H,W,3)."""

from functools import partial
from math import prod
from typing import Annotated, Literal

import chex
import jax
import jax.numpy as jnp
from craftax.craftax import constants as full_constants
from craftax.craftax.craftax_state import EnvParams
from craftax.craftax_classic import constants as classic_constants
from craftax.craftax_classic.envs.craftax_state import EnvParams as ClassicEnvParams
from craftax.craftax_env import make_craftax_env_from_name
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.actions import DiscreteSpace
from rl2.envs.base import BaseEnv, EnvStep
from rl2.envs.observations import CraftaxObservation, Observation

type CraftaxParams = EnvParams | ClassicEnvParams


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class CraftaxConfig:
    id: Literal["Craftax"] = "Craftax"
    env_id: Literal[
        "Craftax-Pixels-v1", "Craftax-Classic-Pixels-v1", "Craftax-Symbolic-v1", "Craftax-Classic-Symbolic-v1"
    ] = "Craftax-Pixels-v1"
    max_steps: Annotated[int, Field(gt=0, lt=2**31, strict=True)] | None = None


class CraftaxEnv(BaseEnv[chex.ArrayTree, CraftaxParams]):
    def __init__(self, config: CraftaxConfig) -> None:
        self.grid_shape: tuple[int, int, int] | None = None
        if "Symbolic" in config.env_id:
            if "Classic" in config.env_id:
                self.grid_shape = (*classic_constants.OBS_DIM, len(classic_constants.BlockType) + 4)
            else:
                channels = len(full_constants.BlockType) + len(full_constants.ItemType) + 5 * 8 + 1
                self.grid_shape = (*full_constants.OBS_DIM, channels)
        self.env = make_craftax_env_from_name(config.env_id, auto_reset=False)
        self.render_env = (
            make_craftax_env_from_name(config.env_id.replace("Symbolic", "Pixels"), auto_reset=False)
            if "Symbolic" in config.env_id
            else self.env
        )
        self.params = self.env.default_params
        if config.max_steps is not None:
            self.params = self.params.replace(max_timesteps=config.max_steps)
        self.action_space = DiscreteSpace(self.env.num_actions)

    def _observation(self, observation: jax.Array) -> Observation:  # (D,) symbolic or (H,W,3) pixels
        if self.grid_shape is None:
            return observation  # (H,W,3)
        size = prod(self.grid_shape)
        return CraftaxObservation(
            grid=observation[:size].reshape(self.grid_shape),  # (H,W,C)
            features=observation[size:],  # (F,)
        )

    @partial(jax.jit, static_argnames="self")
    def reset(self, key: jax.Array, params: CraftaxParams) -> tuple[Observation, chex.ArrayTree]:  # key: ()
        """Return (*O,) float32 observation leaves and (...) state leaves."""
        observation, state = self.env.reset(key, params)  # (*O,); state leaves: (...)
        return self._observation(observation), state  # (*O,); state leaves: (...)

    @partial(jax.jit, static_argnames="self")
    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: chex.ArrayTree,  # leaves: (...)
        action: jax.Array,  # ()
        params: CraftaxParams,
    ) -> EnvStep[chex.ArrayTree]:
        """Advance and autoreset; info retains (*O,) final_observation leaves."""
        step_key, reset_key = jax.random.split(key)  # each () typed PRNG key
        # Separate time limits from termination for PPO timeout bootstrapping.
        observation, state, reward, terminated, _ = self.env.step_env(
            step_key, state, action, params.replace(max_timesteps=2**31 - 1)
        )  # (*O,); state leaves: (...); reward/terminated: ()
        final_observation = self._observation(observation)  # leaves: (*O,)
        truncated = state.timestep >= params.max_timesteps  # ()
        done = terminated | truncated  # ()
        final_state = state  # leaves: (...), before autoreset

        def reset_done(_: None) -> tuple[Observation, chex.ArrayTree]:
            return self.reset(reset_key, params)  # (*O,); state leaves: (...)

        def continue_episode(_: None) -> tuple[Observation, chex.ArrayTree]:
            return final_observation, state  # (*O,); state leaves: (...)

        observation, state = jax.lax.cond(done, reset_done, continue_episode, None)  # (*O,); leaves: (...)
        return EnvStep(
            observation=observation,  # (*O,)
            state=state,  # leaves: (...)
            reward=reward,  # ()
            done=done,  # ()
            info={
                "terminated": terminated,  # ()
                "truncated": truncated,  # ()
                "final_observation": final_observation,  # (*O,)
                "final_state": final_state,  # leaves: (...)
            },
        )

    @staticmethod
    def render_observation(observation: jax.Array) -> jax.Array:  # (H, W, 3) -> (H, W, 3)
        return jnp.rint(jnp.clip(observation, 0, 1) * 255).astype(jnp.uint8)  # (H, W, 3)

    @partial(jax.jit, static_argnames="self")
    def render(self, state: chex.ArrayTree) -> jax.Array:  # state leaves: (...) -> (H, W, 3)
        """Render the native player-centered view and inventory HUD."""
        observation = self.render_env.get_obs(state)  # (H, W, 3)
        return self.render_observation(observation)  # (H, W, 3)
