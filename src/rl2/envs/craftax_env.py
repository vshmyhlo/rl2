"""Craftax and Craftax-Classic symbolic/pixel observations for PPO.

O is (F,) for symbolic features or (H, W, 3) for pixels."""

from functools import partial
from typing import Annotated, Literal

import chex
import jax
import jax.numpy as jnp
from craftax.craftax.craftax_state import EnvParams
from craftax.craftax_classic.envs.craftax_state import EnvParams as ClassicEnvParams
from craftax.craftax_env import make_craftax_env_from_name
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.actions import DiscreteSpace
from rl2.envs.base import BaseEnv, EnvStep

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

    @partial(jax.jit, static_argnames="self")
    def reset(self, key: jax.Array, params: CraftaxParams) -> tuple[jax.Array, chex.ArrayTree]:  # key: ()
        """Return (*O,) float32 observation and (...) state leaves."""
        observation, state = self.env.reset(key, params)  # (*O,); state leaves: (...)
        return observation, state  # (*O,); state leaves: (...)

    @partial(jax.jit, static_argnames="self")
    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: chex.ArrayTree,  # leaves: (...)
        action: jax.Array,  # ()
        params: CraftaxParams,
    ) -> EnvStep[chex.ArrayTree]:
        """Advance and autoreset; info retains (*O,) final_observation."""
        step_key, reset_key = jax.random.split(key)  # each () typed PRNG key
        # Separate time limits from termination for PPO timeout bootstrapping.
        observation, state, reward, terminated, _ = self.env.step_env(
            step_key, state, action, params.replace(max_timesteps=2**31 - 1)
        )  # (*O,); state leaves: (...); reward/terminated: ()
        final_observation = observation  # (*O,)
        truncated = state.timestep >= params.max_timesteps  # ()
        done = terminated | truncated  # ()
        final_state = state  # leaves: (...), before autoreset

        def reset_done(_: None) -> tuple[jax.Array, chex.ArrayTree]:
            return self.reset(reset_key, params)  # (*O,); state leaves: (...)

        def continue_episode(_: None) -> tuple[jax.Array, chex.ArrayTree]:
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
