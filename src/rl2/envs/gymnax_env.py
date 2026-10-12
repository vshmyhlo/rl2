"""Single-environment Gymnax Breakout transitions and observation conversion."""

from functools import partial

import gymnax
import jax
import jax.numpy as jnp
from gymnax.environments.minatar.breakout import EnvParams, EnvState

from rl2.envs.base import BaseEnv, EnvStep


def encode_observations(observations: jax.Array) -> jax.Array:  # (..., H, W, C) -> (..., C, H, W)
    """Match the shared CNN's channel-first uint8 input and /255 normalization."""
    return jnp.moveaxis(observations, -1, -3).astype(jnp.uint8) * 255  # (..., C, H, W)


class GymnaxEnv(BaseEnv[EnvState, EnvParams]):
    """Breakout-MinAtar with explicit state, independent RNGs, and same-step reset."""

    @staticmethod
    def render_observation(observation: jax.Array) -> jax.Array:  # (4, H, W) -> (H*16, W*16, 3)
        return render_observation(observation)  # (H*16, W*16, 3)

    @partial(jax.jit, static_argnames="self")
    def render(self, state: EnvState) -> jax.Array:  # state leaves: (...) -> (H*16, W*16, 3)
        observation = encode_observations(self.env.get_obs(state, self.params))  # (4, H, W)
        return render_observation(observation)  # (H*16, W*16, 3)

    def __init__(self, env_id: str, *, max_steps_in_episode: int | None = None) -> None:
        if env_id != "Breakout-MinAtar":
            raise ValueError("PPO currently supports only Gymnax Breakout-MinAtar")
        self.env, self.params = gymnax.make(env_id)
        if max_steps_in_episode is not None:
            if max_steps_in_episode <= 0:
                raise ValueError("max_steps_in_episode must be positive")
            self.params = self.params.replace(max_steps_in_episode=max_steps_in_episode)
        self.num_actions = self.env.num_actions

    @partial(jax.jit, static_argnames="self")
    def reset(self, key: jax.Array, params: EnvParams) -> tuple[jax.Array, EnvState]:  # key: ()
        """Return (C, H, W) uint8 observation and (...) state leaves."""
        observation, state = self.env.reset(key, params)  # (H, W, C); state leaves: (...)
        return encode_observations(observation), state  # (C, H, W); state leaves: (...)

    @partial(jax.jit, static_argnames="self")
    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: EnvState,  # leaves: (...)
        action: jax.Array,  # ()
        params: EnvParams,
    ) -> EnvStep[EnvState]:
        """Advance and autoreset; info retains (C, H, W) final_observation."""
        step_key, reset_key = jax.random.split(key)  # each () typed PRNG key
        # Separate time limits from termination for PPO timeout bootstrapping.
        observation, state, reward, terminated, _ = self.env.step_env(
            step_key, state, action, params.replace(max_steps_in_episode=2**31 - 1)
        )  # (H, W, C); state leaves: (...); reward/terminated: ()
        final_observation = encode_observations(observation)  # (C, H, W)
        truncated = state.time >= params.max_steps_in_episode  # ()
        done = terminated | truncated  # ()
        final_state = state  # leaves: (...), before autoreset

        def reset_done(_: None) -> tuple[jax.Array, EnvState]:
            return self.reset(reset_key, params)  # (C, H, W); state leaves: (...)

        def continue_episode(_: None) -> tuple[jax.Array, EnvState]:
            return final_observation, state  # (C, H, W); state leaves: (...)

        observation, state = jax.lax.cond(done, reset_done, continue_episode, None)  # (C,H,W); leaves: (...)
        return EnvStep(
            observation,  # (C, H, W)
            state,  # leaves: (...)
            reward,  # ()
            done,  # ()
            {
                "terminated": terminated,  # ()
                "truncated": truncated,  # ()
                "final_observation": final_observation,  # (C, H, W)
                "final_state": final_state,  # leaves: (...)
            },
        )


def render_observation(observation: jax.Array) -> jax.Array:  # (C=4, H, W) -> (H*16, W*16, 3)
    """Color the paddle, ball, trail, and bricks for TensorBoard playback."""
    palette = jnp.array([[255, 255, 255], [255, 80, 80], [100, 100, 100], [80, 160, 255]], dtype=jnp.uint8)  # (C,3)
    pixels = jnp.max(jnp.where(observation[..., None] > 0, palette[:, None, None, :], 0), axis=0)  # (H,W,3)
    return pixels.repeat(16, axis=0).repeat(16, axis=1)  # (H*16, W*16, 3)
