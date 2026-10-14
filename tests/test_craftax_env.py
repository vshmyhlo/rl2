from operator import itemgetter
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from rl2.envs.craftax_env import CraftaxConfig, CraftaxEnv


@struct.dataclass
class Params:
    max_timesteps: int = 2


class State(NamedTuple):
    timestep: jax.Array  # ()
    terminal: jax.Array  # ()
    color: jax.Array  # ()


class Game:
    def get_obs(self, state: State) -> jax.Array:  # scalar state leaves -> (H=2, W=3, 3)
        return jnp.full((2, 3, 3), state.color)  # (H, W, 3)

    def reset(self, key: jax.Array, params: Params) -> tuple[jax.Array, State]:  # key: ()
        color = jax.random.uniform(key)  # ()
        return jnp.full((2, 3, 3), color), State(jnp.array(0), jnp.array(False), color)  # (H,W,3); scalar leaves

    def step_env(
        self,
        key: jax.Array,  # ()
        state: State,  # scalar leaves
        action: jax.Array,  # ()
        params: Params,
    ) -> tuple[jax.Array, State, jax.Array, jax.Array, dict[str, jax.Array]]:
        state = state._replace(timestep=state.timestep + 1, color=state.color / 2 + action)  # scalar leaves
        done = state.terminal | (state.timestep >= params.max_timesteps)  # ()
        return jnp.full((2, 3, 3), state.color), state, jnp.array(0.25), done, {}  # (H,W,3); scalar leaves


def test_render_observation_converts_only_video_pixels() -> None:
    pixels = (jnp.arange(18, dtype=jnp.float32).reshape(2, 3, 3) - 1) / 15  # (H=2, W=3, C=3)
    frame = CraftaxEnv.render_observation(pixels)  # (H, W, C)
    expected = np.rint(np.clip(np.asarray(pixels), 0, 1) * 255).astype(np.uint8)  # (H, W, C)
    np.testing.assert_array_equal(frame, expected)
    assert frame.dtype == jnp.uint8


def test_terminal_timeout_both_and_partial_autoreset() -> None:
    env = object.__new__(CraftaxEnv)
    env.env = Game()
    env.params = Params(max_timesteps=2)
    key = jax.random.key(0)  # ()
    observations, initial = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(key, 4), env.params
    )  # (B=4,2,3,3); leaves: (B,)
    chex.assert_trees_all_equal(
        (observations, initial), jax.vmap(env.env.reset, in_axes=(0, None))(jax.random.split(key, 4), env.params)
    )
    assert np.unique(initial.color).size == 4
    state = initial._replace(timestep=jnp.array([0, 1, 1, 0]), terminal=jnp.array([True, False, True, False]))
    # state leaves: (B,), respectively terminal, timeout, both, continuing
    key = jax.random.key(1)  # ()
    actions = jnp.arange(4, dtype=jnp.int32)  # (B,) native action IDs
    next_observations, next_state, rewards, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        jax.random.split(key, 4), state, actions, env.params
    )  # (B,H,W,C); leaves: (B,...); rewards/done/flags: (B,); final_observation: (B,H,W,C)
    np.testing.assert_array_equal(info["terminated"], [True, False, True, False])
    np.testing.assert_array_equal(info["truncated"], [False, True, True, False])
    np.testing.assert_array_equal(done, [True, True, True, False])
    np.testing.assert_array_equal(next_state.timestep, [0, 0, 0, 1])
    np.testing.assert_array_equal(info["final_state"].timestep, [1, 2, 2, 1])
    final = jax.tree.map(itemgetter(0), info["final_state"])  # scalar state leaves
    np.testing.assert_array_equal(env.render(final), env.render_observation(info["final_observation"][0]))
    np.testing.assert_array_equal(rewards, [0.25] * 4)
    assert next_observations.shape == info["final_observation"].shape == (4, 2, 3, 3)
    reset_keys = jax.vmap(jax.random.split)(jax.random.split(key, 4))[:, 1]  # (B,) typed keys
    reset_observations, reset_state = jax.vmap(env.reset, in_axes=(0, None))(
        reset_keys, env.params
    )  # (B,2,3,3); leaves: (B,)
    np.testing.assert_array_equal(next_observations[:3], reset_observations[:3])
    np.testing.assert_array_equal(next_state.color[:3], reset_state.color[:3])
    np.testing.assert_array_equal(next_observations[3], info["final_observation"][3])
    expected_final = jnp.broadcast_to((state.color / 2 + actions)[:, None, None, None], (4, 2, 3, 3))  # (B,H,W,3)
    np.testing.assert_array_equal(info["final_observation"], expected_final)
    next_observations, next_state, rewards, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        jax.random.split(key, 4), initial, jnp.zeros(4, jnp.int32), env.params
    )  # (B,H,W,C); leaves: (B,...); rewards/done/flags: (B,); final_observation: (B,H,W,C)
    np.testing.assert_array_equal(next_state.timestep, 1)
    np.testing.assert_array_equal(next_observations, info["final_observation"])


@pytest.mark.parametrize(
    "env_id,shape,num_actions",
    [("Craftax-Classic-Pixels-v1", (63, 63, 3), 17), ("Craftax-Pixels-v1", (130, 110, 3), 43)],
    ids=["classic", "full"],
)
def test_real_craftax_reset_step_and_video(env_id: str, shape: tuple[int, ...], num_actions: int) -> None:
    pytest.importorskip("craftax")
    env = CraftaxEnv(CraftaxConfig(env_id=env_id, max_steps=1))
    key = jax.random.key(0)  # ()
    observations, state = env.reset(key, env.params)  # (H,W,3); leaves: (...)
    assert observations.shape == shape and observations.dtype == jnp.float32
    assert env.action_space.n == num_actions
    chex.assert_trees_all_equal((observations, state), env.env.reset(key, env.params))
    next_observations, next_state, rewards, done, info = env.step(
        jax.random.key(1), state, jnp.array(0), env.params
    )  # (H,W,C); leaves: (...); scalars; final_observation: (H,W,C)
    assert next_observations.shape == info["final_observation"].shape == shape
    chex.assert_shape((rewards, done, info["terminated"], info["truncated"]), ())
    np.testing.assert_array_equal(info["terminated"], False)
    np.testing.assert_array_equal(info["truncated"], True)
    np.testing.assert_array_equal(next_state.timestep, 0)
    frame = env.render(info["final_state"])  # (H,W,3)
    np.testing.assert_array_equal(frame, env.render_observation(info["final_observation"]))
    assert frame.shape == shape and frame.dtype == jnp.uint8
