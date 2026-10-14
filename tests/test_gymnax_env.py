import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2.envs import EnvStep
from rl2.envs.gymnax_env import GymnaxEnv, render_observation


def test_single_reset_step_preserves_native_observations() -> None:
    env = GymnaxEnv("Breakout-MinAtar")
    params = env.params.replace(max_steps_in_episode=1)
    key = jax.random.key(5)  # ()
    observation, state = env.reset(key, params)  # (10,10,4); leaves: (...)
    raw, raw_state = env.env.reset(key, params)  # (10,10,4); leaves: (...)
    chex.assert_trees_all_equal(state, raw_state)
    np.testing.assert_array_equal(observation, raw)
    assert observation.shape == (10, 10, 4) and observation.dtype == jnp.float32
    assert env.action_space.n == 3
    action = jnp.array(2)  # () native action ID
    native = env.env.step_env(jax.random.split(key)[0], state, action, params.replace(max_steps_in_episode=2**31 - 1))
    result = env.step(key, state, action, params)  # observation: (10,10,4); state leaves: (...); reward/done: ()
    assert isinstance(result, EnvStep)
    observation, state, reward, done, info = result
    assert result.observation is observation and result.state is state
    assert result.reward is reward and result.done is done and result.info is info
    # observation/final_observation: (10,10,4); reward/done/flags: (); state leaves: (...)
    assert observation.shape == info["final_observation"].shape == (10, 10, 4)
    chex.assert_shape((reward, done, info["terminated"], info["truncated"], state.time), ())
    assert bool(done) and bool(info["truncated"]) and not bool(info["terminated"])
    assert int(state.time) == 0
    assert int(info["final_state"].time) == 1
    chex.assert_trees_all_equal((info["final_observation"], info["final_state"], reward), native[:3])
    frame = env.render(info["final_state"])  # (160,160,3)
    np.testing.assert_array_equal(frame, render_observation(info["final_observation"]))
    assert frame.shape == (160, 160, 3) and frame.dtype == jnp.uint8


def test_terminal_timeout_both_and_partial_autoreset() -> None:
    env = GymnaxEnv("Breakout-MinAtar", max_steps_in_episode=2)
    _, state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(jax.random.key(0), 4), env.params
    )  # leaves: (B=4,...)
    # Missing the paddle, timeout, both, and continuing, respectively.
    state = state.replace(
        ball_y=jnp.array([8, 3, 8, 3]),
        ball_x=jnp.zeros(4, jnp.int32),
        ball_dir=jnp.full(4, 2),
        time=jnp.array([0, 1, 1, 0]),
    )
    result = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        jax.random.split(jax.random.key(1), 4), state, jnp.zeros(4, jnp.int32), env.params
    )  # (B,H,W,C); leaves: (B,...); rewards/done/flags: (B,); final_observation: (B,H,W,C)
    assert isinstance(result, EnvStep)
    next_observations, next_state, rewards, done, info = result  # (B,H,W,C); leaves: (B,...); rewards/done: (B,)
    np.testing.assert_array_equal(info["terminated"], [True, False, True, False])
    np.testing.assert_array_equal(info["truncated"], [False, True, True, False])
    np.testing.assert_array_equal(done, [True, True, True, False])
    np.testing.assert_array_equal(next_state.time, [0, 0, 0, 1])
    np.testing.assert_array_equal(info["final_state"].time, [1, 2, 2, 1])
    assert next_observations.shape == info["final_observation"].shape == (4, 10, 10, 4)
    np.testing.assert_array_equal(next_observations[3], info["final_observation"][3])
    assert not np.array_equal(next_observations[0], info["final_observation"][0])
    assert rewards.shape == (4,)


def test_rejects_non_native_environment() -> None:
    with pytest.raises(ValueError, match="Gymnax"):
        GymnaxEnv("ALE/Breakout-v5")
