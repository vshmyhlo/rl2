import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2.gymnax_env import GymnaxEnv, render_observation


def test_batched_reset_matches_gymnax_and_encoder_scale() -> None:
    env = GymnaxEnv("Breakout-MinAtar")
    key = jax.random.key(5)  # ()
    observations, state = env.reset(key, 2)  # (B=2,4,10,10); leaves: (B,...)
    raw, raw_state = jax.vmap(env.env.reset, in_axes=(0, None))(jax.random.split(key, 2), env.params)
    chex.assert_trees_all_equal(state, raw_state)
    np.testing.assert_array_equal(observations / 255, np.moveaxis(raw, -1, 1))
    assert observations.shape == (2, 4, 10, 10)
    assert observations.dtype == jnp.uint8
    assert env.num_actions == 3
    frame = render_observation(observations[0])  # (160,160,3)
    assert frame.shape == (160, 160, 3) and frame.dtype == jnp.uint8


def test_terminal_timeout_both_and_partial_autoreset() -> None:
    env = GymnaxEnv("Breakout-MinAtar", max_steps_in_episode=2)
    _, state = env.reset(jax.random.key(0), 4)  # leaves: (B=4,...)
    # Missing the paddle, timeout, both, and continuing, respectively.
    state = state.replace(
        ball_y=jnp.array([8, 3, 8, 3]),
        ball_x=jnp.zeros(4, jnp.int32),
        ball_dir=jnp.full(4, 2),
        time=jnp.array([0, 1, 1, 0]),
    )
    result = env.step(jax.random.key(1), state, jnp.zeros(4, jnp.int32))
    np.testing.assert_array_equal(result.terminated, [True, False, True, False])
    np.testing.assert_array_equal(result.truncated, [False, True, True, False])
    np.testing.assert_array_equal(result.state.time, [0, 0, 0, 1])
    assert result.observations.shape == result.final_observations.shape == (4, 4, 10, 10)
    np.testing.assert_array_equal(result.observations[3], result.final_observations[3])
    assert not np.array_equal(result.observations[0], result.final_observations[0])
    assert result.rewards.shape == (4,)


def test_rejects_non_native_environment() -> None:
    with pytest.raises(ValueError, match="Gymnax"):
        GymnaxEnv("ALE/Breakout-v5")
