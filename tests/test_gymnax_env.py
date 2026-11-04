import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from rl2.envs import EnvStep
from rl2.envs.gymnax_env import GymnaxConfig, GymnaxEnv, render_observation


@pytest.mark.parametrize("env_id,channels,actions", [("Breakout-MinAtar", 4, 3), ("SpaceInvaders-MinAtar", 6, 4)])
def test_single_reset_step_preserves_native_observations(env_id: str, channels: int, actions: int) -> None:
    env = GymnaxEnv(GymnaxConfig(id=env_id, max_steps=1))
    params = env.params
    assert params.max_steps_in_episode == 1
    key = jax.random.key(5)  # ()
    observation, state = env.reset(key, params)  # (10,10,C); leaves: (...)
    raw, raw_state = env.env.reset(key, params)  # (10,10,C); leaves: (...)
    chex.assert_trees_all_equal(state, raw_state)
    np.testing.assert_array_equal(observation, raw)
    assert observation.shape == (10, 10, channels) and observation.dtype == jnp.float32
    assert env.action_space.n == actions
    action = jnp.array(2)  # () native action ID
    native = env.env.step_env(jax.random.split(key)[0], state, action, params.replace(max_steps_in_episode=2**31 - 1))
    result = env.step(key, state, action, params)  # observation: (10,10,C); state leaves: (...); reward/done: ()
    assert isinstance(result, EnvStep)
    observation, state, reward, done, info = result  # (10,10,C); leaves: (...); scalars: ()
    assert result.observation is observation and result.state is state
    assert result.reward is reward and result.done is done and result.info is info
    # observation/final_observation: (10,10,C); reward/done/flags: (); state leaves: (...)
    assert observation.shape == info["final_observation"].shape == (10, 10, channels)
    chex.assert_shape((reward, done, info["terminated"], info["truncated"], state.time), ())
    assert bool(done) and bool(info["truncated"]) and not bool(info["terminated"])
    assert int(state.time) == 0
    assert int(info["final_state"].time) == 1
    chex.assert_trees_all_equal((info["final_observation"], info["final_state"], reward), native[:3])
    frame = env.render(info["final_state"])  # (160,160,3)
    np.testing.assert_array_equal(frame, render_observation(info["final_observation"]))
    assert frame.shape == (160, 160, 3) and frame.dtype == jnp.uint8


def test_terminal_timeout_both_and_partial_autoreset() -> None:
    env = GymnaxEnv(GymnaxConfig(max_steps=2))
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


def test_space_invaders_batched_collision_and_fire() -> None:
    env = GymnaxEnv(GymnaxConfig(id="SpaceInvaders-MinAtar"))
    keys = jax.random.split(jax.random.key(0), 2)  # (B=2,)
    _, state = jax.vmap(env.reset, in_axes=(0, None))(keys, env.params)  # (B,10,10,6); leaves: (B,...)
    state = state.replace(e_bullet_map=state.e_bullet_map.at[0, 8, 5].set(1))  # leaves: (B,...)
    result = env.step_batch(keys, state, jnp.full(2, 3, jnp.int32), env.params)  # observation: (B,10,10,6)
    assert result.observation.shape == result.info["final_observation"].shape == (2, 10, 10, 6)
    assert result.reward.shape == (2,)
    np.testing.assert_array_equal(result.done, [True, False])
    np.testing.assert_array_equal(result.info["terminated"], [True, False])
    np.testing.assert_array_equal(result.info["truncated"], [False, False])
    np.testing.assert_array_equal(result.state.time, [0, 1])
    # Firing creates a friendly bullet; the incoming enemy bullet survives in the final observation.
    np.testing.assert_array_equal(result.info["final_observation"][:, 8, 5, 4], [1, 1])
    assert result.info["final_observation"][0, 9, 5, 5] == 1
    frame = env.render_observation(result.info["final_observation"][0])  # (160,160,3)
    np.testing.assert_array_equal(frame[8 * 16, 5 * 16], [80, 255, 80])


@pytest.mark.parametrize(
    "settings", [{"id": "ALE/Breakout-v5"}, {"max_steps": 0}, {"max_steps": True}, {"max_steps": 2**31}]
)
def test_invalid_config(settings: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        GymnaxConfig(**settings)


def test_default_horizon_preserves_native_limit() -> None:
    env = GymnaxEnv(GymnaxConfig())
    assert env.params.max_steps_in_episode == env.env.default_params.max_steps_in_episode


def test_four_rooms_visual_observations_and_episode_endings() -> None:
    env = GymnaxEnv(GymnaxConfig(id="FourRooms-misc", max_steps=2))
    params = env.params.replace(fail_prob=0.0)
    keys = jax.random.split(jax.random.key(0), 3)  # (B=3,)
    observations, state = jax.vmap(env.reset, in_axes=(0, None))(keys, params)  # (B,13,13,2); leaves: (B,...)
    native, _ = env.env.reset(keys[0], params)  # (13,13,2); leaves: (...)
    np.testing.assert_array_equal(observations[0], native)
    assert observations.shape == (3, 13, 13, 2) and observations.dtype == jnp.float32
    assert env.action_space.n == 4
    # Move right into the goal, hit the time limit, or continue normally.
    state = state.replace(
        pos=jnp.array([[8, 8], [4, 1], [4, 1]]),  # (B,2)
        time=jnp.array([0, 1, 0]),  # (B,)
    )
    result = jax.jit(env.step_batch)(keys, state, jnp.ones(3, jnp.int32), params)
    assert result.observation.shape == result.info["final_observation"].shape == (3, 13, 13, 2)
    assert result.reward.shape == (3,) and result.reward.dtype == jnp.float32
    np.testing.assert_array_equal(result.reward, [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(result.done, [True, True, False])
    np.testing.assert_array_equal(result.info["terminated"], [True, False, False])
    np.testing.assert_array_equal(result.info["truncated"], [False, True, False])
    np.testing.assert_array_equal(result.state.time, [0, 0, 1])
    np.testing.assert_array_equal(result.info["final_state"].time, [1, 2, 1])
    final_state = jax.tree.map(lambda leaf: leaf[2], result.info["final_state"])  # leaves: (...)
    frame = env.render(final_state)  # (208,208,3)
    assert frame.shape == (208, 208, 3) and frame.dtype == jnp.uint8
    np.testing.assert_array_equal(frame[8 * 16, 9 * 16], [80, 200, 80])
    np.testing.assert_array_equal(frame[4 * 16, 2 * 16], [80, 160, 255])
