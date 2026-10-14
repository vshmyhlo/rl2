from dataclasses import replace
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState
from pydantic import ValidationError

from rl2 import train_mappo as ppo
from rl2.envs.robotarium_env import RobotariumConfig, RobotariumEnv

PRESET = Path(__file__).resolve().parents[1] / "configs/mappo/jaxmarl/jaxrobotarium/material_transport.yaml"


def test_batched_shapes_observations_and_timeout() -> None:
    env = RobotariumEnv(RobotariumConfig(max_steps=2, update_frequency=1))
    # Regression: upstream reset stores tracers on the simulator, breaking pure JAX use.
    with jax.checking_leaks():
        observation, state = jax.jit(jax.vmap(env.reset))(
            jax.random.split(jax.random.key(0), 2)
        )  # (E=2, N=4, O=22)/(E, N, A=5); state leaves: (E, ...)
        first = jax.jit(jax.vmap(env.step))(jax.random.split(jax.random.key(1), 2), state, jnp.zeros((2, 4), jnp.int32))
    chex.assert_shape(observation.observations, (2, 4, 22))
    chex.assert_shape(observation.action_masks, (2, 4, 5))
    assert observation.observations.dtype == jnp.float32
    assert observation.action_masks.dtype == jnp.bool_
    np.testing.assert_array_equal(observation.action_masks, True)
    for i in range(4):
        np.testing.assert_array_equal(observation.observations[:, i, :3], state.p_pos[:, i])
        expected = jnp.roll(state.het_rep, -i, axis=1).reshape(2, -1)  # (E, 2N)
        np.testing.assert_array_equal(observation.observations[:, i, -8:], expected)
    np.testing.assert_array_equal(observation.observations[:, :, 12:14], 50)
    np.testing.assert_array_equal(first.done, False)
    chex.assert_trees_all_equal(first.state, first.final_state)
    result = jax.jit(jax.vmap(env.step))(
        jax.random.split(jax.random.key(2), 2), first.state, jnp.zeros((2, 4), jnp.int32)
    )
    chex.assert_shape(result.observation.observations, (2, 4, 22))
    chex.assert_shape(result.rewards, (2, 4))
    chex.assert_shape(result.done, (2,))
    chex.assert_shape(result.agent_dones, (2, 4))
    np.testing.assert_array_equal(result.truncated, True)
    np.testing.assert_array_equal(result.agent_dones, True)
    np.testing.assert_array_equal(result.terminated, False)
    np.testing.assert_array_equal(result.final_state.step, 2)
    np.testing.assert_array_equal(result.state.step, 0)
    np.testing.assert_array_equal(jax.vmap(env.memory_resets)(result.final_state), False)
    raw_final = jax.vmap(env.env.get_obs)(result.final_state)  # agent leaves: (E, O)
    for i, agent in enumerate(env.agents):
        np.testing.assert_array_equal(result.final_observation.observations[:, i], raw_final[agent])


@pytest.mark.parametrize("num_agents", [2, 15], ids=["smallest-team", "all-starting-positions"])
def test_team_size_boundaries(num_agents: int) -> None:
    config = RobotariumConfig()
    env = RobotariumEnv(replace(config, robots=(config.robots[0],) * num_agents, collision_avoidance=False))
    observation, state = env.reset(jax.random.key(0))  # (N, 5N+2)/(N, 5); state leaves: (...)
    chex.assert_shape(observation.observations, (num_agents, 5 * num_agents + 2))
    chex.assert_shape(observation.action_masks, (num_agents, 5))
    assert np.unique(state.p_pos[:, :2], axis=0).shape[0] == num_agents


@pytest.mark.parametrize(
    "overrides",
    [
        {"robots": []},
        {"robots": [{}] * 16},
        {"robots": [{"step_size": 0}, {}]},
        {"robots": [{"capacity": -1}, {}]},
        {"zone1_load": -1},
        {"zone2_load": -1},
        {"max_steps": 0},
        {"update_frequency": 0},
    ],
)
def test_invalid_settings(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RobotariumConfig(**overrides)


def test_pickup_dropoff_rewards_and_movement() -> None:
    env = RobotariumEnv(RobotariumConfig(collision_avoidance=False, update_frequency=1))
    _, state = env.reset(jax.random.key(0))  # observation: (N, O)/(N, A); state leaves: (...)
    # One pickup per source (different capacities), one dropoff, one idle robot.
    state = state.replace(
        p_pos=jnp.array([[0, 0, 0], [-1.3, 0, 0], [1.3, 0, 0], [0.5, 0.5, 0]], jnp.float32),  # (N, 3)
        payload=jnp.array([0, 1, 0, 0], jnp.int32),  # (N,)
        zone1_load=jnp.array([3.0]),  # (1,): less than the robot's capacity
        zone2_load=jnp.array([20.0]),  # (1,)
    )
    actions = jnp.zeros(4, jnp.int32)  # (N,)
    first = env.step(jax.random.key(1), state, actions)
    np.testing.assert_allclose(first.rewards, 2 * 0.25 + 0.75 - 0.1)
    np.testing.assert_array_equal(first.state.payload, [1, 0, 1, 0])
    np.testing.assert_array_equal(first.state.zone1_load, [0])
    np.testing.assert_array_equal(first.state.zone2_load, [5])
    second = env.step(jax.random.key(2), first.state, actions)
    np.testing.assert_allclose(second.rewards, -0.1)  # No repeat pickup while loaded.
    np.testing.assert_array_equal(second.state.zone2_load, [5])
    # Final unloading still incurs the pre-unloading time penalty; later steps give zero.
    last_load = first.state.replace(
        p_pos=first.state.p_pos.at[[0, 2], 0].set(-1.3),  # (N, 3)
        zone2_load=jnp.zeros(1),  # (1,)
    )
    finished = env.step(jax.random.key(3), last_load, actions)
    np.testing.assert_allclose(finished.rewards, 2 * 0.75 - 0.1)
    np.testing.assert_array_equal(finished.state.payload, 0)
    idle = env.step(jax.random.key(4), finished.state, actions)
    np.testing.assert_array_equal(idle.rewards, 0)
    assert not bool(idle.done)
    # Payload is deliberately absent from native observations.
    chex.assert_trees_all_equal(env.env.get_obs(state), env.env.get_obs(state.replace(payload=1 - state.payload)))
    for action, expected in enumerate([[0, 0], [0, 0.45], [0, -0.45], [0.45, 0], [-0.45, 0]]):
        np.testing.assert_allclose(env.env.action_decoder(0, action, state), expected)
    moved = env.step(jax.random.key(5), state, jnp.array([3, 0, 0, 0], jnp.int32))
    assert float(moved.state.p_pos[0, 0]) > 0
    np.testing.assert_array_equal(moved.state.p_pos[1:], state.p_pos[1:])


def test_mappo_update_and_playback() -> None:
    preset = ppo.load_config(PRESET, ppo.Config)
    config = replace(
        preset,
        env=replace(preset.env, robots=preset.env.robots[:2], max_steps=1, update_frequency=1),
        num_envs=1,
        num_steps=1,
        num_minibatches=1,
        update_epochs=1,
        total_steps=1,
        model=ppo.ModelConfig(encoder=ppo.MLPEncoderConfig(embedding_size=4), recurrent=ppo.LSTMConfig(hidden_size=4)),
    )
    env = ppo.make_env(config)
    assert isinstance(env, RobotariumEnv)
    observation, state = jax.vmap(env.reset)(
        jax.random.split(jax.random.key(0), 1)
    )  # (E=1, N=2, O=12)/(E, N, A=5); state leaves: (E, ...)
    model = ppo.ActorCritic(action_space=env.action_space, model=config.model)
    memory = model.initial_carry(2)  # leaves: (B=2, H=4)
    starts = jnp.ones(2, bool)  # (B,)
    params = model.init(
        jax.random.key(1), observation.observations, memory, starts[None], observation.action_masks
    )  # leaves: (...)
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params["params"],
        tx=optax.inject_hyperparams(optax.adam)(learning_rate=config.learning_rate),
    )
    runner = ppo.RolloutCarry(
        env_state=state,  # leaves: (E, ...)
        observations=observation,  # (E, N, O)/(E, N, A)
        model_carry=memory,  # leaves: (B, H)
        episode_starts=starts,  # (B,)
        episode_returns=jnp.zeros(1),  # (E,)
        episode_step_counts=jnp.zeros(1, jnp.int32),  # (E,)
    )
    result = ppo.train_step(runner, train_state, env, config, jax.random.key(2), ppo.init_metrics())
    assert int(result.train_state.step) == 1
    assert int(result.metrics.total.episode_count) == 1
    assert int(result.metrics.total.episode_length_sum) == 1
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(jax.tree.leaves(train_state.params), jax.tree.leaves(result.train_state.params))
    )
    episode = ppo.play_episode(result.train_state, config, jax.random.key(3), memory, record=True, env=env)
    assert episode.length == 1 and len(episode.frames) == 2
    assert all(frame.shape == (320, 480, 3) and frame.dtype == np.uint8 for frame in episode.frames)
