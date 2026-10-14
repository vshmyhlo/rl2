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
from rl2.envs.storm_env import STORMConfig, STORMEnv

PRESETS = Path(__file__).resolve().parents[1] / "configs/mappo/jaxmarl/storm"


@pytest.mark.parametrize("preset,num_agents,grid_size", [("basic", 2, 6), ("advanced", 4, 8)])
def test_shapes_interactions_and_timeout(preset: str, num_agents: int, grid_size: int) -> None:
    config = ppo.load_config(PRESETS / f"{preset}.yaml", ppo.Config)
    env = ppo.make_env(replace(config, env=replace(config.env, max_steps=2)))
    assert isinstance(env, STORMEnv)
    assert env.num_agents == num_agents and env.config.grid_size == grid_size
    observation, state = env.reset(jax.random.key(0))  # (N, 350)/(N, 8); state leaves: (...)
    chex.assert_shape(observation.observations, (num_agents, 350))
    chex.assert_shape(observation.action_masks, (num_agents, 8))
    assert observation.observations.dtype == jnp.float32
    assert observation.action_masks.dtype == jnp.bool_
    np.testing.assert_array_equal(observation.action_masks, True)
    np.testing.assert_array_equal(observation.observations, env.env.get_obs(state).reshape(num_agents, -1))
    first = env.step(jax.random.key(1), state, jnp.full(num_agents, 3, jnp.int32))
    assert not bool(first.done)
    np.testing.assert_array_equal(first.rewards, 0)
    np.testing.assert_array_equal(first.agent_dones, False)
    chex.assert_trees_all_equal(first.state, first.final_state)

    # Place a red-coin holder facing a blue-coin holder; other agents stay clear.
    locations = jnp.array([[2, 2, 0], [2, 3, 2], [0, 0, 0], [0, 1, 0]], jnp.int16)[:num_agents]  # (N, 3)
    grid = jnp.zeros_like(state.grid).at[locations[:, 0], locations[:, 1]].set(env.env._agents)  # (G, G)
    inventories = jnp.zeros_like(state.agent_invs).at[:2].set(jnp.array([[1, 0], [0, 1]], jnp.int8))  # (N, 2)
    state = state.replace(agent_locs=locations, grid=grid, agent_invs=inventories)
    states = jax.tree.map(lambda x: jnp.stack([x, x]), state)  # leaves: (E=2, ...)
    states = states.replace(step=jnp.array([0, 1], jnp.int32))
    actions = jnp.full((2, num_agents), 3, jnp.int32).at[:, 0].set(4)  # (E, N)
    result = jax.jit(jax.vmap(env.step))(jax.random.split(jax.random.key(2), 2), states, actions)
    chex.assert_shape(result.observation.observations, (2, num_agents, 350))
    chex.assert_shape(result.observation.action_masks, (2, num_agents, 8))
    chex.assert_shape(result.rewards, (2, num_agents))
    chex.assert_shape(result.agent_dones, (2, num_agents))
    expected = [1, 1] if preset == "basic" else [0, 5, 0, 0]
    np.testing.assert_allclose(result.rewards, [expected, expected])
    np.testing.assert_array_equal(result.done, [False, True])
    np.testing.assert_array_equal(result.truncated, [False, True])
    np.testing.assert_array_equal(result.terminated, False)
    np.testing.assert_array_equal(result.agent_dones, [[False] * num_agents, [True] * num_agents])
    np.testing.assert_array_equal(result.final_state.step, [1, 2])
    np.testing.assert_array_equal(result.state.step, [1, 0])
    np.testing.assert_array_equal(jax.vmap(env.memory_resets)(result.final_state), False)
    raw = jax.vmap(env.env.get_obs)(result.final_state)  # (E, N, 5, 5, 14)
    np.testing.assert_array_equal(result.final_observation.observations, raw.reshape(2, num_agents, -1))
    assert not np.array_equal(result.observation.observations[1], result.final_observation.observations[1])
    frozen_state = jax.tree.map(lambda x: x[0], result.state)  # leaves: (...)
    frozen = env.step(jax.random.key(3), frozen_state, jnp.full(num_agents, 4, jnp.int32))
    np.testing.assert_array_equal(frozen.rewards, 0)  # A frozen pair cannot earn the payoff again.
    np.testing.assert_array_equal(frozen.final_state.agent_locs[:2], locations[:2])


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_agents": 1},
        {"grid_size": 4},
        {"num_coins": 0},
        {"max_steps": 0},
        {"freeze_penalty": 0},
        {"game": "unknown"},
        {"grid_size": 5, "num_coins": 12},
    ],
)
def test_invalid_settings(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        STORMConfig(**overrides)


def test_simultaneous_mixed_payoffs_and_respawn() -> None:
    env = STORMEnv(STORMConfig(num_agents=4, max_steps=3, freeze_penalty=1))
    _, state = env.reset(jax.random.key(0))  # state leaves: (...)
    locations = jnp.array([[2, 2, 0], [2, 3, 2], [5, 2, 0], [5, 3, 2]], jnp.int16)  # (N=4, 3)
    grid = jnp.zeros_like(state.grid).at[locations[:, 0], locations[:, 1]].set(env.env._agents)  # (8, 8)
    inventories = jnp.array([[1, 1], [1, 0], [0, 1], [0, 1]], jnp.int8)  # (N, 2)
    state = state.replace(agent_locs=locations, grid=grid, agent_invs=inventories)
    interaction = env.step(jax.random.key(1), state, jnp.array([4, 3, 4, 3], jnp.int32))
    np.testing.assert_allclose(interaction.rewards, [4, 1.5, 1, 1])
    respawn = env.step(jax.random.key(2), interaction.state, jnp.full(4, 3, jnp.int32))
    np.testing.assert_array_equal(respawn.rewards, 0)
    np.testing.assert_array_equal(respawn.state.agent_invs, 0)
    np.testing.assert_array_equal(respawn.agent_dones, False)
    np.testing.assert_array_equal(env.memory_resets(respawn.state), False)
    assert not bool(respawn.done)


def test_update_and_playback() -> None:
    config = replace(
        ppo.load_config(PRESETS / "basic.yaml", ppo.Config),
        env=STORMConfig(game="coordination", grid_size=6, max_steps=1),
        num_envs=1,
        num_steps=1,
        num_minibatches=1,
        update_epochs=1,
        total_steps=1,
        model=ppo.ModelConfig(encoder=ppo.MLPEncoderConfig(embedding_size=4), recurrent=ppo.LSTMConfig(hidden_size=4)),
    )
    env = ppo.make_env(config)
    observation, state = jax.vmap(env.reset)(
        jax.random.split(jax.random.key(0), 1)
    )  # (E, N, O)/(E, N, A); state leaves: (E, ...)
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
    assert all(frame.shape == (272, 256, 3) and frame.dtype == np.uint8 for frame in episode.frames)
