from dataclasses import replace

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxmarl.environments.smax.heuristic_enemy_smax_env import State
from jaxmarl.environments.smax.smax_env import State as BattleState
from pydantic import ValidationError

from rl2.envs.smax_env import SMAXConfig, SMAXEnv


@pytest.fixture(scope="module")
def env() -> SMAXEnv:
    return SMAXEnv(SMAXConfig(max_steps=2))


@pytest.fixture(scope="module")
def initial_state(env: SMAXEnv) -> State:
    return env.reset(jax.random.key(0))[1]


def test_reset_packing_and_batched_step(env: SMAXEnv, initial_state: State) -> None:
    observation, state = env.reset(jax.random.key(0))  # (N, O)/(N, A); state leaves: (...)
    raw, _ = env.env.reset(jax.random.key(0))  # agent leaves: (O,)
    assert observation.observation.shape == (3, env.env.observation_space(env.agents[0]).shape[0])
    assert observation.actions_mask.shape == (3, 8)
    assert observation.actions_mask.dtype == jnp.bool_
    for index, agent in enumerate(env.agents):
        np.testing.assert_array_equal(observation.observation[index], raw[agent])
    keys = jax.random.split(jax.random.key(1), 2)  # (E=2,)
    states = jax.tree.map(lambda x: jnp.stack([x, x]), state)  # leaves: (E, ...)
    result = jax.jit(jax.vmap(env.step))(keys, states, jnp.full((2, 3), 4))
    chex.assert_shape(result.reward, (2, 3))
    chex.assert_shape((result.agent_done, result.agent_inactive), (2, 3))
    assert result.agent_inactive.dtype == jnp.bool_
    np.testing.assert_array_equal(result.agent_inactive, False)
    chex.assert_shape(result.done, (2,))
    assert result.observation.observation.shape == (2, *observation.observation.shape)
    np.testing.assert_array_equal(result.done, False)
    np.testing.assert_array_equal(result.state.state.step, 1)
    chex.assert_trees_all_equal(state, initial_state)


@pytest.mark.parametrize("boundary", ["death", "timeout", "victory", "defeat"])
def test_boundaries_preserve_final_state_and_reset_only_completed_battles(
    env: SMAXEnv,
    initial_state: State,
    boundary: str,
) -> None:
    health = initial_state.state.unit_health  # (6,)
    if boundary == "death":
        health = health.at[0].set(0)
    elif boundary == "victory":
        health = health.at[3:].set(0)
    elif boundary == "defeat":
        health = health.at[:3].set(0)
    # Victory/defeat at the horizon exercise termination taking precedence over timeout.
    state = initial_state.replace(
        state=initial_state.state.replace(
            unit_health=health,
            unit_alive=health > 0,
            step=jnp.asarray(0 if boundary == "death" else 1),
        )
    )
    result = env.step(jax.random.key(2), state, jnp.full(3, 4))
    assert bool(result.terminated) == (boundary in ("victory", "defeat"))
    assert bool(result.truncated) == (boundary != "death")
    assert bool(result.done) == (boundary != "death")
    assert int(result.final_state.state.step) == (1 if boundary == "death" else 2)
    chex.assert_shape(result.agent_inactive, (3,))
    np.testing.assert_array_equal(
        result.agent_inactive, [boundary in ("death", "defeat"), boundary == "defeat", boundary == "defeat"]
    )
    if boundary == "death":
        np.testing.assert_array_equal(result.agent_done, [True, False, False])
        np.testing.assert_array_equal(result.observation.actions_mask[0], jnp.arange(8) == 4)
        chex.assert_trees_all_equal(result.state, result.final_state)
    else:
        np.testing.assert_array_equal(result.agent_done, True)
        assert int(result.state.state.step) == 0
        assert bool(result.state.state.unit_alive.all())
        assert not np.array_equal(result.observation.observation, result.final_observation.observation)


def test_render(env: SMAXEnv, initial_state: State) -> None:
    frame = env.render(initial_state)  # (H, W, 3)
    assert frame.shape == (400, 400, 3)
    assert frame.dtype == np.uint8


def test_invalid_config() -> None:
    with pytest.raises(ValidationError, match="max_steps"):
        SMAXConfig(max_steps=0)
    with pytest.raises(ValueError):
        SMAXEnv(replace(SMAXConfig(), map_name="not_a_map"))


@pytest.fixture(scope="module")
def selfplay_env() -> SMAXEnv:
    return SMAXEnv(SMAXConfig(selfplay=True, max_steps=2))


@pytest.fixture(scope="module")
def selfplay_state(selfplay_env: SMAXEnv) -> BattleState:
    return selfplay_env.reset(jax.random.key(0))[1]  # leaves: (...)


@pytest.mark.parametrize("map_name", ["3m", "5m_vs_6m"], ids=["equal_teams", "unequal_teams"])
def test_selfplay_packing_actions_and_batched_autoreset(map_name: str) -> None:
    env = SMAXEnv(SMAXConfig(map_name=map_name, selfplay=True, max_steps=2))
    keys = jax.random.split(jax.random.key(0), 2)  # (E=2,)
    observations, states = jax.jit(jax.vmap(env.reset))(keys)  # (E, N, O)/(E, N, A); leaves: (E, ...)
    state = jax.tree.map(lambda leaf: leaf[0], states)  # leaves: (...)
    raw, _ = env.env.reset(keys[0])  # agent leaves: (O,)
    masks = env.env.get_avail_actions(state)  # agent leaves: (A_agent,)
    num_allies, num_enemies = env.env.num_allies, env.env.num_enemies
    assert env.supports_selfplay
    assert env.num_agents == num_allies + num_enemies
    assert env.agents == tuple([f"ally_{i}" for i in range(num_allies)] + [f"enemy_{i}" for i in range(num_enemies)])
    assert observations.observation.shape == (2, env.num_agents, raw[env.agents[0]].shape[0])
    assert observations.actions_mask.shape == (2, env.num_agents, 5 + max(num_allies, num_enemies))
    assert observations.actions_mask.dtype == jnp.bool_
    for index, agent in enumerate(env.agents):
        native_size = masks[agent].shape[0]
        np.testing.assert_array_equal(observations.observation[0, index], raw[agent])
        np.testing.assert_array_equal(observations.actions_mask[0, index, :native_size], masks[agent])
        np.testing.assert_array_equal(observations.actions_mask[0, index, native_size:], False)

    # Keep allies still and move enemies; the second environment reaches its horizon.
    actions = jnp.full((2, env.num_agents), 4).at[:, num_allies:].set(0)  # (E, N)
    states = states.replace(step=jnp.asarray([0, 1]))  # leaves: (E, ...)
    result = jax.jit(jax.vmap(env.step))(keys, states, actions)
    chex.assert_shape((result.reward, result.agent_done, result.agent_inactive), (2, env.num_agents))
    chex.assert_shape(result.done, (2,))
    assert result.observation.observation.shape == observations.observation.shape
    assert result.observation.actions_mask.shape == observations.actions_mask.shape
    np.testing.assert_array_equal(result.done, [False, True])
    np.testing.assert_array_equal(result.terminated, False)
    np.testing.assert_array_equal(result.agent_inactive, False)
    np.testing.assert_array_equal(result.reward, 0)
    np.testing.assert_array_equal(result.state.step, [1, 0])
    np.testing.assert_array_equal(result.final_state.step, [1, 2])
    assert not np.array_equal(result.final_state.unit_positions[0, num_allies:], state.unit_positions[num_allies:])

    raw_observation, raw_state, raw_reward, _, _ = env.env.step_env(
        jax.random.split(keys[0])[0], state, {agent: actions[0, i] for i, agent in enumerate(env.agents)}
    )  # observation: agent (O,); state leaves: (...); rewards: agent ()
    # Batched and scalar action decoding can differ by float32 rounding.
    chex.assert_trees_all_close(jax.tree.map(lambda leaf: leaf[0], result.final_state), raw_state, atol=1e-6)
    for i, agent in enumerate(env.agents):
        np.testing.assert_allclose(result.final_observation.observation[0, i], raw_observation[agent], atol=1e-6)
        np.testing.assert_array_equal(result.reward[0, i], raw_reward[agent])


@pytest.mark.parametrize("boundary", ["enemy_death", "timeout", "victory", "defeat", "draw"])
def test_selfplay_rewards_and_boundaries(selfplay_env: SMAXEnv, selfplay_state: BattleState, boundary: str) -> None:
    health = selfplay_state.unit_health  # (N=6,)
    if boundary == "enemy_death":
        health = health.at[3].set(0)
    elif boundary == "victory":
        health = health.at[3:].set(0)
    elif boundary == "defeat":
        health = health.at[:3].set(0)
    elif boundary == "draw":
        health = jnp.zeros_like(health)
    state = selfplay_state.replace(
        unit_health=health,  # (N,)
        unit_alive=health > 0,  # (N,)
        step=jnp.asarray(0 if boundary == "enemy_death" else 1),  # ()
    )
    result = selfplay_env.step(jax.random.key(2), state, jnp.full(6, 4))
    assert bool(result.terminated) == (boundary in ("victory", "defeat", "draw"))
    assert bool(result.truncated) == (boundary != "enemy_death")
    assert bool(result.done) == (boundary != "enemy_death")
    assert bool(result.final_state.done) == bool(result.done)
    np.testing.assert_array_equal(result.agent_inactive, health == 0)
    ally_reward = 1 if boundary == "victory" else -1 if boundary == "defeat" else 0
    np.testing.assert_array_equal(result.reward, [ally_reward] * 3 + [-ally_reward] * 3)
    if boundary == "enemy_death":
        np.testing.assert_array_equal(result.agent_done, [False, False, False, True, False, False])
        np.testing.assert_array_equal(result.observation.actions_mask[3], jnp.arange(8) == 4)
        chex.assert_trees_all_equal(result.state, result.final_state)
    else:
        np.testing.assert_array_equal(result.agent_done, True)
        assert int(result.final_state.step) == 2
        assert int(result.state.step) == 0
        np.testing.assert_array_equal(result.state.unit_alive, True)
        assert not np.array_equal(result.observation.observation, result.final_observation.observation)


def test_selfplay_render(selfplay_env: SMAXEnv, selfplay_state: BattleState) -> None:
    frame = selfplay_env.render(selfplay_state)  # (H, W, 3)
    assert frame.shape == (400, 400, 3)
    assert frame.dtype == np.uint8
