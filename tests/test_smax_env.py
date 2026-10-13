from dataclasses import replace

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxmarl.environments.smax.heuristic_enemy_smax_env import State
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
    assert observation.observations.shape == (3, env.env.observation_space(env.agents[0]).shape[0])
    assert observation.action_masks.shape == (3, 8)
    assert observation.action_masks.dtype == jnp.bool_
    for index, agent in enumerate(env.agents):
        np.testing.assert_array_equal(observation.observations[index], raw[agent])
    keys = jax.random.split(jax.random.key(1), 2)  # (E=2,)
    states = jax.tree.map(lambda x: jnp.stack([x, x]), state)  # leaves: (E, ...)
    result = jax.jit(jax.vmap(env.step))(keys, states, jnp.full((2, 3), 4))
    chex.assert_shape(result.rewards, (2, 3))
    chex.assert_shape(result.agent_dones, (2, 3))
    chex.assert_shape(result.done, (2,))
    assert result.observation.observations.shape == (2, *observation.observations.shape)
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
    if boundary == "death":
        np.testing.assert_array_equal(result.agent_dones, [True, False, False])
        np.testing.assert_array_equal(result.observation.action_masks[0], jnp.arange(8) == 4)
        chex.assert_trees_all_equal(result.state, result.final_state)
    else:
        np.testing.assert_array_equal(result.agent_dones, True)
        assert int(result.state.state.step) == 0
        assert bool(result.state.state.unit_alive.all())
        assert not np.array_equal(result.observation.observations, result.final_observation.observations)


def test_render(env: SMAXEnv, initial_state: State) -> None:
    frame = env.render(initial_state)  # (H, W, 3)
    assert frame.shape == (400, 400, 3)
    assert frame.dtype == np.uint8


def test_invalid_config() -> None:
    with pytest.raises(ValidationError, match="max_steps"):
        SMAXConfig(max_steps=0)
    with pytest.raises(ValueError):
        SMAXEnv(replace(SMAXConfig(), map_name="not_a_map"))
