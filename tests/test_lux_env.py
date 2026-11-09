import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from rl2.envs.lux_env import LuxConfig, LuxEnv, LuxState
from rl2.envs.multi_agent import AgentObservation


@pytest.fixture(scope="module")
def env() -> LuxEnv:
    return LuxEnv(LuxConfig(max_units=2, max_steps_in_match=2, match_count=2))


@pytest.fixture(scope="module")
def initial(env: LuxEnv) -> tuple[AgentObservation, LuxState]:
    return jax.jit(env.reset)(jax.random.key(0))  # (N, O)/(N, U, A); state leaves: (...)


def test_shapes_randomization_and_action_encoding(env: LuxEnv, initial: tuple[AgentObservation, LuxState]) -> None:
    observation, state = initial  # observation: (N, O)/(N, U, A); state leaves: (...)
    assert env.num_agents == 2
    assert env.action_space.nvec == (230, 230)
    chex.assert_shape(observation.observation, (2, 1774))
    chex.assert_shape(observation.actions_mask, (2, 2, 230))
    assert observation.observation.dtype == jnp.float32
    assert observation.actions_mask.dtype == jnp.bool_
    np.testing.assert_array_equal(observation.actions_mask[..., 0], True)
    np.testing.assert_array_equal(observation.actions_mask[..., 1:], False)
    repeated = jax.jit(env.reset)(jax.random.key(0))
    chex.assert_trees_all_equal(initial, repeated)
    assert 30 <= int(state.params.unit_sap_cost) <= 50
    assert 3 <= int(state.params.unit_sap_range) <= 7
    commands = env._decode_action(jnp.array([[0, 4], [5, 229]], dtype=jnp.int32))  # (N, U, 3)
    np.testing.assert_array_equal(commands, [[[0, 0, 0], [4, 0, 0]], [[5, -7, -7], [5, 7, 7]]])
    center = env._decode_action(jnp.full((2, 2), 117, dtype=jnp.int32))  # (N, U, 3)
    np.testing.assert_array_equal(center[..., 1:], 0)
    # Default-size contract without compiling another simulator executable.
    default = LuxEnv(LuxConfig(randomize_params=False))
    default_observation, default_state = jax.eval_shape(default.reset, jax.random.key(0))
    chex.assert_shape(default_observation.observation, (2, 1886))
    chex.assert_shape(default_observation.actions_mask, (2, 16, 230))
    chex.assert_shape(default_state.game.units.position, (2, 16, 2))


def test_masks_and_hidden_information(env: LuxEnv, initial: tuple[AgentObservation, LuxState]) -> None:
    _, state = initial
    game = state.game.replace(
        units=state.game.units.replace(
            position=jnp.array([[[0, 0], [1, 1]], [[22, 22], [23, 23]]], dtype=jnp.int16),  # (N, U, 2)
            energy=jnp.array([[[100], [0]], [[100], [-1]]], dtype=jnp.int16),  # (N, U, 1)
        ),
        units_mask=jnp.ones((2, 2), dtype=bool),  # (N, U)
        match_steps=jnp.array(1, dtype=jnp.int32),  # ()
    )
    state = state._replace(game=game, params=state.params.replace(unit_sap_range=jnp.array(3)))
    obs = env.env.get_obs(game, state.params)  # per-team observation leaves: (...)
    packed = env._observation(obs, state)  # (N, O)/(N, U, A)
    np.testing.assert_array_equal(packed.actions_mask[:, 1, 1:], False)
    np.testing.assert_array_equal(packed.actions_mask[:, 0, :5], True)
    assert not bool(packed.actions_mask[0, 0, 5])  # Out-of-range, off-map sap.
    assert bool(packed.actions_mask[0, 0, 117])  # In-bounds sap at the ship's own tile.
    assert int(packed.actions_mask[0, 0, 5:].sum()) == 16  # Corner: offsets 0..3 on each axis.
    changed = state._replace(
        game=game.replace(
            units=game.units.replace(
                position=game.units.position.at[1].set(20),  # (N, U, 2), hidden opponents
                energy=game.units.energy.at[1].set(200),  # (N, U, 1)
            ),
            relic_node_configs=~game.relic_node_configs,  # (R, K, K), hidden scoring tiles
        ),
        params=state.params.replace(nebula_tile_energy_reduction=jnp.array(25)),
    )
    changed_obs = env._observation(env.env.get_obs(changed.game, changed.params), changed)
    np.testing.assert_array_equal(packed.observation[0], changed_obs.observation[0])
    np.testing.assert_array_equal(packed.actions_mask[0], changed_obs.actions_mask[0])
    frame = env.render(state)  # (H, W, 3)
    assert frame.shape == (288, 288, 3) and frame.dtype == np.uint8


def test_native_transition_match_memory_and_batched_game_autoreset(
    env: LuxEnv, initial: tuple[AgentObservation, LuxState]
) -> None:
    _, state = initial
    step = jax.jit(env.step)
    action = jnp.zeros((2, 2), dtype=jnp.int32)  # (N, U)
    key = jax.random.key(1)  # ()
    first = step(key, state, action)
    native = jax.jit(env.env.step_env)(
        jax.random.split(key)[0],
        state.game,
        {"player_0": jnp.zeros((2, 3), dtype=jnp.int16), "player_1": jnp.zeros((2, 3), dtype=jnp.int16)},
        state.params,
    )
    chex.assert_trees_all_equal(first.final_state.game, native[1])
    chex.assert_trees_all_equal(first.state.params, state.params)
    np.testing.assert_array_equal(first.reward, 0)
    np.testing.assert_array_equal(first.agent_done, False)
    # Advance directly to the last action-bearing step of the first match.
    boundary = first.state._replace(game=first.state.game.replace(steps=jnp.array(2), match_steps=jnp.array(2)))
    match = step(jax.random.key(2), boundary, action)
    assert not bool(match.done)
    assert int(match.state.game.match_steps) == 0
    assert float(match.reward.sum()) == 1
    np.testing.assert_array_equal(match.agent_done, False)
    np.testing.assert_array_equal(match.agent_inactive, False)
    chex.assert_trees_all_equal(match.state.params, state.params)
    final = match.state._replace(game=match.state.game.replace(steps=jnp.array(5), match_steps=jnp.array(2)))
    states = jax.tree.map(lambda a, b: jnp.stack((a, b)), first.state, final)  # leaves: (E=2, ...)
    batched = jax.jit(jax.vmap(env.step))(jax.random.split(jax.random.key(3), 2), states, jnp.stack((action, action)))
    chex.assert_shape(batched.observation.observation, (2, 2, 1774))
    chex.assert_shape(batched.observation.actions_mask, (2, 2, 2, 230))
    chex.assert_shape(batched.reward, (2, 2))
    np.testing.assert_array_equal(batched.done, [False, True])
    np.testing.assert_array_equal(batched.terminated, [False, True])
    np.testing.assert_array_equal(batched.truncated, False)
    np.testing.assert_array_equal(batched.agent_done, [[False, False], [True, True]])
    np.testing.assert_array_equal(batched.final_state.game.steps, [2, 6])
    np.testing.assert_array_equal(batched.state.game.steps, [2, 0])
    np.testing.assert_array_equal(batched.reward.sum(-1), [0, 1])
    np.testing.assert_array_equal(batched.final_state.game.team_wins[1].sum(), 2)
    chex.assert_trees_all_equal(batched.final_state.params, states.params)
    assert not np.array_equal(batched.observation.observation[1], batched.final_observation.observation[1])


@pytest.mark.parametrize(
    "overrides",
    [{"max_units": 0}, {"max_units": 17}, {"max_steps_in_match": 1}, {"match_count": 0}, {"max_steps_in_match": 2**30}],
    ids=["empty-team", "too-many-ships", "invalid-map-horizon", "empty-game", "counter-overflow"],
)
def test_invalid_settings(overrides: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        LuxConfig(**overrides)
