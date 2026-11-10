from operator import itemgetter
from unittest.mock import patch

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, PartitionSpec
from PIL import ImageDraw
from pydantic import ValidationError

from rl2.actions import HierarchicalMultiDiscreteSpace
from rl2.envs.lux_env import LuxConfig, LuxEnv, LuxObservation, LuxState
from rl2.envs.multi_agent import AgentObservation


@pytest.fixture(scope="module")
def env() -> LuxEnv:
    return LuxEnv(LuxConfig(max_units=2, max_steps_in_match=2, match_count=2))


@pytest.fixture(scope="module")
def initial(env: LuxEnv) -> tuple[AgentObservation[LuxObservation[jax.Array]], LuxState]:
    return jax.jit(env.reset)(jax.random.key(0))  # (N, ...)/(N, U, A); state leaves: (...)


def test_shapes_randomization_and_action_encoding(
    env: LuxEnv, initial: tuple[AgentObservation[LuxObservation[jax.Array]], LuxState]
) -> None:
    observation, state = initial  # observation: (N, ...)/(N, U, A); state leaves: (...)
    assert env.num_agents == 2
    assert isinstance(env.action_space, HierarchicalMultiDiscreteSpace)
    assert env.action_space.nvec == (230, 230)
    assert env.action_space.group_sizes == (1, 4, 225)
    assert env.action_space.logits_shape == (2, 232)
    native = env.env.get_obs(state.game, state.params)  # per-team leaves: (...)
    for team in range(2):
        chex.assert_trees_all_equal(
            jax.tree.map(itemgetter(team), observation.observation.obs), native[f"player_{team}"]
        )
    space = env.observation_space  # per-team shape/dtype leaves: (...)
    chex.assert_trees_all_equal_shapes_and_dtypes(jax.tree.map(itemgetter(0), observation.observation), space)
    chex.assert_shape(observation.observation.obs.units.position, (2, 2, 2, 2))
    chex.assert_shape(observation.observation.obs.units.energy, (2, 2, 2))
    chex.assert_shape(observation.observation.obs.sensor_mask, (2, 24, 24))
    chex.assert_shape(observation.observation.obs.steps, (2,))
    np.testing.assert_array_equal(observation.observation.team_id, jnp.eye(2))
    np.testing.assert_array_equal(
        observation.observation.public_params,
        jnp.broadcast_to(
            jnp.array(
                [
                    state.params.unit_move_cost,
                    state.params.unit_sap_cost,
                    state.params.unit_sap_range,
                    state.params.unit_sensor_range,
                ]
            ),
            (2, 4),
        ),
    )
    chex.assert_shape(observation.actions_mask, (2, 2, 230))
    assert observation.observation.obs.units.position.dtype == jnp.int16
    assert observation.observation.obs.sensor_mask.dtype == jnp.bool_
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
    chex.assert_shape(default_observation.observation.obs.units.position, (2, 2, 16, 2))
    chex.assert_shape(default_observation.observation.obs.units_mask, (2, 2, 16))
    chex.assert_shape(default_observation.actions_mask, (2, 16, 230))
    chex.assert_shape(default_state.game.units.position, (2, 16, 2))


def test_masks_and_hidden_information(
    env: LuxEnv, initial: tuple[AgentObservation[LuxObservation[jax.Array]], LuxState]
) -> None:
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
    packed = env._observation(obs, state)  # (N, ...)/(N, U, A)
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
    chex.assert_trees_all_equal(*jax.tree.map(itemgetter(0), (packed.observation, changed_obs.observation)))
    np.testing.assert_array_equal(packed.actions_mask[0], changed_obs.actions_mask[0])
    frame = env.render(state)  # (H, W, 3)
    assert frame.shape == (416, 640, 3) and frame.dtype == np.uint8


def test_native_transition_match_memory_and_batched_game_autoreset(
    env: LuxEnv, initial: tuple[AgentObservation[LuxObservation[jax.Array]], LuxState]
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
    chex.assert_shape(batched.observation.observation.obs.units.position, (2, 2, 2, 2, 2))
    chex.assert_shape(batched.observation.observation.obs.steps, (2, 2))
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
    np.testing.assert_array_equal(batched.observation.observation.obs.steps, [[2, 2], [0, 0]])
    np.testing.assert_array_equal(batched.final_observation.observation.obs.steps, [[2, 2], [6, 6]])


def test_batched_autoreset_guards_reset_without_gathering_environments(
    env: LuxEnv, initial: tuple[AgentObservation[LuxObservation[jax.Array]], LuxState]
) -> None:
    key = jax.random.key(7)  # ()
    fresh = jax.jit(env.reset)(key)  # observation/state leaves: (...)
    mesh = Mesh(devices=jax.devices()[:2], axis_names=("batch",), axis_types=(AxisType.Auto,))
    with jax.set_mesh(mesh):
        key, observation, state = jax.device_put((key, *initial), PartitionSpec())  # key: (); other leaves: (...)
        done = jax.device_put(jnp.zeros(2, bool), PartitionSpec("batch"))  # (E=2,)
        reset = (
            jax.jit(
                jax.vmap(env._autoreset, in_axes=(0, None, None, None)),
                in_shardings=(PartitionSpec("batch"), PartitionSpec(), PartitionSpec(), PartitionSpec()),
                out_shardings=PartitionSpec("batch"),
            )
            .lower(done, key, observation, state)
            .compile()
        )
        for flags in ((False, False), (False, True), (True, True)):
            done = jax.device_put(jnp.asarray(flags), PartitionSpec("batch"))  # (E,)
            actual = reset(done, key, observation, state)  # observation/state leaves: (E, ...)
            chex.assert_shape(actual[0].observation.obs.units.position, (2, 2, 2, 2, 2))
            chex.assert_shape(actual[0].actions_mask, (2, 2, 2, 230))
            for index, finished in enumerate(flags):
                selected = jax.tree.map(itemgetter(index), actual)  # leaves: (...)
                chex.assert_trees_all_equal(selected, fresh if finished else initial)
        hlo = reset.as_text()
        assert " conditional(" in hlo  # Reset generation remains guarded by the scalar batch predicate.
        assert " all-gather(" not in hlo


@pytest.mark.parametrize(
    "overrides",
    [{"max_units": 0}, {"max_units": 17}, {"max_steps_in_match": 1}, {"match_count": 0}, {"max_steps_in_match": 2**30}],
    ids=["empty-team", "too-many-ships", "invalid-map-horizon", "empty-game", "counter-overflow"],
)
def test_invalid_settings(overrides: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        LuxConfig(**overrides)


@pytest.mark.parametrize(
    "steps,match_steps,wins,status,turn,points",
    [
        (0, 0, (0, 0), "MATCH 1 / 2", "Turn 0 / 2", "37"),
        (2, 2, (0, 0), "MATCH 1 / 2", "Turn 1 / 2", "37"),
        (3, 0, (1, 0), "MATCH 1 / 2 COMPLETE", "Turn 2 / 2", "--"),
        (6, 0, (2, 0), "GAME OVER - TEAM 0 WINS", "Turn 2 / 2", "--"),
        (6, 0, (1, 1), "GAME OVER - TIE", "Turn 2 / 2", "--"),
    ],
    ids=["initial", "live", "match-boundary", "game-over", "even-match-tie"],
)
def test_render_progress_and_team_statistics(
    env: LuxEnv,
    initial: tuple[AgentObservation[LuxObservation[jax.Array]], LuxState],
    steps: int,
    match_steps: int,
    wins: tuple[int, int],
    status: str,
    turn: str,
    points: str,
) -> None:
    _, state = initial
    game = state.game.replace(
        steps=jnp.asarray(steps),
        match_steps=jnp.asarray(match_steps),  # each ()
        team_points=jnp.array([37, 19]),
        team_wins=jnp.array(wins),  # each (N,)
        units=state.game.units.replace(
            position=jnp.array([[[0, 0], [1, 1]], [[22, 22], [23, 23]]], dtype=jnp.int16),  # (N, U, 2)
            energy=jnp.array([[[80], [-1]], [[60], [40]]], dtype=jnp.int16),  # (N, U, 1)
        ),
        units_mask=jnp.ones((2, 2), dtype=bool),  # (N, U)
        relic_nodes_mask=jnp.array([True, False, False, True, False, False]),  # (R,)
        relic_nodes_map_weights=jnp.zeros((24, 24), dtype=jnp.int16).at[3, 4].set(1),  # (W, H)
    )
    with patch.object(ImageDraw.ImageDraw, "text", autospec=True, side_effect=ImageDraw.ImageDraw.text) as text:
        frame = env.render(state._replace(game=game))  # (416, 640, 3)
    labels = [call.args[2] for call in text.call_args_list]
    assert f"LUX S3  |  {status}" in labels
    assert f"{turn}   |   Game step {steps} / 6" in labels
    assert f"Relic points {points}   |   Wins {wins[0]}" in labels
    assert f"Relic points {'--' if points == '--' else '19'}   |   Wins {wins[1]}" in labels
    assert "Ships 1 / 2   |   Energy 80" in labels
    assert "Ships 2 / 2   |   Energy 100" in labels
    assert "Active relic nodes: 2" in labels
    assert ("Relic points reset at match end." in labels) == (steps > 0 and match_steps == 0)
    assert frame.shape == (416, 640, 3) and frame.dtype == np.uint8
    np.testing.assert_array_equal(frame[72 + 4 * 12 + 1, 16 + 3 * 12 + 1], [165, 140, 65])


def test_render_legend(env: LuxEnv, initial: tuple[AgentObservation[LuxObservation[jax.Array]], LuxState]) -> None:
    _, state = initial
    with patch.object(ImageDraw.ImageDraw, "text", autospec=True, side_effect=ImageDraw.ImageDraw.text) as text:
        frame = env.render(state)  # (416, 640, 3)
    labels = {call.args[2] for call in text.call_args_list}
    assert {"Empty", "Nebula", "Asteroid", "Relic", "Scoring tile", "Team 0", "Team 1"} <= labels
    np.testing.assert_array_equal(frame[380, [22, 130, 238]], [[25, 30, 45], [70, 45, 100], [105, 105, 115]])
    np.testing.assert_array_equal(frame[380, 346], [240, 205, 70])
    np.testing.assert_array_equal(frame[374, 454], [165, 140, 65])
    np.testing.assert_array_equal(frame[380, 454], [14, 18, 28])  # Scoring marker is an outline.
    np.testing.assert_array_equal(frame[402, [22, 130]], [[70, 180, 255], [255, 95, 85]])
