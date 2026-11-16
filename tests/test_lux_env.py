from dataclasses import replace
from operator import itemgetter
from typing import Literal
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
from rl2.envs.lux_env import LuxConfig, LuxEnv, LuxObservation, LuxRenderTransition, LuxState
from rl2.envs.multi_agent import AgentObservation


@pytest.fixture(scope="module")
def env() -> LuxEnv:
    return LuxEnv(LuxConfig(max_units=2, max_steps_in_match=2, match_count=2, render_vision="outline"))


@pytest.fixture(scope="module")
def initial(env: LuxEnv) -> tuple[AgentObservation[LuxObservation[jax.Array]], LuxState]:
    return jax.jit(env.reset)(jax.random.key(0))  # (N, ...)/(N, U, A); state leaves: (...)


@pytest.fixture
def sap_transition(initial: tuple[AgentObservation[LuxObservation[jax.Array]], LuxState]) -> LuxRenderTransition:
    _, state = initial
    game = state.game.replace(
        steps=jnp.asarray(2),
        match_steps=jnp.asarray(2),
        units=state.game.units.replace(
            position=jnp.array([[[3, 4], [0, 0]], [[20, 19], [0, 0]]], dtype=jnp.int16),  # (N, U, 2)
            energy=jnp.full((2, 2, 1), 30, dtype=jnp.int16),  # (N, U, 1)
        ),
        units_mask=jnp.array([[True, False], [True, False]]),  # (N, U)
        map_features=jax.tree.map(jnp.zeros_like, state.game.map_features),  # leaves: (W, H)
        sensor_mask=jnp.zeros_like(state.game.sensor_mask),  # (N, W, H)
        relic_nodes_mask=jnp.zeros_like(state.game.relic_nodes_mask),  # (R,)
        relic_nodes_map_weights=jnp.zeros_like(state.game.relic_nodes_map_weights),  # (W, H)
    )
    return LuxRenderTransition(
        previous_state=state._replace(game=game, params=state.params.replace(unit_sap_cost=30, unit_sap_range=3)),
        action=np.array([[163, 0], [71, 0]], dtype=np.int32),  # (N, U), offsets (3, 1) and (-3, -1)
    )


def test_render_sap_origins_targets_and_splash(env: LuxEnv, sap_transition: LuxRenderTransition) -> None:
    prior = sap_transition.previous_state
    # Both shooters disappear after firing, and the result is a match boundary.
    state = prior._replace(
        game=prior.game.replace(steps=jnp.asarray(3), match_steps=jnp.asarray(0), units_mask=jnp.zeros((2, 2), bool))
    )
    plain = env.render(state)  # (416, 640, 3)
    frame = env.render(state, transition=sap_transition)  # (416, 640, 3)
    assert frame.shape == (416, 640, 3) and frame.dtype == np.uint8
    for origin, target, color in (((3, 4), (6, 5), (70, 180, 255)), ((20, 19), (17, 18), (255, 95, 85))):
        for x, y in (origin, target):
            np.testing.assert_array_equal(frame[72 + y * 12 + 5, 16 + x * 12 + 5], color)
        x, y = target
        # Splash reaches all eight neighbors, but does not tint the next tile out.
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                pixel = (72 + (y + dy) * 12 + 1, 16 + (x + dx) * 12 + 1)
                assert np.any(frame[pixel] != plain[pixel])
        np.testing.assert_array_equal(frame[72 + (y + 2) * 12 + 1, 16 + x * 12 + 1], [25, 30, 45])
    np.testing.assert_array_equal(frame[72:360, 304:], plain[72:360, 304:])  # Scorecards stay unchanged.


def test_render_sap_unsigned_actions(env: LuxEnv, sap_transition: LuxRenderTransition) -> None:
    signed = env.render(sap_transition.previous_state, transition=sap_transition)  # (416, 640, 3)
    unsigned = env.render(
        sap_transition.previous_state,
        transition=sap_transition._replace(action=sap_transition.action.astype(np.uint8)),  # (N, U)
    )  # (416, 640, 3)
    np.testing.assert_array_equal(unsigned, signed)  # Negative offsets must not wrap in unsigned arithmetic.


def test_render_sap_after_collision_reuses_ship_slot(
    env: LuxEnv, initial: tuple[AgentObservation[LuxObservation[jax.Array]], LuxState]
) -> None:
    _, state = initial
    prior = state._replace(
        game=state.game.replace(
            steps=jnp.asarray(2),
            match_steps=jnp.asarray(2),
            units=state.game.units.replace(
                position=jnp.array([[[3, 4], [0, 0]], [[4, 4], [0, 0]]], dtype=jnp.int16),  # (N, U, 2)
                energy=jnp.array([[[50], [0]], [[150], [0]]], dtype=jnp.int16),  # (N, U, 1)
            ),
            units_mask=jnp.array([[True, False], [True, False]]),  # (N, U)
            map_features=jax.tree.map(jnp.zeros_like, state.game.map_features),  # leaves: (W, H)
        ),
        params=state.params.replace(spawn_rate=jnp.asarray(2)),
    )
    action = jnp.array([[119, 0], [4, 0]], dtype=jnp.int32)  # (N, U), sap (0, 2); enemy moves left
    result = jax.jit(env.step)(jax.random.key(21), prior, action)
    assert bool(result.final_state.game.units_mask[0, 0])
    np.testing.assert_array_equal(result.final_state.game.units.position[0, 0], [0, 0])  # Replacement ship.
    frame = env.render(
        result.final_state, transition=LuxRenderTransition(previous_state=prior, action=action)
    )  # (416, 640, 3)
    np.testing.assert_array_equal(frame[149, 57], [70, 180, 255])  # Original target (3, 6), not (0, 2).


@pytest.mark.parametrize("reason", ["absent", "energy", "range", "off-map", "match-reset", "move"])
def test_render_omits_invalid_saps(env: LuxEnv, sap_transition: LuxRenderTransition, reason: str) -> None:
    prior = sap_transition.previous_state
    game, params = prior
    action = np.array([[163, 0], [0, 0]], dtype=np.int32)  # (N, U)
    if reason == "absent":
        game = game.replace(units_mask=game.units_mask.at[0, 0].set(False))
    elif reason == "energy":
        game = game.replace(units=game.units.replace(energy=game.units.energy.at[0, 0, 0].set(29)))
    elif reason == "range":
        params = params.replace(unit_sap_range=2)
    elif reason == "off-map":
        game = game.replace(
            units=game.units.replace(
                position=game.units.position.at[0, 0].set(jnp.array([23, 4], dtype=jnp.int16))  # (N, U, 2)
            )
        )
    elif reason == "match-reset":
        game = game.replace(match_steps=jnp.asarray(0))
    else:
        action[0, 0] = 1
    transition = LuxRenderTransition(previous_state=LuxState(game=game, params=params), action=action)
    plain = env.render(prior)  # (416, 640, 3)
    frame = env.render(prior, transition=transition)  # (416, 640, 3)
    np.testing.assert_array_equal(frame[72:360, 16:304], plain[72:360, 16:304])


def test_render_sap_at_board_edges(env: LuxEnv, sap_transition: LuxRenderTransition) -> None:
    prior = sap_transition.previous_state
    prior = prior._replace(
        game=prior.game.replace(
            units=prior.game.units.replace(
                position=jnp.array([[[0, 0], [0, 0]], [[23, 23], [0, 0]]], dtype=jnp.int16)  # (N, U, 2)
            )
        )
    )
    transition = LuxRenderTransition(
        previous_state=prior,
        action=np.array([[117, 0], [117, 0]], dtype=np.int32),  # (N, U), zero-offset saps
    )
    plain = env.render(prior)  # (416, 640, 3)
    frame = env.render(prior, transition=transition)  # (416, 640, 3)
    assert np.any(frame[72:96, 16:40] != plain[72:96, 16:40])
    assert np.any(frame[336:360, 280:304] != plain[336:360, 280:304])
    np.testing.assert_array_equal(frame[:72], plain[:72])
    np.testing.assert_array_equal(frame[72:360, :16], plain[72:360, :16])
    np.testing.assert_array_equal(frame[360:390], plain[360:390])


def test_observation_converts_rectangular_native_maps_to_height_width(
    env: LuxEnv, initial: tuple[AgentObservation[LuxObservation[jax.Array]], LuxState]
) -> None:
    _, state = initial
    native = env.env.get_obs(state.game, state.params)  # per-team leaves: (...)
    energy = jnp.arange(15, dtype=jnp.int16).reshape((3, 5))  # native (W=3, H=5)
    for team in range(2):
        obs = native[f"player_{team}"]
        native[f"player_{team}"] = obs.replace(
            sensor_mask=(energy % 2 == team),  # native (W, H)
            map_features=obs.map_features.replace(energy=energy + team, tile_type=energy % 3),  # native (W, H)
        )
    world = jax.jit(env._observation)(native, state).observation.world
    assert world.sensor_mask.shape == (2, 5, 3)
    for team in range(2):
        np.testing.assert_array_equal(world.sensor_mask[team], (energy % 2 == team).T)
        np.testing.assert_array_equal(world.map_features.energy[team], (energy + team).T)
        np.testing.assert_array_equal(world.map_features.tile_type[team], (energy % 3).T)
    assert world.sensor_mask.dtype == jnp.bool_
    assert world.map_features.energy.dtype == world.map_features.tile_type.dtype == jnp.int16


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
        view = jax.tree.map(itemgetter(team), observation.observation)  # leaves: (...)
        raw = native[f"player_{team}"]  # leaves: (...)
        for relative, index in ((view.self, team), (view.opponent, 1 - team)):
            chex.assert_trees_all_equal(relative.units, jax.tree.map(itemgetter(index), raw.units))
            np.testing.assert_array_equal(relative.units_mask, raw.units_mask[index])
            np.testing.assert_array_equal(relative.points, raw.team_points[index])
            np.testing.assert_array_equal(relative.wins, raw.team_wins[index])
        for field in ("sensor_mask", "map_features"):
            chex.assert_trees_all_equal(
                getattr(view.world, field), jax.tree.map(lambda leaf: leaf.T, getattr(raw, field))
            )
        for field in ("relic_nodes", "relic_nodes_mask", "steps", "match_steps"):
            chex.assert_trees_all_equal(getattr(view.world, field), getattr(raw, field))
    space = env.observation_space  # per-team shape/dtype leaves: (...)
    chex.assert_trees_all_equal_shapes_and_dtypes(jax.tree.map(itemgetter(0), observation.observation), space)
    chex.assert_shape(observation.observation.self.units.position, (2, 2, 2))
    chex.assert_shape(observation.observation.self.units.energy, (2, 2))
    chex.assert_shape(observation.observation.world.sensor_mask, (2, 24, 24))
    chex.assert_shape(observation.observation.world.steps, (2,))
    assert observation.observation._fields == ("self", "opponent", "world", "public_params")
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
    assert observation.observation.self.units.position.dtype == jnp.int16
    assert observation.observation.world.sensor_mask.dtype == jnp.bool_
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
    chex.assert_shape(default_observation.observation.self.units.position, (2, 16, 2))
    chex.assert_shape(default_observation.observation.self.units_mask, (2, 16))
    chex.assert_shape(default_observation.actions_mask, (2, 16, 230))
    chex.assert_shape(default_state.game.units.position, (2, 16, 2))


def test_relative_teams_preserve_observer_visibility(
    env: LuxEnv, initial: tuple[AgentObservation[LuxObservation[jax.Array]], LuxState]
) -> None:
    _, state = initial
    game = state.game.replace(
        units=state.game.units.replace(
            position=jnp.array([[[0, 0], [1, 1]], [[22, 22], [23, 23]]], dtype=jnp.int16),  # (N, U, 2)
            energy=jnp.array([[[10], [20]], [[30], [40]]], dtype=jnp.int16),  # (N, U, 1)
        ),
        units_mask=jnp.ones((2, 2), dtype=bool),  # (N, U)
        sensor_mask=jnp.zeros((2, 24, 24), dtype=bool).at[0, 22, 22].set(True).at[1, 0, 0].set(True),  # (N, W, H)
        team_points=jnp.array([3, 7], dtype=jnp.int32),  # (N,)
        team_wins=jnp.array([1, 2], dtype=jnp.int32),  # (N,)
    )
    state = state._replace(game=game)
    view = env._observation(env.env.get_obs(game, state.params), state).observation  # leaves: (N, ...)
    chex.assert_trees_all_equal_shapes_and_dtypes(view.self, view.opponent)
    np.testing.assert_array_equal(view.self.units.position, game.units.position)
    np.testing.assert_array_equal(view.self.units.energy, [[10, 20], [30, 40]])
    np.testing.assert_array_equal(view.self.units_mask, True)
    np.testing.assert_array_equal(view.self.points, [3, 7])
    np.testing.assert_array_equal(view.self.wins, [1, 2])
    np.testing.assert_array_equal(view.opponent.units.position, [[[22, 22], [-1, -1]], [[0, 0], [-1, -1]]])
    np.testing.assert_array_equal(view.opponent.units.energy, [[30, -1], [10, -1]])
    np.testing.assert_array_equal(view.opponent.units_mask, [[True, False], [True, False]])
    np.testing.assert_array_equal(view.opponent.points, [7, 3])
    np.testing.assert_array_equal(view.opponent.wins, [2, 1])
    np.testing.assert_array_equal(view.world.sensor_mask, jnp.swapaxes(game.sensor_mask, -2, -1))


@pytest.mark.parametrize(
    ("energy", "present", "match_steps", "move", "sap"),
    [
        (30, False, 1, False, False),
        (30, True, 0, False, False),
        (-1, True, 1, False, False),
        (1, True, 1, False, False),
        (2, True, 1, True, False),
        (29, True, 1, True, False),
        (30, True, 1, True, True),
    ],
    ids=["absent", "match-boundary", "negative-energy", "below-move", "exact-move", "below-sap", "exact-sap"],
)
def test_action_mask_eligibility(
    env: LuxEnv,
    initial: tuple[AgentObservation[LuxObservation[jax.Array]], LuxState],
    energy: int,
    present: bool,
    match_steps: int,
    move: bool,
    sap: bool,
) -> None:
    _, state = initial
    game = state.game.replace(
        units=state.game.units.replace(
            position=jnp.full((2, 2, 2), 12, dtype=jnp.int16),  # (N, U, 2)
            energy=jnp.full((2, 2, 1), energy, dtype=jnp.int16),  # (N, U, 1)
        ),
        units_mask=jnp.full((2, 2), present),  # (N, U)
        match_steps=jnp.asarray(match_steps, dtype=jnp.int32),  # ()
    )
    state = state._replace(game=game, params=state.params.replace(unit_move_cost=2, unit_sap_cost=30, unit_sap_range=3))
    packed = env._observation(env.env.get_obs(game, state.params), state)
    mask = packed.actions_mask  # (N, U, A)
    assert mask.shape == (2, 2, 230) and mask.dtype == jnp.bool_
    np.testing.assert_array_equal(mask[..., 0], True)
    np.testing.assert_array_equal(mask[..., 1:5], move)
    np.testing.assert_array_equal(mask[..., 5:].sum(-1), 49 if sap else 0)
    # Inclusive Chebyshev range: diagonal (+3, +3) is valid; (+4, 0) is not.
    np.testing.assert_array_equal(mask[..., 165], sap)
    np.testing.assert_array_equal(mask[..., 177], False)


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
        map_features=state.game.map_features.replace(
            tile_type=state.game.map_features.tile_type.at[1, 0].set(2)  # (W, H), asteroid to the right
        ),
        sensor_mask=state.game.sensor_mask.at[0, 1, 0].set(True),  # (N, W, H)
        match_steps=jnp.array(1, dtype=jnp.int32),  # ()
    )
    state = state._replace(game=game, params=state.params.replace(unit_sap_range=jnp.array(3)))
    obs = env.env.get_obs(game, state.params)  # per-team observation leaves: (...)
    packed = env._observation(obs, state)  # (N, ...)/(N, U, A)
    np.testing.assert_array_equal(packed.actions_mask[:, 1, 1:], False)
    np.testing.assert_array_equal(packed.actions_mask[:, 0, :5], True)  # Includes off-map/visible-asteroid moves.
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
    chex.assert_shape(batched.observation.observation.self.units.position, (2, 2, 2, 2))
    chex.assert_shape(batched.observation.observation.world.steps, (2, 2))
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
    np.testing.assert_array_equal(batched.observation.observation.world.steps, [[2, 2], [0, 0]])
    np.testing.assert_array_equal(batched.final_observation.observation.world.steps, [[2, 2], [6, 6]])


@pytest.mark.parametrize(
    "next_points,match_steps,expected",
    [
        ((23, 11), 2, (0.5, -0.5)),
        ((22, 12), 2, (0.0, 0.0)),
        ((20, 10), 2, (0.0, 0.0)),
        ((0, 0), 0, (1.0, 0.0)),
    ],
    ids=["relative-gain", "equal-gains", "no-gain", "match-end"],
)
def test_point_reward_from_score_deltas(
    env: LuxEnv,
    initial: tuple[AgentObservation[LuxObservation[jax.Array]], LuxState],
    next_points: tuple[int, int],
    match_steps: int,
    expected: tuple[float, float],
) -> None:
    _, state = initial
    previous = state._replace(
        game=state.game.replace(team_points=jnp.array([20, 10], dtype=jnp.int32), match_steps=jnp.asarray(1))
    )  # state leaves: (...); team_points: (N,)
    wins = jnp.array([int(match_steps == 0), 0], dtype=jnp.int32)  # (N,)
    game = previous.game.replace(
        team_points=jnp.asarray(next_points, dtype=jnp.int32),  # (N,)
        team_wins=previous.game.team_wins + wins,  # (N,)
        match_steps=jnp.asarray(match_steps),  # ()
    )
    shaped = LuxEnv(replace(env.config, point_reward_weight=0.25))
    reward = shaped._reward(previous, game)  # (N,)
    chex.assert_shape(reward, (2,))
    assert reward.dtype == jnp.float32
    np.testing.assert_array_equal(reward, expected)
    np.testing.assert_array_equal(env._reward(previous, game), wins)  # Default disables shaping.


def test_point_reward_across_match_and_game_boundaries(
    env: LuxEnv, initial: tuple[AgentObservation[LuxObservation[jax.Array]], LuxState]
) -> None:
    shaped = LuxEnv(replace(env.config, point_reward_weight=0.25))
    _, state = initial
    game = state.game.replace(
        units=state.game.units.replace(
            position=jnp.array([[[3, 4], [3, 4]], [[20, 19], [20, 19]]], dtype=jnp.int16),  # (N, U, 2)
            energy=jnp.full((2, 2, 1), 300, dtype=jnp.int16),  # (N, U, 1)
        ),
        units_mask=jnp.ones((2, 2), dtype=bool),  # (N, U)
        relic_spawn_schedule=jnp.zeros_like(state.game.relic_spawn_schedule),  # (R,)
        relic_nodes_map_weights=jnp.zeros((24, 24), dtype=jnp.int16).at[3, 4].set(1),  # (W, H)
        team_points=jnp.array([7, 10], dtype=jnp.int32),  # (N,), opponent wins despite our scoring
    )
    states = jax.tree.map(
        lambda *leaves: jnp.stack(leaves),
        *(
            state._replace(
                game=game.replace(
                    steps=jnp.asarray(step),  # ()
                    match_steps=jnp.asarray(match_step),  # ()
                    team_points=game.team_points if match_step else jnp.zeros_like(game.team_points),  # (N,)
                )
            )
            for step, match_step in ((1, 1), (2, 2), (5, 2), (3, 0))
        ),
    )  # leaves: (E=4, ...), ordinary step, match end, game end, new match
    result = jax.jit(jax.vmap(shaped.step))(
        jax.random.split(jax.random.key(3), 4), states, jnp.zeros((4, 2, 2), dtype=jnp.int32)
    )
    chex.assert_shape(result.reward, (4, 2))
    assert result.reward.dtype == jnp.float32
    np.testing.assert_array_equal(result.reward, [[0.25, -0.25], [0, 1], [0, 1], [0, 0]])
    np.testing.assert_array_equal(result.done, [False, False, True, False])
    np.testing.assert_array_equal(result.final_state.game.team_points[:3], [[8, 10], [0, 0], [0, 0]])
    assert int(result.state.game.steps[2]) == 0  # Reward uses terminal state, before game autoreset.


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
            chex.assert_shape(actual[0].observation.self.units.position, (2, 2, 2, 2))
            chex.assert_shape(actual[0].actions_mask, (2, 2, 2, 230))
            for index, finished in enumerate(flags):
                selected = jax.tree.map(itemgetter(index), actual)  # leaves: (...)
                chex.assert_trees_all_equal(selected, fresh if finished else initial)
        hlo = reset.as_text()
        assert " conditional(" in hlo  # Reset generation remains guarded by the scalar batch predicate.
        assert " all-gather(" not in hlo


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_units": 0},
        {"max_units": 17},
        {"max_steps_in_match": 1},
        {"match_count": 0},
        {"max_steps_in_match": 2**30},
        {"point_reward_weight": -1},
    ],
    ids=[
        "empty-team",
        "too-many-ships",
        "invalid-map-horizon",
        "empty-game",
        "counter-overflow",
        "negative-reward-weight",
    ],
)
def test_invalid_settings(overrides: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        LuxConfig(**overrides)


@pytest.mark.parametrize("map_size", [0, 25, 32768], ids=["too-small", "noise-grid-alignment", "coordinate-overflow"])
def test_invalid_map_size(map_size: int) -> None:
    with pytest.raises(ValidationError, match="map_size"):
        LuxConfig(map_size=map_size)


def test_large_map_reset_step_and_render() -> None:
    env = LuxEnv(LuxConfig(map_size=128, max_units=2, randomize_params=False, render_vision="outline"))
    keys = jax.random.split(jax.random.key(42), 2)  # (E=2,)
    observation, state = jax.jit(jax.vmap(env.reset))(keys)  # observation/state leaves: (E, ...)
    assert env.params.map_width == env.params.map_height == 128
    chex.assert_shape(observation.observation.world.sensor_mask, (2, 2, 128, 128))
    chex.assert_shape(state.game.map_features.tile_type, (2, 128, 128))
    transition = jax.jit(jax.vmap(env.step))(keys, state, jnp.zeros((2, 2, 2), dtype=jnp.int32))
    # action: (E, N, U); transition leaves: (E, ...)
    chex.assert_shape(transition.observation.observation.world.map_features.energy, (2, 2, 128, 128))
    chex.assert_shape(transition.observation.actions_mask, (2, 2, 2, 230))
    rendered = jax.tree.map(itemgetter(0), transition.state)  # state leaves: (...)
    game = rendered.game.replace(
        units=rendered.game.units.replace(
            position=rendered.game.units.position.at[1, 0].set(jnp.array([127, 127], jnp.int16)),  # (N, U, 2)
            energy=rendered.game.units.energy.at[1, 0, 0].set(100),  # (N, U, 1)
        ),
        units_mask=jnp.zeros((2, 2), dtype=bool).at[1, 0].set(True),  # (N, U)
        map_features=rendered.game.map_features.replace(
            tile_type=jnp.zeros((128, 128), dtype=jnp.int16).at[3, 4].set(1),  # (W, H)
            energy=jnp.zeros((128, 128), dtype=jnp.int16),  # (W, H)
        ),
        sensor_mask=jnp.zeros((2, 128, 128), dtype=bool),  # (N, W, H)
        relic_nodes_mask=jnp.zeros(6, dtype=bool),  # (R,)
    )
    frame = env.render(rendered._replace(game=game))  # (416, 640, 3)
    assert frame.shape == (416, 640, 3)
    np.testing.assert_array_equal(frame[359, 303], [255, 95, 85])  # Bottom-right ship stays inside the board panel.
    # Nearest-neighbor resizing maps tile (3, 4) to columns 7..8, rows 9..10.
    np.testing.assert_array_equal(frame[81, 22], [25, 30, 45])  # Nebula outline must not spill into tile (2, 4).
    np.testing.assert_array_equal(frame[81:83, 23:25], np.tile([160, 100, 210], (2, 2, 1)))


def test_render_downsampled_map(initial: tuple[AgentObservation[LuxObservation[jax.Array]], LuxState]) -> None:
    _, state = initial
    env = LuxEnv(LuxConfig(map_size=512, max_units=2, randomize_params=False, render_vision="outline"))
    # Tile (1, 1) receives no pixels when Pillow downsamples 512 tiles to 288 pixels.
    game = state.game.replace(
        map_features=state.game.map_features.replace(
            tile_type=jnp.zeros((512, 512), dtype=jnp.int16).at[1, 1].set(1),  # (W, H)
            energy=jnp.zeros((512, 512), dtype=jnp.int16),  # (W, H)
        ),
        sensor_mask=jnp.zeros((2, 512, 512), dtype=bool),  # (N, W, H)
        relic_nodes=jnp.ones((6, 2), dtype=jnp.int16),  # (R, 2)
        relic_nodes_mask=jnp.array([True, False, False, True, False, False]),  # (R,)
        relic_nodes_map_weights=jnp.zeros((512, 512), dtype=jnp.int16).at[1, 1].set(1),  # (W, H)
        units=state.game.units.replace(
            position=jnp.array([[[1, 1], [0, 0]], [[0, 0], [0, 0]]], dtype=jnp.int16),  # (N, U, 2)
            energy=jnp.zeros((2, 2, 1), dtype=jnp.int16),  # (N, U, 1)
        ),
        units_mask=jnp.array([[True, False], [True, False]]),  # (N, U)
    )
    with patch.object(ImageDraw.ImageDraw, "text", autospec=True, side_effect=ImageDraw.ImageDraw.text) as text:
        frame = env.render(
            state._replace(game=game, params=state.params.replace(map_width=512, map_height=512))
        )  # (416, 640, 3)
    expected = np.tile([25, 30, 45], (288, 288, 1))  # (H, W, RGB)
    expected[0, 0] = [255, 95, 85]  # Team 1's one-pixel tile remains visible.
    np.testing.assert_array_equal(frame[72:360, 16:304], expected)
    labels = [call.args[2] for call in text.call_args_list if call.args[2].startswith("Scoring tiles:")]
    assert labels == ["Scoring tiles: 1", "Scoring tiles: 0"]  # Counts still include unrepresented tiles.
    rendered = state._replace(game=game, params=state.params.replace(map_width=512, map_height=512))
    prior = rendered._replace(
        game=game.replace(
            match_steps=jnp.asarray(1),
            units=game.units.replace(energy=jnp.full((2, 2, 1), 100, dtype=jnp.int16)),  # (N, U, 1)
        )
    )
    overlay = env.render(
        rendered,
        transition=LuxRenderTransition(
            previous_state=prior,
            action=np.array([[117, 0], [0, 0]], dtype=np.int32),  # (N, U)
        ),
    )  # (416, 640, 3)
    assert overlay.shape == frame.shape and overlay.dtype == np.uint8
    assert np.any(overlay[73, 17] != frame[73, 17])  # Splash tile (2, 2) survives the omitted center.
    np.testing.assert_array_equal(overlay[72:360, 304:], frame[72:360, 304:])


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


@pytest.mark.parametrize(
    "case,expected",
    [
        ("occupied", (1, 2)),
        ("inactive_relic", (1, 1)),
        ("non_scoring", (0, 0)),
        ("absent", (0, 0)),
        ("negative_energy", (0, 1)),
    ],
)
def test_render_scoring_occupancy(
    env: LuxEnv,
    initial: tuple[AgentObservation[LuxObservation[jax.Array]], LuxState],
    case: str,
    expected: tuple[int, int],
) -> None:
    _, state = initial
    game = state.game.replace(
        units=state.game.units.replace(
            position=jnp.array([[[1, 3], [1, 3]], [[2, 4], [3, 5]]], dtype=jnp.int16),  # (N, U, 2)
            energy=jnp.array([[[0], [10]], [[0], [10]]], dtype=jnp.int16),  # (N, U, 1), zero energy still counts
        ),
        units_mask=jnp.ones((2, 2), dtype=bool),  # (N, U)
        relic_nodes_mask=jnp.array([True, True, False, True, True, False]),  # (R,)
        relic_nodes_map_weights=jnp.zeros((24, 24), dtype=jnp.int16)
        .at[1, 3]
        .set(1)
        .at[2, 4]
        .set(1)
        .at[3, 5]
        .set(2),  # (W, H)
    )
    if case == "inactive_relic":
        game = game.replace(relic_nodes_mask=jnp.array([True, False, False, True, False, False]))  # (R,)
    elif case == "non_scoring":
        game = game.replace(relic_nodes_map_weights=jnp.zeros((24, 24), dtype=jnp.int16))  # (W, H)
    elif case == "absent":
        game = game.replace(units_mask=jnp.zeros((2, 2), dtype=bool))  # (N, U)
    elif case == "negative_energy":
        game = game.replace(
            units=game.units.replace(energy=jnp.full((2, 2, 1), -1, dtype=jnp.int16).at[1, 0, 0].set(0))  # (N, U, 1)
        )
    with patch.object(ImageDraw.ImageDraw, "text", autospec=True, side_effect=ImageDraw.ImageDraw.text) as text:
        env.render(state._replace(game=game))
    labels = [call.args[2] for call in text.call_args_list if call.args[2].startswith("Scoring tiles:")]
    assert labels == [f"Scoring tiles: {count}" for count in expected]


@pytest.mark.parametrize("reference", ["baseline", "reference_" * 20], ids=["short-name", "long-name"])
def test_render_team_labels(
    env: LuxEnv,
    initial: tuple[AgentObservation[LuxObservation[jax.Array]], LuxState],
    reference: str,
) -> None:
    _, state = initial
    original = env.render(state)  # (416, 640, 3)
    team_labels = (f"Reference: {reference}", "Trained policy")
    with patch.object(ImageDraw.ImageDraw, "text", autospec=True, side_effect=ImageDraw.ImageDraw.text) as text:
        frame = env.render(state, team_labels=team_labels)  # (416, 640, 3)
    assert frame.shape == (416, 640, 3) and frame.dtype == np.uint8
    # Labels stay inside the existing team boxes; the board and legends are unchanged.
    np.testing.assert_array_equal(frame[:, :324], original[:, :324])
    np.testing.assert_array_equal(frame[:72], original[:72])
    np.testing.assert_array_equal(frame[267:], original[267:])
    labels = {call.args[1]: call for call in text.call_args_list}
    for team, color in enumerate(((70, 180, 255), (255, 95, 85))):
        call = labels[336, 106 + 100 * team]
        label = call.args[2]
        assert call.kwargs["fill"] == color
        assert call.kwargs["font"].getlength(label) <= 276
        if team == 0 and reference != "baseline":
            assert label.endswith("...") and team_labels[team].startswith(label[:-3])
        else:
            assert label == team_labels[team]
    assert labels[336, 125].args[2].startswith("Relic points")
    assert labels[336, 145].args[2].startswith("Ships")


def test_render_ship_energy(env: LuxEnv, initial: tuple[AgentObservation[LuxObservation[jax.Array]], LuxState]) -> None:
    _, state = initial
    game = state.game.replace(
        units=state.game.units.replace(
            position=jnp.array([[[3, 4], [5, 4]], [[3, 7], [5, 7]]], dtype=jnp.int16),  # (N, U, 2)
            energy=jnp.array([[[0], [50]], [[100], [100]]], dtype=jnp.int16),  # (N, U, 1)
        ),
        units_mask=jnp.array([[True, True], [True, False]]),  # (N, U)
        map_features=jax.tree.map(jnp.zeros_like, state.game.map_features),  # leaves: (W, H)
        sensor_mask=jnp.zeros((2, 24, 24), dtype=bool),  # (N, W, H)
        relic_nodes_mask=jnp.zeros(6, dtype=bool),  # (R,)
    )
    frame = env.render(state._replace(game=game, params=state.params.replace(max_unit_energy=100)))  # (416, 640, 3)
    blue, red, empty = (70, 180, 255), (255, 95, 85), (25, 30, 45)
    np.testing.assert_array_equal(frame[130, 54:62], np.tile(empty, (8, 1)))  # Empty bar, tile (3, 4).
    np.testing.assert_array_equal(frame[130, 78:86], [blue] * 4 + [empty] * 4)  # Half-full bar, tile (5, 4).
    np.testing.assert_array_equal(frame[166, 54:62], np.tile(red, (8, 1)))  # Full bar, tile (3, 7).
    np.testing.assert_array_equal(frame[128, 80:85], np.tile(blue, (5, 1)))  # Last row of ship marker.
    np.testing.assert_array_equal(frame[129, 78:86], np.tile(empty, (8, 1)))  # Exactly one row before bar.
    np.testing.assert_array_equal(frame[165, 54:62], np.tile(empty, (8, 1)))  # Same gap for team 1.
    np.testing.assert_array_equal(frame[156:168, 76:88], np.tile(empty, (12, 12, 1)))  # Absent ship is not drawn.
    assert frame.shape == (416, 640, 3)  # No roster panel.


def test_render_stack_badge(initial: tuple[AgentObservation[LuxObservation[jax.Array]], LuxState]) -> None:
    _, state = initial
    env = LuxEnv(LuxConfig(randomize_params=False))
    game = state.game.replace(
        units=state.game.units.replace(
            position=jnp.full((2, 16, 2), 23, dtype=jnp.int16),  # (N, U, 2), bottom-right tile
            energy=jnp.broadcast_to(jnp.arange(16, dtype=jnp.int16)[None, :, None] * 25, (2, 16, 1)),  # (N, U, 1)
        ),
        units_mask=jnp.zeros((2, 16), dtype=bool).at[0].set(True),  # (N, U)
    )
    with patch.object(ImageDraw.ImageDraw, "text", autospec=True, side_effect=ImageDraw.ImageDraw.text) as text:
        frame = env.render(state._replace(game=game, params=state.params.replace(max_units=16)))  # (416, 640, 3)
    labels = {call.args[1]: call.args[2] for call in text.call_args_list}
    assert labels[298, 354] == "16"  # One count badge instead of overlapping ship markers/bars.
    assert frame.shape == (416, 640, 3) and frame.dtype == np.uint8
    np.testing.assert_array_equal(frame[348:360, 304:308], np.tile([14, 18, 28], (12, 4, 1)))  # Badge stays inside map.


def test_render_energy_field(
    env: LuxEnv, initial: tuple[AgentObservation[LuxObservation[jax.Array]], LuxState]
) -> None:
    _, state = initial
    game = state.game.replace(
        map_features=state.game.map_features.replace(
            energy=jnp.zeros((24, 24), dtype=jnp.int16)
            .at[1:6, 4]
            .set(jnp.array([-10, -5, 0, 15, 30], dtype=jnp.int16))
            .at[1:6, 2]
            .set(30),  # (W, H)
            tile_type=jnp.zeros((24, 24), dtype=jnp.int16).at[1, 2].set(1).at[2, 2].set(2),  # (W, H)
        ),
        units=state.game.units.replace(
            position=state.game.units.position.at[0, 0].set(jnp.array([4, 2], dtype=jnp.int16)),  # (N, U, 2)
            energy=state.game.units.energy.at[0, 0, 0].set(100),  # (N, U, 1)
        ),
        units_mask=jnp.zeros((2, 2), dtype=bool).at[0, 0].set(True),  # (N, U)
        relic_nodes=state.game.relic_nodes.at[0].set(jnp.array([3, 2], dtype=jnp.int16)),  # (R, 2)
        relic_nodes_mask=jnp.array([True, False, False, True, False, False]),  # (R,)
        relic_nodes_map_weights=jnp.zeros((24, 24), dtype=jnp.int16).at[5, 2].set(1),  # (W, H)
    )
    state = state._replace(game=game, params=state.params.replace(min_energy_per_tile=-10, max_energy_per_tile=30))
    frame = env.render(state)  # (416, 640, 3)
    np.testing.assert_array_equal(
        frame[72 + 4 * 12 + 6, 16 + np.arange(1, 6) * 12 + 6],
        [[205, 115, 45], [115, 72, 45], [25, 30, 45], [35, 100, 100], [45, 170, 155]],
    )
    np.testing.assert_array_equal(frame[72 + 1 * 12 + 6, 16 + 4 * 12 + 6], [25, 30, 45])  # No x/y swap.
    np.testing.assert_array_equal(frame[96, 28], [160, 100, 210])  # Nebula border.
    np.testing.assert_array_equal(frame[102, 34], [45, 170, 155])  # Energy remains visible inside nebula.
    np.testing.assert_array_equal(frame[102, 46], [105, 105, 115])  # Asteroid stays solid.
    np.testing.assert_array_equal(frame[102, 58], [240, 205, 70])  # Relic above heatmap.
    np.testing.assert_array_equal(frame[102, 70], [70, 180, 255])  # Ship above heatmap.
    np.testing.assert_array_equal(frame[97, 77], [165, 140, 65])  # Scoring outline above heatmap.

    # Changing the frame's extrema must not rescale an unchanged tile.
    game = game.replace(map_features=game.map_features.replace(energy=game.map_features.energy.clip(-5, 15)))
    next_frame = env.render(state._replace(game=game))  # (416, 640, 3)
    np.testing.assert_array_equal(next_frame[126, 46], frame[126, 46])


def test_render_team_visibility(
    env: LuxEnv, initial: tuple[AgentObservation[LuxObservation[jax.Array]], LuxState]
) -> None:
    _, state = initial
    sensor_mask = (
        jnp.zeros((2, 24, 24), dtype=bool).at[0, 1:4, 4:7].set(True).at[0, 2, 5].set(False).at[1, 3:6, 4:7].set(True)
    )  # (N, W, H), overlapping regions and an unseen hole in team 0's coverage
    game = state.game.replace(
        sensor_mask=sensor_mask,  # (N, W, H)
        map_features=jax.tree.map(jnp.zeros_like, state.game.map_features),  # leaves: (W, H)
        units_mask=jnp.zeros((2, 2), dtype=bool),  # (N, U)
        relic_nodes_mask=jnp.zeros(6, dtype=bool),  # (R,)
    )
    frame = env.render(state._replace(game=game))  # (416, 640, 3)
    blue, red, unseen = (70, 180, 255), (255, 95, 85), (25, 30, 45)
    np.testing.assert_array_equal(frame[120, 34], blue)
    np.testing.assert_array_equal(frame[120, 70], red)
    np.testing.assert_array_equal(frame[120, [54, 57]], [blue, red])  # Shared boundary keeps both colors.
    np.testing.assert_array_equal(frame[138, [39, 40]], [blue, unseen])  # Outline borders the unseen hole.
    np.testing.assert_array_equal(frame[126, 40], unseen)  # No edge between adjacent visible tiles.
    np.testing.assert_array_equal(frame[126, 34], unseen)  # Tile interiors retain their energy colors.
    np.testing.assert_array_equal(frame[84, 70], unseen)  # No x/y swap.
    np.testing.assert_array_equal(frame[120, 90], unseen)  # Unseen tiles stay unoutlined.

    for visible in (False, True):
        game = game.replace(sensor_mask=jnp.full((2, 24, 24), visible))  # mask: (N, W, H)
        frame = env.render(state._replace(game=game))  # (416, 640, 3)
        expected = np.broadcast_to(unseen, (286, 286, 3))  # (H, W, RGB)
        np.testing.assert_array_equal(frame[73:359, 17:303], expected)
        np.testing.assert_array_equal(frame[72, [16, 19]], [blue, red] if visible else [unseen, unseen])
        np.testing.assert_array_equal(frame[359, [17, 20]], [blue, red] if visible else [unseen, unseen])


@pytest.mark.parametrize("coverage", ["partial", "none", "all"])
def test_render_fog_of_war(env: LuxEnv, sap_transition: LuxRenderTransition, coverage: str) -> None:
    fog_env = LuxEnv(LuxConfig(max_units=2))
    assert fog_env.config.render_vision == "fog_of_war"
    state = sap_transition.previous_state
    game = state.game.replace(
        map_features=state.game.map_features.replace(
            energy=jnp.full((24, 24), 10, dtype=jnp.int16),  # (W, H)
            tile_type=jnp.zeros((24, 24), dtype=jnp.int16).at[3, 4].set(1).at[4, 4].set(2),  # (W, H)
        ),
        relic_nodes=state.game.relic_nodes.at[0].set(jnp.array([4, 5], dtype=jnp.int16)),  # (R, 2)
        relic_nodes_mask=jnp.array([True, False, False, True, False, False]),  # (R,)
        relic_nodes_map_weights=jnp.zeros((24, 24), dtype=jnp.int16).at[5, 5].set(1),  # (W, H)
    )
    full = env.render(state._replace(game=game), transition=sap_transition)  # (416, 640, 3), no outlines
    sensor_mask = np.full((2, 24, 24), coverage == "all", dtype=bool)  # (N, W, H)
    if coverage == "partial":
        sensor_mask[0, 2:7, 3:7] = True
        sensor_mask[1, 5:9, 5:9] = True  # Overlap and tiles visible only to team 1.
        sensor_mask[:, 3, 4] = False  # Hole hides a ship, nebula border, and sap origin.
        sensor_mask[:, 4:6, 5] = False  # Hidden relic/scoring/sap pixels.
    game = game.replace(sensor_mask=jnp.asarray(sensor_mask))
    with patch.object(ImageDraw.ImageDraw, "text", autospec=True, side_effect=ImageDraw.ImageDraw.text) as text:
        frame = fog_env.render(state._replace(game=game), transition=sap_transition)  # (416, 640, 3)
    assert frame.shape == (416, 640, 3) and frame.dtype == np.uint8
    visible = np.repeat(np.repeat(sensor_mask.any(axis=0).T, 12, axis=0), 12, axis=1)  # (288, 288)
    board, full_board = frame[72:360, 16:304], full[72:360, 16:304]  # each (288, 288, 3)
    np.testing.assert_array_equal(board[visible], full_board[visible])
    np.testing.assert_array_equal(board[~visible], np.tile([14, 18, 28], ((~visible).sum(), 1)))
    np.testing.assert_array_equal(frame[72:390, 304:], full[72:390, 304:])  # Scorecards and legend.
    assert "Fog of war; beams = sap" in {call.args[2] for call in text.call_args_list}


@pytest.mark.parametrize("map_size", [128, 512], ids=["fractional-upscale", "downsample"])
def test_render_fog_scaled_map(
    initial: tuple[AgentObservation[LuxObservation[jax.Array]], LuxState], map_size: int
) -> None:
    _, state = initial
    env = LuxEnv(LuxConfig(map_size=map_size, max_units=2))
    game = state.game.replace(
        map_features=jax.tree.map(
            lambda leaf: jnp.zeros((map_size, map_size), dtype=leaf.dtype), state.game.map_features
        ),  # leaves: (W, H)
        sensor_mask=jnp.zeros((2, map_size, map_size), dtype=bool).at[1, : map_size // 2, :].set(True),  # (N, W, H)
        units_mask=jnp.zeros((2, 2), dtype=bool),  # (N, U)
        relic_nodes_mask=jnp.zeros(6, dtype=bool),  # (R,)
        relic_nodes_map_weights=jnp.zeros((map_size, map_size), dtype=jnp.int16),  # (W, H)
    )
    frame = env.render(
        state._replace(game=game, params=state.params.replace(map_width=map_size, map_height=map_size))
    )  # (416, 640, 3)
    np.testing.assert_array_equal(frame[72:360, 16:160], np.tile([25, 30, 45], (288, 144, 1)))
    np.testing.assert_array_equal(frame[72:360, 160:304], np.tile([14, 18, 28], (288, 144, 1)))


def test_render_vision_validation() -> None:
    with pytest.raises(ValidationError, match="render_vision"):
        LuxConfig(render_vision="invalid")


@pytest.mark.parametrize("vision", ["outline", "fog_of_war"])
def test_render_legend(
    initial: tuple[AgentObservation[LuxObservation[jax.Array]], LuxState], vision: Literal["outline", "fog_of_war"]
) -> None:
    _, state = initial
    env = LuxEnv(LuxConfig(max_units=2, render_vision=vision))
    with patch.object(ImageDraw.ImageDraw, "text", autospec=True, side_effect=ImageDraw.ImageDraw.text) as text:
        frame = env.render(state)  # (416, 640, 3)
    labels = {call.args[2] for call in text.call_args_list}
    assert {"Empty", "Nebula", "Asteroid", "Relic", "Scoring tile", "Team 0", "Team 1"} <= labels
    assert "Tile energy / turn" in labels
    assert "SHIP ENERGY" not in labels
    assert (
        "Full map (hidden info); outlines = vision" if vision == "outline" else "Fog of war; either team's vision"
    ) in labels
    assert f"Nebula drain: {int(state.params.nebula_tile_energy_reduction)} / turn" in labels
    assert {str(int(state.params.min_energy_per_tile)), "0", str(int(state.params.max_energy_per_tile))} <= labels
    np.testing.assert_array_equal(frame[333, [324, 474, 624]], [[205, 115, 45], [25, 30, 45], [45, 170, 155]])
    np.testing.assert_array_equal(frame[380, [22, 130, 238]], [[25, 30, 45], [25, 30, 45], [105, 105, 115]])
    np.testing.assert_array_equal(frame[374, 130], [160, 100, 210])
    np.testing.assert_array_equal(frame[380, 346], [240, 205, 70])
    np.testing.assert_array_equal(frame[374, 454], [165, 140, 65])
    np.testing.assert_array_equal(frame[380, 454], [14, 18, 28])  # Scoring marker is an outline.
    np.testing.assert_array_equal(frame[402, [22, 130]], [[70, 180, 255], [255, 95, 85]])
