"""Lux model contracts: ship symmetry, spatial command layout, and recurrent replay."""

from dataclasses import replace
from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from jax.sharding import AxisType, Mesh, PartitionSpec
from test_lux_encoder import observation as observation  # noqa: PLC0414

from rl2.actions import HierarchicalMultiDiscreteSpace
from rl2.envs.lux_env import LuxConfig, LuxEnv, LuxObservation
from rl2.model.config import LSTMConfig, LuxModelConfig
from rl2.model.lux import LuxActorCritic, LuxAttentionBlock, LuxCarry, LuxFeatures, LuxModel, _ShipMemory


@pytest.fixture(scope="module")
def model() -> LuxActorCritic:
    return LuxActorCritic(
        action_space=HierarchicalMultiDiscreteSpace(nvec=(14,) * 3, group_sizes=(1, 4, 9)),
        model=LuxModelConfig(
            embedding_size=8,
            channels=4,
            heads=4,
            head_dim=2,
            recurrent=LSTMConfig(hidden_size=4),
            ship_recurrent=LSTMConfig(hidden_size=4, num_layers=2),
            attention_implementation="xla",
        ),
    )


def history(model: LuxActorCritic, observation: LuxObservation[jax.Array], batch_size: int) -> LuxCarry:
    """Return nonzero (B, ...) state with distinct ship histories and valid metadata."""
    carry = model.initial_carry(batch_size)
    return carry._replace(
        team=jax.tree.map(lambda leaf: leaf + 0.5, carry.team),  # (B, H)
        ships=jax.tree.map(
            lambda leaf: leaf + jnp.arange(leaf.shape[1])[None, :, None] + 0.5, carry.ships
        ),  # (B, U, Hs)
        ship_alive=jnp.ones_like(carry.ship_alive),  # (B, U)
        match_start=jnp.full_like(carry.match_start, observation.world.steps - observation.world.match_steps),  # (B,)
    )


def action_mask(observation: LuxObservation[jax.Array]) -> jax.Array:  # leaves: (*S, ...) -> (*S, U, A=14)
    return (
        jnp.broadcast_to(observation.self.units_mask[..., None], (*observation.self.units_mask.shape, 14))
        .at[..., 0]
        .set(True)
    )  # (*S, U, A)


@pytest.fixture(scope="module")
def params(model: LuxActorCritic, observation: LuxObservation[jax.Array]) -> dict[str, Any]:
    sequence = jax.tree.map(lambda leaf: leaf[None, None], observation)  # leaves: (B=1, T=1, ...)
    return model.init(
        jax.random.key(4), sequence, model.initial_carry(1), jnp.ones((1, 1), dtype=bool), action_mask(sequence)
    )


def test_recurrent_modules_initialize_carry_without_parameters(model: LuxActorCritic) -> None:
    config = model.model.ship_recurrent
    memory = _ShipMemory(config=config, dtype=jnp.bfloat16)
    block = LuxAttentionBlock(
        features=2 * model.model.channels,
        recurrent=config,
        attention_implementation="xla",
        dtype=jnp.bfloat16,
    )
    for module in (memory, block):
        carry = module.initial_carry(actor_count=2, unit_count=3)  # per-layer leaves: (B, U, Hs)
        assert len(carry) == config.num_layers
        for leaf in jax.tree.leaves(carry):
            assert leaf.shape == (2, 3, config.hidden_size) and leaf.dtype == jnp.float32
            np.testing.assert_array_equal(leaf, 0)
        bound = module.apply({}, 2, 3, method="initial_carry")  # per-layer leaves: (B, U, Hs)
        chex.assert_trees_all_equal(bound, carry)
    combined = model.initial_carry(2)
    assert len(combined.ships) == model.model.num_blocks
    for state in combined.ships:
        chex.assert_trees_all_equal(state, block.initial_carry(actor_count=2, unit_count=3))
    chex.assert_trees_all_equal(model.apply({}, 2, method="initial_carry"), combined)


def test_sequence_step_replay_reset_and_bfloat16(
    model: LuxActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    sequence = jax.tree.map(lambda leaf: jnp.broadcast_to(leaf, (2, 2, *leaf.shape)), observation)  # (B, T, ...)
    starts = jnp.array([[False, False], [False, True]])  # (B, T)
    memory = history(model, observation, 2)  # leaves: (B, ...)
    masks = action_mask(sequence).at[0, :, 0, 5:].set(False)  # (B, T, U, A), no sap for this ship
    output = jax.jit(model.apply)(params, sequence, memory, starts, masks)
    assert output.policy.raw_logits.shape == (2, 2, 3, 16)
    assert output.policy.logits.shape == (2, 2, 3, 14)
    assert output.value.shape == (2, 2) and output.value.dtype == jnp.float32
    first, second = jax.tree.map(lambda leaf: leaf[:, 0], sequence), jax.tree.map(lambda leaf: leaf[:, 1], sequence)
    # first/second leaves: (B, ...)
    step = jax.jit(lambda obs, carry, reset, mask: model.apply(params, obs, carry, reset, mask, method="step"))
    first_output = step(first, memory, starts[:, 0], masks[:, 0])
    second_output = step(second, first_output.carry, starts[:, 1], masks[:, 1])
    for actual, expected in zip(
        (output.policy.logits, output.value),
        (
            jnp.stack((first_output.policy.logits, second_output.policy.logits), 1),
            jnp.stack((first_output.value, second_output.value), 1),
        ),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, atol=2e-6)
    chex.assert_trees_all_close(output.carry, second_output.carry, atol=2e-6)
    fresh = step(second, model.initial_carry(2), jnp.ones(2, dtype=bool), masks[:, 1])
    np.testing.assert_allclose(second_output.value[1], fresh.value[1], atol=2e-6)
    actions, log_prob = output.policy.sample_and_log_prob(key=jax.random.key(5))  # (B, T, U); (B, T)
    assert actions.shape == (2, 2, 3) and log_prob.shape == (2, 2)
    assert bool(jnp.take_along_axis(masks, actions[..., None], axis=-1).all())
    np.testing.assert_array_equal(actions[..., 2], 0)  # Absent ship always stays.
    np.testing.assert_allclose(output.policy.log_prob(actions), log_prob, atol=1e-6)
    reduced = jax.jit(model.clone(dtype=jnp.bfloat16).apply)(params, sequence, memory, starts, masks)
    assert reduced.policy.raw_logits.dtype == reduced.value.dtype == jnp.float32
    reduced_model = model.clone(dtype=jnp.bfloat16)
    reduced_step = jax.jit(
        lambda variables, obs, carry, reset, mask: reduced_model.apply(
            variables, obs, carry, reset, mask, method="step"
        )
    )
    reduced_first = reduced_step(params, first, memory, starts[:, 0], masks[:, 0])
    reduced_second = reduced_step(params, second, reduced_first.carry, starts[:, 1], masks[:, 1])
    # Keep parameters dynamic in both paths, as in training, including bf16 casts.
    np.testing.assert_allclose(reduced.value, jnp.stack((reduced_first.value, reduced_second.value), 1), atol=2e-3)
    np.testing.assert_allclose(
        reduced.policy.raw_logits,
        jnp.stack((reduced_first.policy.raw_logits, reduced_second.policy.raw_logits), 1),
        atol=2e-3,
    )
    chex.assert_trees_all_close(reduced.carry, reduced_second.carry, atol=2e-3)
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves((reduced.carry.team, reduced.carry.ships)))
    np.testing.assert_allclose(reduced.policy.raw_logits, output.policy.raw_logits, atol=0.01, rtol=0.04)


def test_ship_permutation_equivariance(
    model: LuxActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    permutation = jnp.array([1, 2, 0])  # (U,)
    team = observation.self._replace(
        units=jax.tree.map(lambda leaf: leaf[permutation], observation.self.units),  # position: (U, 2); energy: (U,)
        units_mask=observation.self.units_mask[permutation],  # (U,)
    )
    reordered_world = observation._replace(
        opponent=observation.opponent._replace(
            units=jax.tree.map(lambda leaf: leaf[permutation], observation.opponent.units),
            units_mask=observation.opponent.units_mask[permutation],  # (V,)
        ),
        world=observation.world._replace(
            relic_nodes=observation.world.relic_nodes[::-1],  # (R, 2)
            relic_nodes_mask=observation.world.relic_nodes_mask[::-1],  # (R,)
        ),
    )
    batch = jax.tree.map(
        lambda *leaves: jnp.stack(leaves), observation, observation._replace(self=team), reordered_world
    )  # (B=3, ...)
    memory = history(model, observation, 3)  # leaves: (B, ...)
    memory = memory._replace(
        ships=jax.tree.map(lambda leaf: leaf.at[1].set(leaf[1, permutation]), memory.ships),  # (B, U, Hs)
        ship_alive=memory.ship_alive.at[1].set(memory.ship_alive[1, permutation]),  # (B, U)
    )
    output = jax.jit(lambda obs: model.apply(params, obs, memory, jnp.zeros(3, bool), action_mask(obs), method="step"))(
        batch
    )
    np.testing.assert_allclose(output.policy.logits[1], output.policy.logits[0, permutation], atol=2e-6)
    np.testing.assert_allclose(output.policy.logits[2], output.policy.logits[0], atol=2e-6)
    np.testing.assert_allclose(output.value[1:], jnp.broadcast_to(output.value[0], (2,)), atol=2e-6)
    for leaf in jax.tree.leaves((output.carry.team, output.carry.match_start)):
        np.testing.assert_allclose(leaf[1:], jnp.broadcast_to(leaf[0], leaf[1:].shape), atol=2e-6)
    for leaf in jax.tree.leaves((output.carry.ships, output.carry.ship_alive)):
        np.testing.assert_allclose(leaf[1], leaf[0, permutation], atol=2e-6)
        np.testing.assert_allclose(leaf[2], leaf[0], atol=2e-6)


def test_empty_team_has_no_policy_gradient(
    model: LuxActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    empty = observation._replace(self=observation.self._replace(units_mask=jnp.zeros(3, dtype=bool)))
    empty = jax.tree.map(lambda leaf: leaf[None, None], empty)  # leaves: (B=1, T=1, ...)

    def loss(variables: dict[str, Any]) -> tuple[jax.Array, jax.Array]:  # parameter leaves: (...) -> (); (B, T)
        output = model.apply(variables, empty, model.initial_carry(1), jnp.ones((1, 1), bool), action_mask(empty))
        return output.policy.log_prob(
            jnp.zeros((1, 1, 3), dtype=jnp.int32)
        ).sum(), output.policy.entropy()  # (); (B, T)

    (log_prob, entropy), gradient = jax.jit(jax.value_and_grad(loss, has_aux=True))(params)
    assert float(log_prob) == 0
    np.testing.assert_array_equal(entropy, 0)
    for leaf in jax.tree.leaves(gradient):
        np.testing.assert_array_equal(leaf, 0)


def test_sap_scores_follow_relative_target_coordinates(
    model: LuxActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    network = LuxModel(embedding_size=8, channels=4, action_space=model.action_space)
    variables = jax.tree.map(lambda leaf: leaf, params["params"]["lux"])
    variables["sap_query"] = {"kernel": jnp.zeros((8, 4)), "bias": jnp.ones(4)}
    variables["sap_key"] = {"kernel": jnp.zeros((4, 4)).at[0].set(1), "bias": jnp.zeros(4)}
    variables["sap_offset"] = jax.tree.map(jnp.zeros_like, variables["sap_offset"])
    x, y = jnp.meshgrid(jnp.arange(8), jnp.arange(12), indexing="xy")  # each (H, W)
    spatial = jnp.zeros((1, 12, 8, 4)).at[0, :, :, 0].set(10 * x + y)  # (B, H, W, C)
    features = LuxFeatures(global_features=jnp.zeros((1, 8)), ships=jnp.zeros((1, 3, 8)), spatial=spatial)
    obs = jax.tree.map(lambda leaf: leaf[None], observation)  # leaves: (B, ...)
    policy, value = network.apply(
        {"params": variables}, jnp.zeros((1, 4)), jnp.zeros((1, 3, 4)), features, obs, action_mask(obs), method="heads"
    )
    # First ship is at (1, 2); offsets enumerate x first, then y, matching Lux's command IDs.
    np.testing.assert_allclose(policy.raw_logits[0, 0, 7:], 2 * jnp.array([1, 2, 3, 11, 12, 13, 21, 22, 23]))
    assert value.shape == (1,)


def test_ship_spawn_corner_uses_xy_coordinates_on_rectangular_map(
    model: LuxActorCritic, observation: LuxObservation[jax.Array]
) -> None:
    observation = observation._replace(
        self=observation.self._replace(
            units=observation.self.units.replace(
                position=jnp.array([[7, 11], [5, 8], [-1, -1]], dtype=jnp.int16),  # (U, 2), first ship at far corner
                energy=jnp.full(3, 100, dtype=jnp.int16),  # (U,), native spawn energy
            ),
        ),
        world=observation.world._replace(steps=jnp.asarray(107), match_steps=jnp.asarray(7)),  # scalar counters
    )
    carry = history(model, observation, batch_size=1)
    sequence = jax.tree.map(lambda leaf: leaf[None, None], observation)  # leaves: (B=1, T=1, ...)
    lifecycle = model._ship_lifecycle(sequence, carry, jnp.zeros((1, 1), dtype=bool))
    np.testing.assert_array_equal(lifecycle.reset, [[[True, False, True]]])


def test_ship_spawn_uses_observed_parameters(model: LuxActorCritic, observation: LuxObservation[jax.Array]) -> None:
    sequence = jax.tree.map(lambda leaf: jnp.broadcast_to(leaf, (2, 2, *leaf.shape)), observation)
    # leaves: (B=2, T=2, ...); identical ships, different game parameters.
    sequence = sequence._replace(
        self=sequence.self._replace(
            units=sequence.self.units.replace(
                position=jnp.zeros((2, 2, 3, 2), jnp.int16),  # (B, T, U, 2)
                energy=jnp.full((2, 2, 3), 75, jnp.int16),  # (B, T, U)
            ),
        ),
        world=sequence.world._replace(
            steps=jnp.array([[106, 107], [106, 107]]),  # (B, T)
            match_steps=jnp.array([[6, 7], [6, 7]]),  # (B, T)
        ),
        init_unit_energy=jnp.array([[75, 75], [100, 100]]),  # (B, T)
        spawn_rate=jnp.array([[5, 5], [3, 3]]),  # (B, T)
    )
    carry = model.initial_carry(2)._replace(
        ship_alive=jnp.ones((2, 3), dtype=bool),  # (B, U)
        match_start=jnp.full(2, 100),  # (B,)
    )
    lifecycle = jax.jit(model._ship_lifecycle)(sequence, carry, jnp.zeros((2, 2), dtype=bool))
    assert lifecycle.reset.shape == lifecycle.alive.shape == (2, 2, 3)
    assert lifecycle.match_start.shape == (2, 2)
    np.testing.assert_array_equal(
        lifecycle.reset,
        [[[True, True, True], [False, False, True]], [[False, False, True], [False, False, True]]],
    )


def test_sap_offsets_can_prefer_nearby_targets_symmetrically(
    model: LuxActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    # Reuse parameters with a wider sap range; identical tile features isolate offset scoring.
    network = LuxModel(
        embedding_size=8,
        channels=4,
        action_space=HierarchicalMultiDiscreteSpace(nvec=(30,) * 3, group_sizes=(1, 4, 25)),
    )
    variables = jax.tree.map(lambda leaf: leaf, params["params"]["lux"])
    variables["sap_query"] = {
        "kernel": jnp.zeros((8, 4)),  # (F, C)
        "bias": jnp.array([1.0, 0, 0, 0]),  # (C,)
    }
    variables["sap_key"] = jax.tree.map(jnp.zeros_like, variables["sap_key"])
    variables["sap_offset_hidden"] = {
        "kernel": jnp.zeros((4, 4)).at[2, 0].set(1).at[3, 1].set(1),  # (4, C), absolute dx/dy
        "bias": jnp.zeros(4),  # (C,)
    }
    variables["sap_offset"] = {"kernel": jnp.zeros((4, 4)).at[:2, 0].set(-1)}  # (C, C)
    features = LuxFeatures(
        global_features=jnp.zeros((1, 8)),  # (B, F)
        ships=jnp.zeros((1, 3, 8)),  # (B, U, 2C)
        spatial=jnp.zeros((1, 12, 8, 4)),  # (B, H, W, C)
    )
    obs = jax.tree.map(lambda leaf: leaf[None], observation)  # leaves: (B, ...)
    masks = jnp.broadcast_to(obs.self.units_mask[..., None], (1, 3, 30)).at[..., 0].set(True)  # (B, U, A)
    policy, value = network.apply(
        {"params": variables}, jnp.zeros((1, 4)), jnp.zeros((1, 3, 4)), features, obs, masks, method="heads"
    )
    assert policy.logits.shape == (1, 3, 30) and value.shape == (1,)
    scores = policy.raw_logits[0, 0, 7:].reshape((5, 5))  # (dx, dy), center is offset (0, 0)
    assert float(scores[2, 2]) > float(scores[2, 3]) > float(scores[2, 4]) > float(scores[4, 4])
    np.testing.assert_allclose(scores, scores[::-1], atol=1e-6)
    np.testing.assert_allclose(scores, scores[:, ::-1], atol=1e-6)
    np.testing.assert_allclose(scores, scores.T, atol=1e-6)


def test_bfloat16_sap_preserves_close_target_scores(
    model: LuxActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    network = LuxModel(embedding_size=8, channels=4, action_space=model.action_space, dtype=jnp.bfloat16)
    variables = jax.tree.map(lambda leaf: leaf, params["params"]["lux"])
    variables["sap_query"] = {"kernel": jnp.zeros((8, 4)), "bias": jnp.ones(4)}  # (F, C); (C,)
    variables["sap_key"] = {"kernel": jnp.eye(4), "bias": jnp.zeros(4)}  # (C, C); (C,)
    variables["sap_offset"] = jax.tree.map(jnp.zeros_like, variables["sap_offset"])
    spatial = (
        jnp.zeros((1, 12, 8, 4), jnp.bfloat16)
        .at[..., 0]
        .set(1)
        .at[0, :, :, 1]
        .set(jnp.arange(8, dtype=jnp.bfloat16)[None, :] / 256)
    )  # (B, H, W, C)
    obs = jax.tree.map(lambda leaf: leaf[None], observation)  # leaves: (B, ...)
    scores = network.apply(
        {"params": variables}, jnp.zeros((1, 3, 8), jnp.bfloat16), spatial, obs, method="_sap_logits"
    )  # (B, U, K)
    assert scores.shape == (1, 3, 9) and scores.dtype == jnp.float32
    # Each component is representable in bf16; rounding the dot product would tie x=0 and x=1.
    expected = jnp.repeat(0.5 + jnp.arange(3, dtype=jnp.float32) / 512, 3)  # (K,)
    np.testing.assert_array_equal(scores[0, 0], expected)


def test_single_ship_small_map_reuses_parameters(
    model: LuxActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    small = observation._replace(
        self=observation.self._replace(
            units=jax.tree.map(lambda leaf: leaf[:1], observation.self.units),
            units_mask=observation.self.units_mask[:1],
        ),
        opponent=observation.opponent._replace(
            units=jax.tree.map(lambda leaf: leaf[:1], observation.opponent.units),
            units_mask=observation.opponent.units_mask[:1],
        ),
        world=observation.world._replace(
            sensor_mask=observation.world.sensor_mask[:8, :],  # (H=8, W=8)
            map_features=jax.tree.map(lambda leaf: leaf[:8, :], observation.world.map_features),  # (H, W)
        ),
    )
    small = jax.tree.map(lambda leaf: leaf[None, None], small)  # leaves: (B=1, T=1, ...)
    single = model.clone(action_space=HierarchicalMultiDiscreteSpace(nvec=(14,), group_sizes=(1, 4, 9)))
    output = jax.jit(single.apply)(params, small, single.initial_carry(1), jnp.ones((1, 1), bool), action_mask(small))
    assert output.policy.logits.shape == (1, 1, 1, 14)
    assert output.value.shape == (1, 1)


@pytest.mark.parametrize(
    "groups", [(1, 2, 9), (1, 4, 8), (1, 4, 4), (1, 4, 1)], ids=["move_count", "non_square", "even_side", "zero_range"]
)
def test_rejects_incompatible_command_groups(groups: tuple[int, ...], observation: LuxObservation[jax.Array]) -> None:
    model = LuxModel(
        embedding_size=8,
        channels=4,
        action_space=HierarchicalMultiDiscreteSpace(nvec=(sum(groups),) * 3, group_sizes=groups),
    )
    with pytest.raises(ValueError, match="stay/move/sap"):
        model.init(jax.random.key(0), observation, method="encode")


def test_sharded_policy_and_value_gradients(
    model: LuxActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    if jax.device_count() < 2:
        pytest.skip("Requires two devices; use JAX_NUM_CPU_DEVICES=2 on CPU")
    second = observation._replace(public_params=jnp.array([5, 50, 7, 7], jnp.int16))  # (4,)
    sequence = jax.tree.map(lambda a, b: jnp.stack((a, b))[:, None], observation, second)  # leaves: (B=2, T=1, ...)

    def objective(
        variables: dict[str, Any], obs: LuxObservation[jax.Array]
    ) -> tuple[jax.Array, jax.Array]:  # obs leaves: (B, T, ...) -> (); (B, T)
        output = model.apply(variables, obs, model.initial_carry(2), jnp.ones((2, 1), bool), action_mask(obs))
        actions = jnp.broadcast_to(jnp.array([5, 1, 0]), (2, 1, 3))  # (B, T, U), sap/move/stay
        loss = -output.policy.log_prob(actions).mean() + jnp.square(output.value).mean()  # ()
        return loss, output.value  # (); (B, T)

    differentiated = jax.value_and_grad(objective, has_aux=True)
    expected = jax.jit(differentiated)(params, sequence)
    mesh = Mesh(devices=jax.devices()[:2], axis_names=("batch",), axis_types=(AxisType.Auto,))
    with jax.set_mesh(mesh):
        actual = jax.jit(
            differentiated,
            in_shardings=(PartitionSpec(), PartitionSpec("batch")),
            out_shardings=((PartitionSpec(), PartitionSpec("batch")), PartitionSpec()),
        )(jax.device_put(params, PartitionSpec()), jax.device_put(sequence, PartitionSpec("batch")))
    for value, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(value, reference, atol=2e-5, rtol=2e-4)
    for name in (
        "sap_query",
        "sap_key",
        "sap_offset_hidden",
        "sap_offset",
        "move_output",
        "type_output",
        "value_output",
    ):
        magnitude = sum(float(jnp.abs(leaf).sum()) for leaf in jax.tree.leaves(actual[1]["params"]["lux"][name]))
        assert 0 < magnitude < 1e6
    for index in range(model.model.num_blocks):
        for name in ("ship_memory", "map_attention", "set_attention"):
            leaves = jax.tree.leaves(actual[1]["params"][f"block_{index}"][name])
            assert 0 < sum(float(jnp.abs(leaf).sum()) for leaf in leaves) < 1e6
    relic_gradient = actual[1]["params"]["lux"]["encoder"]["relic_projection"]
    assert 0 < sum(float(jnp.abs(leaf).sum()) for leaf in jax.tree.leaves(relic_gradient)) < 1e6
    assert actual[0][1].addressable_shards[0].data.shape == (1, 1)


def test_each_policy_row_uses_its_own_ship_features(
    model: LuxActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    network = LuxModel(embedding_size=8, channels=4, action_space=model.action_space)
    variables = {"params": params["params"]["lux"]}
    obs = jax.tree.map(lambda leaf: leaf[None], observation)  # leaves: (B=1, ...)
    features = network.apply(variables, obs, method="encode")
    changed = features._replace(ships=features.ships.at[:, 0].add(jnp.arange(8)))  # (B, U, D)

    def heads(encoded: LuxFeatures) -> tuple[jax.Array, jax.Array]:
        policy, value = network.apply(
            variables, jnp.zeros((1, 4)), jnp.zeros((1, 3, 4)), encoded, obs, action_mask(obs), method="heads"
        )
        return policy.raw_logits, value  # (B, U, L); (B,)

    expected_logits, expected_value = heads(features)  # (B, U, L); (B,)
    actual_logits, actual_value = heads(changed)  # (B, U, L); (B,)
    assert float(jnp.max(jnp.abs(actual_logits[:, 0] - expected_logits[:, 0]))) > 1e-5
    np.testing.assert_array_equal(actual_logits[:, 1:], expected_logits[:, 1:])
    np.testing.assert_array_equal(actual_value, expected_value)


def test_padding_capacity_does_not_change_ship_policies_or_team_memory(
    model: LuxActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    compact = observation._replace(
        self=observation.self._replace(
            units=jax.tree.map(lambda leaf: leaf[:2], observation.self.units),  # positions: (U=2, 2); energy: (U,)
            units_mask=observation.self.units_mask[:2],  # (U,)
        ),
        opponent=observation.opponent._replace(
            units=jax.tree.map(lambda leaf: leaf[:2], observation.opponent.units),  # positions: (V=2, 2); energy: (V,)
            units_mask=observation.opponent.units_mask[:2],  # (V,)
        ),
        world=observation.world._replace(
            relic_nodes=observation.world.relic_nodes[:1],  # (R=1, 2)
            relic_nodes_mask=observation.world.relic_nodes_mask[:1],  # (R,)
        ),
    )
    padded, compact = jax.tree.map(lambda leaf: leaf[None, None], (observation, compact))  # leaves: (B=1, T=1, ...)
    smaller = model.clone(action_space=HierarchicalMultiDiscreteSpace(nvec=(14,) * 2, group_sizes=(1, 4, 9)))
    expected = jax.jit(model.apply)(params, padded, model.initial_carry(1), jnp.ones((1, 1), bool), action_mask(padded))
    actual = jax.jit(smaller.apply)(
        params, compact, smaller.initial_carry(1), jnp.ones((1, 1), bool), action_mask(compact)
    )
    assert actual.policy.logits.shape == (1, 1, 2, 14)
    np.testing.assert_allclose(actual.policy.logits, expected.policy.logits[..., :2, :], atol=2e-6)
    expected_carry = expected.carry._replace(
        ships=jax.tree.map(lambda leaf: leaf[:, :2], expected.carry.ships),  # (B, U=2, Hs)
        ship_alive=expected.carry.ship_alive[:, :2],  # (B, U=2)
    )
    chex.assert_trees_all_close((actual.value, actual.carry), (expected.value, expected_carry), atol=2e-6)
    np.testing.assert_allclose(actual.policy.entropy(), expected.policy.entropy(), atol=2e-6)


def test_ship_lifetime_resets_and_inactive_state(
    model: LuxActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    # Rows: survivor, absent, negative energy, newborn, game reset, boundary,
    # skipped boundary, possible same-step respawn. Only ship 0's history differs.
    batch_size = 8
    sequence = jax.tree.map(lambda leaf: jnp.broadcast_to(leaf, (batch_size, 1, *leaf.shape)), observation)
    # sequence leaves: (B=8, T=1, ...)
    sequence = sequence._replace(
        self=sequence.self._replace(
            units_mask=sequence.self.units_mask.at[1, 0, 0].set(False),  # (B, T, U)
            units=sequence.self.units.replace(
                energy=sequence.self.units.energy.at[2, 0, 0].set(-1).at[7, 0, 0].set(100),  # (B, T, U)
                position=sequence.self.units.position.at[7, 0, 0].set(0),  # (B, T, U, 2)
            ),
        ),
        world=sequence.world._replace(
            match_steps=sequence.world.match_steps.at[5, 0].set(0).at[7, 0].set(46),  # (B, T)
            steps=sequence.world.steps.at[7, 0].set(147),  # (B, T), same match offset 101
        ),
    )
    memory = history(model, observation, batch_size)
    memory = memory._replace(
        ship_alive=memory.ship_alive.at[3, 0].set(False),  # (B, U)
        match_start=memory.match_start.at[6].set(100),  # (B,)
    )
    fresh = memory._replace(ships=jax.tree.map(lambda leaf: leaf.at[:, 0].set(0), memory.ships))  # (B, U, Hs)
    starts = jnp.zeros((batch_size, 1), bool).at[4, 0].set(True)  # (B, T)

    def apply(carry: LuxCarry) -> tuple[LuxCarry, LuxFeatures]:  # carry: (B, ...); features: (B, T, ...)
        result = model.apply(params, sequence, carry, starts, method="_encode")
        return result.carry, result.features

    evaluate = jax.jit(apply)
    actual, output = evaluate(memory)
    expected, fresh_output = evaluate(fresh)
    assert output.ships.shape == (8, 1, 3, 8)
    assert len(actual.ships) == model.model.num_blocks
    assert all(len(state) == model.model.ship_recurrent.num_layers for state in actual.ships)
    assert actual.ship_alive.dtype == jnp.bool_ and actual.match_start.dtype == jnp.int32
    # Reset happens before attention: discarded memory cannot affect any entity or summary.
    chex.assert_trees_all_close(
        jax.tree.map(lambda leaf: leaf[1:], output), jax.tree.map(lambda leaf: leaf[1:], fresh_output), atol=2e-6
    )
    assert float(jnp.max(jnp.abs(output.ships[0] - fresh_output.ships[0]))) > 1e-4
    for state, reference in zip(jax.tree.leaves(actual.ships), jax.tree.leaves(expected.ships), strict=True):
        assert state.shape == (8, 3, 4) and state.dtype == jnp.float32
        np.testing.assert_allclose(state[1:, 0], reference[1:, 0], atol=2e-6)
        np.testing.assert_array_equal(state[[1, 2, 5], 0], 0)
        np.testing.assert_array_equal(state[:, 2], 0)
    # The first stage has not mixed entities yet, so other ships' state is independent.
    for state, reference in zip(jax.tree.leaves(actual.ships[0]), jax.tree.leaves(expected.ships[0]), strict=True):
        np.testing.assert_allclose(state[:, 1], reference[:, 1], atol=2e-6)
    chex.assert_trees_all_equal(actual.team, memory.team)


def test_recurrent_attention_replay_is_causal_and_shares_ship_history(
    model: LuxActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    sequence = jax.tree.map(lambda leaf: jnp.broadcast_to(leaf, (1, 4, *leaf.shape)), observation)  # (B=1, T=4, ...)
    sequence = sequence._replace(
        self=sequence.self._replace(units_mask=sequence.self.units_mask.at[0, 1, 0].set(False)),  # (B, T, U)
        world=sequence.world._replace(
            steps=sequence.world.steps + jnp.arange(4),  # (B, T)
            match_steps=sequence.world.match_steps + jnp.arange(4),  # (B, T)
        ),
    )
    initial = history(model, observation, 1)
    changed = initial._replace(ships=jax.tree.map(lambda leaf: leaf.at[:, 0].add(2), initial.ships))  # (B, U, Hs)
    starts = jnp.zeros((1, 4), bool)  # (B, T)
    masks = action_mask(sequence)  # (B, T, U, A)
    evaluate = jax.jit(model.apply)
    output = evaluate(params, sequence, initial, starts, masks)
    alternative = evaluate(params, sequence, changed, starts, masks)
    assert float(jnp.max(jnp.abs(output.policy.raw_logits[:, 0, 0] - alternative.policy.raw_logits[:, 0, 0]))) > 1e-5
    # Attention now communicates ship 0's history to other ships and the team summary.
    assert float(jnp.max(jnp.abs(output.policy.raw_logits[:, 0, 1] - alternative.policy.raw_logits[:, 0, 1]))) > 1e-5
    future = sequence._replace(
        self=sequence.self._replace(
            units=sequence.self.units.replace(energy=sequence.self.units.energy.at[:, -1, 0].set(300))  # (B, T, U)
        )
    )
    future_output = evaluate(params, future, initial, starts, masks)
    np.testing.assert_allclose(output.policy.raw_logits[:, :-1], future_output.policy.raw_logits[:, :-1], atol=2e-6)
    np.testing.assert_allclose(output.value[:, :-1], future_output.value[:, :-1], atol=2e-6)

    @jax.jit
    def replay(carry: LuxCarry) -> tuple[LuxCarry, jax.Array]:  # carry: (B, ...); logits: (T, B, U, L)
        def step(state: LuxCarry, obs: LuxObservation[jax.Array]) -> tuple[LuxCarry, jax.Array]:
            # obs: (B, ...); state: (B, ...); logits: (B, U, L)
            result = model.apply(params, obs, state, jnp.zeros(1, bool), action_mask(obs), method="step")
            return result.carry, result.policy.raw_logits  # carry: (B, ...); logits: (B, U, L)

        return jax.lax.scan(step, carry, jax.tree.map(lambda leaf: jnp.swapaxes(leaf, 0, 1), sequence))

    carry, logits = replay(initial)  # logits: (T, B, U, L)
    chex.assert_trees_all_close(carry, output.carry, atol=2e-6)
    np.testing.assert_allclose(jnp.swapaxes(logits, 0, 1), output.policy.raw_logits, atol=2e-6)


def test_native_collision_and_same_step_respawn_clear_ship_memory(
    model: LuxActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    env = LuxEnv(LuxConfig(map_size=8, max_units=3, randomize_params=False))
    _, state = jax.jit(env.reset)(jax.random.key(19))
    # Equal-energy opposing ships collide at the home tile on a spawning step.
    # Both slots remain present in consecutive observations despite being replaced.
    game = state.game.replace(
        units=state.game.units.replace(
            position=jnp.full((2, 3, 2), -1, jnp.int16).at[:, 0].set(0),  # (N, U, 2)
            energy=jnp.zeros((2, 3, 1), jnp.int16).at[:, 0].set(100),  # (N, U, 1)
        ),
        units_mask=jnp.zeros((2, 3), bool).at[:, 0].set(True),  # (N, U)
        match_steps=jnp.array(3, jnp.int32),  # ()
        steps=jnp.array(3, jnp.int32),  # ()
    )
    result = jax.jit(env.step)(jax.random.key(20), state._replace(game=game), jnp.zeros((2, 3), jnp.int32))
    obs = result.observation.observation  # leaves: (B=N=2, ...)
    np.testing.assert_array_equal(obs.self.units_mask[:, 0], True)
    np.testing.assert_array_equal(obs.self.units.energy[:, 0], 100)
    np.testing.assert_array_equal(obs.self.units.position[:, 0], [[0, 0], [7, 7]])
    assert int(result.state.game.match_steps) == 4
    memory = history(model, observation, 2)._replace(match_start=jnp.zeros(2, jnp.int32))
    fresh = memory._replace(ships=jax.tree.map(jnp.zeros_like, memory.ships))  # (B, U, Hs)

    def apply(carry: LuxCarry) -> tuple[LuxCarry, jax.Array]:  # carry: (B, ...); logits: (B, U, L)
        output = model.apply(params, obs, carry, jnp.zeros(2, bool), action_mask(obs), method="step")
        return output.carry, output.policy.raw_logits  # carry: (B, ...); logits: (B, U, L)

    evaluate = jax.jit(apply)
    chex.assert_trees_all_close(evaluate(memory), evaluate(fresh), atol=2e-6)


class AttentionOutputs(NamedTuple):
    first_map: jax.Array  # (B*T, E, D)
    first_set: jax.Array  # (B*T, E, D)
    second_map: jax.Array  # (B*T, E, D)
    second_set: jax.Array  # (B*T, E, D)


@pytest.mark.parametrize("stage", [0, 1], ids=["first_block", "second_block"])
def test_ship_history_conditions_the_following_attention(
    model: LuxActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array], stage: int
) -> None:
    sequence = jax.tree.map(lambda leaf: leaf[None, None], observation)  # leaves: (B=1, T=1, ...)
    memory = history(model, observation, 1)
    changed = memory._replace(
        ships=tuple(
            jax.tree.map(lambda leaf: leaf.at[:, 0].add(2), state) if index == stage else state
            for index, state in enumerate(memory.ships)
        )  # leaves: (B, U, Hs)
    )

    def capture(module: nn.Module, method: str) -> bool:
        return method == "__call__" and module.name in {"map_attention", "set_attention"}

    @jax.jit
    def attend(carry: LuxCarry) -> AttentionOutputs:  # carry leaves: (B, ...)
        _, state = model.apply(
            params,
            sequence,
            carry,
            jnp.zeros((1, 1), bool),
            method="_encode",
            capture_intermediates=capture,
            mutable=["intermediates"],
        )
        layers = state["intermediates"]
        return AttentionOutputs(
            first_map=layers["block_0"]["map_attention"]["__call__"][0],
            first_set=layers["block_0"]["set_attention"]["__call__"][0],
            second_map=layers["block_1"]["map_attention"]["__call__"][0],
            second_set=layers["block_1"]["set_attention"]["__call__"][0],
        )

    original, updated = attend(memory), attend(changed)
    assert all(leaf.shape == (1, 9, 8) for leaf in original)
    if stage == 0:
        assert float(jnp.max(jnp.abs(original.first_map[:, 0] - updated.first_map[:, 0]))) > 1e-4
        assert float(jnp.max(jnp.abs(original.first_set[:, 1] - updated.first_set[:, 1]))) > 1e-4
    else:
        np.testing.assert_array_equal(original.first_map, updated.first_map)
        np.testing.assert_array_equal(original.first_set, updated.first_set)
    assert float(jnp.max(jnp.abs(original.second_map[:, 0] - updated.second_map[:, 0]))) > 1e-4
    assert float(jnp.max(jnp.abs(original.second_set[:, 1] - updated.second_set[:, 1]))) > 1e-4


@pytest.mark.parametrize("num_blocks", [1, 3], ids=["minimum_depth", "additional_block"])
def test_configurable_block_depth(
    model: LuxActorCritic, observation: LuxObservation[jax.Array], num_blocks: int
) -> None:
    model = model.clone(model=replace(model.model, num_blocks=num_blocks))
    sequence = jax.tree.map(lambda leaf: leaf[None, None], observation)  # leaves: (B=1, T=1, ...)
    output, variables = jax.eval_shape(
        model.init_with_output,
        jax.random.key(4),
        sequence,
        model.initial_carry(1),
        jnp.ones((1, 1), bool),
        action_mask(sequence),
    )
    assert output.policy.raw_logits.shape == (1, 1, 3, 16)
    assert output.value.shape == (1, 1)
    assert len(output.carry.ships) == num_blocks
    for index, state in enumerate(output.carry.ships):
        assert all(leaf.shape == (1, 3, 4) for leaf in jax.tree.leaves(state))
        assert {"ship_memory", "input_norm", "map_attention", "set_attention"} == set(
            variables["params"][f"block_{index}"]
        )
