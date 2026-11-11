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
from rl2.envs.lux_env import LuxObservation
from rl2.model.config import LuxModelConfig
from rl2.model.lux import (
    LuxActorCritic,
    LuxAttentionBlock,
    LuxCarry,
    LuxEncoder,
    LuxFeatures,
    LuxHeads,
    _GlobalMemory,
)


@pytest.fixture(scope="module")
def model() -> LuxActorCritic:
    return LuxActorCritic(
        action_space=HierarchicalMultiDiscreteSpace(nvec=(14,) * 3, group_sizes=(1, 4, 9)),
        model=LuxModelConfig(
            conv_channels=4,
            heads=4,
            head_dim=2,
            attention_implementation="xla",
        ),
    )


def history(model: LuxActorCritic, batch_size: int) -> LuxCarry:
    """Return nonzero per-block (B, Hg) cell/hidden states."""
    return jax.tree.map(lambda leaf: leaf + 0.5, model.initial_carry(batch_size))


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
    block = LuxAttentionBlock(
        attention_implementation="xla",
        heads=model.model.heads,
        head_dim=model.model.head_dim,
        dtype=jnp.bfloat16,
    )
    memory = _GlobalMemory(features=model.model.heads * model.model.head_dim, dtype=jnp.bfloat16)
    carry = block.initial_carry(actor_count=2)  # leaves: (B, Hg)
    for module in (memory, block):
        chex.assert_trees_all_equal(module.initial_carry(2), carry)
        chex.assert_trees_all_equal(module.apply({}, 2, method="initial_carry"), carry)
    assert len(carry) == 2
    for leaf in carry:
        assert leaf.shape == (2, model.model.heads * model.model.head_dim) and leaf.dtype == jnp.float32
        np.testing.assert_array_equal(leaf, 0)
    combined = model.initial_carry(2)
    assert combined._fields == ("global_",)
    assert len(combined.global_) == model.model.num_blocks
    for state in combined.global_:
        chex.assert_trees_all_equal(state, carry)
    chex.assert_trees_all_equal(model.apply({}, 2, method="initial_carry"), combined)


def test_recurrent_modules_return_output_and_carry(model: LuxActorCritic, params: dict[str, Any]) -> None:
    width = model.model.heads * model.model.head_dim
    tokens = jax.random.normal(jax.random.key(5), (1, 2, 4, width))  # (B, T, E, D), one of each token type
    token_types = jnp.arange(4, dtype=jnp.int32)  # (E,)
    mask = jnp.array([[[True, True, True, False]] * 2])  # (B, T, E)
    block = LuxAttentionBlock(
        attention_implementation="xla",
        heads=model.model.heads,
        head_dim=model.model.head_dim,
    )
    carry = block.initial_carry(actor_count=1)  # leaves: (B, Hg)
    variables = params["params"]["encoder"]["block_0"]
    global_result = jax.jit(_GlobalMemory(features=model.model.heads * model.model.head_dim).apply)(
        {"params": variables["global_memory"]}, tokens[..., 0, :], carry, jnp.ones((1, 2), bool)
    )  # token: (B, T, D); carry: (B, Hg)

    def capture(module: nn.Module, method: str) -> bool:
        return method == "__call__" and module.name == "input_norm"

    block_result, state = jax.jit(block.apply, static_argnames=("capture_intermediates", "mutable"))(
        {"params": variables},
        tokens,
        mask,
        carry,
        jnp.ones((1, 2), bool),
        jnp.ones((1, 2, 2, width)),
        token_types,
        capture_intermediates=capture,
        mutable=("intermediates",),
    )  # tokens: (B, T, E, D); carry: (B, Hg)
    for result, shape in ((global_result, (1, 2, width)), (block_result, (1, 2, 4, width))):
        assert type(result) is tuple and len(result) == 2
        output, updated_carry = result  # (B, T, [E,] D); carry leaves: (B, Hg)
        assert output.shape == shape
        assert len(updated_carry) == 2
        for leaf in updated_carry:
            assert leaf.shape == (1, model.model.heads * model.model.head_dim) and leaf.dtype == jnp.float32
    chex.assert_trees_all_equal(global_result[1], block_result[1])
    np.testing.assert_array_equal(block_result[0][..., 3, :], 0)  # Masked relic remains inactive.
    embeddings = variables["token_type_embedding"]["embedding"]  # (4, D)
    assert embeddings.shape == (4, width)
    typed = tokens.at[..., 0, :].set(global_result[0]) + embeddings[token_types]  # (B, T, E, D)
    expected = nn.LayerNorm().apply({"params": variables["input_norm"]}, typed.reshape((2, 4, width)))  # (B*T, E, D)
    np.testing.assert_allclose(state["intermediates"]["input_norm"]["__call__"][0], expected, atol=2e-6)


def test_sequence_step_replay_reset_and_bfloat16(
    model: LuxActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    sequence = jax.tree.map(lambda leaf: jnp.broadcast_to(leaf, (2, 2, *leaf.shape)), observation)  # (B, T, ...)
    starts = jnp.array([[False, False], [False, True]])  # (B, T)
    memory = history(model, 2)  # leaves: (B, ...)
    masks = action_mask(sequence).at[0, :, 0, 5:].set(False)  # (B, T, U, A), no sap for this ship
    output, output_carry = jax.jit(model.apply)(params, sequence, memory, starts, masks)
    assert output.policy.raw_logits.shape == (2, 2, 3, 16)
    assert output.policy.logits.shape == (2, 2, 3, 14)
    assert output.value.shape == (2, 2) and output.value.dtype == jnp.float32
    first, second = jax.tree.map(lambda leaf: leaf[:, 0], sequence), jax.tree.map(lambda leaf: leaf[:, 1], sequence)
    # first/second leaves: (B, ...)
    step = jax.jit(lambda obs, carry, reset, mask: model.apply(params, obs, carry, reset, mask, method="step"))
    first_output, first_output_carry = step(first, memory, starts[:, 0], masks[:, 0])
    second_output, second_output_carry = step(second, first_output_carry, starts[:, 1], masks[:, 1])
    for actual, expected in zip(
        (output.policy.logits, output.value),
        (
            jnp.stack((first_output.policy.logits, second_output.policy.logits), 1),
            jnp.stack((first_output.value, second_output.value), 1),
        ),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, atol=2e-6)
    chex.assert_trees_all_close(output_carry, second_output_carry, atol=2e-6)
    fresh, _ = step(second, model.initial_carry(2), jnp.ones(2, dtype=bool), masks[:, 1])
    np.testing.assert_allclose(second_output.value[1], fresh.value[1], atol=2e-6)
    actions, log_prob = output.policy.sample_and_log_prob(key=jax.random.key(5))  # (B, T, U); (B, T)
    assert actions.shape == (2, 2, 3) and log_prob.shape == (2, 2)
    assert bool(jnp.take_along_axis(masks, actions[..., None], axis=-1).all())
    np.testing.assert_array_equal(actions[..., 2], 0)  # Absent ship always stays.
    np.testing.assert_allclose(output.policy.log_prob(actions), log_prob, atol=1e-6)
    reduced, reduced_carry = jax.jit(model.clone(dtype=jnp.bfloat16).apply)(params, sequence, memory, starts, masks)
    assert reduced.policy.raw_logits.dtype == reduced.value.dtype == jnp.float32
    reduced_model = model.clone(dtype=jnp.bfloat16)
    reduced_step = jax.jit(
        lambda variables, obs, carry, reset, mask: reduced_model.apply(
            variables, obs, carry, reset, mask, method="step"
        )
    )
    reduced_first, reduced_first_carry = reduced_step(params, first, memory, starts[:, 0], masks[:, 0])
    reduced_second, reduced_second_carry = reduced_step(params, second, reduced_first_carry, starts[:, 1], masks[:, 1])
    # Keep parameters dynamic in both paths, as in training, including bf16 casts.
    np.testing.assert_allclose(reduced.value, jnp.stack((reduced_first.value, reduced_second.value), 1), atol=2e-3)
    np.testing.assert_allclose(
        reduced.policy.raw_logits,
        jnp.stack((reduced_first.policy.raw_logits, reduced_second.policy.raw_logits), 1),
        atol=2e-3,
    )
    chex.assert_trees_all_close(reduced_carry, reduced_second_carry, atol=2e-3)
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(reduced_carry))
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
    memory = history(model, 3)  # leaves: (B, ...)
    output, output_carry = jax.jit(
        lambda obs: model.apply(params, obs, memory, jnp.zeros(3, bool), action_mask(obs), method="step")
    )(batch)
    np.testing.assert_allclose(output.policy.logits[1], output.policy.logits[0, permutation], atol=2e-6)
    np.testing.assert_allclose(output.policy.logits[2], output.policy.logits[0], atol=2e-6)
    np.testing.assert_allclose(output.value[1:], jnp.broadcast_to(output.value[0], (2,)), atol=2e-6)
    for leaf in jax.tree.leaves(output_carry):
        np.testing.assert_allclose(leaf[1:], jnp.broadcast_to(leaf[0], leaf[1:].shape), atol=2e-6)


def test_empty_team_has_no_policy_gradient(
    model: LuxActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    empty = observation._replace(self=observation.self._replace(units_mask=jnp.zeros(3, dtype=bool)))
    empty = jax.tree.map(lambda leaf: leaf[None, None], empty)  # leaves: (B=1, T=1, ...)

    def loss(variables: dict[str, Any]) -> tuple[jax.Array, jax.Array]:  # parameter leaves: (...) -> (); (B, T)
        output, _ = model.apply(variables, empty, model.initial_carry(1), jnp.ones((1, 1), bool), action_mask(empty))
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
    network = LuxHeads(features=8, conv_channels=4, action_space=model.action_space)
    variables = jax.tree.map(lambda leaf: leaf, params["params"]["heads"])
    variables["sap_query"] = {"kernel": jnp.zeros((8, 4)), "bias": jnp.ones(4)}
    variables["sap_key"] = {"kernel": jnp.zeros((4, 4)).at[0].set(1), "bias": jnp.zeros(4)}
    variables["sap_offset"] = jax.tree.map(jnp.zeros_like, variables["sap_offset"])
    x, y = jnp.meshgrid(jnp.arange(8), jnp.arange(12), indexing="xy")  # each (H, W)
    spatial = jnp.zeros((1, 12, 8, 4)).at[0, :, :, 0].set(10 * x + y)  # (B, H, W, C)
    features = LuxFeatures(global_=jnp.zeros((1, 8)), ships=jnp.zeros((1, 3, 8)), spatial=spatial)
    obs = jax.tree.map(lambda leaf: leaf[None], observation)  # leaves: (B, ...)
    policy, value = network.apply({"params": variables}, features, obs, action_mask(obs))
    # First ship is at (1, 2); offsets enumerate x first, then y, matching Lux's command IDs.
    np.testing.assert_allclose(policy.raw_logits[0, 0, 7:], 2 * jnp.array([1, 2, 3, 11, 12, 13, 21, 22, 23]))
    assert value.shape == (1,)


def test_sap_offsets_can_prefer_nearby_targets_symmetrically(
    model: LuxActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    # Reuse parameters with a wider sap range; identical tile features isolate offset scoring.
    network = LuxHeads(
        features=8,
        conv_channels=4,
        action_space=HierarchicalMultiDiscreteSpace(nvec=(30,) * 3, group_sizes=(1, 4, 25)),
    )
    variables = jax.tree.map(lambda leaf: leaf, params["params"]["heads"])
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
        global_=jnp.zeros((1, 8)),  # (B, F)
        ships=jnp.zeros((1, 3, 8)),  # (B, U, 2C)
        spatial=jnp.zeros((1, 12, 8, 4)),  # (B, H, W, C)
    )
    obs = jax.tree.map(lambda leaf: leaf[None], observation)  # leaves: (B, ...)
    masks = jnp.broadcast_to(obs.self.units_mask[..., None], (1, 3, 30)).at[..., 0].set(True)  # (B, U, A)
    policy, value = network.apply({"params": variables}, features, obs, masks)
    assert policy.logits.shape == (1, 3, 30) and value.shape == (1,)
    scores = policy.raw_logits[0, 0, 7:].reshape((5, 5))  # (dx, dy), center is offset (0, 0)
    assert float(scores[2, 2]) > float(scores[2, 3]) > float(scores[2, 4]) > float(scores[4, 4])
    np.testing.assert_allclose(scores, scores[::-1], atol=1e-6)
    np.testing.assert_allclose(scores, scores[:, ::-1], atol=1e-6)
    np.testing.assert_allclose(scores, scores.T, atol=1e-6)


def test_bfloat16_sap_preserves_close_target_scores(
    model: LuxActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    network = LuxHeads(features=8, conv_channels=4, action_space=model.action_space, dtype=jnp.bfloat16)
    variables = jax.tree.map(lambda leaf: leaf, params["params"]["heads"])
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
    output, _ = jax.jit(single.apply)(
        params, small, single.initial_carry(1), jnp.ones((1, 1), bool), action_mask(small)
    )
    assert output.policy.logits.shape == (1, 1, 1, 14)
    assert output.value.shape == (1, 1)


@pytest.mark.parametrize(
    "groups", [(1, 2, 9), (1, 4, 8), (1, 4, 4), (1, 4, 1)], ids=["move_count", "non_square", "even_side", "zero_range"]
)
def test_rejects_incompatible_command_groups(groups: tuple[int, ...], observation: LuxObservation[jax.Array]) -> None:
    model = LuxHeads(
        features=8,
        conv_channels=4,
        action_space=HierarchicalMultiDiscreteSpace(nvec=(sum(groups),) * 3, group_sizes=groups),
    )
    features = LuxFeatures(
        global_=jnp.zeros(8),  # (F,)
        ships=jnp.zeros((3, 8)),  # (U, D)
        spatial=jnp.zeros((12, 8, 4)),  # (H, W, C)
    )
    with pytest.raises(ValueError, match="stay/move/sap"):
        model.init(jax.random.key(0), features, observation, action_mask(observation))


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
        output, _ = model.apply(variables, obs, model.initial_carry(2), jnp.ones((2, 1), bool), action_mask(obs))
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
        magnitude = sum(float(jnp.abs(leaf).sum()) for leaf in jax.tree.leaves(actual[1]["params"]["heads"][name]))
        assert 0 < magnitude < 1e6
    for index in range(model.model.num_blocks):
        for name in ("global_memory", "map_attention", "set_attention"):
            leaves = jax.tree.leaves(actual[1]["params"]["encoder"][f"block_{index}"][name])
            assert 0 < sum(float(jnp.abs(leaf).sum()) for leaf in leaves) < 1e6
    relic_gradient = actual[1]["params"]["encoder"]["relic_projection"]
    assert 0 < sum(float(jnp.abs(leaf).sum()) for leaf in jax.tree.leaves(relic_gradient)) < 1e6
    assert actual[0][1].addressable_shards[0].data.shape == (1, 1)


def test_each_policy_row_uses_its_own_ship_features(
    model: LuxActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    network = LuxHeads(features=8, conv_channels=4, action_space=model.action_space)
    variables = {"params": params["params"]["heads"]}
    obs = jax.tree.map(lambda leaf: leaf[None], observation)  # leaves: (B=1, ...)
    encoder = LuxEncoder(model=model.model)
    sequence = jax.tree.map(lambda leaf: leaf[:, None], obs)  # leaves: (B=1, T=1, ...)
    features, _ = encoder.apply(
        {"params": params["params"]["encoder"]}, sequence, encoder.initial_carry(1), jnp.ones((1, 1), bool)
    )  # features: (B, T, ...)
    features = jax.tree.map(lambda leaf: leaf[:, 0], features)  # leaves: (B, ...)
    changed = features._replace(ships=features.ships.at[:, 0].add(jnp.arange(8)))  # (B, U, D)

    def heads(encoded: LuxFeatures) -> tuple[jax.Array, jax.Array]:
        policy, value = network.apply(variables, encoded, obs, action_mask(obs))
        return policy.raw_logits, value  # (B, U, L); (B,)

    expected_logits, expected_value = heads(features)  # (B, U, L); (B,)
    actual_logits, actual_value = heads(changed)  # (B, U, L); (B,)
    assert float(jnp.max(jnp.abs(actual_logits[:, 0] - expected_logits[:, 0]))) > 1e-5
    np.testing.assert_array_equal(actual_logits[:, 1:], expected_logits[:, 1:])
    np.testing.assert_array_equal(actual_value, expected_value)


def test_global_history_reaches_heads_only_through_tokens(
    model: LuxActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    variables = jax.tree.map(lambda leaf: leaf, params)
    for index in range(model.model.num_blocks):
        projection = variables["params"]["encoder"][f"block_{index}"]["global_memory"]["output"]
        projection["kernel"] = jnp.zeros_like(projection["kernel"])  # (Hg, D)
    sequence = jax.tree.map(lambda leaf: leaf[None, None], observation)  # leaves: (B=1, T=1, ...)
    carry = history(model, 1)
    changed = carry._replace(global_=jax.tree.map(lambda leaf: leaf + 2, carry.global_))  # leaves: (B, Hg)
    evaluate = jax.jit(model.apply)
    original, original_carry = evaluate(variables, sequence, carry, jnp.zeros((1, 1), bool), action_mask(sequence))
    updated, updated_carry = evaluate(variables, sequence, changed, jnp.zeros((1, 1), bool), action_mask(sequence))
    assert original.policy.raw_logits.shape == updated.policy.raw_logits.shape == (1, 1, 3, 16)
    assert original.value.shape == updated.value.shape == (1, 1)
    np.testing.assert_array_equal(original.policy.raw_logits, updated.policy.raw_logits)
    np.testing.assert_array_equal(original.value, updated.value)
    # Recurrence still retains history even when its token projection is disabled.
    for original_state, updated_state in zip(original_carry.global_, updated_carry.global_, strict=True):
        assert float(jnp.max(jnp.abs(original_state[1] - updated_state[1]))) > 1e-4


def test_padding_capacity_does_not_change_ship_policies_or_global_memory(
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
    expected, expected_carry = jax.jit(model.apply)(
        params, padded, model.initial_carry(1), jnp.ones((1, 1), bool), action_mask(padded)
    )
    actual, actual_carry = jax.jit(smaller.apply)(
        params, compact, smaller.initial_carry(1), jnp.ones((1, 1), bool), action_mask(compact)
    )
    assert actual.policy.logits.shape == (1, 1, 2, 14)
    np.testing.assert_allclose(actual.policy.logits, expected.policy.logits[..., :2, :], atol=2e-6)
    chex.assert_trees_all_close((actual.value, actual_carry), (expected.value, expected_carry), atol=2e-6)
    np.testing.assert_allclose(actual.policy.entropy(), expected.policy.entropy(), atol=2e-6)


def test_global_history_survives_match_changes_and_absent_ships_but_not_game_resets(
    model: LuxActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    sequence = jax.tree.map(lambda leaf: jnp.broadcast_to(leaf, (3, 3, *leaf.shape)), observation)  # (B, T, ...)
    sequence = sequence._replace(
        self=sequence.self._replace(units_mask=sequence.self.units_mask.at[2].set(False)),  # (B, T, U)
        world=sequence.world._replace(
            steps=sequence.world.steps + jnp.arange(3),  # (B, T)
            match_steps=(sequence.world.match_steps + jnp.arange(3)).at[1, 1:].set(jnp.array([0, 1])),  # (B, T)
        ),
    )
    memory = history(model, 3)
    changed = memory._replace(global_=jax.tree.map(lambda leaf: leaf + 2, memory.global_))  # (B, Hg)
    starts = jnp.zeros((3, 3), bool).at[:, -1].set(True)  # (B, T)
    encode = jax.jit(LuxEncoder(model=model.model).apply)
    variables = {"params": params["params"]["encoder"]}
    expected, expected_carry = encode(variables, sequence, memory, starts)
    actual, actual_carry = encode(variables, sequence, changed, starts)
    # Rows cover ordinary history, a match boundary, and an entirely absent team.
    difference = jnp.max(jnp.abs(actual.global_[:, :2] - expected.global_[:, :2]), axis=-1)  # (B, T=2)
    assert bool((difference > 1e-5).all())
    chex.assert_trees_all_close(
        jax.tree.map(lambda leaf: leaf[:, -1], actual),
        jax.tree.map(lambda leaf: leaf[:, -1], expected),
        atol=2e-6,
    )
    chex.assert_trees_all_close(actual_carry, expected_carry, atol=2e-6)


def test_recurrent_attention_replay_is_causal_and_shares_global_history(
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
    initial = history(model, 1)
    changed = initial._replace(global_=jax.tree.map(lambda leaf: leaf + 2, initial.global_))  # (B, Hg)
    starts = jnp.zeros((1, 4), bool)  # (B, T)
    masks = action_mask(sequence)  # (B, T, U, A)
    evaluate = jax.jit(model.apply)
    output, output_carry = evaluate(params, sequence, initial, starts, masks)
    alternative, _ = evaluate(params, sequence, changed, starts, masks)
    assert float(jnp.max(jnp.abs(output.policy.raw_logits[:, 0, 0] - alternative.policy.raw_logits[:, 0, 0]))) > 1e-5
    # Attention communicates global history to ship tokens.
    assert float(jnp.max(jnp.abs(output.policy.raw_logits[:, 0, 1] - alternative.policy.raw_logits[:, 0, 1]))) > 1e-5
    future = sequence._replace(
        self=sequence.self._replace(
            units=sequence.self.units.replace(energy=sequence.self.units.energy.at[:, -1, 0].set(300))  # (B, T, U)
        )
    )
    future_output, _ = evaluate(params, future, initial, starts, masks)
    np.testing.assert_allclose(output.policy.raw_logits[:, :-1], future_output.policy.raw_logits[:, :-1], atol=2e-6)
    np.testing.assert_allclose(output.value[:, :-1], future_output.value[:, :-1], atol=2e-6)

    @jax.jit
    def replay(carry: LuxCarry) -> tuple[LuxCarry, jax.Array]:  # carry: (B, ...); logits: (T, B, U, L)
        def step(state: LuxCarry, obs: LuxObservation[jax.Array]) -> tuple[LuxCarry, jax.Array]:
            # obs: (B, ...); state: (B, ...); logits: (B, U, L)
            result, result_carry = model.apply(params, obs, state, jnp.zeros(1, bool), action_mask(obs), method="step")
            return result_carry, result.policy.raw_logits  # carry: (B, ...); logits: (B, U, L)

        return jax.lax.scan(step, carry, jax.tree.map(lambda leaf: jnp.swapaxes(leaf, 0, 1), sequence))

    carry, logits = replay(initial)  # logits: (T, B, U, L)
    chex.assert_trees_all_close(carry, output_carry, atol=2e-6)
    np.testing.assert_allclose(jnp.swapaxes(logits, 0, 1), output.policy.raw_logits, atol=2e-6)


class AttentionOutputs(NamedTuple):
    first_map: jax.Array  # (B*T, E, D)
    first_set: jax.Array  # (B*T, E, D)
    second_map: jax.Array  # (B*T, E, D)
    second_set: jax.Array  # (B*T, E, D)


@pytest.mark.parametrize("stage", [0, 1], ids=["first_block", "second_block"])
def test_history_conditions_the_following_attention(
    model: LuxActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array], stage: int
) -> None:
    sequence = jax.tree.map(lambda leaf: leaf[None, None], observation)  # leaves: (B=1, T=1, ...)
    memory = history(model, 1)
    changed = memory._replace(
        global_=tuple(
            jax.tree.map(lambda leaf: leaf + 2, state) if index == stage else state
            for index, state in enumerate(memory.global_)
        )  # per-block leaves: (B, Hg)
    )

    def capture(module: nn.Module, method: str) -> bool:
        return method == "__call__" and module.name in {"map_attention", "set_attention"}

    encoder = LuxEncoder(model=model.model)

    @jax.jit
    def attend(carry: LuxCarry) -> AttentionOutputs:  # carry leaves: (B, ...)
        _, state = encoder.apply(
            {"params": params["params"]["encoder"]},
            sequence,
            carry,
            jnp.zeros((1, 1), bool),
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
        assert float(jnp.max(jnp.abs(original.first_set[:, 2] - updated.first_set[:, 2]))) > 1e-4
    else:
        np.testing.assert_array_equal(original.first_map, updated.first_map)
        np.testing.assert_array_equal(original.first_set, updated.first_set)
    assert float(jnp.max(jnp.abs(original.second_map[:, 0] - updated.second_map[:, 0]))) > 1e-4
    assert float(jnp.max(jnp.abs(original.second_set[:, 2] - updated.second_set[:, 2]))) > 1e-4


@pytest.mark.parametrize("num_blocks", [1, 3], ids=["minimum_depth", "additional_block"])
def test_configurable_block_depth(
    model: LuxActorCritic, observation: LuxObservation[jax.Array], num_blocks: int
) -> None:
    model = model.clone(model=replace(model.model, num_blocks=num_blocks))
    sequence = jax.tree.map(lambda leaf: leaf[None, None], observation)  # leaves: (B=1, T=1, ...)
    (output, output_carry), variables = jax.eval_shape(
        model.init_with_output,
        jax.random.key(4),
        sequence,
        model.initial_carry(1),
        jnp.ones((1, 1), bool),
        action_mask(sequence),
    )
    assert output.policy.raw_logits.shape == (1, 1, 3, 16)
    assert output.value.shape == (1, 1)
    assert len(output_carry.global_) == num_blocks
    assert all(
        leaf.shape == (1, model.model.heads * model.model.head_dim) for leaf in jax.tree.leaves(output_carry.global_)
    )
    for index in range(num_blocks):
        assert {"global_memory", "token_type_embedding", "input_norm", "map_attention", "set_attention"} == set(
            variables["params"]["encoder"][f"block_{index}"]
        )


def test_attention_width_is_independent_of_cnn_channels(
    model: LuxActorCritic, observation: LuxObservation[jax.Array]
) -> None:
    model = model.clone(model=replace(model.model, heads=2, head_dim=3))
    sequence = jax.tree.map(lambda leaf: leaf[None, None], observation)  # leaves: (B=1, T=1, ...)
    (output, output_carry), variables = jax.eval_shape(
        model.init_with_output,
        jax.random.key(4),
        sequence,
        model.initial_carry(1),
        jnp.ones((1, 1), bool),
        action_mask(sequence),
    )
    assert output.policy.raw_logits.shape == (1, 1, 3, 16)
    assert output.value.shape == (1, 1)
    encoder = variables["params"]["encoder"]
    assert all(leaf.shape == (1, model.model.heads * model.model.head_dim) for leaf in jax.tree.leaves(output_carry))
    assert encoder["ship_projection"]["kernel"].shape == (7, 6)
    assert encoder["map_encoder"]["map_downsample_1"]["kernel"].shape == (3, 3, 6, 6)
    assert encoder["map_projection"]["kernel"].shape == (6, 6)
    assert encoder["output_projection"]["kernel"].shape == (24, 6)
    heads = variables["params"]["heads"]
    assert heads["ship_hidden"]["kernel"].shape == (12, 6)
    assert heads["value_hidden"]["kernel"].shape == (6, 6)
    for index in range(model.model.num_blocks):
        attention = variables["params"]["encoder"][f"block_{index}"]["set_attention"]
        assert attention["ffn_in"]["kernel"].shape == (6, 24)
        assert attention["ffn_out"]["kernel"].shape == (24, 6)
