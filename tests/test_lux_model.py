"""Lux model contracts: ship symmetry, spatial command layout, and recurrent replay."""

from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, PartitionSpec
from test_lux_encoder import observation as observation  # noqa: PLC0414

from rl2.actions import HierarchicalMultiDiscreteSpace
from rl2.envs.lux_env import LuxObservation
from rl2.model.lux import LuxFeatures, LuxModel
from rl2.model.mappo import ActorCritic, LSTMConfig, LuxModelConfig


@pytest.fixture(scope="module")
def model() -> ActorCritic:
    return ActorCritic(
        action_space=HierarchicalMultiDiscreteSpace(nvec=(14,) * 3, group_sizes=(1, 4, 9)),
        model=LuxModelConfig(embedding_size=8, channels=4, recurrent=LSTMConfig(hidden_size=4)),
    )


def action_mask(observation: LuxObservation[jax.Array]) -> jax.Array:  # leaves: (*S, ...) -> (*S, U, A=14)
    return (
        jnp.broadcast_to(observation.self.units_mask[..., None], (*observation.self.units_mask.shape, 14))
        .at[..., 0]
        .set(True)
    )  # (*S, U, A)


@pytest.fixture(scope="module")
def params(model: ActorCritic, observation: LuxObservation[jax.Array]) -> dict[str, Any]:
    sequence = jax.tree.map(lambda leaf: leaf[None, None], observation)  # leaves: (B=1, T=1, ...)
    return model.init(
        jax.random.key(4), sequence, model.initial_carry(1), jnp.ones((1, 1), dtype=bool), action_mask(sequence)
    )


def test_sequence_step_replay_reset_and_bfloat16(
    model: ActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    sequence = jax.tree.map(lambda leaf: jnp.broadcast_to(leaf, (2, 2, *leaf.shape)), observation)  # (B, T, ...)
    starts = jnp.array([[False, False], [False, True]])  # (B, T)
    memory = jax.tree.map(lambda leaf: leaf + 0.5, model.initial_carry(2))  # leaves: (B, H)
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
    np.testing.assert_allclose(reduced.value, output.value, atol=0.04, rtol=0.04)
    np.testing.assert_allclose(reduced.policy.raw_logits, output.policy.raw_logits, atol=0.01, rtol=0.04)


def test_ship_permutation_equivariance(
    model: ActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    permutation = jnp.array([1, 2, 0])  # (U,)
    team = observation.self._replace(
        units=jax.tree.map(lambda leaf: leaf[permutation], observation.self.units),  # position: (U, 2); energy: (U,)
        units_mask=observation.self.units_mask[permutation],  # (U,)
    )
    batch = jax.tree.map(lambda a, b: jnp.stack((a, b)), observation, observation._replace(self=team))  # (B=2, ...)
    output = jax.jit(
        lambda obs: model.apply(params, obs, model.initial_carry(2), jnp.ones(2, bool), action_mask(obs), method="step")
    )(batch)
    np.testing.assert_allclose(output.policy.logits[1], output.policy.logits[0, permutation], atol=2e-6)
    np.testing.assert_allclose(output.value[1], output.value[0], atol=2e-6)
    for leaf in jax.tree.leaves(output.carry):
        np.testing.assert_allclose(leaf[0], leaf[1], atol=2e-6)


def test_empty_team_has_no_policy_gradient(
    model: ActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
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
    model: ActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    network = LuxModel(embedding_size=8, channels=4, action_space=model.action_space)
    variables = jax.tree.map(lambda leaf: leaf, params["params"]["lux"])
    variables["sap_query"] = {"kernel": jnp.zeros((8, 4)), "bias": jnp.ones(4)}
    variables["sap_key"] = {"kernel": jnp.zeros((4, 4)).at[0].set(1), "bias": jnp.zeros(4)}
    variables["sap_offset"] = {"kernel": jnp.zeros((2, 4))}
    x, y = jnp.meshgrid(jnp.arange(8), jnp.arange(12), indexing="ij")  # each (W, H)
    spatial = jnp.zeros((1, 8, 12, 4)).at[0, :, :, 0].set(10 * x + y)  # (B, W, H, C)
    features = LuxFeatures(global_features=jnp.zeros((1, 8)), ships=jnp.zeros((1, 3, 8)), spatial=spatial)
    obs = jax.tree.map(lambda leaf: leaf[None], observation)  # leaves: (B, ...)
    policy, value = network.apply(
        {"params": variables}, jnp.zeros((1, 4)), features, obs, action_mask(obs), method="heads"
    )
    # First ship is at (1, 2); offsets enumerate x first, then y, matching Lux's command IDs.
    np.testing.assert_allclose(policy.raw_logits[0, 0, 7:], 2 * jnp.array([1, 2, 3, 11, 12, 13, 21, 22, 23]))
    assert value.shape == (1,)


def test_single_ship_small_map_reuses_parameters(
    model: ActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
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
            sensor_mask=observation.world.sensor_mask[:, :8],  # (W=8, H=8)
            map_features=jax.tree.map(lambda leaf: leaf[:, :8], observation.world.map_features),  # (W, H)
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
    model: ActorCritic, params: dict[str, Any], observation: LuxObservation[jax.Array]
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
    for name in ("sap_query", "sap_key", "sap_offset", "move_output", "type_output", "value_output"):
        magnitude = sum(float(jnp.abs(leaf).sum()) for leaf in jax.tree.leaves(actual[1]["params"]["lux"][name]))
        assert 0 < magnitude < 1e6
    assert actual[0][1].addressable_shards[0].data.shape == (1, 1)
