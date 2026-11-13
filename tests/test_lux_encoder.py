from functools import partial
from math import prod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, PartitionSpec
from luxai_s3.state import MapTile, UnitState

from rl2.envs.lux_env import LuxObservation, LuxTeamObservation, LuxWorldObservation
from rl2.model.config import LuxModelConfig
from rl2.model.lux import LuxEncoder, LuxFeatures, _preprocess_observation, _rasterize


@pytest.fixture(scope="module")
def observation() -> LuxObservation[jax.Array]:
    x, y = jnp.meshgrid(jnp.arange(8), jnp.arange(12), indexing="xy")  # each (H=12, W=8)
    visible = x < 6  # (H, W)
    own = LuxTeamObservation(
        units=UnitState(
            position=jnp.array([[1, 2], [5, 8], [-1, -1]], jnp.int16), energy=jnp.array([90, 160, -1], jnp.int16)
        ),
        units_mask=jnp.array([True, True, False]),  # (U=3,)
        points=jnp.array(10),  # ()
        wins=jnp.array(1),  # ()
    )  # positions: (U, 2); energy: (U,)
    opponent = own._replace(
        units=UnitState(
            position=jnp.array([[2, 4], [5, 7], [-1, -1]], jnp.int16), energy=jnp.array([40, 80, -1], jnp.int16)
        ),
        points=jnp.array(8),  # ()
        wins=jnp.array(0),  # ()
    )  # positions: (U, 2); energy: (U,)
    return LuxObservation(
        self=own,
        opponent=opponent,
        world=LuxWorldObservation(
            sensor_mask=visible,  # (H, W)
            map_features=MapTile(
                energy=jnp.where(visible, x - y, -1).astype(jnp.int16),  # (H, W)
                tile_type=jnp.where(visible, (x + y) % 3, -1).astype(jnp.int16),  # (H, W)
            ),
            relic_nodes=jnp.array([[3, 3], [-1, -1]], jnp.int16),  # (R=2, 2)
            relic_nodes_mask=jnp.array([True, False]),  # (R,)
            steps=jnp.array(145),  # ()
            match_steps=jnp.array(44),  # ()
        ),
        public_params=jnp.array([3, 30, 4, 2], jnp.int16),  # (4,)
    )


@pytest.fixture(scope="module")
def encoder() -> LuxEncoder:
    return LuxEncoder(
        model=LuxModelConfig(conv_channels=4, num_blocks=1, heads=4, head_dim=2, attention_implementation="xla")
    )


@pytest.fixture(scope="module")
def params(encoder: LuxEncoder, observation: LuxObservation[jax.Array]) -> dict[str, Any]:
    sequence = jax.tree.map(lambda leaf: leaf[None, None], observation)  # leaves: (B=1, T=1, ...)
    return encoder.init(jax.random.key(0), sequence, encoder.initial_carry(1), jnp.ones((1, 1), bool))


def encode_independent(
    encoder: LuxEncoder,
    params: dict[str, Any],
    observation: LuxObservation[jax.Array],  # leaves: (*S, ...)
) -> LuxFeatures:  # leaves: (*S, ...)
    """Encode each observation as a one-step sequence with fresh memory."""
    batch_shape = observation.public_params.shape[:-1]
    batch_size = prod(batch_shape)
    sequence = jax.tree.map(
        lambda leaf: leaf.reshape((batch_size, 1, *leaf.shape[len(batch_shape) :])), observation
    )  # leaves: (B, T=1, ...)
    features, _ = encoder.apply(
        params, sequence, encoder.initial_carry(batch_size), jnp.ones((batch_size, 1), bool)
    )  # features: (B, T=1, ...); carry leaves: (B, Hg)
    return jax.tree.map(lambda leaf: leaf.reshape((*batch_shape, *leaf.shape[2:])), features)  # leaves: (*S, ...)


@pytest.mark.parametrize("batch_shape", [(), (2, 1)], ids=["unbatched", "batch_and_time"])
def test_preprocess_observation(observation: LuxObservation[jax.Array], batch_shape: tuple[int, ...]) -> None:
    observation = jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf, (*batch_shape, *leaf.shape)), observation
    )  # leaves: (*S, ...)
    actual = jax.jit(_preprocess_observation)(observation)  # leaves: (*S, ...)
    assert isinstance(actual, LuxObservation)
    assert jax.tree.structure(actual) == jax.tree.structure(observation)
    for original, processed in zip(jax.tree.leaves(observation), jax.tree.leaves(actual), strict=True):
        assert processed.shape == original.shape
    for original, processed, expected_energy in (
        (observation.self, actual.self, [0.225, 0.4, 0.0]),
        (observation.opponent, actual.opponent, [0.1, 0.2, 0.0]),
    ):
        assert processed.units.energy.dtype == jnp.float32
        assert processed.units.position.dtype == original.units.position.dtype
        np.testing.assert_allclose(
            processed.units.energy, jnp.broadcast_to(jnp.array(expected_energy), (*batch_shape, 3))
        )
        np.testing.assert_array_equal(processed.units.position[..., :2, :], original.units.position[..., :2, :])
        np.testing.assert_array_equal(processed.units.position[..., 2, :], 0)
        np.testing.assert_array_equal(processed.units_mask, original.units_mask)
        np.testing.assert_array_equal(processed.points, original.points)
        np.testing.assert_array_equal(processed.wins, original.wins)
        np.testing.assert_array_equal(original.units.energy[..., 2], -1)
        np.testing.assert_array_equal(original.units.position[..., 2, :], -1)
    np.testing.assert_array_equal(actual.world.relic_nodes[..., 0, :], observation.world.relic_nodes[..., 0, :])
    np.testing.assert_array_equal(actual.world.relic_nodes[..., 1, :], 0)
    unchanged = actual._replace(self=observation.self, opponent=observation.opponent)
    unchanged = unchanged._replace(world=unchanged.world._replace(relic_nodes=observation.world.relic_nodes))
    for original, processed in zip(jax.tree.leaves(observation), jax.tree.leaves(unchanged), strict=True):
        np.testing.assert_array_equal(processed, original)


def test_shapes_batch_replay_and_bfloat16(
    encoder: LuxEncoder, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    expected = jax.jit(partial(encode_independent, encoder))(
        params, observation
    )  # global: (F=8,); ships: (U, D); spatial: (H, W, C)
    assert expected.global_.shape == (8,)
    assert expected.ships.shape == (3, 8)
    assert expected.spatial.shape == (12, 8, 4)
    sequence = jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf, (2, 2, *leaf.shape)), observation
    )  # leaves: (B, T, ...)
    actual, carry = jax.jit(encoder.apply)(
        params, sequence, encoder.initial_carry(2), jnp.ones((2, 2), bool)
    )  # features: (B, T, ...); carry leaves: (B, Hg)
    assert len(carry.global_) == encoder.model.num_blocks
    assert all(leaf.shape == (2, 8) and leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(carry))
    assert actual.global_.shape == (2, 2, 8)
    assert actual.ships.shape == (2, 2, 3, 8)
    assert actual.spatial.shape == (2, 2, 12, 8, 4)
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(actual))
    encoded = encoder.apply(params, sequence, method="_prepare")
    expected_mask = jnp.concatenate(
        (
            jnp.ones((2, 2, 1), bool),
            sequence.self.units_mask,
            sequence.opponent.units_mask,
            sequence.world.relic_nodes_mask,
        ),
        axis=-1,
    )  # (B, T, E)
    assert encoded.tokens.shape == (*expected_mask.shape, 8)
    assert encoded.tokens_mask.shape == expected_mask.shape
    np.testing.assert_array_equal(encoded.tokens_mask, expected_mask)
    np.testing.assert_array_equal(encoded.token_types, [0, 1, 1, 1, 2, 2, 2, 3, 3])
    np.testing.assert_array_equal(encoded.tokens[~encoded.tokens_mask], 0)
    assert encoded.spatial.shape == (2, 2, 12, 8, 4)
    assert encoded.map_tokens.shape == (2, 2, 6, 8)
    for value, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(value, jnp.broadcast_to(reference, value.shape), atol=2e-6)
    steps = jax.tree.map(lambda leaf: leaf[:, 0], sequence)  # leaves: (B, ...)
    step_features = encode_independent(encoder, params, steps)  # leaves: (B, ...)
    reduced = jax.jit(partial(encode_independent, encoder.clone(dtype=jnp.bfloat16)))(params, steps)  # leaves: (B, ...)
    for value, reference, low_precision in zip(
        jax.tree.leaves(step_features), jax.tree.leaves(actual), jax.tree.leaves(reduced), strict=True
    ):
        assert value.shape == low_precision.shape == reference[:, 0].shape
        assert low_precision.dtype == jnp.bfloat16
        np.testing.assert_allclose(value, reference[:, 0], atol=2e-6)
        np.testing.assert_allclose(low_precision.astype(jnp.float32), reference[:, 0], atol=0.06, rtol=0.06)


def test_hidden_values_are_ignored(
    encoder: LuxEncoder, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    def corrupt(team: LuxTeamObservation[jax.Array]) -> LuxTeamObservation[jax.Array]:
        return team._replace(
            units=UnitState(
                position=jnp.where(team.units_mask[:, None], team.units.position, 30000),  # (U, 2)
                energy=jnp.where(team.units_mask, team.units.energy, 30000),  # (U,)
            )
        )

    world = observation.world
    changed = observation._replace(
        self=corrupt(observation.self),
        opponent=corrupt(observation.opponent),
        world=world._replace(
            map_features=jax.tree.map(lambda leaf: jnp.where(world.sensor_mask, leaf, 30000), world.map_features),
            relic_nodes=jnp.where(world.relic_nodes_mask[:, None], world.relic_nodes, 30000),  # (R, 2)
        ),
    )
    pair = jax.tree.map(lambda *leaves: jnp.stack(leaves), observation, changed)  # leaves: (B=2, ...)
    outputs = jax.jit(partial(encode_independent, encoder))(params, pair)  # leaves: (B, ...)
    for leaf in jax.tree.leaves(outputs):
        np.testing.assert_array_equal(leaf[0], leaf[1])


def test_entity_ordering(encoder: LuxEncoder, params: dict[str, Any], observation: LuxObservation[jax.Array]) -> None:
    observation = observation._replace(
        world=observation.world._replace(
            relic_nodes=jnp.array([[3, 3], [4, 5]], jnp.int16),  # (R=2, 2)
            relic_nodes_mask=jnp.ones(2, dtype=bool),  # (R,)
        )
    )

    def permute(team: LuxTeamObservation[jax.Array]) -> LuxTeamObservation[jax.Array]:
        return team._replace(
            units=jax.tree.map(lambda leaf: leaf[jnp.array([1, 0, 2])], team.units),  # (U, 2)/(U,)
            units_mask=team.units_mask[jnp.array([1, 0, 2])],  # (U,)
        )

    unordered = observation._replace(
        opponent=permute(observation.opponent),
        world=observation.world._replace(
            relic_nodes=observation.world.relic_nodes[::-1],  # (R, 2)
            relic_nodes_mask=observation.world.relic_nodes_mask[::-1],  # (R,)
        ),
    )
    own_permuted = observation._replace(self=permute(observation.self))
    batch = jax.tree.map(lambda *leaves: jnp.stack(leaves), observation, unordered, own_permuted)  # leaves: (B=3, ...)
    encoded = encoder.apply(params, batch, method="_prepare")  # leaves: (B, ...)
    np.testing.assert_allclose(encoded.tokens[1], encoded.tokens[0, jnp.array([0, 1, 2, 3, 5, 4, 6, 8, 7])], atol=2e-6)
    np.testing.assert_allclose(encoded.tokens[2], encoded.tokens[0, jnp.array([0, 2, 1, 3, 4, 5, 6, 7, 8])], atol=2e-6)
    output = jax.jit(partial(encode_independent, encoder))(params, batch)  # leaves: (B, ...)
    np.testing.assert_allclose(output.global_[0], output.global_[1], atol=2e-6)
    np.testing.assert_allclose(output.global_[0], output.global_[2], atol=2e-6)
    np.testing.assert_allclose(output.ships[0], output.ships[1], atol=2e-6)
    np.testing.assert_allclose(output.ships[2], output.ships[0, jnp.array([1, 0, 2])], atol=2e-6)


def test_visible_inputs_affect_encoding(
    encoder: LuxEncoder, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    world = observation.world
    alternatives = [
        observation._replace(
            world=world._replace(
                map_features=world.map_features.replace(energy=world.map_features.energy.at[2, 3].set(20))
            )
        ),
        observation._replace(
            world=world._replace(
                map_features=world.map_features.replace(tile_type=world.map_features.tile_type.at[2, 3].set(0))
            )
        ),
        observation._replace(world=world._replace(sensor_mask=world.sensor_mask.at[2, 3].set(False))),
        observation._replace(
            world=world._replace(relic_nodes=world.relic_nodes.at[0].set(jnp.array([4, 5], jnp.int16)))
        ),
        observation._replace(
            self=observation.self._replace(
                units=observation.self.units.replace(energy=observation.self.units.energy.at[0].set(300))
            )
        ),
        observation._replace(
            opponent=observation.opponent._replace(
                units=observation.opponent.units.replace(
                    position=observation.opponent.units.position.at[0].set(jnp.array([1, 1], jnp.int16))
                )
            )
        ),
        observation._replace(self=observation.self._replace(points=jnp.array(90), wins=jnp.array(3))),
        observation._replace(world=world._replace(steps=jnp.array(401), match_steps=jnp.array(1))),
        observation._replace(public_params=jnp.array([5, 50, 7, 7], jnp.int16)),
    ]
    batch = jax.tree.map(lambda *leaves: jnp.stack(leaves), observation, *alternatives)  # leaves: (B=10, ...)
    output = jax.jit(partial(encode_independent, encoder))(params, batch).global_  # (B, F)
    assert bool(jnp.all(jnp.max(jnp.abs(output[1:] - output[:1]), axis=-1) > 1e-5))


def test_single_block_remembers_entities_and_map(
    encoder: LuxEncoder, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    alternatives = (
        observation._replace(
            self=observation.self._replace(
                units=observation.self.units.replace(energy=observation.self.units.energy.at[0].set(300))
            )
        ),
        observation._replace(
            world=observation.world._replace(
                map_features=observation.world.map_features.replace(
                    energy=observation.world.map_features.energy.at[2, 3].set(20)
                )
            )
        ),
        observation._replace(
            world=observation.world._replace(
                relic_nodes=observation.world.relic_nodes.at[0].set(jnp.array([4, 5], jnp.int16))
            )
        ),
    )
    # These observations have identical global context, but different entity/map details.
    sequence = jax.tree.map(
        lambda *leaves: jnp.stack(leaves)[:, None], observation, *alternatives
    )  # leaves: (B=4, T=1, ...)
    apply = jax.jit(encoder.apply)
    _, carry = apply(params, sequence, encoder.initial_carry(4), jnp.ones((4, 1), bool))
    for leaf in jax.tree.leaves(carry):  # (B, Hg)
        assert bool((jnp.max(jnp.abs(leaf[1:] - leaf[:1]), axis=-1) > 1e-6).all())
    identical = jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf, (4, 1, *leaf.shape)), observation
    )  # leaves: (B, T, ...)
    output, _ = apply(params, identical, carry, jnp.zeros((4, 1), bool))
    for leaf in (output.global_, output.ships[..., 0, :]):  # (B, T, D)
        assert bool((jnp.max(jnp.abs(leaf[1:] - leaf[:1]), axis=-1) > 1e-6).all())
    reset, reset_carry = apply(params, identical, carry, jnp.ones((4, 1), bool))
    for leaf in jax.tree.leaves((reset, reset_carry)):
        np.testing.assert_allclose(leaf[1:], jnp.broadcast_to(leaf[:1], leaf[1:].shape), atol=2e-6)


def test_gradients_and_empty_observation(
    encoder: LuxEncoder, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    def objective(variables: dict[str, Any], obs: LuxObservation[jax.Array]) -> jax.Array:  # obs leaves: (...) -> ()
        output = encode_independent(encoder, variables, obs).global_  # (F,)
        return jnp.vdot(output, jnp.arange(8, dtype=jnp.float32))  # ()

    gradient = jax.jit(jax.grad(objective))(params, observation)  # parameter leaves: (...)
    for layer in (
        gradient["params"]["map_encoder"]["tile_embedding"],
        gradient["params"]["map_encoder"]["map_stem"],
        gradient["params"]["ship_projection"],
        gradient["params"]["context_projection"],
        gradient["params"]["output_projection"],
    ):  # parameter leaves: (...)
        norm = sum(float(jnp.sum(jnp.abs(leaf))) for leaf in jax.tree.leaves(layer))
        assert 0 < norm < 1e6
    empty = observation._replace(
        self=observation.self._replace(units_mask=jnp.zeros_like(observation.self.units_mask)),  # (U,)
        opponent=observation.opponent._replace(units_mask=jnp.zeros_like(observation.opponent.units_mask)),  # (U,)
        world=observation.world._replace(
            sensor_mask=jnp.zeros_like(observation.world.sensor_mask),  # (H, W)
            relic_nodes_mask=jnp.zeros_like(observation.world.relic_nodes_mask),  # (R,)
        ),
    )
    output = encode_independent(encoder, params, empty).global_  # (F,)
    assert float(jnp.max(jnp.abs(output))) < 3  # Layer-normalized eight-dimensional output.
    empty_gradient = jax.jit(jax.grad(objective))(params, empty)  # parameter leaves: (...)
    assert all(float(jnp.max(jnp.abs(leaf))) < 1e6 for leaf in jax.tree.leaves(empty_gradient))
    for leaf in jax.tree.leaves(empty_gradient["params"]["ship_projection"]):
        np.testing.assert_array_equal(leaf, jnp.zeros_like(leaf))


def test_rasterization_accumulates_stacked_ships_and_ignores_absent_slots() -> None:
    positions = jnp.array([[[1, 2], [1, 2], [-1, -1]]])  # (B=1, U=3, 2)
    mask = jnp.array([[True, True, False]])  # (B, U)
    values = jnp.array([[[1.0, 0.25], [1.0, 0.5], [1.0, -1.0]]])  # (B, U, C=2)
    actual = _rasterize(positions, mask, values, height=3, width=4)  # (B, H, W, C)
    expected = jnp.zeros((1, 3, 4, 2)).at[0, 2, 1].set(jnp.array([2.0, 0.75]))  # (B, H, W, C)
    np.testing.assert_array_equal(actual, expected)


def test_entities_gather_height_width_tiles(
    encoder: LuxEncoder, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    obs = jax.tree.map(lambda leaf: leaf[None], observation)  # leaves: (B=1, ...)
    obs = obs._replace(world=obs.world._replace(relic_nodes=obs.world.relic_nodes.at[0, 0].set(jnp.array([5, 8]))))
    obs = _preprocess_observation(obs)  # leaves: (B, ...)
    spatial = jnp.zeros((1, 12, 8, 4))  # (B, H, W, C)
    reference_units, reference_relics = encoder.apply(
        params, obs, spatial, method="_encode_entities"
    )  # (B, N, 2C); (B, R, 2C)
    units, relics = encoder.apply(
        params, obs, spatial.at[0, 8, 5].set(1), method="_encode_entities"
    )  # (B, N, 2C); (B, R, 2C)
    np.testing.assert_array_equal(
        jnp.any(units != reference_units, axis=-1), [[False, True, False, False, False, False]]
    )
    np.testing.assert_array_equal(jnp.any(relics != reference_relics, axis=-1), [[True, False]])


def test_sharded_encoder_matches_outputs_and_gradients(
    encoder: LuxEncoder, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    if jax.device_count() < 2:
        pytest.skip("Requires two devices; use JAX_NUM_CPU_DEVICES=2 on CPU")
    second = observation._replace(public_params=jnp.array([5, 50, 7, 7], jnp.int16))  # (4,)
    batch = jax.tree.map(lambda *leaves: jnp.stack(leaves), observation, second)  # leaves: (B=2, ...)

    def objective(
        variables: dict[str, Any], obs: LuxObservation[jax.Array]
    ) -> tuple[jax.Array, jax.Array]:  # obs leaves: (B, ...) -> (); (B, F)
        output = encode_independent(encoder, variables, obs).global_  # (B, F)
        return jnp.mean(output * jnp.arange(8, dtype=jnp.float32)), output  # (); (B, F)

    differentiated = jax.value_and_grad(objective, has_aux=True)
    expected = jax.jit(differentiated)(params, batch)  # (loss, embeddings), parameter gradients
    mesh = Mesh(devices=jax.devices()[:2], axis_names=("batch",), axis_types=(AxisType.Auto,))
    with jax.set_mesh(mesh):
        actual = jax.jit(
            differentiated,
            in_shardings=(PartitionSpec(), PartitionSpec("batch")),
            out_shardings=((PartitionSpec(), PartitionSpec("batch")), PartitionSpec()),
        )(jax.device_put(params, PartitionSpec()), jax.device_put(batch, PartitionSpec("batch")))
    for value, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(value, reference, atol=2e-5, rtol=2e-4)
    assert actual[0][1].addressable_shards[0].data.shape == (1, 8)


@pytest.mark.parametrize("empty", [False, True], ids=["unequal_set_lengths", "all_sets_empty"])
def test_set_padding_does_not_change_features(
    encoder: LuxEncoder, params: dict[str, Any], observation: LuxObservation[jax.Array], empty: bool
) -> None:
    own_count, opponent_count, relic_count = (0, 0, 0) if empty else (2, 1, 1)

    def select(team: LuxTeamObservation[jax.Array], count: int, compact: bool) -> LuxTeamObservation[jax.Array]:
        if compact:
            return team._replace(
                units=jax.tree.map(lambda leaf: leaf[:count], team.units),  # (count, 2)/(count,)
                units_mask=team.units_mask[:count],  # (count,)
            )
        return team._replace(units_mask=team.units_mask & (jnp.arange(team.units_mask.size) < count))  # (U/V,)

    padded = observation._replace(
        self=select(observation.self, own_count, False),
        opponent=select(observation.opponent, opponent_count, False),
        world=observation.world._replace(
            relic_nodes_mask=observation.world.relic_nodes_mask & (jnp.arange(2) < relic_count)
        ),
    )
    compact = observation._replace(
        self=select(observation.self, own_count, True),
        opponent=select(observation.opponent, opponent_count, True),
        world=observation.world._replace(
            relic_nodes=observation.world.relic_nodes[:relic_count],  # (R, 2)
            relic_nodes_mask=observation.world.relic_nodes_mask[:relic_count],  # (R,)
        ),
    )
    apply = jax.jit(partial(encode_independent, encoder))
    reference, actual = apply(params, padded), apply(params, compact)  # leaves: (...)
    assert actual.ships.shape == (own_count, 8)
    np.testing.assert_allclose(actual.global_, reference.global_, atol=2e-6)
    np.testing.assert_allclose(actual.ships, reference.ships[:own_count], atol=2e-6)
    np.testing.assert_allclose(actual.spatial, reference.spatial, atol=2e-6)
    np.testing.assert_array_equal(reference.ships[own_count:], 0)
