from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tokamax  # noqa: F401 - load Triton before luxai_s3 imports TensorFlow
from jax.sharding import AxisType, Mesh, PartitionSpec
from luxai_s3.state import MapTile, UnitState

from rl2.envs.lux_env import LuxObservation, LuxTeamObservation, LuxWorldObservation
from rl2.model.lux import LuxEncoder, _rasterize


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
        init_unit_energy=jnp.array(100, jnp.int32),  # ()
        spawn_rate=jnp.array(3, jnp.int32),  # ()
    )


@pytest.fixture(scope="module")
def encoder() -> LuxEncoder:
    return LuxEncoder(features=8, channels=4)


@pytest.fixture(scope="module")
def params(encoder: LuxEncoder, observation: LuxObservation[jax.Array]) -> dict[str, Any]:
    return encoder.init(jax.random.key(0), observation)


def test_shapes_batch_replay_and_bfloat16(
    encoder: LuxEncoder, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    expected = jax.jit(encoder.apply)(params, observation)  # (F=8,)
    sequence = jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf, (2, 2, *leaf.shape)), observation
    )  # leaves: (B, T, ...)
    actual = jax.jit(encoder.apply)(params, sequence)  # (B, T, F)
    assert actual.shape == (2, 2, 8) and actual.dtype == jnp.float32
    encoded = encoder.apply(params, sequence, method="prepare")
    assert encoded.spatial.shape == (2, 2, 12, 8, 4)
    assert encoded.map_tokens.shape == (2, 2, 6, 8)
    np.testing.assert_allclose(actual, jnp.broadcast_to(expected, actual.shape), atol=2e-6)
    steps = jax.tree.map(lambda leaf: leaf[:, 0], sequence)  # leaves: (B, ...)
    np.testing.assert_allclose(encoder.apply(params, steps), actual[:, 0], atol=2e-6)
    reduced = jax.jit(encoder.clone(dtype=jnp.bfloat16).apply)(params, steps)  # (B, F)
    assert reduced.shape == (2, 8) and reduced.dtype == jnp.bfloat16
    np.testing.assert_allclose(reduced.astype(jnp.float32), actual[:, 0], atol=0.06, rtol=0.06)


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
    outputs = jax.jit(lambda obs: encoder.apply(params, obs, method="encode_features"))(pair)
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
    output = jax.jit(lambda obs: encoder.apply(params, obs, method="encode_features"))(batch)
    np.testing.assert_allclose(output.global_features[0], output.global_features[1], atol=2e-6)
    np.testing.assert_allclose(output.global_features[0], output.global_features[2], atol=2e-6)
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
    output = jax.jit(encoder.apply)(params, batch)  # (B, F)
    assert bool(jnp.all(jnp.max(jnp.abs(output[1:] - output[:1]), axis=-1) > 1e-5))


def test_gradients_and_empty_observation(
    encoder: LuxEncoder, params: dict[str, Any], observation: LuxObservation[jax.Array]
) -> None:
    def objective(variables: dict[str, Any], obs: LuxObservation[jax.Array]) -> jax.Array:  # obs leaves: (...) -> ()
        output = encoder.apply(variables, obs)  # (F,)
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
    output = encoder.apply(params, empty)  # (F,)
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
    positions = jnp.concatenate((obs.self.units.position, obs.opponent.units.position), axis=1)  # (B, N=6, 2)
    positions = jnp.maximum(positions, 0)  # (B, N, 2), masked slots at the origin
    energy = jnp.zeros((1, 6))  # (B, N)
    spatial = jnp.zeros((1, 12, 8, 4))  # (B, H, W, C)
    reference_units, reference_relics = encoder.apply(
        params, obs, positions, energy, spatial, method="_encode_entities"
    )  # (B, N, 2C); (B, R, 2C)
    units, relics = encoder.apply(
        params, obs, positions, energy, spatial.at[0, 8, 5].set(1), method="_encode_entities"
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
        output = encoder.apply(variables, obs)  # (B, F)
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
    apply = jax.jit(lambda obs: encoder.apply(params, obs, method="encode_features"))
    reference, actual = apply(padded), apply(compact)
    assert actual.ships.shape == (own_count, 8)
    np.testing.assert_allclose(actual.global_features, reference.global_features, atol=2e-6)
    np.testing.assert_allclose(actual.ships, reference.ships[:own_count], atol=2e-6)
    np.testing.assert_allclose(actual.spatial, reference.spatial, atol=2e-6)
    np.testing.assert_array_equal(reference.ships[own_count:], 0)
