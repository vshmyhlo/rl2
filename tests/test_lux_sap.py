"""Sap logits and gradients against the original gathered-feature calculation."""

from math import prod, sqrt
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from test_lux_encoder import observation as observation  # noqa: PLC0414

from rl2.actions import HierarchicalMultiDiscreteSpace
from rl2.envs.lux_env import LuxObservation
from rl2.model.lux import LuxHeads


def gathered_sap_logits(
    module: LuxHeads,
    ships: jax.Array,  # (*S, U, F)
    spatial: jax.Array,  # (*S, H, W, C)
    observation: LuxObservation[jax.Array],  # leaves: (*S, ...)
) -> jax.Array:  # (*S, U, As)
    batch_shape = ships.shape[:-2]
    batch_size = prod(batch_shape)
    own_ship_count = ships.shape[-2]
    height, width = spatial.shape[-3:-1]
    axis = jnp.arange(-module.sap_range, module.sap_range + 1)  # (2*sap_range+1,)
    offsets = jnp.stack(jnp.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))  # (As, 2)
    positions = jnp.where(observation.self.units_mask[..., None], observation.self.units.position, 0)  # (*S, U, 2)
    targets = positions.reshape((batch_size, own_ship_count, 1, 2)) + offsets  # (B, U, As, 2)
    targets = jnp.clip(targets, 0, jnp.array([width - 1, height - 1]))  # (B, U, As, 2)
    keys = module.sap_key(spatial).reshape((batch_size, height, width, module.conv_channels))  # (B, H, W, C)
    keys = keys[jnp.arange(batch_size)[:, None, None], targets[..., 1], targets[..., 0]]  # (B, U, As, C)
    normalized_offsets = jnp.asarray(offsets, dtype=module.dtype) / module.sap_range  # (As, 2)
    offset_features = jnp.concatenate((normalized_offsets, jnp.abs(normalized_offsets)), axis=-1)  # (As, 4)
    relative = module.sap_offset(nn.silu(module.sap_offset_hidden(offset_features)))  # (As, C)
    keys = keys + relative  # (B, U, As, C)
    queries = module.sap_query(ships).reshape((batch_size, own_ship_count, module.conv_channels))  # (B, U, C)
    sap = jnp.einsum("bukc,buc->buk", keys, queries, preferred_element_type=jnp.float32) / sqrt(
        module.conv_channels
    )  # (B, U, As)
    return sap.reshape((*batch_shape, own_ship_count, offsets.shape[0]))  # (*S, U, As)


@pytest.mark.parametrize(
    ("batch_shape", "dtype"),
    [((), jnp.float32), ((2, 1), jnp.bfloat16)],
    ids=["unbatched_float32", "sequence_bfloat16"],
)
def test_sap_logits_and_gradients_match_gathered_features(
    observation: LuxObservation[jax.Array],  # leaves: (...)
    batch_shape: tuple[int, ...],
    dtype: jax.typing.DTypeLike,
) -> None:
    network = LuxHeads(
        features=8,
        conv_channels=4,
        action_space=HierarchicalMultiDiscreteSpace(nvec=(14,) * 3, group_sizes=(1, 4, 9)),
        dtype=dtype,
    )
    # Include corners, clipped duplicate targets, and an absent ship on a rectangular map.
    observation = observation._replace(
        self=observation.self._replace(
            units=observation.self.units.replace(position=jnp.array([[0, 0], [3, 2], [-1, -1]], jnp.int16))
        )
    )
    obs = jax.tree.map(lambda leaf: jnp.broadcast_to(leaf, (*batch_shape, *leaf.shape)), observation)  # (*S, ...)
    ships = jax.random.normal(jax.random.key(40), (*batch_shape, 3, 8)).astype(dtype)  # (*S, U, F)
    spatial = jax.random.normal(jax.random.key(41), (*batch_shape, 3, 4, 4)).astype(dtype)  # (*S, H, W, C)
    weights = jax.random.normal(jax.random.key(42), (*batch_shape, 3, 9))  # (*S, U, As)
    variables = network.init(jax.random.key(43), ships, spatial, obs, method="_sap_logits")  # parameter leaves

    def loss(
        params: dict[str, Any],
        ships: jax.Array,  # (*S, U, F)
        spatial: jax.Array,  # (*S, H, W, C)
        *,
        reference: bool,
    ) -> tuple[jax.Array, jax.Array]:  # (); (*S, U, As)
        scores = network.apply(
            params, ships, spatial, obs, method=gathered_sap_logits if reference else "_sap_logits"
        )  # (*S, U, As)
        return jnp.sum(scores * weights), scores  # (); (*S, U, As)

    evaluate = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True), static_argnames="reference")
    (_, actual), gradients = evaluate(variables, ships, spatial, reference=False)  # (*S, U, As); parameter/input leaves
    (_, expected), expected_gradients = evaluate(variables, ships, spatial, reference=True)  # same shapes
    assert actual.shape == (*batch_shape, 3, 9) and actual.dtype == jnp.float32
    # Distributing the dot product removes the old bf16 rounding of key + relative.
    rtol, atol = (0.03, 0.003) if dtype == jnp.bfloat16 else (2e-5, 2e-6)
    for result, reference in zip(
        jax.tree.leaves((actual, gradients)), jax.tree.leaves((expected, expected_gradients)), strict=True
    ):
        np.testing.assert_allclose(result.astype(jnp.float32), reference.astype(jnp.float32), rtol=rtol, atol=atol)
