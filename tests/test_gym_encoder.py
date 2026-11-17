"""Gym observation flattening and pixel normalization."""

import jax
import jax.numpy as jnp
import numpy as np

from rl2.model.encoders.gym import GymMLPEncoder


def test_mlp_encodes_nested_observations_like_normalized_flat_features() -> None:
    encoder = GymMLPEncoder(hidden_size=4, embedding_size=4)
    obs = {
        "pixels": (jnp.array([[[0, 255]], [[255, 0]]], dtype=jnp.uint8),),  # (B, H, W)
        "state": jnp.array([[2, -3], [-4, 5]], dtype=jnp.float32),  # (B, S)
    }
    flat = jnp.array([[0, 1, 2, -3], [1, 0, -4, 5]], dtype=jnp.float32)  # (B, F)
    params = encoder.init(jax.random.key(0), obs)
    apply = jax.jit(encoder.apply)
    output = apply(params, obs)  # (B, E)
    expected = apply(params, flat)  # (B, E)
    assert output.shape == (2, 4)
    assert output.dtype == jnp.float32
    np.testing.assert_allclose(jax.device_get(output), jax.device_get(expected), atol=1e-6)
