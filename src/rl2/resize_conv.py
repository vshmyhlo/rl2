"""Spatial resizing followed by a linear convolution."""

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn


class ResizeConv(nn.Module):
    """Antialiased bilinear resize followed by a 3x3 convolution."""

    channels: int
    spatial_shape: tuple[int, int]
    dtype: jax.typing.DTypeLike = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        chex.assert_rank(x, 4)
        chex.assert_type(x, self.dtype)
        chex.assert_type(self.channels, int)
        chex.assert_scalar_positive(self.channels)
        chex.assert_equal(len(self.spatial_shape), 2)
        chex.assert_type(self.spatial_shape, int)
        for size in (*self.spatial_shape, *x.shape[1:]):
            chex.assert_scalar_positive(size)
        x = jax.image.resize(
            x,
            (x.shape[0], *self.spatial_shape, x.shape[-1]),
            method="bilinear",
            antialias=True,
        )
        x = nn.Conv(
            self.channels,
            (3, 3),
            padding="SAME",
            kernel_init=nn.initializers.variance_scaling(2.0, "fan_in", "truncated_normal"),
            dtype=self.dtype,
            name="conv",
        )(x)
        chex.assert_shape(x, (None, *self.spatial_shape, self.channels))
        chex.assert_type(x, self.dtype)
        return x
