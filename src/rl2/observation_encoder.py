"""Normalized residual convolutional encoder for Atari observations."""

import math

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from numpy.typing import NDArray


class ResidualBlock(nn.Module):
    """Pre-activation residual block with per-pixel channel normalization."""

    channels: int
    dtype: jax.typing.DTypeLike = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        chex.assert_shape(x, (None, None, None, self.channels))
        chex.assert_type(x, self.dtype)
        chex.assert_scalar_positive(self.channels)
        residual = x
        for index in range(2):
            x = nn.relu(nn.LayerNorm(dtype=self.dtype)(x))
            x = nn.Conv(
                self.channels,
                (3, 3),
                padding="SAME",
                kernel_init=nn.initializers.variance_scaling(2.0 if index == 0 else 1.0, "fan_in", "truncated_normal"),
                dtype=self.dtype,
            )(x)
        return (residual + x) * jnp.asarray(2**-0.5, dtype=self.dtype)


class ConvObservationEncoder(nn.Module):
    """Encode uint8 observations shaped [batch, frames, height, width, (RGB)].

    After the configured stages, repeat the final channel width with additional
    downsampling stages until the flattened size is at most max_flattened_size.
    Set the limit to None to use only encoder_channels. The limit must be at
    least the final channel width, which is the minimum size at 1x1 resolution.
    Each stage downsamples with antialiased bilinear resizing, rounding each
    halved spatial dimension up.
    Stage count and projection weights are fixed by the initialization shape.
    """

    encoder_channels: tuple[int, ...] = (128, 256, 384, 512)
    embedding_size: int = 768
    dtype: jax.typing.DTypeLike = jnp.float32
    max_flattened_size: int | None = 8192

    @nn.compact
    def __call__(self, obs: jax.Array | NDArray[np.uint8]) -> jax.Array:
        chex.assert_rank(obs, {4, 5})
        chex.assert_type(obs, jnp.uint8)
        chex.assert_scalar_positive(len(self.encoder_channels))
        for size in (*self.encoder_channels, self.embedding_size):
            chex.assert_type(size, int)
            chex.assert_scalar_positive(size)
        if self.max_flattened_size is not None:
            chex.assert_type(self.max_flattened_size, int)
            chex.assert_scalar_non_negative(self.max_flattened_size - self.encoder_channels[-1])
        if obs.ndim == 5:  # Raw RGB: combine stacked frames and color channels.
            chex.assert_shape(obs, (None, None, None, None, 3))
            x = jnp.transpose(obs, (0, 2, 3, 1, 4))
            x = x.reshape((*x.shape[:3], obs.shape[1] * obs.shape[-1]))
        else:
            x = jnp.moveaxis(obs, 1, -1)
        x = (x.astype(jnp.float32) / 255.0).astype(self.dtype)
        # Flax keeps parameters and LayerNorm statistics in float32 by default.
        # IMPALA-style stages; normalization is independent of rollout/minibatch size.
        # Variance scaling avoids expensive QR initialization of large visual kernels.
        visual_init = nn.initializers.variance_scaling(2.0, "fan_in", "truncated_normal")
        stage_channels = self.encoder_channels
        # Each resize halves the spatial dimensions, rounding up at every stage.
        stride = 2 ** len(stage_channels)
        height, width = ((size + stride - 1) // stride for size in x.shape[1:3])
        if self.max_flattened_size is not None:
            while height * width * stage_channels[-1] > self.max_flattened_size:
                stage_channels += (stage_channels[-1],)
                height, width = (height + 1) // 2, (width + 1) // 2
        for stage, channels in enumerate(stage_channels):
            x = nn.Conv(channels, (3, 3), padding="SAME", kernel_init=visual_init, dtype=self.dtype)(x)
            x = jax.image.resize(
                x,
                (x.shape[0], (x.shape[1] + 1) // 2, (x.shape[2] + 1) // 2, channels),
                method="bilinear",
                antialias=True,
            )
            for block in range(2):
                x = ResidualBlock(channels, dtype=self.dtype, name=f"stage_{stage}_block_{block}")(x)
        x = nn.relu(nn.LayerNorm(name="encoder_norm", dtype=self.dtype)(x))
        # Keep the remaining spatial positions distinct in the projection.
        x = nn.Dense(self.embedding_size, kernel_init=visual_init, dtype=self.dtype)(
            x.reshape((x.shape[0], math.prod(x.shape[1:])))
        )
        x = nn.relu(nn.LayerNorm(name="shared_norm", dtype=self.dtype)(x))
        chex.assert_shape(x, (obs.shape[0], self.embedding_size))
        chex.assert_type(x, self.dtype)
        return x
