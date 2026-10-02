"""Residual convolutional decoder mirroring the observation encoder."""

import math

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from rl2.observation_encoder import DEFAULT_STAGES, ConvStages, ResidualBlock, validate_stages


class ConvObservationDecoder(nn.Module):
    """Decode [batch, embedding] into [batch, frames, height, width, (RGB)].

    Supply the same explicit stages as the encoder, in encoder order. Decoding
    traverses them in reverse: residual blocks, optional 3x3 projection, then
    bilinear resizing to the corresponding pre-downsampling spatial size.
    The final 7x7 convolution mirrors the encoder stem. Outputs are unbounded
    floating-point predictions on the normalized pixel scale, not uint8 pixels.
    """

    observation_shape: tuple[int, ...]
    stages: ConvStages = DEFAULT_STAGES
    dtype: jax.typing.DTypeLike = jnp.float32

    @nn.compact
    def __call__(self, latent: jax.Array) -> jax.Array:
        chex.assert_rank(latent, 2)
        chex.assert_type(latent, jnp.floating)
        chex.assert_scalar_positive(latent.shape[-1])
        validate_stages(self.stages)
        if len(self.observation_shape) not in (3, 4):
            raise ValueError("observation_shape must be [frames, height, width] or [frames, height, width, 3]")
        for size in self.observation_shape:
            chex.assert_type(size, int)
            chex.assert_scalar_positive(size)
        if len(self.observation_shape) == 4 and self.observation_shape[-1] != 3:
            raise ValueError("RGB observations must have exactly three color channels")
        frames, height, width = self.observation_shape[:3]
        # Remember exact sizes: doubling alone cannot invert rounding on odd inputs.
        spatial_shapes = [(height, width)]
        for _ in self.stages:
            height, width = (height + 1) // 2, (width + 1) // 2
            spatial_shapes.append((height, width))
        base_shape = (height, width, self.stages[-1].channels)
        visual_init = nn.initializers.variance_scaling(2.0, "fan_in", "truncated_normal")
        x = nn.Dense(math.prod(base_shape), kernel_init=visual_init, dtype=self.dtype, name="projection")(latent)
        x = x.reshape((latent.shape[0], *base_shape))
        x = nn.relu(nn.LayerNorm(name="projection_norm", dtype=self.dtype)(x))
        for index in reversed(range(len(self.stages))):
            stage = self.stages[index]
            for block in reversed(range(stage.blocks)):
                x = ResidualBlock(stage.channels, dtype=self.dtype, name=f"stage_{index}_block_{block}")(x)
            previous_channels = self.stages[max(index - 1, 0)].channels
            if stage.project:
                x = nn.Conv(
                    previous_channels,
                    (3, 3),
                    padding="SAME",
                    kernel_init=visual_init,
                    dtype=self.dtype,
                    name=f"stage_{index}_projection",
                )(x)
            x = jax.image.resize(
                x,
                (x.shape[0], *spatial_shapes[index], previous_channels),
                method="bilinear",
                antialias=True,
            )
        x = nn.relu(nn.LayerNorm(name="output_norm", dtype=self.dtype)(x))
        colors = self.observation_shape[3] if len(self.observation_shape) == 4 else 1
        x = nn.Conv(
            frames * colors,
            (7, 7),
            padding="SAME",
            kernel_init=nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal"),
            dtype=self.dtype,
            name="output",
        )(x)
        if len(self.observation_shape) == 4:
            x = x.reshape((*x.shape[:3], frames, colors))
            x = jnp.transpose(x, (0, 3, 1, 2, 4))
        else:
            x = jnp.moveaxis(x, -1, 1)
        chex.assert_shape(x, (latent.shape[0], *self.observation_shape))
        chex.assert_type(x, self.dtype)
        return x
