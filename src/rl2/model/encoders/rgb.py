"""Normalized residual convolutional encoder for Atari observations."""

import math
from dataclasses import dataclass

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from numpy.typing import NDArray

from rl2.resize_conv import ResizeConv


@dataclass(frozen=True)
class ConvStage:
    """A resize-convolution stage followed by optional 3x3 residual blocks.

    ``resize_factor`` multiplies each spatial dimension, rounding up (0.5 halves,
    1.0 preserves, 2.0 doubles). ``kernel_size`` applies to the entry convolution
    only. All stages use SAME padding and bilinear interpolation.
    """

    channels: int
    blocks: int = 2
    kernel_size: int = 3
    resize_factor: float = 0.5

    def __post_init__(self) -> None:
        chex.assert_type((self.channels, self.blocks, self.kernel_size), int)
        chex.assert_scalar_positive(self.channels)
        chex.assert_scalar_non_negative(self.blocks)
        chex.assert_scalar_positive(self.kernel_size)
        if not math.isfinite(self.resize_factor) or self.resize_factor <= 0:
            raise ValueError("resize_factor must be positive and finite")

    def output_shape(self, height: int, width: int) -> tuple[int, int]:
        """Compute the resized spatial shape, rounding each dimension up."""
        return math.ceil(height * self.resize_factor), math.ceil(width * self.resize_factor)


type ConvStages = tuple[ConvStage, ...]

DEFAULT_STAGES: ConvStages = (
    ConvStage(32, blocks=2),
    ConvStage(64, blocks=2),
    ConvStage(128, blocks=2),
    ConvStage(256, blocks=2),
    ConvStage(256, blocks=1),
    ConvStage(256, blocks=1),
)


def validate_stages(stages: ConvStages) -> None:
    """Validate a fixed stage list shared by an encoder and its decoder."""
    chex.assert_scalar_positive(len(stages))


class ResidualBlock(nn.Module):
    """Two Conv-LayerNorm-SiLU layers followed by a scaled residual sum."""

    channels: int
    dtype: jax.typing.DTypeLike = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        chex.assert_shape(x, (None, None, None, self.channels))
        chex.assert_type(x, self.dtype)
        chex.assert_scalar_positive(self.channels)
        residual = x
        for _ in range(2):
            x = nn.Conv(
                self.channels,
                (3, 3),
                padding="SAME",
                kernel_init=nn.initializers.variance_scaling(2.0, "fan_in", "truncated_normal"),
                dtype=self.dtype,
            )(x)
            x = nn.silu(nn.LayerNorm(dtype=self.dtype)(x))
        return (residual + x) * jnp.asarray(2**-0.5, dtype=self.dtype)


class ConvObservationStage(nn.Module):
    """Resize-convolve, then apply optional residual blocks."""

    channels: int
    spatial_shape: tuple[int, int]
    blocks: int = 2
    dtype: jax.typing.DTypeLike = jnp.float32
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        chex.assert_type((self.channels, self.blocks), int)
        chex.assert_scalar_positive(self.channels)
        chex.assert_scalar_non_negative(self.blocks)
        x = ResizeConv(
            self.channels,
            self.spatial_shape,
            dtype=self.dtype,
            kernel_size=self.kernel_size,
            name="resize_conv",
        )(x)
        for block in range(self.blocks):
            x = ResidualBlock(self.channels, dtype=self.dtype, name=f"block_{block}")(x)
        return x


class ConvObservationEncoder(nn.Module):
    """Encode uint8 observations shaped [batch, frames, height, width, (RGB)].

    A 7x7 stem convolution followed by LayerNorm and SiLU extracts features at
    the original resolution using the first configured channel width.
    Stages use bilinear resize followed by SAME convolution, with configurable
    kernels and resize factors. Optional residual
    blocks preserve spatial size. Defaults resize by two and convolve with 3x3.
    Stage count and projection weights are fixed by the initialization shape.
    """

    stages: ConvStages = DEFAULT_STAGES
    embedding_size: int = 768
    dtype: jax.typing.DTypeLike = jnp.float32

    @nn.compact
    def __call__(self, obs: jax.Array | NDArray[np.uint8]) -> jax.Array:
        validate_stages(self.stages)
        chex.assert_type(self.embedding_size, int)
        chex.assert_scalar_positive(self.embedding_size)
        for size in obs.shape[1:4]:
            chex.assert_scalar_positive(size)
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
        x = nn.Conv(
            self.stages[0].channels,
            (7, 7),
            padding="SAME",
            kernel_init=visual_init,
            dtype=self.dtype,
            name="stem",
        )(x)
        x = nn.silu(nn.LayerNorm(name="stem_norm", dtype=self.dtype)(x))
        for index, stage in enumerate(self.stages):
            spatial_shape = stage.output_shape(x.shape[1], x.shape[2])
            x = ConvObservationStage(
                channels=stage.channels,
                spatial_shape=spatial_shape,
                blocks=stage.blocks,
                kernel_size=stage.kernel_size,
                dtype=self.dtype,
                name=f"stage_{index}",
            )(x)
        # Keep the remaining spatial positions distinct in the projection.
        x = nn.Dense(self.embedding_size, kernel_init=visual_init, dtype=self.dtype)(
            x.reshape((x.shape[0], math.prod(x.shape[1:])))
        )
        x = nn.silu(nn.LayerNorm(name="shared_norm", dtype=self.dtype)(x))
        return x
