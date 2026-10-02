"""Normalized residual convolutional encoder for Atari observations."""

import math
from dataclasses import dataclass

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from numpy.typing import NDArray


@dataclass(frozen=True)
class ConvStage:
    """One resize stage, with an optional 3x3 projection and residual blocks."""

    channels: int
    blocks: int = 2
    project: bool = True

    def __post_init__(self) -> None:
        chex.assert_type((self.channels, self.blocks), int)
        chex.assert_scalar_positive(self.channels)
        chex.assert_scalar_positive(self.blocks)
        chex.assert_type(self.project, bool)


type ConvStages = tuple[ConvStage, ...]

DEFAULT_STAGES: ConvStages = (
    ConvStage(32, blocks=2, project=True),
    ConvStage(64, blocks=2, project=True),
    ConvStage(128, blocks=2, project=True),
    ConvStage(256, blocks=2, project=True),
    ConvStage(256, blocks=1, project=False),
    ConvStage(256, blocks=1, project=False),
)


def validate_stages(stages: ConvStages) -> None:
    """Validate a fixed stage list shared by an encoder and its decoder."""
    chex.assert_scalar_positive(len(stages))
    channels = stages[0].channels
    for stage in stages:
        if not stage.project and stage.channels != channels:
            raise ValueError("A stage that changes channel width must set project=True")
        channels = stage.channels


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

    A 7x7 stem convolution followed by LayerNorm and ReLU extracts features at
    the original resolution using the first configured channel width.
    Every stage is explicitly specified by channels, residual-block count, and
    whether to apply a 3x3 projection. No stages are inferred from input size.
    Resizing uses antialiased bilinear interpolation and rounds each halved
    spatial dimension up.
    Stage count and projection weights are fixed by the initialization shape.
    """

    stages: ConvStages = DEFAULT_STAGES
    embedding_size: int = 768
    dtype: jax.typing.DTypeLike = jnp.float32

    @nn.compact
    def __call__(self, obs: jax.Array | NDArray[np.uint8]) -> jax.Array:
        chex.assert_rank(obs, {4, 5})
        chex.assert_type(obs, jnp.uint8)
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
        x = nn.relu(nn.LayerNorm(name="stem_norm", dtype=self.dtype)(x))
        for index, stage in enumerate(self.stages):
            x = jax.image.resize(
                x,
                (x.shape[0], (x.shape[1] + 1) // 2, (x.shape[2] + 1) // 2, x.shape[-1]),
                method="bilinear",
                antialias=True,
            )
            if stage.project:
                x = nn.Conv(stage.channels, (3, 3), padding="SAME", kernel_init=visual_init, dtype=self.dtype)(x)
            for block in range(stage.blocks):
                x = ResidualBlock(stage.channels, dtype=self.dtype, name=f"stage_{index}_block_{block}")(x)
        x = nn.relu(nn.LayerNorm(name="encoder_norm", dtype=self.dtype)(x))
        # Keep the remaining spatial positions distinct in the projection.
        x = nn.Dense(self.embedding_size, kernel_init=visual_init, dtype=self.dtype)(
            x.reshape((x.shape[0], math.prod(x.shape[1:])))
        )
        x = nn.relu(nn.LayerNorm(name="shared_norm", dtype=self.dtype)(x))
        chex.assert_shape(x, (obs.shape[0], self.embedding_size))
        chex.assert_type(x, self.dtype)
        return x
