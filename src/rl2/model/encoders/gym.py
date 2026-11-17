"""Encoders for Gymnasium Box, Dict, and Tuple observations."""

import jax
import jax.numpy as jnp
from flax import linen as nn

from rl2.envs.gym_env import GymObservation
from rl2.model.encoders.conv2d import ConvObservationEncoder, ConvStages
from rl2.model.encoders.mlp import MLPEncoder


def _flatten_leaf(leaf: jax.Array) -> jax.Array:  # (B, *O) -> (B, F)
    features = leaf.reshape((leaf.shape[0], -1)).astype(jnp.float32)  # (B, F)
    return features / 255.0 if leaf.dtype == jnp.uint8 else features  # (B, F)


class GymMLPEncoder(nn.Module):
    """Flatten observation leaves and encode them with an MLP."""

    hidden_size: int
    embedding_size: int
    dtype: jax.typing.DTypeLike | None = None

    @nn.compact
    def __call__(self, obs: GymObservation[jax.Array]) -> jax.Array:  # leaves: (B, *O) -> (B, E)
        features = jax.tree.map(_flatten_leaf, obs)  # leaves: (B, F)
        combined = jnp.concatenate(jax.tree.leaves(features), axis=-1)  # (B, sum(F))
        return MLPEncoder(hidden_size=self.hidden_size, embedding_size=self.embedding_size, dtype=self.dtype)(
            combined
        )  # (B, E)


class GymConvEncoder(nn.Module):
    """Encode HWC image leaves with CNNs and fuse them with vector features."""

    stages: ConvStages
    embedding_size: int
    dtype: jax.typing.DTypeLike | None = None

    @nn.compact
    def __call__(self, obs: GymObservation[jax.Array]) -> jax.Array:  # leaves: (B, *O) -> (B, E)
        features = []  # each (B, F)
        for index, leaf in enumerate(jax.tree.leaves(obs)):
            if leaf.ndim == 4:
                feature = ConvObservationEncoder(
                    channels_last=True,
                    stages=self.stages,
                    embedding_size=self.embedding_size,
                    dtype=self.dtype,
                    name=f"image_{index}",
                )(leaf)  # (B, E)
            else:
                feature = _flatten_leaf(leaf)  # (B, F)
            features.append(feature)
        combined = jnp.concatenate(features, axis=-1)  # (B, sum(F))
        return MLPEncoder(hidden_size=self.embedding_size, embedding_size=self.embedding_size, dtype=self.dtype)(
            combined
        )  # (B, E)
