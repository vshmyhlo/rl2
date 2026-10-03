"""Minimal CNN + Mamba3 model for Karel program generation.

Initialize through __call__(initial, target, tokens), with states [B,H,W,6]
and teacher-forced token IDs [T,B]. The context predicts the first token;
each supplied token predicts the next. Output logits are [T+1,B,len(TOKENS)].
For a reference program p without EOS, use p as input and p + [EOS] as labels.
Mask padded labels in the training loss.

For generation, prefill(initial, target) starts fresh and returns the carry and
first-token logits. Then step(sampled_token, carry) returns next-token logits.
Stop externally at EOS or the environment's token limit.
"""

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from rl2.karel import TOKENS
from rl2.mamba3 import Mamba3Stack, Mamba3StackCarry

type KarelModelOutput = tuple[Mamba3StackCarry, jax.Array]


class KarelProgramModel(nn.Module):
    """A single image-pair prefix followed by autoregressive program tokens.

    Spatial dimensions must match initialization because the CNN features are
    flattened before projection. Set max_markers to match the environment.
    """

    d_model: int = 256
    num_layers: int = 4
    d_state: int = 64
    headdim: int = 64
    conv_channels: tuple[int, ...] = (32, 64, 64)
    max_markers: int = 10

    def setup(self) -> None:
        chex.assert_type(self.max_markers, int)
        chex.assert_scalar_positive(self.max_markers)
        chex.assert_scalar_positive(len(self.conv_channels))
        for channels in self.conv_channels:
            chex.assert_type(channels, int)
            chex.assert_scalar_positive(channels)
        self.convs = tuple(
            nn.Conv(channels, (3, 3), padding="SAME", name=f"conv_{index}")
            for index, channels in enumerate(self.conv_channels)
        )
        self.context_projection = nn.Dense(self.d_model)
        self.context_norm = nn.LayerNorm()
        self.token_embedding = nn.Embed(len(TOKENS), self.d_model)
        self.backbone = Mamba3Stack(
            d_model=self.d_model,
            num_layers=self.num_layers,
            d_state=self.d_state,
            headdim=self.headdim,
        )
        self.head = nn.Dense(len(TOKENS))

    def encode_pair(self, initial: jax.Array, target: jax.Array) -> jax.Array:
        """Encode aligned initial/target grids into one [B,D] context token."""
        chex.assert_shape(initial, (None, None, None, 6))
        chex.assert_equal_shape((initial, target))
        chex.assert_type((initial, target), jnp.int32)
        for size in initial.shape[:3]:
            chex.assert_scalar_positive(size)
        scale = jnp.asarray([1, 1, 1, 1, 1, self.max_markers] * 2, dtype=jnp.float32)
        x = jnp.concatenate((initial, target), axis=-1).astype(jnp.float32) / scale
        for conv in self.convs:
            x = nn.silu(conv(x))
        x = self.context_projection(x.reshape((x.shape[0], -1)))
        return self.context_norm(x)

    def __call__(self, initial: jax.Array, target: jax.Array, tokens: jax.Array) -> KarelModelOutput:
        """Teacher forcing: [context, embed(tokens)] predicts [tokens, EOS]."""
        context = self.encode_pair(initial, target)
        chex.assert_shape(tokens, (None, context.shape[0]))
        chex.assert_type(tokens, jnp.int32)
        inputs = jnp.concatenate((context[None], self.token_embedding(tokens)), axis=0)
        carry, features = self.backbone(inputs)
        return carry, self.head(features)

    def prefill(self, initial: jax.Array, target: jax.Array) -> KarelModelOutput:
        """Reset history, encode the pair once, and predict the first token [B,V]."""
        chex.assert_shape(initial, (None, None, None, 6))
        tokens = jnp.empty((0, initial.shape[0]), dtype=jnp.int32)
        carry, logits = self(initial, target, tokens)
        return carry, logits[0]

    def step(self, token: jax.Array, carry: Mamba3StackCarry) -> KarelModelOutput:
        """Consume one previously generated token [B] and predict the next [B,V]."""
        chex.assert_rank(token, 1)
        chex.assert_type(token, jnp.int32)
        if carry is None:
            raise ValueError("Use prefill() to condition on a pair before step()")
        carry, features = self.backbone.step(self.token_embedding(token), carry)
        return carry, self.head(features)
