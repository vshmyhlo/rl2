"""Flattened-grid projection + Mamba3 or transformer for Karel program generation.

Initialize through __call__(initial, target, tokens), with states [B,H,W,6]
and teacher-forced token IDs [T,B]. The context predicts the first token;
each supplied token predicts the next. Output logits are [T+1,B,len(TOKENS)].
For a complete reference program p ending in m), use p[:-1] as input and p as
labels. Batch with the dedicated <pad> ID and mask padded labels in the loss.

For generation, prefill(initial, target) starts fresh and returns the carry and
first-token logits. Then step(sampled_token, carry) returns next-token logits.
Stop externally at m) or the environment's token limit. The trainer excludes
PAD from sampling and loss probabilities; this model returns raw logits.
The output head starts at zero, giving a uniform prior over the 50 program
tokens after PAD masking. The encoder and backbone retain random initialization.
"""

from typing import Literal

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from rl2.karel import TOKENS
from rl2.mamba3 import Mamba3Stack, Mamba3StackCarry
from rl2.transformer import AttentionImplementation, TransformerStack, TransformerStackCarry

type BackboneType = Literal["mamba3", "transformer"]
type KarelModelCarry = Mamba3StackCarry | TransformerStackCarry
type KarelModelOutput = tuple[KarelModelCarry, jax.Array]


class KarelProgramModel(nn.Module):
    """A single image-pair prefix followed by autoregressive program tokens.

    Spatial dimensions must match initialization because the paired grids are
    flattened before a single linear projection. Set max_markers to match the environment.
    dtype controls compute and transformer KV storage; parameters stay float32.
    Logits are always float32 for sampling and loss arithmetic. cuDNN attention
    requires a transformer, reduced-precision dtype, and a supported NVIDIA GPU.
    """

    d_model: int = 256
    num_layers: int = 4
    d_state: int = 64
    headdim: int = 64
    max_markers: int = 10
    backbone_type: BackboneType = "mamba3"
    num_heads: int = 8
    num_kv_heads: int | None = None
    max_seq_len: int = 128  # Transformer window, including the image-pair prefix.
    dtype: jax.typing.DTypeLike = jnp.float32
    attention_implementation: AttentionImplementation = "xla"

    def setup(self) -> None:
        if self.backbone_type not in ("mamba3", "transformer"):
            raise ValueError("backbone_type must be 'mamba3' or 'transformer'")
        if self.attention_implementation not in ("xla", "cudnn"):
            raise ValueError("attention_implementation must be 'xla' or 'cudnn'")
        if self.attention_implementation == "cudnn" and self.backbone_type != "transformer":
            raise ValueError("cuDNN attention requires the transformer backbone")
        chex.assert_type(self.max_markers, int)
        chex.assert_scalar_positive(self.max_markers)
        self.context_projection = nn.Dense(self.d_model, dtype=self.dtype)
        self.context_norm = nn.LayerNorm(dtype=self.dtype)
        self.token_embedding = nn.Embed(len(TOKENS), self.d_model, dtype=self.dtype)
        if self.backbone_type == "mamba3":
            self.backbone = Mamba3Stack(
                d_model=self.d_model,
                num_layers=self.num_layers,
                d_state=self.d_state,
                headdim=self.headdim,
                dtype=self.dtype,
            )
        else:
            self.backbone = TransformerStack(
                d_model=self.d_model,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                max_seq_len=self.max_seq_len,
                dtype=self.dtype,
                attention_implementation=self.attention_implementation,
            )
        # Equal logits give an exactly uniform initial policy after PAD masking.
        # The head learns first; gradients reach the backbone once it is nonzero.
        self.head = nn.Dense(
            len(TOKENS),
            dtype=self.dtype,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
        )

    def encode_pair(self, initial: jax.Array, target: jax.Array) -> jax.Array:
        """Normalize, flatten [B,H,W,12], then linearly project and LayerNorm to [B,D]."""
        chex.assert_shape(initial, (None, None, None, 6))
        chex.assert_equal_shape((initial, target))
        chex.assert_type((initial, target), jnp.int32)
        for size in initial.shape[:3]:
            chex.assert_scalar_positive(size)
        scale = jnp.asarray([1, 1, 1, 1, 1, self.max_markers] * 2, dtype=jnp.float32)
        x = jnp.concatenate((initial, target), axis=-1).astype(jnp.float32) / scale
        x = self.context_projection(x.reshape((x.shape[0], -1)))
        return self.context_norm(x)

    def __call__(self, initial: jax.Array, target: jax.Array, tokens: jax.Array) -> KarelModelOutput:
        """Teacher forcing: [context, embed(p[:-1])] predicts complete program p."""
        context = self.encode_pair(initial, target)
        chex.assert_shape(tokens, (None, context.shape[0]))
        chex.assert_type(tokens, jnp.int32)
        inputs = jnp.concatenate((context[None], self.token_embedding(tokens)), axis=0)
        carry, features = self.backbone(inputs)
        return carry, self.head(features).astype(jnp.float32)

    def prefill(self, initial: jax.Array, target: jax.Array) -> KarelModelOutput:
        """Reset history, encode the pair once, and predict the first token [B,V]."""
        chex.assert_shape(initial, (None, None, None, 6))
        tokens = jnp.empty((0, initial.shape[0]), dtype=jnp.int32)
        carry, logits = self(initial, target, tokens)
        return carry, logits[0]

    def step(self, token: jax.Array, carry: KarelModelCarry) -> KarelModelOutput:
        """Consume one previously generated token [B] and predict the next [B,V]."""
        chex.assert_rank(token, 1)
        chex.assert_type(token, jnp.int32)
        if carry is None:
            raise ValueError("Use prefill() to condition on a pair before step()")
        carry, features = self.backbone.step(self.token_embedding(token), carry)
        return carry, self.head(features).astype(jnp.float32)
