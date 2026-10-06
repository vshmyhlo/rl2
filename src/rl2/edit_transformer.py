"""Causal Transformer for Karel AST editing with cached event-history decoding."""

import math
from functools import partial
from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from numpy.typing import NDArray

from rl2.attention import AttentionType
from rl2.karel_ast import AST_ACTIONS
from rl2.karel_ast_edit import FEEDBACK_SIZE
from rl2.shape_checker import ShapeChecker
from rl2.transformer import ARTransformer, TransformerStackCarry

type Array = jax.Array | NDArray[Any]


PAD_EVENT, SEED_EVENT, ACTION_EVENT, UPDATE_EVENT = range(4)


class Events(NamedTuple):
    """Time-major history, or one batched event for cached decoding.

    NamedTuple makes this a JAX pytree; tree.map preserves its type and fields.
    kind/value: int32 [T,B] (or [B]); output: int32 [T,B,H,W,6];
    feedback: float32 [T,B,8]. Every event carries the latest execution image;
    ACTION events carry resulting feedback scalars; UPDATE is used only for the initial seed report.
    SEED values are the initial program's DFS grammar actions in policy IDs.
    ACTION values are sampled location/grammar/STOP IDs. PAD is trailing only
    and does not advance the cache. Values on UPDATE/PAD and feedback on
    SEED/PAD are ignored.
    """

    kind: Array
    value: Array
    output: Array
    feedback: Array


class History(NamedTuple):
    initial: Array  # [B,H,W,6], supplied to every token's image encoder.
    target: Array
    events: Events


class EditCarry(NamedTuple):
    """KV caches and fixed task grids used alongside every event's current image."""

    transformer: TransformerStackCarry
    initial: Array
    target: Array


type ModelOutput = tuple[EditCarry, jax.Array]


class EditTransformer(nn.Module):
    """Causal seed -> initial update -> action/result stream with a KV cache.

    __call__ returns (carry, logits [T,B,V]); row t consumes event t and predicts the
    next event. Loss applies only when that next event is a sampled action;
    updates and seed tokens are provided by the host and are never targets.
    No AST encoder or tree-relative attention is used. Location IDs refer to the
    current host AST's preorder, reconstructed by applying the preceding edits.
    Padding produces zero logits. Prefill selects the last real event per
    example, returning zero for an entirely padded example.
    """

    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 8
    num_kv_heads: int | None = None
    max_nodes: int = 64
    max_seq_len: int = 256
    max_markers: int = 10
    dtype: jax.typing.DTypeLike = jnp.float32
    attention_implementation: AttentionType = "xla"

    def setup(self) -> None:
        """Create task and feedback encoders, event embeddings, and the causal policy."""
        for value in (self.max_nodes, self.max_markers):
            chex.assert_type(value, int)
            chex.assert_scalar_positive(value)
        self.grid_conv = nn.Conv(32, kernel_size=(1, 1), dtype=self.dtype)
        self.context_projection = nn.Dense(self.d_model, dtype=self.dtype)
        self.context_norm = nn.LayerNorm(dtype=self.dtype)
        self.feedback_projection = nn.Dense(self.d_model, dtype=self.dtype)
        self.feedback_norm = nn.LayerNorm(dtype=self.dtype)
        self.token_embedding = nn.Embed(1 + self.max_nodes + len(AST_ACTIONS), self.d_model, dtype=self.dtype)
        self.kind_embedding = nn.Embed(4, self.d_model, dtype=self.dtype)
        self.backbone = ARTransformer(
            dim=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
            attention_implementation=self.attention_implementation,
        )
        self.head = nn.Dense(
            1 + self.max_nodes + len(AST_ACTIONS), dtype=self.dtype, kernel_init=nn.initializers.zeros_init()
        )

    def encode_grids(self, initial: jax.Array, target: jax.Array, output: jax.Array) -> jax.Array:
        """Mix normalized input/target/result channels per cell, preserving spatial positions."""
        chex.assert_scalar_non_negative(initial.ndim - 4)
        chex.assert_shape(initial, (*initial.shape[:-3], None, None, 6))
        chex.assert_equal_shape((initial, target, output))
        chex.assert_type((initial, target, output), jnp.int32)
        for size in initial.shape[-3:-1]:
            chex.assert_scalar_positive(size)
        scale = jnp.asarray([1, 1, 1, 1, 1, self.max_markers] * 3, jnp.float32)
        grids = jnp.concatenate((initial, target, output), axis=-1).astype(jnp.float32) / scale
        features = nn.gelu(self.grid_conv(grids))
        return features.reshape(*initial.shape[:-3], math.prod(features.shape[-3:]))

    def encode_events(self, events: Events, initial: jax.Array, target: jax.Array) -> jax.Array:
        """Add the task/current-image embedding to every event's token or feedback embedding."""
        sc = ShapeChecker(C=6, F=FEEDBACK_SIZE, D=self.d_model)
        sc.check((initial, target), "BHWC", jnp.int32)
        sc.check((events.kind, events.value), "TB", jnp.int32)
        sc.check(events.output, "TBHWC", jnp.int32)
        sc.check(events.feedback, "TBF", jnp.float32)
        active = events.kind != PAD_EVENT
        has_token = (events.kind == SEED_EVENT) | (events.kind == ACTION_EVENT)
        has_feedback = (events.kind == ACTION_EVENT) | (events.kind == UPDATE_EVENT)
        # Mask unused fields before nonlinear operations, so ignored NaNs and
        # out-of-range embedding IDs cannot contaminate parameter gradients.
        feedback = jnp.where(has_feedback[..., None], events.feedback, 0)
        token_ids = jnp.where(has_token, events.value, 0)
        output = jnp.where(active[..., None, None, None], events.output, 0)
        grids = self.encode_grids(
            jnp.broadcast_to(initial, events.output.shape),
            jnp.broadcast_to(target, events.output.shape),
            output,
        )
        sc.check(grids, "TBG", self.dtype)
        context = self.context_norm(self.context_projection(grids))
        update = self.feedback_norm(self.feedback_projection(feedback))
        token = self.token_embedding(token_ids)
        sc.check((context, update, token), "TBD", self.dtype)
        x = jnp.where(has_token[..., None], token, 0)
        x += jnp.where(has_feedback[..., None], update, 0)
        x += context + self.kind_embedding(events.kind)
        encoded = jnp.where(active[..., None], x, 0)
        sc.check(encoded, "TBD", self.dtype)
        return encoded

    def __call__(self, history: History) -> ModelOutput:
        """Encode the complete causal history and return its cache and next-action logits."""
        sc = ShapeChecker(C=6, D=self.d_model, V=1 + self.max_nodes + len(AST_ACTIONS))
        sc.check((history.initial, history.target), "BHWC", jnp.int32)
        inputs = self.encode_events(history.events, history.initial, history.target)
        sc.check(inputs, "TBD", self.dtype)
        active = history.events.kind != PAD_EVENT
        x_len = jnp.sum(active, axis=0, dtype=jnp.int32)
        sc.check(x_len, "B", jnp.int32)
        carry, features = self.backbone(jnp.swapaxes(inputs, 0, 1), x_len)
        features = jnp.swapaxes(features, 0, 1)
        sc.check(features, "TBD", self.dtype)
        logits = self.head(features).astype(jnp.float32)
        logits = jnp.where(active[..., None], logits, 0)
        sc.check(logits, "TBV", jnp.float32)
        return EditCarry(carry, history.initial, history.target), logits

    def prefill(self, history: History) -> ModelOutput:
        """Consume the seed program and initial execution update once."""
        carry, logits = self(history)
        lengths = jnp.sum(history.events.kind != PAD_EVENT, axis=0, dtype=jnp.int32)
        last = jnp.maximum(lengths - 1, 0)
        return carry, logits[last, jnp.arange(logits.shape[1])]

    def step(self, event: Events, carry: EditCarry) -> ModelOutput:
        """Append each non-PAD event; padding preserves the cache and returns zero logits."""
        sc = ShapeChecker(C=6, F=FEEDBACK_SIZE, D=self.d_model, V=1 + self.max_nodes + len(AST_ACTIONS))
        sc.check((carry.initial, carry.target), "BHWC", jnp.int32)
        sc.check((event.kind, event.value), "B", jnp.int32)
        sc.check(event.output, "BHWC", jnp.int32)
        sc.check(event.feedback, "BF", jnp.float32)
        sequence = jax.tree.map(partial(jnp.expand_dims, axis=0), event)
        x_active = event.kind != PAD_EVENT
        sc.check(x_active, "B", jnp.bool_)
        transformer, features = self.backbone.step(
            self.encode_events(sequence, carry.initial, carry.target)[0], x_active, carry.transformer
        )
        sc.check(features, "BD", self.dtype)
        logits = self.head(features).astype(jnp.float32)
        logits = jnp.where(x_active[:, None], logits, 0)
        sc.check(logits, "BV", jnp.float32)
        return carry._replace(transformer=transformer), logits
