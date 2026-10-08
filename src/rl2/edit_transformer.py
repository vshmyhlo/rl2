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


PAD_EVENT, EDIT_EVENT, FEEDBACK_EVENT = range(3)


class Event(NamedTuple):
    """Time-major history, or one batched event for cached decoding.

    NamedTuple makes this a JAX pytree; tree.map preserves its type and fields.
    T is the number of events, B the batch size, and H/W the grid dimensions.
    For one cached decoding step, omit the leading T dimension from every field.

    Attributes:
        kind: int32 [T,B] event types: PAD_EVENT (0) for trailing padding,
            EDIT_EVENT (1) for AST editing tokens, or FEEDBACK_EVENT (2) for
            execution reports after complete edits, including the initial program.
            Padding does not advance the cache.
        action: int32 [T,B] policy token IDs for EDIT: location, grammar, or STOP.
            The initial program is supplied as forced depth-first grammar EDIT
            tokens. Ignored for FEEDBACK and PAD.
        grid: int32 [T,B,3,H,W,6] images in order: initial, target, current.
            Current is the latest execution result, including where execution
            ended. A completing EDIT retains the previous execution image; the
            following FEEDBACK introduces the new result.
        feedback: float32 [T,B,8] execution feedback in this order: score,
            success, runtime error, execution limit, ticks, length, score delta,
            and sequence tokens left. Used only by FEEDBACK events; ignored for
            EDIT and PAD.
    """

    kind: Array
    action: Array
    grid: Array
    feedback: Array


type ModelOutput = tuple[TransformerStackCarry, jax.Array]


class EditTransformer(nn.Module):
    """Causal EDIT tokens and program FEEDBACK stream with a KV cache.

    __call__ returns (carry, logits [T,B,V]); row t consumes event t and predicts the
    next event. Loss applies only when that next event is a sampled action;
    feedback reports and initial program tokens are supplied by the host and
    are never policy targets.
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
        self.kind_embedding = nn.Embed(3, self.d_model, dtype=self.dtype)
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

    def encode_grids(self, grid: jax.Array) -> jax.Array:
        """Mix [initial, target, current] channels per cell, preserving spatial positions."""
        sc = ShapeChecker(I=3, C=6, P=18, K=32)
        leading = "TB" if grid.ndim == 6 else "B"
        sc.check(grid, leading + "IHWC", jnp.int32)
        for size in sc["HW"]:
            chex.assert_scalar_positive(size)
        scale = jnp.asarray([1, 1, 1, 1, 1, self.max_markers] * 3, jnp.float32)
        grids = jnp.moveaxis(grid, -4, -2).reshape(sc[leading + "HWP"]).astype(jnp.float32) / scale
        sc.check(grids, leading + "HWP", jnp.float32)
        features = nn.gelu(self.grid_conv(grids))
        sc.check(features, leading + "HWK", self.dtype)
        encoded = features.reshape(*sc[leading], math.prod(features.shape[-3:]))
        sc.check(encoded, leading + "G", self.dtype)
        return encoded

    def encode_event(self, event: Event) -> jax.Array:
        """Add the task/current-image embedding to every event's token or feedback embedding."""
        sc = ShapeChecker(I=3, C=6, F=FEEDBACK_SIZE, D=self.d_model)
        sc.check((event.kind, event.action), "TB", jnp.int32)
        sc.check(event.grid, "TBIHWC", jnp.int32)
        sc.check(event.feedback, "TBF", jnp.float32)
        active = event.kind != PAD_EVENT
        has_token = event.kind == EDIT_EVENT
        has_feedback = event.kind == FEEDBACK_EVENT
        # Mask unused fields before nonlinear operations, so ignored NaNs and
        # out-of-range embedding IDs cannot contaminate parameter gradients.
        feedback = jnp.where(has_feedback[..., None], event.feedback, 0)
        token_ids = jnp.where(has_token, event.action, 0)
        grid = jnp.where(active[..., None, None, None, None], event.grid, 0)
        grids = self.encode_grids(grid)
        sc.check(grids, "TBG", self.dtype)
        context = self.context_norm(self.context_projection(grids))
        update = self.feedback_norm(self.feedback_projection(feedback))
        token = self.token_embedding(token_ids)
        sc.check((context, update, token), "TBD", self.dtype)
        x = jnp.where(has_token[..., None], token, 0)
        x += jnp.where(has_feedback[..., None], update, 0)
        x += context + self.kind_embedding(event.kind)
        encoded = jnp.where(active[..., None], x, 0)
        sc.check(encoded, "TBD", self.dtype)
        return encoded

    def __call__(self, event: Event) -> ModelOutput:
        """Encode the complete causal history and return its cache and next-action logits."""
        sc = ShapeChecker(C=6, D=self.d_model, V=1 + self.max_nodes + len(AST_ACTIONS))
        inputs = self.encode_event(event)
        sc.check(inputs, "TBD", self.dtype)
        active = event.kind != PAD_EVENT
        x_len = jnp.sum(active, axis=0, dtype=jnp.int32)
        sc.check(x_len, "B", jnp.int32)
        carry, features = self.backbone(jnp.swapaxes(inputs, 0, 1), x_len)
        features = jnp.swapaxes(features, 0, 1)
        sc.check(features, "TBD", self.dtype)
        logits = self.head(features).astype(jnp.float32)
        logits = jnp.where(active[..., None], logits, 0)
        sc.check(logits, "TBV", jnp.float32)
        return carry, logits

    def prefill(self, event: Event) -> ModelOutput:
        """Consume the initial program EDIT tokens and its FEEDBACK report once."""
        carry, logits = self(event)
        lengths = jnp.sum(event.kind != PAD_EVENT, axis=0, dtype=jnp.int32)
        last = jnp.maximum(lengths - 1, 0)
        return carry, logits[last, jnp.arange(logits.shape[1])]

    def step(self, event: Event, carry: TransformerStackCarry) -> ModelOutput:
        """Append an event with its three images; PAD preserves the KV cache and yields zero logits."""
        sc = ShapeChecker(I=3, C=6, F=FEEDBACK_SIZE, D=self.d_model, V=1 + self.max_nodes + len(AST_ACTIONS))
        sc.check((event.kind, event.action), "B", jnp.int32)
        sc.check(event.grid, "BIHWC", jnp.int32)
        sc.check(event.feedback, "BF", jnp.float32)
        sequence = jax.tree.map(partial(jnp.expand_dims, axis=0), event)
        x_active = event.kind != PAD_EVENT
        sc.check(x_active, "B", jnp.bool_)
        carry, features = self.backbone.step(self.encode_event(sequence)[0], x_active, carry)
        sc.check(features, "BD", self.dtype)
        logits = self.head(features).astype(jnp.float32)
        logits = jnp.where(x_active[:, None], logits, 0)
        sc.check(logits, "BV", jnp.float32)
        return carry, logits
