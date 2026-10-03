"""AST-aware Transformer for parallel typed-hole expansions, conditioned on a grid pair.

The entire current tree is re-encoded each round with bidirectional attention.
Each layer/head adds learned tree-relation, distance, and relative-depth biases
to attention scores, derived from the existing preorder depths.
Only the present partial tree is visible, including unresolved sibling holes;
there is no causal action-history cache and no access to future expansions.
Constructor/value heads predict an expansion at every current hole in parallel.
The transition system supplies the legal-action mask; the network cannot bypass it.
"""

import math

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from rl2.karel_ast import AST_ACTIONS, CONSTRUCTORS, NUM_NODE_TYPES, VALUES, ASTFeatures, Field
from rl2.karel_syntax import MAX_BLOCK_DEPTH
from rl2.transformer import AttentionImplementation
from rl2.tree_attention import TreeAttentionBias, TreeRelations, tree_relations


class _ASTBlock(nn.Module):
    d_model: int
    num_heads: int
    num_kv_heads: int
    d_intermediate: int
    num_layers: int
    dtype: jax.typing.DTypeLike
    attention_implementation: AttentionImplementation

    @nn.compact
    def __call__(self, x: jax.Array, present: jax.Array, relations: TreeRelations) -> jax.Array:
        chex.assert_shape(x, (None, None, self.d_model))
        chex.assert_type(x, jnp.floating)
        chex.assert_shape(present, x.shape[:2])
        chex.assert_type(present, jnp.bool_)
        batch, length, _ = x.shape
        chex.assert_shape(relations, (batch, length, length))
        bias = TreeAttentionBias(self.num_heads, name="tree_bias")(relations)
        head_dim = self.d_model // self.num_heads
        normalized = nn.RMSNorm(dtype=self.dtype, name="attention_norm")(x)
        query = nn.Dense(self.d_model, use_bias=False, dtype=self.dtype, name="query")(normalized)
        key = nn.Dense(self.num_kv_heads * head_dim, use_bias=False, dtype=self.dtype, name="key")(normalized)
        value = nn.Dense(self.num_kv_heads * head_dim, use_bias=False, dtype=self.dtype, name="value")(normalized)
        query = query.reshape(batch, length, self.num_heads, head_dim)
        key, value = (v.reshape(batch, length, self.num_kv_heads, head_dim) for v in (key, value))
        # Mask keys, not queries: padded queries still see the always-present pair
        # prefix, avoiding an all-masked softmax. Their outputs never become keys.
        mask = jnp.broadcast_to(present[:, None, None, :], (batch, 1, length, length))
        if self.attention_implementation == "cudnn" and length % 2:
            query, key, value = (jnp.pad(v, ((0, 0), (0, 1), (0, 0), (0, 0))) for v in (query, key, value))
            mask = jnp.pad(mask, ((0, 0), (0, 0), (0, 1), (0, 1)))
            mask = mask.at[:, :, -1, 0].set(True)
            bias = jnp.pad(bias, ((0, 0), (0, 0), (0, 1), (0, 1)))
        attention_dtype = jnp.float32 if self.attention_implementation == "xla" else self.dtype
        attended = (
            jax.nn.dot_product_attention(
                query.astype(attention_dtype),
                key.astype(attention_dtype),
                value.astype(attention_dtype),
                bias=bias.astype(attention_dtype),
                mask=mask,
                is_causal=False,
                implementation=self.attention_implementation,
            )[:, :length]
            .reshape(batch, length, self.d_model)
            .astype(self.dtype)
        )
        out_init = nn.initializers.variance_scaling(1 / (2 * self.num_layers), "fan_in", "truncated_normal")
        attended = nn.Dense(self.d_model, use_bias=False, dtype=self.dtype, kernel_init=out_init, name="attention_out")(
            attended
        )
        x = x.astype(jnp.float32) + attended.astype(jnp.float32)
        normalized = nn.RMSNorm(dtype=self.dtype, name="mlp_norm")(x)
        gate = nn.Dense(self.d_intermediate, use_bias=False, dtype=self.dtype, name="gate")(normalized)
        up = nn.Dense(self.d_intermediate, use_bias=False, dtype=self.dtype, name="up")(normalized)
        hidden = nn.silu(gate) * up
        hidden = nn.Dense(self.d_model, use_bias=False, dtype=self.dtype, kernel_init=out_init, name="down")(hidden)
        return x + hidden.astype(jnp.float32)


class ASTTransformer(nn.Module):
    """One position per AST node, plus a flattened initial/target grid prefix.

    Inputs are [B,H,W,6] int32 grids and batched ASTFeatures with N=max_nodes.
    Returns [B,N,len(AST_ACTIONS)] float32, grammar-masked logits. PAD is excluded
    from active decisions. Non-hole positions use constant dummy PAD logits;
    callers stop expansion when the AST has no holes.
    """

    d_model: int = 320
    num_layers: int = 7
    num_heads: int = 5
    num_kv_heads: int | None = None
    max_nodes: int = 128
    max_depth: int = 64
    max_markers: int = 10
    dtype: jax.typing.DTypeLike = jnp.float32
    attention_implementation: AttentionImplementation = "xla"

    def setup(self) -> None:
        for name in ("d_model", "num_layers", "num_heads", "max_nodes", "max_markers"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer")
            chex.assert_scalar_positive(getattr(self, name))
        if type(self.max_depth) is not int:
            raise TypeError("max_depth must be an integer")
        chex.assert_scalar_in(self.max_depth, 0, MAX_BLOCK_DEPTH)
        kv_heads = self.num_heads if self.num_kv_heads is None else self.num_kv_heads
        if type(kv_heads) is not int:
            raise TypeError("num_kv_heads must be an integer")
        chex.assert_scalar_positive(kv_heads)
        chex.assert_is_divisible(self.d_model, self.num_heads)
        chex.assert_is_divisible(self.num_heads, kv_heads)
        if jnp.dtype(self.dtype) not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
            raise ValueError("dtype must be float32, bfloat16, or float16")
        if self.attention_implementation not in ("xla", "cudnn"):
            raise ValueError("attention_implementation must be 'xla' or 'cudnn'")
        if self.attention_implementation == "cudnn":
            if jnp.dtype(self.dtype) == jnp.float32:
                raise ValueError("cuDNN requires BF16 or float16")
            chex.assert_is_divisible(self.d_model // self.num_heads, 8)
        self.context_projection = nn.Dense(self.d_model, dtype=self.dtype)
        self.context_norm = nn.LayerNorm(dtype=self.dtype)
        self.node_embedding = nn.Embed(NUM_NODE_TYPES, self.d_model, dtype=self.dtype)
        self.field_embedding = nn.Embed(len(Field), self.d_model, dtype=self.dtype)
        self.depth_embedding = nn.Embed(self.max_depth + 1, self.d_model, dtype=self.dtype)
        self.child_embedding = nn.Embed(3, self.d_model, dtype=self.dtype)
        self.value_embedding = nn.Embed(len(VALUES) + 1, self.d_model, dtype=self.dtype)
        self.hole_embedding = nn.Embed(2, self.d_model, dtype=self.dtype)
        self.position_embedding = nn.Embed(self.max_nodes + 1, self.d_model, dtype=self.dtype)
        width = math.ceil((8 * self.d_model / 3) / 128) * 128
        self.layers = tuple(
            _ASTBlock(
                self.d_model,
                self.num_heads,
                kv_heads,
                width,
                self.num_layers,
                self.dtype,
                self.attention_implementation,
                name=f"layers_{i}",
            )
            for i in range(self.num_layers)
        )
        self.final_norm = nn.RMSNorm(dtype=self.dtype)
        # Initially uniform over the legal productions for each hole type.
        self.constructor_head = nn.Dense(len(CONSTRUCTORS), dtype=self.dtype, kernel_init=nn.initializers.zeros_init())
        self.value_head = nn.Dense(len(VALUES), dtype=self.dtype, kernel_init=nn.initializers.zeros_init())

    def __call__(self, initial: jax.Array, target: jax.Array, tree: ASTFeatures) -> jax.Array:
        """Predict parallel hole expansions from the grid pair and current partial AST.

        Args:
            initial: Int32 starting grids of shape [B, H, W, 6]. Channels are
                robot headings (north, east, south, west), walls, and marker counts.
            target: Int32 goal grids with the same shape and channel layout.
                The pair is concatenated along channels, marker counts are
                normalized by max_markers, and the flattened pair is projected
                into one context token.
            tree: Batched ASTFeatures before the next expansion. Node/type,
                field, depth, child-index, and value IDs are int32 [B, max_nodes];
                node_mask is bool with the same shape; is_hole is computed from
                node_mask, node_type, and value.
                action_mask is bool [B, max_nodes, A], where
                A = len(AST_ACTIONS), and marks legal expansions per hole.

        Returns:
            Float32 logits of shape [B, max_nodes, A], ordered as AST_ACTIONS (PAD, then
            constructors, then values). Illegal actions have logit -inf.
            PAD is excluded on active rows; rows with no legal actions use
            PAD=0 and all other logits=-inf as a safe batching fallback.
        """
        chex.assert_shape(initial, (None, None, None, 6))
        chex.assert_equal_shape((initial, target))
        chex.assert_type((initial, target), jnp.int32)
        for size in initial.shape[:3]:
            chex.assert_scalar_positive(size)
        batch = initial.shape[0]
        chex.assert_shape(tree[:6], (batch, self.max_nodes))
        chex.assert_type(tree[:5], jnp.int32)
        chex.assert_type((tree.is_hole, tree.node_mask, tree.action_mask), jnp.bool_)
        chex.assert_shape(tree.action_mask, (batch, self.max_nodes, len(AST_ACTIONS)))
        scale = jnp.asarray([1, 1, 1, 1, 1, self.max_markers] * 2, jnp.float32)
        pair = jnp.concatenate((initial, target), axis=-1).astype(jnp.float32) / scale
        context = self.context_norm(self.context_projection(pair.reshape(batch, -1)))
        # Canonicalize padding before embedding so even dirty padded IDs cannot
        # affect live nodes or index outside an embedding table.
        ids = [jnp.where(tree.node_mask, feature, 0) for feature in tree[:5]]
        nodes = (
            self.node_embedding(ids[0])
            + self.field_embedding(ids[1])
            + self.depth_embedding(ids[2])
            + self.child_embedding(ids[3])
            + self.value_embedding(ids[4])
            + self.hole_embedding(tree.is_hole.astype(jnp.int32))
        )
        x = jnp.concatenate((context[:, None], nodes), axis=1)
        x = x + self.position_embedding(jnp.arange(self.max_nodes + 1, dtype=jnp.int32))[None]
        present = jnp.concatenate((jnp.ones((batch, 1), jnp.bool_), tree.node_mask), axis=1)
        relations = tree_relations(tree.depth, tree.node_mask)
        for layer in self.layers:
            x = layer(x, present, relations)
        nodes = self.final_norm(x[:, 1:])
        logits = jnp.concatenate(
            (
                jnp.full((batch, self.max_nodes, 1), -jnp.inf, jnp.float32),
                self.constructor_head(nodes).astype(jnp.float32),
                self.value_head(nodes).astype(jnp.float32),
            ),
            axis=-1,
        )
        masked = jnp.where(tree.action_mask, logits, -jnp.inf)
        return masked.at[..., 0].set(jnp.where(tree.action_mask.any(axis=-1), -jnp.inf, 0.0))
