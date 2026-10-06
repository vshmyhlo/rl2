"""AST-aware Transformer for parallel typed-hole expansions, conditioned on a grid pair.

The entire current tree is re-encoded each round with a shared BDTransformer.
It uses learned absolute positions with RoPE disabled, standard backbone
projections (no Q/K normalization), and one final RMSNorm inside the stack.
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

from rl2.attention import AttentionType
from rl2.karel_ast import AST_ACTIONS, CONSTRUCTORS, NUM_NODE_TYPES, VALUES, ASTFeatures, Field
from rl2.karel_syntax import MAX_BLOCK_DEPTH
from rl2.shape_checker import ShapeChecker
from rl2.transformer import BDTransformer
from rl2.tree_attention import TreeAttentionBias, tree_relations


class ASTTransformer(nn.Module):
    """One position per AST node, plus a flattened initial/target grid prefix.

    Inputs are [B,H,W,6] int32 grids and batched ASTFeatures with N<=max_nodes.
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
    attention_implementation: AttentionType = "xla"

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
        self.tree_biases = tuple(
            TreeAttentionBias(self.num_heads, name=f"tree_bias_{i}") for i in range(self.num_layers)
        )
        self.backbone = BDTransformer(
            dim=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=kv_heads,
            max_seq_len=self.max_nodes + 1,
            mlp_expansion=width / self.d_model,
            norm_epsilon=1e-6,
            use_rope=False,
            dtype=self.dtype,
            attention_implementation=self.attention_implementation,
        )
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
                field, depth, child-index, and value IDs are int32 [B, N], with
                1 <= N <= max_nodes (a trimmed sequence bucket);
                node_mask is bool with the same shape; is_hole is computed from
                node_mask, node_type, and value.
                action_mask is bool [B, N, A], where
                A = len(AST_ACTIONS), and marks legal expansions per hole.

        Returns:
            Float32 logits of shape [B, N, A], ordered as AST_ACTIONS (PAD, then
            constructors, then values). Illegal actions have logit -inf.
            PAD is excluded on active rows; rows with no legal actions use
            PAD=0 and all other logits=-inf as a safe batching fallback.
        """
        sc = ShapeChecker(C=6, D=self.d_model, A=len(AST_ACTIONS))
        sc.check((initial, target), "BHWC", jnp.int32)
        for size in initial.shape[:3]:
            chex.assert_scalar_positive(size)
        batch = initial.shape[0]
        sc.check(tree.node_mask, "BN", jnp.bool_)
        nodes_count = tree.node_mask.shape[1]
        chex.assert_scalar_in(nodes_count, 1, self.max_nodes)
        sc.check(tree[:5], "BN", jnp.int32)
        sc.check(tree.is_hole, "BN", jnp.bool_)
        sc.check(tree.action_mask, "BNA", jnp.bool_)
        scale = jnp.asarray([1, 1, 1, 1, 1, self.max_markers] * 2, jnp.float32)
        pair = jnp.concatenate((initial, target), axis=-1).astype(jnp.float32) / scale
        sc.check(pair, "BHWP", jnp.float32)
        context = self.context_norm(self.context_projection(pair.reshape(batch, -1)))
        sc.check(context, "BD", self.dtype)
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
        sc.check(nodes, "BND", self.dtype)
        x = jnp.concatenate((context[:, None], nodes), axis=1)
        x = x + self.position_embedding(jnp.arange(nodes_count + 1, dtype=jnp.int32))[None]
        # Start the shared residual stream in float32, as in the former AST blocks.
        x = x.astype(jnp.float32)
        sc.check(x, "BTD", jnp.float32)
        lengths = 1 + tree.node_mask.sum(axis=-1, dtype=jnp.int32)
        sc.check(lengths, "B", jnp.int32)
        relations = tree_relations(tree.depth, tree.node_mask)
        sc.check(relations, "BTT", jnp.int32)
        biases = tuple(tree_bias(relations) for tree_bias in self.tree_biases)
        bias_sc = ShapeChecker(B=batch, H=self.num_heads, T=nodes_count + 1)
        bias_sc.check(biases, "BHTT", jnp.float32)
        x = self.backbone(x, lengths, biases=biases)
        sc.check(x, "BTD", self.dtype)
        nodes = x[:, 1:]
        sc.check(nodes, "BND", self.dtype)
        logits = jnp.concatenate(
            (
                jnp.full((batch, nodes_count, 1), -jnp.inf, jnp.float32),
                self.constructor_head(nodes).astype(jnp.float32),
                self.value_head(nodes).astype(jnp.float32),
            ),
            axis=-1,
        )
        sc.check(logits, "BNA", jnp.float32)
        masked = jnp.where(tree.action_mask, logits, -jnp.inf)
        output = masked.at[..., 0].set(jnp.where(tree.action_mask.any(axis=-1), -jnp.inf, 0.0))
        sc.check(output, "BNA", jnp.float32)
        return output
