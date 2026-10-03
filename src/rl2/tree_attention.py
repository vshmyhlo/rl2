"""Relative attention biases derived from the preorder depths of a rooted tree."""

from enum import IntEnum
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn


class Relation(IntEnum):
    """Relationship of the key node to the query node."""

    SELF = 0
    PARENT = 1
    CHILD = 2
    ANCESTOR = 3
    DESCENDANT = 4
    SIBLING = 5
    OTHER = 6
    CONTEXT_TO_NODE = 7
    NODE_TO_CONTEXT = 8
    CONTEXT_SELF = 9
    PADDING = 10


MAX_DISTANCE = 16
MAX_RELATIVE_DEPTH = 16


class TreeRelations(NamedTuple):
    """Int32 [B, N+1, N+1] lookup indices, including the context prefix.

    Axes are batch, query, key. Distances saturate at 16; signed depth
    differences saturate at +/-16 and are shifted by 16 for table lookup.
    Only actual AST pairs use distance/depth biases; padding has no bias.
    """

    kind: jax.Array
    distance: jax.Array
    relative_depth: jax.Array


def tree_relations(depth: jax.Array, node_mask: jax.Array) -> TreeRelations:
    """Compute relations for valid rooted-tree preorder input in O(B*N^2).

    Depths count AST edges, including list nodes and holes. Live nodes form a
    contiguous prefix; masked depths may contain arbitrary values. Relations
    are recomputed once per forward pass and shared by all attention layers.
    """
    chex.assert_rank(depth, 2)
    chex.assert_equal_shape((depth, node_mask))
    chex.assert_type(depth, jnp.int32)
    chex.assert_type(node_mask, jnp.bool_)
    batch, size = depth.shape
    chex.assert_scalar_positive(size)
    depth = jnp.where(node_mask, depth, 0)
    index = jnp.arange(size)
    # For i < j, the LCA depth is min(depth[i], min(depth[i+1:j+1])-1).
    # Preorder can rise by only one edge at a time; crossing out of a subtree
    # visits a child of the shared ancestor, hence the subtraction by one.
    interval = jnp.where(index[None, :] > index[:, None], depth[:, None, :], size)
    minima = jax.lax.associative_scan(jnp.minimum, interval, axis=-1)
    upper = jnp.minimum(depth[:, :, None], minima - 1)
    lca = jnp.where(index[None, :] > index[:, None], upper, jnp.swapaxes(upper, -1, -2))
    lca = jnp.where(index[None, :] == index[:, None], depth[:, :, None], lca)
    up = depth[:, :, None] - lca
    down = depth[:, None, :] - lca
    kind = jnp.full((batch, size, size), Relation.OTHER, jnp.int32)
    kind = jnp.where((up == 1) & (down == 1), Relation.SIBLING, kind)
    kind = jnp.where(up == 0, jnp.where(down == 1, Relation.CHILD, Relation.DESCENDANT), kind)
    kind = jnp.where(down == 0, jnp.where(up == 1, Relation.PARENT, Relation.ANCESTOR), kind)
    kind = jnp.where((up == 0) & (down == 0), Relation.SELF, kind)
    kind = jnp.pad(kind, ((0, 0), (1, 0), (1, 0)))
    kind = kind.at[:, 0, :].set(Relation.CONTEXT_TO_NODE)
    kind = kind.at[:, :, 0].set(Relation.NODE_TO_CONTEXT)
    kind = kind.at[:, 0, 0].set(Relation.CONTEXT_SELF)
    present = jnp.concatenate((jnp.ones((batch, 1), jnp.bool_), node_mask), axis=-1)
    kind = jnp.where(present[:, :, None] & present[:, None, :], kind, Relation.PADDING)
    distance = jnp.clip(up + down, 0, MAX_DISTANCE)
    relative_depth = jnp.clip(depth[:, None, :] - depth[:, :, None], -MAX_RELATIVE_DEPTH, MAX_RELATIVE_DEPTH)
    padding = ((0, 0), (1, 0), (1, 0))
    return TreeRelations(kind, jnp.pad(distance, padding), jnp.pad(relative_depth + MAX_RELATIVE_DEPTH, padding))


class TreeAttentionBias(nn.Module):
    """Per-head additive logit bias, zero-initialized independently per layer."""

    num_heads: int

    @nn.compact
    def __call__(self, relations: TreeRelations) -> jax.Array:
        """Return float32 [B, H, N+1, N+1] biases before the attention softmax."""
        chex.assert_scalar_positive(self.num_heads)
        chex.assert_rank(relations, 3)
        chex.assert_equal_shape(relations)
        chex.assert_type(relations, jnp.int32)
        chex.assert_equal(relations.kind.shape[1], relations.kind.shape[2])
        kind = self.param("relation", nn.initializers.zeros_init(), (len(Relation), self.num_heads))
        distance = self.param("distance", nn.initializers.zeros_init(), (MAX_DISTANCE + 1, self.num_heads))
        depth = self.param("relative_depth", nn.initializers.zeros_init(), (2 * MAX_RELATIVE_DEPTH + 1, self.num_heads))
        geometry = distance[relations.distance] + depth[relations.relative_depth]
        bias = kind[relations.kind] + jnp.where((relations.kind <= Relation.OTHER)[..., None], geometry, 0)
        bias = jnp.where((relations.kind != Relation.PADDING)[..., None], bias, 0)
        return jnp.moveaxis(bias, -1, 1)
