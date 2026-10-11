# Portions adapted from Mctx, Copyright 2021 DeepMind Technologies Limited.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Mctx PUCT with sparse batched backups for large action spaces."""

import jax
import jax.numpy as jnp
import mctx
from mctx._src import search as mctx_search

from rl2.alphazero.search import PolicyValueFn

type SearchTree = mctx.Tree[None]
type BackupState = tuple[SearchTree, jax.Array, jax.Array]
type SearchState = tuple[jax.Array, SearchTree]


def _backward(tree: SearchTree, leaf_index: jax.Array) -> SearchTree:
    """Back up (B,) leaf indices, masking individual updates once a root is reached.

    Mctx vmaps a per-game while loop. Its implicit masking selects entire
    (B, N, A) children arrays at every depth, copying gigabytes for chess.
    A single batched loop only needs to mask the (B,) values being scattered.

    Args:
        tree: (B, N, ...) array leaves forming an Mctx tree.
        leaf_index: (B,) int32 node indices to back up.

    Returns:
        (B, N, ...) array leaves forming the updated tree.
    """
    batch = jnp.arange(leaf_index.shape[0])

    def ongoing(carry: BackupState) -> jax.Array:
        return jnp.any(carry[2] != mctx.Tree.ROOT_INDEX)

    def advance(carry: BackupState) -> BackupState:
        tree, leaf_value, index = carry
        active = index != mctx.Tree.ROOT_INDEX
        # Inactive games use valid dummy indices and retain the old entries.
        parent = jnp.where(active, tree.parents[batch, index], 0)
        action = jnp.where(active, tree.action_from_parent[batch, index], 0)
        count = tree.node_visits[batch, parent]
        leaf_value = (
            tree.children_rewards[batch, parent, action] + tree.children_discounts[batch, parent, action] * leaf_value
        )
        parent_value = (tree.node_values[batch, parent] * count + leaf_value) / (count + 1.0)
        tree = tree.replace(
            node_values=tree.node_values.at[batch, parent].set(
                jnp.where(active, parent_value, tree.node_values[batch, parent])
            ),
            node_visits=tree.node_visits.at[batch, parent].set(count + active),
            children_values=tree.children_values.at[batch, parent, action].set(
                jnp.where(active, tree.node_values[batch, index], tree.children_values[batch, parent, action])
            ),
            children_visits=tree.children_visits.at[batch, parent, action].add(active.astype(jnp.int32)),
        )
        return tree, leaf_value, parent

    return jax.lax.while_loop(ongoing, advance, (tree, tree.node_values[batch, leaf_index], leaf_index))[0]


def puct_policy(
    model: PolicyValueFn,
    key: jax.Array,
    root: mctx.RootFnOutput,
    recurrent_fn: mctx.RecurrentFn,
    num_simulations: int,
    *,
    invalid_actions: jax.Array,
    temperature: jax.Array | float,
) -> mctx.PolicyOutput[None]:
    """Run Mctx's default MuZero PUCT policy without adding root noise.

    Args:
        model: Bound policy/value function passed to recurrent_fn.
        key: (2,) uint32 PRNG key.
        root: Batched root; self-play must mix root noise into prior_logits first.
            prior_logits: (B, A) float32 policy logits.
            value: (B,) float32 root values.
            embedding: (B, ...) state array leaves.
        recurrent_fn: Mctx transition function.
        num_simulations: Number of node expansions per game.
        invalid_actions: (B, A) boolean mask of invalid root actions.
        temperature: () action-sampling temperature, or a Python scalar.

    Returns:
        A PolicyOutput containing:
            action: (B,) sampled action IDs.
            action_weights: (B, A) float32 normalized visit counts.
            search_tree: (B, N, ...) array leaves, where N = num_simulations + 1.
    """
    # Preserve Mctx's RNG split and zero-noise probability round trip so search
    # visits and sampled actions agree with muzero_policy(dirichlet_fraction=0).
    action_key, _, search_key = jax.random.split(key, 3)
    tiny = jnp.finfo(root.prior_logits.dtype).tiny
    logits = jnp.log(jnp.maximum(jax.nn.softmax(root.prior_logits), tiny))
    logits -= logits.max(axis=-1, keepdims=True)
    root = root.replace(prior_logits=jnp.where(invalid_actions, jnp.finfo(logits.dtype).min, logits))
    batch_size = root.value.shape[0]
    batch = jnp.arange(batch_size)

    def simulate(simulation: int, carry: SearchState) -> SearchState:
        key, tree = carry
        key, simulate_key, expand_key = jax.random.split(key, 3)
        parent, action = mctx_search.simulate(
            jax.random.split(simulate_key, batch_size), tree, mctx.muzero_action_selection, num_simulations
        )
        next_node = tree.children_index[batch, parent, action]
        next_node = jnp.where(next_node == mctx.Tree.UNVISITED, simulation + 1, next_node)
        tree = mctx_search.expand(model, expand_key, tree, recurrent_fn, parent, action, next_node)
        return key, _backward(tree, next_node)

    # Mctx exposes no backup hook. Reuse its internal allocation, traversal and
    # expansion helpers; reference-policy tests guard this small compatibility
    # surface when the dependency is upgraded.
    tree = mctx_search.instantiate_tree_from_root(root, num_simulations, invalid_actions, extra_data=None)
    _, tree = jax.lax.fori_loop(0, num_simulations, simulate, (search_key, tree))
    weights = tree.summary().visit_probs
    logits = jnp.log(jnp.maximum(weights, tiny))
    logits = (logits - logits.max(axis=-1, keepdims=True)) / jnp.maximum(tiny, temperature)
    return mctx.PolicyOutput(
        action=jax.random.categorical(action_key, logits), action_weights=weights, search_tree=tree
    )
