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
"""Data-parallel AlphaZero sketch using PGX rules and local Mctx-style PUCT search.

Run: uv run python -m rl2.alphazero.train --config configs/alphazero/chess.yaml
For a cheap start, set env_id: tic_tac_toe and max_moves: 9 in the YAML config.

Training always uses a mesh over all visible JAX devices, including one device.
Self-play games and training batches are sharded on the batch axis; parameters,
optimizer state, RNG keys, and scalar metrics are fully replicated. num_envs and
batch_size are global sizes and must be divisible by the device count. Losses
average eligible positions globally, before gradient clipping and optimization.
Standalone collect_selfplay, sample_batch, and train_step calls require a
jax.set_mesh context with an Auto batch axis. Evaluation retains its fixed suite.

Array dimensions used throughout this module:

- B: Parallel game states; config.num_envs during self-play.
- T: Recorded moves per game; config.max_moves, including terminal padding.
- N: Search-node slots per game; num_simulations + 1, including the root.
- A: Available action IDs; env.num_actions, including currently illegal actions.
- M: Positions sampled for one optimizer update; config.batch_size.
- P: Players per game; env.num_players (2 for the supported games).
- O: Tuple of observation dimensions; *O expands these axes, for example
  (3, 3, 2) for tic-tac-toe, so (B, T, *O) means (B, T, 3, 3, 2).

Rollouts have shape (B, T, ...), with B * T slots including padding. Minibatches
have shape (M, ...): M is configured independently of B * T and counts samples,
not unique positions. Sampling excludes padding and uses replacement, so M can
exceed B * T. Sharding partitions the global B or M axis across devices.
Only lax.scan's temporary output is time-first (T, B, ...); it is immediately
converted to (B, T, ...). Within a per-game vmap body, the B axis is absent.
() denotes a scalar, ... denotes remaining array axes, and (2,) RNG keys contain
two uint32 words; their length is unrelated to P.

Each iteration starts fresh games and samples updates from that rollout only.
Games still active at max_moves are adjudicated as draws and train both heads;
terminal padding trains neither. Orbax checkpoints save every checkpoint_interval
iterations (default: 10) and on normal completion, retaining only the latest
checkpoint. Reusing a run directory resumes the latest saved iteration.
Progress metrics aggregate over log_interval
iterations (default: 1), with a final partial window emitted on normal completion.
All intervals use multiples of the cumulative completed iteration count, including
restored progress; the first logging window after resume may be shorter.

MLflow metrics use cumulative played positions as the step axis, excluding
terminal padding. Each logging window contains complete training iterations:

- train/policy_loss: Mean policy cross-entropy against MCTS action weights,
  averaged over optimizer updates in the window. Padding is masked out.
- train/value_loss: Mean squared error against final game outcomes, averaged
  over updates, including zero targets for move-limit draws. Environment-truncated
  games and updates with no eligible positions are excluded. Omitted when no
  update in the window has value targets; measured zeros are retained.
- train/loss: Sum of policy and value losses, averaged over all updates, with
  zero value contribution for updates without value targets. AdamW
  weight decay is applied by the optimizer and is not included in this loss.
- rollout/positions: Total played positions collected in the window.
- rollout/value_positions: Total positions with outcome targets, including move-limit draws,
  collected in the window, before sampling training minibatches.
- rollout/completed_games: Total completed games in the window, including
  move-limit draws. Environment-truncated games are excluded.
- train/iteration: Latest completed iteration number, starting at one.
- train/progress_fraction: Completed iterations, including restored progress,
  divided by configured iterations; a unitless fraction in [0, 1], without averaging.
- rollout/completed_games_total: Cumulative completed games, including move-limit draws and restored
  checkpoint progress.
- train/learning_rate: Configured constant optimizer learning rate.
- train/environment_steps_per_second: Window positions divided by window wall time.
- train/iteration_seconds_mean: Mean host elapsed time dispatching self-play and
  optimization per iteration; excludes metric transfers, logging, evaluation,
  and checkpoint saving. Device execution can continue after dispatch returns.
- train/window_seconds: Wall time since the previous window boundary, including
  intervening logging, evaluation, and checkpoint overhead, through the current
  window's metric transfer.
Unless noted, these metrics are emitted for every logging window. Metrics stay
on-device until logging, evaluation, or checkpointing needs host values. Timings
do not explicitly synchronize optimizer outputs; outstanding device work may
overlap the next boundary.

At startup, model/params_millions records the parameter count in millions;
config records the resolved configuration as text.
Stdout reports the latest iteration and cumulative steps, window self-play
totals, mean losses, and window wall time (seconds).

Every eval_interval iterations (default: 30),
the current model plays the most recently saved checkpoint, before a new save.
Evaluation pauses training and uses a separate fixed seed: 50 random legal
opening prefixes, each played twice with the agents swapping colors, equal
num_simulations, no root noise, and greedy visit-count actions. max_moves caps
each game after the opening.
If no checkpoint exists yet, evaluation is skipped until the next interval.

Evaluation metrics are logged immediately under eval/, without window averaging:
- wins, draws, losses, completed_games: Counts from terminated evaluation games.
- games, truncated_games, truncation_rate: Total attempted games and those
  unfinished at the move limit (or truncated by the environment).
- score: (wins + 0.5 * draws) / completed_games; win_rate, draw_rate, loss_rate
  also use completed_games. Omitted when no game finishes. Truncation can bias
  this conditional score, so always inspect truncation_rate alongside it.
- score_lower_bound, score_upper_bound: Conservative 95% bounds for all-game
  score over sampled openings, using opening pairs as independent samples and
  allowing unfinished games any outcome. These can be wide for small suites.
- candidate_iteration, opponent_iteration: Iterations identifying both models.
- seconds: Wall time for the evaluation, including any initial JIT compilation.
Evaluation does not alter training RNG, progress counters, or checkpoint policy.
This scaffold has no persistent replay.
PGX owns observation/action encodings and in-game draw rules. Values always use the
player-to-move perspective. This is not an exact reproduction of the paper.
"""

import argparse
import logging
import sys
from dataclasses import asdict
from functools import partial
from time import monotonic
from typing import NamedTuple, Protocol

import chex
import jax
import jax.numpy as jnp
import optax
import pgx
import yaml
from flax import struct
from flax.training.train_state import TrainState
from jax.sharding import AxisType, Mesh, PartitionSpec
from omegaconf.errors import OmegaConfBaseException

from rl2.alphazero import evaluation
from rl2.alphazero.checkpoints import Checkpoint, checkpoint_manager, restore_checkpoint, save_checkpoint
from rl2.alphazero.config import Config
from rl2.alphazero.model import Parameters, PolicyValueApplyFn, PolicyValueNet
from rl2.configuration import load_config
from rl2.mlflow_logging import MLflowLogger
from rl2.running_stats import RunningSum

logger = logging.getLogger(__name__)

type Metrics = dict[str, jax.Array]


@struct.dataclass
class RootFnOutput:
    prior_logits: jax.Array  # (B, A)
    value: jax.Array  # (B,)
    env_states: pgx.State  # Array leaves: (B, ...)


@struct.dataclass
class RecurrentFnOutput:
    reward: jax.Array  # (B,)
    discount: jax.Array  # (B,)
    prior_logits: jax.Array  # (B, A)
    value: jax.Array  # (B,)


@struct.dataclass
class SearchTree:
    """Batched search storage; node 0 is the root and -1 denotes a missing node."""

    node_visits: jax.Array  # (B, N)
    node_values: jax.Array  # (B, N)
    parents: jax.Array  # (B, N)
    action_from_parent: jax.Array  # (B, N)
    children_index: jax.Array  # (B, N, A)
    children_prior_logits: jax.Array  # (B, N, A)
    children_values: jax.Array  # (B, N, A)
    children_visits: jax.Array  # (B, N, A)
    children_rewards: jax.Array  # (B, N, A)
    children_discounts: jax.Array  # (B, N, A)
    env_states: pgx.State  # Array leaves: (B, N, ...)
    root_invalid_actions: jax.Array  # (B, A)


@struct.dataclass
class PolicyOutput:
    action: jax.Array  # (B,)
    action_weights: jax.Array  # (B, A)
    search_tree: SearchTree  # Array leaves: (B, N, ...)


type SearchState = tuple[jax.Array, SearchTree]


class SimulationState(NamedTuple):
    key: jax.Array  # (2,)
    parent: jax.Array  # ()
    action: jax.Array  # ()
    next_node: jax.Array  # ()


class TrainMetrics(NamedTuple):
    """Per-update losses averaged over eligible positions and value-update count."""

    loss: jax.Array  # ()
    policy_loss: jax.Array  # ()
    value_loss: jax.Array  # ()
    value_updates: jax.Array  # () int32; one when the update has value targets, otherwise zero.


class TrainStepResult(NamedTuple):
    train_state: TrainState
    key: jax.Array  # (2,)
    metrics: TrainMetrics  # Scalar leaves: ().


class WindowMetrics(NamedTuple):
    """Sums of update losses, rollout counters, and iteration timing in a logging window."""

    update_metrics: TrainMetrics  # Scalar leaves: ().
    completed_games: jax.Array  # ()
    positions: jax.Array  # ()
    value_positions: jax.Array  # ()
    iterations: jax.Array  # ()
    iteration_seconds: jax.Array  # ()


class BackupState(NamedTuple):
    tree: SearchTree  # Leaves: (B, N, ...)
    leaf_value: jax.Array  # (B,)
    index: jax.Array  # (B,)


@struct.dataclass
class Batch:
    """Training targets for a rollout or a sampled minibatch."""

    observations: jax.Array  # (B, T, *O) or (M, *O)
    policy_targets: jax.Array  # (B, T, A) or (M, A)
    value_targets: jax.Array  # (B, T) or (M,)
    policy_mask: jax.Array  # (B, T) or (M,)
    value_mask: jax.Array  # (B, T) or (M,)


class SelfplayStep(NamedTuple):
    observations: jax.Array  # (B, *O)
    policy_targets: jax.Array  # (B, A)
    player_ids: jax.Array  # (B,)
    policy_mask: jax.Array  # (B,)


class PolicyValueFn(Protocol):
    """Bound model mapping observations to policy logits and player-to-move values."""

    def __call__(self, observations: jax.Array, /) -> tuple[jax.Array, jax.Array]:  # (B, *O) -> (B, A), (B,)
        """Map (B, *O) observations to (B, A) float32 logits and (B,) float32 values."""
        ...


class RecurrentFn(Protocol):
    def __call__(
        self,
        predict: PolicyValueFn,
        actions: jax.Array,  # (B,)
        env_states: pgx.State,  # Array leaves: (B, ...)
        /,
    ) -> tuple[RecurrentFnOutput, pgx.State]:
        """Return (B, ...) predictions and child state leaves."""
        ...


def init_tree(
    root: RootFnOutput,  # Array leaves: (B, ...)
    num_simulations: int,
    invalid_actions: jax.Array,  # (B, A)
) -> SearchTree:
    """Allocate (B, N, ...) storage and count the root prediction as its first visit."""
    chex.assert_rank(root.prior_logits, 2)
    batch_size, num_actions = root.prior_logits.shape
    chex.assert_shape(root.value, (batch_size,))
    chex.assert_shape(invalid_actions, (batch_size, num_actions))
    node_shape = (batch_size, num_simulations + 1)
    edge_shape = (*node_shape, num_actions)

    def allocate(leaf: jax.Array) -> jax.Array:  # (B, ...) -> (B, N, ...)
        chex.assert_shape(leaf, (batch_size, *leaf.shape[1:]))
        return jnp.zeros((*node_shape, *leaf.shape[1:]), leaf.dtype).at[:, 0].set(leaf)  # (B, N, ...)

    return SearchTree(
        node_visits=jnp.zeros(node_shape, jnp.int32).at[:, 0].set(1),  # (B, N)
        node_values=jnp.zeros(node_shape, root.value.dtype).at[:, 0].set(root.value),  # (B, N)
        parents=jnp.full(node_shape, -1, jnp.int32),  # (B, N)
        action_from_parent=jnp.full(node_shape, -1, jnp.int32),  # (B, N)
        children_index=jnp.full(edge_shape, -1, jnp.int32),  # (B, N, A)
        children_prior_logits=jnp.zeros(edge_shape, root.prior_logits.dtype)
        .at[:, 0]
        .set(root.prior_logits),  # (B, N, A)
        children_values=jnp.zeros(edge_shape, root.value.dtype),  # (B, N, A)
        children_visits=jnp.zeros(edge_shape, jnp.int32),  # (B, N, A)
        children_rewards=jnp.zeros(edge_shape, root.value.dtype),  # (B, N, A)
        children_discounts=jnp.zeros(edge_shape, root.value.dtype),  # (B, N, A)
        env_states=jax.tree.map(allocate, root.env_states),  # Leaves: (B, N, ...)
        root_invalid_actions=invalid_actions,  # (B, A)
    )


def select_action(
    key: jax.Array,  # (2,)
    tree: SearchTree,  # Unbatched leaves: (N, ...)
    node: jax.Array,  # ()
) -> jax.Array:  # ()
    """Select by Mctx's default PUCT with parent/sibling Q normalization."""
    visits = tree.children_visits[node]  # (A,)
    node_visits = tree.node_visits[node]  # ()
    pb_c = 1.25 + jnp.log((node_visits + 19652.0 + 1.0) / 19652.0)  # ()
    priors = jax.nn.softmax(tree.children_prior_logits[node])  # (A,)
    policy_score = jnp.sqrt(node_visits) * pb_c * priors / (visits + 1)  # (A,)
    qvalues = tree.children_rewards[node] + tree.children_discounts[node] * tree.children_values[node]  # (A,)
    value = tree.node_values[node]  # ()
    safe_qvalues = jnp.where(visits > 0, qvalues, value)  # (A,)
    minimum = jnp.minimum(value, safe_qvalues.min())  # ()
    maximum = jnp.maximum(value, safe_qvalues.max())  # ()
    value_score = (jnp.where(visits > 0, qvalues, minimum) - minimum) / jnp.maximum(maximum - minimum, 1e-8)  # (A,)
    noise = 1e-7 * jax.random.uniform(key, priors.shape)  # (A,)
    scores = value_score + policy_score + noise  # (A,)
    scores = jnp.where(tree.root_invalid_actions & (node == 0), -jnp.inf, scores)  # (A,)
    return jnp.argmax(scores).astype(jnp.int32)  # ()


@jax.vmap
def simulate(
    key: jax.Array,  # (2,) per game; caller supplies (B, 2).
    tree: SearchTree,  # Unbatched leaves: (N, ...)
) -> tuple[jax.Array, jax.Array]:  # (), () per game; caller receives (B,), (B,).
    """Traverse an acyclic search tree to an unvisited edge."""

    def ongoing(carry: SimulationState) -> jax.Array:  # ()
        return carry.next_node != -1  # ()

    def advance(carry: SimulationState) -> SimulationState:
        key, action_key = jax.random.split(carry.key)  # Each: (2,)
        node = carry.next_node  # ()
        action = select_action(action_key, tree, node)  # ()
        return SimulationState(
            key=key,
            parent=node,
            action=action,
            next_node=tree.children_index[node, action],  # ()
        )

    initial = SimulationState(
        key=key,
        parent=jnp.int32(-1),
        action=jnp.int32(-1),
        next_node=jnp.int32(0),
    )
    result = jax.lax.while_loop(ongoing, advance, initial)
    return result.parent, result.action  # (), ()


def expand(
    predict: PolicyValueFn,
    tree: SearchTree,  # Leaves: (B, N, ...)
    recurrent_fn: RecurrentFn,
    parent: jax.Array,  # (B,)
    action: jax.Array,  # (B,)
    next_node: jax.Array,  # (B,)
) -> SearchTree:
    """Evaluate (B,) unvisited edges and store predictions and transitions in new nodes."""
    batch_size, _, num_actions = tree.children_index.shape
    batch_indices = jnp.arange(batch_size)  # (B,)

    def select(leaf: jax.Array) -> jax.Array:  # (B, N, ...) -> (B, ...)
        return leaf[batch_indices, parent]  # (B, ...)

    output, next_env_states = recurrent_fn(predict, action, jax.tree.map(select, tree.env_states))
    chex.assert_shape(output.prior_logits, (batch_size, num_actions))
    chex.assert_shape((output.reward, output.discount, output.value), (batch_size,))

    def update(old: jax.Array, new: jax.Array) -> jax.Array:  # (B, N, ...), (B, ...) -> (B, N, ...)
        chex.assert_shape(new, (batch_size, *old.shape[2:]))
        return old.at[batch_indices, next_node].set(new)  # (B, N, ...)

    return tree.replace(
        children_prior_logits=update(tree.children_prior_logits, output.prior_logits),  # (B, N, A)
        node_values=update(tree.node_values, output.value),  # (B, N)
        node_visits=tree.node_visits.at[batch_indices, next_node].set(1),  # (B, N)
        env_states=jax.tree.map(update, tree.env_states, next_env_states),  # Leaves: (B, N, ...)
        children_index=tree.children_index.at[batch_indices, parent, action].set(next_node),  # (B, N, A)
        children_rewards=tree.children_rewards.at[batch_indices, parent, action].set(output.reward),  # (B, N, A)
        children_discounts=tree.children_discounts.at[batch_indices, parent, action].set(output.discount),  # (B, N, A)
        parents=tree.parents.at[batch_indices, next_node].set(parent),  # (B, N)
        action_from_parent=tree.action_from_parent.at[batch_indices, next_node].set(action),  # (B, N)
    )


def masked_logits(
    logits: jax.Array,  # (..., A)
    legal: jax.Array,  # (..., A)
) -> jax.Array:  # (..., A)
    """Shift logits by their legal row maximum and mask illegal actions.

    Args:
        logits: (..., A) floating-point logits over A actions.
        legal: (..., A) boolean mask indicating legal actions.

    Returns:
        (..., A) shifted logits with illegal actions set to the minimum finite
        float32 value. Rows with no legal actions contain only this finite value.
    """
    # Finite masking also keeps terminal nodes with no legal moves well-defined.
    minimum = jnp.finfo(jnp.float32).min
    logits = jnp.where(legal, logits, minimum)  # (..., A)
    return jnp.where(legal, logits - logits.max(axis=-1, keepdims=True), minimum)  # (..., A)


def recurrent_step(
    predict: PolicyValueFn,
    actions: jax.Array,  # (B,)
    env_states: pgx.State,  # Leaves: (B, ...)
    *,
    env: pgx.Env,
) -> tuple[RecurrentFnOutput, pgx.State]:
    """Expand batched search nodes for deterministic, alternating-turn, two-player games.

    Args:
        predict: Bound observation-to-(logits, value) callable, forwarded through
            the search's predict argument. Values use the player-to-move perspective.
        actions: (B,) int32 action IDs.
        env_states: (B, ...) array leaves forming a batched PGX state pytree.
        env: Game rules and action/observation encodings.

    Returns:
        (output, next_env_states), where output contains:
            reward: (B,) float32 immediate reward for the parent state's acting player.
            discount: (B,) float32; -1 for ongoing games, 0 on termination or truncation.
            prior_logits: (B, A) float32 masked child policy logits, A = env.num_actions.
            value: (B,) float32 child player-to-move value, zero on termination or truncation.
        next_env_states: (B, ...) array leaves forming the child PGX state pytree.
        Backup uses reward + discount * child_value, negating the opponent's value.
    """
    next_env_states = jax.vmap(env.step)(env_states, actions)  # Leaves: (B, ...)
    finished = env_states.terminated | env_states.truncated  # (B,)

    def preserve_finished(old: jax.Array, new: jax.Array) -> jax.Array:  # Each: (B, ...)
        return jnp.where(finished.reshape(finished.shape + (1,) * (old.ndim - 1)), old, new)  # (B, ...)

    # PGX checks action legality even after a game has ended. Keep finished
    # nodes absorbing, including truncated states whose mask still has illegal moves.
    next_env_states = jax.tree.map(
        preserve_finished, env_states.replace(rewards=jnp.zeros_like(env_states.rewards)), next_env_states
    )  # Leaves: (B, ...)
    logits, value = predict(next_env_states.observation)  # (B, A), (B,)
    # Check before masking: where would silently broadcast a scalar or (1,) value.
    chex.assert_shape(logits, (env_states.current_player.shape[0], env.num_actions))
    chex.assert_shape(value, env_states.current_player.shape)
    reward = jnp.take_along_axis(next_env_states.rewards, env_states.current_player[:, None], axis=1)[:, 0]  # (B,)
    done = next_env_states.terminated | next_env_states.truncated  # (B,)
    output = RecurrentFnOutput(
        reward=reward,  # (B,)
        discount=jnp.where(done, 0.0, -1.0),  # (B,)
        prior_logits=masked_logits(logits, next_env_states.legal_action_mask),  # (B, A)
        value=jnp.where(done, 0.0, value),  # (B,)
    )
    return output, next_env_states


def backward(
    tree: SearchTree,  # Leaves: (B, N, ...)
    leaf_index: jax.Array,  # (B,)
) -> SearchTree:  # Leaves: (B, N, ...)
    """Back up (B,) leaf indices, masking individual updates once a root is reached.

    Mctx vmaps a per-game while loop. Its implicit masking selects entire
    (B, N, A) children arrays at every depth, copying gigabytes for chess.
    A single batched loop only needs to mask the (B,) values being scattered.

    Args:
        tree: (B, N, ...) array leaves forming a search tree.
        leaf_index: (B,) int32 node indices to back up.

    Returns:
        (B, N, ...) array leaves forming the updated tree.
    """
    batch_indices = jnp.arange(leaf_index.shape[0])  # (B,)

    def ongoing(carry: BackupState) -> jax.Array:  # ()
        return jnp.any(carry.index != 0)  # ()

    def advance(carry: BackupState) -> BackupState:
        tree, leaf_value, index = carry  # Tree leaves: (B, N, ...); values and indices: (B,)
        active = index != 0  # (B,)
        # Inactive games use valid dummy indices and retain the old entries.
        parent = jnp.where(active, tree.parents[batch_indices, index], 0)  # (B,)
        action = jnp.where(active, tree.action_from_parent[batch_indices, index], 0)  # (B,)
        count = tree.node_visits[batch_indices, parent]  # (B,)
        leaf_value = (
            tree.children_rewards[batch_indices, parent, action]
            + tree.children_discounts[batch_indices, parent, action] * leaf_value
        )  # (B,)
        parent_value = (tree.node_values[batch_indices, parent] * count + leaf_value) / (count + 1.0)  # (B,)
        tree = tree.replace(
            node_values=tree.node_values.at[batch_indices, parent].set(
                jnp.where(active, parent_value, tree.node_values[batch_indices, parent])
            ),  # (B, N)
            node_visits=tree.node_visits.at[batch_indices, parent].set(count + active),  # (B, N)
            children_values=tree.children_values.at[batch_indices, parent, action].set(
                jnp.where(
                    active, tree.node_values[batch_indices, index], tree.children_values[batch_indices, parent, action]
                )
            ),  # (B, N, A)
            children_visits=tree.children_visits.at[batch_indices, parent, action].add(
                active.astype(jnp.int32)
            ),  # (B, N, A)
        )  # Leaves: (B, N, ...)
        return BackupState(tree=tree, leaf_value=leaf_value, index=parent)

    initial = BackupState(tree=tree, leaf_value=tree.node_values[batch_indices, leaf_index], index=leaf_index)
    return jax.lax.while_loop(ongoing, advance, initial).tree  # Leaves: (B, N, ...)


def puct_policy(
    predict: PolicyValueFn,
    key: jax.Array,  # (2,)
    root: RootFnOutput,  # Leaves: (B, ...)
    recurrent_fn: RecurrentFn,
    num_simulations: int,
    *,
    invalid_actions: jax.Array,  # (B, A)
    temperature: jax.Array | float,  # ()
) -> PolicyOutput:
    """Run a local implementation of Mctx's default MuZero PUCT policy without adding root noise.

    Args:
        predict: Bound policy/value function passed to recurrent_fn.
        key: (2,) uint32 PRNG key.
        root: Batched root; self-play must mix root noise into prior_logits first.
            prior_logits: (B, A) float32 policy logits.
            value: (B,) float32 root values.
            env_states: (B, ...) state array leaves.
        recurrent_fn: Batched deterministic transition and prediction function.
        num_simulations: Number of node expansions per game; zero gives uniform legal weights.
            Rows with no legal actions fall back to uniform weights over all actions.
        invalid_actions: (B, A) boolean mask of invalid root actions.
        temperature: () action-sampling temperature, or a Python scalar.

    Returns:
        A PolicyOutput containing:
            action: (B,) sampled action IDs.
            action_weights: (B, A) float32 normalized visit counts.
            search_tree: (B, N, ...) node arrays and a (B, A) root action mask.
    """
    # Preserve Mctx's RNG split and zero-noise probability round trip so search
    # visits and sampled actions agree with muzero_policy(dirichlet_fraction=0).
    action_key, _, search_key = jax.random.split(key, 3)  # Each: (2,)
    tiny = jnp.finfo(root.prior_logits.dtype).tiny
    logits = jnp.log(jnp.maximum(jax.nn.softmax(root.prior_logits), tiny))  # (B, A)
    logits -= logits.max(axis=-1, keepdims=True)  # (B, A)
    root = root.replace(prior_logits=jnp.where(invalid_actions, jnp.finfo(logits.dtype).min, logits))  # (B, A) logits
    batch_size = root.value.shape[0]

    def run_simulation(simulation: int, carry: SearchState) -> SearchState:
        key, tree = carry  # Key: (2,); tree leaves: (B, N, ...)
        key, simulate_key, _ = jax.random.split(key, 3)  # Each: (2,); preserve Mctx's selection RNG.
        parent, action = simulate(jax.random.split(simulate_key, batch_size), tree)  # Each: (B,)
        # Each simulation reaches a missing edge and allocates one new node per game.
        next_node = jnp.full((batch_size,), simulation + 1, jnp.int32)  # (B,)
        tree = expand(predict, tree, recurrent_fn, parent, action, next_node)  # Leaves: (B, N, ...)
        return key, backward(tree, next_node)

    tree = init_tree(root, num_simulations, invalid_actions)  # Leaves: (B, N, ...)
    _, tree = jax.lax.fori_loop(0, num_simulations, run_simulation, (search_key, tree))  # Tree leaves: (B, N, ...)
    counts = tree.children_visits[:, 0].astype(root.value.dtype)  # (B, A)
    total = counts.sum(axis=-1, keepdims=True)  # (B, 1)
    legal = ~invalid_actions  # (B, A)
    fallback = (legal | ~legal.any(axis=-1, keepdims=True)).astype(counts.dtype)  # (B, A)
    fallback /= fallback.sum(axis=-1, keepdims=True)  # (B, A)
    weights = jnp.where(total > 0, counts / jnp.maximum(total, 1), fallback)  # (B, A)
    logits = jnp.log(jnp.maximum(weights, tiny))  # (B, A)
    logits = (logits - logits.max(axis=-1, keepdims=True)) / jnp.maximum(tiny, temperature)  # (B, A)
    logits = jnp.where(weights > 0, logits, -jnp.inf)  # (B, A)
    return PolicyOutput(
        action=jax.random.categorical(action_key, logits),  # (B,)
        action_weights=weights,  # (B, A)
        search_tree=tree,  # Leaves: (B, N, ...)
    )


def selfplay_prior_logits(
    logits: jax.Array,  # (B, A)
    legal: jax.Array,  # (B, A)
    key: jax.Array,  # (2,)
    *,
    dirichlet_alpha: float,
    dirichlet_fraction: float,
) -> jax.Array:
    """Mix the policy with Dirichlet noise normalized over legal actions only.

    Args:
        logits: (B, A) float32 network policy logits.
        legal: (B, A) boolean legal-action mask.
        key: (2,) uint32 PRNG key.
        dirichlet_alpha: Positive concentration for each legal action.
        dirichlet_fraction: Noise mixture weight in [0, 1].

    Returns:
        (B, A) float32 noisy, masked logits; empty legal rows remain finite.
    """
    prior_logits = masked_logits(logits, legal)  # (B, A)
    if dirichlet_fraction == 0.0:
        return prior_logits
    # Independent Gamma samples normalized over legal moves form a Dirichlet
    # sample on that subset. Log space avoids underflow for small concentrations.
    log_noise = jax.random.loggamma(key, dirichlet_alpha, shape=logits.shape)  # (B, A)
    noise = jax.nn.softmax(masked_logits(log_noise, legal))  # (B, A)
    probs = (1.0 - dirichlet_fraction) * jax.nn.softmax(prior_logits) + dirichlet_fraction * noise  # (B, A)
    return masked_logits(jnp.log(jnp.maximum(probs, jnp.finfo(probs.dtype).tiny)), legal)  # (B, A)


@partial(
    jax.jit,
    static_argnames=("env", "apply_fn", "config"),
    out_shardings=(PartitionSpec("batch"), PartitionSpec()),
)
def collect_selfplay(
    params: Parameters,
    key: jax.Array,  # (2,)
    *,
    env: pgx.Env,
    apply_fn: PolicyValueApplyFn,
    config: Config,
) -> tuple[Batch, Metrics]:
    """Collect padded games, adjudicating games still active at max_moves as draws.

    Environment truncations remain excluded from value supervision.

    Args:
        params: Model parameter pytree.
        key: (2,) uint32 PRNG key.
        env: Game rules and encodings.
        apply_fn: Policy/value apply function from the training state.
        config: Self-play settings.

    Returns:
        Batch and metrics, with fields:
            observations: (B, T, *O) float32 recorded observations.
            policy_targets: (B, T, A) float32 MCTS action weights.
            value_targets: (B, T) float32 player-relative outcomes, zero for cutoff draws.
            policy_mask: (B, T) bool non-padding positions.
            value_mask: (B, T) bool positions with terminal or adjudicated outcomes.
            completed_games: () integer count including move-limit draws.
            positions: () integer count of non-padding positions.
            value_positions: () integer count of positions with outcome targets.
    """
    params, key = jax.lax.with_sharding_constraint((params, key), PartitionSpec())  # Parameter shapes; key: (2,).
    init_key, play_key = jax.random.split(key)  # Each: (2,)
    env_states = jax.vmap(env.init)(jax.random.split(init_key, config.num_envs))  # Leaves: (B, ...)
    env_states = jax.lax.with_sharding_constraint(env_states, PartitionSpec("batch"))  # Leaves: (B, ...)
    predict = partial(apply_fn, {"params": params})

    def step(
        env_states: pgx.State,  # Leaves: (B, ...)
        inputs: tuple[jax.Array, jax.Array],  # (2,) key and () move index.
    ) -> tuple[pgx.State, SelfplayStep]:
        key, move_i = inputs  # key: (2,); move_i: () scalar move index.
        noise_key, search_key = jax.random.split(key)  # Each: (2,)
        logits, value = predict(env_states.observation)  # (B, A), (B,)
        done = env_states.terminated | env_states.truncated  # (B,)
        root = RootFnOutput(
            prior_logits=selfplay_prior_logits(
                logits,
                env_states.legal_action_mask,
                noise_key,
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_fraction=config.dirichlet_fraction,
            ),  # (B, A)
            value=jnp.where(done, 0.0, value),  # (B,)
            env_states=env_states,  # Leaves: (B, ...)
        )
        # PGX supplies exact transitions for PUCT, with no learned dynamics.
        policy = puct_policy(
            predict,
            search_key,
            root,
            partial(recurrent_step, env=env),
            config.num_simulations,
            invalid_actions=~env_states.legal_action_mask,
            temperature=jnp.where(move_i < config.exploration_moves, 1.0, 0.0),
        )
        next_env_states = jax.vmap(env.step)(env_states, policy.action)  # Leaves: (B, ...)

        def preserve_finished(
            old: jax.Array,  # (B, ...)
            new: jax.Array,  # (B, ...)
        ) -> jax.Array:  # (B, ...)
            """Preserve (B, ...) state leaves for games that have already stopped."""
            # Preserve final rewards: PGX otherwise clears them on later steps.
            return jnp.where(done.reshape((config.num_envs,) + (1,) * (old.ndim - 1)), old, new)  # (B, ...)

        next_env_states = jax.tree.map(preserve_finished, env_states, next_env_states)  # Leaves: (B, ...)

        return next_env_states, SelfplayStep(
            observations=env_states.observation.astype(jnp.float32),  # (B, *O)
            policy_targets=policy.action_weights,  # (B, A)
            player_ids=env_states.current_player,  # (B,)
            policy_mask=~done,  # (B,)
        )

    env_states, rollout = jax.lax.scan(  # Rollout leaves: (T, B, ...).
        step,
        env_states,
        (jax.random.split(play_key, config.max_moves), jnp.arange(config.max_moves, dtype=jnp.int32)),
    )

    def batch_first(x: jax.Array) -> jax.Array:  # (T, B, ...) -> (B, T, ...)
        return x.swapaxes(0, 1)  # (B, T, ...)

    observations, policy_targets, player_ids, policy_mask = jax.tree.map(batch_first, rollout)  # Leaves: (B, T, ...).
    # Map each game's final rewards to its recorded actors.
    value_targets = jnp.take_along_axis(env_states.rewards, player_ids, axis=1)  # (B, T)
    move_limit_draws = ~(env_states.terminated | env_states.truncated)  # (B,)
    completed = env_states.terminated | move_limit_draws  # (B,)
    value_mask = policy_mask & completed[:, None]  # (B, T)
    # A win on the final allowed move keeps its outcome; only active games draw.
    value_targets = jnp.where(policy_mask & env_states.terminated[:, None], value_targets, 0.0)  # (B, T)
    batch = Batch(
        observations=observations,  # (B, T, *O)
        policy_targets=policy_targets,  # (B, T, A)
        value_targets=value_targets,  # (B, T)
        policy_mask=policy_mask,  # (B, T)
        value_mask=value_mask,  # (B, T)
    )

    return batch, {
        "completed_games": completed.sum(),  # ()
        "positions": policy_mask.sum(),  # ()
        "value_positions": value_mask.sum(),  # ()
    }


def loss_fn(
    params: Parameters,
    apply_fn: PolicyValueApplyFn,
    batch: Batch,  # Leaves: (M, ...), sampled positions.
) -> tuple[jax.Array, TrainMetrics]:  # Loss and metric leaves: ().
    """Return () loss and scalar metrics; each head averages its eligible positions."""
    logits, values = apply_fn({"params": params}, batch.observations)  # (M, A), (M,)
    policy_errors = optax.softmax_cross_entropy(logits, batch.policy_targets)  # (M,)
    value_errors = jnp.square(values - batch.value_targets)  # (M,)
    policy_loss = jnp.where(batch.policy_mask, policy_errors, 0.0).sum() / jnp.maximum(batch.policy_mask.sum(), 1)  # ()
    value_count = batch.value_mask.sum()  # ()
    value_loss = jnp.where(batch.value_mask, value_errors, 0.0).sum() / jnp.maximum(value_count, 1)  # ()
    loss = policy_loss + value_loss  # ()
    return loss, TrainMetrics(
        loss=loss,  # ()
        policy_loss=policy_loss,  # ()
        value_loss=value_loss,  # ()
        value_updates=(value_count > 0).astype(jnp.int32),  # ()
    )


@partial(
    jax.jit,
    static_argnames=("batch_size", "updates_per_iteration"),
    out_shardings=PartitionSpec(),
)
def train_step(
    train_state: TrainState,
    key: jax.Array,  # (2,)
    batch: Batch,  # Leaves: (B, T, ...), rollout positions including padding.
    *,
    batch_size: int,
    updates_per_iteration: int,
) -> TrainStepResult:
    """Run sampled updates; return replicated state, (2,) RNG key, and () metric sums."""
    train_state = jax.lax.with_sharding_constraint(train_state, PartitionSpec())
    metrics = TrainMetrics(
        loss=jnp.float32(0),  # ()
        policy_loss=jnp.float32(0),  # ()
        value_loss=jnp.float32(0),  # ()
        value_updates=jnp.int32(0),  # ()
    )

    def update(
        _: int,
        carry: TrainStepResult,
    ) -> TrainStepResult:
        train_state, key, metrics = carry
        key, sample_key = jax.random.split(key)  # Each: (2,)
        minibatch = sample_batch(batch, sample_key, batch_size)  # Leaves: (M, ...), M = batch_size.
        (_, update_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            train_state.params, train_state.apply_fn, minibatch
        )
        grads = jax.lax.with_sharding_constraint(grads, PartitionSpec())  # Parameter shapes.
        return TrainStepResult(
            train_state=train_state.apply_gradients(grads=grads),
            key=key,  # (2,)
            metrics=RunningSum.update(metrics, update_metrics),  # Scalar leaves: ().
        )

    return jax.lax.fori_loop(
        0,
        updates_per_iteration,
        update,
        TrainStepResult(train_state=train_state, key=key, metrics=metrics),
    )


@partial(
    jax.jit,
    static_argnames=("batch_size",),
    out_shardings=PartitionSpec("batch"),
)
def sample_batch(batch: Batch, key: jax.Array, batch_size: int) -> Batch:  # key: (2,)
    """Sample (M, ...) positions from (B, T, ...) rollouts uniformly with replacement.

    M = batch_size, independently of B * T. At least one non-padding position must exist.
    """
    batch = jax.lax.with_sharding_constraint(batch, PartitionSpec("batch"))  # Leaves: (B, T, ...).
    key = jax.lax.with_sharding_constraint(key, PartitionSpec())  # (2,)
    # Invert cumulative counts instead of generating a (M, B * T)
    # Gumbel array. Right insertion skips padding, including leading zero counts.
    counts = jnp.cumsum(batch.policy_mask, dtype=jnp.int32)  # (B * T,) in game-major order.
    ranks = jax.random.randint(key, (batch_size,), 0, counts[-1])  # (M,)
    indices = jnp.searchsorted(counts, ranks, side="right")  # (M,)
    games, moves = jnp.divmod(indices, batch.policy_mask.shape[1])  # Each: (M,)

    def select(x: jax.Array) -> jax.Array:  # (B, T, ...) -> (M, ...)
        return x[games, moves]  # (M, ...)

    return jax.tree.map(select, batch)


def training_mesh(config: Config) -> Mesh:
    """Use all visible devices with equally sized game and minibatch shards."""
    devices = jax.devices()
    for name in ("num_envs", "batch_size"):
        size = getattr(config, name)
        if size % len(devices):
            raise ValueError(f"{name} ({size}) must be divisible by device_count ({len(devices)})")
    return Mesh(devices=devices, axis_names=("batch",), axis_types=(AxisType.Auto,))


def train(config: Config) -> TrainState:
    with (
        jax.set_mesh(training_mesh(config)),
        checkpoint_manager(config.ckpt_dir) as manager,
        MLflowLogger(experiment="alphazero", run_name=config.run_id) as writer,
    ):
        env = pgx.make(config.env_id)
        model = PolicyValueNet(
            num_actions=env.num_actions, channels=config.model.channels, num_blocks=config.model.num_blocks
        )
        key, init_key, env_key = jax.random.split(jax.random.PRNGKey(config.seed), 3)  # Each: (2,)
        observation = env.init(env_key).observation[None].astype(jnp.float32)  # (1, *O)
        params = model.init(init_key, observation)["params"]
        train_state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adamw(config.learning_rate, weight_decay=config.weight_decay),
            ),
        )
        train_state, key = jax.device_put((train_state, key), PartitionSpec())  # State leaves; key: (2,).

        checkpoint = restore_checkpoint(
            manager,
            Checkpoint(train_state=train_state, key=key, iteration=0, steps=0, completed_games=0),
        )
        start_iteration = steps = completed_games = 0
        if checkpoint is not None:
            train_state, key, start_iteration, steps, completed_games = checkpoint
            train_state, key = jax.device_put((train_state, key), PartitionSpec())  # State leaves; key: (2,).
        opponent_params = train_state.params if checkpoint is not None else None
        opponent_iteration = start_iteration
        resolved_config = asdict(config)
        writer.log_params(resolved_config)
        writer.log_text(yaml.safe_dump(resolved_config), "config.yaml")
        parameter_count = sum(parameter.size for parameter in jax.tree.leaves(train_state.params))
        writer.log_metrics({"model/params_millions": parameter_count / 1_000_000}, steps)
        logger.info("Run ID: %s", config.run_id)
        logger.info("MLflow run: %s", writer.run_id)
        if checkpoint is not None:
            logger.info("Resumed iteration %d, step %d", start_iteration, steps)
        window_start = monotonic()
        window_start_steps = steps
        window_start_completed_games = completed_games
        window_metrics = WindowMetrics(
            update_metrics=TrainMetrics(
                loss=jnp.float32(0),  # ()
                policy_loss=jnp.float32(0),  # ()
                value_loss=jnp.float32(0),  # ()
                value_updates=jnp.int32(0),  # ()
            ),
            completed_games=jnp.int32(0),  # ()
            positions=jnp.int32(0),  # ()
            value_positions=jnp.int32(0),  # ()
            iterations=jnp.int32(0),  # ()
            iteration_seconds=jnp.float32(0),  # ()
        )  # Scalar leaves: ().
        for iteration in range(start_iteration + 1, config.iterations + 1):
            iteration_start = monotonic()
            key, play_key = jax.random.split(key)  # Each: (2,)
            batch, rollout_metrics = collect_selfplay(
                train_state.params, play_key, env=env, apply_fn=train_state.apply_fn, config=config
            )
            train_state, key, update_metrics = train_step(
                train_state,
                key,
                batch,
                batch_size=config.batch_size,
                updates_per_iteration=config.updates_per_iteration,
            )
            iteration_metrics = WindowMetrics(
                update_metrics=update_metrics,  # Scalar leaves: ().
                completed_games=rollout_metrics["completed_games"],  # ()
                positions=rollout_metrics["positions"],  # ()
                value_positions=rollout_metrics["value_positions"],  # ()
                iterations=jnp.int32(1),  # ()
                iteration_seconds=jnp.float32(monotonic() - iteration_start),  # ()
            )  # Scalar leaves: ().
            window_metrics = RunningSum.update(window_metrics, iteration_metrics)  # Scalar leaves: ().
            log_due = iteration % config.log_interval == 0 or iteration == config.iterations
            eval_due = iteration % config.eval_interval == 0
            checkpoint_due = iteration % config.checkpoint_interval == 0 or iteration == config.iterations
            if log_due or checkpoint_due or (eval_due and opponent_params is not None):
                host_metrics = jax.device_get(window_metrics)  # Scalar leaves: ().
                # Use actual played positions as the x-axis, excluding terminal padding.
                steps = window_start_steps + int(host_metrics.positions)
                completed_games = window_start_completed_games + int(host_metrics.completed_games)
            if log_due:
                now = monotonic()
                window_seconds = now - window_start
                window_iterations = int(host_metrics.iterations)
                window_updates = window_iterations * config.updates_per_iteration
                host_updates = host_metrics.update_metrics  # Scalar leaves: ().
                mean_losses = {
                    "loss": float(host_updates.loss) / window_updates,
                    "policy_loss": float(host_updates.policy_loss) / window_updates,
                }
                if host_updates.value_updates:
                    mean_losses["value_loss"] = float(host_updates.value_loss) / int(host_updates.value_updates)
                window_rollout = {
                    "completed_games": int(host_metrics.completed_games),
                    "positions": int(host_metrics.positions),
                    "value_positions": int(host_metrics.value_positions),
                }
                scalars = {
                    **{f"train/{name}": value for name, value in mean_losses.items()},
                    **{f"rollout/{name}": value for name, value in window_rollout.items()},
                    "train/iteration": iteration,
                    "train/progress_fraction": iteration / config.iterations,
                    "rollout/completed_games_total": completed_games,
                    "train/learning_rate": config.learning_rate,
                    "train/environment_steps_per_second": window_rollout["positions"] / max(window_seconds, 1e-9),
                    "train/iteration_seconds_mean": float(host_metrics.iteration_seconds) / window_iterations,
                    "train/window_seconds": window_seconds,
                }
                writer.log_metrics(scalars, steps)
                logger.info(
                    "Iteration %d/%d | steps=%d | %s | seconds=%.2f",
                    iteration,
                    config.iterations,
                    steps,
                    " | ".join(f"{name}={value:.6g}" for name, value in (window_rollout | mean_losses).items()),
                    window_seconds,
                )
                window_start = now
                window_start_steps = steps
                window_start_completed_games = completed_games
                window_metrics = RunningSum.init(window_metrics)  # Scalar leaves: ().
            if eval_due:
                if opponent_params is None:
                    logger.info("Skipping evaluation: no previous checkpoint yet")
                else:
                    evaluation_start = monotonic()
                    evaluation_metrics = evaluation.evaluate(
                        opponent_params=opponent_params,
                        candidate_params=train_state.params,
                        env=env,
                        apply_fn=train_state.apply_fn,
                        config=config,
                    )
                    evaluation_metrics.update(
                        candidate_iteration=iteration,
                        opponent_iteration=opponent_iteration,
                        seconds=monotonic() - evaluation_start,
                    )
                    writer.log_metrics({f"eval/{name}": value for name, value in evaluation_metrics.items()}, steps)
                    logger.info(
                        "Evaluation | %s",
                        " | ".join(f"{name}={value:.6g}" for name, value in evaluation_metrics.items()),
                    )
            if checkpoint_due:
                save_checkpoint(
                    manager,
                    Checkpoint(
                        train_state=train_state,
                        key=key,
                        iteration=iteration,
                        steps=steps,
                        completed_games=completed_games,
                    ),
                    config,
                )
                opponent_params = train_state.params
                opponent_iteration = iteration
        return train_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default="configs/alphazero/chess.yaml", help="Path to a YAML config")
    args = parser.parse_args()
    try:
        config = load_config(args.config, Config)
    except (OSError, TypeError, ValueError, yaml.YAMLError, OmegaConfBaseException) as error:
        parser.error(str(error))
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout, force=True
    )
    train(config)


if __name__ == "__main__":
    main()
