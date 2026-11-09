# Portions Copyright 2021 DeepMind Technologies Limited.
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
"""Data-parallel AlphaZero sketch using PGX rules and local PUCT search.

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
Self-play and search operate on one game. A per-game scan produces (T, ...) leaves;
the collector's outer vmap adds B directly, yielding (B, T, ...) leaves.
() denotes a scalar, ... denotes remaining array axes, and (2,) RNG keys contain
two uint32 words; their length is unrelated to P.

Transition convention: env_state --action--> (reward, discount, child_env_state).
logits and value describe env_state; child_logits and child_value describe
child_env_state. Reward belongs to the player acting in env_state, and backup
uses reward + discount * child_value. Values use their own state's player-to-move
perspective; discount is -1 for an ongoing game and 0 when it has finished.
Tree edges follow node_index --action--> child_index. node_action_* arrays
describe outgoing edges; children_* arrays describe their destinations.
parent_* fields are incoming links used only to walk back toward the root.

Each iteration starts fresh games and samples updates from that rollout only.
Games still active at max_moves are adjudicated as draws and train both heads;
terminal padding trains neither. Orbax checkpoints save every checkpoint_interval
iterations (default: 10) and on normal completion, retaining only the latest
checkpoint. Reusing a run directory resumes the latest saved iteration.
Progress metrics aggregate over log_interval
iterations (default: 1), with a final partial window emitted on normal completion.
All intervals use multiples of the cumulative completed iteration count, including
restored progress; the first logging window after resume may be shorter.

MLflow metrics use cumulative environment steps as the step axis, excluding
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
- rollout/environment_steps: Number of moves played across all games in the window, excluding padding.
- rollout/value_target_count: Total positions with outcome targets, including move-limit draws,
  collected in the window, before sampling training minibatches.
- rollout/completed_games: Total completed games in the window, including
  move-limit draws. Environment-truncated games are excluded.
- train/iteration: Latest completed iteration number, starting at one.
- train/progress_fraction: Completed iterations, including restored progress,
  divided by configured iterations; a unitless fraction in [0, 1], without averaging.
- rollout/completed_games_total: Cumulative completed games, including move-limit draws and restored
  checkpoint progress.
- train/learning_rate: Configured constant optimizer learning rate.
- train/environment_steps_per_second: Window environment steps divided by window wall time.
- train/iteration_seconds: Window wall time divided by completed iterations in
  the window; includes the same overhead as train/window_seconds.
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


@struct.dataclass
class SearchRoot:
    env_state: pgx.State  # Array leaves: (...) Game state from which search starts.
    logits: jax.Array  # (A,) Policy scores guiding action exploration at the root.
    value: jax.Array  # () Estimated return from the root player's perspective.


@struct.dataclass
class SearchTransition:
    """Result of taking action in env_state; predictions belong to child_env_state."""

    reward: jax.Array  # () Immediate reward for the player who took the action.
    discount: jax.Array  # () child_value multiplier: -1 for ongoing games, 0 when finished.
    child_env_state: pgx.State  # Array leaves: (...) State reached by taking the action.
    child_logits: jax.Array  # (A,) Policy scores for actions at child_env_state.
    child_value: jax.Array  # () Estimated return for child_env_state's player to move.


@struct.dataclass
class SearchTree:
    """Single-game search storage; node 0 is the root and -1 denotes a missing node."""

    node_env_states: pgx.State  # Array leaves: (N, ...) Game states stored at each node.
    node_logits: jax.Array  # (N, A) Policy scores for actions at each node's state.
    node_values: jax.Array  # (N,) Mean returns for each node's player to move.
    node_visits: jax.Array  # (N,) Visit counts, including each node's initial prediction.
    parent_indices: jax.Array  # (N,) Source node of each incoming edge; -1 for root/unused slots.
    parent_actions: jax.Array  # (N,) Action taken at that parent to reach this node; -1 if unset.
    node_action_visits: jax.Array  # (N, A) Number of times each node-action edge was traversed.
    node_action_rewards: jax.Array  # (N, A) Immediate rewards for actions taken by this node's player.
    node_action_discounts: jax.Array  # (N, A) children_values multipliers: -1 ongoing, 0 finished.
    children_indices: jax.Array  # (N, A) Destination node for each action; -1 if unexpanded.
    children_values: jax.Array  # (N, A) Cached mean values for the destination's player to move.


class SearchState(NamedTuple):
    """Carry between simulations; the tree contains all search results so far."""

    key: jax.Array  # (2,) RNG key for the next simulation.
    tree: SearchTree  # Array leaves: (N, ...) Accumulated search state.


class SimulationState(NamedTuple):
    """Selected node --action--> child edge during one root-to-leaf traversal."""

    key: jax.Array  # (2,) RNG key for the next action selection during traversal.
    node_index: jax.Array  # () Source node of the selected edge.
    action: jax.Array  # () Action selected at node_index.
    child_index: jax.Array  # () Destination node to visit; -1 if the edge is unexpanded.


class BackupState(NamedTuple):
    tree: SearchTree  # Array leaves: (N, ...) Tree being updated along the path to the root.
    node_index: jax.Array  # () Current node in the backup traversal.
    node_return: jax.Array  # () Backed-up return for node_index's player to move.


@struct.dataclass
class PolicyOutput:
    action: jax.Array  # () Action sampled from root visit weights with temperature applied.
    action_weights: jax.Array  # (A,) Normalized root visits used as policy targets; uniform legal fallback if none.
    tree: SearchTree  # Array leaves: (N, ...) Tree and accumulated statistics after search.


class SelfplayStep(NamedTuple):
    observation: jax.Array  # (*O,) Observation at env_state, before the action.
    action_weights: jax.Array  # (A,) Search action weights at env_state.
    current_player: jax.Array  # () Player acting in env_state.
    policy_mask: jax.Array  # () True for a played move; false for padding after the game ends.


@struct.dataclass
class Batch:
    """Training targets for a rollout or a sampled minibatch."""

    observation: jax.Array  # (T, *O), (B, T, *O), or (M, *O) Observations before each action.
    action_weights: jax.Array  # (T, A), (B, T, A), or (M, A) Normalized search visit counts over actions.
    value_targets: jax.Array  # (T,), (B, T), or (M,) Final outcomes for each position's acting player.
    policy_mask: jax.Array  # (T,), (B, T), or (M,) True for positions eligible for policy training.
    value_mask: jax.Array  # (T,), (B, T), or (M,) True for positions with valid outcome targets.


class RolloutMetrics(NamedTuple):
    """Counts summed across games in one self-play rollout."""

    completed_games: jax.Array  # ()
    environment_steps: jax.Array  # ()
    value_target_count: jax.Array  # () Number of rollout positions with valid value targets.


class UpdateMetrics(NamedTuple):
    """Loss means and a value-update count per update; sums when accumulated."""

    loss: jax.Array  # ()
    policy_loss: jax.Array  # ()
    value_loss: jax.Array  # ()
    value_update_count: jax.Array  # () int32; number of updates with valid value targets (0 or 1 per update).


class TrainStepResult(NamedTuple):
    train_state: TrainState  # Model parameters, optimizer state, and update counter after the updates so far.
    key: jax.Array  # (2,) RNG key for sampling the next minibatch.
    update_metrics: UpdateMetrics  # Scalar leaves: () Metric sums over updates completed in this iteration.


class WindowMetrics(NamedTuple):
    """Sums of update losses, rollout counters, and iterations in a logging window."""

    update_metrics: UpdateMetrics  # Scalar leaves: ().
    rollout_metrics: RolloutMetrics  # Scalar leaves: ().
    iterations: jax.Array  # ()


class PolicyValueFn(Protocol):
    """Bound model mapping one observation to policy logits and a player-to-move value."""

    def __call__(self, observation: jax.Array, /) -> tuple[jax.Array, jax.Array]:  # (*O,) -> (A,), ()
        """Map (*O,) observation to (A,) float32 logits and () float32 value."""
        ...


class TransitionFn(Protocol):
    def __call__(
        self,
        predict: PolicyValueFn,
        env_state: pgx.State,  # Unbatched array leaves.
        action: jax.Array,  # ()
        /,
    ) -> SearchTransition:
        """Take action in env_state; return reward, discount, and child-state predictions."""
        ...


def init_search_tree(
    root: SearchRoot,
    num_simulations: int,
) -> SearchTree:
    """Allocate (N, ...) storage and count the root prediction as its first visit."""
    chex.assert_rank(root.logits, 1)
    chex.assert_shape(root.value, ())
    chex.assert_shape(root.env_state.legal_action_mask, root.logits.shape)
    node_shape = (num_simulations + 1,)
    edge_shape = (*node_shape, root.logits.shape[0])

    def allocate(leaf: jax.Array) -> jax.Array:  # (...) -> (N, ...)
        return jnp.zeros((*node_shape, *leaf.shape), leaf.dtype).at[0].set(leaf)  # (N, ...)

    return SearchTree(
        node_env_states=jax.tree.map(allocate, root.env_state),  # Leaves: (N, ...)
        node_logits=jnp.zeros(edge_shape, root.logits.dtype).at[0].set(root.logits),  # (N, A)
        node_values=jnp.zeros(node_shape, root.value.dtype).at[0].set(root.value),  # (N,)
        node_visits=jnp.zeros(node_shape, jnp.int32).at[0].set(1),  # (N,)
        parent_indices=jnp.full(node_shape, -1, jnp.int32),  # (N,)
        parent_actions=jnp.full(node_shape, -1, jnp.int32),  # (N,)
        node_action_visits=jnp.zeros(edge_shape, jnp.int32),  # (N, A)
        node_action_rewards=jnp.zeros(edge_shape, root.value.dtype),  # (N, A)
        node_action_discounts=jnp.zeros(edge_shape, root.value.dtype),  # (N, A)
        children_indices=jnp.full(edge_shape, -1, jnp.int32),  # (N, A)
        children_values=jnp.zeros(edge_shape, root.value.dtype),  # (N, A)
    )


def compute_policy_score(
    action_probs: jax.Array,  # (A,)
    node_visits: jax.Array,  # ()
    action_visits: jax.Array,  # (A,)
) -> jax.Array:  # (A,)
    """Return (A,) PUCT exploration bonuses from policy probabilities and visit counts."""
    exploration_coefficient = 1.25 + jnp.log((node_visits + 19652.0 + 1.0) / 19652.0)  # ()
    return jnp.sqrt(node_visits) * exploration_coefficient * action_probs / (action_visits + 1)  # (A,)


def select_action(
    key: jax.Array,  # (2,)
    tree: SearchTree,  # Unbatched leaves: (N, ...)
    node_index: jax.Array,  # ()
) -> jax.Array:  # ()
    """Return the () action ID with the highest PUCT score at node_index.

    Each score combines an estimated return (reward + discount * child value)
    with an exploration bonus favoring probable, less-visited actions. Returns
    are normalized using the node's value and visited actions; unvisited actions
    get a value score of zero. Tiny random noise breaks ties.

    Illegal actions are excluded. If none are legal, allow all as a fallback
    for finished states.
    """
    action_visits = tree.node_action_visits[node_index]  # (A,)
    node_visits = tree.node_visits[node_index]  # ()
    action_probs = jax.nn.softmax(tree.node_logits[node_index])  # (A,)
    policy_score = compute_policy_score(action_probs, node_visits, action_visits)  # (A,)
    action_values = (
        tree.node_action_rewards[node_index] + tree.node_action_discounts[node_index] * tree.children_values[node_index]
    )  # (A,)
    node_value = tree.node_values[node_index]  # ()
    visited_action_values = jnp.where(action_visits > 0, action_values, node_value)  # (A,)
    minimum = jnp.minimum(node_value, visited_action_values.min())  # ()
    maximum = jnp.maximum(node_value, visited_action_values.max())  # ()
    value_score = (jnp.where(action_visits > 0, action_values, minimum) - minimum) / jnp.maximum(
        maximum - minimum, 1e-8
    )  # (A,)
    noise = 1e-7 * jax.random.uniform(key, action_probs.shape)  # (A,)
    scores = value_score + policy_score + noise  # (A,)
    legal = tree.node_env_states.legal_action_mask[node_index]  # (A,)
    scores = jnp.where(legal | ~legal.any(), scores, -jnp.inf)  # (A,)
    return jnp.argmax(scores).astype(jnp.int32)  # ()


def simulate(
    key: jax.Array,  # (2,)
    tree: SearchTree,  # Unbatched leaves: (N, ...)
) -> SimulationState:
    """Traverse from the root to an unexpanded edge or a terminal/truncated node."""

    def ongoing(carry: SimulationState) -> jax.Array:  # ()
        done = (
            tree.node_env_states.terminated[carry.child_index] | tree.node_env_states.truncated[carry.child_index]
        )  # ()
        return (carry.child_index != -1) & ~done  # ()

    def advance(carry: SimulationState) -> SimulationState:
        key, action_key = jax.random.split(carry.key)  # Each: (2,)
        node_index = carry.child_index  # ()
        action = select_action(action_key, tree, node_index)  # ()
        return SimulationState(
            key=key,
            node_index=node_index,
            action=action,
            child_index=tree.children_indices[node_index, action],  # ()
        )

    initial = SimulationState(
        key=key,
        # Valid placeholders for finished roots in batches with other games still expanding.
        node_index=jnp.int32(0),
        action=jnp.int32(0),
        child_index=jnp.int32(0),
    )
    return jax.lax.while_loop(ongoing, advance, initial)


def expand(
    predict: PolicyValueFn,
    tree: SearchTree,  # Unbatched leaves: (N, ...)
    transition_fn: TransitionFn,
    node_index: jax.Array,  # ()
    action: jax.Array,  # ()
    child_index: jax.Array,  # ()
) -> SearchTree:  # Unbatched leaves: (N, ...)
    """Evaluate node_index --action--> child_index and store the transition."""
    _, num_actions = tree.children_indices.shape

    def select(leaf: jax.Array) -> jax.Array:  # (N, ...) -> (...)
        return leaf[node_index]  # (...)

    env_state = jax.tree.map(select, tree.node_env_states)  # Leaves: (...)
    transition = transition_fn(predict, env_state, action)
    chex.assert_shape(transition.child_logits, (num_actions,))
    chex.assert_shape((transition.reward, transition.discount, transition.child_value), ())

    def update(old: jax.Array, new: jax.Array) -> jax.Array:  # (N, ...), (...) -> (N, ...)
        chex.assert_shape(new, old.shape[1:])
        return old.at[child_index].set(new)  # (N, ...)

    return tree.replace(
        node_env_states=jax.tree.map(update, tree.node_env_states, transition.child_env_state),  # Leaves: (N, ...)
        node_logits=update(tree.node_logits, transition.child_logits),  # (N, A)
        node_values=update(tree.node_values, transition.child_value),  # (N,)
        node_visits=tree.node_visits.at[child_index].set(1),  # (N,)
        parent_indices=tree.parent_indices.at[child_index].set(node_index),  # (N,)
        parent_actions=tree.parent_actions.at[child_index].set(action),  # (N,)
        node_action_rewards=tree.node_action_rewards.at[node_index, action].set(transition.reward),  # (N, A)
        node_action_discounts=tree.node_action_discounts.at[node_index, action].set(transition.discount),  # (N, A)
        children_indices=tree.children_indices.at[node_index, action].set(child_index),  # (N, A)
    )


def masked_logits(
    logits: jax.Array,  # (A,)
    legal: jax.Array,  # (A,)
) -> jax.Array:  # (A,)
    """Shift logits by their legal maximum and mask illegal actions.

    Args:
        logits: (A,) floating-point logits over A actions.
        legal: (A,) boolean mask indicating legal actions.

    Returns:
        (A,) shifted logits with illegal actions set to the minimum finite
        float32 value. States with no legal actions contain only this finite value.
    """
    chex.assert_rank(logits, 1)
    chex.assert_shape(legal, logits.shape)
    # Finite masking also keeps terminal nodes with no legal moves well-defined.
    minimum = jnp.finfo(jnp.float32).min
    logits = jnp.where(legal, logits, minimum)  # (A,)
    return jnp.where(legal, logits - logits.max(), minimum)  # (A,)


def transition_step(
    predict: PolicyValueFn,
    env_state: pgx.State,  # Leaves: (...)
    action: jax.Array,  # ()
    *,
    env: pgx.Env,
) -> SearchTransition:
    """Take one action and predict policy/value at the resulting state in a two-player game.

    Args:
        predict: Bound observation-to-(logits, value) callable; values use the player-to-move perspective.
        env_state: (...) array leaves forming an unbatched PGX state pytree.
        action: () int32 action ID selected at env_state.
        env: Game rules and action/observation encodings.

    Returns:
        A SearchTransition containing:
            reward: () float32 immediate reward for env_state's acting player.
            discount: () float32; -1 for ongoing games, 0 on termination or truncation.
            child_env_state: (...) array leaves forming the resulting PGX state pytree.
            child_logits: (A,) float32 masked policy logits at child_env_state, A = env.num_actions.
            child_value: () float32 value for child_env_state's player, zero on termination or truncation.
        Backup uses reward + discount * child_value, negating the opponent's value.
    """
    child_env_state = env.step(env_state, action)  # Leaves: (...)
    finished = env_state.terminated | env_state.truncated  # ()

    def preserve_finished(old: jax.Array, new: jax.Array) -> jax.Array:  # Each: (...)
        return jnp.where(finished, old, new)  # (...)

    # PGX checks action legality even after a game has ended. Keep finished
    # nodes absorbing, including truncated states whose mask still has illegal moves.
    child_env_state = jax.tree.map(
        preserve_finished, env_state.replace(rewards=jnp.zeros_like(env_state.rewards)), child_env_state
    )  # Leaves: (...)
    child_logits, child_value = predict(child_env_state.observation)  # (A,), ()
    chex.assert_shape(child_logits, (env.num_actions,))
    chex.assert_shape(child_value, ())
    chex.assert_shape(action, ())
    reward = child_env_state.rewards[env_state.current_player]  # ()
    done = child_env_state.terminated | child_env_state.truncated  # ()
    return SearchTransition(
        reward=reward,  # ()
        discount=jnp.where(done, 0.0, -1.0),  # ()
        child_env_state=child_env_state,  # Leaves: (...)
        child_logits=masked_logits(child_logits, child_env_state.legal_action_mask),  # (A,)
        child_value=jnp.where(done, 0.0, child_value),  # ()
    )


def backup(
    tree: SearchTree,  # Unbatched leaves: (N, ...)
    leaf_index: jax.Array,  # ()
) -> SearchTree:  # Unbatched leaves: (N, ...)
    """Back up one leaf's value and visit counts along its path to the root.

    Args:
        tree: (N, ...) node arrays, including each node's environment state.
        leaf_index: () int32 node index to back up.

    Returns:
        (N, ...) updated node arrays; environment states are unchanged.
    """

    def ongoing(carry: BackupState) -> jax.Array:  # ()
        return carry.node_index != 0  # ()

    def advance(carry: BackupState) -> BackupState:
        tree, node_index, node_return = carry  # Tree leaves: (N, ...); index and return: ().
        parent_index = tree.parent_indices[node_index]  # ()
        parent_action = tree.parent_actions[node_index]  # ()
        parent_visits = tree.node_visits[parent_index]  # ()
        parent_return = (
            tree.node_action_rewards[parent_index, parent_action]
            + tree.node_action_discounts[parent_index, parent_action] * node_return
        )  # ()
        parent_mean_value = (tree.node_values[parent_index] * parent_visits + parent_return) / (
            parent_visits + 1.0
        )  # ()
        tree = tree.replace(
            node_values=tree.node_values.at[parent_index].set(parent_mean_value),  # (N,)
            node_visits=tree.node_visits.at[parent_index].add(1),  # (N,)
            children_values=tree.children_values.at[parent_index, parent_action].set(
                tree.node_values[node_index]
            ),  # (N, A)
            node_action_visits=tree.node_action_visits.at[parent_index, parent_action].add(1),  # (N, A)
        )  # Leaves: (N, ...)
        return BackupState(tree=tree, node_index=parent_index, node_return=parent_return)

    initial = BackupState(tree=tree, node_index=leaf_index, node_return=tree.node_values[leaf_index])
    return jax.lax.while_loop(ongoing, advance, initial).tree  # Leaves: (N, ...)


def puct_policy(
    predict: PolicyValueFn,
    key: jax.Array,  # (2,)
    env_state: pgx.State,  # Leaves: (...)
    transition_fn: TransitionFn,
    num_simulations: int,
    *,
    temperature: jax.Array | float,  # ()
    dirichlet_alpha: float = 0.3,
    dirichlet_fraction: float = 0.0,
) -> PolicyOutput:
    """Predict the root and run PUCT search with optional legal-action Dirichlet noise.

    Args:
        predict: Bound policy/value function used at the root and passed to transition_fn.
        key: (2,) uint32 PRNG key.
        env_state: (...) state array leaves; finished root values are zeroed.
        transition_fn: Single-game transition and prediction function.
        num_simulations: Number of simulations; each expands at most one node.
            Finished leaves are revisited without expansion. Zero gives uniform legal weights.
            States with no legal actions fall back to uniform weights over all actions.
        temperature: () action-sampling temperature, or a Python scalar.
        dirichlet_alpha: Positive concentration for each legal root action.
        dirichlet_fraction: Noise mixture weight in [0, 1]; zero disables root noise.

    Returns:
        A PolicyOutput containing:
            action: () sampled action ID.
            action_weights: (A,) float32 normalized visit counts.
            tree: (N, ...) node arrays, including each node's environment state.
    """
    action_key, noise_key, search_key = jax.random.split(key, 3)  # Each: (2,)
    logits, value = predict(env_state.observation)  # (A,), ()
    chex.assert_shape(logits, env_state.legal_action_mask.shape)
    chex.assert_shape(value, ())
    done = env_state.terminated | env_state.truncated  # ()
    root = SearchRoot(
        env_state=env_state,  # Leaves: (...)
        logits=prepare_logits(
            logits,
            env_state.legal_action_mask,
            noise_key,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_fraction=dirichlet_fraction,
        ),  # (A,)
        value=jnp.where(done, 0.0, value),  # ()
    )

    def advance(simulation: int, carry: SearchState) -> SearchState:
        tree = carry.tree  # Leaves: (N, ...)
        key, simulation_key = jax.random.split(carry.key)  # Each: (2,)
        simulation_state = simulate(simulation_key, tree)
        new_node_index = jnp.asarray(simulation + 1, jnp.int32)  # ()

        def pending(carry: tuple[SearchTree, jax.Array]) -> jax.Array:  # ((N, ...) tree, () flag) -> ()
            return carry[1]  # ()

        def expand_leaf(carry: tuple[SearchTree, jax.Array]) -> tuple[SearchTree, jax.Array]:
            tree, _ = carry  # Tree leaves: (N, ...); flag: ().
            return expand(
                predict, tree, transition_fn, simulation_state.node_index, simulation_state.action, new_node_index
            ), jnp.bool_(False)  # (N, ...), ()

        # A single-pass while keeps model parameters shared under vmap, unlike a batched cond.
        tree, _ = jax.lax.while_loop(pending, expand_leaf, (tree, simulation_state.child_index == -1))
        leaf_index = jnp.where(simulation_state.child_index == -1, new_node_index, simulation_state.child_index)  # ()
        tree = tree.replace(
            node_visits=tree.node_visits.at[leaf_index].add(
                (simulation_state.child_index != -1).astype(jnp.int32)
            ),  # (N,)
        )
        return SearchState(key=key, tree=backup(tree, leaf_index))

    tree = init_search_tree(root, num_simulations)  # Leaves: (N, ...)
    tree = jax.lax.fori_loop(
        0, num_simulations, advance, SearchState(key=search_key, tree=tree)
    ).tree  # Leaves: (N, ...)
    counts = tree.node_action_visits[0].astype(root.value.dtype)  # (A,)
    total = counts.sum()  # ()
    legal = env_state.legal_action_mask  # (A,)
    fallback = (legal | ~legal.any()).astype(counts.dtype)  # (A,)
    fallback /= fallback.sum()  # (A,)
    weights = jnp.where(total > 0, counts / jnp.maximum(total, 1), fallback)  # (A,)
    action_logits = jnp.log(jnp.maximum(weights, jnp.finfo(weights.dtype).tiny))  # (A,)
    action_logits = (action_logits - action_logits.max()) / jnp.maximum(
        jnp.finfo(weights.dtype).tiny, temperature
    )  # (A,)
    action_logits = jnp.where(weights > 0, action_logits, -jnp.inf)  # (A,)
    return PolicyOutput(
        action=jax.random.categorical(action_key, action_logits),  # ()
        action_weights=weights,  # (A,)
        tree=tree,  # Leaves: (N, ...)
    )


def prepare_logits(
    logits: jax.Array,  # (A,)
    legal: jax.Array,  # (A,)
    key: jax.Array,  # (2,)
    *,
    dirichlet_alpha: float,
    dirichlet_fraction: float,
) -> jax.Array:
    """Mask policy logits and optionally mix in Dirichlet noise over legal actions.

    Args:
        logits: (A,) float32 network policy logits.
        legal: (A,) boolean legal-action mask.
        key: (2,) uint32 PRNG key.
        dirichlet_alpha: Positive concentration for each legal action.
        dirichlet_fraction: Noise mixture weight in [0, 1].

    Returns:
        (A,) float32 noisy, masked logits; states with no legal actions remain finite.
    """
    logits = masked_logits(logits, legal)  # (A,)
    if dirichlet_fraction == 0.0:
        return logits
    # Independent Gamma samples normalized over legal moves form a Dirichlet
    # sample on that subset. Log space avoids underflow for small concentrations.
    log_noise = jax.random.loggamma(key, dirichlet_alpha, shape=logits.shape)  # (A,)
    noise = jax.nn.softmax(masked_logits(log_noise, legal))  # (A,)
    probs = (1.0 - dirichlet_fraction) * jax.nn.softmax(logits) + dirichlet_fraction * noise  # (A,)
    return masked_logits(jnp.log(jnp.maximum(probs, jnp.finfo(probs.dtype).tiny)), legal)  # (A,)


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
) -> tuple[Batch, RolloutMetrics]:
    """Collect padded games, adjudicating games still active at max_moves as draws.

    Environment truncations remain excluded from value supervision.

    Args:
        params: Model parameter pytree.
        key: (2,) uint32 PRNG key.
        env: Game rules and encodings.
        apply_fn: Policy/value apply function from the training state.
        config: Self-play settings.

    Returns:
        Batch and RolloutMetrics, with fields:
            observation: (B, T, *O) float32 recorded observations.
            action_weights: (B, T, A) float32 MCTS action weights.
            value_targets: (B, T) float32 player-relative outcomes, zero for cutoff draws.
            policy_mask: (B, T) bool non-padding positions.
            value_mask: (B, T) bool positions with terminal or adjudicated outcomes.
            completed_games: () integer count including move-limit draws.
            environment_steps: () integer count of moves played across all games, excluding padding.
            value_target_count: () integer count of positions with outcome targets.
    """
    params, key = jax.lax.with_sharding_constraint((params, key), PartitionSpec())  # Parameter shapes; key: (2,).
    keys = jax.random.split(key, config.num_envs)  # (B, 2)
    keys = jax.lax.with_sharding_constraint(keys, PartitionSpec("batch"))  # (B, 2)
    collect = partial(collect_one_game, params, env=env, apply_fn=apply_fn, config=config)
    batch, metrics = jax.vmap(collect)(keys)  # Batch leaves: (B, T, ...); metric leaves: (B,)
    return batch, jax.tree.map(jnp.sum, metrics)  # Batch leaves: (B, T, ...); metric leaves: ().


def collect_one_game(
    params: Parameters,
    key: jax.Array,  # (2,)
    *,
    env: pgx.Env,
    apply_fn: PolicyValueApplyFn,
    config: Config,
) -> tuple[Batch, RolloutMetrics]:
    """Collect one padded game; active games at the move limit are adjudicated draws.

    Args:
        params: Model parameter pytree shared across games.
        key: (2,) uint32 game PRNG key.
        env: Game rules and encodings.
        apply_fn: Batched network, adapted to one observation at its boundary.
        config: Self-play settings; num_envs is used only by collect_selfplay.

    Returns:
        Batch and RolloutMetrics, with fields:
            observation: (T, *O) float32 observations before each move.
            action_weights: (T, A) float32 normalized search visit counts.
            value_targets: (T,) float32 actor-relative outcomes; zero for padding and draws.
            policy_mask: (T,) bool non-padding positions.
            value_mask: (T,) bool positions with terminal or adjudicated outcomes, excluding truncations.
            completed_games: () int32; one for a terminal or adjudicated game, otherwise zero.
            environment_steps: () integer count of non-padding moves.
            value_target_count: () integer count of supervised value targets.
    """
    init_key, play_key = jax.random.split(key)  # Each: (2,)
    env_state = env.init(init_key)  # Unbatched array leaves.

    def predict(observation: jax.Array) -> tuple[jax.Array, jax.Array]:  # (*O,) -> (A,), ()
        logits, value = apply_fn({"params": params}, observation[None])  # (1, A), (1,)
        chex.assert_rank(logits, 2)
        chex.assert_shape(logits, (1, logits.shape[-1]))
        chex.assert_shape(value, (1,))
        return logits[0], value[0]  # (A,), ()

    def step(
        env_state: pgx.State,  # Leaves: (...)
        inputs: tuple[jax.Array, jax.Array],  # (2,) key and () move index.
    ) -> tuple[pgx.State, SelfplayStep]:
        key, move_index = inputs  # key: (2,); move_index: ().
        done = env_state.terminated | env_state.truncated  # ()
        # PGX supplies exact transitions for PUCT, with no learned dynamics.
        policy = puct_policy(
            predict,
            key,
            env_state,
            partial(transition_step, env=env),
            config.num_simulations,
            temperature=jnp.where(move_index < config.exploration_moves, 1.0, 0.0),
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_fraction=config.dirichlet_fraction,
        )
        next_env_state = env.step(env_state, policy.action)  # Leaves: (...)

        def preserve_finished(
            old: jax.Array,  # (...)
            new: jax.Array,  # (...)
        ) -> jax.Array:  # (...)
            """Preserve (...) state leaves for games that have already stopped."""
            # Preserve final rewards: PGX otherwise clears them on later steps.
            return jnp.where(done, old, new)  # (...)

        next_env_state = jax.tree.map(preserve_finished, env_state, next_env_state)  # Leaves: (...)

        return next_env_state, SelfplayStep(
            observation=env_state.observation.astype(jnp.float32),  # (*O,)
            action_weights=policy.action_weights,  # (A,)
            current_player=env_state.current_player,  # ()
            policy_mask=~done,  # ()
        )

    final_env_state, rollout = jax.lax.scan(  # State leaves: (...); rollout leaves: (T, ...).
        step,
        env_state,
        (jax.random.split(play_key, config.max_moves), jnp.arange(config.max_moves, dtype=jnp.int32)),
    )

    observation, action_weights, current_player, policy_mask = rollout  # Leaves: (T, ...)
    # Map final rewards to the player who acted at each move.
    value_targets = final_env_state.rewards[current_player]  # (T,)
    move_limit_draws = ~(final_env_state.terminated | final_env_state.truncated)  # ()
    completed = final_env_state.terminated | move_limit_draws  # ()
    value_mask = policy_mask & completed  # (T,)
    # A win on the final allowed move keeps its outcome; only active games draw.
    value_targets = jnp.where(policy_mask & final_env_state.terminated, value_targets, 0.0)  # (T,)
    batch = Batch(
        observation=observation,  # (T, *O)
        action_weights=action_weights,  # (T, A)
        value_targets=value_targets,  # (T,)
        policy_mask=policy_mask,  # (T,)
        value_mask=value_mask,  # (T,)
    )

    return batch, RolloutMetrics(
        completed_games=completed.astype(jnp.int32),  # ()
        environment_steps=policy_mask.sum(),  # ()
        value_target_count=value_mask.sum(),  # ()
    )


def loss_fn(
    params: Parameters,
    apply_fn: PolicyValueApplyFn,
    batch: Batch,  # Leaves: (M, ...), sampled positions.
) -> tuple[jax.Array, UpdateMetrics]:  # Loss and metric leaves: ().
    """Return () loss and scalar metrics; each head averages its eligible positions."""
    logits, values = apply_fn({"params": params}, batch.observation)  # (M, A), (M,)
    policy_errors = optax.softmax_cross_entropy(logits, batch.action_weights)  # (M,)
    value_errors = jnp.square(values - batch.value_targets)  # (M,)
    policy_loss = jnp.where(batch.policy_mask, policy_errors, 0.0).sum() / jnp.maximum(batch.policy_mask.sum(), 1)  # ()
    value_count = batch.value_mask.sum()  # ()
    value_loss = jnp.where(batch.value_mask, value_errors, 0.0).sum() / jnp.maximum(value_count, 1)  # ()
    loss = policy_loss + value_loss  # ()
    return loss, UpdateMetrics(
        loss=loss,  # ()
        policy_loss=policy_loss,  # ()
        value_loss=value_loss,  # ()
        value_update_count=(value_count > 0).astype(jnp.int32),  # ()
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
    metrics = UpdateMetrics(
        loss=jnp.float32(0),  # ()
        policy_loss=jnp.float32(0),  # ()
        value_loss=jnp.float32(0),  # ()
        value_update_count=jnp.int32(0),  # ()
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
            update_metrics=RunningSum.update(metrics, update_metrics),  # Scalar leaves: ().
        )

    return jax.lax.fori_loop(
        0,
        updates_per_iteration,
        update,
        TrainStepResult(train_state=train_state, key=key, update_metrics=metrics),
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
            update_metrics=UpdateMetrics(
                loss=jnp.float32(0),  # ()
                policy_loss=jnp.float32(0),  # ()
                value_loss=jnp.float32(0),  # ()
                value_update_count=jnp.int32(0),  # ()
            ),
            rollout_metrics=RolloutMetrics(
                completed_games=jnp.int32(0),  # ()
                environment_steps=jnp.int32(0),  # ()
                value_target_count=jnp.int32(0),  # ()
            ),
            iterations=jnp.int32(0),  # ()
        )  # Scalar leaves: ().
        for iteration in range(start_iteration + 1, config.iterations + 1):
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
                rollout_metrics=rollout_metrics,  # Scalar leaves: ().
                iterations=jnp.int32(1),  # ()
            )  # Scalar leaves: ().
            window_metrics = RunningSum.update(window_metrics, iteration_metrics)  # Scalar leaves: ().
            log_due = iteration % config.log_interval == 0 or iteration == config.iterations
            eval_due = iteration % config.eval_interval == 0
            checkpoint_due = iteration % config.checkpoint_interval == 0 or iteration == config.iterations
            if log_due or checkpoint_due or (eval_due and opponent_params is not None):
                host_metrics = jax.device_get(window_metrics)  # Scalar leaves: ().
                # Use actual environment steps as the x-axis, excluding terminal padding.
                steps = window_start_steps + int(host_metrics.rollout_metrics.environment_steps)
                completed_games = window_start_completed_games + int(host_metrics.rollout_metrics.completed_games)
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
                if host_updates.value_update_count:
                    mean_losses["value_loss"] = float(host_updates.value_loss) / int(host_updates.value_update_count)
                window_rollout = {name: int(value) for name, value in host_metrics.rollout_metrics._asdict().items()}
                scalars = {
                    **{f"train/{name}": value for name, value in mean_losses.items()},
                    "train/iteration": iteration,
                    "train/progress_fraction": iteration / config.iterations,
                    "train/learning_rate": config.learning_rate,
                    "train/environment_steps_per_second": window_rollout["environment_steps"]
                    / max(window_seconds, 1e-9),
                    "train/iteration_seconds": window_seconds / window_iterations,
                    "train/window_seconds": window_seconds,
                    **{f"rollout/{name}": value for name, value in window_rollout.items()},
                    "rollout/completed_games_total": completed_games,
                }
                writer.log_metrics(scalars, steps)
                logger.info(
                    "Iteration %d/%d | steps=%d | %s | seconds=%.2f",
                    iteration,
                    config.iterations,
                    steps,
                    " | ".join(f"{name}={value:.6g}" for name, value in (mean_losses | window_rollout).items()),
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
