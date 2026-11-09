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
jax.set_mesh context with an Auto batch axis. Construct PGX environments before
entering the mesh so their module-level arrays remain usable within device shards.
Evaluation retains its fixed suite.

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
Self-play scans over moves with an explicit game batch; search and inference use
batched arrays directly. Only single-state PGX APIs and per-game RNG operations
use vmap. Scan outputs are transposed from (T, B, ...) to (B, T, ...).
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
from rl2.utils import mask_logits

logger = logging.getLogger(__name__)


@struct.dataclass
class SearchRoot:
    env_state: pgx.State  # Array leaves: (B, ...) Game state from which search starts.
    logits: jax.Array  # (B, A) Policy scores guiding action exploration at the root.
    value: jax.Array  # (B,) Estimated return from the root player's perspective.


@struct.dataclass
class SearchTransition:
    """Result of taking action in env_state; predictions belong to child_env_state."""

    reward: jax.Array  # (B,) Immediate reward for the player who took the action.
    discount: jax.Array  # (B,) child_value multiplier: -1 for ongoing games, 0 when finished.
    child_env_state: pgx.State  # Array leaves: (B, ...) State reached by taking the action.
    child_logits: jax.Array  # (B, A) Policy scores for actions at child_env_state.
    child_value: jax.Array  # (B,) Estimated return for child_env_state's player to move.


@struct.dataclass
class SearchTree:
    """Batched search storage; node 0 is the root and -1 denotes a missing node.

    The single-game puct_policy adapter removes B from every returned field.
    """

    node_env_states: pgx.State  # Array leaves: (B, N, ...) Game states stored at each node.
    node_logits: jax.Array  # (B, N, A) Policy scores for actions at each node's state.
    node_values: jax.Array  # (B, N) Mean returns for each node's player to move.
    node_visits: jax.Array  # (B, N) Visit counts, including each node's initial prediction.
    parent_indices: jax.Array  # (B, N) Source node of each incoming edge; -1 for root/unused slots.
    parent_actions: jax.Array  # (B, N) Action taken at that parent to reach this node; -1 if unset.
    node_action_visits: jax.Array  # (B, N, A) Number of times each node-action edge was traversed.
    node_action_rewards: jax.Array  # (B, N, A) Immediate rewards for actions taken by this node's player.
    node_action_discounts: jax.Array  # (B, N, A) children_values multipliers: -1 ongoing, 0 finished.
    children_indices: jax.Array  # (B, N, A) Destination node for each action; -1 if unexpanded.
    children_values: jax.Array  # (B, N, A) Cached mean values for the destination's player to move.


class SearchState(NamedTuple):
    """Carry between simulations; the tree contains all search results so far."""

    key: jax.Array  # (B, 2) RNG key for the next simulation.
    tree: SearchTree  # Array leaves: (B, N, ...) Accumulated search state.


class SimulationState(NamedTuple):
    """Selected node --action--> child edge during one root-to-leaf traversal."""

    key: jax.Array  # (B, 2) RNG key for the next action selection during traversal.
    node_index: jax.Array  # (B,) Source node of the selected edge.
    action: jax.Array  # (B,) Action selected at node_index.
    child_index: jax.Array  # (B,) Destination node to visit; -1 if the edge is unexpanded.


class BackupState(NamedTuple):
    tree: SearchTree  # Array leaves: (B, N, ...) Tree being updated along the path to the root.
    node_index: jax.Array  # (B,) Current node in the backup traversal.
    node_return: jax.Array  # (B,) Backed-up return for node_index's player to move.


@struct.dataclass
class PolicyOutput:
    """Search results; the single-game puct_policy adapter removes the B axis."""

    action: jax.Array  # (B,) Action sampled from root visit weights with temperature applied.
    action_weights: jax.Array  # (B, A) Normalized root visits used as policy targets; uniform legal fallback if none.
    tree: SearchTree  # Array leaves: (B, N, ...) Tree and accumulated statistics after search.


class SelfplayStep(NamedTuple):
    observation: jax.Array  # (B, *O) Observation at env_state, before the action.
    action_weights: jax.Array  # (B, A) Search action weights at env_state.
    current_player: jax.Array  # (B,) Player acting in env_state.
    policy_mask: jax.Array  # (B,) True for a played move; false for padding after the game ends.


@struct.dataclass
class Batch:
    """Training targets for a rollout or a sampled minibatch."""

    observation: jax.Array  # (T, *O), (B, T, *O), or (M, *O) Observations before each action.
    action_weights: jax.Array  # (T, A), (B, T, A), or (M, A) Normalized search visit counts over actions.
    value_targets: jax.Array  # (T,), (B, T), or (M,) Final outcomes for each position's acting player.
    policy_mask: jax.Array  # (T,), (B, T), or (M,) True for positions eligible for policy training.
    value_mask: jax.Array  # (T,), (B, T), or (M,) True for positions with valid outcome targets.


class RolloutMetrics(NamedTuple):
    """Per-game counts in collect_games; summed across games in collect_selfplay."""

    completed_games: jax.Array  # (B,) or ()
    environment_steps: jax.Array  # (B,) or ()
    value_target_count: jax.Array  # (B,) or () Number of rollout positions with valid value targets.


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
    """Bound model accepting the observation batch supplied by its caller."""

    def __call__(self, observation: jax.Array, /) -> tuple[jax.Array, jax.Array]:  # (B, *O) or (*O,)
        """Map (B, *O) to (B, A), (B,); the puct_policy adapter instead accepts (*O,) -> (A,), ()."""
        ...


def init_search_tree(root: SearchRoot, num_simulations: int) -> SearchTree:
    """Allocate (B, N, ...) storage, counting each root prediction as one visit."""
    chex.assert_rank(root.logits, 2)
    chex.assert_shape(root.value, root.logits.shape[:1])
    chex.assert_shape(root.env_state.legal_action_mask, root.logits.shape)
    node_shape = (root.logits.shape[0], num_simulations + 1)
    edge_shape = (*node_shape, root.logits.shape[-1])

    def allocate(leaf: jax.Array) -> jax.Array:  # (B, ...) -> (B, N, ...)
        return jnp.zeros((*node_shape, *leaf.shape[1:]), leaf.dtype).at[:, 0].set(leaf)  # (B, N, ...)

    return SearchTree(
        node_env_states=jax.tree.map(allocate, root.env_state),  # Leaves: (B, N, ...)
        node_logits=allocate(root.logits),  # (B, N, A)
        node_values=allocate(root.value),  # (B, N)
        node_visits=jnp.zeros(node_shape, jnp.int32).at[:, 0].set(1),  # (B, N)
        parent_indices=jnp.full(node_shape, -1, jnp.int32),  # (B, N)
        parent_actions=jnp.full(node_shape, -1, jnp.int32),  # (B, N)
        node_action_visits=jnp.zeros(edge_shape, jnp.int32),  # (B, N, A)
        node_action_rewards=jnp.zeros(edge_shape, root.value.dtype),  # (B, N, A)
        node_action_discounts=jnp.zeros(edge_shape, root.value.dtype),  # (B, N, A)
        children_indices=jnp.full(edge_shape, -1, jnp.int32),  # (B, N, A)
        children_values=jnp.zeros(edge_shape, root.value.dtype),  # (B, N, A)
    )


def prepare_logits(
    logits: jax.Array,  # (B, A)
    legal: jax.Array,  # (B, A)
    key: jax.Array,  # (B, 2)
    *,
    dirichlet_alpha: float,
    dirichlet_fraction: float,
) -> jax.Array:  # (B, A)
    """Mask (B, A) logits and mix independent Dirichlet noise over each game's legal moves."""
    logits = mask_logits(logits, legal)  # (B, A)
    if dirichlet_fraction == 0.0:
        return logits
    # Log-space Gamma samples avoid underflow for small concentrations.
    log_noise = jax.vmap(partial(jax.random.loggamma, a=dirichlet_alpha, shape=logits.shape[1:]))(key)  # (B, A)
    noise = jax.nn.softmax(mask_logits(log_noise, legal))  # (B, A)
    probs = (1.0 - dirichlet_fraction) * jax.nn.softmax(logits) + dirichlet_fraction * noise  # (B, A)
    return mask_logits(jnp.log(jnp.maximum(probs, jnp.finfo(probs.dtype).tiny)), legal)  # (B, A)


def select_puct_action(
    key: jax.Array,  # (B, 2)
    tree: SearchTree,  # Leaves: (B, N, ...)
    node_index: jax.Array,  # (B,)
) -> jax.Array:  # (B,)
    """Select each game's legal action using normalized returns, PUCT bonus, and random tie breaks."""
    rows = jnp.arange(tree.node_visits.shape[0])  # (B,)
    index = (rows, node_index)
    action_visits = tree.node_action_visits[index]  # (B, A)
    node_visits = tree.node_visits[index]  # (B,)
    action_probs = jax.nn.softmax(tree.node_logits[index])  # (B, A)
    coefficient = 1.25 + jnp.log((node_visits + 19652.0 + 1.0) / 19652.0)  # (B,)
    policy_score = (jnp.sqrt(node_visits) * coefficient)[:, None] * action_probs / (action_visits + 1)  # (B, A)
    action_values = (
        tree.node_action_rewards[index] + tree.node_action_discounts[index] * tree.children_values[index]
    )  # (B, A)
    node_value = tree.node_values[index][:, None]  # (B, 1)
    visited_values = jnp.where(action_visits > 0, action_values, node_value)  # (B, A)
    minimum = jnp.minimum(node_value, visited_values.min(axis=-1, keepdims=True))  # (B, 1)
    maximum = jnp.maximum(node_value, visited_values.max(axis=-1, keepdims=True))  # (B, 1)
    value_score = (jnp.where(action_visits > 0, action_values, minimum) - minimum) / jnp.maximum(
        maximum - minimum, 1e-8
    )  # (B, A)
    noise = 1e-7 * jax.vmap(partial(jax.random.uniform, shape=action_probs.shape[1:]))(key)  # (B, A)
    legal = tree.node_env_states.legal_action_mask[index]  # (B, A)
    scores = jnp.where(
        legal | ~legal.any(axis=-1, keepdims=True), value_score + policy_score + noise, -jnp.inf
    )  # (B, A)
    return jnp.argmax(scores, axis=-1).astype(jnp.int32)  # (B,)


def simulate(key: jax.Array, tree: SearchTree) -> SimulationState:  # Key: (B, 2); tree: (B, N, ...)
    """Traverse each game to an unexpanded edge or finished node, freezing completed paths."""
    rows = jnp.arange(tree.node_visits.shape[0])  # (B,)

    def active(carry: SimulationState) -> jax.Array:  # (B,)
        index = (rows, jnp.maximum(carry.child_index, 0))
        done = tree.node_env_states.terminated[index] | tree.node_env_states.truncated[index]  # (B,)
        return (carry.child_index != -1) & ~done  # (B,)

    def condition(carry: SimulationState) -> jax.Array:  # ()
        return active(carry).any()  # ()

    def body(carry: SimulationState) -> SimulationState:
        running = active(carry)  # (B,)
        keys = jax.vmap(jax.random.split)(carry.key)  # (B, 2, 2)
        node_index = jnp.maximum(carry.child_index, 0)  # (B,)
        action = select_puct_action(keys[:, 1], tree, node_index)  # (B,)
        return SimulationState(
            key=jnp.where(running[:, None], keys[:, 0], carry.key),  # (B, 2)
            node_index=jnp.where(running, node_index, carry.node_index),  # (B,)
            action=jnp.where(running, action, carry.action),  # (B,)
            child_index=jnp.where(running, tree.children_indices[rows, node_index, action], carry.child_index),  # (B,)
        )

    zeros = jnp.zeros(rows.shape, jnp.int32)  # (B,)
    return jax.lax.while_loop(
        condition, body, SimulationState(key=key, node_index=zeros, action=zeros, child_index=zeros)
    )


def step_env(env_state: pgx.State, action: jax.Array, *, env: pgx.Env) -> pgx.State:
    """Step (B, ...) states with (B,) actions, preserving finished states and final rewards."""

    def step_one(state: pgx.State, action: jax.Array) -> pgx.State:  # Leaves: (...); action: ()
        def preserve() -> pgx.State:
            return state  # Leaves: (...)

        def advance() -> pgx.State:
            return env.step(state, action)  # Leaves: (...)

        return jax.lax.cond(state.terminated | state.truncated, preserve, advance)

    return jax.vmap(step_one)(env_state, action)  # Leaves: (B, ...)


def transition_step(
    predict: PolicyValueFn,
    env_state: pgx.State,  # Leaves: (B, ...)
    action: jax.Array,  # (B,)
    *,
    env: pgx.Env,
) -> SearchTransition:
    """Step a batch, returning actor-relative rewards and opponent-relative child predictions.

    Finished parents are absorbing with zero subsequent reward; finished children
    have zero logits/value and discount. Ongoing discounts are -1.
    """
    parent_done = env_state.terminated | env_state.truncated  # (B,)
    child = step_env(env_state, action, env=env)  # Leaves: (B, ...)
    child = child.replace(rewards=jnp.where(parent_done[:, None], 0.0, child.rewards))  # Leaves: (B, ...)
    done = child.terminated | child.truncated  # (B,)

    def predict_child() -> tuple[jax.Array, jax.Array]:
        logits, value = predict(child.observation)  # (B, A), (B,)
        chex.assert_shape(logits, child.legal_action_mask.shape)
        chex.assert_shape(value, done.shape)
        return jnp.where(done[:, None], 0.0, mask_logits(logits, child.legal_action_mask)), jnp.where(
            done, 0.0, value
        )  # (B, A), (B,)

    def zeros() -> tuple[jax.Array, jax.Array]:
        return jnp.zeros(child.legal_action_mask.shape, jnp.float32), jnp.zeros(done.shape, jnp.float32)  # (B, A), (B,)

    logits, value = jax.lax.cond(done.all(), zeros, predict_child)  # (B, A), (B,)
    chex.assert_shape(action, done.shape)
    reward = child.rewards[jnp.arange(done.shape[0]), env_state.current_player]  # (B,)
    return SearchTransition(
        reward=reward,  # (B,)
        discount=jnp.where(done, 0.0, -1.0),  # (B,)
        child_env_state=child,  # Leaves: (B, ...)
        child_logits=logits,  # (B, A)
        child_value=value,  # (B,)
    )


def expand(
    predict: PolicyValueFn,
    tree: SearchTree,  # Leaves: (B, N, ...)
    node_index: jax.Array,  # (B,)
    action: jax.Array,  # (B,)
    child_index: jax.Array,  # (B,), -1 for games not expanding.
    *,
    env: pgx.Env,
) -> SearchTree:
    """Evaluate new children and scatter updates only for expanding games."""
    rows = jnp.arange(tree.node_visits.shape[0])  # (B,)
    expanding = child_index >= 0  # (B,)
    # Distinct out-of-bounds rows drop inactive writes while keeping indices unique.
    write_rows = jnp.where(expanding, rows, rows + rows.size)  # (B,)

    def select(leaf: jax.Array) -> jax.Array:  # (B, N, ...) -> (B, ...)
        return leaf[rows, node_index]  # (B, ...)

    state = jax.tree.map(select, tree.node_env_states)  # Leaves: (B, ...)
    transition = transition_step(predict, state, action, env=env)
    chex.assert_shape(transition.child_logits, (rows.size, tree.node_logits.shape[-1]))
    chex.assert_shape((transition.reward, transition.discount, transition.child_value), rows.shape)

    def update(old: jax.Array, new: jax.Array) -> jax.Array:  # (B, N, ...), (B, ...) -> (B, N, ...)
        chex.assert_shape(new, (old.shape[0], *old.shape[2:]))
        return old.at[write_rows, jnp.maximum(child_index, 0)].set(new, mode="drop", unique_indices=True)  # (B, N, ...)

    edge = (write_rows, node_index, action)
    return tree.replace(
        node_env_states=jax.tree.map(update, tree.node_env_states, transition.child_env_state),  # Leaves: (B, N, ...)
        node_logits=update(tree.node_logits, transition.child_logits),  # (B, N, A)
        node_values=update(tree.node_values, transition.child_value),  # (B, N)
        node_visits=update(tree.node_visits, jnp.ones(rows.shape, jnp.int32)),  # (B, N)
        parent_indices=update(tree.parent_indices, node_index),  # (B, N)
        parent_actions=update(tree.parent_actions, action),  # (B, N)
        node_action_rewards=tree.node_action_rewards.at[edge].set(
            transition.reward, mode="drop", unique_indices=True
        ),  # (B, N, A)
        node_action_discounts=tree.node_action_discounts.at[edge].set(
            transition.discount, mode="drop", unique_indices=True
        ),  # (B, N, A)
        children_indices=tree.children_indices.at[edge].set(child_index, mode="drop", unique_indices=True),  # (B, N, A)
    )


def backup(tree: SearchTree, leaf_index: jax.Array) -> SearchTree:  # Tree: (B, N, ...); index: (B,)
    """Back up each leaf to its root, leaving games with shorter paths unchanged."""
    rows = jnp.arange(tree.node_visits.shape[0])  # (B,)

    def condition(carry: BackupState) -> jax.Array:  # ()
        return (carry.node_index != 0).any()  # ()

    def body(carry: BackupState) -> BackupState:
        tree, node_index, node_return = carry  # Leaves: (B, N, ...); (B,); (B,).
        active = node_index != 0  # (B,)
        parent = jnp.maximum(tree.parent_indices[rows, node_index], 0)  # (B,)
        action = jnp.maximum(tree.parent_actions[rows, node_index], 0)  # (B,)
        edge = (rows, parent, action)
        visits = tree.node_visits[rows, parent]  # (B,)
        parent_return = tree.node_action_rewards[edge] + tree.node_action_discounts[edge] * node_return  # (B,)
        mean = (tree.node_values[rows, parent] * visits + parent_return) / (visits + 1.0)  # (B,)
        write_rows = jnp.where(active, rows, rows + rows.size)  # (B,)
        write_edge = (write_rows, parent, action)
        tree = tree.replace(
            node_values=tree.node_values.at[write_rows, parent].set(mean, mode="drop", unique_indices=True),  # (B, N)
            node_visits=tree.node_visits.at[write_rows, parent].add(1, mode="drop", unique_indices=True),  # (B, N)
            children_values=tree.children_values.at[write_edge].set(
                tree.node_values[rows, node_index], mode="drop", unique_indices=True
            ),  # (B, N, A)
            node_action_visits=tree.node_action_visits.at[write_edge].add(
                1, mode="drop", unique_indices=True
            ),  # (B, N, A)
        )
        return BackupState(tree=tree, node_index=parent, node_return=parent_return)

    initial = BackupState(tree=tree, node_index=leaf_index, node_return=tree.node_values[rows, leaf_index])
    return jax.lax.while_loop(condition, body, initial).tree


def batched_puct_policy(
    predict: PolicyValueFn,
    key: jax.Array,  # (B, 2)
    env_state: pgx.State,  # Leaves: (B, ...)
    num_simulations: int,
    *,
    env: pgx.Env,
    temperature: jax.Array | float,  # ()
    dirichlet_alpha: float = 0.3,
    dirichlet_fraction: float = 0.0,
) -> PolicyOutput:
    """Search (B, ...) states with batched inference and independent per-game random streams.

    Returns (B,) actions, (B, A) normalized visit weights, and (B, N, ...) trees.
    Finished roots skip search; finished leaves are revisited without expansion.
    Zero visits fall back to uniform legal weights (all actions if none are legal).
    Temperature affects sampled actions only, not training weights.
    """
    keys = jax.vmap(partial(jax.random.split, num=3))(key)  # (B, 3, 2)
    done = env_state.terminated | env_state.truncated  # (B,)
    rows = jnp.arange(done.shape[0])  # (B,)

    def finished_predictions() -> tuple[jax.Array, jax.Array]:
        logits = jnp.zeros(env_state.legal_action_mask.shape, jnp.float32)  # (B, A)
        return mask_logits(logits, env_state.legal_action_mask), jnp.zeros(done.shape, jnp.float32)  # (B, A), (B,)

    def predict_root() -> tuple[jax.Array, jax.Array]:
        logits, value = predict(env_state.observation)  # (B, A), (B,)
        chex.assert_shape(logits, env_state.legal_action_mask.shape)
        chex.assert_shape(value, done.shape)
        logits = prepare_logits(
            logits,
            env_state.legal_action_mask,
            keys[:, 1],
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_fraction=dirichlet_fraction,
        )  # (B, A)
        fallback, _ = finished_predictions()  # (B, A), (B,)
        return jnp.where(done[:, None], fallback, logits), jnp.where(done, 0.0, value)  # (B, A), (B,)

    logits, value = jax.lax.cond(done.all(), finished_predictions, predict_root)  # (B, A), (B,)
    tree = init_search_tree(SearchRoot(env_state=env_state, logits=logits, value=value), num_simulations)

    def body(simulation: int, carry: SearchState) -> SearchState:
        keys = jax.vmap(jax.random.split)(carry.key)  # (B, 2, 2)
        selected = simulate(keys[:, 1], carry.tree)
        expanding = selected.child_index == -1  # (B,)
        new_index = jnp.where(expanding, simulation + 1, -1)  # (B,)

        def expand_leaf(tree: SearchTree) -> SearchTree:
            return expand(predict, tree, selected.node_index, selected.action, new_index, env=env)

        def keep_tree(tree: SearchTree) -> SearchTree:
            return tree

        tree = jax.lax.cond(expanding.any(), expand_leaf, keep_tree, carry.tree)
        leaf = jnp.where(expanding, new_index, selected.child_index)  # (B,)
        tree = tree.replace(
            node_visits=tree.node_visits.at[rows, leaf].add(
                (~expanding & ~done).astype(jnp.int32), unique_indices=True
            )  # (B, N)
        )
        return SearchState(key=keys[:, 0], tree=backup(tree, leaf))

    def search() -> SearchTree:
        return jax.lax.fori_loop(0, num_simulations, body, SearchState(key=keys[:, 2], tree=tree)).tree

    def keep_root() -> SearchTree:
        return tree

    tree = jax.lax.cond(done.all(), keep_root, search)
    counts = tree.node_action_visits[:, 0].astype(value.dtype)  # (B, A)
    total = counts.sum(axis=-1, keepdims=True)  # (B, 1)
    legal = env_state.legal_action_mask  # (B, A)
    fallback = (legal | ~legal.any(axis=-1, keepdims=True)).astype(counts.dtype)  # (B, A)
    fallback /= fallback.sum(axis=-1, keepdims=True)  # (B, A)
    weights = jnp.where(total > 0, counts / jnp.maximum(total, 1), fallback)  # (B, A)
    action_logits = jnp.log(jnp.maximum(weights, jnp.finfo(weights.dtype).tiny))  # (B, A)
    action_logits = (action_logits - action_logits.max(axis=-1, keepdims=True)) / jnp.maximum(
        jnp.finfo(weights.dtype).tiny, temperature
    )  # (B, A)
    action_logits = jnp.where(weights > 0, action_logits, -jnp.inf)  # (B, A)
    return PolicyOutput(
        action=jax.vmap(jax.random.categorical)(keys[:, 0], action_logits),  # (B,)
        action_weights=weights,  # (B, A)
        tree=tree,  # Leaves: (B, N, ...)
    )


def puct_policy(
    predict: PolicyValueFn,
    key: jax.Array,  # (2,)
    env_state: pgx.State,  # Leaves: (...)
    num_simulations: int,
    *,
    env: pgx.Env,
    temperature: jax.Array | float,  # ()
    dirichlet_alpha: float = 0.3,
    dirichlet_fraction: float = 0.0,
) -> PolicyOutput:
    """Adapt single-game evaluation to batched search; predict accepts (*O,) and returns (A,), ()."""

    def predict_batch(observation: jax.Array) -> tuple[jax.Array, jax.Array]:  # (1, *O) -> (1, A), (1,)
        logits, value = predict(observation[0])  # (A,), ()
        chex.assert_shape(logits, env_state.legal_action_mask.shape)
        chex.assert_shape(value, ())
        return logits[None], value[None]  # (1, A), (1,)

    def add_batch(leaf: jax.Array) -> jax.Array:  # (...) -> (1, ...)
        return leaf[None]  # (1, ...)

    def remove_batch(leaf: jax.Array) -> jax.Array:  # (1, ...) -> (...)
        return leaf[0]  # (...)

    output = batched_puct_policy(
        predict_batch,
        key[None],
        jax.tree.map(add_batch, env_state),
        num_simulations,
        env=env,
        temperature=temperature,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_fraction=dirichlet_fraction,
    )
    return jax.tree.map(remove_batch, output)


def collect_games(
    params: Parameters,
    keys: jax.Array,  # (B, 2)
    *,
    env: pgx.Env,
    apply_fn: PolicyValueApplyFn,
    config: Config,
) -> tuple[Batch, RolloutMetrics]:
    """Collect (B, T, ...) padded games and (B,) per-game counters.

    Move-limit draws supervise both heads; environment truncations exclude value
    supervision. Finished states retain final rewards throughout padding.
    """
    split_keys = jax.vmap(jax.random.split)(keys)  # (B, 2, 2)
    env_state = jax.vmap(env.init)(split_keys[:, 0])  # Leaves: (B, ...)
    move_keys = jax.vmap(partial(jax.random.split, num=config.max_moves))(split_keys[:, 1])  # (B, T, 2)

    def predict(observation: jax.Array) -> tuple[jax.Array, jax.Array]:  # (B, *O) -> (B, A), (B,)
        return apply_fn({"params": params}, observation)  # (B, A), (B,)

    def scan_body(state: pgx.State, inputs: tuple[jax.Array, jax.Array]) -> tuple[pgx.State, SelfplayStep]:
        key, move = inputs  # (B, 2), ()
        done = state.terminated | state.truncated  # (B,)

        def pad() -> tuple[pgx.State, jax.Array]:
            return state, jnp.zeros(state.legal_action_mask.shape, jnp.float32)  # Leaves: (B, ...); (B, A)

        def play() -> tuple[pgx.State, jax.Array]:
            policy = batched_puct_policy(
                predict,
                key,
                state,
                config.num_simulations,
                env=env,
                temperature=jnp.where(move < config.exploration_moves, 1.0, 0.0),
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_fraction=config.dirichlet_fraction,
            )
            return step_env(state, policy.action, env=env), jnp.where(
                done[:, None], 0.0, policy.action_weights
            )  # (B, ...), (B, A)

        next_state, weights = jax.lax.cond(done.all(), pad, play)  # Leaves: (B, ...); (B, A)
        return next_state, SelfplayStep(
            observation=state.observation.astype(jnp.float32),  # (B, *O)
            action_weights=weights,  # (B, A)
            current_player=state.current_player,  # (B,)
            policy_mask=~done,  # (B,)
        )

    final, rollout = jax.lax.scan(  # State leaves: (B, ...); rollout leaves: (T, B, ...).
        scan_body, env_state, (move_keys.swapaxes(0, 1), jnp.arange(config.max_moves, dtype=jnp.int32))
    )

    def batch_first(leaf: jax.Array) -> jax.Array:  # (T, B, ...) -> (B, T, ...)
        return leaf.swapaxes(0, 1)  # (B, T, ...)

    observation, weights, player, policy_mask = jax.tree.map(batch_first, rollout)  # Leaves: (B, T, ...)
    completed = final.terminated | ~final.truncated  # (B,)
    value_mask = policy_mask & completed[:, None]  # (B, T)
    targets = jnp.take_along_axis(final.rewards, player, axis=1)  # (B, T)
    targets = jnp.where(policy_mask & final.terminated[:, None], targets, 0.0)  # (B, T)
    return Batch(
        observation=observation,  # (B, T, *O)
        action_weights=weights,  # (B, T, A)
        value_targets=targets,  # (B, T)
        policy_mask=policy_mask,  # (B, T)
        value_mask=value_mask,  # (B, T)
    ), RolloutMetrics(
        completed_games=completed.astype(jnp.int32),  # (B,)
        environment_steps=policy_mask.sum(axis=1),  # (B,)
        value_target_count=value_mask.sum(axis=1),  # (B,)
    )


def collect_one_game(
    params: Parameters,
    key: jax.Array,  # (2,)
    *,
    env: pgx.Env,
    apply_fn: PolicyValueApplyFn,
    config: Config,
) -> tuple[Batch, RolloutMetrics]:
    """Collect one game through the batched collector, returning (T, ...) targets and () counters."""
    result = collect_games(params, key[None], env=env, apply_fn=apply_fn, config=config)

    def remove_batch(leaf: jax.Array) -> jax.Array:  # (1, ...) -> (...)
        return leaf[0]  # (...)

    return jax.tree.map(remove_batch, result)


@partial(
    jax.jit, static_argnames=("env", "apply_fn", "config"), out_shardings=(PartitionSpec("batch"), PartitionSpec())
)
def collect_selfplay(
    params: Parameters,
    key: jax.Array,  # (2,)
    *,
    env: pgx.Env,
    apply_fn: PolicyValueApplyFn,
    config: Config,
) -> tuple[Batch, RolloutMetrics]:
    """Collect (B, T, ...) targets and () counters summed across all games/devices.

    Policy masks exclude padding; value masks also exclude environment truncations.
    Games still active at max_moves are adjudicated draws with zero value targets.
    """
    params, key = jax.lax.with_sharding_constraint((params, key), PartitionSpec())  # Parameter shapes; (2,).
    keys = jax.random.split(key, config.num_envs)  # (B, 2)
    keys = jax.lax.with_sharding_constraint(keys, PartitionSpec("batch"))  # (B, 2)

    @partial(
        jax.shard_map,
        in_specs=(PartitionSpec(), PartitionSpec("batch")),
        out_specs=(PartitionSpec("batch"), PartitionSpec("batch")),
        check_vma=False,
    )
    def collect_shard(params: Parameters, keys: jax.Array) -> tuple[Batch, RolloutMetrics]:  # Keys: (b, 2)
        # b is the local game count; all inference and search stay within this shard.
        return collect_games(params, keys, env=env, apply_fn=apply_fn, config=config)

    batch, metrics = collect_shard(params, keys)  # Batch: (B, T, ...); metrics: (B,).
    return batch, jax.tree.map(jnp.sum, metrics)


def sampling_counts(policy_mask: jax.Array) -> jax.Array:  # (B, T) -> (B * T,)
    """Return replicated (B * T,) cumulative valid-position counts in game-major order."""
    # Partitioned prefix sums can become serial GPU loops, and sharded search
    # tables communicate at every binary-search probe. Replicate only the mask.
    policy_mask = jax.lax.with_sharding_constraint(policy_mask, PartitionSpec())  # (B, T)
    counts = jnp.cumsum(policy_mask, dtype=jnp.int32)  # (B * T,)
    return jax.lax.with_sharding_constraint(counts, PartitionSpec())  # (B * T,)


@partial(
    jax.jit,
    static_argnames=("batch_size",),
    out_shardings=PartitionSpec("batch"),
)
def sample_batch(
    batch: Batch,  # Leaves: (B, T, ...)
    key: jax.Array,  # (2,)
    batch_size: int,
    *,
    counts: jax.Array | None = None,  # (B * T,), precomputed by sampling_counts for this batch.
) -> Batch:
    """Sample (M, ...) positions from (B, T, ...) rollouts uniformly with replacement.

    M = batch_size, independently of B * T. At least one non-padding position must exist.
    counts: (B * T,) optional replicated cumulative counts, reused across updates.
    """
    batch = jax.lax.with_sharding_constraint(batch, PartitionSpec("batch"))  # Leaves: (B, T, ...).
    key = jax.lax.with_sharding_constraint(key, PartitionSpec())  # (2,)
    # Invert cumulative counts instead of generating a (M, B * T)
    # Gumbel array. Right insertion skips padding, including leading zero counts.
    if counts is None:
        counts = sampling_counts(batch.policy_mask)  # (B * T,)
    ranks = jax.random.randint(key, (batch_size,), 0, counts[-1])  # (M,)
    indices = jnp.searchsorted(counts, ranks, side="right")  # (M,)
    games, moves = jnp.divmod(indices, batch.policy_mask.shape[1])  # Each: (M,)

    def select(x: jax.Array) -> jax.Array:  # (B, T, ...) -> (M, ...)
        return x[games, moves]  # (M, ...)

    return jax.tree.map(select, batch)


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
    counts = sampling_counts(batch.policy_mask)  # (B * T,), shared by every update in this iteration.
    metrics = UpdateMetrics(
        loss=jnp.float32(0),  # ()
        policy_loss=jnp.float32(0),  # ()
        value_loss=jnp.float32(0),  # ()
        value_update_count=jnp.int32(0),  # ()
    )

    def body_fn(
        _: int,
        carry: TrainStepResult,
    ) -> TrainStepResult:
        train_state, key, metrics = carry
        key, sample_key = jax.random.split(key)  # Each: (2,)
        minibatch = sample_batch(batch, sample_key, batch_size, counts=counts)  # Leaves: (M, ...), M = batch_size.
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
        body_fn,
        TrainStepResult(train_state=train_state, key=key, update_metrics=metrics),
    )


def training_mesh(config: Config) -> Mesh:
    """Use all visible devices with equally sized game and minibatch shards."""
    devices = jax.devices()
    for name in ("num_envs", "batch_size"):
        size = getattr(config, name)
        if size % len(devices):
            raise ValueError(f"{name} ({size}) must be divisible by device_count ({len(devices)})")
    return Mesh(devices=devices, axis_names=("batch",), axis_types=(AxisType.Auto,))


def train(config: Config) -> TrainState:
    env = pgx.make(config.env_id)
    with (
        jax.set_mesh(training_mesh(config)),
        checkpoint_manager(config.ckpt_dir) as manager,
        MLflowLogger(experiment="alphazero", run_name=config.run_id) as writer,
    ):
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
