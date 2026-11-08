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
"""Data-parallel AlphaZero sketch using PGX rules and Mctx PUCT search.

Run: uv run python -m rl2.alphazero.train --config configs/alphazero/chess.yaml
For a cheap start, set env_id: tic_tac_toe and max_moves: 9 in the YAML config.

Training always uses a mesh over all visible JAX devices, including one device.
Self-play games and training batches are sharded on the batch axis; parameters,
optimizer state, RNG keys, and scalar metrics are fully replicated. num_envs and
batch_size are global sizes and must be divisible by the device count. Losses
average eligible positions globally, before gradient clipping and optimization.
Standalone collect_selfplay, sample_batch, and train_step calls require a
jax.set_mesh context with an Auto batch axis. Evaluation retains its fixed suite.

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
import mctx
import optax
import pgx
import yaml
from flax import struct
from flax.training.train_state import TrainState
from jax.sharding import AxisType, Mesh, PartitionSpec
from mctx._src import search as mctx_search
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
type SearchTree = mctx.Tree[None]
type SearchState = tuple[jax.Array, SearchTree]


class TrainMetrics(NamedTuple):
    """Per-update losses averaged over eligible positions and value-update count."""

    loss: jax.Array  # ()
    policy_loss: jax.Array  # ()
    value_loss: jax.Array  # ()
    value_updates: jax.Array  # () int32; one when the update has value targets, otherwise zero.


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
    # B: batch size; A: number of actions; observation_shape: environment observation dimensions.
    observations: jax.Array  # (B, *observation_shape)
    policy_targets: jax.Array  # (B, A)
    value_targets: jax.Array  # (B,)
    policy_mask: jax.Array  # (B,)
    value_mask: jax.Array  # (B,)


class SelfplayStep(NamedTuple):
    observations: jax.Array  # (B, *observation_shape)
    policies: jax.Array  # (B, A)
    players: jax.Array  # (B,)
    valid: jax.Array  # (B,)


class PolicyValueFn(Protocol):
    """Bound model mapping observations to policy logits and player-to-move values."""

    def __call__(self, observation: jax.Array, /) -> tuple[jax.Array, jax.Array]:  # (B, H, W, C) -> (B, A), (B,)
        """Map (B, H, W, C) observations to (B, A) float32 logits and (B,) float32 values."""
        ...


def masked_logits(
    logits: jax.Array,  # (..., A)
    legal: jax.Array,  # (..., A)
) -> jax.Array:  # (..., A)
    """Shift logits by their row maximum and mask illegal actions.

    Args:
        logits: (..., A) floating-point logits over A actions.
        legal: (..., A) boolean mask indicating legal actions.

    Returns:
        (..., A) shifted logits with illegal actions set to the minimum finite
        float32 value. Rows with no legal actions contain only this finite value.
    """
    # Finite masking also keeps terminal nodes with no legal moves well-defined.
    return jnp.where(legal, logits - logits.max(axis=-1, keepdims=True), jnp.finfo(jnp.float32).min)  # (..., A)


def recurrent_step(
    model: PolicyValueFn,
    actions: jax.Array,  # (B,)
    states: pgx.State,  # Leaves: (B, ...)
    *,
    env: pgx.Env,
) -> tuple[mctx.RecurrentFnOutput, pgx.State]:
    """Expand batched Mctx nodes for deterministic, alternating-turn, two-player games.

    Args:
        model: Bound observation-to-(logits, value) callable, forwarded through
            Mctx's params argument. Values use the player-to-move perspective.
        actions: (B,) int32 action IDs.
        states: (B, ...) array leaves forming a batched PGX state pytree.
        env: Game rules and action/observation encodings.

    Returns:
        (output, next_states), where output contains:
            reward: (B,) float32 immediate reward for the parent state's acting player.
            discount: (B,) float32; -1 for ongoing games, 0 on termination or truncation.
            prior_logits: (B, A) float32 masked child policy logits, A = env.num_actions.
            value: (B,) float32 child player-to-move value, zero on termination or truncation.
        next_states: (B, ...) array leaves forming the child PGX state pytree.
        Backup uses reward + discount * child_value, negating the opponent's value.
    """
    next_states = jax.vmap(env.step)(states, actions)  # Leaves: (B, ...)
    finished = states.terminated | states.truncated  # (B,)

    def preserve_finished(old: jax.Array, new: jax.Array) -> jax.Array:  # Each: (B, ...)
        return jnp.where(finished.reshape(finished.shape + (1,) * (old.ndim - 1)), old, new)  # (B, ...)

    # PGX checks action legality even after a game has ended. Keep finished
    # nodes absorbing, including truncated states whose mask still has illegal moves.
    next_states = jax.tree.map(
        preserve_finished, states.replace(rewards=jnp.zeros_like(states.rewards)), next_states
    )  # Leaves: (B, ...)
    logits, value = model(next_states.observation)  # (B, A), (B,)
    # Check before masking: where would silently broadcast a scalar or (1,) value.
    chex.assert_shape(logits, (states.current_player.shape[0], env.num_actions))
    chex.assert_shape(value, states.current_player.shape)
    reward = jnp.take_along_axis(next_states.rewards, states.current_player[:, None], axis=1)[:, 0]  # (B,)
    done = next_states.terminated | next_states.truncated  # (B,)
    output = mctx.RecurrentFnOutput(
        reward=reward,  # (B,)
        discount=jnp.where(done, 0.0, -1.0),  # (B,)
        prior_logits=masked_logits(logits, next_states.legal_action_mask),  # (B, A)
        value=jnp.where(done, 0.0, value),  # (B,)
    )
    return output, next_states


def _backward(
    tree: SearchTree,  # Leaves: (B, N, ...)
    leaf_index: jax.Array,  # (B,)
) -> SearchTree:  # Leaves: (B, N, ...)
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
    batch = jnp.arange(leaf_index.shape[0])  # (B,)

    def ongoing(carry: BackupState) -> jax.Array:  # ()
        return jnp.any(carry.index != mctx.Tree.ROOT_INDEX)  # ()

    def advance(carry: BackupState) -> BackupState:
        tree, leaf_value, index = carry  # Tree leaves: (B, N, ...); values and indices: (B,)
        active = index != mctx.Tree.ROOT_INDEX  # (B,)
        # Inactive games use valid dummy indices and retain the old entries.
        parent = jnp.where(active, tree.parents[batch, index], 0)  # (B,)
        action = jnp.where(active, tree.action_from_parent[batch, index], 0)  # (B,)
        count = tree.node_visits[batch, parent]  # (B,)
        leaf_value = (
            tree.children_rewards[batch, parent, action] + tree.children_discounts[batch, parent, action] * leaf_value
        )  # (B,)
        parent_value = (tree.node_values[batch, parent] * count + leaf_value) / (count + 1.0)  # (B,)
        tree = tree.replace(
            node_values=tree.node_values.at[batch, parent].set(
                jnp.where(active, parent_value, tree.node_values[batch, parent])
            ),  # (B, N)
            node_visits=tree.node_visits.at[batch, parent].set(count + active),  # (B, N)
            children_values=tree.children_values.at[batch, parent, action].set(
                jnp.where(active, tree.node_values[batch, index], tree.children_values[batch, parent, action])
            ),  # (B, N, A)
            children_visits=tree.children_visits.at[batch, parent, action].add(active.astype(jnp.int32)),  # (B, N, A)
        )  # Leaves: (B, N, ...)
        return BackupState(tree=tree, leaf_value=leaf_value, index=parent)

    initial = BackupState(tree=tree, leaf_value=tree.node_values[batch, leaf_index], index=leaf_index)
    return jax.lax.while_loop(ongoing, advance, initial).tree  # Leaves: (B, N, ...)


def puct_policy(
    model: PolicyValueFn,
    key: jax.Array,  # (2,)
    root: mctx.RootFnOutput,  # Leaves: (B, ...)
    recurrent_fn: mctx.RecurrentFn,
    num_simulations: int,
    *,
    invalid_actions: jax.Array,  # (B, A)
    temperature: jax.Array | float,  # ()
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
    action_key, _, search_key = jax.random.split(key, 3)  # Each: (2,)
    tiny = jnp.finfo(root.prior_logits.dtype).tiny
    logits = jnp.log(jnp.maximum(jax.nn.softmax(root.prior_logits), tiny))  # (B, A)
    logits -= logits.max(axis=-1, keepdims=True)  # (B, A)
    root = root.replace(prior_logits=jnp.where(invalid_actions, jnp.finfo(logits.dtype).min, logits))  # (B, A) logits
    batch_size = root.value.shape[0]
    batch = jnp.arange(batch_size)  # (B,)

    def simulate(simulation: int, carry: SearchState) -> SearchState:
        key, tree = carry  # Key: (2,); tree leaves: (B, N, ...)
        key, simulate_key, expand_key = jax.random.split(key, 3)  # Each: (2,)
        parent, action = mctx_search.simulate(
            jax.random.split(simulate_key, batch_size), tree, mctx.muzero_action_selection, num_simulations
        )  # Each: (B,)
        next_node = tree.children_index[batch, parent, action]  # (B,)
        next_node = jnp.where(next_node == mctx.Tree.UNVISITED, simulation + 1, next_node)  # (B,)
        tree = mctx_search.expand(
            model, expand_key, tree, recurrent_fn, parent, action, next_node
        )  # Leaves: (B, N, ...)
        return key, _backward(tree, next_node)

    # Mctx exposes no backup hook. Reuse its internal allocation, traversal and
    # expansion helpers; reference-policy tests guard this small compatibility
    # surface when the dependency is upgraded.
    tree = mctx_search.instantiate_tree_from_root(
        root, num_simulations, invalid_actions, extra_data=None
    )  # Leaves: (B, N, ...)
    _, tree = jax.lax.fori_loop(0, num_simulations, simulate, (search_key, tree))  # Tree leaves: (B, N, ...)
    weights = tree.summary().visit_probs  # (B, A)
    logits = jnp.log(jnp.maximum(weights, tiny))  # (B, A)
    logits = (logits - logits.max(axis=-1, keepdims=True)) / jnp.maximum(tiny, temperature)  # (B, A)
    return mctx.PolicyOutput(
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

    B: games; T: max_moves; A: actions; O: environment observation dimensions.
    Environment truncations remain excluded from value supervision.

    Args:
        params: Model parameter pytree.
        key: (2,) uint32 PRNG key.
        env: Game rules and encodings.
        apply_fn: Policy/value apply function from the training state.
        config: Self-play settings.

    Returns:
        Batch and metrics, with fields:
            observations: (B * T, *O) float32 recorded observations, grouped by game.
            policy_targets: (B * T, A) float32 MCTS action weights.
            value_targets: (B * T,) float32 player-relative outcomes, zero for cutoff draws.
            policy_mask: (B * T,) bool non-padding positions.
            value_mask: (B * T,) bool positions with terminal or adjudicated outcomes.
            completed_games: () integer count including move-limit draws.
            positions: () integer count of non-padding positions.
            value_positions: () integer count of positions with outcome targets.
    """
    params, key = jax.lax.with_sharding_constraint((params, key), PartitionSpec())  # Parameter shapes; key: (2,).
    init_key, play_key = jax.random.split(key)  # Each: (2,)
    states = jax.vmap(env.init)(jax.random.split(init_key, config.num_envs))  # Leaves: (B, ...)
    states = jax.lax.with_sharding_constraint(states, PartitionSpec("batch"))  # Leaves: (B, ...)
    predict = partial(apply_fn, {"params": params})

    def recurrent_fn(
        model: PolicyValueFn,
        _key: jax.Array,  # (2,)
        actions: jax.Array,  # (B,)
        states: pgx.State,  # Leaves: (B, ...)
    ) -> tuple[mctx.RecurrentFnOutput, pgx.State]:
        # Mctx requires a PRNG key even for deterministic transitions.
        return recurrent_step(model, actions, states, env=env)

    def step(
        states: pgx.State,  # Leaves: (B, ...)
        inputs: tuple[jax.Array, jax.Array],  # (2,) key and () move index.
    ) -> tuple[pgx.State, SelfplayStep]:
        key, move_i = inputs  # key: (2,); move_i: () scalar move index.
        noise_key, search_key = jax.random.split(key)  # Each: (2,)
        logits, value = predict(states.observation)  # (B, A), (B,)
        done = states.terminated | states.truncated  # (B,)
        root = mctx.RootFnOutput(
            prior_logits=selfplay_prior_logits(
                logits,
                states.legal_action_mask,
                noise_key,
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_fraction=config.dirichlet_fraction,
            ),  # (B, A)
            value=jnp.where(done, 0.0, value),  # (B,)
            embedding=states,  # Leaves: (B, ...)
        )
        # PGX supplies exact transitions for Mctx's PUCT, with no learned dynamics.
        policy = puct_policy(
            predict,
            search_key,
            root,
            recurrent_fn,
            config.num_simulations,
            invalid_actions=~states.legal_action_mask,
            temperature=jnp.where(move_i < config.exploration_moves, 1.0, 0.0),
        )
        next_states = jax.vmap(env.step)(states, policy.action)  # Leaves: (B, ...)

        def preserve_finished(
            old: jax.Array,  # (B, ...)
            new: jax.Array,  # (B, ...)
        ) -> jax.Array:  # (B, ...)
            """Preserve (B, ...) state leaves for games that have already stopped."""
            # Preserve final rewards: PGX otherwise clears them on later steps.
            return jnp.where(done.reshape((config.num_envs,) + (1,) * (old.ndim - 1)), old, new)  # (B, ...)

        next_states = jax.tree.map(preserve_finished, states, next_states)  # Leaves: (B, ...)

        return next_states, SelfplayStep(
            observations=states.observation.astype(jnp.float32),  # (B, *O)
            policies=policy.action_weights,  # (B, A)
            players=states.current_player,  # (B,)
            valid=~done,  # (B,)
        )

    # Recorded arrays: (T, B, *O), (T, B, A), (T, B), (T, B).
    states, (observations, policies, players, valid) = jax.lax.scan(
        step,
        states,
        (jax.random.split(play_key, config.max_moves), jnp.arange(config.max_moves, dtype=jnp.int32)),
    )
    # T: recorded time steps; B: games. Map final rewards to each recorded actor.
    values = states.rewards[jnp.arange(config.num_envs)[None, :], players]  # (T, B)
    move_limit_draws = ~(states.terminated | states.truncated)  # (B,)
    completed = states.terminated | move_limit_draws  # (B,)
    value_mask = valid & completed[None, :]  # (T, B)
    # A win on the final allowed move keeps its outcome; only active games draw.
    values = jnp.where(valid & states.terminated[None, :], values, 0.0)  # (T, B)
    batch = Batch(
        observations=observations,  # (T, B, *O)
        policy_targets=policies,  # (T, B, A)
        value_targets=values,  # (T, B)
        policy_mask=valid,  # (T, B)
        value_mask=value_mask,  # (T, B)
    )

    def flatten(x: jax.Array) -> jax.Array:  # (T, B, ...) -> (B * T, ...)
        # Keep each game's positions on its device when flattening the rollout.
        return x.swapaxes(0, 1).reshape((-1, *x.shape[2:]))  # (B * T, ...)

    return jax.tree.map(flatten, batch), {
        "completed_games": completed.sum(),  # ()
        "positions": valid.sum(),  # ()
        "value_positions": value_mask.sum(),  # ()
    }


def loss_fn(
    params: Parameters,
    apply_fn: PolicyValueApplyFn,
    batch: Batch,  # Leaves: (B, ...).
) -> tuple[jax.Array, TrainMetrics]:  # Loss and metric leaves: ().
    """Return () loss and scalar metrics; each head averages its eligible positions."""
    logits, values = apply_fn({"params": params}, batch.observations)  # (B, A), (B,)
    policy_errors = optax.softmax_cross_entropy(logits, batch.policy_targets)  # (B,)
    value_errors = jnp.square(values - batch.value_targets)  # (B,)
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
    static_argnames=("batch_size",),
    out_shardings=PartitionSpec(),
)
def train_step(
    train_state: TrainState,
    batch: Batch,  # Leaves: (N, ...), rollout positions including padding.
    key: jax.Array,  # (2,)
    *,
    batch_size: int,
) -> tuple[TrainState, TrainMetrics]:
    """Sample a minibatch and update replicated parameters using globally masked means."""
    train_state = jax.lax.with_sharding_constraint(train_state, PartitionSpec())
    minibatch = sample_batch(batch, key, batch_size)  # Leaves: (B, ...), B = batch_size.
    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params, train_state.apply_fn, minibatch)
    grads = jax.lax.with_sharding_constraint(grads, PartitionSpec())  # Parameter shapes.
    return train_state.apply_gradients(grads=grads), metrics


@partial(
    jax.jit,
    static_argnames=("batch_size",),
    out_shardings=PartitionSpec("batch"),
)
def sample_batch(batch: Batch, key: jax.Array, batch_size: int) -> Batch:  # key: (2,)
    """Sample real positions uniformly with replacement; at least one must exist."""
    batch = jax.lax.with_sharding_constraint(batch, PartitionSpec("batch"))  # Leaves: (N, ...).
    key = jax.lax.with_sharding_constraint(key, PartitionSpec())  # (2,)
    # Invert cumulative counts instead of generating a (batch_size, rollout_size)
    # Gumbel array. Right insertion skips padding, including leading zero counts.
    counts = jnp.cumsum(batch.policy_mask, dtype=jnp.int32)  # (N,) rollout positions including padding.
    ranks = jax.random.randint(key, (batch_size,), 0, counts[-1])  # (B,)
    indices = jnp.searchsorted(counts, ranks, side="right")  # (B,)

    def select(x: jax.Array) -> jax.Array:  # (N, ...) -> (B, ...)
        return x[indices]  # (B, ...)

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
        # initialize
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
        for iteration in range(start_iteration, config.iterations):
            iteration_start = monotonic()
            key, play_key = jax.random.split(key)  # Each: (2,)
            batch, rollout_metrics = collect_selfplay(
                train_state.params, play_key, env=env, apply_fn=train_state.apply_fn, config=config
            )
            update_metrics = TrainMetrics(
                loss=jnp.float32(0),  # ()
                policy_loss=jnp.float32(0),  # ()
                value_loss=jnp.float32(0),  # ()
                value_updates=jnp.int32(0),  # ()
            )  # Scalar leaves: ().
            for _ in range(config.updates_per_iteration):
                key, sample_key = jax.random.split(key)  # Each: (2,)
                train_state, metrics = train_step(train_state, batch, sample_key, batch_size=config.batch_size)
                update_metrics = RunningSum.update(update_metrics, metrics)  # Scalar leaves: ().
            iteration_metrics = WindowMetrics(
                update_metrics=update_metrics,  # Scalar leaves: ().
                completed_games=rollout_metrics["completed_games"],  # ()
                positions=rollout_metrics["positions"],  # ()
                value_positions=rollout_metrics["value_positions"],  # ()
                iterations=jnp.int32(1),  # ()
                iteration_seconds=jnp.float32(monotonic() - iteration_start),  # ()
            )  # Scalar leaves: ().
            window_metrics = RunningSum.update(window_metrics, iteration_metrics)  # Scalar leaves: ().
            log_due = (iteration + 1) % config.log_interval == 0 or iteration + 1 == config.iterations
            eval_due = (iteration + 1) % config.eval_interval == 0
            checkpoint_due = (iteration + 1) % config.checkpoint_interval == 0 or iteration + 1 == config.iterations
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
                    "train/iteration": iteration + 1,
                    "train/progress_fraction": (iteration + 1) / config.iterations,
                    "rollout/completed_games_total": completed_games,
                    "train/learning_rate": config.learning_rate,
                    "train/environment_steps_per_second": window_rollout["positions"] / max(window_seconds, 1e-9),
                    "train/iteration_seconds_mean": float(host_metrics.iteration_seconds) / window_iterations,
                    "train/window_seconds": window_seconds,
                }
                writer.log_metrics(scalars, steps)
                logger.info(
                    "Iteration %d/%d | steps=%d | %s | seconds=%.2f",
                    iteration + 1,
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
                        candidate_iteration=iteration + 1,
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
                        iteration=iteration + 1,
                        steps=steps,
                        completed_games=completed_games,
                    ),
                    config,
                )
                opponent_params = train_state.params
                opponent_iteration = iteration + 1
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
