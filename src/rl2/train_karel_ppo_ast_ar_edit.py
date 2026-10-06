"""PPO for execution-guided autoregressive Karel AST editing.

Uses the same seed -> initial report -> action/result stream, legal masks,
score-difference rewards, and KV-cached decoding as train_karel_ast_ar_edit.
A shared Transformer supplies policy logits and a scalar critic at every
pre-action position. GAE assigns credit across individual editing decisions,
including location selection, unfinished expansions, and STOP.

Rollouts contain complete episodes. STOP and the finite sequence budget both
end the editing task, so neither bootstraps a value beyond the final decision.
Seed/report-only predictions and trailing padding have no loss. Minibatches
retain complete causal histories, with losses averaged over valid decisions.
Advantages are normalized per minibatch, and the value loss is an unclipped
half squared error, matching ppo.py. Repeated tasks are optional (group_size=1
is supported); there is no group-relative baseline.
"""

import argparse
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from functools import partial
from operator import itemgetter
from pathlib import Path
from time import monotonic
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax import linen as nn
from flax.training.train_state import TrainState
from numpy.typing import NDArray
from tensorboardX import SummaryWriter

from rl2 import train_karel_ast_ar_edit as edit
from rl2.attention import AttentionType
from rl2.jax_cache import configure_compilation_cache
from rl2.karel import REWARD_COMPONENTS, TASK_CATEGORIES, KarelConfig, KarelProgramEnv
from rl2.karel_ast import AST_ACTIONS, KarelAST
from rl2.karel_ast_edit import FEEDBACK_SIZE, Evaluation
from rl2.karel_ast_edit_vector import KarelASTEditVectorEnv
from rl2.shape_checker import ShapeChecker
from rl2.train_karel_ast_ar_edit import (
    ACTION_EVENT,
    UPDATE_EVENT,
    EditCarry,
    Events,
    History,
    act,
    empty_events,
    format_group_programs,
    initial_history,
    mask_logits,
    take_history,
)
from rl2.train_karel_ast_grpo import (
    Array,
    TrainingProgress,
    _restore_checkpoint,
    _save_checkpoint,
    learning_rate_schedule,
)
from rl2.utils import read_bytes, read_optional, write_bytes


@dataclass(frozen=True)
class Config(edit.Config):
    group_size: int = 1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_coef: float = 0.5

    def _validate_group_size(self) -> None:
        """PPO uses a critic and only needs one episode per task."""
        # The base class already validates positive integer rollout counts.

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("gamma", "gae_lambda"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not np.isfinite(self.value_coef) or self.value_coef < 0:
            raise ValueError("value_coef must be finite and nonnegative")


type ModelOutput = tuple[EditCarry, jax.Array, jax.Array]
type Metrics = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


def check_history(history: History) -> None:
    """Validate the full time-major event stream and fixed task grids."""
    sc = ShapeChecker(C=6, F=FEEDBACK_SIZE)
    sc.check([history.initial, history.target], "BHWC", dtype=np.int32)
    sc.check([history.events.kind, history.events.value], "TB", dtype=np.int32)
    sc.check(history.events.output, "TBHWC", dtype=np.int32)
    sc.check(history.events.feedback, "TBF", dtype=np.float32)


class EditActorCritic(edit.EditTransformer):
    """The editing policy's shared causal backbone with an additional value head."""

    def setup(self) -> None:
        super().setup()
        self.value_head = nn.Dense(1, dtype=self.dtype, kernel_init=nn.initializers.orthogonal(1.0))

    def __call__(self, history: History) -> ModelOutput:
        check_history(history)
        sc = ShapeChecker(
            T=history.events.kind.shape[0],
            B=history.initial.shape[0],
            D=self.d_model,
            V=1 + self.max_nodes + len(AST_ACTIONS),
        )
        inputs = self.encode_events(history.events, history.initial, history.target)
        sc.check(inputs, "TBD", dtype=self.dtype)
        x_len = jnp.full(sc["B"], inputs.shape[0], jnp.int32)
        sc.check(x_len, "B", jnp.int32)
        carry, features = self.backbone(jnp.swapaxes(inputs, 0, 1), x_len)
        features = jnp.swapaxes(features, 0, 1)
        sc.check(features, "TBD", dtype=self.dtype)
        logits = self.head(features).astype(jnp.float32)
        values = self.value_head(features)[..., 0].astype(jnp.float32)
        sc.check(logits, "TBV", dtype=jnp.float32)
        sc.check(values, "TB", dtype=jnp.float32)
        return EditCarry(carry, history.initial, history.target), logits, values

    def prefill(self, history: History) -> ModelOutput:
        carry, logits, values = self(history)
        return carry, logits[-1], values[-1]

    def step(self, event: Events, carry: EditCarry) -> ModelOutput:
        sc = ShapeChecker(C=6, F=FEEDBACK_SIZE, D=self.d_model, V=1 + self.max_nodes + len(AST_ACTIONS))
        sc.check([carry.initial, carry.target], "BHWC", dtype=jnp.int32)
        sc.check([event.kind, event.value], "B", dtype=jnp.int32)
        sc.check(event.output, "BHWC", dtype=jnp.int32)
        sc.check(event.feedback, "BF", dtype=jnp.float32)
        sequence = jax.tree.map(partial(jnp.expand_dims, axis=0), event)
        x_active = jnp.ones(sc["B"], jnp.bool_)
        sc.check(x_active, "B", jnp.bool_)
        transformer, features = self.backbone.step(
            self.encode_events(sequence, carry.initial, carry.target)[0], x_active, carry.transformer
        )
        sc.check(features, "BD", dtype=self.dtype)
        logits = self.head(features).astype(jnp.float32)
        values = self.value_head(features)[..., 0].astype(jnp.float32)
        sc.check(logits, "BV", dtype=jnp.float32)
        sc.check(values, "B", dtype=jnp.float32)
        return carry._replace(transformer=transformer), logits, values


@partial(jax.jit, static_argnames="log_compiles")
def prefill(state: TrainState, history: History, *, log_compiles: bool = False) -> ModelOutput:
    if log_compiles:
        print(f"JIT trace PPO edit prefill: events={history.events.kind.shape}", flush=True)
    return state.apply_fn({"params": state.params}, history, method=EditActorCritic.prefill)


@partial(jax.jit, static_argnames="log_compiles")
def decode_step(state: TrainState, event: Events, carry: EditCarry, *, log_compiles: bool = False) -> ModelOutput:
    if log_compiles:
        print(f"JIT trace PPO edit step: batch={event.kind.shape[0]}", flush=True)
    return state.apply_fn({"params": state.params}, event, carry, method=EditActorCritic.step)


def load_config(path: str | Path) -> Config:
    """Load local or remote YAML into validated training and environment settings."""
    settings = yaml.safe_load(read_bytes(str(path))) or {}
    # Older checkpoints stored the retired gradient-accumulation setting.
    settings.pop("replay_batch_size", None)
    if "env" in settings:
        settings["env"] = KarelConfig(**settings["env"])
    return Config(**settings)


def create_state(config: Config, initial: Array, target: Array) -> TrainState:
    """Initialize policy parameters and Adam state using the supplied task-grid shapes."""
    sc = ShapeChecker(H=config.env.height, W=config.env.width, C=6)
    sc.check([initial, target], "BHWC", dtype=jnp.int32)
    chex.assert_scalar_positive(initial.shape[0])
    model = EditActorCritic(
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        max_nodes=config.max_nodes,
        max_seq_len=config.max_seq_len,
        max_markers=config.env.max_markers,
        dtype=jnp.bfloat16 if config.bf16 else jnp.float32,
        attention_implementation=config.attention_implementation,
    )
    events = empty_events(1, initial.shape[0], config)
    events.kind[:] = UPDATE_EVENT
    params = model.init(jax.random.key(config.seed), History(initial, target, events))["params"]

    def optimizer(learning_rate: float | jax.Array) -> optax.GradientTransformation:
        """Clip the global gradient norm before applying Adam at the scheduled rate."""
        if isinstance(learning_rate, jax.Array):
            sc.check(learning_rate, "", dtype=jnp.float32)
        return optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(learning_rate, eps=1e-5))

    return TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.inject_hyperparams(optimizer)(config.learning_rate)
    )


class EditBatch(NamedTuple):
    """Array-only episode histories and policy targets, suitable for a jitted update."""

    history: History  # Events [T,B]; each row predicts the following event.
    actions: Array  # [T,B], STOP=0 is a real action wherever mask is true.
    old_log_probs: Array
    legal: Array  # [T,B,V], exactly the rollout masks; dummy STOP on non-decisions.
    mask: Array  # Only sampled policy actions, never seed/update/PAD events.
    rewards: Array  # [T,B], score differences on completed edits; zero elsewhere.
    values: Array  # [T,B], frozen rollout-time critic predictions.
    advantages: Array  # [T,B], per-decision GAE before minibatch normalization.
    returns: Array  # [T,B], frozen critic targets.


class Rollout(NamedTuple):
    """Training arrays together with host-only scores, diagnostics, and final ASTs."""

    batch: EditBatch
    rewards: NDArray[np.float32]
    diagnostics: dict[str, float]
    key: jax.Array
    programs: tuple[KarelAST, ...]


type EpisodeResults = tuple[
    EditBatch, list[Evaluation], NDArray[np.float32], NDArray[np.int32], jax.Array, tuple[KarelAST, ...]
]


def run_episodes(
    state: TrainState,
    tasks: list[KarelProgramEnv],
    envs: KarelASTEditVectorEnv,
    key: jax.Array,
    config: Config,
) -> EpisodeResults:
    """Step active episodes and feed each action together with its resulting observation.

    Completing an edit immediately refreshes the action token's image and
    feedback. Its logits choose the next action with no intervening event.
    Finished members only consume trailing PAD, with no further policy loss.
    """
    sc = ShapeChecker()
    sc.check(key, "")
    if not jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
        raise TypeError("Expected a typed JAX PRNG key from jax.random.key")
    count = len(tasks)
    if not count or envs.num_envs != count or envs.config != config.edit_config:
        raise ValueError("Expected matching nonempty tasks and editing environments")
    observations = envs.reset(tasks)
    seed_scores = np.asarray([observation.feedback[0] for observation in observations], np.float32)
    history = initial_history(observations, config)
    carry, logits, values = prefill(state, history, log_compiles=config.log_compiles)
    position = history.events.kind.shape[0] - 1
    stored = empty_events(config.max_seq_len, count, config)
    for destination, prefix in zip(stored, history.events):
        destination[: position + 1] = prefix
    shape = (config.max_seq_len, count)
    actions = np.zeros(shape, np.int32)
    old_log_probs = np.zeros(shape, np.float32)
    mask = np.zeros(shape, np.bool_)
    rewards = np.zeros(shape, np.float32)
    old_values = np.zeros(shape, np.float32)
    legal = np.zeros((*shape, 1 + config.max_nodes + len(AST_ACTIONS)), np.bool_)
    legal[..., 0] = True
    active = np.ones(count, np.bool_)
    for _ in range(config.max_seq_len - position - 1):
        ready = active.copy()
        for index in np.flatnonzero(ready):
            legal[position, index] = observations[index].legal
        key, sample_key = jax.random.split(key)
        sampled, log_probs = jax.device_get(act(mask_logits(logits, legal[position]), sample_key))
        actions[position] = np.where(ready, sampled, 0)
        old_log_probs[position] = np.where(ready, log_probs, 0)
        mask[position] = ready
        old_values[position] = np.where(ready, np.asarray(values), 0)
        transitions = envs.step(sampled, ready)
        event = jax.tree.map(itemgetter(0), empty_events(1, count, config))
        event.output[:] = np.stack([observation.output for observation in observations])
        for index in np.flatnonzero(ready):
            action = int(sampled[index])
            event.kind[index] = ACTION_EVENT
            event.value[index] = action
            transition = transitions[index]
            assert transition is not None
            rewards[position, index] = transition.reward
            observations[index] = transition.observation
            event.output[index] = transition.observation.output
            event.feedback[index] = transition.observation.feedback
            active[index] = not (transition.terminated or transition.truncated)
        if not active.any():
            break
        for destination, value in zip(stored, event):
            destination[position + 1] = value
        position += 1
        carry, logits, values = decode_step(state, event, carry, log_compiles=config.log_compiles)
    summaries = envs.summaries()
    if active.any() or not all(summary.tree.complete for summary in summaries):
        raise RuntimeError("Editing exhausted its bounded history with unfinished episodes")
    # Bucketing only adds trailing padding, which cannot affect earlier causal logits.
    length = min(config.max_seq_len, ((position + 1 + 31) // 32) * 32)
    stored.output[position + 1 :] = np.stack([observation.output for observation in observations])
    history = history._replace(events=jax.tree.map(itemgetter(slice(length)), stored))
    batch = EditBatch(
        history,
        actions[:length],
        old_log_probs[:length],
        legal[:length],
        mask[:length],
        rewards[:length],
        old_values[:length],
        np.zeros((length, count), np.float32),
        np.zeros((length, count), np.float32),
    )
    check_batch(batch)
    results = [summary.result for summary in summaries]
    completed_edits = np.asarray([summary.completed_edits for summary in summaries], np.int32)
    return batch, results, seed_scores, completed_edits, key, tuple(summary.tree for summary in summaries)


def collect_rollout(
    state: TrainState, envs: KarelASTEditVectorEnv, rng: np.random.Generator, key: jax.Array, config: Config
) -> Rollout:
    """Sample task groups, run editing episodes, and compute advantages and diagnostics."""
    count = config.num_tasks * config.group_size
    if envs.num_envs != count or envs.config != config.edit_config:
        raise ValueError("Expected num_tasks * group_size environments with configured limits")
    tasks: list[KarelProgramEnv] = []
    for seed in rng.integers(0, 2**31, size=config.num_tasks):
        task = KarelProgramEnv(config.env)
        task.reset(seed=int(seed))
        tasks.extend([task] * config.group_size)
    batch, results, seed_scores, completed_edits, key, programs = run_episodes(state, tasks, envs, key, config)
    scores = np.asarray([result.score for result in results], np.float32)
    rewards = batch.rewards.sum(axis=0)
    grouped = rewards.reshape(config.num_tasks, config.group_size)
    advantages, returns = jax.device_get(gae(batch.rewards, batch.values, batch.mask, config.gamma, config.gae_lambda))
    batch = batch._replace(advantages=advantages, returns=returns)
    successes = np.asarray([result.success for result in results])
    sequence_lengths = config.edit_config.prefill_length + batch.mask.sum(axis=0)
    # Final candidate sizes; AST depth counts edges from the root, including list nodes.
    node_count_mean = float(np.mean([len(tree.nodes) for tree in programs]))
    depth_mean = float(np.mean([max(node.depth for node in tree.nodes) for tree in programs]))
    diagnostics = {
        "value/explained_variance": explained_variance(batch.values, batch.returns, batch.mask),
        "charts/reward_mean": float(rewards.mean()),
        "charts/score_mean": float(scores.mean()),
        "charts/seed_score_mean": float(seed_scores.mean()),
        "charts/success_rate": float(successes.mean()),
        "charts/group_success_rate": float(successes.reshape(config.num_tasks, config.group_size).any(axis=1).mean()),
        "charts/reward_diverse_group_fraction": float((np.ptp(grouped, axis=1) > 0).mean()),
        "charts/edits_mean": float(completed_edits.mean()),
        "charts/sequence_budget_exhausted_rate": float(np.mean(sequence_lengths == config.max_seq_len)),
        "charts/sequence_length_mean": float(sequence_lengths.mean()),
        "charts/decisions_mean": float(batch.mask.sum()) / count,
        "charts/program_token_length_mean": float(np.mean([len(tree.tokens()) for tree in programs])),
        "charts/program_node_count_mean": node_count_mean,
        "charts/program_node_ratio_mean": node_count_mean / config.max_nodes,
        "charts/program_depth_mean": depth_mean,
        "charts/program_depth_ratio_mean": depth_mean / config.max_depth,
        **{
            f"charts/reward_{name}_mean": float(np.mean([result.components[name] for result in results]))
            for name in REWARD_COMPONENTS
        },
        **{
            f"charts/{error}_rate": float(np.mean([result.error == error for result in results]))
            for error in ("runtime_error", "execution_limit", "syntax_error", "token_limit")
        },
    }
    for category in TASK_CATEGORIES:
        stats = [
            task.sampling_stats for task in tasks[:: config.group_size] if task.sampling_stats.category == category
        ]
        diagnostics[f"sampling/{category}_fraction"] = len(stats) / config.num_tasks
        if stats:
            diagnostics[f"sampling/{category}_attempts_mean"] = float(np.mean([s.attempts for s in stats]))
            diagnostics[f"sampling/{category}_seconds_mean"] = float(np.mean([s.seconds for s in stats]))
    return Rollout(batch, rewards, diagnostics, key, programs)


def check_batch(batch: EditBatch) -> None:
    """Check replay inputs and all per-decision training targets."""
    check_history(batch.history)
    sc = ShapeChecker(T=batch.history.events.kind.shape[0], B=batch.history.initial.shape[0])
    sc.check(batch.actions, "TB", dtype=np.int32)
    sc.check(
        [batch.old_log_probs, batch.rewards, batch.values, batch.advantages, batch.returns], "TB", dtype=np.float32
    )
    sc.check(batch.mask, "TB", dtype=np.bool_)
    sc.check(batch.legal, "TBV", dtype=np.bool_)


@jax.jit
def gae(rewards: Array, values: Array, mask: Array, gamma: float, gae_lambda: float) -> tuple[jax.Array, jax.Array]:
    """GAE over contiguous decision spans, with zero bootstrap at each episode end.

    The sequence budget is part of this finite-horizon task's state. Its end
    is terminal for credit assignment even though the environment calls it a
    truncation. Non-decision entries reset the reverse scan and return zeros.
    """
    sc = ShapeChecker()
    sc.check([rewards, values], "TB", dtype=jnp.float32)
    sc.check(mask, "TB", dtype=jnp.bool_)
    rewards = jnp.where(mask, rewards, 0)
    values = jnp.where(mask, values, 0)

    def step(
        carry: tuple[jax.Array, jax.Array], transition: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        sc.check(carry, "B", dtype=jnp.float32)
        reward, value, valid = transition
        sc.check([reward, value], "B", dtype=jnp.float32)
        sc.check(valid, "B", dtype=jnp.bool_)
        next_advantage, next_value = carry
        advantage = jnp.where(valid, reward + gamma * next_value - value + gamma * gae_lambda * next_advantage, 0)
        return (advantage, value), advantage

    zeros = jnp.zeros(rewards.shape[1], jnp.float32)
    _, advantages = jax.lax.scan(step, (zeros, zeros), (rewards, values, mask), reverse=True)
    returns = jnp.where(mask, advantages + values, 0)
    sc.check([advantages, returns], "TB", dtype=jnp.float32)
    return advantages, returns


def explained_variance(values: Array, returns: Array, mask: Array) -> float:
    """Evaluate rollout-time critic predictions on actual decisions only."""
    sc = ShapeChecker()
    sc.check([values, returns], "TB", dtype=np.float32)
    sc.check(mask, "TB", dtype=np.bool_)
    values, returns = np.asarray(values)[mask], np.asarray(returns)[mask]
    if not returns.size:
        return float("nan")
    variance = np.var(returns)
    return float(1 - np.var(returns - values) / variance) if variance > 0 else float("nan")


def objective(logits: jax.Array, values: jax.Array, batch: EditBatch, config: Config) -> tuple[jax.Array, Metrics]:
    """Decision-weighted clipped PPO policy loss and un-clipped critic regression."""
    check_batch(batch)
    sc = ShapeChecker(T=batch.actions.shape[0], B=batch.actions.shape[1], V=1 + config.max_nodes + len(AST_ACTIONS))
    sc.check(logits, "TBV", dtype=jnp.float32)
    sc.check(values, "TB", dtype=jnp.float32)
    weights = batch.mask.astype(jnp.float32) / jnp.maximum(batch.mask.sum(), 1)
    advantages = jnp.where(batch.mask, jax.lax.stop_gradient(batch.advantages), 0)
    mean = jnp.sum(weights * advantages)
    centered = jnp.where(batch.mask, advantages - mean, 0)
    std = jnp.sqrt(jnp.sum(weights * jnp.square(centered)))
    advantages = centered / (std + 1e-8)
    # Ignore even nonfinite filler targets and outputs outside real decisions.
    masked_logits = mask_logits(jnp.where(batch.mask[..., None], logits, 0), batch.legal)
    policy_total, (policy, entropy, kl, clipped) = edit.objective(
        masked_logits, batch.actions, batch.old_log_probs, advantages, weights, config
    )
    error = jnp.where(batch.mask, values - jax.lax.stop_gradient(batch.returns), 0)
    value_loss = 0.5 * jnp.sum(weights * jnp.square(error))
    loss = policy_total + config.value_coef * value_loss
    metrics = (policy, value_loss, entropy, kl, clipped)
    sc.check([loss, *metrics], "", dtype=jnp.float32)
    return loss, metrics


@partial(jax.jit, static_argnames="config")
def update(state: TrainState, batch: EditBatch, config: Config) -> tuple[TrainState, Metrics]:
    """Replay complete histories and reject excessive-KL, empty, or nonfinite updates."""
    check_batch(batch)
    if config.log_compiles:
        print(f"JIT trace PPO edit update: sequence_shape={batch.actions.shape}", flush=True)

    def loss_fn(params: optax.Params) -> tuple[jax.Array, Metrics]:
        _, logits, values = state.apply_fn({"params": params}, batch.history)
        return objective(logits, values, batch, config)

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    accepted = (
        jnp.all(jnp.any(batch.mask, axis=0))
        & jnp.isfinite(loss)
        & jnp.all(jnp.isfinite(jnp.asarray(metrics)))
        & jnp.isfinite(optax.tree.norm(grads))
    )
    if config.target_kl is not None:
        accepted &= metrics[3] <= config.target_kl

    def apply_update(current: TrainState) -> TrainState:
        return current.apply_gradients(grads=grads)

    def skip_update(current: TrainState) -> TrainState:
        return current

    return jax.lax.cond(accepted, apply_update, skip_update, state), metrics


def select_episodes(batch: EditBatch, indices: NDArray[np.int64]) -> EditBatch:
    """Select episode columns without losing the causal context or PPO targets."""
    check_batch(batch)
    sc = ShapeChecker()
    sc.check(indices, "I", dtype=np.int64)
    if (
        not len(indices)
        or len(np.unique(indices)) != len(indices)
        or np.any((indices < 0) | (indices >= batch.actions.shape[1]))
    ):
        raise ValueError("Expected unique, in-range episode indices")
    selected = EditBatch(take_history(batch.history, indices), *(array[:, indices] for array in batch[1:]))
    check_batch(selected)
    return selected


def generate(state: TrainState, task: KarelProgramEnv, key: jax.Array, config: Config) -> KarelAST:
    """Edit seed_program for a freshly reset task, leaving that task untouched.

    Use load_model() to restore an editing policy, reset a matching environment,
    then call generate(). To optimize a supplied program, set seed_program to
    its source using dataclasses.replace(config, seed_program=source).
    Returns the final candidate, which need not be successful or the best visited.
    """
    sc = ShapeChecker()
    sc.check(key, "")
    if not jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
        raise TypeError("Expected a typed JAX PRNG key from jax.random.key")
    if task.config != config.env:
        raise ValueError("Task must use the configured environment limits")
    with KarelASTEditVectorEnv(config.edit_config, 1) as envs:
        _, _, _, _, _, programs = run_episodes(state, [task], envs, key, config)
    return programs[0]


def load_model(
    run_dir: str | Path, *, attention_implementation: AttentionType | None = None
) -> tuple[Config, TrainState]:
    """Load checkpoint policy weights for inference, optionally overriding the attention backend.

    Optimizer state is freshly initialized; training resume is handled by train().
    """
    directory = str(run_dir).rstrip("/")
    config = load_config(f"{directory}/config.yaml")
    if attention_implementation is not None:
        config = replace(config, attention_implementation=attention_implementation)
    dummy = np.zeros((1, config.env.height, config.env.width, 6), np.int32)
    state = create_state(config, dummy, dummy)
    data = read_bytes(f"{directory}/checkpoint.msgpack")
    restored = _restore_checkpoint(data, state, np.random.default_rng(config.seed))
    return config, state.replace(params=restored.state.params)


def train(config: Config) -> TrainState:
    """Run or resume editing-policy training with grouped rollouts, KL stopping, logs, and checkpoints."""
    configure_compilation_cache()
    batch_size = config.num_tasks * config.group_size
    rng = np.random.default_rng(config.seed)
    key = jax.random.key(config.seed)
    dummy = np.zeros((1, config.env.height, config.env.width, 6), np.int32)
    state = create_state(config, dummy, dummy)
    run_name = config.run_id or f"karel_ppo_ast_ar_edit_seed{config.seed}_{datetime.now(UTC):%Y%m%d-%H%M%S-%f}"
    config = replace(config, run_id=run_name)
    run_dir = f"{config.log_dir.rstrip('/')}/{run_name}"
    checkpoint = read_optional(f"{run_dir}/checkpoint.msgpack")
    progress = TrainingProgress(state, key, 0, 0, 0, 0)
    if checkpoint is not None:
        progress = _restore_checkpoint(checkpoint, state, rng)
        print(f"Resuming {run_dir} at rollout {progress.iteration}, step {progress.steps}", flush=True)
    else:
        if read_optional(f"{run_dir}/params.msgpack") is not None:
            raise ValueError("Run has only inference weights, not resumable training state. Use a new run_id.")
    state, key, start_iteration, steps, logged_groups, episodes = progress
    if start_iteration >= config.total_updates:
        print(f"Run already completed {start_iteration} rollouts (target {config.total_updates}).", flush=True)
        return state
    write_bytes(f"{run_dir}/config.yaml", yaml.safe_dump(asdict(config)).encode())
    if checkpoint is None:
        _save_checkpoint(run_dir, progress, rng)
    # Hide events from unsaved rollouts after an interruption, retaining the
    # checkpoint's own step. New runs do not need a TensorBoard restart marker.
    writer = SummaryWriter(logdir=run_dir, purge_step=steps + 1 if checkpoint is not None else None)
    with ExitStack() as resources:
        resources.callback(writer.close)
        envs = resources.enter_context(
            KarelASTEditVectorEnv(config.edit_config, batch_size, workers=config.env_workers)
        )
        writer.add_text("config", f"```yaml\n{yaml.safe_dump(asdict(config))}```", steps)
        writer.add_text("devices", str(jax.devices()), steps)
        parameter_count = sum(parameter.size for parameter in jax.tree.leaves(state.params))
        writer.add_scalar("model/params_millions", parameter_count / 1_000_000, steps)
        print(
            f"TensorBoard run: {run_dir}\nJAX devices: {jax.devices()}\nModel parameters: {parameter_count:,}",
            flush=True,
        )
        print(
            "Edit rewards are score differences; PPO uses per-decision GAE and a learned critic.",
            flush=True,
        )
        start = monotonic()
        last_checkpoint_time = start
        start_steps = steps
        for iteration in range(start_iteration, config.total_updates):
            rollout_start = monotonic()
            batch, rewards, diagnostics, key, programs = collect_rollout(state, envs, rng, key, config)
            rollout_seconds = monotonic() - rollout_start
            episodes += batch_size
            steps = episodes
            learning_rate = float(learning_rate_schedule(config)(iteration))
            state = state.replace(
                opt_state=state.opt_state._replace(
                    hyperparams={**state.opt_state.hyperparams, "learning_rate": jnp.asarray(learning_rate)}
                )
            )
            optimization_start = monotonic()
            metrics: list[Metrics] = []
            early_stop = False
            updates_done = 0
            for _ in range(config.update_epochs):
                for indices in np.split(rng.permutation(batch_size), config.num_minibatches):
                    minibatch = select_episodes(batch, indices)
                    previous_step = state.step
                    state, metric = update(state, minibatch, config)
                    metrics.append(metric)
                    # The compiled guard also rejects nonfinite updates. Its
                    # optimizer step is the authoritative acceptance signal.
                    if int(state.step) == int(previous_step):
                        early_stop = True
                        break
                    updates_done += 1
                if early_stop:
                    break
            log_iteration = (iteration + 1) % config.log_interval == 0
            if log_iteration:
                jax.block_until_ready((state, metrics))
                policy_loss, value_loss, entropy, approx_kl, clip_fraction = np.mean(jax.device_get(metrics), axis=0)
                for tag, scalar in {
                    **diagnostics,
                    "losses/policy": policy_loss,
                    "losses/value": value_loss,
                    "policy/entropy": entropy,
                    "policy/approx_kl": approx_kl,
                    "policy/clip_fraction": clip_fraction,
                    "policy/early_stop": early_stop,
                    "charts/learning_rate": learning_rate,
                    "charts/updates_per_rollout": updates_done,
                    "charts/total_episodes": episodes,
                    "charts/steps_per_second": (steps - start_steps) / (monotonic() - start),
                    "time/rollout_seconds": rollout_seconds,
                    "time/optimization_seconds": monotonic() - optimization_start,
                }.items():
                    writer.add_scalar(tag, float(scalar), steps)
                if config.log_program_count and (iteration + 1) % config.log_program_interval == 0:
                    samples = format_group_programs(
                        programs,
                        rewards,
                        config=config,
                        group_index=logged_groups % config.num_tasks,
                    )
                    logged_groups += 1
                    writer.add_text("samples/generated_programs", samples, steps)
            if (
                monotonic() - last_checkpoint_time >= config.checkpoint_interval_seconds
                or iteration + 1 == config.total_updates
            ):
                _save_checkpoint(
                    run_dir,
                    TrainingProgress(state, key, iteration + 1, steps, logged_groups, episodes),
                    rng,
                )
                last_checkpoint_time = monotonic()
            if log_iteration:
                writer.flush()
                print(
                    f"iteration={iteration + 1} step={steps} success={diagnostics['charts/success_rate']:.3f} "
                    f"reward={diagnostics['charts/reward_mean']:.3f} "
                    f"score={diagnostics['charts/score_mean']:.3f} "
                    f"syntax={diagnostics['charts/reward_syntax_mean']:.3f} "
                    f"runtime={diagnostics['charts/reward_runtime_mean']:.3f} "
                    f"distance={diagnostics['charts/reward_distance_mean']:.3f} "
                    f"trajectory={diagnostics['charts/reward_trajectory_mean']:.3f} "
                    f"length_penalty={diagnostics['charts/reward_length_mean']:.4f} "
                    f"execution_penalty={diagnostics['charts/reward_execution_mean']:.4f} "
                    f"reward_diverse_groups={diagnostics['charts/reward_diverse_group_fraction']:.3f} "
                    f"decisions_mean={diagnostics['charts/decisions_mean']:.3f} "
                    f"sequence_length_mean={diagnostics['charts/sequence_length_mean']:.3f} "
                    f"program_node_count_mean={diagnostics['charts/program_node_count_mean']:.3f} "
                    f"program_node_ratio_mean={diagnostics['charts/program_node_ratio_mean']:.3f} "
                    f"program_depth_mean={diagnostics['charts/program_depth_mean']:.3f} "
                    f"program_depth_ratio_mean={diagnostics['charts/program_depth_ratio_mean']:.3f} "
                    f"edits_mean={diagnostics['charts/edits_mean']:.3f} "
                    f"sequence_budget_exhausted_rate={diagnostics['charts/sequence_budget_exhausted_rate']:.3f} "
                    f"policy={policy_loss:.3f} value={value_loss:.3f} entropy={entropy:.3f} kl={approx_kl:.4f} "
                    f"updates={updates_done} early_stop={early_stop}",
                    flush=True,
                )
        return state


def main() -> None:
    """Parse the config path and launch the autoregressive AST editing trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/karel_ppo_ast_ar_edit.yaml", help="Path to a YAML config")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
