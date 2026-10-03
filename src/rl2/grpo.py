"""GRPO over grammar-constrained AST expansions for Karel program synthesis.

Each pair has a group of independently generated ASTs. Each round expands all
current holes from one shared tree encoding; new children wait for the next round.
Completed trees are printed and submitted to KarelProgramEnv for its unchanged
terminal rewards. Rollouts store the partial tree BEFORE each round, including
its per-hole masks. Updates replay those states and normalize losses per program.
Round microbatches accumulate gradients before one optimizer update, bounding
memory without changing program weighting or clipping gradients per microbatch.
"""

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import monotonic
from typing import Any, NamedTuple

import chex
import gcsfs
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax import serialization
from flax.training.train_state import TrainState
from numpy.typing import NDArray
from tensorboardX import SummaryWriter

from rl2.ast_transformer import ASTTransformer
from rl2.karel import ACTIONS, REWARD_COMPONENTS, TOKEN_TO_ID, KarelConfig, KarelProgramEnv
from rl2.karel_ast import AST_ACTIONS, ASTFeatures, KarelAST, batch_features
from rl2.transformer import AttentionImplementation

type Array = jax.Array | NDArray[Any]
type Grids = Array
type Metrics = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


@dataclass(frozen=True)
class Config:
    seed: int = 1
    total_updates: int = 1000  # Number of rollout batches, independent of program length.
    num_tasks: int = 8
    group_size: int = 8
    num_minibatches: int = 4
    update_epochs: int = 2
    d_model: int = 320
    num_layers: int = 7
    max_nodes: int = 128
    max_depth: int = 64
    decision_batch_size: int = 32  # Partial-tree rounds per microbatch; each predicts all holes.
    num_heads: int = 5
    num_kv_heads: int | None = None
    bf16: bool = False
    attention_implementation: AttentionImplementation = "xla"
    learning_rate: float = 0.00025
    anneal_lr: bool = True
    clip_coef: float = 0.2
    target_kl: float | None = 0.02
    entropy_coef: float = 0.0
    kl_coef: float = 0.0  # Zero disables the frozen-reference KL penalty.
    max_grad_norm: float = 0.5
    log_dir: str = "runs"
    run_id: str | None = None  # Same ID resumes; None creates a timestamped run.
    checkpoint_interval_seconds: float = 300.0  # Save at the next completed rollout boundary.
    log_program_interval: int = 10  # First rollout, then every N rollouts.
    log_program_count: int = 8  # First N programs in one group; zero disables samples.
    env: KarelConfig = field(default_factory=KarelConfig)

    def __post_init__(self) -> None:
        if self.run_id is not None:
            if not isinstance(self.run_id, str):
                raise TypeError("run_id must be a string or null")
            if not self.run_id.strip() or self.run_id in (".", "..") or any(c in self.run_id for c in "/\\"):
                raise ValueError("run_id must be a nonempty directory name, without slashes or traversal")
        chex.assert_scalar_positive(self.checkpoint_interval_seconds)
        if not np.isfinite(self.checkpoint_interval_seconds):
            raise ValueError("checkpoint_interval_seconds must be finite")
        KarelAST.empty(self.max_nodes, self.max_depth, self.env.max_program_tokens)
        if type(self.decision_batch_size) is not int:
            raise TypeError("decision_batch_size must be an integer")
        chex.assert_scalar_positive(self.decision_batch_size)
        if type(self.bf16) is not bool:
            raise TypeError("bf16 must be a boolean")
        if self.attention_implementation not in ("xla", "cudnn"):
            raise ValueError("attention_implementation must be 'xla' or 'cudnn'")
        if self.attention_implementation == "cudnn" and not self.bf16:
            raise ValueError("cuDNN attention requires bf16=True")
        for value in (self.log_program_interval, self.log_program_count):
            if type(value) is not int:
                raise TypeError("Program logging settings must be integers")
        chex.assert_scalar_positive(self.log_program_interval)
        chex.assert_scalar_non_negative(self.log_program_count)
        for value in (self.total_updates, self.num_tasks, self.group_size, self.num_minibatches, self.update_epochs):
            if type(value) is not int:
                raise TypeError("Rollout and update counts must be integers")
            chex.assert_scalar_positive(value)
        if self.group_size < 2:
            raise ValueError("GRPO requires group_size >= 2")
        if self.num_tasks * self.group_size % self.num_minibatches:
            raise ValueError("num_minibatches must divide num_tasks * group_size")
        chex.assert_type(self.seed, int)
        chex.assert_scalar_non_negative(self.seed)
        chex.assert_scalar_in(self.clip_coef, 0, 1, included=False)
        for value in (self.learning_rate, self.max_grad_norm):
            chex.assert_scalar_positive(value)
            if not np.isfinite(value):
                raise ValueError("Optimizer settings must be finite")
        for value in (self.entropy_coef, self.kl_coef):
            chex.assert_scalar_non_negative(value)
            if not np.isfinite(value):
                raise ValueError("Loss coefficients must be finite")
        if self.target_kl is not None:
            chex.assert_scalar_positive(self.target_kl)
            if not np.isfinite(self.target_kl):
                raise ValueError("target_kl must be finite or null")

    @property
    def max_rounds(self) -> int:
        """Each round resolves one AST depth level; no more rounds than nodes."""
        return min(self.max_nodes, self.max_depth + 1)


class GRPOBatch(NamedTuple):
    initial: Array  # [B,H,W,6], grouped by task during collection.
    target: Array
    tree: ASTFeatures  # [R,B,N] features, [R,B,N,A] masks; snapshots BEFORE each round.
    actions: Array  # [R,B,N] constructor/value IDs; PAD=0 at non-hole positions.
    old_log_probs: Array
    mask: Array  # Includes the final expansion, excludes PAD.
    advantages: Array  # [B], normalized within task groups before minibatching.


def _read_bytes(path: str) -> bytes:
    if path.startswith("gs://"):
        return gcsfs.GCSFileSystem().cat_file(path)
    return Path(path).read_bytes()


def _write_bytes(path: str, data: bytes) -> None:
    if path.startswith("gs://"):
        gcsfs.GCSFileSystem().pipe_file(path, data)
    else:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Commit a whole checkpoint, never truncate the previous good file.
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as f:
                temporary = Path(f.name)
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _read_optional(path: str) -> bytes | None:
    """Only a missing object means a new run; storage/authentication errors propagate."""
    try:
        return _read_bytes(path)
    except FileNotFoundError:
        return None


class TrainingProgress(NamedTuple):
    state: TrainState
    reference_params: optax.Params | None
    key: jax.Array
    iteration: int  # Number of fully completed rollout batches.
    steps: int  # Cumulative sampled hole decisions, for TensorBoard.
    logged_groups: int
    episodes: int


def _save_checkpoint(run_dir: str, progress: TrainingProgress, rng: np.random.Generator) -> None:
    """Save a complete rollout boundary; checkpoint.msgpack is authoritative."""
    chex.assert_shape(progress.key, ())
    payload = {
        "version": 1,
        "state": serialization.to_state_dict(progress.state),
        "reference_params": serialization.to_state_dict(progress.reference_params),
        "key": np.asarray(jax.random.key_data(progress.key)),
        "key_impl": str(jax.random.key_impl(progress.key)),
        # PCG64 contains 128-bit integers, which cannot be encoded directly by msgpack.
        "numpy_rng": json.dumps(rng.bit_generator.state),
        "iteration": progress.iteration,
        "steps": progress.steps,
        "logged_groups": progress.logged_groups,
        "episodes": progress.episodes,
    }
    _write_bytes(f"{run_dir}/checkpoint.msgpack", serialization.msgpack_serialize(payload))


def _restore_checkpoint(data: bytes, config: Config, state: TrainState, rng: np.random.Generator) -> TrainingProgress:
    payload = serialization.msgpack_restore(data)
    if payload["version"] != 1:
        raise ValueError("Unsupported GRPO checkpoint version")
    try:
        restored = serialization.from_state_dict(state, payload["state"])
        chex.assert_trees_all_equal_shapes_and_dtypes(
            (state.params, state.opt_state), (restored.params, restored.opt_state)
        )
    except (ValueError, AssertionError) as error:
        raise ValueError(
            "Checkpoint model/optimizer structure is incompatible with this config. Use a new run_id."
        ) from error
    reference = payload["reference_params"]
    if not config.kl_coef:
        reference = None
    elif reference is None:
        # Enabling KL on a previously unregularized run anchors it at resume.
        reference = restored.params
    else:
        reference = serialization.from_state_dict(state.params, reference)
        chex.assert_trees_all_equal_shapes_and_dtypes(reference, state.params)
    chex.assert_rank(payload["key"], 1)
    chex.assert_type(payload["key"], np.uint32)
    key = jax.random.wrap_key_data(jnp.asarray(payload["key"]), impl=payload["key_impl"])
    counters = tuple(payload[name] for name in ("iteration", "steps", "logged_groups", "episodes"))
    for counter in counters:
        if type(counter) is not int or counter < 0:
            raise ValueError("Invalid checkpoint counters")
    rng.bit_generator.state = json.loads(payload["numpy_rng"])
    return TrainingProgress(restored, reference, key, *counters)


def load_config(path: str | Path) -> Config:
    settings = yaml.safe_load(_read_bytes(str(path))) or {}
    if "env" in settings:
        settings["env"] = KarelConfig(**settings["env"])
    return Config(**settings)


def learning_rate_schedule(config: Config) -> optax.Schedule:
    return optax.linear_schedule(
        config.learning_rate,
        0.0 if config.anneal_lr else config.learning_rate,
        config.total_updates,
    )


def create_state(config: Config, initial: Grids, target: Grids) -> TrainState:
    chex.assert_shape(initial, (None, None, None, 6))
    chex.assert_equal_shape((initial, target))
    chex.assert_type((initial, target), jnp.int32)
    model = ASTTransformer(
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        max_nodes=config.max_nodes,
        max_depth=config.max_depth,
        max_markers=config.env.max_markers,
        dtype=jnp.bfloat16 if config.bf16 else jnp.float32,
        attention_implementation=config.attention_implementation,
    )
    features = batch_features(
        tuple(
            KarelAST.empty(config.max_nodes, config.max_depth, config.env.max_program_tokens).features()
            for _ in initial
        )
    )
    params = model.init(jax.random.key(config.seed), initial, target, features)["params"]

    def optimizer(learning_rate: float | jax.Array) -> optax.GradientTransformation:
        return optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(learning_rate, eps=1e-5))

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.inject_hyperparams(optimizer)(config.learning_rate),
    )


@jax.jit
def group_advantages(rewards: Array) -> jax.Array:
    """Normalize within each task, preserving exactly zero advantages for ties."""
    chex.assert_rank(rewards, 2)
    chex.assert_type(rewards, jnp.float32)
    chex.assert_scalar_positive(rewards.shape[1] - 1)
    # Subtract a member first: reducing identical fractional rewards directly can
    # round their mean away from that value and create a spurious advantage.
    shifted = rewards - rewards[:, :1]
    centered = shifted - shifted.mean(axis=1, keepdims=True)
    std = jnp.sqrt(jnp.square(centered).mean(axis=1, keepdims=True))
    return centered / (std + 1e-8)


def generation_logits(logits: jax.Array) -> jax.Array:
    """Exclude PAD on live rows; retain a constant PAD fallback on finished rows."""
    chex.assert_type(logits, jnp.float32)
    chex.assert_scalar_positive(logits.shape[-1] - 1)
    live = jnp.isfinite(logits[..., 1:]).any(axis=-1)
    return logits.at[..., 0].set(jnp.where(live, -jnp.inf, 0.0))


def action_log_prob(logits: jax.Array, actions: Array) -> jax.Array:
    chex.assert_shape(logits, (*actions.shape, len(AST_ACTIONS)))
    chex.assert_type(logits, jnp.float32)
    chex.assert_type(actions, jnp.int32)
    log_probs = jax.nn.log_softmax(generation_logits(logits))
    selected = jnp.take_along_axis(log_probs, actions[..., None], axis=-1)[..., 0]
    return jnp.where(actions == 0, 0.0, selected)


@jax.jit
def predict(state: TrainState, initial: Array, target: Array, tree: ASTFeatures) -> jax.Array:
    return state.apply_fn({"params": state.params}, initial, target, tree)


@jax.jit
def act(logits: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
    chex.assert_shape(logits, (None, None, len(AST_ACTIONS)))
    chex.assert_type(logits, jnp.float32)
    actions = jax.random.categorical(key, generation_logits(logits)).astype(jnp.int32)
    return actions, action_log_prob(logits, actions)


def collect_rollout(
    state: TrainState, envs: list[KarelProgramEnv], rng: np.random.Generator, key: jax.Array, config: Config
) -> tuple[GRPOBatch, NDArray[np.float32], dict[str, float], jax.Array]:
    batch_size = config.num_tasks * config.group_size
    if len(envs) != batch_size or any(env.config != config.env for env in envs):
        raise ValueError("Expected num_tasks * group_size environments with the configured limits")
    # Same reset seed gives every member of a group the same sampled pair.
    seeds = np.repeat(rng.integers(0, 2**31, size=config.num_tasks), config.group_size)
    pairs = [env.reset(seed=int(seed)) for env, seed in zip(envs, seeds)]
    initial = np.stack([pair.initial for pair in pairs])
    target = np.stack([pair.target for pair in pairs])
    shape = (config.max_rounds, batch_size, config.max_nodes)
    actions = np.full(shape, KarelProgramEnv.pad_token_id, dtype=np.int32)
    old_log_probs = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.bool_)
    rewards = np.zeros(batch_size, dtype=np.float32)
    reward_components = {name: np.zeros(batch_size, dtype=np.float32) for name in REWARD_COMPONENTS}
    successes = np.zeros(batch_size, dtype=np.bool_)
    active = np.ones(batch_size, dtype=np.bool_)
    errors: list[str | None] = [None] * batch_size
    trees = [KarelAST.empty(config.max_nodes, config.max_depth, config.env.max_program_tokens) for _ in envs]
    snapshots: list[ASTFeatures] = []
    source_lengths = np.zeros(batch_size, np.int32)
    for t in range(shape[0]):
        features = batch_features(tuple(tree.features() for tree in trees))
        snapshots.append(features)
        logits = predict(state, initial, target, features)
        key, sample_key = jax.random.split(key)
        sampled, log_probs = jax.device_get(act(logits, sample_key))
        mask[t] = features.action_mask.any(axis=-1)
        actions[t] = sampled
        old_log_probs[t] = log_probs
        for index in np.flatnonzero(active):
            trees[index] = trees[index].expand_round(sampled[index])
            if trees[index].complete:
                source = trees[index].tokens()
                source_lengths[index] = len(source)
                # The AST mask guarantees this fits the unchanged environment.
                assert len(source) <= config.env.max_program_tokens
                for token in source:
                    _, reward, terminated, truncated, info = envs[index].step(TOKEN_TO_ID[token])
                    if terminated or truncated:
                        break
                if not (terminated or truncated):
                    raise RuntimeError("A completed AST did not terminate its Karel environment")
                rewards[index] = reward
                active[index] = False
                errors[index] = info["error"]
                successes[index] = bool(info["success"])
                for name, values in reward_components.items():
                    values[index] = info[f"reward_{name}"]
        if not active.any():
            break
    if active.any():
        raise RuntimeError("AST generation exhausted its budget with unfinished holes")
    # Pad only the time dimension. Finished rows get the model's safe dummy
    # distribution, and their probabilities/entropy/gradients are masked out.
    stored = ASTFeatures(*(np.stack(fields) for fields in zip(*snapshots)))
    stored = ASTFeatures(*(np.pad(x, ((0, shape[0] - len(snapshots)),) + ((0, 0),) * (x.ndim - 1)) for x in stored))
    grouped_rewards = rewards.reshape((config.num_tasks, config.group_size))
    advantages = np.asarray(group_advantages(grouped_rewards)).reshape(-1)
    diagnostics = {
        "charts/reward_mean": float(rewards.mean()),
        **{f"charts/reward_{name}_mean": float(values.mean()) for name, values in reward_components.items()},
        "charts/success_rate": float(successes.mean()),
        "charts/group_success_rate": float(successes.reshape((config.num_tasks, config.group_size)).any(axis=1).mean()),
        "charts/informative_group_fraction": float((np.ptp(grouped_rewards, axis=1) > 0).mean()),
        "charts/episode_length_mean": float(mask.sum(axis=(0, 2)).mean()),
        "charts/generation_rounds_mean": float(mask.any(axis=-1).sum(axis=0).mean()),
        "charts/program_token_length_mean": float(source_lengths.mean()),
        "charts/truncation_rate": errors.count("token_limit") / batch_size,
        "charts/syntax_error_rate": errors.count("syntax_error") / batch_size,
        "charts/runtime_error_rate": errors.count("runtime_error") / batch_size,
        "charts/execution_limit_rate": errors.count("execution_limit") / batch_size,
    }
    return GRPOBatch(initial, target, stored, actions, old_log_probs, mask, advantages), rewards, diagnostics, key


def format_program(tokens: Sequence[str]) -> str:
    """Indent DSL blocks and put actions on separate lines without changing tokens.

    Conditions stay inline. Invalid/incomplete programs are formatted best-effort,
    never repaired; unmatched closers cannot produce negative indentation. Wrap
    long malformed headers and cap display indentation to keep logs readable.
    """
    lines: list[str] = []
    current: list[str] = []
    depth = 0

    def flush() -> None:
        if current:
            lines.append("  " * min(depth, 16) + " ".join(current))
            current.clear()

    for token in tokens:
        if token in ("m)", "w)", "i)", "e)", "r)"):
            flush()
            depth = max(0, depth - 1)
            current.append(token)
            flush()
        elif token in ("m(", "w(", "i(", "e(", "r("):
            current.append(token)
            flush()
            depth += 1
        elif token in ACTIONS:
            flush()
            current.append(token)
            flush()
        else:
            if token in ("DEF", "WHILE", "IF", "IFELSE", "ELSE", "REPEAT"):
                flush()
            if sum(len(part) + 1 for part in current) + len(token) + 2 * min(depth, 16) > 100:
                flush()
            current.append(token)
    flush()
    return "\n".join(lines)


def format_group_programs(
    batch: GRPOBatch, rewards: Array, *, config: Config, group_size: int, group_index: int, count: int
) -> str:
    """Format actual rollout ASTs from one task group, without resampling or ranking.

    Rows retain sampling order. Replay AST expansions and print their source;
    reward and action/source lengths belong to that rollout.
    """
    chex.assert_rank(batch.actions, 3)
    chex.assert_equal_shape((batch.actions, batch.mask))
    chex.assert_type(batch.actions, jnp.int32)
    chex.assert_type(batch.mask, jnp.bool_)
    chex.assert_shape(rewards, (batch.actions.shape[1],))
    chex.assert_type(rewards, jnp.float32)
    for value in (group_size, group_index, count):
        chex.assert_type(value, int)
    chex.assert_scalar_positive(group_size)
    chex.assert_is_divisible(batch.actions.shape[1], group_size)
    chex.assert_scalar_in(group_index, 0, batch.actions.shape[1] // group_size - 1)
    chex.assert_scalar_positive(count)
    count = min(count, group_size)
    actions, mask, rewards = jax.device_get((batch.actions, batch.mask, rewards))
    lines = [f"## Group {group_index}: {count}/{group_size} programs for the same initial/target pair."]
    for sample in range(count):
        index = group_index * group_size + sample
        rounds = np.flatnonzero(mask[:, index].any(axis=-1))
        tree = KarelAST.empty(config.max_nodes, config.max_depth, config.env.max_program_tokens)
        for step in rounds:
            tree = tree.expand_round(actions[step, index])
        program = tree.source()
        lines.append(
            f"### Sample {sample}: reward={float(rewards[index]):.4f}, actions={int(mask[:, index].sum())}, rounds={len(rounds)}, "
            f"tokens={len(tree.tokens())}\n\n```text\n{program}\n```"
        )
    return "\n\n".join(lines)


def objective(
    logits: jax.Array,
    batch: GRPOBatch,
    config: Config,
    reference_log_probs: jax.Array | None = None,
    normalization: jax.Array | None = None,
) -> tuple[jax.Array, Metrics]:
    """Clip each hole decision; average all decisions equally within each program."""
    chex.assert_rank(batch.actions, 3)
    chex.assert_equal_shape((batch.actions, batch.old_log_probs, batch.mask))
    chex.assert_shape(batch.advantages, (batch.actions.shape[1],))
    chex.assert_type((batch.old_log_probs, batch.advantages), jnp.float32)
    chex.assert_type(batch.mask, jnp.bool_)
    mask = batch.mask & (batch.actions != KarelProgramEnv.pad_token_id)
    log_probs = action_log_prob(logits, batch.actions)
    if normalization is None:
        normalization = mask.astype(jnp.float32) / jnp.maximum(mask.sum(axis=(0, 2)), 1)[None, :, None] / mask.shape[1]
    chex.assert_shape(normalization, mask.shape)
    chex.assert_type(normalization, jnp.float32)

    def average(values: jax.Array) -> jax.Array:
        chex.assert_equal_shape((values, batch.mask))
        chex.assert_type(values, jnp.float32)
        return (jnp.where(mask, values, 0.0) * normalization).sum()

    log_ratio = jnp.where(mask, log_probs - jax.lax.stop_gradient(batch.old_log_probs), 0.0)
    ratio = jnp.exp(log_ratio)
    advantage = jax.lax.stop_gradient(batch.advantages)[None, :, None]
    policy_loss = -average(
        jnp.minimum(ratio * advantage, jnp.clip(ratio, 1 - config.clip_coef, 1 + config.clip_coef) * advantage)
    )
    policy_logits = generation_logits(logits)
    # PAD and grammar-invalid tokens have probability zero. Replace their log
    # probabilities before multiplying to avoid 0 * -inf and NaN gradients.
    entropy_log_probs = jnp.where(jnp.isfinite(policy_logits), jax.nn.log_softmax(policy_logits), 0.0)
    entropy = average(-(jax.nn.softmax(policy_logits) * entropy_log_probs).sum(axis=-1))
    approx_kl = average(jnp.expm1(log_ratio) - log_ratio)
    clip_fraction = average((jnp.abs(ratio - 1) > config.clip_coef).astype(jnp.float32))
    reference_kl = jnp.asarray(0.0, dtype=jnp.float32)
    if config.kl_coef:
        if reference_log_probs is None:
            raise ValueError("Reference log probabilities are required when kl_coef > 0")
        chex.assert_equal_shape((reference_log_probs, batch.actions))
        chex.assert_type(reference_log_probs, jnp.float32)
        delta = jnp.where(mask, jax.lax.stop_gradient(reference_log_probs) - log_probs, 0.0)
        reference_kl = average(jnp.expm1(delta) - delta)
    loss = policy_loss - config.entropy_coef * entropy + config.kl_coef * reference_kl
    return loss, (policy_loss, entropy, approx_kl, clip_fraction, reference_kl)


@partial(jax.jit, static_argnames="config")
def update(
    state: TrainState, batch: GRPOBatch, config: Config, reference_params: optax.Params | None = None
) -> tuple[TrainState, Metrics]:
    """Accumulate round microbatch gradients; apply one program-level update."""
    if config.kl_coef and reference_params is None:
        raise ValueError("Frozen reference parameters are required when kl_coef > 0")
    chex.assert_rank(batch.actions, 3)
    chex.assert_equal_shape((batch.actions, batch.old_log_probs, batch.mask))
    chex.assert_type(batch.actions, jnp.int32)
    chex.assert_type((batch.old_log_probs, batch.advantages), jnp.float32)
    chex.assert_type(batch.mask, jnp.bool_)
    chex.assert_type(batch.tree[:5], jnp.int32)
    chex.assert_type((batch.tree.is_hole, batch.tree.node_mask, batch.tree.action_mask), jnp.bool_)
    steps, programs, _ = batch.actions.shape
    count = steps * programs
    width = config.decision_batch_size
    padded = ((count + width - 1) // width) * width

    def flatten(value: Array) -> jax.Array:
        chex.assert_shape(value, (steps, programs, *value.shape[2:]))
        value = jnp.asarray(value).reshape(count, *value.shape[2:])
        return jnp.pad(value, ((0, padded - count),) + ((0, 0),) * (value.ndim - 1))

    trees = ASTFeatures(*(flatten(value) for value in batch.tree))
    actions, old_log_probs, mask = (flatten(value) for value in (batch.actions, batch.old_log_probs, batch.mask))
    weights = flatten(
        batch.mask.astype(jnp.float32) / jnp.maximum(batch.mask.sum(axis=(0, 2)), 1)[None, :, None] / programs
    )
    zeros = jax.tree.map(jnp.zeros_like, state.params)
    metric_zeros = tuple(jnp.asarray(0.0, jnp.float32) for _ in range(5))

    def scan(
        accumulated: tuple[optax.Params, Metrics], indices: jax.Array
    ) -> tuple[tuple[optax.Params, Metrics], None]:
        chex.assert_shape(indices, (width,))
        chex.assert_type(indices, jnp.int32)
        rows = indices % programs
        tree = ASTFeatures(*(value[indices] for value in trees))
        chunk = GRPOBatch(
            batch.initial[rows],
            batch.target[rows],
            ASTFeatures(*(x[None] for x in tree)),
            actions[indices][None],
            old_log_probs[indices][None],
            mask[indices][None],
            batch.advantages[rows],
        )

        def evaluate() -> tuple[optax.Params, Metrics]:
            reference = None
            if config.kl_coef:
                logits = state.apply_fn({"params": reference_params}, chunk.initial, chunk.target, tree)
                reference = action_log_prob(logits, chunk.actions[0])[None]

            def loss_fn(params: optax.Params) -> tuple[jax.Array, Metrics]:
                logits = state.apply_fn({"params": params}, chunk.initial, chunk.target, tree)
                return objective(logits[None], chunk, config, reference, weights[indices][None])

            (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            return grads, metrics

        grads, metrics = jax.lax.cond(chunk.mask.any(), evaluate, lambda: (zeros, metric_zeros))
        return (
            jax.tree.map(jnp.add, accumulated[0], grads),
            tuple(a + b for a, b in zip(accumulated[1], metrics)),
        ), None

    (grads, metrics), _ = jax.lax.scan(
        scan, (zeros, metric_zeros), jnp.arange(padded, dtype=jnp.int32).reshape(-1, width)
    )
    if config.target_kl is None:
        return state.apply_gradients(grads=grads), metrics
    state = jax.lax.cond(metrics[2] > config.target_kl, lambda: state, lambda: state.apply_gradients(grads=grads))
    return state, metrics


def generate(
    state: TrainState,
    initial: Grids,
    target: Grids,
    key: jax.Array,
    max_nodes: int = 128,
    max_depth: int = 64,
    max_program_tokens: int = 128,
) -> tuple[KarelAST, ...]:
    """Sample complete trees; illegal expansions and unfinished output are errors."""
    chex.assert_shape(initial, (None, None, None, 6))
    chex.assert_equal_shape((initial, target))
    chex.assert_type((initial, target), jnp.int32)
    chex.assert_scalar_positive(initial.shape[0])
    chex.assert_shape(key, ())
    if not jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
        raise TypeError("Expected a typed JAX PRNG key from jax.random.key")
    trees = [KarelAST.empty(max_nodes, max_depth, max_program_tokens) for _ in initial]
    for _ in range(min(max_nodes, max_depth + 1)):
        logits = predict(state, initial, target, batch_features(tuple(tree.features() for tree in trees)))
        key, sample_key = jax.random.split(key)
        actions = np.asarray(jax.random.categorical(sample_key, logits), dtype=np.int32)
        trees = [tree.expand_round(action) for tree, action in zip(trees, actions)]
        if all(tree.complete for tree in trees):
            return tuple(trees)
    raise RuntimeError("AST generation failed to finish within its node budget")


def load_model(
    run_dir: str | Path, *, attention_implementation: AttentionImplementation | None = None
) -> tuple[Config, TrainState]:
    """Restore weights for inference; override attention with xla to load on CPU.

    This inference helper starts a fresh optimizer. train() automatically restores
    full training state when the configured run_id has a checkpoint.
    """
    directory = str(run_dir).rstrip("/")
    checkpoint = _read_optional(f"{directory}/checkpoint.msgpack")
    payload = serialization.msgpack_restore(checkpoint) if checkpoint is not None else None
    if payload is not None and payload["version"] != 1:
        raise ValueError("Unsupported GRPO checkpoint version")
    config = load_config(f"{directory}/config.yaml")
    if attention_implementation is not None:
        config = replace(config, attention_implementation=attention_implementation)
    dummy = np.zeros((1, config.env.height, config.env.width, 6), np.int32)
    state = create_state(config, dummy, dummy)
    params = (
        serialization.from_bytes(state.params, _read_bytes(f"{directory}/params.msgpack"))
        if payload is None
        else serialization.from_state_dict(state.params, payload["state"]["params"])
    )
    chex.assert_trees_all_equal_shapes_and_dtypes(params, state.params)
    return config, state.replace(params=params)


def train(config: Config) -> TrainState:
    batch_size = config.num_tasks * config.group_size
    envs = [KarelProgramEnv(config.env) for _ in range(batch_size)]
    rng = np.random.default_rng(config.seed)
    key = jax.random.key(config.seed)
    dummy = np.zeros((1, config.env.height, config.env.width, 6), np.int32)
    state = create_state(config, dummy, dummy)
    reference_params = state.params if config.kl_coef else None
    run_name = config.run_id or f"karel_grpo_seed{config.seed}_{datetime.now(UTC):%Y%m%d-%H%M%S-%f}"
    config = replace(config, run_id=run_name)
    run_dir = f"{config.log_dir.rstrip('/')}/{run_name}"
    checkpoint = _read_optional(f"{run_dir}/checkpoint.msgpack")
    progress = TrainingProgress(state, reference_params, key, 0, 0, 0, 0)
    if checkpoint is not None:
        progress = _restore_checkpoint(checkpoint, config, state, rng)
        print(f"Resuming {run_dir} at rollout {progress.iteration}, step {progress.steps}", flush=True)
    else:
        if _read_optional(f"{run_dir}/params.msgpack") is not None:
            raise ValueError("Run has only inference weights, not resumable training state. Use a new run_id.")
    state, reference_params, key, start_iteration, steps, logged_groups, episodes = progress
    if start_iteration >= config.total_updates:
        print(f"Run already completed {start_iteration} rollouts (target {config.total_updates}).", flush=True)
        return state
    _write_bytes(f"{run_dir}/config.yaml", yaml.safe_dump(asdict(config)).encode())
    if checkpoint is None:
        _save_checkpoint(run_dir, progress, rng)
    # Hide events from unsaved rollouts after an interruption, retaining the
    # checkpoint's own step. New runs do not need a TensorBoard restart marker.
    writer = SummaryWriter(logdir=run_dir, purge_step=steps + 1 if checkpoint is not None else None)
    try:
        writer.add_text("config", f"```yaml\n{yaml.safe_dump(asdict(config))}```", steps)
        writer.add_text("devices", str(jax.devices()), steps)
        parameter_count = sum(parameter.size for parameter in jax.tree.leaves(state.params))
        writer.add_scalar("model/params_millions", parameter_count / 1_000_000, steps)
        print(
            f"TensorBoard run: {run_dir}\nJAX devices: {jax.devices()}\nModel parameters: {parameter_count:,}",
            flush=True,
        )
        print(
            "Task/trajectory rewards minus length/execution costs: equal-reward groups have zero GRPO advantages.",
            flush=True,
        )
        start = monotonic()
        last_checkpoint_time = start
        start_steps = steps
        for iteration in range(start_iteration, config.total_updates):
            rollout_start = monotonic()
            batch, rewards, diagnostics, key = collect_rollout(state, envs, rng, key, config)
            rollout_seconds = monotonic() - rollout_start
            steps += int(batch.mask.sum())
            episodes += batch_size
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
                    minibatch = GRPOBatch(
                        batch.initial[indices],
                        batch.target[indices],
                        ASTFeatures(*(x[:, indices] for x in batch.tree)),
                        batch.actions[:, indices],
                        batch.old_log_probs[:, indices],
                        batch.mask[:, indices],
                        batch.advantages[indices],
                    )
                    state, metric = update(state, minibatch, config, reference_params)
                    metrics.append(metric)
                    if config.target_kl is not None and float(metric[2]) > config.target_kl:
                        early_stop = True
                        break
                    updates_done += 1
                if early_stop:
                    break
            jax.block_until_ready((state, metrics))
            policy_loss, entropy, approx_kl, clip_fraction, reference_kl = np.mean(jax.device_get(metrics), axis=0)
            for tag, scalar in {
                **diagnostics,
                "losses/policy": policy_loss,
                "policy/entropy": entropy,
                "policy/approx_kl": approx_kl,
                "policy/clip_fraction": clip_fraction,
                "policy/reference_kl": reference_kl,
                "policy/early_stop": early_stop,
                "charts/learning_rate": learning_rate,
                "charts/updates_per_rollout": updates_done,
                "charts/total_episodes": episodes,
                "charts/steps_per_second": (steps - start_steps) / (monotonic() - start),
                "time/rollout_seconds": rollout_seconds,
                "time/optimization_seconds": monotonic() - optimization_start,
            }.items():
                writer.add_scalar(tag, float(scalar), steps)
            if config.log_program_count and (iteration == 0 or (iteration + 1) % config.log_program_interval == 0):
                samples = format_group_programs(
                    batch,
                    rewards,
                    config=config,
                    group_size=config.group_size,
                    group_index=logged_groups % config.num_tasks,
                    count=config.log_program_count,
                )
                logged_groups += 1
                writer.add_text("samples/generated_programs", samples, steps)
            if (
                monotonic() - last_checkpoint_time >= config.checkpoint_interval_seconds
                or iteration + 1 == config.total_updates
            ):
                _save_checkpoint(
                    run_dir,
                    TrainingProgress(state, reference_params, key, iteration + 1, steps, logged_groups, episodes),
                    rng,
                )
                last_checkpoint_time = monotonic()
            writer.flush()
            print(
                f"iteration={iteration + 1} step={steps} success={diagnostics['charts/success_rate']:.3f} "
                f"reward={diagnostics['charts/reward_mean']:.3f} "
                f"syntax={diagnostics['charts/reward_syntax_mean']:.3f} "
                f"runtime={diagnostics['charts/reward_runtime_mean']:.3f} "
                f"distance={diagnostics['charts/reward_distance_mean']:.3f} "
                f"trajectory={diagnostics['charts/reward_trajectory_mean']:.3f} "
                f"length_penalty={diagnostics['charts/reward_length_mean']:.4f} "
                f"execution_penalty={diagnostics['charts/reward_execution_mean']:.4f} "
                f"informative_groups={diagnostics['charts/informative_group_fraction']:.3f} "
                f"policy={policy_loss:.3f} entropy={entropy:.3f} kl={approx_kl:.4f} "
                f"updates={updates_done} early_stop={early_stop}",
                flush=True,
            )
        return state
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grpo_karel.yaml", help="Path to a YAML config")
    parser.add_argument("--platform", choices=("cpu", "cuda", "metal"), help="Require a JAX backend")
    args = parser.parse_args()
    if args.platform is not None:
        jax.config.update("jax_platforms", args.platform)
        jax.devices()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
