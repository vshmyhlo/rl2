"""Execution-guided autoregressive Karel editing with improvement rewards.

The policy consumes a causal sequence with a persistent KV cache:
    seed DFS tokens -> initial execution report -> action/result -> action/result -> ... -> STOP
Every token, including seed prefill, receives the original input,
target, and current execution image. These 18 channels pass through a shared
32-channel 1x1 convolution, GELU, flattening, and projection. This image embedding
is added to each event embedding. The current image starts at the seed's final
or last-valid output and changes only when a complete edit is re-evaluated.
Each action carries its resulting image and
eight feedback scalars: score, success, runtime error, execution limit,
normalized ticks, source length, last score difference, and remaining sequence budget.
KarelASTEditEnv owns AST edits, masks, execution, rewards, and termination. The
policy infers the current program from its seed and edit history.

Reward timing and example trajectories
--------------------------------------
Execute the seed before the first action, with no reward. Execute again only
after all holes in a replacement are filled, always from the original task input.
A completed replacement receives r = score(new program) - score(previous program).
Location choices, unfinished grammar expansions, and STOP receive zero reward.
An execution failure is scored from its last valid grid and can be repaired by
later edits. The environment accepts regressive edits and does not auto-stop on
success. Feedback retains the latest execution result while filling holes.

Example 1: the target requires one right turn (max_seq_len=17).

    S0: turnLeft; seed score=S_left; sequence tokens left=12
    S0 -- select the turnLeft statement --> (r=0, S1)
    S1: <Statement>; unchanged execution feedback; sequence tokens left=11
    S1 -- turnRight --> (r=S_right-S_left, S2)
    S2: turnRight; execute and refresh score/grid/delta; sequence tokens left=10
    S2 -- STOP --> (r=0, Terminal)

    SEED(Program,ConsNonEmpty,turnLeft,End)
        -> UPDATE(seed score,delta=0) -> ACTION(select; resulting observation)
        -> ACTION(turnRight; new score/grid,delta=S_right-S_left) -> ACTION(STOP)

Example 2: replace a statement with REPEAT 3 { move }.

    S0 -- select statement --> (r=0, S1: <Statement>)
    S1 -- REPEAT --> (r=0, S2: REPEAT <Count> { <Nonempty body> })
    S2 -- R=3 --> (r=0, S3: REPEAT 3 { <Nonempty body> })
    S3 -- ConsNonEmpty --> (r=0, S4: REPEAT 3 { <Statement>; <List tail> })
    S4 -- move --> (r=0, S5: REPEAT 3 { move; <List tail> })
    S5 -- End --> (r=S_repeat-S_old, S6: REPEAT 3 { move })
    S6 -- STOP --> (r=0, Terminal)

Only End completes this replacement and triggers execution. Choosing Cons at S5
would extend it, if node, depth, source, and sequence budgets allow completion.
If a completion spends the last action, it receives its score difference and
ends the episode immediately; no extra STOP or terminal-score bonus is added.
Feedback events consume context space but do not consume policy decisions.

Training uses regular episode-level GRPO credit assignment. Sum all edit deltas
into one episode return, normalize returns within each task group, and assign
that same advantage to every policy decision in the episode, including STOP.
There is no return-to-go credit assignment or learned critic. Equal-return
groups have zero policy advantages. The clipped loss averages decisions within
each episode, then across episodes; host events and padding have zero loss.
The episode return telescopes to final_score - seed_score. Since each task group
shares the seed score, its normalized advantages are equivalent to normalizing
final scores, while delta rewards and feedback remain available during editing.
"""

import argparse
import math
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, replace
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

from rl2.jax_cache import configure_compilation_cache
from rl2.karel import (
    REWARD_COMPONENTS,
    TASK_CATEGORIES,
    KarelConfig,
    KarelProgramEnv,
)
from rl2.karel_ast import AST_ACTIONS, KarelAST, program_actions
from rl2.karel_ast_edit import FEEDBACK_SIZE, EditConfig, Evaluation, Observation
from rl2.karel_ast_edit_vector import KarelASTEditVectorEnv
from rl2.train_karel_ast_grpo import (
    Array,
    Metrics,
    TrainingProgress,
    _restore_checkpoint,
    _save_checkpoint,
    group_advantages,
    learning_rate_schedule,
)
from rl2.transformer import AttentionImplementation, TransformerStack, TransformerStackCarry
from rl2.utils import read_bytes, read_optional, write_bytes


@dataclass(frozen=True)
class Config:
    # Reproducibility and training duration.
    seed: int = 1
    total_updates: int = 1000  # Number of rollout batches, independent of program length.

    # Task environment and program editing.
    env: KarelConfig = field(default_factory=KarelConfig)
    seed_program: str = "DEF run m( turnLeft m)"
    max_nodes: int = 128  # AST node budget for grammar masking; also sizes edit locations.
    max_depth: int = 64  # AST depth limit for grammar masking only.
    max_seq_len: int = 256  # Total seed tokens, one initial report, and action/result tokens per episode.

    # Rollout groups and update batching.
    num_tasks: int = 8
    group_size: int = 8
    env_workers: int = 0  # Spawned environment shards; zero steps locally.
    num_minibatches: int = 4
    update_epochs: int = 2

    # Transformer architecture and numerical backend.
    d_model: int = 320
    num_layers: int = 7
    num_heads: int = 5
    num_kv_heads: int | None = None
    bf16: bool = False
    attention_implementation: AttentionImplementation = "xla"

    # Optimizer and clipped policy objective.
    learning_rate: float = 0.00025
    anneal_lr: bool = True
    clip_coef: float = 0.2
    target_kl: float | None = 0.02
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5

    # Run storage, checkpointing, and logging.
    log_dir: str = "runs"
    run_id: str | None = None  # Same ID resumes; None creates a timestamped run.
    checkpoint_interval_seconds: float = 300.0  # Save at the next completed rollout boundary.
    log_interval: int = 20  # TensorBoard scalars, stdout, and flushes every N completed rollouts.
    log_program_interval: int = 20  # Program samples on logging iterations divisible by this interval.
    log_program_count: int = 8  # First N programs in one group; zero disables samples.
    log_compiles: bool = False  # Print bucket shapes on new prediction/update JIT traces.

    def __post_init__(self) -> None:
        """Validate training settings, backend compatibility, and seed-program budgets."""
        if self.run_id is not None:
            if not isinstance(self.run_id, str):
                raise TypeError("run_id must be a string or null")
            if not self.run_id.strip() or self.run_id in (".", "..") or any(c in self.run_id for c in "/\\"):
                raise ValueError("run_id must be a nonempty directory name, without slashes or traversal")
        chex.assert_scalar_positive(self.checkpoint_interval_seconds)
        if not np.isfinite(self.checkpoint_interval_seconds):
            raise ValueError("checkpoint_interval_seconds must be finite")
        if type(self.bf16) is not bool:
            raise TypeError("bf16 must be a boolean")
        if type(self.log_compiles) is not bool:
            raise TypeError("log_compiles must be a boolean")
        if type(self.env_workers) is not int:
            raise TypeError("env_workers must be an integer")
        chex.assert_scalar_non_negative(self.env_workers)
        if self.attention_implementation not in ("xla", "cudnn"):
            raise ValueError("attention_implementation must be 'xla' or 'cudnn'")
        if self.attention_implementation == "cudnn" and not self.bf16:
            raise ValueError("cuDNN attention requires bf16=True")
        for value in (self.log_interval, self.log_program_interval, self.log_program_count):
            if type(value) is not int:
                raise TypeError("Logging settings must be integers")
        chex.assert_scalar_positive(self.log_interval)
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
        chex.assert_scalar_non_negative(self.entropy_coef)
        if not np.isfinite(self.entropy_coef):
            raise ValueError("entropy_coef must be finite")
        if self.target_kl is not None:
            chex.assert_scalar_positive(self.target_kl)
            if not np.isfinite(self.target_kl):
                raise ValueError("target_kl must be finite or null")

        _ = self.edit_config

    @property
    def edit_config(self) -> EditConfig:
        """Build the independent editing environment's validated settings."""
        return EditConfig(self.env, self.max_nodes, self.max_depth, self.max_seq_len, self.seed_program)


PAD_EVENT, SEED_EVENT, ACTION_EVENT, UPDATE_EVENT = range(4)


class Events(NamedTuple):
    """Time-major history, or one batched event for cached decoding.

    NamedTuple makes this a JAX pytree; tree.map preserves its type and fields.
    kind/value: int32 [T,B] (or [B]); output: int32 [T,B,H,W,6];
    feedback: float32 [T,B,8]. Every event carries the latest execution image;
    ACTION events carry resulting feedback scalars; UPDATE is used only for the initial seed report.
    SEED values are the initial program's DFS grammar actions in policy IDs.
    ACTION values are sampled location/grammar/STOP IDs. PAD is trailing only.
    """

    kind: Array
    value: Array
    output: Array
    feedback: Array


class History(NamedTuple):
    initial: Array  # [B,H,W,6], supplied to every token's image encoder.
    target: Array
    events: Events


class EditCarry(NamedTuple):
    """KV caches and fixed task grids used alongside every event's current image."""

    transformer: TransformerStackCarry
    initial: Array
    target: Array


type ModelOutput = tuple[EditCarry, jax.Array]


class EditTransformer(nn.Module):
    """Causal seed -> initial update -> action/result stream with a KV cache.

    __call__ returns logits [T,B,V]; row t consumes event t and predicts the
    next event. Loss applies only when that next event is a sampled action;
    updates and seed tokens are provided by the host and are never targets.
    No AST encoder or tree-relative attention is used. Location IDs refer to the
    current host AST's preorder, reconstructed by applying the preceding edits.
    """

    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 8
    num_kv_heads: int | None = None
    max_nodes: int = 64
    max_seq_len: int = 256
    max_markers: int = 10
    dtype: jax.typing.DTypeLike = jnp.float32
    attention_implementation: AttentionImplementation = "xla"

    def setup(self) -> None:
        """Create task and feedback encoders, event embeddings, and the causal policy."""
        for value in (self.max_nodes, self.max_markers):
            chex.assert_type(value, int)
            chex.assert_scalar_positive(value)
        self.grid_conv = nn.Conv(32, kernel_size=(1, 1), dtype=self.dtype)
        self.context_projection = nn.Dense(self.d_model, dtype=self.dtype)
        self.context_norm = nn.LayerNorm(dtype=self.dtype)
        self.feedback_projection = nn.Dense(self.d_model, dtype=self.dtype)
        self.feedback_norm = nn.LayerNorm(dtype=self.dtype)
        self.token_embedding = nn.Embed(1 + self.max_nodes + len(AST_ACTIONS), self.d_model, dtype=self.dtype)
        self.kind_embedding = nn.Embed(4, self.d_model, dtype=self.dtype)
        self.backbone = TransformerStack(
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
            attention_implementation=self.attention_implementation,
        )
        self.head = nn.Dense(
            1 + self.max_nodes + len(AST_ACTIONS), dtype=self.dtype, kernel_init=nn.initializers.zeros_init()
        )

    def encode_grids(self, initial: jax.Array, target: jax.Array, output: jax.Array) -> jax.Array:
        """Mix normalized input/target/result channels per cell, preserving spatial positions."""
        chex.assert_scalar_non_negative(initial.ndim - 4)
        chex.assert_shape(initial, (*initial.shape[:-3], None, None, 6))
        chex.assert_equal_shape((initial, target, output))
        chex.assert_type((initial, target, output), jnp.int32)
        for size in initial.shape[-3:-1]:
            chex.assert_scalar_positive(size)
        scale = jnp.asarray([1, 1, 1, 1, 1, self.max_markers] * 3, jnp.float32)
        grids = jnp.concatenate((initial, target, output), axis=-1).astype(jnp.float32) / scale
        features = nn.gelu(self.grid_conv(grids))
        return features.reshape(*initial.shape[:-3], math.prod(features.shape[-3:]))

    def encode_events(self, events: Events, initial: jax.Array, target: jax.Array) -> jax.Array:
        """Add the task/current-image embedding to every event's token or feedback embedding."""
        chex.assert_rank(events.kind, 2)
        chex.assert_equal_shape((events.kind, events.value))
        chex.assert_type((events.kind, events.value, events.output), jnp.int32)
        time, batch = events.kind.shape
        chex.assert_shape(initial, (batch, None, None, 6))
        chex.assert_equal_shape((initial, target))
        chex.assert_type((initial, target), jnp.int32)
        chex.assert_shape(events.output, (time, *initial.shape))
        chex.assert_shape(events.feedback, (time, batch, FEEDBACK_SIZE))
        chex.assert_type(events.feedback, jnp.float32)
        grids = self.encode_grids(
            jnp.broadcast_to(initial, events.output.shape),
            jnp.broadcast_to(target, events.output.shape),
            events.output,
        )
        context = self.context_norm(self.context_projection(grids))
        update = self.feedback_norm(self.feedback_projection(events.feedback))
        token = self.token_embedding(events.value)
        x = jnp.where((events.kind == UPDATE_EVENT)[..., None], 0, token)
        has_feedback = (events.kind == ACTION_EVENT) | (events.kind == UPDATE_EVENT)
        x += jnp.where(has_feedback[..., None], update, 0)
        x += context + self.kind_embedding(events.kind)
        return jnp.where((events.kind != PAD_EVENT)[..., None], x, 0)

    def __call__(self, history: History) -> ModelOutput:
        """Encode the complete causal history and return its cache and next-action logits."""
        inputs = self.encode_events(history.events, history.initial, history.target)
        carry, features = self.backbone(inputs)
        return EditCarry(carry, history.initial, history.target), self.head(features).astype(jnp.float32)

    def prefill(self, history: History) -> ModelOutput:
        """Consume the seed program and initial execution update once."""
        carry, logits = self(history)
        return carry, logits[-1]

    def step(self, event: Events, carry: EditCarry) -> ModelOutput:
        """Append one event per episode to the KV cache and predict the next action."""
        chex.assert_rank(event.kind, 1)
        if carry is None:
            raise ValueError("Use prefill before step")
        sequence = jax.tree.map(partial(jnp.expand_dims, axis=0), event)
        transformer, features = self.backbone.step(
            self.encode_events(sequence, carry.initial, carry.target)[0], carry.transformer
        )
        return carry._replace(transformer=transformer), self.head(features).astype(jnp.float32)


def empty_events(time: int, batch: int, config: Config) -> Events:
    """Allocate writable host arrays initialized to padding for a batch of event streams."""
    chex.assert_scalar_non_negative(time)
    chex.assert_scalar_positive(batch)
    return Events(
        np.zeros((time, batch), np.int32),
        np.zeros((time, batch), np.int32),
        np.zeros((time, batch, config.env.height, config.env.width, 6), np.int32),
        np.zeros((time, batch, FEEDBACK_SIZE), np.float32),
    )


def initial_history(observations: list[Observation], config: Config) -> History:
    """Assemble task grids, seed grammar tokens, and the seed's execution feedback."""
    if not observations:
        raise ValueError("Cannot initialize an empty history")
    seed = program_actions(tuple(config.seed_program.split()))
    events = empty_events(len(seed) + 1, len(observations), config)
    events.kind[:-1] = SEED_EVENT
    events.value[:-1] = (1 + config.max_nodes + np.asarray(seed, np.int32))[:, None]
    events.kind[-1] = UPDATE_EVENT
    current = np.stack([observation.output for observation in observations])
    events.output[:] = current
    events.feedback[-1] = np.stack([observation.feedback for observation in observations])
    return History(np.stack([o.initial for o in observations]), np.stack([o.target for o in observations]), events)


@partial(jax.jit, static_argnames="log_compiles")
def prefill(state: TrainState, history: History, *, log_compiles: bool = False) -> ModelOutput:
    """Initialize rollout KV caches and first-decision logits with the current policy."""
    if log_compiles:
        print(f"JIT trace edit prefill: events={history.events.kind.shape}", flush=True)
    return state.apply_fn({"params": state.params}, history, method=EditTransformer.prefill)


@partial(jax.jit, static_argnames="log_compiles")
def decode_step(state: TrainState, event: Events, carry: EditCarry, *, log_compiles: bool = False) -> ModelOutput:
    """Advance cached rollout histories by one action, feedback, or padding event."""
    if log_compiles:
        print(f"JIT trace edit step: batch={event.kind.shape[0]}", flush=True)
    return state.apply_fn({"params": state.params}, event, carry, method=EditTransformer.step)


@jax.jit
def mask_logits(logits: jax.Array, legal: Array) -> jax.Array:
    """Exclude illegal actions from sampling and policy-loss probability calculations."""
    chex.assert_equal_shape((logits, legal))
    chex.assert_type(logits, jnp.float32)
    chex.assert_type(legal, jnp.bool_)
    return jnp.where(legal, logits, -jnp.inf)


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
    chex.assert_shape(initial, (None, config.env.height, config.env.width, 6))
    chex.assert_equal_shape((initial, target))
    chex.assert_type((initial, target), jnp.int32)
    chex.assert_scalar_positive(initial.shape[0])
    model = EditTransformer(
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
        return optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(learning_rate, eps=1e-5))

    return TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.inject_hyperparams(optimizer)(config.learning_rate)
    )


def action_log_prob(logits: jax.Array, actions: Array) -> jax.Array:
    """Extract selected-action log probabilities across batch or sequence dimensions."""
    chex.assert_shape(logits, (*actions.shape, None))
    chex.assert_type(logits, jnp.float32)
    chex.assert_type(actions, jnp.int32)
    return jnp.take_along_axis(jax.nn.log_softmax(logits), actions[..., None], axis=-1)[..., 0]


@jax.jit
def act(logits: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Sample one action per episode and retain its log probability for policy replay."""
    chex.assert_shape(logits, (None, None))
    chex.assert_type(logits, jnp.float32)
    chex.assert_shape(key, ())
    if not jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
        raise TypeError("Expected a typed JAX PRNG key from jax.random.key")
    actions = jax.random.categorical(key, logits).astype(jnp.int32)
    return actions, action_log_prob(logits, actions)


class EditBatch(NamedTuple):
    """Array-only episode histories and policy targets, suitable for a jitted update."""

    history: History  # Events [T,B]; each row predicts the following event.
    actions: Array  # [T,B], STOP=0 is a real action wherever mask is true.
    old_log_probs: Array
    legal: Array  # [T,B,V], exactly the rollout masks; dummy STOP on non-decisions.
    mask: Array  # Only sampled policy actions, never seed/update/PAD events.
    rewards: Array  # [T,B], score differences on completed edits; zero elsewhere.
    advantages: Array  # [B], group-normalized episode returns, shared by all decisions.


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
    count = len(tasks)
    if not count or envs.num_envs != count or envs.config != config.edit_config:
        raise ValueError("Expected matching nonempty tasks and editing environments")
    observations = envs.reset(tasks)
    seed_scores = np.asarray([observation.feedback[0] for observation in observations], np.float32)
    history = initial_history(observations, config)
    carry, logits = prefill(state, history, log_compiles=config.log_compiles)
    position = history.events.kind.shape[0] - 1
    stored = empty_events(config.max_seq_len, count, config)
    for destination, prefix in zip(stored, history.events):
        destination[: position + 1] = prefix
    shape = (config.max_seq_len, count)
    actions = np.zeros(shape, np.int32)
    old_log_probs = np.zeros(shape, np.float32)
    mask = np.zeros(shape, np.bool_)
    rewards = np.zeros(shape, np.float32)
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
        carry, logits = decode_step(state, event, carry, log_compiles=config.log_compiles)
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
        np.zeros(count, np.float32),
    )
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
    batch = batch._replace(advantages=np.asarray(group_advantages(grouped)).reshape(-1))
    successes = np.asarray([result.success for result in results])
    sequence_lengths = config.edit_config.prefill_length + batch.mask.sum(axis=0)
    # Final candidate sizes; AST depth counts edges from the root, including list nodes.
    node_count_mean = float(np.mean([len(tree.nodes) for tree in programs]))
    depth_mean = float(np.mean([max(node.depth for node in tree.nodes) for tree in programs]))
    diagnostics = {
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


def objective(
    logits: jax.Array, actions: Array, old_log_probs: Array, advantages: Array, weights: Array, config: Config
) -> tuple[jax.Array, Metrics]:
    """Compute the weighted clipped policy loss, entropy bonus, and update diagnostics."""
    chex.assert_shape(logits, (*actions.shape, 1 + config.max_nodes + len(AST_ACTIONS)))
    chex.assert_equal_shape((actions, old_log_probs, advantages, weights))
    chex.assert_type(actions, jnp.int32)
    chex.assert_type((logits, old_log_probs, advantages, weights), jnp.float32)
    log_probs = action_log_prob(logits, actions)
    log_ratio = jnp.where(weights > 0, log_probs - jax.lax.stop_gradient(old_log_probs), 0.0)
    ratio = jnp.exp(log_ratio)
    advantages = jax.lax.stop_gradient(advantages)
    policy = -jnp.sum(
        weights
        * jnp.minimum(ratio * advantages, jnp.clip(ratio, 1 - config.clip_coef, 1 + config.clip_coef) * advantages)
    )
    safe_log_probs = jnp.where(jnp.isfinite(logits), jax.nn.log_softmax(logits), 0.0)
    entropy = -jnp.sum(weights * jnp.sum(jax.nn.softmax(logits) * safe_log_probs, axis=-1))
    kl = jnp.sum(weights * (jnp.expm1(log_ratio) - log_ratio))
    clipped = jnp.sum(weights * (jnp.abs(ratio - 1) > config.clip_coef))
    return policy - config.entropy_coef * entropy, (policy, entropy, kl, clipped)


def take_history(history: History, indices: NDArray[np.int64]) -> History:
    """Select episode columns while preserving their complete event histories."""
    chex.assert_rank(indices, 1)
    chex.assert_type(indices, np.integer)
    return History(
        history.initial[indices],
        history.target[indices],
        jax.tree.map(itemgetter((slice(None), indices)), history.events),
    )


@partial(jax.jit, static_argnames="config")
def update(state: TrainState, batch: EditBatch, config: Config) -> tuple[TrainState, Metrics]:
    """Compute gradients and conditionally apply them within one compiled update.

    Gradients flow through complete histories, including prior actions and
    execution updates. Weights computed here give each episode equal total weight.
    Empty-decision episodes and nonfinite updates leave optimizer state unchanged.
    """
    chex.assert_rank(batch.actions, 2)
    chex.assert_equal_shape((batch.actions, batch.old_log_probs, batch.mask))
    chex.assert_type(batch.actions, jnp.int32)
    chex.assert_type(batch.mask, jnp.bool_)
    chex.assert_type((batch.old_log_probs, batch.advantages), jnp.float32)
    time, count = batch.actions.shape
    chex.assert_scalar_positive(time)
    chex.assert_scalar_positive(count)
    chex.assert_equal_shape((batch.actions, batch.rewards))
    chex.assert_shape(batch.advantages, (count,))
    chex.assert_type(batch.rewards, jnp.float32)
    chex.assert_shape(batch.history.events.kind, (time, count))
    counts = jnp.sum(batch.mask, axis=0)
    weights = batch.mask.astype(jnp.float32) / jnp.maximum(counts[None], 1) / count
    advantages = jnp.broadcast_to(batch.advantages, batch.actions.shape)
    if config.log_compiles:
        print(f"JIT trace edit update: sequence_shape={batch.actions.shape}", flush=True)

    def loss_fn(params: optax.Params) -> tuple[jax.Array, Metrics]:
        """Replay the minibatch under candidate parameters and evaluate its masked loss."""
        _, logits = state.apply_fn({"params": params}, batch.history)
        return objective(
            mask_logits(logits, batch.legal),
            batch.actions,
            batch.old_log_probs,
            advantages,
            weights,
            config,
        )

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    def apply_update(current: TrainState) -> TrainState:
        """Apply the computed gradients and advance optimizer state for an accepted update."""
        return current.apply_gradients(grads=grads)

    def skip_update(current: TrainState) -> TrainState:
        """Preserve optimizer state when KL is excessive or the update is nonfinite."""
        return current

    accepted = jnp.all(counts > 0) & jnp.all(jnp.isfinite(jnp.asarray(metrics))) & jnp.isfinite(optax.tree.norm(grads))
    if config.target_kl is not None:
        accepted &= metrics[2] <= config.target_kl
    state = jax.lax.cond(accepted, apply_update, skip_update, state)
    return state, metrics


def select_episodes(batch: EditBatch, indices: NDArray[np.int64]) -> EditBatch:
    """Create a minibatch of distinct episodes, retaining histories and rollout metadata."""
    chex.assert_rank(indices, 1)
    chex.assert_type(indices, np.integer)
    if (
        not len(indices)
        or len(np.unique(indices)) != len(indices)
        or np.any((indices < 0) | (indices >= batch.actions.shape[1]))
    ):
        raise ValueError("Expected unique, in-range episode indices")
    return EditBatch(
        take_history(batch.history, indices),
        batch.actions[:, indices],
        batch.old_log_probs[:, indices],
        batch.legal[:, indices],
        batch.mask[:, indices],
        batch.rewards[:, indices],
        batch.advantages[indices],
    )


def format_group_programs(programs: tuple[KarelAST, ...], rewards: Array, *, config: Config, group_index: int) -> str:
    """Format the seed and sampled final programs from one task group for TensorBoard."""
    chex.assert_shape(rewards, (len(programs),))
    chex.assert_type(rewards, np.float32)
    chex.assert_scalar_in(group_index, 0, config.num_tasks - 1)
    start = group_index * config.group_size
    samples = [f"Seed program:\n```text\n{config.seed_program}\n```"]
    for index in range(start, start + min(config.log_program_count, config.group_size)):
        samples.append(
            f"Sample {index - start}: reward={float(rewards[index]):.4f}\n\n```text\n{programs[index].source()}\n```"
        )
    return "\n\n".join(samples)


def generate(state: TrainState, task: KarelProgramEnv, key: jax.Array, config: Config) -> KarelAST:
    """Edit seed_program for a freshly reset task, leaving that task untouched.

    Use load_model() to restore an editing policy, reset a matching environment,
    then call generate(). To optimize a supplied program, set seed_program to
    its source using dataclasses.replace(config, seed_program=source).
    Returns the final candidate, which need not be successful or the best visited.
    """
    chex.assert_shape(key, ())
    if not jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
        raise TypeError("Expected a typed JAX PRNG key from jax.random.key")
    if task.config != config.env:
        raise ValueError("Task must use the configured environment limits")
    with KarelASTEditVectorEnv(config.edit_config, 1) as envs:
        _, _, _, _, _, programs = run_episodes(state, [task], envs, key, config)
    return programs[0]


def load_model(
    run_dir: str | Path, *, attention_implementation: AttentionImplementation | None = None
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
    run_name = config.run_id or f"karel_ast_ar_edit_seed{config.seed}_{datetime.now(UTC):%Y%m%d-%H%M%S-%f}"
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
            "Edit rewards are score differences; GRPO normalizes total returns within each task group.",
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
                policy_loss, entropy, approx_kl, clip_fraction = np.mean(jax.device_get(metrics), axis=0)
                for tag, scalar in {
                    **diagnostics,
                    "losses/policy": policy_loss,
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
                    f"policy={policy_loss:.3f} entropy={entropy:.3f} kl={approx_kl:.4f} "
                    f"updates={updates_done} early_stop={early_stop}",
                    flush=True,
                )
        return state


def main() -> None:
    """Parse the config path and launch the autoregressive AST editing trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/karel_ast_ar_edit.yaml", help="Path to a YAML config")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
