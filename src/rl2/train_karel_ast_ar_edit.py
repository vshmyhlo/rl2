"""Execution-guided autoregressive Karel editing with improvement rewards.

The policy consumes a causal sequence with a persistent KV cache:
    EDIT(seed DFS tokens) -> FEEDBACK -> EDIT(location) -> EDIT(grammar)... -> FEEDBACK -> ... -> EDIT(STOP)
Every EDIT and FEEDBACK event receives the previous action token. A FEEDBACK
event repeats the action that completed the program, including the final seed token.
Only FEEDBACK receives the original input, target, and current execution image.
These 18 channels pass through a shared 32-channel 1x1 convolution, GELU,
flattening, and projection. The image embedding is added only to FEEDBACK events.
The current image starts at the seed's final or last-valid output.
Each FEEDBACK introduces the new image and eight scalars: score, success, runtime error, execution limit,
normalized ticks, source length, last score difference, and remaining sequence budget.
KarelASTEditEnv owns AST edits, masks, execution, rewards, and termination. The
policy infers the current program from its seed and edit history.
Set allow_stop=False to mask STOP and keep editing until the remaining sequence
budget cannot fit another complete replacement.

Reward timing and example trajectories
--------------------------------------
Execute the fixed turnLeft program before the first action, with no reward. Execute again only
after all holes in a replacement are filled, always from the original task input.
A completed replacement receives r = score(new program) - score(previous program).
Scores subtract weighted AST depth / max_depth, AST node count / max_nodes,
and consumed execution ticks / max_execution_steps. All three penalty weights
live in env; depth is normalized by the editor's candidate AST depth limit.
Location choices, unfinished grammar expansions, and STOP receive zero reward.
An execution failure is scored from its last valid grid and can be repaired by
later edits. The environment accepts regressive edits and does not auto-stop on
success. Feedback retains the latest execution result while filling holes.

Example 1: the target requires one right turn (max_seq_len=17).

    S0: turnLeft; seed score=S_left; sequence tokens left=12
    S0 -- select the turnLeft statement --> (r=0, S1)
    S1: <Statement>; unchanged execution feedback; sequence tokens left=11
    S1 -- turnRight --> (r=S_right-S_left, S2)
    S2: turnRight; execute and emit FEEDBACK; sequence tokens left=9
    S2 -- STOP --> (r=0, Terminal)

    EDIT(Program,ConsNonEmpty,turnLeft,End)
        -> FEEDBACK(seed score,delta=0) -> EDIT(select) -> EDIT(turnRight)
        -> FEEDBACK(new score/grid,delta=S_right-S_left) -> EDIT(STOP)

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
If a completion and its feedback consume the remaining budget, the episode
ends with its score difference; no extra STOP or terminal-score bonus is added.
Feedback events consume context space but do not consume policy decisions.

Training uses regular episode-level GRPO credit assignment. Sum all edit deltas
into one episode return, normalize returns within each task group, and assign
that same advantage to every policy decision in the episode, including STOP.
There is no return-to-go credit assignment or learned critic. Equal-return
groups have zero policy advantages. The clipped loss averages decisions within
each episode, then across episodes; host events and padding have zero loss.
The episode return telescopes to final_score - seed_score. Since each task group
shares the seed score, its normalized advantages are equivalent to normalizing
final scores. Training normalizes final scores directly to avoid float32 delta
accumulation breaking ties; delta rewards and feedback remain available during editing.
"""

import argparse
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
from flax.training.train_state import TrainState
from numpy.typing import NDArray
from tensorboardX import SummaryWriter

from rl2.attention import AttentionType
from rl2.edit_transformer import (
    EDIT_EVENT,
    FEEDBACK_EVENT,
    EditTransformer,
    Event,
    ModelOutput,
)
from rl2.jax_cache import configure_compilation_cache
from rl2.karel import (
    TASK_CATEGORIES,
    KarelConfig,
    KarelProgramEnv,
)
from rl2.karel_ast import AST_ACTIONS, KarelAST
from rl2.karel_ast_edit import (
    EDIT_REWARD_COMPONENTS,
    FEEDBACK_SIZE,
    INITIAL_ACTIONS,
    INITIAL_PROGRAM,
    EditConfig,
    EditStep,
    Evaluation,
    ExecutedObservation,
    Observation,
)
from rl2.karel_ast_edit_vector import KarelASTEditVectorEnv
from rl2.shape_checker import ShapeChecker
from rl2.train_karel_ast_grpo import (
    Array,
    Metrics,
    TrainingProgress,
    _restore_checkpoint,
    _save_checkpoint,
    group_advantages,
    learning_rate_schedule,
)
from rl2.transformer import TransformerStackCarry
from rl2.utils import read_bytes, read_optional, write_bytes


@dataclass(frozen=True)
class Config:
    # Reproducibility and training duration.
    seed: int = 1
    total_updates: int = 1000  # Number of rollout batches, independent of program length.

    # Task environment and program editing.
    env: KarelConfig = field(default_factory=KarelConfig)
    max_nodes: int = 128  # AST node budget for grammar masking; also sizes edit locations.
    max_depth: int = 64  # AST depth limit for grammar masking and penalty normalization.
    max_seq_len: int = 256  # Total prefill tokens, edit actions, and execution feedback reports per episode.
    allow_stop: bool = True  # False keeps editing until no complete edit fits the sequence budget.

    # Rollout groups and update batching.
    num_tasks: int = 8
    max_unique_tasks: int | None = None  # Fixed task pool size; 1 overfits one problem, None samples fresh tasks.
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
    attention_implementation: AttentionType = "xla"

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
    log_compiles: bool = False  # Print named dimensions (b=batch, t=time) on new prediction/update JIT traces.

    def __post_init__(self) -> None:
        """Validate training settings, backend compatibility, and initial program completion budgets."""
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
        if self.max_unique_tasks is not None:
            if type(self.max_unique_tasks) is not int:
                raise TypeError("max_unique_tasks must be an integer or null")
            if self.max_unique_tasks < 1:
                raise ValueError("max_unique_tasks must be positive or null")
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
        self._validate_group_size()
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

    def _validate_group_size(self) -> None:
        """Require multiple samples for the GRPO baseline; PPO overrides this constraint."""
        if self.group_size < 2:
            raise ValueError("GRPO requires group_size >= 2")

    @property
    def edit_config(self) -> EditConfig:
        """Build the independent editing environment's validated settings."""
        return EditConfig(self.env, self.max_nodes, self.max_depth, self.max_seq_len, allow_stop=self.allow_stop)


def empty_event(time: int, batch: int, config: Config) -> Event:
    """Allocate writable host arrays initialized to padding for a batch of event streams."""
    chex.assert_scalar_non_negative(time)
    chex.assert_scalar_positive(batch)
    return Event(
        np.zeros((time, batch), np.int32),
        np.zeros((time, batch), np.int32),
        np.zeros((time, batch, 3, config.env.height, config.env.width, 6), np.int32),
        np.zeros((time, batch, FEEDBACK_SIZE), np.float32),
    )


def initial_event(observations: list[ExecutedObservation], config: Config) -> Event:
    """Prefill the fixed turnLeft program, then its execution feedback."""
    if not observations:
        raise ValueError("Cannot initialize an empty history")
    event = empty_event(len(INITIAL_ACTIONS) + 1, len(observations), config)
    event.kind[:-1] = EDIT_EVENT
    event.action[:-1] = (1 + config.max_nodes + np.asarray(INITIAL_ACTIONS, np.int32))[:, None]
    event.kind[-1] = FEEDBACK_EVENT
    event.action[-1] = event.action[-2]
    event.grid[-1] = np.stack([np.stack((o.initial, o.target, o.output)) for o in observations])
    event.feedback[-1] = np.stack([observation.feedback for observation in observations])
    sc = ShapeChecker(
        T=len(INITIAL_ACTIONS) + 1,
        B=len(observations),
        I=3,
        H=config.env.height,
        W=config.env.width,
        C=6,
        F=FEEDBACK_SIZE,
    )
    sc.check((event.kind, event.action), "TB", np.int32)
    sc.check(event.grid, "TBIHWC", np.int32)
    sc.check(event.feedback, "TBF", np.float32)
    return event


@partial(jax.jit, static_argnames="log_compiles")
def prefill(state: TrainState, event: Event, *, log_compiles: bool = False) -> ModelOutput:
    """Initialize rollout KV caches and first-decision logits with the current policy."""
    if log_compiles:
        time, batch = event.kind.shape
        print(f"JIT trace edit prefill: b={batch}, t={time}", flush=True)
    return state.apply_fn({"params": state.params}, event, method=EditTransformer.prefill)


@partial(jax.jit, static_argnames="log_compiles")
def decode_step(
    state: TrainState,
    event: Event,
    carry: TransformerStackCarry,
    *,
    log_compiles: bool = False,
) -> ModelOutput:
    """Advance cached rollout histories by one action, feedback, or padding event."""
    if log_compiles:
        print(f"JIT trace edit step: b={event.kind.shape[0]}", flush=True)
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
    if "allow_stop" in settings and type(settings["allow_stop"]) is not bool:
        raise TypeError("allow_stop must be a boolean")
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
    event = empty_event(1, initial.shape[0], config)
    params = model.init(jax.random.key(config.seed), event)["params"]

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

    event: Event  # Events [T,B]; each row predicts the following event.
    actions: Array  # [T,B], STOP=0 is a real action wherever mask is true.
    old_log_probs: Array
    legal: Array  # [T,B,V], exactly the rollout masks; dummy STOP on non-decisions.
    mask: Array  # True on rows predicting sampled edits; false before forced reports and padding.
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


def _empty_edit_batch(count: int, config: Config) -> EditBatch:
    """Allocate replay buffers with dummy STOP masks on non-decision rows."""
    shape = (config.max_seq_len, count)
    legal = np.zeros((*shape, 1 + config.max_nodes + len(AST_ACTIONS)), np.bool_)
    legal[..., 0] = True
    return EditBatch(
        empty_event(config.max_seq_len, count, config),
        np.zeros(shape, np.int32),
        np.zeros(shape, np.float32),
        legal,
        np.zeros(shape, np.bool_),
        np.zeros(shape, np.float32),
        np.zeros(count, np.float32),
    )


class _EpisodeState:
    """Track which episodes need an edit, execution feedback, or trailing padding."""

    def __init__(self, executions: list[ExecutedObservation], config: Config) -> None:
        self.config = config
        self.executions = executions
        self.observations: list[Observation] = list(executions)
        self.active = np.ones(len(executions), np.bool_)
        self.pending_feedback = np.zeros(len(executions), np.bool_)
        self.last_actions = np.full(len(executions), 1 + config.max_nodes + INITIAL_ACTIONS[-1], np.int32)

    @property
    def ready(self) -> NDArray[np.bool_]:
        return self.active & ~self.pending_feedback

    @property
    def done(self) -> bool:
        return not self.active.any() and not self.pending_feedback.any()

    def grids(self) -> NDArray[np.int32]:
        grids = np.stack([np.stack((o.initial, o.target, o.output)) for o in self.executions])
        sc = ShapeChecker(B=len(self.executions), I=3, H=self.config.env.height, W=self.config.env.width, C=6)
        sc.check(grids, "BIHWC", dtype=np.int32)
        return grids

    def advance(
        self, actions: NDArray[np.int32], transitions: list[EditStep | None]
    ) -> tuple[Event, NDArray[np.float32]]:
        """Emit actions on both kinds, with execution images and scalars only on feedback."""
        count = len(self.executions)
        sc = ShapeChecker(B=count, I=3, H=self.config.env.height, W=self.config.env.width, C=6, F=FEEDBACK_SIZE)
        sc.check(actions, "B", dtype=np.int32)
        chex.assert_equal(len(transitions), count)
        event = jax.tree.map(itemgetter(0), empty_event(1, count, self.config))
        if self.pending_feedback.any():
            event.grid[self.pending_feedback] = self.grids()[self.pending_feedback]
        rewards = np.zeros(count, np.float32)
        for index in np.flatnonzero(self.pending_feedback):
            event.kind[index] = FEEDBACK_EVENT
            event.action[index] = self.last_actions[index]
            event.feedback[index] = self.executions[index].feedback
        self.pending_feedback[:] = False
        for index, transition in enumerate(transitions):
            if transition is None:
                continue
            event.kind[index] = EDIT_EVENT
            event.action[index] = actions[index]
            self.last_actions[index] = actions[index]
            rewards[index] = transition.reward
            self.observations[index] = transition.observation
            if isinstance(transition.observation, ExecutedObservation):
                self.executions[index] = transition.observation
                # Budget exhaustion still emits feedback; STOP does not.
                self.pending_feedback[index] = not transition.terminated
            self.active[index] = not (transition.terminated or transition.truncated)
        sc.check([event.kind, event.action], "B", dtype=np.int32)
        sc.check(event.feedback, "BF", dtype=np.float32)
        sc.check(event.grid, "BIHWC", dtype=np.int32)
        sc.check(rewards, "B", dtype=np.float32)
        return event, rewards


def run_episodes(
    state: TrainState,
    tasks: list[KarelProgramEnv],
    envs: KarelASTEditVectorEnv,
    key: jax.Array,
    config: Config,
) -> EpisodeResults:
    """Sample edits, advance episode event streams, and record policy replay targets.

    Feedback pauses sampling for its episode. Each event is stored one row after
    the policy targets and rewards of the decision that produced it.
    """
    count = len(tasks)
    if not count or envs.num_envs != count or envs.config != config.edit_config:
        raise ValueError("Expected matching nonempty tasks and editing environments")
    sc = ShapeChecker(B=count, V=1 + config.max_nodes + len(AST_ACTIONS))
    sc.check(key, "")
    episodes = _EpisodeState(envs.reset(tasks), config)
    seed_scores = np.asarray([observation.feedback[0] for observation in episodes.executions], np.float32)
    event = initial_event(episodes.executions, config)
    carry, logits = prefill(state, event, log_compiles=config.log_compiles)
    length = event.kind.shape[0]
    batch = _empty_edit_batch(count, config)
    for destination, prefix in zip(batch.event, event):
        destination[:length] = prefix

    for position in range(length - 1, config.max_seq_len - 1):
        ready = episodes.ready
        for index in np.flatnonzero(ready):
            batch.legal[position, index] = episodes.observations[index].action_mask
        sc.check(logits, "BV", dtype=np.float32)
        if ready.any():
            key, sample_key = jax.random.split(key)
            sampled, log_probs = jax.device_get(act(mask_logits(logits, batch.legal[position]), sample_key))
            sc.check(sampled, "B", dtype=np.int32)
            sc.check(log_probs, "B", dtype=np.float32)
            batch.actions[position] = np.where(ready, sampled, 0)
            batch.old_log_probs[position] = np.where(ready, log_probs, 0)
        batch.mask[position] = ready

        transitions = envs.step(batch.actions[position], ready)
        event, batch.rewards[position] = episodes.advance(batch.actions[position], transitions)
        for destination, value in zip(batch.event, event):
            destination[position + 1] = value
        length = position + 2
        if episodes.done:
            break
        carry, logits = decode_step(state, event, carry, log_compiles=config.log_compiles)

    summaries = envs.summaries()
    if not episodes.done or not all(summary.tree.complete for summary in summaries):
        raise RuntimeError("Editing exhausted its bounded history with unfinished episodes")
    # Bucketing only adds trailing padding, which cannot affect earlier causal logits.
    length = min(config.max_seq_len, ((length + 31) // 32) * 32)
    batch = batch._replace(
        event=jax.tree.map(itemgetter(slice(length)), batch.event),
        actions=batch.actions[:length],
        old_log_probs=batch.old_log_probs[:length],
        legal=batch.legal[:length],
        mask=batch.mask[:length],
        rewards=batch.rewards[:length],
    )
    results = [summary.result for summary in summaries]
    completed_edits = np.asarray([summary.completed_edits for summary in summaries], np.int32)
    return batch, results, seed_scores, completed_edits, key, tuple(summary.tree for summary in summaries)


def sample_task_groups(rng: np.random.Generator, config: Config) -> list[KarelProgramEnv]:
    """Sample groups with replacement from fresh tasks or a seed-defined fixed pool.

    Pool entries are generated lazily from (run seed, task index), so restarting
    or restoring the rollout RNG preserves the pool without extra checkpoint state.
    num_tasks still controls groups per rollout, even when the pool is smaller.
    """
    limit = config.max_unique_tasks
    indices = rng.integers(0, 2**31 if limit is None else limit, size=config.num_tasks)
    sc = ShapeChecker(N=config.num_tasks)
    sc.check(indices, "N", np.int64)
    tasks: list[KarelProgramEnv] = []
    for index in indices:
        seed = int(index)
        if limit is not None:
            seed = int(np.random.SeedSequence([config.seed, seed]).generate_state(1).item())
        task = KarelProgramEnv(config.env)
        task.reset(seed=seed)
        tasks.extend([task] * config.group_size)
    return tasks


def collect_rollout(
    state: TrainState, envs: KarelASTEditVectorEnv, rng: np.random.Generator, key: jax.Array, config: Config
) -> Rollout:
    """Sample task groups, run editing episodes, and compute advantages and diagnostics."""
    count = config.num_tasks * config.group_size
    if envs.num_envs != count or envs.config != config.edit_config:
        raise ValueError("Expected num_tasks * group_size environments with configured limits")
    tasks = sample_task_groups(rng, config)
    batch, results, seed_scores, completed_edits, key, programs = run_episodes(state, tasks, envs, key, config)
    sc = ShapeChecker(B=count, T=batch.actions.shape[0])
    sc.check(batch.rewards, "TB", np.float32)
    sc.check(batch.actions, "TB", np.int32)
    sc.check(batch.mask, "TB", np.bool_)
    sc.check(seed_scores, "B", np.float32)
    scores = np.asarray([result.score for result in results], np.float64)
    sc.check(scores, "B", np.float64)
    rewards = batch.rewards.sum(axis=0)
    # All group members share a seed score. Normalize the equivalent final
    # scores, avoiding path-dependent rounding when float32 edit deltas sum.
    # Center in host float64 first so casting does not erase small real gains.
    grouped = scores.reshape(config.num_tasks, config.group_size)
    shifted = (grouped - grouped[:, :1]).astype(np.float32)
    advantages = np.asarray(group_advantages(shifted)).reshape(-1)
    sc.check((rewards, advantages), "B", np.float32)
    batch = batch._replace(advantages=advantages)
    successes = np.asarray([result.success for result in results])
    sequence_lengths = config.edit_config.prefill_length + batch.mask.sum(axis=0) + completed_edits
    # STOP is the only voluntary termination. With STOP disabled, exhausting
    # the budget can leave one unused token, too little for a complete edit.
    budget_exhausted = (sequence_lengths == config.max_seq_len) | ~np.any(batch.mask & (batch.actions == 0), axis=0)
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
        "charts/sequence_budget_exhausted_rate": float(budget_exhausted.mean()),
        "charts/sequence_length_mean": float(sequence_lengths.mean()),
        "charts/decisions_mean": float(batch.mask.sum()) / count,
        "charts/program_token_length_mean": float(np.mean([len(tree.tokens()) for tree in programs])),
        "charts/program_node_count_mean": node_count_mean,
        "charts/program_node_ratio_mean": node_count_mean / config.max_nodes,
        "charts/program_depth_mean": depth_mean,
        "charts/program_depth_ratio_mean": depth_mean / config.max_depth,
        **{
            f"charts/reward_{name}_mean": float(np.mean([result.components[name] for result in results]))
            for name in EDIT_REWARD_COMPONENTS
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


def take_event(event: Event, indices: NDArray[np.int64]) -> Event:
    """Select episode columns while preserving their complete event histories."""
    chex.assert_rank(indices, 1)
    chex.assert_type(indices, np.integer)
    return jax.tree.map(itemgetter((slice(None), indices)), event)


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
    chex.assert_shape(batch.event.kind, (time, count))
    counts = jnp.sum(batch.mask, axis=0)
    weights = batch.mask.astype(jnp.float32) / jnp.maximum(counts[None], 1) / count
    advantages = jnp.broadcast_to(batch.advantages, batch.actions.shape)
    if config.log_compiles:
        print(f"JIT trace edit update: b={count}, t={time}", flush=True)

    def loss_fn(params: optax.Params) -> tuple[jax.Array, Metrics]:
        """Replay the minibatch under candidate parameters and evaluate its masked loss."""
        _, logits = state.apply_fn({"params": params}, batch.event)
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
        take_event(batch.event, indices),
        batch.actions[:, indices],
        batch.old_log_probs[:, indices],
        batch.legal[:, indices],
        batch.mask[:, indices],
        batch.rewards[:, indices],
        batch.advantages[indices],
    )


def format_group_programs(programs: tuple[KarelAST, ...], rewards: Array, *, config: Config, group_index: int) -> str:
    """Format the initial and sampled final programs from one task group for TensorBoard."""
    chex.assert_shape(rewards, (len(programs),))
    chex.assert_type(rewards, np.float32)
    chex.assert_scalar_in(group_index, 0, config.num_tasks - 1)
    start = group_index * config.group_size
    samples = [f"Initial program:\n```text\n{' '.join(INITIAL_PROGRAM)}\n```"]
    for index in range(start, start + min(config.log_program_count, config.group_size)):
        samples.append(
            f"Sample {index - start}: reward={float(rewards[index]):.4f}\n\n```text\n{programs[index].source()}\n```"
        )
    return "\n\n".join(samples)


def generate(state: TrainState, task: KarelProgramEnv, key: jax.Array, config: Config) -> KarelAST:
    """Build and edit a program for a freshly reset task, leaving that task untouched.

    Use load_model() to restore an editing policy, reset a matching environment,
    then call generate(). Every episode starts from the fixed turnLeft program and its execution feedback.
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


def check_resume_config(config: Config, saved: Config) -> None:
    """Reject changes that reinterpret saved weights even when their shapes match."""

    def model_settings(settings: Config) -> dict[str, int]:
        return {
            **{name: getattr(settings, name) for name in ("d_model", "num_layers", "num_heads", "max_nodes")},
            "num_kv_heads": settings.num_heads if settings.num_kv_heads is None else settings.num_kv_heads,
            **{f"env.{name}": getattr(settings.env, name) for name in ("height", "width", "max_markers")},
        }

    previous = model_settings(saved)
    changed = [name for name, value in model_settings(config).items() if value != previous[name]]
    if changed:
        raise ValueError(f"Checkpoint model settings are incompatible: {', '.join(changed)}. Use a new run_id.")


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
        # check_resume_config(config, load_config(f"{run_dir}/config.yaml"))
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
                    f"depth_penalty={diagnostics['charts/reward_depth_mean']:.4f} "
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
