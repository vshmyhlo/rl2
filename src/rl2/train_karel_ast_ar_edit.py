"""Execution-guided GRPO for autoregressive Karel subtree editing.

A macro action selects a preorder location and fills its typed hole with DFS
constructor/value decisions. Only complete replacements execute. STOP returns
the current program; max_decisions bounds total policy actions per episode.
Every candidate executes from the task's original input, including after a
failed or regressive edit.
The policy consumes a causal sequence with a persistent KV cache:
    task grids -> seed DFS tokens -> execution update -> edit actions
               -> execution update -> more edit actions -> ... -> STOP
Each execution update projects the final/partial grid and seven feedback scalars
into one input token. The host maintains the AST and masks; the model infers the
current program from its seed and edit history, with no tree encoder.
GRPO uses final rewards, clips individual decoder decisions, and averages their
loss within each episode. This is an editing policy, not an MCTS implementation.

Feedback timing and example trajectories
----------------------------------------
Execute the seed program before the first decision. Thereafter, execute only
when a replacement has no unresolved holes. Always restart execution from the
original input grid. During replacement generation, execution feedback stays
from the previous complete program. Partial-tree changes are represented by
action tokens; scalar feedback is supplied only at execution-update boundaries,
where the length feature is the complete candidate's source length. STOP returns
the current candidate without executing it again. The final candidate's reward
trains all decisions in the episode; intermediate evaluations supply observations,
not separate per-edit training rewards.

Below, S describes the host state (current AST, last execution feedback, actions
left); the model observes its serialized event history rather than S directly.
Program bodies are shown without the surrounding DEF run m( ... m) wrapper.
Assume max_decisions=12; locations, grammar expansions and STOP each cost one action.
Each transition reads S -> action -> (r, next S). Here r describes the equivalent
terminal-only training reward: zero until termination, then R(final program).
R(p) is the environment's execution score for program p, including configured
reward terms and penalties. Evaluating a completed edit updates R(p) in the
observation even when r=0 because the editing episode is still active. The
implementation stores one final reward per episode, not a per-transition reward
array; GRPO normalizes that final reward within the task group and assigns the
resulting advantage to every decision. Seed evaluation supplies S0's feedback.

Example 1: the target requires one right turn.

    S0 -- select the turnLeft statement --> (r=0, S1)
    S1 -- turnRight --> (r=0, S2)
    S2 -- STOP --> (r=R(turnRight), Terminal)

    S0: turnLeft; feedback from seed execution; actions left=12
    S1: <Statement>; same execution feedback; actions left=11
    S2: turnRight; replacement complete, so execute from original input;
        refreshed feedback contains R(turnRight) and success; actions left=10
    Terminal: return turnRight and train using its reward.

    Corresponding model input/decision stream:
        TASK(initial, target) -> SEED(Program, ConsNonEmpty, turnLeft, End)
        -> UPDATE(seed result) -> ACTION(select statement) -> ACTION(turnRight)
        -> UPDATE(turnRight result) -> ACTION(STOP)
    UPDATE is supplied by the host before sampling the following action. STOP is
    predicted from the last update; a final action need not be fed back afterward.

Example 2: the target requires moving three cells along a clear path.

    S0 -- select the statement --> (r=0, S1)
    S1 -- REPEAT --> (r=0, S2)
    S2 -- R=3 --> (r=0, S3)
    S3 -- ConsNonEmpty --> (r=0, S4)
    S4 -- move --> (r=0, S5)
    S5 -- End --> (r=0, S6)
    S6 -- STOP --> (r=R(REPEAT 3 { move }), Terminal)

    S0: turnLeft; feedback from seed execution; actions left=12
    S1: <Statement>
    S2: REPEAT <Count> { <Nonempty body> }
    S3: REPEAT 3 { <Nonempty body> }
    S4: REPEAT 3 { <Statement>; <List tail> }
    S5: REPEAT 3 { move; <List tail> }
    S6: REPEAT 3 { move }; replacement complete, so execute from original input;
        refreshed feedback contains R(REPEAT 3 { move }) and success; actions left=6
    Terminal: return the repeat program and train using its reward.

S1-S5 in example 2 retain S0's execution feedback, while actions left decrease
from 11 to 7. Choosing Cons instead of End at S5 would create another statement
hole and list-tail
hole, extending the replacement if the budgets allow it. Grammar masks reserve
enough node/source capacity and enough remaining policy actions to finish all
holes. There is no separate limit on completed edits. STOP or exhaustion of the
action budget ends the episode; reaching the target does not force a STOP.
If a completion consumes the last available action, it leads directly to
(r=R(completed program), Terminal), with no additional STOP decision. With only
one action remaining on a complete tree, only STOP is legal. Execution-update
events do not consume policy actions; cache space for them is reserved separately.

If execution fails, refresh feedback with the last valid grid and error status.
The AST is still complete, so the next decision can select another subtree to
repair or STOP, provided the action budget has not already ended the episode.
"""

import argparse
import math
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from functools import partial
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
    TOKEN_TO_ID,
    ExecutionStats,
    KarelConfig,
    KarelPair,
    KarelProgramEnv,
    KarelProgramError,
    execute_program,
)
from rl2.karel_ast import AST_ACTIONS, KarelAST, Node, program_actions
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

FEEDBACK_SIZE = 7  # reward, success, runtime error, execution limit, ticks, source length, actions left


@dataclass(frozen=True)
class Config:
    # Reproducibility and training duration.
    seed: int = 1
    total_updates: int = 1000  # Number of rollout batches, independent of program length.

    # Task environment and program editing.
    env: KarelConfig = field(default_factory=KarelConfig)
    seed_program: str = "DEF run m( turnLeft m)"
    max_nodes: int = 128  # AST node budget for grammar masking; also sizes edit locations and sequence capacity.
    max_depth: int = 64  # AST depth limit for grammar masking only.
    max_decisions: int = 64  # Total sampled actions, including locations, grammar actions and STOP.

    # Rollout groups and update batching.
    num_tasks: int = 8
    group_size: int = 8
    num_minibatches: int = 4
    update_epochs: int = 2

    # Transformer architecture and numerical backend.
    d_model: int = 320
    num_layers: int = 7
    num_heads: int = 5
    num_kv_heads: int | None = None
    bf16: bool = False
    attention_implementation: AttentionImplementation = "xla"

    # Optimizer and GRPO objective.
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
        KarelAST.empty(self.max_nodes, self.max_depth, self.env.max_program_tokens)
        if type(self.bf16) is not bool:
            raise TypeError("bf16 must be a boolean")
        if type(self.log_compiles) is not bool:
            raise TypeError("log_compiles must be a boolean")
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

        if type(self.max_decisions) is not int:
            raise TypeError("Action budget must be an integer")
        chex.assert_scalar_positive(self.max_decisions)
        if not isinstance(self.seed_program, str):
            raise TypeError("seed_program must be source text")
        seed_tree(self)

    @property
    def max_seq_len(self) -> int:
        """Task + maximum seed + initial update + all actions and edit updates."""
        # Each completed edit needs at least a location action and one expansion.
        return 2 + self.max_nodes + self.max_decisions + self.max_decisions // 2


def seed_tree(config: Config) -> KarelAST:
    """Build the episode's starting AST by replaying its source as grammar actions."""
    tree = KarelAST.empty(config.max_nodes, config.max_depth, config.env.max_program_tokens)
    for action in program_actions(tuple(config.seed_program.split())):
        tree = tree.expand(action)
    return tree


def open_subtree(tree: KarelAST, position: int) -> KarelAST:
    """Replace a complete subtree by its typed hole; compact unreachable nodes.

    Position is in the pre-edit preorder, not the append-only storage order.
    The untouched context keeps its original depth/field/child-index metadata.
    """
    if not tree.complete:
        raise ValueError("Select an edit location only on a complete tree")
    if type(position) is not int:
        raise TypeError("Location must be an integer preorder position")
    if not 0 <= position < len(tree.nodes):
        raise ValueError("Edit location is outside the tree")
    selected = tree.preorder()[position]
    nodes: list[Node] = []

    def copy(index: int) -> int:
        """Copy reachable nodes, replacing the selected subtree with one typed hole."""
        node = tree.nodes[index]
        destination = len(nodes)
        nodes.append(replace(node, constructor=0, value=0, children=()) if index == selected else node)
        if index != selected:
            children = tuple(copy(child) for child in node.children)
            nodes[destination] = replace(node, children=children)
        return destination

    copy(0)
    result = replace(tree, nodes=tuple(nodes))
    if not result.allowed_actions().any():
        raise ValueError("Selected subtree cannot be completed within the budgets")
    return result


class Observation(NamedTuple):
    """Host-side edit state used to build masks and execution-update events."""

    initial: Array
    target: Array
    output: Array
    feedback: Array
    legal: Array


PAD_EVENT, SEED_EVENT, ACTION_EVENT, UPDATE_EVENT = range(4)


class Events(NamedTuple):
    """Time-major history, or one batched event for cached decoding.

    kind/value: int32 [T,B] (or [B]); output: int32 [T,B,H,W,6];
    feedback: float32 [T,B,7]. Only UPDATE events use output/feedback.
    SEED values are the initial program's DFS grammar actions in policy IDs.
    ACTION values are sampled location/grammar/STOP IDs. PAD is trailing only.
    """

    kind: Array
    value: Array
    output: Array
    feedback: Array


class History(NamedTuple):
    initial: Array  # [B,H,W,6], encoded once as the task prefix.
    target: Array
    events: Events


type ModelOutput = tuple[TransformerStackCarry, jax.Array]


class EditTransformer(nn.Module):
    """Causal task -> seed -> update -> actions -> update stream with a KV cache.

    __call__ returns logits [T+1,B,V]; the task prefix predicts row zero,
    then each input event predicts the following action (when it is a decision).
    Updates and seed tokens are provided by the host and have no policy loss.
    No AST encoder or tree-relative attention is used. Location IDs refer to the
    current host AST's preorder, reconstructed by applying the preceding edits.
    """

    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 8
    num_kv_heads: int | None = None
    max_nodes: int = 64
    max_seq_len: int = 354
    max_markers: int = 10
    dtype: jax.typing.DTypeLike = jnp.float32
    attention_implementation: AttentionImplementation = "xla"

    def setup(self) -> None:
        """Create task and feedback encoders, event embeddings, and the causal policy."""
        for value in (self.max_nodes, self.max_markers):
            chex.assert_type(value, int)
            chex.assert_scalar_positive(value)
        self.context_projection = nn.Dense(self.d_model, dtype=self.dtype)
        self.context_norm = nn.LayerNorm(dtype=self.dtype)
        self.update_projection = nn.Dense(self.d_model, dtype=self.dtype)
        self.update_norm = nn.LayerNorm(dtype=self.dtype)
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

    def encode_pair(self, initial: jax.Array, target: jax.Array) -> jax.Array:
        """Encode each task's normalized input and target grids as one prefix token."""
        chex.assert_shape(initial, (None, None, None, 6))
        chex.assert_equal_shape((initial, target))
        chex.assert_type((initial, target), jnp.int32)
        for size in initial.shape[:3]:
            chex.assert_scalar_positive(size)
        scale = jnp.asarray([1, 1, 1, 1, 1, self.max_markers] * 2, jnp.float32)
        grids = jnp.concatenate((initial, target), axis=-1).astype(jnp.float32) / scale
        return self.context_norm(self.context_projection(grids.reshape(initial.shape[0], -1)))

    def encode_events(self, events: Events) -> jax.Array:
        """Embed seed/action IDs and execution feedback, zeroing trailing padding."""
        chex.assert_rank(events.kind, 2)
        chex.assert_equal_shape((events.kind, events.value))
        chex.assert_type((events.kind, events.value, events.output), jnp.int32)
        time, batch = events.kind.shape
        chex.assert_shape(events.output, (time, batch, None, None, 6))
        chex.assert_shape(events.feedback, (time, batch, FEEDBACK_SIZE))
        chex.assert_type(events.feedback, jnp.float32)
        scale = jnp.asarray([1, 1, 1, 1, 1, self.max_markers], jnp.float32)
        output = events.output.astype(jnp.float32) / scale
        width = math.prod(events.output.shape[2:])
        update = self.update_norm(
            self.update_projection(jnp.concatenate((output.reshape(time, batch, width), events.feedback), axis=-1))
        )
        token = self.token_embedding(events.value)
        x = jnp.where((events.kind == UPDATE_EVENT)[..., None], update, token)
        x += self.kind_embedding(events.kind)
        return jnp.where((events.kind != PAD_EVENT)[..., None], x, 0)

    def __call__(self, history: History) -> ModelOutput:
        """Encode the complete causal history and return its cache and next-action logits."""
        context = self.encode_pair(history.initial, history.target)
        chex.assert_shape(history.events.kind, (None, context.shape[0]))
        inputs = jnp.concatenate((context[None], self.encode_events(history.events)), axis=0)
        carry, features = self.backbone(inputs)
        return carry, self.head(features).astype(jnp.float32)

    def prefill(self, history: History) -> ModelOutput:
        """Consume the task, seed program and initial execution update once."""
        carry, logits = self(history)
        return carry, logits[-1]

    def step(self, event: Events, carry: TransformerStackCarry) -> ModelOutput:
        """Append one event per episode to the KV cache and predict the next action."""
        chex.assert_rank(event.kind, 1)
        if carry is None:
            raise ValueError("Use prefill before step")
        sequence = Events(*(jnp.asarray(value)[None] for value in event))
        carry, features = self.backbone.step(self.encode_events(sequence)[0], carry)
        return carry, self.head(features).astype(jnp.float32)


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
    events.output[-1] = np.stack([observation.output for observation in observations])
    events.feedback[-1] = np.stack([observation.feedback for observation in observations])
    return History(np.stack([o.initial for o in observations]), np.stack([o.target for o in observations]), events)


@partial(jax.jit, static_argnames="log_compiles")
def prefill(state: TrainState, history: History, *, log_compiles: bool = False) -> ModelOutput:
    """Initialize rollout KV caches and first-decision logits with the current policy."""
    if log_compiles:
        print(f"JIT trace edit prefill: events={history.events.kind.shape}", flush=True)
    return state.apply_fn({"params": state.params}, history, method=EditTransformer.prefill)


@partial(jax.jit, static_argnames="log_compiles")
def decode_step(
    state: TrainState, event: Events, carry: TransformerStackCarry, *, log_compiles: bool = False
) -> ModelOutput:
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


@dataclass(frozen=True)
class Evaluation:
    output: NDArray[np.int32]
    reward: float
    success: bool
    error: str | None
    ticks: int
    components: dict[str, float]


def evaluate(tree: KarelAST, task: KarelProgramEnv, env: KarelProgramEnv, initial: NDArray[np.int32]) -> Evaluation:
    """Reuse the environment's exact scoring; replay once to obtain grid/tick feedback.

    Task remains freshly reset, allowing arbitrary successive candidates to use
    the same initial state and cached target distance map. No reference labels
    enter either the observation or the reward.
    """
    chex.assert_shape(initial, (task.config.height, task.config.width, 6))
    chex.assert_type(initial, np.int32)
    tokens = tree.tokens()
    env.reset_from(task)
    reward, info = 0.0, {}
    for token in tokens:
        _, reward, terminated, truncated, info = env.step(TOKEN_TO_ID[token])
        if terminated or truncated:
            break
    if info.get("error") in ("syntax_error", "token_limit"):
        raise RuntimeError("AST edit produced invalid or over-budget source")
    stats = ExecutionStats()
    try:
        output = execute_program(
            tokens,
            initial,
            max_steps=task.config.max_execution_steps,
            max_markers=task.config.max_markers,
            execution_stats=stats,
        )
    except KarelProgramError as error:
        if error.partial_state is None:
            raise
        output = error.partial_state
    return Evaluation(
        output,
        reward,
        bool(info["success"]),
        info["error"],
        stats.steps,
        {name: float(info[f"reward_{name}"]) for name in REWARD_COMPONENTS},
    )


def observe(tree: KarelAST, pair: KarelPair, result: Evaluation, decisions_left: int, config: Config) -> Observation:
    """Build host feedback and action masks that reserve enough budget to finish edits."""
    chex.assert_scalar_in(decisions_left, 0, config.max_decisions)
    legal = np.zeros(1 + config.max_nodes + len(AST_ACTIONS), np.bool_)
    if tree.complete:
        legal[0] = True
        # Reserve the location action plus the cheapest typed replacement.
        # In this grammar minimum-node completions also minimize source length.
        for position, index in enumerate(tree.preorder()):
            legal[1 + position] = 1 + int(tree._costs(index).min()) <= decisions_left
    else:
        # One grammar expansion resolves exactly one node. Reserve actions for
        # ALL remaining holes, not only the next hole or the sampled constructor.
        frontier = tree.frontier
        assert frontier is not None
        costs = tree._costs(frontier)
        resolved = sum(not node.is_hole for node in tree.nodes)
        required = tree._minimum_completion() - resolved
        legal[1 + config.max_nodes :] = tree.allowed_actions() & (required - costs.min() + costs <= decisions_left)
    feedback = np.asarray(
        [
            result.reward,
            result.success,
            result.error == "runtime_error",
            result.error == "execution_limit",
            result.ticks / config.env.max_execution_steps,
            # During replacement, length describes the current partial tree's minimum completion.
            tree._minimum_completion(source=True) / config.env.max_program_tokens,
            decisions_left / config.max_decisions,
        ],
        np.float32,
    )
    return Observation(pair.initial, pair.target, result.output, feedback, legal)


def load_config(path: str | Path) -> Config:
    """Load local or remote YAML into validated training and environment settings."""
    settings = yaml.safe_load(read_bytes(str(path))) or {}
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
    """Sample one action per episode and retain its log probability for GRPO replay."""
    chex.assert_shape(logits, (None, None))
    chex.assert_type(logits, jnp.float32)
    chex.assert_shape(key, ())
    if not jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
        raise TypeError("Expected a typed JAX PRNG key from jax.random.key")
    actions = jax.random.categorical(key, logits).astype(jnp.int32)
    return actions, action_log_prob(logits, actions)


class GRPOBatch(NamedTuple):
    history: History  # Events [T-1,B]; implicit task prefix makes T predictions.
    actions: Array  # [T,B], STOP=0 is a real action wherever mask is true.
    old_log_probs: Array
    legal: Array  # [T,B,V], exactly the rollout masks; dummy STOP on non-decisions.
    mask: Array  # Only sampled policy actions, never seed/update/PAD events.
    advantages: Array  # [B]
    programs: tuple[KarelAST, ...]


def run_episodes(
    state: TrainState,
    tasks: list[KarelProgramEnv],
    envs: list[KarelProgramEnv],
    pairs: list[KarelPair],
    key: jax.Array,
    config: Config,
) -> tuple[GRPOBatch, list[Evaluation], NDArray[np.float32], NDArray[np.int32], jax.Array]:
    """One causal stream per episode; forced updates interleave asynchronously.

    The logits after an action completing an edit do NOT choose the next action:
    first consume its execution UPDATE, then sample using the new logits. Other
    batch members may still be emitting grammar actions at the same stream index.
    Finished members only consume trailing PAD, with no further policy loss.
    """
    count = len(tasks)
    if not count or len(envs) != count or len(pairs) != count:
        raise ValueError("Expected matching nonempty tasks, environments and pairs")
    trees = [seed_tree(config) for _ in tasks]
    results = [evaluate(tree, task, env, pair.initial) for tree, task, env, pair in zip(trees, tasks, envs, pairs)]
    seed_rewards = np.asarray([result.reward for result in results], np.float32)
    remaining = np.full(count, config.max_decisions, np.int32)
    completed_edits = np.zeros(count, np.int32)
    observations = [
        observe(tree, pair, result, config.max_decisions, config) for tree, pair, result in zip(trees, pairs, results)
    ]
    history = initial_history(observations, config)
    carry, logits = prefill(state, history, log_compiles=config.log_compiles)
    position = history.events.kind.shape[0]
    stored = empty_events(config.max_seq_len - 1, count, config)
    for destination, prefix in zip(stored, history.events):
        destination[:position] = prefix
    shape = (config.max_seq_len, count)
    actions = np.zeros(shape, np.int32)
    old_log_probs = np.zeros(shape, np.float32)
    mask = np.zeros(shape, np.bool_)
    legal = np.zeros((*shape, 1 + config.max_nodes + len(AST_ACTIONS)), np.bool_)
    legal[..., 0] = True
    active = np.ones(count, np.bool_)
    pending_update = np.zeros(count, np.bool_)
    for _ in range(config.max_decisions + config.max_decisions // 2):
        ready = active & ~pending_update
        updates = active & pending_update
        observations = [
            observe(tree, pair, result, int(left), config)
            for tree, pair, result, left in zip(trees, pairs, results, remaining)
        ]
        for index in np.flatnonzero(ready):
            legal[position, index] = observations[index].legal
        key, sample_key = jax.random.split(key)
        sampled, log_probs = jax.device_get(act(mask_logits(logits, legal[position]), sample_key))
        actions[position] = np.where(ready, sampled, 0)
        old_log_probs[position] = np.where(ready, log_probs, 0)
        mask[position] = ready
        event = Events(*(value[0] for value in empty_events(1, count, config)))
        for index in np.flatnonzero(updates):
            event.kind[index] = UPDATE_EVENT
            event.output[index] = observations[index].output
            event.feedback[index] = observations[index].feedback
            pending_update[index] = False
        for index in np.flatnonzero(ready):
            action = int(sampled[index])
            remaining[index] -= 1
            event.kind[index] = ACTION_EVENT
            event.value[index] = action
            tree = trees[index]
            if tree.complete:
                if action == 0:
                    active[index] = False
                    continue
                trees[index] = open_subtree(tree, action - 1)
            else:
                tree = tree.expand(action - 1 - config.max_nodes)
                trees[index] = tree
                if tree.complete:
                    results[index] = evaluate(tree, tasks[index], envs[index], pairs[index].initial)
                    completed_edits[index] += 1
                    active[index] = remaining[index] > 0
                    pending_update[index] = active[index]
        if not active.any():
            break
        for destination, value in zip(stored, event):
            destination[position] = value
        position += 1
        carry, logits = decode_step(state, event, carry, log_compiles=config.log_compiles)
    if active.any() or not all(tree.complete for tree in trees):
        raise RuntimeError("Editing exhausted its bounded history with unfinished episodes")
    # Bucketing only adds trailing padding, which cannot affect earlier causal logits.
    length = min(config.max_seq_len, ((position + 1 + 31) // 32) * 32)
    history = History(history.initial, history.target, Events(*(value[: length - 1] for value in stored)))
    batch = GRPOBatch(
        history,
        actions[:length],
        old_log_probs[:length],
        legal[:length],
        mask[:length],
        np.zeros(count, np.float32),
        tuple(trees),
    )
    return batch, results, seed_rewards, completed_edits, key


def collect_rollout(
    state: TrainState, envs: list[KarelProgramEnv], rng: np.random.Generator, key: jax.Array, config: Config
) -> tuple[GRPOBatch, NDArray[np.float32], dict[str, float], jax.Array]:
    """Sample task groups, run editing episodes, and compute advantages and diagnostics."""
    count = config.num_tasks * config.group_size
    if len(envs) != count or any(env.config != config.env for env in envs):
        raise ValueError("Expected num_tasks * group_size environments with configured limits")
    tasks: list[KarelProgramEnv] = []
    pairs: list[KarelPair] = []
    for seed in rng.integers(0, 2**31, size=config.num_tasks):
        task = KarelProgramEnv(config.env)
        pair = task.reset(seed=int(seed))
        tasks.extend([task] * config.group_size)
        pairs.extend([pair] * config.group_size)
    batch, results, seed_rewards, completed_edits, key = run_episodes(state, tasks, envs, pairs, key, config)
    rewards = np.asarray([result.reward for result in results], np.float32)
    grouped = rewards.reshape(config.num_tasks, config.group_size)
    batch = batch._replace(advantages=np.asarray(group_advantages(grouped)).reshape(-1))
    successes = np.asarray([result.success for result in results])
    diagnostics = {
        "charts/reward_mean": float(rewards.mean()),
        "charts/seed_reward_mean": float(seed_rewards.mean()),
        "charts/reward_improvement_mean": float((rewards - seed_rewards).mean()),
        "charts/success_rate": float(successes.mean()),
        "charts/group_success_rate": float(successes.reshape(config.num_tasks, config.group_size).any(axis=1).mean()),
        "charts/informative_group_fraction": float((np.ptp(grouped, axis=1) > 0).mean()),
        "charts/edits_mean": float(completed_edits.mean()),
        "charts/action_budget_exhausted_rate": float((batch.mask.sum(axis=0) == config.max_decisions).mean()),
        "charts/decisions_mean": float(batch.mask.sum()) / count,
        "charts/program_token_length_mean": float(np.mean([len(tree.tokens()) for tree in batch.programs])),
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
    return batch, rewards, diagnostics, key


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


class Replay(NamedTuple):
    history: History
    actions: Array
    old_log_probs: Array
    legal: Array
    advantages: Array  # [T,B], broadcast from terminal episode advantages.
    weights: Array  # [T,B], zero on seed/update/PAD predictions.


def replay_batch(batch: GRPOBatch) -> Replay:
    """Prepare full-history replay with equal episode weights and loss only on decisions."""
    chex.assert_rank(batch.actions, 2)
    chex.assert_equal_shape((batch.actions, batch.old_log_probs, batch.mask))
    chex.assert_type(batch.actions, np.int32)
    chex.assert_type(batch.mask, np.bool_)
    chex.assert_type((batch.old_log_probs, batch.advantages), np.float32)
    chex.assert_shape(batch.advantages, (len(batch.programs),))
    chex.assert_shape(batch.actions, (None, len(batch.programs)))
    counts = batch.mask.sum(axis=0)
    if not len(counts) or np.any(counts == 0):
        raise ValueError("Every episode must contain at least one decision")
    weights = (batch.mask / counts[None] / len(counts)).astype(np.float32)
    advantages = np.broadcast_to(batch.advantages, batch.actions.shape).copy()
    return Replay(batch.history, batch.actions, batch.old_log_probs, batch.legal, advantages, weights)


def take_history(history: History, indices: NDArray[np.int64]) -> History:
    """Select episode columns while preserving their complete event histories."""
    chex.assert_rank(indices, 1)
    chex.assert_type(indices, np.integer)
    return History(
        history.initial[indices], history.target[indices], Events(*(value[:, indices] for value in history.events))
    )


@partial(jax.jit, static_argnames="config")
def update(state: TrainState, replay: Replay, config: Config) -> tuple[TrainState, Metrics]:
    """Compute gradients and conditionally apply them within one compiled update.

    Gradients flow through complete histories, including prior actions and
    execution updates. Replay weights give each episode equal total weight.
    """
    if config.log_compiles:
        print(f"JIT trace edit update: sequence_shape={replay.actions.shape}", flush=True)

    def loss_fn(params: optax.Params) -> tuple[jax.Array, Metrics]:
        """Replay the minibatch under candidate parameters and evaluate its masked loss."""
        _, logits = state.apply_fn({"params": params}, replay.history)
        return objective(
            mask_logits(logits, replay.legal),
            replay.actions,
            replay.old_log_probs,
            replay.advantages,
            replay.weights,
            config,
        )

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    def apply_update(current: TrainState) -> TrainState:
        """Apply the computed gradients and advance optimizer state for an accepted update."""
        return current.apply_gradients(grads=grads)

    def skip_update(current: TrainState) -> TrainState:
        """Preserve parameters and optimizer state when the KL guard rejects an update."""
        return current

    if config.target_kl is None:
        state = apply_update(state)
    else:
        state = jax.lax.cond(metrics[2] <= config.target_kl, apply_update, skip_update, state)
    return state, metrics


def select_episodes(batch: GRPOBatch, indices: NDArray[np.int64]) -> GRPOBatch:
    """Create a minibatch of distinct episodes, retaining histories and rollout metadata."""
    chex.assert_rank(indices, 1)
    chex.assert_type(indices, np.integer)
    if (
        not len(indices)
        or len(np.unique(indices)) != len(indices)
        or np.any((indices < 0) | (indices >= len(batch.programs)))
    ):
        raise ValueError("Expected unique, in-range episode indices")
    return GRPOBatch(
        take_history(batch.history, indices),
        batch.actions[:, indices],
        batch.old_log_probs[:, indices],
        batch.legal[:, indices],
        batch.mask[:, indices],
        batch.advantages[indices],
        tuple(batch.programs[index] for index in indices),
    )


def format_group_programs(batch: GRPOBatch, rewards: Array, *, config: Config, group_index: int) -> str:
    """Format the seed and sampled final programs from one task group for TensorBoard."""
    chex.assert_shape(rewards, (len(batch.programs),))
    chex.assert_type(rewards, np.float32)
    chex.assert_scalar_in(group_index, 0, config.num_tasks - 1)
    start = group_index * config.group_size
    samples = [f"Seed program:\n```text\n{config.seed_program}\n```"]
    for index in range(start, start + min(config.log_program_count, config.group_size)):
        samples.append(
            f"Sample {index - start}: reward={float(rewards[index]):.4f}\n\n```text\n{batch.programs[index].source()}\n```"
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
    env = KarelProgramEnv(config.env)
    pair = env.reset_from(task)
    batch, _, _, _, _ = run_episodes(state, [task], [env], [pair], key, config)
    return batch.programs[0]


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
    """Run or resume GRPO training with grouped rollouts, KL stopping, logs, and checkpoints."""
    configure_compilation_cache()
    batch_size = config.num_tasks * config.group_size
    envs = [KarelProgramEnv(config.env) for _ in range(batch_size)]
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
                    state, metric = update(state, replay_batch(minibatch), config)
                    metrics.append(metric)
                    if config.target_kl is not None and float(metric[2]) > config.target_kl:
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
                        batch,
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
                    f"syntax={diagnostics['charts/reward_syntax_mean']:.3f} "
                    f"runtime={diagnostics['charts/reward_runtime_mean']:.3f} "
                    f"distance={diagnostics['charts/reward_distance_mean']:.3f} "
                    f"trajectory={diagnostics['charts/reward_trajectory_mean']:.3f} "
                    f"length_penalty={diagnostics['charts/reward_length_mean']:.4f} "
                    f"execution_penalty={diagnostics['charts/reward_execution_mean']:.4f} "
                    f"informative_groups={diagnostics['charts/informative_group_fraction']:.3f} "
                    f"decisions_mean={diagnostics['charts/decisions_mean']:.3f} "
                    f"edits_mean={diagnostics['charts/edits_mean']:.3f} "
                    f"action_budget_exhausted_rate={diagnostics['charts/action_budget_exhausted_rate']:.3f} "
                    f"policy={policy_loss:.3f} entropy={entropy:.3f} kl={approx_kl:.4f} "
                    f"updates={updates_done} early_stop={early_stop}",
                    flush=True,
                )
        return state
    finally:
        writer.close()


def main() -> None:
    """Parse the config path and launch the autoregressive AST editing trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/karel_ast_ar_edit.yaml", help="Path to a YAML config")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
