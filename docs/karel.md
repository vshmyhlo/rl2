# Karel synthesis reference

Current implementation, recorded 2026-10-03. Given an initial and target grid,
generate one program that transforms the initial state into the target.
The interpreter and task generator are local implementations.

## Environment and programs

[karel.py](../src/rl2/karel.py) samples a random world and grammar-generated
program, executes it, and rejects failed or unchanged outcomes. This gives a
reachable target and a known solution. Sampling is with replacement; uniqueness
across resets is not guaranteed. `reference_program` exposes the solution for
debugging or supervision, but it is not part of the policy observation.

The training YAML selects a category **once per task, before retries**:

| Category | Weight | Required initial-to-target change |
|---|---:|---|
| `easy` | 0.20 | Any nonidentical, reachable target |
| `navigation` | 0.30 | Shortest-path position distance ≥ `navigation_min_distance` (3) |
| `marker` | 0.25 | Sum of absolute marker-count differences ≥ `marker_min_edits` (3) |
| `combined` | 0.25 | Position distance ≥ `combined_min_distance` (2) and marker edits ≥ `combined_min_edits` (2) |

The `task_<category>_weight` settings are relative probabilities, normalized by
their sum; zero disables a category. Categories can overlap: "easy" means
unrestricted sampling, not necessarily a one-action solution. Position distance
accounts for walls, and marker edits count all cells. These thresholds bound
necessary primitive actions, independently of reference-program length.

The existing random world/program generator is reused. Candidates with execution
errors, unchanged targets, or insufficient changes are rejected within the chosen
category. The YAML allows 10,000 attempts because combined tasks are rare;
exhaustion raises an error naming the category and thresholds, with **no easy
fallback**. Restrictive grids or program budgets can make a category impossible.
The base `KarelConfig` defaults to easy-only sampling for compatibility; the mix
is enabled explicitly in the training YAML. Single-category sampling introduces
no extra RNG draw, preserving the previous easy-only seeded stream.

`env.sampling_stats` reports the selected category, candidate attempts, and elapsed
sampling seconds. MLflow records `sampling/<category>_fraction` over unique
tasks (not group copies), plus `_attempts_mean`, `_seconds_mean`, and
`_acceptance_rate` for categories present in that rollout. Acceptance rate is
accepted tasks divided by candidate attempts. The accepted target's BFS map is
reused at reset, and group copies preserve metadata with independent episode state.
These diagnostics are excluded from the model observation and reward.

- `reset(seed=...)` returns `KarelPair(initial, target)`, without an info wrapper.
- Each state is `int32[H, W, 6]`: four robot-heading channels, walls, marker counts.
- `step(token_id)` returns `(None, reward, terminated, truncated, info)`.
- `m)` submits and terminates the program. There is no separate EOS token.
- The vocabulary has 50 program tokens plus PAD. PAD is only for batching and is
  rejected by the environment and excluded from the policy distribution.
- Intermediate rewards are zero. `info["success"]` means exact target equality;
  `info["error"]` distinguishes syntax, runtime, execution-limit, and token-limit failures.

The DSL supports movement, turns, marker pickup/placement, `IF`, `IFELSE`,
`WHILE`, and `REPEAT R=0..R=19`. Blocks must be nonempty. Delimiters are
`m(` / `m)`, `i(` / `i)`, `e(` / `e)`, `w(` / `w)`, and `r(` / `r)`;
conditions use `c(` / `c)` and allow one negation. The interpreter allows nesting to depth 64.
The shortest valid program has five tokens: `DEF run m( turnLeft m)`.

## Terminal rewards

```text
D(s,t) = position_weight * shortest_path(robot_s, robot_t)
       + orientation_weight * int(heading_s != heading_t)
       + marker_weight * sum_cells(abs(markers_s - markers_t))
```

All weights default to 1 and must be positive and finite. Position distance
counts orthogonal moves through free cells; marker errors include every cell. Initial and target differ,
so the normalization denominator is positive. A BFS distance map from the target
robot is cached once per reset and reused for both initial and final/partial
states. Unreachable positions raise `ValueError`; sampled tasks and executions
always stay in the target's connected component.

The reward sums weighted syntax, runtime, and distance scores, each originally in `[0, 1]`, plus a
**weighted exact-success bonus** (default +1) for normal completion at the exact target, and a
weighted trajectory bonus, minus separate length and execution costs.

`env.syntax_weight`, `env.runtime_weight`, `env.distance_weight`, and
`env.success_weight` default to
`1.0` and must be finite and nonnegative. Set a weight to `0` to disable that
reward contribution. These scale whole reward terms; `position_weight`,
`orientation_weight`, and `marker_weight` still define the underlying distance.
Execution, error classification, and the `info["success"]` flag remain independent
of reward weights. Setting `success_weight: 0` disables the bonus without
changing success-rate metrics. Disabled syntax reward also
skips syntax edit-distance scoring.

`reward = syntax_weight * syntax + runtime_weight * runtime + distance_weight * distance + success_weight * success + trajectory - length_cost - execution_cost`.

The table and examples below use the default term weights. Logged
`reward_syntax`, `reward_runtime`, `reward_distance`, and `reward_success` are **weighted
contributions**, so all logged reward components continue to sum to total reward.

```text
progress(s) = clip(1 - 0.5 * D(s,target) / D(initial,target), 0, 1)
```

| Outcome | Syntax | Runtime | Distance | Before efficiency penalties |
| --- | --- | --- | --- | --- |
| Invalid syntax or token limit | `1/(1+d)` | `0` | `0` | `1/(1+d)` |
| Runtime error or execution limit | `1` | `0` | `progress(last_valid_state)` | `1 + progress(last_valid_state) + b` |
| Completes without exact match | `1` | `1` | `progress(final_state)` | `2 + progress(final_state) + b` |
| Completes at exact target | `1` | `1` | `1` | `4 + b` (includes +1 success bonus) |

Two negative reward components prefer compact, fast programs:

- `reward_length = -length_penalty_weight * generated_tokens / max_program_tokens`.
  Counts every generated token, including terminal `m)` when present, excluding PAD.
  Applies to all outcomes, including syntax errors and generation limits.
- `reward_execution = -execution_penalty_weight * execution_steps / max_execution_steps`.
  Uses consumed interpreter ticks: statements and while-condition checks, including
  failed primitive attempts. Syntax/generation failures consume zero; execution
  limits consume exactly the budget. This measures deterministic execution cost,
  not elapsed wall-clock time.

Both weights default to `0.05`, are finite and nonnegative, and can be set to zero
independently. The table and progress examples below show scores before these
deductions. Exact success continues to depend on execution and target equality,
not the numerical reward. `ExecutionStats` can also collect ticks for standalone
`execute_program(..., execution_stats=stats)` calls.

Here `b` is the weighted trajectory bonus. Let `D0` be initial distance,
`DT` distance after the last successful primitive action, and
`B = sum(max(0, D_next - D_previous))` over those actions:

```text
b = trajectory_weight * max(0, D0 - DT) / (D0 + B)
```

Distance uses the same position, heading, and marker weights as the other terms.
Only successful primitive actions count; control-flow checks and failed actions
do not. Runtime/execution-limit failures retain credit up to the last valid state;
syntax/token-limit failures get zero. Net-zero cycles and regressions earn zero.
Necessary turns can temporarily increase distance, so this term has a modest
weight. Set `trajectory_weight: 0` to disable it. No full trajectory is retained.

Here `d` is the minimum token edits to any syntactically valid program,
independently of the target or reference solution. One edit earns 0.5; four edits
earn 0.2. Repairs are never executed. Distance progress uses the last valid state
before failure, including updated position, heading, and markers. It does not
use the best state visited. Execution exceptions expose `partial_state`.

Progress maps the original `[-1, 1]` score to `[0, 1]`: unchanged distance
scores **0.5**, a 50% increase scores **0.25**, and doubled distance or worse
scores **0**. With no distance increases, completed exact solutions earn
**4.25**, half-distance solutions **2.875**, and unchanged programs **2.5**.
A runtime failure after halving distance earns **1.875**. Failure at the target
earns **2.25** but is not an exact success. Reversals reduce the trajectory bonus.
Only normal completion earns the runtime term and can unlock the exact-success bonus.
A terminal `m)` on the last allowed token still evaluates normally.

[karel_syntax.py](../src/rl2/karel_syntax.py) computes exact edit distance with a
minimum-cost parsing chart. A cheap bound on repair length avoids expanding the
grammar to enforce depth 64 unless needed. Hypothetical repairs are not limited
by the episode's token budget or the sampler's depth setting.

## Model and training

[train_karel_ast_grpo.py](../src/rl2/train_karel_ast_grpo.py) trains [ASTTransformer](../src/rl2/ast_transformer.py)
with grouped execution rewards. In each generation round the policy expands
all current typed AST holes in parallel, choosing among 15 constructors and
25 values per hole. New children wait until the next round. A separate PAD action
is used only for batching. Generation finishes when no holes remain; there is
no EOS action.

The encoder normalizes marker counts, flattens the 12-channel initial/target
pair, and projects it into one context token. The Transformer attends to that
prefix and the entire current partial tree. Node features encode type,
constructor/value, parent field, depth, child index, hole status, and preorder
position. Every round recomputes the tree representation; there is no KV
cache. Zero-initialized output heads give a uniform prior over allowed actions.

The encoder uses the shared [BDTransformer](../src/rl2/transformer.py) with
RoPE disabled, retaining the learned preorder-position embeddings. The shared
blocks use pre-RMSNorm and SwiGLU, with no additional Q/K normalization.
Projection weights use normal initialization with standard deviation 0.02.
The AST wrapper retains its MLP width rounded up from `8 * d_model / 3` to a
multiple of 128 and its normalization epsilon of `1e-6`. Residuals start in
FP32, and the stack applies the final RMSNorm once. The usual
`1/sqrt(head_dim)` attention scaling and tree-relative bias remain in place.

[Tree-relative attention](../src/rl2/tree_attention.py) adds a learned bias to
each query/key score before softmax:

`score(i,j) = dot(q_i,k_j)/sqrt(head_dim) + relation[i,j] + distance[i,j] + depth[i,j]`.

Each layer and query head has its own three lookup tables. Relationships are
self, parent, child, ancestor, descendant, sibling, or other, describing the
**key relative to the query**. Tree distance counts edges through the lowest
common ancestor and saturates at 16. Relative depth is `depth(key)-depth(query)`,
clipped to `[-16,16]`. Lists and unresolved holes count as ordinary AST nodes.
The image-pair prefix has separate context-to-node, node-to-context, and
context-to-context biases, without tree distance/depth terms. Padding has zero
bias and remains excluded as an attention key.

Relations are derived from preorder depths in quadratic work once per forward
pass and reused by all layers; rollouts store no extra fields or pairwise
matrices. Bias tables start at zero, preserving the initial uniform policy.
They add 2,135 parameters with 7 layers and 5 heads. Both XLA and cuDNN attention
receive the additive bias, including during gradient computation.
Padding is passed as int32 query/key sequence lengths (image prefix plus live
AST nodes), using cuDNN's native padding mode instead of an explicit boolean
attention mask. Any extra cuDNN alignment position is excluded from the lengths.

[KarelAST](../src/rl2/karel_ast.py) always masks illegal constructors/values and
reserves enough space to finish within three independent limits:

- `max_nodes`: allocated AST nodes and total expansion actions, including lists and values.
- `max_depth`: AST edges, including list nodes, rather than only nested control blocks.
- `env.max_program_tokens`: printed source tokens, including `m)`.

Before sampling, the runtime reserves the cheapest completion for every hole
and splits remaining node/source capacity among holes that can use it. Each
hole's mask limits choices to its allocated share, so any simultaneous
combination fits. Preorder breaks allocation ties; unused shares are available
again next round. This conservative allocation can exclude an expansion that
would fit if siblings chose cheaper actions. No sampled action is repaired or
rejected after sampling.
Each immutable tree caches its preorder and read-only parallel mask, so expansion
validation reuses the mask computed for model features. Grammar costs are cached
by hole type, remaining depth, and cost kind; new trees get fresh mask caches.

The model returns `[B, N, A]` logits, one distribution per node. Only holes have
legal actions; resolved nodes and padding receive a dummy PAD distribution and
contribute no loss. A round advances one AST depth level, so generation takes at
most `min(max_nodes, max_depth + 1)` rounds. `Cons`/`End` lists still grow one link
per round. DFS expansion and teacher-forcing helpers remain available as source
conversion utilities; training and inference use parallel rounds.

Completed trees are printed and submitted through `KarelProgramEnv.step()`.
All existing reward terms remain unchanged: length penalties count **source
tokens**, and execution penalties count interpreter ticks. Legal expansions
produce valid syntax within the source budget, but runtime failures and execution
limits remain possible. Sampled reference solutions define the tasks; they are
not supervised training labels.

Each task and target distance map is sampled/computed once per group.
`KarelProgramEnv.reset_from()` copies a freshly reset environment's task, map,
and RNG state into independent episode state for the other group members.

Rollouts store the partial tree and all per-hole legal-action masks **before**
every round. Actions and log probabilities have shape `[R, B, N]`.
Sampling, replayed log probabilities, and entropy therefore use the
same masks. GRPO normalizes rewards within each task group and applies a clipped
objective with clipping **per hole**, rather than a product of probabilities
for the whole round. Losses average over all nonpadding hole decisions across
rounds within each program, then equally over programs. Whole programs stay
together in minibatches. Updates pack only program-round snapshots containing
decisions into one model batch. Row capacity rounds up in fixed steps of
`ceil(max_rounds * programs_per_minibatch / 4)`, capped at the original capacity.
This gives at most four buckets; a nondivisible capacity shortens the last bucket,
and empty replay uses the smallest bucket. Explicit per-decision weights preserve
the original per-program normalization. Filler rows have zero loss weight.
The model also trims trailing node padding to a 32/64/96/128/… sequence bucket,
capped at `max_nodes`, keeping all live context nodes, including resolved nodes.
Rollout inference uses the same sequence bucketing; stored rollouts keep their
fixed `[R,B,N]` shapes. Each minibatch still uses a single forward/backward pass
and one optimizer update. There is no gradient accumulation or value head.
For example, 17 rounds × 64 programs allocates 1,088 snapshots, with row buckets
272/544/816/1088. If only 400 have decisions and their largest tree has 25 nodes,
the model receives `[544,32]`
node features plus one context token per row. No program budget is reduced.
This uses one shape per minibatch, rather than splitting it into size groups.
Model/checkpoint parameter shapes stay unchanged. Changing categorical sampling
shapes can change sampled programs for a given RNG key compared with older code.
The retired `decision_batch_size` and `kl_coef` settings are ignored when loading
older saved configs. Identical-reward groups have exactly zero advantages.
KL early stopping compares against the policy that collected the rollout.
There is no frozen-reference policy or reference-KL penalty.

The YAML enables BF16 computation and cuDNN attention. Parameters, optimizer
state, normalization statistics, and loss/log-probability arithmetic remain
float32. cuDNN requires a supported NVIDIA GPU; for CPU set
`attention_implementation: xla`. BF16 can remain enabled.

## AST editing with improvement rewards

`rl2.karel_ast_edit.KarelASTEditEnv` provides the editing state machine used by
`train_karel_ast_ar_edit.py`. Its `EditConfig` holds the task configuration,
AST limits and total sequence budget (`max_seq_len`). Every episode starts with the fixed
`DEF run m( turnLeft m)` program; its source is not configurable. `reset(seed=...)` samples a
task; `reset(task=...)` copies a freshly reset `KarelProgramEnv` so several
editing episodes can share a task without changing it.

`KarelASTEditVectorEnv` batches reset and masked stepping. Set `env_workers` in
the trainer config to run persistent spawned processes (the sample config uses
8); `0` runs locally. Workers own contiguous shards of the episode batch, so
the process count does not grow with every individual episode. The trainer
calls `step(actions, mask)` to run ready actions in parallel and wait for
results in episode order. Finished members consuming trailing padding are paused:
they do not step, spend budget, or reset. Task sampling remains in the trainer;
seed evaluation, edits, and candidate execution run in workers. Workers close
on training completion or failure. Single-program `generate()` runs locally.

`step(action)` returns an `EditStep` containing the observation, reward,
termination/truncation flags. The observation is tagged `executed` after reset or a
completed edit, and `editing` while a replacement is incomplete. STOP returns
`executed` with `terminated=True`, without executing again.
STOP is action 0; location actions are preorder index + 1; grammar actions are
offset by `1 + max_nodes`. Legal masks reserve enough capacity to complete all
holes and reserve one slot for the resulting feedback. Initial program EDIT
tokens, sampled EDIT tokens, and FEEDBACK reports consume the same sequence budget. Only completed replacements execute, always from the original input.

Reset executes the fixed turnLeft program and establishes a score without issuing reward. Each completed
edit receives `new_score - previous_score`; all other actions receive zero.
STOP terminates normally. Finishing an edit whose feedback uses the final sequence
token truncates with that edit's reward and a complete program. There is no additional
terminal bonus or bootstrap beyond the editing budget. Summed rewards equal
`final_score - seed_score`.

The trainer uses regular episode-level GRPO credit assignment. For edit rewards
`+0.5, -0.3, +0.6`, the episode return is `+0.8`. Returns are centered and scaled
within each task group; every decision in an episode, including STOP, receives
that episode's advantage. There is no return-to-go credit assignment or critic.
PPO clipping and equal episode weighting remain. Group size must be at least
two; equal-return groups have zero policy advantages. Since a group shares its
seed score, normalizing total deltas is equivalent to normalizing final scores.
The other trainers' reward formulations are unchanged.

Execution feedback now includes both the absolute score and its last change,
alongside output grid, success/error flags, ticks, source length, and sequence tokens
remaining. Each FEEDBACK event stacks initial, target, and last-valid output grids into
18 channels. A shared 32-channel 1x1 convolution with GELU mixes channels within
each cell; flattened spatial features are projected and added to the event
embedding. During seed prefill, only the FEEDBACK event uses the seed
execution image. There are three event kinds: PAD, EDIT, and FEEDBACK. The
fixed turnLeft program is supplied as four forced EDIT tokens followed by its first FEEDBACK.
The first policy decision uses that feedback to select an edit location (or STOP).
Every completed edit emits another FEEDBACK event, which supplies the new current
image and projected execution scalars. EDIT tokens ignore grids and feedback
scalars. FEEDBACK ignores action IDs. Reports consume
context space but are never policy targets; each episode pauses sampling while
consuming its report. PPO skips these forced transitions when computing GAE.
The KV cache contains only attention state. Events store all three grid images,
so cached and full-sequence encoding use identical observations.
Checkpoints using the former four-kind embedding cannot be resumed with this three-kind model.
The sample run ID includes the grid size, `delta_conv_all_steps_v6`, and seed.
In this trainer, `charts/reward_mean` reports total improvement,
`charts/score_mean` reports final score, and `charts/seed_score_mean` reports
the seed baseline. `charts/degenerate_group_fraction` measures the fraction of groups
with equal final scores; these groups produce zero policy advantages.

## Logging and checkpoints

`log_interval: 20` writes MLflow metrics and prints progress every 20
completed rollouts. Metrics describe that rollout,
not an average over the interval. The cadence continues across resumes; startup
metadata is written once per invocation. Checkpoint timing is independent.

`log_compiles: true` (enabled in the YAML) prints one concise line on each new
prediction/update JIT trace: `JIT trace predict/update: bucket_shape=(rows, nodes)`.
Cached calls stay quiet. These prints go to stdout independently of
`log_interval`; set the flag to false to disable them. This does not enable
JAX's compilation logs. A trace may use a persistent compilation cache,
so trace counts alone are not counts of newly compiled executables.

MLflow artifacts under `samples/generated_programs/` shows the first 8 programs from
one rollout group in sampling order, with rewards, AST action counts, round
counts, source token counts, and indented source. Groups rotate at each logging event. Logging
uses existing rollouts without extra sampling; programs are not printed to
stdout. Configure `log_program_interval` (20 by default) and `log_program_count`
(0 disables). Samples are written only on scalar-logging iterations also
divisible by `log_program_interval`; use a multiple of `log_interval` to log
samples less frequently.

`charts/reward_{syntax,runtime,distance,success,trajectory,length,execution}_mean`
logs each reward contribution, including zeros on failures. Their sum equals
`charts/reward_mean`. Runtime is a binary completion indicator;
distance measures progress from the final or last valid execution state. `charts/success_rate` measures
exact solutions, and `charts/group_success_rate` measures groups containing at
least one solution. `charts/degenerate_group_fraction` measures the fraction of groups
with identical rewards. Syntax errors and generation truncations should be zero.

MLflow and stdout `step` count completed program episodes, including failed
and truncated programs. Each rollout batch adds `num_tasks * group_size` steps;
`iteration` counts rollout batches. `time/steps_per_second` measures episodes
per second. `charts/episode_length_mean` counts AST expansions, excluding PAD.
`charts/generation_rounds_mean` counts encoding rounds per program, while
`charts/program_token_length_mean` counts printed source tokens.

`ckpt_dir` selects the Orbax checkpoint directory; `run_id` identifies the MLflow
run. Reusing the directory resumes its latest checkpoint. Set `run_id: null` in
YAML for a timestamped ID; the presets include it in `ckpt_dir`.
Checkpoints are written at completed rollout boundaries every
`checkpoint_interval_seconds`, plus initial and final checkpoints. Local and
GCS paths are supported; the latest two checkpoints are retained.
Legacy `checkpoint.msgpack` files can still be restored.

The checkpoint contains model weights, optimizer state/update count, JAX and
NumPy RNG states, completed-rollout count, and logging/episode counters.
The NumPy RNG drives both task
sampling and minibatch shuffling; there is no separate dataset cursor or live
environment state to save at a rollout boundary. No duplicate weights-only
export is written. `config.yaml` records the current invocation separately.

Resume uses the **current config**. Learning rate, batch/group sizes, rewards,
logging settings, and other settings may change as long as model/optimizer
shapes and structure remain compatible. `total_updates` is the total rollout
target, not an additional count; an already completed run is a no-op. Increasing
it extends training and recomputes the learning-rate schedule with that horizon.
Changing the seed does not reset restored RNG streams. Reference weights in
older checkpoints are ignored. With unchanged settings/backend, restoring a checkpoint preserves
the training trajectory. Unsaved rollouts are replayed; MLflow retains any metrics
already logged beyond the saved step. Older checkpoints restore the step counter
from their saved episode count.

`rl2.train_karel_ast_grpo.load_model()` reads checkpoint weights for inference with a fresh
optimizer; `rl2.train_karel_ast_grpo.generate()` samples complete trees in parallel rounds.
Legacy `params.msgpack` exports can still be loaded for inference, but cannot
resume training because they lack optimizer/RNG state. Old source-token model
checkpoints remain incompatible with this AST policy.
AST checkpoints from the former AST-specific blocks have a different
parameter/optimizer structure from the shared-backbone model, which also changes
initialization and removes Q/K RMSNorm. Start with a new `run_id` for this architecture.

## Configuration and verification

[karel_ast_grpo.yaml](../configs/karel_ast_grpo.yaml) is the source of truth. Currently:

- 8×8 grids; 128 AST nodes, AST depth 16, 128 source tokens, 256 execution ticks.
- Reference sampler: control depth 2, up to 3 statements per block, 10 markers per cell.
- Transformer: width 320, 7 layers, 5 heads of dimension 64; about 9.2M parameters.
- 32 tasks × 8 samples = 256 programs per rollout; 4 minibatches of 64 programs,
  all rounds per minibatch in one forward/backward pass, 1 update epoch.
- 100,000 rollouts; learning rate 0.0001 with linear decay per rollout; clipping 0.2;
  target KL 0.02; gradient norm limit 0.5; entropy coefficient 0.01.

```bash
uv run python -m rl2.train_karel_ast_grpo --config configs/karel_ast_grpo.yaml
JAX_PLATFORMS=cpu uv run pytest tests/test_karel.py tests/test_karel_syntax.py tests/test_karel_ast.py tests/test_ast_transformer.py tests/test_grpo.py
```

Tests cover reward boundaries, simultaneous AST expansion and shared budgets,
per-hole replay probabilities,
PAD masking, single-pass/roundwise gradient equivalence, and training/checkpoint
smoke tests. These checks do not establish learning performance. No supervised
warmup is implemented.
