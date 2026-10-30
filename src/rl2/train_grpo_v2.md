# Tool-calling GRPO

`train_grpo_v2.py` finetunes Qwen3 with a JAX learner, a vLLM inference server,
and a small custom tool-calling interaction loop. Each `tasksets` entry contains
a relative `weight` (default 1) and a `taskset` with `id: alfworld`, `appworld`, or
`reasoning_gym`; the other environment presets are unsupported. For example:

```yaml
tasksets:
  - weight: 3
    taskset:
      id: alfworld
      config_file: data/alfworld/base_config.yaml
  - weight: 1
    taskset:
      id: reasoning_gym
      task: leg_counting
      task_config:
        min_animals: 1
        max_animals: 5
```

For every requested group, select an entry with probability `weight / sum(weights)`,
then sample one task from it. All attempts in the group use that task. Weights
apply to groups, independently of catalog sizes; batches need not match the exact
proportions. Weights must be nonnegative and finite, with at least one positive
weight. Zero-weight entries are neither initialized nor sampled. Entries can
repeat an environment ID with different settings. Replace the old top-level
`taskset` mapping with a one-entry `tasksets` list for single-taskset runs.
The `taskset.*` settings below refer to the nested mapping within each entry.

Prepare the official ALFWorld training data with `alfworld-download`, set
`ALFWORLD_DATA`, and point `taskset.config_file` at a configuration whose
`dataset.data_path` selects the training split. Relative dataset paths resolve
against that configuration file. All six task types and every eligible solvable
training game IDs are discovered at startup; expert observations are disabled.

For [AppWorld](https://github.com/StonyBrookNLP/appworld), install `appworld` in a separate Python environment, then run
`appworld install` and `APPWORLD_ROOT=/data/appworld appworld download data` there.
Set `taskset.python` to that environment's Python executable and `taskset.root`
to the data root in `configs/grpo_v2/appworld.yaml`, and use that preset below.
The adapter discovers IDs from the official training split and exposes `execute(code=...)`.
Each rollout owns a subprocess and temporary output directory. Python variables
persist between calls; API discovery and task completion use the preloaded `apis`
object. Completion awards `float(world.evaluate().success)` once; earlier steps
and rollouts that stop without completion receive zero reward.

Run on Linux x86_64 with Python 3.12 and two visible GPUs:

```bash
uv run python -m rl2.train_grpo_v2 --config configs/grpo_v2/alfworld.yaml
```

For [Reasoning Gym](https://github.com/open-thought/reasoning-gym), use
`--config configs/grpo_v2/reasoning_gym.yaml`; no separate installation is needed.
Set `taskset.task` to a registered dataset name and `taskset.task_config` to its
difficulty parameters. Each group samples a 32-bit dataset seed and generates one puzzle on demand.
Every attempt in a group receives the same question. The top-level `seed`
determines the stream, and the checkpointed task RNG restores its continuation.
Repeated puzzles are possible. There is no `taskset.size`; `task_config` cannot
override `seed` or `size`.
Reasoning Gym also supports internal mixtures: use `task: composite` and put a list of
`{name: leg_counting, weight: 1, config: {}}` entries in `task_config.datasets`.

All tasksets provide an indefinite stream, creating one task per requested group.
ALFWorld and AppWorld retain their discovered ID catalogs and sample uniformly
with replacement; task objects and environments are created on demand.
Task discovery completes during actor startup, outside the async rollout loop.
The checkpointed task RNG restores both weighted selection and each stream's
continuation; keep taskset order, weights, settings, and finite catalogs unchanged
when resuming. A single active taskset preserves its original sampling sequence.

The model calls `submit_answer(answer=...)` with a string in the question's
requested format. One valid submission ends the episode and receives the
dataset's native `score_answer` reward, including partial credit. Plain text
without a tool call and truncated responses receive zero reward. The preset
shares ALFWorld's task budgets, including up to 50 turns for recovery from malformed calls,
but retains Tokamax and one prefetched batch. Each
rollout owns a Ray worker for CPU-bound generation and scoring; timeouts and
cancellation terminate it. Keep task settings and seed unchanged when resuming.

Ray reserves one GPU for the actor's vLLM server and `learner_gpus` GPUs for the
learner (default 1). Multi-GPU training uses fully replicated data parallelism:
base weights, LoRA factors, and optimizer state are replicated on every learner
GPU; each microbatch is split across those GPUs, with gradients averaged globally.
The JIT update declares input/output `PartitionSpec`s and runs inside the learner's
`jax.set_mesh` context; direct callers must also provide a mesh context with a `batch` axis.
Set `learner_gpus` in the YAML and expose at least `learner_gpus + 1` GPUs.
The batcher runs on CPU. The actor and batcher use NumPy arrays and keep
`JAX_PLATFORMS=cpu`; only the learner places training arrays on its GPUs.

The vLLM server uses an 8192-token context limit, at most 128 sequences and 8192
prefill/decode tokens per scheduler iteration, and a 90% GPU memory budget.
Prefix caching reuses shared transcript prefixes; chunked prefill interleaves
long prompt processing with decoding. These settings are fixed in `serve_vllm()`;
the agent also respects the configured `max_tokens` transcript budget.
The server keeps idle HTTP connections alive for 30 seconds, longer than the client's
5-second expiry. Retry messages include the exception type and HTTP status when available;
request payloads are not logged. Request timeouts still use `rollout_timeout`.

## Interaction

Tasks and environments inherit the `Task` and `Environment` abstract base classes in `agent_task.py`.
A task supplies its identity and an async `init_env()` context manager. The training
caller opens the environment within the rollout timeout and passes it to `Agent.run(env)`.
The environment’s async `reset()` returns a `ResetResult` containing `tools`,
`system_prompt`, and `prompt`. Reset must start an active episode with no reward
and at least one valid step available. The prompt includes the initial observation.
Its async `step(action)` returns a `StepResult` containing the observation, per-step
reward, and done flag. The rollout return sums all step rewards, including both
intermediate and terminal rewards. Tool responses contain only the observation
text; rewards and done flags remain internal to the rollout loop.
`ALFWorldTask.init_env()` owns worker startup and cleanup; ALFWorld
validates its own tool arguments. The agent loop has no ALFWorld imports.

Each rollout owns one ALFWorld Ray actor using Ray's default CPU scheduling and follows this loop:

1. Reset the selected game and give the model the action-tool schemas, instructions, goal, starting observation, and admissible commands.
2. Sample a response through `AsyncOpenAI.completions.create()` against vLLM's `/v1/completions` endpoint.
3. Parse Qwen3 `<tool_call>` JSON blocks and dispatch the calls to the ALFWorld worker.
4. Append observation text in `<tool_response>` blocks and repeat until done or a limit is reached.

The tools match the original ALFWorld interface: `look`, `go_to`, `take`, `move`,
`open_container`, `close_container`, `clean`, `heat`, `cool`, `use`, `examine`, and
`inventory`. Every `look` executes the game command. `move` selects
the supported placement grammar from the current admissible commands.

Thinking is enabled, and tool calls inside thinking are ignored. The model is
instructed to use one tool per turn; multiple calls execute sequentially until
the episode ends. Malformed calls receive tool-error feedback without changing
the game. A response without tool calls ends the episode. There are no MCP
servers, renderer pools, or Verifiers objects in this path. The tokenizer and
SDK client are shared across concurrent rollouts; episode state is independent.
ALFWorld calls await Ray object references; AppWorld uses asynchronous subprocess
I/O. Tokenization runs in threads.

The transcript is append-only, using Qwen3 message boundaries. Sampled history,
including thinking and end-of-turn tokens, is never re-tokenized. Every sampled
token has its original log probability and a true loss mask. Prompts,
observations, and inserted boundaries have a false mask. Example logs decode the
full token sequence from the beginning, preserving system instructions, tool
schemas, observations, thinking, and special tokens. Tool results that were
never appended to the model context are excluded.

ALFWorld reward is its `won` value. Stops are `environment_done`, `max_turns`,
`max_tokens`, `context_length`, or `agent_completed`. A truncated model response
is retained for training but its tool calls are not executed. The next generation budget is capped
by remaining context; observations that cannot fit end the rollout without
truncating existing tokens. Episodes with no sampled tokens fail the group.

`rollout_timeout` bounds the whole episode; `environment_timeout` bounds task
discovery and each reset/step request. HTTP failures, malformed sampling data,
worker errors, and timeouts abort the group and cancel its other rollouts.
Environments close normally on success; errors and cancellations terminate ALFWorld
actors immediately. AppWorld workers receive EOF and have five seconds to close
before their process group is killed. Training enables Ray's Linux subreaper to clean up TextWorld
child processes even after a forced termination. The SDK retries transient HTTP failures
up to three times; there are no fallback rewards for infrastructure failures.

## Training and resuming

The actor samples `group_size` independent attempts at the same task.
`length_penalty` charges successful rollouts for generated tokens across all turns,
including thinking and tool-call text, excluding prompts and observations:

```yaml
length_penalty:
  reference_tokens: 8192
  penalty: 0.05
```

When enabled, rewards must be binary (`0` for failure, `1` for success); other
rewards raise an error. Training reward is
`reward * (1 - penalty * min(num_output_tokens / reference_tokens, 1))`.
Failures remain zero; every success remains positive. The cost starts with the
first generated token and caps at `penalty`. `reference_tokens` is a positive
fixed scale, independent of `max_tokens`; `penalty` must be in `[0, 1)`.
Defaults are 8192 and 0 (disabled). ALFWorld enables 0.05; Reasoning Gym leaves
it disabled because its scoring supports partial credit. Disabled penalties
preserve arbitrary task rewards. There is no warmup schedule.

Advantages are `training_reward - group_mean`, with no standard-deviation scaling,
even when the penalty is disabled. This preserves the penalty coefficient's
strength in all-success groups; all-failure groups have zero advantages.
The previous `start_fraction` / `end_fraction` transcript ramp is removed.
This also changes advantage scale from earlier runs; use a new run directory
and compare against a no-penalty baseline with the same mean-centering.
Policy-loss reduction remains a mean over sampled tokens per rollout, then
across rollouts. Monitor raw success and successful-rollout lengths together.

Groups with equal adjusted rewards have zero advantages. The batcher gathers
`num_groups` concurrently and preserves request order. Set
`drop_degenerate_groups: true` (default false) to discard groups with equal
adjusted rewards and sample replacement tasks until all `num_groups` slots
contain non-degenerate groups. There is no retry limit; collection can continue
indefinitely if every group is degenerate. Batch metrics describe retained groups.
Set `continual_group_filling: true` (default false) to keep sampling the same task
until its group has different adjusted rewards. After the initial parallel
`group_size` attempts, each additional attempt replaces the oldest sample;
advantages are recomputed over the newest `group_size` samples.
`max_resampling_attempts: 100` limits additional samples after the initial group
(nonnegative; `0` disables retries). At the limit, the latest group is returned
even if still degenerate. When both options are enabled, the batcher's
`drop_degenerate_groups` filter discards such groups and samples replacement tasks.
Metrics and training include only retained samples.
`rollout/reward_*` reports raw task rewards; `rollout/adjusted_reward_*` reports
adjusted rewards.
The batcher pads complete transcripts to `padding_multiple` (default 256),
capped at the model context length. Oversized transcripts are rejected.

GRPO uses [verl-style decoupled token-TIS](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html#decoupled-mode-with-token-level-importance-sampling-decoupled-token-is),
averaged over sampled tokens within each rollout and then across rollouts.
Before optimizing each batch, the learner scores its transcripts once using the
optimizer's microbatch shape and temperature. These anchor scores stay fixed
across all minibatches and epochs; the original vLLM sampling scores are retained.
For each sampled token:

- PPO ratio: `r = exp(clip(current_logp - anchor_logp, -20, 20))`.
- Detached correction: `w = min(exp(clip(anchor_logp - rollout_logp, -20, 20)), rollout_is_threshold)`.
- Loss: `-w * min(r * advantage, clip(r, 1 - clip_coef, 1 + clip_coef) * advantage)`.

`rollout_is_threshold` defaults to `2.0` and must be positive. Correction weights
are zero on context/padding tokens, are not batch-normalized, and do not change
loss denominators. There is no rejection sampling or reference-policy KL penalty.
Token-level correction and truncation trade bias for lower variance; they do not
establish numerical parity or exactly correct the trajectory distribution.
The anchor pass adds a forward evaluation; entirely zero-advantage batches skip
its execution. With one optimizer update, the PPO ratio starts near one but
retains gradients; rollout correction still weights those gradients.
`train/approx_kl` and `train/clip_fraction` measure current-versus-anchor drift.
Separate `train/rollout_log_prob_error_mean`, `train/rollout_importance_weight_mean`,
and `train/rollout_importance_weight_clipped_fraction` expose the anchor/rollout
gap and applied correction. All policy diagnostics exclude skipped minibatches.

Both finetuning modes use AdamW with reference defaults `learning_rate: 1e-6`,
Optax's `beta1: 0.9`, `beta2: 0.999`, and `eps: 1e-8`, and `weight_decay: 0.0`
(disabled). Set `weight_decay` in the YAML to any nonnegative value.
When enabled, decay applies to all trainable parameters, including LoRA factors in LoRA mode.

`adamw_batch_size_scale: 1.0` leaves AdamW unchanged. Set this factor (`k`) to the
new/reference optimizer batch-size ratio (for example, `0.5` or `2.0`) to scale
learning rate and weight decay by
`sqrt(k)`, use `beta1 = 1 - 0.1*k` and `beta2 = 1 - 0.001*k`, and divide epsilon
(`1e-8`) by `sqrt(k)`. `learning_rate` and `weight_decay` remain reference values;
`train/learning_rate` reports the effective scaled rate. Require `0 < k <= 10`
to keep beta1 nonnegative; `adamw_batch_size_scale: 10` disables first-moment smoothing.
This is an approximate scaling rule; weight-decay scaling assumes equal numbers
of processed rollouts across runs.

`adamw_batch_size_scale` only scales the optimizer. Configure the actual batch separately:
`num_groups * group_size / minibatches_per_epoch` rollouts per update.
Changing only `grad_accum_steps` does not change that batch size.

Set `finetuning_mode: full` to optimize all model weights, including
embeddings, normalization, and the output head, with FP32 master weights and Adam
moments. This uses substantially more GPU memory than LoRA. Tied embeddings and
output heads remain shared. `lora_rank` and `lora_alpha` apply only in LoRA mode.
Transformer computation uses BF16; the output
projection accumulates FP32 logits in both JAX and vLLM. In the default
`finetuning_mode: lora`, base weights are frozen;
only FP32 LoRA factors on every attention and MLP projection are optimized.
`lora_rank` defaults to 16 and `lora_alpha` to 32; the weight delta is
`(lora_alpha / lora_rank) * B @ A`. A is randomly initialized and B starts at zero,
so training starts from the base policy. Embeddings, normalization, and the output
head stay frozen. Both JAX and vLLM compute separate base and LoRA branches.
The FP32 factors are cast to BF16; alpha/r is folded into BF16 B and rounded again.
The shrink result and expanded delta each round to BF16 before the delta is added
to the BF16 base output. JAX keeps these rounding boundaries in the gradient graph.
Only the FP32 master factors receive gradients; Adam moments stay FP32 and base weights stay BF16.
Ranks from 1 through 512 are accepted, with vLLM capacity rounded up as needed.
The native-LoRA GPU audit passes tiny models but exceeds full-model parity tolerances.
Optimizer state, gradients, log probabilities, and loss use FP32. Learner
`attention_backend` defaults to `cudnn`, which had lower worst-case sampled-token
error in the A100 vLLM audit; `tokamax` and `xla` remain available. `scan_layers`
and `gradient_checkpointing` default to true. `nvida.env` selects O1, 90% GPU memory
preallocation, full FP32 matmul precision, and disables TF32. BF16 model computation
and FP32 logits remain enabled. See the
[publication parity suite](../../docs/qwen3.md#parity-with-the-grpo-v2-vllm-server).
The [current audit](../../qwen3_vllm_parity.md) still finds out-of-range ratios
with unchanged weights at 4,096 tokens, for both Tokamax and cuDNN. Corrected RoPE
frequencies and FP32 log-softmax reduce specific discrepancies but do not certify
training parity. Preserve the actor's sampling scores as the rollout-correction denominator.
Temperature is restricted to `[0.01, 2.0]`; vLLM returns
`processed_logprobs` with neutral top-p/top-k sampling, matching the learner.

Each iteration makes `update_epochs` passes with `minibatches_per_epoch` optimizer
minibatches per pass. `grad_accum_steps` splits each minibatch into microbatches.
Minibatches with all-zero advantages skip forward/backward computation and preserve
parameters, Adam state, and the optimizer step. `train/skipped_update_fraction` reports the
skipped fraction. Skipped minibatches contribute zero to policy loss and gradient norm.
Policy diagnostics average only evaluated minibatches and are omitted when all updates are skipped.
`minibatches_per_epoch * grad_accum_steps` must divide `num_groups * group_size`.
The resulting microbatch size must also be divisible by `learner_gpus`.
Shuffling uses the saved seed and optimizer step. The learner selects a context-managed
publisher from `grpo_v2/weight_transfer.py` at startup. Both modes expose the same
callable, read the current parameters on every call, and own their HTTP and transfer
resources. Publication runs in the learner; the actor handles rollouts.
In LoRA mode, publication copies only FP32 LoRA
factors from one replica into PEFT safetensors and configuration files in a temporary
local directory shared with vLLM. vLLM starts with LoRA enabled, casts adapters to
BF16, and replaces the `training-policy` adapter through `/v1/load_lora_adapter`
with `load_inplace: true`. Rollout requests select that adapter; the base model is
loaded once. A small server wrapper clears vLLM 0.32's persistent `load_inplace`
flag after a successful load, preventing repeated adapter reads during decoding.
In full mode, vLLM starts with its NCCL weight-transfer backend and requests select
the base model name. A persistent two-rank NCCL group connects the first learner
GPU to the single vLLM worker. Each publication reads the current JAX replica
through DLPack and casts weights to BF16 on the GPU, including the output head.
vLLM's trainer engine streams packed GPU buffers over NCCL; HTTP carries only
parameter names, shapes, dtypes, and transfer control. There are no host weight
copies or temporary weight files. Packing uses two buffers with a 64 MiB flush
threshold; a chunk can exceed that threshold by one tensor. Transfer completion
is awaited before the learner can reuse buffers. FP32 master weights stay in the
learner and checkpoints. The NCCL group is closed when the learner exits.
To check repeated NCCL publication on two GPUs, run
`QWEN3_VLLM_TESTS=1 CUDA_VISIBLE_DEVICES=0 QWEN3_VLLM_DEVICE=1 uv run pytest tests/test_qwen3_vllm.py -k nccl_full_weight_publication`.
The learner waits for the actor and vLLM to become ready before initializing its publisher.
The initial weights are published before any rollouts. `parameter_sync_step: 1`
(default) publishes after each training batch; `4` publishes every fourth batch,
regardless of PPO minibatch count or update epochs. All completed batches count,
including batches with skipped updates. Scheduled publication runs only if the
optimizer step has advanced since the last publication. The cumulative iteration
count preserves the schedule across restarts. Startup and resume also publish
immediately. Checkpoints and training completion do not force an extra publication
before the interval ends.
`learner/weight_transfer_seconds` is omitted when no publication runs;
`learner/update_seconds` excludes publication time.
Each LoRA publication removes its
temporary directory after the blocking weight update returns, including on failure.
The loaded weights stay in vLLM memory. This trainer runs on one host; LoRA also
requires a shared local filesystem.
`prefetch_batches: 0` (default) collects each batch without overlapping training,
avoiding weight changes within a rollout. With `parameter_sync_step: 1`, each
batch uses the latest learner weights; longer sync intervals can retain older weights.
Positive `prefetch_batches: N` lets the batcher maintain a pool of up to
`(N + 1) * num_groups` pending rollout groups, including the current batch.
Each learner request consumes `num_groups` groups and leaves up to N batches' worth
running or ready while the learner optimizes. `prefetch_mode: next_in_order` (default)
selects groups in submission order. Set `prefetch_mode: next_ready` to assemble each
batch from ready groups across the pool, without waiting for slower groups submitted
alongside them. When multiple groups are ready, their selection order is unspecified.
Groups stay intact, and degenerate groups are
replaced within their pool slots when filtering is enabled. Both modes drain the
pool at checkpoint intervals. Prefetching can extend beyond `total_iterations`;
unused groups are canceled during shutdown. Prefetched rollouts can use older
weights, and multi-turn rollouts can span weight updates.
`weight_update_pause_mode` accepts `wait` (default) or `keep`, operating
on individual generation requests. Groups can start during publication; vLLM
queues new generation requests until resume and handles in-flight requests
according to the selected pause mode. Publication awaits `/pause` with cache clearing,
updates the weights, then calls `/resume`. Full mode uses vLLM's
`/start_weight_update`, `/update_weights`, and `/finish_weight_update` lifecycle.
Transfer failures leave generation paused,
propagate to the learner, and trigger training cleanup.
With vLLM 0.32, `keep` also honors `clear_cache=true`: running requests are
requeued and their KV state is recomputed after resume. Tokens within one request
can still be sampled under different weight versions.
Use `wait` for a consistent weight version within each generation request; a
multi-turn rollout can still span updates.

`ckpt_dir/checkpoints` stores Orbax checkpoints every `checkpoint_interval`
iterations and at normal completion, retaining only the latest checkpoint.
Orbax stages the arrays before returning, then writes asynchronously while training
continues. A subsequent save waits for any previous save, and shutdown waits for
outstanding writes and propagates save failures. `checkpoint/save_blocking_seconds`
measures save dispatch, including staging and any previous-save wait, excluding
the new checkpoint's background writes.
Prefetching drains at checkpoint intervals so the saved task RNG does not
skip unconsumed batches. A final checkpoint outside that interval can include RNG
advancement from unused prefetched groups. Restarting with the same directory restores model weights (including LoRA factors in LoRA mode),
optimizer state, shuffle key, task RNG, and progress counters. `total_iterations` is a cumulative
target; `null` runs until interrupted. Environments restart fresh, so resumption
is not bit-for-bit reproducible. Use a new directory for a fresh experiment or
incompatible model changes. Checkpoints require matching `model` and `finetuning_mode`;
LoRA also requires matching `lora_rank` and `lora_alpha`. Existing version 2
checkpoints without `finetuning_mode` are treated as LoRA. Legacy version 1
checkpoints cannot be resumed. Use a new `ckpt_dir` when switching modes.
Local and `gs://` paths are supported.

## Logging and profiling

Console logging defaults to INFO. The driver and workers use standard Python logging
with timestamp, severity, source location, and message, omitting Ray IDs
and other metadata. vLLM HTTP access logs and periodic statistics INFO messages
are suppressed; Prometheus metrics, warnings, and errors remain enabled.
The rollout actor sets HTTPX logging to WARNING to suppress routine
HTTP request messages while retaining warnings and errors. ALFWorld loader
initialization output is suppressed on success and replayed on failure; output
from environment steps remains enabled. Existing cluster jobs need a new image and
resubmission to pick up logging changes.

MLflow uses experiment `grpo_v2` and the standard tracking environment variables.
The [module docstring](train_grpo_v2.py) defines metric names and aggregation.
Every `example_log_interval` iterations (default 10, positive integer), it saves
full token transcripts, final observations, rewards, and advantages. The batcher
formats up to eight examples only for iterations that log them, using the consumed
groups even when prefetches finish out of order.
Stop fractions include all `StopCondition` labels, with zero for absent labels.
`rollout/total_tokens_mean` is the mean full transcript length in the learner
batch, counting context and sampled tokens and excluding padding, before the
next-token shift.
`batcher/padded_sequence_tokens` records the padded batch width; the learner's
forward pass uses one fewer position for next-token prediction.

`enable_tracing` defaults to true and requires an OTLP endpoint. `submit.py`
supplies the cluster's Tempo endpoint. Spans cover iterations, rollout groups,
episodes, environment reset/step calls, model requests, batching, updates,
weight transfer, logging, and checkpoints. Model spans count full request input
tokens; `rollout/input_tokens_mean` counts context tokens appended per episode.
See [cluster tracing access](../../infra/k3s/README.md#opentelemetry-traces-with-tempo).

`enable_ray_dashboard` and `enable_ray_timeline` default to false. Enable them
in the training YAML. Submitted jobs expose their dashboard and saved timeline
through [Training jobs](http://localhost:8085/); forward port 8085 through SSH or
your IDE. Local dashboards use port 8265. Timelines are saved every 60 seconds
and at graceful shutdown, under `ckpt_dir/profiles` unless overridden by
`RL2_RAY_TIMELINE_PATH`. These traces measure host-side intervals, not GPU kernels.

