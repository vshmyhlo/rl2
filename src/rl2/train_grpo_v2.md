# Tool-calling GRPO

`train_grpo_v2.py` trains Qwen3 LoRA adapters with a JAX learner, a vLLM inference server,
and a small custom tool-calling interaction loop. V2 accepts
`taskset.id: alfworld` or `appworld`; the other environment presets are unsupported.

Prepare the official ALFWorld training data with `alfworld-download`, set
`ALFWORLD_DATA`, and point `taskset.config_file` at a configuration whose
`dataset.data_path` selects the training split. Relative dataset paths resolve
against that configuration file. All six task types and every eligible solvable
training game are loaded; expert observations are disabled.

For [AppWorld](https://github.com/StonyBrookNLP/appworld), install `appworld` in a separate Python environment, then run
`appworld install` and `APPWORLD_ROOT=/data/appworld appworld download data` there.
Set `taskset.python` to that environment's Python executable and `taskset.root`
to the data root in `configs/grpo_v2/appworld.yaml`, and use that preset below.
The adapter loads the official training split and exposes `execute(code=...)`.
Each rollout owns a subprocess and temporary output directory. Python variables
persist between calls; API discovery and task completion use the preloaded `apis`
object. Completion awards `float(world.evaluate().success)` once; earlier steps
and rollouts that stop without completion receive zero reward.

Run on Linux x86_64 with Python 3.12 and two visible GPUs:

```bash
uv run python -m rl2.train_grpo_v2 --config configs/grpo_v2/alfworld.yaml
```

Ray reserves one GPU for the actor's vLLM server and `learner_gpus` GPUs for the
learner (default 1). Multi-GPU training uses fully replicated data parallelism:
base weights, LoRA factors, and optimizer state are replicated on every learner
GPU; each microbatch is split across those GPUs, with gradients averaged globally.
The JIT update declares input/output `PartitionSpec`s and runs inside the learner's
`jax.set_mesh` context; direct callers must also provide a mesh context with a `batch` axis.
Set `learner_gpus` in the YAML and expose at least `learner_gpus + 1` GPUs.
The batcher runs on CPU. The actor and batcher use NumPy arrays and keep
`JAX_PLATFORMS=cpu`; only the learner places training arrays on its GPUs.

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

The actor samples `group_size` independent attempts at the same task. Set
`length_penalty_coef` (default `0.0`, nonnegative) to subtract a reward cost per
sampled token across all turns, including thinking and end-of-turn tokens, only
when the environment reward is nonzero:
`training_reward = reward - length_penalty_coef * num_output_tokens`.
Zero environment rewards remain zero regardless of rollout length.
Advantages are `(training_reward - group_mean) / (population_std + 1e-4)`;
groups with equal adjusted rewards have zero advantages. The batcher gathers
`num_groups` concurrently and preserves request order. Set
`drop_degenerate_groups: true` (default false) to discard groups with equal
adjusted rewards and sample replacement tasks until all `num_groups` slots
contain non-degenerate groups. There is no retry limit; collection can continue
indefinitely if every group is degenerate. Batch metrics describe retained groups.
`rollout/reward_*` reports raw task rewards; `rollout/adjusted_reward_*` reports
adjusted rewards. Within groups with equal nonzero task rewards, normalization can amplify even
a small length penalty into substantial advantages favoring shorter rollouts.
The batcher pads complete transcripts to `padding_multiple` (default 256),
capped at the model context length. Oversized transcripts are rejected.

GRPO uses a clipped importance-ratio objective, averaged over sampled tokens
within each rollout and then across rollouts. There is no reference-policy KL
penalty. Transformer computation uses BF16; the output
projection accumulates FP32 logits in both JAX and vLLM. Base weights are frozen;
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
`attention_backend` accepts `tokamax` (default), `xla`, or `cudnn`; `scan_layers` and
`gradient_checkpointing` default to true. These defaults match the standalone
Qwen3 model and the [publication parity suite](../../docs/qwen3.md#parity-with-the-grpo-v2-vllm-server).
BF16 parity is approximate: the real Qwen3-0.6B native-LoRA test reaches 0.179 maximum
log-probability error at 256 tokens, above its 0.12 limit. See the
[measured limits](../../docs/qwen3.md#native-lora-audit-2026-10-27);
zero false clipping on these samples does not certify longer production contexts.
Temperature is restricted to `[0.01, 2.0]`; vLLM returns
`processed_logprobs` with neutral top-p/top-k sampling, matching the learner.

Each iteration makes `update_epochs` passes with `minibatches_per_epoch` optimizer
minibatches per pass. `grad_accum_steps` splits each minibatch into microbatches.
Minibatches with all-zero advantages skip forward/backward computation and preserve
parameters, Adam state, and the optimizer step. `train/skipped_update_fraction` reports the
skipped fraction; skipped minibatches contribute zero to the other optimization metrics.
`minibatches_per_epoch * grad_accum_steps` must divide `num_groups * group_size`.
The resulting microbatch size must also be divisible by `learner_gpus`.
Shuffling uses the saved seed and optimizer step. Publication copies only FP32 LoRA
factors from one replica into PEFT safetensors and configuration files in a temporary
local directory shared with vLLM. vLLM starts with LoRA enabled, casts adapters to
BF16, and replaces the `training-policy` adapter through `/v1/load_lora_adapter`
with `load_inplace: true`. Rollout requests select that adapter; the base model is
loaded once. A small server wrapper clears vLLM 0.32's persistent `load_inplace`
flag after a successful load, preventing repeated adapter reads during decoding.
The initial adapter is published before any rollouts. Later publications occur only
when the optimizer step advances; iterations with all minibatches skipped omit
`learner/weight_transfer_seconds`. Each publication removes its
temporary directory after the blocking weight update returns, including on failure.
The single adapter stays in vLLM memory. Learner and server must share a local filesystem.
`prefetch_batches: N`
(default 1) generates up to N batches ahead while the learner optimizes; each
batch contains `num_groups` rollout groups. The learner consumes batches in
submission order, waiting for the oldest pending batch. Prefetched rollouts can use older
weights, and multi-turn rollouts can span weight updates. Set it to 0 to collect
each batch after publishing the latest weights.
`weight_update_pause_mode` accepts `wait` (default), `abort`, or `keep`, operating
on individual generation requests. Groups can start during publication; vLLM
queues new generation requests until resume and handles in-flight requests
according to the selected pause mode. The actor awaits `/pause` with cache clearing,
reloads the adapter, then calls `/resume`. Reload failures leave generation paused,
propagate to the learner, and trigger training cleanup.
With vLLM 0.32, `keep` also honors `clear_cache=true`: running requests are
requeued and their KV state is recomputed after resume. Tokens within one request
can still be sampled under different adapter versions.
Use `wait` for a consistent adapter version within each generation request; a
multi-turn rollout can still span updates.

`ckpt_dir/checkpoints` stores Orbax checkpoints every `checkpoint_interval`
iterations and at normal completion, retaining only the latest checkpoint.
Prefetching drains on checkpoint iterations so the saved task RNG does not
skip unconsumed batches. Restarting with the same directory restores frozen base weights, LoRA factors,
optimizer state, shuffle key, task RNG, and progress counters. `total_iterations` is a cumulative
target; `null` runs until interrupted. Environments restart fresh, so resumption
is not bit-for-bit reproducible. Use a new directory for a fresh experiment or
incompatible model changes. LoRA checkpoints require matching `model`, `lora_rank`,
and `lora_alpha`; older full-parameter checkpoints cannot be resumed.
Local and `gs://` paths are supported.

## Logging and profiling

Console logging defaults to INFO. The driver and workers use standard Python logging
with timestamp, severity, source location, and message, omitting Ray IDs
and other metadata. vLLM uses its default logging configuration with HTTP
access logs disabled. The rollout actor sets HTTPX logging to WARNING to suppress routine
HTTP request messages while retaining warnings and errors. ALFWorld loader
initialization output is suppressed on success and replayed on failure; output
from environment steps remains enabled. Existing cluster jobs need a new image and
resubmission to pick up logging changes.

MLflow uses experiment `grpo_v2` and the standard tracking environment variables.
The [module docstring](train_grpo_v2.py) defines metric names and aggregation.
Every `example_log_interval` iterations (default 10, positive integer), it saves
full token transcripts, final observations, rewards, and advantages. Every batch
carries up to eight examples so logs match the consumed batch even when prefetches
finish out of order.
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

