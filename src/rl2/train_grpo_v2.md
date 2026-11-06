# Tool-calling GRPO

`train_grpo_v2.py` finetunes Qwen3 with a JAX learner, a vLLM inference server,
and a small custom tool-calling interaction loop. `tasksets` is a mapping keyed by
unique, lowercase `snake_case` taskset identifiers (starting with a letter). Each entry contains
a relative `weight` (default 1) and a `taskset` with `id: alfworld`, `appworld`,
`codegym`, `reasoning_gym`, or `webshop`; the other environment presets are unsupported. For example:

```yaml
tasksets:
  household:
    weight: 3
    taskset:
      id: alfworld
      config_file: data/alfworld/base_config.yaml
  easy_leg_counting:
    weight: 1
    taskset:
      id: reasoning_gym
      task: leg_counting
      task_config:
        min_animals: 1
        max_animals: 5
```

For every requested group, select an entry using deterministic smooth weighted
round robin: add `weight / sum(weights)` to each entry's score, select the highest
score (breaking ties by identifier), and subtract 1 from that score. Scores start
at zero. Equal weights cycle through identifiers in sorted order; unequal weights
spread selections according to their proportions. Then sample one task from the
selected entry using the seeded task RNG. All attempts in the group use that task.
Weights apply to requested groups, independently of catalog sizes; individual
batches need not match the exact proportions. Weights must be nonnegative and finite, with at least one positive
weight. Zero-weight entries are neither initialized nor sampled. Entries can
repeat an environment ID with different settings under distinct keys. The keys
identify configured tasksets; nested `taskset.id` selects the environment type.
Replace each item in an old `tasksets` list with a named mapping entry. For a
single-taskset run, use a one-entry mapping. YAML duplicate keys are rejected.
The `taskset.*` settings below refer to the nested mapping within each entry.

Prepare the official ALFWorld training data with `alfworld-download`, set
`ALFWORLD_DATA`, and point `taskset.config_file` at a configuration whose
`dataset.data_path` selects the training split. Relative dataset paths resolve
against that configuration file. All six task types and every eligible solvable
training game IDs are discovered at startup; expert observations are disabled.

For [WebShop](https://github.com/princeton-nlp/WebShop), use `configs/grpo_v2/webshop.yaml`
and set `taskset.root` to the upstream checkout on each Ray node. Prepare the full
product catalog, human instructions, and full search index using upstream's setup.
Install its runtime dependencies (including Java for Pyserini and the spaCy model)
in the same Python environment as Ray. Upstream's old pinned requirements are not
a lockfile for this project's Python/dependency versions; use a compatible installation.
There is no separate `taskset.python` executable. The actor adds the checkout to
its `PYTHONPATH`; other tasksets do not import WebShop's optional dependencies.

Each rollout owns one Ray actor wrapping the text Gym environment. The model uses
`search(keywords=...)` and `click(value=...)`; observations include available actions.
Task discovery uses all human goals after the 500 test and 1,000 development goals.
Product prices and goal ordering use the upstream fixed seed, so attempts share the
same task. Step rewards pass through unchanged; purchase terminates the episode,
and subsequent steps cannot repeat its reward. Timeouts and cancellation kill the actor.

For [AppWorld](https://github.com/StonyBrookNLP/appworld), install `appworld` in a separate Python environment, then run
`appworld install` and `APPWORLD_ROOT=/data/appworld appworld download data` there.
Set `taskset.python` to that environment's Python executable and `taskset.root`
to the data root in `configs/grpo_v2/appworld.yaml`, and use that preset below.
The adapter discovers IDs from the official training split and exposes `execute(code=...)`.
Each rollout owns a subprocess and temporary output directory. Python variables
persist between calls; API discovery and task completion use the preloaded `apis`
object. Completion awards `float(world.evaluate().success)` once; earlier steps
and rollouts that stop without completion receive zero reward.

For [CodeGym](https://github.com/StigLidu/CodeGym), deploy the
[cluster server](../../infra/k3s/codegym/README.md), then use
`configs/grpo_v2/codegym.yaml`. `taskset.manager_url` addresses the upstream
HTTP manager; its allocated worker ports must be reachable on the same host.
The preset uses the internal cluster DNS name and all four pinned English
training shards. `taskset.dataset_file` also accepts local JSONL or Parquet files.
Rows must contain system function documentation followed by a user message
with a `Question:` section, an `ability` specification, and optionally
`plugin_call_max_round` (default 256).

The model calls individual functions such as `Observe()` and `Done(answer=...)`.
The server generates their JSON schemas from the deployed Python source at image
build time and serves them through `/tools/{environment_name}`. Only actions in
`func_mapping` are exposed; dispatch code determines argument names and required
fields, annotations provide types, and docstrings provide descriptions. Missing
or unsupported types remain unrestricted. The adapter retains the question and
uses these tool descriptions instead of the dataset’s textual function listing.
Each attempt
owns an independent server session; reset and rollout exit stop it and release
the worker. Server failures propagate as infrastructure errors. CodeGym's
binary score is `float(upstream_reward > 0)`; step rewards are changes in that
score, so a persistent success is counted once. Episodes stop when the server
finishes or the row's call limit is reached, subject to the agent's turn, token,
and timeout budgets. No final-text answer grading is performed.

The preset matches `mix.yaml` training settings, including full finetuning,
8192 transcript tokens, 32 rollouts per batch, and four prefetched batches.
It needs capacity for up to 160 concurrent episodes. Increase the deployed
40-worker pool and its allowed worker ports before running this preset, or set
`prefetch_batches: 0` to fit that pool. Account for other jobs sharing the server.

Run on Linux x86_64 with Python 3.12 and two visible GPUs:

```bash
uv run python -m rl2.train_grpo_v2 --config configs/grpo_v2/alfworld.yaml
```

For [Reasoning Gym](https://github.com/open-thought/reasoning-gym), use
`--config configs/grpo_v2/reasoning_gym.yaml`; the datasets need no separate installation.
Set `taskset.task` to a registered dataset name and `taskset.task_config` to its
difficulty parameters. Each group samples a 32-bit dataset seed and generates one puzzle on demand.
Every attempt in a group receives the same question. The top-level `seed`
determines the stream, and the checkpointed task RNG restores its continuation.
Repeated puzzles are possible. There is no `taskset.size`; `task_config` cannot
override `seed` or `size`, including in composite child configs.
Reasoning Gym also supports internal mixtures: use `task: composite` and put a list of
`{name: leg_counting, weight: 1, config: {}}` entries in `task_config.datasets`.

All tasksets provide an indefinite stream, creating one task per requested group.
ALFWorld, AppWorld, and WebShop retain their discovered ID catalogs and sample uniformly
with replacement; task objects and environments are created on demand.
CodeGym samples uniformly from its loaded task rows with replacement.
Task discovery completes during actor startup, outside the async rollout loop.
Checkpoints save round-robin scores and the task RNG to restore taskset selection
and each stream's continuation. Older checkpoints without scores start a fresh
round-robin schedule. Keep taskset identifiers, weights, settings, and finite catalogs
unchanged when resuming. Sampling orders entries by identifier, so YAML key order
does not affect the sequence. A single active taskset preserves its original sampling sequence.

For Reasoning Gym, the model sees `python(code=...)` and `submit_answer(answer=...)`.
The calculator tool is disabled and rejected; Python handles arithmetic too.
The Python tool lazily creates one network-disabled gVisor pod and persistent
interpreter per rollout, with SymPy 1.14.0 (`import sympy`) for symbolic math and
Z3 4.15.4 (`import z3`) for constraint solving preinstalled alongside the standard library.
Variables and files persist across calls; use `print()` for output.
Observations are JSON with `output` (stdout/stderr), `exit_code`, and
`output_truncated`. Python exceptions return exit code 1 with zero reward and preserve
state. Successful calls also receive zero reward. Each execution has a 5-second limit;
startup has a separate 90-second limit, and the environment/rollout deadlines still apply.
Execution timeouts, interpreter exit, or invalid responses end that rollout with zero
reward and a JSON `{"error": "..."}` observation; startup and cleanup failures still
propagate as infrastructure errors.
Reset and rollout exit delete the sandbox,
including when generation ends without submitting an answer.

Python requires the [sandbox prerequisites](sandbox/README.md#prerequisites), including
the built `rl2.local/python-sandbox:1` image on each sandbox node, `kubectl`, and
Kubernetes credentials in the rollout caller. Taskset fields
`python_namespace` (default `sandbox`) and `python_kubectl` (default `[kubectl]`)
configure access. For training launched directly on the k3s host, use
`python_kubectl: [sudo, -n, k3s, kubectl]`. For cluster training, the image includes
`kubectl`; submit with `--python-sandbox` to mount the dedicated service-account token.
Apply `infra/k3s/training/python-sandbox.yaml` once to grant pod management and exec
access only in `sandbox`. The node still needs the offline runsc runtime configured.
Credentials stay in the trusted caller, never in the Python sandbox.

The model calls `submit_answer(answer=...)` with a string in the question's
requested format. One valid submission ends the episode and receives the
dataset's native `score_answer` reward, including partial credit. Game of Life
answers must decode to a JSON list of rows (lists); malformed structures return
formatting feedback and allow resubmission. Incomplete grids retain native partial credit. Plain text
without a tool call and truncated responses receive zero reward. The preset
shares ALFWorld's task budgets, including up to 50 turns for tool use and recovery from malformed calls,
but retains Tokamax and one prefetched batch. Each
rollout owns a Ray worker for CPU-bound generation and scoring; timeouts and
cancellation terminate it. Keep task settings and seed unchanged when resuming.

Ray reserves one GPU for the actor's vLLM server and `learner_gpus` GPUs for the
learner (default 1). Multi-GPU training uses fully replicated data parallelism:
base weights, LoRA factors, and optimizer state are replicated on every learner
GPU; each full batch is split across those GPUs, with gradients averaged globally.
The JIT update declares input/output `PartitionSpec`s and runs inside the learner's
`jax.set_mesh` context; direct callers must also provide a mesh context with a `batch` axis.
Set `learner_gpus` in the YAML and expose at least `learner_gpus + 1` GPUs.
The batcher runs on CPU. The actor and batcher use NumPy arrays and keep
`JAX_PLATFORMS=cpu`; only the learner places training arrays on its GPUs.

The optional `vllm` configuration accepts `max_model_len`, `max_num_seqs`,
`max_num_batched_tokens`, `gpu_memory_utilization`, `enable_prefix_caching`,
`enable_chunked_prefill`, and `async_scheduling`. Omitted values use vLLM's
defaults. Prefix caching reuses shared transcript prefixes; chunked prefill
interleaves long prompt processing with decoding. The agent also respects the
configured `max_tokens` transcript budget.
The [A100 mix benchmark](../../docs/vllm_mix_benchmark_a100.md) records the
serving settings selected for `configs/grpo_v2/mix.yaml` and their limitations.
The server enables scale-out endpoints for the agent's `/inference/v1/generate` requests.
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
All five supported environments set `single_tool_call=True`: responses with multiple
calls receive corrective feedback before any call executes.
Its async `step(action)` returns a `StepResult` containing the observation, per-step
reward, and done flag. The rollout return sums all step rewards, including both
intermediate and terminal rewards. Tool responses contain only the observation
text; rewards and done flags remain internal to the rollout loop.
Each task’s `init_env()` owns environment startup and cleanup, including on errors
and cancellation. Following the contract in [agent_task.py](agent_task.py), each adapter
returns formatting, parsing, and tool-schema errors as actionable `ToolCallError`
feedback without advancing or terminating the environment. The model can correct
and retry within its turn/token budget. Well-formed incorrect answers receive normal
task scoring; infrastructure failures remain errors. Steps after termination require
a reset. The agent loop contains no environment-specific dispatch.

Each rollout owns one ALFWorld Ray actor using Ray's default CPU scheduling and follows this loop:

1. Reset the selected game and give the model the action-tool schemas, instructions, goal, starting observation, and admissible commands.
2. Sample token IDs, log-probabilities, and weight-version spans from vLLM's `/inference/v1/generate` endpoint using the shared async SDK client.
3. Parse Qwen3 `<tool_call>` JSON blocks and dispatch the calls to the ALFWorld worker.
4. Append observation text in `<tool_response>` blocks and repeat until done or a limit is reached.

The tools match the original ALFWorld interface: `look`, `go_to`, `take`, `move`,
`open_container`, `close_container`, `clean`, `heat`, `cool`, `use`, `examine`, and
`inventory`. Every `look` executes the game command. `move` selects
the supported placement grammar from the current admissible commands.

Thinking is enabled, and tool calls inside thinking are ignored. The model is
required to use one tool per turn. Malformed calls receive tool-error feedback without changing
the game. A response without tool calls ends the episode. There are no MCP
servers, renderer pools, or Verifiers objects in this path. The tokenizer and
SDK client are shared across concurrent rollouts; episode state is independent.
ALFWorld, WebShop, and Reasoning Gym scoring await Ray object references; AppWorld
uses asynchronous subprocess I/O, and CodeGym uses asynchronous HTTP. Tokenization runs in threads.

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

`rollout_timeout` bounds the whole episode; `environment_timeout` bounds Ray/subprocess
task discovery and each reset/step request. CodeGym dataset loading uses the dataset
library’s I/O timeouts during startup. A Ray `ActorDiedError` or HTTPX `NetworkError`
(including CodeGym `ReadError`) or `RemoteProtocolError` restarts the affected episode
in a fresh environment with the same task and seed, up to two retries within the
original rollout deadline. Failed attempts are discarded, without fallback rewards.
Individual environment steps are not replayed: the server may have already executed
an action before the connection failed. Actor constructor exceptions are not retried.
Exhausted retries, other worker errors, HTTP status errors,
malformed sampling data, and timeouts abort the group and cancel its other rollouts.
Environments close normally on success; errors and cancellations terminate ALFWorld and WebShop
actors immediately. AppWorld workers receive EOF and are awaited on normal exit;
errors and cancellations kill their process group before awaiting exit. Training enables Ray's Linux subreaper to clean up TextWorld
child processes even after a forced termination. The OpenAI SDK retries transient vLLM HTTP
failures up to three times; there are no fallback rewards for infrastructure failures.

## Training and resuming

The actor samples `group_size` independent attempts at the same task.
`length_penalty` proportionally reduces positive rewards for generated tokens across all turns,
including thinking and tool-call text, excluding prompts and observations:

```yaml
length_penalty:
  reference_tokens: 8192
  penalty: 0.05
```

When enabled, rewards must be in `[0, 1]`, including partial credit; other
rewards raise an error. Training reward is
`reward * (1 - penalty * min(num_output_tokens / reference_tokens, 1))`.
Zero rewards remain zero; positive rewards remain positive. The cost starts with the
first generated token and caps at `reward * penalty`. `reference_tokens` is a positive
fixed scale, independent of `max_tokens`; `penalty` must be in `[0, 1)`.
Defaults are 8192 and 0 (disabled). ALFWorld and the mixed-task preset enable 0.05;
the standalone Reasoning Gym preset leaves it disabled. Disabled penalties
preserve arbitrary task rewards. There is no warmup schedule.

Advantages are `training_reward - group_mean`, with no standard-deviation scaling,
even when the penalty is disabled. This preserves the penalty coefficient's
strength in all-success groups; all-failure groups have zero advantages.
The previous `start_fraction` / `end_fraction` transcript ramp is removed.
This also changes advantage scale from earlier runs; use a new run directory
and compare against a no-penalty baseline with the same mean-centering.
Policy-loss reduction is a mean over unmasked sampled tokens per rollout, then
across all rollouts; fully masked rollouts contribute zero. Monitor raw success and successful-rollout lengths together.

Groups with equal adjusted rewards have zero advantages. The batcher gathers
`num_groups` concurrently and preserves request order. Set
`drop_degenerate_groups: true` (default false) to discard groups with equal
adjusted rewards and sample replacement tasks until all `num_groups` slots
contain non-degenerate groups. There is no retry limit; collection can continue
indefinitely if every group is degenerate. `batcher/skipped_groups` counts discarded
groups per batch, including skips during prefetch, and is zero when none are skipped.
The count appears in MLflow and the iteration log. Rollout metrics describe retained groups.
Each group collects `group_size` rollouts concurrently for the same task.
`rollout/reward_*` reports raw task rewards; `rollout/adjusted_reward_*` reports
adjusted rewards.
The batcher pads complete transcripts to `padding_multiple` (default 256),
capped at the model context length. Oversized transcripts are rejected.

GRPO uses [verl-style decoupled token-TIS](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html#decoupled-mode-with-token-level-importance-sampling-decoupled-token-is),
averaged over unmasked sampled tokens within each rollout and then across all rollouts.
Before optimizing each batch, the learner scores its transcripts once using the
optimizer's full batch shape and temperature. These anchor scores stay fixed
during the update; the original vLLM sampling scores are retained.
For each unmasked sampled token:

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
gap and applied correction. All policy diagnostics exclude skipped updates.

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
`num_groups * group_size` rollouts per batch and
`num_groups * group_size * grad_accum_steps` rollouts per optimizer update.

Set `finetuning: {mode: full}` to optimize all model weights, including
embeddings, normalization, and the output head, with FP32 master weights and Adam
moments. This uses substantially more GPU memory than LoRA. Tied embeddings and
output heads remain shared. The full variant rejects LoRA settings.
Transformer computation uses BF16; the output
projection accumulates FP32 logits in both JAX and vLLM. In the default
`finetuning: {mode: lora, rank: 16, alpha: 32}`, base weights are frozen;
only FP32 LoRA factors on every attention and MLP projection are optimized.
`finetuning.rank` defaults to 16 and `finetuning.alpha` to 32; the weight delta is
`(alpha / rank) * B @ A`. A is randomly initialized and B starts at zero,
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
preallocation, and leaves matmul precision at backend defaults. BF16 model
computation and FP32 logits remain enabled. See the
[publication parity suite](../../docs/qwen3.md#parity-with-the-grpo-v2-vllm-server).
The [current audit](../../qwen3_vllm_parity.md) still finds out-of-range ratios
with unchanged weights at 4,096 tokens, for both Tokamax and cuDNN. Corrected RoPE
frequencies and FP32 log-softmax reduce specific discrepancies but do not certify
training parity. Preserve the actor's sampling scores as the rollout-correction denominator.
Temperature is restricted to `[0.01, 2.0]`; vLLM returns
`processed_logprobs` with neutral top-p/top-k sampling, matching the learner.

Each outer iteration fetches `grad_accum_steps` full rollout batches (default 1)
in an inner loop, averages their gradients, and applies one optimizer update.
Clipping and AdamW apply once to the averaged gradient.
Parameters remain fixed within each accumulation window. There is no splitting or
repeated optimization of a batch; `update_epochs` and `minibatches_per_epoch` are unsupported.
The full batch must fit in learner memory, and `learner_gpus` must divide
`num_groups * group_size`.

Every batch counts toward the window. Batches with no unmasked sampled tokens carrying
nonzero advantages contribute zero gradients and zero policy loss, and omit policy diagnostics.
Entirely inactive windows preserve parameters, Adam state, and the optimizer step.
`train/gradient_norm` reports the averaged gradient's norm only when an optimizer update runs.
`train/skipped_update_fraction` is 1 for an entirely inactive window, otherwise 0.
`total_iterations`, checkpoint intervals, example intervals, and metric steps count
outer iterations. Checkpoints and normal training completion occur after the inner loop.
Rollout statistics pool all consumed batches; policy diagnostics average the evaluated
batches, while policy loss also includes zero contributions from inactive batches.
Accumulation buffers are local to the inner loop and are not checkpointed.
`grad_accum_steps` may change when resuming at an iteration boundary.

The learner selects a context-managed
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
The initial weights are published before any rollouts. `parameter_sync_steps: 1`
(default) publishes after every optimizer update; `3` publishes after every third
optimizer update. With `grad_accum_steps: 4`, that is every 12 batches when all
windows are active. Both settings must be positive; no divisibility constraint applies.
The former `parameter_sync_step` setting is now named `parameter_sync_steps`.
Skipped optimizer updates do not advance the publication schedule. Publication runs
when the cumulative optimizer update count reaches a multiple of `parameter_sync_steps`
and has advanced since the last publication. This schedule is preserved across restarts.
Startup and resume also publish immediately. Checkpoints and training completion do not force an extra publication
before the interval ends.
`learner/weight_transfer_seconds` is omitted when no publication runs;
`learner/update_seconds` excludes publication time.
Each LoRA publication removes its
temporary directory after the blocking weight update returns, including on failure.
The loaded weights stay in vLLM memory. This trainer runs on one host; LoRA also
requires a shared local filesystem.
`prefetch_batches: 0` (default) collects each batch without overlapping training,
avoiding weight changes within a rollout. With `parameter_sync_steps: 1`, each
batch uses the latest learner weights; longer sync intervals can retain older weights.
Positive `prefetch_batches: N` maintains a fixed pool of `(N + 1) * num_groups`
pending rollout groups. Every popped group, whether accepted or stale, immediately
queues a replacement, including the last group of a batch. The full pool remains
running or ready while the learner optimizes. Each batch uses ready groups across
the pool, without waiting for slower groups submitted alongside them. When multiple
groups are ready, ties are broken by submission order.
Groups stay intact, and degenerate groups are
replaced within their pool slots when filtering is enabled. The pool target is capped
by the number of groups still needed through the next checkpoint or `total_iterations`,
so it shrinks to zero at those boundaries. Final checkpoints therefore do
not advance the task sampler past consumed groups. Prefetched rollouts can use older
weights, and multi-turn rollouts can span weight updates.
`max_staleness` is a nonnegative integer, defaulting to `100`. At batch consumption,
only sampled tokens whose age exceeds this many successful weight publications
are masked from the loss and policy diagnostics. `0` trains only tokens sampled
under the current published weights; `null` is not accepted. All transcripts,
group rewards, and advantages are retained, including fully masked rollouts.
Stale tokens remain visible as context for later tokens. Fully stale batches are
consumed normally and skip optimizer updates rather than resampling groups.

The publisher labels weights through `/update_weight_version` after loading them
and before `/resume`, starting at `"0"`. vLLM returns exact spans of generated tokens
for each version. The agent expands them into `TrainingSequence.weight_versions`,
an `int32` array aligned with the full transcript; context tokens use `-1`.
Missing, invalid, or incomplete version metadata fails the rollout.
Versions advance on successful publications, not optimizer steps or skipped syncs,
and reset on resume with an empty rollout pool. This also handles versions that
change within a completion under `keep` or between turns under `wait`.
All `*_transcript_token_fraction` metrics use the consumed batch's **non-padding
transcript tokens** as their denominator, before the next-token shift:

| Metric (`batcher/` prefix) | Tokens counted |
| --- | --- |
| `context_transcript_token_fraction` | Initial context, user/tool turns, and inserted formatting |
| `stale_transcript_token_fraction` | Sampled tokens excluded by `max_staleness` |
| `unmasked_transcript_token_fraction` | Sampled tokens retained by the loss mask |
| `masked_transcript_token_fraction` | Context plus stale sampled tokens |

Context + stale + unmasked = 1; masked + unmasked = 1. Zero advantages do not
change these fractions. `batcher/stale_tokens` reports the stale sampled-token count.
Padding remains covered separately by `batcher/token_utilization_fraction`.

`batcher/weight_age_transcript_token_fraction/<age>` groups sampled tokens by
publications behind the version at batch consumption, before that iteration's
optimization and publication. The learner passes `consumption_weight_version` to
the batcher, which uses it for both staleness masking and age metrics.
Labels are `0` (current) through `max_staleness`,
plus `older` for all greater ages. With `max_staleness: 4`, there are at most six
buckets, regardless of how many versions are published.
`batcher/weight_age_masked_transcript_token_fraction/<age>` and
`batcher/weight_age_unmasked_transcript_token_fraction/<age>` split each bucket by
the training loss mask. All use the same non-padding transcript denominator.
Context has no weight age: age totals sum to 1 minus the context fraction. Masked
age fractions sum to the stale fraction; unmasked age fractions sum to the overall
unmasked fraction. Absent buckets are omitted; measured zeros are preserved.

These replace `batcher/stale_token_fraction` and the
`batcher/weight_version{,_masked,_unmasked}_token_fraction/<version>` metrics,
whose denominator was sampled tokens only. The old names are no longer emitted.
The absolute-version `batcher/weight_version{,_masked,_unmasked}_transcript_token_fraction/<version>`
metrics are also replaced by the bounded age buckets above.
The iteration log reports `stale_tokens`; `batcher/skipped_groups` continues to
count only degenerate-group discards.
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
Prefetching drains before periodic and final checkpoints so the saved sampling
state does not skip unconsumed batches. Restarting with the same directory restores model weights (including LoRA factors in LoRA mode),
optimizer state, shuffle key, task RNG,
round-robin scores, and progress counters. `total_iterations` is a cumulative
target; `null` runs until interrupted. Environments restart fresh, so resumption
is not bit-for-bit reproducible. Use a new directory for a fresh experiment or
incompatible model changes. Checkpoints require matching `model` and `finetuning`,
including `rank` and `alpha` for LoRA. Existing version 2 checkpoints with
a string `finetuning_mode` and top-level LoRA settings remain supported; those without
`finetuning_mode` are treated as LoRA. Legacy version 1
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
Per-taskset metrics append the configured `tasksets` key as a categorical suffix:
`rollout/reward_mean/leg_counting`, for example. Use lowercase `snake_case` keys.
Each value averages raw rewards over that taskset's retained rollouts in the
consumed batch, before the length penalty, without reweighting by sampling weights.
Discarded groups and replaced attempts are excluded. Tasksets absent from a batch
are omitted; measured zero rewards are logged. `rollout/reward_mean` remains the
mean over all retained rollouts in the batch.
Every `example_log_interval` iterations (default 10, positive integer), it saves
full token transcripts, final observations, rewards, and advantages. The batcher
keeps the last rollout in sampling order from the latest consumed group for each
configured taskset name, including across logging intervals. At each interval it
formats one example per taskset seen so far, labeled with its taskset name and
task ID. Unconsumed prefetched groups and discarded groups are excluded. This
example cache starts empty when training restarts.
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

