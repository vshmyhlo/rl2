# Environments for agentic GRPO

Start with Prime Intellect's Environments Hub and Verifiers. The project already includes `verifiers`, and its environments support tool execution, multi-turn rollouts, and rewards.

| Collection | What you get | Why consider it |
|---|---|---|
| [Prime Environments Hub](https://app.primeintellect.ai/dashboard/environments) + [Verifiers](https://docs.primeintellect.ai/verifiers/v1/overview) | Installable environments, tool execution, stateful multi-turn rollouts, reward functions, and environment mixtures. | First choice for plugging existing tasks into the actor. |
| [OpenEnv](https://github.com/huggingface/OpenEnv) | Containerized environments with `reset`/`step` interfaces; coding, text games, browser tasks, and other domains. | Useful for separating environment servers from model inference. |
| [AgentGym](https://github.com/WooooDyy/AgentGym) + [AgentGym-RL](https://github.com/WooooDyy/AgentGym-RL) | WebShop, ALFWorld, ScienceWorld, TextCraft, SQL, and tool-use environments. The RL framework explicitly supports GRPO. | Broad collection for multi-step decision-making experiments. |
| [AppWorld](https://github.com/StonyBrookNLP/appworld) | Stateful simulated apps with executable APIs, task evaluation, and explicit train/dev/test splits. | Realistic workflows involving several tools and persistent state. |
| [τ²-bench](https://github.com/sierra-research/tau2-bench) | Customer-service conversations, tools, policies, and simulated users across retail, airline, and telecom domains. | Multi-turn interaction with users and tools; budget for user-simulator inference. |
| [BrowserGym](https://github.com/ServiceNow/BrowserGym) | A common interface to MiniWoB++, WebArena, WorkArena, and other browser benchmarks. | Browser agents; requires more environment setup than simple Python tools. |

## Starting point

### ToolRL RLLA-4K

Run `uv run python -m rl2.train_grpo_v2 --config configs/grpo_v2/toolrl.yaml` with the same two-GPU setup described below. The preset selects `taskset: {id: toolrl}` and downloads the official 3,920-row training Parquet, pinned to revision `8cee13ec0ca72f0461da372a93a6fd8140dbb840`. Set `taskset.dataset_file` to a local Parquet path for offline use. The upstream 80-row test file is excluded from training; this trainer does not yet run held-out evaluation.

`toolrl_env.py` supplies a Verifiers taskset, task rewards, and a custom Qwen3 renderer. Each rollout generates one response (`max_turns: 1`, up to 1,024 tokens), conditioned on the dataset's system instructions and conversation history. Tool descriptions remain in the prompt; generated calls are graded as text without execution. The renderer preserves `<think>`, `<tool_call>`, and `<response>` tags, including malformed output, while retaining exact sampled tokens and log-probabilities for GRPO. Reference answers are grading data only.

Rewards follow ToolRL's default [reward design](https://github.com/qiancheng0/ToolRL/blob/8cee13ec0ca72f0461da372a93a6fd8140dbb840/verl/utils/reward_score/rlla.py): format earns 0–1, and tool names, parameter keys, and values earn partial correctness credit scaled to −3–3. Invalid generated calls earn −3 correctness; empty prompts, malformed reference responses, and invalid reference calls fail dataset loading. Response-only examples earn format reward only, without grading response semantics. Length bonuses and reward schedules are not enabled.

Upstream reward limitations are preserved: correctness grades the first `<tool_call>` block anywhere in the completion, including inside reasoning, and ignores later blocks. Format reward is independent and does not strictly validate tag nesting. A format failure therefore does not necessarily imply a low total reward.

The [ToolRL dataset](https://github.com/qiancheng0/ToolRL#-dataset) combines ToolACE, Hammer-Masked, and xLAM. This integration retains its original output format; it does not convert calls to Qwen3's native `arguments` schema or implement an interactive tool environment.

### CodeGym

The preset starts a [CodeGym environment manager](https://github.com/StigLidu/CodeGym/blob/e85a439761ae2df813d5997a81bbd9f9e2ed27b7/online_server/README.md) inside the Ray actor's process tree, alongside vLLM. Prepare the server once, then train with two GPUs:

```bash
uv run python -m rl2.codegym_setup --root /runs/codegym
uv run python -m rl2.train_grpo_v2 --config configs/grpo_v2/codegym.yaml
```

Use a writable directory and set `taskset.server.root` to the same path. Setup downloads the three upstream server modules, requirements, and license at revision `e85a439761ae2df813d5997a81bbd9f9e2ed27b7`; no Git clone is needed. It changes the worker bind address to IPv4 loopback, installs server dependencies plus Gymnasium/NumPy in `<root>/.venv`, and exports the matching `envs_en` dataset as `envs/codegym_v1/{source}__{env_name}.py`. A completion marker skips repeated setup at the same revisions. Setup needs network access; actor startup uses the prepared files. Keep the prepared directory at its original path because its virtual environment contains absolute paths.

For k3s, prepare it on the shared `/runs` volume before training in the same job (later jobs reuse it):

```bash
./submit.py --gpus 2 -- sh -c 'uv run python -m rl2.codegym_setup --root /runs/codegym && uv run python -m rl2.train_grpo_v2 --config configs/grpo_v2/codegym.yaml'
```

The manager binds `127.0.0.1:8001`; vLLM uses port 8000. `server.workers: 40` covers the preset's 32 concurrent rollouts. Startup waits for every worker, with `server.startup_timeout: 120` seconds by default. Shutdown and startup failures terminate the manager's process group, including workers. Upstream worker logs are under `<root>/logs/`. Worker ports are allocated from 10000 and are not published outside the training pod.

For an external server, omit `taskset.server` (or set it to `null`) and set `taskset.manager_url` to its HTTP(S) endpoint. The client must reach both the manager and its allocated worker ports on the same hostname.

The preset loads all 80,130 rows of the `task_en_instruction_en_env` subset’s `train` split from [VanishD/CodeGym](https://huggingface.co/datasets/VanishD/CodeGym), the community reproduction linked by upstream. Both tasks and environment code use revision `85286359a342f7a288aea74273772b69b9b784c2`. The task file pattern selects all four English training shards. Set `taskset.dataset_file` to a JSONL/JSON or Parquet path or glob to override it, and provide matching environment modules. Rows need `prompt` messages and `ability: codegym_v1@environment_name@JSON_config`; optional `plugin_call_max_round` caps each task's turns.

`codegym_env.py` preserves the dataset's `<|FunctionCallBegin|>[{"name": ..., "parameters": {...}}]<|FunctionCallEnd|>` format. Each rollout leases an independent worker, starts the task, feeds back observations, and ends when `/is_done` reports completion or a turn limit is reached. Invalid calls receive feedback and consume a turn. Server failures abort the group; they are not scored as incorrect answers. Sessions are stopped before workers are released, including on errors or cancellation.

Environment code is exported unchanged. The pinned dataset contains a syntax error in `Codeforces_5537_I__MinPathCostEnv.py`; tasks referencing it will fail during environment loading.

Reward matches the upstream client: 1 for a positive environment reward, otherwise 0. The task's `answer` and `reward_model` fields are unused. Verifiers' `agent.interaction()` captures the complete conversation; only sampled assistant tokens contribute to the loss. The preset allows 32 model turns with 512 tokens per turn, subject to the existing agent timeout and model context limit. CodeGym's code and dataset use [CC BY-NC 4.0](https://github.com/StigLidu/CodeGym/blob/e85a439761ae2df813d5997a81bbd9f9e2ed27b7/LICENSE).

### WebShop, ALFWorld, τ²-bench, and AppWorld

Run `uv run python -m rl2.train_grpo_v2 --config configs/grpo_v2/<name>.yaml`, where `<name>` is `webshop`, `alfworld`, `tau2`, or `appworld`. Edit the preset's `taskset.python` to an absolute Python executable in a separate environment with that benchmark installed. The trainer launches the included `benchmark_workers/*_worker.py` scripts directly; those environments do not need rl2, JAX, Ray, or vLLM installed.

| Preset | Complete training subset | Setup and configuration |
|---|---|---|
| `webshop` | All human goals at shuffled indices `[1500, len(goals))`; excludes upstream's 500 test and 1,000 development goals | Follow [WebShop setup](https://github.com/princeton-nlp/WebShop#-setup) with `./setup.sh -d all`. Set `taskset.root` to the repository. The worker explicitly loads `items_shuffle.json`, `items_ins_v2.json`, human instructions, and the full search index; no small-product fallback. |
| `alfworld` | Every solvable training game accepted by the official loader, across all six task types | Install `alfworld`, run `alfworld-download`, set `ALFWORLD_DATA`, and point `taskset.config_file` to upstream's [base_config.yaml](https://github.com/alfworld/alfworld/blob/master/configs/base_config.yaml). The worker forces `num_train_games=-1` and all task types, and disables expert observations. |
| `tau2` | Official `train` IDs across retail, airline, and telecom: 74 + 30 + 74 at the inspected upstream revision | Install [tau2-bench](https://github.com/sierra-research/tau2-bench), configure its data directory and simulator credentials in runtime environment variables, and set `taskset.user_llm`. `taskset.domains` can select domains. Missing training splits fail loading; `base`, `test`, and telecom `full` are never substituted. |
| `appworld` | All IDs from `load_task_ids("train")` | Install `appworld`, run `appworld install` and `appworld download data`, and set `taskset.root` to the downloaded data root. See [AppWorld setup](https://github.com/StonyBrookNLP/appworld#floppy_disk-installation). |

Each rollout starts its own process and resets the selected training task, giving GRPO independent environment state. Dependencies and data must already be installed on the actor host. AppWorld executes Python in the upstream environment with per-rollout temporary outputs; separate processes follow its [parallel-world guidance](https://github.com/StonyBrookNLP/appworld/blob/main/guides/parallelizing_worlds.md). Process separation is not a security sandbox; run code-executing benchmarks on an isolated benchmark host/container. Worker shutdown closes the environment; stalled workers and their child processes are killed after five seconds. Infrastructure failures abort the rollout group.

The action formats are native WebShop `search[...]`/`click[...]`, ALFWorld commands, τ² plain user messages or JSON tool calls, and AppWorld fenced Python blocks. Observations, available actions, domain policy, and API discovery instructions enter the prompt; reference solutions and evaluator state do not. Rewards are WebShop's fractional purchase reward, ALFWorld's `won`, τ²'s upstream evaluator reward, and AppWorld's `evaluate().success`. τ² uses its user simulator, including its inference cost, and finalizes with `done` at the trainer's turn limit. All sampled assistant tokens and their log-probabilities remain in the GRPO trace; environment output is masked from the loss.

These presets use four rollouts, 50 turns, and 1,024 tokens per response. `rollout_timeout` controls Verifiers' conversation timeout; `environment_timeout` bounds each worker request and initial task discovery. The model context limit still applies. Full WebShop catalog loading is repeated per process and can require substantial host RAM and startup time; size concurrency and timeouts accordingly. The trainer samples from every loaded training task without a fixed task-count cap; finite runs do not guarantee visiting every task.

Adapter tests use lightweight upstream fakes plus a real worker/Verifiers transcript integration. Full benchmark installations, external user-model calls, and GPU training are separate deployment checks.

### Addition and shared training pipeline

The prototype uses `verifiers==0.4.0` and its `verifiers.v1` API. `AdditionTaskset.load()` supplies five local addition tasks; it and the MCP `add` tool live in `addition_env.py`. The actor uses the documented `vf.make_agent(vf.AgentConfig(...))` factory and `agent.run(task)` interface. The actor owns vLLM and runs Verifiers' `null` chat harness in local subprocesses for up to three model turns. An exact final integer answer earns 1, otherwise 0.

The actor samples four independent rollouts of the same task concurrently (`group_size=4` per request, minimum 2), then scores the completed group. Each rollout receives `(reward - group_mean) / (group_std + 1e-4)`, using population standard deviation. Equal-reward groups have zero advantages. The learner requests a batch with configured `num_groups` and `group_size` (defaults: 1 and 4). The async batcher awaits `num_groups` calls to `Actor.fetch_next_group(group_size=...)` together and prepares the training arrays on CPU in request order, using the model context limit supplied by the learner. The learner moves the prepared arrays onto its device before optimization. The actor uses async methods so groups overlap while awaiting model and tool responses, following [Ray’s async actor pattern](https://docs.ray.io/en/latest/ray-core/actors/async_api.html). vLLM stays running between requests. Each group contains: shared task ID/prompt plus each rollout's training sequence, reward, advantage, final answer, turn/token counts, duration, and stop condition. Full Verifiers traces are discarded after scoring; generation or scoring failures propagate to the learner.

Run with `uv run python -m rl2.train_grpo_v2 --config configs/grpo_v2/addition.yaml` on a Linux x86_64 host with Python 3.12 and two visible GPUs. The main dependencies include the rollout stack. Ray reserves one GPU for `Actor` (inherited by its vLLM subprocess) and the other for `Learner`; the batcher runs on CPU. The first rollout provisions the harness's Python dependencies using uv. No external dataset or paid inference service is needed.

The actor and batcher set `JAX_PLATFORMS=cpu` through their Ray runtime environments before importing JAX. CPU array placement alone does not prevent CUDA initialization, and deserializing JAX arrays restores them onto the receiving worker's default device. This keeps those workers from reserving GPU memory needed by the learner and vLLM; vLLM retains its assigned CUDA device.

The YAML config uses the same interpolation and validation as PPO/MAPPO. It controls the model, task sampling seed, batch dimensions, sampling limits, temperature, run ID, and optional `total_iterations` (`null` runs until interrupted). `ckpt_dir` stores profiling artifacts and Orbax checkpoints under `checkpoints/`. Checkpoints save every `checkpoint_interval` completed iterations (default 100) and at normal completion, retaining the latest two. Restarting with the same `ckpt_dir` automatically restores the latest checkpoint: model parameters, optimizer state and step, shuffle key, task sampling RNG, cumulative iteration/rollout counts, and observed stop-condition labels. Restored weights are published before collecting new rollouts. `total_iterations` is the cumulative target, so raising it extends a completed run; `null` runs until interrupted. Interrupted or failed iterations resume from the last completed checkpoint. Live environments and vLLM generation state restart fresh, so continuation is not bit-for-bit reproducible. The current config controls resumed training; saved config is informational. Use a new directory for incompatible model changes or a fresh run. Local and `gs://` checkpoint paths are supported.

### Ray dashboard

The dashboard starts automatically at <http://localhost:8265> for local runs. The project's `ray[default]` dependency includes the dashboard dependencies in submitted images.

For `submit.py` jobs, open [Training jobs](http://localhost:8085/) and click **Ray dashboard** beside a running job. The list refreshes every 10 seconds and also shows queued jobs and timeline links. Install or update the [shared viewer](../../infra/k3s/README.md#shared-on-disk-checkpoints) once; forward only **8085** through your IDE or SSH when browsing remotely. Each dashboard opens at `/dashboard/<job-name>/`, so multiple jobs can be viewed without separate port-forwards.

Submission sets `RL2_RAY_DASHBOARD_HOST=0.0.0.0` inside the Pod; local runs bind to loopback by default. The dashboard is available only while Ray is running.

Ray 2.59 enables token authentication by default. Expand **Dashboard login token** in the jobs list, or use the command printed by `submit.py`, and enter its token in Ray's login dialog:

```bash
sudo k3s kubectl -n training exec job/<job-name> -c train -- ray get-auth-token
```

Logins are scoped to each job. For local runs, use `uv run ray get-auth-token`.

### Ray timeline

Timeline collection is disabled by default. Set this in the training YAML to enable profiling and snapshot exports, including jobs launched with `submit.py`:

```yaml
enable_ray_timeline: true
```

`submit.py` supplies the output path and viewer URL but does not enable collection; the training config controls it.

The driver logs the trace path and saves a snapshot every 60 seconds and before Ray shuts down, including when training raises an exception. Local snapshots are replaced atomically; abrupt termination can lose events since the last successful export. Ray's event-retention limits still apply, so snapshots are not an unlimited history.

`submit.py` sets `RL2_RAY_TIMELINE_PATH=/runs/ray-timelines/<job-name>.json` on the persistent volume and prints `http://localhost:8085/?job=<job-name>`. The driver logs the same URL from `RL2_RAY_TIMELINE_URL`. After the [shared viewer is installed](../../infra/k3s/README.md#shared-on-disk-checkpoints), forward port **8085** in your IDE or SSH and click that link. It waits for the first snapshot, then opens Perfetto; **Load latest** refreshes the trace. The link also works after the job finishes.

Without the submission environment variables, local runs save `<ckpt_dir>/profiles/ray-<unique-id>.json`; set `ckpt_dir` to a writable directory. You can always download the JSON and open it at [ui.perfetto.dev](https://ui.perfetto.dev/). While a Pod is running:

```bash
sudo k3s kubectl -n training cp <pod>:<logged-trace-path> ./ray-timeline.json -c train
```

After the Pod exits, retrieve the file from the PVC's [host backing directory](../../infra/k3s/README.md#shared-on-disk-checkpoints); paths below `/runs/` are relative to that directory. Live `uv run ray timeline` export also works inside the running container.

Worker tracks include actor calls, `batcher.prepare_batch`, and the learner's `fetch_batch`, `optimize`, `publish_weights`, and `checkpoint` phases with iteration metadata. These measure host-side intervals, not GPU kernels or subprocess internals. Custom spans use Ray's private profiling API. The trainer explicitly uses a positive event-reporting interval: setting `RAY_task_events_report_interval_ms=0` disables reporting in Ray 2.59 despite its outdated timeline docstring.

OpenTelemetry tracing is enabled when `OTEL_EXPORTER_OTLP_ENDPOINT` or
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` is set. `submit.py` supplies the cluster's
Tempo gRPC endpoint and adds `k8s.job.name` to every span. New GRPO jobs therefore
export automatically; set `OTEL_SDK_DISABLED=true` to disable tracing.
For local runs, forward Tempo's port 4317 and launch with:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317 \
  uv run python -m rl2.train_grpo_v2 --config configs/grpo_v2/addition.yaml
```

In [Grafana Explore](http://localhost:8082/explore), select **Tempo** and search
`{ resource.service.name = "rl2" && name = "train.iteration" }`.
Each iteration has a separate trace tagged with `run_id`, `iteration`, and `model`.
Spans cover Ray calls, batch preparation, rollout groups and individual rollouts,
scoring, optimization, weight publication, logging, and checkpoints.
Groups carry `group_id` and `task_id`; rollouts add `rollout_index` and
`stop_condition`. These are host-side intervals, not GPU kernel timings or
individual model/tool calls inside a rollout.

Ray's startup hook configures an OTLP gRPC batch exporter in the driver and each
worker. Ray propagates context across actor calls; child asyncio tasks inherit
the group context and open their own spans inside the task. Exceptions and
cancellation end spans and restore the previous context. Graceful worker exit
flushes queued spans; abrupt process termination can lose buffered spans.
See [Tempo access instructions](../../infra/k3s/README.md#opentelemetry-traces-with-tempo).

MLflow uses the shared `MLflowLogger`, experiment `grpo_v2`, and standard `MLFLOW_TRACKING_URI` / experiment environment variables. The [module docstring](train_grpo_v2.py) documents every metric key, its meaning, aggregation, and logging cadence. Model size and learner setup time are logged at the restored iteration (zero for a new run); iteration metrics are logged after each completed iteration, and `time/checkpoint_seconds` only on saves. Setup timing starts before learner model loading and includes checkpoint restoration, NCCL initialization, and initial weight publication. Iteration timing and throughput include collection, optimization, and weight transfer, excluding metric/example logging and checkpoint saves. Cumulative iteration, update, and rollout counters continue across restarts. Config parameters, resolved YAML at `config/<iteration>.yaml`, learner console output, and rollout examples every 10 iterations are also saved.

Stop-condition fractions are logged as `charts/stop_condition_fraction/<label>` over all rollouts in each batch, with `unspecified` for `None`. Custom labels are included automatically; labels seen earlier in the training loop log zero when absent from a later batch.

Timing metrics are scoped under `time/actor/`, `time/batcher/`, and `time/learner/`. The batcher logs wall time awaiting all rollout groups (including actor return transfers) as `time/batcher/fetch_rollout_seconds` and CPU array preparation, without explicit synchronization, as `time/batcher/prepare_batch_seconds`. Learner throughput uses full iteration wall time; actor rollout duration remains a mean over rollouts.

Ray return transfers are timed from immediately before the producer returns to immediately after the receiver’s `await`/`ray.get` completes. Per-group Actor → Batcher logs report seconds. MLflow records mean group latency as `time/batcher/actor_to_batcher_seconds` and batch transfer latency as `time/learner/batcher_to_learner_seconds`. Each group is measured as it arrives, before waiting for other groups. These intervals include Ray serialization, object-store/transport work, deserialization, and receiver scheduling delay; they exclude rollout generation and batch assembly. They are end-to-end return-path latency, not isolated serialization CPU time. Timestamps use the shared monotonic clock of this single-node setup; they are not valid across hosts.

Each `Rollout.sequence` contains exactly one complete multi-turn token sequence: exact token IDs, aligned sampling log-probabilities, and a loss mask excluding prompts and tool results. Missing tokens/log-probabilities, inconsistent lengths, and multiple branches fail the fetch. `TrainClientConfig` renders Qwen3 prompts and parses tool calls locally, using vLLM 0.27.0's native `/inference/v1/generate` endpoint to obtain sampled tokens and log-probabilities. Advantages retain their group normalization when rollouts are flattened for training.

The learner loads BF16 Qwen3 weights through `qwen3.py`, keeps FP32 master parameters and Adam state, and uses BF16 model computation. It publishes BF16 weights initially and after each optimized batch. Publishing uses vLLM's [NCCL weight-transfer backend](https://docs.vllm.ai/en/v0.27.0/training/weight_transfer/) between the two GPUs on the same host. DLPack exposes JAX arrays as PyTorch views; NCCL packs tensors with a target buffer size of at least 64 MiB, enlarged to fit the largest tensor. The learner completes each rollout batch before optimization and weight publication; collection and updates do not overlap. The actor uses vLLM pause/resume without a local rollout lock, clears cached state, and applies the weights. `weight_update_pause_mode` selects how vLLM handles in-flight requests: `wait` (default) finishes them, `abort` cancels them, and `keep` freezes them until resume. These modes operate on individual requests, not complete multi-turn rollout groups. Failed updates prevent further rollouts. The loopback server enables the weight-transfer endpoints with `VLLM_SERVER_DEV_MODE=1` and `--weight-transfer-config '{"backend":"nccl"}'`.

BF16 mixed-precision training is always enabled; no precision flag is needed. Model projections and activations use BF16, while master weights, gradient accumulation, Adam moments, log-probabilities, and the GRPO loss use FP32. Normalization and other sensitive model operations also use FP32 internally. Keeping master weights in FP32 preserves updates smaller than BF16 can represent. vLLM rollouts also use BF16.

GRPO replays each complete transcript with causal next-token alignment. The loss is the clipped importance-ratio objective, averaged over sampled tokens within each rollout and then over rollouts; prompts, tool results, and padding have no direct loss. No reference-policy KL penalty is applied. Variable-length transcripts are right-padded to multiples of `padding_multiple` (default: `256`; set to `1` to disable rounding), capped at the model context length; overlong transcripts are rejected without truncation. Each new JIT trace of the training update logs its minibatch shape, gradient accumulation steps, temperature, and clipping coefficient; cached executions stay silent.

`gradient_checkpointing` defaults to `true`: the learner recomputes each decoder layer during backpropagation to reduce activation memory at the cost of extra computation. Set `gradient_checkpointing: false` in YAML to disable it. KV-cache decoding is unchanged.

`update_epochs` (default 1) sets passes over each rollout batch. `minibatches_per_epoch` (default 4) splits each pass into equal optimizer minibatches. Before each epoch, the learner applies one fresh permutation to every batch field, then slices contiguous minibatches. Shuffling is reproducible from the saved base key (initialized from `seed`) and the optimizer step, including across rollout batches and restarts. `grad_accum_steps` (default 1) splits each minibatch into equal microbatches; their gradients are averaged before global-norm clipping and one Adam step. The product `minibatches_per_epoch * grad_accum_steps` must divide `num_groups * group_size`. Microbatch size is therefore `num_groups * group_size / minibatches_per_epoch / grad_accum_steps`, and optimizer steps per rollout batch are `update_epochs * minibatches_per_epoch`. Rollout log-probabilities and advantages stay fixed across all updates; metrics average over all optimizer minibatches and epochs. All presets use one epoch, four optimizer minibatches, and one accumulation step (microbatch sizes one for Addition and eight for ToolRL/CodeGym). Learning rate is `1e-6`, clip coefficient `0.2`, and gradient norm limit `1.0`. vLLM uses neutral generation defaults and `processed_logprobs`, matching the learner’s temperature-scaled full-softmax probabilities. Temperature is restricted to `[0.01, 2.0]` so vLLM cannot silently clamp it.

The development host inherits `NCCL_NET=gIB`, whose plugin fails without `libibverbs`. For this single-host setup, bypass it with `NCCL_NET=Socket NCCL_NET_PLUGIN=none uv run python -m rl2.train_grpo_v2 --config configs/grpo_v2/addition.yaml` ([NCCL network settings](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-net)).

For GRPO, select environments that can reset the same task into independent copies and produce multiple trajectories with meaningful reward differences. Check rollout cost, termination rules, and available training splits.

Verifiers owns the conversation/tool loop inside `Actor.collect_group()`, while retaining the Ray actors, batcher, and learner. The [Verifiers v1 overview](https://docs.primeintellect.ai/verifiers/v1/overview#trace) describes traces and token/log-probability capture with prime-rl; the [prime-rl training guide](https://docs.primeintellect.ai/prime-rl/training) covers training runs.

Checked 2026-10-19: the lockfile and installed package use `0.4.0`, the [latest stable PyPI release](https://pypi.org/project/verifiers/); `0.4.1.dev41` is a prerelease. The [standalone-agent](https://docs.primeintellect.ai/verifiers/v1/agent) and [taskset](https://docs.primeintellect.ai/verifiers/v1/tasksets) interfaces above match the current docs. The local subprocess runtime is for prototyping; the [architecture guide](https://docs.primeintellect.ai/verifiers/v1/architecture) recommends isolated runtimes for production training.
