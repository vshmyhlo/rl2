# Qwen3-0.6B on BFCL V4

Run commands from the repository root with Python 3.12 on Linux x86-64 and an NVIDIA GPU.
This evaluates the Hugging Face checkpoint through vLLM, independently of the
JAX implementation. The runner lives in `src/rl2/bfcl/runner.py`; the root `bfcl` extra
installs evaluation and serving dependencies into the project environment.

Code is grouped under `src/rl2/bfcl/`: `runner.py` defines the protocol and CLI,
`qwen.py` registers the model with BFCL, and `job.py` orchestrates Ray tasks and
the server actor. Tests live in `tests/bfcl/`.

## Setup and run

```bash
uv sync --extra bfcl --locked
uv run --extra bfcl --locked rl2-bfcl download

# Terminal 1: keep the server running; Ctrl-C releases the GPU.
uv run --extra bfcl --locked rl2-bfcl serve
```

Wait for vLLM to report that application startup is complete, then in terminal 2:

```bash
# Four fixed cases covering calls, tool selection, abstention, and multi-turn use.
uv run --extra bfcl --locked rl2-bfcl generate --smoke
uv run --extra bfcl --locked rl2-bfcl evaluate --smoke

# Complete single-turn and multi-turn categories, with strict coverage checking.
uv run --extra bfcl --locked rl2-bfcl generate
uv run --extra bfcl --locked rl2-bfcl evaluate
```

Results, input/state logs, official scores, and `protocol.json` go into
`runs/bfcl/qwen3-0.6b/`; smoke artifacts use `qwen3-0.6b-smoke/`.
Use `--run-dir PATH` on both generation and evaluation for a new run.
Generation resumes missing cases without overwriting existing responses.
Changed configurations require a fresh directory. Use the same `--port` on
the server and generation commands when changing the default `1053`.
Supply any Hub credentials through `HF_TOKEN` at runtime.

## Kubernetes Job

With the [local k3s queue and build service](../infra/k3s/README.md#build-and-queue-training-jobs) installed,
submit one GPU Pod containing the evaluation launcher:

```bash
./submit.py --extra bfcl --memory-gib 32 -- \
  python -m rl2.bfcl.job --run-dir /runs/bfcl/qwen3-0.6b
```

For a four-case check, use a separate directory and add `--smoke`:

```bash
./submit.py --extra bfcl --memory-gib 32 -- \
  python -m rl2.bfcl.job --run-dir /runs/bfcl/qwen3-0.6b-smoke --smoke
```

Add `--dry-run` before `--` to preview without building or submitting.
The image includes the root `bfcl` extra. One container downloads the pinned model,
starts vLLM, waits up to 900 seconds for readiness, generates responses, stops
vLLM, and scores saved results. Failures stop the pipeline; cancellation cleans
up subprocess groups. Kubernetes holds the GPU allocation until the Pod ends.
`--startup-timeout SECONDS` changes the readiness timeout.

`src/rl2/bfcl/job.py` starts a local Ray runtime inside the container: `VLLMServer` is a
GPU actor, while `download`, `generate`, and `score` are CPU tasks. The driver
waits for generation before stopping the actor and starting scoring. Task retries
and actor restarts are disabled. No external Ray cluster is required.
The image sets `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` so Ray workers reuse its installed
dependencies. It also sets `RAY_DEDUP_LOGS=0` to preserve repeated messages from
different workers. Set both variables before launching locally through `uv run`.

Results and `download.log`, `serve.log`, `generate.log`, and `evaluate.log` persist
in the requested `/runs` directory. Each stage streams stdout and stderr to Pod
logs with a `[download]`, `[serve]`, `[generate]`, or `[evaluate]` prefix, including
output from child workers. Follow with `kubectl logs -n <namespace> -f job/<job-name>`.
The model cache defaults to
`<run-directory-parent>/.huggingface` unless `HF_HOME` is supplied.

Resubmitting the same image and directory resumes missing cases after preemption;
changed protocol settings or source hashes require a new directory. Concurrent
launchers cannot share a run directory.

## Protocol

- **Evaluator:** `bfcl-eval==2025.12.17`, the version specified by the
  [official leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html).
  Serving uses **vLLM 0.27.0** and **Transformers 5.5.4** for compatibility
  with the project's Hugging Face Hub and Triton dependencies. The root
  `uv.lock` pins all versions. BFCL's NumPy 1.26 constraint is overridden
  to use the project's NumPy 2. This preserves benchmark logic but is not
  an exact reproduction of the leaderboard's serving environment.
- **Model:** `Qwen/Qwen3-0.6B` at
  `c1899de289a04d12100db370d81485cdf75e47ca`, BF16, no quantization,
  one GPU, the checkpoint's configured 40,960-token context. Both BFCL and vLLM
  load that local snapshot and use the same context budget.
- **Handler:** the existing BFCL `QwenFCHandler`, registered under
  `Qwen/Qwen3-0.6B-FC`. Prompts, tool schemas, history handling, parsing,
  execution, and grading remain upstream behavior. BFCL sends formatted text
  to `/v1/completions`; vLLM does not need a tool or reasoning parser.
- **Decoding:** thinking enabled, BFCL's requested temperature **0.001**,
  clamped to **0.01** by vLLM. Both values are recorded in the manifest.
  Upstream maximum **4,096 output tokens per response** (reduced for remaining
  context). vLLM loads the pinned model's generation defaults, including
  `top_p=0.95` and `top_k=20`; server seed is 0. No extra output repair,
  best-of selection, or success-based retries are added.
- **Scope:** all 4,441 `single_turn` and `multi_turn` cases, including live,
  parallel, irrelevance, missing-information, and long-context categories.
  Memory and web-search categories are excluded. Report category scores and
  counts as **BFCL V4 single-turn/multi-turn results**, not an overall V4 score.
  Ignore the scorer's aggregate over unrun categories. Smoke uses partial
  grading solely to verify the pipeline; four cases are not an accuracy estimate.
- **Throughput:** continuous batching, prefix caching, 32 concurrent requests,
  and 90% GPU memory allocation. `VLLM_USE_FLASHINFER_SAMPLER=0` selects the
  native GPU sampler, avoiding FlashInfer's C++/NVCC build requirement.
  Attention and CUDA graphs remain enabled. Numerical/batching variation means a fixed
  server seed does not guarantee bitwise-identical repeated generations.

This is the pinned **BFCL local-handler protocol**, not Qwen's recommended
thinking recipe: Qwen recommends temperature 0.6 and a larger output budget.
Treat experiments with different decoding or thinking disabled as separate runs
and explicitly report those changes. Inspect failed trajectories for truncation,
repetition, malformed calls, and infrastructure errors before interpreting scores.
Keep benchmark cases out of training and prompt tuning.

References: [BFCL runner](https://github.com/ShishirPatil/gorilla/tree/f7cf735/berkeley-function-call-leaderboard),
[Qwen handler](https://github.com/ShishirPatil/gorilla/blob/f7cf735/berkeley-function-call-leaderboard/bfcl_eval/model_handler/local_inference/qwen_fc.py),
[Qwen recommendations](https://huggingface.co/Qwen/Qwen3-0.6B#best-practices).

## Tests

```bash
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run --extra bfcl --locked pytest tests/bfcl/ tests/test_submit.py -q
```

Unit tests verify registration, command wiring, scope, manifest protection, and
Job process lifecycle without loading a model. The smoke commands above exercise actual GPU inference,
tool execution, and the official scorer.
