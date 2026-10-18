# Qwen3-0.6B on BFCL V4

Run commands from the repository root with Python 3.12 on Linux x86-64 and an NVIDIA GPU.
This evaluates the Hugging Face checkpoint through vLLM, independently of the
JAX implementation. The runner lives in `src/rl2/bfcl.py`; the root `bfcl` extra
installs evaluation and serving dependencies into the project environment.

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
  one GPU, native 32,768-token context. Both BFCL and vLLM load that local snapshot.
- **Handler:** the existing BFCL `QwenFCHandler`, registered under
  `Qwen/Qwen3-0.6B-FC`. Prompts, tool schemas, history handling, parsing,
  execution, and grading remain upstream behavior. BFCL sends formatted text
  to `/v1/completions`; vLLM does not need a tool or reasoning parser.
- **Decoding:** thinking enabled, BFCL's default temperature **0.001**,
  upstream maximum **4,096 output tokens per response** (reduced for remaining
  context). vLLM loads the pinned model's generation defaults, including
  `top_p=0.95` and `top_k=20`; server seed is 0. No extra output repair,
  best-of selection, or success-based retries are added.
- **Scope:** all `single_turn` and `multi_turn` cases, including live,
  parallel, irrelevance, missing-information, and long-context categories.
  Memory and web-search categories are excluded. Report category scores and
  counts as **BFCL V4 single-turn/multi-turn results**, not an overall V4 score.
  Ignore the scorer's aggregate over unrun categories. Smoke uses partial
  grading solely to verify the pipeline; four cases are not an accuracy estimate.
- **Throughput:** continuous batching, prefix caching, 32 concurrent requests,
  and 90% GPU memory allocation. Numerical/batching variation means a fixed
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
uv run --extra bfcl --locked pytest tests/test_bfcl.py -q
```

Unit tests verify registration, command wiring, scope, and manifest protection
without loading a model. The smoke commands above exercise actual GPU inference,
tool execution, and the official scorer.
