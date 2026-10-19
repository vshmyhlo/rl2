# Gemma reasoning with GRPO

Train Gemma 3 270M on [Reasoning Gym](https://github.com/open-thought/reasoning-gym)
with an optional calculator:

```bash
uv run python -m rl2.train_grpo_reasoning --config configs/grpo_reasoning/gemma3.yaml
```

The trainer supports only Gemma, using Google DeepMind's `gemma` library for
model computation, KV caches, tokenization, and native checkpoint I/O. The config
starts from `gs://gemma-data/checkpoints/gemma3-270m-it`. A local native
Gemma/Orbax checkpoint directory is also accepted. `tokenizer: null` selects the
official Gemma 3 tokenizer; an override names a local or `gs://` SentencePiece
model. Its vocabulary and special tokens must match the checkpoint.

For CPU use, set `dtype: float32` and launch with `JAX_PLATFORMS=cpu`. Parameters,
optimizer state, and loss remain float32; `dtype` controls model computation and
KV-cache storage. Prompt plus completion budgets must fit 32,768 tokens. Start
with small batches: the 262,144-token vocabulary makes logits memory intensive.

Set `attention_backend: cudnn` with `dtype: bfloat16` to use
[JAX cuDNN attention](https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html)
on a supported NVIDIA GPU such as H100. JAX requires compute capability 9.0 or
higher for Gemma 3 270M's 256-wide heads, and the GPU/cuDNN combination must support
that width for both forward and backward passes; newer hardware alone is not
sufficient (see the [cuDNN support matrix](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention.html#support-matrix)).
It applies to training, prefill, and cached decoding;
`attention_backend: xla` (the default) uses upstream Gemma attention. Checkpoints
are compatible across backends. cuDNN is opt-in and does not fall back on CPU.

On Linux, set `attention_backend: tokamax` to use
[Tokamax attention](https://github.com/openxla/tokamax) for training, prefill, and
cached decoding. Tokamax selects an available implementation automatically,
falling back to XLA when needed. Both `float32` and `bfloat16` are supported.

## Prompts and calculator

Prompts use the Gemma 3 user/model turn format used by `gm.text.ChatSampler`.
Each question samples an example count uniformly from zero through the length of
`prompt_examples`, then includes that many examples in listed order. The supplied
config uses zero through six examples with equal probability, alternating worked
or direct answers with calculator exchanges. An empty list gives zero-shot
prompts. All completions in a group share the sampled prompt;
sampling uses the checkpointed RNG. One BOS is added at tokenization.
Only the newly generated completion is scored. The prompt asks the model to put
its final answer inside `<answer>...</answer>`; untagged numeric answers can earn
partial training reward.
Generation ends at EOS, end-of-turn, or the token budget.

The supplied config enables `calculator_max_calls: 2`; set it to `0` to disable
calculator access. A model-generated call pauses decoding at the closing tag:

```text
<calculator>12 * 13</calculator>
<result>156</result>
<answer>156</answer>
```

The trainer inserts the `<result>` line, then resumes the same KV cache. The
model supplies the call and final answer. Tool use is optional and has no separate
reward. Both prompt variants include instructions for the tool and answer format.
This is an inline text protocol inside the model turn, not a native FunctionGemma
tool message. The model must not end its turn at a calculator call; the trainer
pauses it automatically at `</calculator>`.

Arithmetic accepts numbers, parentheses, and `+ - * / // % **`. It does not execute
Python code. Invalid expressions and exhausted call limits return an error inside
`<result>`. Expressions are limited to 256 characters and 64 AST nodes, magnitudes
to `1e100`, and exponents to an absolute value of 100. Floating results use 15
significant digits.

`max_new_tokens` bounds the full completion transcript, including inserted
results. Results must fit completely and leave at least one model-token slot;
otherwise the rollout ends as truncated. Tool results participate in attention
but incur no policy/KL loss or duration penalty. Calculator requests themselves
are model-generated tokens and do incur loss and any enabled duration cost.

The small training decoder records token log probabilities and loss masks, which
`ChatSampler.chat()` does not return directly. Sampling uses Gemma's
`RandomSampling` with the configured temperature. Calculator decoding synchronizes
with the host after sampled tokens to detect calls; it is slower than the tool-free
JAX scan. Steps containing only inserted results skip random sampling, log-softmax,
and host token transfers. No API server or Verifiers rollout is involved.

## Training

Each rollout samples `group_size` independent completions for each of `num_tasks`
new questions. Select the dataset with `task` and pass options in `task_config`.
The supplied arithmetic config uses two-digit multiplication with two operands,
so calculator access is useful. It disables duration penalties initially to avoid
rewarding the model for skipping tools. Enable a small penalty after inspecting
correctness and tool use. One task with four completions and singleton optimizer
minibatches is a memory-conservative smoke-test setup, not a tuned training recipe.

The last complete answer tag is passed to the dataset's `score_answer`. With
`allow_untagged_numeric: true`, missing tags fall back to a final numeric line,
equation result (such as `22 * 60 = 1320`), or explicit `Answer: 1320` statement.
Extraction considers only the response after the last calculator exchange;
tool calls/results alone earn no reward. Malformed tags do not trigger fallback,
and a tagged answer takes precedence over untagged numbers. Numeric strings are
passed unchanged to the dataset scorer.

Format credit requires exactly one nonempty `<answer>...</answer>` pair at the
end of the response, without nested tags. The training reward is:

```text
(1 - format_reward_weight) * task_reward + format_reward_weight * valid_format
    - duration_penalty_coef * model_generated_tokens / max_new_tokens
```

The supplied config enables numeric fallback and uses
`format_reward_weight: 0.1`: correct tagged answers earn 1.0, correct untagged
answers 0.9, and incorrect answers with valid formatting 0.1. Defaults for other
configs retain strict tagged scoring (`allow_untagged_numeric: false`,
`format_reward_weight: 0.0`).

EOS and end-of-turn count as generated tokens; prompt tokens, tool results, and
padding do not. Group advantages use population standard deviation plus `1e-4`;
equal-reward groups have zero policy advantage. The clipped GRPO loss averages
model-token losses within each completion, then averages completions. Replay uses
the sampling temperature and exact transcript, with gradients through context.

`beta` controls sampled KL against the initial checkpoint; `0` skips reference
replay. `num_minibatches` divides `num_tasks * group_size` and bounds both update
and reference-replay batches. `update_epochs` reuses fixed behavior probabilities.
Adam uses constant learning rate and global gradient clipping. Training runs on
one JAX device; there is no distributed training or held-out evaluation.

## Logging and checkpoints

MLflow logs task reward, duration penalty, total reward, success rate, truncation,
completion length, losses, and timing. `charts/success_rate` measures correctness
regardless of format; `charts/format_rate` measures format compliance, and
`charts/formatted_success_rate` requires both. Set `MLFLOW_TRACKING_URI` for a tracking
server; otherwise storage is local. The experiment defaults to `grpo_reasoning`.
Steps count questions seen, not completions. At `log_interval` and completion,
`samples/completions/<step>.md` shows the shared prompt and the highest-scoring
2–4 completions from the first question's group, including inserted tool results.
`log_completion_count` controls how many are shown.

Configuration, devices, parameter count, and session stdout/stderr are also
logged. Timing includes dispatch, compilation, host calculator work, and waiting;
it is wall time, not GPU kernel time. Startup reports initialization stages and
prints early rollout progress while compilation may occur.

`ckpt_dir/checkpoints/` stores Orbax training state: parameters, Adam state,
RNG, rollout count, and prompts seen. Checkpoints are saved at the first completed
rollout after `checkpoint_interval_seconds` and at completion; the latest two are
retained. The preset uses `ckpt_dir: /runs/grpo_reasoning/${run_id}`. Reuse the same
directory to resume, or change `run_id` for a fresh run with this template. The supplied
mixed-shot config has a separate run ID to avoid resuming earlier runs.
`run_id: null` creates a timestamped run.

Training settings may change on resume; compatible parameter/optimizer trees
and valid progress metadata are required. New learning rate and gradient clipping
settings apply to restored Adam state. Legacy `checkpoint.msgpack` training states
remain readable when compatible. The source checkpoint still supplies the frozen
KL reference. A `gs://` checkpoint directory supports training checkpoints.

Local runs export final weights to `ckpt_dir/model-N`, readable with
`gm.ckpts.load_params` or reusable as `checkpoint`. Keep the tokenizer setting
when reusing an export with a custom tokenizer.
