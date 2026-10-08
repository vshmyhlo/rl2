# Reasoning Gym GRPO

Fine-tune a converted GDN2 language-model checkpoint on procedurally generated
[Reasoning Gym](https://github.com/open-thought/reasoning-gym) questions:

```bash
uv run --extra gdn2 python -m rl2.train_grpo_reasoning --config configs/grpo_reasoning.yaml
```

The default config uses `checkpoints/gdn2/paper-matched-30b` and the cached
TinyLlama SentencePiece tokenizer. See the [GDN2 instructions](../src/rl2/gdn2/README.md)
for checkpoint conversion and tokenizer setup. Set `checkpoint` and `tokenizer`
to use other compatible local files. Vocabulary sizes must match.

Choose a dataset with `task` and pass its options through `task_config`, for example:

```yaml
task: leg_counting
task_config:
  min_animals: 1
  max_animals: 3
```

Each rollout takes `num_tasks` new questions and samples `group_size` independent
completions per question. Prompts request a final answer inside
`<answer>...</answer>`. The last complete answer tag is sent to the dataset's
`score_answer` method; missing tags receive zero reward. Multiline answers are
supported. Rewards are standardized within each question's group using population
standard deviation plus `1e-4`; equal-reward groups have zero policy advantage.
A base language model may initially get very sparse rewards, so inspect the logged
completions and informative-group fraction when selecting tasks and token budgets.

Sampling uses the full vocabulary with positive `temperature`. Replay uses that
same temperature and teacher-forces complete prompt/response sequences, including
gradients through prompt processing. The clipped GRPO loss averages tokens within
each completion, then averages completions. Prompt tokens and padding incur no
loss. The first EOS does incur loss. Responses reaching `max_new_tokens` without
EOS are scored as generated and reported as truncated. Oversized prompts raise an
error; raise `max_prompt_tokens` to accommodate them.

`beta` controls the sampled KL penalty against the frozen initial checkpoint.
Set it to zero to disable reference replay. `num_minibatches` controls both update
and reference-replay batch sizes and must divide `num_tasks * group_size`.
`update_epochs` reuses the rollout with fixed behavior log probabilities. Adam
uses a constant learning rate and global gradient clipping. This trainer runs on
one JAX device and does not implement distributed training or held-out evaluation.

For NVIDIA GPUs, install the optional kernel dependencies and set `backend: triton`:

```bash
uv sync --extra cuda12 --extra gdn2 --extra gdn2-triton
uv run --no-sync python -m rl2.train_grpo_reasoning --config configs/grpo_reasoning.yaml
```

The portable `jax` backend also supports CPU; select it with `JAX_PLATFORMS=cpu`
and use `dtype: float32` for CPU experiments. The default config uses bfloat16
projections with float32 parameters, optimizer state, recurrence, and loss.

TensorBoard scalars and sample completions are written to `log_dir/run_id`.
A null `run_id` creates a timestamped directory. Reuse that directory name to
resume `checkpoint.msgpack`, including optimizer state, sampling/shuffling RNG,
and the next dataset index. Checkpoints are atomically replaced at completed
rollout boundaries, at the configured time interval and at completion. You may
increase `total_updates` or change logging settings when resuming; settings that
affect training must match. Keep the source checkpoint and tokenizer unchanged.

Local runs export final weights to `log_dir/run_id/model-N`, where `N` is the
completed rollout count. These use the standard GDN2 checkpoint format:

```bash
uv run --extra gdn2 python -m rl2.gdn2.generate \
  --checkpoint runs/MY_RUN/model-1000 \
  --prompt 'Solve the following problem. Put only your final answer inside <answer>...</answer>.

Question: How many legs do two dogs have?

Answer:'
```

A `gs://` log directory supports resumable training checkpoints and TensorBoard
logging, but automatic inference exports are only written for local runs.
