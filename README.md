# rl2

Tinkering with RL and other things.

Barebones Atari PPO in [JAX + Flax](src/rl2/ppo.py), using Optax and
Gymnasium/ALE. Shared residual CNN + LSTM, clipped policy loss, GAE, entropy bonus,
gradient clipping, and shuffled sequence minibatches. `observation_size: 84` resizes
observations to 84×84 with area interpolation while preserving RGB channels.
Set it to `null` to retain the original resolution. Resizing happens before frame
stacking and applies to training and video policy inputs; recorded video retains
its original resolution.

`atari_preprocessing: false` (default) keeps one emulator frame per agent step.
Set it to `true` for grayscale, action repeat 4, temporal max pooling, and random
no-ops at reset. Resize-only observations do not enable any of those behaviors.
Training rewards remain sign-clipped. `frame_stack: false` uses one frame per step,
with the LSTM carrying history; `true` stacks 4 frames.

The encoder uses four IMPALA-inspired stages with widths 128, 256, 384, and 512.
Each stage has a 3×3 convolution, stride-2 spatial max pooling, and two residual
blocks. Blocks use channel-only pre-activation LayerNorm/ReLU and scale residual
sums by 1/√2. Normalization does not use batch statistics. The final 6×6 feature
map retains spatial position before a 768-unit projection and 1536-unit LSTM.
Separate policy and value heads each have 512 hidden units and LayerNorm/ReLU;
outputs are linear. The default 84×84 RGB model has approximately 50.56M parameters;
the exact count is printed and logged at startup. Changing input resolution or
LSTM size changes the count. The encoder follows the residual design from
[IMPALA](https://proceedings.mlr.press/v80/espeholt18a.html), with added normalization.
This architecture requires new training; its parameter shapes differ from the old CNN.

Each environment has its own LSTM cell and hidden state, carried across steps and
rollouts and reset before the first observation of each new episode. Video games
have separate memory. PPO trains on complete `num_steps` sequences, shuffling
environments rather than timesteps; gradients stop at rollout and episode boundaries.
`num_envs` must be divisible by `num_minibatches`. With the default config, each
minibatch contains 8 environments × 128 steps = 1024 transitions. Memory collected
under the previous policy is retained when parameters change between rollouts.

Edit [configs/ppo.yaml](configs/ppo.yaml), then run from the repository root:

`env_id` selects the Atari game; the default is `ALE/SpaceInvaders-v5`.
For example, set `env_id: ALE/Pong-v5` to play Pong instead.

```sh
uv run rl2                              # load configs/ppo.yaml
uv run rl2 --config configs/custom.yaml # load your own config
uv run rl2 --help
```

`vector_env: async` (default) runs the `num_envs` Atari environments in separate
spawned processes with shared observation memory. Set `vector_env: sync` to run
them sequentially in the training process. PPO batch sizes stay the same; compare
steps/second to see which mode is faster on your machine.

`anneal_lr: true` linearly decreases `learning_rate` toward zero over the full
rollout budget. Rollout `i` (starting at zero) uses `learning_rate * (1 - i / N)`,
where `N = total_steps // (num_envs * num_steps)`. All minibatch updates within
that rollout use the same rate; the final rollout uses `learning_rate / N`.
Set `anneal_lr: false` for a constant rate. The actual rate used is logged as
`charts/learning_rate` in TensorBoard and `lr` in the console.

`target_kl: 0.01` enables early stopping within each rollout's optimization loop.
If a minibatch's approximate KL exceeds this threshold, its update is skipped
and the remaining minibatches/epochs are stopped. Training resumes with a fresh
rollout. Set `target_kl: null` to always run all `update_epochs`. This is a sampled
check before an update, not a hard bound on the final policy's KL. Learning-rate
decay follows rollout progress even when updates are skipped.

For a quick CPU smoke run, copy the config and set `total_steps: 32`, `num_envs: 2`,
`num_steps: 16`, `num_minibatches: 2`, and `update_epochs: 1`:

```sh
uv run rl2 --config configs/smoke.yaml
```

`total_steps` counts agent transitions across all environments (one Atari frame
each, or roughly four with preprocessing), rounded down to full rollouts.
Logs show total raw return and total episode length, each averaged over the last
100 completed episodes, plus
losses, entropy and steps/second. Episode length counts agent steps. Time limits
bootstrap from the final observation; game overs stop bootstrapping.
With preprocessing enabled, the final frame at ALE timeouts is captured directly
from the emulator to avoid Gymnasium's stale frame buffer when action repeat ends early.

Each run also logs to a separate directory under `log_dir`
(default: `gs://cohere-dev/vlad/rl2`). GCS logging uses Google Application Default
Credentials with write access to the bucket. For local development, authenticate
with `gcloud auth application-default login` before training.
Set `log_dir: runs` to log locally instead.

To view GCS logs locally, download them and open TensorBoard:

```sh
gcloud storage rsync --recursive gs://cohere-dev/vlad/rl2 runs
uv run tensorboard --logdir runs
```

Repeat the sync to refresh downloaded logs, or skip it when logging locally.
Visit http://localhost:6006 to compare runs. Scalars show policy/value losses,
entropy, steps/second, `charts/return_mean_100`, `charts/episode_length_mean_100`,
and `charts/total_episodes` (cumulative completed training episodes across all environments),
plotted against agent transitions. Episode metrics include completed episodes only
(including timeouts) and appear after the first episode ends.
The Text tab contains each run's YAML config.

PPO diagnostics are logged every rollout:

- `policy/approx_kl`: sampled old-to-current policy KL estimate, averaged over
  evaluated minibatches (including the minibatch that triggers early stopping).
- `policy/clip_fraction`: fraction of sampled action probability ratios outside
  `[1 - clip_coef, 1 + clip_coef]`, averaged over the same updates.
- `value/explained_variance`: `1 - Var(returns - values) / Var(returns)` over the
  full rollout, using rollout-time value predictions and GAE returns. A value of
  1 is a perfect variance fit, 0 is no explained variance, and negative is worse
  than a constant prediction. It is NaN when the returns have zero variance.
- `policy/early_stop`: 1 if KL stopped optimization on this rollout, otherwise 0.
- `charts/updates_per_rollout`: number of gradient updates actually applied.

Every `video_every_episodes: 100` completed training episodes (across all environments),
the trainer plays one separate game with the current policy and logs it under
`gameplay` in TensorBoard's Images tab as an animated GIF. Recording happens after
the rollout update, includes the whole game, and uses a separate environment and
random seed. Recorded games do not count toward training steps or episode metrics.
Set `video_every_episodes: 0` to disable videos. Recording and encoding pause training.
`video_speed: 2.0` plays recorded games at double speed; use `1.0` for normal speed
or `0.5` for half speed. This controls playback FPS, not environment stepping.

Evaluation runs after the rollout update whenever another `eval_every_episodes: 100`
training episodes have completed across all environments. Each evaluation plays
`eval_episodes: 100` full games with the frozen policy and pauses training until it
finishes. If one rollout crosses several thresholds, its updated policy is evaluated
once. Evaluation games do not count toward training episodes or steps, and use fresh
LSTM memory and a separate RNG. `eval_seed: 10000` fixes the evaluation seed set across
checkpoints. Set `eval_every_episodes: 0` to disable evaluation (also the default for
older configs that omit this setting).

Evaluation requires training with `atari_preprocessing: true` and
`observation_size: 84` or `null`. It uses sampled PPO actions, sticky actions 0.25,
action repeat 4, random reset no-ops, unclipped full-game returns, and a 108,000-emulator-frame
limit. TensorBoard logs `eval/return_mean`, `eval/return_median`, `eval/return_std`,
and `eval/return_sem` against training steps. The Text tab's `eval/report` includes
all episode scores, seeds, protocol settings, and actual training step/episode
counts; these reports work with local and GCS log directories. Evaluation duration
is logged under `time/evaluation_seconds`. Episode SEM describes the fixed policy's
evaluation variability; paper comparisons still require matching training budgets,
protocols, and multiple independent training seeds. See
[src/rl2/atari_eval.py](src/rl2/atari_eval.py) for the complete protocol.

Python 3.12+; `uv` installs dependencies, including ALE's bundled ROMs. The default
JAX install runs on CPU. For NVIDIA GPUs on Linux, use the `cuda12` extra, which
installs JAX with CUDA 12 and cuDNN libraries (a compatible NVIDIA driver is required):

```sh
uv run --extra cuda12 rl2 --platform cuda
```

`--platform cuda` requires CUDA and fails if it is unavailable. Omit `--platform`
to use `JAX_PLATFORMS` or JAX's automatic device selection. See the
[JAX installation requirements](https://docs.jax.dev/en/latest/installation.html)
for supported GPUs and drivers.

On Apple Silicon with macOS 14+, the optional `metal` extra uses the community
[metaljax backend](https://github.com/eterevsky/metaljax) (beta):

```sh
JAX_PLATFORMS=metal uv run --extra metal rl2
```

The trainer prints its JAX devices and saves them in TensorBoard's Text tab;
the Metal run should report `MetalDevice`. Atari emulation still runs on CPU.
Use `JAX_PLATFORMS=cpu uv run rl2` to explicitly select CPU. The Metal extra
requires JAX 0.11.x; use `--extra metal` whenever running with the Metal backend.

This is a minimal trainer with no checkpoints.

Tests: `uv run pytest`.
Metal tests: `JAX_PLATFORMS=metal uv run --extra metal pytest`.

Reproducibility: `seed` controls model initialization, action sampling, minibatch
shuffling, and Atari seeds (`seed + environment index`). Video games use separate
random keys and environments. Keep the same code, YAML, backend, hardware, and
locked dependencies (`uv run --locked ...`) when reproducing a run. Exact equality
across CPU/Metal or different library versions is not guaranteed; timestamps,
run directory names, and throughput will differ even for identical training.

The reproducibility regression compares six fresh processes with 128 training
transitions each, shortened Atari episodes to exercise resets, and up to 16 gradient
updates. It checks exact model/optimizer state, observations, sampled actions,
log-probabilities, values, LSTM states, episode reset masks, and metrics across
repeated runs, sync/async execution, and videos on/off, plus a different-seed
control. Run it on the desired backend:

```sh
JAX_PLATFORMS=cpu uv run --locked pytest tests/test_reproducibility.py -v
JAX_PLATFORMS=metal uv run --locked --extra metal pytest tests/test_reproducibility.py -v
```


## Interactive Reasoning Gym

Play generated math, logic, and programming tasks in the terminal:

```sh
uv run rl2-reasoning                       # choose a task interactively
uv run rl2-reasoning --list                # list available tasks
uv run rl2-reasoning countdown --size 5 --seed 42
uv run rl2-reasoning countdown --describe  # show default task settings
uv run rl2-reasoning countdown --config '{"min_numbers": 3, "max_numbers": 3}'
```

Enter an answer to get the task's score (0–1) and a reference answer, then
continue to the next question. `/skip` reveals the reference answer; `/quit`,
Ctrl-C, or EOF ends the session. `/multi` accepts a multiline answer (for example,
code); finish with `/submit` on its own line. `/help` lists the commands.
The final mean score includes submitted answers only, with skips counted separately.
Use the same task, config, and seed to replay the same questions.

Reasoning Gym is installed with the project dependencies. If installation fails
because `cc` is missing but GCC is installed, use `CC=gcc uv sync`.
