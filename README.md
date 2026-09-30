# rl2

Tinkering with RL and other things.

Barebones Atari PPO in [JAX + Flax](src/rl2/ppo.py), using Optax and
Gymnasium/ALE. Shared CNN + LSTM, clipped policy loss, GAE, entropy bonus, gradient
clipping, and shuffled sequence minibatches. `atari_preprocessing: false` (default)
uses native RGB images (210×160 for Pong) and one emulator frame per agent step.
Set it to `true` for 84×84 grayscale, action repeat 4, max pooling, and random
no-ops at reset. Raw images require more memory and computation.
Training rewards are sign-clipped in both modes. `frame_stack: false` (default)
uses one frame per step, with the LSTM carrying history; `true` stacks 4 frames.
Both options apply to training and video games, independently of each other.

Each convolution is followed by channel-only LayerNorm and ReLU. Normalization
is independent at each spatial location and does not use batch statistics.
The CNN feeds a shared 512-unit dense layer, a shared LSTM (`lstm_hidden_size: 256`),
then separate policy and value MLP heads with 256-unit hidden layers.
Each dense hidden layer uses LayerNorm followed by ReLU. The final outputs are
linear: action logits for the policy and one scalar for the value.

Each environment has its own LSTM cell and hidden state, carried across steps and
rollouts and reset before the first observation of each new episode. Video games
have separate memory. PPO trains on complete `num_steps` sequences, shuffling
environments rather than timesteps; gradients stop at rollout and episode boundaries.
`num_envs` must be divisible by `num_minibatches`. With the default config, each
minibatch contains 2 environments × 128 steps = 256 transitions. Memory collected
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

Each run also logs to a separate directory under `log_dir` (default: `runs`).
Open the local TensorBoard UI in another terminal:

```sh
uv run tensorboard --logdir runs
```

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

Tests: `uv run python -m unittest discover -s tests`.
Metal tests: `JAX_PLATFORMS=metal uv run --extra metal python -m unittest discover -s tests`.

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
JAX_PLATFORMS=cpu uv run --locked python -m unittest discover -s tests -p test_reproducibility.py -v
JAX_PLATFORMS=metal uv run --locked --extra metal python -m unittest discover -s tests -p test_reproducibility.py -v
```
