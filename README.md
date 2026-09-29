# rl2

Tinkering with RL and other things.

Barebones Atari PPO in [JAX + Flax](src/rl2/ppo.py), using Optax and
Gymnasium/ALE. Shared CNN, clipped policy loss, GAE, entropy bonus, gradient
clipping, and shuffled minibatches. Atari observations use 84×84 grayscale,
4-frame stacks and action repeat 4; training rewards are sign-clipped.

Edit [configs/ppo.yaml](configs/ppo.yaml), then run from the repository root:

```sh
uv run rl2                              # load configs/ppo.yaml
uv run rl2 --config configs/custom.yaml # load your own config
uv run rl2 --help
```

`vector_env: async` (default) runs the `num_envs` Atari environments in separate
spawned processes with shared observation memory. Set `vector_env: sync` to run
them sequentially in the training process. PPO batch sizes stay the same; compare
steps/second to see which mode is faster on your machine.

For a quick CPU smoke run, copy the config and set `total_steps: 32`, `num_envs: 2`,
`num_steps: 16`, `num_minibatches: 2`, and `update_epochs: 1`:

```sh
uv run rl2 --config configs/smoke.yaml
```

`total_steps` counts agent transitions across all environments (roughly four
Atari frames each), rounded down to full rollouts. Logs show total raw return and
total episode length, each averaged over the last 100 completed episodes, plus
losses, entropy and steps/second. Episode length counts agent steps. Time limits
bootstrap from the final observation; game overs stop bootstrapping.

Each run also logs to a separate directory under `log_dir` (default: `runs`).
Open the local TensorBoard UI in another terminal:

```sh
uv run tensorboard --logdir runs
```

Visit http://localhost:6006 to compare runs. Scalars show policy/value losses,
entropy, steps/second, `charts/return_mean_100`, and `charts/episode_length_mean_100`,
plotted against agent transitions. Episode metrics include completed episodes only
(including timeouts) and appear after the first episode ends.
The Text tab contains each run's YAML config.

Every `video_every_episodes: 100` completed training episodes (across all environments),
the trainer plays one separate game with the current policy and logs it under
`gameplay` in TensorBoard's Images tab as an animated GIF. Recording happens after
the rollout update, includes the whole game, and uses a separate environment and
random seed. Recorded games do not count toward training steps or episode metrics.
Set `video_every_episodes: 0` to disable videos. Recording and encoding pause training.

Python 3.12+; `uv` installs dependencies, including ALE's bundled ROMs. The default
JAX install runs on CPU; for NVIDIA GPUs, install the appropriate
[JAX accelerator package](https://docs.jax.dev/en/latest/installation.html).

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
Both pass with JAX 0.11.2, Flax 0.12.10, and metaljax 0.11.9 on macOS 26.6.2.
