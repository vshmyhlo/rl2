# rl2

Tinkering with RL and other things.

Barebones Atari PPO in [JAX + Flax](src/rl2/ppo.py), using Optax and
Gymnasium/ALE. Shared CNN, clipped policy loss, GAE, entropy bonus, gradient
clipping, and shuffled minibatches. Atari observations use 84×84 grayscale,
4-frame stacks and action repeat 4; training rewards are sign-clipped.

```sh
uv run rl2                              # train on ALE/Pong-v5
uv run rl2 --env-id ALE/Breakout-v5
uv run rl2 --help
```

Quick CPU smoke run:

```sh
uv run rl2 --total-steps 32 --num-envs 2 --num-steps 16 \
  --num-minibatches 2 --update-epochs 1
```

`--total-steps` counts agent transitions across all environments (roughly four
Atari frames each), rounded down to full rollouts. Logs show the mean raw return
over the last 100 completed games, losses, entropy and steps/second. Time limits
bootstrap from the final observation; game overs stop bootstrapping.

Python 3.12+; `uv` installs dependencies, including ALE's bundled ROMs. The default
JAX install runs on CPU; for NVIDIA GPUs, install the appropriate
[JAX accelerator package](https://docs.jax.dev/en/latest/installation.html).
This is a minimal trainer: no checkpoints, evaluation loop or experiment tracker.

Tests: `uv run python -m unittest discover -s tests`.
