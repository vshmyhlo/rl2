# rl2

Tinkering with RL and other things.

## Results / Open Questions

Entries are ordered from most recent at the top to oldest at the bottom.

### Sample-Efficient Learning

Can RL agents learn effective policies with fewer environment interactions?
Explore ways to make better use of each collected experience.

### Solving Atari57 with a Single Trained Model

Try solving Atari57 with a single trained model using
[multi_atari.py](src/rl2/multi_atari.py).

### Solving Reasoning Problems with reasoning-gym

Try solving reasoning problems with
[reasoning-gym](https://github.com/open-thought/reasoning-gym).

### Learning When to Think or Act in RL

Can RL policies learn to “think” or plan before acting, similar to thinking in
LLMs, with the policy itself deciding how long to think or plan before taking
an action?

### Solving Atari with Program Induction/Synthesis

In Progress

Can program induction or synthesis learn programs that solve Atari games?

### Solving Sparse Reward Atari Environments

In Progress

Solving sparse reward Atari environments, such as Montezuma’s Revenge.

### Basic PPO for Atari

[Basic PPO](src/rl2/ppo.py) with [the Atari configuration](configs/ppo_atari.yaml)
achieves a human-normalized score of **161%** on Space Invaders.

![PPO gameplay on Space Invaders](images/spaceinvaders.gif)

In `configs/ppo.yaml`, `model` is a tagged union. The default is
`model: {type: lstm, hidden_size: 1536}`. To use a
[GDN2 residual stack](src/rl2/gdn2/), replace that block with:

```yaml
model:
  type: gdn2
  hidden_size: 768
  num_layers: 2
  num_heads: 12
  head_dim: 64
  intermediate_size: 1536
  conv_size: 4
```

To use a [Mamba3 residual stack](src/rl2/mamba3.py), select:

```yaml
model:
  type: mamba3
  hidden_size: 768
  num_layers: 2
  intermediate_size: 1536
  state_size: 128
  expand: 2
  head_dim: 64
  num_groups: 1
  mimo_rank: 1
  rope_fraction: 0.5
```

Mamba3 uses the exact `intermediate_size`; zero disables its feed-forward layers.
`mimo_rank: 1` selects SISO and larger ranks enable MIMO. `hidden_size * expand`
must be divisible by `head_dim`, and the resulting head count must be divisible
by `num_groups`. `state_size` must be even and large enough for a rotary pair;
`rope_fraction` is either 0.5 or 1.0.

Pydantic rejects settings outside the selected model type; the old top-level
`lstm_hidden_size` and `gdn2_*` fields are no longer accepted. All three choices use
the same CNN and policy/value heads and train from scratch. GDN2 and Mamba3 use
portable JAX implementations and need no tokenizer or pretrained checkpoint.
`bf16` controls compute precision; parameters and recurrent memory remain float32.
Episode resets clear every layer's memory, including GDN2 convolution histories,
during both rollout steps and sequence replay.

```bash
uv run ppo --config configs/ppo.yaml
```

PPO saves Orbax checkpoints every 10 minutes at the next completed rollout and
at normal completion, retaining the latest two under the TensorBoard run directory's
`checkpoints/` subdirectory (local paths and `gs://` paths are supported).
Configure the full run path with placeholders in YAML:

```yaml
run_id: spaceinvaders-mamba3
log_dir: gs://cohere-dev/vlad/rl2/ppo/${run_id}/${path_name:${env_id}}
```

PPO and AlphaZero load YAML through OmegaConf, resolve references to config fields,
then validate with Pydantic. `${path_name:${env_id}}` replaces `/` with `_` for
the directory name; `${env_id}` alone keeps the original environment ID.
Use `log_dir: runs/${run_id}` to group only by run ID. A plain `log_dir` is used
literally, with no run ID appended. Missing run IDs are generated before interpolation.

Restart with the same config, log directory, and run ID to load the latest completed
checkpoint automatically. Omitted IDs are generated and printed.
`checkpoint_interval_seconds` defaults to `600.0`. Model parameters,
optimizer state, training RNGs, rollout count, and completed-episode statistics are
restored. Environments and recurrent memory restart with fresh episodes. TensorBoard
continues at the restored step. `total_steps` remains the overall training target;
increasing it extends the run and recalculates decay schedules for the new target.


## Useful links and references

- [Google DeepMind mctx](https://github.com/google-deepmind/mctx): Monte Carlo tree
  search in JAX, with batched, JIT-compiled implementations supporting AlphaZero,
  MuZero, and Gumbel MuZero.
- [Programmatically Interpretable Reinforcement Learning](https://proceedings.mlr.press/v80/verma18a.html)
  (Verma et al., ICML 2018): Uses a trained neural policy to guide the search for
  interpretable policies written in a domain-specific language. Relevant to
  learning programs that play Atari; evaluated on simulated driving in TORCS.
- [OCAtari: Object-Centric Atari 2600 Reinforcement Learning Environments](https://arxiv.org/abs/2306.08649)
  (Delfosse et al., 2023): Extracts structured object observations from Atari games
  using RAM or vision. Useful for studying programmatic policies over objects
  and their attributes while separating control learning from perception.
