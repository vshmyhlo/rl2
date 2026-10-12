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

...

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
- [JaxMARL](https://github.com/bold-lab-ai/JaxMARL): Multi-agent RL environments
  and baseline algorithms in JAX, including SMAX and Hanabi, with
  GPU-parallel simulation and PPO-based baselines such as IPPO and MAPPO.
- [Pgx](https://github.com/sotetsuk/pgx): JAX-native game environments, including
  chess, shogi, Go, and backgammon, with GPU-accelerated parallel simulation
  and AlphaZero and PPO training examples.
- [XLand-MiniGrid](https://github.com/dunnolab/xland-minigrid): JAX-based grid-world
  environments and benchmarks for meta-reinforcement learning, inspired by XLand
  and MiniGrid, with composable rules and goals and multi-GPU PPO baselines.
- [DiscoRL](https://github.com/google-deepmind/disco_rl): A minimal JAX framework
  for meta-learning RL update rules, with pretrained Disco103 weights and notebooks
  for training agents with the discovered rule or learning new rules.
- [minimax](https://github.com/facebookresearch/minimax): JAX-based unsupervised
  environment design baselines for automatic curricula, including PAIRED,
  Prioritized Level Replay, and ACCEL, with multi-device training support.
- [JaxUED](https://github.com/DramaCow/jaxued): Unsupervised environment design
  in JAX, with readable single-file implementations of domain randomization,
  Prioritized Level Replay, ACCEL, and PAIRED, plus reusable level-sampling utilities.
- [Brax](https://github.com/google/brax): JAX-based physics simulation and RL
  training, including PPO and SAC, with accelerator-parallel execution. Active
  development focuses on training, with support for MuJoCo Playground environments.
- [Gymnax](https://github.com/RobertTLange/gymnax): JAX RL environments spanning
  classic control, MinAtar, and meta-RL tasks, with vectorized rollouts.
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground):
  GPU-accelerated environments for robot learning and sim-to-real transfer.
- [Jumanji](https://github.com/instadeepai/jumanji): JAX RL environments for
  logic puzzles, packing, and routing problems.
- [Craftax](https://github.com/MichaelTMatthews/Craftax): JAX RL environment
  extending Crafter with roguelike mechanics inspired by NetHack.
- [Navix](https://github.com/epignatelli/navix): Accelerated MiniGrid environments
  implemented in JAX.
- [Kinetix](https://github.com/FLAIROx/Kinetix): JAX environments for RL on
  general 2D physics tasks.
- [JAXtari](https://github.com/k4ntz/JAXtari): GPU-accelerated Atari environments
  in JAX, with pixel and object observations and customizable game mechanics.
