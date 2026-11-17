# Drone environments

```sh
uv run ppo --config configs/ppo/crazyflow/reach_pos.yaml
uv run mappo --config configs/mappo/crazyrl/surround.yaml
```

Presets for every supported task live in those two directories. They are starting
points, not tuned results. Evaluation videos show drone/target positions.

| Adapter | Tasks | Policy actions | Observations |
| --- | --- | --- | --- |
| Crazyflow / PPO | `reach_pos`, `reach_vel`, `landing`, `figure_eight` | Four bounded attitude controls | Position, quaternion, velocity, angular velocity, goal differences |
| CrazyRL / MAPPO entry point | `hover`, `circle`, `surround`, `escort`, `catch` | Three continuous xyz commands per drone in `[-1, 1]` | Native per-agent observations |

Crazyflow uses the functional simulation pipeline with `so_rpy` dynamics and the
`cf2x_L250` drone. Controls are roll, pitch, yaw, collective thrust. `env.freq`
(default 50 Hz) must divide the 500 Hz physics frequency; `env.max_steps` counts
policy/environment steps. Figure-eight lookahead uses `n_samples`, `samples_dt`,
and `trajectory_seconds`; the trajectory must cover the episode duration.
Resets randomize position/velocity using the caller's key. Rewards and termination
rules use upstream task functions. Same-step autoreset retains pre-reset state and
observations for timeout bootstrapping.

CrazyRL infers the number of agents from `env.initial_positions`. Surround,
Escort and Catch require at least two drones. Target positions, circle/escort
intermediate points, map size, and Catch target speed are configurable.
`max_steps` may shorten the native 200-step horizon. Resets retain upstream fixed
initial positions; scalar rewards retain upstream weighting. These environments
use simplified position dynamics, separate from Crazyflow's physics.

The MAPPO entry point remains parameter-sharing **IPPO** with local critics.
Continuous policies use the existing bounded truncated Gaussian distribution,
float32 policy outputs, and joint log probabilities/differential entropy summed
over action coordinates. Differential entropy can be negative. Continuous masks
have zero length. Existing categorical policies are unchanged.

## Dependencies

Both sources are pinned in `pyproject.toml`/`uv.lock`. CrazyRL's dependency metadata
is overridden to use this project's Python/JAX versions and runtime dependencies;
its bundled training scripts are unused. CrazyRL eagerly imports its NumPy renderer,
so OpenGL runtime libraries are required even for JAX training. The training image
already includes them (`libgl1`, `libegl1`, `libopengl0`).

`rl2` enables `SCIPY_ARRAY_API=1` for Crazyflow before importing SciPy. In notebooks
or external programs that import SciPy first, set this variable before starting
Python. The rollout path uses explicit JAX state; it does not mutate Gymnasium
wrappers or call host physics callbacks.

## HLO inspection

Inspected optimized executables on NVIDIA A100 GPUs for single and 64-environment
reset/step paths of all nine tasks. CrazyRL used two drones per environment;
Crazyflow used ten physics substeps per environment step. These are new adapters,
so there is no prior adapter implementation to compare.

- Crazyflow batched steps: one ten-iteration physics loop; 4,960 bytes of temporary
  storage. Reset generation is conditional on completed episodes. Most copies are
  output-buffer copies of native state leaves and retained final state; no large
  intermediate or host callback appeared.
- CrazyRL batched steps: no loops; 640–1,184 bytes of temporary storage. Position,
  reward, collision and reset selection operations fuse. Empty continuous masks
  carry no data.
- Two-GPU reset/step executables for 64-world figure-eight and Circle batches
  contain no cross-device collectives. Step temporary storage is 5,472 and 672
  bytes respectively.
- Figure-eight constants are constructed outside the training mesh to allow
  shard-local reset execution. A regression test covers the initially failing
  automatic/manual mesh transition.

Memory figures describe complete adapter outputs, including final state; training
can eliminate unused leaves. This inspection is not a throughput benchmark or a
learning-quality evaluation.
