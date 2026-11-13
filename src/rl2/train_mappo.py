"""Recurrent parameter-sharing IPPO for SMAX, JaxNav, MPE, STORM, JaxRobotarium, and Lux.

The actor and critic share a local observation encoder/LSTM with separate heads.
Actions are masked during collection and updates. Agent death/inactivity resets
memory without ending the reward stream or GAE trace. Environment completion ends
traces; timeouts bootstrap from final local observations unless also terminated.
Despite the historical MAPPO filename, the critic uses no centralized state.
Lux observations separate self/opponent teams and retain native dtypes through
rollout storage, with map arrays in H, W order and positions in x, y order.
The default generic model uses an MLP; model.type=lux uses masked entity-set
self-attention, shared ship heads, and spatial sap-target scoring.
Lux map and entity attention default to model.heads=8 and model.head_dim=32,
with feature width model.heads * model.head_dim and CNN width model.conv_channels.
Lux cuDNN attention requires bf16=true; select xla for float32 or CPU execution.
Both models retain recurrent memory and one value per agent/team.
Lux memory consists of a global-token LSTM cell/hidden state per attention block,
retained across matches and reset at full-game boundaries.

Optional map_size_curriculum (Lux and JaxNav) overrides env.map_size:
    map_size_curriculum:
      - {start_iteration: 0, map_size: 8}
      - {start_iteration: 500, map_size: 16}
      - {start_iteration: 2500, map_size: 24}
Thresholds count completed iterations: sizes above run for 500 iterations, then
2000, then the remainder. Stages start at zero and have strictly increasing
thresholds. An empty/omitted curriculum keeps env.map_size fixed. Resume selects
the stage using restored iterations and the current configuration. Size changes
restart environments, recurrent memory, and opponent assignments, discarding
unfinished episodes without counting them as completed. Parameters, optimizer,
opponent history, and lifetime counters persist. JAX recompiles for new shapes.
Evaluation and video use the active training size; log windows may span stages.

Shape dimensions:
    E: Parallel environments (config.num_envs).
    N: Controlled agents per environment (env.num_agents).
    B: Agent sequences; E * N during rollout, E for training with an opponent pool.
    T: Rollout steps per environment (config.num_steps).
    M: Agent sequences per minibatch (training B / config.num_minibatches).
    S: Leading batch axes: (B,) per step, (T, B) during scans, (B, T) for training.
    O: Local observation features.
    A: Action categories per discrete component.
    C: Action component shape; () for scalar actions or (D,) for D components.
    D: Discrete action components per agent (e.g., ships controlled by a team in Lux).
    H: Recurrent hidden width (Lux: model.heads * model.head_dim; generic: model.recurrent.hidden_size).
    L: Raw policy logits per action component.
    K: Raw PRNG key words.
    Spatial arrays use H, W for height, width; Lux maps have shape (*S, H, W).
    In video arrays, F is frame capacity and (H, W, 3) is image height, width, RGB.
    Flattened loss arrays use N = prod(S) for transition count.
    *S and *C unpack shape tuples; ... denotes remaining leaf dimensions; () is scalar.

Multi-discrete policies produce one joint action per agent: log probabilities and
entropies sum over D.
Grouped choices normalize action types and their conditional commands separately;
their joint entropy includes conditional entropy weighted by type probability.
Recurrent minibatches preserve entire (B, T, ...) agent sequences. Advantages are
standardized per minibatch; a single transition retains its raw advantage.
KL-rejected minibatches skip backpropagation and optimizer updates.

Optional selfplay config enables a historical opponent pool for environments with
supports_selfplay=True and exactly two controlled agents. Defaults: pool_size=8,
snapshot_interval=100 completed iterations, current_policy_fraction=0.5. Slot zero
retains the initial policy; scheduled snapshots replace the oldest unreferenced
slot, or are skipped when all replaceable slots are active. Opponents are sampled
uniformly from populated slots. Each environment samples a learner side and an
opponent at full-game reset, retaining assignments and separate memory across
iterations and matches. Only learner transitions enter PPO, including current-policy
games: training B=E, while rollout memory retains E*N agents. num_envs must then
be divisible by num_minibatches * device_count. Historical inference maps over
environments to avoid materializing a parameter copy per environment.

Training uses all devices on JAX's default backend (all visible GPUs when available).
Parameters and optimizer state are replicated; environments are sharded along E
and recurrent minibatches along B, keeping each agent sequence intact. num_envs
is global and must be divisible by device_count; num_envs * num_agents must be
divisible by num_minibatches * device_count. Losses, advantage normalization,
gradient clipping, KL stopping, and metrics reduce over the global batch.

An iteration collects num_envs * num_steps environment transitions and runs PPO
updates. total_iterations sets the cumulative iteration budget, including restored
history. MLflow steps count transitions summed across environments.
Team returns sum mean agent rewards, counting shared rewards once. Episodes count
environment completions. Lux uses
match-win increments; averaging both teams gives match_count / 2 per full game,
so self-play return is not a measure of playing strength.
With a pool, training episode returns sum only the selected learner's raw rewards.
Loss and explained-variance metrics use only learner transitions. Evaluation and
standalone video behavior are unchanged.

Progress metrics are logged every log_interval iterations (default: 60) and on
normal completion.
Distribution summaries default to the mean without a suffix; _std and _sem mark
standard deviation and standard error. _total marks lifetime scope, including
restored history; lifetime return and episode length remain means, not sums.

Window metrics:
    train/policy_loss: Mean clipped surrogate policy loss.
    train/value_loss: Half mean squared value error, before value_coef.
    train/total_loss: Policy loss + value_coef * value loss - entropy_coef * entropy.
    train/entropy: Masked joint action entropy in nats, summed over components.
    train/approx_kl: Mean exp(log_ratio) - 1 - log_ratio in nats versus rollout policy.
    train/ppo_clip_fraction: Fraction of action probability ratios outside PPO's clip range.
        These average over transitions, then evaluated minibatches, including
        KL-rejected minibatches. Total loss uses each iteration's entropy weight.
    train/explained_variance: Mean rollout value explained variance over
        iterations with nonconstant GAE returns; omitted when none are defined.
    train/updates_per_iteration: Accepted minibatch updates per iteration.
    train/kl_early_stop_fraction: Fraction of iterations stopped by target_kl;
        omitted when KL stopping is disabled.
    rollout/episode_count: Number of completed episodes in the window.
    rollout/episode_return: Mean raw return of completed episodes.
    rollout/episode_length: Mean length of completed episodes in steps.
        Episode means are omitted when no episodes completed.
    rollout/episode_end_fraction/terminated: Fraction of completed window episodes
        with task termination, including episodes also flagged as truncated.
    rollout/episode_end_fraction/truncated: Fraction of completed window episodes
        truncated without task termination. Both fractions use environment episode
        counts, reset each window, and are omitted when no episodes completed.
    train/iterations_per_second, train/env_steps_per_second,
    train/episodes_per_second: Window counts divided
        by wall seconds. Includes collection, updates, and intervening logging,
        evaluation, video, and checkpoints; excludes the current log and later save.

Lifetime metrics (include restored history):
    train/iterations_total: Completed iterations.
    train/episodes_total: Completed training episodes.
    train/env_steps_total: Environment transitions.
    train/episode_return_total: Raw return averaged over completed episodes.
    train/episode_length_total: Episode length averaged over completed episodes.
        Lifetime episode means are omitted before any episode completes.
    train/progress_fraction: Completed iterations / total_iterations, capped at one.

Latest settings and session timings:
    train/map_size: Latest iteration's map side length in cells, at each progress
        log's environment step; omitted for environments without map_size.
    train/pool_param_iteration_min, train/pool_param_iteration_max: Minimum and
        maximum snapshot iteration across populated opponent-pool slots, including
        the preserved initial snapshot, after any insertion this iteration. Logged
        at each progress log's environment step; omitted when selfplay is disabled.
    train/learning_rate, train/entropy_coef: Latest iteration's scheduled settings,
        evaluated at its starting completed-iteration count against total_iterations.
    actor/video_seconds: Gameplay recording, encoding, and upload wall seconds.
    checkpoint/save_seconds: Blocking wall seconds to prepare and submit an async
        checkpoint, including any wait for the previous save; excludes background
        writes and the final shutdown wait.
        Operation timings are logged only when that operation runs, at its current
        environment step, and exclude logging the timing itself.
    model/params_millions: Parameter count in millions at startup, including resume.

eval_ref_policies maps lowercase snake_case names to full Orbax checkpoint directory
paths, such as /runs/mappo/<run_id>/checkpoints/<iteration>. Local and gs:// storage
are supported; relative local paths resolve from the working directory. Each path
points to the checkpoint root; the loader reads its state/ child.
Only policy parameters are loaded, once at startup when evaluation is enabled,
using the current model configuration and device placement. Saved configurations,
optimizers, and opponent pools are ignored. References require a self-play environment
with exactly two controlled agents. Evaluation only plays against references;
without references, evaluation is omitted for every environment.
Each reference gets eval_episodes fixed seeds. The current policy alternates team
sides by episode index; returns sum only its team's rewards (Lux: match-win rewards).
Evaluation metrics use <name> from eval_ref_policies:
    eval/episode_return/<name>, eval/episode_return_std/<name>, eval/episode_return_sem/<name>: Raw return
        mean, sample standard deviation, and standard error over eval_episodes seeds.
        Uncertainty metrics are omitted for one episode.
    eval/evaluation_seconds/<name>: Wall seconds from evaluation through gameplay
        and report upload, excluding metric logging and subsequent joint rating.
These metrics are logged only when that reference is evaluated, at the current
environment-step count, on the evaluation schedule below.
    eval/evaluation_seconds: Wall seconds for the full evaluation, including
        environment setup, all references, gameplay/report uploads, joint ratings,
        and their metric logging; excludes logging this timing. Logged once per
        evaluation at the current environment step; omitted without references.
Joint ratings are logged once after all references finish, at the same step:
    eval/trueskill, eval/trueskill_std, eval/trueskill_conservative: Current
        policy's posterior mu, sigma, and mu - 3 * sigma, in TrueSkill rating points.
    eval/reference_trueskill/<name>, eval/reference_trueskill_std/<name>,
    eval/reference_trueskill_conservative/<name>: Each reference's posterior mu,
        sigma, and mu - 3 * sigma from the same rating pool.
    These are posterior parameters, not averages over games; omitted without references.
Each full episode is one result: higher total team reward wins; equal rewards draw
(Lux: compare total match wins). A single learner rating is shared across references;
each result updates both participants. All ratings start fresh at each checkpoint
evaluation with package defaults: mu=25, sigma=25/3, beta=25/6, tau=25/300,
draw_probability=0.1. Games run grouped by reference, then results sorted by reference
name and episode index are shuffled once with Python Random(eval_seed) for rating.
The replay is excluded. This makes ratings independent of reference execution order,
but sequential TrueSkill still depends on the chosen shuffle. Ratings do not carry
across checkpoint steps or process restarts; rerunning the same games does not accumulate
evidence. No reference-versus-reference games are played.

Enabled video, evaluation, and checkpoint schedules use completed iteration counts,
including on resume, and also run on normal completion. video_interval and
eval_interval default to 0 (disabled), and checkpoint_interval to 600. A zero video
interval disables standalone gameplay. Evaluation replays its highest-return episode
(per reference policy) for one video, independently of video_interval. The first
episode wins ties. Scores include every full episode; the replay does not enter metrics.
Lux evaluation replays label the trained policy and named reference inside their team boxes.
video_max_frames (default: 900; None: unlimited) caps recordings. GIFs loop at video_fps
with frame boundaries rounded to 10 ms and a minimum 10 ms frame delay.
Each evaluation/video episode compiles reset, inference, and environment steps
together. Video buffers states up to the frame cap or configured episode horizon,
then transfers them once for host rendering; uncapped video stores the full episode.

Artifacts: config/<step>.yaml, devices/<step>.txt, logs/<session>.log,
model/parameter_count/<step>.txt, gameplay/<step>.gif, eval/report/<name>/<step>.json,
eval/gameplay/<name>/<step>.gif, and eval/ratings/<step>.json. Per-reference reports
include the reference name and checkpoint, both teams' individual returns and episode
lengths, aggregate scores, the zero-based video_episode_index, configuration, seed,
and training counters. The ratings report contains all final joint ratings, reference
checkpoint paths, evaluation seed, and environment step. Configuration parameters use
dotted keys; existing MLflow parameter values are preserved on resume.

Checkpoints save the full configuration, parameters, optimizer state, RNG, iteration
and transition counters, and cumulative episode statistics. Restore uses the current
configuration without compatibility checks. Legacy checkpoints lacking transition
counters estimate them using current rollout dimensions. Live environments and
memory restart on resume; unfinished episode statistics are discarded. Checkpoint
writes overlap training; shutdown waits for pending writes and surfaces failures
before finalizing MLflow.
Pool parameters and snapshot iterations are checkpointed separately. Resume uses
the current pool capacity, preserving the initial snapshot and newest history;
enabling a pool on an older checkpoint starts it from the restored learner. Live
opponent assignments and memory restart with environments. Disabling the feature
ignores saved pool state. Saved lifetime return totals retain their original meaning.
"""

import argparse
import json
import os
from collections.abc import Callable
from contextlib import ExitStack, closing
from dataclasses import asdict, fields, replace
from dataclasses import field as dataclass_field
from functools import partial
from math import sqrt
from operator import itemgetter
from pathlib import Path
from random import Random
from statistics import fmean, stdev
from time import monotonic
from typing import Annotated, Any, Literal, NamedTuple, Self, cast

# Orbax's epath otherwise loads TensorFlow for gs:// paths when it is installed.
# Use our gcsfs dependency instead of TensorFlow's native runtime.
# Set this before importing Orbax, since epath caches its backend selection.
os.environ.setdefault("EPATH_USE_TF", "0")

# isort: split

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import rlax
import trueskill
import yaml
from flax import struct
from flax.training.train_state import TrainState
from jax.sharding import AxisType, Mesh, PartitionSpec
from numpy.typing import NDArray
from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass

from rl2.actions import (
    CategoricalPolicy,
    DiscreteSpace,
    HierarchicalMultiCategoricalPolicy,
    HierarchicalMultiDiscreteSpace,
    MultiAgentActionSpace,
    Policy,
)
from rl2.configuration import BaseConfig, load_config
from rl2.envs.jaxnav_env import JaxNavConfig, JaxNavEnv
from rl2.envs.lux_env import LuxConfig, LuxEnv, LuxObservation
from rl2.envs.mpe_env import MPEConfig, MPEEnv
from rl2.envs.multi_agent import AgentObservation, BaseMultiAgentEnv
from rl2.envs.robotarium_env import RobotariumConfig, RobotariumEnv
from rl2.envs.smax_env import SMAXConfig, SMAXEnv
from rl2.envs.storm_env import STORMConfig, STORMEnv
from rl2.jax_cache import configure_compilation_cache
from rl2.lstm import LSTMStackCarry
from rl2.mlflow_logging import MLflowLogger
from rl2.model.actor_critic import ActorCriticOutput
from rl2.model.config import GenericModelConfig, LSTMConfig, LuxModelConfig, MLPEncoderConfig, ModelConfig
from rl2.model.generic import GenericActorCritic
from rl2.model.lux import LuxActorCritic, LuxCarry
from rl2.running_stats import RunningSum
from rl2.selfplay.pool import (
    MatchAssignment,
    OpponentPool,
    SelfPlayConfig,
    init_pool,
    insert_snapshot,
    resize_pool,
    sample_opponents,
)
from rl2.utils import explained_variance, gae

__all__ = [
    "ActorCriticOutput",
    "CategoricalPolicy",
    "DiscreteSpace",
    "GenericModelConfig",
    "HierarchicalMultiCategoricalPolicy",
    "HierarchicalMultiDiscreteSpace",
    "LSTMConfig",
    "LuxModelConfig",
    "MLPEncoderConfig",
    "ModelConfig",
    "make_actor_critic",
]


type RecurrentCarry = LSTMStackCarry | LuxCarry
type Observation = jax.Array | LuxObservation[jax.Array]
type PPOEnv = BaseMultiAgentEnv[Any, Observation]


@struct.dataclass
class PPOBatch:
    """Batch-major recurrent PPO inputs; S = (B, T).

    Attributes:
        observation: (*S, ...) local observation leaves.
        actions_mask: (*S, *C, A) boolean legal-action flags.
        action: (*S, *C) integer action IDs.
        old_log_prob: (*S,) rollout action log probabilities.
        value: (*S,) rollout-time value predictions for explained variance.
        advantage: (*S,) unnormalized advantage estimates.
        return_target: (*S,) value targets.
        model_carry: (B, H) initial LSTM cell/hidden leaves per generic layer or Lux global attention block.
        episode_start: (*S,) boolean agent or environment reset flags.
    """

    observation: Observation  # leaves: (*S, ...)
    actions_mask: jax.Array  # (*S, *C, A) bool
    action: jax.Array  # (*S, *C) integer action IDs
    old_log_prob: jax.Array  # (*S,)
    value: jax.Array  # (*S,)
    advantage: jax.Array  # (*S,)
    return_target: jax.Array  # (*S,)
    model_carry: RecurrentCarry  # leaves: (B, ...), model-specific state
    episode_start: jax.Array  # (*S,)

    def select_sequences(self, index: jax.Array) -> Self:  # (M,) agent-sequence indices
        """Return (M, T, ...) sequence fields and (M, ...) initial memory leaves."""
        return jax.tree.map(itemgetter(index), self)


class RolloutCarry(NamedTuple):
    """Live environment state and recurrent memory between rollouts.

    Attributes:
        env_state: (E, ...) batched environment state leaves.
        observation: (E, N, ...) local observation leaves and (E, N, *C, A) legal-action masks.
        model_carry: (B, H) LSTM cell/hidden leaves per generic layer or Lux global attention block.
        episode_start: (B,) boolean agent or environment reset flags.
        episode_return: (E,) accumulated raw team rewards for unfinished episodes.
        episode_step_count: (E,) step counts for unfinished episodes.
    """

    env_state: chex.ArrayTree  # leaves: (E, ...)
    observation: AgentObservation[Observation]  # observation: (E, N, ...); actions_mask: (E, N, *C, A)
    model_carry: RecurrentCarry  # leaves: (B, ...)
    episode_start: jax.Array  # (B,) reset on agent death or environment completion
    episode_return: jax.Array  # (E,) unfinished team returns
    episode_step_count: jax.Array  # (E,) unfinished episode lengths
    selfplay: MatchAssignment | None = None  # leaves: (E,), assignments persist until environment completion


class RolloutResult(NamedTuple):
    """Collected training data, advanced RNG key, and completed episode statistics.

    Attributes:
        batch: (B, T, ...) batch-major PPO arrays with (B, ...) initial memory leaves.
        key: () advanced typed PRNG key, also used for minibatch shuffling.
        episode_return_sum: () sum of raw returns for completed episodes.
        episode_length_sum: () sum of lengths for completed episodes.
        episode_count: () number of completed episodes.
        terminated_episode_count: () number of completed episodes with task termination.
    """

    batch: PPOBatch  # observation: (B, T, ...); targets/starts: (B, T); model_carry leaves: (B, ...)
    key: jax.Array  # () typed PRNG key
    episode_return_sum: jax.Array  # ()
    episode_length_sum: jax.Array  # ()
    episode_count: jax.Array  # ()
    terminated_episode_count: jax.Array  # ()


@struct.dataclass
class MinibatchMetrics:
    """Scalar diagnostics for one evaluated minibatch, including KL-rejected updates."""

    policy_loss: jax.Array  # ()
    value_loss: jax.Array  # ()
    total_loss: jax.Array  # ()
    entropy: jax.Array  # ()
    approx_kl: jax.Array  # ()
    clip_fraction: jax.Array  # ()


class EpisodeMetrics(NamedTuple):
    """Completed-episode statistics, reduced from (E,) leaves to () aggregates."""

    episode_count: jax.Array  # () or (E,) before reduction
    episode_return_sum: jax.Array  # () or (E,) before reduction
    episode_length_sum: jax.Array  # () or (E,) before reduction


class WindowMetrics(NamedTuple):
    episodes: EpisodeMetrics  # scalar leaves
    terminated_episode_count: jax.Array  # ()
    minibatch_metrics: MinibatchMetrics  # scalar leaves, including KL-rejected minibatches
    explained_variance_sum: jax.Array  # () summed over rollouts
    explained_variance_count: jax.Array  # () rollouts with nonconstant returns
    minibatch_count: jax.Array  # () evaluated minibatches
    rollout_count: jax.Array  # () completed rollouts
    update_count: jax.Array  # () accepted minibatches
    early_stop_count: jax.Array  # () rollouts stopped by the KL threshold


class TrainingMetrics(NamedTuple):
    window: WindowMetrics  # scalar leaves
    total: EpisodeMetrics  # scalar leaves


class TrainStepResult(NamedTuple):
    """Advanced training state and accumulated metrics."""

    train_state: TrainState
    rollout_carry: RolloutCarry  # observation: (E, N, ...); state: (E, ...); model_carry: (B, ...)
    key: jax.Array  # () typed PRNG key
    metrics: TrainingMetrics  # scalar leaves


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class MapSizeStage:
    start_iteration: Annotated[int, Field(ge=0, strict=True)]
    map_size: Annotated[int, Field(gt=0, strict=True)]


@dataclass(frozen=True)
class Config(BaseConfig):
    env: Annotated[
        SMAXConfig | JaxNavConfig | MPEConfig | STORMConfig | RobotariumConfig | LuxConfig, Field(discriminator="id")
    ]
    seed: int
    total_iterations: Annotated[int, Field(gt=0)]
    num_envs: Annotated[int, Field(gt=0)]
    num_steps: Annotated[int, Field(gt=0)]
    num_minibatches: Annotated[int, Field(gt=0)]
    update_epochs: Annotated[int, Field(gt=0)]
    learning_rate: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    anneal_lr: bool
    gamma: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    gae_lambda: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    clip_coef: Annotated[float, Field(ge=0, lt=1, allow_inf_nan=False)]
    target_kl: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None
    entropy_coef: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    value_coef: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    max_grad_norm: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    video_fps: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    model: ModelConfig = GenericModelConfig()
    bf16: bool = False
    video_max_frames: Annotated[int, Field(gt=0)] | None = 900
    video_interval: Annotated[int, Field(ge=0)] = 0
    eval_interval: Annotated[int, Field(ge=0)] = 0
    eval_episodes: Annotated[int, Field(gt=0)] = 100
    eval_seed: Annotated[int, Field(ge=0, lt=2**32)] = 10_000
    eval_ref_policies: dict[
        Annotated[str, Field(pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")],
        Annotated[str, Field(pattern=r"\S")],
    ] = dataclass_field(default_factory=dict, hash=False)
    lr_decay: Literal["linear", "cosine"] = "linear"
    entropy_decay: Literal["constant", "cosine"] = "constant"
    checkpoint_interval: Annotated[int, Field(gt=0)] = 600
    log_interval: Annotated[int, Field(gt=0)] = 60
    selfplay: SelfPlayConfig | None = None
    map_size_curriculum: tuple[MapSizeStage, ...] = ()

    @model_validator(mode="after")
    def validate_rollout(self) -> Self:
        if isinstance(self.env, LuxConfig) != isinstance(self.model, LuxModelConfig):
            raise ValueError("The lux model requires the luxai_s3 environment and vice versa")
        if isinstance(self.model, LuxModelConfig) and self.model.attention_implementation == "cudnn" and not self.bf16:
            raise ValueError("Lux cuDNN attention requires bf16=true; use attention_implementation=xla for float32")
        if self.map_size_curriculum:
            if not isinstance(self.env, (LuxConfig, JaxNavConfig)):
                raise ValueError("map_size_curriculum requires a Lux or JaxNav environment")
            if self.map_size_curriculum[0].start_iteration != 0:
                raise ValueError("map_size_curriculum must start at iteration 0")
            previous = -1
            for stage in self.map_size_curriculum:
                if stage.start_iteration <= previous:
                    raise ValueError("map_size_curriculum thresholds must be strictly increasing")
                replace(self.env, map_size=stage.map_size)
                previous = stage.start_iteration
        return self


def config_for_iteration(config: Config, completed_iterations: int) -> Config:
    """Resolve the active map size without changing the saved curriculum."""
    for stage in reversed(config.map_size_curriculum):
        if completed_iterations >= stage.start_iteration:
            env = cast(LuxConfig | JaxNavConfig, config.env)
            return (
                config if env.map_size == stage.map_size else replace(config, env=replace(env, map_size=stage.map_size))
            )
    return config


def make_actor_critic(
    action_space: MultiAgentActionSpace,
    model: ModelConfig,
    dtype: jax.typing.DTypeLike = jnp.float32,
) -> GenericActorCritic | LuxActorCritic:
    """Select a concrete model; validate the Lux model/action-space pairing."""
    if isinstance(model, LuxModelConfig):
        if not isinstance(action_space, HierarchicalMultiDiscreteSpace):
            raise TypeError("The lux model requires a hierarchical multi-discrete action space")
        return LuxActorCritic(action_space=action_space, model=model, dtype=dtype)
    return GenericActorCritic(action_space=action_space, model=model, dtype=dtype)


def make_env(config: Config) -> PPOEnv:
    env: PPOEnv
    if isinstance(config.env, SMAXConfig):
        env = SMAXEnv(config.env)
    elif isinstance(config.env, JaxNavConfig):
        env = JaxNavEnv(config.env)
    elif isinstance(config.env, MPEConfig):
        env = MPEEnv(config.env)
    elif isinstance(config.env, LuxConfig):
        env = LuxEnv(config.env)
    elif isinstance(config.env, STORMConfig):
        env = STORMEnv(config.env)
    else:
        env = RobotariumEnv(config.env)
    if config.selfplay is not None:
        if not env.supports_selfplay:
            raise ValueError("selfplay pool requires an environment with supports_selfplay=True")
        if env.num_agents != 2:
            raise ValueError("selfplay pool requires two controlled agents; multi-agent teams need explicit grouping")
    if config.eval_ref_policies and (not env.supports_selfplay or env.num_agents != 2):
        raise ValueError("eval_ref_policies requires a self-play environment with two controlled agents")
    sequence_count = config.num_envs * (1 if config.selfplay is not None else env.num_agents)
    if sequence_count % config.num_minibatches:
        label = "num_envs" if config.selfplay is not None else "num_envs * num_agents"
        raise ValueError(f"{label} must be divisible by num_minibatches to preserve sequences")
    return env


class TrainingProgress(NamedTuple):
    train_state: TrainState
    key: jax.Array  # () typed PRNG key
    rollout_count: int  # Number of completed rollouts, independent of optimizer steps.
    episode_count: int
    episode_return_sum: float
    episode_length_sum: int
    environment_step_count: int | None = None  # None for legacy checkpoints without transition counters.
    opponent_pool: OpponentPool | None = None


def init_metrics(progress: TrainingProgress | None = None) -> TrainingMetrics:
    """Initialize () metric leaves, preserving restored totals with 64-bit dtypes."""
    with jax.enable_x64():
        total = EpisodeMetrics(
            episode_count=jnp.asarray(progress.episode_count if progress else 0, dtype=jnp.int64),  # ()
            episode_return_sum=jnp.asarray(progress.episode_return_sum if progress else 0.0, dtype=jnp.float64),  # ()
            episode_length_sum=jnp.asarray(progress.episode_length_sum if progress else 0, dtype=jnp.int64),  # ()
        )
        window = WindowMetrics(
            episodes=RunningSum.init(total),  # scalar leaves
            terminated_episode_count=jnp.zeros((), jnp.int64),  # ()
            minibatch_metrics=MinibatchMetrics(
                **{field.name: jnp.zeros((), jnp.float64) for field in fields(MinibatchMetrics)}
            ),
            explained_variance_sum=jnp.zeros((), jnp.float64),  # ()
            explained_variance_count=jnp.zeros((), jnp.int64),  # ()
            minibatch_count=jnp.zeros((), jnp.int64),  # ()
            rollout_count=jnp.zeros((), jnp.int64),  # ()
            update_count=jnp.zeros((), jnp.int64),  # ()
            early_stop_count=jnp.zeros((), jnp.int64),  # ()
        )
    return TrainingMetrics(window=window, total=total)  # scalar leaves


def checkpoint_manager(run_dir: str) -> ocp.CheckpointManager:
    directory = f"{run_dir.rstrip('/')}/checkpoints"
    if not directory.startswith("gs://"):
        directory = str(Path(directory).resolve())
    return ocp.CheckpointManager(
        directory,
        options=ocp.CheckpointManagerOptions(max_to_keep=2, enable_async_checkpointing=True),
    )


def save_checkpoint(manager: ocp.CheckpointManager, progress: TrainingProgress, config: Config) -> float:
    """Submit an async checkpoint and return the blocking preparation/submission seconds."""
    started = monotonic()
    key_data = jax.random.key_data(progress.key)  # (K,) raw PRNG words
    pool_items = (
        {}
        if progress.opponent_pool is None
        else {"opponent_pool": ocp.args.StandardSave(progress.opponent_pool._asdict())}
    )
    manager.save(
        progress.rollout_count,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave({"train_state": progress.train_state, "key": key_data}),
            **pool_items,
            metadata=ocp.args.JsonSave(
                {
                    "version": 2,
                    "algorithm": f"{config.env.id.lower()}_ippo",
                    "config": asdict(config),
                    "iteration": progress.rollout_count,
                    "completed_episodes": progress.episode_count,
                    "training_return_sum": progress.episode_return_sum,
                    "training_length_sum": progress.episode_length_sum,
                    "environment_steps": progress.environment_step_count,
                    "opponent_pool": progress.opponent_pool is not None,
                }
            ),
        ),
        force=True,
    )
    seconds = monotonic() - started
    print(f"Started checkpoint at iteration {progress.rollout_count}; blocked for {seconds:.1f}s", flush=True)
    return seconds


def restore_checkpoint(
    manager: ocp.CheckpointManager, train_state: TrainState, config: Config
) -> TrainingProgress | None:
    rollout_count = manager.latest_step()
    if rollout_count is None:
        return None
    metadata = manager.restore(rollout_count, args=ocp.args.Composite(metadata=ocp.args.JsonRestore())).metadata
    if metadata["version"] != 2:
        raise ValueError("Unsupported PPO checkpoint version; the shared encoder/LSTM requires a new run directory")
    if metadata["iteration"] != rollout_count or rollout_count < 0 or metadata["completed_episodes"] < 0:
        raise ValueError("Invalid PPO checkpoint progress")
    for field in ("training_return_sum", "training_length_sum"):
        if metadata.get(field) is None:
            raise ValueError(f"PPO checkpoint requires {field}; use a new run directory")
    restored = manager.restore(
        rollout_count,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(
                {
                    "train_state": train_state,
                    "key": jax.random.key_data(jax.random.key(config.seed)),
                }
            )
        ),
    ).state
    pool = None
    if config.selfplay is not None and metadata.get("opponent_pool", False):
        saved_pool = manager.restore(
            rollout_count, args=ocp.args.Composite(opponent_pool=ocp.args.StandardRestore())
        ).opponent_pool
        pool = resize_pool(OpponentPool(**saved_pool), config.selfplay.pool_size)
    return TrainingProgress(
        train_state=restored["train_state"],
        key=jax.random.wrap_key_data(restored["key"]),
        rollout_count=rollout_count,
        episode_count=metadata["completed_episodes"],
        episode_return_sum=metadata["training_return_sum"],
        episode_length_sum=metadata["training_length_sum"],
        environment_step_count=metadata.get("environment_steps"),
        opponent_pool=pool,
    )


def load_eval_ref_policies(config: Config, params: optax.Params) -> dict[str, optax.Params]:  # parameter-shaped leaves
    """Restore only fixed opponent weights, with the current parameters' shapes and sharding."""
    if not config.eval_interval or not config.eval_ref_policies:
        return {}
    target = {"train_state": {"params": params}}  # leaves: parameter shapes
    references: dict[str, optax.Params] = {}  # leaves: parameter shapes
    with ocp.PyTreeCheckpointer() as checkpointer:
        for name, checkpoint in config.eval_ref_policies.items():
            directory = f"{checkpoint.rstrip('/')}/state"
            if not directory.startswith("gs://"):
                directory = str(Path(directory).resolve())
            restored = checkpointer.restore(
                directory,
                args=ocp.args.PyTreeRestore(
                    item=target,
                    restore_args=ocp.checkpoint_utils.construct_restore_args(target),
                    partial_restore=True,
                ),
            )  # leaves: parameter shapes
            references[name] = restored["train_state"]["params"]
            print(f"Loaded evaluation reference {name}: {checkpoint}", flush=True)
    return references  # leaves: parameter shapes


def decay_schedule(initial_value: float, total_iterations: int, decay: Literal["linear", "cosine"]) -> optax.Schedule:
    schedule = (
        optax.cosine_decay_schedule(initial_value, 1)
        if decay == "cosine"
        else optax.linear_schedule(initial_value, 0.0, 1)
    )

    def by_iterations(count: jax.Array) -> jax.Array:  # () -> ()
        # Normalize before Optax's clipping to support budgets beyond int32.
        return schedule(jnp.asarray(count, dtype=jnp.float32) / float(total_iterations))  # ()

    return by_iterations


def learning_rate_schedule(config: Config) -> optax.Schedule:
    if not config.anneal_lr:
        return optax.constant_schedule(config.learning_rate)
    return decay_schedule(config.learning_rate, config.total_iterations, config.lr_decay)


def entropy_coef_schedule(config: Config) -> optax.Schedule:
    if config.entropy_decay == "constant":
        return optax.constant_schedule(config.entropy_coef)
    return decay_schedule(config.entropy_coef, config.total_iterations, "cosine")


class EpisodeState(NamedTuple):
    key: jax.Array  # () typed PRNG key
    observation: AgentObservation[Observation]  # leaves: (N, ...); mask: (N, *C, A)
    env_state: Any  # leaves: (...)
    model_carry: RecurrentCarry  # leaves: (N, ...)
    episode_starts: jax.Array  # (N,)
    team_returns: jax.Array  # (N,)
    length: jax.Array  # ()
    done: jax.Array  # ()


class EpisodeResult(NamedTuple):
    episode_return: float
    length: int
    frames: list[NDArray[np.uint8]]  # each (H, W, 3)
    opponent_return: float | None  # None without a reference opponent


class EvaluationMatch(NamedTuple):
    current: optax.Params  # leaves: parameter shapes
    opponent: optax.Params  # leaves: matching parameter shapes
    current_team: jax.Array  # () team index


class CompiledEpisode(NamedTuple):
    episode_return: jax.Array  # ()
    length: jax.Array  # ()
    states: Any  # leaves: (F, ...), initial then pre-autoreset states; None without recording
    opponent_return: jax.Array | None  # () with a reference opponent


@partial(jax.jit, static_argnames=("env", "apply_fn", "recording_capacity", "stop_at_frame_limit"))
def run_episode(
    env: PPOEnv,
    apply_fn: Callable[..., tuple[ActorCriticOutput[Policy], RecurrentCarry]],
    params: optax.Params,  # leaves: parameter shapes, or EvaluationMatch
    key: jax.Array,  # () typed PRNG key
    initial_carry: RecurrentCarry,  # leaves: (N, ...)
    *,
    recording_capacity: int,
    stop_at_frame_limit: bool,
) -> CompiledEpisode:
    """Compile reset and the full episode; recording_capacity=0 disables state recording."""
    key, reset_key = jax.random.split(key)  # each () typed PRNG key
    observation, env_state = env.reset(reset_key)  # observation/state leaves: (...)
    state = EpisodeState(
        key=key,  # ()
        observation=observation,  # (N, ...)/(N, *C, A)
        env_state=env_state,  # leaves: (...)
        model_carry=initial_carry,  # leaves: (N, ...)
        episode_starts=jnp.ones(env.num_agents, dtype=bool),  # (N,)
        team_returns=jnp.zeros(env.num_agents, dtype=jnp.float32),  # (N,)
        length=jnp.asarray(0, dtype=jnp.int32),  # ()
        done=jnp.asarray(False),  # ()
    )
    saved = None
    if recording_capacity > 0:
        saved = jax.tree.map(
            lambda leaf: jnp.zeros_like(leaf, shape=(recording_capacity, *jnp.shape(leaf))).at[0].set(leaf), env_state
        )  # leaves: (F, ...), initial state at index zero

    def condition(loop_state: tuple[EpisodeState, Any]) -> jax.Array:  # ()
        state, _ = loop_state
        if recording_capacity > 0 and stop_at_frame_limit:
            return ~state.done & (state.length < recording_capacity - 1)  # ()
        return ~state.done  # ()

    def body(loop_state: tuple[EpisodeState, Any]) -> tuple[EpisodeState, Any]:
        state, saved = loop_state  # saved leaves: (F, ...) or None
        key, action_key, env_key = jax.random.split(state.key, 3)  # each () typed PRNG key
        if isinstance(params, EvaluationMatch):
            model_carry = state.model_carry  # leaves: (N=2, ...)
            actions = []
            for team, weights in ((params.current_team, params.current), (1 - params.current_team, params.opponent)):
                indices = team[None]  # (1,)
                observation, team_model_carry, starts, mask = jax.tree.map(
                    itemgetter(indices),
                    (
                        state.observation.observation,
                        state.model_carry,
                        state.episode_starts,
                        state.observation.actions_mask,
                    ),
                )  # observation: (1, ...); team_model_carry: (1, ...); starts: (1,); mask: (1, *C, A)
                (policy, _), team_model_carry = apply_fn(
                    {"params": weights}, observation, team_model_carry, starts, mask, method="step"
                )  # team_model_carry leaves: (1, ...); logits: (1, *C, A); value: (1,)
                action = policy.sample(key=jax.random.fold_in(action_key, team))  # (1, *C)
                model_carry = jax.tree.map(
                    lambda full, part, indices=indices: full.at[indices].set(part), model_carry, team_model_carry
                )  # leaves: (N, ...)
                actions.append(action[0])  # (*C,)
            action = jnp.stack(actions)[jnp.arange(2) ^ params.current_team]  # (N=2, *C)
        else:
            (policy, _), model_carry = apply_fn(
                {"params": params},
                state.observation.observation,
                state.model_carry,
                state.episode_starts,
                state.observation.actions_mask,
                method="step",
            )  # model_carry leaves: (N, ...); logits: (N, *C, A); value: (N,)
            action = policy.sample(key=action_key)  # (N, *C)
        result = env.step(env_key, state.env_state, action)
        state = state._replace(
            key=key,  # ()
            observation=result.observation,  # (N, ...)/(N, *C, A)
            env_state=result.state,  # leaves: (...)
            model_carry=model_carry,  # leaves: (N, ...)
            episode_starts=jnp.asarray(result.agent_done, dtype=bool),  # (N,)
            team_returns=state.team_returns + result.reward,  # (N,)
            length=state.length + 1,  # ()
            done=jnp.asarray(result.done, dtype=bool),  # ()
        )
        if recording_capacity > 0:
            # Keep scoring after the video cap without writing past its buffer.
            saved = jax.tree.map(
                lambda buffer, leaf: buffer.at[state.length].set(leaf, mode="drop"), saved, result.final_state
            )  # leaves: (F, ...)
        return state, saved

    state, saved = jax.lax.while_loop(condition, body, (state, saved))
    return CompiledEpisode(
        episode_return=(
            state.team_returns[params.current_team]
            if isinstance(params, EvaluationMatch)
            else state.team_returns.mean()
        ),  # ()
        length=state.length,  # ()
        states=saved,  # leaves: (F, ...) or None
        opponent_return=(
            state.team_returns[1 - params.current_team] if isinstance(params, EvaluationMatch) else None
        ),  # () or None
    )


def play_episode(
    train_state: TrainState,
    config: Config,
    key: jax.Array,  # () typed PRNG key
    initial_carry: RecurrentCarry,  # leaves: (N, ...)
    *,
    record: bool,
    env: PPOEnv,
    stop_at_frame_limit: bool = False,
    opponent_params: optax.Params | None = None,  # leaves: parameter shapes
    current_team: int = 0,
    team_labels: tuple[str, str] | None = None,
) -> EpisodeResult:
    """Run one compiled episode, then transfer once and render its valid recorded states."""
    max_episode_steps = (
        (config.env.max_steps_in_match + 1) * config.env.match_count
        if isinstance(config.env, LuxConfig)
        else config.env.max_steps
    )
    recording_capacity = 0
    if record:
        recording_capacity = max_episode_steps + 1
        if config.video_max_frames is not None:
            recording_capacity = min(recording_capacity, config.video_max_frames)
    result = run_episode(
        env,
        train_state.apply_fn,
        train_state.params
        if opponent_params is None
        else EvaluationMatch(
            current=train_state.params,
            opponent=opponent_params,
            current_team=jnp.asarray(current_team, dtype=jnp.int32),  # ()
        ),
        key,
        initial_carry,
        recording_capacity=recording_capacity,
        stop_at_frame_limit=stop_at_frame_limit,
    )
    result = jax.device_get(result)  # return/length: (); states: (F, ...) or None
    frames = []  # each (H, W, 3) uint8
    if record:
        for index in range(min(int(result.length) + 1, recording_capacity)):
            env_state = jax.tree.map(itemgetter(index), result.states)  # leaves: (...)
            frame = (
                env.render(env_state, team_labels=team_labels) if isinstance(env, LuxEnv) else env.render(env_state)
            )  # (H, W, 3)
            frames.append(np.asarray(frame))  # (H, W, 3)
    return EpisodeResult(
        episode_return=float(result.episode_return),
        length=int(result.length),
        frames=frames,
        opponent_return=None if result.opponent_return is None else float(result.opponent_return),
    )


def log_video(
    train_state: TrainState,
    config: Config,
    writer: MLflowLogger,
    episode_count: int,
    step_count: int,
    *,
    initial_carry: RecurrentCarry,  # leaves: (N, ...)
    env: PPOEnv | None = None,
) -> None:
    """Record a separate episode up to the frame cap and log its gameplay GIF."""
    print(f"Recording gameplay at step {step_count} after {episode_count} training episodes...", flush=True)
    started = monotonic()
    episode = play_episode(
        train_state,
        config,
        jax.random.fold_in(jax.random.key(config.seed), episode_count),
        initial_carry,
        record=True,
        stop_at_frame_limit=True,
        env=make_env(config) if env is None else env,
    )
    writer.log_video(episode.frames, f"gameplay/{step_count}.gif", fps=config.video_fps)
    seconds = monotonic() - started
    writer.log_metrics({"actor/video_seconds": seconds}, step_count)
    print(
        f"Recorded {len(episode.frames)} gameplay frames after {episode_count} training episodes in {seconds:.1f}s",
        flush=True,
    )


def log_reference_evaluation(
    train_state: TrainState,
    config: Config,
    writer: MLflowLogger,
    episode_count: int,
    step_count: int,
    *,
    initial_carry: RecurrentCarry,  # leaves: (N, ...)
    opponent_params: optax.Params,  # leaves: parameter shapes
    opponent_name: str,
    env: PPOEnv | None = None,
) -> list[tuple[float, float]]:
    """Log reference games and replay; return (current return, opponent return) pairs."""
    print(f"Evaluating {config.eval_episodes} games at step {step_count}...", flush=True)
    suffix = f"/{opponent_name}"
    started = monotonic()
    episode_results: list[dict[str, float | int]] = []
    env = make_env(config) if env is None else env
    for episode_index in range(config.eval_episodes):
        episode = play_episode(
            train_state,
            config,
            jax.random.fold_in(jax.random.key(config.eval_seed), episode_index),
            initial_carry,
            record=False,
            env=env,
            opponent_params=opponent_params,
            current_team=episode_index % 2,
        )
        opponent_return = cast(float, episode.opponent_return)
        episode_results.append(
            {
                "return": episode.episode_return,
                "opponent_return": opponent_return,
                "length": episode.length,
                "current_team": episode_index % 2,
            }
        )
    episode_return_sample = [result["return"] for result in episode_results]
    video_episode_index = max(range(config.eval_episodes), key=episode_return_sample.__getitem__)
    team_labels = ("Trained policy", f"Reference: {opponent_name}")
    if video_episode_index % 2:
        team_labels = team_labels[::-1]
    replay = play_episode(
        train_state,
        config,
        jax.random.fold_in(jax.random.key(config.eval_seed), video_episode_index),
        initial_carry,
        record=True,
        stop_at_frame_limit=True,
        env=env,
        opponent_params=opponent_params,
        current_team=video_episode_index % 2,
        team_labels=team_labels,
    )
    if replay.frames:
        writer.log_video(replay.frames, f"eval/gameplay{suffix}/{step_count}.gif", fps=config.video_fps)
    std = stdev(episode_return_sample) if len(episode_return_sample) > 1 else None
    result = {
        "env_id": config.env.id,
        "return_mean": fmean(episode_return_sample),
        "return_std": std,
        "return_sem": std / sqrt(config.eval_episodes) if std is not None else None,
        "episodes": episode_results,
        "video_episode_index": video_episode_index,
        "eval_seed": config.eval_seed,
        "training_config": asdict(config),
        "training_steps": step_count,
        "training_episodes": episode_count,
        "opponent_name": opponent_name,
        "opponent_checkpoint": config.eval_ref_policies[opponent_name],
    }
    scalars = {
        f"eval/episode_{name.removesuffix('_mean')}{suffix}": result[name]
        for name in ("return_mean", "return_std", "return_sem")
        if result[name] is not None
    }
    writer.log_text(json.dumps(result, indent=2, allow_nan=False), f"eval/report{suffix}/{step_count}.json")
    seconds = monotonic() - started
    scalars[f"eval/evaluation_seconds{suffix}"] = seconds
    writer.log_metrics(scalars, step_count)
    print(
        f"Evaluation: episode_return={result['return_mean']:.1f} over {config.eval_episodes} games in {seconds:.1f}s",
        flush=True,
    )
    return [(result["return"], result["opponent_return"]) for result in episode_results]


def rate_evaluation(
    outcomes: dict[str, list[tuple[float, float]]],
    seed: int,
) -> tuple[trueskill.Rating, dict[str, trueskill.Rating]]:
    """Rate (current return, opponent return) pairs once in a seeded shuffled order."""
    rating_env = trueskill.TrueSkill()
    current = rating_env.create_rating()
    references = {name: rating_env.create_rating() for name in sorted(outcomes)}
    matches = [(name, index) for name in references for index in range(len(outcomes[name]))]
    Random(seed).shuffle(matches)
    for name, index in matches:
        current_return, opponent_return = outcomes[name][index]
        (current,), (references[name],) = rating_env.rate(
            [(current,), (references[name],)],
            ranks=[int(current_return < opponent_return), int(opponent_return < current_return)],
        )
    return current, references


def log_evaluation(
    train_state: TrainState,
    config: Config,
    writer: MLflowLogger,
    episode_count: int,
    step_count: int,
    *,
    initial_carry: RecurrentCarry,  # leaves: (N, ...)
    opponents: dict[str, optax.Params],  # values: parameter-shaped leaves
    env: PPOEnv | None = None,
) -> None:
    """Evaluate all references and jointly rate this checkpoint's match results."""
    if not opponents:
        return
    started = monotonic()
    env = make_env(config) if env is None else env
    outcomes = {
        name: log_reference_evaluation(
            train_state,
            config,
            writer,
            episode_count,
            step_count,
            initial_carry=initial_carry,
            opponent_params=params,
            opponent_name=name,
            env=env,
        )
        for name, params in opponents.items()
    }
    current, references = rate_evaluation(outcomes, config.eval_seed)
    ratings = [("trueskill", "", current)] + [
        ("reference_trueskill", f"/{name}", rating) for name, rating in references.items()
    ]
    scalars = {
        f"eval/{prefix}{measurement}{suffix}": value
        for prefix, suffix, rating in ratings
        for measurement, value in (
            ("", rating.mu),
            ("_std", rating.sigma),
            ("_conservative", rating.mu - 3 * rating.sigma),
        )
    }
    writer.log_text(
        json.dumps(
            {
                "ratings": scalars,
                "eval_seed": config.eval_seed,
                "training_steps": step_count,
                "opponent_checkpoints": config.eval_ref_policies,
            },
            indent=2,
            allow_nan=False,
        ),
        f"eval/ratings/{step_count}.json",
    )
    writer.log_metrics(scalars, step_count)
    writer.log_metrics({"eval/evaluation_seconds": monotonic() - started}, step_count)


def update(
    train_state: TrainState,
    batch: PPOBatch,  # observation: (*S, ...); actions_mask: (*S, *C, A); targets/starts: (*S,); model_carry leaves: (B, ...)
    config: Config,
    iteration_count: int | jax.Array = 0,  # () completed iterations
) -> tuple[TrainState, MinibatchMetrics]:
    entropy_coef = jnp.asarray(entropy_coef_schedule(config)(iteration_count), dtype=jnp.float32)  # ()
    advantage = batch.advantage  # (*S,)
    chex.assert_scalar_positive(advantage.size)
    # Centering a single sample always gives zero and erases its policy gradient.
    if advantage.size > 1:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)  # (*S,)

    def loss_fn(params: optax.Params) -> tuple[jax.Array, MinibatchMetrics]:  # (); scalar metric leaves
        (policy, value), _ = train_state.apply_fn(
            {"params": params}, batch.observation, batch.model_carry, batch.episode_start, batch.actions_mask
        )  # model_carry leaves: (B, ...); policy logits: (*S, *C, A); value: (*S,)
        log_prob = policy.log_prob(batch.action)  # (*S,)
        log_ratio = log_prob - batch.old_log_prob  # (*S,)
        ratio = jnp.exp(log_ratio)  # (*S,)
        flat_ratio, flat_advantage = jax.tree.map(jnp.ravel, (ratio, advantage))  # (N,), N = prod(S)
        policy_loss = rlax.clipped_surrogate_pg_loss(flat_ratio, flat_advantage, config.clip_coef)  # ()
        value_loss = rlax.l2_loss(value, batch.return_target).mean()  # ()
        entropy = policy.entropy().mean()  # ()
        loss = policy_loss + config.value_coef * value_loss - entropy_coef * entropy  # ()
        approx_kl = (jnp.expm1(log_ratio) - log_ratio).mean()  # ()
        clip_fraction = (jnp.abs(ratio - 1) > config.clip_coef).mean()  # ()
        return loss, MinibatchMetrics(
            policy_loss=policy_loss,  # ()
            value_loss=value_loss,  # ()
            total_loss=loss,  # ()
            entropy=entropy,  # ()
            approx_kl=approx_kl,  # ()
            clip_fraction=clip_fraction,  # ()
        )  # loss: (); metrics: scalar leaves

    loss, backward, metrics = jax.vjp(loss_fn, train_state.params, has_aux=True)  # loss: (); metric leaves: ()

    def apply_gradients() -> TrainState:
        # KL-rejected minibatches need diagnostics but no backpropagation.
        (grads,) = backward(jnp.ones_like(loss))  # parameter-shaped leaves
        grads = jax.lax.with_sharding_constraint(grads, PartitionSpec())  # parameter-shaped leaves
        return train_state.apply_gradients(grads=grads)

    if config.target_kl is None:
        return apply_gradients(), metrics
    train_state = jax.lax.cond(
        metrics.approx_kl > config.target_kl,
        lambda: train_state,
        apply_gradients,
    )
    return train_state, metrics


def collect_rollout(
    rollout_carry: RolloutCarry,  # observation: (E, N, ...); state: (E, ...); model_carry: (B, ...)
    train_state: TrainState,
    env: PPOEnv,
    config: Config,
    key: jax.Array,  # () typed PRNG key
    opponent_pool: OpponentPool | None = None,
) -> tuple[RolloutCarry, RolloutResult]:
    """Collect (B, T, ...) transitions entirely on device and compute GAE.

    Timeouts bootstrap from final observations; agent end flags reset recurrence.
    Team rewards and GAE traces continue until environment completion.
    Completed episode returns, lengths, and count are reduced to scalars on device.
    """
    actor_count = config.num_envs * env.num_agents

    class Runner(NamedTuple):
        current: RolloutCarry  # state: (E, ...); model_carry: (B, ...)
        key: jax.Array  # () typed PRNG key
        episode_metrics: EpisodeMetrics  # scalar leaves
        terminated_episode_count: jax.Array  # ()

    class Transition(NamedTuple):
        """S is (B,) per step, (T, B) after scanning, then (B, T) for training."""

        observation: Observation  # leaves: (*S, ...)
        actions_mask: jax.Array  # (*S, *C, A)
        action: jax.Array  # (*S, *C)
        log_prob: jax.Array  # (*S,)
        value: jax.Array  # (*S,)
        reward: jax.Array  # (*S,)
        episode_end: jax.Array  # (*S,)
        episode_start: jax.Array  # (*S,)

    def flatten(array: jax.Array) -> jax.Array:  # (E, N, ...) -> (B, ...)
        return array.reshape((actor_count, *array.shape[2:]))  # (B, ...)

    def step(
        runner: Runner,  # state: (E, ...); model_carry: (B, ...); key/completed totals: ()
        _: None,
    ) -> tuple[Runner, Transition]:
        current, key, episode_metrics, terminated_episode_count = (
            runner  # state: (E, ...); model_carry: (B, ...); totals: ()
        )
        key, action_key, env_key = jax.random.split(key, 3)  # each () typed PRNG key
        observation = jax.tree.map(flatten, current.observation)  # (B, ...)/(B, *C, A)
        (policy, prediction), model_carry = train_state.apply_fn(
            {"params": train_state.params},
            observation.observation,
            current.model_carry,
            current.episode_start,
            observation.actions_mask,
            method="step",
        )  # model_carry leaves: (B, ...); logits: (B, *C, A); prediction: (B,)
        action, log_prob = policy.sample_and_log_prob(key=action_key)  # (B, *C); (B,)
        assignment = current.selfplay
        if opponent_pool is not None:
            assignment = cast(MatchAssignment, assignment)
            key, opponent_key = jax.random.split(key)  # each ()
            opponent_keys = jax.random.split(opponent_key, config.num_envs)  # (E,)
            opponent_indices = 2 * jnp.arange(config.num_envs) + 1 - assignment.learner_team  # (E,)

            def opponent_step(env_index: jax.Array) -> tuple[jax.Array, RecurrentCarry]:  # () -> (*C), leaves: (...)
                index = opponent_indices[env_index]  # ()
                slot = assignment.opponent[env_index]  # ()

                def historical() -> tuple[jax.Array, RecurrentCarry]:
                    inputs = jax.tree.map(lambda leaf: leaf[index][None], observation)  # (1, ...)/(1, *C, A)
                    opponent_model_carry = jax.tree.map(lambda leaf: leaf[index][None], current.model_carry)  # (1, ...)
                    output, opponent_model_carry = train_state.apply_fn(
                        {"params": jax.tree.map(lambda leaf: leaf[slot], opponent_pool.params)},
                        inputs.observation,
                        opponent_model_carry,
                        current.episode_start[index][None],
                        inputs.actions_mask,
                        method="step",
                    )
                    command = output.policy.sample(key=opponent_keys[env_index])[0]  # (*C,)
                    return command, jax.tree.map(lambda leaf: leaf[0], opponent_model_carry)  # (*C,); leaves: (...)

                return jax.lax.cond(
                    slot >= 0,
                    historical,
                    lambda: (action[index], jax.tree.map(lambda leaf: leaf[index], model_carry)),
                )

            # Map one environment at a time so inference does not gather E copies of the parameters.
            opponent_action, opponent_model_carry = jax.lax.map(opponent_step, jnp.arange(config.num_envs))
            # opponent_action: (E, *C); opponent_model_carry leaves: (E, ...)
            action = action.at[opponent_indices].set(opponent_action)  # (B, *C)
            model_carry = jax.tree.map(
                lambda all_agents, opponents: all_agents.at[opponent_indices].set(opponents),
                model_carry,
                opponent_model_carry,
            )
        env_key = jax.random.split(env_key, config.num_envs)  # (E,) typed PRNG keys
        result = jax.vmap(env.step)(
            env_key,
            current.env_state,
            action.reshape((config.num_envs, env.num_agents, *action.shape[1:])),  # (E, N, *C)
        )
        timeout = jnp.repeat(result.truncated & ~result.terminated, env.num_agents)  # (B,)
        # Weak boolean outputs would change the next iteration's compilation signature.
        episode_start = jnp.asarray(result.agent_done, dtype=bool).reshape(actor_count)  # (B,)

        def bootstrap(_: None) -> jax.Array:  # (B,)
            final = jax.tree.map(flatten, result.final_observation)  # (B, ...)/(B, *C, A)
            # Agent inactivity resets memory without ending the team's reward stream.
            reset = result.agent_inactive.reshape(actor_count)  # (B,)
            value = train_state.apply_fn(
                {"params": train_state.params},
                final.observation,
                model_carry,
                reset,
                final.actions_mask,
                method="step",
            )[0].value  # (B,)
            return jnp.where(timeout, value, 0.0)  # (B,)

        def no_bootstrap(_: None) -> jax.Array:
            return jnp.zeros(actor_count)  # (B,)

        raw_reward = result.reward.reshape(actor_count)  # (B,)
        reward = raw_reward + config.gamma * jax.lax.cond(timeout.any(), bootstrap, no_bootstrap, None)  # (B,)
        episode_reward = (
            result.reward.mean(axis=-1)
            if assignment is None
            else result.reward[jnp.arange(config.num_envs), assignment.learner_team]
        )  # (E,)
        episode_return = current.episode_return + episode_reward  # (E,)
        episode_step_count = current.episode_step_count + 1  # (E,)
        episode_metrics = RunningSum.update(
            episode_metrics,
            jax.tree.map(
                jnp.sum,
                EpisodeMetrics(
                    episode_count=result.done,  # (E,)
                    episode_return_sum=jnp.where(result.done, episode_return, 0.0),  # (E,)
                    episode_length_sum=jnp.where(result.done, episode_step_count, 0),  # (E,)
                ),
            ),
        )  # each ()
        next_assignment = assignment
        if opponent_pool is not None:
            key, assignment_key = jax.random.split(key)  # each ()
            sampled = sample_opponents(
                assignment_key, opponent_pool, cast(SelfPlayConfig, config.selfplay), config.num_envs
            )
            next_assignment = jax.tree.map(lambda new, old: jnp.where(result.done, new, old), sampled, assignment)
        advanced = RolloutCarry(
            env_state=result.state,  # leaves: (E, ...)
            observation=result.observation,  # observation: (E, N, ...); actions_mask: (E, N, *C, A)
            model_carry=model_carry,  # leaves: (B, ...)
            episode_start=episode_start,  # (B,)
            episode_return=jnp.where(result.done, 0.0, episode_return),  # (E,)
            episode_step_count=jnp.where(result.done, 0, episode_step_count),  # (E,)
            selfplay=next_assignment,  # leaves: (E,)
        )
        transition = Transition(
            observation=observation.observation,  # (B, ...)
            actions_mask=observation.actions_mask,  # (B, *C, A)
            action=action,  # (B, *C)
            log_prob=log_prob,  # (B,)
            value=prediction,  # (B,)
            reward=reward,  # (B,)
            episode_end=jnp.repeat(result.done, env.num_agents),  # (B,)
            episode_start=current.episode_start,  # (B,)
        )
        if assignment is not None:
            learner_indices = 2 * jnp.arange(config.num_envs) + assignment.learner_team  # (E,)
            transition = jax.tree.map(lambda leaf: leaf[learner_indices], transition)  # leaves: (E, ...)
        return Runner(
            current=advanced,
            key=key,  # ()
            episode_metrics=episode_metrics,  # scalar leaves
            terminated_episode_count=terminated_episode_count + jnp.sum(result.done & result.terminated),  # ()
        ), transition

    episode_metrics = EpisodeMetrics(
        episode_count=jnp.zeros((), dtype=int),  # ()
        episode_return_sum=jnp.zeros((), dtype=rollout_carry.episode_return.dtype),  # ()
        episode_length_sum=jnp.zeros((), dtype=int),  # ()
    )
    (advanced, key, episode_metrics, terminated_episode_count), trajectory = jax.lax.scan(
        step,
        Runner(
            current=rollout_carry,
            key=key,  # ()
            episode_metrics=episode_metrics,  # scalar leaves
            terminated_episode_count=jnp.zeros((), dtype=int),  # ()
        ),
        None,
        length=config.num_steps,
    )  # trajectory leaves: (T, B, ...)
    # Peek at the next value without advancing the rollout's recurrent state.
    observation = jax.tree.map(flatten, advanced.observation)  # (B, ...)/(B, *C, A)
    next_value = train_state.apply_fn(
        {"params": train_state.params},
        observation.observation,
        advanced.model_carry,
        advanced.episode_start,
        observation.actions_mask,
        method="step",
    )[0].value  # (B,)
    initial_carry = rollout_carry.model_carry  # leaves: (B, ...)
    if advanced.selfplay is not None:
        learner_indices = 2 * jnp.arange(config.num_envs) + advanced.selfplay.learner_team  # (E,)
        next_value = next_value[learner_indices]  # (E,)
        initial_indices = (
            2 * jnp.arange(config.num_envs) + cast(MatchAssignment, rollout_carry.selfplay).learner_team
        )  # (E,)
        initial_carry = jax.tree.map(lambda leaf: leaf[initial_indices], initial_carry)  # leaves: (E, ...)
    advantage, return_target = gae(
        trajectory.reward, trajectory.episode_end, trajectory.value, next_value, config.gamma, config.gae_lambda
    )  # each (T, B)
    trajectory, advantage, return_target = jax.tree.map(
        lambda leaf: jnp.swapaxes(leaf, 0, 1), (trajectory, advantage, return_target)
    )  # trajectory leaves: (B, T, ...); advantage/return_target: (B, T)
    batch = PPOBatch(
        observation=trajectory.observation,  # (B, T, ...)
        actions_mask=trajectory.actions_mask,  # (B, T, *C, A)
        action=trajectory.action,  # (B, T, *C)
        old_log_prob=trajectory.log_prob,  # (B, T)
        value=trajectory.value,  # (B, T)
        advantage=advantage,  # (B, T)
        return_target=return_target,  # (B, T)
        model_carry=initial_carry,  # leaves: (B, ...), B=E with an opponent pool
        episode_start=trajectory.episode_start,  # (B, T)
    )
    return advanced, RolloutResult(
        batch=batch,  # observation: (B, T, ...); targets/starts: (B, T); model_carry leaves: (B, ...)
        key=key,  # ()
        episode_return_sum=episode_metrics.episode_return_sum,  # ()
        episode_length_sum=episode_metrics.episode_length_sum,  # ()
        episode_count=episode_metrics.episode_count,  # ()
        terminated_episode_count=terminated_episode_count,  # ()
    )


def train_step(
    rollout_carry: RolloutCarry,  # observation: (E, N, ...); state: (E, ...); model_carry: (B, ...)
    train_state: TrainState,
    env: PPOEnv,
    config: Config,
    key: jax.Array,  # () typed PRNG key
    metrics: TrainingMetrics,  # scalar leaves
    iteration_count: int | jax.Array = 0,  # () completed iterations
    opponent_pool: OpponentPool | None = None,
) -> TrainStepResult:
    """Collect a rollout, run PPO epochs, and accumulate window/lifetime metrics."""
    rollout_carry, rollout = (
        collect_rollout(rollout_carry, train_state, env, config, key)
        if opponent_pool is None
        else collect_rollout(rollout_carry, train_state, env, config, key, opponent_pool)
    )
    batch = jax.lax.with_sharding_constraint(
        rollout.batch, PartitionSpec("batch")
    )  # (B, T, ...); model_carry: (B, ...)
    learning_rate = jnp.asarray(learning_rate_schedule(config)(iteration_count), dtype=jnp.float32)  # ()
    train_state = train_state.replace(
        opt_state=train_state.opt_state._replace(
            hyperparams={**train_state.opt_state.hyperparams, "learning_rate": learning_rate}  # ()
        )
    )

    class UpdateCarry(NamedTuple):
        train_state: TrainState
        key: jax.Array  # () typed PRNG key
        permutation: jax.Array  # (B,)
        minibatch_metrics: MinibatchMetrics  # scalar leaves
        minibatch_count: jax.Array  # () evaluated minibatches
        kl_exceeded: jax.Array  # () bool

    def condition(current: UpdateCarry) -> jax.Array:
        return (current.minibatch_count < config.update_epochs * config.num_minibatches) & ~current.kl_exceeded  # ()

    def body(current: UpdateCarry) -> UpdateCarry:
        def shuffle(key: jax.Array) -> tuple[jax.Array, jax.Array]:  # () typed PRNG key
            key, shuffle_key = jax.random.split(key)  # each () typed PRNG key
            return key, jax.random.permutation(shuffle_key, batch.action.shape[0])  # (); (B,)

        def keep_permutation(key: jax.Array) -> tuple[jax.Array, jax.Array]:  # () typed PRNG key
            return key, current.permutation  # (); (B,)

        minibatch_index = current.minibatch_count % config.num_minibatches  # ()
        key, permutation = jax.lax.cond(minibatch_index == 0, shuffle, keep_permutation, current.key)  # (); (B,)
        index = permutation.reshape(config.num_minibatches, -1)[minibatch_index]  # (M,), M = B / minibatches
        minibatch = batch.select_sequences(index)  # sequences: (M, T, ...); model_carry leaves: (M, ...)
        minibatch = jax.lax.with_sharding_constraint(
            minibatch, PartitionSpec("batch")
        )  # (M, T, ...); model_carry: (M, ...)
        train_state, metrics = update(current.train_state, minibatch, config, iteration_count)
        train_state = jax.lax.with_sharding_constraint(train_state, PartitionSpec())
        kl_exceeded = jnp.asarray(False) if config.target_kl is None else metrics.approx_kl > config.target_kl  # ()
        return UpdateCarry(
            train_state=train_state,
            key=key,  # ()
            permutation=permutation,  # (B,)
            minibatch_metrics=RunningSum.update(current.minibatch_metrics, metrics),  # scalar leaves
            minibatch_count=current.minibatch_count + 1,  # ()
            kl_exceeded=kl_exceeded,  # ()
        )

    updated = jax.lax.while_loop(
        condition,
        body,
        UpdateCarry(
            train_state=train_state,
            key=rollout.key,  # ()
            permutation=jnp.zeros(batch.action.shape[0], dtype=jnp.int32),  # (B,)
            minibatch_metrics=MinibatchMetrics(
                **{field.name: jnp.zeros((), jnp.float32) for field in fields(MinibatchMetrics)}
            ),
            minibatch_count=jnp.asarray(0, dtype=jnp.int32),  # ()
            kl_exceeded=jnp.asarray(False),  # ()
        ),
    )
    episodes = EpisodeMetrics(
        episode_count=rollout.episode_count,  # ()
        episode_return_sum=rollout.episode_return_sum,  # ()
        episode_length_sum=rollout.episode_length_sum,  # ()
    )
    has_return_variance = jnp.var(batch.return_target) > 0  # ()
    window = WindowMetrics(
        episodes=episodes,  # scalar leaves
        terminated_episode_count=rollout.terminated_episode_count,  # ()
        minibatch_metrics=updated.minibatch_metrics,  # scalar leaves
        explained_variance_sum=jnp.where(
            has_return_variance, explained_variance(batch.value, batch.return_target), 0.0
        ),  # ()
        explained_variance_count=has_return_variance.astype(jnp.int32),  # ()
        minibatch_count=updated.minibatch_count,  # ()
        rollout_count=jnp.asarray(1, dtype=jnp.int32),  # ()
        update_count=updated.minibatch_count - updated.kl_exceeded.astype(jnp.int32),  # ()
        early_stop_count=updated.kl_exceeded.astype(jnp.int32),  # ()
    )
    # Widen only metric arithmetic; model and environment computation keep their dtypes.
    with jax.enable_x64():
        metrics = RunningSum.update(metrics, TrainingMetrics(window=window, total=episodes))  # scalar leaves
    train_state, key, metrics = jax.lax.with_sharding_constraint(
        (updated.train_state, updated.key, metrics), PartitionSpec()
    )  # key: (); metrics: scalar leaves
    rollout_carry = jax.lax.with_sharding_constraint(rollout_carry, PartitionSpec("batch"))  # (E, ...)/(B, ...)
    return TrainStepResult(
        train_state=train_state,
        rollout_carry=rollout_carry,  # state: (E, ...); model_carry: (B, ...)
        key=key,  # ()
        metrics=metrics,  # scalar leaves
    )


# Compile the rollout and updates together; only the training loop relinquishes ownership.
train_step_donated = jax.jit(
    train_step,
    static_argnames=("env", "config"),
    donate_argnames=("train_state",),
    in_shardings=(PartitionSpec("batch"), PartitionSpec(), PartitionSpec(), PartitionSpec(), PartitionSpec()),
    out_shardings=TrainStepResult(
        train_state=PartitionSpec(),
        rollout_carry=PartitionSpec("batch"),
        key=PartitionSpec(),
        metrics=PartitionSpec(),
    ),
)


train_step_with_pool_donated = jax.jit(
    train_step,
    static_argnames=("env", "config"),
    donate_argnames=("train_state",),
    in_shardings=(
        PartitionSpec("batch"),
        PartitionSpec(),
        PartitionSpec(),
        PartitionSpec(),
        PartitionSpec(),
        PartitionSpec(),
    ),
    out_shardings=TrainStepResult(
        train_state=PartitionSpec(), rollout_carry=PartitionSpec("batch"), key=PartitionSpec(), metrics=PartitionSpec()
    ),
)


def training_mesh(config: Config, num_agents: int) -> Mesh:
    """Use every available device with equal environment and agent-sequence shards."""
    devices = jax.devices()
    if config.num_envs % len(devices):
        raise ValueError(f"num_envs ({config.num_envs}) must be divisible by device_count ({len(devices)})")
    learner_agents = 1 if config.selfplay is not None else num_agents
    if config.num_envs * learner_agents % (config.num_minibatches * len(devices)):
        raise ValueError(
            f"num_envs ({config.num_envs}) * learner_agents ({learner_agents}) must be divisible by "
            f"num_minibatches ({config.num_minibatches}) * device_count ({len(devices)})"
        )
    return Mesh(devices=devices, axis_names=("batch",), axis_types=(AxisType.Auto,))


def init_rollout_carry(
    env: PPOEnv,
    model: GenericActorCritic | LuxActorCritic,
    config: Config,
    key: jax.Array,  # () typed PRNG key
    opponent_pool: OpponentPool | None = None,
) -> RolloutCarry:
    """Return fresh (E, ...) environments and (E*N, ...) memory on the training mesh."""
    batched = PartitionSpec("batch")
    reset_keys = jax.random.split(key, config.num_envs)  # (E,)
    reset = jax.jit(jax.vmap(env.reset), in_shardings=batched, out_shardings=batched)
    observation, env_state = reset(jax.device_put(reset_keys, batched))  # leaves: (E, N, ...); (E, ...)
    assignment = None
    if opponent_pool is not None:
        assignment = sample_opponents(
            jax.random.fold_in(key, config.num_envs),
            opponent_pool,
            cast(SelfPlayConfig, config.selfplay),
            config.num_envs,
        )  # leaves: (E,)
    actor_count = config.num_envs * env.num_agents
    carry = RolloutCarry(
        env_state=env_state,  # leaves: (E, ...)
        observation=observation,  # observation: (E, N, ...); masks: (E, N, *C, A)
        model_carry=model.initial_carry(actor_count),  # leaves: (E*N, ...)
        episode_start=jnp.ones(actor_count, dtype=bool),  # (E*N,)
        episode_return=jnp.zeros(config.num_envs, dtype=jnp.float32),  # (E,)
        episode_step_count=jnp.zeros(config.num_envs, dtype=jnp.int32),  # (E,)
        selfplay=assignment,  # leaves: (E,)
    )
    return jax.device_put(carry, batched)  # leaves: (E, ...) or (E*N, ...)


def train(config: Config) -> TrainState:
    """Train with a fully resolved config, as returned by load_config."""
    configure_compilation_cache()
    active_config = config_for_iteration(config, 0)
    env = make_env(active_config)
    with ExitStack() as resources, jax.set_mesh(training_mesh(config, env.num_agents)):
        replicated = PartitionSpec()
        batched = PartitionSpec("batch")
        batch_size = config.num_envs * config.num_steps
        writer = MLflowLogger(experiment="mappo", run_name=config.run_id)
        resources.enter_context(writer)
        manager = resources.enter_context(closing(checkpoint_manager(config.ckpt_dir)))
        key, init_key, reset_key = jax.random.split(jax.random.key(config.seed), 3)  # each () typed PRNG key
        model = make_actor_critic(
            action_space=env.action_space,
            model=config.model,
            dtype=jnp.bfloat16 if config.bf16 else jnp.float32,
        )
        rollout_carry = init_rollout_carry(env, model, active_config, reset_key)  # leaves: (E, ...) or (B, ...)
        observation = rollout_carry.observation  # leaves: (E, N, ...)
        lr_schedule = learning_rate_schedule(config)
        entropy_schedule = entropy_coef_schedule(config)
        optimizer = optax.inject_hyperparams(
            lambda learning_rate: optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(learning_rate, eps=1e-5),
            )
        )
        train_state = TrainState.create(
            apply_fn=model.apply,
            params=model.init(
                init_key,
                jax.tree.map(lambda leaf: leaf[:1, :1], observation.observation),  # leaves: (B=1, T=1, ...)
                model.initial_carry(1),  # leaves: (B=1, ...)
                rollout_carry.episode_start[:1, None],  # (B=1, T=1)
                observation.actions_mask[:1, :1],  # (B=1, T=1, *C, A)
            )["params"],
            tx=optimizer(config.learning_rate),
        )
        train_state = jax.device_put(train_state, replicated)
        progress = restore_checkpoint(manager, train_state, config)
        metrics = init_metrics(progress)  # scalar leaves
        start_rollout_index = 0
        if progress is not None:
            train_state, key, start_rollout_index = (
                progress.train_state,
                progress.key,
                progress.rollout_count,
            )  # key: ()
            # Live environment states are not serialized. Start fresh episodes
            # with fresh recurrent memory, retaining the policy and training RNGs.
            resumed_config = config_for_iteration(config, start_rollout_index)
            if resumed_config.env != active_config.env:
                active_config = resumed_config
                env = make_env(active_config)
            rollout_carry = init_rollout_carry(
                env, model, active_config, jax.random.fold_in(key, start_rollout_index)
            )  # leaves: (E, ...) or (B, ...)
        train_state = jax.device_put(train_state, replicated)
        eval_references = load_eval_ref_policies(config, train_state.params)  # leaves: parameter shapes
        key, metrics = jax.device_put((key, metrics), replicated)  # key: (); metrics: scalar leaves
        opponent_pool = None
        assignment = None
        if config.selfplay is not None:
            opponent_pool = (
                progress.opponent_pool
                if progress is not None and progress.opponent_pool is not None
                else init_pool(train_state.params, config.selfplay.pool_size, start_rollout_index)
            )
            opponent_pool = jax.device_put(opponent_pool, replicated)
            key, assignment_key = jax.random.split(key)  # each ()
            assignment = sample_opponents(assignment_key, opponent_pool, config.selfplay, config.num_envs)
        initial_step_count = (
            progress.environment_step_count
            if progress is not None and progress.environment_step_count is not None
            else start_rollout_index * batch_size
        )
        if progress is not None:
            print(f"Resumed run {config.run_id} at step {initial_step_count}", flush=True)
        writer.log_text(yaml.safe_dump(asdict(config)), f"config/{initial_step_count}.yaml")
        writer.log_params(asdict(config))
        print(f"Run ID: {config.run_id}\nMLflow run: {writer.run_id}", flush=True)
        devices = str(jax.devices())
        print(f"JAX devices: {devices}", flush=True)
        writer.log_text(devices, f"devices/{initial_step_count}.txt")
        parameter_count = sum(parameter.size for parameter in jax.tree.leaves(train_state.params))
        writer.log_metrics({"model/params_millions": parameter_count / 1_000_000}, initial_step_count)
        print(f"Model parameters: {parameter_count:,}", flush=True)
        writer.log_text(str(parameter_count), f"model/parameter_count/{initial_step_count}.txt")
        rollout_carry = rollout_carry._replace(selfplay=assignment)  # assignment leaves: (E,)
        rollout_carry = jax.device_put(rollout_carry, batched)  # state: (E, ...); model_carry: (B, ...)

        window_start = monotonic()
        window_start_step_count = initial_step_count
        step_count = initial_step_count
        for rollout_index in range(start_rollout_index, config.total_iterations):
            next_config = config_for_iteration(active_config, rollout_index)
            if next_config.env != active_config.env:
                active_config = next_config
                env = make_env(active_config)
                key, reset_key = jax.random.split(key)  # each ()
                rollout_carry = init_rollout_carry(
                    env, model, active_config, reset_key, opponent_pool
                )  # leaves: (E, ...) or (B, ...)
                print(
                    f"Map size changed to {cast(LuxConfig | JaxNavConfig, active_config.env).map_size} "
                    f"after {rollout_index} iterations",
                    flush=True,
                )
            # Schedules need a fraction, while exact lifetime counters stay as host integers.
            schedule_step = jnp.asarray(rollout_index, dtype=jnp.float32)  # (); avoids JIT's default int32 limit
            if opponent_pool is None:
                result = train_step_donated(rollout_carry, train_state, env, active_config, key, metrics, schedule_step)
            else:
                result = train_step_with_pool_donated(
                    rollout_carry, train_state, env, active_config, key, metrics, schedule_step, opponent_pool
                )
            train_state, rollout_carry, key, metrics = result  # state: (E, ...); model_carry: (B, ...); key/metrics: ()
            rollout_count = rollout_index + 1
            if config.selfplay is not None and rollout_count % config.selfplay.snapshot_interval == 0:
                opponent_pool = insert_snapshot(
                    cast(OpponentPool, opponent_pool),
                    train_state.params,
                    rollout_count,
                    cast(MatchAssignment, rollout_carry.selfplay).opponent,
                )
            is_final_rollout = rollout_count == config.total_iterations
            step_count += batch_size
            if config.video_interval and (rollout_count % config.video_interval == 0 or is_final_rollout):
                log_video(
                    train_state,
                    active_config,
                    writer,
                    int(metrics.total.episode_count),
                    step_count,
                    initial_carry=model.initial_carry(env.num_agents),
                    env=env,
                )
            if (
                config.eval_interval
                and (rollout_count % config.eval_interval == 0 or is_final_rollout)
                and eval_references
            ):
                log_evaluation(
                    train_state,
                    active_config,
                    writer,
                    int(metrics.total.episode_count),
                    step_count,
                    initial_carry=model.initial_carry(env.num_agents),
                    env=env,
                    opponents=eval_references,
                )
            if rollout_count % config.log_interval == 0 or is_final_rollout:
                # Host logging requires the completed device metrics.
                window, total = jax.device_get(metrics)  # scalar leaves
                now = monotonic()
                window_seconds = now - window_start
                mean_metrics = {
                    field.name: float(getattr(window.minibatch_metrics, field.name)) / int(window.minibatch_count)
                    for field in fields(MinibatchMetrics)
                }
                window_rollout_count = int(window.rollout_count)
                window_episode_count = int(window.episodes.episode_count)
                episode_count = int(total.episode_count)
                learning_rate = float(lr_schedule(schedule_step))
                entropy_coef = float(entropy_schedule(schedule_step))
                rate_seconds = max(window_seconds, 1e-9)
                sps = (step_count - window_start_step_count) / rate_seconds
                scalars: dict[str, float | int] = {
                    "train/updates_per_iteration": int(window.update_count) / window_rollout_count,
                    "train/policy_loss": mean_metrics["policy_loss"],
                    "train/value_loss": mean_metrics["value_loss"],
                    "train/total_loss": mean_metrics["total_loss"],
                    "train/entropy": mean_metrics["entropy"],
                    "train/approx_kl": mean_metrics["approx_kl"],
                    "train/ppo_clip_fraction": mean_metrics["clip_fraction"],
                    "train/iterations_per_second": window_rollout_count / rate_seconds,
                    "train/env_steps_per_second": sps,
                    "train/episodes_per_second": window_episode_count / rate_seconds,
                    "train/learning_rate": learning_rate,
                    "train/entropy_coef": entropy_coef,
                    "train/episodes_total": episode_count,
                    "train/env_steps_total": step_count,
                    "train/iterations_total": rollout_count,
                    "train/progress_fraction": min(rollout_count / config.total_iterations, 1.0),
                    "rollout/episode_count": window_episode_count,
                }
                if isinstance(active_config.env, (LuxConfig, JaxNavConfig)):
                    scalars["train/map_size"] = active_config.env.map_size
                if opponent_pool is not None:
                    pool_iterations = jax.device_get(opponent_pool.iteration)  # (K,)
                    populated_iterations = pool_iterations[pool_iterations >= 0]  # (populated_slots,)
                    scalars["train/pool_param_iteration_min"] = int(populated_iterations.min())
                    scalars["train/pool_param_iteration_max"] = int(populated_iterations.max())
                if config.target_kl is not None:
                    scalars["train/kl_early_stop_fraction"] = int(window.early_stop_count) / window_rollout_count
                if int(window.explained_variance_count):
                    scalars["train/explained_variance"] = float(window.explained_variance_sum) / int(
                        window.explained_variance_count
                    )
                if window_episode_count:
                    terminated_episode_count = int(window.terminated_episode_count)
                    scalars["rollout/episode_end_fraction/terminated"] = terminated_episode_count / window_episode_count
                    scalars["rollout/episode_end_fraction/truncated"] = (
                        window_episode_count - terminated_episode_count
                    ) / window_episode_count
                    scalars["rollout/episode_return"] = float(window.episodes.episode_return_sum) / window_episode_count
                    scalars["rollout/episode_length"] = int(window.episodes.episode_length_sum) / window_episode_count
                if episode_count:
                    scalars["train/episode_return_total"] = float(total.episode_return_sum) / episode_count
                    scalars["train/episode_length_total"] = int(total.episode_length_sum) / episode_count
                writer.log_metrics(scalars, step_count)
                episode_return_text = f"{scalars['rollout/episode_return']:.1f}" if window_episode_count else "n/a"
                episode_length_text = f"{scalars['rollout/episode_length']:.1f}" if window_episode_count else "n/a"
                explained_variance_text = (
                    f"{scalars['train/explained_variance']:.3f}" if "train/explained_variance" in scalars else "n/a"
                )
                print(
                    f"iteration={rollout_count} step={step_count} episodes={episode_count} episode_return={episode_return_text} "
                    f"episode_length={episode_length_text} "
                    f"progress={scalars['train/progress_fraction']:.3f} "
                    f"iterations/s={scalars['train/iterations_per_second']:.3f} "
                    f"env_steps/s={sps:.3f} episodes/s={scalars['train/episodes_per_second']:.3f} "
                    f"lr={learning_rate:.3g} "
                    f"policy={mean_metrics['policy_loss']:.3f} value={mean_metrics['value_loss']:.3f} "
                    f"entropy={mean_metrics['entropy']:.3f} kl={mean_metrics['approx_kl']:.4f} "
                    f"clipfrac={mean_metrics['clip_fraction']:.3f} ev={explained_variance_text} "
                    f"updates={scalars['train/updates_per_iteration']:.2f} "
                    f"window={window_seconds:.1f}s",
                    flush=True,
                )
                window_start = now
                window_start_step_count = step_count
                with jax.enable_x64():
                    metrics = metrics._replace(window=RunningSum.init(metrics.window))  # scalar leaves
            if rollout_count % config.checkpoint_interval == 0 or is_final_rollout:
                checkpoint_seconds = save_checkpoint(
                    manager,
                    TrainingProgress(
                        train_state=train_state,
                        key=key,
                        rollout_count=rollout_count,
                        episode_count=int(metrics.total.episode_count),
                        episode_return_sum=float(metrics.total.episode_return_sum),
                        episode_length_sum=int(metrics.total.episode_length_sum),
                        environment_step_count=step_count,
                        opponent_pool=opponent_pool,
                    ),
                    config,
                )
                writer.log_metrics({"checkpoint/save_seconds": checkpoint_seconds}, step_count)
        return train_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mappo/smax.yaml", help="Path to a YAML config")
    args = parser.parse_args()
    train(load_config(args.config, Config))


if __name__ == "__main__":
    main()
