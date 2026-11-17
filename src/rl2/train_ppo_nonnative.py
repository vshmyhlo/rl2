"""Recurrent PPO for host-stepped Gymnasium and VisFly environments.

B denotes environments, T rollout steps, O observation leaf dimensions, A discrete
actions, D continuous action size, and H recurrent width. Sequence arrays use
(B, T, ...). Minibatches preserve entire environment sequences. Episode ends reset
memory and stop GAE traces; timeouts bootstrap from final observations unless the
episode also terminated.

The default value_coef=1.0 weights the critic by 0.5 * mean squared value error.

Training uses all devices on JAX's default backend (all visible GPUs when available).
Parameters and optimizer state are replicated; PPO minibatches are sharded along
B, keeping each recurrent sequence intact. Environments execute on the host;
policy inference and optimization are compiled separately. num_envs is global and
must be divisible by num_minibatches * device_count. Losses, advantage normalization,
gradient clipping, and KL stopping use global minibatch reductions. Advantages
are standardized per minibatch, except single transitions retain their raw advantage.
Training enters a jax.set_mesh context; standalone updates and train steps require
an active mesh with a batch axis, including on CPU.

total_steps and MLflow steps count environment transitions across all environments.
An iteration collects one batch of num_envs * num_steps transitions and runs PPO
epochs. An update applies gradients from one accepted minibatch; KL-rejected
minibatches do not count as updates.
KL-rejected minibatches skip backpropagation and optimizer work.
Window/lifetime metrics and latest settings are logged every log_interval iterations
(default: 10). Checkpoints are saved every checkpoint_interval iterations (default: 100).
Periodic operations use completed iteration counts, including resumed training,
and also run on normal completion. Learning-rate and entropy schedules use completed
environment steps divided by the current batch size, rounded down. No metrics have
categorical suffixes. Only complete batches fitting within total_steps are collected.

Window metrics:
    train/policy_loss, train/value_loss: Mean clipped surrogate loss and half
        squared value error over minibatch transitions, then evaluated minibatches.
    train/entropy_loss, train/entropy: Negative entropy weighted by the iteration's
        entropy coefficient, and unweighted entropy in nats, respectively.
    train/total_loss: Policy loss + value_coef * value loss + entropy loss.
    train/approx_kl: Mean exp(log_ratio) - 1 - log_ratio, in nats versus rollout policy.
    train/clip_fraction: Fraction of action probability ratios outside the clip range.
        All losses/diagnostics average transitions, then evaluated minibatches,
        including KL-rejected minibatches; total loss uses each iteration's entropy weight.
    train/explained_variance: Mean over iterations with nonconstant returns;
        omitted if none are defined.
    train/updates_per_iteration, train/kl_early_stop_fraction: Accepted updates
        and KL stop frequency, averaged over iterations; omit stops when disabled.
    rollout/episode_count, rollout/return, rollout/episode_length_steps:
        Completed episodes and their raw return/length means; omit unavailable means.
    train/iterations_per_second, train/env_steps_per_second, train/updates_per_second:
        Completed batches, environment transitions across all environments, and
        accepted minibatch updates divided by window wall seconds, respectively.
        Includes collection, optimization, and intervening logging, evaluation,
        playback, and checkpoints. Zero accepted updates yields zero updates/second.

Lifetime metrics:
    train/iterations_total, train/updates_total, train/episodes_total, train/env_steps_total:
        Completed iteration, accepted update, episode, and environment transition counts.
    train/return, train/episode_length_steps: Raw return and length means
        over all completed episodes; omitted before the first completion.
    train/progress_fraction: Fraction of the environment-step budget completed.
        Lifetime counters and episode sums include restored checkpoint history.

Latest settings and session timings:
    train/learning_rate, train/entropy_coef: Latest iteration's scheduled settings.
    actor/video_seconds, eval/evaluation_seconds: Wall seconds for gameplay through
        video encoding/upload, or evaluation through gameplay/report uploads.
        Logged per operation at video_interval or eval_interval iterations;
        omitted when the operation does not run. Excludes logging the timing itself.
    checkpoint/save_seconds: Blocking wall seconds to prepare and submit an async
        checkpoint, including any wait for a previous save inside manager.save.
        Excludes background writes and the final drain at manager close. Logged per
        save at checkpoint_interval iterations; omitted when no save runs.
    train/metrics_seconds: Wall seconds from collecting window metrics on the host
        through scalar reduction, metric logging, progress printing, and window reset.
        Includes the required device-to-host wait; adds no timing-only synchronization.
        Logged after each log_interval window (also the final window), at the same
        environment step; excludes logging this timing itself. No window means omitted.
    model/params_millions: Parameter count at startup, including resume.

Evaluation uses fixed seeds and logs raw episode return mean, sample standard
deviation, and standard error as eval/return, eval/return_std, and
eval/return_sem after each evaluation. eval/training_episodes_total is the lifetime
completed training episode count at evaluation. Uncertainty metrics are omitted
for a single episode. With eval_video=true, evaluation replays its highest-return episode with the same
seed for recording, choosing the first on ties, independently of video_interval.
The replay is excluded from score statistics and included in eval/evaluation_seconds.
Standalone gameplay is disabled by default (video_interval=0); a positive
video_interval enables recording at that iteration interval and on normal completion.
video_max_frames caps recording (default: 1000). Standalone gameplay and evaluation
replays stop at that cap; scored evaluation episodes always run to completion.
Gameplay is saved as looping GIFs at video_fps, with frame boundaries rounded to
10 ms and a 10 ms minimum frame delay. Evaluation and playback create fresh,
single-environment instances seeded independently of training. Frames are rendered
before actions; autoreset frames are excluded. video_max_frames=None records the
full episode. VisFly requires its simulator dependencies, scene assets, and RGB
render settings for video; scoring alone does not require RGB rendering.

Artifacts: config/<step>.yaml, devices/<step>.txt, logs/<session>.log,
model/parameter_count/<step>.txt, eval/report/<step>.json, gameplay/<step>.gif,
and eval/gameplay/<step>.gif. Config parameters use dotted keys; existing values
are preserved on resume. Startup console logs include per-environment observation
shapes/dtypes and the action space (discrete count or continuous shape/bounds),
including on resume. Checkpoints retain parameters, optimizer state, RNG,
completed iterations, environment steps, and episode totals. Legacy checkpoints
without an environment-step count infer it using the current batch size, so their
historical count is exact only when that size is unchanged. Saved configuration is
ignored on restore. Live environments and recurrent memory restart on resume;
unfinished episode statistics are discarded.
"""

import argparse
import json
import os
from collections.abc import Callable
from contextlib import ExitStack, closing
from dataclasses import asdict, fields, replace
from functools import partial
from math import sqrt
from operator import itemgetter
from pathlib import Path
from statistics import fmean, stdev
from time import monotonic
from typing import Annotated, Literal, NamedTuple, Self

# Orbax's epath otherwise loads TensorFlow for gs:// paths when it is installed.
# That native runtime can crash a later Triton import; use our gcsfs dependency.
# Set this before importing Orbax, since epath caches its backend selection.
os.environ.setdefault("EPATH_USE_TF", "0")

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import rlax
import yaml
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from jax.sharding import AxisType, Mesh, PartitionSpec
from numpy.typing import NDArray
from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass

from rl2.actions import ActionSpace, BetaPolicy, CategoricalPolicy, DiscreteSpace, Policy, TruncatedGaussianPolicy
from rl2.configuration import BaseConfig, load_config
from rl2.envs.gym_env import EnvConfig, GymEnv, VisFlyConfig
from rl2.envs.nonnative import BaseNonnativeEnv, Observation
from rl2.jax_cache import configure_compilation_cache
from rl2.lstm import LSTMStack, LSTMStackCarry
from rl2.mlflow_logging import MLflowLogger
from rl2.model.encoders.conv2d import DEFAULT_STAGES, ConvObservationEncoder, ConvStages
from rl2.model.encoders.mlp import MLPEncoder
from rl2.running_stats import RunningSum
from rl2.utils import explained_variance, gae

type RecurrentCarry = LSTMStackCarry
type PPOEnv = BaseNonnativeEnv


@struct.dataclass
class PPOBatch:
    """PPO inputs; S is (B, T) for sequences or (B,) for step-only policies.

    Attributes:
        observations: (*S, *O) leaves with observation dimensions O.
        actions: (*S,) integer IDs or (*S, *action_shape) bounded continuous actions.
        old_log_probs: (*S,) rollout action log probabilities.
        values: (*S,) rollout-time value predictions for explained variance.
        advantages: (*S,) unnormalized advantage estimates.
        returns: (*S,) value targets.
        model_carry: (B, H) initial LSTM memory leaves, where H is model.recurrent.hidden_size.
        episode_starts: (*S,) boolean episode reset flags.
    """

    observations: Observation  # leaves: (*S, *O)
    actions: jax.Array  # (*S,) integer IDs or (*S, *action_shape) bounded continuous actions
    old_log_probs: jax.Array  # (*S,)
    values: jax.Array  # (*S,)
    advantages: jax.Array  # (*S,)
    returns: jax.Array  # (*S,)
    model_carry: RecurrentCarry  # leaves: (B, H), H is model.recurrent.hidden_size
    episode_starts: jax.Array  # (*S,)


class RolloutCarry(NamedTuple):
    """Live environment state and recurrent memory between rollouts.

    Attributes:
        observations: (B, *O) current observation leaves with observation dimensions O.
        model_carry: (B, H) LSTM memory leaves, where H is model.recurrent.hidden_size.
        episode_starts: (B,) boolean episode reset flags.
        episode_returns: (B,) accumulated raw rewards for unfinished episodes.
        episode_step_counts: (B,) step counts for unfinished episodes.
    """

    observations: Observation  # leaves: (B, *O)
    model_carry: RecurrentCarry  # leaves: (B, H)
    episode_starts: jax.Array  # (B,)
    episode_returns: jax.Array  # (B,)
    episode_step_counts: jax.Array  # (B,)


class RolloutResult(NamedTuple):
    """Collected training data, advanced RNG key, and completed episode statistics.

    Attributes:
        batch: (B, T, ...) batch-major PPO arrays with (B, H) initial memory leaves.
        key: () advanced typed PRNG key, also used for minibatch shuffling.
        episode_return_sum: () sum of raw returns for completed episodes.
        episode_length_sum: () sum of lengths for completed episodes.
        episode_count: () number of completed episodes.
    """

    batch: PPOBatch  # observations: (B, T, *O); targets/starts: (B, T); model_carry leaves: (B, H)
    key: jax.Array  # () typed PRNG key
    episode_return_sum: jax.Array  # ()
    episode_length_sum: jax.Array  # ()
    episode_count: jax.Array  # ()


@struct.dataclass
class MinibatchMetrics:
    """Scalar diagnostics for one evaluated minibatch, including KL-rejected updates."""

    policy_loss: jax.Array  # ()
    value_loss: jax.Array  # ()
    entropy_loss: jax.Array  # () weighted by the rollout's entropy coefficient
    entropy: jax.Array  # ()
    approx_kl: jax.Array  # ()
    clip_fraction: jax.Array  # ()


class EpisodeMetrics(NamedTuple):
    """Completed-episode statistics, reduced from (B,) leaves to () aggregates."""

    episode_count: jax.Array  # () or (B,) before reduction
    episode_return_sum: jax.Array  # () or (B,) before reduction
    episode_length_sum: jax.Array  # () or (B,) before reduction


class WindowMetrics(NamedTuple):
    episodes: EpisodeMetrics  # scalar leaves
    minibatch_metrics: MinibatchMetrics  # scalar leaves, including KL-rejected minibatches
    explained_variance_sum: jax.Array  # () summed over rollouts with defined variance
    explained_variance_count: jax.Array  # () rollouts with defined variance
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
    rollout_carry: RolloutCarry  # observations: (B, *O); memory: (B, H)
    key: jax.Array  # () typed PRNG key
    metrics: TrainingMetrics  # scalar leaves


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LSTMConfig:
    type: Literal["lstm"] = "lstm"
    hidden_size: Annotated[int, Field(gt=0, strict=True)] = 1536
    num_layers: Annotated[int, Field(gt=0, strict=True)] = 1
    intermediate_size: Annotated[int, Field(ge=0, strict=True)] = 0


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class Conv2DEncoderConfig:
    """2D convolutions over channels-last symbolic grids or RGB images."""

    type: Literal["conv2d"] = "conv2d"
    stages: Annotated[ConvStages, Field(min_length=1)] = DEFAULT_STAGES
    embedding_size: Annotated[int, Field(gt=0, strict=True)] = 768


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class MLPEncoderConfig:
    type: Literal["mlp"] = "mlp"
    hidden_size: Annotated[int, Field(gt=0, strict=True)] = 256
    embedding_size: Annotated[int, Field(gt=0, strict=True)] = 256


type EncoderConfig = Annotated[Conv2DEncoderConfig | MLPEncoderConfig, Field(discriminator="type")]


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class ModelConfig:
    encoder: EncoderConfig
    recurrent: LSTMConfig = LSTMConfig()
    continuous_policy: Literal["truncated_gaussian", "beta"] = "truncated_gaussian"


@dataclass(frozen=True)
class Config(BaseConfig):
    model: ModelConfig
    env: EnvConfig
    seed: int
    total_steps: int
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
    max_grad_norm: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    video_fps: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    value_coef: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 1.0
    bf16: bool = True
    video_max_frames: Annotated[int, Field(gt=0)] | None = 1000
    video_interval: Annotated[int, Field(ge=0)] = 0
    eval_interval: Annotated[int, Field(ge=0)] = 0
    eval_episodes: Annotated[int, Field(gt=0)] = 100
    eval_video: bool = False
    eval_seed: Annotated[int, Field(ge=0, lt=2**32)] = 10_000
    lr_decay: Literal["linear", "cosine"] = "linear"
    entropy_decay: Literal["constant", "cosine"] = "constant"
    checkpoint_interval: Annotated[int, Field(gt=0)] = 100
    log_interval: Annotated[int, Field(gt=0)] = 10

    @model_validator(mode="after")
    def validate_rollout(self) -> Self:
        if isinstance(self.env, VisFlyConfig) and self.num_envs % self.env.num_scenes:
            raise ValueError("num_envs must be divisible by env.num_scenes")
        if self.num_envs % self.num_minibatches:
            raise ValueError("num_envs must be divisible by num_minibatches to preserve sequences")
        if self.total_steps < self.num_envs * self.num_steps:
            raise ValueError("total_steps must cover at least one rollout")
        return self


def make_env(config: Config, *, seed: int | None = None, evaluation: bool = False, record: bool = False) -> PPOEnv:
    env_config = config.env
    if evaluation and isinstance(env_config, VisFlyConfig):
        env_config = replace(env_config, num_scenes=1)
    return GymEnv(
        config=env_config,
        num_envs=1 if evaluation else config.num_envs,
        seed=config.seed if seed is None else seed,
        record=record,
    )


class TrainingProgress(NamedTuple):
    train_state: TrainState
    key: jax.Array  # () typed PRNG key
    rollout_count: int  # Number of completed rollouts, independent of optimizer steps.
    episode_count: int
    episode_return_sum: float
    episode_length_sum: int
    env_steps: int


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
    manager.save(
        progress.rollout_count,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave({"train_state": progress.train_state, "key": key_data}),
            metadata=ocp.args.JsonSave(
                {
                    "version": 4,
                    "config": asdict(config),
                    "iteration": progress.rollout_count,
                    "completed_episodes": progress.episode_count,
                    "training_return_sum": progress.episode_return_sum,
                    "training_length_sum": progress.episode_length_sum,
                    "environment_steps": progress.env_steps,
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
    if metadata["version"] not in (1, 2, 3, 4):
        raise ValueError("Unsupported PPO checkpoint version")
    if metadata["iteration"] != rollout_count or rollout_count < 0 or metadata["completed_episodes"] < 0:
        raise ValueError("Invalid PPO checkpoint progress")
    for field in ("training_return_sum", "training_length_sum"):
        if metadata.get(field) is None:
            raise ValueError(f"PPO checkpoint requires {field}; use a new run directory")
    env_steps = (
        metadata["environment_steps"]
        if metadata["version"] >= 4
        else rollout_count * config.num_envs * config.num_steps
    )
    if env_steps < 0:
        raise ValueError("Invalid PPO checkpoint environment steps")
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
    # Version 1 also stored a NumPy shuffle RNG; all randomness now uses the saved JAX key.
    return TrainingProgress(
        train_state=restored["train_state"],
        key=jax.random.wrap_key_data(restored["key"]),
        rollout_count=rollout_count,
        episode_count=metadata["completed_episodes"],
        episode_return_sum=metadata["training_return_sum"],
        episode_length_sum=metadata["training_length_sum"],
        env_steps=env_steps,
    )


def learning_rate_schedule(config: Config) -> optax.Schedule:
    if not config.anneal_lr:
        return optax.constant_schedule(config.learning_rate)
    planned_rollout_count = config.total_steps // (config.num_envs * config.num_steps)
    if config.lr_decay == "cosine":
        return optax.cosine_decay_schedule(config.learning_rate, planned_rollout_count)
    return optax.linear_schedule(config.learning_rate, 0.0, planned_rollout_count)


def entropy_coef_schedule(config: Config) -> optax.Schedule:
    if config.entropy_decay == "constant":
        return optax.constant_schedule(config.entropy_coef)
    planned_rollout_count = config.total_steps // (config.num_envs * config.num_steps)
    return optax.cosine_decay_schedule(config.entropy_coef, planned_rollout_count)


class ActorCriticOutput(NamedTuple):
    carry: RecurrentCarry  # leaves: (B, H)
    policy: Policy  # parameter leaves: (*S, A) discrete or (*S, D) continuous
    values: jax.Array  # (*S,)


class ObservationEncoder(nn.Module):
    """Combine image and vector leaves from Box, Dict, or Tuple observations."""

    config: EncoderConfig
    dtype: jax.typing.DTypeLike | None = None

    @nn.compact
    def __call__(self, obs: Observation) -> jax.Array:  # leaves: (B, *O) -> (B, E)
        features = []  # each (B, F)
        for index, leaf in enumerate(jax.tree.leaves(obs)):
            if self.config.type == "conv2d" and leaf.ndim == 4:
                feature = ConvObservationEncoder(
                    channels_last=True,
                    stages=self.config.stages,
                    embedding_size=self.config.embedding_size,
                    dtype=self.dtype,
                    name=f"image_{index}",
                )(leaf)  # (B, E)
            else:
                feature = leaf.reshape((leaf.shape[0], -1)).astype(jnp.float32)  # (B, F)
                if leaf.dtype == jnp.uint8:
                    feature = feature / 255.0  # (B, F)
            features.append(feature)
        combined = jnp.concatenate(features, axis=-1)  # (B, sum(F))
        return MLPEncoder(
            hidden_size=self.config.hidden_size if self.config.type == "mlp" else self.config.embedding_size,
            embedding_size=self.config.embedding_size,
            dtype=self.dtype,
        )(combined)  # (B, E)


class ActorCritic(nn.Module):
    """Shared observation encoder/LSTM with separate policy and value heads."""

    action_space: ActionSpace
    model: ModelConfig
    dtype: jax.typing.DTypeLike | None = None

    @nn.nowrap
    def _make_recurrent(self, *, parent: nn.Module | None = None) -> LSTMStack:
        return LSTMStack(
            features=self.model.recurrent.hidden_size,
            num_layers=self.model.recurrent.num_layers,
            intermediate_size=self.model.recurrent.intermediate_size,
            dtype=self.dtype,
            name="lstm",
            parent=parent,
        )

    @nn.nowrap
    def initial_carry(self, num_envs: int) -> RecurrentCarry:
        return self._make_recurrent().initial_carry(num_envs)  # leaves: (B, H)

    def setup(self) -> None:
        self.encoder = ObservationEncoder(config=self.model.encoder, dtype=self.dtype, name="encoder")
        self.recurrent = self._make_recurrent(parent=self)
        self.recurrent_input = nn.Dense(features=self.model.recurrent.hidden_size, dtype=self.dtype, name="lstm_input")
        init = nn.initializers.orthogonal(sqrt(2))
        self.policy_hidden = nn.Dense(features=512, kernel_init=init, name="policy_hidden", dtype=self.dtype)
        self.policy_norm = nn.LayerNorm(name="policy_norm", dtype=self.dtype)
        self.value_hidden = nn.Dense(features=512, kernel_init=init, name="value_hidden", dtype=self.dtype)
        self.value_norm = nn.LayerNorm(name="value_norm", dtype=self.dtype)
        policy_size = self.action_space.n if isinstance(self.action_space, DiscreteSpace) else self.action_space.size
        if not isinstance(self.action_space, DiscreteSpace):
            if self.model.continuous_policy == "beta":
                policy_size *= 2
            else:
                self.policy_log_std = self.param("policy_log_std", nn.initializers.zeros, (policy_size,))  # (D,)
        self.policy_output = nn.Dense(
            features=policy_size, kernel_init=nn.initializers.orthogonal(0.01), name="policy_output", dtype=self.dtype
        )
        self.value_output = nn.Dense(
            features=1, kernel_init=nn.initializers.orthogonal(1.0), name="value_output", dtype=self.dtype
        )

    def _encode(self, obs: Observation) -> jax.Array:  # leaves: (B, *O) -> (B, H)
        x = self.encoder(obs)  # (B, F), F is model.encoder.embedding_size
        return jnp.asarray(self.recurrent_input(x), dtype=self.dtype)  # (B, H)

    def _heads(self, x: jax.Array) -> tuple[Policy, jax.Array]:  # (*S, H) -> policy leaves: (*S, A/D); (*S,)
        x = jnp.asarray(x, dtype=self.dtype)  # (*S, H), S is (B,) or (B, T)
        policy = nn.relu(self.policy_norm(self.policy_hidden(x)))  # (*S, 512)
        critic = nn.relu(self.value_norm(self.value_hidden(x)))  # (*S, 512)
        # Heads use the compute dtype; float32 outputs keep PPO loss arithmetic precise.
        output = self.policy_output(policy).astype(jnp.float32)  # (*S, A/D/2D)
        distribution: Policy
        if isinstance(self.action_space, DiscreteSpace):
            distribution = CategoricalPolicy(output)
        elif self.model.continuous_policy == "beta":
            # Concentrations >= 1 avoid singular densities at the action bounds.
            alpha, beta = jnp.split(jax.nn.softplus(output) + 1.0, 2, axis=-1)  # each (*S, D)
            distribution = BetaPolicy(alpha=alpha, beta=beta, action_space=self.action_space)
        else:
            low, high = jnp.asarray((self.action_space.low, self.action_space.high), jnp.float32)  # each (D,)
            scale = high / 2 - low / 2  # (D,)
            # Keep the Gaussian center inside the bounds to avoid tail-only truncation.
            mean = (high / 2 + low / 2) + scale * jnp.tanh(output)  # (*S, D)
            log_std = jnp.broadcast_to(
                jnp.clip(self.policy_log_std, -20.0, 2.0) + jnp.log(scale), output.shape
            )  # (*S, D)
            distribution = TruncatedGaussianPolicy(mean=mean, log_std=log_std, action_space=self.action_space)
        values = self.value_output(critic).squeeze(-1).astype(jnp.float32)  # (*S,)
        return distribution, values  # policy leaves: (*S, A/D); (*S,)

    def __call__(
        self,
        obs: Observation,  # leaves: (B, T, *O), O is encoder-specific
        model_carry: RecurrentCarry,  # leaves: (B, H)
        episode_starts: jax.Array,  # (B, T)
    ) -> ActorCriticOutput:  # carry leaves: (B, H); policy leaves: (B, T, A/D); values: (B, T)
        """Process observation sequences, resetting memory at episode starts."""
        env_count, step_count = episode_starts.shape
        x = self._encode(jax.tree.map(lambda x: x.reshape((-1, *x.shape[2:])), obs))  # (B * T, H)
        x = x.reshape((env_count, step_count, -1))  # (B, T, H)
        model_carry, x = self.recurrent(x, model_carry, episode_starts)  # (B, H) leaves; (B, T, H)
        policy, values = self._heads(x)  # policy leaves: (B, T, A/D); (B, T)
        return ActorCriticOutput(carry=model_carry, policy=policy, values=values)

    def step(
        self,
        obs: Observation,  # leaves: (B, *O), O is encoder-specific
        model_carry: RecurrentCarry,  # leaves: (B, H)
        episode_starts: jax.Array,  # (B,)
    ) -> ActorCriticOutput:  # carry leaves: (B, H); policy leaves: (B, A/D); values: (B,)
        """Process one observation per sequence, resetting memory at episode starts."""
        x = self._encode(obs)  # (B, H)
        model_carry, x = self.recurrent.step(x, model_carry, episode_starts)  # model_carry leaves: (B, H); x: (B, H)
        policy, values = self._heads(x)  # policy leaves: (B, A/D); (B,)
        return ActorCriticOutput(carry=model_carry, policy=policy, values=values)


@partial(jax.jit, static_argnames=("apply_fn",))
def policy_step(
    apply_fn: Callable[..., ActorCriticOutput],
    params: optax.Params,
    observations: Observation,  # leaves: (B, *O)
    memory: RecurrentCarry,  # leaves: (B, H)
    starts: jax.Array,  # (B,)
) -> ActorCriticOutput:
    return apply_fn({"params": params}, observations, memory, starts, method="step")


@jax.jit
def sample_actions(policy: Policy, key: jax.Array) -> tuple[jax.Array, jax.Array]:  # key: ()
    return policy.sample_and_log_prob(key=key)  # (B, *action_shape); (B,)


class EpisodeResult(NamedTuple):
    episode_return: float
    length: int
    frames: list[NDArray[np.uint8]]  # each (H, W, 3)


def play_episode(
    train_state: TrainState,
    config: Config,
    key: jax.Array,  # () typed PRNG key
    initial_carry: RecurrentCarry,  # leaves: (1, H)
    *,
    record: bool,
    stop_at_frame_limit: bool = False,
) -> EpisodeResult:
    """Run a fresh host environment; record frames before actions, excluding autoresets."""
    seed = int(jax.random.bits(key, dtype=jnp.uint32))
    frames = []  # each (H, W, 3)
    episode_return = 0.0
    length = 0
    memory = initial_carry  # leaves: (1, H)
    starts = jnp.ones(1, dtype=bool)  # (1,)
    with closing(make_env(config, seed=seed, evaluation=True, record=record)) as env:
        observations = env.reset()  # leaves: (1, *O)
        while True:
            if record and (config.video_max_frames is None or len(frames) < config.video_max_frames):
                frames.append(env.render())
            if (
                record
                and stop_at_frame_limit
                and config.video_max_frames is not None
                and len(frames) >= config.video_max_frames
            ):
                break
            key, action_key = jax.random.split(key)  # each ()
            memory, policy, _ = policy_step(train_state.apply_fn, train_state.params, observations, memory, starts)
            actions, _ = sample_actions(policy, action_key)  # (1, *action_shape); (1,)
            transition = env.step(actions)
            observations = transition.observations  # leaves: (1, *O)
            episode_return += float(transition.rewards[0])
            length += 1
            if bool((transition.terminated | transition.truncated)[0]):
                break
            starts = jnp.zeros(1, dtype=bool)  # (1,)
    return EpisodeResult(episode_return=episode_return, length=length, frames=frames)


def log_video(
    train_state: TrainState,
    config: Config,
    writer: MLflowLogger,
    episode_count: int,
    step_count: int,
    *,
    initial_carry: RecurrentCarry,  # leaves: (1, H)
) -> None:
    """Record a separate episode up to the frame cap and log its gameplay GIF."""
    print(f"Recording gameplay at step {step_count} after {episode_count} training episodes...", flush=True)
    started = monotonic()
    _, _, frames = play_episode(
        train_state,
        config,
        jax.random.fold_in(jax.random.key(config.seed), episode_count),
        initial_carry,
        record=True,
        stop_at_frame_limit=True,
    )
    writer.log_video(frames, f"gameplay/{step_count}.gif", fps=config.video_fps)
    seconds = monotonic() - started
    writer.log_metrics({"actor/video_seconds": seconds}, step_count)
    print(
        f"Recorded {len(frames)} gameplay frames after {episode_count} training episodes in {seconds:.1f}s", flush=True
    )


def log_evaluation(
    train_state: TrainState,
    config: Config,
    writer: MLflowLogger,
    episode_count: int,
    step_count: int,
    *,
    initial_carry: RecurrentCarry,  # leaves: (1, H)
) -> None:
    """Evaluate fixed seeds, log raw scores, and optionally replay the best episode."""
    print(f"Evaluating {config.eval_episodes} games at step {step_count}...", flush=True)
    started = monotonic()
    episode_results: list[dict[str, float | int]] = []
    for episode_index in range(config.eval_episodes):
        episode_return, episode_step_count, _ = play_episode(
            train_state,
            config,
            jax.random.fold_in(jax.random.key(config.eval_seed), episode_index),
            initial_carry,
            record=False,
        )
        episode_results.append({"return": episode_return, "length": episode_step_count})
    returns = [result["return"] for result in episode_results]
    best_episode_index = max(range(len(returns)), key=returns.__getitem__)
    if config.eval_video:
        _, _, frames = play_episode(  # frames: each (H, W, 3)
            train_state,
            config,
            jax.random.fold_in(jax.random.key(config.eval_seed), best_episode_index),
            initial_carry,
            record=True,
            stop_at_frame_limit=True,
        )
        if frames:
            writer.log_video(frames, f"eval/gameplay/{step_count}.gif", fps=config.video_fps)
    std = stdev(returns) if len(returns) > 1 else None
    result = {
        "env_id": config.env.id,
        "return_mean": fmean(returns),
        "return_std": std,
        "return_sem": std / sqrt(config.eval_episodes) if std is not None else None,
        "episodes": episode_results,
        "eval_seed": config.eval_seed,
        "training_config": asdict(config),
        "training_steps": step_count,
        "training_episodes": episode_count,
    }
    scalars = {"eval/return": result["return_mean"], "eval/training_episodes_total": episode_count}
    if std is not None:
        scalars.update({"eval/return_std": std, "eval/return_sem": result["return_sem"]})
    writer.log_text(json.dumps(result, indent=2, allow_nan=False), f"eval/report/{step_count}.json")
    seconds = monotonic() - started
    scalars["eval/evaluation_seconds"] = seconds
    writer.log_metrics(scalars, step_count)
    print(
        f"Evaluation: return={result['return_mean']:.1f} over {config.eval_episodes} games in {seconds:.1f}s",
        flush=True,
    )


def update(
    train_state: TrainState,
    batch: PPOBatch,  # observations: (*S, *O); targets/starts: (*S,); model_carry leaves: (B, H)
    config: Config,
    schedule_index: int | jax.Array = 0,  # () completed environment steps // current batch size, for schedules
) -> tuple[TrainState, MinibatchMetrics]:
    """Apply a global-minibatch update inside the active batch mesh."""
    schedule_index = jnp.asarray(schedule_index)  # ()
    chex.assert_type(schedule_index, int)
    entropy_coef = jnp.asarray(entropy_coef_schedule(config)(schedule_index), dtype=jnp.float32)  # ()
    advantages = batch.advantages  # (*S,)
    chex.assert_scalar_positive(advantages.size)
    # Centering a single sample always gives zero and erases its policy gradient.
    if advantages.size > 1:
        # Equal fractional values can round to a different mean in float32.
        tied = jnp.all(advantages == advantages.reshape(-1)[0])  # ()
        advantages = jnp.where(tied, 0.0, (advantages - advantages.mean()) / (advantages.std() + 1e-8))  # (*S,)

    def loss_fn(params: optax.Params) -> tuple[jax.Array, MinibatchMetrics]:  # (); scalar metric leaves
        _, policy, values = train_state.apply_fn(
            {"params": params}, batch.observations, batch.model_carry, batch.episode_starts
        )  # model_carry leaves: (B, H); policy leaves: (*S, A/D); values: (*S,)
        log_probs = policy.log_prob(batch.actions)  # (*S,)
        log_ratio = log_probs - batch.old_log_probs  # (*S,)
        ratio = jnp.exp(log_ratio)  # (*S,)
        flat_ratio, flat_advantages = jax.tree.map(jnp.ravel, (ratio, advantages))  # (N,), N = prod(S)
        policy_loss = rlax.clipped_surrogate_pg_loss(flat_ratio, flat_advantages, config.clip_coef)  # ()
        value_loss = rlax.l2_loss(values, batch.returns).mean()  # ()
        entropy = policy.entropy().mean()  # ()
        entropy_loss = -entropy_coef * entropy  # ()
        loss = policy_loss + config.value_coef * value_loss + entropy_loss  # ()
        approx_kl = (jnp.expm1(log_ratio) - log_ratio).mean()  # ()
        clip_fraction = (jnp.abs(ratio - 1) > config.clip_coef).mean()  # ()
        return loss, MinibatchMetrics(
            policy_loss=policy_loss,  # ()
            value_loss=value_loss,  # ()
            entropy_loss=entropy_loss,  # ()
            entropy=entropy,  # ()
            approx_kl=approx_kl,  # ()
            clip_fraction=clip_fraction,  # ()
        )  # loss: (); metrics: scalar leaves

    loss, backward, metrics = jax.vjp(loss_fn, train_state.params, has_aux=True)  # loss: (); metric leaves: ()

    def apply_gradients() -> TrainState:
        # Defer backpropagation until the KL check accepts the minibatch.
        (grads,) = backward(jnp.ones_like(loss))  # parameter-shaped leaves
        # Reduce gradients across the global minibatch before clipping and Adam.
        grads = jax.lax.with_sharding_constraint(grads, PartitionSpec())
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
    rollout_carry: RolloutCarry,  # observations: (B, *O); memory: (B, H)
    train_state: TrainState,
    envs: PPOEnv,
    config: Config,
    key: jax.Array,  # () typed PRNG key
) -> tuple[RolloutCarry, RolloutResult]:
    """Collect (B, T, ...) transitions with host environments and JAX policy/GAE.

    Timeouts bootstrap from final observations; both end flags reset recurrence.
    Completed episode returns, lengths, and count are reduced to scalars on device.
    """

    class Runner(NamedTuple):
        current: RolloutCarry  # memory: (B, H)
        key: jax.Array  # () typed PRNG key
        episode_metrics: EpisodeMetrics  # scalar leaves

    class Transition(NamedTuple):
        """S is (B,) per step, (T, B) after scanning, then (B, T) in the batch."""

        observations: Observation  # leaves: (*S, *O)
        actions: jax.Array  # (*S,) integer IDs or (*S, *action_shape) bounded continuous actions
        log_probs: jax.Array  # (*S,)
        values: jax.Array  # (*S,)
        rewards: jax.Array  # (*S,)
        episode_ends: jax.Array  # (*S,)
        episode_starts: jax.Array  # (*S,)

    def step(
        runner: Runner,  # memory: (B, H); key/completed totals: ()
        _: None,
    ) -> tuple[Runner, Transition]:
        current, key, episode_metrics = runner  # memory: (B, H); key/completed totals: ()
        key, action_key = jax.random.split(key)  # each () typed PRNG key
        memory, policy, predictions = policy_step(
            train_state.apply_fn, train_state.params, current.observations, current.model_carry, current.episode_starts
        )  # memory: (B, H); policy: (B, A/D); predictions: (B,)
        actions, log_probs = sample_actions(policy, action_key)  # (B, *action_shape); (B,)
        transition = envs.step(actions)
        observations = transition.observations  # leaves: (B, *O)
        raw_rewards = transition.rewards  # (B,)
        episode_ends = transition.terminated | transition.truncated  # (B,)
        timeout = transition.truncated & ~transition.terminated  # (B,)
        rewards = raw_rewards  # (B,)
        if bool(jnp.any(timeout)):
            values = policy_step(
                train_state.apply_fn,
                train_state.params,
                transition.final_observations,
                memory,
                jnp.zeros(config.num_envs, dtype=bool),
            ).values  # (B,)
            rewards = rewards + config.gamma * jnp.where(timeout, values, 0.0)  # (B,)
        episode_returns = current.episode_returns + raw_rewards  # (B,)
        episode_step_counts = current.episode_step_counts + 1  # (B,)
        episode_metrics = RunningSum.update(
            episode_metrics,
            jax.tree.map(
                jnp.sum,
                EpisodeMetrics(
                    episode_count=episode_ends,  # (B,)
                    episode_return_sum=jnp.where(episode_ends, episode_returns, 0.0),  # (B,)
                    episode_length_sum=jnp.where(episode_ends, episode_step_counts, 0),  # (B,)
                ),
            ),
        )  # each ()
        advanced = RolloutCarry(
            observations=observations,  # (B, *O)
            model_carry=memory,  # leaves: (B, H)
            episode_starts=episode_ends,  # (B,)
            episode_returns=jnp.where(episode_ends, 0.0, episode_returns),  # (B,)
            episode_step_counts=jnp.where(episode_ends, 0, episode_step_counts),  # (B,)
        )
        return Runner(current=advanced, key=key, episode_metrics=episode_metrics), Transition(
            observations=current.observations,  # (B, *O)
            actions=actions,  # (B, *action_shape)
            log_probs=log_probs,  # (B,)
            values=predictions,  # (B,)
            rewards=rewards,  # (B,)
            episode_ends=episode_ends,  # (B,)
            episode_starts=current.episode_starts,  # (B,)
        )

    episode_metrics = EpisodeMetrics(
        episode_count=jnp.zeros((), dtype=int),  # ()
        episode_return_sum=jnp.zeros((), dtype=rollout_carry.episode_returns.dtype),  # ()
        episode_length_sum=jnp.zeros((), dtype=int),  # ()
    )
    runner = Runner(current=rollout_carry, key=key, episode_metrics=episode_metrics)
    transitions = []
    for _ in range(config.num_steps):
        runner, transition = step(runner, None)
        transitions.append(transition)
    advanced, key, episode_metrics = runner
    trajectory = jax.tree.map(lambda *leaves: jnp.stack(leaves), *transitions)  # leaves: (T, B, ...)
    next_value = policy_step(
        train_state.apply_fn, train_state.params, advanced.observations, advanced.model_carry, advanced.episode_starts
    ).values  # (B,)
    advantages, returns = gae(
        trajectory.rewards, trajectory.episode_ends, trajectory.values, next_value, config.gamma, config.gae_lambda
    )  # each (T, B)
    trajectory, advantages, returns = jax.tree.map(
        lambda leaf: jnp.swapaxes(leaf, 0, 1), (trajectory, advantages, returns)
    )  # trajectory leaves: (B, T, ...); advantages/returns: (B, T)
    batch = PPOBatch(
        observations=trajectory.observations,  # (B, T, *O)
        actions=trajectory.actions,  # (B, T, *action_shape)
        old_log_probs=trajectory.log_probs,  # (B, T)
        values=trajectory.values,  # (B, T)
        advantages=advantages,  # (B, T)
        returns=returns,  # (B, T)
        model_carry=rollout_carry.model_carry,  # leaves: (B, H)
        episode_starts=trajectory.episode_starts,  # (B, T)
    )
    return advanced, RolloutResult(
        batch=batch,  # observations: (B, T, *O); targets/starts: (B, T); model_carry leaves: (B, H)
        key=key,  # ()
        episode_return_sum=episode_metrics.episode_return_sum,  # ()
        episode_length_sum=episode_metrics.episode_length_sum,  # ()
        episode_count=episode_metrics.episode_count,  # ()
    )


@partial(jax.jit, static_argnames=("config",), donate_argnames=("train_state",))
def optimize_rollout(
    rollout_carry: RolloutCarry,  # observations: (B, *O); memory: (B, H)
    rollout: RolloutResult,  # batch leaves: (B, T, ...); key/statistics: ()
    train_state: TrainState,
    config: Config,
    metrics: TrainingMetrics,  # scalar leaves
    schedule_index: int | jax.Array = 0,  # ()
) -> TrainStepResult:
    """Run PPO epochs and accumulate metrics inside the active batch mesh."""
    batch = jax.lax.with_sharding_constraint(rollout.batch, PartitionSpec("batch"))  # (B, T, ...); memory: (B, H)
    learning_rate = jnp.asarray(learning_rate_schedule(config)(schedule_index), dtype=jnp.float32)  # ()
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
            return key, jax.random.permutation(shuffle_key, config.num_envs)  # (); (B,)

        def keep_permutation(key: jax.Array) -> tuple[jax.Array, jax.Array]:  # () typed PRNG key
            return key, current.permutation  # (); (B,)

        minibatch_index = current.minibatch_count % config.num_minibatches  # ()
        key, permutation = jax.lax.cond(minibatch_index == 0, shuffle, keep_permutation, current.key)  # (); (B,)
        indices = permutation.reshape(config.num_minibatches, -1)[minibatch_index]  # (B,), B is minibatch size
        minibatch = jax.tree.map(itemgetter(indices), batch)  # (B, T, ...); memory: (B, H)
        minibatch = jax.lax.with_sharding_constraint(minibatch, PartitionSpec("batch"))
        train_state, metrics = update(current.train_state, minibatch, config, schedule_index)
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
            permutation=jnp.zeros(config.num_envs, dtype=jnp.int32),  # (B,)
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
    variance = explained_variance(batch.values, batch.returns)  # () undefined for constant returns
    variance_defined = ~jnp.isnan(variance)  # ()
    window = WindowMetrics(
        episodes=episodes,  # scalar leaves
        minibatch_metrics=updated.minibatch_metrics,  # scalar leaves
        explained_variance_sum=jnp.where(variance_defined, variance, 0.0),  # ()
        explained_variance_count=variance_defined.astype(jnp.int32),  # ()
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
    rollout_carry = jax.lax.with_sharding_constraint(rollout_carry, PartitionSpec("batch"))  # leaves: (B, ...)
    return TrainStepResult(
        train_state=train_state,
        rollout_carry=rollout_carry,  # memory: (B, H)
        key=key,  # ()
        metrics=metrics,  # scalar leaves
    )


def train_step(
    rollout_carry: RolloutCarry,  # observations: (B, *O); memory: (B, H)
    train_state: TrainState,
    envs: PPOEnv,
    config: Config,
    key: jax.Array,  # () typed PRNG key
    metrics: TrainingMetrics,  # scalar leaves
    schedule_index: int | jax.Array = 0,  # () completed environment steps // batch size
) -> TrainStepResult:
    """Collect on the host, then run compiled PPO epochs over the collected batch."""
    rollout_carry, rollout = collect_rollout(rollout_carry, train_state, envs, config, key)
    return optimize_rollout(rollout_carry, rollout, train_state, config, metrics, schedule_index)


def training_mesh(config: Config) -> Mesh:
    """Use every available device and require equally sized environment shards."""
    devices = jax.devices()
    if config.num_envs % (config.num_minibatches * len(devices)):
        raise ValueError(
            f"num_envs ({config.num_envs}) must be divisible by num_minibatches "
            f"({config.num_minibatches}) * device_count ({len(devices)})"
        )
    return Mesh(devices=devices, axis_names=("batch",), axis_types=(AxisType.Auto,))


def train(config: Config) -> TrainState:
    """Train with a fully resolved config, as returned by load_config."""
    with ExitStack() as resources, jax.set_mesh(training_mesh(config)):
        configure_compilation_cache()
        replicated = PartitionSpec()
        batched = PartitionSpec("batch")
        batch_size = config.num_envs * config.num_steps
        envs = resources.enter_context(closing(make_env(config)))
        manager = resources.enter_context(closing(checkpoint_manager(config.ckpt_dir)))
        key, init_key = jax.random.split(jax.random.key(config.seed))  # each () typed PRNG key
        obs = envs.reset()  # leaves: (B, *O)
        model = ActorCritic(
            action_space=envs.action_space,
            model=config.model,
            dtype=jnp.bfloat16 if config.bf16 else jnp.float32,
        )
        model_carry = model.initial_carry(config.num_envs)  # leaves: (B, H)
        episode_starts = jnp.ones(config.num_envs, dtype=bool)  # (B,)
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
                init_key, jax.tree.map(lambda x: x[:1, None], obs), model.initial_carry(1), episode_starts[:1, None]
            )["params"],
            tx=optimizer(config.learning_rate),
        )
        train_state = jax.device_put(train_state, replicated)
        progress = restore_checkpoint(manager, train_state, config)
        metrics = init_metrics(progress)  # scalar leaves
        start_rollout_index = 0
        if progress is not None:
            train_state = jax.device_put(progress.train_state, replicated)
            key = progress.key  # ()
            start_rollout_index = progress.rollout_count
            # The environment and recurrent memory were freshly initialized above.
        key, metrics = jax.device_put((key, metrics), replicated)  # key: (); metrics: scalar leaves
        initial_step_count = progress.env_steps if progress is not None else 0
        writer = MLflowLogger(experiment="ppo_nonnative", run_name=config.run_id)
        resources.enter_context(writer)
        if progress is not None:
            print(f"Resumed run {config.run_id} at step {initial_step_count}", flush=True)
        writer.log_text(yaml.safe_dump(asdict(config)), f"config/{initial_step_count}.yaml")
        writer.log_params(asdict(config))
        print(f"Run ID: {config.run_id}\nMLflow run: {writer.run_id}", flush=True)
        devices = str(jax.devices())
        print(f"JAX devices: {devices}", flush=True)
        writer.log_text(devices, f"devices/{initial_step_count}.txt")
        observation_space = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), obs)
        print(f"Observation space: {observation_space}", flush=True)
        print(f"Action space: {envs.action_space}", flush=True)
        parameter_count = sum(parameter.size for parameter in jax.tree.leaves(train_state.params))
        writer.log_metrics({"model/params_millions": parameter_count / 1_000_000}, initial_step_count)
        print(f"Model parameters: {parameter_count:,}", flush=True)
        writer.log_text(str(parameter_count), f"model/parameter_count/{initial_step_count}.txt")
        rollout_carry = RolloutCarry(
            observations=obs,  # (B, *O)
            model_carry=model_carry,  # leaves: (B, H)
            episode_starts=episode_starts,  # (B,)
            episode_returns=jnp.zeros(config.num_envs, dtype=jnp.float32),  # (B,)
            episode_step_counts=jnp.zeros(config.num_envs, dtype=jnp.int32),  # (B,)
        )
        rollout_carry = jax.device_put(rollout_carry, batched)  # leaves: (B, ...)

        window_start = monotonic()
        window_start_step_count = initial_step_count
        planned_rollout_count = start_rollout_index + max(config.total_steps - initial_step_count, 0) // batch_size
        step_count = initial_step_count
        for rollout_index in range(start_rollout_index, planned_rollout_count):
            schedule_index = step_count // batch_size
            train_state, rollout_carry, key, metrics = train_step(
                rollout_carry, train_state, envs, config, key, metrics, schedule_index
            )  # memory: (B, H); key/metrics: ()
            rollout_count = rollout_index + 1
            is_final_rollout = rollout_count == planned_rollout_count
            step_count += batch_size
            if config.video_interval and (rollout_count % config.video_interval == 0 or is_final_rollout):
                log_video(
                    train_state,
                    config,
                    writer,
                    int(metrics.total.episode_count),
                    step_count,
                    initial_carry=model.initial_carry(1),
                )
            if config.eval_interval and (rollout_count % config.eval_interval == 0 or is_final_rollout):
                log_evaluation(
                    train_state,
                    config,
                    writer,
                    int(metrics.total.episode_count),
                    step_count,
                    initial_carry=model.initial_carry(1),
                )
            if rollout_count % config.log_interval == 0 or is_final_rollout:
                metrics_started = monotonic()
                # Synchronize the window before measuring throughput.
                window, total = jax.device_get(metrics)  # scalar leaves
                now = monotonic()
                window_seconds = now - window_start
                mean_metrics = {
                    field.name: float(getattr(window.minibatch_metrics, field.name)) / int(window.minibatch_count)
                    for field in fields(MinibatchMetrics)
                }
                window_iteration_count = int(window.rollout_count)
                window_update_count = int(window.update_count)
                window_episode_count = int(window.episodes.episode_count)
                episode_count = int(total.episode_count)
                learning_rate = float(lr_schedule(schedule_index))
                entropy_coef = float(entropy_schedule(schedule_index))
                rate_seconds = max(window_seconds, 1e-9)
                sps = (step_count - window_start_step_count) / rate_seconds
                scalars: dict[str, float | int] = {
                    "train/updates_per_iteration": window_update_count / window_iteration_count,
                    "train/policy_loss": mean_metrics["policy_loss"],
                    "train/value_loss": mean_metrics["value_loss"],
                    "train/entropy_loss": -mean_metrics["entropy"],
                    "train/total_loss": mean_metrics["policy_loss"]
                    + config.value_coef * mean_metrics["value_loss"]
                    + mean_metrics["entropy_loss"],
                    "train/entropy": mean_metrics["entropy"],
                    "train/approx_kl": mean_metrics["approx_kl"],
                    "train/clip_fraction": mean_metrics["clip_fraction"],
                    "train/iterations_per_second": window_iteration_count / rate_seconds,
                    "train/env_steps_per_second": sps,
                    "train/updates_per_second": window_update_count / rate_seconds,
                    "train/learning_rate": learning_rate,
                    "train/entropy_coef": entropy_coef,
                    "train/episodes_total": episode_count,
                    "train/env_steps_total": step_count,
                    "train/iterations_total": rollout_count,
                    "train/updates_total": int(train_state.step),
                    "train/progress_fraction": step_count / config.total_steps,
                    "rollout/episode_count": window_episode_count,
                }
                if config.target_kl is not None:
                    scalars["train/kl_early_stop_fraction"] = int(window.early_stop_count) / window_iteration_count
                if window.explained_variance_count:
                    scalars["train/explained_variance"] = float(window.explained_variance_sum) / int(
                        window.explained_variance_count
                    )
                if window_episode_count:
                    scalars["rollout/return"] = float(window.episodes.episode_return_sum) / window_episode_count
                    scalars["rollout/episode_length_steps"] = (
                        int(window.episodes.episode_length_sum) / window_episode_count
                    )
                if episode_count:
                    scalars["train/return"] = float(total.episode_return_sum) / episode_count
                    scalars["train/episode_length_steps"] = int(total.episode_length_sum) / episode_count
                writer.log_metrics(scalars, step_count)
                episode_return_text = f"{scalars['rollout/return']:.1f}" if window_episode_count else "n/a"
                episode_length_text = (
                    f"{scalars['rollout/episode_length_steps']:.1f}" if window_episode_count else "n/a"
                )
                variance_text = (
                    f"{scalars['train/explained_variance']:.3f}" if window.explained_variance_count else "n/a"
                )
                early_stop_text = (
                    f"{scalars['train/kl_early_stop_fraction']:.2f}" if config.target_kl is not None else "n/a"
                )
                print(
                    f"iteration={rollout_count} step={step_count} episodes={episode_count} return={episode_return_text} "
                    f"episode_length={episode_length_text} "
                    f"progress={scalars['train/progress_fraction']:.3f} "
                    f"iterations/s={scalars['train/iterations_per_second']:.3f} "
                    f"env_steps/s={sps:.3f} updates/s={scalars['train/updates_per_second']:.3f} "
                    f"lr={learning_rate:.3g} "
                    f"policy={mean_metrics['policy_loss']:.3f} value={mean_metrics['value_loss']:.3f} "
                    f"entropy={mean_metrics['entropy']:.3f} kl={mean_metrics['approx_kl']:.4f} "
                    f"clipfrac={mean_metrics['clip_fraction']:.3f} ev={variance_text} "
                    f"updates/iteration={scalars['train/updates_per_iteration']:.2f} "
                    f"early_stop={early_stop_text} "
                    f"window={window_seconds:.1f}s",
                    flush=True,
                )
                window_start = now
                window_start_step_count = step_count
                with jax.enable_x64():
                    metrics = metrics._replace(window=RunningSum.init(metrics.window))  # scalar leaves
                writer.log_metrics({"train/metrics_seconds": monotonic() - metrics_started}, step_count)
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
                        env_steps=step_count,
                    ),
                    config,
                )
                writer.log_metrics({"checkpoint/save_seconds": checkpoint_seconds}, step_count)
        return train_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ppo_nonnative.yaml", help="Path to a YAML config")
    args = parser.parse_args()
    train(load_config(args.config, Config))


if __name__ == "__main__":
    main()
