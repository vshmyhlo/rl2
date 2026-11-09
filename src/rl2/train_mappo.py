"""Recurrent parameter-sharing IPPO for SMAX, JaxNav, MPE, STORM, JaxRobotarium, and Lux.

The actor and critic share a local observation encoder/LSTM with separate heads.
Actions are masked during collection and updates. Agent death/inactivity resets
memory without ending the reward stream or GAE trace. Environment completion ends
traces; timeouts bootstrap from final local observations unless also terminated.
Despite the historical MAPPO filename, the critic uses no centralized state.

E denotes environments, N agents, B = E * N agent sequences, T rollout steps,
O local observation features, A categories, and H recurrent width. C is empty
for scalar actions or (D,) for D discrete components. Multi-discrete policies
produce one joint action per agent: log probabilities and entropies sum over D.
Recurrent minibatches preserve entire (B, T, ...) agent sequences. Advantages are
standardized per minibatch; a single transition retains its raw advantage.
KL-rejected minibatches skip backpropagation and optimizer updates.

An iteration collects num_envs * num_steps environment transitions and runs PPO
updates. total_steps and MLflow steps count transitions summed across environments;
observations count each agent separately. Team returns sum mean agent rewards,
counting shared rewards once. Episodes count environment completions. Lux uses
match-win increments; averaging both teams gives match_count / 2 per full game,
so self-play return is not a measure of playing strength.

Progress metrics are logged every log_interval iterations (default: 60) and on
normal completion. No metrics have categorical suffixes.

Window metrics:
    train/policy_loss: Mean clipped surrogate policy loss.
    train/value_loss: Half mean squared value error, before value_coef.
    train/total_loss: Policy loss + value_coef * value loss - entropy_coef * entropy.
    train/entropy: Masked joint action entropy in nats, summed over components.
    train/approx_kl: Mean exp(log_ratio) - 1 - log_ratio in nats versus rollout policy.
    train/clip_fraction: Fraction of action probability ratios outside the clip range.
        These average over transitions, then evaluated minibatches, including
        KL-rejected minibatches. Total loss uses each iteration's entropy weight.
    train/explained_variance_mean: Mean rollout value explained variance over
        iterations with nonconstant GAE returns; omitted when none are defined.
    train/updates_per_iteration_mean: Accepted minibatch updates per iteration.
    train/kl_early_stop_fraction: Fraction of iterations stopped by target_kl;
        omitted when KL stopping is disabled.
    rollout/episode_count: Number of completed episodes in the window.
    rollout/return_mean: Mean raw return of completed episodes.
    rollout/episode_length_steps_mean: Mean length of completed episodes in steps.
        Episode means are omitted when no episodes completed.
    train/iterations_per_second, train/env_steps_per_second,
    train/observations_per_second, train/episodes_per_second: Window counts divided
        by wall seconds. Includes collection, updates, and intervening logging,
        evaluation, video, and checkpoints; excludes the current log and later save.

Lifetime metrics (include restored history):
    train/iterations_total: Completed iterations.
    train/updates_total: Accepted optimizer updates.
    train/episodes_total: Completed training episodes.
    train/env_steps_total: Environment transitions.
    train/observations_total: Agent observations.
    train/return_mean: Raw return averaged over completed episodes.
    train/episode_length_steps_mean: Episode length averaged over completed episodes.
        Lifetime episode means are omitted before any episode completes.
    train/progress_fraction: Completed environment transitions / total_steps,
        capped at one. The budget remainder smaller than a rollout is unused.

Latest settings and session timings:
    train/learning_rate, train/entropy_coef: Latest iteration's scheduled settings,
        evaluated at its starting environment-step count against total_steps.
    train/elapsed_seconds: Wall seconds since this session's rollout loop started,
        through the current metric read, including intervening overhead.
    actor/video_seconds: Gameplay recording, encoding, and upload wall seconds.
    eval/evaluation_seconds: Evaluation through gameplay/report upload wall seconds.
    checkpoint/save_seconds: Checkpoint save wall seconds through completed writes.
        Operation timings are logged only when that operation runs, at its current
        environment step, and exclude logging the timing itself.
    model/params_millions: Parameter count in millions at startup, including resume.

Evaluation logs eval/return_mean, eval/return_std, and eval/return_sem: raw return
mean, sample standard deviation, and standard error over eval_episodes fixed seeds.
Uncertainty metrics are omitted for one episode. eval/training_episodes_total is
the lifetime completed training episode count at evaluation.

Enabled video, evaluation, and checkpoint schedules use completed iteration counts,
including on resume, and also run on normal completion. video_interval defaults to
600, eval_interval to 0 (disabled), and checkpoint_interval to 600. A zero video
interval disables standalone gameplay. Evaluation records its last episode
independently of video_interval. video_max_frames (default: 900; None: unlimited)
caps recordings; evaluation still finishes the full episode. GIFs loop at video_fps
with frame boundaries rounded to 10 ms and a minimum 10 ms frame delay.

Artifacts: config/<step>.yaml, devices/<step>.txt, logs/<session>.log,
model/parameter_count/<step>.txt, eval/report/<step>.json, gameplay/<step>.gif,
and eval/gameplay/<step>.gif. Evaluation reports include individual returns and
lengths, aggregate scores, configuration, seed, and training counters. Configuration
parameters use dotted keys; existing MLflow parameter values are preserved on resume.

Checkpoints save the full configuration, parameters, optimizer state, RNG, iteration
and transition counters, and cumulative episode statistics. Restore uses the current
configuration without compatibility checks. Legacy checkpoints lacking transition
counters estimate them using current rollout dimensions. Live environments and
memory restart on resume; unfinished episode statistics are discarded.
"""

import argparse
import json
import os
from collections.abc import Callable
from contextlib import ExitStack, closing
from dataclasses import asdict, fields
from datetime import timedelta
from functools import partial
from math import prod, sqrt
from operator import itemgetter
from pathlib import Path
from statistics import fmean, stdev
from time import monotonic
from typing import Annotated, Any, Literal, NamedTuple, Self

# Orbax's epath otherwise loads TensorFlow for gs:// paths when it is installed.
# That native runtime can crash a later Triton import; use our gcsfs dependency.
# Set this before importing Orbax, since epath caches its backend selection.
os.environ.setdefault("EPATH_USE_TF", "0")

import chex
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import rlax
import yaml
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass

from rl2.actions import CategoricalPolicy, DiscreteSpace, MultiCategoricalPolicy, MultiDiscreteSpace
from rl2.configuration import BaseConfig, load_config
from rl2.envs.jaxnav_env import JaxNavConfig, JaxNavEnv
from rl2.envs.lux_env import LuxConfig, LuxEnv
from rl2.envs.mpe_env import MPEConfig, MPEEnv
from rl2.envs.multi_agent import AgentObservation, BaseMultiAgentEnv
from rl2.envs.robotarium_env import RobotariumConfig, RobotariumEnv
from rl2.envs.smax_env import SMAXConfig, SMAXEnv
from rl2.envs.storm_env import STORMConfig, STORMEnv
from rl2.jax_cache import configure_compilation_cache
from rl2.lstm import LSTMStack, LSTMStackCarry
from rl2.mlflow_logging import MLflowLogger
from rl2.playback import EpisodeResult, PlaybackState, finish_episode
from rl2.running_stats import RunningSum
from rl2.utils import explained_variance, gae

type RecurrentCarry = LSTMStackCarry
type PPOEnv = BaseMultiAgentEnv[Any]


@struct.dataclass
class PPOBatch:
    """Batch-major recurrent PPO inputs; S = (B, T).

    Attributes:
        observation: (*S, O) local observation features.
        actions_mask: (*S, *C, A) boolean legal-action flags.
        action: (*S, *C) integer action IDs.
        old_log_prob: (*S,) rollout action log probabilities.
        value: (*S,) rollout-time value predictions for explained variance.
        advantage: (*S,) unnormalized advantage estimates.
        return_target: (*S,) value targets.
        model_carry: (B, H) initial LSTM memory leaves, where H is model.recurrent.hidden_size.
        episode_start: (*S,) boolean agent or environment reset flags.
    """

    observation: jax.Array  # (*S, O)
    actions_mask: jax.Array  # (*S, *C, A) bool
    action: jax.Array  # (*S, *C) integer action IDs
    old_log_prob: jax.Array  # (*S,)
    value: jax.Array  # (*S,)
    advantage: jax.Array  # (*S,)
    return_target: jax.Array  # (*S,)
    model_carry: RecurrentCarry  # leaves: (B, H), H is model.recurrent.hidden_size
    episode_start: jax.Array  # (*S,)

    def select_sequences(self, index: jax.Array) -> Self:  # (M,) agent-sequence indices
        """Return (M, T, ...) sequence fields and (M, H) initial memory leaves."""
        return jax.tree.map(itemgetter(index), self)


class RolloutCarry(NamedTuple):
    """Live environment state and recurrent memory between rollouts.

    Attributes:
        env_state: (E, ...) batched environment state leaves.
        observation: (E, N, O) local features and (E, N, *C, A) legal-action masks.
        model_carry: (B, H) LSTM memory leaves, where H is model.recurrent.hidden_size.
        episode_start: (B,) boolean agent or environment reset flags.
        episode_return: (E,) accumulated raw team rewards for unfinished episodes.
        episode_step_count: (E,) step counts for unfinished episodes.
    """

    env_state: chex.ArrayTree  # leaves: (E, ...)
    observation: AgentObservation  # observation: (E, N, O); actions_mask: (E, N, *C, A)
    model_carry: RecurrentCarry  # leaves: (B, H)
    episode_start: jax.Array  # (B,) reset on agent death or environment completion
    episode_return: jax.Array  # (E,) unfinished team returns
    episode_step_count: jax.Array  # (E,) unfinished episode lengths


class RolloutResult(NamedTuple):
    """Collected training data, advanced RNG key, and completed episode statistics.

    Attributes:
        batch: (B, T, ...) batch-major PPO arrays with (B, H) initial memory leaves.
        key: () advanced typed PRNG key, also used for minibatch shuffling.
        episode_return_sum: () sum of raw returns for completed episodes.
        episode_length_sum: () sum of lengths for completed episodes.
        episode_count: () number of completed episodes.
    """

    batch: PPOBatch  # observation: (B, T, O); targets/starts: (B, T); model_carry leaves: (B, H)
    key: jax.Array  # () typed PRNG key
    episode_return_sum: jax.Array  # ()
    episode_length_sum: jax.Array  # ()
    episode_count: jax.Array  # ()


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
    rollout_carry: RolloutCarry  # observation: (E, N, O); state: (E, ...); memory: (B, H)
    key: jax.Array  # () typed PRNG key
    metrics: TrainingMetrics  # scalar leaves


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LSTMConfig:
    type: Literal["lstm"] = "lstm"
    hidden_size: Annotated[int, Field(gt=0, strict=True)] = 128
    num_layers: Annotated[int, Field(gt=0, strict=True)] = 1
    intermediate_size: Annotated[int, Field(ge=0, strict=True)] = 0


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class MLPEncoderConfig:
    type: Literal["mlp"] = "mlp"
    embedding_size: Annotated[int, Field(gt=0, strict=True)] = 128


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class ModelConfig:
    encoder: MLPEncoderConfig
    recurrent: LSTMConfig = LSTMConfig()


@dataclass(frozen=True)
class Config(BaseConfig):
    model: ModelConfig
    env: Annotated[
        SMAXConfig | JaxNavConfig | MPEConfig | STORMConfig | RobotariumConfig | LuxConfig, Field(discriminator="id")
    ]
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
    value_coef: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    max_grad_norm: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    video_fps: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    bf16: bool = True
    video_max_frames: Annotated[int, Field(gt=0)] | None = 900
    video_interval: Annotated[int, Field(ge=0)] = 600
    eval_interval: Annotated[int, Field(ge=0)] = 0
    eval_episodes: Annotated[int, Field(gt=0)] = 100
    eval_seed: Annotated[int, Field(ge=0, lt=2**32)] = 10_000
    lr_decay: Literal["linear", "cosine"] = "linear"
    entropy_decay: Literal["constant", "cosine"] = "constant"
    checkpoint_interval: Annotated[int, Field(gt=0)] = 600
    log_interval: Annotated[int, Field(gt=0)] = 60

    @model_validator(mode="after")
    def validate_rollout(self) -> Self:
        if self.total_steps < self.num_envs * self.num_steps:
            raise ValueError("total_steps must cover at least one rollout")
        return self


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
    if config.num_envs * env.num_agents % config.num_minibatches:
        raise ValueError("num_envs * num_agents must be divisible by num_minibatches to preserve sequences")
    return env


class TrainingProgress(NamedTuple):
    train_state: TrainState
    key: jax.Array  # () typed PRNG key
    rollout_count: int  # Number of completed rollouts, independent of optimizer steps.
    episode_count: int
    episode_return_sum: float
    episode_length_sum: int
    environment_step_count: int | None = None  # None for legacy checkpoints without transition counters.
    observation_count: int | None = None


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
        options=ocp.CheckpointManagerOptions(max_to_keep=2),
    )


def save_checkpoint(manager: ocp.CheckpointManager, progress: TrainingProgress, config: Config) -> float:
    """Wait for a completed rollout checkpoint to finish saving and return elapsed seconds."""
    started = monotonic()
    key_data = jax.random.key_data(progress.key)  # (K,) raw PRNG words
    manager.save(
        progress.rollout_count,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave({"train_state": progress.train_state, "key": key_data}),
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
                    "observations": progress.observation_count,
                }
            ),
        ),
        force=True,
    )
    manager.wait_until_finished()
    seconds = monotonic() - started
    print(f"Saved checkpoint at iteration {progress.rollout_count} in {seconds:.1f}s", flush=True)
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
    return TrainingProgress(
        train_state=restored["train_state"],
        key=jax.random.wrap_key_data(restored["key"]),
        rollout_count=rollout_count,
        episode_count=metadata["completed_episodes"],
        episode_return_sum=metadata["training_return_sum"],
        episode_length_sum=metadata["training_length_sum"],
        environment_step_count=metadata.get("environment_steps"),
        observation_count=metadata.get("observations"),
    )


def decay_schedule(initial_value: float, total_steps: int, decay: Literal["linear", "cosine"]) -> optax.Schedule:
    schedule = (
        optax.cosine_decay_schedule(initial_value, 1)
        if decay == "cosine"
        else optax.linear_schedule(initial_value, 0.0, 1)
    )

    def by_environment_steps(count: jax.Array) -> jax.Array:  # () -> ()
        # Normalize before Optax's clipping to support budgets beyond int32.
        return schedule(jnp.asarray(count, dtype=jnp.float32) / float(total_steps))  # ()

    return by_environment_steps


def learning_rate_schedule(config: Config) -> optax.Schedule:
    if not config.anneal_lr:
        return optax.constant_schedule(config.learning_rate)
    return decay_schedule(config.learning_rate, config.total_steps, config.lr_decay)


def entropy_coef_schedule(config: Config) -> optax.Schedule:
    if config.entropy_decay == "constant":
        return optax.constant_schedule(config.entropy_coef)
    return decay_schedule(config.entropy_coef, config.total_steps, "cosine")


class ActorCriticOutput(NamedTuple):
    carry: RecurrentCarry  # leaves: (B, H)
    policy: CategoricalPolicy | MultiCategoricalPolicy  # logits: (*S, *C, A)
    value: jax.Array  # (*S,)


class ActorCritic(nn.Module):
    """Shared local observation encoder/LSTM with separate policy and value heads."""

    action_space: DiscreteSpace | MultiDiscreteSpace
    model: ModelConfig
    dtype: jax.typing.DTypeLike = jnp.float32

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
    def initial_carry(self, actor_count: int) -> RecurrentCarry:
        return self._make_recurrent().initial_carry(actor_count)  # leaves: (B, H)

    def setup(self) -> None:
        self.encoder = nn.Dense(features=self.model.encoder.embedding_size, dtype=self.dtype, name="encoder")
        self.recurrent = self._make_recurrent(parent=self)
        self.recurrent_input = nn.Dense(features=self.model.recurrent.hidden_size, dtype=self.dtype, name="lstm_input")
        init = nn.initializers.orthogonal(sqrt(2))
        self.policy_hidden = nn.Dense(features=512, kernel_init=init, name="policy_hidden", dtype=self.dtype)
        self.policy_norm = nn.LayerNorm(name="policy_norm", dtype=self.dtype)
        self.value_hidden = nn.Dense(features=512, kernel_init=init, name="value_hidden", dtype=self.dtype)
        self.value_norm = nn.LayerNorm(name="value_norm", dtype=self.dtype)
        self.policy_output = nn.Dense(
            features=(
                self.action_space.n
                if isinstance(self.action_space, DiscreteSpace)
                else prod(self.action_space.logits_shape)
            ),
            kernel_init=nn.initializers.orthogonal(0.01),
            name="policy_output",
            dtype=self.dtype,
        )
        self.value_output = nn.Dense(
            features=1, kernel_init=nn.initializers.orthogonal(1.0), name="value_output", dtype=self.dtype
        )

    def _encode(self, observation: jax.Array) -> jax.Array:  # (*S, O) -> (*S, H)
        x = nn.relu(self.encoder(observation))  # (*S, F), F is model.encoder.embedding_size
        return self.recurrent_input(x).astype(self.dtype)  # (*S, H)

    def _heads(
        self,
        x: jax.Array,  # (*S, H)
        actions_mask: jax.Array,  # (*S, *C, A)
    ) -> tuple[CategoricalPolicy | MultiCategoricalPolicy, jax.Array]:  # logits: (*S, *C, A); value: (*S,)
        x = x.astype(self.dtype)  # (*S, H), S is (B,) or (B, T)
        policy = nn.relu(self.policy_norm(self.policy_hidden(x)))  # (*S, 512)
        critic = nn.relu(self.value_norm(self.value_hidden(x)))  # (*S, 512)
        # Heads use the compute dtype; float32 outputs keep PPO loss arithmetic precise.
        logits = self.policy_output(policy).astype(jnp.float32)  # (*S, prod(C)*A), before component reshape
        if isinstance(self.action_space, MultiDiscreteSpace):
            logits = logits.reshape((*x.shape[:-1], *self.action_space.logits_shape))  # (*S, D, A)
            counts = jnp.asarray(self.action_space.nvec)  # (D,)
            actions_mask = actions_mask & (jnp.arange(logits.shape[-1]) < counts[:, None])  # (*S, D, A)
        logits = jnp.where(actions_mask, logits, -1e10)  # (*S, *C, A)
        value = self.value_output(critic).squeeze(-1).astype(jnp.float32)  # (*S,)
        distribution = (
            MultiCategoricalPolicy(logits=logits)
            if isinstance(self.action_space, MultiDiscreteSpace)
            else CategoricalPolicy(logits=logits)
        )
        return distribution, value  # logits: (*S, *C, A); value: (*S,)

    def __call__(
        self,
        observation: jax.Array,  # (B, T, O)
        model_carry: RecurrentCarry,  # leaves: (B, H)
        episode_start: jax.Array,  # (B, T)
        actions_mask: jax.Array,  # (B, T, *C, A)
    ) -> ActorCriticOutput:  # carry leaves: (B, H); logits: (B, T, *C, A); value: (B, T)
        """Process observation sequences, resetting memory at episode starts."""
        x = self._encode(observation)  # (B, T, H)
        # The shared LSTM scans time-major inputs; MAPPO exposes batch-major arrays.
        recurrent_x, recurrent_starts = jax.tree.map(
            lambda leaf: jnp.swapaxes(leaf, 0, 1), (x, episode_start)
        )  # (T, B, H); (T, B)
        model_carry, recurrent_x = self.recurrent(
            recurrent_x, model_carry, recurrent_starts
        )  # model_carry leaves: (B, H); recurrent_x: (T, B, H)
        x = jnp.swapaxes(recurrent_x, 0, 1)  # (B, T, H)
        policy, value = self._heads(x, actions_mask)  # logits: (B, T, *C, A); (B, T)
        return ActorCriticOutput(carry=model_carry, policy=policy, value=value)

    def step(
        self,
        observation: jax.Array,  # (B, O)
        model_carry: RecurrentCarry,  # leaves: (B, H)
        episode_start: jax.Array,  # (B,)
        actions_mask: jax.Array,  # (B, *C, A)
    ) -> ActorCriticOutput:  # carry leaves: (B, H); logits: (B, *C, A); value: (B,)
        """Process one observation per sequence, resetting memory at episode starts."""
        x = self._encode(observation)  # (B, H)
        model_carry, x = self.recurrent.step(x, model_carry, episode_start)  # model_carry leaves: (B, H); x: (B, H)
        policy, value = self._heads(x, actions_mask)  # logits: (B, *C, A); (B,)
        return ActorCriticOutput(carry=model_carry, policy=policy, value=value)


@partial(jax.jit, static_argnames="apply_fn")
def sample_policy_step(
    apply_fn: Callable[..., ActorCriticOutput],
    params: optax.Params,
    observation: jax.Array,  # (B, O)
    carry: RecurrentCarry,  # leaves: (B, H)
    episode_start: jax.Array,  # (B,)
    key: jax.Array,  # () typed PRNG key
    actions_mask: jax.Array,  # (B, *C, A)
) -> tuple[RecurrentCarry, jax.Array]:  # carry leaves: (B, H); action: (B, *C)
    """Compile inference and sampling together; reuse across playback steps and episodes."""
    carry, policy, _ = apply_fn(
        {"params": params}, observation, carry, episode_start, actions_mask, method="step"
    )  # carry leaves: (B, H); logits: (B, *C, A); value: (B,)
    return carry, policy.sample(seed=key)  # carry leaves: (B, H); action: (B, *C)


type EpisodeState = PlaybackState[AgentObservation, Any, RecurrentCarry]


def playback_step(
    env: PPOEnv,
    apply_fn: Callable[..., ActorCriticOutput],
    params: optax.Params,
    state: EpisodeState,  # observation/state/memory leaves as in PlaybackState
) -> EpisodeState:
    key, action_key, env_key = jax.random.split(state.key, 3)  # each () typed PRNG key
    memory, action = sample_policy_step(
        apply_fn,
        params,
        state.observation.observation,
        state.memory,
        state.episode_starts,
        action_key,
        state.observation.actions_mask,
    )  # memory leaves: (N, H); action: (N, *C)
    result = env.step(env_key, state.env_state, action)
    return state._replace(
        key=key,  # ()
        observation=result.observation,  # (N, O)/(N, *C, A)
        env_state=result.state,  # leaves: (...)
        final_state=result.final_state,  # leaves: (...)
        memory=memory,  # leaves: (N, H)
        episode_starts=result.agent_done,  # (N,)
        episode_return=state.episode_return + result.reward.mean(),  # ()
        length=state.length + 1,  # ()
        done=result.done,  # ()
    )


def play_episode(
    train_state: TrainState,
    config: Config,
    key: jax.Array,  # () typed PRNG key
    initial_carry: RecurrentCarry,  # leaves: (N, H)
    *,
    record: bool,
    env: PPOEnv,
    stop_at_frame_limit: bool = False,
) -> EpisodeResult:
    """Run on device, transferring bounded state chunks for optional host rendering."""
    key, reset_key = jax.random.split(key)  # each () typed PRNG key
    observation, env_state = env.reset(reset_key)  # observation/state leaves: (...)
    state = PlaybackState(
        key=key,  # ()
        observation=observation,  # (N, O)/(N, *C, A)
        env_state=env_state,  # leaves: (...)
        final_state=env_state,  # leaves: (...)
        memory=initial_carry,  # leaves: (N, H)
        episode_starts=jnp.ones(env.num_agents, dtype=bool),  # (N,)
        episode_return=jnp.asarray(0.0, dtype=jnp.float32),  # ()
        length=jnp.asarray(0, dtype=jnp.int32),  # ()
        done=jnp.asarray(False),  # ()
    )
    return finish_episode(
        playback_step,
        env,
        train_state.apply_fn,
        train_state.params,
        state,
        record=record,
        frame_limit=config.video_max_frames,
        stop_at_frame_limit=stop_at_frame_limit,
    )


def log_video(
    train_state: TrainState,
    config: Config,
    writer: MLflowLogger,
    episode_count: int,
    step_count: int,
    *,
    initial_carry: RecurrentCarry,  # leaves: (N, H)
    env: PPOEnv | None = None,
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
        env=make_env(config) if env is None else env,
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
    initial_carry: RecurrentCarry,  # leaves: (N, H)
    env: PPOEnv | None = None,
) -> None:
    """Evaluate fixed environment seeds, log raw scores, and record the last episode."""
    print(f"Evaluating {config.eval_episodes} games at step {step_count}...", flush=True)
    started = monotonic()
    episode_results: list[dict[str, float | int]] = []
    env = make_env(config) if env is None else env
    for episode_index in range(config.eval_episodes):
        episode_return, episode_step_count, frames = play_episode(
            train_state,
            config,
            jax.random.fold_in(jax.random.key(config.eval_seed), episode_index),
            initial_carry,
            record=episode_index == config.eval_episodes - 1,
            env=env,
        )
        episode_results.append({"return": episode_return, "length": episode_step_count})
        if frames:
            writer.log_video(frames, f"eval/gameplay/{step_count}.gif", fps=config.video_fps)
    episode_return_sample = [result["return"] for result in episode_results]
    std = stdev(episode_return_sample) if len(episode_return_sample) > 1 else None
    result = {
        "env_id": config.env.id,
        "return_mean": fmean(episode_return_sample),
        "return_std": std,
        "return_sem": std / sqrt(config.eval_episodes) if std is not None else None,
        "episodes": episode_results,
        "eval_seed": config.eval_seed,
        "training_config": asdict(config),
        "training_steps": step_count,
        "training_episodes": episode_count,
    }
    scalars = {
        f"eval/{name}": result[name] for name in ("return_mean", "return_std", "return_sem") if result[name] is not None
    }
    scalars["eval/training_episodes_total"] = episode_count
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
    batch: PPOBatch,  # observation: (*S, O); actions_mask: (*S, *C, A); targets/starts: (*S,); model_carry leaves: (B, H)
    config: Config,
    environment_step_count: int | jax.Array = 0,  # () completed environment transitions before this iteration
) -> tuple[TrainState, MinibatchMetrics]:
    entropy_coef = jnp.asarray(entropy_coef_schedule(config)(environment_step_count), dtype=jnp.float32)  # ()
    advantage = batch.advantage  # (*S,)
    chex.assert_scalar_positive(advantage.size)
    # Centering a single sample always gives zero and erases its policy gradient.
    if advantage.size > 1:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)  # (*S,)

    def loss_fn(params: optax.Params) -> tuple[jax.Array, MinibatchMetrics]:  # (); scalar metric leaves
        _, policy, value = train_state.apply_fn(
            {"params": params}, batch.observation, batch.model_carry, batch.episode_start, batch.actions_mask
        )  # model_carry leaves: (B, H); policy logits: (*S, *C, A); value: (*S,)
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
    rollout_carry: RolloutCarry,  # observation: (E, N, O); state: (E, ...); memory: (B, H)
    train_state: TrainState,
    env: PPOEnv,
    config: Config,
    key: jax.Array,  # () typed PRNG key
) -> tuple[RolloutCarry, RolloutResult]:
    """Collect (B, T, ...) transitions entirely on device and compute GAE.

    Timeouts bootstrap from final observations; agent end flags reset recurrence.
    Team rewards and GAE traces continue until environment completion.
    Completed episode returns, lengths, and count are reduced to scalars on device.
    """
    actor_count = config.num_envs * env.num_agents

    class Runner(NamedTuple):
        current: RolloutCarry  # state: (E, ...); memory: (B, H)
        key: jax.Array  # () typed PRNG key
        episode_metrics: EpisodeMetrics  # scalar leaves

    class Transition(NamedTuple):
        """S is (B,) per step, (T, B) after scanning, then (B, T) for training."""

        observation: jax.Array  # (*S, O)
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
        runner: Runner,  # state: (E, ...); memory: (B, H); key/completed totals: ()
        _: None,
    ) -> tuple[Runner, Transition]:
        current, key, episode_metrics = runner  # state: (E, ...); memory: (B, H); key/completed totals: ()
        key, action_key, env_key = jax.random.split(key, 3)  # each () typed PRNG key
        observation = jax.tree.map(flatten, current.observation)  # (B, O)/(B, *C, A)
        memory, policy, prediction = train_state.apply_fn(
            {"params": train_state.params},
            observation.observation,
            current.model_carry,
            current.episode_start,
            observation.actions_mask,
            method="step",
        )  # memory leaves: (B, H); logits: (B, *C, A); prediction: (B,)
        action, log_prob = policy.sample_and_log_prob(seed=action_key)  # (B, *C); (B,)
        env_key = jax.random.split(env_key, config.num_envs)  # (E,) typed PRNG keys
        result = jax.vmap(env.step)(
            env_key,
            current.env_state,
            action.reshape((config.num_envs, env.num_agents, *action.shape[1:])),  # (E, N, *C)
        )
        timeout = jnp.repeat(result.truncated & ~result.terminated, env.num_agents)  # (B,)
        episode_start = result.agent_done.reshape(actor_count)  # (B,)

        def bootstrap(_: None) -> jax.Array:  # (B,)
            final = jax.tree.map(flatten, result.final_observation)  # (B, O)/(B, *C, A)
            # Agent inactivity resets memory without ending the team's reward stream.
            reset = result.agent_inactive.reshape(actor_count)  # (B,)
            value = train_state.apply_fn(
                {"params": train_state.params},
                final.observation,
                memory,
                reset,
                final.actions_mask,
                method="step",
            ).value  # (B,)
            return jnp.where(timeout, value, 0.0)  # (B,)

        def no_bootstrap(_: None) -> jax.Array:
            return jnp.zeros(actor_count)  # (B,)

        raw_reward = result.reward.reshape(actor_count)  # (B,)
        reward = raw_reward + config.gamma * jax.lax.cond(timeout.any(), bootstrap, no_bootstrap, None)  # (B,)
        episode_return = current.episode_return + result.reward.mean(axis=-1)  # (E,)
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
        advanced = RolloutCarry(
            env_state=result.state,  # leaves: (E, ...)
            observation=result.observation,  # observation: (E, N, O); actions_mask: (E, N, *C, A)
            model_carry=memory,  # leaves: (B, H)
            episode_start=episode_start,  # (B,)
            episode_return=jnp.where(result.done, 0.0, episode_return),  # (E,)
            episode_step_count=jnp.where(result.done, 0, episode_step_count),  # (E,)
        )
        return Runner(current=advanced, key=key, episode_metrics=episode_metrics), Transition(
            observation=observation.observation,  # (B, O)
            actions_mask=observation.actions_mask,  # (B, *C, A)
            action=action,  # (B, *C)
            log_prob=log_prob,  # (B,)
            value=prediction,  # (B,)
            reward=reward,  # (B,)
            episode_end=jnp.repeat(result.done, env.num_agents),  # (B,)
            episode_start=current.episode_start,  # (B,)
        )

    episode_metrics = EpisodeMetrics(
        episode_count=jnp.zeros((), dtype=int),  # ()
        episode_return_sum=jnp.zeros((), dtype=rollout_carry.episode_return.dtype),  # ()
        episode_length_sum=jnp.zeros((), dtype=int),  # ()
    )
    (advanced, key, episode_metrics), trajectory = jax.lax.scan(
        step, Runner(current=rollout_carry, key=key, episode_metrics=episode_metrics), None, length=config.num_steps
    )  # trajectory leaves: (T, B, ...)
    # Peek at the next value without advancing the rollout's recurrent state.
    observation = jax.tree.map(flatten, advanced.observation)  # (B, O)/(B, *C, A)
    next_value = train_state.apply_fn(
        {"params": train_state.params},
        observation.observation,
        advanced.model_carry,
        advanced.episode_start,
        observation.actions_mask,
        method="step",
    ).value  # (B,)
    advantage, return_target = gae(
        trajectory.reward, trajectory.episode_end, trajectory.value, next_value, config.gamma, config.gae_lambda
    )  # each (T, B)
    trajectory, advantage, return_target = jax.tree.map(
        lambda leaf: jnp.swapaxes(leaf, 0, 1), (trajectory, advantage, return_target)
    )  # trajectory leaves: (B, T, ...); advantage/return_target: (B, T)
    batch = PPOBatch(
        observation=trajectory.observation,  # (B, T, O)
        actions_mask=trajectory.actions_mask,  # (B, T, *C, A)
        action=trajectory.action,  # (B, T, *C)
        old_log_prob=trajectory.log_prob,  # (B, T)
        value=trajectory.value,  # (B, T)
        advantage=advantage,  # (B, T)
        return_target=return_target,  # (B, T)
        model_carry=rollout_carry.model_carry,  # leaves: (B, H)
        episode_start=trajectory.episode_start,  # (B, T)
    )
    return advanced, RolloutResult(
        batch=batch,  # observation: (B, T, O); targets/starts: (B, T); model_carry leaves: (B, H)
        key=key,  # ()
        episode_return_sum=episode_metrics.episode_return_sum,  # ()
        episode_length_sum=episode_metrics.episode_length_sum,  # ()
        episode_count=episode_metrics.episode_count,  # ()
    )


def train_step(
    rollout_carry: RolloutCarry,  # observation: (E, N, O); state: (E, ...); memory: (B, H)
    train_state: TrainState,
    env: PPOEnv,
    config: Config,
    key: jax.Array,  # () typed PRNG key
    metrics: TrainingMetrics,  # scalar leaves
    environment_step_count: int | jax.Array = 0,  # () completed environment transitions before this iteration
) -> TrainStepResult:
    """Collect a rollout, run PPO epochs, and accumulate window/lifetime metrics."""
    rollout_carry, rollout = collect_rollout(rollout_carry, train_state, env, config, key)
    batch = rollout.batch  # observation: (B, T, O); targets/starts: (B, T); model_carry leaves: (B, H)
    learning_rate = jnp.asarray(learning_rate_schedule(config)(environment_step_count), dtype=jnp.float32)  # ()
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
        minibatch = batch.select_sequences(index)  # sequences: (M, T, ...); memory leaves: (M, H)
        train_state, metrics = update(current.train_state, minibatch, config, environment_step_count)
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
    return TrainStepResult(
        train_state=updated.train_state,
        rollout_carry=rollout_carry,  # state: (E, ...); memory: (B, H)
        key=updated.key,  # ()
        metrics=metrics,  # scalar leaves
    )


# Compile the rollout and updates together; only the training loop relinquishes ownership.
train_step_donated = jax.jit(train_step, static_argnames=("env", "config"), donate_argnames=("train_state",))


def train(config: Config) -> TrainState:
    """Train with a fully resolved config, as returned by load_config."""
    with ExitStack() as resources:
        configure_compilation_cache()
        batch_size = config.num_envs * config.num_steps
        env = make_env(config)
        manager = resources.enter_context(closing(checkpoint_manager(config.ckpt_dir)))
        key, init_key, reset_key = jax.random.split(jax.random.key(config.seed), 3)  # each () typed PRNG key
        reset_key = jax.random.split(reset_key, config.num_envs)  # (E,) typed PRNG keys
        observation, env_state = jax.vmap(env.reset)(reset_key)  # (E, N, O)/(E, N, *C, A); state leaves: (E, ...)
        model = ActorCritic(
            action_space=env.action_space,
            model=config.model,
            dtype=jnp.bfloat16 if config.bf16 else jnp.float32,
        )
        actor_count = config.num_envs * env.num_agents
        model_carry = model.initial_carry(actor_count)  # leaves: (B, H)
        episode_start = jnp.ones(actor_count, dtype=bool)  # (B,)
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
                observation.observation[:1, :1],  # (B=1, T=1, O)
                model.initial_carry(1),  # leaves: (B=1, H)
                episode_start[:1, None],  # (B=1, T=1)
                observation.actions_mask[:1, :1],  # (B=1, T=1, *C, A)
            )["params"],
            tx=optimizer(config.learning_rate),
        )
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
            reset_key = jax.random.split(jax.random.fold_in(key, start_rollout_index), config.num_envs)  # (E,) keys
            observation, env_state = jax.vmap(env.reset)(reset_key)  # (E, N, O)/(E, N, *C, A); state leaves: (E, ...)
        initial_step_count = (
            progress.environment_step_count
            if progress is not None and progress.environment_step_count is not None
            else start_rollout_index * batch_size
        )
        observation_count = (
            progress.observation_count
            if progress is not None and progress.observation_count is not None
            else initial_step_count * env.num_agents
        )
        writer = MLflowLogger(experiment="mappo", run_name=config.run_id)
        resources.enter_context(writer)
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
        rollout_carry = RolloutCarry(
            env_state=env_state,  # leaves: (E, ...)
            observation=observation,  # (E, N, O)/(E, N, *C, A)
            model_carry=model_carry,  # leaves: (B, H)
            episode_start=episode_start,  # (B,)
            episode_return=jnp.zeros(config.num_envs, dtype=jnp.float32),  # (E,)
            episode_step_count=jnp.zeros(config.num_envs, dtype=jnp.int32),  # (E,)
        )

        start = window_start = monotonic()
        window_start_step_count = initial_step_count
        remaining_rollout_count = max(0, (config.total_steps - initial_step_count) // batch_size)
        planned_rollout_count = start_rollout_index + remaining_rollout_count
        step_count = initial_step_count
        for rollout_index in range(start_rollout_index, planned_rollout_count):
            # Schedules need a fraction, while exact lifetime counters stay as host integers.
            schedule_step = jnp.asarray(step_count, dtype=jnp.float32)  # (); avoids JIT's default int32 limit
            train_state, rollout_carry, key, metrics = train_step_donated(
                rollout_carry, train_state, env, config, key, metrics, schedule_step
            )  # state: (E, ...); memory: (B, H); key/metrics: ()
            rollout_count = rollout_index + 1
            is_final_rollout = rollout_count == planned_rollout_count
            step_count += batch_size
            observation_count += batch_size * env.num_agents
            if config.video_interval and (rollout_count % config.video_interval == 0 or is_final_rollout):
                log_video(
                    train_state,
                    config,
                    writer,
                    int(metrics.total.episode_count),
                    step_count,
                    initial_carry=model.initial_carry(env.num_agents),
                    env=env,
                )
            if config.eval_interval and (rollout_count % config.eval_interval == 0 or is_final_rollout):
                log_evaluation(
                    train_state,
                    config,
                    writer,
                    int(metrics.total.episode_count),
                    step_count,
                    initial_carry=model.initial_carry(env.num_agents),
                    env=env,
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
                # Include evaluation/video overhead in this session's elapsed time.
                elapsed_seconds = now - start
                scalars: dict[str, float | int] = {
                    "train/updates_per_iteration_mean": int(window.update_count) / window_rollout_count,
                    "train/policy_loss": mean_metrics["policy_loss"],
                    "train/value_loss": mean_metrics["value_loss"],
                    "train/total_loss": mean_metrics["total_loss"],
                    "train/entropy": mean_metrics["entropy"],
                    "train/approx_kl": mean_metrics["approx_kl"],
                    "train/clip_fraction": mean_metrics["clip_fraction"],
                    "train/iterations_per_second": window_rollout_count / rate_seconds,
                    "train/env_steps_per_second": sps,
                    "train/observations_per_second": sps * env.num_agents,
                    "train/episodes_per_second": window_episode_count / rate_seconds,
                    "train/learning_rate": learning_rate,
                    "train/entropy_coef": entropy_coef,
                    "train/episodes_total": episode_count,
                    "train/env_steps_total": step_count,
                    "train/observations_total": observation_count,
                    "train/iterations_total": rollout_count,
                    "train/updates_total": int(train_state.step),
                    "train/progress_fraction": min(step_count / config.total_steps, 1.0),
                    "rollout/episode_count": window_episode_count,
                    "train/elapsed_seconds": elapsed_seconds,
                }
                if config.target_kl is not None:
                    scalars["train/kl_early_stop_fraction"] = int(window.early_stop_count) / window_rollout_count
                if int(window.explained_variance_count):
                    scalars["train/explained_variance_mean"] = float(window.explained_variance_sum) / int(
                        window.explained_variance_count
                    )
                if window_episode_count:
                    scalars["rollout/return_mean"] = float(window.episodes.episode_return_sum) / window_episode_count
                    scalars["rollout/episode_length_steps_mean"] = (
                        int(window.episodes.episode_length_sum) / window_episode_count
                    )
                if episode_count:
                    scalars["train/return_mean"] = float(total.episode_return_sum) / episode_count
                    scalars["train/episode_length_steps_mean"] = int(total.episode_length_sum) / episode_count
                writer.log_metrics(scalars, step_count)
                episode_return_text = f"{scalars['rollout/return_mean']:.1f}" if window_episode_count else "n/a"
                episode_length_text = (
                    f"{scalars['rollout/episode_length_steps_mean']:.1f}" if window_episode_count else "n/a"
                )
                explained_variance_text = (
                    f"{scalars['train/explained_variance_mean']:.3f}"
                    if "train/explained_variance_mean" in scalars
                    else "n/a"
                )
                print(
                    f"iteration={rollout_count} step={step_count} episodes={episode_count} return={episode_return_text} "
                    f"episode_length={episode_length_text} "
                    f"progress={scalars['train/progress_fraction']:.3f} "
                    f"iterations/s={scalars['train/iterations_per_second']:.3f} "
                    f"env_steps/s={sps:.3f} episodes/s={scalars['train/episodes_per_second']:.3f} "
                    f"lr={learning_rate:.3g} "
                    f"policy={mean_metrics['policy_loss']:.3f} value={mean_metrics['value_loss']:.3f} "
                    f"entropy={mean_metrics['entropy']:.3f} kl={mean_metrics['approx_kl']:.4f} "
                    f"clipfrac={mean_metrics['clip_fraction']:.3f} ev={explained_variance_text} "
                    f"updates={scalars['train/updates_per_iteration_mean']:.2f} "
                    f"window={window_seconds:.1f}s "
                    f"elapsed={timedelta(seconds=int(elapsed_seconds))}",
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
                        observation_count=observation_count,
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
