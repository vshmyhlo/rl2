"""Recurrent parameter-sharing IPPO for SMAX and JaxNav, with local actor and critic inputs.

The actor and critic share an encoder and LSTM, with separate policy and value
heads and parameters shared across agents. Both use only local observations.
Unavailable actions are masked during rollout collection and PPO updates.
SMAX agent death or JaxNav collision resets local recurrence but does not end
the team's reward stream or GAE trace.
Environment completion ends traces; time limits bootstrap from final local observations
before autoreset, unless the episode also terminated.

E denotes environments, N agents, B = E * N agent sequences, T rollout steps,
O local observation features, A discrete actions, and H recurrent width.
Environment states and episode statistics use E; model batches use B.
Recurrent minibatches preserve entire agent sequences.

Internal counters use *_count (batched counters: *_counts), zero-based positions
use *_index, and accumulated values use *_sum or *_sums for pytrees.
Configuration sizes retain num_* names; checkpoint keys and log tags remain stable.

total_steps and MLflow's metric step count environment transitions.
Observation counts include every agent, so each environment step contributes N
observations. Episodes count completed environments, not individual agent outcomes.
Team returns sum mean agent rewards over an episode, counting SMAX's shared
reward once. JaxNav uses its native mixture of individual and team rewards.
Metric names use total_ for all-training statistics and episodes
for episode counts. Window statistics have no scope affix.

Progress scalars computed over the logging window:
    losses/policy: PPO policy loss.
    losses/value: Value prediction loss before applying value_coef.
    losses/entropy: Negative action distribution entropy before applying entropy_coef.
    losses/total: Policy loss + value_coef * value loss + weighted entropy loss.
    policy/entropy: Masked action distribution entropy.
    policy/approx_kl: Approximate KL divergence from the rollout policy.
    policy/clip_fraction: Fraction of probability ratios outside the clip range.
    policy/early_stop: Fraction of rollouts stopped early by the KL threshold.
    value/explained_variance: Rollout value predictions versus GAE returns.
    charts/updates_per_rollout: Mean number of accepted minibatch updates.
    charts/iterations_per_second: Rollouts and their PPO updates per second.
    charts/environment_steps_per_second: Environment transitions per second across all environments.
    charts/observations_per_second: Agent observations per second across all environments and agents.
    charts/episodes_per_second: Completed training episodes per second.
    charts/episodes: Training episodes completed in the window.
    charts/return_mean: Mean raw team return of episodes completed in the window.
    charts/episode_length_mean: Mean episode length in environment steps.

Progress scalars computed across all training:
    charts/training_progress: Fraction of planned rollout iterations completed, from 0 to 1.
    charts/total_iterations: Cumulative completed rollouts, including resumed training.
    charts/total_updates: Cumulative accepted optimizer updates, including resumed training.
    charts/total_episodes: Cumulative completed training episodes.
    charts/total_environment_steps: Cumulative training environment transitions.
    charts/total_observations: Cumulative agent observations across all environments and agents.
    charts/total_return_mean: Mean raw team return over all completed training episodes.
    charts/total_episode_length_mean: Mean length over all completed training episodes, in environment steps.

Progress scalars reporting the latest rollout's settings:
    charts/learning_rate: Current learning rate.
    charts/entropy_coef: Current entropy bonus coefficient.

Session timing:
    time/elapsed_seconds: Wall time since this training session's rollout loop started.
    time/video_seconds: Periodic gameplay recording, encoding, and upload wall time.
    time/checkpoint_seconds: Checkpoint save wall time, including completion of writes.

Startup scalars (also logged when resuming):
    model/params_millions: Model parameter count in millions.

Evaluation scalars (when evaluation is enabled; unavailable scores are omitted):
    eval/return_mean: Mean raw evaluation team return.
    eval/return_std: Sample standard deviation of raw evaluation team returns.
    eval/return_sem: Standard error of the mean raw evaluation team return.
    eval/total_training_episodes: Completed training episodes at evaluation time.
    time/evaluation_seconds: Evaluation wall time, including gameplay and report uploads.

Artifacts (indexed by training step):
    config/<step>.yaml: Full training configuration as YAML, logged at startup and on resume.
    devices/<step>.txt: JAX devices used by this session.
    model/parameter_count/<step>.txt: Exact model parameter count.
    eval/report/<step>.json: Complete JSON report after each evaluation, including per-episode
        results, aggregate scores, training configuration, evaluation seed,
        and training environment steps/completed episodes.

Video/images:
    gameplay: RGB gameplay from a separate episode under the current policy,
        rendered from environment state and saved as gameplay/<step>.mp4.
        Recording ends at episode termination/truncation or video_max_frames
        (default: 900; None removes the frame cap). Playback uses video_fps frames per second.
    eval/gameplay: RGB video of the last episode in each evaluation run, using
        the same frame cap and playback speed. Evaluation finishes the full
        episode even when recording reaches the frame cap.
    No standalone still images, audio, histograms, embeddings, or graphs are logged.

Episode means are omitted when no corresponding episodes or history are available.
Evaluation uses fixed seeds; return_std and return_sem are omitted for one episode.

Logging windows span log_interval iterations (default: 60); progress metrics are
emitted at scheduled rollout boundaries and on normal completion. Losses, entropy,
KL, and clip fraction average over evaluated minibatches, including KL-rejected
updates. Explained variance, updates per rollout, and early-stop frequency average
over rollouts. Throughput uses window wall time, including intervening logging,
evaluation, video, and checkpoint overhead. Episode window means include only
episodes completed within that window. Learning rate and entropy coefficient report
the latest rollout's settings.

Evaluation and video schedules are independent of progress logging. Videos are
recorded every video_interval iterations (default: 600; 0 disables periodic
recording). Evaluation uses eval_interval (default: 0, disabled) and records its
last episode independently of video_interval. Checkpoints use checkpoint_interval
(default: 600). One iteration is a rollout and its PPO updates. All schedules use
the total completed iteration count, including on resume. Enabled evaluation,
logging, and checkpoints also run on normal completion. Defaults here refer to
Config; YAML configurations may override them.

Checkpoints retain parameters, optimizer state, RNG, completed iterations, and
cumulative episode statistics. Live environments and recurrent memory restart
with fresh episodes on resume; unfinished episode statistics are discarded.
"""

import argparse
import json
import os
from collections.abc import Callable
from contextlib import ExitStack, closing
from dataclasses import asdict, fields
from datetime import timedelta
from functools import partial
from math import sqrt
from operator import itemgetter
from pathlib import Path
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
from numpy.typing import NDArray
from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass

from rl2.actions import CategoricalPolicy, DiscreteSpace
from rl2.configuration import load_settings
from rl2.envs.jaxnav_env import JaxNavConfig, JaxNavEnv
from rl2.envs.multi_agent import AgentObservation
from rl2.envs.smax_env import SMAXConfig, SMAXEnv
from rl2.jax_cache import configure_compilation_cache
from rl2.lstm import LSTMStack, LSTMStackCarry
from rl2.mlflow_logging import MLflowLogger
from rl2.running_stats import RunningSum
from rl2.utils import explained_variance, gae

type RecurrentCarry = LSTMStackCarry
type PPOEnv = SMAXEnv | JaxNavEnv


@struct.dataclass
class PPOBatch:
    """PPO inputs; S is (T, B) for sequences or (B,) for step-only policies.

    Attributes:
        observations: (*S, O) local observation features.
        action_masks: (*S, A) boolean legal-action flags.
        actions: (*S,) integer action IDs.
        old_log_probs: (*S,) rollout action log probabilities.
        values: (*S,) rollout-time value predictions for explained variance.
        advantages: (*S,) unnormalized advantage estimates.
        returns: (*S,) value targets.
        model_carry: (B, H) initial LSTM memory leaves, where H is model.recurrent.hidden_size.
        episode_starts: (*S,) boolean agent or environment reset flags.
    """

    observations: jax.Array  # (*S, O)
    action_masks: jax.Array  # (*S, A) bool
    actions: jax.Array  # (*S,) integer action IDs
    old_log_probs: jax.Array  # (*S,)
    values: jax.Array  # (*S,)
    advantages: jax.Array  # (*S,)
    returns: jax.Array  # (*S,)
    model_carry: RecurrentCarry  # leaves: (B, H), H is model.recurrent.hidden_size
    episode_starts: jax.Array  # (*S,)

    def select_sequences(self, indices: jax.Array) -> Self:  # (M,) agent-sequence indices
        """Return (T, M, ...) sequence fields and (M, H) initial memory leaves."""
        return self.replace(
            **jax.tree.map(
                itemgetter((slice(None), indices)),
                {field.name: getattr(self, field.name) for field in fields(self) if field.name != "model_carry"},
            ),  # observations: (T, M, O); masks: (T, M, A); other sequence fields: (T, M)
            model_carry=jax.tree.map(itemgetter(indices), self.model_carry),  # leaves: (M, H)
        )


class RolloutCarry(NamedTuple):
    """Live environment state and recurrent memory between rollouts.

    Attributes:
        env_state: (E, ...) batched environment state leaves.
        observations: (E, N, O) local features and (E, N, A) legal-action masks.
        model_carry: (B, H) LSTM memory leaves, where H is model.recurrent.hidden_size.
        episode_starts: (B,) boolean agent or environment reset flags.
        episode_returns: (E,) accumulated raw team rewards for unfinished episodes.
        episode_step_counts: (E,) step counts for unfinished episodes.
    """

    env_state: chex.ArrayTree  # leaves: (E, ...)
    observations: AgentObservation  # observations: (E, N, O); masks: (E, N, A)
    model_carry: RecurrentCarry  # leaves: (B, H)
    episode_starts: jax.Array  # (B,) reset on agent death or environment completion
    episode_returns: jax.Array  # (E,) unfinished team returns
    episode_step_counts: jax.Array  # (E,) unfinished episode lengths


class RolloutResult(NamedTuple):
    """Collected training data, advanced RNG key, and completed episode statistics.

    Attributes:
        batch: (T, B, ...) time-major PPO arrays with (B, H) initial memory leaves.
        key: () advanced typed PRNG key, also used for minibatch shuffling.
        episode_return_sum: () sum of raw returns for completed episodes.
        episode_length_sum: () sum of lengths for completed episodes.
        episode_count: () number of completed episodes.
    """

    batch: PPOBatch  # observations: (T, B, O); targets/starts: (T, B); model_carry leaves: (B, H)
    key: jax.Array  # () typed PRNG key
    episode_return_sum: jax.Array  # ()
    episode_length_sum: jax.Array  # ()
    episode_count: jax.Array  # ()


@struct.dataclass
class MinibatchMetrics:
    """Scalar diagnostics for one evaluated minibatch, including KL-rejected updates."""

    policy_loss: jax.Array  # ()
    value_loss: jax.Array  # ()
    entropy_loss: jax.Array  # ()
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
    rollout_carry: RolloutCarry  # observations: (E, N, O); state: (E, ...); memory: (B, H)
    key: jax.Array  # () typed PRNG key
    metrics: TrainingMetrics  # scalar leaves


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LSTMConfig:
    type: Literal["lstm"] = "lstm"
    hidden_size: Annotated[int, Field(gt=0, strict=True)] = 128
    num_layers: Annotated[int, Field(gt=0, strict=True)] = 1
    intermediate_size: Annotated[int, Field(ge=0, strict=True)] = 0


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class ModelConfig:
    hidden_size: Annotated[int, Field(gt=0, strict=True)] = 128
    recurrent: LSTMConfig = LSTMConfig()


@dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class Config:
    env_id: Literal["SMAX", "JaxNav"]
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
    ckpt_dir: str
    video_fps: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    bf16: bool = True
    video_max_frames: Annotated[int, Field(gt=0)] | None = 900
    video_interval: Annotated[int, Field(ge=0)] = 600
    eval_interval: Annotated[int, Field(ge=0)] = 0
    eval_episodes: Annotated[int, Field(gt=0)] = 100
    eval_seed: Annotated[int, Field(ge=0, lt=2**32)] = 10_000
    model: ModelConfig = ModelConfig()
    lr_decay: Literal["linear", "cosine"] = "linear"
    entropy_decay: Literal["constant", "cosine"] = "constant"
    run_id: str | None = None
    checkpoint_interval: Annotated[int, Field(gt=0)] = 600
    log_interval: Annotated[int, Field(gt=0)] = 60
    smax: SMAXConfig = SMAXConfig()
    jaxnav: JaxNavConfig = JaxNavConfig()

    def __post_init__(self) -> None:
        if self.run_id is not None and (
            not self.run_id.strip() or self.run_id in (".", "..") or any(c in self.run_id for c in "/\\")
        ):
            raise ValueError("run_id must be a nonempty directory name without slashes or traversal")

    @model_validator(mode="after")
    def validate_rollout(self) -> Self:
        if self.env_id != "SMAX" and self.smax != SMAXConfig():
            raise ValueError("smax settings require env_id: SMAX")
        if self.env_id != "JaxNav" and self.jaxnav != JaxNavConfig():
            raise ValueError("jaxnav settings require env_id: JaxNav")
        if self.total_steps < self.num_envs * self.num_steps:
            raise ValueError("total_steps must cover at least one rollout")
        return self


def make_env(config: Config) -> PPOEnv:
    env = SMAXEnv(config.smax) if config.env_id == "SMAX" else JaxNavEnv(config.jaxnav)
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
                    "algorithm": f"{config.env_id.lower()}_ippo",
                    "config": asdict(config),
                    "iteration": progress.rollout_count,
                    "completed_episodes": progress.episode_count,
                    "training_return_sum": progress.episode_return_sum,
                    "training_length_sum": progress.episode_length_sum,
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
    if metadata["version"] != 2 or metadata.get("algorithm") != f"{config.env_id.lower()}_ippo":
        raise ValueError("Unsupported PPO checkpoint version; the shared encoder/LSTM requires a new run directory")
    if metadata["iteration"] != rollout_count or rollout_count < 0 or metadata["completed_episodes"] < 0:
        raise ValueError("Invalid PPO checkpoint progress")
    saved_config = metadata["config"]
    if saved_config.get("env_id", config.env_id) != config.env_id:
        raise ValueError("Cannot resume PPO with a different environment; use a new run directory")
    if "model" in saved_config and ModelConfig(**saved_config["model"]) != config.model:
        raise ValueError("Cannot resume PPO with different model settings; use a new run directory")
    if SMAXConfig(**saved_config["smax"]) != config.smax:
        raise ValueError("Cannot resume PPO with different SMAX settings; use a new run directory")
    if JaxNavConfig(**saved_config.get("jaxnav", {})) != config.jaxnav:
        raise ValueError("Cannot resume PPO with different JaxNav settings; use a new run directory")
    # Progress and schedules count rollouts, so each must still represent the same number of steps.
    if (
        "num_envs" in saved_config
        and "num_steps" in saved_config
        and saved_config["num_envs"] * saved_config["num_steps"] != config.num_envs * config.num_steps
    ):
        raise ValueError("Cannot resume PPO with a different num_envs * num_steps; use a new run directory")
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
    )


def load_config(path: str | Path) -> Config:
    """Load, resolve all interpolations and the run ID, and validate a training config."""
    return Config(**load_settings(path))


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
    policy: CategoricalPolicy  # logits: (*S, A)
    values: jax.Array  # (*S,)


class ActorCritic(nn.Module):
    """Shared local observation encoder/LSTM with separate policy and value heads."""

    action_space: DiscreteSpace
    model: ModelConfig = ModelConfig()
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
        self.encoder = nn.Dense(features=self.model.hidden_size, dtype=self.dtype, name="encoder")
        self.recurrent = self._make_recurrent(parent=self)
        self.recurrent_input = nn.Dense(features=self.model.recurrent.hidden_size, dtype=self.dtype, name="lstm_input")
        init = nn.initializers.orthogonal(sqrt(2))
        self.policy_hidden = nn.Dense(features=512, kernel_init=init, name="policy_hidden", dtype=self.dtype)
        self.policy_norm = nn.LayerNorm(name="policy_norm", dtype=self.dtype)
        self.value_hidden = nn.Dense(features=512, kernel_init=init, name="value_hidden", dtype=self.dtype)
        self.value_norm = nn.LayerNorm(name="value_norm", dtype=self.dtype)
        self.policy_output = nn.Dense(
            features=self.action_space.n,
            kernel_init=nn.initializers.orthogonal(0.01),
            name="policy_output",
            dtype=self.dtype,
        )
        self.value_output = nn.Dense(
            features=1, kernel_init=nn.initializers.orthogonal(1.0), name="value_output", dtype=self.dtype
        )

    def _encode(self, obs: jax.Array) -> jax.Array:  # (*S, O) -> (*S, H)
        x = nn.relu(self.encoder(obs))  # (*S, F), F is model.hidden_size
        return self.recurrent_input(x).astype(self.dtype)  # (*S, H)

    def _heads(
        self,
        x: jax.Array,  # (*S, H)
        action_masks: jax.Array,  # (*S, A)
    ) -> tuple[CategoricalPolicy, jax.Array]:  # logits: (*S, A); values: (*S,)
        x = x.astype(self.dtype)  # (*S, H), S is (B,) or (T, B)
        policy = nn.relu(self.policy_norm(self.policy_hidden(x)))  # (*S, 512)
        critic = nn.relu(self.value_norm(self.value_hidden(x)))  # (*S, 512)
        # Heads use the compute dtype; float32 outputs keep PPO loss arithmetic precise.
        logits = self.policy_output(policy).astype(jnp.float32)  # (*S, A)
        logits = jnp.where(action_masks, logits, -1e10)  # (*S, A)
        values = self.value_output(critic).squeeze(-1).astype(jnp.float32)  # (*S,)
        return CategoricalPolicy(logits), values  # logits: (*S, A); values: (*S,)

    def __call__(
        self,
        obs: jax.Array,  # (T, B, O)
        model_carry: RecurrentCarry,  # leaves: (B, H)
        episode_starts: jax.Array,  # (T, B)
        action_masks: jax.Array,  # (T, B, A)
    ) -> ActorCriticOutput:  # carry leaves: (B, H); logits: (T, B, A); values: (T, B)
        """Process observation sequences, resetting memory at episode starts."""
        x = self._encode(obs)  # (T, B, H)
        # Batch the encoder and heads across time; the LSTM handles its recurrence.
        model_carry, x = self.recurrent(x, model_carry, episode_starts)  # model_carry leaves: (B, H); x: (T, B, H)
        policy, values = self._heads(x, action_masks)  # logits: (T, B, A); (T, B)
        return ActorCriticOutput(carry=model_carry, policy=policy, values=values)

    def step(
        self,
        obs: jax.Array,  # (B, O)
        model_carry: RecurrentCarry,  # leaves: (B, H)
        episode_starts: jax.Array,  # (B,)
        action_masks: jax.Array,  # (B, A)
    ) -> ActorCriticOutput:  # carry leaves: (B, H); logits: (B, A); values: (B,)
        """Process one observation per sequence, resetting memory at episode starts."""
        x = self._encode(obs)  # (B, H)
        model_carry, x = self.recurrent.step(x, model_carry, episode_starts)  # model_carry leaves: (B, H); x: (B, H)
        policy, values = self._heads(x, action_masks)  # logits: (B, A); (B,)
        return ActorCriticOutput(carry=model_carry, policy=policy, values=values)


@partial(jax.jit, static_argnames="apply_fn")
def sample_policy_step(
    apply_fn: Callable[..., ActorCriticOutput],
    params: optax.Params,
    observations: jax.Array,  # (B, O)
    carry: RecurrentCarry,  # leaves: (B, H)
    episode_starts: jax.Array,  # (B,)
    key: jax.Array,  # () typed PRNG key
    action_masks: jax.Array,  # (B, A)
) -> tuple[RecurrentCarry, jax.Array]:  # carry leaves: (B, H); actions: (B,)
    """Compile inference and sampling together; reuse across playback steps and episodes."""
    carry, policy, _ = apply_fn(
        {"params": params}, observations, carry, episode_starts, action_masks, method="step"
    )  # carry leaves: (B, H); logits: (B, A); values: (B,)
    return carry, policy.sample(seed=key)  # carry leaves: (B, H); actions: (B,)


class EpisodeResult(NamedTuple):
    episode_return: float
    length: int
    frames: list[NDArray[np.uint8]]  # each (H, W, 3)


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
    """Run one episode, optionally recording (H, W, 3) uint8 RGB frames."""
    key, reset_key = jax.random.split(key)  # each () typed PRNG key
    observation, env_state = env.reset(reset_key)  # (N, O)/(N, A); state leaves: (...)
    memory = initial_carry  # leaves: (N, H)
    episode_starts = jnp.ones(env.num_agents, dtype=bool)  # (N,)
    frames: list[NDArray[np.uint8]] = []  # entries: (H, W, 3)
    if record:
        frames.append(np.asarray(env.render(env_state)))
    episode_return = 0.0
    episode_step_count = 0
    while not (stop_at_frame_limit and config.video_max_frames is not None and len(frames) >= config.video_max_frames):
        key, action_key, env_key = jax.random.split(key, 3)  # each () typed PRNG key
        memory, actions = sample_policy_step(
            train_state.apply_fn,
            train_state.params,
            observation.observations,
            memory,
            episode_starts,
            action_key,
            observation.action_masks,
        )  # memory leaves: (N, H); actions: (N,)
        result = env.step(env_key, env_state, actions)
        observation, env_state = result.observation, result.state  # (N, O)/(N, A); state leaves: (...)
        episode_starts = result.agent_dones  # (N,)
        episode_return += float(result.rewards.mean())
        episode_step_count += 1
        if record and (config.video_max_frames is None or len(frames) < config.video_max_frames):
            frames.append(np.asarray(env.render(result.final_state)))
        if bool(result.done):
            break
    return EpisodeResult(episode_return=episode_return, length=episode_step_count, frames=frames)


def log_video(
    train_state: TrainState,
    config: Config,
    writer: MLflowLogger,
    episode_count: int,
    step_count: int,
    *,
    initial_carry: RecurrentCarry,  # leaves: (N, H)
) -> None:
    """Record a separate episode up to the frame cap and log its gameplay video."""
    print(f"Recording gameplay at step {step_count} after {episode_count} training episodes...", flush=True)
    started = monotonic()
    _, _, frames = play_episode(
        train_state,
        config,
        jax.random.fold_in(jax.random.key(config.seed), episode_count),
        initial_carry,
        record=True,
        stop_at_frame_limit=True,
        env=make_env(config),
    )
    writer.log_video(frames, f"gameplay/{step_count}.mp4", fps=config.video_fps)
    seconds = monotonic() - started
    writer.log_metrics({"time/video_seconds": seconds}, step_count)
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
) -> None:
    """Evaluate fixed environment seeds, log raw scores, and record the last episode."""
    print(f"Evaluating {config.eval_episodes} games at step {step_count}...", flush=True)
    started = monotonic()
    episode_results: list[dict[str, float | int]] = []
    env = make_env(config)
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
            writer.log_video(frames, f"eval/gameplay/{step_count}.mp4", fps=config.video_fps)
    returns = np.array([result["return"] for result in episode_results])  # (E,), E evaluation episodes
    std = float(returns.std(ddof=1)) if config.eval_episodes > 1 else None
    result = {
        "env_id": config.env_id,
        "return_mean": float(returns.mean()),
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
    scalars["eval/total_training_episodes"] = episode_count
    writer.log_text(json.dumps(result, indent=2, allow_nan=False), f"eval/report/{step_count}.json")
    seconds = monotonic() - started
    scalars["time/evaluation_seconds"] = seconds
    writer.log_metrics(scalars, step_count)
    print(
        f"Evaluation: return={result['return_mean']:.1f} over {config.eval_episodes} games in {seconds:.1f}s",
        flush=True,
    )


def update(
    train_state: TrainState,
    batch: PPOBatch,  # observations: (*S, O); masks: (*S, A); targets/starts: (*S,); model_carry leaves: (B, H)
    config: Config,
    rollout_index: int | jax.Array = 0,  # () zero-based rollout position for schedules
) -> tuple[TrainState, MinibatchMetrics]:
    rollout_index = jnp.asarray(rollout_index)  # ()
    chex.assert_type(rollout_index, int)
    entropy_coef = jnp.asarray(entropy_coef_schedule(config)(rollout_index), dtype=jnp.float32)  # ()
    advantages = batch.advantages  # (*S,)
    chex.assert_scalar_positive(advantages.size)
    # Centering a single sample always gives zero and erases its policy gradient.
    if advantages.size > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # (*S,)

    def loss_fn(params: optax.Params) -> tuple[jax.Array, MinibatchMetrics]:  # (); scalar metric leaves
        _, policy, values = train_state.apply_fn(
            {"params": params}, batch.observations, batch.model_carry, batch.episode_starts, batch.action_masks
        )  # model_carry leaves: (B, H); policy logits: (*S, A); values: (*S,)
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

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
    if config.target_kl is None:
        return train_state.apply_gradients(grads=grads), metrics
    train_state = jax.lax.cond(
        metrics.approx_kl > config.target_kl,
        lambda: train_state,
        lambda: train_state.apply_gradients(grads=grads),
    )
    return train_state, metrics


def collect_rollout(
    rollout_carry: RolloutCarry,  # observations: (E, N, O); state: (E, ...); memory: (B, H)
    train_state: TrainState,
    envs: PPOEnv,
    config: Config,
    key: jax.Array,  # () typed PRNG key
) -> tuple[RolloutCarry, RolloutResult]:
    """Collect (T, B, ...) transitions entirely on device and compute GAE.

    Timeouts bootstrap from final observations; agent end flags reset recurrence.
    Team rewards and GAE traces continue until environment completion.
    Completed episode returns, lengths, and count are reduced to scalars on device.
    """
    actor_count = config.num_envs * envs.num_agents

    class Runner(NamedTuple):
        current: RolloutCarry  # state: (E, ...); memory: (B, H)
        key: jax.Array  # () typed PRNG key
        episode_metrics: EpisodeMetrics  # scalar leaves

    class Transition(NamedTuple):
        """S is (B,) per step or (T, B) after scanning."""

        observations: jax.Array  # (*S, O)
        action_masks: jax.Array  # (*S, A)
        actions: jax.Array  # (*S,)
        log_probs: jax.Array  # (*S,)
        values: jax.Array  # (*S,)
        rewards: jax.Array  # (*S,)
        episode_ends: jax.Array  # (*S,)
        episode_starts: jax.Array  # (*S,)

    def flatten(array: jax.Array) -> jax.Array:  # (E, N, ...) -> (B, ...)
        return array.reshape((actor_count, *array.shape[2:]))  # (B, ...)

    def step(
        runner: Runner,  # state: (E, ...); memory: (B, H); key/completed totals: ()
        _: None,
    ) -> tuple[Runner, Transition]:
        current, key, episode_metrics = runner  # state: (E, ...); memory: (B, H); key/completed totals: ()
        key, action_key, env_key = jax.random.split(key, 3)  # each () typed PRNG key
        observation = jax.tree.map(flatten, current.observations)  # (B, O)/(B, A)
        memory, policy, predictions = train_state.apply_fn(
            {"params": train_state.params},
            observation.observations,
            current.model_carry,
            current.episode_starts,
            observation.action_masks,
            method="step",
        )  # memory leaves: (B, H); logits: (B, A); predictions: (B,)
        actions, log_probs = policy.sample_and_log_prob(seed=action_key)  # each (B,)
        env_keys = jax.random.split(env_key, config.num_envs)  # (E,) typed PRNG keys
        result = jax.vmap(envs.step)(
            env_keys,
            current.env_state,
            actions.reshape((config.num_envs, envs.num_agents)),  # (E, N)
        )
        timeout = jnp.repeat(result.truncated & ~result.terminated, envs.num_agents)  # (B,)
        episode_starts = result.agent_dones.reshape(actor_count)  # (B,)

        def bootstrap(_: None) -> jax.Array:  # (B,)
            final = jax.tree.map(flatten, result.final_observation)  # (B, O)/(B, A)
            # Agent inactivity resets memory without ending the team's reward stream.
            resets = jax.vmap(envs.memory_resets)(result.final_state).reshape(actor_count)  # (B,)
            values = train_state.apply_fn(
                {"params": train_state.params},
                final.observations,
                memory,
                resets,
                final.action_masks,
                method="step",
            ).values  # (B,)
            return jnp.where(timeout, values, 0.0)  # (B,)

        def no_bootstrap(_: None) -> jax.Array:
            return jnp.zeros(actor_count)  # (B,)

        raw_rewards = result.rewards.reshape(actor_count)  # (B,)
        rewards = raw_rewards + config.gamma * jax.lax.cond(timeout.any(), bootstrap, no_bootstrap, None)  # (B,)
        episode_returns = current.episode_returns + result.rewards.mean(axis=-1)  # (E,)
        episode_step_counts = current.episode_step_counts + 1  # (E,)
        episode_metrics = RunningSum.update(
            episode_metrics,
            jax.tree.map(
                jnp.sum,
                EpisodeMetrics(
                    episode_count=result.done,  # (E,)
                    episode_return_sum=jnp.where(result.done, episode_returns, 0.0),  # (E,)
                    episode_length_sum=jnp.where(result.done, episode_step_counts, 0),  # (E,)
                ),
            ),
        )  # each ()
        advanced = RolloutCarry(
            env_state=result.state,  # leaves: (E, ...)
            observations=result.observation,  # observations: (E, N, O); masks: (E, N, A)
            model_carry=memory,  # leaves: (B, H)
            episode_starts=episode_starts,  # (B,)
            episode_returns=jnp.where(result.done, 0.0, episode_returns),  # (E,)
            episode_step_counts=jnp.where(result.done, 0, episode_step_counts),  # (E,)
        )
        return Runner(current=advanced, key=key, episode_metrics=episode_metrics), Transition(
            observations=observation.observations,  # (B, O)
            action_masks=observation.action_masks,  # (B, A)
            actions=actions,  # (B,)
            log_probs=log_probs,  # (B,)
            values=predictions,  # (B,)
            rewards=rewards,  # (B,)
            episode_ends=jnp.repeat(result.done, envs.num_agents),  # (B,)
            episode_starts=current.episode_starts,  # (B,)
        )

    episode_metrics = EpisodeMetrics(
        episode_count=jnp.zeros((), dtype=int),  # ()
        episode_return_sum=jnp.zeros((), dtype=rollout_carry.episode_returns.dtype),  # ()
        episode_length_sum=jnp.zeros((), dtype=int),  # ()
    )
    (advanced, key, episode_metrics), trajectory = jax.lax.scan(
        step, Runner(current=rollout_carry, key=key, episode_metrics=episode_metrics), None, length=config.num_steps
    )  # trajectory leaves: (T, B, ...)
    # Peek at the next value without advancing the rollout's recurrent state.
    observation = jax.tree.map(flatten, advanced.observations)  # (B, O)/(B, A)
    next_value = train_state.apply_fn(
        {"params": train_state.params},
        observation.observations,
        advanced.model_carry,
        advanced.episode_starts,
        observation.action_masks,
        method="step",
    ).values  # (B,)
    advantages, returns = gae(
        trajectory.rewards, trajectory.episode_ends, trajectory.values, next_value, config.gamma, config.gae_lambda
    )  # each (T, B)
    batch = PPOBatch(
        observations=trajectory.observations,  # (T, B, O)
        action_masks=trajectory.action_masks,  # (T, B, A)
        actions=trajectory.actions,  # (T, B)
        old_log_probs=trajectory.log_probs,  # (T, B)
        values=trajectory.values,  # (T, B)
        advantages=advantages,  # (T, B)
        returns=returns,  # (T, B)
        model_carry=rollout_carry.model_carry,  # leaves: (B, H)
        episode_starts=trajectory.episode_starts,  # (T, B)
    )
    return advanced, RolloutResult(
        batch=batch,  # observations: (T, B, O); targets/starts: (T, B); model_carry leaves: (B, H)
        key=key,  # ()
        episode_return_sum=episode_metrics.episode_return_sum,  # ()
        episode_length_sum=episode_metrics.episode_length_sum,  # ()
        episode_count=episode_metrics.episode_count,  # ()
    )


def train_step(
    rollout_carry: RolloutCarry,  # observations: (E, N, O); state: (E, ...); memory: (B, H)
    train_state: TrainState,
    envs: PPOEnv,
    config: Config,
    key: jax.Array,  # () typed PRNG key
    metrics: TrainingMetrics,  # scalar leaves
    rollout_index: int | jax.Array = 0,  # () zero-based rollout position for schedules
) -> TrainStepResult:
    """Collect a rollout, run PPO epochs, and accumulate window/lifetime metrics."""
    rollout_carry, rollout = collect_rollout(rollout_carry, train_state, envs, config, key)
    batch = rollout.batch  # observations: (T, B, O); targets/starts: (T, B); model_carry leaves: (B, H)
    learning_rate = jnp.asarray(learning_rate_schedule(config)(rollout_index), dtype=jnp.float32)  # ()
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

    def should_update(current: UpdateCarry) -> jax.Array:
        return (current.minibatch_count < config.update_epochs * config.num_minibatches) & ~current.kl_exceeded  # ()

    def update_minibatch(current: UpdateCarry) -> UpdateCarry:
        def shuffle(key: jax.Array) -> tuple[jax.Array, jax.Array]:  # () typed PRNG key
            key, shuffle_key = jax.random.split(key)  # each () typed PRNG key
            return key, jax.random.permutation(shuffle_key, batch.actions.shape[1])  # (); (B,)

        def keep_permutation(key: jax.Array) -> tuple[jax.Array, jax.Array]:  # () typed PRNG key
            return key, current.permutation  # (); (B,)

        minibatch_index = current.minibatch_count % config.num_minibatches  # ()
        key, permutation = jax.lax.cond(minibatch_index == 0, shuffle, keep_permutation, current.key)  # (); (B,)
        indices = permutation.reshape(config.num_minibatches, -1)[minibatch_index]  # (M,), M = B / minibatches
        minibatch = batch.select_sequences(indices)  # sequences: (T, M, ...); memory: (M, D)
        train_state, metrics = update(current.train_state, minibatch, config, rollout_index)
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
        should_update,
        update_minibatch,
        UpdateCarry(
            train_state=train_state,
            key=rollout.key,  # ()
            permutation=jnp.zeros(batch.actions.shape[1], dtype=jnp.int32),  # (B,)
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
    window = WindowMetrics(
        episodes=episodes,  # scalar leaves
        minibatch_metrics=updated.minibatch_metrics,  # scalar leaves
        explained_variance_sum=explained_variance(batch.values, batch.returns),  # ()
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
train_step_donated = jax.jit(train_step, static_argnames=("envs", "config"), donate_argnames=("train_state",))


def train(config: Config) -> TrainState:
    """Train with a fully resolved config, as returned by load_config."""
    with ExitStack() as resources:
        configure_compilation_cache()
        batch_size = config.num_envs * config.num_steps
        envs = make_env(config)
        manager = resources.enter_context(closing(checkpoint_manager(config.ckpt_dir)))
        key, init_key, reset_key = jax.random.split(jax.random.key(config.seed), 3)  # each () typed PRNG key
        reset_keys = jax.random.split(reset_key, config.num_envs)  # (E,) typed PRNG keys
        obs, env_state = jax.vmap(envs.reset)(reset_keys)  # (E, N, O)/(E, N, A); state leaves: (E, ...)
        model = ActorCritic(
            action_space=envs.action_space,
            model=config.model,
            dtype=jnp.bfloat16 if config.bf16 else jnp.float32,
        )
        actor_count = config.num_envs * envs.num_agents
        model_carry = model.initial_carry(actor_count)  # leaves: (B, H)
        episode_starts = jnp.ones(actor_count, dtype=bool)  # (B,)
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
                obs.observations[:1, :1],  # (T=1, B=1, O)
                model.initial_carry(1),  # leaves: (B=1, H)
                episode_starts[None, :1],  # (T=1, B=1)
                obs.action_masks[:1, :1],  # (T=1, B=1, A)
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
            reset_keys = jax.random.split(jax.random.fold_in(key, start_rollout_index), config.num_envs)  # (E,) keys
            obs, env_state = jax.vmap(envs.reset)(reset_keys)  # (E, N, O)/(E, N, A); state leaves: (E, ...)
            print(f"Resumed run {config.run_id} at step {start_rollout_index * batch_size}", flush=True)
        initial_step_count = start_rollout_index * batch_size
        writer = MLflowLogger(experiment="mappo", run_name=config.run_id, resume=progress is not None)
        resources.enter_context(writer)
        writer.log_text(yaml.safe_dump(asdict(config)), f"config/{initial_step_count}.yaml")
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
            observations=obs,  # (E, N, O)/(E, N, A)
            model_carry=model_carry,  # leaves: (B, H)
            episode_starts=episode_starts,  # (B,)
            episode_returns=jnp.zeros(config.num_envs, dtype=jnp.float32),  # (E,)
            episode_step_counts=jnp.zeros(config.num_envs, dtype=jnp.int32),  # (E,)
        )

        start = window_start = monotonic()
        window_start_step_count = initial_step_count
        planned_rollout_count = config.total_steps // batch_size
        for rollout_index in range(start_rollout_index, planned_rollout_count):
            train_state, rollout_carry, key, metrics = train_step_donated(
                rollout_carry, train_state, envs, config, key, metrics, rollout_index
            )  # state: (E, ...); memory: (B, H); key/metrics: ()
            rollout_count = rollout_index + 1
            is_final_rollout = rollout_count == planned_rollout_count
            step_count = rollout_count * batch_size
            if config.video_interval and rollout_count % config.video_interval == 0:
                log_video(
                    train_state,
                    config,
                    writer,
                    int(metrics.total.episode_count),
                    step_count,
                    initial_carry=model.initial_carry(envs.num_agents),
                )
            if config.eval_interval and (rollout_count % config.eval_interval == 0 or is_final_rollout):
                log_evaluation(
                    train_state,
                    config,
                    writer,
                    int(metrics.total.episode_count),
                    step_count,
                    initial_carry=model.initial_carry(envs.num_agents),
                )
            if rollout_count % config.log_interval == 0 or is_final_rollout:
                # Synchronize the window before measuring throughput.
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
                learning_rate = float(lr_schedule(rollout_index))
                entropy_coef = float(entropy_schedule(rollout_index))
                rate_seconds = max(window_seconds, 1e-9)
                sps = (step_count - window_start_step_count) / rate_seconds
                # Include evaluation/video overhead in this session's elapsed time.
                elapsed_seconds = now - start
                scalars: dict[str, float | int] = {
                    "value/explained_variance": float(window.explained_variance_sum) / window_rollout_count,
                    "charts/updates_per_rollout": int(window.update_count) / window_rollout_count,
                    "policy/early_stop": int(window.early_stop_count) / window_rollout_count,
                    "losses/policy": mean_metrics["policy_loss"],
                    "losses/value": mean_metrics["value_loss"],
                    "losses/entropy": -mean_metrics["entropy"],
                    "losses/total": mean_metrics["policy_loss"]
                    + config.value_coef * mean_metrics["value_loss"]
                    + mean_metrics["entropy_loss"],
                    "policy/entropy": mean_metrics["entropy"],
                    "policy/approx_kl": mean_metrics["approx_kl"],
                    "policy/clip_fraction": mean_metrics["clip_fraction"],
                    "charts/iterations_per_second": window_rollout_count / rate_seconds,
                    "charts/environment_steps_per_second": sps,
                    "charts/observations_per_second": sps * envs.num_agents,
                    "charts/episodes_per_second": window_episode_count / rate_seconds,
                    "charts/learning_rate": learning_rate,
                    "charts/entropy_coef": entropy_coef,
                    "charts/total_episodes": episode_count,
                    "charts/total_environment_steps": step_count,
                    "charts/total_observations": step_count * envs.num_agents,
                    "charts/total_iterations": rollout_count,
                    "charts/total_updates": int(train_state.step),
                    "charts/training_progress": rollout_count / planned_rollout_count,
                    "charts/episodes": window_episode_count,
                    "time/elapsed_seconds": elapsed_seconds,
                }
                if window_episode_count:
                    scalars["charts/return_mean"] = float(window.episodes.episode_return_sum) / window_episode_count
                    scalars["charts/episode_length_mean"] = (
                        int(window.episodes.episode_length_sum) / window_episode_count
                    )
                if episode_count:
                    scalars["charts/total_return_mean"] = float(total.episode_return_sum) / episode_count
                    scalars["charts/total_episode_length_mean"] = int(total.episode_length_sum) / episode_count
                writer.log_metrics(scalars, step_count)
                episode_return_text = f"{scalars['charts/return_mean']:.1f}" if window_episode_count else "n/a"
                episode_length_text = f"{scalars['charts/episode_length_mean']:.1f}" if window_episode_count else "n/a"
                print(
                    f"iteration={rollout_count} step={step_count} episodes={episode_count} return={episode_return_text} "
                    f"episode_length={episode_length_text} "
                    f"progress={scalars['charts/training_progress']:.3f} "
                    f"iterations/s={scalars['charts/iterations_per_second']:.3f} "
                    f"env_steps/s={sps:.3f} episodes/s={scalars['charts/episodes_per_second']:.3f} "
                    f"lr={learning_rate:.3g} "
                    f"policy={mean_metrics['policy_loss']:.3f} value={mean_metrics['value_loss']:.3f} "
                    f"entropy={mean_metrics['entropy']:.3f} kl={mean_metrics['approx_kl']:.4f} "
                    f"clipfrac={mean_metrics['clip_fraction']:.3f} ev={scalars['value/explained_variance']:.3f} "
                    f"updates={scalars['charts/updates_per_rollout']:.2f} "
                    f"early_stop={scalars['policy/early_stop']:.2f} "
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
                    ),
                    config,
                )
                writer.log_metrics({"time/checkpoint_seconds": checkpoint_seconds}, step_count)
        return train_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mappo/smax.yaml", help="Path to a YAML config")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
