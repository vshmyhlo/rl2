"""Recurrent PPO for Gymnax Breakout-MinAtar with a residual CNN and an LSTM backbone.

TensorBoard outputs use training environment steps as the global step.
Metric names use total_ for all-training statistics, _last_100 for recent episode
history, and episodes for episode counts. Window statistics have no scope affix.

Progress scalars computed over the logging window:
    losses/policy: PPO policy loss.
    losses/value: Value prediction loss before applying value_coef.
    losses/entropy: Negative action distribution entropy before applying entropy_coef.
    losses/total: Policy loss + value_coef * value loss + weighted entropy loss.
    policy/entropy: Action distribution entropy.
    policy/approx_kl: Approximate KL divergence from the rollout policy.
    policy/clip_fraction: Fraction of probability ratios outside the clip range.
    policy/early_stop: Fraction of rollouts stopped early by the KL threshold.
    value/explained_variance: Rollout value predictions versus GAE returns.
    charts/updates_per_rollout: Mean number of accepted minibatch updates.
    charts/iterations_per_second: Rollouts and their PPO updates per second in the logging window.
    charts/observations_per_second: Training observations across all environments per second in the logging window.
    charts/episodes_per_second: Completed training episodes per second in the logging window.
    charts/episodes: Training episodes completed in the logging window.
    charts/return_mean: Mean raw return of episodes completed in the window.
    charts/episode_length_mean: Mean episode length in the window, in steps.

Progress scalars computed across all training:
    charts/total_episodes: Cumulative completed training episodes.
    charts/total_observations: Cumulative training observations across all environments.
    charts/total_return_mean: Mean raw return over all completed training episodes.
    charts/total_episode_length_mean: Mean length over all completed training episodes, in steps.

Progress scalars computed over recent episode history (independent of logging windows):
    charts/return_mean_last_100: Mean raw return over up to 100 latest training episodes.
    charts/episode_length_mean_last_100: Mean length of those episodes, in steps.

Progress scalars reporting the latest rollout's settings:
    charts/learning_rate: Current learning rate.
    charts/entropy_coef: Current entropy bonus coefficient.

Startup scalars (also logged when resuming):
    model/params_millions: Model parameter count in millions.

Evaluation scalars (when evaluation is enabled; unavailable scores are omitted):
    eval/return_mean: Mean raw evaluation return.
    eval/return_std: Standard deviation of raw evaluation returns.
    eval/return_sem: Standard error of the mean raw evaluation return.
    eval/total_training_episodes: Completed training episodes at evaluation time.
    training/total_return_mean: Mean raw return over all completed training episodes.
    training/total_episode_length_mean: Mean length over all completed training episodes, in steps.
    training/total_episodes: Completed episodes in the training score history.
    training/return_mean_last_100: Mean raw return over up to 100 latest training episodes.
    training/episode_length_mean_last_100: Mean length of those episodes, in steps.
    training/episodes_last_100: Number of episodes in that recent history.
    time/evaluation_seconds: Evaluation wall time.

Text:
    config: Full training configuration as YAML, logged at startup and on resume.
    eval/report: Complete JSON report after each evaluation, including per-episode
        results, aggregate scores, training configuration and scores,
        evaluation seeds and training steps/episodes.

Video/images:
    gameplay: RGB gameplay from a separate episode under the current policy,
        recorded with add_video and stored as an animated GIF image summary.
        Recording ends at episode termination/truncation or video_max_frames
        (default: 900; None removes the frame cap). Playback uses video_speed
        and a playback rate of 10 environment steps per second.
    eval/gameplay: RGB video of the last episode in each evaluation run, using
        the same frame cap and playback speed. Evaluation finishes the full
        episode even when recording reaches the frame cap.
    No standalone still images, audio, histograms, embeddings, or graphs are logged.

Episode means are omitted when no corresponding episodes or history are available.

Logging windows span log_interval_seconds (default: 60); progress metrics are emitted
at rollout boundaries and on normal completion. Losses, entropy, KL, and clip fraction
average over evaluated minibatches, including KL-rejected updates. Explained
variance, updates per rollout, and early-stop frequency average over rollouts.
Seconds per iteration and throughput use window wall time, including intervening
logging, evaluation, video, and checkpoint overhead.
Episode window means include only episodes completed within that window. Existing
cumulative and last-100 episode statistics retain their meanings. Learning rate
and entropy coefficient report the latest rollout's settings. Evaluation and
video schedules are independent of progress logging. Videos are recorded every
video_interval_seconds (default: 600; 0 disables recording), at rollout boundaries.
The video timer restarts after recording and on resume, without catch-up videos.
"""

import argparse
import json
import os
from collections import deque
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
import yaml
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from gymnax.environments.minatar.breakout import EnvState
from numpy import stack, uint8
from numpy.typing import NDArray
from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass
from tensorboardX import SummaryWriter

from rl2.configuration import load_settings
from rl2.gymnax_env import GymnaxEnv, render_observation
from rl2.jax_cache import configure_compilation_cache
from rl2.lstm import LSTMStack, LSTMStackCarry
from rl2.observation_encoder import DEFAULT_STAGES, ConvObservationEncoder, ConvStage, ConvStages
from rl2.training_scores import TrainingScores
from rl2.utils import explained_variance, gae

type RecurrentCarry = LSTMStackCarry


@struct.dataclass
class PPOBatch:
    """PPO inputs; S is (T, B) for sequences or (B,) for step-only policies.

    Attributes:
        observations: (*S, *O) observations with observation dimensions O.
        actions: (*S,) integer action IDs.
        old_log_probs: (*S,) rollout action log probabilities.
        values: (*S,) rollout-time value predictions for explained variance.
        advantages: (*S,) unnormalized advantage estimates.
        returns: (*S,) value targets.
        carry: (B, D) initial LSTM memory leaves, where D is model.hidden_size.
        episode_starts: (*S,) boolean episode reset flags.
    """

    observations: jax.Array  # (*S, *O)
    actions: jax.Array  # (*S,)
    old_log_probs: jax.Array  # (*S,)
    values: jax.Array  # (*S,)
    advantages: jax.Array  # (*S,)
    returns: jax.Array  # (*S,)
    carry: RecurrentCarry  # leaves: (B, D), D is model.hidden_size
    episode_starts: jax.Array  # (*S,)


class RolloutCarry(NamedTuple):
    """Live environment state carried between rollouts.

    Attributes:
        env_state: (B, ...) batched Gymnax state leaves.
        observations: (B, *O) current observations with observation dimensions O.
        carry: (B, D) LSTM memory leaves, where D is model.hidden_size.
        episode_starts: (B,) boolean episode reset flags.
        episode_returns: (B,) accumulated raw rewards for unfinished episodes.
        episode_lengths: (B,) step counts for unfinished episodes.
    """

    env_state: EnvState  # leaves: (B, ...)
    observations: jax.Array  # (B, *O)
    carry: RecurrentCarry  # leaves: (B, ...)
    episode_starts: jax.Array  # (B,)
    episode_returns: jax.Array  # (B,)
    episode_lengths: jax.Array  # (B,)


class RolloutResult(NamedTuple):
    """Collected training data, advanced RNG key, and completed episode statistics.

    Attributes:
        batch: (T, B, ...) time-major PPO arrays with (B, ...) initial memory leaves.
        key: () advanced typed PRNG key, also used for minibatch shuffling.
        completed_returns: (T, B) raw returns, zero for unfinished episodes.
        completed_lengths: (T, B) lengths, zero for unfinished episodes.
        completed: (T, B) completed episode mask.
    """

    batch: PPOBatch  # observations: (T, B, *O); targets/starts: (T, B); carry leaves: (B, ...)
    key: jax.Array  # () typed PRNG key
    completed_returns: jax.Array  # (T, B)
    completed_lengths: jax.Array  # (T, B)
    completed: jax.Array  # (T, B)


@struct.dataclass
class PPOMetrics:
    """Scalar diagnostics for one evaluated minibatch, including KL-rejected updates."""

    policy_loss: jax.Array  # ()
    value_loss: jax.Array  # ()
    entropy_loss: jax.Array  # ()
    entropy: jax.Array  # ()
    approx_kl: jax.Array  # ()
    clip_fraction: jax.Array  # ()


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LSTMConfig:
    type: Literal["lstm"] = "lstm"
    hidden_size: Annotated[int, Field(gt=0, strict=True)] = 1536
    num_layers: Annotated[int, Field(gt=0, strict=True)] = 1
    intermediate_size: Annotated[int, Field(ge=0, strict=True)] = 0


@dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class Config:
    env_id: Literal["Breakout-MinAtar"]
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
    log_dir: str
    video_speed: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    bf16: bool = True
    video_max_frames: Annotated[int, Field(gt=0)] | None = 900
    video_interval_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 600.0
    eval_interval_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    eval_episodes: Annotated[int, Field(gt=0)] = 100
    eval_seed: Annotated[int, Field(ge=0, lt=2**32)] = 10_000
    encoder_stages: ConvStages = DEFAULT_STAGES
    model: LSTMConfig = LSTMConfig()
    lr_decay: Literal["linear", "cosine"] = "linear"
    entropy_decay: Literal["constant", "cosine"] = "constant"
    run_id: str | None = None
    checkpoint_interval_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 600.0
    log_interval_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 60.0

    def __post_init__(self) -> None:
        if self.run_id is not None and (
            not self.run_id.strip() or self.run_id in (".", "..") or any(c in self.run_id for c in "/\\")
        ):
            raise ValueError("run_id must be a nonempty directory name without slashes or traversal")

    @model_validator(mode="after")
    def validate_rollout(self) -> Self:
        if self.num_envs % self.num_minibatches:
            raise ValueError("num_envs must be divisible by num_minibatches to preserve sequences")
        if self.total_steps < self.num_envs * self.num_steps:
            raise ValueError("total_steps must cover at least one rollout")
        return self


class TrainingProgress(NamedTuple):
    state: TrainState
    key: jax.Array  # () typed PRNG key
    iteration: int  # Number of completed rollouts, independent of optimizer steps.
    completed_episodes: int
    recent_returns: tuple[float, ...]
    recent_lengths: tuple[int, ...]
    training_return_sum: float | None = None
    training_length_sum: int | None = None


def checkpoint_manager(run_dir: str) -> ocp.CheckpointManager:
    directory = f"{run_dir.rstrip('/')}/checkpoints"
    if not directory.startswith("gs://"):
        directory = str(Path(directory).resolve())
    return ocp.CheckpointManager(
        directory,
        options=ocp.CheckpointManagerOptions(max_to_keep=2),
    )


def save_checkpoint(manager: ocp.CheckpointManager, progress: TrainingProgress, config: Config) -> None:
    """Save a completed rollout; live environment state is deliberately restarted on resume."""
    key_data = jax.random.key_data(progress.key)  # (K,) raw PRNG words
    manager.save(
        progress.iteration,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave({"train_state": progress.state, "key": key_data}),
            metadata=ocp.args.JsonSave(
                {
                    "version": 2,
                    "config": asdict(config),
                    "iteration": progress.iteration,
                    "completed_episodes": progress.completed_episodes,
                    "recent_returns": list(progress.recent_returns),
                    "recent_lengths": list(progress.recent_lengths),
                    "training_return_sum": progress.training_return_sum,
                    "training_length_sum": progress.training_length_sum,
                }
            ),
        ),
        force=True,
    )


def restore_checkpoint(manager: ocp.CheckpointManager, state: TrainState, config: Config) -> TrainingProgress | None:
    iteration = manager.latest_step()
    if iteration is None:
        return None
    metadata = manager.restore(iteration, args=ocp.args.Composite(metadata=ocp.args.JsonRestore())).metadata
    if metadata["version"] not in (1, 2):
        raise ValueError("Unsupported PPO checkpoint version")
    if metadata["iteration"] != iteration or iteration < 0 or metadata["completed_episodes"] < 0:
        raise ValueError("Invalid PPO checkpoint progress")
    saved_config = metadata["config"]
    if saved_config.get("env_id", config.env_id) != config.env_id:
        raise ValueError("Cannot resume PPO with a different environment; use a new run directory")
    # Progress and schedules count rollouts, so each must still represent the same number of steps.
    if (
        "num_envs" in saved_config
        and "num_steps" in saved_config
        and saved_config["num_envs"] * saved_config["num_steps"] != config.num_envs * config.num_steps
    ):
        raise ValueError("Cannot resume PPO with a different num_envs * num_steps; use a new run directory")
    restored = manager.restore(
        iteration,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(
                {
                    "train_state": state,
                    "key": jax.random.key_data(jax.random.key(config.seed)),
                }
            )
        ),
    ).state
    # Version 1 also stored a NumPy shuffle RNG; all randomness now uses the saved JAX key.
    return TrainingProgress(
        restored["train_state"],
        jax.random.wrap_key_data(restored["key"]),
        iteration,
        metadata["completed_episodes"],
        tuple(metadata["recent_returns"]),
        tuple(metadata["recent_lengths"]),
        metadata.get("training_return_sum"),
        metadata.get("training_length_sum"),
    )


def load_config(path: str | Path) -> Config:
    """Load, resolve all interpolations and the run ID, and validate a training config."""
    settings = load_settings(path)
    if "encoder_stages" in settings:
        settings["encoder_stages"] = tuple(ConvStage(**stage) for stage in settings["encoder_stages"])
    return Config(**settings)


def learning_rate_schedule(config: Config) -> optax.Schedule:
    num_rollouts = config.total_steps // (config.num_envs * config.num_steps)
    if config.anneal_lr and config.lr_decay == "cosine":
        return optax.cosine_decay_schedule(config.learning_rate, num_rollouts)
    return optax.linear_schedule(
        config.learning_rate,
        0.0 if config.anneal_lr else config.learning_rate,
        num_rollouts,
    )


def entropy_coef_schedule(config: Config) -> optax.Schedule:
    if config.entropy_decay == "constant":
        return optax.constant_schedule(config.entropy_coef)
    num_rollouts = config.total_steps // (config.num_envs * config.num_steps)
    return optax.cosine_decay_schedule(config.entropy_coef, num_rollouts)


class ActorCritic(nn.Module):
    num_actions: int
    model: LSTMConfig = LSTMConfig()
    dtype: jax.typing.DTypeLike = jnp.float32
    encoder_stages: ConvStages = DEFAULT_STAGES
    embedding_size: int = 768

    @nn.nowrap
    def _make_recurrent(self, *, parent: nn.Module | None = None) -> LSTMStack:
        return LSTMStack(
            self.model.hidden_size,
            self.model.num_layers,
            self.model.intermediate_size,
            dtype=self.dtype,
            name="lstm",
            parent=parent,
        )

    @nn.nowrap
    def initial_carry(self, num_envs: int) -> RecurrentCarry:
        return self._make_recurrent().initial_carry(num_envs)

    def setup(self) -> None:
        self.encoder = ConvObservationEncoder(
            stages=self.encoder_stages,
            embedding_size=self.embedding_size,
            dtype=self.dtype,
            name="encoder",
        )
        self.recurrent = self._make_recurrent(parent=self)
        self.recurrent_input = nn.Dense(self.model.hidden_size, dtype=self.dtype, name="lstm_input")
        init = nn.initializers.orthogonal(sqrt(2))
        self.policy_hidden = nn.Dense(512, kernel_init=init, name="policy_hidden", dtype=self.dtype)
        self.policy_norm = nn.LayerNorm(name="policy_norm", dtype=self.dtype)
        self.value_hidden = nn.Dense(512, kernel_init=init, name="value_hidden", dtype=self.dtype)
        self.value_norm = nn.LayerNorm(name="value_norm", dtype=self.dtype)
        self.policy_output = nn.Dense(
            self.num_actions, kernel_init=nn.initializers.orthogonal(0.01), name="policy_output", dtype=self.dtype
        )
        self.value_output = nn.Dense(
            1, kernel_init=nn.initializers.orthogonal(1.0), name="value_output", dtype=self.dtype
        )

    def _encode(self, obs: jax.Array) -> jax.Array:  # (B, F, H, W[, C]) -> (B, D)
        x = self.encoder(obs)  # (B, E), E is embedding_size
        return self.recurrent_input(x).astype(self.dtype)  # (B, D), D is model.hidden_size

    def _heads(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:  # (B, D) -> (B, A), (B,)
        x = x.astype(self.dtype)  # (B, D)
        policy = nn.relu(self.policy_norm(self.policy_hidden(x)))  # (B, 512)
        critic = nn.relu(self.value_norm(self.value_hidden(x)))  # (B, 512)
        # Heads use the compute dtype; float32 outputs keep PPO loss arithmetic precise.
        logits = self.policy_output(policy).astype(jnp.float32)  # (B, A)
        values = self.value_output(critic).squeeze(-1).astype(jnp.float32)  # (B,)
        return logits, values  # (B, A), (B,)

    def __call__(
        self,
        obs: jax.Array,  # (T, B, F, H, W[, C])
        carry: RecurrentCarry,  # leaves: (B, ...)
        episode_starts: jax.Array,  # (T, B)
    ) -> tuple[RecurrentCarry, jax.Array, jax.Array]:  # carry leaves: (B, ...); (T, B, A); (T, B)
        """Process time-stacked observations [T,B,F,H,W,(C)] and reset masks [T,B]."""
        chex.assert_rank(obs, {5, 6})
        steps, environments = obs.shape[:2]
        x = self._encode(obs.reshape((-1, *obs.shape[2:])))  # (T * B, D)
        x = x.reshape((steps, environments, -1))  # (T, B, D)
        # Batch the CNN and heads across time; the LSTM handles its recurrence.
        carry, x = self.recurrent(x, carry, episode_starts)  # carry leaves: (B, ...); x: (T, B, D)
        logits, values = self._heads(x.reshape((steps * environments, -1)))  # (T * B, A), (T * B,)
        logits = logits.reshape((steps, environments, self.num_actions))  # (T, B, A)
        values = values.reshape((steps, environments))  # (T, B)
        return carry, logits, values  # carry leaves: (B, ...); (T, B, A); (T, B)

    def step(
        self,
        obs: jax.Array,  # (B, F, H, W[, C])
        carry: RecurrentCarry,  # leaves: (B, ...)
        episode_starts: jax.Array,  # (B,)
    ) -> tuple[RecurrentCarry, jax.Array, jax.Array]:  # carry leaves: (B, ...); (B, A); (B,)
        """Process one observation per environment [B,F,H,W,(C)] with reset masks [B]."""
        chex.assert_rank(obs, {4, 5})
        x = self._encode(obs)  # (B, D)
        carry, x = self.recurrent.step(x, carry, episode_starts)  # carry leaves: (B, ...); x: (B, D)
        logits, values = self._heads(x)  # (B, A), (B,)
        return carry, logits, values  # carry leaves: (B, ...); (B, A); (B,)


def action_log_prob(
    logits: jax.Array,  # (*S, A), S is (B,) or (T, B)
    actions: jax.Array,  # (*S,)
) -> jax.Array:  # (*S,)
    return jnp.take_along_axis(jax.nn.log_softmax(logits), actions[..., None], axis=-1)[..., 0]  # (*S,)


@jax.jit
def act(
    state: TrainState,
    obs: jax.Array,  # (B, *O)
    carry: RecurrentCarry,  # leaves: (B, ...)
    episode_starts: jax.Array,  # (B,)
    key: jax.Array,  # () typed PRNG key
) -> tuple[jax.Array, jax.Array, jax.Array, RecurrentCarry]:  # (B,), (B,), (B,); carry leaves: (B, ...)
    carry, logits, values = state.apply_fn(
        {"params": state.params},
        obs,
        carry,
        episode_starts,
        method="step",
    )
    actions = jax.random.categorical(key, logits)  # (B,)
    return actions, action_log_prob(logits, actions), values, carry  # (B,), (B,), (B,); carry leaves: (B, ...)


@jax.jit
def value(
    state: TrainState,
    obs: jax.Array,  # (B, *O)
    carry: RecurrentCarry,  # leaves: (B, ...)
    episode_starts: jax.Array,  # (B,)
) -> jax.Array:  # (B,)
    # Peek at the next value without advancing the rollout's recurrent state.
    return state.apply_fn({"params": state.params}, obs, carry, episode_starts, method="step")[2]  # (B,)


def play_episode(
    state: TrainState,
    config: Config,
    key: jax.Array,  # () typed PRNG key
    initial_carry: RecurrentCarry,  # leaves: (1, ...)
    *,
    record: bool,
    env: GymnaxEnv,
    stop_at_frame_limit: bool = False,
) -> tuple[float, int, list[NDArray[uint8]]]:
    """Run one episode; frames: list of (H, W, 3) uint8 RGB images."""
    key, reset_key = jax.random.split(key)  # each () typed PRNG key
    observations, env_state = env.reset(reset_key, 1)  # (1, C, H, W); leaves: (1, ...)
    memory = initial_carry  # leaves: (1, ...)
    frames: list[NDArray[uint8]] = []  # entries: (H, W, 3)
    if record:
        frames.append(np.asarray(render_observation(observations[0])))
    episode_return = 0.0
    length = 0
    while not (stop_at_frame_limit and config.video_max_frames is not None and len(frames) >= config.video_max_frames):
        key, action_key, env_key = jax.random.split(key, 3)  # each () typed PRNG key
        actions, _, _, memory = act(
            state, observations, memory, jnp.full(1, length == 0, dtype=bool), action_key
        )  # 3 x (1,); memory leaves: (1, ...)
        transition = env.step(env_key, env_state, actions)
        observations, env_state = transition.observations, transition.state  # (1,C,H,W); leaves: (1,...)
        episode_return += float(transition.rewards[0])
        length += 1
        if record and (config.video_max_frames is None or len(frames) < config.video_max_frames):
            frames.append(np.asarray(render_observation(transition.final_observations[0])))
        if bool((transition.terminated | transition.truncated)[0]):
            break
    return episode_return, length, frames


def log_video(
    state: TrainState,
    config: Config,
    writer: SummaryWriter,
    episode: int,
    steps: int,
    *,
    initial_carry: RecurrentCarry,  # leaves: (1, ...)
) -> None:
    _, _, frames = play_episode(
        state,
        config,
        jax.random.fold_in(jax.random.key(config.seed), episode),
        initial_carry,
        record=True,
        stop_at_frame_limit=True,
        env=GymnaxEnv(config.env_id),
    )
    video = stack(frames).transpose(0, 3, 1, 2)[None]  # (1, T, 3, H, W)
    writer.add_video("gameplay", video, steps, fps=10 * config.video_speed)
    print(f"Recorded {len(frames)} gameplay frames after {episode} training episodes", flush=True)


def log_evaluation(
    state: TrainState,
    config: Config,
    writer: SummaryWriter,
    episode: int,
    steps: int,
    *,
    initial_carry: RecurrentCarry,  # leaves: (1, ...)
    training_scores: TrainingScores | None = None,
) -> None:
    """Evaluate fixed Gymnax seeds, log raw scores, and record the last episode."""
    started = monotonic()
    episodes: list[dict[str, float | int]] = []
    env = GymnaxEnv(config.env_id)
    for index in range(config.eval_episodes):
        score, length, frames = play_episode(
            state,
            config,
            jax.random.fold_in(jax.random.key(config.eval_seed), index),
            initial_carry,
            record=index == config.eval_episodes - 1,
            env=env,
        )
        episodes.append({"return": score, "length": length})
        if frames:
            video = stack(frames).transpose(0, 3, 1, 2)[None]  # (1, T, 3, H, W)
            writer.add_video("eval/gameplay", video, steps, fps=10 * config.video_speed)
    returns = np.array([result["return"] for result in episodes])  # (E,), E evaluation episodes
    std = float(returns.std(ddof=1)) if config.eval_episodes > 1 else None
    result = {
        "env_id": config.env_id,
        "return_mean": float(returns.mean()),
        "return_std": std,
        "return_sem": std / sqrt(config.eval_episodes) if std is not None else None,
        "episodes": episodes,
        "eval_seed": config.eval_seed,
        "training_config": asdict(config),
        "training_steps": steps,
        "training_episodes": episode,
        "training_scores": training_scores.as_report() if training_scores is not None else None,
    }
    for name in ("return_mean", "return_std", "return_sem"):
        if result[name] is not None:
            writer.add_scalar(f"eval/{name}", result[name], steps)
    writer.add_scalar("eval/total_training_episodes", episode, steps)
    if result["training_scores"] is not None:
        for name, score in result["training_scores"].items():
            if score is not None:
                writer.add_scalar(f"training/{name}", score, steps)
    writer.add_scalar("time/evaluation_seconds", monotonic() - started, steps)
    writer.add_text("eval/report", f"```json\n{json.dumps(result, indent=2, allow_nan=False)}\n```", steps)
    print(f"Evaluation: return={result['return_mean']:.1f} over {config.eval_episodes} games", flush=True)


@partial(jax.jit, static_argnames="config")
def update(
    state: TrainState,
    batch: PPOBatch,  # observations: (*S, *O); targets/starts: (*S,); carry leaves: (B, ...)
    config: Config,
    iteration: int | jax.Array = 0,  # ()
) -> tuple[TrainState, PPOMetrics]:
    iteration = jnp.asarray(iteration)  # ()
    chex.assert_type(iteration, int)
    entropy_coef = jnp.asarray(entropy_coef_schedule(config)(iteration), dtype=jnp.float32)  # ()
    advantages = batch.advantages  # (*S,)
    chex.assert_type(batch.actions, int)
    chex.assert_scalar_positive(advantages.size)
    # Centering a single sample always gives zero and erases its policy gradient.
    if advantages.size > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # (*S,)

    def loss_fn(params: optax.Params) -> tuple[jax.Array, PPOMetrics]:
        _, logits, values = state.apply_fn(
            {"params": params}, batch.observations, batch.carry, batch.episode_starts
        )  # carry leaves: (B, ...); logits: (*S, A); values: (*S,)
        log_probs = action_log_prob(logits, batch.actions)  # (*S,)
        log_ratio = log_probs - batch.old_log_probs  # (*S,)
        ratio = jnp.exp(log_ratio)  # (*S,)
        clipped = jnp.clip(ratio, 1 - config.clip_coef, 1 + config.clip_coef)  # (*S,)
        policy_loss = -jnp.minimum(ratio * advantages, clipped * advantages).mean()  # ()
        value_loss = 0.5 * jnp.square(values - batch.returns).mean()  # ()
        entropy = -(jax.nn.softmax(logits) * jax.nn.log_softmax(logits)).sum(-1).mean()  # ()
        entropy_loss = -entropy_coef * entropy  # ()
        loss = policy_loss + config.value_coef * value_loss + entropy_loss  # ()
        approx_kl = (jnp.expm1(log_ratio) - log_ratio).mean()  # ()
        clip_fraction = (jnp.abs(ratio - 1) > config.clip_coef).mean()  # ()
        return loss, PPOMetrics(
            policy_loss=policy_loss,  # ()
            value_loss=value_loss,  # ()
            entropy_loss=entropy_loss,  # ()
            entropy=entropy,  # ()
            approx_kl=approx_kl,  # ()
            clip_fraction=clip_fraction,  # ()
        )  # loss: (); metrics: scalar leaves

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    if config.target_kl is None:
        return state.apply_gradients(grads=grads), metrics
    state = jax.lax.cond(
        metrics.approx_kl > config.target_kl,
        lambda: state,
        lambda: state.apply_gradients(grads=grads),
    )
    return state, metrics


@partial(jax.jit, static_argnames=("envs", "config"))
def collect_rollout(
    rollout_carry: RolloutCarry,  # observations: (B, *O); state/memory leaves: (B, ...)
    train_state: TrainState,
    envs: GymnaxEnv,
    config: Config,
    key: jax.Array,  # () typed PRNG key
) -> tuple[RolloutCarry, RolloutResult]:
    """Collect (T, B, ...) transitions entirely on device and compute GAE.

    Timeouts bootstrap from final observations; both end flags reset recurrence.
    Completed episode arrays retain fixed (T, B) shapes for JIT compilation.
    """

    def step(
        runner: tuple[RolloutCarry, jax.Array],  # state leaves: (B, ...); key: ()
        _: None,
    ) -> tuple[tuple[RolloutCarry, jax.Array], tuple[jax.Array, ...]]:
        current, key = runner
        key, action_key, env_key = jax.random.split(key, 3)  # each () typed PRNG key
        actions, log_probs, predictions, memory = act(
            train_state, current.observations, current.carry, current.episode_starts, action_key
        )  # 3 x (B,); memory leaves: (B, ...)
        transition = envs.step(env_key, current.env_state, actions)
        done = transition.terminated | transition.truncated  # (B,)
        timeout = transition.truncated & ~transition.terminated  # (B,)

        def bootstrap(_: None) -> jax.Array:
            predictions = value(
                train_state, transition.final_observations, memory, jnp.zeros(config.num_envs, dtype=bool)
            )  # (B,)
            return jnp.where(timeout, predictions, 0.0)  # (B,)

        def no_bootstrap(_: None) -> jax.Array:
            return jnp.zeros(config.num_envs)  # (B,)

        rewards = transition.rewards + config.gamma * jax.lax.cond(timeout.any(), bootstrap, no_bootstrap, None)  # (B,)
        episode_returns = current.episode_returns + transition.rewards  # (B,)
        episode_lengths = current.episode_lengths + 1  # (B,)
        advanced = RolloutCarry(
            env_state=transition.state,  # leaves: (B, ...)
            observations=transition.observations,  # (B, *O)
            carry=memory,  # leaves: (B, ...)
            episode_starts=done,  # (B,)
            episode_returns=jnp.where(done, 0.0, episode_returns),  # (B,)
            episode_lengths=jnp.where(done, 0, episode_lengths),  # (B,)
        )
        return (advanced, key), (
            current.observations,  # (B, *O)
            actions,  # (B,)
            log_probs,  # (B,)
            predictions,  # (B,)
            rewards,  # (B,)
            done,  # (B,)
            current.episode_starts,  # each (B,)
            jnp.where(done, episode_returns, 0.0),  # (B,)
            jnp.where(done, episode_lengths, 0),  # (B,)
        )

    (advanced, key), trajectory = jax.lax.scan(step, (rollout_carry, key), None, length=config.num_steps)
    observations, actions, log_probs, values, rewards, dones, starts, completed_returns, completed_lengths = trajectory
    # observations: (T, B, *O); other trajectory arrays: (T, B)
    next_value = value(train_state, advanced.observations, advanced.carry, advanced.episode_starts)  # (B,)
    advantages, returns = gae(rewards, dones, values, next_value, config.gamma, config.gae_lambda)  # each (T, B)
    batch = PPOBatch(
        observations=observations,  # (T, B, *O)
        actions=actions,  # (T, B)
        old_log_probs=log_probs,  # (T, B)
        values=values,  # (T, B)
        advantages=advantages,  # (T, B)
        returns=returns,  # (T, B)
        carry=rollout_carry.carry,  # leaves: (B, ...)
        episode_starts=starts,  # (T, B)
    )
    return advanced, RolloutResult(batch, key, completed_returns, completed_lengths, dones)


def train(config: Config) -> TrainState:
    """Train with a fully resolved config, as returned by load_config."""
    with ExitStack() as resources:
        configure_compilation_cache()
        batch_size = config.num_envs * config.num_steps
        envs = GymnaxEnv(config.env_id)
        run_name = config.run_id
        run_dir = config.log_dir
        manager = resources.enter_context(closing(checkpoint_manager(run_dir)))
        key, init_key, reset_key = jax.random.split(jax.random.key(config.seed), 3)  # each () typed PRNG key
        obs, env_state = envs.reset(reset_key, config.num_envs)  # (B, *O); state leaves: (B, ...)
        model = ActorCritic(
            envs.num_actions,
            config.model,
            dtype=jnp.bfloat16 if config.bf16 else jnp.float32,
            encoder_stages=config.encoder_stages,
        )
        carry = model.initial_carry(config.num_envs)  # leaves: (B, ...)
        episode_start = jnp.ones(config.num_envs, dtype=bool)  # (B,)
        lr_schedule = learning_rate_schedule(config)
        entropy_schedule = entropy_coef_schedule(config)
        optimizer = optax.inject_hyperparams(
            lambda learning_rate: optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(learning_rate, eps=1e-5),
            )
        )
        state = TrainState.create(
            apply_fn=model.apply,
            params=model.init(init_key, obs[None, :1], model.initial_carry(1), episode_start[None, :1])["params"],
            tx=optimizer(config.learning_rate),
        )
        progress = restore_checkpoint(manager, state, config)
        start_iteration = 0
        recent_returns: deque[float] = deque(maxlen=100)
        recent_lengths: deque[int] = deque(maxlen=100)
        completed_episodes = 0
        training_return_sum: float | None = 0.0
        training_length_sum: int | None = 0
        if progress is not None:
            (
                state,
                key,
                start_iteration,
                completed_episodes,
                saved_returns,
                saved_lengths,
                training_return_sum,
                training_length_sum,
            ) = progress
            recent_returns.extend(saved_returns)
            recent_lengths.extend(saved_lengths)
            # Live environment states are not serialized. Start fresh episodes
            # with fresh recurrent memory, retaining the policy and training RNGs.
            obs, env_state = envs.reset(jax.random.fold_in(key, start_iteration), config.num_envs)  # (B, *O)
            print(f"Resumed run {run_name} at step {start_iteration * batch_size}", flush=True)
        initial_steps = start_iteration * batch_size
        writer = resources.enter_context(
            closing(SummaryWriter(logdir=run_dir, purge_step=initial_steps + 1 if progress is not None else None))
        )
        writer.add_text("config", f"```yaml\n{yaml.safe_dump(asdict(config))}```", initial_steps)
        print(f"Run ID: {run_name}\nTensorBoard run: {run_dir}", flush=True)
        devices = str(jax.devices())
        print(f"JAX devices: {devices}", flush=True)
        writer.add_text("devices", devices, initial_steps)
        parameter_count = sum(parameter.size for parameter in jax.tree.leaves(state.params))
        writer.add_scalar("model/params_millions", parameter_count / 1_000_000, initial_steps)
        print(f"Model parameters: {parameter_count:,}", flush=True)
        writer.add_text("model/parameter_count", str(parameter_count), initial_steps)
        rollout_carry = RolloutCarry(
            env_state=env_state,  # leaves: (B, ...)
            observations=obs,  # (B, *O)
            carry=carry,  # leaves: (B, ...)
            episode_starts=episode_start,  # (B,)
            episode_returns=jnp.zeros(config.num_envs, dtype=jnp.float32),  # (B,)
            episode_lengths=jnp.zeros(config.num_envs, dtype=jnp.int32),  # (B,)
        )

        start = monotonic()
        window_start = start
        window_steps = initial_steps
        window_rollouts = 0
        window_metric_count = 0
        window_metric_sums = dict.fromkeys((field.name for field in fields(PPOMetrics)), 0.0)
        window_rollout_sums: dict[str, float] = {}
        window_episodes = 0
        window_return_sum = 0.0
        window_length_sum = 0
        last_checkpoint_time = start
        next_video_time = start + config.video_interval_seconds
        next_eval_time = start + config.eval_interval_seconds
        num_iterations = config.total_steps // batch_size
        for iteration in range(start_iteration, num_iterations):
            rollout_carry, rollout = collect_rollout(
                rollout_carry, state, envs, config, key
            )  # carry: (B, ...); rollout arrays: (T, B, ...)
            batch = rollout.batch  # observations: (T, B, *O); targets/starts: (T, B); carry leaves: (B, ...)
            key = rollout.key  # () typed PRNG key
            completed, returns, lengths = jax.device_get(
                (rollout.completed, rollout.completed_returns, rollout.completed_lengths)
            )  # each (T, B), host arrays
            finished_returns, finished_lengths = returns[completed], lengths[completed]  # each (D,)
            recent_returns.extend(finished_returns.tolist())
            recent_lengths.extend(finished_lengths.tolist())
            finished_count = finished_returns.size
            finished_return_sum = float(finished_returns.sum())
            finished_length_sum = int(finished_lengths.sum())
            window_episodes += finished_count
            window_return_sum += finished_return_sum
            window_length_sum += finished_length_sum
            if training_return_sum is not None:
                training_return_sum += finished_return_sum
            if training_length_sum is not None:
                training_length_sum += finished_length_sum
            completed_episodes += finished_count
            learning_rate = float(lr_schedule(iteration))
            entropy_coef = float(entropy_schedule(iteration))
            state = state.replace(
                opt_state=state.opt_state._replace(
                    hyperparams={
                        **state.opt_state.hyperparams,
                        "learning_rate": jnp.asarray(learning_rate),
                    }
                )
            )
            early_stop = False
            updates_done = 0
            for _ in range(config.update_epochs):
                key, shuffle_key = jax.random.split(key)  # each () typed PRNG key
                permutation = jax.random.permutation(shuffle_key, config.num_envs)  # (B,)
                for indices in jnp.split(permutation, config.num_minibatches):  # each (M,)
                    # M is the number of environments in this minibatch.
                    minibatch = batch.replace(
                        **jax.tree.map(
                            itemgetter((slice(None), indices)),
                            {
                                field.name: getattr(batch, field.name)
                                for field in fields(batch)
                                if field.name != "carry"
                            },
                        ),  # observations: (T, M, *O); other sequence fields: (T, M)
                        carry=jax.tree.map(itemgetter(indices), batch.carry),  # leaves: (M, ...)
                    )
                    state, metric = update(state, minibatch, config, iteration)
                    for name in window_metric_sums:
                        window_metric_sums[name] += float(getattr(metric, name))
                    window_metric_count += 1
                    if config.target_kl is not None and float(metric.approx_kl) > config.target_kl:
                        early_stop = True
                        break
                    updates_done += 1
                if early_stop:
                    break
            window_rollouts += 1
            # Evaluate rollout-time predictions against the full rollout's GAE returns.
            explained_var = explained_variance(batch.values, batch.returns)
            steps = (iteration + 1) * batch_size
            for tag, scalar in {
                "value/explained_variance": explained_var,
                "charts/updates_per_rollout": updates_done,
                "policy/early_stop": early_stop,
            }.items():
                window_rollout_sums[tag] = window_rollout_sums.get(tag, 0.0) + float(scalar)
            if config.video_interval_seconds and monotonic() >= next_video_time:
                log_video(state, config, writer, completed_episodes, steps, initial_carry=model.initial_carry(1))
                # Restart after recording so long videos never cause catch-up recordings.
                next_video_time = monotonic() + config.video_interval_seconds
            if config.eval_interval_seconds and (iteration + 1 == num_iterations or monotonic() >= next_eval_time):
                log_evaluation(
                    state,
                    config,
                    writer,
                    completed_episodes,
                    steps,
                    initial_carry=model.initial_carry(1),
                    training_scores=TrainingScores(
                        completed_episodes,
                        training_return_sum,
                        tuple(float(score) for score in recent_returns),
                        length_sum=training_length_sum,
                        recent_lengths=tuple(recent_lengths),
                    ),
                )
                # Restart after evaluation so long evaluations never cause catch-up runs.
                next_eval_time = monotonic() + config.eval_interval_seconds
            now = monotonic()
            window_seconds = now - window_start
            if window_seconds >= config.log_interval_seconds or iteration + 1 == num_iterations:
                mean_metrics = {name: total / window_metric_count for name, total in window_metric_sums.items()}
                rate_seconds = max(window_seconds, 1e-9)
                sps = (steps - window_steps) / rate_seconds
                # Include evaluation/video overhead in this session's elapsed time.
                elapsed_seconds = now - start
                scalars = {tag: total / window_rollouts for tag, total in window_rollout_sums.items()}
                scalars.update(
                    {
                        "losses/policy": mean_metrics["policy_loss"],
                        "losses/value": mean_metrics["value_loss"],
                        "losses/entropy": -mean_metrics["entropy"],
                        "losses/total": mean_metrics["policy_loss"]
                        + config.value_coef * mean_metrics["value_loss"]
                        + mean_metrics["entropy_loss"],
                        "policy/entropy": mean_metrics["entropy"],
                        "policy/approx_kl": mean_metrics["approx_kl"],
                        "policy/clip_fraction": mean_metrics["clip_fraction"],
                        "charts/iterations_per_second": window_rollouts / rate_seconds,
                        "charts/observations_per_second": sps,
                        "charts/episodes_per_second": window_episodes / rate_seconds,
                        "charts/learning_rate": learning_rate,
                        "charts/entropy_coef": entropy_coef,
                        "charts/total_episodes": completed_episodes,
                        "charts/total_observations": steps,
                        "charts/episodes": window_episodes,
                        "time/elapsed_seconds": elapsed_seconds,
                    }
                )
                if window_episodes:
                    scalars["charts/return_mean"] = window_return_sum / window_episodes
                    scalars["charts/episode_length_mean"] = window_length_sum / window_episodes
                if recent_returns:
                    scalars["charts/return_mean_last_100"] = sum(recent_returns) / len(recent_returns)
                    scalars["charts/episode_length_mean_last_100"] = sum(recent_lengths) / len(recent_lengths)
                    if training_return_sum is not None:
                        scalars["charts/total_return_mean"] = training_return_sum / completed_episodes
                    if training_length_sum is not None:
                        scalars["charts/total_episode_length_mean"] = training_length_sum / completed_episodes
                for tag, scalar in scalars.items():
                    writer.add_scalar(tag, float(scalar), steps)
                writer.flush()
                score = f"{window_return_sum / window_episodes:.1f}" if window_episodes else "n/a"
                length = f"{window_length_sum / window_episodes:.1f}" if window_episodes else "n/a"
                print(
                    f"step={steps} episodes={completed_episodes} return={score} "
                    f"episode_length={length} "
                    f"iterations/s={scalars['charts/iterations_per_second']:.3f} "
                    f"observations/s={sps:.3f} episodes/s={scalars['charts/episodes_per_second']:.3f} "
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
                window_steps = steps
                window_rollouts = 0
                window_metric_count = 0
                window_metric_sums = dict.fromkeys(window_metric_sums, 0.0)
                window_rollout_sums.clear()
                window_episodes = 0
                window_return_sum = 0.0
                window_length_sum = 0
            if (
                monotonic() - last_checkpoint_time >= config.checkpoint_interval_seconds
                or iteration + 1 == num_iterations
            ):
                save_checkpoint(
                    manager,
                    TrainingProgress(
                        state,
                        key,
                        iteration + 1,
                        completed_episodes,
                        tuple(float(v) for v in recent_returns),
                        tuple(int(v) for v in recent_lengths),
                        training_return_sum,
                        training_length_sum,
                    ),
                    config,
                )
                last_checkpoint_time = monotonic()
        return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ppo.yaml", help="Path to a YAML config")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
