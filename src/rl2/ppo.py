"""Recurrent PPO with a residual Atari CNN and an LSTM, GDN2, or Mamba3 backbone.

Progress metrics aggregate over log_interval_seconds (default: 60), emitted at
rollout boundaries and on normal completion. Losses, entropy, KL, and clip fraction
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
from typing import Annotated, Any, Literal, NamedTuple, Self, SupportsFloat, cast

# Orbax's epath otherwise loads TensorFlow for gs:// paths when it is installed.
# That native runtime can crash a later Triton import; use our gcsfs dependency.
# Set this before importing Orbax, since epath caches its backend selection.
os.environ.setdefault("EPATH_USE_TF", "0")

import ale_py
import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import yaml
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from numpy import asarray, integer, uint8
from numpy.typing import NDArray
from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass
from tensorboardX import SummaryWriter

from rl2 import atari_eval
from rl2.configuration import load_settings
from rl2.gdn2 import (
    GatedDeltaNet2Backend,
    GatedDeltaNet2Config,
    GatedDeltaNet2Recurrent,
    GatedDeltaNet2StackCarry,
)
from rl2.jax_cache import configure_compilation_cache
from rl2.jax_env import JaxVectorEnv
from rl2.lstm import LSTMStack, LSTMStackCarry
from rl2.mamba3 import Mamba3Stack, Mamba3StackCarry
from rl2.multi_atari import register_envs
from rl2.observation_encoder import DEFAULT_STAGES, ConvObservationEncoder, ConvStage, ConvStages
from rl2.sequence_model import RecurrentSequenceModel
from rl2.utils import explained_variance, gae

type Array = jax.Array | NDArray[Any]
type RecurrentCarry = LSTMStackCarry | GatedDeltaNet2StackCarry | Mamba3StackCarry
type ModelType = Literal["lstm", "gdn2", "mamba3"]


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
        carry: (B, ...) leaves of the initial recurrent memory, with backend-specific trailing dimensions.
        episode_starts: (*S,) boolean episode reset flags.
    """

    observations: Array  # (*S, *O)
    actions: Array  # (*S,)
    old_log_probs: Array  # (*S,)
    values: Array  # (*S,)
    advantages: Array  # (*S,)
    returns: Array  # (*S,)
    carry: RecurrentCarry  # leaves: (B, ...), backend-specific trailing dimensions
    episode_starts: Array  # (*S,)


class RolloutCarry(NamedTuple):
    """Live environment state carried between rollouts.

    Attributes:
        observations: (B, *O) current observations with observation dimensions O.
        carry: (B, ...) recurrent memory leaves with backend-specific trailing dimensions.
        episode_starts: (B,) boolean episode reset flags.
        episode_returns: (B,) accumulated raw rewards for unfinished episodes.
        episode_lengths: (B,) step counts for unfinished episodes.
    """

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
        completed_returns: Raw returns in step order, then environment order.
        completed_lengths: Episode lengths in the same order as completed_returns.
    """

    batch: PPOBatch  # observations: (T, B, *O); targets/starts: (T, B); carry leaves: (B, ...)
    key: jax.Array  # () typed PRNG key
    completed_returns: tuple[float, ...]
    completed_lengths: tuple[int, ...]


@struct.dataclass
class PPOMetrics:
    """Scalar diagnostics for one evaluated minibatch, including KL-rejected updates."""

    policy_loss: Array  # ()
    value_loss: Array  # ()
    entropy: Array  # ()
    approx_kl: Array  # ()
    clip_fraction: Array  # ()


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LSTMConfig:
    type: Literal["lstm"] = "lstm"
    hidden_size: Annotated[int, Field(gt=0, strict=True)] = 1536
    num_layers: Annotated[int, Field(gt=0, strict=True)] = 1
    intermediate_size: Annotated[int, Field(ge=0, strict=True)] = 0


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class GDN2Config:
    type: Literal["gdn2"] = "gdn2"
    # Triton accelerates rollout steps and differentiable sequence replay on NVIDIA GPUs.
    backend: GatedDeltaNet2Backend = "jax"
    hidden_size: Annotated[int, Field(gt=0, strict=True)] = 768
    num_layers: Annotated[int, Field(gt=0, strict=True)] = 2
    num_heads: Annotated[int, Field(gt=0, strict=True)] = 12
    head_dim: Annotated[int, Field(gt=0, strict=True)] = 64
    intermediate_size: Annotated[int, Field(gt=0, strict=True)] = 1536
    conv_size: Annotated[int, Field(gt=0, strict=True)] = 4

    def mixer_config(self, dtype: jax.typing.DTypeLike) -> GatedDeltaNet2Config:
        return GatedDeltaNet2Config(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            conv_size=self.conv_size,
            dtype=dtype,
        )


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class Mamba3Config:
    type: Literal["mamba3"] = "mamba3"
    hidden_size: Annotated[int, Field(gt=0, strict=True)] = 768
    num_layers: Annotated[int, Field(gt=0, strict=True)] = 2
    intermediate_size: Annotated[int, Field(ge=0, strict=True)] = 1536
    state_size: Annotated[int, Field(gt=0, strict=True)] = 128
    expand: Annotated[int, Field(gt=0, strict=True)] = 2
    head_dim: Annotated[int, Field(gt=0, strict=True)] = 64
    num_groups: Annotated[int, Field(gt=0, strict=True)] = 1
    mimo_rank: Annotated[int, Field(gt=0, strict=True)] = 1
    rope_fraction: Literal[0.5, 1.0] = 0.5

    def __post_init__(self) -> None:
        inner_size = self.hidden_size * self.expand
        if inner_size % self.head_dim:
            raise ValueError("hidden_size * expand must be divisible by head_dim")
        if (inner_size // self.head_dim) % self.num_groups:
            raise ValueError("number of heads must be divisible by num_groups")
        if self.state_size % 2 or int(self.state_size * self.rope_fraction) < 2:
            raise ValueError("state_size must be even and allow at least one rotary pair")


type ModelConfig = Annotated[LSTMConfig | GDN2Config | Mamba3Config, Field(discriminator="type")]


@dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class Config:
    env_id: str
    frame_stack: bool
    atari_preprocessing: bool
    seed: int
    total_steps: int
    num_envs: Annotated[int, Field(gt=0)]
    vector_env: Literal["sync", "async"]
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
    observation_size: int | None = None
    eval_interval_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    eval_episodes: int = 100
    eval_seed: int = 10_000
    encoder_stages: ConvStages = DEFAULT_STAGES
    model: ModelConfig = LSTMConfig()
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

    @model_validator(mode="after")
    def validate_evaluation(self) -> Self:
        if self.eval_interval_seconds:
            atari_eval.EvaluationConfig(episodes=self.eval_episodes, seed=self.eval_seed)
            atari_eval.validate_training_config(self)
        return self


class TrainingProgress(NamedTuple):
    state: TrainState
    key: jax.Array  # () typed PRNG key
    iteration: int  # Number of completed rollouts, independent of optimizer steps.
    completed_episodes: int
    recent_returns: tuple[float, ...]
    recent_lengths: tuple[int, ...]
    training_return_sum: float | None = None


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
    model: ModelConfig = LSTMConfig()
    dtype: jax.typing.DTypeLike = jnp.float32
    encoder_stages: ConvStages = DEFAULT_STAGES
    embedding_size: int = 768

    @nn.nowrap
    def _make_recurrent(self, *, parent: nn.Module | None = None) -> RecurrentSequenceModel[RecurrentCarry]:
        recurrent: LSTMStack | GatedDeltaNet2Recurrent | Mamba3Stack
        if self.model.type == "lstm":
            recurrent = LSTMStack(
                self.model.hidden_size,
                self.model.num_layers,
                self.model.intermediate_size,
                dtype=self.dtype,
                name="lstm",
                parent=parent,
            )
        elif self.model.type == "gdn2":
            recurrent = GatedDeltaNet2Recurrent(
                self.model.mixer_config(self.dtype),
                self.model.num_layers,
                self.model.intermediate_size,
                backend=self.model.backend,
                name="gdn2",
                parent=parent,
            )
        else:
            recurrent = Mamba3Stack(
                d_model=self.model.hidden_size,
                num_layers=self.model.num_layers,
                d_intermediate=self.model.intermediate_size,
                mlp_multiple_of=1,
                d_state=self.model.state_size,
                expand=self.model.expand,
                headdim=self.model.head_dim,
                ngroups=self.model.num_groups,
                mimo_rank=self.model.mimo_rank,
                rope_fraction=self.model.rope_fraction,
                dtype=self.dtype,
                name="mamba3",
                parent=parent,
            )
        # The selected model and its carry always travel together through PPO.
        return cast(RecurrentSequenceModel[RecurrentCarry], recurrent)

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
        self.recurrent_input = nn.Dense(self.model.hidden_size, dtype=self.dtype, name=f"{self.model.type}_input")
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

    def _encode(self, obs: Array) -> jax.Array:
        chex.assert_rank(obs, {4, 5})
        x = self.encoder(obs)
        x = self.recurrent_input(x).astype(self.dtype)
        return x

    def _heads(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        # Flatten time and batch for sequence calls, keeping the same heads as step().
        x = x.astype(self.dtype)
        policy = nn.relu(self.policy_norm(self.policy_hidden(x)))
        critic = nn.relu(self.value_norm(self.value_hidden(x)))
        # Heads use the compute dtype; float32 outputs keep PPO loss arithmetic precise.
        logits = self.policy_output(policy).astype(jnp.float32)
        values = self.value_output(critic).squeeze(-1).astype(jnp.float32)
        return logits, values

    def __call__(
        self,
        obs: Array,
        carry: RecurrentCarry,
        episode_starts: Array,
    ) -> tuple[RecurrentCarry, jax.Array, jax.Array]:
        """Process time-stacked observations [T,B,F,H,W,(C)] and reset masks [T,B]."""
        chex.assert_rank(obs, {5, 6})
        steps, environments = obs.shape[:2]
        x = self._encode(obs.reshape((-1, *obs.shape[2:])))
        x = x.reshape((steps, environments, -1))
        # Batch the CNN and heads across time; each backbone handles its recurrence.
        carry, x = self.recurrent(x, carry, episode_starts)
        logits, values = self._heads(x.reshape((steps * environments, -1)))
        logits = logits.reshape((steps, environments, self.num_actions))
        values = values.reshape((steps, environments))
        return carry, logits, values

    def step(
        self,
        obs: Array,
        carry: RecurrentCarry,
        episode_starts: Array,
    ) -> tuple[RecurrentCarry, jax.Array, jax.Array]:
        """Process one observation per environment [B,F,H,W,(C)] with reset masks [B]."""
        chex.assert_rank(obs, {4, 5})
        x = self._encode(obs)
        carry, x = self.recurrent.step(x, carry, episode_starts)
        logits, values = self._heads(x)
        return carry, logits, values


def make_model(config: Config, num_actions: int) -> ActorCritic:
    return ActorCritic(
        num_actions,
        config.model,
        dtype=jnp.bfloat16 if config.bf16 else jnp.float32,
        encoder_stages=config.encoder_stages,
    )


def initial_model_carry(config: Config, num_envs: int) -> RecurrentCarry:
    return make_model(config, num_actions=1).initial_carry(num_envs)


class AtariPreprocessing(gym.wrappers.AtariPreprocessing):
    def step(
        self,
        action: int | integer[Any],
    ) -> tuple[NDArray[Any], SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated or truncated:
            # Gymnasium breaks action repeat before capturing the final screen.
            # Return the actual final screen, including for timeout bootstrapping.
            capture = self.ale.getScreenGrayscale if self.grayscale_obs else self.ale.getScreenRGB
            capture(self.obs_buffer[0])
            self.obs_buffer[1].fill(0)
            obs = self._get_obs()
        return obs, reward, terminated, truncated, info


def make_env(
    env_id: str,
    render_mode: str | None = None,
    frame_stack: bool = False,
    atari_preprocessing: bool = False,
    observation_size: int | None = None,
    grayscale_obs: bool = True,
) -> gym.Env[NDArray[uint8], int | integer[Any]]:
    chex.assert_type(grayscale_obs, bool)
    if observation_size is not None and (type(observation_size) is not int or observation_size < 1):
        raise ValueError("observation_size must be a positive integer or null")
    gym.register_envs(ale_py)
    register_envs()
    env = gym.make(env_id, frameskip=1, render_mode=render_mode)
    try:
        if atari_preprocessing:
            env = AtariPreprocessing(env, grayscale_obs=grayscale_obs)
        if observation_size is not None:
            env = gym.wrappers.ResizeObservation(env, (observation_size, observation_size))
        if frame_stack:
            return gym.wrappers.FrameStackObservation(env, stack_size=4)
        return gym.wrappers.ReshapeObservation(env, (1, *env.observation_space.shape))
    except BaseException:
        env.close()
        raise


def action_log_prob(logits: Array, actions: Array) -> jax.Array:
    return jnp.take_along_axis(jax.nn.log_softmax(logits), actions[..., None], axis=-1)[..., 0]


@jax.jit
def act(
    state: TrainState,
    obs: Array,
    carry: RecurrentCarry,
    episode_starts: Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, RecurrentCarry]:
    carry, logits, values = state.apply_fn(
        {"params": state.params},
        obs,
        carry,
        episode_starts,
        method="step",
    )
    actions = jax.random.categorical(key, logits)
    return actions, action_log_prob(logits, actions), values, carry


@jax.jit
def value(state: TrainState, obs: Array, carry: RecurrentCarry, episode_starts: Array) -> jax.Array:
    # Peek at the next value without advancing the rollout's recurrent state.
    return state.apply_fn({"params": state.params}, obs, carry, episode_starts, method="step")[2]


def log_video(
    state: TrainState,
    config: Config,
    writer: SummaryWriter,
    episode: int,
    steps: int,
    *,
    initial_carry: RecurrentCarry,  # leaves: (1, ...)
) -> None:
    env = make_env(
        config.env_id,
        render_mode="rgb_array",
        frame_stack=config.frame_stack,
        atari_preprocessing=config.atari_preprocessing,
        observation_size=config.observation_size,
    )
    try:
        obs, _ = env.reset(seed=config.seed + episode)  # (*O,)
        key = jax.random.fold_in(jax.random.key(config.seed), episode)  # () typed PRNG key
        frames = [env.render()]  # list of (H, W, C) frames
        carry = initial_carry  # leaves: (1, ...)
        while config.video_max_frames is None or len(frames) < config.video_max_frames:
            key, action_key = jax.random.split(key)  # each () typed PRNG key
            actions, _, _, carry = act(state, obs[None], carry, jnp.zeros(1, dtype=bool), action_key)  # (1,), memory
            obs, _, terminated, truncated, _ = env.step(int(actions[0]))  # obs: (*O,)
            frames.append(env.render())
            if terminated or truncated:
                break
        # One recorded frame per agent step, accounting for optional action repeat.
        video = jnp.stack(frames).transpose(0, 3, 1, 2)[None]  # (1, T, C, H, W)
        writer.add_video(
            "gameplay",
            asarray(video),  # (1, T, C, H, W); TensorBoard expects a host array
            steps,
            fps=config.video_speed * env.metadata["render_fps"] / (4 if config.atari_preprocessing else 1),
        )
        print(f"Recorded {len(frames)} gameplay frames after {episode} training episodes", flush=True)
    finally:
        env.close()


def log_evaluation(
    state: TrainState,
    config: Config,
    writer: SummaryWriter,
    episode: int,
    steps: int,
    *,
    training_scores: "atari_eval.TrainingScores | None" = None,
) -> None:
    """Evaluate the current policy and save scores plus the complete report to TensorBoard."""
    print(f"Evaluating {config.eval_episodes} games after {episode} training episodes", flush=True)
    started = monotonic()
    result = atari_eval.evaluate(
        state,
        config,
        atari_eval.EvaluationConfig(episodes=config.eval_episodes, seed=config.eval_seed),
        training_scores=training_scores,
    )
    for name in ("return_mean", "return_median", "return_std", "return_sem", "human_normalized_score_percent"):
        if result[name] is not None:
            writer.add_scalar(f"eval/{name}", result[name], steps)
    writer.add_scalar("eval/training_episodes", episode, steps)
    if result["training_scores"] is not None:
        for name, score in result["training_scores"].items():
            if score is not None:
                writer.add_scalar(f"training/{name}", score, steps)
    writer.add_scalar("time/evaluation_seconds", monotonic() - started, steps)
    result["training_steps"] = steps
    result["training_episodes"] = episode
    writer.add_text("eval/report", f"```json\n{json.dumps(result, indent=2, allow_nan=False)}\n```", steps)
    normalized = result["human_normalized_score_percent"]
    normalized_text = f" human_normalized={normalized:.1f}%" if normalized is not None else " human_normalized=n/a"
    print(
        f"Evaluation: return={result['return_mean']:.1f}{normalized_text} over {config.eval_episodes} games", flush=True
    )


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
        loss = policy_loss + config.value_coef * value_loss - entropy_coef * entropy  # ()
        approx_kl = (jnp.expm1(log_ratio) - log_ratio).mean()  # ()
        clip_fraction = (jnp.abs(ratio - 1) > config.clip_coef).mean()  # ()
        return loss, PPOMetrics(
            policy_loss=policy_loss,  # ()
            value_loss=value_loss,  # ()
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


def collect_rollout(
    rollout_state: RolloutCarry,  # observations: (B, *O); carry leaves: (B, ...); episode fields: (B,)
    state: TrainState,
    envs: JaxVectorEnv,
    config: Config,
    key: jax.Array,  # () typed PRNG key
) -> tuple[RolloutCarry, RolloutResult]:
    """Collect T steps, bootstrap timeouts before resetting, and compute GAE.

    The caller owns the environments, configured with autoreset disabled, and
    initializes rollout_state after reset. Collection advances the environments
    without updating policy parameters or writing logs/checkpoints.

    Args:
        rollout_state: (B, ...) live state; see RolloutCarry for field shapes.
        state: Policy parameters and model application function.
        envs: Vector environments wrapped to exchange JAX arrays.
        config: Rollout length T, environment count B, and discount settings.
        key: () typed PRNG key for action sampling.

    Returns:
        carry: (B, ...) continuation state; see RolloutCarry for field shapes.
        rollout: (T, B, ...) training data and statistics; see RolloutResult for field shapes.
    """
    obs = rollout_state.observations  # (B, *O)
    carry = rollout_state.carry  # leaves: (B, ...)
    episode_start = rollout_state.episode_starts  # (B,)
    episode_returns = rollout_state.episode_returns  # (B,)
    episode_lengths = rollout_state.episode_lengths  # (B,)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    observation_steps: list[jax.Array] = []  # T entries of (B, *O)
    action_steps: list[jax.Array] = []  # T entries of (B,)
    log_prob_steps: list[jax.Array] = []  # T entries of (B,)
    value_steps: list[jax.Array] = []  # T entries of (B,)
    reward_steps: list[jax.Array] = []  # T entries of (B,)
    done_steps: list[jax.Array] = []  # T entries of (B,)
    episode_start_steps: list[jax.Array] = []  # T entries of (B,)

    # Snapshot memory for replay; update() differentiates only model parameters.
    rollout_carry = carry  # leaves: (B, ...), backend-specific trailing dimensions
    for _ in range(config.num_steps):
        observation_steps.append(obs)  # (B, *O)
        episode_start_steps.append(episode_start)  # (B,)
        key, action_key = jax.random.split(key)  # each () typed PRNG key
        action, log_prob, prediction, carry = act(
            state, obs, carry, episode_start, action_key
        )  # 3 x (B,), carry leaves: (B, ...)
        action_steps.append(action)  # (B,)
        log_prob_steps.append(log_prob)  # (B,)
        value_steps.append(prediction)  # (B,)
        obs, reward, terminated, truncated, _ = envs.step(action)  # (B, *O), 3 x (B,), info
        done = terminated | truncated  # (B,)
        done_steps.append(done)  # (B,)
        rollout_reward = reward  # (B,) raw environment rewards
        # Bootstrap time limits from the final observation, before resetting.
        timeout = truncated & ~terminated  # (B,)
        if timeout.any():
            rollout_reward = (
                reward + config.gamma * value(state, obs, carry, jnp.zeros(config.num_envs, dtype=bool)) * timeout
            )  # (B,)
        reward_steps.append(rollout_reward)  # (B,)
        episode_start = done  # (B,)
        episode_returns = episode_returns + reward  # (B,)
        episode_lengths = episode_lengths + 1  # (B,)
        finished_returns = episode_returns[done]  # (D,), D completed episodes this step
        finished_lengths = episode_lengths[done]  # (D,)
        finished_count = finished_returns.size
        completed_returns.extend(finished_returns.tolist())
        completed_lengths.extend(finished_lengths.tolist())
        episode_returns = jnp.where(done, 0.0, episode_returns)  # (B,)
        episode_lengths = jnp.where(done, 0, episode_lengths)  # (B,)
        if finished_count:
            obs, _ = envs.reset(options={"reset_mask": done})  # (B, *O)

    observations = jnp.stack(observation_steps)  # (T, B, *O)
    actions = jnp.stack(action_steps)  # (T, B)
    log_probs = jnp.stack(log_prob_steps)  # (T, B)
    values = jnp.stack(value_steps)  # (T, B)
    rewards = jnp.stack(reward_steps)  # (T, B)
    dones = jnp.stack(done_steps)  # (T, B)
    episode_starts = jnp.stack(episode_start_steps)  # (T, B)
    next_value = value(state, obs, carry, episode_start)  # (B,)
    advantages, returns = gae(
        rewards,
        dones,
        values,
        next_value,
        config.gamma,
        config.gae_lambda,
    )  # each (T, B)
    batch = PPOBatch(
        observations=observations,  # (T, B, *O)
        actions=actions,  # (T, B)
        old_log_probs=log_probs,  # (T, B)
        values=values,  # (T, B)
        advantages=advantages,  # (T, B)
        returns=returns,  # (T, B)
        carry=rollout_carry,  # leaves: (B, ...)
        episode_starts=episode_starts,  # (T, B)
    )
    return (
        RolloutCarry(
            observations=obs,  # (B, *O)
            carry=carry,  # leaves: (B, ...)
            episode_starts=episode_start,  # (B,)
            episode_returns=episode_returns,  # (B,)
            episode_lengths=episode_lengths,  # (B,)
        ),
        RolloutResult(
            batch=batch,  # observations: (T, B, *O); targets/starts: (T, B); carry leaves: (B, ...)
            key=key,  # () typed PRNG key
            completed_returns=tuple(completed_returns),
            completed_lengths=tuple(completed_lengths),
        ),
    )


def train(config: Config) -> TrainState:
    """Train with a fully resolved config, as returned by load_config."""
    with ExitStack() as resources:
        configure_compilation_cache()
        batch_size = config.num_envs * config.num_steps
        vector_cls = gym.vector.AsyncVectorEnv if config.vector_env == "async" else gym.vector.SyncVectorEnv
        # Spawn avoids forking JAX's threads or accelerator runtime.
        vector_options = {"context": "spawn"} if config.vector_env == "async" else {}
        envs = resources.enter_context(
            closing(
                JaxVectorEnv(
                    vector_cls(
                        [
                            partial(
                                make_env,
                                config.env_id,
                                frame_stack=config.frame_stack,
                                atari_preprocessing=config.atari_preprocessing,
                                observation_size=config.observation_size,
                            )
                            for _ in range(config.num_envs)
                        ],
                        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
                        **vector_options,
                    )
                )
            )
        )
        run_name = config.run_id
        run_dir = config.log_dir
        manager = resources.enter_context(closing(checkpoint_manager(run_dir)))
        obs, _ = envs.reset(seed=config.seed)  # (B, *O)
        key, init_key = jax.random.split(jax.random.key(config.seed))  # each () typed PRNG key
        model = make_model(config, envs.single_action_space.n)
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
        if progress is not None:
            state, key, start_iteration, completed_episodes, saved_returns, saved_lengths, training_return_sum = (
                progress
            )
            recent_returns.extend(saved_returns)
            recent_lengths.extend(saved_lengths)
            # Gym environments and wrappers are not serialized. Start fresh episodes
            # with fresh recurrent memory, retaining the policy and training RNGs.
            obs, _ = envs.reset(seed=config.seed + start_iteration * config.num_envs)  # (B, *O)
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
        rollout_state = RolloutCarry(
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
            rollout_state, rollout = collect_rollout(
                rollout_state, state, envs, config, key
            )  # carry: (B, ...); rollout arrays: (T, B, ...)
            batch = rollout.batch  # observations: (T, B, *O); targets/starts: (T, B); carry leaves: (B, ...)
            key = rollout.key  # () typed PRNG key
            recent_returns.extend(rollout.completed_returns)
            recent_lengths.extend(rollout.completed_lengths)
            finished_count = len(rollout.completed_returns)
            finished_return_sum = sum(rollout.completed_returns)
            window_episodes += finished_count
            window_return_sum += finished_return_sum
            window_length_sum += sum(rollout.completed_lengths)
            if training_return_sum is not None:
                training_return_sum += finished_return_sum
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
            metrics: list[PPOMetrics] = []  # scalar leaves per minibatch
            early_stop = False
            updates_done = 0
            for _ in range(config.update_epochs):
                key, shuffle_key = jax.random.split(key)  # each () typed PRNG key
                permutation = jax.random.permutation(shuffle_key, config.num_envs)  # (B,)
                for indices in jnp.split(permutation, config.num_minibatches):  # each (M,)
                    # M is the number of environments in this minibatch.
                    minibatch = batch.replace(
                        observations=batch.observations[:, indices],  # (T, M, *O)
                        actions=batch.actions[:, indices],  # (T, M)
                        old_log_probs=batch.old_log_probs[:, indices],  # (T, M)
                        values=batch.values[:, indices],  # (T, M)
                        advantages=batch.advantages[:, indices],  # (T, M)
                        returns=batch.returns[:, indices],  # (T, M)
                        carry=jax.tree.map(itemgetter(indices), batch.carry),  # leaves: (M, ...)
                        episode_starts=batch.episode_starts[:, indices],  # (T, M)
                    )
                    state, metric = update(state, minibatch, config, iteration)
                    metrics.append(metric)
                    if config.target_kl is not None and float(metric.approx_kl) > config.target_kl:
                        early_stop = True
                        break
                    updates_done += 1
                if early_stop:
                    break
            for name in window_metric_sums:
                window_metric_sums[name] += sum(float(getattr(metric, name)) for metric in metrics)
            window_metric_count += len(metrics)
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
                    training_scores=atari_eval.TrainingScores(
                        completed_episodes, training_return_sum, tuple(float(score) for score in recent_returns)
                    ),
                )
                # Restart after evaluation so long evaluations never cause catch-up runs.
                next_eval_time = monotonic() + config.eval_interval_seconds
            now = monotonic()
            window_seconds = now - window_start
            if window_seconds >= config.log_interval_seconds or iteration + 1 == num_iterations:
                mean_metrics = {name: total / window_metric_count for name, total in window_metric_sums.items()}
                sps = (steps - window_steps) / max(window_seconds, 1e-9)
                # Estimate from this session's completed rollouts, including evaluation/video overhead.
                elapsed_seconds = now - start
                eta_seconds = elapsed_seconds * (num_iterations - iteration - 1) / (iteration + 1 - start_iteration)
                scalars = {tag: total / window_rollouts for tag, total in window_rollout_sums.items()}
                scalars.update(
                    {
                        "losses/policy": mean_metrics["policy_loss"],
                        "losses/value": mean_metrics["value_loss"],
                        "policy/entropy": mean_metrics["entropy"],
                        "policy/approx_kl": mean_metrics["approx_kl"],
                        "policy/clip_fraction": mean_metrics["clip_fraction"],
                        "charts/steps_per_second": sps,
                        "charts/learning_rate": learning_rate,
                        "charts/entropy_coef": entropy_coef,
                        "charts/total_episodes": completed_episodes,
                        "charts/window_episodes": window_episodes,
                        "time/iteration_seconds": window_seconds / window_rollouts,
                        "time/window_seconds": window_seconds,
                        "time/elapsed_seconds": elapsed_seconds,
                        "time/eta_seconds": eta_seconds,
                    }
                )
                if window_episodes:
                    scalars["charts/return_mean_window"] = window_return_sum / window_episodes
                    scalars["charts/episode_length_mean_window"] = window_length_sum / window_episodes
                if recent_returns:
                    scalars["charts/return_mean_100"] = sum(recent_returns) / len(recent_returns)
                    scalars["charts/episode_length_mean_100"] = sum(recent_lengths) / len(recent_lengths)
                    if training_return_sum is not None:
                        scalars["charts/return_mean"] = training_return_sum / completed_episodes
                for tag, scalar in scalars.items():
                    writer.add_scalar(tag, float(scalar), steps)
                writer.flush()
                score = f"{window_return_sum / window_episodes:.1f}" if window_episodes else "n/a"
                length = f"{window_length_sum / window_episodes:.1f}" if window_episodes else "n/a"
                print(
                    f"step={steps} episodes={completed_episodes} return={score} "
                    f"episode_length={length} sps={sps:.0f} lr={learning_rate:.3g} "
                    f"policy={mean_metrics['policy_loss']:.3f} value={mean_metrics['value_loss']:.3f} "
                    f"entropy={mean_metrics['entropy']:.3f} kl={mean_metrics['approx_kl']:.4f} "
                    f"clipfrac={mean_metrics['clip_fraction']:.3f} ev={scalars['value/explained_variance']:.3f} "
                    f"updates={scalars['charts/updates_per_rollout']:.2f} "
                    f"early_stop={scalars['policy/early_stop']:.2f} "
                    f"s/iteration={scalars['time/iteration_seconds']:.2f} window={window_seconds:.1f}s "
                    f"elapsed={timedelta(seconds=int(elapsed_seconds))} eta={timedelta(seconds=int(eta_seconds))}",
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
