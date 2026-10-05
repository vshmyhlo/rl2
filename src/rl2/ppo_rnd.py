"""Recurrent PPO with Random Network Distillation for exploration.

Copied from ppo.py. Values have a final axis of (extrinsic, intrinsic); the
intrinsic return continues across resets while recurrent memory still resets.
"""

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import monotonic
from typing import Any, SupportsFloat

import ale_py
import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax import linen as nn
from flax.training.train_state import TrainState
from numpy.typing import NDArray
from tensorboardX import SummaryWriter

from rl2.multi_atari import register_envs
from rl2.utils import RunningMeanStd

type Array = jax.Array | NDArray[Any]
type LSTMCarry = tuple[jax.Array, jax.Array]
type PPOBatch = tuple[Array, Array, Array, Array, Array, LSTMCarry, Array]
type PPOMetrics = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


@dataclass(frozen=True)
class Config:
    env_id: str
    frame_stack: bool
    atari_preprocessing: bool
    seed: int
    total_steps: int
    num_envs: int
    vector_env: str
    num_steps: int
    num_minibatches: int
    update_epochs: int
    lstm_hidden_size: int
    learning_rate: float
    anneal_lr: bool
    gamma: float
    gae_lambda: float
    clip_coef: float
    target_kl: float | None
    entropy_coef: float
    value_coef: float
    max_grad_norm: float
    log_dir: str
    video_every_episodes: int
    video_speed: float
    bf16: bool = True
    video_max_frames: int | None = 900
    observation_size: int | None = None
    eval_every_minutes: float = 0.0
    eval_episodes: int = 100
    eval_seed: int = 10_000
    intrinsic_gamma: float = 0.99
    extrinsic_coef: float = 2.0
    intrinsic_coef: float = 1.0
    rnd_learning_rate: float = 0.0001
    rnd_update_fraction: float = 0.25
    rnd_warmup_steps: int = 128
    rnd_update_epochs: int = 4
    encoder_channels: tuple[int, ...] = (128, 256, 384, 512)
    embedding_size: int = 768


def load_config(path: str | Path) -> Config:
    with open(path) as file:
        settings = yaml.safe_load(file)
    if "encoder_channels" in settings:
        # Config is a static JIT argument, so YAML sequences must be hashable.
        settings["encoder_channels"] = tuple(settings["encoder_channels"])
    return Config(**settings)


def learning_rate_schedule(config: Config) -> optax.Schedule:
    num_rollouts = config.total_steps // (config.num_envs * config.num_steps)
    return optax.linear_schedule(
        config.learning_rate,
        0.0 if config.anneal_lr else config.learning_rate,
        num_rollouts,
    )


def initial_carry(num_envs: int, hidden_size: int) -> LSTMCarry:
    return (jnp.zeros((num_envs, hidden_size)), jnp.zeros((num_envs, hidden_size)))


def select_carry(carry: LSTMCarry, indices: NDArray[np.int64]) -> LSTMCarry:
    return carry[0][indices], carry[1][indices]


def rnd_frames(obs: Array) -> NDArray[np.uint8]:
    """Extract only the newest frame, keeping RGB channels when present."""
    frames = np.asarray(obs)[:, -1]
    return frames[..., None] if frames.ndim == 3 else frames


def normalize_rnd_observations(frames: Array, moments: RunningMeanStd) -> NDArray[np.float32]:
    return np.clip((np.asarray(frames, dtype=np.float32) - moments.mean) / np.sqrt(moments.var + 1e-8), -5, 5).astype(
        np.float32
    )


def normalize_intrinsic_rewards(
    rewards: NDArray[np.float32],
    discounted: NDArray[np.float64],
    moments: RunningMeanStd,
    gamma: float,
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    """Scale by discounted-return std without centering or resetting at game over."""
    returns = np.empty_like(rewards, dtype=np.float64)
    for t, reward in enumerate(rewards):
        discounted = gamma * discounted + reward
        returns[t] = discounted
    moments.update(returns.reshape(-1))
    return (rewards / np.sqrt(moments.var + 1e-8)).astype(np.float32), discounted


class RNDNetwork(nn.Module):
    predictor: bool = False
    channels: tuple[int, ...] = (32, 64, 64)
    feature_size: int = 512
    dtype: jax.typing.DTypeLike = jnp.float32

    @nn.compact
    def __call__(self, obs: Array) -> jax.Array:
        chex.assert_rank(obs, 4)
        chex.assert_type(obs, jnp.floating)
        x = jnp.asarray(obs, dtype=self.dtype)
        init = nn.initializers.variance_scaling(2.0, "fan_in", "truncated_normal")
        for channels, kernel, stride in zip(self.channels, (8, 4, 3), (4, 2, 1)):
            x = nn.leaky_relu(
                nn.Conv(channels, (kernel, kernel), strides=(stride, stride), kernel_init=init, dtype=self.dtype)(x)
            )
        x = x.reshape((x.shape[0], -1))
        if self.predictor:
            for _ in range(2):
                x = nn.relu(nn.Dense(self.feature_size, kernel_init=init, dtype=self.dtype)(x))
        x = nn.Dense(self.feature_size, kernel_init=init, dtype=self.dtype)(x)
        chex.assert_shape(x, (obs.shape[0], self.feature_size))
        chex.assert_type(x, self.dtype)
        # Intrinsic rewards and predictor losses accumulate in float32.
        return x.astype(jnp.float32)


@jax.jit
def rnd_reward(predictor: TrainState, target: TrainState, obs: Array) -> jax.Array:
    predicted = predictor.apply_fn({"params": predictor.params}, obs)
    expected = target.apply_fn({"params": target.params}, obs)
    return jnp.mean(jnp.square(predicted - expected), axis=-1)


@jax.jit
def update_rnd(
    predictor: TrainState, target: TrainState, obs: Array, key: jax.Array, update_fraction: float
) -> tuple[TrainState, jax.Array]:
    expected = jax.lax.stop_gradient(target.apply_fn({"params": target.params}, obs))
    mask = jax.random.uniform(key, (obs.shape[0],)) < update_fraction

    def loss_fn(params: optax.Params) -> jax.Array:
        predicted = predictor.apply_fn({"params": params}, obs)
        errors = jnp.mean(jnp.square(predicted - expected), axis=-1)
        return (errors * mask).sum() / jnp.maximum(mask.sum(), 1)

    loss, grads = jax.value_and_grad(loss_fn)(predictor.params)
    # An empty sample must not advance Adam's momentum or step counter.
    predictor = jax.lax.cond(mask.any(), lambda: predictor.apply_gradients(grads=grads), lambda: predictor)
    return predictor, loss


def warmup_rnd_observations(envs: gym.vector.VectorEnv, config: Config, moments: RunningMeanStd) -> Array:
    """Gather random frames for normalization, then start fresh training episodes."""
    obs, _ = envs.reset(seed=config.seed)
    moments.update(rnd_frames(obs))
    rng = np.random.default_rng(config.seed)
    for _ in range(config.rnd_warmup_steps):
        obs, _, terminated, truncated, _ = envs.step(rng.integers(envs.single_action_space.n, size=config.num_envs))
        moments.update(rnd_frames(obs))
        done = terminated | truncated
        if done.any():
            envs.reset(options={"reset_mask": done})
    obs, _ = envs.reset(seed=config.seed)
    return obs


class ResetLSTM(nn.Module):
    features: int
    dtype: jax.typing.DTypeLike = jnp.float32

    @nn.compact
    def __call__(
        self,
        carry: LSTMCarry,
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[LSTMCarry, jax.Array]:
        x, episode_starts = inputs
        carry = jax.tree.map(lambda c: jnp.where(episode_starts[:, None], 0, c), carry)
        # Float32 carry preserves recurrent accumulation and scan dtype stability.
        carry, x = nn.OptimizedLSTMCell(self.features, dtype=self.dtype)(carry, x)
        return carry, x.astype(self.dtype)


class ResidualBlock(nn.Module):
    """Pre-activation residual block with per-pixel channel normalization."""

    channels: int
    dtype: jax.typing.DTypeLike = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        for index in range(2):
            x = nn.relu(nn.LayerNorm(dtype=self.dtype)(x))
            x = nn.Conv(
                self.channels,
                (3, 3),
                padding="SAME",
                kernel_init=nn.initializers.variance_scaling(2.0 if index == 0 else 1.0, "fan_in", "truncated_normal"),
                dtype=self.dtype,
            )(x)
        return (residual + x) * jnp.asarray(2**-0.5, dtype=self.dtype)


class ActorCritic(nn.Module):
    num_actions: int
    lstm_hidden_size: int
    dtype: jax.typing.DTypeLike = jnp.float32
    encoder_channels: tuple[int, ...] = (128, 256, 384, 512)
    embedding_size: int = 768

    @nn.compact
    def __call__(
        self,
        obs: Array,
        carry: LSTMCarry,
        episode_starts: Array,
    ) -> tuple[LSTMCarry, jax.Array, jax.Array]:
        # [steps, environments, frames, height, width, (RGB channels)].
        chex.assert_rank(obs, {5, 6})
        chex.assert_type(obs, jnp.uint8)
        steps, environments = obs.shape[:2]
        chex.assert_shape(carry, (environments, self.lstm_hidden_size))
        chex.assert_type(carry, jnp.float32)
        chex.assert_shape(episode_starts, (steps, environments))
        chex.assert_type(episode_starts, jnp.bool_)
        obs = obs.reshape((-1, *obs.shape[2:]))
        if obs.ndim == 5:  # Raw RGB: combine stacked frames and color channels.
            x = jnp.transpose(obs, (0, 2, 3, 1, 4))
            x = x.reshape((*x.shape[:3], -1))
        else:
            x = jnp.moveaxis(obs, 1, -1)
        x = (x.astype(jnp.float32) / 255.0).astype(self.dtype)
        # Flax keeps parameters and LayerNorm statistics in float32 by default.
        init = nn.initializers.orthogonal(np.sqrt(2))
        # IMPALA-style stages; normalization is independent of rollout/minibatch size.
        # Variance scaling avoids expensive QR initialization of large visual kernels.
        visual_init = nn.initializers.variance_scaling(2.0, "fan_in", "truncated_normal")
        for stage, channels in enumerate(self.encoder_channels):
            x = nn.Conv(channels, (3, 3), padding="SAME", kernel_init=visual_init, dtype=self.dtype)(x)
            x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
            for block in range(2):
                x = ResidualBlock(channels, dtype=self.dtype, name=f"stage_{stage}_block_{block}")(x)
        x = nn.relu(nn.LayerNorm(name="encoder_norm", dtype=self.dtype)(x))
        # Preserve spatial position (6x6 at 84x84 input) for aiming and movement.
        x = nn.Dense(self.embedding_size, kernel_init=visual_init, dtype=self.dtype)(x.reshape((x.shape[0], -1)))
        x = nn.relu(nn.LayerNorm(name="shared_norm", dtype=self.dtype)(x))
        carry, x = nn.scan(
            ResetLSTM,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )(self.lstm_hidden_size, dtype=self.dtype, name="lstm")(
            carry,
            (x.reshape((steps, environments, -1)), episode_starts),
        )
        policy = nn.Dense(512, kernel_init=init, name="policy_hidden", dtype=self.dtype)(x)
        policy = nn.relu(nn.LayerNorm(name="policy_norm", dtype=self.dtype)(policy))
        critic = nn.Dense(512, kernel_init=init, name="value_hidden", dtype=self.dtype)(x)
        critic = nn.relu(nn.LayerNorm(name="value_norm", dtype=self.dtype)(critic))
        # Heads use the compute dtype; float32 outputs keep PPO loss arithmetic precise.
        logits = nn.Dense(
            self.num_actions, kernel_init=nn.initializers.orthogonal(0.01), name="policy_output", dtype=self.dtype
        )(policy)
        value = nn.Dense(2, kernel_init=nn.initializers.orthogonal(1.0), name="value_output", dtype=self.dtype)(critic)
        return carry, logits.astype(jnp.float32), value.astype(jnp.float32)


class AtariPreprocessing(gym.wrappers.AtariPreprocessing):
    def step(
        self,
        action: int | np.integer[Any],
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
) -> gym.Env[NDArray[np.uint8], int | np.integer[Any]]:
    if observation_size is not None and (type(observation_size) is not int or observation_size < 1):
        raise ValueError("observation_size must be a positive integer or null")
    gym.register_envs(ale_py)
    register_envs()
    env = gym.make(env_id, frameskip=1, render_mode=render_mode)
    if atari_preprocessing:
        env = AtariPreprocessing(env)
    if observation_size is not None:
        env = gym.wrappers.ResizeObservation(env, (observation_size, observation_size))
    if frame_stack:
        return gym.wrappers.FrameStackObservation(env, stack_size=4)
    return gym.wrappers.ReshapeObservation(env, (1, *env.observation_space.shape))


def action_log_prob(logits: Array, actions: Array) -> jax.Array:
    return jnp.take_along_axis(jax.nn.log_softmax(logits), actions[..., None], axis=-1)[..., 0]


@jax.jit
def act(
    state: TrainState,
    obs: Array,
    carry: LSTMCarry,
    episode_starts: Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, LSTMCarry]:
    carry, logits, values = state.apply_fn(
        {"params": state.params},
        obs[None],
        carry,
        episode_starts[None],
    )
    logits, values = logits[0], values[0]
    actions = jax.random.categorical(key, logits)
    return actions, action_log_prob(logits, actions), values, carry


@jax.jit
def value(state: TrainState, obs: Array, carry: LSTMCarry, episode_starts: Array) -> jax.Array:
    # Peek at the next value without advancing the rollout's recurrent state.
    return state.apply_fn({"params": state.params}, obs[None], carry, episode_starts[None])[2][0]


def log_video(
    state: TrainState,
    config: Config,
    writer: SummaryWriter,
    episode: int,
    steps: int,
) -> None:
    env = make_env(
        config.env_id,
        render_mode="rgb_array",
        frame_stack=config.frame_stack,
        atari_preprocessing=config.atari_preprocessing,
        observation_size=config.observation_size,
    )
    try:
        obs, _ = env.reset(seed=config.seed + episode)
        key = jax.random.fold_in(jax.random.key(config.seed), episode)
        frames = [env.render()]
        carry = initial_carry(1, config.lstm_hidden_size)
        while config.video_max_frames is None or len(frames) < config.video_max_frames:
            key, action_key = jax.random.split(key)
            actions, _, _, carry = act(state, obs[None], carry, jnp.zeros(1, dtype=bool), action_key)
            obs, _, terminated, truncated, _ = env.step(int(actions[0]))
            frames.append(env.render())
            if terminated or truncated:
                break
        # One recorded frame per agent step, accounting for optional action repeat.
        video = np.stack(frames).transpose(0, 3, 1, 2)[None]
        writer.add_video(
            "gameplay",
            video,
            steps,
            fps=config.video_speed * env.metadata["render_fps"] / (4 if config.atari_preprocessing else 1),
        )
        print(f"Recorded {len(frames)} gameplay frames after {episode} training episodes", flush=True)
    finally:
        env.close()


def log_evaluation(state: TrainState, config: Config, writer: SummaryWriter, episode: int, steps: int) -> None:
    """Evaluate the current policy and save scores plus the complete report to TensorBoard."""
    # Local import: atari_eval reuses PPO's observation wrapper and recurrent types.
    from rl2.atari_eval import EvaluationConfig, evaluate

    print(f"Evaluating {config.eval_episodes} games after {episode} training episodes", flush=True)
    started = monotonic()
    result = evaluate(state, config, EvaluationConfig(episodes=config.eval_episodes, seed=config.eval_seed))
    for name in ("return_mean", "return_median", "return_std", "return_sem", "human_normalized_score_percent"):
        if result[name] is not None:
            writer.add_scalar(f"eval/{name}", result[name], steps)
    writer.add_scalar("eval/training_episodes", episode, steps)
    writer.add_scalar("time/evaluation_seconds", monotonic() - started, steps)
    result["training_steps"] = steps
    result["training_episodes"] = episode
    writer.add_text("eval/report", f"```json\n{json.dumps(result, indent=2, allow_nan=False)}\n```", steps)
    normalized = result["human_normalized_score_percent"]
    normalized_text = f" human_normalized={normalized:.1f}%" if normalized is not None else " human_normalized=n/a"
    print(
        f"Evaluation: return={result['return_mean']:.1f}{normalized_text} over {config.eval_episodes} games", flush=True
    )


@jax.jit
def gae(
    rewards: Array,
    dones: Array,
    values: Array,
    next_value: Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Truncation bootstrap is already included in rewards; dones stop traces."""

    def step(
        carry: tuple[jax.Array, jax.Array],
        transition: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        advantage, next_value = carry
        reward, done, value = transition
        discount = gamma * (1.0 - done)
        delta = reward + discount * next_value - value
        advantage = delta + discount * gae_lambda * advantage
        return (advantage, value), advantage

    _, advantages = jax.lax.scan(
        step,
        (jnp.zeros_like(next_value), next_value),
        (rewards, dones, values),
        reverse=True,
    )
    return advantages, advantages + values


def explained_variance(values: Array, returns: Array) -> float:
    variance = np.var(returns)
    return float(1 - np.var(returns - values) / variance) if variance > 0 else np.nan


@partial(jax.jit, static_argnames="config")
def update(state: TrainState, batch: PPOBatch, config: Config) -> tuple[TrainState, PPOMetrics]:
    obs, actions, old_log_probs, advantages, returns, carry, episode_starts = batch
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def loss_fn(params: optax.Params) -> tuple[jax.Array, PPOMetrics]:
        _, logits, values = state.apply_fn({"params": params}, obs, carry, episode_starts)
        log_probs = action_log_prob(logits, actions)
        log_ratio = log_probs - old_log_probs
        ratio = jnp.exp(log_ratio)
        clipped = jnp.clip(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
        policy_loss = -jnp.minimum(ratio * advantages, clipped * advantages).mean()
        value_losses = 0.5 * jnp.square(values - returns).mean(axis=(0, 1))
        value_loss, intrinsic_value_loss = value_losses[0], value_losses[1]
        entropy = -(jax.nn.softmax(logits) * jax.nn.log_softmax(logits)).sum(-1).mean()
        loss = policy_loss + config.value_coef * (value_loss + intrinsic_value_loss) - config.entropy_coef * entropy
        approx_kl = (jnp.expm1(log_ratio) - log_ratio).mean()
        clip_fraction = (jnp.abs(ratio - 1) > config.clip_coef).mean()
        return loss, (policy_loss, value_loss, entropy, approx_kl, clip_fraction, intrinsic_value_loss)

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    if config.target_kl is None:
        return state.apply_gradients(grads=grads), metrics
    state = jax.lax.cond(
        metrics[3] > config.target_kl,
        lambda: state,
        lambda: state.apply_gradients(grads=grads),
    )
    return state, metrics


def train(config: Config) -> TrainState:
    batch_size = config.num_envs * config.num_steps
    if min(config.num_envs, config.num_steps, config.num_minibatches, config.update_epochs) < 1:
        raise ValueError("Environment, rollout, minibatch, and epoch counts must be positive")
    if config.num_envs % config.num_minibatches:
        raise ValueError("num_envs must be divisible by num_minibatches to preserve sequences")
    if config.lstm_hidden_size < 1:
        raise ValueError("lstm_hidden_size must be positive")
    if not config.encoder_channels or any(type(width) is not int or width < 1 for width in config.encoder_channels):
        raise ValueError("encoder_channels must contain positive integers")
    if type(config.embedding_size) is not int or config.embedding_size < 1:
        raise ValueError("embedding_size must be a positive integer")
    if config.total_steps < batch_size:
        raise ValueError("total_steps must cover at least one rollout")
    if config.video_max_frames is not None and (
        type(config.video_max_frames) is not int or config.video_max_frames < 1
    ):
        raise ValueError("video_max_frames must be a positive integer or null for a full episode")
    if config.video_every_episodes < 0:
        raise ValueError("video_every_episodes must be nonnegative (0 disables videos)")
    if not np.isfinite(config.video_speed) or config.video_speed <= 0:
        raise ValueError("video_speed must be positive and finite")
    if config.vector_env not in ("sync", "async"):
        raise ValueError("vector_env must be 'sync' or 'async'")
    if config.target_kl is not None and (not np.isfinite(config.target_kl) or config.target_kl <= 0):
        raise ValueError("target_kl must be positive and finite, or null to disable stopping")
    if not np.isfinite(config.intrinsic_gamma) or not 0 <= config.intrinsic_gamma < 1:
        raise ValueError("intrinsic_gamma must be finite and in [0, 1)")
    for name in ("extrinsic_coef", "intrinsic_coef"):
        coefficient = getattr(config, name)
        if not np.isfinite(coefficient) or coefficient < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not np.isfinite(config.rnd_learning_rate) or config.rnd_learning_rate <= 0:
        raise ValueError("rnd_learning_rate must be positive and finite")
    if not np.isfinite(config.rnd_update_fraction) or not 0 < config.rnd_update_fraction <= 1:
        raise ValueError("rnd_update_fraction must be in (0, 1]")
    if type(config.rnd_warmup_steps) is not int or config.rnd_warmup_steps < 0:
        raise ValueError("rnd_warmup_steps must be a nonnegative integer")
    if type(config.rnd_update_epochs) is not int or config.rnd_update_epochs < 1:
        raise ValueError("rnd_update_epochs must be a positive integer")

    if (
        type(config.eval_every_minutes) not in (int, float)
        or not np.isfinite(config.eval_every_minutes)
        or config.eval_every_minutes < 0
    ):
        raise ValueError("eval_every_minutes must be finite and nonnegative (0 disables evaluation)")
    if config.eval_every_minutes:
        from rl2.atari_eval import EvaluationConfig, validate_training_config

        EvaluationConfig(episodes=config.eval_episodes, seed=config.eval_seed)
        validate_training_config(config)

    vector_cls = gym.vector.AsyncVectorEnv if config.vector_env == "async" else gym.vector.SyncVectorEnv
    # Spawn avoids forking JAX's threads or accelerator runtime.
    vector_options = {"context": "spawn"} if config.vector_env == "async" else {}
    envs = vector_cls(
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
    writer = None
    try:
        run_name = f"ppo_rnd_{config.env_id.replace('/', '_')}_seed{config.seed}_{datetime.now(UTC):%Y%m%d-%H%M%S-%f}"
        run_dir = f"{config.log_dir.rstrip('/')}/{run_name}"
        writer = SummaryWriter(logdir=run_dir)
        writer.add_text("config", f"```yaml\n{yaml.safe_dump(asdict(config))}```", 0)
        print(f"TensorBoard run: {run_dir}", flush=True)
        devices = str(jax.devices())
        print(f"JAX devices: {devices}", flush=True)
        writer.add_text("devices", devices, 0)
        obs, _ = envs.reset(seed=config.seed)
        obs_moments = RunningMeanStd(rnd_frames(obs).shape[1:])
        print(f"Warming up RND normalization for {config.rnd_warmup_steps} vector steps", flush=True)
        obs = warmup_rnd_observations(envs, config, obs_moments)
        reward_moments = RunningMeanStd()
        discounted_intrinsic = np.zeros(config.num_envs, dtype=np.float64)
        key, init_key, predictor_key, target_key = jax.random.split(jax.random.key(config.seed), 4)
        # Keep RND's random stream independent of policy sampling and evaluation.
        rnd_key = jax.random.fold_in(key, 1)
        compute_dtype = jnp.bfloat16 if config.bf16 else jnp.float32
        predictor_model = RNDNetwork(predictor=True, dtype=compute_dtype)
        target_model = RNDNetwork(dtype=compute_dtype)
        rnd_example = normalize_rnd_observations(rnd_frames(obs[:1]), obs_moments)
        predictor = TrainState.create(
            apply_fn=predictor_model.apply,
            params=predictor_model.init(predictor_key, rnd_example)["params"],
            tx=optax.adam(config.rnd_learning_rate, eps=1e-5),
        )
        target = TrainState.create(
            apply_fn=target_model.apply,
            params=target_model.init(target_key, rnd_example)["params"],
            tx=optax.set_to_zero(),
        )
        model = ActorCritic(
            envs.single_action_space.n,
            config.lstm_hidden_size,
            dtype=compute_dtype,
            encoder_channels=config.encoder_channels,
            embedding_size=config.embedding_size,
        )
        carry = initial_carry(config.num_envs, config.lstm_hidden_size)
        episode_start = np.ones(config.num_envs, dtype=bool)
        lr_schedule = learning_rate_schedule(config)
        optimizer = optax.inject_hyperparams(
            lambda learning_rate: optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(learning_rate, eps=1e-5),
            )
        )
        state = TrainState.create(
            apply_fn=model.apply,
            params=model.init(
                init_key, obs[None, :1], initial_carry(1, config.lstm_hidden_size), episode_start[None, :1]
            )["params"],
            tx=optimizer(config.learning_rate),
        )
        parameter_count = sum(parameter.size for parameter in jax.tree.leaves(state.params))
        writer.add_scalar("model/params_millions", parameter_count / 1_000_000, 0)
        print(f"Model parameters: {parameter_count:,}", flush=True)
        writer.add_text("model/parameter_count", str(parameter_count), 0)
        rng = np.random.default_rng(config.seed)
        episode_returns = np.zeros(config.num_envs)
        episode_lengths = np.zeros(config.num_envs, dtype=np.int64)
        recent_returns = deque(maxlen=100)
        recent_lengths = deque(maxlen=100)
        completed_episodes = 0
        next_video_episode = config.video_every_episodes
        shape = (config.num_steps, config.num_envs)
        observations = np.empty((*shape, *obs.shape[1:]), dtype=np.uint8)
        actions = np.empty(shape, dtype=np.int32)
        log_probs, rewards, intrinsic_rewards = [np.empty(shape, dtype=np.float32) for _ in range(3)]
        values = np.empty((*shape, 2), dtype=np.float32)
        next_frames = np.empty((*shape, *rnd_frames(obs).shape[1:]), dtype=np.uint8)
        dones = np.empty(shape, dtype=bool)
        episode_starts = np.empty(shape, dtype=bool)

        # Keep asynchronous initialization out of the first rollout's timing.
        jax.block_until_ready((state, predictor, target, carry, key))
        start = monotonic()
        eval_interval_seconds = config.eval_every_minutes * 60
        next_eval_time = start + eval_interval_seconds
        for iteration in range(config.total_steps // batch_size):
            rollout_start = monotonic()
            env_seconds = 0.0
            model_seconds = 0.0
            # Truncated BPTT: preserve memory, but gradients stop at rollout boundaries.
            rollout_carry = jax.tree.map(jax.lax.stop_gradient, carry)
            for t in range(config.num_steps):
                observations[t] = obs
                episode_starts[t] = episode_start
                model_start = monotonic()
                key, action_key = jax.random.split(key)
                action, log_prob, prediction, carry = act(state, obs, carry, episode_start, action_key)
                actions[t], log_probs[t], values[t] = jax.device_get((action, log_prob, prediction))
                jax.block_until_ready(carry)
                model_seconds += monotonic() - model_start
                env_start = monotonic()
                obs, reward, terminated, truncated, _ = envs.step(actions[t])
                env_seconds += monotonic() - env_start
                # Save the actual successor, including terminal screens, before reset.
                next_frames[t] = rnd_frames(obs)
                dones[t] = terminated | truncated
                rewards[t] = np.sign(reward)
                # Bootstrap time limits from the final observation, before resetting.
                timeout = truncated & ~terminated
                if timeout.any():
                    model_start = monotonic()
                    rewards[t] += (
                        config.gamma
                        * np.asarray(
                            value(
                                state,
                                obs,
                                carry,
                                np.zeros(config.num_envs, dtype=bool),
                            )
                        )[:, 0]
                        * timeout
                    )
                    model_seconds += monotonic() - model_start
                episode_start = dones[t].copy()
                episode_returns += reward
                episode_lengths += 1
                recent_returns.extend(episode_returns[dones[t]])
                recent_lengths.extend(episode_lengths[dones[t]])
                completed_episodes += int(dones[t].sum())
                episode_returns[dones[t]] = 0
                episode_lengths[dones[t]] = 0
                if dones[t].any():
                    env_start = monotonic()
                    obs, _ = envs.reset(options={"reset_mask": dones[t].copy()})
                    env_seconds += monotonic() - env_start

            model_start = monotonic()
            next_value = jax.block_until_ready(value(state, obs, carry, episode_start))
            # Snapshot normalization implicitly by updating moments only after training.
            # Chunk by environment to avoid placing the entire image rollout on device.
            for indices in np.split(np.arange(config.num_envs), config.num_minibatches):
                frames = next_frames[:, indices].reshape((-1, *next_frames.shape[2:]))
                normalized = normalize_rnd_observations(frames, obs_moments)
                intrinsic_rewards[:, indices] = np.asarray(rnd_reward(predictor, target, normalized)).reshape(
                    config.num_steps, -1
                )
            raw_intrinsic_mean = float(intrinsic_rewards.mean())
            intrinsic_rewards, discounted_intrinsic = normalize_intrinsic_rewards(
                intrinsic_rewards, discounted_intrinsic, reward_moments, config.intrinsic_gamma
            )
            model_seconds += monotonic() - model_start
            extrinsic_advantages, extrinsic_returns = jax.device_get(
                gae(
                    rewards,
                    dones,
                    values[..., 0],
                    next_value[..., 0],
                    config.gamma,
                    config.gae_lambda,
                )
            )
            # The next value at game over belongs to the reset observation with fresh
            # recurrent memory. Intrinsic traces and discounted reward stats continue.
            intrinsic_advantages, intrinsic_returns = jax.device_get(
                gae(
                    intrinsic_rewards,
                    np.zeros_like(dones),
                    values[..., 1],
                    next_value[..., 1],
                    config.intrinsic_gamma,
                    config.gae_lambda,
                )
            )
            advantages = config.extrinsic_coef * extrinsic_advantages + config.intrinsic_coef * intrinsic_advantages
            returns = np.stack((extrinsic_returns, intrinsic_returns), axis=-1)
            rollout_seconds = monotonic() - rollout_start
            optimization_start = monotonic()
            # Keep uint8 rollouts on the host; transfer only each minibatch to JAX.
            batch = (observations, actions, log_probs, advantages, returns)
            learning_rate = float(lr_schedule(iteration))
            state = state.replace(
                opt_state=state.opt_state._replace(
                    hyperparams={
                        **state.opt_state.hyperparams,
                        "learning_rate": jnp.asarray(learning_rate),
                    }
                )
            )
            metrics = []
            early_stop = False
            updates_done = 0
            for _ in range(config.update_epochs):
                for indices in np.split(rng.permutation(config.num_envs), config.num_minibatches):
                    minibatch = (
                        *[x[:, indices] for x in batch],
                        select_carry(rollout_carry, indices),
                        episode_starts[:, indices],
                    )
                    state, metric = update(state, minibatch, config)
                    metrics.append(metric)
                    if config.target_kl is not None and float(metric[3]) > config.target_kl:
                        early_stop = True
                        break
                    updates_done += 1
                if early_stop:
                    break
            # Predictor training has its own epoch budget, unaffected by PPO KL stops.
            rnd_losses = []
            for _ in range(config.rnd_update_epochs):
                for indices in np.split(rng.permutation(config.num_envs), config.num_minibatches):
                    frames = next_frames[:, indices].reshape((-1, *next_frames.shape[2:]))
                    normalized = normalize_rnd_observations(frames, obs_moments)
                    rnd_key, update_key = jax.random.split(rnd_key)
                    predictor, rnd_loss = update_rnd(
                        predictor, target, normalized, update_key, config.rnd_update_fraction
                    )
                    rnd_losses.append(rnd_loss)
            # Metrics alone may be ready before the optimizer's parameter updates.
            jax.block_until_ready((state, metrics, predictor, rnd_losses))
            for frames in next_frames:
                obs_moments.update(frames)
            optimization_seconds = monotonic() - optimization_start
            policy_loss, value_loss, entropy, approx_kl, clip_fraction, intrinsic_value_loss = np.mean(
                jax.device_get(metrics),
                axis=0,
            )
            # Evaluate rollout-time predictions against the full rollout's GAE returns.
            explained_var = explained_variance(values[..., 0], extrinsic_returns)
            steps = (iteration + 1) * batch_size
            sps = steps / (monotonic() - start)
            for tag, scalar in {
                "losses/policy": policy_loss,
                "losses/value": value_loss,
                "losses/intrinsic_value": intrinsic_value_loss,
                "losses/rnd_predictor": float(np.mean(jax.device_get(rnd_losses))),
                "rnd/reward_raw_mean": raw_intrinsic_mean,
                "rnd/reward_normalized_mean": float(intrinsic_rewards.mean()),
                "rnd/return_std": float(np.sqrt(reward_moments.var)),
                "rnd/predictor_updates": int(predictor.step),
                "value/intrinsic_explained_variance": explained_variance(values[..., 1], intrinsic_returns),
                "policy/entropy": entropy,
                "charts/steps_per_second": sps,
                "time/rollout_seconds": rollout_seconds,
                "time/rollout_env_seconds": env_seconds,
                "time/rollout_model_seconds": model_seconds,
                "time/optimization_seconds": optimization_seconds,
                "policy/approx_kl": approx_kl,
                "policy/clip_fraction": clip_fraction,
                "value/explained_variance": explained_var,
                "charts/learning_rate": learning_rate,
                "charts/updates_per_rollout": updates_done,
                "policy/early_stop": early_stop,
                "charts/total_episodes": completed_episodes,
            }.items():
                writer.add_scalar(tag, float(scalar), steps)
            if recent_returns:
                writer.add_scalar("charts/return_mean_100", float(np.mean(recent_returns)), steps)
                writer.add_scalar("charts/episode_length_mean_100", float(np.mean(recent_lengths)), steps)
            while config.video_every_episodes and completed_episodes >= next_video_episode:
                log_video(state, config, writer, next_video_episode, steps)
                next_video_episode += config.video_every_episodes
            if eval_interval_seconds and monotonic() >= next_eval_time:
                log_evaluation(state, config, writer, completed_episodes, steps)
                # Restart after evaluation so long evaluations never cause catch-up runs.
                next_eval_time = monotonic() + eval_interval_seconds
            writer.flush()
            score = f"{np.mean(recent_returns):.1f}" if recent_returns else "n/a"
            length = f"{np.mean(recent_lengths):.1f}" if recent_lengths else "n/a"
            print(
                f"step={steps} episodes={completed_episodes} return={score} "
                f"episode_length={length} sps={sps:.0f} lr={learning_rate:.3g} "
                f"policy={policy_loss:.3f} value={value_loss:.3f} entropy={entropy:.3f} "
                f"kl={approx_kl:.4f} clipfrac={clip_fraction:.3f} ev={explained_var:.3f} "
                f"updates={updates_done} early_stop={early_stop}",
                flush=True,
            )
        return state
    finally:
        envs.close()
        if writer is not None:
            writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ppo_rnd_montezuma.yaml", help="Path to a YAML config")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
