"""Recurrent PPO with a residual Atari CNN and an LSTM or GDN2 backbone."""

import argparse
import json
from collections import deque
from dataclasses import asdict
from datetime import datetime
from functools import partial
from pathlib import Path
from time import monotonic
from typing import Annotated, Any, Literal, SupportsFloat, cast

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
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass
from tensorboardX import SummaryWriter

from rl2.gdn2 import GatedDeltaNet2Config, GatedDeltaNet2Stack, GatedDeltaNet2StackCarry
from rl2.jax_cache import configure_compilation_cache
from rl2.lstm import LSTM, LSTMCarry, initial_carry
from rl2.multi_atari import register_envs
from rl2.observation_encoder import DEFAULT_STAGES, ConvObservationEncoder, ConvStage, ConvStages
from rl2.shape_checker import ShapeChecker

type Array = jax.Array | NDArray[Any]
type RecurrentCarry = LSTMCarry | GatedDeltaNet2StackCarry
type ModelType = Literal["lstm", "gdn2"]
type PPOBatch = tuple[Array, Array, Array, Array, Array, RecurrentCarry, Array]
type PPOMetrics = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LSTMConfig:
    type: Literal["lstm"] = "lstm"
    hidden_size: Annotated[int, Field(gt=0, strict=True)] = 1536


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class GDN2Config:
    type: Literal["gdn2"] = "gdn2"
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


type ModelConfig = Annotated[LSTMConfig | GDN2Config, Field(discriminator="type")]


@dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
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
    encoder_stages: ConvStages = DEFAULT_STAGES
    model: ModelConfig = LSTMConfig()


def load_config(path: str | Path) -> Config:
    with open(path) as file:
        settings = yaml.safe_load(file)
    if "encoder_stages" in settings:
        settings["encoder_stages"] = tuple(ConvStage(**stage) for stage in settings["encoder_stages"])
    return Config(**settings)


def learning_rate_schedule(config: Config) -> optax.Schedule:
    num_rollouts = config.total_steps // (config.num_envs * config.num_steps)
    return optax.linear_schedule(
        config.learning_rate,
        0.0 if config.anneal_lr else config.learning_rate,
        num_rollouts,
    )


class ResetGDN2(nn.Module):
    config: GatedDeltaNet2Config
    num_layers: int
    intermediate_size: int

    @nn.compact
    def __call__(
        self,
        carry: GatedDeltaNet2StackCarry,
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[GatedDeltaNet2StackCarry, jax.Array]:
        x, episode_starts = inputs
        sc = ShapeChecker(D=self.config.hidden_size)
        sc.check(x, "BD", self.config.dtype)
        sc.check(episode_starts, "B", jnp.bool_)

        def reset(leaf: jax.Array) -> jax.Array:
            sc = ShapeChecker(B=x.shape[0])
            sc.check(leaf, "B" + "HKV"[: leaf.ndim - 1], jnp.float32)
            mask = episode_starts.reshape((x.shape[0],) + (1,) * (leaf.ndim - 1))
            return jnp.where(mask, 0, leaf)

        # Reset matrix memory AND short-convolution history in every layer.
        carry = jax.tree.map(reset, carry)
        backbone = GatedDeltaNet2Stack(self.config, self.num_layers, self.intermediate_size)
        carry, x = backbone.step(x, jnp.ones((x.shape[0],), dtype=jnp.bool_), carry)
        sc.check(x, "BD", self.config.dtype)
        return carry, x


class ActorCritic(nn.Module):
    num_actions: int
    model: ModelConfig = LSTMConfig()
    dtype: jax.typing.DTypeLike = jnp.float32
    encoder_stages: ConvStages = DEFAULT_STAGES
    embedding_size: int = 768

    @nn.nowrap
    def initial_carry(self, num_envs: int) -> RecurrentCarry:
        if self.model.type == "lstm":
            return initial_carry(num_envs, self.model.hidden_size)
        return GatedDeltaNet2Stack(
            self.model.mixer_config(self.dtype), self.model.num_layers, self.model.intermediate_size
        ).initial_carry(num_envs)

    def setup(self) -> None:
        self.encoder = ConvObservationEncoder(
            stages=self.encoder_stages,
            embedding_size=self.embedding_size,
            dtype=self.dtype,
            name="encoder",
        )
        if self.model.type == "lstm":
            self.recurrent = LSTM(self.model.hidden_size, dtype=self.dtype, name="lstm")
        else:
            self.gdn2_input = nn.Dense(self.model.hidden_size, dtype=self.dtype, name="gdn2_input")
            self.recurrent = ResetGDN2(
                self.model.mixer_config(self.dtype), self.model.num_layers, self.model.intermediate_size, name="gdn2"
            )
        init = nn.initializers.orthogonal(np.sqrt(2))
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
        sc = ShapeChecker(E=self.embedding_size)
        sc.check(obs, "BFHW" if obs.ndim == 4 else "BFHWC", jnp.uint8)
        x = self.encoder(obs)
        sc.check(x, "BE", self.dtype)
        if self.model.type == "gdn2":
            x = self.gdn2_input(x).astype(self.dtype)
            projection_sc = ShapeChecker(B=obs.shape[0], D=self.model.hidden_size)
            projection_sc.check(x, "BD", self.dtype)
        return x

    def _heads(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        # Flatten time and batch for sequence calls, keeping the same heads as step().
        sc = ShapeChecker(A=self.num_actions)
        x = x.astype(self.dtype)
        sc.check(x, "BD", self.dtype)
        policy = nn.relu(self.policy_norm(self.policy_hidden(x)))
        critic = nn.relu(self.value_norm(self.value_hidden(x)))
        # Heads use the compute dtype; float32 outputs keep PPO loss arithmetic precise.
        logits = self.policy_output(policy).astype(jnp.float32)
        values = self.value_output(critic).squeeze(-1).astype(jnp.float32)
        sc.check(logits, "BA", jnp.float32)
        sc.check(values, "B", jnp.float32)
        return logits, values

    def __call__(
        self,
        obs: Array,
        carry: RecurrentCarry,
        episode_starts: Array,
    ) -> tuple[RecurrentCarry, jax.Array, jax.Array]:
        """Process time-stacked observations [T,B,F,H,W,(C)] and reset masks [T,B]."""
        chex.assert_rank(obs, {5, 6})
        sc = ShapeChecker(A=self.num_actions)
        sc.check(obs, "TBFHW" if obs.ndim == 5 else "TBFHWC", jnp.uint8)
        sc.check(episode_starts, "TB", jnp.bool_)
        steps, environments = sc["TB"]
        x = self._encode(obs.reshape((-1, *obs.shape[2:])))
        x = x.reshape((steps, environments, -1))
        # Batch the CNN and heads across time; scan only recurrent computation.
        if self.model.type == "lstm":
            carry, x = self.recurrent(x, cast(LSTMCarry, carry), episode_starts)
        else:

            def recurrent_step(
                cell: ResetGDN2,
                memory: GatedDeltaNet2StackCarry,
                inputs: tuple[jax.Array, jax.Array],
            ) -> tuple[GatedDeltaNet2StackCarry, jax.Array]:
                return cell(memory, inputs)

            carry, x = nn.scan(
                recurrent_step,
                variable_broadcast="params",
                split_rngs={"params": False},
                in_axes=0,
                out_axes=0,
            )(self.recurrent, cast(GatedDeltaNet2StackCarry, carry), (x, episode_starts))
        logits, values = self._heads(x.reshape((steps * environments, -1)))
        logits = logits.reshape((steps, environments, self.num_actions))
        values = values.reshape((steps, environments))
        sc.check(logits, "TBA", jnp.float32)
        sc.check(values, "TB", jnp.float32)
        return carry, logits, values

    def step(
        self,
        obs: Array,
        carry: RecurrentCarry,
        episode_starts: Array,
    ) -> tuple[RecurrentCarry, jax.Array, jax.Array]:
        """Process one observation per environment [B,F,H,W,(C)] with reset masks [B]."""
        chex.assert_rank(obs, {4, 5})
        sc = ShapeChecker(A=self.num_actions)
        sc.check(obs, "BFHW" if obs.ndim == 4 else "BFHWC", jnp.uint8)
        sc.check(episode_starts, "B", jnp.bool_)
        x = self._encode(obs)
        if self.model.type == "lstm":
            carry, x = self.recurrent.step(x, cast(LSTMCarry, carry), episode_starts)
        else:
            carry, x = self.recurrent(cast(GatedDeltaNet2StackCarry, carry), (x, episode_starts))
        logits, values = self._heads(x)
        sc.check(logits, "BA", jnp.float32)
        sc.check(values, "B", jnp.float32)
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
    grayscale_obs: bool = True,
) -> gym.Env[NDArray[np.uint8], int | np.integer[Any]]:
    chex.assert_type(grayscale_obs, bool)
    if observation_size is not None and (type(observation_size) is not int or observation_size < 1):
        raise ValueError("observation_size must be a positive integer or null")
    gym.register_envs(ale_py)
    register_envs()
    env = gym.make(env_id, frameskip=1, render_mode=render_mode)
    if atari_preprocessing:
        env = AtariPreprocessing(env, grayscale_obs=grayscale_obs)
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
        carry = initial_model_carry(config, 1)
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
        value_loss = 0.5 * jnp.square(values - returns).mean()
        entropy = -(jax.nn.softmax(logits) * jax.nn.log_softmax(logits)).sum(-1).mean()
        loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
        approx_kl = (jnp.expm1(log_ratio) - log_ratio).mean()
        clip_fraction = (jnp.abs(ratio - 1) > config.clip_coef).mean()
        return loss, (policy_loss, value_loss, entropy, approx_kl, clip_fraction)

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
    configure_compilation_cache()
    batch_size = config.num_envs * config.num_steps
    if min(config.num_envs, config.num_steps, config.num_minibatches, config.update_epochs) < 1:
        raise ValueError("Environment, rollout, minibatch, and epoch counts must be positive")
    if config.num_envs % config.num_minibatches:
        raise ValueError("num_envs must be divisible by num_minibatches to preserve sequences")
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
        run_name = f"{config.env_id.replace('/', '_')}_seed{config.seed}_{datetime.now():%Y%m%d-%H%M%S-%f}"
        run_dir = f"{config.log_dir.rstrip('/')}/{run_name}"
        writer = SummaryWriter(logdir=run_dir)
        writer.add_text("config", f"```yaml\n{yaml.safe_dump(asdict(config))}```", 0)
        print(f"TensorBoard run: {run_dir}", flush=True)
        devices = str(jax.devices())
        print(f"JAX devices: {devices}", flush=True)
        writer.add_text("devices", devices, 0)
        obs, _ = envs.reset(seed=config.seed)
        key, init_key = jax.random.split(jax.random.key(config.seed))
        model = make_model(config, envs.single_action_space.n)
        carry = model.initial_carry(config.num_envs)
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
            params=model.init(init_key, obs[None, :1], model.initial_carry(1), episode_start[None, :1])["params"],
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
        log_probs, values, rewards = [np.empty(shape, dtype=np.float32) for _ in range(3)]
        dones = np.empty(shape, dtype=bool)
        episode_starts = np.empty(shape, dtype=bool)

        # Keep asynchronous initialization out of the first rollout's timing.
        jax.block_until_ready((state, carry, key))
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
                        )
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
            model_seconds += monotonic() - model_start
            advantages, returns = jax.device_get(
                gae(
                    rewards,
                    dones,
                    values,
                    next_value,
                    config.gamma,
                    config.gae_lambda,
                )
            )
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
                        jax.tree.map(lambda c: c[indices], rollout_carry),
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
            # Metrics alone may be ready before the optimizer's parameter updates.
            jax.block_until_ready((state, metrics))
            optimization_seconds = monotonic() - optimization_start
            policy_loss, value_loss, entropy, approx_kl, clip_fraction = np.mean(
                jax.device_get(metrics),
                axis=0,
            )
            # Evaluate rollout-time predictions against the full rollout's GAE returns.
            explained_var = explained_variance(values, returns)
            steps = (iteration + 1) * batch_size
            sps = steps / (monotonic() - start)
            for tag, scalar in {
                "losses/policy": policy_loss,
                "losses/value": value_loss,
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
    parser.add_argument("--config", default="configs/ppo.yaml", help="Path to a YAML config")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
