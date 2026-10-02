"""Recurrent PPO with a normalized residual Atari CNN and LSTM."""

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
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
from rl2.observation_encoder import ConvObservationEncoder

type Array = jax.Array | NDArray[Any]
type LSTMCarry = tuple[jax.Array, jax.Array]
type PPOBatch = tuple[Array, Array, Array, Array, Array, LSTMCarry, Array]
type PPOMetrics = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


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
    encoder_max_flattened_size: int | None = 8192


def load_config(path: str | Path) -> Config:
    with open(path) as file:
        return Config(**yaml.safe_load(file))


def learning_rate_schedule(config: Config) -> optax.Schedule:
    num_rollouts = config.total_steps // (config.num_envs * config.num_steps)
    return optax.linear_schedule(
        config.learning_rate,
        0.0 if config.anneal_lr else config.learning_rate,
        num_rollouts,
    )


def initial_carry(num_envs: int, hidden_size: int) -> LSTMCarry:
    return (jnp.zeros((num_envs, hidden_size)), jnp.zeros((num_envs, hidden_size)))


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


class ActorCritic(nn.Module):
    num_actions: int
    lstm_hidden_size: int
    dtype: jax.typing.DTypeLike = jnp.float32
    encoder_channels: tuple[int, ...] = (32, 64, 128, 256)
    embedding_size: int = 768
    encoder_max_flattened_size: int | None = 8192

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
        x = ConvObservationEncoder(
            encoder_channels=self.encoder_channels,
            embedding_size=self.embedding_size,
            max_flattened_size=self.encoder_max_flattened_size,
            dtype=self.dtype,
            name="encoder",
        )(obs.reshape((-1, *obs.shape[2:])))
        init = nn.initializers.orthogonal(np.sqrt(2))
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
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0), name="value_output", dtype=self.dtype)(critic)
        return carry, logits.astype(jnp.float32), value.squeeze(-1).astype(jnp.float32)


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
    batch_size = config.num_envs * config.num_steps
    if min(config.num_envs, config.num_steps, config.num_minibatches, config.update_epochs) < 1:
        raise ValueError("Environment, rollout, minibatch, and epoch counts must be positive")
    if config.num_envs % config.num_minibatches:
        raise ValueError("num_envs must be divisible by num_minibatches to preserve sequences")
    if config.lstm_hidden_size < 1:
        raise ValueError("lstm_hidden_size must be positive")
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
        model = ActorCritic(
            envs.single_action_space.n,
            config.lstm_hidden_size,
            dtype=jnp.bfloat16 if config.bf16 else jnp.float32,
            encoder_max_flattened_size=config.encoder_max_flattened_size,
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
    parser.add_argument(
        "--platform",
        choices=("cpu", "cuda", "metal"),
        help="Require a JAX backend (default: JAX_PLATFORMS or automatic selection; CUDA needs --extra cuda12)",
    )
    args = parser.parse_args()
    if args.platform is not None:
        # Select before initializing JAX; an unavailable backend must fail instead of falling back to CPU.
        jax.config.update("jax_platforms", args.platform)
        jax.devices()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
