"""Clipped PPO with a shared Atari CNN."""

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from time import monotonic

import ale_py
from flax import linen as nn
from flax.training.train_state import TrainState
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tensorboardX import SummaryWriter
import yaml


@dataclass(frozen=True)
class Config:
    env_id: str
    seed: int
    total_steps: int
    num_envs: int
    num_steps: int
    num_minibatches: int
    update_epochs: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_coef: float
    entropy_coef: float
    value_coef: float
    max_grad_norm: float
    log_dir: str
    video_every_episodes: int


def load_config(path):
    with open(path) as file:
        return Config(**yaml.safe_load(file))


class ActorCritic(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, obs):
        x = jnp.moveaxis(obs, 1, -1).astype(jnp.float32) / 255.0
        init = nn.initializers.orthogonal(np.sqrt(2))
        for channels, kernel, stride in [(32, 8, 4), (64, 4, 2), (64, 3, 1)]:
            x = nn.relu(nn.Conv(channels, (kernel, kernel), (stride, stride),
                                padding="VALID", kernel_init=init)(x))
        x = nn.relu(nn.Dense(512, kernel_init=init)(x.reshape((x.shape[0], -1))))
        logits = nn.Dense(self.num_actions, kernel_init=nn.initializers.orthogonal(0.01))(x)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)
        return logits, value.squeeze(-1)


def make_env(env_id, render_mode=None):
    gym.register_envs(ale_py)
    env = gym.make(env_id, frameskip=1, render_mode=render_mode)
    env = gym.wrappers.AtariPreprocessing(env)
    return gym.wrappers.FrameStackObservation(env, stack_size=4)


def action_log_prob(logits, actions):
    return jnp.take_along_axis(jax.nn.log_softmax(logits), actions[:, None], axis=-1)[:, 0]


@jax.jit
def act(state, obs, key):
    logits, values = state.apply_fn({"params": state.params}, obs)
    actions = jax.random.categorical(key, logits)
    return actions, action_log_prob(logits, actions), values


@jax.jit
def value(state, obs):
    return state.apply_fn({"params": state.params}, obs)[1]


def log_video(state, config, writer, episode, steps):
    env = make_env(config.env_id, render_mode="rgb_array")
    try:
        obs, _ = env.reset(seed=config.seed + episode)
        key = jax.random.fold_in(jax.random.key(config.seed), episode)
        frames = [env.render()]
        done = False
        while not done:
            key, action_key = jax.random.split(key)
            actions, _, _ = act(state, obs[None], action_key)
            obs, _, terminated, truncated, _ = env.step(int(actions[0]))
            frames.append(env.render())
            done = terminated or truncated
        # One RGB frame per agent step; Atari preprocessing repeats each action 4 times.
        video = np.stack(frames).transpose(0, 3, 1, 2)[None]
        writer.add_video("gameplay", video, steps, fps=env.metadata["render_fps"] / 4)
        print(f"Recorded game after {episode} training episodes", flush=True)
    finally:
        env.close()


@jax.jit
def gae(rewards, dones, values, next_value, gamma, gae_lambda):
    """Truncation bootstrap is already included in rewards; dones stop traces."""
    def step(carry, transition):
        advantage, next_value = carry
        reward, done, value = transition
        discount = gamma * (1.0 - done)
        delta = reward + discount * next_value - value
        advantage = delta + discount * gae_lambda * advantage
        return (advantage, value), advantage

    _, advantages = jax.lax.scan(
        step, (jnp.zeros_like(next_value), next_value),
        (rewards, dones, values), reverse=True,
    )
    return advantages, advantages + values


@partial(jax.jit, static_argnames="config")
def update(state, batch, config):
    obs, actions, old_log_probs, advantages, returns = batch
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def loss_fn(params):
        logits, values = state.apply_fn({"params": params}, obs)
        log_probs = action_log_prob(logits, actions)
        ratio = jnp.exp(log_probs - old_log_probs)
        clipped = jnp.clip(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
        policy_loss = -jnp.minimum(ratio * advantages, clipped * advantages).mean()
        value_loss = 0.5 * jnp.square(values - returns).mean()
        entropy = -(jax.nn.softmax(logits) * jax.nn.log_softmax(logits)).sum(-1).mean()
        loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
        return loss, (policy_loss, value_loss, entropy)

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), metrics


def train(config: Config):
    batch_size = config.num_envs * config.num_steps
    if min(config.num_envs, config.num_steps, config.num_minibatches, config.update_epochs) < 1:
        raise ValueError("Environment, rollout, minibatch, and epoch counts must be positive")
    if batch_size % config.num_minibatches:
        raise ValueError("num_envs * num_steps must be divisible by num_minibatches")
    if config.total_steps < batch_size:
        raise ValueError("total_steps must cover at least one rollout")
    if config.video_every_episodes < 0:
        raise ValueError("video_every_episodes must be nonnegative (0 disables videos)")

    envs = gym.vector.SyncVectorEnv(
        [partial(make_env, config.env_id) for _ in range(config.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )
    writer = None
    try:
        run_name = f"{config.env_id.replace('/', '_')}_seed{config.seed}_{datetime.now():%Y%m%d-%H%M%S-%f}"
        run_dir = Path(config.log_dir) / run_name
        writer = SummaryWriter(logdir=str(run_dir))
        writer.add_text("config", f"```yaml\n{yaml.safe_dump(asdict(config))}```", 0)
        print(f"TensorBoard run: {run_dir}", flush=True)
        obs, _ = envs.reset(seed=config.seed)
        key, init_key = jax.random.split(jax.random.key(config.seed))
        model = ActorCritic(envs.single_action_space.n)
        state = TrainState.create(
            apply_fn=model.apply,
            params=model.init(init_key, obs[:1])["params"],
            tx=optax.chain(optax.clip_by_global_norm(config.max_grad_norm),
                           optax.adam(config.learning_rate, eps=1e-5)),
        )
        rng = np.random.default_rng(config.seed)
        episode_returns = np.zeros(config.num_envs)
        episode_lengths = np.zeros(config.num_envs, dtype=np.int64)
        recent_returns = deque(maxlen=100)
        recent_lengths = deque(maxlen=100)
        completed_episodes = 0
        next_video_episode = config.video_every_episodes
        start = monotonic()
        shape = (config.num_steps, config.num_envs)
        observations = np.empty((*shape, *obs.shape[1:]), dtype=np.uint8)
        actions = np.empty(shape, dtype=np.int32)
        log_probs, values, rewards = [np.empty(shape, dtype=np.float32) for _ in range(3)]
        dones = np.empty(shape, dtype=bool)

        for iteration in range(config.total_steps // batch_size):
            for t in range(config.num_steps):
                observations[t] = obs
                key, action_key = jax.random.split(key)
                actions[t], log_probs[t], values[t] = jax.device_get(act(state, obs, action_key))
                obs, reward, terminated, truncated, _ = envs.step(actions[t])
                dones[t] = terminated | truncated
                rewards[t] = np.sign(reward)
                # Bootstrap time limits from the final observation, before resetting.
                timeout = truncated & ~terminated
                if timeout.any():
                    rewards[t] += config.gamma * np.asarray(value(state, obs)) * timeout
                episode_returns += reward
                episode_lengths += 1
                recent_returns.extend(episode_returns[dones[t]])
                recent_lengths.extend(episode_lengths[dones[t]])
                completed_episodes += int(dones[t].sum())
                episode_returns[dones[t]] = 0
                episode_lengths[dones[t]] = 0
                if dones[t].any():
                    obs, _ = envs.reset(options={"reset_mask": dones[t].copy()})

            advantages, returns = jax.device_get(gae(
                rewards, dones, values, value(state, obs), config.gamma, config.gae_lambda,
            ))
            # Keep uint8 rollouts on the host; transfer only each minibatch to JAX.
            batch = [x.reshape((batch_size, *x.shape[2:])) for x in
                     (observations, actions, log_probs, advantages, returns)]
            metrics = []
            for _ in range(config.update_epochs):
                for indices in np.split(rng.permutation(batch_size), config.num_minibatches):
                    state, metric = update(state, tuple(x[indices] for x in batch), config)
                    metrics.append(metric)
            policy_loss, value_loss, entropy = np.mean(jax.device_get(metrics), axis=0)
            steps = (iteration + 1) * batch_size
            sps = steps / (monotonic() - start)
            for tag, scalar in {"losses/policy": policy_loss, "losses/value": value_loss,
                                "policy/entropy": entropy, "charts/steps_per_second": sps}.items():
                writer.add_scalar(tag, float(scalar), steps)
            if recent_returns:
                writer.add_scalar("charts/return_mean_100", float(np.mean(recent_returns)), steps)
                writer.add_scalar("charts/episode_length_mean_100", float(np.mean(recent_lengths)), steps)
            while config.video_every_episodes and completed_episodes >= next_video_episode:
                log_video(state, config, writer, next_video_episode, steps)
                next_video_episode += config.video_every_episodes
            writer.flush()
            score = f"{np.mean(recent_returns):.1f}" if recent_returns else "n/a"
            length = f"{np.mean(recent_lengths):.1f}" if recent_lengths else "n/a"
            print(f"step={steps} return={score} episode_length={length} sps={sps:.0f} "
                  f"policy={policy_loss:.3f} value={value_loss:.3f} entropy={entropy:.3f}",
                  flush=True)
        return state
    finally:
        envs.close()
        if writer is not None:
            writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ppo.yaml", help="Path to a YAML config")
    train(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
