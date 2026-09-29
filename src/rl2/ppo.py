"""Clipped PPO with a shared Atari CNN."""

from collections import deque
from dataclasses import dataclass
from functools import partial
from time import monotonic

import ale_py
from flax import linen as nn
from flax.training.train_state import TrainState
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro


@dataclass(frozen=True)
class Config:
    env_id: str = "ALE/Pong-v5"
    seed: int = 1
    total_steps: int = 10_000_000
    num_envs: int = 8
    num_steps: int = 128
    num_minibatches: int = 4
    update_epochs: int = 4
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.1
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


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


def make_env(env_id):
    gym.register_envs(ale_py)
    env = gym.make(env_id, frameskip=1)
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

    envs = gym.vector.SyncVectorEnv(
        [partial(make_env, config.env_id) for _ in range(config.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )
    try:
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
        recent_returns = deque(maxlen=100)
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
                recent_returns.extend(episode_returns[dones[t]])
                episode_returns[dones[t]] = 0
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
            score = f"{np.mean(recent_returns):.1f}" if recent_returns else "n/a"
            print(f"step={steps} return={score} sps={steps / (monotonic() - start):.0f} "
                  f"policy={policy_loss:.3f} value={value_loss:.3f} entropy={entropy:.3f}",
                  flush=True)
        return state
    finally:
        envs.close()


def main():
    train(tyro.cli(Config))


if __name__ == "__main__":
    main()
