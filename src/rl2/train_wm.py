"""Train a Mamba world model on Space Invaders with a uniform random policy.

Run with ``uv run python -m rl2.train_wm``. Observations are single grayscale
frames, normalized to [0, 1]; rewards retain their environment scale. Each
rollout receives one Adam update using next-frame MSE, reward MSE, and terminal
binary cross entropy. Mamba history crosses chunks with truncated BPTT and
resets at episode boundaries. No policy is learned.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import monotonic

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization, struct
from flax.training.train_state import TrainState
from numpy.typing import NDArray
from tensorboardX import SummaryWriter

from rl2.mamba3 import Mamba3Carry
from rl2.ppo import make_env
from rl2.wm import MambaWorldModel


@dataclass(frozen=True)
class Config:
    seed: int = 1
    total_steps: int = 100_000
    num_envs: int = 8
    num_steps: int = 32
    observation_size: int = 84
    d_model: int = 128
    d_state: int = 64
    headdim: int = 32
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    bf16: bool = False
    log_dir: str = "runs/wm"
    log_every: int = 10
    checkpoint_every: int = 100

    def __post_init__(self) -> None:
        for value in (
            self.total_steps,
            self.num_envs,
            self.num_steps,
            self.observation_size,
            self.d_model,
            self.d_state,
            self.headdim,
            self.log_every,
            self.checkpoint_every,
        ):
            chex.assert_type(value, int)
            chex.assert_scalar_positive(value)
        chex.assert_is_divisible(self.total_steps, self.num_envs)
        chex.assert_is_divisible(2 * self.d_model, self.headdim)
        chex.assert_is_divisible(self.d_state, 2)
        if self.d_state < 4:
            raise ValueError("d_state must be at least 4 for Mamba's default rotary fraction")
        chex.assert_type(self.seed, int)
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        for value in (self.learning_rate, self.max_grad_norm):
            chex.assert_scalar_positive(value)
            if not np.isfinite(value):
                raise ValueError("learning_rate and max_grad_norm must be finite")


@struct.dataclass
class Batch:
    """Time-major transitions; next_observations always precede any reset."""

    observations: jax.Array
    actions: jax.Array
    next_observations: jax.Array
    rewards: jax.Array
    terminated: jax.Array
    episode_starts: jax.Array

    def validate(self) -> None:
        chex.assert_scalar_positive(self.observations.ndim - 2)
        chex.assert_equal_shape((self.observations, self.next_observations))
        chex.assert_type((self.observations, self.next_observations), jnp.uint8)
        leading = self.observations.shape[:2]
        for size in leading:
            chex.assert_scalar_positive(size)
        chex.assert_shape((self.actions, self.rewards, self.terminated, self.episode_starts), leading)
        chex.assert_type(self.actions, jnp.int32)
        chex.assert_type(self.rewards, jnp.float32)
        chex.assert_type((self.terminated, self.episode_starts), jnp.bool_)


def collect_rollout(
    envs: gym.vector.VectorEnv,
    observation: NDArray[np.uint8],
    episode_start: NDArray[np.bool_],
    rng: np.random.Generator,
    num_steps: int,
) -> tuple[Batch, NDArray[np.uint8], NDArray[np.bool_]]:
    """Collect random transitions from environments with autoreset disabled."""
    chex.assert_type(num_steps, int)
    chex.assert_scalar_positive(num_steps)
    chex.assert_shape(observation, (envs.num_envs, *envs.single_observation_space.shape))
    chex.assert_type(observation, np.uint8)
    chex.assert_shape(episode_start, (envs.num_envs,))
    chex.assert_type(episode_start, np.bool_)
    if envs.autoreset_mode != gym.vector.AutoresetMode.DISABLED:
        raise ValueError("collect_rollout requires disabled autoreset to preserve terminal observations")
    if not isinstance(envs.single_action_space, gym.spaces.Discrete) or envs.single_action_space.start != 0:
        raise ValueError("Expected a zero-based discrete action space")
    shape = (num_steps, *observation.shape)
    observations, next_observations = np.empty(shape, np.uint8), np.empty(shape, np.uint8)
    actions = np.empty((num_steps, envs.num_envs), np.int32)
    rewards = np.empty_like(actions, dtype=np.float32)
    terminated = np.empty_like(actions, dtype=np.bool_)
    episode_starts = np.empty_like(terminated)
    for t in range(num_steps):
        observations[t], episode_starts[t] = observation, episode_start
        actions[t] = rng.integers(envs.single_action_space.n, size=envs.num_envs, dtype=np.int32)
        observation, reward, terminal, truncated, _ = envs.step(actions[t])
        # Copy the target before reset can replace a vector environment buffer.
        next_observations[t], rewards[t], terminated[t] = observation, reward, terminal
        episode_start = terminal | truncated
        if episode_start.any():
            observation, _ = envs.reset(options={"reset_mask": episode_start.copy()})
    batch = Batch(*map(jnp.asarray, (observations, actions, next_observations, rewards, terminated, episode_starts)))
    batch.validate()
    return batch, observation, episode_start


@partial(jax.jit, static_argnames=("model",))
def update(
    state: TrainState, model: MambaWorldModel, batch: Batch, carry: Mamba3Carry
) -> tuple[TrainState, Mamba3Carry, dict[str, jax.Array]]:
    """One supervised update; terminal labels exclude time-limit truncations."""
    batch.validate()
    chex.assert_trees_all_equal_shapes_and_dtypes(carry, model.initial_carry(batch.actions.shape[1]))
    carry = jax.tree.map(jax.lax.stop_gradient, carry)
    observations = batch.observations.astype(jnp.float32) / 255.0
    targets = batch.next_observations.astype(jnp.float32) / 255.0

    def loss_fn(params: optax.Params) -> tuple[jax.Array, tuple[Mamba3Carry, dict[str, jax.Array]]]:
        final_carry, _, prediction = state.apply_fn(
            {"params": params}, observations, batch.actions, carry, batch.episode_starts, method=model.observe
        )
        observation_loss = jnp.mean(jnp.square(prediction.observation - targets))
        reward_loss = jnp.mean(jnp.square(prediction.reward - batch.rewards))
        termination_loss = jnp.mean(
            optax.sigmoid_binary_cross_entropy(prediction.termination_logits, batch.terminated.astype(jnp.float32))
        )
        loss = observation_loss + reward_loss + termination_loss
        metrics = {
            "loss": loss,
            "observation_loss": observation_loss,
            "reward_loss": reward_loss,
            "termination_loss": termination_loss,
        }
        return loss, (final_carry, metrics)

    (_, (carry, metrics)), gradients = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    metrics["grad_norm"] = optax.tree.norm(gradients)
    # As in recurrent policy training, this carry was computed before the update.
    return state.apply_gradients(grads=gradients), jax.tree.map(jax.lax.stop_gradient, carry), metrics


def save_checkpoint(state: TrainState, run_dir: Path) -> None:
    """Atomically replace the latest parameters, optimizer state, and update count."""
    temporary = run_dir / "checkpoint.msgpack.tmp"
    temporary.write_bytes(serialization.to_bytes(state))
    temporary.replace(run_dir / "checkpoint.msgpack")


def train(config: Config) -> Path:
    """Train on freshly collected random rollouts and return the artifact directory."""
    env_id = "ALE/SpaceInvaders-v5"
    envs = gym.vector.SyncVectorEnv(
        [
            partial(make_env, env_id, atari_preprocessing=True, observation_size=config.observation_size)
            for _ in range(config.num_envs)
        ],
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )
    writer = None
    try:
        observation, _ = envs.reset(seed=config.seed)
        episode_start = np.ones(config.num_envs, dtype=np.bool_)
        rng = np.random.default_rng(config.seed)
        model = MambaWorldModel(
            observation_shape=envs.single_observation_space.shape,
            num_actions=int(envs.single_action_space.n),
            d_model=config.d_model,
            d_state=config.d_state,
            headdim=config.headdim,
            dtype=jnp.bfloat16 if config.bf16 else jnp.float32,
        )
        params = model.init(
            jax.random.key(config.seed),
            jnp.asarray(observation[:1], dtype=jnp.float32) / 255.0,
            jnp.zeros(1, dtype=jnp.int32),
        )["params"]
        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate)),
        )
        carry = model.initial_carry(config.num_envs)
        run_dir = Path(config.log_dir) / f"SpaceInvaders_seed{config.seed}_{datetime.now(UTC):%Y%m%d-%H%M%S-%f}"
        run_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            **asdict(config),
            "env_id": env_id,
            "observation_shape": model.observation_shape,
            "num_actions": model.num_actions,
        }
        (run_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        writer = SummaryWriter(logdir=str(run_dir))
        writer.add_text("config", json.dumps(metadata, indent=2), 0)
        print(f"Run: {run_dir}\nJAX devices: {jax.devices()}", flush=True)
        start = monotonic()
        steps, iteration = 0, 0
        while steps < config.total_steps:
            num_steps = min(config.num_steps, (config.total_steps - steps) // config.num_envs)
            batch, observation, episode_start = collect_rollout(envs, observation, episode_start, rng, num_steps)
            state, carry, metrics = update(state, model, batch, carry)
            values = {name: float(value) for name, value in jax.device_get(metrics).items()}
            if not all(np.isfinite(value) for value in values.values()):
                raise FloatingPointError(f"Non-finite training metrics: {values}")
            steps += num_steps * config.num_envs
            iteration += 1
            for name, value in values.items():
                writer.add_scalar(f"train/{name}", value, steps)
            writer.add_scalar("rollout/mean_reward", float(batch.rewards.mean()), steps)
            writer.add_scalar("rollout/termination_rate", float(batch.terminated.mean()), steps)
            writer.add_scalar("charts/steps_per_second", steps / (monotonic() - start), steps)
            if iteration == 1 or iteration % config.log_every == 0 or steps == config.total_steps:
                print(
                    f"steps={steps}/{config.total_steps} loss={values['loss']:.4f} "
                    f"observation={values['observation_loss']:.4f} reward={values['reward_loss']:.4f} "
                    f"termination={values['termination_loss']:.4f}",
                    flush=True,
                )
            if iteration % config.checkpoint_every == 0 or steps == config.total_steps:
                save_checkpoint(state, run_dir)
        return run_dir
    finally:
        envs.close()
        if writer is not None:
            writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    defaults = Config()
    for name in (
        "seed",
        "total_steps",
        "num_envs",
        "num_steps",
        "observation_size",
        "d_model",
        "d_state",
        "headdim",
        "log_every",
        "checkpoint_every",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name))
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=defaults.bf16)
    parser.add_argument("--log-dir", default=defaults.log_dir)
    train(Config(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
