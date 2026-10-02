"""Train a Mamba world model on Space Invaders with a uniform random policy.

Run with ``uv run python -m rl2.train_wm --config configs/train_wm_atari.yaml``.
The convolutional encoder accepts uint8 frames and normalizes them internally;
reconstruction targets use [0, 1] and rewards retain their environment scale.
Each rollout receives one Adam update using
next-frame MSE, reward MSE, and terminal
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
import yaml
from flax import serialization, struct
from flax.training.train_state import TrainState
from google.cloud import storage
from numpy.typing import NDArray
from tensorboardX import SummaryWriter

from rl2.mamba3 import Mamba3StackCarry
from rl2.ppo import make_env
from rl2.wm import MambaWorldModel


@dataclass(frozen=True)
class Config:
    env_id: str
    observation_size: int | None
    frame_stack: bool
    atari_preprocessing: bool
    seed: int
    total_steps: int
    num_envs: int
    vector_env: str
    num_steps: int
    d_model: int
    num_layers: int
    d_intermediate: int | None
    encoder_channels: tuple[int, ...]
    d_state: int
    headdim: int
    learning_rate: float
    max_grad_norm: float
    log_dir: str
    log_every: int
    log_flush_secs: int
    checkpoint_dir: str
    checkpoint_every: int
    video_every_steps: int
    video_num_steps: int
    video_fps: float
    bf16: bool = True
    encoder_max_flattened_size: int | None = 8192

    def __post_init__(self) -> None:
        for value in (
            self.total_steps,
            self.num_envs,
            self.num_steps,
            self.d_model,
            self.num_layers,
            self.d_state,
            self.headdim,
            self.log_every,
            self.log_flush_secs,
            self.checkpoint_every,
            self.video_num_steps,
        ):
            chex.assert_type(value, int)
            chex.assert_scalar_positive(value)
        chex.assert_type(self.video_every_steps, int)
        chex.assert_scalar_non_negative(self.video_every_steps)
        if self.d_intermediate is not None:
            chex.assert_type(self.d_intermediate, int)
            chex.assert_scalar_non_negative(self.d_intermediate)
        chex.assert_scalar_positive(len(self.encoder_channels))
        for channels in self.encoder_channels:
            chex.assert_type(channels, int)
            chex.assert_scalar_positive(channels)
        if self.encoder_max_flattened_size is not None:
            chex.assert_type(self.encoder_max_flattened_size, int)
            chex.assert_scalar_non_negative(self.encoder_max_flattened_size - self.encoder_channels[-1])
        if self.observation_size is not None:
            chex.assert_type(self.observation_size, int)
            chex.assert_scalar_positive(self.observation_size)
        chex.assert_type((self.frame_stack, self.atari_preprocessing, self.bf16), bool)
        if self.vector_env not in ("sync", "async"):
            raise ValueError("vector_env must be 'sync' or 'async'")
        chex.assert_is_divisible(self.total_steps, self.num_envs)
        chex.assert_is_divisible(2 * self.d_model, self.headdim)
        chex.assert_is_divisible(self.d_state, 2)
        if self.d_state < 4:
            raise ValueError("d_state must be at least 4 for Mamba's default rotary fraction")
        chex.assert_type(self.seed, int)
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        for value in (self.learning_rate, self.max_grad_norm, self.video_fps):
            chex.assert_scalar_positive(value)
            if not np.isfinite(value):
                raise ValueError("learning_rate, max_grad_norm, and video_fps must be finite")


def load_config(path: str | Path) -> Config:
    """Load world-model training settings from a YAML mapping."""
    with open(path) as file:
        settings = yaml.safe_load(file)
    if not isinstance(settings, dict):
        raise TypeError("The YAML config must contain a mapping of training settings")
    if "encoder_channels" in settings:
        settings["encoder_channels"] = tuple(settings["encoder_channels"])
    return Config(**settings)


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
    state: TrainState, model: MambaWorldModel, batch: Batch, carry: Mamba3StackCarry
) -> tuple[TrainState, Mamba3StackCarry, dict[str, jax.Array]]:
    """One supervised update; terminal labels exclude time-limit truncations."""
    batch.validate()
    chex.assert_trees_all_equal_shapes_and_dtypes(carry, model.initial_carry(batch.actions.shape[1]))
    carry = jax.tree.map(jax.lax.stop_gradient, carry)
    targets = batch.next_observations.astype(jnp.float32) / 255.0

    def loss_fn(params: optax.Params) -> tuple[jax.Array, tuple[Mamba3StackCarry, dict[str, jax.Array]]]:
        final_carry, _, prediction = state.apply_fn(
            {"params": params}, batch.observations, batch.actions, carry, batch.episode_starts, method=model.observe
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


type ImaginationCarry = tuple[Mamba3StackCarry, jax.Array]


@partial(jax.jit, static_argnames=("model",))
def imagine_frames(
    state: TrainState,
    model: MambaWorldModel,
    observation: jax.Array,
    carry: Mamba3StackCarry,
    episode_start: jax.Array,
    actions: jax.Array,
) -> jax.Array:
    """Return the real seed frame followed by fixed-horizon imagined observations.

    Inputs describe one environment. Only the seed observation is encoded;
    subsequent steps feed back predicted latents. Predicted termination does
    not reset or stop this diagnostic rollout.
    """
    chex.assert_shape(observation, (1, *model.observation_shape))
    chex.assert_type(observation, jnp.uint8)
    chex.assert_trees_all_equal_shapes_and_dtypes(carry, model.initial_carry(1))
    chex.assert_shape(episode_start, (1,))
    chex.assert_type(episode_start, jnp.bool_)
    chex.assert_shape(actions, (None, 1))
    chex.assert_type(actions, jnp.int32)
    chex.assert_scalar_positive(actions.shape[0])
    normalized = observation.astype(jnp.float32) / 255.0
    latent = state.apply_fn({"params": state.params}, observation, method=model.encode)
    starts = jnp.zeros(actions.shape, dtype=jnp.bool_).at[0].set(episode_start)

    def step(memory: ImaginationCarry, inputs: tuple[jax.Array, jax.Array]) -> tuple[ImaginationCarry, jax.Array]:
        history, latent = memory
        action, reset = inputs
        chex.assert_shape(latent, (1, model.d_model))
        chex.assert_type(latent, jnp.float32)
        chex.assert_shape((action, reset), (1,))
        chex.assert_type((action, reset), (jnp.int32, jnp.bool_))
        history, latent, prediction = state.apply_fn(
            {"params": state.params}, latent, action, history, reset, method=model.imagine
        )
        return (history, latent), prediction.observation[0]

    _, frames = jax.lax.scan(step, (carry, latent), (actions, starts))
    return jnp.concatenate((normalized, frames), axis=0)


def log_video(
    state: TrainState,
    model: MambaWorldModel,
    config: Config,
    writer: SummaryWriter,
    observation: NDArray[np.uint8],
    carry: Mamba3StackCarry,
    episode_start: NDArray[np.bool_],
    steps: int,
) -> None:
    """Log a random-action imagination clip without stepping training environments."""
    chex.assert_shape(observation, (1, *model.observation_shape))
    chex.assert_type(observation, np.uint8)
    chex.assert_shape(episode_start, (1,))
    chex.assert_type(episode_start, np.bool_)
    chex.assert_type(steps, int)
    chex.assert_scalar_non_negative(steps)
    # A separate, fixed random stream allows comparisons without changing the
    # training collector's actions or consuming its random state.
    rng = np.random.default_rng(config.seed)
    actions = rng.integers(model.num_actions, size=(config.video_num_steps, 1), dtype=np.int32)
    frames = np.asarray(
        imagine_frames(state, model, jnp.asarray(observation), carry, jnp.asarray(episode_start), jnp.asarray(actions))
    )
    # make_env returns [stack, height, width] or [stack, height, width, RGB].
    chex.assert_rank(frames, {4, 5})
    chex.assert_type(frames, np.float32)
    frames = frames[:, -1]  # Show only the newest frame of each predicted stack.
    if frames.ndim == 3:
        frames = frames[:, None]
    else:
        chex.assert_shape(frames, (config.video_num_steps + 1, None, None, 3))
        frames = frames.transpose(0, 3, 1, 2)
    video = np.rint(np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)[None]
    writer.add_video("imagination/random_policy", video, steps, fps=config.video_fps)
    writer.flush()
    print(f"Recorded {config.video_num_steps} imagined transitions at step {steps}", flush=True)


def write_artifact(path: str, data: bytes) -> None:
    """Upload a complete GCS object or replace a local file atomically."""
    if path.startswith("gs://"):
        blob = storage.Blob.from_uri(path, client=storage.Client())
        blob.upload_from_string(data)
    else:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f"{destination.name}.tmp")
        temporary.write_bytes(data)
        temporary.replace(destination)


def save_checkpoint(state: TrainState, run_dir: str | Path) -> None:
    """Save the latest parameters, optimizer state, and update count in the checkpoint run directory."""
    write_artifact(f"{str(run_dir).rstrip('/')}/checkpoint.msgpack", serialization.to_bytes(state))


def train(config: Config) -> str:
    """Train on freshly collected random rollouts and return the artifact directory."""
    vector_cls = gym.vector.AsyncVectorEnv if config.vector_env == "async" else gym.vector.SyncVectorEnv
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
        observation, _ = envs.reset(seed=config.seed)
        episode_start = np.ones(config.num_envs, dtype=np.bool_)
        rng = np.random.default_rng(config.seed)
        model = MambaWorldModel(
            observation_shape=envs.single_observation_space.shape,
            num_actions=int(envs.single_action_space.n),
            d_model=config.d_model,
            num_layers=config.num_layers,
            d_intermediate=config.d_intermediate,
            encoder_channels=config.encoder_channels,
            encoder_max_flattened_size=config.encoder_max_flattened_size,
            d_state=config.d_state,
            headdim=config.headdim,
            dtype=jnp.bfloat16 if config.bf16 else jnp.float32,
        )
        params = model.init(
            jax.random.key(config.seed),
            jnp.asarray(observation[:1]),
            jnp.zeros(1, dtype=jnp.int32),
        )["params"]
        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate)),
        )
        carry = model.initial_carry(config.num_envs)
        run_name = f"{config.env_id.replace('/', '_')}_seed{config.seed}_{datetime.now(UTC):%Y%m%d-%H%M%S-%f}"
        run_dir = f"{config.log_dir.rstrip('/')}/{run_name}"
        checkpoint_run_dir = f"{config.checkpoint_dir.rstrip('/')}/{run_name}"
        metadata = {
            **asdict(config),
            "observation_shape": model.observation_shape,
            "num_actions": model.num_actions,
        }
        writer = SummaryWriter(logdir=run_dir, flush_secs=config.log_flush_secs)
        metadata_bytes = (json.dumps(metadata, indent=2) + "\n").encode()
        write_artifact(f"{run_dir}/config.json", metadata_bytes)
        write_artifact(f"{checkpoint_run_dir}/config.json", metadata_bytes)
        writer.add_text("config", f"```yaml\n{yaml.safe_dump(asdict(config))}```", 0)
        print(f"TensorBoard run: {run_dir}", flush=True)
        print(f"Checkpoint: {checkpoint_run_dir}/checkpoint.msgpack", flush=True)
        devices = str(jax.devices())
        print(f"JAX devices: {devices}", flush=True)
        writer.add_text("devices", devices, 0)
        start = monotonic()
        steps, iteration = 0, 0
        next_video_step = config.video_every_steps
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
                writer.flush()
            if iteration % config.checkpoint_every == 0 or steps == config.total_steps:
                save_checkpoint(state, checkpoint_run_dir)
            if config.video_every_steps and steps >= next_video_step:
                video_carry = jax.tree.map(lambda leaf: leaf[:1], carry)
                log_video(state, model, config, writer, observation[:1], video_carry, episode_start[:1], steps)
                next_video_step = (steps // config.video_every_steps + 1) * config.video_every_steps
        return run_dir
    finally:
        envs.close()
        if writer is not None:
            writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_wm_atari.yaml", help="Path to a YAML config")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
