"""Train a Mamba or LSTM world model with a uniform random policy.

Run with ``uv run python -m rl2.train_wm --config configs/train_wm_atari.yaml``.
The convolutional encoder accepts uint8 frames and normalizes them internally;
reconstruction targets use [0, 1] and rewards retain their environment scale.
Each rollout receives one Adam update using
posterior reconstruction, balanced categorical KL, reward MSE, and terminal
binary cross entropy. Recurrent history crosses chunks with truncated BPTT and
resets at episode boundaries. No policy is learned.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import monotonic
from typing import Literal

import chex
import cv2
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

from rl2.jax_cache import configure_compilation_cache
from rl2.observation_encoder import ConvStage, ConvStages, validate_stages
from rl2.ppo import make_env
from rl2.wm import (
    LSTMWorldModel,
    MambaWorldModel,
    WorldModelState,
    categorical_entropy,
    categorical_kl,
    check_keys,
    latent_kl_losses,
)

ObservationLoss = Literal["l1", "l2", "charbonnier"]


@dataclass(frozen=True)
class Config:
    env_id: str
    observation_size: int | None
    frame_stack: bool
    atari_preprocessing: bool
    grayscale_obs: bool
    seed: int
    total_steps: int
    num_envs: int
    vector_env: str
    num_steps: int
    d_model: int
    num_layers: int
    d_intermediate: int | None
    encoder_stages: ConvStages
    d_state: int
    headdim: int
    stochastic_size: int
    stochastic_classes: int
    unimix: float
    dynamics_kl_scale: float
    representation_kl_scale: float
    free_nats: float
    learning_rate: float
    max_grad_norm: float
    log_dir: str
    log_every: int
    log_flush_secs: int
    checkpoint_dir: str
    checkpoint_every: int
    video_every_steps: int
    video_num_steps: int
    video_prefill_frames: int
    video_fps: float
    bf16: bool = True
    observation_loss: ObservationLoss = "l2"
    charbonnier_epsilon: float = 1e-3
    model: Literal["mamba", "lstm"] = "mamba"

    def __post_init__(self) -> None:
        for value in (
            self.total_steps,
            self.num_envs,
            self.num_steps,
            self.d_model,
            self.num_layers,
            self.stochastic_size,
            self.stochastic_classes,
            self.log_every,
            self.log_flush_secs,
            self.checkpoint_every,
            self.video_num_steps,
            self.video_prefill_frames,
        ):
            chex.assert_type(value, int)
            chex.assert_scalar_positive(value)
        chex.assert_type(self.video_every_steps, int)
        chex.assert_scalar_non_negative(self.video_every_steps)
        validate_observation_loss(self.observation_loss, self.charbonnier_epsilon)
        if self.d_intermediate is not None:
            chex.assert_type(self.d_intermediate, int)
            chex.assert_scalar_non_negative(self.d_intermediate)
        if self.stochastic_classes < 2:
            raise ValueError("stochastic_classes must be at least two")
        if not np.isfinite(self.unimix) or not 0 <= self.unimix < 1:
            raise ValueError("unimix must be finite and in [0, 1)")
        for value in (self.dynamics_kl_scale, self.representation_kl_scale, self.free_nats):
            chex.assert_scalar_non_negative(value)
            if not np.isfinite(value):
                raise ValueError("KL scales and free_nats must be finite")
        validate_stages(self.encoder_stages)
        if self.observation_size is not None:
            chex.assert_type(self.observation_size, int)
            chex.assert_scalar_positive(self.observation_size)
        chex.assert_type((self.frame_stack, self.atari_preprocessing, self.grayscale_obs, self.bf16), bool)
        if self.vector_env not in ("sync", "async"):
            raise ValueError("vector_env must be 'sync' or 'async'")
        chex.assert_is_divisible(self.total_steps, self.num_envs)
        if self.model not in ("mamba", "lstm"):
            raise ValueError("model must be 'mamba' or 'lstm'")
        if self.model == "mamba":
            for value in (self.d_state, self.headdim):
                chex.assert_type(value, int)
                chex.assert_scalar_positive(value)
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
    if "encoder_stages" in settings:
        settings["encoder_stages"] = tuple(ConvStage(**stage) for stage in settings["encoder_stages"])
    return Config(**settings)


def learning_rate_schedule(config: Config) -> optax.Schedule:
    """Cosine learning-rate decay to zero over the training run.

    The schedule takes the number of completed optimizer updates, starting at
    zero. Include a short final rollout in the update count. Even a one-update
    run uses the initial learning rate; zero is reached after its final update.
    """
    rollout_size = config.num_envs * config.num_steps
    num_updates = (config.total_steps + rollout_size - 1) // rollout_size
    return optax.cosine_decay_schedule(config.learning_rate, decay_steps=num_updates)


@struct.dataclass
class Batch:
    """Time-major transitions; next_observations always precede any reset.

    Attributes:
        observations: (T, B, *image) uint8 current frames.
        actions: (T, B) int32 action IDs.
        next_observations: (T, B, *image) uint8 targets, including terminal frames.
        rewards: (T, B) float32 incoming rewards.
        terminated: (T, B) bool terminal labels, excluding truncation.
        episode_starts: (T, B) bool masks resetting memory before each transition.
    """

    observations: jax.Array
    actions: jax.Array
    next_observations: jax.Array
    rewards: jax.Array
    terminated: jax.Array
    episode_starts: jax.Array

    def validate(self) -> None:
        chex.assert_type((self.observations, self.next_observations), jnp.uint8)
        leading = self.observations.shape[:2]
        for size in leading:
            chex.assert_scalar_positive(size)
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
    chex.assert_type(observation, np.uint8)
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


def validate_observation_loss(loss: ObservationLoss, epsilon: float) -> None:
    """Validate the reconstruction penalty and its smoothing scale."""
    if loss not in ("l1", "l2", "charbonnier"):
        raise ValueError("observation_loss must be 'l1', 'l2', or 'charbonnier'")
    chex.assert_scalar_positive(epsilon)
    if not np.isfinite(epsilon):
        raise ValueError("charbonnier_epsilon must be finite")
    limits = np.finfo(np.float32)
    if not float(limits.tiny) <= epsilon <= float(limits.max):
        raise ValueError("charbonnier_epsilon must be representable as a positive normal float32")


def observation_reconstruction_loss(
    prediction: jax.Array,
    target: jax.Array,
    loss: ObservationLoss,
    epsilon: float,
) -> jax.Array:
    """Sum pixel/channel penalties per frame, then average time and batch.

    Args:
        prediction: (T, B, *observation_shape) floating predictions.
        target: (T, B, *observation_shape) floating targets normalized to [0, 1].
        loss: 'l1' uses abs(error); 'l2' uses 0.5 * error**2, preserving the
            original Gaussian objective; 'charbonnier' uses
            sqrt(error**2 + epsilon**2) - epsilon (zero at a perfect match).
        epsilon: Positive Charbonnier smoothing scale in normalized pixel units.

    Returns:
        () float32 reconstruction loss, without clipping predictions.
    """
    chex.assert_type((prediction, target), jnp.floating)
    validate_observation_loss(loss, epsilon)
    error = prediction.astype(jnp.float32) - target.astype(jnp.float32)
    if loss == "l1":
        penalty = jnp.abs(error)
    elif loss == "l2":
        penalty = 0.5 * jnp.square(error)
    else:
        # Scale before squaring and rationalize sqrt(error**2 + epsilon**2)
        # minus epsilon to avoid overflow, underflow, and cancellation. The
        # scale cancels algebraically, so it need not participate in autodiff.
        scale = jax.lax.stop_gradient(jnp.maximum(jnp.abs(error), epsilon))
        scaled_error, scaled_epsilon = error / scale, epsilon / scale
        radius = jnp.sqrt(jnp.square(scaled_error) + jnp.square(scaled_epsilon))
        penalty = error * (scaled_error / (radius + scaled_epsilon))
    return jnp.mean(jnp.sum(penalty, axis=tuple(range(2, penalty.ndim))))


@partial(jax.jit, static_argnames=("model", "observation_loss", "charbonnier_epsilon"))
def update(
    state: TrainState,
    model: MambaWorldModel,
    batch: Batch,
    carry: WorldModelState,
    key: jax.Array,
    dynamics_kl_scale: float,
    representation_kl_scale: float,
    free_nats: float,
    observation_loss: ObservationLoss = "l2",
    charbonnier_epsilon: float = 1e-3,
) -> tuple[TrainState, WorldModelState, dict[str, jax.Array]]:
    """Train posterior reconstruction and balanced KL using a fresh sampling key.

    Carry is detached only at chunk boundaries. The selected reconstruction
    term sums pixels per frame; the observation_loss metric retains pixel MSE for
    readable logs. KL sums categorical variables and applies free_nats per
    transition before batch averaging, independently for the two gradient paths.
    """
    batch.validate()
    check_keys(key)
    carry = jax.tree.map(jax.lax.stop_gradient, carry)
    targets = batch.next_observations.astype(jnp.float32) / 255.0
    keys = jax.random.split(key, batch.actions.shape[0])

    def loss_fn(params: optax.Params) -> tuple[jax.Array, tuple[WorldModelState, dict[str, jax.Array]]]:
        final_carry, output = state.apply_fn(
            {"params": params},
            batch.observations,
            batch.actions,
            batch.next_observations,
            keys,
            carry,
            batch.episode_starts,
            method=model.observe,
        )
        prediction = output.prediction
        observation_mse = jnp.mean(jnp.square(prediction.observation.astype(jnp.float32) - targets))
        reconstruction_loss = observation_reconstruction_loss(
            prediction.observation, targets, observation_loss, charbonnier_epsilon
        )
        reward_loss = jnp.mean(jnp.square(prediction.reward - batch.rewards))
        termination_loss = jnp.mean(
            optax.sigmoid_binary_cross_entropy(prediction.termination_logits, batch.terminated.astype(jnp.float32))
        )
        prior, posterior = output.prior_logits, output.posterior_logits
        dynamics_kl = categorical_kl(posterior, prior)
        dynamics_loss, representation_loss = latent_kl_losses(posterior, prior, free_nats)
        kl_loss = dynamics_kl_scale * dynamics_loss + representation_kl_scale * representation_loss
        loss = reconstruction_loss + reward_loss + termination_loss + kl_loss
        metrics = {
            "loss": loss,
            "observation_loss": observation_mse,
            "reconstruction_loss": reconstruction_loss,
            "reward_loss": reward_loss,
            "termination_loss": termination_loss,
            "kl_loss": kl_loss,
            "dynamics_kl_loss": dynamics_loss,
            "representation_kl_loss": representation_loss,
            "kl": dynamics_kl.mean(),
            "prior_entropy": categorical_entropy(prior).mean(),
            "posterior_entropy": categorical_entropy(posterior).mean(),
        }
        return loss, (final_carry, metrics)

    (_, (carry, metrics)), gradients = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    metrics["grad_norm"] = optax.tree.norm(gradients)
    # As in recurrent policy training, this carry was computed before the update.
    return state.apply_gradients(grads=gradients), jax.tree.map(jax.lax.stop_gradient, carry), metrics


def update_video_history(history: Batch | None, batch: Batch, num_frames: int) -> Batch:
    """Retain recent transitions for a num_frames diagnostic video window.

    History spans rollout chunks. Episode masks are retained so selection can
    reject windows crossing resets. At least one transition is kept even for
    a single-frame prefill, which uses its next observation.
    """
    chex.assert_type(num_frames, int)
    chex.assert_scalar_positive(num_frames)
    batch.validate()
    keep = max(1, num_frames - 1)
    if history is None:
        return jax.tree.map(lambda x: x[-keep:], batch)
    history.validate()
    return jax.tree.map(lambda old, new: jnp.concatenate((old, new), axis=0)[-keep:], history, batch)


def select_video_window(
    history: Batch,
    episode_start: NDArray[np.bool_],
    num_frames: int,
) -> tuple[NDArray[np.uint8], NDArray[np.int32]] | None:
    """Select a recent, same-episode window ending in a live environment state.

    Args:
        history: Consecutive recent transitions, possibly spanning chunks.
        episode_start: (B,) bool mask, true if the next collector input is
            a reset observation. This excludes both terminal and truncated endpoints.
        num_frames: Total real frames (prefill plus prediction targets), connected
            by num_frames-1 recorded actions.

    Returns:
        observations: (num_frames, 1, *image) uint8 frames.
        actions: (num_frames-1, 1) int32 actions for the first eligible environment.
        Returns None if no full window is available. Terminal targets are never
        used as imagination seeds.
    """
    history.validate()
    chex.assert_type(num_frames, int)
    chex.assert_scalar_positive(num_frames)
    chex.assert_type(episode_start, np.bool_)
    length = num_frames - 1
    if history.actions.shape[0] < length:
        return None
    start = history.actions.shape[0] - length
    eligible = ~episode_start
    if length:
        eligible &= ~np.asarray(history.episode_starts[start + 1 :]).any(axis=0)
        eligible &= ~np.asarray(history.terminated[start:]).any(axis=0)
    else:
        eligible &= ~np.asarray(history.terminated[-1])
    candidates = np.flatnonzero(eligible)
    if not candidates.size:
        return None
    env = int(candidates[0])
    if length:
        observations = jnp.concatenate(
            (history.observations[start : start + 1, env : env + 1], history.next_observations[start:, env : env + 1]),
            axis=0,
        )
    else:
        observations = history.next_observations[-1:, env : env + 1]
    actions = history.actions[start:, env : env + 1]
    return np.asarray(observations), np.asarray(actions)


@partial(jax.jit, static_argnames=("model", "num_prefill_frames"))
def comparison_frames(
    state: TrainState,
    model: MambaWorldModel,
    observations: jax.Array,
    actions: jax.Array,
    num_prefill_frames: int,
    key: jax.Array,
) -> jax.Array:
    """Compare real targets, posterior reconstructions, and prior imagination.

    Args:
        state: Current training parameters and the model's apply function.
        model: World model used to condition, observe, imagine, and decode.
        observations: (N+T, 1, *image) uint8 consecutive real frames from one episode.
        actions: (N+T-1, 1) int32 recorded actions connecting those frames.
        num_prefill_frames: N real context frames; N and T must both be positive.
        key: () typed JAX sampling key (or a legacy key), independent of training randomness.

    Returns:
        (3, N+T, *image) float32 frames, ordered real, posterior, prior. All panels show
        the real context for the first N frames. Both model branches start from
        exactly the same posterior state, rebuilt with current weights. Thereafter
        posterior reconstruction sees each real target; prior imagination sees
        only recorded actions and its own sampled latent history. Predicted
        termination does not stop the comparison.
    """
    chex.assert_type(observations, jnp.uint8)
    chex.assert_type(num_prefill_frames, int)
    chex.assert_scalar_positive(num_prefill_frames)
    chex.assert_scalar_positive(observations.shape[0] - num_prefill_frames)
    chex.assert_type(actions, jnp.int32)
    check_keys(key)
    prefill_observations = observations[:num_prefill_frames]
    prefill_actions = actions[: num_prefill_frames - 1]
    future_actions = actions[num_prefill_frames - 1 :]
    prefill_key, imagination_key, posterior_key = jax.random.split(key, 3)
    if num_prefill_frames > 1:
        carry, _ = state.apply_fn(
            {"params": state.params},
            prefill_observations[:-1],
            prefill_actions,
            prefill_observations[1:],
            jax.random.split(prefill_key, num_prefill_frames - 1),
            method=model.observe,
        )
    else:
        carry = state.apply_fn(
            {"params": state.params},
            prefill_observations[0],
            model.initial_carry(1),
            jnp.ones(1, jnp.bool_),
            prefill_key,
            method=model.condition,
        )

    _, posterior = state.apply_fn(
        {"params": state.params},
        observations[num_prefill_frames - 1 : -1],
        future_actions,
        observations[num_prefill_frames:],
        jax.random.split(posterior_key, future_actions.shape[0]),
        carry,
        method=model.observe,
    )

    def step(memory: WorldModelState, inputs: tuple[jax.Array, jax.Array]) -> tuple[WorldModelState, jax.Array]:
        action, sample_key = inputs
        chex.assert_type(action, jnp.int32)
        check_keys(sample_key)
        memory, prediction = state.apply_fn(
            {"params": state.params},
            memory,
            action,
            sample_key,
            method=model.imagine,
        )
        return memory, prediction.observation[0]

    _, frames = jax.lax.scan(step, carry, (future_actions, jax.random.split(imagination_key, future_actions.shape[0])))
    real = observations[:, 0].astype(jnp.float32) / 255.0
    reconstruction = jnp.concatenate((real[:num_prefill_frames], posterior.prediction.observation[:, 0]), axis=0)
    imagination = jnp.concatenate((real[:num_prefill_frames], frames), axis=0)
    return jnp.stack((real, reconstruction, imagination))


def log_video(
    state: TrainState,
    model: MambaWorldModel,
    config: Config,
    writer: SummaryWriter,
    observations: NDArray[np.uint8],
    actions: NDArray[np.int32],
    steps: int,
) -> None:
    """Log labeled real/posterior/prior panels using a recorded same-episode window.

    Observations: (prefill+future, 1, *image) uint8 frames.
    Actions: (prefill+future-1, 1) int32 actions.
    Frames are shown side by side in that order, with a
    header marking shared real context before prediction begins. No environments
    are stepped and no training random state is consumed. Returns None.
    """
    num_frames = config.video_prefill_frames + config.video_num_steps
    chex.assert_type(observations, np.uint8)
    chex.assert_type(actions, np.int32)
    chex.assert_type(steps, int)
    chex.assert_scalar_non_negative(steps)
    frames = np.asarray(
        comparison_frames(
            state,
            model,
            jnp.asarray(observations),
            jnp.asarray(actions),
            config.video_prefill_frames,
            jax.random.key(config.seed),
        )
    )
    chex.assert_type(frames, np.float32)
    targets = frames[0, config.video_prefill_frames :]
    for panel, name in enumerate(("posterior", "prior"), start=1):
        error = frames[panel, config.video_prefill_frames :] - targets
        writer.add_scalar(f"diagnostics/{name}_mse", float(np.mean(np.square(error))), steps)
    frames = frames[:, :, -1]  # Show the newest frame from each observation stack.
    if frames.ndim == 4:
        frames = np.repeat(frames[..., None], 3, axis=-1)
    pixels = np.rint(np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)
    _, _, height, width, _ = pixels.shape
    header_height = 18
    panels = np.zeros((3, num_frames, height + header_height, width, 3), np.uint8)
    panels[:, :, header_height:] = pixels
    for panel, label in enumerate(("Real", "Posterior", "Prior")):
        for t in range(num_frames):
            cv2.putText(
                panels[panel, t],
                "Context" if t < config.video_prefill_frames else label,
                (2, 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    video = np.concatenate(tuple(panels), axis=2).transpose(0, 3, 1, 2)[None]
    writer.add_video("imagination/real_posterior_prior", video, steps, fps=config.video_fps)
    writer.flush()
    print(
        f"Recorded comparison: {config.video_prefill_frames} context + {config.video_num_steps} predicted frames "
        f"at step {steps} (real | posterior | prior)",
        flush=True,
    )


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
    """Train on fresh random-policy rollouts and return the log directory."""
    configure_compilation_cache()
    vector_cls = gym.vector.AsyncVectorEnv if config.vector_env == "async" else gym.vector.SyncVectorEnv
    vector_options = {"context": "spawn"} if config.vector_env == "async" else {}
    envs = vector_cls(
        [
            partial(
                make_env,
                config.env_id,
                frame_stack=config.frame_stack,
                atari_preprocessing=config.atari_preprocessing,
                grayscale_obs=config.grayscale_obs,
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
        model_class = LSTMWorldModel if config.model == "lstm" else MambaWorldModel
        model = model_class(
            observation_shape=envs.single_observation_space.shape,
            num_actions=int(envs.single_action_space.n),
            d_model=config.d_model,
            num_layers=config.num_layers,
            d_intermediate=config.d_intermediate,
            encoder_stages=config.encoder_stages,
            d_state=config.d_state,
            headdim=config.headdim,
            stochastic_size=config.stochastic_size,
            stochastic_classes=config.stochastic_classes,
            unimix=config.unimix,
            dtype=jnp.bfloat16 if config.bf16 else jnp.float32,
        )
        parameter_key, sample_key, training_key = jax.random.split(jax.random.key(config.seed), 3)
        params = model.init(
            parameter_key,
            jnp.asarray(observation[:1]),
            jnp.zeros(1, dtype=jnp.int32),
            jnp.asarray(observation[:1]),
            sample_key,
        )["params"]
        lr_schedule = learning_rate_schedule(config)
        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(lr_schedule)),
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
        parameter_count = sum(parameter.size for parameter in jax.tree.leaves(state.params))
        writer.add_scalar("model/params_millions", parameter_count / 1_000_000, 0)
        print(f"Model parameters: {parameter_count:,}", flush=True)
        writer.add_text("model/parameter_count", str(parameter_count), 0)
        start = monotonic()
        steps, iteration = 0, 0
        next_video_step = config.video_every_steps
        video_history: Batch | None = None
        while steps < config.total_steps:
            num_steps = min(config.num_steps, (config.total_steps - steps) // config.num_envs)
            batch, observation, episode_start = collect_rollout(envs, observation, episode_start, rng, num_steps)
            if config.video_every_steps:
                video_history = update_video_history(
                    video_history, batch, config.video_prefill_frames + config.video_num_steps
                )
            learning_rate = float(lr_schedule(state.step))
            state, carry, metrics = update(
                state,
                model,
                batch,
                carry,
                jax.random.fold_in(training_key, state.step),
                config.dynamics_kl_scale,
                config.representation_kl_scale,
                config.free_nats,
                observation_loss=config.observation_loss,
                charbonnier_epsilon=config.charbonnier_epsilon,
            )
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
            writer.add_scalar("charts/learning_rate", learning_rate, steps)
            if iteration == 1 or iteration % config.log_every == 0 or steps == config.total_steps:
                print(
                    f"steps={steps}/{config.total_steps} loss={values['loss']:.4f} "
                    f"observation={values['observation_loss']:.4f} reward={values['reward_loss']:.4f} "
                    f"termination={values['termination_loss']:.4f} kl={values['kl']:.4f} lr={learning_rate:.3g}",
                    flush=True,
                )
                writer.flush()
            if iteration % config.checkpoint_every == 0 or steps == config.total_steps:
                save_checkpoint(state, checkpoint_run_dir)
            if config.video_every_steps and steps >= next_video_step:
                assert video_history is not None
                num_video_frames = config.video_prefill_frames + config.video_num_steps
                window = select_video_window(video_history, episode_start, num_video_frames)
                if window is not None:
                    log_video(state, model, config, writer, *window, steps)
                else:
                    print(
                        f"Skipped video at step {steps}: no {num_video_frames}-frame live episode window",
                        flush=True,
                    )
                next_video_step = (steps // config.video_every_steps + 1) * config.video_every_steps
        return run_dir
    finally:
        try:
            envs.close()
        finally:
            if writer is not None:
                writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_wm_atari.yaml", help="Path to a YAML config")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
