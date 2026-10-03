"""Minimal outcome-supervised GRPO for Karel program synthesis.

Each rollout samples group_size programs per initial/target pair. Advantages
are normalized within each pair's group; the clipped objective averages tokens
within programs, then programs within the minibatch. Terminal m) is included, PAD
is excluded, and syntax/token-limit failures receive a syntax edit-distance score.

This sketch starts from random weights with no grammar mask or supervised
warmup. Terminal rewards measure progress toward the target; exact success is
logged separately. Equal-reward groups have zero advantages; syntax errors can
now earn different rewards according to their minimum repair costs.
An optional KL penalty uses a frozen copy of the initial model as reference.
See https://arxiv.org/abs/2402.03300 for the GRPO objective.
"""

import argparse
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import monotonic
from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax.training.train_state import TrainState
from numpy.typing import NDArray
from tensorboardX import SummaryWriter

from rl2.karel import TOKENS, KarelConfig, KarelProgramEnv
from rl2.karel_model import KarelProgramModel
from rl2.mamba3 import Mamba3StackCarry

type Array = jax.Array | NDArray[Any]
type Metrics = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


@dataclass(frozen=True)
class Config:
    seed: int = 1
    total_updates: int = 1000  # Number of rollout batches, independent of program length.
    num_tasks: int = 8
    group_size: int = 8
    num_minibatches: int = 4
    update_epochs: int = 2
    d_model: int = 256
    num_layers: int = 4
    d_state: int = 64
    headdim: int = 64
    conv_channels: tuple[int, ...] = (32, 64, 64)
    learning_rate: float = 0.00025
    anneal_lr: bool = True
    clip_coef: float = 0.2
    target_kl: float | None = 0.02
    entropy_coef: float = 0.0
    kl_coef: float = 0.0  # Zero disables the frozen-reference KL penalty.
    max_grad_norm: float = 0.5
    log_dir: str = "runs"
    env: KarelConfig = field(default_factory=KarelConfig)

    def __post_init__(self) -> None:
        for value in (self.total_updates, self.num_tasks, self.group_size, self.num_minibatches, self.update_epochs):
            if type(value) is not int:
                raise TypeError("Rollout and update counts must be integers")
            chex.assert_scalar_positive(value)
        if self.group_size < 2:
            raise ValueError("GRPO requires group_size >= 2")
        if self.num_tasks * self.group_size % self.num_minibatches:
            raise ValueError("num_minibatches must divide num_tasks * group_size")
        chex.assert_type(self.seed, int)
        chex.assert_scalar_non_negative(self.seed)
        chex.assert_scalar_in(self.clip_coef, 0, 1, included=False)
        for value in (self.learning_rate, self.max_grad_norm):
            chex.assert_scalar_positive(value)
            if not np.isfinite(value):
                raise ValueError("Optimizer settings must be finite")
        for value in (self.entropy_coef, self.kl_coef):
            chex.assert_scalar_non_negative(value)
            if not np.isfinite(value):
                raise ValueError("Loss coefficients must be finite")
        if self.target_kl is not None:
            chex.assert_scalar_positive(self.target_kl)
            if not np.isfinite(self.target_kl):
                raise ValueError("target_kl must be finite or null")


class GRPOBatch(NamedTuple):
    initial: Array  # [B,H,W,6], grouped by task during collection.
    target: Array
    actions: Array  # [T,B], including terminal m); padded with <pad> afterwards.
    old_log_probs: Array
    mask: Array  # [T,B], includes terminal m) or every token on truncation, excludes PAD.
    advantages: Array  # [B], computed before shuffling/minibatching.


def load_config(path: str | Path) -> Config:
    with open(path) as file:
        settings = yaml.safe_load(file) or {}
    if "env" in settings:
        settings["env"] = KarelConfig(**settings["env"])
    if "conv_channels" in settings:
        settings["conv_channels"] = tuple(settings["conv_channels"])
    return Config(**settings)


def learning_rate_schedule(config: Config) -> optax.Schedule:
    return optax.linear_schedule(
        config.learning_rate, 0.0 if config.anneal_lr else config.learning_rate, config.total_updates
    )


@jax.jit
def group_advantages(rewards: Array) -> jax.Array:
    """Normalize within each task, preserving exactly zero advantages for ties."""
    chex.assert_rank(rewards, 2)
    chex.assert_type(rewards, jnp.float32)
    chex.assert_scalar_positive(rewards.shape[1] - 1)
    # Subtract a member first: reducing identical fractional rewards directly can
    # round their mean away from that value and create a spurious advantage.
    shifted = rewards - rewards[:, :1]
    centered = shifted - shifted.mean(axis=1, keepdims=True)
    std = jnp.sqrt(jnp.square(centered).mean(axis=1, keepdims=True))
    return centered / (std + 1e-8)


def generation_logits(logits: jax.Array) -> jax.Array:
    """Exclude the batching-only PAD symbol from every policy distribution."""
    chex.assert_shape(logits, (*logits.shape[:-1], len(TOKENS)))
    chex.assert_type(logits, jnp.float32)
    return logits.at[..., KarelProgramEnv.pad_token_id].set(-jnp.inf)


def action_log_prob(logits: jax.Array, actions: Array) -> jax.Array:
    chex.assert_shape(logits, (*actions.shape, len(TOKENS)))
    chex.assert_type(logits, jnp.float32)
    chex.assert_type(actions, jnp.int32)
    log_probs = jax.nn.log_softmax(generation_logits(logits))
    selected = jnp.take_along_axis(log_probs, actions[..., None], axis=-1)[..., 0]
    # Padded labels have no probability/loss contribution; avoid -inf arithmetic.
    return jnp.where(actions == KarelProgramEnv.pad_token_id, 0.0, selected)


@jax.jit
def prefill(state: TrainState, initial: Array, target: Array) -> tuple[Mamba3StackCarry, jax.Array]:
    return state.apply_fn({"params": state.params}, initial, target, method=KarelProgramModel.prefill)


@jax.jit
def decode_step(state: TrainState, actions: Array, carry: Mamba3StackCarry) -> tuple[Mamba3StackCarry, jax.Array]:
    return state.apply_fn({"params": state.params}, actions, carry, method=KarelProgramModel.step)


@jax.jit
def act(logits: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
    chex.assert_shape(logits, (None, len(TOKENS)))
    chex.assert_type(logits, jnp.float32)
    actions = jax.random.categorical(key, generation_logits(logits)).astype(jnp.int32)
    return actions, action_log_prob(logits, actions)


def collect_rollout(
    state: TrainState, envs: list[KarelProgramEnv], rng: np.random.Generator, key: jax.Array, config: Config
) -> tuple[GRPOBatch, NDArray[np.float32], dict[str, float], jax.Array]:
    batch_size = config.num_tasks * config.group_size
    if len(envs) != batch_size or any(env.config != config.env for env in envs):
        raise ValueError("Expected num_tasks * group_size environments with the configured limits")
    # Same reset seed gives every member of a group the same sampled pair.
    seeds = np.repeat(rng.integers(0, 2**31, size=config.num_tasks), config.group_size)
    pairs = [env.reset(seed=int(seed)) for env, seed in zip(envs, seeds)]
    initial = np.stack([pair.initial for pair in pairs])
    target = np.stack([pair.target for pair in pairs])
    shape = (config.env.max_program_tokens, batch_size)
    actions = np.full(shape, KarelProgramEnv.pad_token_id, dtype=np.int32)
    old_log_probs = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.bool_)
    rewards = np.zeros(batch_size, dtype=np.float32)
    successes = np.zeros(batch_size, dtype=np.bool_)
    active = np.ones(batch_size, dtype=np.bool_)
    errors: list[str | None] = [None] * batch_size
    carry, logits = prefill(state, initial, target)
    for t in range(shape[0]):
        key, sample_key = jax.random.split(key)
        sampled, log_probs = jax.device_get(act(logits, sample_key))
        mask[t] = active
        actions[t, active] = sampled[active]
        old_log_probs[t, active] = log_probs[active]
        for index in np.flatnonzero(active):
            _, reward, terminated, truncated, info = envs[index].step(int(sampled[index]))
            rewards[index] += reward
            if terminated or truncated:
                active[index] = False
                errors[index] = info["error"]
                successes[index] = bool(info["success"])
        if not active.any():
            break
        carry, logits = decode_step(state, actions[t], carry)
    grouped_rewards = rewards.reshape((config.num_tasks, config.group_size))
    advantages = np.asarray(group_advantages(grouped_rewards)).reshape(-1)
    diagnostics = {
        "charts/reward_mean": float(rewards.mean()),
        "charts/success_rate": float(successes.mean()),
        "charts/group_success_rate": float(successes.reshape((config.num_tasks, config.group_size)).any(axis=1).mean()),
        "charts/informative_group_fraction": float((np.ptp(grouped_rewards, axis=1) > 0).mean()),
        "charts/episode_length_mean": float(mask.sum(axis=0).mean()),
        "charts/truncation_rate": errors.count("token_limit") / batch_size,
        "charts/syntax_error_rate": errors.count("syntax_error") / batch_size,
        "charts/runtime_error_rate": errors.count("runtime_error") / batch_size,
        "charts/execution_limit_rate": errors.count("execution_limit") / batch_size,
    }
    return GRPOBatch(initial, target, actions, old_log_probs, mask, advantages), rewards, diagnostics, key


def objective(
    logits: jax.Array, batch: GRPOBatch, config: Config, reference_log_probs: jax.Array | None = None
) -> tuple[jax.Array, Metrics]:
    """Token-clipped GRPO, with equal weight per program regardless of length."""
    chex.assert_rank(batch.actions, 2)
    chex.assert_equal_shape((batch.actions, batch.old_log_probs, batch.mask))
    chex.assert_shape(batch.advantages, (batch.actions.shape[1],))
    chex.assert_type((batch.old_log_probs, batch.advantages), jnp.float32)
    chex.assert_type(batch.mask, jnp.bool_)
    mask = batch.mask & (batch.actions != KarelProgramEnv.pad_token_id)
    log_probs = action_log_prob(logits, batch.actions)

    def average(values: jax.Array) -> jax.Array:
        chex.assert_equal_shape((values, batch.mask))
        chex.assert_type(values, jnp.float32)
        return (jnp.where(mask, values, 0).sum(axis=0) / jnp.maximum(mask.sum(axis=0), 1)).mean()

    log_ratio = jnp.where(mask, log_probs - jax.lax.stop_gradient(batch.old_log_probs), 0.0)
    ratio = jnp.exp(log_ratio)
    advantage = jax.lax.stop_gradient(batch.advantages)[None]
    policy_loss = -average(
        jnp.minimum(ratio * advantage, jnp.clip(ratio, 1 - config.clip_coef, 1 + config.clip_coef) * advantage)
    )
    policy_logits = generation_logits(logits)
    # PAD has probability zero. Replace its log-probability before multiplying
    # to avoid 0 * -inf and NaN gradients in the entropy term.
    entropy_log_probs = jax.nn.log_softmax(policy_logits).at[..., KarelProgramEnv.pad_token_id].set(0.0)
    entropy = average(-(jax.nn.softmax(policy_logits) * entropy_log_probs).sum(axis=-1))
    approx_kl = average(jnp.expm1(log_ratio) - log_ratio)
    clip_fraction = average((jnp.abs(ratio - 1) > config.clip_coef).astype(jnp.float32))
    reference_kl = jnp.asarray(0.0, dtype=jnp.float32)
    if config.kl_coef:
        if reference_log_probs is None:
            raise ValueError("Reference log probabilities are required when kl_coef > 0")
        chex.assert_equal_shape((reference_log_probs, batch.actions))
        chex.assert_type(reference_log_probs, jnp.float32)
        delta = jnp.where(mask, jax.lax.stop_gradient(reference_log_probs) - log_probs, 0.0)
        reference_kl = average(jnp.expm1(delta) - delta)
    loss = policy_loss - config.entropy_coef * entropy + config.kl_coef * reference_kl
    return loss, (policy_loss, entropy, approx_kl, clip_fraction, reference_kl)


@partial(jax.jit, static_argnames="config")
def update(
    state: TrainState, batch: GRPOBatch, config: Config, reference_params: optax.Params | None = None
) -> tuple[TrainState, Metrics]:
    reference_log_probs = None
    if config.kl_coef:
        if reference_params is None:
            raise ValueError("Frozen reference parameters are required when kl_coef > 0")
        _, reference_logits = state.apply_fn(
            {"params": reference_params}, batch.initial, batch.target, batch.actions[:-1]
        )
        reference_log_probs = action_log_prob(reference_logits, batch.actions)

    def loss_fn(params: optax.Params) -> tuple[jax.Array, Metrics]:
        # Context predicts action[0]; action[t-1] predicts action[t], including m).
        _, logits = state.apply_fn({"params": params}, batch.initial, batch.target, batch.actions[:-1])
        return objective(logits, batch, config, reference_log_probs)

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    if config.target_kl is None:
        return state.apply_gradients(grads=grads), metrics
    state = jax.lax.cond(metrics[2] > config.target_kl, lambda: state, lambda: state.apply_gradients(grads=grads))
    return state, metrics


def train(config: Config) -> TrainState:
    batch_size = config.num_tasks * config.group_size
    envs = [KarelProgramEnv(config.env) for _ in range(batch_size)]
    rng = np.random.default_rng(config.seed)
    key, init_key = jax.random.split(jax.random.key(config.seed))
    model = KarelProgramModel(
        d_model=config.d_model,
        num_layers=config.num_layers,
        d_state=config.d_state,
        headdim=config.headdim,
        conv_channels=config.conv_channels,
        max_markers=config.env.max_markers,
    )
    initial, target = envs[0].reset(seed=config.seed)
    optimizer = optax.inject_hyperparams(
        lambda learning_rate: optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm), optax.adam(learning_rate, eps=1e-5)
        )
    )
    state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(init_key, initial[None], target[None], jnp.empty((0, 1), dtype=jnp.int32))["params"],
        tx=optimizer(config.learning_rate),
    )
    reference_params = state.params if config.kl_coef else None
    schedule = learning_rate_schedule(config)
    run_name = f"karel_grpo_seed{config.seed}_{datetime.now(UTC):%Y%m%d-%H%M%S-%f}"
    run_dir = f"{config.log_dir.rstrip('/')}/{run_name}"
    writer = SummaryWriter(logdir=run_dir)
    try:
        writer.add_text("config", f"```yaml\n{yaml.safe_dump(asdict(config))}```", 0)
        writer.add_text("devices", str(jax.devices()), 0)
        parameter_count = sum(parameter.size for parameter in jax.tree.leaves(state.params))
        writer.add_scalar("model/params_millions", parameter_count / 1_000_000, 0)
        print(
            f"TensorBoard run: {run_dir}\nJAX devices: {jax.devices()}\nModel parameters: {parameter_count:,}",
            flush=True,
        )
        print(
            "Syntax-distance and progress rewards: equal-reward groups have zero GRPO advantages.",
            flush=True,
        )
        start = monotonic()
        steps = 0
        for iteration in range(config.total_updates):
            rollout_start = monotonic()
            batch, _, diagnostics, key = collect_rollout(state, envs, rng, key, config)
            rollout_seconds = monotonic() - rollout_start
            steps += int(batch.mask.sum())
            learning_rate = float(schedule(iteration))
            state = state.replace(
                opt_state=state.opt_state._replace(
                    hyperparams={**state.opt_state.hyperparams, "learning_rate": jnp.asarray(learning_rate)}
                )
            )
            optimization_start = monotonic()
            metrics: list[Metrics] = []
            early_stop = False
            updates_done = 0
            for _ in range(config.update_epochs):
                for indices in np.split(rng.permutation(batch_size), config.num_minibatches):
                    minibatch = GRPOBatch(
                        batch.initial[indices],
                        batch.target[indices],
                        batch.actions[:, indices],
                        batch.old_log_probs[:, indices],
                        batch.mask[:, indices],
                        batch.advantages[indices],
                    )
                    state, metric = update(state, minibatch, config, reference_params)
                    metrics.append(metric)
                    if config.target_kl is not None and float(metric[2]) > config.target_kl:
                        early_stop = True
                        break
                    updates_done += 1
                if early_stop:
                    break
            jax.block_until_ready((state, metrics))
            policy_loss, entropy, approx_kl, clip_fraction, reference_kl = np.mean(jax.device_get(metrics), axis=0)
            for tag, scalar in {
                **diagnostics,
                "losses/policy": policy_loss,
                "policy/entropy": entropy,
                "policy/approx_kl": approx_kl,
                "policy/clip_fraction": clip_fraction,
                "policy/reference_kl": reference_kl,
                "policy/early_stop": early_stop,
                "charts/learning_rate": learning_rate,
                "charts/updates_per_rollout": updates_done,
                "charts/total_episodes": (iteration + 1) * batch_size,
                "charts/steps_per_second": steps / (monotonic() - start),
                "time/rollout_seconds": rollout_seconds,
                "time/optimization_seconds": monotonic() - optimization_start,
            }.items():
                writer.add_scalar(tag, float(scalar), steps)
            writer.flush()
            print(
                f"iteration={iteration + 1} step={steps} success={diagnostics['charts/success_rate']:.3f} "
                f"reward={diagnostics['charts/reward_mean']:.3f} "
                f"informative_groups={diagnostics['charts/informative_group_fraction']:.3f} "
                f"policy={policy_loss:.3f} entropy={entropy:.3f} kl={approx_kl:.4f} "
                f"updates={updates_done} early_stop={early_stop}",
                flush=True,
            )
        return state
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grpo_karel.yaml", help="Path to a YAML config")
    parser.add_argument("--platform", choices=("cpu", "cuda", "metal"), help="Require a JAX backend")
    args = parser.parse_args()
    if args.platform is not None:
        jax.config.update("jax_platforms", args.platform)
        jax.devices()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
