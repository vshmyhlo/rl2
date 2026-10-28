"""LoRA-only GRPO training with a JAX learner, vLLM rollouts, and Ray actors.

Rollout generation needs no explicit lock around weight publication: vLLM's
pause/resume mechanism queues new requests and handles in-flight requests
according to weight_update_pause_mode.

MLflow's metric step counts completed training iterations: one rollout batch,
its optimizer updates, and publication of the updated weights. All metrics below
are logged every iteration except startup scalars, weight-transfer timing, and checkpoint timing.

Startup scalars (at the restored iteration, or step 0 for a new run):
    model/parameters_billions: Model parameter count in billions.
    model/trainable_parameters_billions: LoRA parameter count in billions.
    learner/setup_seconds: Model/optimizer setup, checkpoint restore, and initial adapter publication wall time.

Optimization scalars (averaged over minibatches and update epochs):
    train/policy_loss: Clipped GRPO policy loss.
    train/approx_kl: Mean exp(log_ratio) - 1 - log_ratio, where log_ratio is learner minus sampling log-probability.
    train/clip_fraction: Fraction of sampled-token probability ratios outside the clip range.
    train/gradient_norm: Global LoRA gradient norm after microbatch averaging, before clipping.
    train/skipped_update_fraction: Fraction of optimizer minibatches skipped because all advantages are zero.
Loss, KL, and clip fraction average sampled tokens within each rollout, then rollouts.
Skipped minibatches contribute zero to loss, KL, clip fraction, and gradient norm metrics.

Rollout batch scalars:
    rollout/reward_mean: Mean raw rollout reward.
    rollout/reward_std: Population standard deviation of raw rollout rewards.
    rollout/reward_min: Minimum raw rollout reward.
    rollout/reward_max: Maximum raw rollout reward.
    rollout/adjusted_reward_mean: Mean rollout reward after the length penalty.
    rollout/adjusted_reward_std: Population standard deviation of adjusted rollout rewards.
    rollout/adjusted_reward_min: Minimum adjusted rollout reward.
    rollout/adjusted_reward_max: Maximum adjusted rollout reward.
    rollout/degenerate_group_fraction: Fraction of groups with identical adjusted rewards and zero advantages.
    rollout/stop_fraction/<label>: Fraction of batch rollouts with each StopCondition.
    rollout/turns_mean: Mean model responses per rollout.
    rollout/input_tokens_mean: Mean context tokens appended per rollout, excluding sampled tokens.
    rollout/output_tokens_mean: Mean sampled tokens per rollout across all model calls.
    rollout/total_tokens_mean: Mean full transcript tokens per learner batch row, including context and sampled tokens, excluding padding.
    batcher/token_utilization_fraction: Non-padding transcript tokens divided by batch size times padded sequence length.
    batcher/padded_sequence_tokens: Padded sequence width chosen during batch construction, before the next-token shift.
All StopCondition labels are logged each iteration, including zero for absent labels.

Training progress and settings:
    train/iterations_total: Cumulative completed training iterations.
    train/updates_total: Cumulative optimizer updates, excluding skipped minibatches.
    train/rollouts_total: Cumulative rollouts used for training.
    train/learning_rate: Configured optimizer learning rate.

Timing and throughput (logged every iteration unless noted):
    learner/iterations_per_second: Reciprocal of iteration wall time.
    learner/rollouts_per_second: Batch rollout count divided by iteration wall time.
    learner/output_tokens_per_second: Batch sampled-token count divided by iteration wall time.
    actor/rollout_seconds_mean: Mean agent execution time per rollout, including model calls and environment steps.
    batcher/fetch_rollout_seconds: Wall time fetching all rollout groups, including actor return transfers.
    batcher/prepare_batch_seconds: Wall time preparing the CPU training arrays without explicit synchronization.
    learner/batch_fetch_seconds: Wall time fetching the rollout batch, including generation, scoring, and Ray transfers.
    learner/update_seconds: Device transfer and optimizer update wall time.
    learner/weight_transfer_seconds: Wall time exporting and reloading LoRA adapters in vLLM; logged only on publication.
    learner/iteration_seconds: Batch fetching, optimization, and weight publication wall time.
    checkpoint/save_seconds: Checkpoint save wall time, including completion of writes; logged on saves.
    batcher/actor_transfer_seconds_mean: Mean group transfer latency from actor return to batcher receipt.
    learner/batcher_transfer_seconds: Time from batcher return to learner receipt, including prefetch buffering.
Ray transfer latencies include serialization, transport, deserialization, and
scheduling, using a shared single-node monotonic clock. Timing intervals overlap
and should not be summed. Iteration timing and throughput exclude that iteration's
metric and example logging and checkpoint saves. Cumulative counters include resumed training.
Output-token counts include every sampled token, including thinking and end-of-turn tokens.
"""

import argparse
import asyncio
import json
import logging
import os
import posixpath
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import AsyncGenerator, Generator
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, closing, contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
from functools import partial
from itertools import islice
from pathlib import Path
from statistics import mean, pstdev
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Literal, NamedTuple, get_args
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4

import httpx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import ray
import yaml
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from openai import AsyncOpenAI
from pydantic import Field, TypeAdapter
from pydantic.dataclasses import dataclass as config_dataclass
from ray.actor import ActorHandle
from safetensors.numpy import save_file
from transformers import AutoTokenizer
from vllm.config.lora import MaxLoRARanks

from rl2.agent_loop import Agent, Rollout, StopCondition, TrainingSequence
from rl2.agent_task import Task
from rl2.alfworld_env import ALFWorldConfig, stream_alfworld_tasks
from rl2.appworld_env import AppWorldConfig, stream_appworld_tasks
from rl2.configuration import BaseConfig, load_config
from rl2.grpo_v2.reasoning_gym import ReasoningGymConfig, stream_reasoning_gym_tasks
from rl2.grpo_v2.state import LoRATrainState
from rl2.mlflow_logging import MLflowLogger
from rl2.qwen3 import (
    AttentionBackend,
    LoRAParameters,
    LoRAWeights,
    Parameters,
    Qwen3LM,
    load_qwen3_checkpoint,
)
from rl2.ray_timeline import record_timeline
from rl2.tracing import flush_tracing, span, tracing_enabled

logger = logging.getLogger("rl2.train_grpo_v2")
MODEL = "Qwen/Qwen3-0.6B"
SERVER_URL = "http://127.0.0.1:8000"
LORA_NAME = "training-policy"


@config_dataclass(frozen=True)
class Config(BaseConfig):
    model: str = MODEL
    taskset: Annotated[ALFWorldConfig | AppWorldConfig | ReasoningGymConfig, Field(discriminator="id")] = (
        ALFWorldConfig(config_file="data/alfworld/base_config.yaml")
    )
    seed: Annotated[int, Field(ge=0)] = 1
    learner_gpus: Annotated[int, Field(ge=1)] = 1
    num_groups: Annotated[int, Field(ge=1)] = 1
    group_size: Annotated[int, Field(ge=2)] = 4
    prefetch_batches: Annotated[int, Field(ge=0)] = 0  # Positive values trade policy staleness for overlap.
    drop_degenerate_groups: bool = False
    length_penalty_coef: Annotated[float, Field(ge=0)] = 0.0  # Per-token cost for nonzero environment rewards.
    total_iterations: Annotated[int | None, Field(ge=1)] = None
    checkpoint_interval: Annotated[int, Field(gt=0)] = 100
    example_log_interval: Annotated[int, Field(gt=0)] = 10
    max_tokens: Annotated[int, Field(ge=1)] = 512  # Total transcript tokens, including system, prompt, and tools.
    padding_multiple: Annotated[int, Field(ge=1)] = 256
    max_turns: Annotated[int, Field(ge=1)] = 3
    rollout_timeout: Annotated[float, Field(gt=0)] = 120
    environment_timeout: Annotated[float, Field(gt=0)] = 600
    temperature: Annotated[float, Field(ge=0.01, le=2.0)] = 1.0  # vLLM clamps smaller positive values to 0.01.
    learning_rate: Annotated[float, Field(gt=0)] = 1e-5
    lora_rank: Annotated[int, Field(ge=1, le=512)] = 16
    lora_alpha: Annotated[float, Field(gt=0)] = 32.0
    clip_coef: Annotated[float, Field(gt=0, lt=1)] = 0.2
    max_grad_norm: Annotated[float, Field(gt=0)] = 1.0
    update_epochs: Annotated[int, Field(ge=1)] = 1
    minibatches_per_epoch: Annotated[int, Field(ge=1)] = 4
    grad_accum_steps: Annotated[int, Field(ge=1)] = 1
    attention_backend: AttentionBackend = "cudnn"  # Lower worst-case sampled-token error in the A100 vLLM audit.
    scan_layers: bool = True
    gradient_checkpointing: bool = True
    enable_ray_dashboard: bool = False
    enable_ray_timeline: bool = False
    enable_tracing: bool = True
    # In-flight vLLM requests during weight updates (individual requests, not multi-turn rollout groups):
    # wait: finish requests before applying weights (default).
    # abort: cancel requests before applying weights.
    # keep: freeze requests, then resume them after applying weights.
    weight_update_pause_mode: Literal["wait", "abort", "keep"] = "wait"

    def __post_init__(self) -> None:
        if self.num_groups * self.group_size % (self.minibatches_per_epoch * self.grad_accum_steps):
            raise ValueError("minibatches_per_epoch * grad_accum_steps must divide num_groups * group_size")
        if self.num_groups * self.group_size % (self.minibatches_per_epoch * self.grad_accum_steps * self.learner_gpus):
            raise ValueError("learner_gpus must divide the microbatch size")


DEFAULT_CONFIG = Config()


@dataclass(frozen=True)
class TrainingProgress:
    train_state: LoRATrainState  # Frozen base weights, LoRA factors/moments; () optimizer step.
    shuffle_key: jax.Array  # () typed base PRNG key, folded with the optimizer step.
    iteration: int
    total_rollouts: int
    sampling_state: dict[str, Any]


def checkpoint_manager(run_dir: str) -> ocp.CheckpointManager:
    directory = f"{run_dir.rstrip('/')}/checkpoints"
    if not directory.startswith("gs://"):
        directory = str(Path(directory).resolve())
    return ocp.CheckpointManager(directory, options=ocp.CheckpointManagerOptions(max_to_keep=1))


def save_checkpoint(manager: ocp.CheckpointManager, progress: TrainingProgress, config: Config) -> float:
    """Save a completed iteration and wait for writes before reusing learner buffers."""
    started = time.monotonic()
    manager.save(
        progress.iteration,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave(
                {
                    "train_state": progress.train_state,
                    "shuffle_key": jax.random.key_data(progress.shuffle_key),  # (K,) raw PRNG words.
                }
            ),
            metadata=ocp.args.JsonSave(
                {
                    "version": 2,
                    "algorithm": "grpo_v2",
                    "config": asdict(config),
                    "iteration": progress.iteration,
                    "total_rollouts": progress.total_rollouts,
                    "sampling_state": progress.sampling_state,
                }
            ),
        ),
        force=True,
    )
    manager.wait_until_finished()
    seconds = time.monotonic() - started
    logger.info("Saved checkpoint at iteration %s in %.1fs", progress.iteration, seconds)
    return seconds


def restore_checkpoint(
    manager: ocp.CheckpointManager, train_state: LoRATrainState, config: Config
) -> TrainingProgress | None:
    iteration = manager.latest_step()
    if iteration is None:
        return None
    metadata = manager.restore(iteration, args=ocp.args.Composite(metadata=ocp.args.JsonRestore())).metadata
    if metadata["version"] != 2 or metadata["algorithm"] != "grpo_v2":
        raise ValueError("Unsupported GRPO v2 checkpoint; LoRA training requires a new checkpoint directory")
    for name in ("model", "lora_rank", "lora_alpha"):
        if metadata["config"].get(name) != getattr(config, name):
            raise ValueError(f"GRPO v2 checkpoint {name} does not match the current configuration")
    if metadata["iteration"] != iteration or iteration < 0 or metadata["total_rollouts"] < 0:
        raise ValueError("Invalid GRPO v2 checkpoint progress")
    restored = manager.restore(
        iteration,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(
                {
                    "train_state": train_state,
                    "shuffle_key": jax.device_put(
                        jax.random.key_data(jax.random.key(config.seed)), train_state.step.sharding
                    ),  # (K,)
                }
            )
        ),
    ).state
    return TrainingProgress(
        train_state=restored["train_state"],
        shuffle_key=jax.random.wrap_key_data(restored["shuffle_key"]),  # ()
        iteration=iteration,
        total_rollouts=metadata["total_rollouts"],
        sampling_state=metadata["sampling_state"],
    )


@dataclass(frozen=True)
class RolloutGroup:
    task_id: str | None  # Source task ID shared by the group; None if the task has no ID.
    rollouts: list[Rollout]  # Independent attempts at the same task, in sampling order, scored as one group.
    length_penalty_coef: float = 0.0

    @property
    def training_rewards(self) -> list[float]:
        return [
            rollout.reward - self.length_penalty_coef * rollout.num_output_tokens if rollout.reward != 0 else 0.0
            for rollout in self.rollouts
        ]

    @property
    def is_degenerate(self) -> bool:
        rewards = self.training_rewards
        return all(reward == rewards[0] for reward in rewards)


@dataclass(frozen=True)
class Transfer[T]:
    value: T
    ready_at: float  # Same-node monotonic timestamp immediately before Ray serializes the return value.


def format_rollout_examples(groups: list[RolloutGroup]) -> str:
    """Format up to eight full decoded token sequences in sampling order."""
    sections: list[str] = []
    examples = islice(((group, rollout) for group in groups for rollout in group.rollouts), 8)
    for index, (group, rollout) in enumerate(examples, start=1):
        sections.append(
            f"{'=' * 72}\n"
            f"Example {index} | task={group.task_id} | reward={rollout.reward:.3f} | "
            f"advantage={rollout.advantage:.3f}\n"
            f"Turns: {rollout.num_turns} | Output tokens: {rollout.num_output_tokens} | "
            f"Stop: {rollout.stop_condition}\n\n"
            f"Full token sequence (special tokens preserved):\n{rollout.interaction or '(empty)'}"
        )
    return "\n\n".join(sections)


class GRPOBatch[Array: np.ndarray | jax.Array](NamedTuple):
    token_ids: Array  # (B, T) int32 right-padded complete transcripts.
    attention_mask: Array  # (B, T) bool context and sampled tokens, excluding padding.
    loss_mask: Array  # (B, T) bool sampled tokens only.
    old_log_probs: Array  # (B, T) float32 temperature-scaled sampling-policy log-probabilities.
    advantages: Array  # (B,) float32 normalized within each rollout group.


@dataclass(frozen=True)
class RolloutBatch:
    training_batch: GRPOBatch[np.ndarray]  # Leaves: (B, T), except advantages: (B,).
    rollout_metrics: dict[str, float]
    stop_counts: dict[StopCondition, int]
    num_output_tokens: int
    examples: str
    transfer_metrics: dict[str, float | int]


def build_rollout_batch(
    groups: list[RolloutGroup],
    training_batch: GRPOBatch[np.ndarray],  # Leaves: (B, T), except advantages: (B,).
    transfer_metrics: dict[str, float | int],
    *,
    include_examples: bool = False,
) -> RolloutBatch:
    """Keep only training arrays and the summaries used by learner logging."""
    rollouts = [rollout for group in groups for rollout in group.rollouts]
    rewards = [rollout.reward for rollout in rollouts]
    training_rewards = [reward for group in groups for reward in group.training_rewards]
    return RolloutBatch(
        training_batch=training_batch,
        rollout_metrics={
            "rollout/reward_mean": mean(rewards),
            "rollout/reward_std": pstdev(rewards),
            "rollout/reward_min": min(rewards),
            "rollout/reward_max": max(rewards),
            "rollout/adjusted_reward_mean": mean(training_rewards),
            "rollout/adjusted_reward_std": pstdev(training_rewards),
            "rollout/adjusted_reward_min": min(training_rewards),
            "rollout/adjusted_reward_max": max(training_rewards),
            "rollout/degenerate_group_fraction": mean(group.is_degenerate for group in groups),
            "rollout/turns_mean": mean(rollout.num_turns for rollout in rollouts),
            "rollout/input_tokens_mean": mean(rollout.num_input_tokens for rollout in rollouts),
            "rollout/output_tokens_mean": mean(rollout.num_output_tokens for rollout in rollouts),
            "rollout/total_tokens_mean": float(training_batch.attention_mask.sum(axis=1).mean()),
            "batcher/token_utilization_fraction": float(training_batch.attention_mask.mean()),
            "batcher/padded_sequence_tokens": float(training_batch.attention_mask.shape[1]),
            "actor/rollout_seconds_mean": mean(rollout.rollout_seconds for rollout in rollouts),
        },
        stop_counts=Counter(rollout.stop_condition for rollout in rollouts),
        num_output_tokens=sum(rollout.num_output_tokens for rollout in rollouts),
        examples=format_rollout_examples(groups) if include_examples else "",
        transfer_metrics=transfer_metrics,
    )


# Use a value alias so Ray can serialize script-defined actors without importing __main__.Metrics.
Metrics = dict[str, jax.Array]  # Scalar leaves.


def token_log_probs(
    state: LoRATrainState,
    params: LoRAParameters,  # (R, I), (O, R) adapter factors.
    batch: GRPOBatch[jax.Array],  # (B, T), (B,) leaves.
    temperature: float,
) -> jax.Array:  # (B, T-1)
    """(B, T-1) log-probabilities for tokens at positions 1 through T-1."""
    logits = state.apply_fn(
        {"params": state.base_params},
        batch.token_ids[:, :-1],
        attention_mask=batch.attention_mask[:, :-1],
        lora=params,
        lora_scale=state.lora_scale,
    ).logits  # (B, T-1, V)
    logits = logits.astype(jnp.float32) / temperature  # (B, T-1, V)
    log_probs = jax.nn.log_softmax(logits, axis=-1)  # (B, T-1, V); stable shifted arithmetic, as in vLLM.
    # Fuse selection's backward pass with softmax instead of scattering into a full FP32 vocabulary buffer.
    targets = jax.nn.one_hot(batch.token_ids[:, 1:], logits.shape[-1], dtype=jnp.bool_)  # (B, T-1, V)
    return jnp.where(targets, log_probs, 0).sum(axis=-1)  # (B, T-1)


def objective(
    log_probs: jax.Array,  # (B, T-1)
    batch: GRPOBatch[jax.Array],  # (B, T), (B,) leaves.
    clip_coef: float,
) -> tuple[jax.Array, Metrics]:  # (), scalar metric leaves
    mask = batch.loss_mask[:, 1:]  # (B, T-1)

    def average(values: jax.Array) -> jax.Array:  # (B, T-1) -> ()
        return (jnp.where(mask, values, 0).sum(axis=1) / jnp.maximum(mask.sum(axis=1), 1)).mean()  # ()

    log_ratio = jnp.where(mask, log_probs - batch.old_log_probs[:, 1:], 0)  # (B, T-1)
    ratio = jnp.exp(log_ratio)  # (B, T-1)
    advantages = batch.advantages[:, None]  # (B, 1)
    loss = -average(jnp.minimum(ratio * advantages, jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef) * advantages))  # ()
    return loss, {
        "train/policy_loss": loss,  # ()
        "train/approx_kl": average(jnp.expm1(log_ratio) - log_ratio),  # ()
        "train/clip_fraction": average((jnp.abs(ratio - 1) > clip_coef).astype(jnp.float32)),  # ()
    }


def create_state(
    model: Qwen3LM,
    params: Parameters,  # Frozen HF parameter shapes.
    config: Config,
) -> LoRATrainState:
    """Freeze base weights and optimize FP32 LoRA factors on attention and MLP projections."""
    adapters: LoRAParameters = {}  # (R, I) and (O, R) leaves.
    key = jax.random.key(config.seed)  # ()
    for name, weight in sorted(params.items()):  # HF parameter shapes.
        if not (name.startswith("model.layers.") and name.endswith("_proj.weight")):
            continue
        key, adapter_key = jax.random.split(key)  # Each ().
        output_size, input_size = weight.shape
        adapters[name] = LoRAWeights(
            a=jax.random.uniform(
                adapter_key,
                (config.lora_rank, input_size),
                dtype=jnp.float32,
                minval=-(input_size**-0.5),
                maxval=input_size**-0.5,
            ),  # (R, I)
            b=jnp.zeros((output_size, config.lora_rank), dtype=jnp.float32),  # (O, R)
        )
    if not adapters:
        raise ValueError("No attention or MLP projection weights found for LoRA")
    return LoRATrainState.create(
        apply_fn=model.apply,
        params=adapters,
        base_params=params,
        lora_scale=config.lora_alpha / config.lora_rank,
        tx=optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate)),
    ).replace(step=jnp.asarray(0, dtype=jnp.int32))  # () Match the device counter returned by update.


@partial(
    jax.jit,
    static_argnames=("grad_accum_steps", "temperature", "clip_coef"),
    in_shardings=(PartitionSpec(), PartitionSpec("batch")),
    out_shardings=PartitionSpec(),
    donate_argnums=(0,),
)
def update(
    state: LoRATrainState,
    batch: GRPOBatch[jax.Array],  # One optimizer minibatch: (B, T), (B,) leaves.
    grad_accum_steps: int,
    temperature: float,
    clip_coef: float,
) -> tuple[LoRATrainState, Metrics]:
    """Consume state buffers, skipping zero-advantage updates; requires a `batch` mesh context and positional args."""
    if grad_accum_steps < 1 or batch.token_ids.shape[0] % grad_accum_steps:
        raise ValueError("grad_accum_steps must be positive and divide the minibatch size")
    if batch.token_ids.shape[0] % (grad_accum_steps * jax.sharding.get_abstract_mesh().shape["batch"]):
        raise ValueError("learner device count must divide the microbatch size")
    # Python logging runs on tracing, not on cached executions.
    logger.info(
        "JIT trace update: batch_shape=%s grad_accum_steps=%s temperature=%s clip_coef=%s",
        batch.token_ids.shape,
        grad_accum_steps,
        temperature,
        clip_coef,
    )

    def apply_update() -> tuple[LoRATrainState, Metrics]:
        microbatch_size = batch.token_ids.shape[0] // grad_accum_steps
        microbatches = jax.tree.map(
            lambda value: value.reshape(grad_accum_steps, microbatch_size, *value.shape[1:]), batch
        )  # (A, B/A, T); advantages: (A, B/A). A = grad_accum_steps.
        microbatches = jax.lax.with_sharding_constraint(
            microbatches, PartitionSpec(None, "batch")
        )  # (A, B/A, T), (A, B/A); shard examples within each microbatch.

        def accumulate(
            grads: LoRAParameters,  # (R, I) and (O, R) leaves.
            microbatch: GRPOBatch[jax.Array],  # (B/A, T), (B/A,) leaves.
        ) -> tuple[LoRAParameters, Metrics]:
            def loss_fn(params: LoRAParameters) -> tuple[jax.Array, Metrics]:  # LoRA leaves -> (), scalar metrics.
                return objective(token_log_probs(state, params, microbatch, temperature), microbatch, clip_coef)

            (_, metrics), gradient = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            return jax.tree.map(jnp.add, grads, gradient), metrics  # LoRA factor shapes; scalar metrics.

        gradients, metrics = jax.lax.scan(
            accumulate, jax.tree.map(jnp.zeros_like, state.params), microbatches
        )  # LoRA factor shapes; metric leaves: (A,).
        gradients = jax.tree.map(lambda value: value / grad_accum_steps, gradients)  # LoRA factor shapes.
        gradients = jax.lax.with_sharding_constraint(gradients, PartitionSpec())  # LoRA factor shapes.
        metrics = jax.tree.map(jnp.mean, metrics)  # Scalar leaves.
        metrics["train/gradient_norm"] = optax.global_norm(gradients)  # () Before clipping.
        metrics["train/skipped_update_fraction"] = jnp.asarray(0.0, dtype=jnp.float32)  # ()
        return state.apply_gradients(grads=gradients), metrics

    def skip_update() -> tuple[LoRATrainState, Metrics]:
        return state, {
            "train/policy_loss": jnp.asarray(0.0, dtype=jnp.float32),  # ()
            "train/approx_kl": jnp.asarray(0.0, dtype=jnp.float32),  # () Not evaluated.
            "train/clip_fraction": jnp.asarray(0.0, dtype=jnp.float32),  # () Not evaluated.
            "train/gradient_norm": jnp.asarray(0.0, dtype=jnp.float32),  # ()
            "train/skipped_update_fraction": jnp.asarray(1.0, dtype=jnp.float32),  # ()
        }

    return jax.lax.cond(jnp.any(batch.advantages != 0), apply_update, skip_update)


def score_group(task: Task, rollouts: list[Rollout], length_penalty_coef: float = 0.0) -> RolloutGroup:
    if len(rollouts) < 2:
        raise ValueError("A rollout group needs at least two rollouts")
    group = RolloutGroup(
        task_id=f"{task.name}:{task.id}",
        rollouts=rollouts,
        length_penalty_coef=length_penalty_coef,
    )
    rewards = group.training_rewards
    reward_mean, reward_std = mean(rewards), pstdev(rewards)
    return replace(
        group,
        rollouts=[
            replace(rollout, advantage=(reward - reward_mean) / (reward_std + 1e-4))
            for rollout, reward in zip(rollouts, rewards, strict=True)
        ],
    )


@asynccontextmanager
async def create_agent(config: Config = DEFAULT_CONFIG) -> AsyncGenerator[Agent]:
    tokenizer = await asyncio.to_thread(AutoTokenizer.from_pretrained, config.model)
    async with AsyncOpenAI(
        base_url=f"{SERVER_URL}/v1", api_key="EMPTY", timeout=config.rollout_timeout, max_retries=3
    ) as client:
        models = await client.models.list()
        model = next(model for model in models.data if model.id == config.model)
        yield Agent(
            client=client,
            tokenizer=tokenizer,
            model=LORA_NAME,
            context_length=TypeAdapter(int).validate_python(
                (model.model_extra or {}).get("max_model_len"), strict=True
            ),
            max_tokens=config.max_tokens,
            max_turns=config.max_turns,
            temperature=config.temperature,
            rollout_timeout=config.rollout_timeout,
            environment_timeout=config.environment_timeout,
        )


@contextmanager
def serve_vllm(model: str = MODEL, lora_rank: int = DEFAULT_CONFIG.lora_rank) -> Generator[subprocess.Popen[bytes]]:
    max_lora_rank = next((rank for rank in get_args(MaxLoRARanks) if rank >= lora_rank), None)
    if max_lora_rank is None:
        raise ValueError(f"vLLM supports LoRA ranks up to {max(get_args(MaxLoRARanks))}")
    logger.info("Starting vLLM server for %s", model)
    # Inherit the GPU assigned to this actor through CUDA_VISIBLE_DEVICES.
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "rl2.vllm_support.server",
            "--model",
            model,
            "--host",
            os.environ.get("RL2_VLLM_HOST", "127.0.0.1"),
            "--port",
            "8000",
            "--disable-uvicorn-access-log",
            "--dtype",
            "bfloat16",
            "--hf-overrides",
            '{"head_dtype":"float32"}',
            "--generation-config",
            "vllm",
            "--logprobs-mode",
            "processed_logprobs",
            "--enable-lora",
            "--max-lora-rank",
            str(max_lora_rank),
            "--lora-dtype",
            "bfloat16",
        ],
        env={
            **os.environ,
            "VLLM_SERVER_DEV_MODE": "1",
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
        },
    ) as server:
        try:
            yield server
        finally:
            logger.info("Stopping vLLM server: pid=%s", server.pid)
            server.terminate()
    logger.info("vLLM server exited")


def wait_for_server(server: subprocess.Popen[bytes], timeout: float = 600) -> None:
    logger.info("Waiting for vLLM at %s", SERVER_URL)
    deadline = time.monotonic() + timeout
    while (remaining := deadline - time.monotonic()) > 0:
        if (returncode := server.poll()) is not None:
            raise RuntimeError(f"vLLM exited with code {returncode} before becoming ready; see server logs above")
        try:
            with urlopen(f"{SERVER_URL}/health", timeout=min(5, remaining)):
                logger.info("vLLM is ready")
                return
        except (URLError, TimeoutError) as error:
            logger.debug("vLLM health check failed: %s", error)
            time.sleep(min(1, max(0, deadline - time.monotonic())))
    logger.error("vLLM readiness timed out after %ss at %s", timeout, SERVER_URL)
    raise TimeoutError(f"vLLM was not ready within {timeout}s at {SERVER_URL}")


# Shared imports must not initialize CUDA in rollout workers.
@ray.remote(runtime_env={"env_vars": {"JAX_PLATFORMS": "cpu"}})
class Batcher:
    def __init__(self, actor: ActorHandle, config: Config = DEFAULT_CONFIG) -> None:
        self.actor = actor
        self.config = config

    def prepare_batch(self, groups: list[RolloutGroup], context_length: int) -> GRPOBatch[np.ndarray]:
        """Flatten rollout groups in order and right-pad their transcripts into NumPy arrays.

        B is the total rollout count; T is the longest transcript length rounded up
        to a multiple of config.padding_multiple, capped at min(config.max_tokens, context_length).
        Padding is zero/False; transcripts and rollout advantages are preserved.

        Returns:
            GRPOBatch with token_ids: (B, T) int32 complete transcripts;
            attention_mask: (B, T) bool non-padding positions;
            loss_mask: (B, T) bool sampled-token positions;
            old_log_probs: (B, T) float32 sampling-policy log-probabilities;
            advantages: (B,) float32 per-rollout advantages.

        Raises:
            ValueError: If there are no rollouts, a transcript exceeds the token budget,
                or a transcript lacks an initial context token followed by a sampled token.
        """
        rollouts = [rollout for group in groups for rollout in group.rollouts]
        if not rollouts:
            raise ValueError("A training batch must contain rollouts")
        length = max(len(rollout.sequence.token_ids) for rollout in rollouts)
        token_limit = min(self.config.max_tokens, context_length)
        if length > token_limit:
            raise ValueError(f"Rollout length {length} exceeds token budget {token_limit}")
        multiple = self.config.padding_multiple
        width = min((length + multiple - 1) // multiple * multiple, token_limit)
        shape = (len(rollouts), width)
        # Host copies avoid compiling JAX padding and reductions for every transcript length.
        batch = GRPOBatch(
            token_ids=np.zeros(shape, dtype=np.int32),  # (B, T)
            attention_mask=np.zeros(shape, dtype=np.bool_),  # (B, T)
            loss_mask=np.zeros(shape, dtype=np.bool_),  # (B, T)
            old_log_probs=np.zeros(shape, dtype=np.float32),  # (B, T)
            advantages=np.asarray([rollout.advantage for rollout in rollouts], dtype=np.float32),  # (B,)
        )
        for index, rollout in enumerate(rollouts):
            sequence: TrainingSequence = rollout.sequence
            length = len(sequence.token_ids)
            mask = sequence.loss_mask  # (L,) unpadded transcript length.
            if length < 2 or mask[0] or not mask[1:].any():
                raise ValueError("Each rollout needs context followed by at least one sampled token")
            batch.token_ids[index, :length] = sequence.token_ids  # (L,)
            batch.loss_mask[index, :length] = sequence.loss_mask  # (L,)
            batch.old_log_probs[index, :length] = sequence.old_log_probs  # (L,)
            batch.attention_mask[index, :length] = True  # (L,)
        return batch  # (B, T), (B,) leaves.

    async def fetch_next_batch(
        self, num_groups: int, group_size: int, context_length: int, *, include_examples: bool = False
    ) -> Transfer[RolloutBatch]:
        if num_groups < 1:
            raise ValueError("num_groups must be at least 1")
        started = time.monotonic()

        async def fetch_group() -> tuple[RolloutGroup, float]:
            while True:
                result = await self.actor.fetch_next_group.remote(group_size=group_size)
                seconds = time.monotonic() - result.ready_at
                if not self.config.drop_degenerate_groups or not result.value.is_degenerate:
                    return result.value, seconds

        # Timestamp each arrival before waiting for slower groups.
        async with asyncio.TaskGroup() as pending:
            tasks = [pending.create_task(fetch_group()) for _ in range(num_groups)]
        results = [task.result() for task in tasks]
        fetched = time.monotonic()
        groups = [group for group, _ in results]
        with span("batcher.prepare_batch"):
            training_batch = self.prepare_batch(groups, context_length=context_length)  # (B, T), (B,) leaves.
        prepared = time.monotonic()
        batch = build_rollout_batch(
            groups=groups,
            training_batch=training_batch,
            include_examples=include_examples,
            transfer_metrics={
                "batcher/fetch_rollout_seconds": fetched - started,
                "batcher/prepare_batch_seconds": prepared - fetched,
                "batcher/actor_transfer_seconds_mean": mean(seconds for _, seconds in results),
            },
        )
        return Transfer(value=batch, ready_at=time.monotonic())


@ray.remote(num_gpus=1)
class Learner:
    def __init__(self, batcher: ActorHandle, actor: ActorHandle, config: Config = DEFAULT_CONFIG) -> None:
        self.config = config
        self.batcher = batcher
        self.actor = actor
        self.shuffle_key = jax.random.key(config.seed)  # () typed base PRNG key.
        self._pending_batches: list[ray.ObjectRef] = []

    def start(self) -> None:
        self.mesh = Mesh(devices=jax.devices("gpu"), axis_names=("batch",))
        with (
            jax.set_mesh(self.mesh),
            MLflowLogger(experiment="grpo_v2", run_name=self.config.run_id) as writer,
            ExitStack() as resources,
        ):
            manager = resources.enter_context(closing(checkpoint_manager(self.config.ckpt_dir)))
            writer.log_params(asdict(self.config))
            logger.info("Run ID: %s; MLflow run: %s", self.config.run_id, writer.run_id)
            start = time.monotonic()
            logger.info("Loading learner model %s on %s assigned GPUs", self.config.model, self.config.learner_gpus)
            self.model, variables = load_qwen3_checkpoint(
                self.config.model,
                dtype="bfloat16",
                attention_backend=self.config.attention_backend,
                scan_layers=self.config.scan_layers,
                gradient_checkpointing=self.config.gradient_checkpointing,
            )
            self.state = jax.device_put(
                create_state(self.model, variables["params"], self.config),
                NamedSharding(self.mesh, PartitionSpec()),
            )
            del variables
            progress = restore_checkpoint(manager, self.state, self.config)
            iteration = 0
            total_rollouts = 0
            if progress is not None:
                self.state = progress.train_state
                self.shuffle_key = progress.shuffle_key  # ()
                iteration = progress.iteration
                total_rollouts = progress.total_rollouts
                ray.get(self.actor.set_sampling_state.remote(progress.sampling_state))
                logger.info("Resumed run %s at iteration %s", self.config.run_id, iteration)
            writer.log_text(yaml.safe_dump(asdict(self.config)), f"config/{iteration}.yaml")
            logger.info(
                "Learner loaded %s base tensors and %s LoRA adapters",
                len(self.state.base_params),
                len(self.state.params),
            )
            resources.callback(self.cancel_pending_batches)
            self.publish_weights()
            total_updates = int(self.state.step)
            writer.log_metrics(
                {
                    "model/parameters_billions": sum(p.size for p in jax.tree.leaves(self.state.base_params))
                    / 1_000_000_000,
                    "model/trainable_parameters_billions": sum(p.size for p in jax.tree.leaves(self.state.params))
                    / 1_000_000_000,
                    "learner/setup_seconds": time.monotonic() - start,
                },
                step=iteration,
            )
            logger.info("Starting GRPO training loop")
            while self.config.total_iterations is None or iteration < self.config.total_iterations:
                iteration += 1
                checkpoint_due = (
                    iteration % self.config.checkpoint_interval == 0 or iteration == self.config.total_iterations
                )
                # Each iteration is a separate trace, including its remote calls and child tasks.
                with span(
                    "train.iteration",
                    {"run_id": self.config.run_id or writer.run_id, "iteration": iteration, "model": self.config.model},
                    root=True,
                ):
                    started = time.monotonic()
                    with span("learner.fetch_batch"):
                        batch = self.fetch_next_batch(iteration=iteration)
                    fetched = time.monotonic()
                    with span("learner.optimize"):
                        update_metrics = self.optimize(batch.training_batch)
                    updated_step = int(self.state.step)
                    weights_updated = updated_step != total_updates
                    total_updates = updated_step
                    optimized = time.monotonic()
                    if weights_updated:
                        with span("learner.publish_weights"):
                            self.publish_weights()
                    now = time.monotonic()
                    num_rollouts = batch.training_batch.token_ids.shape[0]
                    total_rollouts += num_rollouts
                    seconds = max(now - started, 1e-9)
                    scalars = {
                        **update_metrics,
                        **batch.transfer_metrics,
                        **batch.rollout_metrics,
                        **{
                            f"rollout/stop_fraction/{condition}": batch.stop_counts.get(condition, 0) / num_rollouts
                            for condition in get_args(StopCondition)
                        },
                        "train/iterations_total": iteration,
                        "train/updates_total": total_updates,
                        "train/learning_rate": self.config.learning_rate,
                        "train/rollouts_total": total_rollouts,
                        "learner/iterations_per_second": 1 / seconds,
                        "learner/rollouts_per_second": num_rollouts / seconds,
                        "learner/output_tokens_per_second": batch.num_output_tokens / seconds,
                        "learner/batch_fetch_seconds": fetched - started,
                        "learner/update_seconds": optimized - fetched,
                        "learner/iteration_seconds": now - started,
                    }
                    if weights_updated:
                        scalars["learner/weight_transfer_seconds"] = now - optimized
                    logger.info(
                        "iteration=%s reward=%.3f degenerate_group_fraction=%.3f batch=%.2fs "
                        "group_transfer=%.6fs batch_transfer=%.6fs loss=%.4f update=%.2fs%s",
                        iteration,
                        scalars["rollout/reward_mean"],
                        scalars["rollout/degenerate_group_fraction"],
                        scalars["learner/batch_fetch_seconds"],
                        scalars["batcher/actor_transfer_seconds_mean"],
                        scalars["learner/batcher_transfer_seconds"],
                        scalars["train/policy_loss"],
                        scalars["learner/update_seconds"],
                        f" weights={scalars['learner/weight_transfer_seconds']:.2f}s" if weights_updated else "",
                    )
                    if iteration % self.config.example_log_interval == 0:
                        with span("learner.log_examples"):
                            examples = f"Rollout examples | step {iteration}\n\n{batch.examples}"
                            writer.log_text(examples, f"examples/{iteration}.txt")
                    if checkpoint_due:
                        with span("learner.checkpoint"):
                            scalars["checkpoint/save_seconds"] = save_checkpoint(
                                manager,
                                TrainingProgress(
                                    train_state=self.state,
                                    shuffle_key=self.shuffle_key,  # ()
                                    iteration=iteration,
                                    total_rollouts=total_rollouts,
                                    sampling_state=ray.get(self.actor.get_sampling_state.remote()),
                                ),
                                self.config,
                            )
                    with span("learner.log_metrics"):
                        writer.log_metrics(scalars, step=iteration)

    def cancel_pending_batches(self) -> None:
        while self._pending_batches:
            ray.cancel(self._pending_batches.pop(0), recursive=True)

    def fetch_next_batch(self, *, iteration: int) -> RolloutBatch:
        prefetch_batches = min(self.config.prefetch_batches, (-iteration) % self.config.checkpoint_interval)
        if self.config.total_iterations is not None:
            prefetch_batches = min(prefetch_batches, self.config.total_iterations - iteration)
        while len(self._pending_batches) < prefetch_batches + 1:
            self._pending_batches.append(
                self.batcher.fetch_next_batch.remote(
                    num_groups=self.config.num_groups,
                    group_size=self.config.group_size,
                    context_length=self.model.config.max_position_embeddings,
                    include_examples=True,
                )
            )
        result = ray.get(self._pending_batches[0])
        self._pending_batches.pop(0)
        seconds = time.monotonic() - result.ready_at
        batch = result.value
        return replace(
            batch,
            transfer_metrics={**batch.transfer_metrics, "learner/batcher_transfer_seconds": seconds},
        )

    def publish_weights(self) -> None:
        host_params = jax.device_get(self.state.params)  # Host (R, I) and (O, R) leaves.
        adapters = jax.tree.map(
            lambda value: np.asarray(value, dtype=np.float32), host_params
        )  # Host (R, I) and (O, R) leaves.
        weights = {
            f"base_model.model.{name.removesuffix('.weight')}.lora_{factor}.weight": value
            for name, adapter in adapters.items()
            for factor, value in (("A", adapter.a), ("B", adapter.b))
        }  # PEFT (R, I) and (O, R) leaves.
        # vLLM retains this single adapter in memory after the synchronous load.
        with TemporaryDirectory(prefix="rl2-lora-") as temporary_directory:
            directory = Path(temporary_directory)
            save_file(weights, directory / "adapter_model.safetensors")
            (directory / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "peft_type": "LORA",
                        "task_type": "CAUSAL_LM",
                        "base_model_name_or_path": self.config.model,
                        "r": self.config.lora_rank,
                        "lora_alpha": self.config.lora_alpha,
                        "target_modules": sorted({name.split(".")[-2] for name in adapters}),
                        "bias": "none",
                        "lora_dropout": 0.0,
                        "inference_mode": True,
                        "use_rslora": False,
                    }
                )
            )
            ray.get(self.actor.update_weights.remote(str(directory)))

    def optimize(self, batch: GRPOBatch[np.ndarray]) -> dict[str, float]:  # Batch leaves: (B, T), (B,).
        if batch.token_ids.shape[0] % self.config.minibatches_per_epoch:
            raise ValueError("minibatches_per_epoch must divide the rollout batch size")
        minibatch_size = batch.token_ids.shape[0] // self.config.minibatches_per_epoch
        if minibatch_size % (self.config.grad_accum_steps * self.mesh.size):
            raise ValueError("grad_accum_steps * learner device count must divide the minibatch size")
        batched = NamedSharding(self.mesh, PartitionSpec("batch"))
        device_batch: GRPOBatch[jax.Array] = jax.device_put(batch, batched)  # (B, T), (B,) leaves.
        history: list[Metrics] = []
        for _ in range(self.config.update_epochs):
            key = jax.random.fold_in(self.shuffle_key, self.state.step)  # () typed PRNG key.
            permutation = jax.random.permutation(key, batch.token_ids.shape[0])  # (B,)
            shuffled = jax.tree.map(
                lambda value, permutation=permutation: value[permutation], device_batch
            )  # (B, T), (B,) leaves.
            for start in range(0, batch.token_ids.shape[0], minibatch_size):
                minibatch = jax.tree.map(
                    lambda value, start=start: value[start : start + minibatch_size], shuffled
                )  # (B/N, T), (B/N,) leaves; N = minibatches_per_epoch.
                minibatch = jax.device_put(minibatch, batched)  # (B/N, T), (B/N,) leaves.
                self.state, metrics = update(
                    self.state,
                    minibatch,
                    self.config.grad_accum_steps,
                    self.config.temperature,
                    self.config.clip_coef,
                )
                history.append(metrics)
        metrics = jax.tree.map(lambda *values: jnp.stack(values).mean(), *history)  # Scalar leaves.
        return {name: float(value) for name, value in metrics.items()}


# Restrict JAX before import/backend initialization; vLLM still uses the assigned CUDA device.
@ray.remote(num_gpus=1, runtime_env={"env_vars": {"JAX_PLATFORMS": "cpu"}})
class Actor:
    def __init__(self, config: Config = DEFAULT_CONFIG) -> None:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.config = config
        self.rng = random.Random(config.seed)
        self._server = ExitStack()
        self._agent_resources = AsyncExitStack()
        self._agent: Agent | None = None
        self._agent_lock = asyncio.Lock()
        self._runs: set[asyncio.Task[Rollout]] = set()
        self._closed = False
        self.tasks: Generator[Task]
        if isinstance(config.taskset, ReasoningGymConfig):
            self.tasks = stream_reasoning_gym_tasks(config.taskset, self.rng)
        elif isinstance(config.taskset, AppWorldConfig):
            self.tasks = stream_appworld_tasks(config.taskset, self.rng, config.environment_timeout)
        else:
            self.tasks = stream_alfworld_tasks(config.taskset, self.rng, config.environment_timeout)
        logger.info("Starting actor and vLLM on its assigned GPU")
        with self._server as resources:
            server = resources.enter_context(serve_vllm(model=config.model, lora_rank=config.lora_rank))
            wait_for_server(server)
            self._server = resources.pop_all()
        logger.info("Actor ready for rollout requests: taskset=%s", config.taskset.id)

    async def _get_agent(self) -> Agent:
        async with self._agent_lock:
            if self._closed:
                raise RuntimeError("Actor is closed")
            if self._agent is None:
                self._agent = await self._agent_resources.enter_async_context(create_agent(self.config))
            return self._agent

    async def aclose(self) -> None:
        async with self._agent_lock, AsyncExitStack() as resources, self._agent_resources:
            # Ray termination does not call a user-defined shutdown hook.
            resources.push_async_callback(asyncio.to_thread, self._server.close)
            self._closed = True
            runs = list(self._runs)
            for run in runs:
                run.cancel()
            # Drain tasks owned by collect_group's TaskGroup before closing agent resources.
            await asyncio.gather(*runs, return_exceptions=True)

    def get_sampling_state(self) -> dict[str, Any]:
        version, state, gauss_next = self.rng.getstate()
        return {"version": version, "state": list(state), "gauss_next": gauss_next}

    def set_sampling_state(self, state: dict[str, Any]) -> None:
        self.rng.setstate((state["version"], tuple(state["state"]), state["gauss_next"]))

    async def fetch_next_group(self, group_size: int) -> Transfer[RolloutGroup]:
        task = next(self.tasks)
        group = await self.collect_group(task, group_size=group_size)
        return Transfer(value=group, ready_at=time.monotonic())

    async def update_weights(self, adapter_path: str) -> None:
        with span("actor.update_weights"):
            async with httpx.AsyncClient(base_url=SERVER_URL, timeout=600) as client:
                with span("actor.update_weights.pause", {"mode": self.config.weight_update_pause_mode}):
                    response = await client.post(
                        "/pause", params={"mode": self.config.weight_update_pause_mode, "clear_cache": "true"}
                    )
                    response.raise_for_status()
                with span("actor.update_weights.load_adapter"):
                    response = await client.post(
                        "/v1/load_lora_adapter",
                        json={"lora_name": LORA_NAME, "lora_path": adapter_path, "load_inplace": True},
                    )
                    response.raise_for_status()
                with span("actor.update_weights.resume"):
                    response = await client.post("/resume")
                    response.raise_for_status()

    async def collect_group(self, task: Task, group_size: int) -> RolloutGroup:
        if group_size < 2:
            raise ValueError("group_size must be at least 2")
        span_data = {"group_id": uuid4().hex, "task_id": str(task.id), "group_size": str(group_size)}
        started = time.monotonic()
        try:
            with span("actor.collect_group", {**span_data, "run_id": self.config.run_id or "unspecified"}):
                agent = await self._get_agent()
                async with asyncio.TaskGroup() as pending:

                    async def run_rollout(index: int) -> Rollout:
                        with span("actor.rollout", {**span_data, "rollout_index": index}) as rollout_span:
                            async with AsyncExitStack() as resources:
                                async with asyncio.timeout(self.config.rollout_timeout):
                                    env = await resources.enter_async_context(task.init_env())
                                    result = await agent.run(env=env)
                            rollout_span.set_attribute("stop_condition", result.stop_condition)
                            return result

                    runs = [pending.create_task(run_rollout(index)) for index in range(group_size)]
                    self._runs.update(runs)
                    for run in runs:
                        run.add_done_callback(self._runs.discard)
                with span("actor.score_group"):
                    group = score_group(
                        task, [run.result() for run in runs], length_penalty_coef=self.config.length_penalty_coef
                    )
        except Exception:
            logger.exception(
                "Group failed: task=%s group_size=%s elapsed=%.2fs",
                task.id,
                group_size,
                time.monotonic() - started,
            )
            raise
        return group


def configure_console_logging() -> None:
    log_format = "%(asctime)s\t%(levelname)s %(filename)s:%(lineno)d -- %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, force=True)
    for handler in logging.getLogger("ray").handlers:
        handler.setFormatter(logging.Formatter(log_format))


def train(config: Config) -> None:
    configure_console_logging()
    os.environ["RAY_AUTH_MODE"] = "disabled"
    os.environ["RAY_DASHBOARD_ENABLE_PROFILING"] = "1" if config.enable_ray_dashboard else "0"
    timeline_path = os.environ.get(
        "RL2_RAY_TIMELINE_PATH", posixpath.join(config.ckpt_dir, "profiles", f"ray-{uuid4().hex}.json")
    )
    # The HTTP endpoint is shared over loopback by workers on this node.
    with (
        ExitStack() as resources,
        ray.init(
            address="local",
            include_dashboard=config.enable_ray_dashboard,
            dashboard_host=os.environ.get("RL2_RAY_DASHBOARD_HOST", "127.0.0.1"),
            dashboard_port=8265,
            runtime_env={
                "env_vars": {"RAY_PROFILING": "1"} if config.enable_ray_timeline else {},
                "worker_process_setup_hook": configure_console_logging,
            },
            # Zero disables event reporting in Ray 2.59 despite the timeline docstring.
            _system_config={
                "enable_timeline": config.enable_ray_timeline,
                "task_events_report_interval_ms": 1000,
                "kill_child_processes_on_worker_exit_with_raylet_subreaper": True,
            },
            _tracing_startup_hook="rl2.tracing:setup_tracing" if config.enable_tracing and tracing_enabled() else None,
        ),
        record_timeline(timeline_path) if config.enable_ray_timeline else nullcontext(),
    ):
        resources.callback(flush_tracing)
        logger.info("Creating batcher, learner, and actor")
        actor = Actor.remote(config=config)
        batcher = Batcher.remote(actor=actor, config=config)
        learner = Learner.options(num_gpus=config.learner_gpus).remote(batcher=batcher, actor=actor, config=config)
        try:
            logger.info("Starting learner")
            ray.get(learner.start.remote())
        finally:
            logger.info("Requesting graceful worker termination")
            try:
                ray.get(actor.aclose.remote())
            finally:
                shutdown = [
                    learner.__ray_terminate__.remote(),
                    actor.__ray_terminate__.remote(),
                    batcher.__ray_terminate__.remote(),
                ]
                ray.wait(shutdown, num_returns=len(shutdown))
            logger.info("Worker termination finished")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GRPO with MLflow logging")
    parser.add_argument("--config", required=True, help="Path to YAML configuration")
    args = parser.parse_args()
    train(load_config(args.config, Config))


if __name__ == "__main__":
    main()
