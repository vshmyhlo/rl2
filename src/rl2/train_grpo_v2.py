"""GRPO training with a JAX learner, vLLM rollouts, and Ray actors.

Rollout generation needs no explicit lock around weight publication: vLLM's
pause/resume mechanism queues new requests and handles in-flight requests
according to weight_update_pause_mode.

MLflow's metric step counts completed outer iterations: batch_size rollouts split into
grad_accum_steps equal microbatches, one attempted optimizer update, and any scheduled weight publication.
Below, a consumed batch means one microbatch. All metrics below
are logged every iteration except startup scalars, weight-transfer timing, checkpoint timing,
unevaluated policy diagnostics, gradient norms for skipped updates, and tasksets absent from the iteration.

Startup scalars (at the restored iteration, or step 0 for a new run):
    model/parameters_billions: Model parameter count in billions.
    model/trainable_parameters_billions: Trainable parameter count in billions.
    learner/setup_seconds: Model/optimizer setup, checkpoint restore, and initial weight publication wall time.

Optimization scalars (batch means averaged over the iteration unless noted):
    train/policy_loss: CISPO detached clipped-importance-weighted REINFORCE, GRPO clipped policy loss with truncated rollout importance weights, or StableReinforce score-centered policy loss excluding KL.
    train/approx_kl: GRPO only: mean exp(log_ratio) - 1 - log_ratio, where log_ratio is current minus frozen pre-update learner log-probability, bounded to [-20, 20].
    train/clip_fraction: GRPO only: fraction of current/anchor sampled-token probability ratios outside the PPO clip range.
    train/rollout_log_prob_error_mean: Mean absolute learner minus rollout log-probability difference; uses frozen learner scores in GRPO and current scores in CISPO/StableReinforce.
    train/rollout_importance_weight_mean: CISPO/GRPO: mean applied token importance weight after configured clipping (GRPO also safety-bounds log ratios).
    train/rollout_importance_weight_clipped_fraction: CISPO/GRPO: fraction of raw CISPO ratios outside its configured thresholds, or safety-bounded GRPO weights above loss.rollout_is_threshold.
    train/loss: StableReinforce only: score-centered policy loss plus the configured sampler KL penalty, using uncapped KL values.
    train/score_correction_mean: StableReinforce only: mean additive log-probability surrogate whose gradient centers the score; in nats.
    train/sampler_kl: CISPO/StableReinforce: sampler-to-learner KL estimate in nats. CISPO uses k3 for diagnostics only; StableReinforce reports its configured estimator before kl_coeff. k1 is log(q)-log(p), which can be negative on a finite batch. k3 is exp(d)-1-d with d=log(p)-log(q), capped per token at 1e30 before averaging with the exponent bounded before evaluation. This diagnostic cap does not affect the loss or gradients. Logged even when kl_coeff is zero.
    train/sampler_topk_fraction_mean: StableReinforce only: mean recorded sampler probability mass in the retained top K alternatives, dimensionless.
    train/gradient_norm: Global norm of the window-averaged gradient before clipping; logged only when an optimizer update is applied.
    train/skipped_update_fraction: Logged at accumulation boundaries: 1 if the entire window has no training signal and is skipped, otherwise 0.
Loss and policy diagnostics average unmasked sampled tokens within each rollout, then all rollouts in each batch.
Policy diagnostics average only evaluated batches and are omitted if none are evaluated.
Fully masked rollouts contribute zero; importance weights do not change denominators.
Batches with no unmasked tokens carrying nonzero advantages contribute zero policy loss and omit policy diagnostics,
unless StableReinforce's kl_coeff is positive, in which case retained tokens still train through KL.
Fully inactive StableReinforce batches also contribute zero train/loss and omit its four diagnostics.
Losses use dimensionless rewards; gradient norm is the unnormalized L2 norm over trainable parameter gradients.
Every consumed batch counts toward grad_accum_steps; inactive batches contribute zero gradients.
Gradients average equally across the window before clipping and AdamW; an entirely inactive window preserves optimizer state.

Rollout scalars (pooled over all batches consumed in the iteration):
    rollout/reward_mean: Mean raw rollout reward.
    rollout/reward_mean/<taskset_id>: Mean raw reward over retained iteration rollouts from that configured taskset key; omitted if absent.
    rollout/reward_std: Population standard deviation of raw rollout rewards.
    rollout/reward_min: Minimum raw rollout reward.
    rollout/reward_max: Maximum raw rollout reward.
    rollout/adjusted_reward_mean: Mean rollout reward after the reward-proportional generated-token cost.
    rollout/adjusted_reward_std: Population standard deviation of adjusted rollout rewards.
    rollout/adjusted_reward_min: Minimum adjusted rollout reward.
    rollout/adjusted_reward_max: Maximum adjusted rollout reward.
    rollout/degenerate_group_fraction: Fraction of consumed rollouts whose full source group has identical adjusted rewards and zero advantages.
    rollout/stop_fraction/<label>: Fraction of iteration rollouts with each StopCondition.
    rollout/turns_mean: Mean model responses per rollout.
    rollout/input_tokens_mean: Mean context tokens appended per rollout, excluding sampled tokens.
    rollout/output_tokens_mean: Mean sampled tokens per rollout across all model calls.
    rollout/total_tokens_mean: Mean full transcript tokens per learner batch row, including context and sampled tokens, excluding padding.
    batcher/token_utilization_fraction: Total non-padding transcript tokens divided by total padded token capacity across batches.
    batcher/padded_sequence_tokens: Final consumed batch's padded sequence width, before the next-token shift.
    batcher/skipped_groups: Groups discarded by drop_degenerate_groups while filling the iteration's batches, including prefetch.
    batcher/stale_tokens: Sampled tokens masked from training by max_staleness across the iteration's batches.
    batcher/context_transcript_token_fraction: Context tokens divided by all non-padding transcript tokens.
    batcher/stale_transcript_token_fraction: Stale sampled tokens divided by all non-padding transcript tokens.
    batcher/masked_transcript_token_fraction: Context plus stale sampled tokens divided by all non-padding transcript tokens.
    batcher/unmasked_transcript_token_fraction: Tokens retained by the loss mask divided by all non-padding transcript tokens.
    batcher/weight_age_transcript_token_fraction/<age>: Sampled tokens in this age bucket divided by all non-padding transcript tokens.
    batcher/weight_age_masked_transcript_token_fraction/<age>: Masked sampled tokens in this age bucket divided by all non-padding transcript tokens.
    batcher/weight_age_unmasked_transcript_token_fraction/<age>: Unmasked sampled tokens in this age bucket divided by all non-padding transcript tokens.
Context, stale, and unmasked fractions sum to one; masked is context plus stale.
Unmasked tokens may have zero advantages. Context tokens have no generating weight version.
Age is publications behind the version at batch consumption: 0 through max_staleness, then older.
Weight-age metrics omit absent buckets and retain measured zeros for present buckets.
All StopCondition labels are logged each iteration, including zero for absent labels.

Training progress and settings:
    train/iterations_total: Cumulative completed outer iterations, including entirely inactive windows.
    train/updates_total: Cumulative optimizer updates, excluding skipped updates.
    train/rollouts_total: Cumulative rollouts used for training.
    train/learning_rate: Effective optimizer learning rate after AdamW batch-size scaling.

Timing and throughput (logged every iteration unless noted):
    learner/iterations_per_second: Reciprocal of iteration wall time.
    learner/rollouts_per_second: All consumed rollouts divided by iteration wall time.
    learner/output_tokens_per_second: All consumed sampled tokens divided by iteration wall time.
    actor/rollout_seconds_mean: Mean agent execution time per rollout, including model calls and environment steps.
    batcher/fetch_rollout_seconds: Sum of wall times waiting for each batch's groups, including actor return transfers; excludes prior prefetch time.
    batcher/prepare_batch_seconds: Sum of CPU batch preparation wall times without explicit synchronization.
    learner/batch_fetch_seconds: Sum of batch-fetch wall times, including generation, scoring, and Ray transfers.
    learner/update_seconds: Sum of device transfer, gradient accumulation, and optimizer update wall times, excluding weight publication.
    learner/weight_transfer_seconds: Wall time publishing weights to vLLM; logged only on publication.
    learner/iteration_seconds: Batch fetching, optimization, and weight publication wall time.
    checkpoint/save_blocking_seconds: Blocking save dispatch wall time, including staging and any previous-save wait;
        excludes background writes for this checkpoint; logged on saves.
Timing intervals overlap and should not be summed. Iteration timing and throughput exclude that iteration's
metric and example logging and checkpoint saves. Cumulative counters include resumed training.
Output-token counts include every sampled token, including thinking and end-of-turn tokens.
"""

import argparse
import asyncio
import logging
import os
import posixpath
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from collections.abc import AsyncGenerator, Generator
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, closing, contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path
from statistics import mean, pstdev
from typing import Annotated, Any, Literal, NamedTuple, cast, get_args
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
from flax.training.train_state import TrainState
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from openai import AsyncOpenAI
from pydantic import ConfigDict, Field, TypeAdapter
from pydantic.dataclasses import dataclass as config_dataclass
from ray.actor import ActorHandle
from ray.exceptions import ActorDiedError
from transformers import AutoTokenizer
from vllm.config.lora import MaxLoRARanks

from rl2.agent_loop import Agent, Rollout, StopCondition, TrainingSequence
from rl2.agent_task import Task
from rl2.alfworld_env import ALFWorldConfig, stream_alfworld_tasks
from rl2.appworld_env import AppWorldConfig, stream_appworld_tasks
from rl2.configuration import BaseConfig, load_config
from rl2.grpo_v2 import cispo, stable_reinforce
from rl2.grpo_v2.cispo import CISPOConfig
from rl2.grpo_v2.reasoning_gym import ReasoningGymConfig, stream_reasoning_gym_tasks
from rl2.grpo_v2.stable_reinforce import StableReinforceConfig
from rl2.grpo_v2.state import LoRATrainState, TrainParameters
from rl2.grpo_v2.weight_transfer import full_weight_publisher, lora_weight_publisher
from rl2.lm_envs.codegym.env import CodeGymConfig, stream_codegym_tasks
from rl2.lm_envs.crafter import CrafterConfig, stream_crafter_tasks
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
from rl2.vllm_support.retry_logging import add_retry_cause
from rl2.webshop_env import WebShopConfig, stream_webshop_tasks

logger = logging.getLogger("rl2.train_grpo_v2")
MODEL = "Qwen/Qwen3-0.6B"
SERVER_URL = "http://127.0.0.1:8000"
LORA_NAME = "training-policy"


@config_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LengthPenaltyConfig:
    reference_tokens: Annotated[int, Field(gt=0)] = 8192
    penalty: Annotated[float, Field(ge=0, lt=1)] = 0.0


@config_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class WeightedTasksetConfig:
    taskset: Annotated[
        ALFWorldConfig | AppWorldConfig | CodeGymConfig | CrafterConfig | ReasoningGymConfig | WebShopConfig,
        Field(discriminator="id"),
    ]
    weight: Annotated[float, Field(ge=0, lt=float("inf"))] = 1.0


TasksetConfigs = dict[Annotated[str, Field(pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")], WeightedTasksetConfig]


@config_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class VLLMConfig:
    """Optional serving overrides; omitted values use vLLM's defaults."""

    max_model_len: Annotated[int | None, Field(ge=1)] = None
    max_num_seqs: Annotated[int | None, Field(ge=1)] = None
    max_num_batched_tokens: Annotated[int | None, Field(ge=1)] = None
    gpu_memory_utilization: Annotated[float | None, Field(gt=0, le=1)] = None
    enable_prefix_caching: bool | None = None
    enable_chunked_prefill: bool | None = None
    async_scheduling: bool | None = None


@config_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LoRAConfig:
    mode: Literal["lora"] = "lora"
    rank: Annotated[int, Field(ge=1, le=512)] = 16
    alpha: Annotated[float, Field(gt=0)] = 32.0


@config_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class FullFinetuningConfig:
    mode: Literal["full"] = "full"


@config_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class GRPOLossConfig:
    kind: Literal["grpo"] = "grpo"
    clip_coef: Annotated[float, Field(gt=0, lt=1)] = 0.2
    rollout_is_threshold: Annotated[float, Field(gt=0)] = 2.0


LossConfig = Annotated[CISPOConfig | GRPOLossConfig | StableReinforceConfig, Field(discriminator="kind")]


@config_dataclass(frozen=True)
class Config(BaseConfig):
    model: str = MODEL
    vllm: VLLMConfig = VLLMConfig()
    tasksets: Annotated[TasksetConfigs, Field(min_length=1)] = field(
        default_factory=lambda: {
            "alfworld": WeightedTasksetConfig(taskset=ALFWorldConfig(config_file="data/alfworld/base_config.yaml"))
        }
    )
    seed: Annotated[int, Field(ge=0)] = 1
    learner_gpus: Annotated[int, Field(ge=1)] = 1
    batch_size: Annotated[int, Field(ge=1)] = 4  # Total rollouts per optimizer update, before accumulation splitting.
    group_size: Annotated[int, Field(ge=2)] = 4
    prefetch_batches: Annotated[int, Field(ge=0)] = 0  # Pending rollout capacity in units of batch_size.
    grad_accum_steps: Annotated[int, Field(ge=1)] = 1  # Equal microbatches per optimizer update.
    parameter_sync_steps: Annotated[int, Field(ge=1)] = 1  # Completed optimizer updates between weight publications.
    max_staleness: Annotated[int, Field(ge=0)] = 100  # Allowed weight publications since each token was sampled.
    drop_degenerate_groups: bool = False
    length_penalty: LengthPenaltyConfig = LengthPenaltyConfig()
    total_iterations: Annotated[int | None, Field(ge=1)] = None
    checkpoint_interval: Annotated[int, Field(gt=0)] = 100
    example_log_interval: Annotated[int, Field(gt=0)] = 10
    max_tokens: Annotated[int, Field(ge=1)] = 512  # Total transcript tokens, including system, prompt, and tools.
    padding_multiple: Annotated[int, Field(ge=1)] = 256
    max_turns: Annotated[int, Field(ge=1)] = 3
    rollout_timeout: Annotated[float, Field(gt=0)] = 120
    environment_timeout: Annotated[float, Field(gt=0)] = 600
    temperature: Annotated[float, Field(ge=0.01, le=2.0)] = 1.0  # vLLM clamps smaller positive values to 0.01.
    learning_rate: Annotated[float, Field(gt=0)] = 1e-6
    weight_decay: Annotated[float, Field(ge=0)] = 0.0
    # New/reference optimizer batch-size ratio; configure batch dimensions separately.
    adamw_batch_size_scale: Annotated[float, Field(gt=0, le=10)] = 1.0
    finetuning: Annotated[LoRAConfig | FullFinetuningConfig, Field(discriminator="mode")] = LoRAConfig()
    loss: LossConfig = CISPOConfig()
    max_grad_norm: Annotated[float, Field(gt=0)] = 1.0
    attention_backend: AttentionBackend = "cudnn"  # Lower worst-case sampled-token error in the A100 vLLM audit.
    scan_layers: bool = True
    gradient_checkpointing: bool = True
    enable_ray_dashboard: bool = False
    enable_ray_timeline: bool = False
    enable_tracing: bool = True
    # In-flight vLLM requests during weight updates (individual requests, not multi-turn rollout groups):
    # wait: finish requests before applying weights (default).
    # keep: freeze requests, then resume them after applying weights.
    weight_update_pause_mode: Literal["wait", "keep"] = "wait"

    def __post_init__(self) -> None:
        if not any(entry.weight > 0 for entry in self.tasksets.values()):
            raise ValueError("tasksets must contain at least one positive weight")
        if self.batch_size % self.grad_accum_steps:
            raise ValueError("grad_accum_steps must divide batch_size")
        if self.microbatch_size % self.learner_gpus:
            raise ValueError("learner_gpus must divide microbatch_size (batch_size // grad_accum_steps)")

    @property
    def microbatch_size(self) -> int:
        """Rollouts per accumulation step, summed across learner GPUs."""
        return self.batch_size // self.grad_accum_steps

    @property
    def adamw_kwargs(self) -> dict[str, float]:
        """Scale reference hyperparameters by the configured batch-size ratio."""
        scale = self.adamw_batch_size_scale**0.5
        return {
            "learning_rate": self.learning_rate * scale,
            "b1": 1 - self.adamw_batch_size_scale * 0.1,
            "b2": 1 - self.adamw_batch_size_scale * 0.001,
            "eps": 1e-8 / scale,
            "weight_decay": self.weight_decay * scale,
        }


DEFAULT_CONFIG = Config()


@dataclass(frozen=True)
class TrainingProgress:
    train_state: TrainState  # Trainable parameter/moment shapes; () optimizer step.
    shuffle_key: jax.Array  # () legacy PRNG key retained for checkpoint compatibility.
    iteration: int
    total_rollouts: int
    sampling_state: dict[str, Any]
    rollout_pool: list[dict[str, Any]] = field(default_factory=list)
    weight_version: int = -1  # Legacy checkpoints restart at publication 0.


def checkpoint_manager(run_dir: str) -> ocp.CheckpointManager:
    """Manage checkpoints under the run directory, retaining only the latest."""
    directory = f"{run_dir.rstrip('/')}/checkpoints"
    if not directory.startswith("gs://"):
        directory = str(Path(directory).resolve())
    return ocp.CheckpointManager(
        directory, options=ocp.CheckpointManagerOptions(max_to_keep=1, enable_async_checkpointing=True)
    )


def save_checkpoint(manager: ocp.CheckpointManager, progress: TrainingProgress, config: Config) -> float:
    """Stage a completed iteration for background writing before reusing learner buffers."""
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
                    "rollout_pool": progress.rollout_pool,
                    "weight_version": progress.weight_version,
                }
            ),
        ),
        force=True,
    )
    seconds = time.monotonic() - started
    logger.info("Dispatched async checkpoint at iteration %s in %.1fs", progress.iteration, seconds)
    return seconds


def restore_checkpoint(
    manager: ocp.CheckpointManager, train_state: TrainState, config: Config
) -> TrainingProgress | None:
    """Restore the latest compatible checkpoint, or return None if none exists."""
    iteration = manager.latest_step()
    if iteration is None:
        return None
    metadata = manager.restore(iteration, args=ocp.args.Composite(metadata=ocp.args.JsonRestore())).metadata
    if metadata["version"] != 2 or metadata["algorithm"] != "grpo_v2":
        raise ValueError("Unsupported GRPO v2 checkpoint; use a new checkpoint directory")
    saved_config = metadata["config"]
    saved_mode = saved_config.get("finetuning", saved_config.get("finetuning_mode", "lora"))
    if isinstance(saved_mode, str):
        # Older version 2 checkpoints stored LoRA settings at the top level.
        saved_mode = {"mode": saved_mode}
        if saved_mode["mode"] == "lora":
            saved_mode |= {"rank": saved_config.get("lora_rank"), "alpha": saved_config.get("lora_alpha")}
    if saved_mode != asdict(config.finetuning):
        raise ValueError("GRPO v2 checkpoint finetuning does not match the current configuration")
    if saved_config.get("model") != config.model:
        raise ValueError("GRPO v2 checkpoint model does not match the current configuration")
    saved_loss = saved_config.get("loss")
    if saved_loss is None:
        # Version 2 checkpoints predating the tagged loss stored these fields at the top level.
        saved_stable = saved_config.get("stable_reinforce")
        saved_loss = (
            {"kind": "stable_reinforce", **saved_stable}
            if saved_stable is not None
            else {
                "kind": "grpo",
                "clip_coef": saved_config.get("clip_coef", 0.2),
                "rollout_is_threshold": saved_config.get("rollout_is_threshold", 2.0),
            }
        )
    if saved_loss != asdict(config.loss):
        raise ValueError("GRPO v2 checkpoint loss does not match the current configuration")
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
        rollout_pool=metadata.get("rollout_pool", []),
        weight_version=metadata.get("weight_version", -1),
    )


@dataclass(frozen=True)
class RolloutGroup:
    task_id: str  # Source task ID shared by the group.
    taskset_id: str  # Configured tasksets key.
    rollouts: list[Rollout]  # Independent attempts at the same task, in sampling order, scored as one group.
    length_penalty: LengthPenaltyConfig = DEFAULT_CONFIG.length_penalty
    source_is_degenerate: bool | None = None  # Preserve full-group status when slicing.

    @property
    def training_rewards(self) -> list[float]:
        """Reduce rewards in [0, 1] proportionally to generated tokens only."""
        if self.length_penalty.penalty == 0:
            return [rollout.reward for rollout in self.rollouts]
        if any(not 0.0 <= rollout.reward <= 1.0 for rollout in self.rollouts):
            raise ValueError("Enabled length penalty requires rewards in [0, 1]")
        return [
            rollout.reward
            * (
                1.0
                - self.length_penalty.penalty
                * min(rollout.num_output_tokens / self.length_penalty.reference_tokens, 1.0)
            )
            for rollout in self.rollouts
        ]

    @property
    def is_degenerate(self) -> bool:
        """Whether all adjusted rewards are equal, yielding zero group advantages."""
        if self.source_is_degenerate is not None:
            return self.source_is_degenerate
        rewards = self.training_rewards
        return all(reward == rewards[0] for reward in rewards)


def format_rollout_examples(groups: list[RolloutGroup]) -> str:
    """Format full decoded token sequences in sampling order."""
    sections: list[str] = []
    examples = ((group, rollout) for group in groups for rollout in group.rollouts)
    for index, (group, rollout) in enumerate(examples, start=1):
        sections.append(
            f"{'=' * 72}\n"
            f"Example {index} | task={group.task_id} | reward={rollout.reward:.3f} | "
            f"advantage={rollout.advantage:.3f} | taskset={group.taskset_id}\n"
            f"Turns: {rollout.num_turns} | Output tokens: {rollout.num_output_tokens} | "
            f"Stop: {rollout.stop_condition}\n\n"
            f"Full token sequence (special tokens preserved):\n{rollout.interaction or '(empty)'}"
        )
    return "\n\n".join(sections)


class GRPOBatch[Array: np.ndarray | jax.Array](NamedTuple):
    token_ids: Array  # (B, T) int32 right-padded complete transcripts.
    attention_mask: Array  # (B, T) bool context and sampled tokens, excluding padding.
    loss_mask: Array  # (B, T) bool sampled tokens within max_staleness only.
    old_log_probs: Array  # (B, T) float32 temperature-scaled sampling-policy log-probabilities.
    advantages: Array  # (B,) float32 mean-centered adjusted rewards within each rollout group.
    sampler_topk_ids: Array | None = None  # (B, T, K) int32 sampling-time alternatives, if enabled.
    sampler_topk_log_probs: Array | None = None  # (B, T, K) float32 scores; zero for context/padding.


@dataclass(frozen=True)
class RolloutBatch:
    training_batch: GRPOBatch[np.ndarray]  # (B, T), (B,), optional (B, T, K) sampler leaves.
    rollout_metrics: dict[str, float]
    stop_counts: dict[StopCondition, int]
    num_output_tokens: int
    examples: str
    timing_metrics: dict[str, float | int]
    taskset_counts: dict[str, int] = field(default_factory=dict)


class BatchGradientResult(NamedTuple):
    gradient_sum: TrainParameters  # Parameter shapes.
    metrics: dict[str, float]
    has_signal: bool


class CollectedGroup(NamedTuple):
    group: RolloutGroup
    skipped_groups: int


class BatchSummary(NamedTuple):
    metrics: dict[str, float]
    num_rollouts: int
    taskset_counts: dict[str, int]


def summarize_iteration(batches: list[BatchSummary]) -> dict[str, float]:
    """Combine rollout statistics without retaining the training arrays."""
    result: dict[str, float] = {}
    for name in sorted({name for batch in batches for name in batch.metrics}):
        measurement = name.split("/")[1]
        present = [batch for batch in batches if name in batch.metrics]
        if name in ("batcher/skipped_groups", "batcher/stale_tokens"):
            result[name] = sum(batch.metrics[name] for batch in present)
        elif measurement.endswith("_min"):
            result[name] = min(batch.metrics[name] for batch in present)
        elif measurement.endswith("_max"):
            result[name] = max(batch.metrics[name] for batch in present)
        elif name == "batcher/padded_sequence_tokens":
            result[name] = present[-1].metrics[name]
        else:
            weights: list[float]
            if name.startswith("rollout/reward_mean/"):
                taskset_id = name.removeprefix("rollout/reward_mean/")
                weights = [batch.taskset_counts.get(taskset_id, 0) for batch in batches]
            elif "transcript_token_fraction" in name:
                weights = [batch.metrics["rollout/total_tokens_mean"] * batch.num_rollouts for batch in batches]
            elif name == "batcher/token_utilization_fraction":
                weights = [batch.metrics["batcher/padded_sequence_tokens"] * batch.num_rollouts for batch in batches]
            else:
                weights = [batch.num_rollouts for batch in batches]
            # Absent age buckets still contribute transcript tokens to the denominator.
            denominator = sum(weights)
            weighted = [
                (batch.metrics, weight) for batch, weight in zip(batches, weights, strict=True) if name in batch.metrics
            ]
            if measurement.endswith("_std"):
                mean_name = name.removesuffix("_std") + "_mean"
                average = sum(metrics[mean_name] * weight for metrics, weight in weighted) / denominator
                variance = (
                    sum(
                        (metrics[name] ** 2 + (metrics[mean_name] - average) ** 2) * weight
                        for metrics, weight in weighted
                    )
                    / denominator
                )
                result[name] = variance**0.5
            else:
                result[name] = sum(metrics[name] * weight for metrics, weight in weighted) / denominator
    return result


def build_rollout_batch(
    groups: list[RolloutGroup],
    training_batch: GRPOBatch[np.ndarray],  # Leaves: (B, T), except advantages: (B,).
    timing_metrics: dict[str, float | int],
    *,
    consumption_weight_version: int,
    max_staleness: int,
    skipped_groups: int = 0,
    latest_examples: dict[str, RolloutGroup] | None = None,
) -> RolloutBatch:
    """Summarize the batch and refresh the optional per-taskset example cache."""
    if latest_examples is None:
        latest_examples = {}
    rollouts = [rollout for group in groups for rollout in group.rollouts]
    rewards = [rollout.reward for rollout in rollouts]
    training_rewards = [reward for group in groups for reward in group.training_rewards]
    rewards_by_taskset: dict[str, list[float]] = defaultdict(list)
    for group in groups:
        rewards_by_taskset[group.taskset_id].extend(rollout.reward for rollout in group.rollouts)
        latest_examples[group.taskset_id] = replace(group, rollouts=group.rollouts[-1:])
    sampled_tokens = sum(int(rollout.sequence.loss_mask.sum()) for rollout in rollouts)
    transcript_tokens = int(training_batch.attention_mask.sum())
    unmasked_tokens = int(training_batch.loss_mask.sum())
    stale_tokens = sampled_tokens - unmasked_tokens
    age_counts: Counter[int] = Counter()
    unmasked_age_counts: Counter[int] = Counter()
    for index, rollout in enumerate(rollouts):
        sequence = rollout.sequence
        token_ages = np.minimum(consumption_weight_version - sequence.weight_versions, max_staleness + 1)  # (L,)
        unmasked = training_batch.loss_mask[index, : len(sequence.token_ids)]  # (L,) unpadded transcript.
        for mask, counts in (
            (sequence.loss_mask, age_counts),
            (sequence.loss_mask & unmasked, unmasked_age_counts),
        ):
            ages, token_counts = np.unique(token_ages[mask], return_counts=True)  # (A,) observed age buckets.
            counts.update({int(age): int(count) for age, count in zip(ages, token_counts, strict=True)})
    age_metrics = {
        f"batcher/weight_age{suffix}_transcript_token_fraction/{age if age <= max_staleness else 'older'}": count
        / transcript_tokens
        for age, total in age_counts.items()
        for suffix, count in (
            ("", total),
            ("_masked", total - unmasked_age_counts[age]),
            ("_unmasked", unmasked_age_counts[age]),
        )
    }
    return RolloutBatch(
        training_batch=training_batch,
        rollout_metrics={
            "rollout/reward_mean": mean(rewards),
            **{f"rollout/reward_mean/{name}": mean(values) for name, values in rewards_by_taskset.items()},
            "rollout/reward_std": pstdev(rewards),
            "rollout/reward_min": min(rewards),
            "rollout/reward_max": max(rewards),
            "rollout/adjusted_reward_mean": mean(training_rewards),
            "rollout/adjusted_reward_std": pstdev(training_rewards),
            "rollout/adjusted_reward_min": min(training_rewards),
            "rollout/adjusted_reward_max": max(training_rewards),
            "rollout/degenerate_group_fraction": sum(len(group.rollouts) * group.is_degenerate for group in groups)
            / len(rollouts),
            "rollout/turns_mean": mean(rollout.num_turns for rollout in rollouts),
            "rollout/input_tokens_mean": mean(rollout.num_input_tokens for rollout in rollouts),
            "rollout/output_tokens_mean": mean(rollout.num_output_tokens for rollout in rollouts),
            "rollout/total_tokens_mean": transcript_tokens / len(rollouts),
            "batcher/token_utilization_fraction": transcript_tokens / training_batch.attention_mask.size,
            "batcher/padded_sequence_tokens": float(training_batch.attention_mask.shape[1]),
            "batcher/skipped_groups": skipped_groups,
            "batcher/stale_tokens": stale_tokens,
            "batcher/context_transcript_token_fraction": (transcript_tokens - sampled_tokens) / transcript_tokens,
            "batcher/stale_transcript_token_fraction": stale_tokens / transcript_tokens,
            "batcher/masked_transcript_token_fraction": (transcript_tokens - unmasked_tokens) / transcript_tokens,
            "batcher/unmasked_transcript_token_fraction": unmasked_tokens / transcript_tokens,
            **age_metrics,
            "actor/rollout_seconds_mean": mean(rollout.rollout_seconds for rollout in rollouts),
        },
        stop_counts=Counter(rollout.stop_condition for rollout in rollouts),
        num_output_tokens=sum(rollout.num_output_tokens for rollout in rollouts),
        examples=format_rollout_examples(list(latest_examples.values())),
        timing_metrics=timing_metrics,
        taskset_counts={name: len(values) for name, values in rewards_by_taskset.items()},
    )


# Use a value alias so Ray can serialize script-defined actors without importing __main__.Metrics.
Metrics = dict[str, jax.Array]  # Scalar leaves.
POLICY_DIAGNOSTICS = (
    "train/approx_kl",
    "train/clip_fraction",
    "train/rollout_log_prob_error_mean",
    "train/rollout_importance_weight_mean",
    "train/rollout_importance_weight_clipped_fraction",
)


def policy_diagnostics(loss_config: LossConfig) -> tuple[str, ...]:
    if isinstance(loss_config, CISPOConfig):
        return cispo.DIAGNOSTICS
    if isinstance(loss_config, StableReinforceConfig):
        return stable_reinforce.DIAGNOSTICS
    return POLICY_DIAGNOSTICS


def token_distribution(
    state: TrainState,
    params: TrainParameters,  # HF parameter shapes or (R, I), (O, R) factors.
    batch: GRPOBatch[jax.Array],  # (B, T), (B,), optional (B, T, K) leaves.
    temperature: float,
) -> jax.Array:  # (B, T-1, V)
    """(B, T-1, V) temperature-scaled learner log-probabilities."""
    if isinstance(state, LoRATrainState):
        logits = state.apply_fn(
            {"params": state.base_params},
            batch.token_ids[:, :-1],
            attention_mask=batch.attention_mask[:, :-1],
            lora=cast(LoRAParameters, params),
            lora_scale=state.lora_scale,
        ).logits  # (B, T-1, V)
    else:
        logits = state.apply_fn(
            {"params": cast(Parameters, params)},
            batch.token_ids[:, :-1],
            attention_mask=batch.attention_mask[:, :-1],
        ).logits  # (B, T-1, V)
    logits = logits.astype(jnp.float32) / temperature  # (B, T-1, V)
    return jax.nn.log_softmax(logits, axis=-1)  # (B, T-1, V); stable shifted arithmetic, as in vLLM.


def token_log_probs(
    state: TrainState,
    params: TrainParameters,  # HF parameter shapes or (R, I), (O, R) factors.
    batch: GRPOBatch[jax.Array],  # (B, T), (B,), optional (B, T, K) leaves.
    temperature: float,
) -> jax.Array:  # (B, T-1)
    """(B, T-1) log-probabilities for tokens at positions 1 through T-1."""
    log_probs = token_distribution(state, params, batch, temperature)  # (B, T-1, V)
    # Fuse selection's backward pass with softmax instead of scattering into a full FP32 vocabulary buffer.
    targets = jax.nn.one_hot(batch.token_ids[:, 1:], log_probs.shape[-1], dtype=jnp.bool_)  # (B, T-1, V)
    return jnp.where(targets, log_probs, 0).sum(axis=-1)  # (B, T-1)


@partial(
    jax.jit,
    static_argnames=("temperature",),
    in_shardings=(PartitionSpec(), PartitionSpec("batch")),
    out_shardings=PartitionSpec("batch"),
)
def compute_anchor_log_probs(
    state: TrainState,
    batch: GRPOBatch[jax.Array],  # (B, T), (B,) leaves.
    temperature: float,
) -> jax.Array:  # (B, T-1)
    """(B, T-1) detached learner scores for this batch's gradient computation.

    Skip scoring when no unmasked token has a nonzero advantage.
    """
    size, width = batch.token_ids.shape
    if size % jax.sharding.get_abstract_mesh().shape["batch"]:
        raise ValueError("learner device count must divide the rollout batch size")

    def evaluate() -> jax.Array:  # (B, T-1)
        return token_log_probs(state, state.params, batch, temperature)  # (B, T-1)

    # No policy diagnostics are logged for an entirely skipped rollout batch.
    def skip() -> jax.Array:  # (B, T-1)
        return jnp.zeros((size, width - 1), jnp.float32)  # (B, T-1), unused by skipped updates.

    scores = jax.lax.cond(
        jnp.any((batch.advantages != 0) & jnp.any(batch.loss_mask[:, 1:], axis=1)), evaluate, skip
    )  # (B, T-1)
    return jax.lax.stop_gradient(scores)  # (B, T-1)


def objective(
    log_probs: jax.Array,  # (B, T-1)
    batch: GRPOBatch[jax.Array],  # (B, T), (B,) leaves; old_log_probs retains the rollout scores.
    anchor_log_probs: jax.Array,  # (B, T-1) learner scores frozen before all batch updates.
    clip_coef: float,
    rollout_is_threshold: float = 2.0,
) -> tuple[jax.Array, Metrics]:  # (), scalar metric leaves
    """Compute PPO loss against frozen anchors with detached rollout correction.

    Upper-truncate anchor/rollout importance weights before applying them to the
    clipped surrogate. Average unmasked tokens per rollout, then all rollouts;
    fully masked rollouts contribute zero.
    """
    mask = batch.loss_mask[:, 1:]  # (B, T-1)
    anchor_log_probs = jax.lax.stop_gradient(anchor_log_probs)  # (B, T-1)

    def average(values: jax.Array) -> jax.Array:  # (B, T-1) -> ()
        return (jnp.where(mask, values, 0).sum(axis=1) / jnp.maximum(mask.sum(axis=1), 1)).mean()  # ()

    rollout_log_ratio = jnp.where(mask, anchor_log_probs - batch.old_log_probs[:, 1:], 0)  # (B, T-1)
    # verl's token-TIS: bound exponentiation, upper-truncate, and detach; no batch normalization or rejection.
    raw_weights = jnp.exp(jnp.clip(rollout_log_ratio, -20, 20))  # (B, T-1)
    weights = jax.lax.stop_gradient(jnp.where(mask, jnp.minimum(raw_weights, rollout_is_threshold), 0))  # (B, T-1)
    log_ratio = jnp.clip(jnp.where(mask, log_probs - anchor_log_probs, 0), -20, 20)  # (B, T-1), verl safety bound.
    ratio = jnp.exp(log_ratio)  # (B, T-1)
    advantages = batch.advantages[:, None]  # (B, 1)
    loss = -average(
        weights * jnp.minimum(ratio * advantages, jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef) * advantages)
    )  # ()
    return loss, {
        "train/policy_loss": loss,  # ()
        "train/approx_kl": average(jnp.expm1(log_ratio) - log_ratio),  # ()
        "train/clip_fraction": average(((ratio < 1 - clip_coef) | (ratio > 1 + clip_coef)).astype(jnp.float32)),  # ()
        "train/rollout_log_prob_error_mean": average(jnp.abs(rollout_log_ratio)),  # ()
        "train/rollout_importance_weight_mean": average(weights),  # ()
        "train/rollout_importance_weight_clipped_fraction": average(
            (raw_weights > rollout_is_threshold).astype(jnp.float32)
        ),  # ()
    }


def create_state(
    model: Qwen3LM,
    params: Parameters,  # HF parameter shapes.
    config: Config,
) -> TrainState:
    """Optimize FP32 full weights or LoRA factors on attention and MLP projections."""
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(**config.adamw_kwargs),
    )
    if config.finetuning.mode == "full":
        return TrainState.create(
            apply_fn=model.apply,
            params=jax.tree.map(lambda value: value.astype(jnp.float32), params),  # HF parameter shapes.
            tx=optimizer,
        ).replace(step=jnp.asarray(0, dtype=jnp.int32))  # ()
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
                (config.finetuning.rank, input_size),
                dtype=jnp.float32,
                minval=-(input_size**-0.5),
                maxval=input_size**-0.5,
            ),  # (R, I)
            b=jnp.zeros((output_size, config.finetuning.rank), dtype=jnp.float32),  # (O, R)
        )
    if not adapters:
        raise ValueError("No attention or MLP projection weights found for LoRA")
    return LoRATrainState.create(
        apply_fn=model.apply,
        params=adapters,
        base_params=params,
        lora_scale=config.finetuning.alpha / config.finetuning.rank,
        tx=optimizer,
    ).replace(step=jnp.asarray(0, dtype=jnp.int32))  # () Match the device counter returned by update.


@partial(
    jax.jit,
    static_argnames=("temperature", "loss_config"),
    in_shardings=(PartitionSpec(), PartitionSpec("batch"), PartitionSpec("batch"), PartitionSpec()),
    out_shardings=PartitionSpec(),
    donate_argnums=(3,),
)
def accumulate_gradients(
    state: TrainState,
    batch: GRPOBatch[jax.Array],  # (B, T), (B,), optional (B, T, K) leaves.
    anchor_log_probs: jax.Array,  # (B, T-1) frozen learner scores aligned with batch rows.
    gradient_sum: TrainParameters,  # Parameter shapes; the running sum for this outer iteration.
    temperature: float,
    loss_config: LossConfig = DEFAULT_CONFIG.loss,
) -> tuple[TrainParameters, Metrics]:
    """Add one full batch's gradients, donating the running sum and preserving state."""
    if batch.token_ids.shape[0] % jax.sharding.get_abstract_mesh().shape["batch"]:
        raise ValueError("learner device count must divide the rollout batch size")
    # Python logging runs on tracing, not on cached executions.
    logger.info(
        "JIT trace accumulate_gradients: batch_shape=%s temperature=%s loss=%s",
        batch.token_ids.shape,
        temperature,
        loss_config.kind,
    )

    def compute_gradients() -> tuple[TrainParameters, Metrics]:
        def loss_fn(params: TrainParameters) -> tuple[jax.Array, Metrics]:  # Parameter leaves -> (), scalars.
            if isinstance(loss_config, StableReinforceConfig):
                if batch.sampler_topk_ids is None or batch.sampler_topk_log_probs is None:
                    raise ValueError("StableReinforce requires sampling-time top-logprobs")
                distribution = token_distribution(state, params, batch, temperature)  # (B, T-1, V)
                selected = jnp.take_along_axis(distribution, batch.token_ids[:, 1:, None], axis=-1)[..., 0]  # (B, T-1)
                head = jnp.take_along_axis(distribution, batch.sampler_topk_ids[:, 1:], axis=-1)  # (B, T-1, K)
                return stable_reinforce.objective(
                    selected,
                    head,
                    batch.sampler_topk_log_probs[:, 1:],
                    batch.old_log_probs[:, 1:],
                    batch.advantages,
                    batch.loss_mask[:, 1:],
                    loss_config,
                )
            log_probs = token_log_probs(state, params, batch, temperature)  # (B, T-1)
            if isinstance(loss_config, CISPOConfig):
                return cispo.objective(
                    log_probs, batch.old_log_probs[:, 1:], batch.advantages, batch.loss_mask[:, 1:], loss_config
                )
            return objective(
                log_probs,
                batch,
                anchor_log_probs,
                loss_config.clip_coef,
                loss_config.rollout_is_threshold,
            )

        (_, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(state.params)  # Scalars; parameter shapes.
        gradients = jax.lax.with_sharding_constraint(gradients, PartitionSpec())  # Parameter shapes.
        return jax.tree.map(jnp.add, gradient_sum, gradients), metrics  # Parameter shapes; scalar metrics.

    def skip_gradients() -> tuple[TrainParameters, Metrics]:
        # Fixed JIT output structure; optimize excludes unevaluated policy diagnostics from logging.
        names = policy_diagnostics(loss_config)
        if isinstance(loss_config, StableReinforceConfig):
            names = ("train/loss", *names)
        return gradient_sum, {  # Parameter shapes; scalar metrics.
            "train/policy_loss": jnp.asarray(0.0, dtype=jnp.float32),  # ()
            **{name: jnp.asarray(0.0, dtype=jnp.float32) for name in names},  # () Not evaluated.
        }

    kl_active = isinstance(loss_config, StableReinforceConfig) and loss_config.kl_coeff > 0
    return jax.lax.cond(
        jnp.any(((batch.advantages != 0) | kl_active) & jnp.any(batch.loss_mask[:, 1:], axis=1)),
        compute_gradients,
        skip_gradients,
    )


@partial(
    jax.jit,
    static_argnames=("grad_accum_steps", "has_signal"),
    in_shardings=(PartitionSpec(), PartitionSpec()),
    out_shardings=PartitionSpec(),
    donate_argnums=(0,),
)
def update(
    state: TrainState,
    gradient_sum: TrainParameters,  # Parameter shapes; summed gradients from the inner loop.
    grad_accum_steps: int,
    has_signal: bool,
) -> tuple[TrainState, Metrics]:
    """Average the accumulated gradients, then clip and apply one optimizer update."""
    if not has_signal:
        return state, {
            "train/skipped_update_fraction": jnp.asarray(1.0, jnp.float32),  # ()
        }
    gradients = jax.tree.map(lambda value: value / grad_accum_steps, gradient_sum)  # Parameter shapes.
    return state.apply_gradients(grads=gradients), {
        "train/gradient_norm": optax.global_norm(gradients),  # () Before clipping.
        "train/skipped_update_fraction": jnp.asarray(0.0, jnp.float32),  # ()
    }


def score_group(
    task: Task,
    rollouts: list[Rollout],
    *,
    taskset_id: str,
    length_penalty: LengthPenaltyConfig = DEFAULT_CONFIG.length_penalty,
) -> RolloutGroup:
    """Mean-center adjusted rewards without rescaling the length cost."""
    if len(rollouts) < 2:
        raise ValueError("A rollout group needs at least two rollouts")
    group = RolloutGroup(
        task_id=f"{task.name}:{task.id}",
        rollouts=rollouts,
        length_penalty=length_penalty,
        taskset_id=taskset_id,
    )
    rewards = group.training_rewards
    reward_mean = mean(rewards)
    return replace(
        group,
        rollouts=[
            replace(rollout, advantage=reward - reward_mean) for rollout, reward in zip(rollouts, rewards, strict=True)
        ],
    )


@asynccontextmanager
async def create_agent(config: Config = DEFAULT_CONFIG) -> AsyncGenerator[Agent]:
    """Yield an agent configured from vLLM's context limit and close its client on exit."""
    tokenizer = await asyncio.to_thread(AutoTokenizer.from_pretrained, config.model)
    if isinstance(config.loss, StableReinforceConfig) and config.loss.top_k >= tokenizer.vocab_size:
        raise ValueError("loss.top_k must be smaller than the model vocabulary")
    async with AsyncOpenAI(
        base_url=f"{SERVER_URL}/v1", api_key="EMPTY", timeout=config.rollout_timeout, max_retries=3
    ) as client:
        models = await client.models.list()
        model = next(model for model in models.data if model.id == config.model)
        yield Agent(
            client=client,
            tokenizer=tokenizer,
            model=LORA_NAME if config.finetuning.mode == "lora" else config.model,
            context_length=TypeAdapter(int).validate_python(
                (model.model_extra or {}).get("max_model_len"), strict=True
            ),
            max_tokens=config.max_tokens,
            max_turns=config.max_turns,
            temperature=config.temperature,
            rollout_timeout=config.rollout_timeout,
            environment_timeout=config.environment_timeout,
            sampler_top_k=config.loss.top_k if isinstance(config.loss, StableReinforceConfig) else 0,
        )


@contextmanager
def serve_vllm(
    model: str = MODEL,
    lora_rank: int | None = LoRAConfig().rank,
    *,
    config: VLLMConfig = DEFAULT_CONFIG.vllm,
    sampler_top_k: int = 0,
) -> Generator[subprocess.Popen[bytes]]:
    """Manage vLLM's process lifetime; lora_rank=None enables full-weight transfer."""
    weight_args = ["--weight-transfer-config", '{"backend":"nccl"}']
    if lora_rank is not None:
        max_lora_rank = next((rank for rank in get_args(MaxLoRARanks) if rank >= lora_rank), None)
        if max_lora_rank is None:
            raise ValueError(f"vLLM supports LoRA ranks up to {max(get_args(MaxLoRARanks))}")
        weight_args = ["--enable-lora", "--max-lora-rank", str(max_lora_rank), "--lora-dtype", "bfloat16"]
    serving_args: list[str] = []
    for name, value in asdict(config).items():
        if value is None:
            continue
        flag = name.replace("_", "-")
        if isinstance(value, bool):
            serving_args.append(f"--{'' if value else 'no-'}{flag}")
        else:
            serving_args.extend([f"--{flag}", str(value)])
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
            "--enable-scale-out",
            "--dtype",
            "bfloat16",
            "--hf-overrides",
            '{"head_dtype":"float32"}',
            "--generation-config",
            "vllm",
            "--logprobs-mode",
            "processed_logprobs",
            *(["--max-logprobs", str(sampler_top_k + 1)] if sampler_top_k else []),
            *serving_args,
            *weight_args,
        ],
        env={
            **os.environ,
            "VLLM_SERVER_DEV_MODE": "1",
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
            "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "30",
        },
    ) as server:
        try:
            yield server
        finally:
            logger.info("Stopping vLLM server: pid=%s", server.pid)
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("vLLM did not stop within 30s; killing pid=%s", server.pid)
                server.kill()
    logger.info("vLLM server exited")


def wait_for_server(server: subprocess.Popen[bytes], timeout: float = 600) -> None:
    """Block until vLLM is healthy, raising if it exits or the timeout expires."""
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
    """Maintain a bounded rollout pool for sequential learner batch requests."""

    def __init__(self, actor: ActorHandle, config: Config = DEFAULT_CONFIG) -> None:
        self.actor = actor
        self.config = config
        self._pending_groups: list[asyncio.Task[CollectedGroup]] = []
        self._latest_examples: dict[str, RolloutGroup] = {}
        self._completed_groups: deque[RolloutGroup] = deque()

    def prepare_batch(
        self, groups: list[RolloutGroup], context_length: int, *, consumption_weight_version: int = 0
    ) -> GRPOBatch[np.ndarray]:
        """Flatten rollout groups in order and right-pad their transcripts into NumPy arrays.

        B is the total rollout count; T is the longest transcript length rounded up
        to a multiple of config.padding_multiple, capped at min(config.max_tokens, context_length).
        Padding is zero/False; transcripts and rollout advantages are preserved.

        Returns:
            GRPOBatch with token_ids: (B, T) int32 complete transcripts;
            attention_mask: (B, T) bool non-padding positions;
            loss_mask: (B, T) bool sampled-token positions within max_staleness;
            old_log_probs: (B, T) float32 sampling-policy log-probabilities;
            advantages: (B,) float32 per-rollout advantages.
            sampler_topk_ids: (B, T, K) int32 alternatives, or None outside StableReinforce;
            sampler_topk_log_probs: (B, T, K) float32 sampler scores, or None outside StableReinforce.

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
        top_k = self.config.loss.top_k if isinstance(self.config.loss, StableReinforceConfig) else 0
        # Host copies avoid compiling JAX padding and reductions for every transcript length.
        batch = GRPOBatch(
            token_ids=np.zeros(shape, dtype=np.int32),  # (B, T)
            attention_mask=np.zeros(shape, dtype=np.bool_),  # (B, T)
            loss_mask=np.zeros(shape, dtype=np.bool_),  # (B, T)
            old_log_probs=np.zeros(shape, dtype=np.float32),  # (B, T)
            advantages=np.asarray([rollout.advantage for rollout in rollouts], dtype=np.float32),  # (B,)
            sampler_topk_ids=np.zeros((*shape, top_k), dtype=np.int32) if top_k else None,  # (B, T, K)
            sampler_topk_log_probs=np.zeros((*shape, top_k), dtype=np.float32) if top_k else None,  # (B, T, K)
        )
        for index, rollout in enumerate(rollouts):
            sequence: TrainingSequence = rollout.sequence
            length = len(sequence.token_ids)
            mask = sequence.loss_mask  # (L,) unpadded transcript length.
            if length < 2 or mask[0] or not mask[1:].any():
                raise ValueError("Each rollout needs context followed by at least one sampled token")
            batch.token_ids[index, :length] = sequence.token_ids  # (L,)
            batch.loss_mask[index, :length] = mask & (
                sequence.weight_versions >= consumption_weight_version - self.config.max_staleness
            )  # (L,)
            batch.old_log_probs[index, :length] = sequence.old_log_probs  # (L,)
            batch.attention_mask[index, :length] = True  # (L,)
            if batch.sampler_topk_ids is not None and batch.sampler_topk_log_probs is not None:
                if (
                    sequence.sampler_topk_ids is None
                    or sequence.sampler_topk_log_probs is None
                    or sequence.sampler_topk_ids.shape != (length, top_k)
                    or sequence.sampler_topk_log_probs.shape != (length, top_k)
                ):
                    raise ValueError("StableReinforce requires matching sampling-time top-logprobs in every rollout")
                batch.sampler_topk_ids[index, :length] = sequence.sampler_topk_ids  # (L, K)
                batch.sampler_topk_log_probs[index, :length] = sequence.sampler_topk_log_probs  # (L, K)
        return batch  # (B, T), (B,), optional (B, T, K) leaves.

    async def _collect_group(self, group_size: int) -> CollectedGroup:
        """Fill one pool slot, replacing degenerate groups when configured."""
        skipped_groups = 0
        while True:
            pending = self.actor.fetch_next_group.remote(group_size=group_size)
            try:
                group = await pending
            except asyncio.CancelledError:
                ray.cancel(pending, recursive=True)
                raise
            if not self.config.drop_degenerate_groups or not group.is_degenerate:
                return CollectedGroup(group=group, skipped_groups=skipped_groups)
            skipped_groups += 1

    def _fill_pending_groups(self, group_size: int, target_pending: int) -> None:
        while len(self._pending_groups) < target_pending:
            # These tasks remain live between learner requests.
            task = asyncio.create_task(self._collect_group(group_size=group_size))
            self._pending_groups.append(task)

    async def _fetch_group(self, *, group_size: int, pool_size: int, groups_until_drain: int) -> CollectedGroup:
        """Pop the next ready group and replenish within the drain limit."""
        self._fill_pending_groups(group_size=group_size, target_pending=min(pool_size, groups_until_drain))
        ready, _ = await asyncio.wait(self._pending_groups, return_when=asyncio.FIRST_COMPLETED)
        task = next(task for task in self._pending_groups if task in ready)
        group = await task
        self._pending_groups.remove(task)
        groups_until_drain -= 1
        self._fill_pending_groups(group_size=group_size, target_pending=min(pool_size, groups_until_drain))
        return group

    def get_pool_state(self) -> list[dict[str, Any]]:
        """Serialize unconsumed rollouts after pending generation drains at a checkpoint."""
        result: list[dict[str, Any]] = []
        for group in self._completed_groups:
            data = asdict(group)
            for rollout in data["rollouts"]:
                rollout["sequence"] = {
                    name: value.tolist() for name, value in rollout["sequence"].items() if value is not None
                }
            result.append(data)
        return result

    def set_pool_state(self, groups: list[dict[str, Any]]) -> None:
        """Restore already-scored rollouts without resampling or recomputing advantages."""
        restored: deque[RolloutGroup] = deque()
        for group in groups:
            rollouts: list[Rollout] = []
            for rollout in group["rollouts"]:
                values = rollout["sequence"]
                sequence = TrainingSequence(
                    token_ids=np.asarray(values["token_ids"], dtype=np.int32),  # (T,)
                    loss_mask=np.asarray(values["loss_mask"], dtype=np.bool_),  # (T,)
                    old_log_probs=np.asarray(values["old_log_probs"], dtype=np.float32),  # (T,)
                    weight_versions=np.asarray(values["weight_versions"], dtype=np.int32),  # (T,)
                    sampler_topk_ids=(
                        np.asarray(values["sampler_topk_ids"], dtype=np.int32)
                        if values.get("sampler_topk_ids") is not None
                        else None
                    ),  # (T, K)
                    sampler_topk_log_probs=(
                        np.asarray(values["sampler_topk_log_probs"], dtype=np.float32)
                        if values.get("sampler_topk_log_probs") is not None
                        else None
                    ),  # (T, K)
                )
                rollouts.append(Rollout(**(rollout | {"sequence": sequence})))
            restored.append(
                RolloutGroup(
                    task_id=group["task_id"],
                    taskset_id=group["taskset_id"],
                    rollouts=rollouts,
                    length_penalty=LengthPenaltyConfig(**group["length_penalty"]),
                    source_is_degenerate=group["source_is_degenerate"],
                )
            )
        self._completed_groups = restored

    async def cancel_pending_groups(self) -> None:
        """Cancel prefetched groups and their dependent rollout tasks."""
        for task in self._pending_groups:
            task.cancel()
        await asyncio.gather(*self._pending_groups, return_exceptions=True)
        self._pending_groups.clear()

    async def fetch_next_batch(
        self,
        batch_size: int,
        group_size: int,
        context_length: int,
        *,
        iteration: int = 1,
        consumption_weight_version: int = 0,
    ) -> RolloutBatch:
        """Consume exactly batch_size rollouts, retaining unused parts of scored groups."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        started = time.monotonic()
        batches_until_drain = (-iteration) % (self.config.checkpoint_interval * self.config.grad_accum_steps)
        if self.config.total_iterations is not None:
            batches_until_drain = min(
                batches_until_drain, max(0, self.config.total_iterations * self.config.grad_accum_steps - iteration)
            )
        if self.config.prefetch_batches == 0:
            batches_until_drain = 0
        buffered = sum(len(group.rollouts) for group in self._completed_groups)
        rollouts_until_drain = max(0, (batches_until_drain + 1) * batch_size - buffered)
        groups_until_drain = (rollouts_until_drain + group_size - 1) // group_size
        pool_rollouts = max(batch_size, self.config.prefetch_batches * self.config.batch_size)
        pool_size = (pool_rollouts + group_size - 1) // group_size
        try:
            groups: list[RolloutGroup] = []
            skipped_groups = 0
            remaining = batch_size
            self._fill_pending_groups(group_size=group_size, target_pending=min(pool_size, groups_until_drain))
            while remaining:
                if not self._completed_groups:
                    item = await self._fetch_group(
                        group_size=group_size,
                        pool_size=pool_size,
                        groups_until_drain=groups_until_drain,
                    )
                    groups_until_drain -= 1
                    skipped_groups += item.skipped_groups
                    self._completed_groups.append(item.group)
                group = self._completed_groups.popleft()
                taken = min(remaining, len(group.rollouts))
                groups.append(replace(group, rollouts=group.rollouts[:taken], source_is_degenerate=group.is_degenerate))
                if taken < len(group.rollouts):
                    self._completed_groups.appendleft(
                        replace(group, rollouts=group.rollouts[taken:], source_is_degenerate=group.is_degenerate)
                    )
                remaining -= taken
            fetched = time.monotonic()
            with span("batcher.prepare_batch"):
                training_batch = await asyncio.to_thread(
                    self.prepare_batch,
                    groups,
                    context_length=context_length,
                    consumption_weight_version=consumption_weight_version,
                )  # (B, T), (B,) leaves.
            prepared = time.monotonic()
            return build_rollout_batch(
                groups=groups,
                training_batch=training_batch,
                consumption_weight_version=consumption_weight_version,
                max_staleness=self.config.max_staleness,
                latest_examples=self._latest_examples,
                skipped_groups=skipped_groups,
                timing_metrics={
                    "batcher/fetch_rollout_seconds": fetched - started,
                    "batcher/prepare_batch_seconds": prepared - fetched,
                },
            )
        except BaseException:
            await self.cancel_pending_groups()
            raise


@ray.remote(num_gpus=1)
class Learner:
    def __init__(self, batcher: ActorHandle, actor: ActorHandle, config: Config = DEFAULT_CONFIG) -> None:
        self.config = config
        self.batcher = batcher
        self.actor = actor
        self.weight_version = 0
        self.shuffle_key = jax.random.key(config.seed)  # () typed base PRNG key.

    def start(self) -> None:
        """Initialize or restore training, then run updates, weight publication, and periodic checkpoints."""
        self.mesh = Mesh(devices=jax.devices("gpu"), axis_names=("batch",))
        with (
            jax.set_mesh(self.mesh),
            MLflowLogger(experiment="grpo_v2", run_name=self.config.run_id) as writer,
            closing(checkpoint_manager(self.config.ckpt_dir)) as manager,
            ExitStack() as resources,
        ):
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
                ray.get(self.batcher.set_pool_state.remote(progress.rollout_pool))
                logger.info("Resumed run %s at iteration %s", self.config.run_id, iteration)
            writer.log_text(yaml.safe_dump(asdict(self.config)), f"config/{iteration}.yaml")
            logger.info(
                "Learner loaded %s trainable tensors in %s mode",
                len(self.state.params),
                self.config.finetuning.mode,
            )
            ray.get(self.actor.ready.remote())
            if self.config.finetuning.mode == "full":
                publisher = full_weight_publisher(
                    lambda: self.state.params,
                    server_url=SERVER_URL,
                    device_index=self.mesh.devices.flat[0].local_hardware_id,
                    pause_mode=self.config.weight_update_pause_mode,
                )
            else:
                publisher = lora_weight_publisher(
                    lambda: self.state.params,
                    server_url=SERVER_URL,
                    model=self.config.model,
                    adapter_name=LORA_NAME,
                    lora_rank=self.config.finetuning.rank,
                    lora_alpha=self.config.finetuning.alpha,
                    pause_mode=self.config.weight_update_pause_mode,
                )
            self._publish_weights = resources.enter_context(publisher)
            resources.callback(self.cancel_pending_groups)
            self.weight_version = progress.weight_version if progress is not None else -1
            self.publish_weights()
            self.weight_version += 1
            writer.log_metrics(
                {
                    "model/parameters_billions": sum(
                        p.size
                        for p in jax.tree.leaves(
                            self.state.base_params if isinstance(self.state, LoRATrainState) else self.state.params
                        )
                    )
                    / 1_000_000_000,
                    "model/trainable_parameters_billions": sum(p.size for p in jax.tree.leaves(self.state.params))
                    / 1_000_000_000,
                    "learner/setup_seconds": time.monotonic() - start,
                },
                step=iteration,
            )
            self.run_iterations(writer, manager, iteration=iteration, total_rollouts=total_rollouts)

    def run_iterations(
        self,
        writer: MLflowLogger,
        manager: ocp.CheckpointManager,
        *,
        iteration: int,
        total_rollouts: int,
    ) -> None:
        """Run accumulation, publication, and logging from the restored progress."""
        published_step = int(self.state.step)
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
                summaries: list[BatchSummary] = []
                update_history: dict[str, list[float]] = defaultdict(list)
                timing_metrics: Counter[str] = Counter()
                stop_counts: Counter[StopCondition] = Counter()
                num_rollouts = 0
                num_output_tokens = 0
                fetch_seconds = 0.0
                update_seconds = 0.0
                previous = started
                gradient_sum = jax.tree.map(jnp.zeros_like, self.state.params)  # Parameter shapes.
                has_signal = False
                for accumulation_step in range(self.config.grad_accum_steps):
                    batch_iteration = (iteration - 1) * self.config.grad_accum_steps + accumulation_step + 1
                    with span("learner.fetch_batch"):
                        batch = self.fetch_next_batch(iteration=batch_iteration)
                    fetched = time.monotonic()
                    with span("learner.optimize"):
                        result = self.optimize(batch.training_batch, gradient_sum)
                        gradient_sum = result.gradient_sum  # Parameter shapes.
                        has_signal |= result.has_signal
                        for name, value in result.metrics.items():
                            update_history[name].append(value)
                    optimized = time.monotonic()
                    fetch_seconds += fetched - previous
                    update_seconds += optimized - fetched
                    previous = optimized
                    batch_rollouts = batch.training_batch.token_ids.shape[0]
                    num_rollouts += batch_rollouts
                    num_output_tokens += batch.num_output_tokens
                    summaries.append(
                        BatchSummary(
                            metrics=batch.rollout_metrics,
                            num_rollouts=batch_rollouts,
                            taskset_counts=batch.taskset_counts,
                        )
                    )
                    timing_metrics.update(batch.timing_metrics)
                    stop_counts.update(batch.stop_counts)
                with span("learner.apply_update"):
                    optimizer_metrics = self.apply_update(gradient_sum, has_signal=has_signal)
                applied = time.monotonic()
                update_seconds += applied - optimized
                total_updates = int(self.state.step)
                sync_due = total_updates % self.config.parameter_sync_steps == 0
                publish = sync_due and total_updates != published_step
                if publish:
                    self.publish_weights()
                    self.weight_version += 1
                    published_step = total_updates
                now = time.monotonic()
                total_rollouts += num_rollouts
                seconds = max(now - started, 1e-9)
                scalars = {
                    **{name: mean(values) for name, values in update_history.items()},
                    **optimizer_metrics,
                    **timing_metrics,
                    **summarize_iteration(summaries),
                    **{
                        f"rollout/stop_fraction/{condition}": stop_counts.get(condition, 0) / num_rollouts
                        for condition in get_args(StopCondition)
                    },
                    "train/iterations_total": iteration,
                    "train/updates_total": total_updates,
                    "train/learning_rate": self.config.adamw_kwargs["learning_rate"],
                    "train/rollouts_total": total_rollouts,
                    "learner/iterations_per_second": 1 / seconds,
                    "learner/rollouts_per_second": num_rollouts / seconds,
                    "learner/output_tokens_per_second": num_output_tokens / seconds,
                    "learner/batch_fetch_seconds": fetch_seconds,
                    "learner/update_seconds": update_seconds,
                    "learner/iteration_seconds": now - started,
                }
                if publish:
                    scalars["learner/weight_transfer_seconds"] = now - applied
                logger.info(
                    "iteration=%s reward=%.3f degenerate_group_fraction=%.3f skipped_groups=%d "
                    "stale_tokens=%d batch=%.2fs loss=%.4f update=%.2fs%s",
                    iteration,
                    scalars["rollout/reward_mean"],
                    scalars["rollout/degenerate_group_fraction"],
                    scalars["batcher/skipped_groups"],
                    scalars["batcher/stale_tokens"],
                    scalars["learner/batch_fetch_seconds"],
                    scalars["train/policy_loss"],
                    scalars["learner/update_seconds"],
                    f" weights={scalars['learner/weight_transfer_seconds']:.2f}s" if publish else "",
                )
                if iteration % self.config.example_log_interval == 0:
                    with span("learner.log_examples"):
                        examples = f"Rollout examples | step {iteration}\n\n{batch.examples}"
                        writer.log_text(examples, f"examples/{iteration}.txt")
                if checkpoint_due:
                    with span("learner.checkpoint"):
                        scalars["checkpoint/save_blocking_seconds"] = save_checkpoint(
                            manager,
                            TrainingProgress(
                                train_state=self.state,
                                shuffle_key=self.shuffle_key,  # ()
                                iteration=iteration,
                                total_rollouts=total_rollouts,
                                sampling_state=ray.get(self.actor.get_sampling_state.remote()),
                                rollout_pool=ray.get(self.batcher.get_pool_state.remote()),
                                weight_version=self.weight_version,
                            ),
                            self.config,
                        )
                with span("learner.log_metrics"):
                    writer.log_metrics(scalars, step=iteration)

    def cancel_pending_groups(self) -> None:
        """Drain the batcher's rollout cancellation before shutting down training."""
        ray.get(self.batcher.cancel_pending_groups.remote())

    def fetch_next_batch(self, *, iteration: int) -> RolloutBatch:
        """Fetch one microbatch assembled from the batcher's pending rollout groups."""
        return ray.get(
            self.batcher.fetch_next_batch.remote(
                batch_size=self.config.microbatch_size,
                group_size=self.config.group_size,
                context_length=self.model.config.max_position_embeddings,
                iteration=iteration,
                consumption_weight_version=self.weight_version,
            )
        )

    def publish_weights(self) -> None:
        self._publish_weights(self.weight_version + 1)

    def optimize(
        self,
        batch: GRPOBatch[np.ndarray],  # (B, T), (B,), optional (B, T, K) leaves.
        gradient_sum: TrainParameters,  # Parameter shapes.
    ) -> BatchGradientResult:
        """Accumulate one batch without changing parameters or optimizer state."""
        if batch.token_ids.shape[0] % self.mesh.size:
            raise ValueError("learner device count must divide the rollout batch size")
        batched = NamedSharding(self.mesh, PartitionSpec("batch"))
        device_batch: GRPOBatch[jax.Array] = jax.device_put(batch, batched)  # (B, T), (B,), optional (B, T, K) leaves.
        loss_config = self.config.loss
        anchors = (
            compute_anchor_log_probs(self.state, device_batch, self.config.temperature)
            if isinstance(loss_config, GRPOLossConfig)
            else jnp.zeros_like(device_batch.old_log_probs[:, 1:])
        )  # (B, T-1); only GRPO needs an anchor forward pass.
        gradient_sum, metrics = accumulate_gradients(
            self.state,
            device_batch,
            anchors,
            gradient_sum,
            self.config.temperature,
            loss_config,
        )  # Parameter shapes; scalar metrics.
        scalars = {name: float(value) for name, value in metrics.items()}
        kl_active = isinstance(loss_config, StableReinforceConfig) and loss_config.kl_coeff > 0
        has_signal = bool(np.any(((batch.advantages != 0) | kl_active) & np.any(batch.loss_mask[:, 1:], axis=1)))
        if not has_signal:
            for name in policy_diagnostics(loss_config):
                del scalars[name]
        return BatchGradientResult(gradient_sum=gradient_sum, metrics=scalars, has_signal=has_signal)

    def apply_update(self, gradient_sum: TrainParameters, *, has_signal: bool) -> dict[str, float]:  # Parameter shapes.
        self.state, metrics = update(self.state, gradient_sum, self.config.grad_accum_steps, has_signal)
        return {name: float(value) for name, value in metrics.items()}


class SampledTask(NamedTuple):
    taskset_id: str
    task: Task


def stream_tasks(
    tasksets: TasksetConfigs,
    rng: random.Random,
    timeout: float,
    *,
    taskset_scores: dict[str, float] | None = None,
) -> Generator[SampledTask]:
    """Discover active tasksets, then select groups by smooth weighted round robin."""
    streams: dict[str, Generator[Task]] = {}
    weights: dict[str, float] = {}
    scores = {} if taskset_scores is None else taskset_scores
    max_weight = max(entry.weight for entry in tasksets.values())
    # Keep sampling reproducible when YAML serialization reorders mapping keys.
    for name, entry in sorted(tasksets.items()):
        if entry.weight == 0:
            continue
        taskset = entry.taskset
        if isinstance(taskset, ReasoningGymConfig):
            streams[name] = stream_reasoning_gym_tasks(taskset, rng)
        elif isinstance(taskset, AppWorldConfig):
            streams[name] = stream_appworld_tasks(taskset, rng, timeout)
        elif isinstance(taskset, CodeGymConfig):
            streams[name] = stream_codegym_tasks(taskset, rng)
        elif isinstance(taskset, CrafterConfig):
            streams[name] = stream_crafter_tasks(taskset, rng)
        elif isinstance(taskset, WebShopConfig):
            streams[name] = stream_webshop_tasks(taskset, rng, timeout)
        else:
            streams[name] = stream_alfworld_tasks(taskset, rng, timeout)
        weights[name] = entry.weight / max_weight
    total_weight = sum(weights.values())
    weights = {name: weight / total_weight for name, weight in weights.items()}

    def stream() -> Generator[SampledTask]:
        while True:
            for name, weight in weights.items():
                scores[name] = scores.get(name, 0.0) + weight
            name = max(weights, key=scores.__getitem__)
            scores[name] -= 1.0
            yield SampledTask(taskset_id=name, task=next(streams[name]))

    return stream()


# Restrict JAX before import/backend initialization; vLLM still uses the assigned CUDA device.
@ray.remote(num_gpus=1, runtime_env={"env_vars": {"JAX_PLATFORMS": "cpu"}})
class Actor:
    def __init__(self, config: Config = DEFAULT_CONFIG) -> None:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.config = config
        self.rng = random.Random(config.seed)
        self.taskset_scores: dict[str, float] = {}
        self._server = ExitStack()
        self._agent_resources = AsyncExitStack()
        self._agent: Agent | None = None
        self._agent_lock = asyncio.Lock()
        self._runs: set[asyncio.Task[Rollout]] = set()
        self._closed = False
        self.tasks = stream_tasks(
            config.tasksets, self.rng, config.environment_timeout, taskset_scores=self.taskset_scores
        )
        logger.info("Starting actor and vLLM on its assigned GPU")
        with self._server as resources:
            server = resources.enter_context(
                serve_vllm(
                    model=config.model,
                    lora_rank=config.finetuning.rank if config.finetuning.mode == "lora" else None,
                    config=config.vllm,
                    **({"sampler_top_k": config.loss.top_k} if isinstance(config.loss, StableReinforceConfig) else {}),
                )
            )
            wait_for_server(server)
            self._server = resources.pop_all()
        logger.info(
            "Actor ready for rollout requests: tasksets=%s",
            [(name, entry.weight) for name, entry in config.tasksets.items()],
        )

    def ready(self) -> None:
        """Awaiting this method waits for construction, including vLLM readiness."""

    async def _get_agent(self) -> Agent:
        """Create the shared agent once, serializing initialization with actor shutdown."""
        async with self._agent_lock:
            if self._closed:
                raise RuntimeError("Actor is closed")
            if self._agent is None:
                self._agent = await self._agent_resources.enter_async_context(create_agent(self.config))
            return self._agent

    async def aclose(self) -> None:
        """Cancel and drain active rollouts before closing the agent and vLLM server."""
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
        """Return task RNG and round-robin scores as a JSON-compatible checkpoint snapshot."""
        version, state, gauss_next = self.rng.getstate()
        return {
            "version": version,
            "state": list(state),
            "gauss_next": gauss_next,
            "taskset_scores": self.taskset_scores.copy(),
        }

    def set_sampling_state(self, state: dict[str, Any]) -> None:
        self.rng.setstate((state["version"], tuple(state["state"]), state["gauss_next"]))
        self.taskset_scores.clear()
        self.taskset_scores.update(state.get("taskset_scores", {}))

    async def fetch_next_group(self, group_size: int) -> RolloutGroup:
        sampled = next(self.tasks)
        return await self.collect_group(sampled.task, group_size=group_size, taskset_id=sampled.taskset_id)

    async def collect_group(self, task: Task, group_size: int, *, taskset_id: str) -> RolloutGroup:
        """Collect and score a fixed-size group of concurrent rollouts."""
        if group_size < 2:
            raise ValueError("group_size must be at least 2")
        span_data = {"group_id": uuid4().hex, "task_id": str(task.id), "group_size": str(group_size)}
        started = time.monotonic()
        try:
            with span("actor.collect_group", {**span_data, "run_id": self.config.run_id or "unspecified"}):
                agent = await self._get_agent()

                async def run_rollout(index: int) -> Rollout:
                    with span("actor.rollout", {**span_data, "rollout_index": index}) as rollout_span:
                        deadline = asyncio.get_running_loop().time() + self.config.rollout_timeout
                        for attempt in range(3):
                            try:
                                async with AsyncExitStack() as resources, asyncio.timeout_at(deadline):
                                    env = await resources.enter_async_context(task.init_env())
                                    result = await agent.run(env=env)
                                break
                            except (ActorDiedError, httpx.NetworkError, httpx.RemoteProtocolError) as error:
                                if attempt == 2 or (isinstance(error, ActorDiedError) and error.actor_init_failed):
                                    raise
                                logger.warning(
                                    "Rollout infrastructure failure (%s); restarting episode: task=%s rollout=%d retry=%d/2",
                                    type(error).__name__,
                                    task.id,
                                    index,
                                    attempt + 1,
                                    exc_info=True,
                                )
                        rollout_span.set_attribute("stop_condition", result.stop_condition)
                        return result

                async with asyncio.TaskGroup() as tg:
                    runs = [tg.create_task(run_rollout(index)) for index in range(group_size)]
                    self._runs.update(runs)
                    for run in runs:
                        run.add_done_callback(self._runs.discard)
                with span("actor.score_group"):
                    group = score_group(
                        task,
                        [run.result() for run in runs],
                        length_penalty=self.config.length_penalty,
                        taskset_id=taskset_id,
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
    logging.getLogger("openai._base_client").addFilter(add_retry_cause)
    for handler in logging.getLogger("ray").handlers:
        handler.setFormatter(logging.Formatter(log_format))


def train(config: Config) -> None:
    """Run actor, batcher, and learner on local Ray, shutting down workers on exit."""
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
