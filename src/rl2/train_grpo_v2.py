"""GRPO training with a JAX learner, vLLM rollouts, and Ray actors.

MLflow's metric step counts completed training iterations: one rollout batch,
its optimizer updates, and publication of the updated weights. All metrics below
are logged every iteration except startup scalars and checkpoint timing.

Startup scalars (at the restored iteration, or step 0 for a new run):
    model/params_millions: Model parameter count in millions.
    time/learner/setup_seconds: Model/optimizer setup, checkpoint restore, NCCL initialization, and initial publication wall time.

Optimization scalars (averaged over minibatches and update epochs):
    losses/policy: Clipped GRPO policy loss.
    policy/approx_kl: Mean exp(log_ratio) - 1 - log_ratio, where log_ratio is learner minus sampling log-probability.
    policy/clip_fraction: Fraction of sampled-token probability ratios outside the clip range.
    policy/grad_norm: Global gradient norm after microbatch averaging, before clipping.
Loss, KL, and clip fraction average sampled tokens within each rollout, then rollouts.

Rollout batch scalars:
    charts/reward_mean: Mean raw rollout reward.
    charts/reward_std: Population standard deviation of raw rollout rewards.
    charts/reward_min: Minimum raw rollout reward.
    charts/reward_max: Maximum raw rollout reward.
    charts/degenerate_group_fraction: Fraction of groups with identical rewards and zero advantages.
    charts/stop_condition_fraction/<label>: Fraction of batch rollouts with each stop condition; None maps to unspecified.
    charts/turns_mean: Mean model responses per rollout, including tool calls.
    charts/input_tokens_mean: Mean newly fed input tokens per rollout; a lower bound if context is rewritten.
    charts/output_tokens_mean: Mean sampled tokens per rollout across all model calls.
Stop-condition labels are added as observed; previously seen labels log zero when
absent from a later batch.

Training progress and settings:
    charts/total_iterations: Cumulative completed training iterations.
    charts/total_updates: Cumulative optimizer updates; update_epochs * minibatches_per_epoch per iteration.
    charts/total_rollouts: Cumulative rollouts used for training.
    charts/learning_rate: Configured optimizer learning rate.

Timing and throughput (logged every iteration unless noted):
    time/learner/iterations_per_second: Reciprocal of iteration wall time.
    time/learner/rollouts_per_second: Batch rollout count divided by iteration wall time.
    time/learner/output_tokens_per_second: Batch sampled-token count divided by iteration wall time.
    time/actor/rollout_seconds: Mean agent execution time per rollout, including model calls and tools.
    time/batcher/fetch_rollout_seconds: Wall time fetching all rollout groups, including actor return transfers.
    time/batcher/prepare_batch_seconds: Wall time preparing the CPU training arrays without explicit synchronization.
    time/learner/batch_fetch_seconds: Wall time fetching the rollout batch, including generation, scoring, and Ray transfers.
    time/learner/update_seconds: Device transfer and optimizer update wall time.
    time/learner/weight_transfer_seconds: Wall time publishing updated learner weights to vLLM.
    time/learner/iteration_seconds: Batch fetching, optimization, and weight publication wall time.
    time/checkpoint_seconds: Checkpoint save wall time, including completion of writes; logged on saves.
    time/batcher/actor_to_batcher_seconds: Mean group transfer latency from actor return to batcher receipt.
    time/learner/batcher_to_learner_seconds: Batch transfer latency from batcher return to learner receipt.
Ray transfer latencies include serialization, transport, deserialization, and
scheduling, using a shared single-node monotonic clock. Timing intervals overlap
and should not be summed. Iteration timing and throughput exclude that iteration's
metric and example logging and checkpoint saves. Cumulative counters include resumed training.
Output-token counts use reported completion usage, counting zero when usage is unavailable.
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
from collections import Counter
from collections.abc import Iterator
from contextlib import AsyncExitStack, ExitStack, closing, contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
from functools import partial
from itertools import islice
from pathlib import Path
from statistics import mean, pstdev
from textwrap import indent
from typing import Annotated, Any, Literal, NamedTuple
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
import torch
import verifiers.v1 as vf
import yaml
from flax.training.train_state import TrainState
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from numpy.typing import NDArray
from opentelemetry import trace as otel_trace
from opentelemetry.trace import StatusCode
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass as config_dataclass
from ray.actor import ActorHandle
from renderers.configs import CustomRendererConfig
from verifiers.v1.configs.agent import TimeoutConfig
from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine
from vllm.utils.network_utils import get_open_port

from rl2.addition_env import AdditionTaskset
from rl2.alfworld_env import ALFWorldConfig, load_alfworld_tasks
from rl2.benchmark_env import (
    AppWorldConfig,
    BenchmarkTask,
    Tau2Config,
    WebShopConfig,
    load_benchmark_tasks,
)
from rl2.codegym_env import CODEGYM_TRAIN_FILE, CodeGymTask, CodeGymTaskset, CodeGymTasksetConfig
from rl2.codegym_server import CodeGymServerConfig, serve_codegym
from rl2.configuration import BaseConfig, load_config
from rl2.mlflow_logging import MLflowLogger
from rl2.qwen3 import Parameters, Qwen3LM, load_qwen3_checkpoint
from rl2.ray_timeline import record_timeline
from rl2.toolrl_env import TOOLRL_TRAIN_FILE, ToolRLTaskset, ToolRLTasksetConfig
from rl2.tracing import flush_tracing, span, tracing_enabled

logger = logging.getLogger(__name__)
MODEL = "Qwen/Qwen3-0.6B"
SERVER_URL = "http://127.0.0.1:8000"
WEIGHT_BUFFER_BYTES = 64 * 1024**2


@config_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class AdditionConfig:
    id: Literal["addition"] = "addition"


@config_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class ToolRLConfig:
    id: Literal["toolrl"] = "toolrl"
    dataset_file: str = TOOLRL_TRAIN_FILE


@config_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class CodeGymConfig:
    manager_url: str
    id: Literal["codegym"] = "codegym"
    dataset_file: str = CODEGYM_TRAIN_FILE
    server: CodeGymServerConfig | None = None

    def __post_init__(self) -> None:
        url = httpx.URL(self.manager_url)
        if url.scheme not in {"http", "https"} or not url.host:
            raise ValueError("CodeGym manager_url must be an HTTP(S) URL")
        if self.server is not None and (
            url.scheme != "http"
            or url.host != "127.0.0.1"
            or url.port is None
            or not 1 <= url.port <= 65535
            or url.port == 8000
            or url.path != "/"
            or url.query
            or url.fragment
            or url.userinfo
        ):
            raise ValueError(
                "Local CodeGym manager_url must be http://127.0.0.1:<port> with a port other than vLLM's 8000"
            )


@config_dataclass(frozen=True)
class Config(BaseConfig):
    model: str = MODEL
    taskset: Annotated[
        AdditionConfig | ToolRLConfig | CodeGymConfig | WebShopConfig | ALFWorldConfig | Tau2Config | AppWorldConfig,
        Field(discriminator="id"),
    ] = AdditionConfig()
    seed: Annotated[int, Field(ge=0)] = 1
    num_groups: Annotated[int, Field(ge=1)] = 1
    group_size: Annotated[int, Field(ge=2)] = 4
    total_iterations: Annotated[int | None, Field(ge=1)] = None
    checkpoint_interval: Annotated[int, Field(gt=0)] = 100
    max_tokens: Annotated[int, Field(ge=1)] = 512
    padding_multiple: Annotated[int, Field(ge=1)] = 256
    max_turns: Annotated[int, Field(ge=1)] = 3
    rollout_timeout: Annotated[float, Field(gt=0)] = 120
    environment_timeout: Annotated[float, Field(gt=0)] = 600
    temperature: Annotated[float, Field(ge=0.01, le=2.0)] = 1.0  # vLLM clamps smaller positive values to 0.01.
    learning_rate: Annotated[float, Field(gt=0)] = 1e-6
    clip_coef: Annotated[float, Field(gt=0, lt=1)] = 0.2
    max_grad_norm: Annotated[float, Field(gt=0)] = 1.0
    update_epochs: Annotated[int, Field(ge=1)] = 1
    minibatches_per_epoch: Annotated[int, Field(ge=1)] = 4
    grad_accum_steps: Annotated[int, Field(ge=1)] = 1
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
        if (
            isinstance(self.taskset, CodeGymConfig)
            and self.taskset.server is not None
            and self.taskset.server.workers < self.num_groups * self.group_size
        ):
            raise ValueError("CodeGym server.workers must cover num_groups * group_size concurrent rollouts")
        if self.taskset.id == "toolrl" and self.max_turns != 1:
            raise ValueError("ToolRL requires max_turns=1; tool calls are graded without execution")


DEFAULT_CONFIG = Config()


@dataclass(frozen=True)
class TrainingProgress:
    train_state: TrainState  # Parameter-shaped params and optimizer moments; () optimizer step.
    shuffle_key: jax.Array  # () typed base PRNG key, folded with the optimizer step.
    iteration: int
    total_rollouts: int
    sampling_state: dict[str, Any]
    stop_conditions: tuple[str, ...]


def checkpoint_manager(run_dir: str) -> ocp.CheckpointManager:
    directory = f"{run_dir.rstrip('/')}/checkpoints"
    if not directory.startswith("gs://"):
        directory = str(Path(directory).resolve())
    return ocp.CheckpointManager(directory, options=ocp.CheckpointManagerOptions(max_to_keep=2))


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
                    "version": 1,
                    "algorithm": "grpo_v2",
                    "config": asdict(config),
                    "iteration": progress.iteration,
                    "total_rollouts": progress.total_rollouts,
                    "sampling_state": progress.sampling_state,
                    "stop_conditions": progress.stop_conditions,
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
    manager: ocp.CheckpointManager, train_state: TrainState, config: Config
) -> TrainingProgress | None:
    iteration = manager.latest_step()
    if iteration is None:
        return None
    metadata = manager.restore(iteration, args=ocp.args.Composite(metadata=ocp.args.JsonRestore())).metadata
    if metadata["version"] != 1 or metadata["algorithm"] != "grpo_v2":
        raise ValueError("Unsupported GRPO v2 checkpoint")
    if metadata["iteration"] != iteration or iteration < 0 or metadata["total_rollouts"] < 0:
        raise ValueError("Invalid GRPO v2 checkpoint progress")
    restored = manager.restore(
        iteration,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(
                {
                    "train_state": train_state,
                    "shuffle_key": jax.random.key_data(jax.random.key(config.seed)),  # (K,)
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
        stop_conditions=tuple(metadata["stop_conditions"]),
    )


@dataclass(frozen=True)
class TrainingSequence:
    token_ids: NDArray[np.int32]  # (T,) exact context and generated tokens in one sequence.
    loss_mask: NDArray[np.bool_]  # (T,) train on sampled tokens only; exclude prompts and tool results.
    old_log_probs: NDArray[np.float32]  # (T,) sampling-policy log-probabilities; ignored where loss_mask is False.


@dataclass(frozen=True)
class Rollout:
    sequence: TrainingSequence  # One complete token sequence spanning all model and tool turns.
    reward: float  # Total task reward before group normalization.
    advantage: float  # (reward - group mean) / (population std + 1e-4).
    completion: str  # Final assistant text for sample logging.
    num_turns: int  # Number of model responses, including tool calls.
    num_input_tokens: int  # Newly fed input tokens; a lower bound when context is rewritten.
    num_output_tokens: int  # Total sampled tokens across model calls.
    rollout_seconds: float  # Agent execution time, including model calls and tools.
    stop_condition: str | None  # Verifiers termination reason; None if unspecified.


@dataclass(frozen=True)
class RolloutGroup:
    task_id: str | None  # Source task ID shared by the group; None if the task has no ID.
    prompt: str  # Shared task prompt for sample logging.
    rollouts: list[Rollout]  # Independent attempts at the same task, in sampling order, scored as one group.


@dataclass(frozen=True)
class Transfer[T]:
    value: T
    ready_at: float  # Same-node monotonic timestamp immediately before Ray serializes the return value.


def format_rollout_examples(groups: list[RolloutGroup]) -> str:
    """Format up to three final replies from the batch in sampling order."""
    sections: list[str] = []
    examples = islice(((group, rollout) for group in groups for rollout in group.rollouts), 3)
    for index, (group, rollout) in enumerate(examples, start=1):
        sections.append(
            f"{'=' * 72}\n"
            f"Example {index} | task={group.task_id} | reward={rollout.reward:.3f} | "
            f"advantage={rollout.advantage:.3f}\n"
            f"Turns: {rollout.num_turns} | Output tokens: {rollout.num_output_tokens} | "
            f"Stop: {rollout.stop_condition}\n\n"
            f"Prompt:\n{indent(group.prompt or '(empty)', '    ')}\n\n"
            f"Final assistant output:\n{indent(rollout.completion or '(empty)', '    ')}"
        )
    return "\n\n".join(sections)


class GRPOBatch[Array: np.ndarray | jax.Array](NamedTuple):
    token_ids: Array  # (B, T) int32 right-padded complete transcripts.
    attention_mask: Array  # (B, T) bool context and sampled tokens, excluding padding.
    loss_mask: Array  # (B, T) bool sampled tokens only.
    old_log_probs: Array  # (B, T) float32 temperature-scaled sampling-policy log-probabilities.
    advantages: Array  # (B,) float32 normalized within each rollout group.


# ---


@dataclass(frozen=True)
class RolloutBatch:
    training_batch: GRPOBatch[np.ndarray]  # Leaves: (B, T), except advantages: (B,).
    rollout_metrics: dict[str, float]
    stop_counts: dict[str, int]
    num_output_tokens: int
    examples: str
    transfer_metrics: dict[str, float | int]


def build_rollout_batch(
    groups: list[RolloutGroup],
    training_batch: GRPOBatch[np.ndarray],  # Leaves: (B, T), except advantages: (B,).
    transfer_metrics: dict[str, float | int],
) -> RolloutBatch:
    """Keep only training arrays and the summaries used by learner logging."""
    rollouts = [rollout for group in groups for rollout in group.rollouts]
    rewards = [rollout.reward for rollout in rollouts]
    return RolloutBatch(
        training_batch=training_batch,
        rollout_metrics={
            "charts/reward_mean": mean(rewards),
            "charts/reward_std": pstdev(rewards),
            "charts/reward_min": min(rewards),
            "charts/reward_max": max(rewards),
            "charts/degenerate_group_fraction": mean(
                all(rollout.reward == group.rollouts[0].reward for rollout in group.rollouts) for group in groups
            ),
            "charts/turns_mean": mean(rollout.num_turns for rollout in rollouts),
            "charts/input_tokens_mean": mean(rollout.num_input_tokens for rollout in rollouts),
            "charts/output_tokens_mean": mean(rollout.num_output_tokens for rollout in rollouts),
            "time/actor/rollout_seconds": mean(rollout.rollout_seconds for rollout in rollouts),
        },
        stop_counts=Counter(
            rollout.stop_condition if rollout.stop_condition is not None else "unspecified" for rollout in rollouts
        ),
        num_output_tokens=sum(rollout.num_output_tokens for rollout in rollouts),
        examples=format_rollout_examples(groups),
        transfer_metrics=transfer_metrics,
    )


# Use a value alias so Ray can serialize script-defined actors without importing __main__.Metrics.
Metrics = dict[str, jax.Array]  # Scalar leaves.


def token_log_probs(
    state: TrainState,
    params: Parameters,  # HF parameter shapes.
    batch: GRPOBatch[jax.Array],  # (B, T), (B,) leaves.
    temperature: float,
) -> jax.Array:  # (B, T-1)
    """(B, T-1) log-probabilities for tokens at positions 1 through T-1."""
    logits = state.apply_fn(
        {"params": params}, batch.token_ids[:, :-1], attention_mask=batch.attention_mask[:, :-1]
    ).logits  # (B, T-1, V)
    logits = logits.astype(jnp.float32) / temperature  # (B, T-1, V)
    # Fuse selection's backward pass with softmax instead of scattering into a full FP32 vocabulary buffer.
    targets = jax.nn.one_hot(batch.token_ids[:, 1:], logits.shape[-1], dtype=jnp.bool_)  # (B, T-1, V)
    selected = jnp.where(targets, logits, 0).sum(axis=-1)  # (B, T-1)
    return selected - jax.nn.logsumexp(logits, axis=-1)  # (B, T-1)


def objective(
    log_probs: jax.Array,  # (B, T-1)
    batch: GRPOBatch[jax.Array],  # (B, T), (B,) leaves.
    clip_coef: float,
) -> tuple[jax.Array, Metrics]:  # (), scalar metric leaves
    mask = batch.loss_mask[:, 1:]  # (B, T-1)

    def average(values: jax.Array) -> jax.Array:  # (B, T-1) -> ()
        return (jnp.where(mask, values, 0).sum(axis=1) / jnp.maximum(mask.sum(axis=1), 1)).mean()  # ()

    log_ratio = jnp.where(mask, log_probs - jax.lax.stop_gradient(batch.old_log_probs[:, 1:]), 0)  # (B, T-1)
    ratio = jnp.exp(log_ratio)  # (B, T-1)
    advantages = jax.lax.stop_gradient(batch.advantages[:, None])  # (B, 1)
    loss = -average(jnp.minimum(ratio * advantages, jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef) * advantages))  # ()
    return loss, {
        "losses/policy": loss,  # ()
        "policy/approx_kl": average(jnp.expm1(log_ratio) - log_ratio),  # ()
        "policy/clip_fraction": average((jnp.abs(ratio - 1) > clip_coef).astype(jnp.float32)),  # ()
    }


def create_state(model: Qwen3LM, params: Parameters, config: Config) -> TrainState:
    """Keep FP32 master weights and Adam state; the model controls compute dtype."""
    return TrainState.create(
        apply_fn=model.apply,
        params=jax.tree.map(lambda value: value.astype(jnp.float32), params),  # HF parameter shapes.
        tx=optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate)),
    ).replace(step=jnp.asarray(0, dtype=jnp.int32))  # () Match the device counter returned by update.


@partial(jax.jit, static_argnames=("grad_accum_steps", "temperature", "clip_coef"), donate_argnums=(0,))
def update(
    state: TrainState,
    batch: GRPOBatch[jax.Array],  # One optimizer minibatch: (B, T), (B,) leaves.
    *,
    grad_accum_steps: int,
    temperature: float,
    clip_coef: float,
) -> tuple[TrainState, Metrics]:
    """Consume state buffers and return the updated state and scalar metrics."""
    if grad_accum_steps < 1 or batch.token_ids.shape[0] % grad_accum_steps:
        raise ValueError("grad_accum_steps must be positive and divide the minibatch size")
    # Python logging runs on tracing, not on cached executions.
    logger.info(
        "JIT trace update: batch_shape=%s grad_accum_steps=%s temperature=%s clip_coef=%s",
        batch.token_ids.shape,
        grad_accum_steps,
        temperature,
        clip_coef,
    )
    microbatch_size = batch.token_ids.shape[0] // grad_accum_steps
    microbatches = jax.tree.map(
        lambda value: value.reshape(grad_accum_steps, microbatch_size, *value.shape[1:]), batch
    )  # (A, B/A, T); advantages: (A, B/A). A = grad_accum_steps.

    def accumulate(
        grads: Parameters,
        microbatch: GRPOBatch[jax.Array],  # (B/A, T), (B/A,) leaves.
    ) -> tuple[Parameters, Metrics]:
        def loss_fn(params: Parameters) -> tuple[jax.Array, Metrics]:  # (), scalar metric leaves
            return objective(token_log_probs(state, params, microbatch, temperature), microbatch, clip_coef)

        (_, metrics), gradient = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return jax.tree.map(jnp.add, grads, gradient), metrics  # HF parameter shapes; scalar metrics.

    gradients, metrics = jax.lax.scan(
        accumulate, jax.tree.map(jnp.zeros_like, state.params), microbatches
    )  # HF parameter shapes; metric leaves: (A,).
    gradients = jax.tree.map(lambda value: value / grad_accum_steps, gradients)  # HF parameter shapes.
    metrics = jax.tree.map(jnp.mean, metrics)  # Scalar leaves.
    metrics["policy/grad_norm"] = optax.global_norm(gradients)  # () Before clipping.
    return state.apply_gradients(grads=gradients), metrics


def training_sequence(trace: vf.Trace) -> TrainingSequence:
    branches = trace.branches
    if len(branches) != 1:
        raise ValueError(f"Expected exactly one training sequence, got {len(branches)} branches")
    branch = branches[0]
    if not trace.agent.trainable or not branch.trainable or not any(branch.sampled_mask):
        raise ValueError("Rollout has no trainable tokens")
    for node in branch.nodes:
        if len(node.token_ids) != len(node.mask):
            raise ValueError("Token IDs and loss mask must have equal lengths")
        if node.sampled and not any(node.mask):
            raise ValueError("Sampled messages must contain training tokens")
        # Verifiers otherwise fills missing log-probabilities with zeros.
        if len(node.logprobs) != sum(node.mask):
            raise ValueError("Each sampled token must have a log-probability")
    return TrainingSequence(
        token_ids=np.asarray(branch.token_ids, dtype=np.int32),  # (T,)
        loss_mask=np.asarray(branch.sampled_mask, dtype=np.bool_),  # (T,)
        old_log_probs=np.asarray(branch.logprobs, dtype=np.float32),  # (T,)
    )


async def score_group(task: vf.Task, traces: list[vf.Trace]) -> RolloutGroup:
    logger.debug("Scoring %s rollouts for task %s", len(traces), task.data.id)
    if len(traces) < 2:
        raise ValueError("A rollout group needs at least two traces")
    for trace in traces:
        if not trace.ok:
            raise RuntimeError(f"Rollout failed: {trace.stop_condition}: {trace.errors}")
    # These tasks can be graded from the completed trace without a live runtime.
    async with asyncio.timeout(30):
        for trace in traces:
            await task.score(trace)
    # Preserve reward differences before the training batch converts advantages to float32.
    rewards = [trace.reward for trace in traces]
    reward_mean, reward_std = mean(rewards), pstdev(rewards)
    advantages = [(reward - reward_mean) / (reward_std + 1e-4) for reward in rewards]
    return RolloutGroup(
        task_id=task.data.id,
        prompt=task.data.prompt_text,
        rollouts=[
            Rollout(
                sequence=training_sequence(trace),
                reward=trace.reward,
                advantage=advantage,
                completion=trace.last_reply,
                num_turns=trace.num_turns,
                num_input_tokens=trace.num_input_tokens,
                num_output_tokens=trace.usage.completion_tokens if trace.usage is not None else 0,
                rollout_seconds=trace.timing.agent.duration,
                stop_condition=trace.stop_condition,
            )
            for trace, advantage in zip(traces, advantages, strict=True)
        ],
    )


def record_rollout_spans(trace: vf.Trace) -> None:
    """Export Verifiers' recorded wall-clock intervals under the current rollout."""
    if not otel_trace.get_current_span().is_recording():
        return
    tracer = otel_trace.get_tracer(__name__)
    model_context = otel_trace.set_span_in_context(otel_trace.get_current_span())
    for name, timing in (
        ("boot", trace.timing.boot),
        ("setup", trace.timing.setup),
        ("agent", trace.timing.agent),
        ("finalize", trace.timing.finalize),
        ("scoring", trace.timing.scoring),
    ):
        if timing.start <= 0 or timing.end < timing.start:
            continue
        current = tracer.start_span(
            f"actor.rollout.{name}",
            start_time=int(timing.start * 1_000_000_000),
        )
        if name == "agent":
            current.set_attribute("model_seconds", trace.timing.agent.model.duration)
            current.set_attribute("harness_seconds", trace.timing.agent.harness.duration)
            model_context = otel_trace.set_span_in_context(current)
        current.end(end_time=int(timing.end * 1_000_000_000))
    for index, call in enumerate(trace.calls):
        if call.time.start <= 0 or call.time.end < call.time.start:
            continue
        current = tracer.start_span(
            "actor.rollout.model",
            context=model_context,
            start_time=int(call.time.start * 1_000_000_000),
            attributes={
                "call_index": index,
                "model": call.model or "unspecified",
                "finish_reason": call.finish_reason or "unspecified",
            },
        )
        if call.usage is not None:
            current.set_attribute("input_tokens", call.usage.input_tokens)
            current.set_attribute("output_tokens", call.usage.completion_tokens)
        if call.error is not None:
            current.set_attribute("error.type", call.error.type)
            current.set_status(StatusCode.ERROR, call.error.type)
        current.end(end_time=int(call.time.end * 1_000_000_000))


def create_agent(config: Config = DEFAULT_CONFIG) -> vf.Agent:
    logger.debug("Creating Verifiers agent: model=%s endpoint=%s", config.model, SERVER_URL)
    return vf.make_agent(
        config=vf.AgentConfig(
            model=config.model,
            harness=vf.HarnessConfig(id="null"),
            runtime=vf.SubprocessConfig(),
            client=vf.TrainClientConfig(
                base_url=f"{SERVER_URL}/v1",
                api_key_var="VLLM_API_KEY",
                renderer_model_name=config.model,
                renderer=CustomRendererConfig(import_path="rl2.toolrl_env.ToolRLRenderer")
                if config.taskset.id == "toolrl"
                else None,
            ),
            sampling=vf.SamplingConfig(
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=1.0,
                chat_template_kwargs={"enable_thinking": config.taskset.id == "toolrl"},
            ),
            max_turns=config.max_turns,
            timeout=TimeoutConfig(setup=120, rollout=config.rollout_timeout, finalize=30, scoring=30),
        )
    )


@contextmanager
def serve_vllm(model: str = MODEL) -> Iterator[subprocess.Popen[bytes]]:
    logger.info("Starting vLLM server for %s", model)
    # Inherit the GPU assigned to this actor through CUDA_VISIBLE_DEVICES.
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--dtype",
            "bfloat16",
            "--generation-config",
            "vllm",
            "--logprobs-mode",
            "processed_logprobs",
            "--weight-transfer-config",
            '{"backend":"nccl"}',
        ],
        env={**os.environ, "VLLM_SERVER_DEV_MODE": "1"},
    ) as server:
        try:
            yield server
        finally:
            logger.info("Stopping vLLM server: pid=%s", server.pid)
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("vLLM did not stop within 10s; killing pid=%s", server.pid)
                server.kill()
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
        to a multiple of config.padding_multiple, capped at context_length. Padding is zero/False, and
        transcripts are never truncated. Existing rollout advantages are preserved.

        Returns:
            GRPOBatch with token_ids: (B, T) int32 complete transcripts;
            attention_mask: (B, T) bool non-padding positions;
            loss_mask: (B, T) bool sampled-token positions;
            old_log_probs: (B, T) float32 sampling-policy log-probabilities;
            advantages: (B,) float32 per-rollout advantages.

        Raises:
            ValueError: If there are no rollouts, a transcript exceeds context_length,
                or a transcript lacks an initial context token followed by a sampled token.
        """
        rollouts = [rollout for group in groups for rollout in group.rollouts]
        if not rollouts:
            raise ValueError("A training batch must contain rollouts")
        length = max(len(rollout.sequence.token_ids) for rollout in rollouts)
        if length > context_length:
            raise ValueError(f"Rollout length {length} exceeds model context length {context_length}")
        multiple = self.config.padding_multiple
        width = min((length + multiple - 1) // multiple * multiple, context_length)
        shape = (len(rollouts), width)
        # Host copies avoid compiling JAX padding and reductions for every transcript length.
        arrays = {
            "token_ids": np.zeros(shape, dtype=np.int32),  # (B, T)
            "attention_mask": np.zeros(shape, dtype=np.bool_),  # (B, T)
            "loss_mask": np.zeros(shape, dtype=np.bool_),  # (B, T)
            "old_log_probs": np.zeros(shape, dtype=np.float32),  # (B, T)
            "advantages": np.asarray([rollout.advantage for rollout in rollouts], dtype=np.float32),  # (B,)
        }
        for index, rollout in enumerate(rollouts):
            sequence = rollout.sequence
            length = len(sequence.token_ids)
            mask = sequence.loss_mask  # (L,) unpadded transcript length.
            if length < 2 or mask[0] or not mask[1:].any():
                raise ValueError("Each rollout needs context followed by at least one sampled token")
            for name in ("token_ids", "loss_mask", "old_log_probs"):
                arrays[name][index, :length] = getattr(sequence, name)  # (L,)
            arrays["attention_mask"][index, :length] = True
        return GRPOBatch(**arrays)  # (B, T), (B,) leaves.

    async def fetch_next_batch(self, num_groups: int, group_size: int, context_length: int) -> Transfer[RolloutBatch]:
        if num_groups < 1:
            raise ValueError("num_groups must be at least 1")
        started = time.monotonic()
        logger.info("Fetching batch: num_groups=%s group_size=%s", num_groups, group_size)

        async def fetch_group() -> tuple[RolloutGroup, float]:
            result = await self.actor.fetch_next_group.remote(group_size=group_size)
            received_at = time.monotonic()
            seconds = received_at - result.ready_at
            logger.info(
                "Ray transfer Actor -> Batcher: task=%s seconds=%.6f",
                result.value.task_id,
                seconds,
            )
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
            transfer_metrics={
                "time/batcher/fetch_rollout_seconds": fetched - started,
                "time/batcher/prepare_batch_seconds": prepared - fetched,
                "time/batcher/actor_to_batcher_seconds": mean(seconds for _, seconds in results),
            },
        )
        logger.info("Batch ready: groups=%s elapsed=%.2fs", num_groups, time.monotonic() - started)
        return Transfer(value=batch, ready_at=time.monotonic())


@ray.remote(num_gpus=1)
class Learner:
    def __init__(self, batcher: ActorHandle, actor: ActorHandle, config: Config = DEFAULT_CONFIG) -> None:
        self.config = config
        self.batcher = batcher
        self.actor = actor
        self.shuffle_key = jax.random.key(config.seed)  # () typed base PRNG key.

    def initialize_weight_transfer(self, resources: ExitStack) -> None:
        init_info = {"master_address": "127.0.0.1", "master_port": get_open_port(), "rank_offset": 1, "world_size": 2}
        logger.info("Initializing learner/vLLM NCCL weight transfer")
        pending = self.actor.initialize_weight_transfer.remote(init_info)
        self._transfer_group = NCCLWeightTransferEngine.trainer_init(init_info)
        resources.callback(self._transfer_group.destroy)
        ray.get(pending)
        logger.info("NCCL weight transfer ready")

    def start(self) -> None:
        with (
            MLflowLogger(experiment="grpo_v2", run_name=self.config.run_id) as writer,
            ExitStack() as resources,
        ):
            manager = resources.enter_context(closing(checkpoint_manager(self.config.ckpt_dir)))
            writer.log_params(asdict(self.config))
            logger.info("Run ID: %s; MLflow run: %s", self.config.run_id, writer.run_id)
            start = time.monotonic()
            logger.info("Loading learner model %s on its assigned GPU", self.config.model)
            device = jax.devices("gpu")[0]
            self.mesh = Mesh(devices=[device], axis_names=("batch",))
            with jax.default_device(device):
                self.model, variables = load_qwen3_checkpoint(
                    self.config.model,
                    dtype="bfloat16",
                    attention_backend="tokamax",
                    gradient_checkpointing=self.config.gradient_checkpointing,
                )
                self.state = create_state(
                    self.model,
                    jax.device_put(variables["params"], NamedSharding(self.mesh, PartitionSpec())),
                    self.config,
                )
            del variables
            progress = restore_checkpoint(manager, self.state, self.config)
            iteration = 0
            total_rollouts = 0
            stop_conditions: set[str] = set()
            if progress is not None:
                self.state = progress.train_state
                self.shuffle_key = progress.shuffle_key  # ()
                iteration = progress.iteration
                total_rollouts = progress.total_rollouts
                stop_conditions.update(progress.stop_conditions)
                ray.get(self.actor.set_sampling_state.remote(progress.sampling_state))
                logger.info("Resumed run %s at iteration %s", self.config.run_id, iteration)
            writer.log_text(yaml.safe_dump(asdict(self.config)), f"config/{iteration}.yaml")
            logger.info("Learner loaded %s parameter tensors", len(self.state.params))
            self.initialize_weight_transfer(resources=resources)
            self.publish_weights()
            writer.log_metrics(
                {
                    "model/params_millions": sum(p.size for p in jax.tree.leaves(self.state.params)) / 1_000_000,
                    "time/learner/setup_seconds": time.monotonic() - start,
                },
                step=iteration,
            )
            logger.info("Starting GRPO training loop")
            while self.config.total_iterations is None or iteration < self.config.total_iterations:
                iteration += 1
                # Each iteration is a separate trace, including its remote calls and child tasks.
                with span(
                    "train.iteration",
                    {"run_id": self.config.run_id or writer.run_id, "iteration": iteration, "model": self.config.model},
                    root=True,
                ):
                    started = time.monotonic()
                    with span("learner.fetch_batch"):
                        batch = self.fetch_next_batch(
                            num_groups=self.config.num_groups, group_size=self.config.group_size
                        )
                    fetched = time.monotonic()
                    with span("learner.optimize"):
                        update_metrics = self.optimize(batch.training_batch)
                    optimized = time.monotonic()
                    with span("learner.publish_weights"):
                        self.publish_weights()
                    now = time.monotonic()
                    num_rollouts = batch.training_batch.token_ids.shape[0]
                    stop_conditions.update(batch.stop_counts)
                    total_rollouts += num_rollouts
                    seconds = max(now - started, 1e-9)
                    scalars = {
                        **update_metrics,
                        **batch.transfer_metrics,
                        **batch.rollout_metrics,
                        **{
                            f"charts/stop_condition_fraction/{condition}": batch.stop_counts.get(condition, 0)
                            / num_rollouts
                            for condition in sorted(stop_conditions)
                        },
                        "charts/total_iterations": iteration,
                        "charts/total_updates": int(self.state.step),
                        "charts/learning_rate": self.config.learning_rate,
                        "charts/total_rollouts": total_rollouts,
                        "time/learner/iterations_per_second": 1 / seconds,
                        "time/learner/rollouts_per_second": num_rollouts / seconds,
                        "time/learner/output_tokens_per_second": batch.num_output_tokens / seconds,
                        "time/learner/batch_fetch_seconds": fetched - started,
                        "time/learner/update_seconds": optimized - fetched,
                        "time/learner/weight_transfer_seconds": now - optimized,
                        "time/learner/iteration_seconds": now - started,
                    }
                    logger.info(
                        "iteration=%s reward=%.3f degenerate_group_fraction=%.3f batch=%.2fs "
                        "group_transfer=%.6fs batch_transfer=%.6fs loss=%.4f update=%.2fs weights=%.2fs",
                        iteration,
                        scalars["charts/reward_mean"],
                        scalars["charts/degenerate_group_fraction"],
                        scalars["time/learner/batch_fetch_seconds"],
                        scalars["time/batcher/actor_to_batcher_seconds"],
                        scalars["time/learner/batcher_to_learner_seconds"],
                        scalars["losses/policy"],
                        scalars["time/learner/update_seconds"],
                        scalars["time/learner/weight_transfer_seconds"],
                    )
                    if iteration % 10 == 0:
                        with span("learner.log_examples"):
                            examples = f"Rollout examples | step {iteration}\n\n{batch.examples}"
                            logger.info("\n%s", examples)
                            writer.log_text(examples, f"examples/{iteration}.txt")
                    if iteration % self.config.checkpoint_interval == 0 or iteration == self.config.total_iterations:
                        with span("learner.checkpoint"):
                            scalars["time/checkpoint_seconds"] = save_checkpoint(
                                manager,
                                TrainingProgress(
                                    train_state=self.state,
                                    shuffle_key=self.shuffle_key,  # ()
                                    iteration=iteration,
                                    total_rollouts=total_rollouts,
                                    sampling_state=ray.get(self.actor.get_sampling_state.remote()),
                                    stop_conditions=tuple(sorted(stop_conditions)),
                                ),
                                self.config,
                            )
                    with span("learner.log_metrics"):
                        writer.log_metrics(scalars, step=iteration)

    def fetch_next_batch(self, num_groups: int, group_size: int) -> RolloutBatch:
        logger.debug("Requesting batch: num_groups=%s group_size=%s", num_groups, group_size)
        result = ray.get(
            self.batcher.fetch_next_batch.remote(
                num_groups=num_groups,
                group_size=group_size,
                context_length=self.model.config.max_position_embeddings,
            )
        )
        received_at = time.monotonic()
        seconds = received_at - result.ready_at
        batch = result.value
        logger.info(
            "Ray transfer Batcher -> Learner: groups=%s seconds=%.6f",
            num_groups,
            seconds,
        )
        return replace(
            batch,
            transfer_metrics={**batch.transfer_metrics, "time/learner/batcher_to_learner_seconds": seconds},
        )

    def publish_weights(self) -> None:
        started = time.monotonic()
        params = jax.tree.map(lambda value: value.astype(jnp.bfloat16), self.state.params)  # HF parameter shapes.
        # vLLM packs weights on separate CUDA streams without waiting on the DLPack consumer stream.
        jax.block_until_ready(params)
        weights = jax.tree.map(torch.from_dlpack, params)  # GPU views; HF parameter shapes.
        buffer_bytes = max(WEIGHT_BUFFER_BYTES, max(weight.nbytes for weight in weights.values()))
        update_info = {
            "names": list(weights),
            "dtype_names": [str(weight.dtype).removeprefix("torch.") for weight in weights.values()],
            "shapes": [list(weight.shape) for weight in weights.values()],
            "packed": True,
            "packed_buffer_size_bytes": buffer_bytes,
        }
        logger.info("Publishing %s parameter tensors via NCCL", len(weights))
        pending = self.actor.update_weights.remote(update_info)
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=iter(weights.items()),
            trainer_args=NCCLTrainerSendWeightsArgs(
                group=self._transfer_group, packed=True, packed_buffer_size_bytes=buffer_bytes
            ),
        )
        torch.cuda.synchronize()
        ray.get(pending)
        logger.info("Weights published: elapsed=%.2fs", time.monotonic() - started)

    def optimize(self, batch: GRPOBatch[np.ndarray]) -> dict[str, float]:  # Batch leaves: (B, T), (B,).
        device_batch: GRPOBatch[jax.Array] = jax.device_put(
            batch, NamedSharding(self.mesh, PartitionSpec("batch"))
        )  # (B, T), (B,) leaves.
        if batch.token_ids.shape[0] % self.config.minibatches_per_epoch:
            raise ValueError("minibatches_per_epoch must divide the rollout batch size")
        minibatch_size = batch.token_ids.shape[0] // self.config.minibatches_per_epoch
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
                self.state, metrics = update(
                    self.state,
                    minibatch,
                    grad_accum_steps=self.config.grad_accum_steps,
                    temperature=self.config.temperature,
                    clip_coef=self.config.clip_coef,
                )
                history.append(metrics)
        metrics = jax.tree.map(lambda *values: jnp.stack(values).mean(), *history)  # Scalar leaves.
        return {name: float(value) for name, value in metrics.items()}


# Restrict JAX before import/backend initialization; vLLM still uses the assigned CUDA device.
@ray.remote(num_gpus=1, runtime_env={"env_vars": {"JAX_PLATFORMS": "cpu"}})
class Actor:
    def __init__(self, config: Config = DEFAULT_CONFIG) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self._server = ExitStack()
        self._agent_resources = AsyncExitStack()
        self._agent: vf.Agent | None = None
        self._agent_lock = asyncio.Lock()
        self._runs: set[asyncio.Task[vf.Trace]] = set()
        self._closed = False
        self._weights_valid = True
        if isinstance(config.taskset, (WebShopConfig, Tau2Config, AppWorldConfig)):
            self.tasks = load_benchmark_tasks(config.taskset, config.max_turns, config.environment_timeout)
        elif isinstance(config.taskset, ALFWorldConfig):
            self.tasks = load_alfworld_tasks(config.taskset, config.max_turns)
        elif config.taskset.id == "codegym":
            self.tasks = list(CodeGymTaskset(config=CodeGymTasksetConfig(data_file=config.taskset.dataset_file)))
        elif config.taskset.id == "toolrl":
            self.tasks = list(ToolRLTaskset(config=ToolRLTasksetConfig(data_file=config.taskset.dataset_file)))
        else:
            self.tasks = list(AdditionTaskset(config=vf.TasksetConfig()))
        logger.info("Starting actor and vLLM on its assigned GPU")
        with self._server as resources:
            if isinstance(config.taskset, CodeGymConfig) and config.taskset.server is not None:
                resources.enter_context(serve_codegym(config.taskset.server, config.taskset.manager_url))
            server = resources.enter_context(serve_vllm(model=config.model))
            wait_for_server(server)
            self._server = resources.pop_all()
        logger.info("Actor ready for rollout requests: tasks=%s", len(self.tasks))

    def __ray_shutdown__(self) -> None:
        logger.info("Shutting down actor")
        self._server.close()
        logger.info("Actor resources released")

    async def _get_agent(self) -> vf.Agent:
        async with self._agent_lock:
            if self._closed:
                raise RuntimeError("Actor is closed")
            if self._agent is None:
                self._agent = await self._agent_resources.enter_async_context(create_agent(self.config))
            return self._agent

    async def aclose(self) -> None:
        # Ray's synchronous shutdown hook cannot await the agent's cleanup.
        async with self._agent_lock, self._agent_resources:
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
        if not self._weights_valid:
            raise RuntimeError("Cannot generate during or after a failed weight update")
        task = self.rng.choice(self.tasks)
        logger.debug("Selected task %s: %r", task.data.id, task.data.prompt)
        group = await self.collect_group(task, group_size=group_size)
        return Transfer(value=group, ready_at=time.monotonic())

    async def initialize_weight_transfer(self, init_info: dict[str, Any]) -> None:
        with span("actor.initialize_weight_transfer"):
            async with httpx.AsyncClient(base_url=SERVER_URL, timeout=600) as client:
                response = await client.post("/init_weight_transfer_engine", json={"init_info": init_info})
                response.raise_for_status()

    async def update_weights(self, update_info: dict[str, Any]) -> None:
        with span("actor.update_weights"):
            self._weights_valid = False
            logger.info("Receiving learner weights: tensors=%s", len(update_info["names"]))
            async with httpx.AsyncClient(base_url=SERVER_URL, timeout=600) as client:
                with span("actor.update_weights.pause", {"mode": self.config.weight_update_pause_mode}):
                    response = await client.post(
                        "/pause", params={"mode": self.config.weight_update_pause_mode, "clear_cache": "true"}
                    )
                    response.raise_for_status()
                with span("actor.update_weights.start"):
                    response = await client.post("/start_weight_update")
                    response.raise_for_status()
                with span("actor.update_weights.transfer", {"tensor_count": len(update_info["names"])}):
                    response = await client.post("/update_weights", json={"update_info": update_info})
                    response.raise_for_status()
                with span("actor.update_weights.finish"):
                    response = await client.post("/finish_weight_update")
                    response.raise_for_status()
                with span("actor.update_weights.resume"):
                    response = await client.post("/resume")
                    response.raise_for_status()
            self._weights_valid = True
            logger.info("Learner weights applied; generation resumed")

    async def collect_group(self, task: vf.Task, group_size: int) -> RolloutGroup:
        if group_size < 2:
            raise ValueError("group_size must be at least 2")
        span_data = {"group_id": uuid4().hex, "task_id": str(task.data.id), "group_size": str(group_size)}
        started = time.monotonic()
        logger.info("Collecting group: task=%s group_size=%s", task.data.id, group_size)
        try:
            with span("actor.collect_group", {**span_data, "run_id": self.config.run_id or "unspecified"}):
                agent = await self._get_agent()
                async with asyncio.TaskGroup() as pending:

                    async def run_rollout(index: int) -> vf.Trace:
                        with span("actor.rollout", {**span_data, "rollout_index": index}) as rollout_span:
                            if isinstance(task, BenchmarkTask) and isinstance(
                                self.config.taskset, (WebShopConfig, Tau2Config, AppWorldConfig)
                            ):
                                result = await task.run(
                                    agent,
                                    self.config.taskset,
                                    self.config.max_turns,
                                    self.config.environment_timeout,
                                )
                            elif isinstance(task, CodeGymTask) and self.config.taskset.id == "codegym":
                                result = await task.run(agent, self.config.taskset.manager_url)
                            else:
                                result = await agent.run(task.defer_scoring())
                            record_rollout_spans(result)
                            rollout_span.set_attribute("stop_condition", result.stop_condition or "unspecified")
                            if not result.ok:
                                rollout_span.set_status(StatusCode.ERROR, "Rollout failed")
                            return result

                    runs = [pending.create_task(run_rollout(index)) for index in range(group_size)]
                    self._runs.update(runs)
                    for run in runs:
                        run.add_done_callback(self._runs.discard)
                with span("actor.score_group"):
                    group = await score_group(task, [run.result() for run in runs])
        except Exception:
            logger.exception(
                "Group failed: task=%s group_size=%s elapsed=%.2fs",
                task.data.id,
                group_size,
                time.monotonic() - started,
            )
            raise
        logger.info(
            "Collected group: task=%s elapsed=%.2fs output_tokens=%s rewards=%s advantages=%s",
            task.data.id,
            time.monotonic() - started,
            sum(rollout.num_output_tokens for rollout in group.rollouts),
            [rollout.reward for rollout in group.rollouts],
            [rollout.advantage for rollout in group.rollouts],
        )
        return group


def train(config: Config) -> None:
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
            logging_config=ray.LoggingConfig(log_level="INFO"),
            runtime_env={"env_vars": {"RAY_PROFILING": "1"} if config.enable_ray_timeline else {}},
            # Zero disables event reporting in Ray 2.59 despite the timeline docstring.
            _system_config={"enable_timeline": config.enable_ray_timeline, "task_events_report_interval_ms": 1000},
            _tracing_startup_hook="rl2.tracing:setup_tracing" if config.enable_tracing and tracing_enabled() else None,
        ),
        record_timeline(timeline_path) if config.enable_ray_timeline else nullcontext(),
    ):
        resources.callback(flush_tracing)
        logger.info("Creating batcher, learner, and actor")
        actor = Actor.remote(config=config)
        batcher = Batcher.remote(actor=actor, config=config)
        learner = Learner.remote(batcher=batcher, actor=actor, config=config)
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
