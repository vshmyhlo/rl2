"""GRPO training with a JAX learner, vLLM rollouts, and Ray actors.

MLflow's metric step counts completed training iterations: one rollout batch,
its optimizer updates, and publication of the updated weights. All metrics below
are logged every iteration except the startup scalars.

Startup scalars (step 0):
    model/params_millions: Model parameter count in millions.
    time/learner/setup_seconds: Model loading, optimizer setup, NCCL initialization, and initial weight publication wall time.

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

Timing and throughput (logged every iteration):
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
    time/batcher/actor_to_batcher_seconds: Mean group transfer latency from actor return to batcher receipt.
    time/learner/batcher_to_learner_seconds: Batch transfer latency from batcher return to learner receipt.
Ray transfer latencies include serialization, transport, deserialization, and
scheduling, using a shared single-node monotonic clock. Timing intervals overlap
and should not be summed. Iteration timing and throughput exclude that iteration's
metric and example logging. Cumulative counters start at zero for each training session.
Output-token counts use reported completion usage, counting zero when usage is unavailable.
"""

import argparse
import asyncio
import logging
import os
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from functools import partial
from itertools import islice
from statistics import mean, pstdev
from textwrap import indent
from typing import Annotated, Any, Literal, NamedTuple
from urllib.error import URLError
from urllib.request import urlopen

import httpx
import jax
import jax.numpy as jnp
import optax
import ray
import torch
import verifiers.v1 as vf
import yaml
from flax.training.train_state import TrainState
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass as config_dataclass
from ray.actor import ActorHandle
from renderers.configs import CustomRendererConfig
from verifiers.v1.configs.agent import TimeoutConfig
from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine
from vllm.utils.network_utils import get_open_port

from rl2.addition_env import AdditionTaskset
from rl2.benchmark_env import (
    ALFWorldConfig,
    AppWorldConfig,
    BenchmarkTask,
    Tau2Config,
    WebShopConfig,
    load_benchmark_tasks,
)
from rl2.codegym_env import CODEGYM_TRAIN_FILE, CodeGymTask, CodeGymTaskset, CodeGymTasksetConfig
from rl2.configuration import BaseConfig, load_config
from rl2.mlflow_logging import MLflowLogger
from rl2.qwen3 import Parameters, Qwen3LM, load_qwen3_checkpoint
from rl2.toolrl_env import TOOLRL_TRAIN_FILE, ToolRLTaskset, ToolRLTasksetConfig
from rl2.utils import group_advantages

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

    def __post_init__(self) -> None:
        url = httpx.URL(self.manager_url)
        if url.scheme not in {"http", "https"} or not url.host:
            raise ValueError("CodeGym manager_url must be an HTTP(S) URL")


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
    max_tokens: Annotated[int, Field(ge=1)] = 512
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
    # In-flight vLLM requests during weight updates (individual requests, not multi-turn rollout groups):
    # wait: finish requests before applying weights (default).
    # abort: cancel requests before applying weights.
    # keep: freeze requests, then resume them after applying weights.
    weight_update_pause_mode: Literal["wait", "abort", "keep"] = "wait"

    def __post_init__(self) -> None:
        if self.num_groups * self.group_size % (self.minibatches_per_epoch * self.grad_accum_steps):
            raise ValueError("minibatches_per_epoch * grad_accum_steps must divide num_groups * group_size")
        if self.taskset.id == "toolrl" and self.max_turns != 1:
            raise ValueError("ToolRL requires max_turns=1; tool calls are graded without execution")


DEFAULT_CONFIG = Config()


@dataclass(frozen=True)
class TrainingSequence:
    token_ids: jax.Array  # (T,) int32 exact context and generated tokens in one sequence.
    loss_mask: jax.Array  # (T,) bool train on sampled tokens only; exclude prompts and tool results.
    old_log_probs: jax.Array  # (T,) float32 sampling-policy log-probabilities; ignored where loss_mask is False.


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


def format_rollout_examples(groups: list[RolloutGroup], step: int) -> str:
    """Format up to three final replies from the batch in sampling order."""
    sections = [f"Rollout examples | step {step}"]
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


class GRPOBatch(NamedTuple):
    token_ids: jax.Array  # (B, T) Right-padded complete transcripts.
    attention_mask: jax.Array  # (B, T) True for context and sampled tokens, excluding padding.
    loss_mask: jax.Array  # (B, T) True only for sampled tokens.
    old_log_probs: jax.Array  # (B, T) Temperature-scaled sampling-policy log-probabilities.
    advantages: jax.Array  # (B,) Normalized within each rollout group.


# ---


@dataclass(frozen=True)
class RolloutBatch:
    groups: list[RolloutGroup]
    training_batch: GRPOBatch  # Leaves: (B, T), except advantages: (B,).
    transfer_metrics: dict[str, float | int]


# Use a value alias so Ray can serialize script-defined actors without importing __main__.Metrics.
Metrics = dict[str, jax.Array]  # Scalar leaves.


def token_log_probs(
    state: TrainState,
    params: Parameters,  # HF parameter shapes.
    batch: GRPOBatch,
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
    batch: GRPOBatch,
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
    batch: GRPOBatch,  # One optimizer minibatch: (B, T), (B,) leaves.
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

    def accumulate(grads: Parameters, microbatch: GRPOBatch) -> tuple[Parameters, Metrics]:
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
    if not branch.trainable or not any(branch.sampled_mask):
        raise ValueError("Rollout has no trainable tokens")
    for node in branch.nodes:
        if len(node.token_ids) != len(node.mask):
            raise ValueError("Token IDs and loss mask must have equal lengths")
        if node.sampled and not any(node.mask):
            raise ValueError("Sampled messages must contain training tokens")
        # Verifiers otherwise fills missing log-probabilities with zeros.
        if len(node.logprobs) != sum(node.mask):
            raise ValueError("Each sampled token must have a log-probability")
    with jax.default_device(jax.devices("cpu")[0]):
        return TrainingSequence(
            token_ids=jnp.asarray(branch.token_ids, dtype=jnp.int32),  # (T,)
            loss_mask=jnp.asarray(branch.sampled_mask, dtype=jnp.bool_),  # (T,)
            old_log_probs=jnp.asarray(branch.logprobs, dtype=jnp.float32),  # (T,)
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
    # Keep reward arithmetic on CPU so vLLM has the actor GPU to itself.
    with jax.default_device(jax.devices("cpu")[0]):
        rewards = jnp.asarray([trace.reward for trace in traces], dtype=jnp.float32)  # (G,)
        advantages = group_advantages(rewards)  # (G,)
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
            for trace, advantage in zip(traces, advantages.tolist(), strict=True)
        ],
    )


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


# Ray leaves GPUs visible to CPU workers; JAX unpickles arrays onto its default device.
@ray.remote(runtime_env={"env_vars": {"JAX_PLATFORMS": "cpu"}})
class Batcher:
    def __init__(self, actor: ActorHandle) -> None:
        self.actor = actor

    def prepare_batch(self, groups: list[RolloutGroup], context_length: int) -> GRPOBatch:
        """Flatten rollout groups in order and right-pad their transcripts into CPU JAX arrays.

        B is the total rollout count; T is the longest transcript length rounded up
        to a multiple of 128, capped at context_length. Padding is zero/False, and
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
        width = min((length + 127) // 128 * 128, context_length)
        with jax.default_device(jax.devices("cpu")[0]):
            padded: list[dict[str, jax.Array]] = []  # (T,) leaves per rollout.
            for rollout in rollouts:
                sequence = rollout.sequence
                length = len(sequence.token_ids)
                if length < 2 or sequence.loss_mask[0] or not sequence.loss_mask[1:].any():
                    raise ValueError("Each rollout needs context followed by at least one sampled token")
                padded.append(
                    jax.tree.map(
                        partial(jnp.pad, pad_width=(0, width - length)),
                        {
                            "token_ids": sequence.token_ids,  # (L,) unpadded transcript length.
                            "attention_mask": jnp.ones(length, dtype=jnp.bool_),  # (L,)
                            "loss_mask": sequence.loss_mask,  # (L,)
                            "old_log_probs": sequence.old_log_probs,  # (L,)
                        },
                    )
                )
            arrays = jax.tree.map(lambda *values: jnp.stack(values), *padded)  # (B, T) leaves.
            return GRPOBatch(
                **arrays,
                advantages=jnp.asarray([rollout.advantage for rollout in rollouts], dtype=jnp.float32),  # (B,)
            )  # (B, T), (B,) leaves.

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

        # Timestamp each arrival before gather waits for any slower groups.
        results = await asyncio.gather(*(fetch_group() for _ in range(num_groups)))
        fetched = time.monotonic()
        groups = [group for group, _ in results]
        training_batch = self.prepare_batch(groups, context_length=context_length)  # (B, T), (B,) leaves.
        prepared = time.monotonic()
        batch = RolloutBatch(
            groups=groups,
            training_batch=training_batch,
            transfer_metrics={
                "time/batcher/fetch_rollout_seconds": fetched - started,
                "time/batcher/prepare_batch_seconds": prepared - fetched,
                "time/batcher/actor_to_batcher_seconds": mean(seconds for _, seconds in results),
            },
        )
        logger.info("Batch ready: groups=%s elapsed=%.2fs", len(batch.groups), time.monotonic() - started)
        return Transfer(value=batch, ready_at=time.monotonic())


@ray.remote(num_gpus=1)
class Learner:
    def __init__(self, batcher: ActorHandle, actor: ActorHandle, config: Config = DEFAULT_CONFIG) -> None:
        self.config = config
        self.batcher = batcher
        self.actor = actor

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
            writer.log_params(asdict(self.config))
            writer.log_text(yaml.safe_dump(asdict(self.config)), "config/0.yaml")
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
            logger.info("Learner loaded %s parameter tensors", len(self.state.params))
            self.initialize_weight_transfer(resources=resources)
            self.publish_weights()
            writer.log_metrics(
                {
                    "model/params_millions": sum(p.size for p in jax.tree.leaves(self.state.params)) / 1_000_000,
                    "time/learner/setup_seconds": time.monotonic() - start,
                },
                step=0,
            )
            logger.info("Starting GRPO training loop")
            iteration = 0
            total_rollouts = 0
            stop_conditions: set[str] = set()
            while self.config.total_iterations is None or iteration < self.config.total_iterations:
                iteration += 1
                started = time.monotonic()
                batch = self.fetch_next_batch(num_groups=self.config.num_groups, group_size=self.config.group_size)
                fetched = time.monotonic()
                update_metrics = self.optimize(batch.training_batch)
                optimized = time.monotonic()
                self.publish_weights()
                now = time.monotonic()
                rollouts = [rollout for group in batch.groups for rollout in group.rollouts]
                rewards = [rollout.reward for rollout in rollouts]
                stop_counts = Counter(
                    rollout.stop_condition if rollout.stop_condition is not None else "unspecified"
                    for rollout in rollouts
                )
                stop_conditions.update(stop_counts)
                total_rollouts += len(rollouts)
                seconds = max(now - started, 1e-9)
                scalars = {
                    **update_metrics,
                    **batch.transfer_metrics,
                    **{
                        f"charts/stop_condition_fraction/{condition}": stop_counts[condition] / len(rollouts)
                        for condition in sorted(stop_conditions)
                    },
                    "charts/reward_mean": mean(rewards),
                    "charts/reward_std": pstdev(rewards),
                    "charts/reward_min": min(rewards),
                    "charts/reward_max": max(rewards),
                    "charts/degenerate_group_fraction": mean(
                        all(rollout.reward == group.rollouts[0].reward for rollout in group.rollouts)
                        for group in batch.groups
                    ),
                    "charts/turns_mean": mean(rollout.num_turns for rollout in rollouts),
                    "charts/input_tokens_mean": mean(rollout.num_input_tokens for rollout in rollouts),
                    "charts/output_tokens_mean": mean(rollout.num_output_tokens for rollout in rollouts),
                    "charts/total_iterations": iteration,
                    "charts/total_updates": int(self.state.step),
                    "charts/learning_rate": self.config.learning_rate,
                    "charts/total_rollouts": total_rollouts,
                    "time/learner/iterations_per_second": 1 / seconds,
                    "time/learner/rollouts_per_second": len(rollouts) / seconds,
                    "time/learner/output_tokens_per_second": sum(r.num_output_tokens for r in rollouts) / seconds,
                    "time/actor/rollout_seconds": mean(rollout.rollout_seconds for rollout in rollouts),
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
                    examples = format_rollout_examples(batch.groups, step=iteration)
                    logger.info("\n%s", examples)
                    writer.log_text(examples, f"examples/{iteration}.txt")
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
            len(batch.groups),
            seconds,
        )
        return RolloutBatch(
            groups=batch.groups,
            training_batch=batch.training_batch,
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

    def optimize(self, batch: GRPOBatch) -> dict[str, float]:  # Batch leaves: (B, T), (B,).
        batch = jax.device_put(batch, NamedSharding(self.mesh, PartitionSpec("batch")))  # (B, T), (B,) leaves.
        if batch.token_ids.shape[0] % self.config.minibatches_per_epoch:
            raise ValueError("minibatches_per_epoch must divide the rollout batch size")
        minibatch_size = batch.token_ids.shape[0] // self.config.minibatches_per_epoch
        history: list[Metrics] = []
        for _ in range(self.config.update_epochs):
            key = jax.random.fold_in(jax.random.key(self.config.seed), self.state.step)  # () typed PRNG key.
            permutation = jax.random.permutation(key, batch.token_ids.shape[0])  # (B,)
            shuffled = jax.tree.map(
                lambda value, permutation=permutation: value[permutation], batch
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
        random.seed(config.seed)
        self._server = ExitStack()
        self._weights_valid = True
        if isinstance(config.taskset, (WebShopConfig, ALFWorldConfig, Tau2Config, AppWorldConfig)):
            self.tasks = load_benchmark_tasks(config.taskset, config.max_turns, config.environment_timeout)
        elif config.taskset.id == "codegym":
            self.tasks = list(CodeGymTaskset(config=CodeGymTasksetConfig(data_file=config.taskset.dataset_file)))
        elif config.taskset.id == "toolrl":
            self.tasks = list(ToolRLTaskset(config=ToolRLTasksetConfig(data_file=config.taskset.dataset_file)))
        else:
            self.tasks = list(AdditionTaskset(config=vf.TasksetConfig()))
        logger.info("Starting actor and vLLM on its assigned GPU")
        with self._server as resources:
            server = resources.enter_context(serve_vllm(model=config.model))
            wait_for_server(server)
            self._server = resources.pop_all()
        logger.info("Actor ready for rollout requests: tasks=%s", len(self.tasks))

    def __ray_shutdown__(self) -> None:
        logger.info("Shutting down actor")
        self._server.close()
        logger.info("Actor resources released")

    async def fetch_next_group(self, group_size: int) -> Transfer[RolloutGroup]:
        if not self._weights_valid:
            raise RuntimeError("Cannot generate during or after a failed weight update")
        task = random.choice(self.tasks)
        logger.debug("Selected task %s: %r", task.data.id, task.data.prompt)
        group = await self.collect_group(task, group_size=group_size)
        return Transfer(value=group, ready_at=time.monotonic())

    async def initialize_weight_transfer(self, init_info: dict[str, Any]) -> None:
        async with httpx.AsyncClient(base_url=SERVER_URL, timeout=600) as client:
            response = await client.post("/init_weight_transfer_engine", json={"init_info": init_info})
            response.raise_for_status()

    async def update_weights(self, update_info: dict[str, Any]) -> None:
        self._weights_valid = False
        logger.info("Receiving learner weights: tensors=%s", len(update_info["names"]))
        async with httpx.AsyncClient(base_url=SERVER_URL, timeout=600) as client:
            response = await client.post(
                "/pause", params={"mode": self.config.weight_update_pause_mode, "clear_cache": "true"}
            )
            response.raise_for_status()
            response = await client.post("/start_weight_update")
            response.raise_for_status()
            response = await client.post("/update_weights", json={"update_info": update_info})
            response.raise_for_status()
            response = await client.post("/finish_weight_update")
            response.raise_for_status()
            response = await client.post("/resume")
            response.raise_for_status()
        self._weights_valid = True
        logger.info("Learner weights applied; generation resumed")

    async def collect_group(self, task: vf.Task, group_size: int) -> RolloutGroup:
        if group_size < 2:
            raise ValueError("group_size must be at least 2")
        started = time.monotonic()
        logger.info("Collecting group: task=%s group_size=%s", task.data.id, group_size)
        try:
            async with create_agent(self.config) as agent, asyncio.TaskGroup() as pending:
                if isinstance(task, BenchmarkTask) and isinstance(
                    self.config.taskset, (WebShopConfig, ALFWorldConfig, Tau2Config, AppWorldConfig)
                ):
                    runs = [
                        pending.create_task(
                            task.run(
                                agent,
                                self.config.taskset,
                                self.config.max_turns,
                                self.config.environment_timeout,
                            )
                        )
                        for _ in range(group_size)
                    ]
                elif isinstance(task, CodeGymTask) and self.config.taskset.id == "codegym":
                    runs = [
                        pending.create_task(task.run(agent, self.config.taskset.manager_url)) for _ in range(group_size)
                    ]
                else:
                    runs = [pending.create_task(agent.run(task.defer_scoring())) for _ in range(group_size)]
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
    # The HTTP endpoint is shared over loopback by workers on this node.
    with ray.init(address="local", logging_config=ray.LoggingConfig(log_level="INFO")):
        logger.info("Creating batcher, learner, and actor")
        actor = Actor.remote(config=config)
        batcher = Batcher.remote(actor=actor)
        learner = Learner.remote(batcher=batcher, actor=actor, config=config)
        try:
            logger.info("Starting learner")
            ray.get(learner.start.remote())
        finally:
            logger.info("Requesting graceful learner and actor termination")
            shutdown = [learner.__ray_terminate__.remote(), actor.__ray_terminate__.remote()]
            ray.wait(shutdown, num_returns=len(shutdown))
            logger.info("Learner and actor termination finished")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GRPO with MLflow logging")
    parser.add_argument("--config", required=True, help="Path to YAML configuration")
    args = parser.parse_args()
    train(load_config(args.config, Config))


if __name__ == "__main__":
    main()
