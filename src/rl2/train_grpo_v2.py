import argparse
import asyncio
import logging
import os
import random
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from functools import partial
from statistics import mean, pstdev
from typing import Annotated, Any, NamedTuple
from urllib.error import URLError
from urllib.request import urlopen

import httpx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ray
import torch
import verifiers.v1 as vf
import yaml
from flax.training.train_state import TrainState
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass as config_dataclass
from ray.actor import ActorHandle
from verifiers.v1.configs.agent import TimeoutConfig
from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine
from vllm.utils.network_utils import get_open_port

from rl2.addition_env import AdditionTask, AdditionTaskset
from rl2.configuration import BaseConfig, load_config
from rl2.mlflow_logging import MLflowLogger
from rl2.qwen3 import Parameters, Qwen3LM, load_qwen3_checkpoint
from rl2.utils import group_advantages

logger = logging.getLogger(__name__)
MODEL = "Qwen/Qwen3-0.6B"
SERVER_URL = "http://127.0.0.1:8000"
WEIGHT_BUFFER_BYTES = 64 * 1024**2


@config_dataclass(frozen=True)
class Config(BaseConfig):
    model: str = MODEL
    seed: Annotated[int, Field(ge=0)] = 1
    num_groups: Annotated[int, Field(ge=1)] = 1
    group_size: Annotated[int, Field(ge=2)] = 4
    total_iterations: Annotated[int | None, Field(ge=1)] = None
    max_tokens: Annotated[int, Field(ge=1)] = 512
    max_turns: Annotated[int, Field(ge=1)] = 3
    temperature: Annotated[float, Field(ge=0.01, le=2.0)] = 1.0  # vLLM clamps smaller positive values to 0.01.
    learning_rate: Annotated[float, Field(gt=0)] = 1e-6
    clip_coef: Annotated[float, Field(gt=0, lt=1)] = 0.2
    max_grad_norm: Annotated[float, Field(gt=0)] = 1.0
    microbatch_size: Annotated[int, Field(ge=1)] = 1
    update_epochs: Annotated[int, Field(ge=1)] = 1

    def __post_init__(self) -> None:
        if self.num_groups * self.group_size % self.microbatch_size:
            raise ValueError("microbatch_size must divide num_groups * group_size")


DEFAULT_CONFIG = Config()


@dataclass(frozen=True)
class TrainingSequence:
    token_ids: NDArray[np.int32]  # (T,) Exact context and generated tokens in one sequence.
    loss_mask: NDArray[np.bool_]  # (T,) Train on sampled tokens only; exclude prompts and tool results.
    old_log_probs: NDArray[np.float32]  # (T,) Sampling-policy log-probabilities; ignored where loss_mask is False.


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


@dataclass(frozen=True)
class RolloutBatch:
    groups: list[RolloutGroup]
    transfer_metrics: dict[str, float | int]


def rollout_array_bytes(group: RolloutGroup) -> int:
    """Logical sequence-array bytes, excluding text/metadata and Ray serialization overhead."""
    return sum(
        array.nbytes
        for rollout in group.rollouts
        for array in (rollout.sequence.token_ids, rollout.sequence.loss_mask, rollout.sequence.old_log_probs)
    )


class GRPOBatch(NamedTuple):
    token_ids: jax.Array  # (B, T) Right-padded complete transcripts.
    attention_mask: jax.Array  # (B, T) True for context and sampled tokens, excluding padding.
    loss_mask: jax.Array  # (B, T) True only for sampled tokens.
    old_log_probs: jax.Array  # (B, T) Temperature-scaled sampling-policy log-probabilities.
    advantages: jax.Array  # (B,) Normalized within each rollout group.


type Metrics = dict[str, jax.Array]  # Scalar leaves.


def prepare_batch(groups: list[RolloutGroup], context_length: int) -> GRPOBatch:
    rollouts = [rollout for group in groups for rollout in group.rollouts]
    if not rollouts:
        raise ValueError("A training batch must contain rollouts")
    length = max(len(rollout.sequence.token_ids) for rollout in rollouts)
    if length > context_length:
        raise ValueError(f"Rollout length {length} exceeds model context length {context_length}")
    width = min(1 << (length - 1).bit_length(), context_length)
    shape = (len(rollouts), width)
    # Stage variable-length host transcripts before a single device transfer.
    tokens = np.zeros(shape, dtype=np.int32)  # (B, T)
    attention_mask, loss_mask = (np.zeros(shape, dtype=np.bool_) for _ in range(2))  # Each (B, T).
    old_log_probs = np.zeros(shape, dtype=np.float32)  # (B, T)
    for index, rollout in enumerate(rollouts):
        sequence = rollout.sequence
        length = len(sequence.token_ids)
        if length < 2 or sequence.loss_mask[0] or not sequence.loss_mask[1:].any():
            raise ValueError("Each rollout needs context followed by at least one sampled token")
        tokens[index, :length] = sequence.token_ids
        attention_mask[index, :length] = True
        loss_mask[index, :length] = sequence.loss_mask
        old_log_probs[index, :length] = sequence.old_log_probs
    arrays = jax.tree.map(
        jnp.asarray,
        {
            "token_ids": tokens,  # (B, T)
            "attention_mask": attention_mask,  # (B, T)
            "loss_mask": loss_mask,  # (B, T)
            "old_log_probs": old_log_probs,  # (B, T)
            "advantages": np.asarray([rollout.advantage for rollout in rollouts], dtype=np.float32),  # (B,)
        },
    )
    return GRPOBatch(**arrays)


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
    selected = jnp.take_along_axis(logits, batch.token_ids[:, 1:, None], axis=-1)[..., 0]  # (B, T-1)
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
    return TrainState.create(
        apply_fn=model.apply,
        params=jax.tree.map(lambda value: value.astype(jnp.float32), params),  # HF parameter shapes.
        tx=optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate)),
    )


@partial(jax.jit, static_argnames=("microbatch_size", "temperature", "clip_coef"))
def update(
    state: TrainState,
    batch: GRPOBatch,
    *,
    microbatch_size: int,
    temperature: float,
    clip_coef: float,
) -> tuple[TrainState, Metrics]:
    if batch.token_ids.shape[0] % microbatch_size:
        raise ValueError("microbatch_size must divide the rollout batch size")
    count = batch.token_ids.shape[0] // microbatch_size
    microbatches = jax.tree.map(
        lambda value: value.reshape(count, microbatch_size, *value.shape[1:]), batch
    )  # (M, B/M, T); advantages: (M, B/M).

    def accumulate(grads: Parameters, microbatch: GRPOBatch) -> tuple[Parameters, Metrics]:
        def loss_fn(params: Parameters) -> tuple[jax.Array, Metrics]:  # (), scalar metric leaves
            return objective(token_log_probs(state, params, microbatch, temperature), microbatch, clip_coef)

        (_, metrics), gradient = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return jax.tree.map(jnp.add, grads, gradient), metrics  # HF parameter shapes; scalar metrics.

    gradients, metrics = jax.lax.scan(
        accumulate, jax.tree.map(jnp.zeros_like, state.params), microbatches
    )  # HF parameter shapes; metric leaves: (M,).
    gradients = jax.tree.map(lambda value: value / count, gradients)  # HF parameter shapes.
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
    return TrainingSequence(
        token_ids=np.asarray(branch.token_ids, dtype=np.int32),  # (T,)
        loss_mask=np.asarray(branch.sampled_mask, dtype=np.bool_),  # (T,)
        old_log_probs=np.asarray(branch.logprobs, dtype=np.float32),  # (T,)
    )


async def score_group(task: AdditionTask, traces: list[vf.Trace]) -> RolloutGroup:
    logger.debug("Scoring %s rollouts for task %s", len(traces), task.data.id)
    if len(traces) < 2:
        raise ValueError("A rollout group needs at least two traces")
    for trace in traces:
        if not trace.ok:
            raise RuntimeError(f"Rollout failed: {trace.stop_condition}: {trace.errors}")
    # Addition rewards need only the completed trace, so score after all rollouts finish.
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
                base_url=f"{SERVER_URL}/v1", api_key_var="VLLM_API_KEY", renderer_model_name=config.model
            ),
            sampling=vf.SamplingConfig(
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=1.0,
                chat_template_kwargs={"enable_thinking": False},
            ),
            max_turns=config.max_turns,
            timeout=TimeoutConfig(setup=120, rollout=120, finalize=30, scoring=30),
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


@ray.remote
class Batcher:
    def __init__(self, actor: ActorHandle) -> None:
        self.actor = actor

    async def fetch_next_batch(self, num_groups: int, group_size: int) -> Transfer[RolloutBatch]:
        if num_groups < 1:
            raise ValueError("num_groups must be at least 1")
        started = time.monotonic()
        logger.info("Fetching batch: num_groups=%s group_size=%s", num_groups, group_size)

        async def fetch_group() -> tuple[RolloutGroup, float]:
            result = await self.actor.fetch_next_group.remote(group_size=group_size)
            received_at = time.monotonic()
            seconds = received_at - result.ready_at
            logger.info(
                "Ray transfer Actor -> Batcher: task=%s seconds=%.6f array_bytes=%s",
                result.value.task_id,
                seconds,
                rollout_array_bytes(result.value),
            )
            return result.value, seconds

        # Timestamp each arrival before gather waits for any slower groups.
        results = await asyncio.gather(*(fetch_group() for _ in range(num_groups)))
        batch = RolloutBatch(
            groups=[group for group, _ in results],
            transfer_metrics={
                "time/actor_to_batcher_seconds_mean": mean(seconds for _, seconds in results),
                "time/actor_to_batcher_seconds_max": max(seconds for _, seconds in results),
                "data/batch_array_bytes": sum(rollout_array_bytes(group) for group, _ in results),
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
            print(f"Run ID: {self.config.run_id}\nMLflow run: {writer.run_id}", flush=True)
            start = time.monotonic()
            logger.info("Loading learner model %s on its assigned GPU", self.config.model)
            device = jax.devices("gpu")[0]
            with jax.default_device(device):
                self.model, variables = load_qwen3_checkpoint(self.config.model, dtype="bfloat16")
                self.state = create_state(self.model, jax.device_put(variables["params"], device), self.config)
            del variables
            logger.info("Learner loaded %s parameter tensors", len(self.state.params))
            self.initialize_weight_transfer(resources=resources)
            self.publish_weights()
            writer.log_metrics(
                {
                    "model/params_millions": sum(p.size for p in jax.tree.leaves(self.state.params)) / 1_000_000,
                    "time/setup_seconds": time.monotonic() - start,
                },
                step=0,
            )
            logger.info("Starting GRPO training loop")
            iteration = 0
            total_rollouts = 0
            while self.config.total_iterations is None or iteration < self.config.total_iterations:
                iteration += 1
                started = time.monotonic()
                batch = self.fetch_next_batch(num_groups=self.config.num_groups, group_size=self.config.group_size)
                fetched = time.monotonic()
                update_metrics = self.optimize(batch.groups)
                optimized = time.monotonic()
                self.publish_weights()
                now = time.monotonic()
                rollouts = [rollout for group in batch.groups for rollout in group.rollouts]
                rewards = [rollout.reward for rollout in rollouts]
                total_rollouts += len(rollouts)
                seconds = max(now - started, 1e-9)
                scalars = {
                    **update_metrics,
                    **batch.transfer_metrics,
                    "charts/reward_mean": mean(rewards),
                    "charts/reward_std": pstdev(rewards),
                    "charts/reward_min": min(rewards),
                    "charts/reward_max": max(rewards),
                    "charts/turns_mean": mean(rollout.num_turns for rollout in rollouts),
                    "charts/input_tokens_mean": mean(rollout.num_input_tokens for rollout in rollouts),
                    "charts/output_tokens_mean": mean(rollout.num_output_tokens for rollout in rollouts),
                    "charts/total_iterations": iteration,
                    "charts/total_updates": int(self.state.step),
                    "charts/learning_rate": self.config.learning_rate,
                    "charts/total_rollouts": total_rollouts,
                    "charts/rollouts_per_second": len(rollouts) / seconds,
                    "charts/output_tokens_per_second": sum(r.num_output_tokens for r in rollouts) / seconds,
                    "time/rollout_seconds_mean": mean(rollout.rollout_seconds for rollout in rollouts),
                    "time/batch_seconds": fetched - started,
                    "time/update_seconds": optimized - fetched,
                    "time/weight_transfer_seconds": now - optimized,
                    "time/iteration_seconds": now - started,
                    "time/elapsed_seconds": now - start,
                }
                print(
                    f"iteration={iteration} reward={scalars['charts/reward_mean']:.3f} "
                    f"batch={scalars['time/batch_seconds']:.2f}s "
                    f"group_transfer_mean={scalars['time/actor_to_batcher_seconds_mean']:.6f}s "
                    f"batch_transfer={scalars['time/batcher_to_learner_seconds']:.6f}s "
                    f"loss={scalars['losses/policy']:.4f} update={scalars['time/update_seconds']:.2f}s "
                    f"weights={scalars['time/weight_transfer_seconds']:.2f}s "
                    f"elapsed={scalars['time/elapsed_seconds']:.2f}s",
                    flush=True,
                )
                writer.log_metrics(scalars, step=iteration)

    def fetch_next_batch(self, num_groups: int, group_size: int) -> RolloutBatch:
        logger.debug("Requesting batch: num_groups=%s group_size=%s", num_groups, group_size)
        result = ray.get(self.batcher.fetch_next_batch.remote(num_groups=num_groups, group_size=group_size))
        received_at = time.monotonic()
        seconds = received_at - result.ready_at
        batch = result.value
        logger.info(
            "Ray transfer Batcher -> Learner: groups=%s seconds=%.6f array_bytes=%s",
            len(batch.groups),
            seconds,
            batch.transfer_metrics["data/batch_array_bytes"],
        )
        return RolloutBatch(
            groups=batch.groups,
            transfer_metrics={**batch.transfer_metrics, "time/batcher_to_learner_seconds": seconds},
        )

    def publish_weights(self) -> None:
        started = time.monotonic()
        params = jax.tree.map(lambda value: value.astype(jnp.bfloat16), self.state.params)  # HF parameter shapes.
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

    def optimize(self, groups: list[RolloutGroup]) -> dict[str, float]:
        batch = prepare_batch(groups, context_length=self.model.config.max_position_embeddings)
        history: list[Metrics] = []
        for _ in range(self.config.update_epochs):
            self.state, metrics = update(
                self.state,
                batch,
                microbatch_size=self.config.microbatch_size,
                temperature=self.config.temperature,
                clip_coef=self.config.clip_coef,
            )
            history.append(metrics)
        metrics = jax.tree.map(lambda *values: jnp.stack(values).mean(), *history)  # Scalar leaves.
        jax.block_until_ready(self.state)
        return {name: float(value) for name, value in metrics.items()}


@ray.remote(num_gpus=1)
class Actor:
    def __init__(self, config: Config = DEFAULT_CONFIG) -> None:
        self.config = config
        random.seed(config.seed)
        self._server = ExitStack()
        self._weights_condition = asyncio.Condition()
        self._active_groups = 0
        self._weights_valid = True
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
        async with self._weights_condition:
            if not self._weights_valid:
                raise RuntimeError("Cannot generate after a failed weight update")
            self._active_groups += 1
        try:
            task = random.choice(self.tasks)
            logger.debug("Selected task %s: %r", task.data.id, task.data.prompt)
            group = await self.collect_group(task, group_size=group_size)
        finally:
            async with self._weights_condition:
                self._active_groups -= 1
                self._weights_condition.notify_all()
        return Transfer(value=group, ready_at=time.monotonic())

    async def initialize_weight_transfer(self, init_info: dict[str, Any]) -> None:
        async with httpx.AsyncClient(base_url=SERVER_URL, timeout=600) as client:
            response = await client.post("/init_weight_transfer_engine", json={"init_info": init_info})
            response.raise_for_status()

    async def update_weights(self, update_info: dict[str, Any]) -> None:
        async with self._weights_condition:
            await self._weights_condition.wait_for(lambda: self._active_groups == 0)
            self._weights_valid = False
            logger.info("Receiving learner weights: tensors=%s", len(update_info["names"]))
            async with httpx.AsyncClient(base_url=SERVER_URL, timeout=600) as client:
                response = await client.post("/pause", params={"mode": "wait", "clear_cache": "true"})
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

    async def collect_group(self, task: AdditionTask, group_size: int) -> RolloutGroup:
        if group_size < 2:
            raise ValueError("group_size must be at least 2")
        started = time.monotonic()
        logger.info("Collecting group: task=%s group_size=%s", task.data.id, group_size)
        try:
            async with create_agent(self.config) as agent, asyncio.TaskGroup() as pending:
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
