import asyncio
import logging
import random
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from urllib.error import URLError
from urllib.request import urlopen

import jax
import jax.numpy as jnp
import numpy as np
import ray
import verifiers.v1 as vf
from numpy.typing import NDArray
from ray.actor import ActorHandle
from verifiers.v1.configs.agent import TimeoutConfig

from rl2.addition_env import AdditionTask, AdditionTaskset
from rl2.utils import group_advantages

logger = logging.getLogger(__name__)
MODEL = "Qwen/Qwen3-0.6B"
SERVER_URL = "http://127.0.0.1:8000"


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


def create_agent() -> vf.Agent:
    logger.debug("Creating Verifiers agent: model=%s endpoint=%s", MODEL, SERVER_URL)
    return vf.make_agent(
        config=vf.AgentConfig(
            model=MODEL,
            harness=vf.HarnessConfig(id="null"),
            runtime=vf.SubprocessConfig(),
            client=vf.TrainClientConfig(
                base_url=f"{SERVER_URL}/v1", api_key_var="VLLM_API_KEY", renderer_model_name=MODEL
            ),
            sampling=vf.SamplingConfig(
                max_tokens=512, temperature=1.0, chat_template_kwargs={"enable_thinking": False}
            ),
            max_turns=3,
            timeout=TimeoutConfig(setup=120, rollout=120, finalize=30, scoring=30),
        )
    )


@contextmanager
def serve_vllm() -> Iterator[subprocess.Popen[bytes]]:
    logger.info("Starting vLLM server for %s", MODEL)
    # Inherit the GPU assigned to this actor through CUDA_VISIBLE_DEVICES.
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            MODEL,
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
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

    async def fetch_next_batch(self, num_groups: int, group_size: int) -> list[RolloutGroup]:
        if num_groups < 1:
            raise ValueError("num_groups must be at least 1")
        started = time.monotonic()
        logger.info("Fetching batch: num_groups=%s group_size=%s", num_groups, group_size)
        batch = await asyncio.gather(
            *(self.actor.fetch_next_group.remote(group_size=group_size) for _ in range(num_groups))
        )
        logger.info("Batch ready: groups=%s elapsed=%.2fs", len(batch), time.monotonic() - started)
        return batch


@ray.remote(num_gpus=1)
class Learner:
    def __init__(self, batcher: ActorHandle) -> None:
        self.batcher = batcher

    def start(self) -> None:
        logger.info("Starting learner training loop")
        iteration = 0
        while True:
            iteration += 1
            started = time.monotonic()
            logger.info("Starting learner iteration %s", iteration)
            batch = self.fetch_next_batch(num_groups=1, group_size=4)
            self.optimize(batch)
            self.publish_weights()
            logger.info("Learner iteration %s complete: elapsed=%.2fs", iteration, time.monotonic() - started)

    def fetch_next_batch(self, num_groups: int, group_size: int) -> list[RolloutGroup]:
        logger.debug("Requesting batch: num_groups=%s group_size=%s", num_groups, group_size)
        batch = ray.get(self.batcher.fetch_next_batch.remote(num_groups=num_groups, group_size=group_size))
        logger.debug("Learner received %s groups", len(batch))
        return batch

    def optimize(self, batch: list[RolloutGroup]) -> None:
        logger.debug("Optimizing batch (placeholder)")
        time.sleep(1)

    def publish_weights(self) -> None:
        logger.debug("Publishing weights (placeholder)")
        time.sleep(1)


@ray.remote(num_gpus=1)
class Actor:
    def __init__(self) -> None:
        self._server = ExitStack()
        self.tasks = list(AdditionTaskset(config=vf.TasksetConfig()))
        logger.info("Starting actor and vLLM on its assigned GPU")
        with self._server as resources:
            server = resources.enter_context(serve_vllm())
            wait_for_server(server)
            self._server = resources.pop_all()
        logger.info("Actor ready for rollout requests: tasks=%s", len(self.tasks))

    def __ray_shutdown__(self) -> None:
        logger.info("Shutting down actor")
        self._server.close()
        logger.info("Actor resources released")

    async def fetch_next_group(self, group_size: int) -> RolloutGroup:
        task = random.choice(self.tasks)
        logger.debug("Selected task %s: %r", task.data.id, task.data.prompt)
        return await self.collect_group(task, group_size=group_size)

    async def collect_group(self, task: AdditionTask, group_size: int) -> RolloutGroup:
        if group_size < 2:
            raise ValueError("group_size must be at least 2")
        started = time.monotonic()
        logger.info("Collecting group: task=%s group_size=%s", task.data.id, group_size)
        try:
            async with create_agent() as agent, asyncio.TaskGroup() as pending:
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


def main() -> None:
    # The HTTP endpoint is shared over loopback by workers on this node.
    with ray.init(address="local", logging_config=ray.LoggingConfig(log_level="INFO")):
        logger.info("Creating batcher, learner, and actor")
        actor = Actor.remote()
        batcher = Batcher.remote(actor=actor)
        learner = Learner.remote(batcher=batcher)
        try:
            logger.info("Starting learner")
            ray.get(learner.start.remote())
        finally:
            logger.info("Requesting graceful actor termination")
            ray.wait([actor.__ray_terminate__.remote()])
            logger.info("Actor termination finished")


if __name__ == "__main__":
    main()
