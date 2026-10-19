import asyncio
import logging
import random
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import jax
import jax.numpy as jnp
import ray
import verifiers.v1 as vf
from ray.actor import ActorHandle
from verifiers.v1.configs.agent import TimeoutConfig

from rl2.addition_env import AdditionTask, AdditionTaskset
from rl2.utils import group_advantages

logger = logging.getLogger(__name__)
MODEL = "Qwen/Qwen3-0.6B"
SERVER_URL = "http://127.0.0.1:8000"


@dataclass(frozen=True)
class TrainingSequence:
    token_ids: list[int]  # (T,) Exact context and generated tokens in one sequence.
    loss_mask: list[bool]  # (T,) Train on sampled tokens only; exclude prompts, tools, and repeated prefixes.
    old_log_probs: list[float]  # (T,) Sampling-policy log-probabilities; ignored where loss_mask is False.


@dataclass(frozen=True)
class Rollout:
    sequences: list[TrainingSequence]  # Training inputs; empty when the backend supplies no token data.
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


def training_sequences(trace: vf.Trace) -> list[TrainingSequence]:
    sequences: list[TrainingSequence] = []
    seen: set[int] = set()
    for branch in trace.branches:
        token_ids = branch.token_ids  # (T,)
        if not branch.trainable or not token_ids:
            continue
        loss_mask: list[bool] = []  # (T,)
        for node in branch.nodes:
            # Shared prefixes remain context, but contribute to the loss only once.
            loss_mask.extend([False] * len(node.mask) if id(node) in seen else node.mask)
            seen.add(id(node))
        if any(loss_mask):
            sequences.append(TrainingSequence(token_ids=token_ids, loss_mask=loss_mask, old_log_probs=branch.logprobs))
    return sequences


async def score_group(task: AdditionTask, traces: list[vf.Trace]) -> RolloutGroup:
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
                sequences=training_sequences(trace),
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
    return vf.make_agent(
        config=vf.AgentConfig(
            model=MODEL,
            harness=vf.HarnessConfig(id="null"),
            runtime=vf.SubprocessConfig(),
            client=vf.EvalClientConfig(base_url=f"{SERVER_URL}/v1", api_key_var="VLLM_API_KEY"),
            sampling=vf.SamplingConfig(
                max_tokens=512, temperature=1.0, chat_template_kwargs={"enable_thinking": False}
            ),
            max_turns=3,
            timeout=TimeoutConfig(setup=120, rollout=120, finalize=30, scoring=30),
        )
    )


@contextmanager
def serve_vllm() -> Iterator[None]:
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
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
        ],
    ) as server:
        try:
            yield
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
    logger.info("vLLM server exited")


def wait_for_server(timeout: float = 600) -> None:
    logger.info("Waiting for vLLM at %s", SERVER_URL)
    deadline = time.monotonic() + timeout
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            with urlopen(f"{SERVER_URL}/health", timeout=min(5, remaining)):
                logger.info("vLLM is ready")
                return
        except (URLError, TimeoutError):
            time.sleep(min(1, max(0, deadline - time.monotonic())))
    raise TimeoutError(f"vLLM was not ready within {timeout}s at {SERVER_URL}")


@ray.remote
class Batcher:
    def submit_group(self, group: RolloutGroup) -> None:
        logger.debug("Batcher received group of %s rollouts (buffering not implemented)", len(group.rollouts))

    def fetch_next_batch(self) -> Any:
        logger.debug("Batcher received batch request (batching not implemented)")


@ray.remote(num_gpus=1)
class Learner:
    def __init__(self, batcher: ActorHandle) -> None:
        self.batcher = batcher

    def start(self) -> None:
        logger.info("Starting learner training loop")
        while True:
            batch = self.fetch_next_batch()
            self.optimize(batch)
            self.publish_weights()

    def fetch_next_batch(self) -> Any:
        logger.debug("Fetching next batch from batcher")
        return ray.get(self.batcher.fetch_next_batch.remote())

    def optimize(self, batch: Any) -> None:
        logger.debug("Optimizing batch (placeholder)")
        time.sleep(1)

    def publish_weights(self) -> None:
        logger.debug("Publishing weights (placeholder)")
        time.sleep(1)


@ray.remote(num_gpus=1)
class Actor:
    def __init__(self, batcher: ActorHandle, group_size: int = 4) -> None:
        if group_size < 2:
            raise ValueError("group_size must be at least 2")
        self.batcher = batcher
        self.group_size = group_size
        self.tasks = list(AdditionTaskset(config=vf.TasksetConfig()))

    def start(self) -> None:
        logger.info("Starting actor and vLLM on its assigned GPU")
        with serve_vllm():
            wait_for_server()
            logger.info("Starting actor rollout loop")
            while True:
                example = self.fetch_next_prompt()
                group = self.collect_group(example)
                logger.debug("Submitting rollout group to batcher")
                ray.get(self.batcher.submit_group.remote(group))

    def fetch_next_prompt(self) -> AdditionTask:
        task = random.choice(self.tasks)
        logger.debug("Selected task %s: %r", task.data.id, task.data.prompt)
        return task

    def collect_group(self, task: AdditionTask) -> RolloutGroup:
        logger.debug("Collecting %s rollouts for task %s", self.group_size, task.data.id)

        async def collect() -> RolloutGroup:
            async with create_agent() as agent, asyncio.TaskGroup() as pending:
                runs = [pending.create_task(agent.run(task.defer_scoring())) for _ in range(self.group_size)]
            group = await score_group(task, [run.result() for run in runs])
            logger.info(
                "Collected group: task=%s rewards=%s advantages=%s",
                task.data.id,
                [rollout.reward for rollout in group.rollouts],
                [rollout.advantage for rollout in group.rollouts],
            )
            return group

        return asyncio.run(collect())


def main() -> None:
    # The HTTP endpoint is shared over loopback by workers on this node.
    with ray.init(address="local", logging_config=ray.LoggingConfig(log_level="INFO")):
        logger.info("Creating batcher, learner, and actor")
        batcher = Batcher.remote()
        learner = Learner.remote(batcher=batcher)
        actor = Actor.remote(batcher=batcher)
        logger.info("Starting learner and actor")
        ray.get([learner.start.remote(), actor.start.remote()])


if __name__ == "__main__":
    main()
