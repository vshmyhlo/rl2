import asyncio
import logging
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import ray
import verifiers.v1 as vf
from ray.actor import ActorHandle
from verifiers.v1.configs.agent import TimeoutConfig

from rl2.addition_env import AdditionData, AdditionTask

logger = logging.getLogger(__name__)
MODEL = "Qwen/Qwen3-0.6B"
SERVER_URL = "http://127.0.0.1:8000"


@dataclass(frozen=True)
class Rollout:
    trace: dict[str, Any]
    reward: float
    stop_condition: str | None


def create_environment() -> vf.Agent:
    return vf.Agent(
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


def serve_vllm() -> None:
    logger.info("Starting vLLM server for %s", MODEL)
    subprocess.run(
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
        check=True,
    )
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
    def submit_rollout(self, rollout: Rollout) -> None:
        logger.debug("Batcher received rollout (buffering not implemented)")

    def fetch_next_batch(self) -> Any:
        logger.debug("Batcher received batch request (batching not implemented)")


@ray.remote
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


@ray.remote
class Actor:
    def __init__(self, batcher: ActorHandle) -> None:
        self.batcher = batcher
        self.tasks = [
            AdditionTask(
                data=AdditionData(
                    id=str(index),
                    prompt=f"What is {a} + {b}?",
                    answer=a + b,
                    system_prompt="Use the add tool, then reply with only the resulting integer.",
                )
            )
            for index, (a, b) in enumerate([(17, 25), (8, 13), (0, 9), (-4, 7), (31, 12)])
        ]

    def start(self) -> None:
        logger.info("Starting actor and submitting vLLM server task")
        self.server = ray.remote(num_gpus=1, max_retries=0)(serve_vllm).remote()
        wait_for_server()
        logger.info("Starting actor rollout loop")
        while True:
            example = self.fetch_next_prompt()
            rollout = self.collect_rollout(example)
            logger.debug("Submitting rollout to batcher")
            ray.get(self.batcher.submit_rollout.remote(rollout))

    def fetch_next_prompt(self) -> AdditionTask:
        task = random.choice(self.tasks)
        logger.debug("Selected task %s: %r", task.data.id, task.data.prompt)
        return task

    def collect_rollout(self, task: AdditionTask) -> Rollout:
        logger.debug("Collecting rollout for task %s", task.data.id)

        async def collect() -> Rollout:
            async with create_environment() as env:
                trace = await env.run(task)
            if not trace.ok:
                raise RuntimeError(f"Rollout failed: {trace.stop_condition}: {trace.errors}")
            logger.info("Collected rollout: task=%s reward=%s", task.data.id, trace.reward)
            return Rollout(
                trace=trace.to_record(float_decimals=None),
                reward=trace.reward,
                stop_condition=trace.stop_condition,
            )

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
