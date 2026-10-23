"""Verifiers rollouts backed by benchmark-specific Python environments."""

import asyncio
import json
import os
import signal
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import TracebackType
from typing import Annotated, Any, Literal, Self

import verifiers.v1 as vf
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic.dataclasses import dataclass

PREFIX = "RL2_BENCHMARK "


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class WebShopConfig:
    python: str
    root: str
    id: Literal["webshop"] = "webshop"


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class Tau2Config:
    python: str
    user_llm: str
    id: Literal["tau2"] = "tau2"
    domains: tuple[Literal["retail", "airline", "telecom"], ...] = ("retail", "airline", "telecom")

    def __post_init__(self) -> None:
        if not self.domains or len(set(self.domains)) != len(self.domains):
            raise ValueError("tau2 domains must be nonempty and unique")


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class AppWorldConfig:
    python: str
    root: str
    id: Literal["appworld"] = "appworld"


BenchmarkConfig = WebShopConfig | Tau2Config | AppWorldConfig


class Observation(BaseModel):
    observation: str
    reward: float = 0.0
    done: bool = False


def worker_command(config: BenchmarkConfig, max_steps: int) -> list[str]:
    worker = Path(__file__).with_name("benchmark_workers") / f"{config.id}_worker.py"
    return [config.python, str(worker), json.dumps(asdict(config) | {"max_steps": max_steps})]


def worker_environment(config: BenchmarkConfig) -> dict[str, str]:
    env = dict(os.environ)
    if isinstance(config, WebShopConfig):
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [config.root, env.get("PYTHONPATH", "")]))
    elif isinstance(config, AppWorldConfig):
        env["APPWORLD_ROOT"] = config.root
    return env


class BenchmarkSession:
    def __init__(self, command: list[str], timeout: float | None, env: dict[str, str]) -> None:
        self.command = command
        self.timeout = timeout
        self.env = env

    async def __aenter__(self) -> Self:
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            start_new_session=True,
            limit=16 * 1024 * 1024,
            env=self.env,
        )
        return self

    async def request(self, **request: Any) -> Any:
        assert self.process.stdin is not None and self.process.stdout is not None
        async with asyncio.timeout(self.timeout):
            self.process.stdin.write((json.dumps(request) + "\n").encode())
            await self.process.stdin.drain()
            while line := await self.process.stdout.readline():
                if line.startswith(PREFIX.encode()):
                    return json.loads(line[len(PREFIX) :])
            raise RuntimeError("Benchmark worker exited without a response; see its stderr")

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=5)
        except TimeoutError:
            # Include children spawned by the benchmark.
            os.killpg(self.process.pid, signal.SIGKILL)
            await self.process.wait()
        if exc_type is None and self.process.returncode != 0:
            raise RuntimeError(f"Benchmark worker failed during cleanup (exit {self.process.returncode})")


class BenchmarkData(vf.TaskData):
    prompt: str
    task_id: str


class BenchmarkTask(vf.Task[BenchmarkData]):
    async def run(self, agent: vf.Agent, config: BenchmarkConfig, max_steps: int, timeout: float) -> vf.Trace:
        async with BenchmarkSession(worker_command(config, max_steps), timeout, worker_environment(config)) as session:
            initial = Observation.model_validate(await session.request(op="reset", task_id=self.data.task_id))
            task = self.with_data(prompt=initial.observation).defer_scoring()
            async with agent.interaction(task) as interaction:
                segment = await interaction.turn()
                for step in range(max_steps):
                    if segment.terminated:
                        break
                    result = Observation.model_validate(await session.request(op="step", action=segment.last_reply))
                    if result.done:
                        interaction.trace.stop(f"{config.id}_done")
                        break
                    if step + 1 == max_steps:
                        interaction.trace.stop(f"{config.id}_max_steps")
                        break
                    segment = await interaction.turn(result.observation)
                reward = TypeAdapter(float).validate_python(await session.request(op="score"))
                interaction.trace.record_reward(f"{config.id}_reward", reward)
            return interaction.trace


def load_benchmark_tasks(config: BenchmarkConfig, max_steps: int, timeout: float) -> list[BenchmarkTask]:
    result = subprocess.run(
        [*worker_command(config, max_steps), "--list"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
        timeout=timeout,
        env=worker_environment(config),
    )
    lines = [line[len(PREFIX) :] for line in result.stdout.splitlines() if line.startswith(PREFIX)]
    if len(lines) != 1:
        raise ValueError("Benchmark worker must return one training task list")
    ids = TypeAdapter(list[Annotated[str, Field(min_length=1)]]).validate_json(lines[0])
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Benchmark training task IDs must be nonempty and unique")
    return [
        BenchmarkTask(data=BenchmarkData(id=f"{config.id}:{task_id}", task_id=task_id, prompt=f"{config.id}:{task_id}"))
        for task_id in ids
    ]
