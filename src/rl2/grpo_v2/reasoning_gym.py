"""Reasoning Gym puzzles as single-submission tool-calling episodes."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from random import Random
from typing import Annotated, Any, ClassVar, Literal

import ray
import reasoning_gym
from pydantic import ConfigDict, Field, TypeAdapter, ValidationError
from pydantic.dataclasses import dataclass as config_dataclass
from ray.actor import ActorHandle
from reasoning_gym.composite import DatasetSpec
from reasoning_gym.dataset import ProceduralDataset

from rl2.agent_task import Environment, ResetResult, StepResult, Task, ToolCall, ToolCallError, ToolSchema


@config_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class ReasoningGymConfig:
    id: Literal["reasoning_gym"] = "reasoning_gym"
    task: Annotated[str, Field(min_length=1)] = "leg_counting"
    task_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if {"seed", "size"} & self.task_config.keys():
            raise ValueError("task_config cannot override seed or size")


def create_dataset(config: ReasoningGymConfig, seed: int) -> ProceduralDataset:
    settings = dict(config.task_config)
    if config.task == "composite" and "datasets" in settings:
        settings["datasets"] = TypeAdapter(list[DatasetSpec]).validate_python(settings["datasets"])
    return reasoning_gym.create_dataset(config.task, seed=seed, size=1, **settings)


def tool_schemas() -> list[ToolSchema]:
    return [
        {
            "type": "function",
            "function": {
                "name": "submit_answer",
                "description": "Submit your final answer once. This ends the episode.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "description": "Answer in the format the question requests."}
                    },
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def validate_tool_call(call: ToolCall) -> None:
    if call.name != "submit_answer":
        raise ToolCallError(f"Unknown Reasoning Gym tool: {call.name}")
    if call.arguments.keys() != {"answer"}:
        raise ToolCallError("submit_answer requires exactly one argument: answer")
    try:
        TypeAdapter(dict[str, str]).validate_python(call.arguments, strict=True)
    except ValidationError as error:
        raise ToolCallError(str(error)) from error


@dataclass(frozen=True, slots=True)
class ReasoningGymTask(Task):
    config: ReasoningGymConfig
    seed: int
    name: ClassVar[str] = "reasoning_gym"

    @property
    def id(self) -> str:
        return f"{self.config.task}/{self.seed}"

    @asynccontextmanager
    async def init_env(self) -> AsyncGenerator[ReasoningGymSession]:
        async with AsyncExitStack() as resources:
            actor = ReasoningGymWorker.remote(task=self)
            resources.push_async_callback(asyncio.to_thread, ray.kill, actor)
            yield ReasoningGymSession(actor=actor)


def stream_reasoning_gym_tasks(config: ReasoningGymConfig, rng: Random) -> Generator[ReasoningGymTask]:
    while True:
        yield ReasoningGymTask(config=config, seed=rng.getrandbits(32))


@ray.remote
class ReasoningGymWorker:
    def __init__(self, task: ReasoningGymTask) -> None:
        self.task = task
        self.dataset = create_dataset(task.config, task.seed)
        self.entry: dict[str, Any] = {}
        self.done = True

    def reset(self) -> ResetResult:
        self.entry = self.dataset[0]
        self.done = False
        return ResetResult(
            tools=tool_schemas(),
            system_prompt=(
                "Solve the reasoning problem. Close </think> before calling submit_answer exactly once "
                "with your final answer as a string in the format requested by the question. "
                "Put reasoning outside the answer argument. Use a complete <tool_call> JSON block."
            ),
            prompt=self.entry["question"],
        )

    def step(self, answer: str) -> StepResult:
        if self.done:
            raise RuntimeError("Reset the episode before submitting an answer")
        reward = float(self.dataset.score_answer(answer=answer, entry=self.entry))
        self.done = True
        return StepResult(observation="Answer submitted.", reward=reward, done=True)


class ReasoningGymSession(Environment):
    def __init__(self, actor: ActorHandle) -> None:
        self.actor = actor

    async def reset(self, *, timeout: float | None = None) -> ResetResult:
        async with asyncio.timeout(timeout):
            return await self.actor.reset.remote()

    async def step(self, action: ToolCall, *, timeout: float | None = None) -> StepResult:
        validate_tool_call(action)
        async with asyncio.timeout(timeout):
            return await self.actor.step.remote(action.arguments["answer"])
