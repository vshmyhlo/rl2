"""Reasoning Gym puzzles as single-submission tool-calling episodes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Generator
from contextlib import AsyncExitStack, aclosing, asynccontextmanager
from dataclasses import asdict, dataclass, field
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
from rl2.calculator import calculate
from rl2.sandbox.kubernetes import PythonSession


@config_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class ReasoningGymConfig:
    id: Literal["reasoning_gym"] = "reasoning_gym"
    task: Annotated[str, Field(min_length=1)] = "leg_counting"
    task_config: dict[str, Any] = field(default_factory=dict)
    python_namespace: Annotated[str, Field(min_length=1)] = "sandbox"
    python_kubectl: Annotated[tuple[Annotated[str, Field(min_length=1)], ...], Field(min_length=1)] = ("kubectl",)

    def __post_init__(self) -> None:
        if {"seed", "size"} & self.task_config.keys():
            raise ValueError("task_config cannot override seed or size")


def create_dataset(config: ReasoningGymConfig, seed: int) -> ProceduralDataset:
    settings = dict(config.task_config)
    if config.task == "composite" and "datasets" in settings:
        datasets = TypeAdapter(list[DatasetSpec]).validate_python(settings["datasets"])
        for dataset in datasets:
            if {"seed", "size"} & dataset.config.keys():
                raise ValueError(f"Composite dataset {dataset.name!r} config cannot override seed or size")
        settings["datasets"] = datasets
    return reasoning_gym.create_dataset(config.task, seed=seed, size=1, **settings)


def tool_schemas() -> list[ToolSchema]:
    return [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": (
                    "Evaluate arithmetic using numbers, parentheses, and + - * / // % ** only. "
                    "Expressions are limited to 256 characters, exponent magnitudes to 100, "
                    "and intermediate/result magnitudes to 1e100. Returns a result or error without ending the episode."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Arithmetic expression to evaluate."}
                    },
                    "required": ["expression"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "python",
                "description": (
                    "Execute Python statements in an isolated, network-disabled sandbox. Variables, imports, "
                    "and files persist between calls in this episode. Use print() to return values. "
                    "Returns JSON with output (stdout/stderr), exit_code (0 for success, 1 for a Python exception), "
                    "and output_truncated. Exceptions preserve state. Calls have a 5-second execution limit; "
                    "a timeout ends the rollout. Code and output are limited to 16384 bytes each."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string", "description": "Python statements to execute."}},
                    "required": ["code"],
                    "additionalProperties": False,
                },
            },
        },
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
        },
    ]


def validate_tool_call(call: ToolCall) -> None:
    arguments = {"calculator": "expression", "python": "code", "submit_answer": "answer"}
    if call.name not in arguments:
        raise ToolCallError(f"Unknown Reasoning Gym tool: {call.name}")
    argument = arguments[call.name]
    if call.arguments.keys() != {argument}:
        raise ToolCallError(f"{call.name} requires exactly one argument: {argument}")
    try:
        TypeAdapter(dict[str, str]).validate_python(call.arguments, strict=True)
    except ValidationError as error:
        raise ToolCallError(str(error)) from error
    if call.name == "python":
        code = call.arguments["code"]
        if len(code.encode()) > 16384 or "\0" in code:
            raise ToolCallError("code must be at most 16384 UTF-8 bytes and contain no NUL characters")


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
            session = await resources.enter_async_context(
                aclosing(ReasoningGymSession(actor=actor, config=self.config))
            )
            yield session


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
                "Solve the reasoning problem. You may use calculator for arithmetic and python for computation "
                "before submitting your answer. Python variables persist across calls in this episode; "
                "use print() to display results. Python has no network access. "
                "Close </think> before each tool call. Call submit_answer exactly once "
                "with your final answer as a string in the format requested by the question. "
                "Put reasoning outside the answer argument. Use a complete <tool_call> JSON block."
            ),
            prompt=self.entry["question"],
        )

    def step(self, action: ToolCall) -> StepResult:
        if self.done:
            raise RuntimeError("Reset the episode before calling a tool")
        if action.name == "calculator":
            return StepResult(observation=calculate(action.arguments["expression"]), reward=0.0, done=False)
        if action.name != "submit_answer":
            raise ToolCallError(f"Unsupported scoring worker tool: {action.name}")
        reward = float(self.dataset.score_answer(answer=action.arguments["answer"], entry=self.entry))
        self.done = True
        return StepResult(observation="Answer submitted.", reward=reward, done=True)


class ReasoningGymSession(Environment):
    def __init__(self, actor: ActorHandle, config: ReasoningGymConfig) -> None:
        self.actor = actor
        self.config = config
        self._resources = AsyncExitStack()
        self._python: PythonSession | None = None
        self._done = True

    async def aclose(self) -> None:
        self._done = True
        await self._resources.aclose()
        self._python = None

    async def reset(self, *, timeout: float | None = None) -> ResetResult:
        await self.aclose()
        async with asyncio.timeout(timeout):
            result = await self.actor.reset.remote()
            self._done = False
            return result

    async def step(self, action: ToolCall, *, timeout: float | None = None) -> StepResult:
        validate_tool_call(action)
        if self._done:
            raise RuntimeError("Reset the episode before calling a tool")
        async with asyncio.timeout(timeout):
            if action.name == "python":
                if self._python is None:
                    self._python = await self._resources.enter_async_context(
                        PythonSession(namespace=self.config.python_namespace, kubectl=self.config.python_kubectl)
                    )
                result = await self._python.execute(action.arguments["code"])
                return StepResult(observation=json.dumps(asdict(result)), reward=0.0, done=False)
            step = await self.actor.step.remote(action)
            self._done = step.done
            return step
