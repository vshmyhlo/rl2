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
from rl2.sandbox.kubernetes import PythonExecutionError, PythonSession, validate_python_code


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
                "name": "python",
                "description": (
                    "Execute Python statements in an isolated, network-disabled sandbox. Variables, imports, "
                    "and files persist between calls in this episode. Use print() to return values. "
                    "Includes the Python standard library, SymPy 1.14.0 (import sympy) for symbolic math "
                    "and equation solving, and Z3 4.15.4 (import z3) for Boolean/integer constraints. "
                    "Packages are preinstalled; no downloads or package installation are available. "
                    "Returns JSON with output (combined stdout/stderr; empty if no output was captured), "
                    "exit_code (0 for success, 1 for a Python exception), and output_truncated "
                    "(true if output was cut off). Python exceptions preserve state and do not end the episode. "
                    "Calls have a 5-second execution limit. A timeout or sandbox execution failure returns "
                    "JSON with an error field and ends the episode. Code and captured output are limited to "
                    "16384 bytes each before JSON encoding."
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
    arguments = {"python": "code", "submit_answer": "answer"}
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
        try:
            validate_python_code(call.arguments["code"])
        except ValueError as error:
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
                "Solve the problem accurately and efficiently. You may use python for arithmetic, symbolic math, "
                "constraint solving, and other computation. Keep reasoning brief.\n\n"
                "Python variables persist across calls. Python has no network access. "
                "Put the problem's input directly into your code; it is not available through "
                "input(), sys.stdin, or pre-existing files. Use print() to display results.\n\n"
                "Before executing code, check that it implements the requested operation and uses the exact input. "
                "After execution, briefly check that the result satisfies the question. "
                "Successful execution alone does not establish correctness.\n\n"
                "If a tool reports an error, identify its cause and change the relevant code. "
                "Do not repeat the same failed call or remove required computation just to make the code run. "
                "If a result contradicts the problem, investigate before submitting. "
                "Once the result is checked, submit without repeating the analysis.\n\n"
                "Use exactly one tool call per turn and wait for its result. "
                "Close </think> before each tool call and use a complete <tool_call> JSON block. "
                "Call submit_answer exactly once, with your final answer as a string in the format requested. "
                "Do not add explanations, extra brackets, or other wrappers inside the answer."
            ),
            prompt=self.entry["question"],
            single_tool_call=True,
        )

    def step(self, action: ToolCall) -> StepResult:
        if self.done:
            raise RuntimeError("Reset the episode before calling a tool")
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
        async with asyncio.timeout(timeout):
            await self.aclose()
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
                try:
                    result = await self._python.execute(action.arguments["code"])
                except PythonExecutionError as error:
                    self._done = True
                    return StepResult(observation=json.dumps({"error": str(error)}), reward=0.0, done=True)
                return StepResult(observation=json.dumps(asdict(result)), reward=0.0, done=False)
            step = await self.actor.step.remote(action)
            self._done = step.done
            return step
