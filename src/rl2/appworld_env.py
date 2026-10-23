"""AppWorld tool-calling episodes, isolated in the configured Python environment."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, ClassVar

from pydantic import Field, TypeAdapter, ValidationError

from rl2.agent_task import Environment, ResetResult, StepResult, Task, ToolCall, ToolCallError, ToolSchema
from rl2.benchmark_env import PREFIX, AppWorldConfig, BenchmarkSession, Observation, worker_environment


def tool_schemas() -> list[ToolSchema]:
    return [
        {
            "type": "function",
            "function": {
                "name": "execute",
                "description": "Execute Python in AppWorld. Variables persist between calls; print results to see them.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string", "description": "Python source without Markdown fences."}},
                    "required": ["code"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def validate_tool_call(call: ToolCall) -> None:
    if call.name != "execute":
        raise ToolCallError(f"Unknown AppWorld tool: {call.name}")
    if call.arguments.keys() != {"code"}:
        raise ToolCallError("execute requires exactly one argument: code")
    try:
        TypeAdapter(dict[str, str]).validate_python(call.arguments, strict=True)
    except ValidationError as error:
        raise ToolCallError(str(error)) from error


def worker_command(config: AppWorldConfig) -> list[str]:
    worker = Path(__file__).with_name("benchmark_workers") / "appworld_worker.py"
    # The agent loop owns the rollout budget; AppWorld requires an integer limit.
    settings = asdict(config) | {"max_steps": sys.maxsize, "tool_calls": True}
    return [config.python, str(worker), json.dumps(settings)]


@dataclass(frozen=True, slots=True)
class AppWorldTask(Task):
    id: str = field()
    config: AppWorldConfig
    name: ClassVar[str] = "appworld"

    @asynccontextmanager
    async def init_env(self) -> AsyncGenerator[AppWorldSession]:
        async with AppWorldSession(task=self) as environment:
            yield environment


def load_appworld_tasks(config: AppWorldConfig, timeout: float) -> list[AppWorldTask]:
    result = subprocess.run(
        [*worker_command(config), "--list"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
        timeout=timeout,
        env=worker_environment(config),
    )
    lines = [line[len(PREFIX) :] for line in result.stdout.splitlines() if line.startswith(PREFIX)]
    if len(lines) != 1:
        raise ValueError("AppWorld worker must return one training task list")
    ids = TypeAdapter(list[Annotated[str, Field(min_length=1)]]).validate_json(lines[0])
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("AppWorld training task IDs must be nonempty and unique")
    return [AppWorldTask(id=task_id, config=config) for task_id in ids]


class AppWorldSession(BenchmarkSession, Environment):
    """One subprocess per rollout, with asynchronous requests and bounded cleanup."""

    def __init__(self, task: AppWorldTask) -> None:
        super().__init__(command=worker_command(task.config), timeout=None, env=worker_environment(task.config))
        self.task = task

    async def reset(self, *, timeout: float | None = None) -> ResetResult:
        async with asyncio.timeout(timeout):
            initial = Observation.model_validate(await self.request(op="reset", task_id=self.task.id))
        assert initial.reward == 0.0, "AppWorld reset must not award a reward"
        assert not initial.done, "AppWorld reset must allow at least one step"
        return ResetResult(
            tools=tool_schemas(),
            system_prompt="Use the execute tool once per turn. Pass Python source in code without Markdown fences.",
            prompt=initial.observation,
        )

    async def step(self, action: ToolCall, *, timeout: float | None = None) -> StepResult:
        validate_tool_call(action)
        async with asyncio.timeout(timeout):
            result = Observation.model_validate(await self.request(op="step", action=action.arguments["code"]))
        return StepResult(observation=result.observation, reward=result.reward, done=result.done)
