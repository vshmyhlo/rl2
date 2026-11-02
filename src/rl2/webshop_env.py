"""WebShop tool-calling episodes backed by one synchronous Ray actor per rollout."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator, Generator
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, closing
from dataclasses import dataclass, field, replace
from pathlib import Path
from random import Random
from types import TracebackType
from typing import Any, ClassVar, Literal, Self

import gymnasium as gym
import ray
from gymnasium.envs.registration import EnvSpec
from pydantic import ConfigDict, TypeAdapter, ValidationError
from pydantic.dataclasses import dataclass as config_dataclass
from ray.actor import ActorHandle

from rl2.agent_task import Environment, ResetResult, StepResult, Task, ToolCall, ToolCallError, ToolSchema


@config_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class WebShopConfig:
    root: str
    id: Literal["webshop"] = "webshop"


def make_environment(root: str) -> gym.Env[str, str]:
    # Gym's entry point keeps optional WebShop imports inside the Ray process.
    return gym.make(
        EnvSpec(
            id="rl2/WebShop-v0",
            entry_point="rl2.grpo_v2.webshop:WebShop",
            order_enforce=False,
            disable_env_checker=True,  # Upstream uses free-form text without Gym spaces.
        ),
        root=root,
    )


def worker_runtime_env(config: WebShopConfig) -> dict[str, Any]:
    return {"env_vars": {"PYTHONPATH": os.pathsep.join(filter(None, [config.root, os.environ.get("PYTHONPATH", "")]))}}


@ray.remote(max_calls=1, max_retries=0)
def discover_tasks(root: str) -> list[str]:
    with closing(make_environment(root)) as environment:
        return environment.get_wrapper_attr("task_ids")()


def tool_schemas() -> list[ToolSchema]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {argument: {"type": "string"}},
                    "required": [argument],
                    "additionalProperties": False,
                },
            },
        }
        for name, argument, description in (
            ("search", "keywords", "Search for products using keywords."),
            ("click", "value", "Click a product, option, or button from the available actions."),
        )
    ]


def validate_tool_call(call: ToolCall) -> None:
    arguments = {"search": "keywords", "click": "value"}
    if call.name not in arguments:
        raise ToolCallError(f"Unknown WebShop tool: {call.name}")
    argument = arguments[call.name]
    if call.arguments.keys() != {argument}:
        raise ToolCallError(f"{call.name} requires exactly one argument: {argument}")
    try:
        TypeAdapter(dict[str, str]).validate_python(call.arguments, strict=True)
    except ValidationError as error:
        raise ToolCallError(str(error)) from error
    if not call.arguments[argument].strip():
        raise ToolCallError(f"{argument} must not be blank")


@dataclass(frozen=True, slots=True)
class WebShopTask(Task):
    id: str = field()
    config: WebShopConfig
    name: ClassVar[str] = "webshop"

    @asynccontextmanager
    async def init_env(self) -> AsyncGenerator[WebShopSession]:
        async with WebShopSession(task=self) as environment:
            yield environment


def stream_webshop_tasks(config: WebShopConfig, rng: Random, timeout: float) -> Generator[WebShopTask]:
    config = replace(config, root=str(Path(config.root).expanduser().resolve()))
    with ExitStack() as resources:
        pending = discover_tasks.options(runtime_env=worker_runtime_env(config)).remote(config.root)
        resources.callback(ray.cancel, pending, force=True)
        ids = ray.get(pending, timeout=timeout)
    if not ids:
        raise ValueError("WebShop training split is empty")

    def stream() -> Generator[WebShopTask]:
        while True:
            yield WebShopTask(id=rng.choice(ids), config=config)

    return stream()


@ray.remote
class WebShopWorker:
    def __init__(self, task: WebShopTask) -> None:
        self.task = task
        self.environment = make_environment(task.config.root)

    def reset(self) -> ResetResult:
        observation, info = self.environment.reset(options={"task_id": self.task.id})
        assert info.get("reward", 0) == 0, "WebShop reset must not award a reward"
        assert not any(info.get(key, False) for key in ("done", "terminated", "truncated")), (
            "WebShop reset must allow at least one step"
        )
        return ResetResult(
            tools=tool_schemas(),
            system_prompt=(
                "Complete the shopping instruction by searching for products and clicking available options. "
                "Use search(keywords) or click(value), with exactly one tool call per turn. "
                "Copy click values from Available actions. Click Buy Now when the product and options match. "
                "Keep reasoning brief and close </think> before a complete <tool_call> JSON block."
            ),
            prompt=observation,
        )

    def step(self, action: ToolCall) -> StepResult:
        argument = "keywords" if action.name == "search" else "value"
        observation, reward, terminated, truncated, _ = self.environment.step(
            f"{action.name}[{action.arguments[argument]}]"
        )
        return StepResult(observation=observation, reward=float(reward), done=bool(terminated or truncated))

    def close(self) -> None:
        self.environment.close()


class WebShopSession(Environment):
    def __init__(self, task: WebShopTask) -> None:
        self.task = task

    async def __aenter__(self) -> Self:
        self.actor: ActorHandle = WebShopWorker.options(runtime_env=worker_runtime_env(self.task.config)).remote(
            task=self.task
        )
        return self

    async def reset(self, *, timeout: float | None = None) -> ResetResult:
        async with asyncio.timeout(timeout):
            return await self.actor.reset.remote()

    async def step(self, action: ToolCall, *, timeout: float | None = None) -> StepResult:
        validate_tool_call(action)
        async with asyncio.timeout(timeout):
            return await self.actor.step.remote(action)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        async with AsyncExitStack() as resources:
            resources.push_async_callback(asyncio.to_thread, ray.kill, self.actor)
            if exc_type is None:
                await self.actor.close.remote()
