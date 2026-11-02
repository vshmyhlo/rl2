"""CodeGym tasks backed by the upstream HTTP manager and environment workers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from random import Random
from typing import Annotated, Any, ClassVar, Literal, Self
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from datasets import load_dataset
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError, model_validator
from pydantic.dataclasses import dataclass as config_dataclass

from rl2.agent_task import Environment, ResetResult, StepResult, Task, ToolCall, ToolCallError, ToolSchema

CODEGYM_TRAIN_FILE = (
    "hf://datasets/VanishD/CodeGym@85286359a342f7a288aea74273772b69b9b784c2/task_en_instruction_en_env/train-*.parquet"
)
MANAGER_URL = "http://codegym.codegym.svc.cluster.local:8000"


@config_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class CodeGymConfig:
    id: Literal["codegym"] = "codegym"
    manager_url: Annotated[str, Field(pattern=r"^https?://", min_length=1)] = MANAGER_URL
    dataset_file: Annotated[str, Field(min_length=1)] = CODEGYM_TRAIN_FILE
    step_timeout: Annotated[float, Field(ge=3)] = 10


class SystemMessage(BaseModel):
    role: Literal["system"]
    content: str = Field(min_length=1)


class UserMessage(BaseModel):
    role: Literal["user"]
    content: str = Field(min_length=1)


class CodeGymRow(BaseModel):
    prompt: tuple[SystemMessage, UserMessage]
    ability: str
    plugin_call_max_round: int = Field(default=256, ge=1)

    @model_validator(mode="after")
    def validate_task(self) -> Self:
        parts = self.ability.split("@", 2)
        if len(parts) != 3 or parts[0] != "codegym_v1" or not parts[1]:
            raise ValueError("Expected codegym_v1@environment_name@JSON_config ability")
        TypeAdapter(dict[str, JsonValue]).validate_json(parts[2])
        if not self.prompt[1].content.partition("Question:")[2].strip():
            raise ValueError("Expected a nonempty question after Question:")
        return self

    @property
    def question(self) -> str:
        return self.prompt[1].content.partition("Question:")[2].strip()


class CodeGymAction(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    name: str = Field(min_length=1)
    parameters: dict[str, JsonValue]


class Observation(BaseModel):
    observation: str


class Done(BaseModel):
    done: bool


class Reward(BaseModel):
    reward: float


async def request_json(client: httpx.AsyncClient, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    response = await client.request(method, path, **kwargs)
    response.raise_for_status()
    return response.json()


@dataclass(frozen=True)
class CodeGymSession:
    client: httpx.AsyncClient
    session_id: str
    observation: str
    tools: list[ToolSchema]


@asynccontextmanager
async def codegym_session(manager_url: str, ability: str) -> AsyncGenerator[CodeGymSession]:
    async with AsyncExitStack() as resources:
        manager = await resources.enter_async_context(httpx.AsyncClient(base_url=manager_url, timeout=120))
        environment_name = ability.split("@", 2)[1]
        response = await request_json(manager, "GET", f"/tools/{environment_name}")
        tools = TypeAdapter(list[ToolSchema]).validate_python(response["tools"])
        if not tools or len({tool["function"]["name"] for tool in tools}) != len(tools):
            raise ValueError("CodeGym server returned empty or duplicate tool definitions")
        instance = await request_json(manager, "GET", "/get_instance")
        resources.push_async_callback(
            request_json, manager, "GET", "/release_instance", params={"uid": instance["uid"]}, timeout=10
        )
        worker_url = httpx.URL(manager_url).copy_with(port=int(instance["port"]))
        worker = await resources.enter_async_context(httpx.AsyncClient(base_url=worker_url, timeout=120))
        session_id = str(uuid4())
        # Stop before releasing the worker, including when /start fails partway through.
        resources.push_async_callback(
            request_json, worker, "POST", "/stop", json={"session_id": session_id}, timeout=10
        )
        env_name, env_str = ability.split("@", 1)
        started = Observation.model_validate(
            await request_json(
                worker, "POST", "/start", json={"session_id": session_id, "env_name": env_name, "env_str": env_str}
            )
        )
        yield CodeGymSession(client=worker, session_id=session_id, observation=started.observation, tools=tools)


@dataclass(frozen=True, slots=True)
class CodeGymTask(Task):
    id: str = field()
    config: CodeGymConfig
    row: CodeGymRow
    name: ClassVar[str] = "codegym"

    @asynccontextmanager
    async def init_env(self) -> AsyncGenerator[CodeGymEnvironment]:
        async with AsyncExitStack() as resources:
            yield CodeGymEnvironment(task=self, resources=resources)


class CodeGymEnvironment(Environment):
    def __init__(self, task: CodeGymTask, resources: AsyncExitStack) -> None:
        self.task = task
        self.resources = resources
        self.session: CodeGymSession
        self.steps = 0
        self.score = 0.0
        self.done = True

    async def status(self) -> tuple[bool, float]:
        params = {"session_id": self.session.session_id}
        done = Done.model_validate(await request_json(self.session.client, "GET", "/is_done", params=params))
        reward = Reward.model_validate(await request_json(self.session.client, "GET", "/reward", params=params))
        return done.done, reward.reward

    async def reset(self, *, timeout: float | None = None) -> ResetResult:
        self.done = True
        async with asyncio.timeout(timeout):
            await self.resources.aclose()
            self.session = await self.resources.enter_async_context(
                codegym_session(self.task.config.manager_url, self.task.row.ability)
            )
            done, raw_reward = await self.status()
        assert not done, "CodeGym reset must allow at least one step"
        assert raw_reward == 0.0, "CodeGym reset must not award a reward"
        self.steps = 0
        self.score = 0.0
        self.done = False
        return ResetResult(
            tools=self.session.tools,
            system_prompt=(
                "Solve the task using the provided CodeGym tools. Call one function per turn and wait for its result. "
                "If the initial observation lacks the task data, use an available observation tool "
                "to inspect the current instance before acting. "
                "Examples in the question and tool descriptions are illustrative, not the current instance. "
                "Use actual tool results rather than assuming their outputs. "
                "Submit your answer through the documented completion function."
            ),
            prompt=f"{self.task.row.question}\n\nInitial observation:\n{self.session.observation}",
            single_tool_call=True,
        )

    async def step(self, action: ToolCall, *, timeout: float | None = None) -> StepResult:
        if self.done:
            raise RuntimeError("Reset the episode before calling a tool")
        functions = {tool["function"]["name"]: tool["function"] for tool in self.session.tools}
        if action.name not in functions:
            raise ToolCallError(f"Unknown CodeGym tool: {action.name}")
        parameters = functions[action.name]["parameters"]
        missing = set(parameters["required"]) - action.arguments.keys()
        extra = action.arguments.keys() - parameters["properties"].keys()
        if missing or (extra and parameters.get("additionalProperties") is False):
            raise ToolCallError(
                f"Invalid arguments for {action.name}: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        try:
            call = CodeGymAction(name=action.name, parameters=action.arguments)
        except ValidationError as error:
            raise ToolCallError(str(error)) from error
        async with asyncio.timeout(timeout):
            result = Observation.model_validate(
                await request_json(
                    self.session.client,
                    "POST",
                    "/step",
                    timeout=max(120, self.task.config.step_timeout + 10),
                    json={
                        "session_id": self.session.session_id,
                        "action": call.model_dump_json(),
                        "timeout": self.task.config.step_timeout,
                    },
                )
            )
            done, raw_reward = await self.status()
        self.steps += 1
        # Match upstream binary episode success without repeatedly adding its cumulative score.
        score = float(raw_reward > 0)
        reward = score - self.score
        self.score = score
        self.done = done or self.steps >= self.task.row.plugin_call_max_round
        return StepResult(
            observation=result.observation,
            reward=reward,
            done=self.done,
        )


def stream_codegym_tasks(config: CodeGymConfig, rng: Random) -> Generator[CodeGymTask]:
    """Load the catalog at startup; sample rows uniformly with replacement per group."""
    file_type = "parquet" if urlsplit(config.dataset_file).path.endswith(".parquet") else "json"
    dataset = load_dataset(file_type, data_files={"train": config.dataset_file}, split="train")
    if not len(dataset):
        raise ValueError("CodeGym dataset is empty")

    def stream() -> Generator[CodeGymTask]:
        while True:
            index = rng.randrange(len(dataset))
            try:
                row = CodeGymRow.model_validate(dataset[index])
            except ValueError as error:
                raise ValueError(f"Invalid CodeGym row {index}: {error}") from error
            yield CodeGymTask(id=str(index), config=config, row=row)

    return stream()
