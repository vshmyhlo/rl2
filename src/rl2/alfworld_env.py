from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager
from dataclasses import dataclass, field, replace
from os.path import expandvars
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar, Literal, Self

import ray
import yaml
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
from pydantic import ConfigDict, TypeAdapter, ValidationError
from pydantic.dataclasses import dataclass as config_dataclass
from ray.actor import ActorHandle

from rl2.agent_task import Environment, EpisodeResult, Task, ToolCall, ToolCallError, ToolSchema


@config_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class ALFWorldConfig:
    config_file: str
    id: Literal["alfworld"] = "alfworld"


def load_environment_config(config_file: str, max_steps: int) -> dict[str, Any]:
    config_path = Path(config_file).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    data_path = Path(expandvars(config["dataset"]["data_path"])).expanduser()
    # Resolve dataset paths relative to the configuration file.
    config["dataset"]["data_path"] = str((config_path.parent / data_path).resolve())
    config["dataset"]["num_train_games"] = -1
    config["env"].update(task_types=[1, 2, 3, 4, 5, 6], domain_randomization=False)
    # DQN mode omits expert demonstrations from observations.
    config["general"]["training_method"] = "dqn"
    config["rl"]["training"]["max_nb_steps_per_episode"] = max_steps
    return config


def discover_games(config_file: str, max_steps: int) -> list[str]:
    config = load_environment_config(config_file, max_steps)
    return sorted(AlfredTWEnv(config, train_eval="train").game_files)


discover_games_remote = ray.remote(max_retries=0)(discover_games)


class ALFWorld:
    def __init__(self, config_file: str, max_steps: int, task_id: str) -> None:
        config = load_environment_config(config_file, max_steps)
        game = Path(task_id).resolve()
        if not game.is_relative_to(config["dataset"]["data_path"]):
            raise ValueError("ALFWorld task is outside the training split")
        # Keep upstream validation, but scan only this episode's directory.
        config["dataset"]["data_path"] = str(game.parent)
        self.loader = AlfredTWEnv(config, train_eval="train")
        if str(game) not in self.loader.game_files:
            raise ValueError("ALFWorld task is not an eligible training game")
        self.loader.game_files = [str(game)]
        self.loader.num_games = 1
        self.resources = ExitStack()

    def observe(self, observation: str, info: dict[str, Any], done: bool = False) -> EpisodeResult:
        self.commands = info["admissible_commands"][0]
        return EpisodeResult(
            observation=observation + "\nAdmissible commands: " + ", ".join(self.commands),
            reward=float(info["won"][0]),
            done=done or bool(info["won"][0]),
        )

    def reset(self) -> EpisodeResult:
        self.resources.close()
        self.env = self.loader.init_env(batch_size=1)
        self.resources.callback(self.env.close)
        observations, info = self.env.reset()
        return self.observe(observations[0], info)

    def step(self, action: str) -> EpisodeResult:
        observations, _, dones, info = self.env.step([action.strip()])
        return self.observe(observations[0], info, bool(dones[0]))

    def close(self) -> None:
        self.resources.close()


@dataclass(frozen=True)
class ActionTool:
    description: str
    parameters: tuple[str, ...]
    commands: tuple[str, ...]


ACTION_TOOLS = {
    "look": ActionTool(
        description="Read the goal and starting observation first; later, look around your current location.",
        parameters=(),
        commands=("look",),
    ),
    "go_to": ActionTool(
        description="Go to a named location, such as 'shelf 1'.",
        parameters=("location",),
        commands=("go to {location}",),
    ),
    "take": ActionTool(
        description="Take an object from a location. Include numeric IDs in both names.",
        parameters=("object", "location"),
        commands=("take {object} from {location}",),
    ),
    "move": ActionTool(
        description="Place a carried object at a location or inside a receptacle.",
        parameters=("object", "location"),
        commands=(
            "move {object} to {location}",
            "put {object} in/on {location}",
            "put {object} into {location}",
            "put {object} in {location}",
        ),
    ),
    "open_container": ActionTool(
        description="Open a named container.",
        parameters=("container",),
        commands=("open {container}",),
    ),
    "close_container": ActionTool(
        description="Close a named container.",
        parameters=("container",),
        commands=("close {container}",),
    ),
    "clean": ActionTool(
        description="Clean a carried object using a sink basin.",
        parameters=("object", "location"),
        commands=("clean {object} with {location}",),
    ),
    "heat": ActionTool(
        description="Heat a carried object using a microwave.",
        parameters=("object", "location"),
        commands=("heat {object} with {location}",),
    ),
    "cool": ActionTool(
        description="Cool a carried object using a fridge.",
        parameters=("object", "location"),
        commands=("cool {object} with {location}",),
    ),
    "use": ActionTool(
        description="Toggle a named device, such as a lamp.",
        parameters=("object",),
        commands=("use {object}",),
    ),
    "examine": ActionTool(
        description="Examine a named object or location.",
        parameters=("target",),
        commands=("examine {target}",),
    ),
    "inventory": ActionTool(
        description="List the objects you are carrying.",
        parameters=(),
        commands=("inventory",),
    ),
}


def tool_schemas() -> list[ToolSchema]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {parameter: {"type": "string"} for parameter in tool.parameters},
                    "required": list(tool.parameters),
                    "additionalProperties": False,
                },
            },
        }
        for name, tool in ACTION_TOOLS.items()
    ]


def validate_tool_call(call: ToolCall) -> None:
    if call.name not in ACTION_TOOLS:
        raise ToolCallError(f"Unknown ALFWorld tool: {call.name}")
    parameters = ACTION_TOOLS[call.name].parameters
    if call.arguments.keys() != set(parameters):
        raise ToolCallError(f"{call.name} requires exactly these arguments: {', '.join(parameters) or '(none)'}")
    try:
        TypeAdapter(dict[str, str]).validate_python(call.arguments, strict=True)
    except ValidationError as error:
        raise ToolCallError(str(error)) from error


class ALFWorldTools:
    def __init__(self, environment: ALFWorld, max_steps: int) -> None:
        self.environment = environment
        self.max_steps = max_steps

    def reset(self) -> EpisodeResult:
        self.current = self.environment.reset()
        self.started = False
        self.steps = 0
        return self.current

    def call(self, call: ToolCall) -> EpisodeResult:
        if self.current.done:
            return self.current
        if call.name == "look" and not self.started:
            self.started = True
            return self.current
        self.started = True
        commands = [command.format(**call.arguments) for command in ACTION_TOOLS[call.name].commands]
        command = next((command for command in commands if command in self.environment.commands), commands[0])
        self.current = self.environment.step(command)
        self.steps += 1
        if self.steps >= self.max_steps:
            self.current = replace(self.current, done=True)
        return self.current


@dataclass(frozen=True, slots=True)
class ALFWorldTask(Task):
    id: str = field()  # A required slot overrides the abstract property.
    config_file: str
    name: ClassVar[str] = "alfworld"
    system_prompt: ClassVar[str] = (
        "Use one action tool per turn. Use the exact object and location names, including IDs."
    )
    prompt: ClassVar[str] = (
        "Call look to read your goal and surroundings, then use the action tools to complete the task."
    )

    @property
    def tools(self) -> list[ToolSchema]:
        return tool_schemas()

    @asynccontextmanager
    async def init_env(self, *, max_steps: int, timeout: float) -> AsyncGenerator[ALFWorldSession]:
        async with ALFWorldSession(task=self, max_steps=max_steps, timeout=timeout) as environment:
            yield environment


def load_alfworld_tasks(config: ALFWorldConfig, max_steps: int, timeout: float) -> list[ALFWorldTask]:
    config_file = str(Path(config.config_file).expanduser().resolve())
    with ExitStack() as resources:
        pending = discover_games_remote.remote(config_file, max_steps)
        resources.callback(ray.cancel, pending, force=True)
        ids = ray.get(pending, timeout=timeout)
    if not ids or any(not task_id for task_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("ALFWorld training task IDs must be nonempty and unique")
    return [ALFWorldTask(id=task_id, config_file=config_file) for task_id in ids]


@ray.remote(num_cpus=1)
class ALFWorldWorker:
    def __init__(self, task: ALFWorldTask, max_steps: int) -> None:
        self.environment = ALFWorld(config_file=task.config_file, max_steps=max_steps, task_id=task.id)
        self.tools = ALFWorldTools(environment=self.environment, max_steps=max_steps)

    def reset(self) -> EpisodeResult:
        return self.tools.reset()

    def step(self, action: ToolCall) -> EpisodeResult:
        return self.tools.call(action)

    def close(self) -> None:
        self.environment.close()

    def __ray_shutdown__(self) -> None:
        self.close()


class ALFWorldSession(Environment):
    """One Ray actor per rollout, with asynchronous requests and bounded cleanup."""

    def __init__(self, task: ALFWorldTask, max_steps: int, timeout: float) -> None:
        self.task = task
        self.max_steps = max_steps
        self.timeout = timeout

    async def __aenter__(self) -> Self:
        self.actor: ActorHandle = ALFWorldWorker.remote(task=self.task, max_steps=self.max_steps)
        return self

    async def reset(self) -> EpisodeResult:
        async with asyncio.timeout(self.timeout):
            return await self.actor.reset.remote()

    async def step(self, action: ToolCall) -> EpisodeResult:
        validate_tool_call(action)
        async with asyncio.timeout(self.timeout):
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
                async with asyncio.timeout(5):
                    await self.actor.close.remote()
