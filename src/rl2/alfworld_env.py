from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, closing
from dataclasses import dataclass, field
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

from rl2.agent_task import Environment, ResetResult, StepResult, Task, ToolCall, ToolCallError, ToolSchema

SYSTEM_PROMPT = """You control an agent in an ALFWorld household environment. Complete the task by acting.
Use exactly one action tool per turn, then wait for the observation. Keep reasoning brief.
Always close </think> before writing a complete <tool_call> JSON block outside reasoning.
Every response must include an action; keep acting until the environment ends the episode.

Explore to discover objects. The initial room description lists locations, not their contents.
Use go_to to visit a location and reveal its contents; open closed containers with open_container.
An unseen object needs to be searched for. An empty inventory or a look response saying
"nothing" does not mean the task object is absent from the room. Check unvisited locations.
Track your location, carried object, and searched locations from actual observations.

Choose an action supported by the latest Admissible commands and call its corresponding tool.
Copy exact names and numeric IDs from observations or commands; never invent objects or locations.
For a listed 'go to shelf 1', call:
<tool_call>{"name": "go_to", "arguments": {"location": "shelf 1"}}</tool_call>
Navigate before interacting. 'Nothing happens' means the action failed; check the current
commands and change your action instead of assuming success or repeating the failed action.

For placement tasks, find and take the object, then go to the destination and use move.
For clean/heat/cool tasks, carry the object to the sinkbasin/microwave/fridge and use
clean/heat/cool before placing it. For two-object tasks, deliver two distinct objects.
To 'examine X with a lamp', take X, find the lamp, and use the lamp while carrying X.
The lamp supplies light; the task is not to pick up or examine the lamp.
"""


@config_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class ALFWorldConfig:
    config_file: str
    id: Literal["alfworld"] = "alfworld"


def load_environment_config(config_file: str) -> dict[str, Any]:
    config_path = Path(config_file).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    data_path = Path(expandvars(config["dataset"]["data_path"])).expanduser()
    # Resolve dataset paths relative to the configuration file.
    config["dataset"]["data_path"] = str((config_path.parent / data_path).resolve())
    config["dataset"]["num_train_games"] = -1
    config["env"].update(task_types=[1, 2, 3, 4, 5, 6], domain_randomization=False)
    # DQN mode omits expert demonstrations from observations.
    config["general"]["training_method"] = "dqn"
    # The agent loop owns the rollout budget.
    config["rl"]["training"]["max_nb_steps_per_episode"] = None
    return config


def discover_games(config_file: str) -> list[str]:
    config = load_environment_config(config_file)
    return sorted(AlfredTWEnv(config, train_eval="train").game_files)


discover_games_remote = ray.remote(max_retries=0)(discover_games)


class ALFWorld:
    def __init__(self, config_file: str, task_id: str) -> None:
        config = load_environment_config(config_file)
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

    def observe(self, observation: str, info: dict[str, Any], done: bool = False) -> StepResult:
        self.commands = info["admissible_commands"][0]
        won = bool(info["won"][0])
        return StepResult(
            observation=observation + "\nAdmissible commands: " + ", ".join(self.commands),
            reward=float(won),
            done=done or won,
        )

    def reset(self) -> StepResult:
        self.resources.close()
        self.env = self.resources.enter_context(closing(self.loader.init_env(batch_size=1)))
        observations, info = self.env.reset()
        return self.observe(observations[0], info)

    def step(self, action: str) -> StepResult:
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
        description="Look at your current surroundings; use go_to to search other locations.",
        parameters=(),
        commands=("look",),
    ),
    "go_to": ActionTool(
        description="Visit a location such as 'shelf 1' and observe its contents or whether it is closed.",
        parameters=("location",),
        commands=("go to {location}",),
    ),
    "take": ActionTool(
        description="Pick up a visible object at your current location. Include numeric IDs in both names.",
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
        description="Toggle a nearby device. For lamp tasks, use the lamp while carrying the target object.",
        parameters=("object",),
        commands=("use {object}",),
    ),
    "examine": ActionTool(
        description="Inspect an accessible object or location. Lamp tasks require use on the lamp instead.",
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
    def __init__(self, environment: ALFWorld) -> None:
        self.environment = environment

    def reset(self) -> ResetResult:
        initial = self.environment.reset()
        assert initial.reward == 0.0, "ALFWorld reset must not award a reward"
        assert not initial.done, "ALFWorld reset must allow at least one step"
        return ResetResult(
            tools=tool_schemas(),
            system_prompt=SYSTEM_PROMPT,
            prompt=(
                f"Task and starting observation:\n{initial.observation}\n\n"
                "Start with one action from the admissible commands. "
                "If the target is not visible, visit a location to search for it."
            ),
        )

    def call(self, call: ToolCall) -> StepResult:
        commands = [command.format(**call.arguments) for command in ACTION_TOOLS[call.name].commands]
        command = next((command for command in commands if command in self.environment.commands), commands[0])
        return self.environment.step(command)


@dataclass(frozen=True, slots=True)
class ALFWorldTask(Task):
    id: str = field()  # A required slot overrides the abstract property.
    config_file: str
    name: ClassVar[str] = "alfworld"

    @asynccontextmanager
    async def init_env(self) -> AsyncGenerator[ALFWorldSession]:
        async with ALFWorldSession(task=self) as environment:
            yield environment


def load_alfworld_tasks(config: ALFWorldConfig, timeout: float) -> list[ALFWorldTask]:
    config_file = str(Path(config.config_file).expanduser().resolve())
    with ExitStack() as resources:
        pending = discover_games_remote.remote(config_file)
        resources.callback(ray.cancel, pending, force=True)
        ids = ray.get(pending, timeout=timeout)
    if not ids or any(not task_id for task_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("ALFWorld training task IDs must be nonempty and unique")
    return [ALFWorldTask(id=task_id, config_file=config_file) for task_id in ids]


@ray.remote
class ALFWorldWorker:
    def __init__(self, task: ALFWorldTask) -> None:
        self.environment = ALFWorld(config_file=task.config_file, task_id=task.id)
        self.tools = ALFWorldTools(environment=self.environment)

    def reset(self) -> ResetResult:
        return self.tools.reset()

    def step(self, action: ToolCall) -> StepResult:
        return self.tools.call(action)

    def close(self) -> None:
        self.environment.close()

    def __ray_shutdown__(self) -> None:
        self.close()


class ALFWorldSession(Environment):
    """One Ray actor per rollout, with asynchronous requests and cleanup."""

    def __init__(self, task: ALFWorldTask) -> None:
        self.task = task

    async def __aenter__(self) -> Self:
        self.actor: ActorHandle = ALFWorldWorker.remote(task=self.task)
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
