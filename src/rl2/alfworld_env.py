from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from contextlib import ExitStack, closing, redirect_stdout, suppress
from dataclasses import asdict, dataclass, replace
from os.path import expandvars
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

import yaml
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
from pydantic import BaseModel, ConfigDict, TypeAdapter, model_validator
from pydantic.dataclasses import dataclass as config_dataclass


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


@dataclass(frozen=True)
class EpisodeResult:
    observation: str
    reward: float
    done: bool


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


def tool_schemas() -> list[dict[str, Any]]:
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


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str
    arguments: dict[str, str]

    @model_validator(mode="after")
    def validate_action(self) -> Self:
        if self.name not in ACTION_TOOLS:
            raise ValueError(f"Unknown ALFWorld tool: {self.name}")
        parameters = ACTION_TOOLS[self.name].parameters
        if self.arguments.keys() != set(parameters):
            raise ValueError(f"{self.name} requires exactly these arguments: {', '.join(parameters) or '(none)'}")
        return self


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


@dataclass(frozen=True)
class ALFWorldTask:
    id: str
    config_file: str


def worker_command(config_file: str, max_steps: int) -> list[str]:
    return [sys.executable, "-m", "rl2.alfworld_env", config_file, str(max_steps)]


def load_alfworld_tasks(config: ALFWorldConfig, max_steps: int, timeout: float) -> list[ALFWorldTask]:
    config_file = str(Path(config.config_file).expanduser().resolve())
    result = subprocess.run(
        [*worker_command(config_file, max_steps), "--list"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
        timeout=timeout,
    )
    ids = TypeAdapter(list[str]).validate_json(result.stdout)
    if not ids or any(not task_id for task_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("ALFWorld training task IDs must be nonempty and unique")
    return [ALFWorldTask(id=task_id, config_file=config_file) for task_id in ids]


class ALFWorldSession:
    """One isolated TextWorld process per rollout, with cancellable async I/O."""

    def __init__(self, task: ALFWorldTask, max_steps: int, timeout: float) -> None:
        self.command = [*worker_command(task.config_file, max_steps), task.id]
        self.timeout = timeout

    async def __aenter__(self) -> Self:
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            start_new_session=True,
            limit=16 * 1024 * 1024,
        )
        return self

    async def request(self, action: ToolCall | None = None) -> EpisodeResult:
        assert self.process.stdin is not None and self.process.stdout is not None
        async with asyncio.timeout(self.timeout):
            self.process.stdin.write(((action.model_dump_json() if action is not None else "null") + "\n").encode())
            await self.process.stdin.drain()
            line = await self.process.stdout.readline()
            if not line:
                raise RuntimeError("ALFWorld worker exited without a response; see its stderr")
            return TypeAdapter(EpisodeResult).validate_json(line)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        if exc_type is not None:
            with suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
        try:
            await asyncio.wait_for(self.process.wait(), timeout=5)
        finally:
            # TextWorld may own children; also reap them after a crash or timeout.
            with suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
            await self.process.wait()
        if exc_type is None and self.process.returncode != 0:
            raise RuntimeError(f"ALFWorld worker failed (exit {self.process.returncode})")


def main() -> None:
    config_file, max_steps, task_id = sys.argv[1:]
    output = sys.stdout
    with redirect_stdout(sys.stderr):
        if task_id == "--list":
            print(json.dumps(discover_games(config_file, int(max_steps))), file=output, flush=True)
            return
        with closing(ALFWorld(config_file=config_file, max_steps=int(max_steps), task_id=task_id)) as environment:
            tools = ALFWorldTools(environment=environment, max_steps=int(max_steps))
            for line in sys.stdin:
                action = TypeAdapter(ToolCall | None).validate_json(line)
                result = tools.reset() if action is None else tools.call(action)
                print(json.dumps(asdict(result)), file=output, flush=True)


if __name__ == "__main__":
    main()
