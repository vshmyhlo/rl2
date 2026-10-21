from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import ExitStack, asynccontextmanager, closing
from dataclasses import asdict, dataclass, replace
from os.path import expandvars
from pathlib import Path
from typing import Any, Literal

import verifiers.v1 as vf
import yaml
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
from mcp.server.lowlevel import Server
from mcp.server.mcpserver import MCPServer
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass as config_dataclass


@config_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class ALFWorldConfig:
    config_file: str
    id: Literal["alfworld"] = "alfworld"


def load_environment_config(config_file: str, max_steps: int) -> dict[str, Any]:
    config_path = Path(config_file).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    data_path = Path(expandvars(config["dataset"]["data_path"])).expanduser()
    # Tool servers run in temporary directories; anchor dataset paths to the config.
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


class ALFWorldState(vf.State):
    done: bool = False
    reward: float = 0.0


class ALFWorldToolsConfig(vf.ToolsetConfig):
    config_file: str
    max_steps: int = Field(gt=0)


class ALFWorldData(vf.TaskData):
    task_id: str


class ALFWorldTools(vf.Toolset[ALFWorldToolsConfig, ALFWorldState]):
    TOOL_PREFIX = None

    def register(self, mcp: MCPServer) -> None:
        super().register(mcp)
        original_lifespan = mcp._lowlevel_server.lifespan

        @asynccontextmanager
        async def lifespan(server: Server) -> AsyncIterator[dict[str, Any]]:
            # Close during HTTP shutdown, before Uvicorn re-raises the termination signal.
            async with self._exit_stack, original_lifespan(server) as state:
                yield state

        mcp._lowlevel_server.lifespan = lifespan

    async def setup_task(self, task: ALFWorldData) -> None:
        self.environment = self._exit_stack.enter_context(
            closing(
                ALFWorld(config_file=self.config.config_file, max_steps=self.config.max_steps, task_id=task.task_id)
            )
        )
        self.current = self.environment.reset()
        self.started = False
        self.steps = 0

    def result(self) -> dict[str, Any]:
        self.state.done = self.current.done
        self.state.reward = self.current.reward
        return asdict(self.current)

    def step(self, command: str, *aliases: str) -> dict[str, Any]:
        if not self.current.done:
            self.started = True
            command = next((value for value in (command, *aliases) if value in self.environment.commands), command)
            self.current = self.environment.step(command)
            self.steps += 1
            if self.steps >= self.config.max_steps:
                self.current = replace(self.current, done=True)
        return self.result()

    @vf.tool
    def look(self) -> dict[str, Any]:
        """Read the goal and starting observation first; later, look around your current location."""
        if not self.started:
            self.started = True
            return self.result()
        return self.step("look")

    @vf.tool
    def go_to(self, location: str) -> dict[str, Any]:
        """Go to a named location, such as 'shelf 1'."""
        return self.step(f"go to {location}")

    @vf.tool
    def take(self, object: str, location: str) -> dict[str, Any]:
        """Take an object from a location. Include numeric IDs in both names."""
        return self.step(f"take {object} from {location}")

    @vf.tool
    def move(self, object: str, location: str) -> dict[str, Any]:
        """Place a carried object at a location or inside a receptacle."""
        return self.step(
            f"move {object} to {location}",
            f"put {object} in/on {location}",
            f"put {object} into {location}",
            f"put {object} in {location}",
        )

    @vf.tool
    def open_container(self, container: str) -> dict[str, Any]:
        """Open a named container."""
        return self.step(f"open {container}")

    @vf.tool
    def close_container(self, container: str) -> dict[str, Any]:
        """Close a named container."""
        return self.step(f"close {container}")

    @vf.tool
    def clean(self, object: str, location: str) -> dict[str, Any]:
        """Clean a carried object using a sink basin."""
        return self.step(f"clean {object} with {location}")

    @vf.tool
    def heat(self, object: str, location: str) -> dict[str, Any]:
        """Heat a carried object using a microwave."""
        return self.step(f"heat {object} with {location}")

    @vf.tool
    def cool(self, object: str, location: str) -> dict[str, Any]:
        """Cool a carried object using a fridge."""
        return self.step(f"cool {object} with {location}")

    @vf.tool
    def use(self, object: str) -> dict[str, Any]:
        """Toggle a named device, such as a lamp."""
        return self.step(f"use {object}")

    @vf.tool
    def examine(self, target: str) -> dict[str, Any]:
        """Examine a named object or location."""
        return self.step(f"examine {target}")

    @vf.tool
    def inventory(self) -> dict[str, Any]:
        """List the objects you are carrying."""
        return self.step("inventory")


class ALFWorldTaskConfig(vf.TaskConfig):
    tools: ALFWorldToolsConfig


class ALFWorldTask(vf.Task[ALFWorldData, ALFWorldState, ALFWorldTaskConfig]):
    @classmethod
    def toolsets(cls, config: ALFWorldTaskConfig) -> list[vf.Toolset]:
        return [ALFWorldTools(config=config.tools)]

    @vf.stop
    def alfworld_done(self, trace: vf.Trace) -> bool:
        return trace.state.done

    @vf.reward
    async def alfworld_reward(self, trace: vf.Trace) -> float:
        return trace.state.reward


def load_alfworld_tasks(config: ALFWorldConfig, max_steps: int) -> list[ALFWorldTask]:
    ids = discover_games(config_file=config.config_file, max_steps=max_steps)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("ALFWorld training task IDs must be nonempty and unique")
    task_config = ALFWorldTaskConfig(
        tools=ALFWorldToolsConfig(config_file=str(Path(config.config_file).expanduser().resolve()), max_steps=max_steps)
    )
    return [
        ALFWorldTask(
            data=ALFWorldData(
                id=f"alfworld:{task_id}",
                task_id=task_id,
                prompt="Call look to read your goal and surroundings, then use the action tools to complete the task.",
                system_prompt="Use one action tool per turn. Use the exact object and location names, including IDs.",
            ),
            config=task_config,
        )
        for task_id in ids
    ]


if __name__ == "__main__":
    ALFWorldTools.run()
