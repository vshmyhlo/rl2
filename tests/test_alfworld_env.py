import asyncio
import json
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from mcp.server.mcpserver import MCPServer

from rl2 import alfworld_env as env


def test_training_games_and_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps({"dataset": {}, "env": {}, "general": {}, "rl": {"training": {}}}))
    upstream = Mock()
    upstream.reset.return_value = (["Put apple on table"], {"admissible_commands": [["look"]]})
    upstream.step.return_value = (["Done"], [0], [True], {"admissible_commands": [[]], "won": [True]})
    loader = Mock(game_files=["train/b", "train/a"])
    loader.init_env.return_value = upstream
    constructor = Mock(return_value=loader)
    monkeypatch.setattr(env, "AlfredTWEnv", constructor)
    with closing(env.ALFWorld(config_file=str(path), max_steps=3)) as environment:
        assert environment.task_ids() == ["train/a", "train/b"]
        settings = constructor.call_args.args[0]
        assert constructor.call_args.kwargs == {"train_eval": "train"}
        assert settings["dataset"]["num_train_games"] == -1
        assert settings["env"]["task_types"] == [1, 2, 3, 4, 5, 6]
        assert settings["general"]["training_method"] == "dqn"
        assert settings["rl"]["training"]["max_nb_steps_per_episode"] == 3
        with pytest.raises(ValueError, match="training split"):
            environment.reset("dev/a")
        assert "look" in environment.reset("train/b")["observation"]
        assert environment.commands == ["look"]
        assert loader.game_files == ["train/b"]
        assert environment.score() == 0.0
        assert environment.step(" look \n")["done"]
        upstream.step.assert_called_once_with(["look"])
        assert environment.score() == 1.0
    upstream.close.assert_called_once()


@pytest.fixture
def environment(monkeypatch: pytest.MonkeyPatch) -> Mock:
    environment = Mock(commands=[])
    environment.reset.return_value = {"observation": "Put mug on desk", "reward": 0.0, "done": False}
    environment.step.return_value = {"observation": "Moved", "reward": 0.0, "done": False}
    monkeypatch.setattr(env, "ALFWorld", Mock(return_value=environment))
    return environment


@pytest.fixture
def task(environment: Mock) -> env.ALFWorldTask:
    environment.task_ids.return_value = ["train/a"]
    return env.load_alfworld_tasks(env.ALFWorldConfig(config_file="config.yaml"), max_steps=2)[0]


def test_manifest(task: env.ALFWorldTask, environment: Mock) -> None:
    assert task.data.id == "alfworld:train/a"
    assert task.data.task_id == "train/a"
    assert "look" in task.data.prompt
    assert task.config.tools.max_steps == 2
    assert Path(task.config.tools.config_file).is_absolute()
    environment.close.assert_called_once()
    environment.reset.assert_not_called()


@pytest.mark.parametrize("ids", [[], ["a", "a"]])
def test_invalid_manifest(ids: list[str], environment: Mock) -> None:
    environment.task_ids.return_value = ids
    with pytest.raises(ValueError, match="nonempty and unique"):
        env.load_alfworld_tasks(env.ALFWorldConfig(config_file="config.yaml"), max_steps=2)
    environment.close.assert_called_once()


@pytest.mark.parametrize(
    "name,arguments,command",
    [
        ("go_to", {"location": "shelf 1"}, "go to shelf 1"),
        ("take", {"object": "mug 1", "location": "shelf 1"}, "take mug 1 from shelf 1"),
        ("move", {"object": "mug 1", "location": "desk 1"}, "move mug 1 to desk 1"),
        ("open_container", {"container": "cabinet 1"}, "open cabinet 1"),
        ("close_container", {"container": "cabinet 1"}, "close cabinet 1"),
        ("clean", {"object": "mug 1", "location": "sinkbasin 1"}, "clean mug 1 with sinkbasin 1"),
        ("heat", {"object": "mug 1", "location": "microwave 1"}, "heat mug 1 with microwave 1"),
        ("cool", {"object": "mug 1", "location": "fridge 1"}, "cool mug 1 with fridge 1"),
        ("use", {"object": "desklamp 1"}, "use desklamp 1"),
        ("examine", {"target": "mug 1"}, "examine mug 1"),
        ("inventory", {}, "inventory"),
        ("look", {}, "look"),
    ],
)
def test_tool_commands(name: str, arguments: dict[str, str], command: str, environment: Mock) -> None:
    async def run() -> None:
        tools = env.ALFWorldTools(config=env.ALFWorldToolsConfig(config_file="config.yaml", max_steps=2))
        async with tools._exit_stack:
            await tools.setup_task(env.ALFWorldData(task_id="train/a"))
            assert tools.look()["observation"] == "Put mug on desk"
            environment.step.assert_not_called()
            result = getattr(tools, name)(**arguments)
            assert result["observation"] == "Moved"
            environment.step.assert_called_once_with(command)

    asyncio.run(run())
    environment.close.assert_called_once()


@pytest.mark.parametrize("command", ["put mug 1 in/on desk 1", "put mug 1 into desk 1", "put mug 1 in desk 1"])
def test_placement_grammar(command: str, environment: Mock) -> None:
    environment.commands = [command]

    async def run() -> None:
        tools = env.ALFWorldTools(config=env.ALFWorldToolsConfig(config_file="config.yaml", max_steps=2))
        async with tools._exit_stack:
            await tools.setup_task(env.ALFWorldData(task_id="train/a"))
            tools.move(object="mug 1", location="desk 1")
            environment.step.assert_called_once_with(command)

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["success", "limit", "reset-error", "step-error", "cancel"])
def test_lifecycle_and_stop(mode: str, task: env.ALFWorldTask, environment: Mock) -> None:
    environment.reset_mock()
    if mode == "success":
        environment.step.return_value = {"observation": "Done", "reward": 1.0, "done": True}
    elif mode in ("reset-error", "step-error"):
        getattr(environment, mode.removesuffix("-error")).side_effect = RuntimeError("failed")

    async def run() -> None:
        tools = task.toolsets(task.config)[0]
        async with tools._exit_stack:
            await tools.setup_task(task.data)
            tools.go_to(location="shelf 1")
            if mode == "cancel":
                raise asyncio.CancelledError
            if mode == "limit":
                tools.go_to(location="desk 1")
            assert task.alfworld_done(SimpleNamespace(state=tools.state))
            assert await task.alfworld_reward(SimpleNamespace(state=tools.state)) == (1.0 if mode == "success" else 0.0)
            count = environment.step.call_count
            tools.go_to(location="shelf 1")
            assert environment.step.call_count == count

    if mode in ("reset-error", "step-error", "cancel"):
        with pytest.raises(asyncio.CancelledError if mode == "cancel" else RuntimeError):
            asyncio.run(run())
    else:
        asyncio.run(run())
    environment.close.assert_called_once()


def test_tool_schemas(task: env.ALFWorldTask) -> None:
    server = MCPServer("alfworld")
    task.toolsets(task.config)[0].register(server)
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    assert set(tools) == {
        "look",
        "go_to",
        "take",
        "move",
        "open_container",
        "close_container",
        "clean",
        "heat",
        "cool",
        "use",
        "examine",
        "inventory",
    }
    assert set(tools["take"].input_schema["required"]) == {"object", "location"}
    assert tools["take"].input_schema["properties"]["object"]["type"] == "string"
