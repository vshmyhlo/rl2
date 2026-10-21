import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from mcp.server.lowlevel import Server
from mcp.server.mcpserver import MCPServer

from rl2 import alfworld_env as env


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        json.dumps(
            {
                "dataset": {"data_path": "games"},
                "env": {"goal_desc_human_anns_prob": 0},
                "general": {},
                "rl": {"training": {}},
            }
        )
    )
    return path


def write_game(directory: Path, *, solvable: bool = True, task_type: str = "pick_and_place_simple") -> Path:
    directory.mkdir(parents=True)
    (directory / "traj_data.json").write_text(json.dumps({"task_type": task_type}))
    game = directory / "game.tw-pddl"
    game.write_text(json.dumps({"solvable": solvable}))
    return game


def test_episode_reset_step_and_cleanup(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    game = config_file.parent / "games" / "train-a" / "game.tw-pddl"
    upstream = Mock()
    upstream.reset.return_value = (["Put apple on table"], {"admissible_commands": [["look"]], "won": [False]})
    upstream.step.return_value = (["Done"], [0], [True], {"admissible_commands": [[]], "won": [True]})
    loader = Mock(game_files=[str(game)])
    loader.init_env.return_value = upstream
    constructor = Mock(return_value=loader)
    monkeypatch.setattr(env, "AlfredTWEnv", constructor)
    with closing(env.ALFWorld(config_file=str(config_file), max_steps=3, task_id=str(game))) as environment:
        settings = constructor.call_args.args[0]
        assert constructor.call_args.kwargs == {"train_eval": "train"}
        assert settings["dataset"]["data_path"] == str(game.parent)
        assert settings["dataset"]["num_train_games"] == -1
        assert settings["env"]["task_types"] == [1, 2, 3, 4, 5, 6]
        assert settings["general"]["training_method"] == "dqn"
        assert settings["rl"]["training"]["max_nb_steps_per_episode"] == 3
        initial = environment.reset()
        assert "look" in initial.observation
        assert environment.commands == ["look"]
        assert initial.reward == 0.0 and not initial.done
        loader.init_env.assert_called_once_with(batch_size=1)
        result = environment.step(" look ")
        assert result.done and result.reward == 1.0
        upstream.step.assert_called_once_with(["look"])
        upstream.reset.return_value = (["Already done"], {"admissible_commands": [[]], "won": [True]})
        assert environment.reset() == env.EpisodeResult(
            observation="Already done\nAdmissible commands: ", reward=1.0, done=True
        )
        upstream.reset.side_effect = RuntimeError("reset failed")
        with pytest.raises(RuntimeError, match="reset failed"):
            environment.reset()
    assert upstream.close.call_count == 3


def test_episode_loads_only_selected_game(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = config_file.parent / "games"
    first = write_game(root / "train-a")
    second = write_game(root / "train-b")
    assert env.discover_games(str(config_file), max_steps=2) == [str(first), str(second)]
    # Loading the selected episode must not read a sibling's metadata again.
    (second.parent / "traj_data.json").write_text("invalid JSON")
    monkeypatch.chdir(config_file.parent.parent)
    with closing(env.ALFWorld(config_file=str(config_file), max_steps=2, task_id=str(first))) as environment:
        assert environment.loader.game_files == [str(first)]
        assert environment.loader.num_games == 1


@pytest.mark.parametrize("reason", ["outside-root", "symlink", "unsolvable", "unsupported", "wrong-file"])
def test_episode_rejects_invalid_game(reason: str, config_file: Path) -> None:
    root = config_file.parent / "games"
    if reason in {"outside-root", "symlink"}:
        game = write_game(config_file.parent / "other-split")
        if reason == "symlink":
            root.mkdir()
            (root / "linked").symlink_to(game.parent, target_is_directory=True)
            game = root / "linked" / game.name
    else:
        game = write_game(
            root / "train-a",
            solvable=reason != "unsolvable",
            task_type="unsupported" if reason == "unsupported" else "pick_and_place_simple",
        )
        if reason == "wrong-file":
            game = game.parent / "traj_data.json"
    with pytest.raises(ValueError, match="training"):
        env.ALFWorld(config_file=str(config_file), max_steps=2, task_id=str(game))


@pytest.mark.parametrize("path_kind", ["relative", "environment"])
def test_dataset_path_is_independent_of_working_directory(
    path_kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    game_file = write_game(tmp_path / "games" / "train-a")
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_file = config_dir / "alfworld.yaml"
    monkeypatch.setenv("RL2_TEST_DATA", str(tmp_path / "games"))
    config_file.write_text(
        json.dumps(
            {
                "dataset": {"data_path": "../games" if path_kind == "relative" else "$RL2_TEST_DATA"},
                "env": {"goal_desc_human_anns_prob": 0},
                "general": {},
                "rl": {"training": {}},
            }
        )
    )
    for directory in (config_dir, tmp_path):
        monkeypatch.chdir(directory)
        assert env.discover_games(str(config_file), max_steps=2) == [str(game_file)]


@pytest.fixture
def environment(monkeypatch: pytest.MonkeyPatch) -> Mock:
    environment = Mock(commands=[])
    environment.reset.return_value = env.EpisodeResult(observation="Put mug on desk", reward=0.0, done=False)
    environment.step.return_value = env.EpisodeResult(observation="Moved", reward=0.0, done=False)
    monkeypatch.setattr(env, "ALFWorld", Mock(return_value=environment))
    return environment


@pytest.fixture
def task(monkeypatch: pytest.MonkeyPatch) -> env.ALFWorldTask:
    monkeypatch.setattr(env, "discover_games", Mock(return_value=["train/a"]))
    return env.load_alfworld_tasks(env.ALFWorldConfig(config_file="config.yaml"), max_steps=2)[0]


def test_manifest(task: env.ALFWorldTask, environment: Mock) -> None:
    assert task.data.id == "alfworld:train/a"
    assert task.data.task_id == "train/a"
    assert "look" in task.data.prompt
    assert task.config.tools.max_steps == 2
    assert Path(task.config.tools.config_file).is_absolute()
    environment.close.assert_not_called()
    environment.reset.assert_not_called()


@pytest.mark.parametrize("ids", [[], ["a", "a"]])
def test_invalid_manifest(ids: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "discover_games", Mock(return_value=ids))
    with pytest.raises(ValueError, match="nonempty and unique"):
        env.load_alfworld_tasks(env.ALFWorldConfig(config_file="config.yaml"), max_steps=2)


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


@pytest.mark.parametrize("mode", ["success", "already-done", "limit", "reset-error", "step-error", "cancel"])
def test_lifecycle_and_stop(mode: str, task: env.ALFWorldTask, environment: Mock) -> None:
    environment.reset_mock()
    if mode == "success":
        environment.step.return_value = env.EpisodeResult(observation="Done", reward=1.0, done=True)
    elif mode == "already-done":
        environment.reset.return_value = env.EpisodeResult(observation="Done", reward=1.0, done=True)
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
            assert await task.alfworld_reward(SimpleNamespace(state=tools.state)) == (
                1.0 if mode in {"success", "already-done"} else 0.0
            )
            if mode == "already-done":
                environment.step.assert_not_called()
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


@pytest.mark.parametrize("cancel", [False, True])
def test_mcp_lifespan_preserves_context_and_closes_environment(cancel: bool, environment: Mock) -> None:
    events: list[str] = []

    def close() -> None:
        events.append("close")

    environment.close.side_effect = close

    @asynccontextmanager
    async def existing_lifespan(server: Server) -> AsyncIterator[dict[str, Any]]:
        try:
            events.append("start")
            yield {"resource": "existing"}
        finally:
            events.append("stop")

    async def run() -> None:
        tools = env.ALFWorldTools(config=env.ALFWorldToolsConfig(config_file="config.yaml", max_steps=2))
        async with tools._exit_stack:
            await tools.setup_task(env.ALFWorldData(task_id="train/a"))
            server = MCPServer("alfworld")
            server._lowlevel_server.lifespan = existing_lifespan
            tools.register(server)
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    async with server._lowlevel_server.lifespan(server._lowlevel_server) as state:
                        assert state == {"resource": "existing"}
                        raise asyncio.CancelledError
            else:
                async with server._lowlevel_server.lifespan(server._lowlevel_server) as state:
                    assert state == {"resource": "existing"}
            # Cleanup must happen during MCP shutdown, before the outer Verifiers stack exits.
            environment.close.assert_called_once()
        environment.close.assert_called_once()

    asyncio.run(run())
    assert events == ["start", "stop", "close"]
