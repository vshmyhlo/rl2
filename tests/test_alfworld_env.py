import asyncio
import json
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

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


@pytest.mark.parametrize("ids", [["a", "b"], [], ["a", "a"], [""]], ids=["valid", "empty", "duplicate", "empty-id"])
def test_task_discovery(ids: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    discover = Mock()
    get = Mock(return_value=ids)
    cancel = Mock()
    monkeypatch.setattr(env, "discover_games_remote", discover)
    monkeypatch.setattr(env.ray, "get", get)
    monkeypatch.setattr(env.ray, "cancel", cancel)
    config = env.ALFWorldConfig(config_file="config.yaml")
    if ids == ["a", "b"]:
        tasks = env.load_alfworld_tasks(config, max_steps=3, timeout=10)
        assert [task.id for task in tasks] == ids
        assert all(task.config_file == str(Path("config.yaml").resolve()) for task in tasks)
    else:
        with pytest.raises(ValueError, match="nonempty and unique"):
            env.load_alfworld_tasks(config, max_steps=3, timeout=10)
    discover.remote.assert_called_once_with(str(Path("config.yaml").resolve()), 3)
    get.assert_called_once_with(discover.remote.return_value, timeout=10)
    cancel.assert_called_once_with(discover.remote.return_value, force=True)


def test_task_discovery_timeout_cancels_work(monkeypatch: pytest.MonkeyPatch) -> None:
    discover = Mock()
    cancel = Mock()
    monkeypatch.setattr(env, "discover_games_remote", discover)
    monkeypatch.setattr(env.ray, "get", Mock(side_effect=env.ray.exceptions.GetTimeoutError))
    monkeypatch.setattr(env.ray, "cancel", cancel)
    with pytest.raises(env.ray.exceptions.GetTimeoutError):
        env.load_alfworld_tasks(env.ALFWorldConfig(config_file="config.yaml"), max_steps=2, timeout=1)
    cancel.assert_called_once_with(discover.remote.return_value, force=True)


def test_worker_reset_step_and_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = Mock()
    environment.reset.return_value = env.EpisodeResult(observation="Goal", reward=0.0, done=False)
    environment.step.return_value = env.EpisodeResult(observation="Success", reward=1.0, done=True)
    environment.commands = ["look"]
    constructor = Mock(return_value=environment)
    monkeypatch.setattr(env, "ALFWorld", constructor)
    worker = env.ALFWorldWorker.__ray_metadata__.modified_class(
        task=env.ALFWorldTask(id="train/a", config_file="config.yaml"), max_steps=2
    )
    with closing(worker):
        rewards = [worker.reset().reward]
        rewards.extend(worker.step(env.ToolCall(name="look", arguments={})).reward for _ in range(2))
    constructor.assert_called_once_with(config_file="config.yaml", max_steps=2, task_id="train/a")
    assert rewards == [0.0, 0.0, 1.0]
    environment.step.assert_called_once_with("look")
    environment.close.assert_called_once()


def test_session_startup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = Mock(remote=Mock(side_effect=RuntimeError("actor creation failed")))
    kill = Mock()
    monkeypatch.setattr(env, "ALFWorldWorker", worker)
    monkeypatch.setattr(env.ray, "kill", kill)
    task = env.ALFWorldTask(id="a", config_file="unused")

    async def run() -> None:
        async with task.init_env(max_steps=2, timeout=5):
            pytest.fail("Failed startup entered the session")

    with pytest.raises(RuntimeError, match="actor creation failed"):
        asyncio.run(run())
    worker.remote.assert_called_once_with(task=task, max_steps=2)
    kill.assert_not_called()


@pytest.mark.parametrize("method", ["reset", "step", "close"])
def test_session_pending_result_keeps_loop_responsive(method: str, monkeypatch: pytest.MonkeyPatch) -> None:
    actor = Mock()
    monkeypatch.setattr(env, "ALFWorldWorker", Mock(remote=Mock(return_value=actor)))
    kill = Mock()
    monkeypatch.setattr(env.ray, "kill", kill)

    async def run() -> None:
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        result = loop.create_future()
        result.set_result(env.EpisodeResult(observation="ready", reward=0.0, done=False))
        pending = loop.create_future()
        for name in ("reset", "step", "close"):
            getattr(actor, name).remote = Mock(return_value=result)

        def submit(*args: object) -> asyncio.Future[env.EpisodeResult]:
            started.set()
            return pending

        getattr(actor, method).remote = Mock(side_effect=submit)
        task = env.ALFWorldTask(id="a", config_file="unused")

        async def use_session() -> None:
            async with task.init_env(max_steps=2, timeout=5) as session:
                assert (await session.reset()).observation == "ready"
                await session.step(env.ToolCall(name="look", arguments={}))

        async with asyncio.TaskGroup() as group:
            group.create_task(use_session())
            await asyncio.wait_for(started.wait(), timeout=5)
            pending.set_result(result.result())

    asyncio.run(run())
    kill.assert_called_once_with(actor)


@pytest.mark.parametrize("mode", ["normal", "crash", "reset_timeout", "step_timeout", "cancel", "close_error"])
def test_session_actor_cleanup(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    actor = Mock()
    worker = Mock(remote=Mock(return_value=actor))
    kill = Mock()
    monkeypatch.setattr(env, "ALFWorldWorker", worker)
    monkeypatch.setattr(env.ray, "kill", kill)
    task = env.ALFWorldTask(id=mode, config_file="unused")

    async def run() -> None:
        async def reset() -> env.EpisodeResult:
            if mode == "crash":
                raise RuntimeError("worker failed")
            if mode == "reset_timeout":
                await asyncio.Future()
            return env.EpisodeResult(observation="ready", reward=0.0, done=False)

        async def step(action: env.ToolCall) -> env.EpisodeResult:
            assert action == env.ToolCall(name="look", arguments={})
            if mode in {"step_timeout", "cancel"}:
                if mode == "cancel":
                    asyncio.get_running_loop().call_soon(current.cancel)
                await asyncio.Future()
            return env.EpisodeResult(observation="stepped", reward=1.0, done=True)

        async def close() -> None:
            if mode == "close_error":
                raise RuntimeError("close failed")

        current = asyncio.current_task()
        assert current is not None
        actor.reset.remote = AsyncMock(side_effect=reset)
        actor.step.remote = AsyncMock(side_effect=step)
        actor.close.remote = AsyncMock(side_effect=close)
        async with task.init_env(max_steps=2, timeout=0.01 if mode.endswith("timeout") else 5) as session:
            result = await session.reset()
            assert result.observation == "ready"
            result = await session.step(env.ToolCall(name="look", arguments={}))
            assert result == env.EpisodeResult(observation="stepped", reward=1.0, done=True)

    error = {
        "crash": RuntimeError,
        "reset_timeout": TimeoutError,
        "step_timeout": TimeoutError,
        "cancel": asyncio.CancelledError,
        "close_error": RuntimeError,
    }.get(mode)
    if error:
        with pytest.raises(error):
            asyncio.run(run())
    else:
        asyncio.run(run())
    worker.remote.assert_called_once_with(task=task, max_steps=2)
    actor.reset.remote.assert_awaited_once_with()
    if mode in {"crash", "reset_timeout"}:
        actor.step.remote.assert_not_called()
    else:
        actor.step.remote.assert_awaited_once_with(env.ToolCall(name="look", arguments={}))
    kill.assert_called_once_with(actor)
    if mode in {"normal", "close_error"}:
        actor.close.remote.assert_awaited_once_with()
    else:
        actor.close.remote.assert_not_called()


@pytest.mark.parametrize(
    ("name", "arguments", "command"),
    [
        ("look", {}, "look"),
        ("go_to", {"location": "shelf 1"}, "go to shelf 1"),
        ("take", {"object": "mug 1", "location": "shelf 1"}, "take mug 1 from shelf 1"),
        ("move", {"object": "mug 1", "location": "desk 1"}, "move mug 1 to desk 1"),
        ("open_container", {"container": "fridge 1"}, "open fridge 1"),
        ("close_container", {"container": "fridge 1"}, "close fridge 1"),
        ("clean", {"object": "mug 1", "location": "sinkbasin 1"}, "clean mug 1 with sinkbasin 1"),
        ("heat", {"object": "mug 1", "location": "microwave 1"}, "heat mug 1 with microwave 1"),
        ("cool", {"object": "mug 1", "location": "fridge 1"}, "cool mug 1 with fridge 1"),
        ("use", {"object": "lamp 1"}, "use lamp 1"),
        ("examine", {"target": "mug 1"}, "examine mug 1"),
        ("inventory", {}, "inventory"),
    ],
)
def test_tool_commands(name: str, arguments: dict[str, str], command: str) -> None:
    environment = Mock(commands=[])
    initial = env.EpisodeResult(observation="Goal", reward=0.0, done=False)
    environment.reset.return_value = initial
    environment.step.return_value = initial
    tools = env.ALFWorldTools(environment=environment, max_steps=2)
    assert tools.reset() == initial
    assert tools.call(env.ToolCall(name="look", arguments={})) == initial
    environment.step.assert_not_called()
    tools.call(env.ToolCall(name=name, arguments=arguments))
    environment.step.assert_called_once_with(command)
    schemas = {schema["function"]["name"]: schema["function"] for schema in env.tool_schemas()}
    assert set(schemas) == {
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
    assert set(schemas[name]["parameters"]["required"]) == set(arguments)
    assert schemas[name]["parameters"]["properties"] == {key: {"type": "string"} for key in arguments}


@pytest.mark.parametrize("command", ["put mug 1 in/on desk 1", "put mug 1 into desk 1", "put mug 1 in desk 1"])
def test_placement_aliases_and_terminal_state(command: str) -> None:
    environment = Mock(commands=[command])
    environment.reset.return_value = env.EpisodeResult(observation="Goal", reward=0.0, done=False)
    environment.step.return_value = env.EpisodeResult(observation="Done", reward=1.0, done=True)
    tools = env.ALFWorldTools(environment=environment, max_steps=2)
    tools.reset()
    result = tools.call(env.ToolCall(name="move", arguments={"object": "mug 1", "location": "desk 1"}))
    assert result.done and result.reward == 1.0
    assert tools.call(env.ToolCall(name="look", arguments={})) == result
    environment.step.assert_called_once_with(command)


def test_tool_step_budget() -> None:
    environment = Mock(commands=["look"])
    initial = env.EpisodeResult(observation="Goal", reward=0.0, done=False)
    environment.reset.return_value = environment.step.return_value = initial
    tools = env.ALFWorldTools(environment=environment, max_steps=1)
    tools.reset()
    assert not tools.call(env.ToolCall(name="look", arguments={})).done
    assert tools.call(env.ToolCall(name="look", arguments={})).done
    assert tools.call(env.ToolCall(name="look", arguments={})).done
    environment.step.assert_called_once_with("look")
    tools.reset()
    assert not tools.call(env.ToolCall(name="look", arguments={})).done


@pytest.mark.parametrize(
    ("name", "arguments"),
    [("unknown", {}), ("take", {"object": "mug 1"}), ("look", {"extra": "x"}), ("use", {"object": 1})],
    ids=["unknown", "missing-argument", "extra-argument", "argument-type"],
)
def test_invalid_action_does_not_reach_worker(name: str, arguments: dict[str, object]) -> None:
    session = env.ALFWorldSession(task=env.ALFWorldTask(id="a", config_file="config.yaml"), max_steps=2, timeout=1)
    with pytest.raises(env.ToolCallError):
        asyncio.run(session.step(env.ToolCall(name=name, arguments=arguments)))
