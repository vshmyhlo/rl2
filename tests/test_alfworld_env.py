import asyncio
import json
import sys
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
                "rl": {"training": {"max_nb_steps_per_episode": 1}},
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
    with closing(env.ALFWorld(config_file=str(config_file), task_id=str(game))) as environment:
        settings = constructor.call_args.args[0]
        assert constructor.call_args.kwargs == {"train_eval": "train"}
        assert settings["dataset"]["data_path"] == str(game.parent)
        assert settings["dataset"]["num_train_games"] == -1
        assert settings["env"]["task_types"] == [1, 2, 3, 4, 5, 6]
        assert settings["general"]["training_method"] == "dqn"
        assert settings["rl"]["training"]["max_nb_steps_per_episode"] is None
        initial = environment.reset()
        assert "look" in initial.observation
        assert environment.commands == ["look"]
        assert initial.reward == 0.0 and not initial.done
        loader.init_env.assert_called_once_with(batch_size=1)
        result = environment.step(" look ")
        assert result.done and result.reward == 1.0
        upstream.step.assert_called_once_with(["look"])
        upstream.reset.return_value = (["Already done"], {"admissible_commands": [[]], "won": [True]})
        assert environment.reset() == env.StepResult(
            observation="Already done\nAdmissible commands: ", reward=1.0, done=True
        )
        upstream.reset.side_effect = RuntimeError("reset failed")
        with pytest.raises(RuntimeError, match="reset failed"):
            environment.reset()
    assert upstream.close.call_count == 3


def test_episode_loads_only_selected_game(
    config_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = config_file.parent / "games"
    first = write_game(root / "train-a")
    second = write_game(root / "train-b")
    assert env.discover_games(str(config_file)) == [str(first), str(second)]
    captured = capsys.readouterr()
    assert not captured.out
    assert not captured.err
    # Loading the selected episode must not read a sibling's metadata again.
    (second.parent / "traj_data.json").write_text("invalid JSON")
    monkeypatch.chdir(config_file.parent.parent)
    with closing(env.ALFWorld(config_file=str(config_file), task_id=str(first))) as environment:
        assert environment.loader.game_files == [str(first)]
        assert environment.loader.num_games == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert not captured.err
    print("training progress")
    print("training diagnostic", file=sys.stderr)
    captured = capsys.readouterr()
    assert captured.out == "training progress\n"
    assert captured.err == "training diagnostic\n"


def test_discovery_failure_preserves_output(config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    game = write_game(config_file.parent / "games" / "train-a")
    (game.parent / "traj_data.json").write_text("invalid JSON")
    with pytest.raises(json.JSONDecodeError):
        env.discover_games(str(config_file))
    captured = capsys.readouterr()
    assert "Initializing AlfredTWEnv" in captured.out
    assert captured.err
    print("training progress")
    print("training diagnostic", file=sys.stderr)
    captured = capsys.readouterr()
    assert captured.out == "training progress\n"
    assert captured.err == "training diagnostic\n"


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
        env.ALFWorld(config_file=str(config_file), task_id=str(game))


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
                "rl": {"training": {"max_nb_steps_per_episode": 1}},
            }
        )
    )
    for directory in (config_dir, tmp_path):
        monkeypatch.chdir(directory)
        assert env.discover_games(str(config_file)) == [str(game_file)]


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
        tasks = env.load_alfworld_tasks(config, timeout=10)
        assert [task.id for task in tasks] == ids
        assert all(task.config_file == str(Path("config.yaml").resolve()) for task in tasks)
    else:
        with pytest.raises(ValueError, match="nonempty and unique"):
            env.load_alfworld_tasks(config, timeout=10)
    discover.remote.assert_called_once_with(str(Path("config.yaml").resolve()))
    get.assert_called_once_with(discover.remote.return_value, timeout=10)
    cancel.assert_called_once_with(discover.remote.return_value, force=True)


def test_task_discovery_timeout_cancels_work(monkeypatch: pytest.MonkeyPatch) -> None:
    discover = Mock()
    cancel = Mock()
    monkeypatch.setattr(env, "discover_games_remote", discover)
    monkeypatch.setattr(env.ray, "get", Mock(side_effect=env.ray.exceptions.GetTimeoutError))
    monkeypatch.setattr(env.ray, "cancel", cancel)
    with pytest.raises(env.ray.exceptions.GetTimeoutError):
        env.load_alfworld_tasks(env.ALFWorldConfig(config_file="config.yaml"), timeout=1)
    cancel.assert_called_once_with(discover.remote.return_value, force=True)


def test_worker_reset_step_and_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = Mock()
    observation = "Your task is to: examine the box with the desklamp.\nAdmissible commands: go to drawer 1, look"
    environment.reset.return_value = env.StepResult(observation=observation, reward=0.0, done=False)
    environment.step.return_value = env.StepResult(observation="Success", reward=1.0, done=True)
    environment.commands = ["look"]
    constructor = Mock(return_value=environment)
    monkeypatch.setattr(env, "ALFWorld", constructor)
    worker = env.ALFWorldWorker.__ray_metadata__.modified_class(
        task=env.ALFWorldTask(id="train/a", config_file="config.yaml")
    )
    with closing(worker):
        initial = worker.reset()
        assert initial.prompt == (
            f"Task and starting observation:\n{observation}\n\n"
            "Start with one action from the admissible commands. "
            "If the target is not visible, visit a location to search for it."
        )
        assert initial.tools == env.tool_schemas()
        assert initial.system_prompt == env.SYSTEM_PROMPT
        result = worker.step(env.ToolCall(name="look", arguments={}))
    constructor.assert_called_once_with(config_file="config.yaml", task_id="train/a")
    assert result == env.StepResult(observation="Success", reward=1.0, done=True)
    environment.step.assert_called_once_with("look")
    environment.close.assert_called_once()


@pytest.mark.parametrize(
    ("reward", "done", "message"),
    [(1.0, False, "must not award a reward"), (0.0, True, "must allow at least one step")],
    ids=["reward-on-reset", "terminal-reset"],
)
def test_reset_rejects_reward_or_terminal_state(reward: float, done: bool, message: str) -> None:
    environment = Mock()
    environment.reset.return_value = env.StepResult(observation="Goal", reward=reward, done=done)
    tools = env.ALFWorldTools(environment=environment)
    with pytest.raises(AssertionError, match=message):
        tools.reset()
    environment.step.assert_not_called()


def test_session_startup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = Mock(remote=Mock(side_effect=RuntimeError("actor creation failed")))
    kill = Mock()
    monkeypatch.setattr(env, "ALFWorldWorker", worker)
    monkeypatch.setattr(env.ray, "kill", kill)
    task = env.ALFWorldTask(id="a", config_file="unused")

    async def run() -> None:
        async with task.init_env():
            pytest.fail("Failed startup entered the session")

    with pytest.raises(RuntimeError, match="actor creation failed"):
        asyncio.run(run())
    worker.remote.assert_called_once_with(task=task)
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
        results = {
            "reset": env.ResetResult(tools=env.tool_schemas(), system_prompt="Use tools.", prompt="ready"),
            "step": env.StepResult(observation="stepped", reward=0.0, done=False),
            "close": None,
        }
        pending = loop.create_future()
        for name, value in results.items():
            result = loop.create_future()
            result.set_result(value)
            getattr(actor, name).remote = Mock(return_value=result)

        def submit(*args: object) -> asyncio.Future[env.ResetResult | env.StepResult | None]:
            started.set()
            return pending

        getattr(actor, method).remote = Mock(side_effect=submit)
        task = env.ALFWorldTask(id="a", config_file="unused")

        async def use_session() -> None:
            async with task.init_env() as session:
                initial = await session.reset()
                assert initial.prompt == "ready"
                assert initial.system_prompt == "Use tools."
                assert initial.tools == env.tool_schemas()
                await session.step(env.ToolCall(name="look", arguments={}))

        async with asyncio.TaskGroup() as group:
            group.create_task(use_session())
            await asyncio.wait_for(started.wait(), timeout=5)
            pending.set_result(results[method])

    asyncio.run(run())
    kill.assert_called_once_with(actor)


def test_cleanup_waits_past_former_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    actor = Mock()
    monkeypatch.setattr(env, "ALFWorldWorker", Mock(remote=Mock(return_value=actor)))
    kill = Mock()
    monkeypatch.setattr(env.ray, "kill", kill)

    async def run() -> None:
        loop = asyncio.get_running_loop()
        now = loop.time()
        monkeypatch.setattr(loop, "time", lambda: now)

        async def close() -> None:
            nonlocal now
            now += 6
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        actor.close.remote = AsyncMock(side_effect=close)
        async with env.ALFWorldTask(id="task", config_file="config.yaml").init_env():
            pass

    asyncio.run(run())
    actor.close.remote.assert_awaited_once_with()
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
        async def reset() -> env.ResetResult:
            if mode == "crash":
                raise RuntimeError("worker failed")
            if mode == "reset_timeout":
                await asyncio.Future()
            return env.ResetResult(tools=env.tool_schemas(), system_prompt="Use tools.", prompt="ready")

        async def step(action: env.ToolCall) -> env.StepResult:
            assert action == env.ToolCall(name="look", arguments={})
            if mode in {"step_timeout", "cancel"}:
                if mode == "cancel":
                    asyncio.get_running_loop().call_soon(current.cancel)
                await asyncio.Future()
            return env.StepResult(observation="stepped", reward=1.0, done=True)

        async def close() -> None:
            if mode == "close_error":
                raise RuntimeError("close failed")

        current = asyncio.current_task()
        assert current is not None
        actor.reset.remote = AsyncMock(side_effect=reset)
        actor.step.remote = AsyncMock(side_effect=step)
        actor.close.remote = AsyncMock(side_effect=close)
        async with task.init_env() as session:
            initial = await session.reset(timeout=0 if mode == "reset_timeout" else None)
            assert initial.prompt == "ready"
            result = await session.step(
                env.ToolCall(name="look", arguments={}), timeout=0 if mode == "step_timeout" else None
            )
            assert result == env.StepResult(observation="stepped", reward=1.0, done=True)

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
    worker.remote.assert_called_once_with(task=task)
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
    initial = env.StepResult(observation="Goal", reward=0.0, done=False)
    environment.reset.return_value = initial
    environment.step.return_value = initial
    tools = env.ALFWorldTools(environment=environment)
    reset = tools.reset()
    assert initial.observation in reset.prompt
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
def test_placement_aliases(command: str) -> None:
    environment = Mock(commands=[command])
    environment.reset.return_value = env.StepResult(observation="Goal", reward=0.0, done=False)
    environment.step.return_value = env.StepResult(observation="Done", reward=1.0, done=True)
    tools = env.ALFWorldTools(environment=environment)
    tools.reset()
    result = tools.call(env.ToolCall(name="move", arguments={"object": "mug 1", "location": "desk 1"}))
    assert result.done and result.reward == 1.0
    environment.step.assert_called_once_with(command)


def test_tool_calls_do_not_replay_terminal_result() -> None:
    environment = Mock(commands=["look"])
    environment.reset.return_value = env.StepResult(observation="Goal", reward=0.0, done=False)
    terminal = env.StepResult(observation="Done", reward=1.0, done=True)
    environment.step.side_effect = [terminal, RuntimeError("Episode finished")]
    tools = env.ALFWorldTools(environment=environment)
    tools.reset()
    assert tools.call(env.ToolCall(name="look", arguments={})) == terminal
    with pytest.raises(RuntimeError, match="Episode finished"):
        tools.call(env.ToolCall(name="look", arguments={}))
    assert environment.step.call_count == 2


def test_tools_do_not_limit_steps() -> None:
    environment = Mock(commands=["look"])
    initial = env.StepResult(observation="Goal", reward=0.0, done=False)
    environment.reset.return_value = environment.step.return_value = initial
    tools = env.ALFWorldTools(environment=environment)
    tools.reset()
    assert not tools.call(env.ToolCall(name="look", arguments={})).done
    for _ in range(3):
        assert tools.call(env.ToolCall(name="look", arguments={})) == initial
    assert environment.step.call_count == 4
    environment.step.assert_called_with("look")
    tools.reset()
    assert not tools.call(env.ToolCall(name="look", arguments={})).done


@pytest.mark.parametrize(
    ("name", "arguments"),
    [("unknown", {}), ("take", {"object": "mug 1"}), ("look", {"extra": "x"}), ("use", {"object": 1})],
    ids=["unknown", "missing-argument", "extra-argument", "argument-type"],
)
def test_invalid_action_does_not_reach_worker(name: str, arguments: dict[str, object]) -> None:
    session = env.ALFWorldSession(task=env.ALFWorldTask(id="a", config_file="config.yaml"))
    with pytest.raises(env.ToolCallError):
        asyncio.run(session.step(env.ToolCall(name=name, arguments=arguments)))
