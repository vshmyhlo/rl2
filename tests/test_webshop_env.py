import asyncio
import os
import random
import runpy
import sys
from contextlib import closing
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, NamedTuple
from unittest.mock import AsyncMock, Mock

import pytest

from rl2 import webshop_env as env


class Upstream(NamedTuple):
    environment: Mock
    constructor: Mock
    engine: SimpleNamespace
    prices: list[float]


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch) -> Upstream:
    environment = Mock()
    environment.server.goals = range(1503)
    environment.reset.return_value = ("Buy a hat", None)
    environment.get_available_actions.return_value = {"clickables": ["Buy Now"]}
    environment.step.side_effect = [("Results", 0.0, False, None), ("Purchased", 0.75, True, None)]
    prices: list[float] = []

    def create_environment(**kwargs: Any) -> Mock:
        prices.append(random.random())
        return environment

    constructor = Mock(side_effect=create_environment)
    engine = SimpleNamespace()
    for name, attributes in {
        "web_agent_site.engine": {"engine": engine},
        "web_agent_site.envs": {"WebAgentTextEnv": constructor},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    # Exercise Gym's entry point without importing optional upstream packages during collection.
    backend = ModuleType("rl2.grpo_v2.webshop")
    backend.__dict__.update(runpy.run_path(str(Path(env.__file__).parent / "grpo_v2/webshop.py")))
    monkeypatch.setitem(sys.modules, "rl2.grpo_v2.webshop", backend)
    return Upstream(environment=environment, constructor=constructor, engine=engine, prices=prices)


def test_worker_episode_and_full_split(upstream: Upstream) -> None:
    config = env.WebShopConfig(root="/data/webshop")
    task = env.WebShopTask(id="1502", config=config)
    worker_class = env.WebShopWorker.__ray_metadata__.modified_class
    with closing(worker_class(task=task)) as worker:
        assert worker.environment.get_wrapper_attr("task_ids")() == ["1500", "1501", "1502"]
        assert upstream.constructor.call_args.kwargs == {
            "observation_mode": "text",
            "file_path": "/data/webshop/data/items_shuffle.json",
            "num_products": None,
            "human_goals": True,
            "limit_goals": -1,
        }
        assert upstream.engine.DEFAULT_ATTR_PATH == "/data/webshop/data/items_ins_v2.json"
        assert upstream.engine.HUMAN_ATTR_PATH == "/data/webshop/data/items_human_ins.json"
        with pytest.raises(RuntimeError, match="Reset"):
            worker.step(env.ToolCall(name="click", arguments={"value": "Buy Now"}))
        initial = worker.reset()
        assert initial.single_tool_call
        assert initial.prompt == 'Buy a hat\nAvailable actions: {"clickables": ["Buy Now"]}'
        schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in initial.tools}
        assert set(schemas) == {"search", "click"}
        assert schemas["search"]["required"] == ["keywords"]
        assert schemas["click"]["required"] == ["value"]
        upstream.environment.reset.assert_called_once_with(session=1502)
        result = worker.step(env.ToolCall(name="search", arguments={"keywords": "red hat"}))
        assert not result.done and result.reward == 0.0
        upstream.environment.step.assert_called_with("search[red hat]")
        result = worker.step(env.ToolCall(name="click", arguments={"value": "Buy Now"}))
        assert result.done and result.reward == 0.75
        upstream.environment.step.assert_called_with("click[Buy Now]")
        with pytest.raises(RuntimeError, match="Reset"):
            worker.step(env.ToolCall(name="click", arguments={"value": "Buy Now"}))
        assert upstream.environment.step.call_count == 2
        worker.reset()
        upstream.environment.step.side_effect = None
        upstream.environment.step.return_value = ("Results", 0, False, None)
        assert worker.step(env.ToolCall(name="search", arguments={"keywords": "hat"})).reward == 0
    upstream.environment.close.assert_called_once()
    assert env.discover_tasks._function(config.root) == ["1500", "1501", "1502"]
    assert upstream.environment.close.call_count == 2
    assert upstream.prices[0] == upstream.prices[1]


@pytest.mark.parametrize("task_id", ["1499", "1503"], ids=["held-out", "past-end"])
def test_reset_rejects_tasks_outside_training_split(task_id: str, upstream: Upstream) -> None:
    with (
        closing(env.make_environment("/data/webshop")) as environment,
        pytest.raises(ValueError, match="training split"),
    ):
        environment.reset(options={"task_id": task_id})
    upstream.environment.reset.assert_not_called()


@pytest.mark.parametrize("info", [{"reward": 1.0}, {"done": True}], ids=["reward", "terminal"])
def test_reset_rejects_inactive_episode(info: dict[str, Any], upstream: Upstream) -> None:
    upstream.environment.reset.return_value = ("Already finished", info)
    task = env.WebShopTask(id="1500", config=env.WebShopConfig(root="/webshop"))
    with closing(env.WebShopWorker.__ray_metadata__.modified_class(task=task)) as worker, pytest.raises(AssertionError):
        worker.reset()


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("unknown", {}),
        ("search", {}),
        ("click", {"value": "Buy Now", "extra": "x"}),
        ("search", {"keywords": 1}),
        ("click", {"value": " "}),
    ],
    ids=["unknown", "missing", "extra", "wrong-type", "blank"],
)
def test_invalid_tool_does_not_reach_actor(name: str, arguments: dict[str, Any]) -> None:
    session = env.WebShopSession(task=env.WebShopTask(id="1500", config=env.WebShopConfig(root="/webshop")))
    with pytest.raises(env.ToolCallError):
        asyncio.run(session.step(env.ToolCall(name=name, arguments=arguments)))


def test_task_discovery_and_reproducible_sampling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = env.WebShopConfig(root="WebShop")
    discover = Mock()
    cancel = Mock()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "/existing")
    monkeypatch.setattr(env, "discover_tasks", discover)
    monkeypatch.setattr(env.ray, "get", Mock(return_value=["1500", "1501", "1502"]))
    monkeypatch.setattr(env.ray, "cancel", cancel)
    stream = env.stream_webshop_tasks(config, random.Random(3), timeout=30)
    rng = random.Random(3)
    tasks = [next(stream) for _ in range(4)]
    assert [task.id for task in tasks] == [rng.choice(["1500", "1501", "1502"]) for _ in tasks]
    assert all(task.config.root == str(tmp_path / "WebShop") for task in tasks)
    assert tasks[0] is not tasks[1]
    discover.options.assert_called_once_with(
        runtime_env={"env_vars": {"PYTHONPATH": str(tmp_path / "WebShop") + os.pathsep + "/existing"}}
    )
    discover.options.return_value.remote.assert_called_once_with(str(tmp_path / "WebShop"))
    env.ray.get.assert_called_once_with(discover.options.return_value.remote.return_value, timeout=30)
    cancel.assert_called_once_with(discover.options.return_value.remote.return_value, force=True)


@pytest.mark.parametrize("failure", ["empty", "empty-id", "duplicate", "timeout"])
def test_task_discovery_failure_cancels_pending_work(failure: str, monkeypatch: pytest.MonkeyPatch) -> None:
    discover = Mock()
    monkeypatch.setattr(env, "discover_tasks", discover)
    monkeypatch.setattr(
        env.ray,
        "get",
        Mock(
            return_value={"empty-id": [""], "duplicate": ["1500", "1500"]}.get(failure, []),
            side_effect=TimeoutError if failure == "timeout" else None,
        ),
    )
    cancel = Mock()
    monkeypatch.setattr(env.ray, "cancel", cancel)
    with pytest.raises(TimeoutError if failure == "timeout" else ValueError):
        env.stream_webshop_tasks(env.WebShopConfig(root="/webshop"), random.Random(3), timeout=30)
    cancel.assert_called_once_with(discover.options.return_value.remote.return_value, force=True)


@pytest.mark.parametrize("mode", ["normal", "crash", "reset_timeout", "step_timeout", "cancel", "close_error"])
def test_session_actor_cleanup(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    actor = Mock()
    worker = Mock()
    worker.options.return_value.remote.return_value = actor
    kill = Mock()
    monkeypatch.setattr(env, "WebShopWorker", worker)
    monkeypatch.setattr(env.ray, "kill", kill)
    task = env.WebShopTask(id="1500", config=env.WebShopConfig(root="/webshop"))
    call = env.ToolCall(name="search", arguments={"keywords": "hat"})

    async def run() -> None:
        async def reset() -> env.ResetResult:
            if mode == "crash":
                raise RuntimeError("worker failed")
            if mode == "reset_timeout":
                await asyncio.Future()
            return env.ResetResult(tools=env.tool_schemas(), system_prompt="Use tools", prompt="Buy a hat")

        async def step(action: env.ToolCall) -> env.StepResult:
            assert action == call
            if mode in {"step_timeout", "cancel"}:
                if mode == "cancel":
                    asyncio.get_running_loop().call_soon(current.cancel)
                await asyncio.Future()
            return env.StepResult(observation="Results", reward=0.0, done=False)

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
            assert initial.prompt == "Buy a hat"
            result = await session.step(call, timeout=0 if mode == "step_timeout" else None)
            assert result == env.StepResult(observation="Results", reward=0.0, done=False)

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
    worker.options.assert_called_once_with(runtime_env=env.worker_runtime_env(task.config))
    worker.options.return_value.remote.assert_called_once_with(task=task)
    kill.assert_called_once_with(actor)
    if mode in {"normal", "close_error"}:
        actor.close.remote.assert_awaited_once_with()
    else:
        actor.close.remote.assert_not_called()
