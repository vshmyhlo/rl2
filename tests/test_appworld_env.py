import asyncio
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from rl2 import appworld_env as env


def test_task_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    config = env.AppWorldConfig(python=sys.executable, root="/data/appworld")
    run = Mock(return_value=SimpleNamespace(stdout='startup noise\nRL2_BENCHMARK ["a", "b"]\n'))
    monkeypatch.setattr(env.subprocess, "run", run)
    rng = random.Random(42)
    initial_state = rng.getstate()
    constructor = Mock(wraps=env.AppWorldTask)
    monkeypatch.setattr(env, "AppWorldTask", constructor)
    stream = env.stream_appworld_tasks(config, rng=rng, timeout=10)
    assert iter(stream) is stream
    constructor.assert_not_called()
    assert rng.getstate() == initial_state
    tasks = [next(stream) for _ in range(5)]
    reference = random.Random(42)
    assert [task.id for task in tasks] == [reference.choice(["a", "b"]) for _ in tasks]
    assert rng.getstate() == reference.getstate()
    assert constructor.call_count == len(tasks)
    assert all(task.name == "appworld" and task.config == config for task in tasks)
    state = rng.getstate()
    continuation = [next(stream) for _ in range(3)]
    rng.setstate(state)
    assert [next(stream) for _ in range(3)] == continuation
    run.assert_called_once()
    command = run.call_args.args[0]
    assert command[:2] == [
        sys.executable,
        str(Path(env.__file__).with_name("benchmark_workers") / "appworld_worker.py"),
    ]
    assert command[-1] == "--list"
    settings = json.loads(command[-2])
    assert settings["tool_calls"] and settings["max_steps"] == sys.maxsize
    assert run.call_args.kwargs["timeout"] == 10
    assert run.call_args.kwargs["env"]["APPWORLD_ROOT"] == config.root


@pytest.mark.parametrize(
    "output",
    [
        "",
        "RL2_BENCHMARK []",
        'RL2_BENCHMARK ["a", "a"]',
        'RL2_BENCHMARK [""]',
        "RL2_BENCHMARK [1]",
        'RL2_BENCHMARK ["a"]\nRL2_BENCHMARK ["b"]',
    ],
    ids=["missing", "empty", "duplicate", "empty-id", "wrong-type", "multiple-manifests"],
)
def test_invalid_task_discovery(output: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env.subprocess, "run", Mock(return_value=SimpleNamespace(stdout=output)))
    with pytest.raises(ValueError):
        env.stream_appworld_tasks(
            env.AppWorldConfig(python=sys.executable, root="/apps"), rng=random.Random(1), timeout=10
        )


@pytest.mark.parametrize(
    ("name", "arguments"),
    [("unknown", {}), ("execute", {}), ("execute", {"code": "pass", "extra": True}), ("execute", {"code": 1})],
    ids=["unknown", "missing-argument", "extra-argument", "argument-type"],
)
def test_invalid_tool_call_does_not_reach_worker(name: str, arguments: dict[str, object]) -> None:
    task = env.AppWorldTask(id="a", config=env.AppWorldConfig(python=sys.executable, root="/apps"))
    session = env.AppWorldSession(task=task)
    session.request = AsyncMock()
    with pytest.raises(env.ToolCallError):
        asyncio.run(session.step(env.ToolCall(name=name, arguments=arguments)))
    session.request.assert_not_called()


@pytest.mark.parametrize("initial", [{"reward": 1.0}, {"done": True}], ids=["reward", "terminal"])
def test_reset_requires_active_episode(initial: dict[str, object]) -> None:
    task = env.AppWorldTask(id="a", config=env.AppWorldConfig(python=sys.executable, root="/apps"))
    session = env.AppWorldSession(task=task)
    session.request = AsyncMock(return_value={"observation": "Goal", **initial})
    with pytest.raises(AssertionError, match="AppWorld reset"):
        asyncio.run(session.reset())


@pytest.mark.parametrize("operation", ["reset", "step"])
def test_request_timeout(operation: str) -> None:
    task = env.AppWorldTask(id="a", config=env.AppWorldConfig(python=sys.executable, root="/apps"))
    session = env.AppWorldSession(task=task)

    async def request(**kwargs: object) -> object:
        await asyncio.Future()

    session.request = AsyncMock(side_effect=request)

    async def run() -> None:
        with pytest.raises(TimeoutError):
            if operation == "reset":
                await session.reset(timeout=0)
            else:
                await session.step(env.ToolCall(name="execute", arguments={"code": "pass"}), timeout=0)

    asyncio.run(run())
    session.request.assert_awaited_once()


def test_subprocess_episode_and_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise the actual worker transport without downloading AppWorld's databases.
    (tmp_path / "appworld.py").write_text("""
import os
from pathlib import Path
from types import SimpleNamespace, TracebackType
from typing import Self

def load_task_ids(split: str) -> list[str]:
    assert split == "train"
    assert os.environ["APPWORLD_ROOT"] == "test-root"
    return ["a"]

class AppWorld:
    def __init__(self, task_id: str, experiment_name: str, max_interactions: int) -> None:
        self.output = experiment_name
        self.calls = 0
        self.done = False
        self.task = SimpleNamespace(supervisor={"email": "a@example.com"}, datetime="2026-01-01", instruction="Buy milk")
    def __enter__(self) -> Self:
        return self
    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        assert Path(self.output).exists()
    def execute(self, code: str) -> str:
        self.calls += 1
        self.done = code == "apis.supervisor.complete_task()"
        return f"{self.calls}:{code}:{self.output}"
    def task_completed(self) -> bool:
        return self.done
    def evaluate(self) -> SimpleNamespace:
        assert self.done
        return SimpleNamespace(success=True)
""")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    config = env.AppWorldConfig(python=sys.executable, root="test-root")
    stream = env.stream_appworld_tasks(config, rng=random.Random(1), timeout=10)
    task = next(stream)
    assert next(stream) == task

    async def run() -> None:
        async with task.init_env() as first, task.init_env() as second:
            for session in (first, second):
                initial = await session.reset(timeout=10)
                assert "Buy milk" in initial.prompt and "a@example.com" in initial.prompt
                assert "show_api_doc" in initial.prompt and "2026-01-01" in initial.prompt
                assert "```python" not in initial.prompt
                assert initial.tools == env.tool_schemas()
                assert "execute" in initial.system_prompt
            action = env.ToolCall(name="execute", arguments={"code": 'print("``` inside Python")'})
            one = await first.step(action, timeout=10)
            two = await second.step(action, timeout=10)
            assert one.observation.startswith('1:print("``` inside Python"):')
            assert two.observation.startswith('1:print("``` inside Python"):')
            assert one.reward == two.reward == 0.0 and not one.done and not two.done
            outputs = [Path(result.observation.rsplit(":", 1)[1]) for result in (one, two)]
            assert outputs[0] != outputs[1] and all(path.exists() for path in outputs)
            result = await first.step(
                env.ToolCall(name="execute", arguments={"code": "apis.supervisor.complete_task()"})
            )
            assert result.done and result.reward == 1.0
            assert result.observation.startswith("2:apis.supervisor.complete_task():")
            await first.reset(timeout=10)
            assert not outputs[0].exists()
            restarted = await first.step(action, timeout=10)
            assert restarted.observation.startswith("1:") and not restarted.done
            outputs.append(Path(restarted.observation.rsplit(":", 1)[1]))
        assert first.process.returncode == second.process.returncode == 0
        assert all(not path.exists() for path in outputs)

    asyncio.run(run())
