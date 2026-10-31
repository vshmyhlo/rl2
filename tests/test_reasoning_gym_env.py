import asyncio
import json
import pickle
import random
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from rl2.grpo_v2 import reasoning_gym as env
from rl2.sandbox.gvisor import PythonResult


def test_task_stream_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    config = env.ReasoningGymConfig()
    rng = random.Random(42)
    initial_state = rng.getstate()
    create_dataset = Mock()
    monkeypatch.setattr(env, "create_dataset", create_dataset)
    stream = env.stream_reasoning_gym_tasks(config, rng=rng)
    assert iter(stream) is stream
    assert rng.getstate() == initial_state
    tasks = [next(stream), next(stream)]
    assert tasks[0].id != tasks[1].id
    assert all(task.id == f"leg_counting/{task.seed}" and 0 <= task.seed < 2**32 for task in tasks)
    assert all(task.name == "reasoning_gym" and task.config == config for task in tasks)
    assert pickle.loads(pickle.dumps(tasks)) == tasks
    create_dataset.assert_not_called()


@pytest.mark.parametrize(
    "settings",
    [
        {"size": 10000},
        {"task": ""},
        {"task_config": {"seed": 1}},
        {"task_config": {"size": 1}},
        {"unknown": True},
        {"python_namespace": ""},
        {"python_kubectl": []},
        {"python_kubectl": [""]},
    ],
)
def test_invalid_config(settings: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        env.ReasoningGymConfig(**settings)


@pytest.mark.parametrize(
    ("config", "error"),
    [
        (env.ReasoningGymConfig(task="unknown"), ValueError),
        (env.ReasoningGymConfig(task_config={"unknown": 1}), TypeError),
        (env.ReasoningGymConfig(task_config={"min_animals": 0}), AssertionError),
    ],
    ids=["unknown-dataset", "unknown-parameter", "invalid-difficulty"],
)
def test_invalid_dataset_fails_at_worker_startup(config: env.ReasoningGymConfig, error: type[Exception]) -> None:
    task = next(env.stream_reasoning_gym_tasks(config, rng=random.Random(1)))
    with pytest.raises(error):
        env.ReasoningGymWorker.__ray_metadata__.modified_class(task=task)


def test_real_dataset_rewards_and_independent_episodes() -> None:
    config = env.ReasoningGymConfig(task_config={"min_animals": 1, "max_animals": 1})
    task = env.ReasoningGymTask(config=config, seed=42)
    dataset = env.create_dataset(config, seed=42)
    assert len(dataset) == 1
    entry = dataset[0]
    assert entry["metadata"]["num_animals"] == 1
    worker_class = env.ReasoningGymWorker.__ray_metadata__.modified_class
    first = worker_class(task=task)
    second = worker_class(task=task)
    submission = env.ToolCall(name="submit_answer", arguments={"answer": entry["answer"]})
    with pytest.raises(RuntimeError, match="Reset"):
        first.step(submission)
    initial = first.reset()
    assert initial == second.reset()
    assert initial.prompt == entry["question"]
    assert initial.tools == env.tool_schemas()
    assert "submit_answer" in initial.system_prompt
    assert first.step(submission) == env.StepResult(observation="Answer submitted.", reward=1.0, done=True)
    assert second.step(env.ToolCall(name="submit_answer", arguments={"answer": "wrong"})) == env.StepResult(
        observation="Answer submitted.", reward=0.0, done=True
    )
    with pytest.raises(RuntimeError, match="Reset"):
        first.step(submission)
    assert first.reset() == initial
    answer = f"The answer is {entry['answer']}"
    result = first.step(env.ToolCall(name="submit_answer", arguments={"answer": answer}))
    assert result.done and 0 < result.reward < 1
    assert result.reward == dataset.score_answer(answer=answer, entry=entry)


def test_calculator_feedback_and_episode_lifecycle() -> None:
    task = env.ReasoningGymTask(config=env.ReasoningGymConfig(), seed=1)
    worker = env.ReasoningGymWorker.__ray_metadata__.modified_class(task=task)
    calculation = env.ToolCall(name="calculator", arguments={"expression": "12 * (3 + 4)"})
    with pytest.raises(RuntimeError, match="Reset"):
        worker.step(calculation)
    initial = worker.reset()
    schemas = {tool["function"]["name"]: tool["function"] for tool in initial.tools}
    assert set(schemas) == {"calculator", "python", "submit_answer"}
    assert schemas["calculator"]["parameters"] == {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "Arithmetic expression to evaluate."}},
        "required": ["expression"],
        "additionalProperties": False,
    }
    assert "calculator" in initial.system_prompt
    scorer = Mock(wraps=worker.dataset.score_answer)
    worker.dataset.score_answer = scorer
    for expression in ("1 / 0", "__import__('os').getcwd()"):
        result = worker.step(env.ToolCall(name="calculator", arguments={"expression": expression}))
        assert result.observation.startswith("Error:")
        assert result.reward == 0.0 and not result.done
    assert worker.step(calculation) == env.StepResult(observation="84", reward=0.0, done=False)
    scorer.assert_not_called()
    submission = env.ToolCall(name="submit_answer", arguments={"answer": worker.entry["answer"]})
    assert worker.step(submission) == env.StepResult(observation="Answer submitted.", reward=1.0, done=True)
    scorer.assert_called_once_with(answer=submission.arguments["answer"], entry=worker.entry)
    with pytest.raises(RuntimeError, match="Reset"):
        worker.step(calculation)
    worker.reset()
    assert worker.step(calculation).observation == "84"


def test_composite_dataset_native_scoring() -> None:
    config = env.ReasoningGymConfig(
        task="composite",
        task_config={"datasets": [{"name": "leg_counting", "weight": 1, "config": {}}]},
    )
    task = env.ReasoningGymTask(config=config, seed=1)
    dataset = env.create_dataset(config, seed=1)
    entry = dataset[0]
    worker = env.ReasoningGymWorker.__ray_metadata__.modified_class(task=task)
    assert worker.reset().prompt == entry["question"]
    assert worker.step(env.ToolCall(name="submit_answer", arguments={"answer": entry["answer"]})).reward == 1.0
    assert isinstance(config.task_config["datasets"][0], dict)


@pytest.mark.parametrize("settings", [{"seed": None}, {"size": 10}], ids=["random-seed", "size"])
def test_composite_cannot_override_sampling(settings: dict[str, object]) -> None:
    config = env.ReasoningGymConfig(
        task="composite",
        task_config={"datasets": [{"name": "leg_counting", "weight": 1, "config": settings}]},
    )
    with pytest.raises(ValueError, match="cannot override seed or size"):
        env.create_dataset(config, seed=1)


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("unknown", {}),
        ("submit_answer", {}),
        ("submit_answer", {"answer": "42", "extra": True}),
        ("submit_answer", {"answer": 42}),
        ("calculator", {}),
        ("calculator", {"expression": "1 + 1", "answer": "2"}),
        ("calculator", {"expression": 42}),
        ("python", {}),
        ("python", {"code": "print(42)", "extra": True}),
        ("python", {"code": 42}),
        ("python", {"code": "é" * 8193}),
        ("python", {"code": "\0"}),
    ],
    ids=[
        "unknown",
        "missing-answer",
        "extra-answer",
        "answer-type",
        "missing-expression",
        "extra-expression",
        "expression-type",
        "missing-code",
        "extra-code",
        "code-type",
        "code-byte-limit",
        "code-nul",
    ],
)
def test_invalid_tool_call_does_not_reach_worker(name: str, arguments: dict[str, object]) -> None:
    actor = Mock()
    session = env.ReasoningGymSession(actor=actor, config=env.ReasoningGymConfig())
    with pytest.raises(env.ToolCallError):
        asyncio.run(session.step(env.ToolCall(name=name, arguments=arguments)))
    actor.step.remote.assert_not_called()


@pytest.mark.parametrize("mode", ["normal", "failure", "reset_timeout", "step_timeout", "cancel"])
def test_session_worker_cleanup(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    actor = Mock()
    actor.reset.remote = AsyncMock(return_value=env.ResetResult(tools=[], system_prompt="", prompt="question"))
    actor.step.remote = AsyncMock(return_value=env.StepResult(observation="Answer submitted.", reward=0.5, done=True))
    constructor = Mock(return_value=actor)
    kill = Mock()
    monkeypatch.setattr(env.ReasoningGymWorker, "remote", constructor)
    monkeypatch.setattr(env.ray, "kill", kill)
    if mode == "failure":
        actor.step.remote.side_effect = RuntimeError("scorer failed")
    elif mode == "cancel":
        actor.step.remote.side_effect = asyncio.CancelledError

    async def pending(*actions: env.ToolCall) -> None:
        await asyncio.Future()

    if mode == "reset_timeout":
        actor.reset.remote.side_effect = pending
    elif mode == "step_timeout":
        actor.step.remote.side_effect = pending

    task = env.ReasoningGymTask(config=env.ReasoningGymConfig(), seed=1)

    async def run() -> None:
        async with task.init_env() as session:
            initial = await session.reset(timeout=0 if mode == "reset_timeout" else None)
            assert initial.prompt == "question"
            result = await session.step(
                env.ToolCall(name="submit_answer", arguments={"answer": "42"}),
                timeout=0 if mode == "step_timeout" else None,
            )
            assert result.done and result.reward == 0.5

    errors = {
        "failure": RuntimeError,
        "reset_timeout": TimeoutError,
        "step_timeout": TimeoutError,
        "cancel": asyncio.CancelledError,
    }
    if mode == "normal":
        asyncio.run(run())
        actor.step.remote.assert_awaited_once_with(env.ToolCall(name="submit_answer", arguments={"answer": "42"}))
    else:
        with pytest.raises(errors[mode]):
            asyncio.run(run())
    constructor.assert_called_once_with(task=task)
    kill.assert_called_once_with(actor)


def test_python_tool_reuses_sandbox_and_reset_isolates_episodes(monkeypatch: pytest.MonkeyPatch) -> None:
    config = env.ReasoningGymConfig(python_namespace="sandbox-test", python_kubectl=("sudo", "-n", "k3s", "kubectl"))
    task = env.ReasoningGymTask(config=config, seed=1)
    worker = env.ReasoningGymWorker.__ray_metadata__.modified_class(task=task)
    actor = Mock()
    actor.reset.remote = AsyncMock(side_effect=worker.reset)
    actor.step.remote = AsyncMock(side_effect=worker.step)
    monkeypatch.setattr(env.ReasoningGymWorker, "remote", Mock(return_value=actor))
    monkeypatch.setattr(env.ray, "kill", Mock())
    first, second = AsyncMock(spec=env.PythonSession), AsyncMock(spec=env.PythonSession)
    first.__aenter__.return_value = first
    second.__aenter__.return_value = second
    first.execute.side_effect = [
        PythonResult(output="", exit_code=0, output_truncated=False),
        PythonResult(output="ValueError: example\n", exit_code=1, output_truncated=False),
        PythonResult(output="42\n", exit_code=0, output_truncated=True),
    ]
    second.execute.return_value = PythonResult(output="NameError: x\n", exit_code=1, output_truncated=False)
    constructor = Mock(side_effect=[first, second])
    monkeypatch.setattr(env, "PythonSession", constructor)

    async def run() -> None:
        async with task.init_env() as session:
            code = env.ToolCall(name="python", arguments={"code": "x = 40"})
            with pytest.raises(RuntimeError, match="Reset"):
                await session.step(code)
            initial = await session.reset()
            schema = next(tool["function"] for tool in initial.tools if tool["function"]["name"] == "python")
            assert schema["parameters"] == {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python statements to execute."}},
                "required": ["code"],
                "additionalProperties": False,
            }
            assert "python" in initial.system_prompt
            await session.step(env.ToolCall(name="calculator", arguments={"expression": "6 * 7"}))
            constructor.assert_not_called()
            for source, output, exit_code, truncated in [
                ("x = 40", "", 0, False),
                ("raise ValueError('example')", "ValueError: example\n", 1, False),
                ("print(x + 2)", "42\n", 0, True),
            ]:
                result = await session.step(env.ToolCall(name="python", arguments={"code": source}))
                assert result.reward == 0.0 and not result.done
                assert json.loads(result.observation) == {
                    "output": output,
                    "exit_code": exit_code,
                    "output_truncated": truncated,
                }
            constructor.assert_called_once_with(namespace="sandbox-test", kubectl=config.python_kubectl)
            assert [call.args[0] for call in first.execute.await_args_list] == [
                "x = 40",
                "raise ValueError('example')",
                "print(x + 2)",
            ]
            assert actor.step.remote.await_count == 1  # Only the calculator reaches the scoring worker.
            submission = env.ToolCall(name="submit_answer", arguments={"answer": worker.entry["answer"]})
            assert (await session.step(submission)).reward == 1.0
            with pytest.raises(RuntimeError, match="Reset"):
                await session.step(code)
            await session.reset()
            first.__aexit__.assert_awaited_once()
            result = await session.step(env.ToolCall(name="python", arguments={"code": "print(x)"}))
            assert json.loads(result.observation)["exit_code"] == 1
            second.execute.assert_awaited_once_with("print(x)")
        assert constructor.call_count == 2
        second.__aexit__.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["failure", "timeout", "cancel", "startup_failure"])
def test_python_tool_cleanup_on_error(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    actor = Mock()
    actor.reset.remote = AsyncMock(return_value=env.ResetResult(tools=[], system_prompt="", prompt="question"))
    monkeypatch.setattr(env.ReasoningGymWorker, "remote", Mock(return_value=actor))
    kill = Mock()
    monkeypatch.setattr(env.ray, "kill", kill)
    sandbox = AsyncMock(spec=env.PythonSession)
    sandbox.__aenter__.return_value = sandbox
    constructor = Mock(return_value=sandbox)
    monkeypatch.setattr(env, "PythonSession", constructor)

    async def pending(code: str) -> None:
        await asyncio.Future()

    if mode == "startup_failure":
        sandbox.__aenter__.side_effect = RuntimeError("startup failed")
    else:
        sandbox.execute.side_effect = {
            "failure": ValueError("invalid sandbox response"),
            "timeout": pending,
            "cancel": asyncio.CancelledError,
        }[mode]

    async def run() -> None:
        task = env.ReasoningGymTask(config=env.ReasoningGymConfig(), seed=1)
        async with task.init_env() as session:
            await session.reset()
            await session.step(env.ToolCall(name="python", arguments={"code": "print(42)"}), timeout=0.01)

    errors = {
        "failure": ValueError,
        "timeout": TimeoutError,
        "cancel": asyncio.CancelledError,
        "startup_failure": RuntimeError,
    }
    with pytest.raises(errors[mode]) as error:
        asyncio.run(run())
    assert not isinstance(error.value, env.ToolCallError)
    if mode == "startup_failure":
        sandbox.__aexit__.assert_not_awaited()  # PythonSession cleans up its own failed startup.
    else:
        sandbox.__aexit__.assert_awaited_once()
    actor.step.remote.assert_not_called()
    kill.assert_called_once_with(actor)


def test_python_code_limit_accepts_boundary() -> None:
    env.validate_tool_call(env.ToolCall(name="python", arguments={"code": "é" * 8192}))
