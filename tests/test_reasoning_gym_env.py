import asyncio
import pickle
import random
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from rl2.grpo_v2 import reasoning_gym as env


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
    [{"size": 10000}, {"task": ""}, {"task_config": {"seed": 1}}, {"task_config": {"size": 1}}, {"unknown": True}],
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
    assert set(schemas) == {"calculator", "submit_answer"}
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
    ],
    ids=[
        "unknown",
        "missing-answer",
        "extra-answer",
        "answer-type",
        "missing-expression",
        "extra-expression",
        "expression-type",
    ],
)
def test_invalid_tool_call_does_not_reach_worker(name: str, arguments: dict[str, object]) -> None:
    actor = Mock()
    session = env.ReasoningGymSession(actor=actor)
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
