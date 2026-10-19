import asyncio
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import verifiers.v1 as vf
from datasets import Dataset
from datasets import config as datasets_config
from verifiers.v1.trace import AgentInfo, TraceTask

from rl2 import toolrl_env as env


def completion(calls: list[dict[str, object]]) -> str:
    return "<think>Choose tools.</think>\n<tool_call>\n" + "\n".join(map(json.dumps, calls)) + "\n</tool_call>"


@pytest.mark.parametrize(
    "prediction,score",
    [
        ('{"name":"add","parameters":{"a":1,"b":2}}', 3.0),
        ('{"name":"add","parameters":{"a":1,"b":9}}', 1.5),
        ('{"name":"add","parameters":{"a":1}}', 0.75),
        ('{"name":"subtract","parameters":{"a":1,"b":2}}', -3.0),
        ('{"name":"add","parameters":[]}', -3.0),
        ('{"name":"add"}', -3.0),
        ("not json", -3.0),
    ],
    ids=[
        "exact",
        "wrong-value",
        "missing-key",
        "wrong-tool",
        "invalid-parameters",
        "missing-parameters",
        "invalid-json",
    ],
)
def test_tool_correctness(prediction: str, score: float) -> None:
    expected = [env.ToolCall(name="add", parameters={"a": 1, "b": 2})]
    assert env.tool_correctness(f"<tool_call>\n{prediction}\n</tool_call>", expected) == score


def test_multiple_calls_match_without_reusing_predictions() -> None:
    calls = [{"name": "add", "parameters": {"a": a}} for a in [1, 2]]
    answer = completion(calls)
    expected = env.parse_tool_calls(answer)
    assert env.tool_correctness(completion(calls[::-1]), expected) == 3.0
    assert env.tool_correctness(completion(calls[:1]), expected) == 0.0
    assert env.tool_correctness(completion(calls + calls[:1]), expected) == pytest.approx(2.6)


def test_parameterless_and_nested_calls() -> None:
    calls = [{"name": "ping", "parameters": {}}, {"name": "query", "parameters": {"filter": {"ids": [1, 2]}}}]
    expected = env.parse_tool_calls(completion(calls))
    assert env.tool_correctness(completion(calls[::-1]), expected) == 3.0
    assert env.match_score([], []) == 1.0
    assert env.match_score([], ["extra"]) == 0.0


@pytest.mark.parametrize(
    "body",
    [
        "",
        "\n<response>Ask for details.</response>",
        '\n<tool_call>\n{"name":"ping","parameters":{}}\n</tool_call>',
        '\n<tool_call>\n{"name":"ping","parameters":{}}\n</tool_call>\n<response>Done.</response>',
    ],
    ids=["thinking-only", "response", "calls", "calls-and-response"],
)
def test_format_reward(body: str) -> None:
    answer = "<think>Plan.</think>" + body
    assert env.format_reward(answer, answer) == 1.0
    assert env.format_reward(answer + "trailing text", answer) == 0.0
    assert env.format_reward(answer.replace("</think>", ""), answer) == 0.0
    if body:
        assert env.format_reward(answer + body, answer) == 0.0


def test_load_parquet_and_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", tmp_path / "cache")
    path = tmp_path / "train.parquet"
    answer = completion([{"name": "add", "parameters": {"a": 1}}])
    response = "<think>Missing input.</think>\n<response>Which number?</response>"
    Dataset.from_list(
        [
            {
                "prompt": [{"role": "system", "content": "Tools: add"}, {"role": "user", "content": "Add one"}],
                "reward_model": {"ground_truth": text, "style": "rule"},
            }
            for text in [answer, response]
        ]
    ).to_parquet(path)
    config = env.ToolRLTasksetConfig(data_file=str(path), task=vf.TaskConfig())
    tasks = list(env.ToolRLTaskset(config=config))
    assert [task.data.id for task in tasks] == ["0", "1"]
    assert [task.data.idx for task in tasks] == [0, 1]
    assert tasks[0].config == config.task
    assert tasks[0].data.prompt_text == "Tools: add\n\nAdd one"
    assert tasks[0].data.prompt[0].content == "Tools: add"
    assert answer not in tasks[0].data.prompt_text
    assert tasks[0].toolsets(config.task) == []
    for task in tasks:
        trace = vf.Trace(
            task=TraceTask(type="ToolRLTask", data=task.data),
            agent=AgentInfo(config=vf.AgentConfig(model="test")),
        )
        trace.root_reply = task.data.answer
        asyncio.run(task.score(trace))
        assert trace.reward == (4.0 if task.data.expected_calls else 1.0)
    trace.root_reply = "<think>Different text.</think>\n<response>Anything.</response>"
    asyncio.run(tasks[1].score(trace))
    assert trace.reward == 1.0  # Upstream does not grade response-only semantics.


@pytest.mark.parametrize("rows", [[], [{"prompt": [], "reward_model": {"ground_truth": "<tool_call>bad</tool_call>"}}]])
def test_invalid_dataset(rows: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "load_dataset", Mock(return_value=rows))
    with pytest.raises(ValueError, match="empty|Invalid ToolRL row 0"):
        list(env.ToolRLTaskset(config=env.ToolRLTasksetConfig()))
