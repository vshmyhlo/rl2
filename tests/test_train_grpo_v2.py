import asyncio
import io
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock
from urllib.error import URLError

import httpx
import pytest
import verifiers.v1 as vf
from verifiers.v1.clients import eval as eval_client
from verifiers.v1.runtimes import subprocess as subprocess_runtime
from verifiers.v1.trace import AgentInfo, TraceTask

from rl2 import train_grpo_v2 as training
from rl2.addition_env import AdditionTools


@pytest.mark.parametrize("fails", [False, True], ids=["launch", "server-failure"])
def test_serve_vllm(fails: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "Qwen/Qwen3-0.6B",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]
    run = Mock(side_effect=subprocess.CalledProcessError(returncode=1, cmd=command) if fails else None)
    monkeypatch.setattr(training.subprocess, "run", run)

    if fails:
        with pytest.raises(subprocess.CalledProcessError):
            training.serve_vllm()
    else:
        training.serve_vllm()

    run.assert_called_once_with(command, check=True)


def test_wait_for_server_retries_until_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    response = io.BytesIO()
    request = Mock(side_effect=[URLError("connection refused"), response])
    monkeypatch.setattr(training, "urlopen", request)
    monkeypatch.setattr(training.time, "sleep", Mock())

    training.wait_for_server()

    assert request.call_count == 2
    request.assert_called_with(f"{training.SERVER_URL}/health", timeout=5)
    assert response.closed


def test_wait_for_server_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training, "urlopen", Mock(side_effect=URLError("connection refused")))
    monkeypatch.setattr(training.time, "monotonic", Mock(side_effect=[0, 0, 1, 2]))
    monkeypatch.setattr(training.time, "sleep", Mock())

    with pytest.raises(TimeoutError, match="vLLM was not ready"):
        training.wait_for_server(timeout=2)


@pytest.fixture
def actor(monkeypatch: pytest.MonkeyPatch) -> Any:
    actor = training.Actor.__ray_metadata__.modified_class(batcher=Mock())
    monkeypatch.setattr(training.random, "choice", Mock(return_value=actor.tasks[0]))
    return actor


def test_environment_examples(actor: Any) -> None:
    assert len(actor.tasks) == 5
    assert [task.data.answer for task in actor.tasks] == [42, 21, 9, 3, 43]
    task = actor.fetch_next_prompt()
    assert task.data.id == "0"
    assert task.data.prompt == "What is 17 + 25?"
    tools = task.toolsets(task.config)
    assert isinstance(tools[0], AdditionTools)
    assert asyncio.run(tools[0].add(a=17, b=25)) == 42
    config = training.create_environment().config
    assert config.harness.id == "null"
    assert config.runtime.type == "subprocess"
    assert config.client.base_url == f"{training.SERVER_URL}/v1"
    assert config.max_turns == 3


@pytest.mark.parametrize(("reply", "reward"), [(" 42\n", 1.0), ("41", 0.0), ("", 0.0)])
def test_correct_answer(reply: str, reward: float, actor: Any) -> None:
    task = actor.tasks[0]
    trace = vf.Trace(
        task=TraceTask(type="AdditionTask", data=task.data),
        agent=AgentInfo(config=training.create_environment().config),
        root_reply=reply,
    )
    assert asyncio.run(task.correct_answer(trace)) == reward


@pytest.mark.parametrize("fails", [False, True], ids=["submit", "failed-rollout"])
def test_actor_submits_only_completed_rollouts(fails: bool, actor: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    env = AsyncMock()
    env.__aenter__.return_value = env
    env.run.return_value = Mock(ok=not fails, reward=1.0, stop_condition="done", errors=[])
    env.run.return_value.to_record.return_value = {"task": {"data": {"id": "0"}}}
    monkeypatch.setattr(training, "create_environment", Mock(return_value=env))
    remote = Mock()
    monkeypatch.setattr(training.ray, "remote", remote)
    monkeypatch.setattr(training.ray, "get", Mock())

    def check_server_started() -> None:
        remote.assert_called_once_with(num_gpus=1, max_retries=0)
        remote.return_value.assert_called_once_with(training.serve_vllm)
        remote.return_value.return_value.remote.assert_called_once_with()

    monkeypatch.setattr(training, "wait_for_server", Mock(side_effect=check_server_started))
    task = actor.tasks[0]
    actor.fetch_next_prompt = Mock(side_effect=[task, StopIteration])
    with pytest.raises(RuntimeError if fails else StopIteration):
        actor.start()
    env.run.assert_awaited_once_with(task)
    env.__aexit__.assert_awaited_once()
    submit = actor.batcher.submit_rollout.remote
    if fails:
        submit.assert_not_called()
    else:
        submit.assert_called_once()
        rollout = submit.call_args.args[0]
        assert rollout.reward == 1.0
        assert rollout.trace["task"]["data"]["id"] == "0"
        assert pickle.loads(pickle.dumps(rollout)) == rollout


@pytest.fixture
def local_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Use installed dependencies instead of provisioning a uv environment in tests.
    async def prepare_script(
        self: subprocess_runtime.SubprocessRuntime,
        script: str | bytes,
        env: dict[str, str] | None = None,
        *,
        activate: bool = True,
    ) -> list[str]:
        path = tmp_path / "harness.py"
        path.write_bytes(script.encode() if isinstance(script, str) else script)
        return [sys.executable, str(path)]

    monkeypatch.setattr(subprocess_runtime, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(subprocess_runtime.SubprocessRuntime, "prepare_uv_script", prepare_script)


@pytest.mark.parametrize("mode", ["success", "tool-error", "provider-error"])
def test_verifiers_tool_rollout(mode: str, actor: Any, local_harness: None, monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{training.SERVER_URL}/v1/chat/completions"
        payload = json.loads(request.content)
        requests.append(payload)
        if mode == "provider-error":
            return httpx.Response(status_code=400, json={"error": {"message": "invalid model"}})
        tool_call = len(requests) == 1 or mode == "tool-error"
        delta: dict[str, Any] = {"role": "assistant", "content": None if tool_call else "42"}
        if tool_call:
            delta["tool_calls"] = [
                {
                    "index": 0,
                    "id": f"call_{len(requests)}",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a":17}' if mode == "tool-error" else '{"a":17,"b":25}'},
                }
            ]
        chunk = {
            "id": f"completion-{len(requests)}",
            "created": 0,
            "object": "chat.completion.chunk",
            "model": training.MODEL,
            "choices": [{"index": 0, "delta": delta, "finish_reason": "tool_calls" if tool_call else "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }
        return httpx.Response(
            status_code=200,
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(eval_client, "build_async_httpx", Mock(return_value=client))
    if mode == "provider-error":
        with pytest.raises(RuntimeError, match="Rollout failed"):
            actor.collect_rollout(actor.tasks[0])
    else:
        rollout = actor.collect_rollout(actor.tasks[0])
        trace = vf.WireTrace.model_validate(rollout.trace)
        assert len(trace.calls) == len(requests) == (2 if mode == "success" else 3)
        assert trace.task.data.answer == 42
        assert trace.reward == rollout.reward == (1.0 if mode == "success" else 0.0)
        assert trace.tool_messages
        if mode == "success":
            assert "42" in trace.tool_messages[0].content
            assert trace.last_reply == "42"
        else:
            assert "b" in trace.tool_messages[0].content
            assert trace.stop_condition == "max_turns"
        assert pickle.loads(pickle.dumps(rollout)) == rollout
    assert client.is_closed
    assert requests[0]["model"] == training.MODEL
    assert requests[0]["max_tokens"] == 512
    assert requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert requests[0]["tools"][0]["function"]["name"] == "add"
