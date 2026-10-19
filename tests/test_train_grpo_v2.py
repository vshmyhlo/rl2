import asyncio
import io
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock
from urllib.error import URLError

import httpx
import pytest
import renderers.base as renderer_base
import verifiers.v1 as vf
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from verifiers.v1.clients import base as client_base
from verifiers.v1.clients.train import ElasticRendererPool
from verifiers.v1.runtimes import subprocess as subprocess_runtime
from verifiers.v1.trace import AgentInfo, TraceTask

from rl2 import train_grpo_v2 as training
from rl2.addition_env import AdditionTaskset, AdditionTools


@pytest.mark.parametrize("failure", [None, "launch", "rollout", "shutdown"])
def test_serve_vllm(failure: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
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
    ]
    server = MagicMock()
    server.__enter__.return_value = server
    launch = Mock(return_value=server, side_effect=OSError("launch failed") if failure == "launch" else None)
    monkeypatch.setattr(training.subprocess, "Popen", launch)
    if failure == "shutdown":
        server.wait.side_effect = subprocess.TimeoutExpired(cmd=command, timeout=10)

    if failure in {"launch", "rollout"}:
        with pytest.raises(OSError if failure == "launch" else RuntimeError), training.serve_vllm():
            raise RuntimeError("rollout failed")
    else:
        with training.serve_vllm():
            server.terminate.assert_not_called()

    # No env override: vLLM inherits Ray's CUDA_VISIBLE_DEVICES assignment.
    launch.assert_called_once_with(command)
    if failure == "launch":
        server.terminate.assert_not_called()
    else:
        server.terminate.assert_called_once_with()
        server.wait.assert_called_once_with(timeout=10)
        server.__exit__.assert_called_once()
        assert server.kill.call_count == (1 if failure == "shutdown" else 0)


def test_actor_and_learner_each_reserve_one_gpu() -> None:
    assert training.Actor._default_options["num_gpus"] == 1
    assert training.Learner._default_options["num_gpus"] == 1
    assert training.Batcher._default_options.get("num_gpus", 0) == 0


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
    actor = training.Actor.__ray_metadata__.modified_class(batcher=Mock(), group_size=2)
    monkeypatch.setattr(training.random, "choice", Mock(return_value=actor.tasks[0]))
    return actor


def test_environment_examples(actor: Any) -> None:
    assert len(actor.tasks) == 5
    assert [task.data.idx for task in actor.tasks] == list(range(5))
    assert [task.data.answer for task in actor.tasks] == [42, 21, 9, 3, 43]
    task = actor.fetch_next_prompt()
    assert task.data.id == "0"
    assert task.data.prompt == "What is 17 + 25?"
    tools = task.toolsets(task.config)
    assert isinstance(tools[0], AdditionTools)
    assert asyncio.run(tools[0].add(a=17, b=25)) == 42
    agent = training.create_agent()
    assert agent.harness.SUPPORTS_MCP
    config = agent.config
    assert config.harness.id == "null"
    assert config.runtime.type == "subprocess"
    assert config.client.base_url == f"{training.SERVER_URL}/v1"
    assert isinstance(config.client, vf.TrainClientConfig)
    assert config.client.renderer_model_name == training.MODEL
    assert config.max_turns == 3


def test_taskset_applies_config_and_prompt_override(tmp_path: Path) -> None:
    prompt = tmp_path / "system.txt"
    prompt.write_text("Use the addition tool.")
    config = vf.TasksetConfig(system_prompt=prompt)
    tasks = list(AdditionTaskset(config=config))
    assert all(task.config is config.task for task in tasks)
    assert all(task.data.system_prompt == "Use the addition tool." for task in tasks)


@pytest.mark.parametrize(("reply", "reward"), [(" 42\n", 1.0), ("41", 0.0), ("", 0.0)])
def test_correct_answer(reply: str, reward: float, actor: Any) -> None:
    task = actor.tasks[0]
    trace = vf.Trace(
        task=TraceTask(type="AdditionTask", data=task.data),
        agent=AgentInfo(config=training.create_agent().config),
        root_reply=reply,
    )
    assert asyncio.run(task.correct_answer(trace)) == reward


@pytest.mark.parametrize("size", [0, 1])
def test_group_requires_multiple_rollouts(size: int) -> None:
    with pytest.raises(ValueError, match="group_size"):
        training.Actor.__ray_metadata__.modified_class(batcher=Mock(), group_size=size)


@pytest.mark.parametrize("failure", [None, "rollout", "score", "tokens"])
def test_actor_submits_only_complete_groups(failure: str | None, actor: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    task = actor.tasks[0]
    traces = [
        vf.Trace(
            task=TraceTask(type="AdditionTask", data=task.data),
            agent=AgentInfo(config=training.create_agent().config),
            root_reply=reply,
            nodes=[
                vf.MessageNode(
                    message=vf.AssistantMessage(content=reply),
                    sampled=True,
                    token_ids=[42 if reply == "42" else 41],
                    mask=[True],
                    logprobs=[-0.2],
                )
            ],
            ok=True,
            stop_condition="agent_completed",
        )
        for reply in ["42", "41"]
    ]
    if failure == "rollout":
        traces[1].ok = False
    if failure == "score":
        monkeypatch.setattr(task, "score", AsyncMock(side_effect=ValueError("scoring failed")))
    if failure == "tokens":
        traces[1].nodes = []
    env = AsyncMock()
    env.__aenter__.return_value = env
    seen: list[training.AdditionTask] = []

    async def run(sample: training.AdditionTask) -> vf.Trace:
        index = len(seen)
        seen.append(sample)
        # Finish out of order to check reward/trace/advantage alignment.
        if index == 0:
            await asyncio.sleep(0)
        return traces[index]

    env.run.side_effect = run
    monkeypatch.setattr(training, "create_agent", Mock(return_value=env))
    remote = Mock()
    monkeypatch.setattr(training.ray, "remote", remote)
    monkeypatch.setattr(training.ray, "get", Mock())

    server = MagicMock()
    serve = Mock(return_value=server)
    monkeypatch.setattr(training, "serve_vllm", serve)

    def check_server_started() -> None:
        serve.assert_called_once_with()
        server.__enter__.assert_called_once_with()
        remote.assert_not_called()

    monkeypatch.setattr(training, "wait_for_server", Mock(side_effect=check_server_started))
    actor.fetch_next_prompt = Mock(side_effect=[task, StopIteration])
    error = {None: StopIteration, "rollout": RuntimeError, "score": ValueError, "tokens": ValueError}[failure]
    with pytest.raises(error):
        actor.start()
    assert len(seen) == actor.group_size
    assert seen[0] is not seen[1] and all(sample is not task for sample in seen)
    assert all(sample.data == task.data and sample.scoring_deferred for sample in seen)
    assert not task.scoring_deferred
    env.__aexit__.assert_awaited_once()
    server.__exit__.assert_called_once()
    submit = actor.batcher.submit_group.remote
    if failure:
        submit.assert_not_called()
    else:
        submit.assert_called_once()
        group = submit.call_args.args[0]
        assert group.task_id == task.data.id
        assert [rollout.reward for rollout in group.rollouts] == [1.0, 0.0]
        assert [rollout.advantage for rollout in group.rollouts] == pytest.approx([1.0, -1.0], abs=3e-4)
        assert group.prompt == task.data.prompt_text
        assert [rollout.completion for rollout in group.rollouts] == ["42", "41"]
        assert [rollout.sequence.token_ids for rollout in group.rollouts] == [[42], [41]]
        assert pickle.loads(pickle.dumps(group)) == group


@pytest.fixture
def token_trace(actor: Any) -> vf.Trace:
    return vf.Trace(
        task=TraceTask(type="AdditionTask", data=actor.tasks[0].data),
        agent=AgentInfo(config=training.create_agent().config),
        nodes=[
            vf.MessageNode(message=vf.UserMessage(content="question"), token_ids=[10], mask=[False]),
            vf.MessageNode(
                parent=0,
                message=vf.AssistantMessage(content="tool call"),
                sampled=True,
                token_ids=[20],
                mask=[True],
                logprobs=[-0.2],
            ),
            vf.MessageNode(
                parent=1,
                message=vf.ToolMessage(content="42", tool_call_id="call_add"),
                token_ids=[30],
                mask=[False],
            ),
            vf.MessageNode(
                parent=2,
                message=vf.AssistantMessage(content="42"),
                sampled=True,
                token_ids=[40, 41],
                mask=[True, True],
                logprobs=[-0.4, -0.5],
            ),
        ],
    )


def test_training_sequence_preserves_tokens_and_masks_context(token_trace: vf.Trace) -> None:
    sequence = training.training_sequence(token_trace)
    assert sequence.token_ids == [10, 20, 30, 40, 41]
    assert sequence.loss_mask == [False, True, False, True, True]
    assert sequence.old_log_probs == [0.0, -0.2, 0.0, -0.4, -0.5]
    assert len(sequence.token_ids) == len(sequence.loss_mask) == len(sequence.old_log_probs)
    assert pickle.loads(pickle.dumps(sequence)) == sequence


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("empty", "exactly one"),
        ("branched", "exactly one"),
        ("no-tokens", "no trainable tokens"),
        ("mask-length", "equal lengths"),
        ("missing-logprobs", "log-probability"),
        ("missing-turn-tokens", "Sampled messages"),
    ],
)
def test_training_sequence_rejects_invalid_tokens(failure: str, message: str, token_trace: vf.Trace) -> None:
    trace = token_trace
    if failure == "empty":
        trace.nodes = []
    elif failure == "branched":
        trace.nodes.append(
            vf.MessageNode(
                parent=1,
                message=vf.AssistantMessage(content="alternative"),
                sampled=True,
                token_ids=[50],
                mask=[True],
                logprobs=[-0.6],
            )
        )

    elif failure == "no-tokens":
        for node in trace.nodes:
            node.token_ids, node.mask, node.logprobs = [], [], []
    elif failure == "mask-length":
        trace.nodes[-1].mask = [True]
    elif failure == "missing-logprobs":
        trace.nodes[-1].logprobs = []
    elif failure == "missing-turn-tokens":
        trace.nodes[-1].token_ids, trace.nodes[-1].mask, trace.nodes[-1].logprobs = [], [], []
    with pytest.raises(ValueError, match=message):
        training.training_sequence(trace)


def test_group_cancels_pending_rollouts(actor: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    env = AsyncMock()
    env.__aenter__.return_value = env
    started = asyncio.Event()
    cancelled: list[bool] = []
    count = 0

    async def run(task: training.AdditionTask) -> vf.Trace:
        nonlocal count
        count += 1
        if count == 1:
            await started.wait()
            raise ValueError("generation failed")
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.append(True)

    env.run.side_effect = run
    monkeypatch.setattr(training, "create_agent", Mock(return_value=env))
    with pytest.raises(ExceptionGroup, match="TaskGroup"):
        actor.collect_group(actor.tasks[0])
    assert cancelled == [True]
    env.__aexit__.assert_awaited_once()


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


@pytest.fixture
def tokenizer(monkeypatch: pytest.MonkeyPatch) -> PreTrainedTokenizerFast:
    # Exercise the real Qwen3 renderer without downloading a model tokenizer.
    backend = Tokenizer(
        models.BPE(vocab={c: i for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}, merges=[])
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        name_or_path=training.MODEL,
        eos_token="<|im_end|>",
        additional_special_tokens=[
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
            "<think>",
            "</think>",
        ],
    )
    monkeypatch.setattr(renderer_base, "load_tokenizer", Mock(return_value=tokenizer))
    monkeypatch.setattr(ElasticRendererPool, "_renderers", {})
    monkeypatch.setattr(ElasticRendererPool, "_locks", {})
    return tokenizer


@pytest.mark.parametrize("mode", ["success", "tool-error", "provider-error"])
def test_verifiers_tool_rollout(
    mode: str, actor: Any, local_harness: None, tokenizer: PreTrainedTokenizerFast, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[dict[str, Any]] = []
    arguments = '{"a":17}' if mode == "tool-error" else '{"a":17,"b":25}'
    tool_text = '<tool_call>\n{"name":"add","arguments":' + arguments + "}\n</tool_call><|im_end|>"
    tool_ids = tokenizer.encode(tool_text)  # (C_tool,)
    answer_ids = tokenizer.encode("42<|im_end|>")  # (C_answer,)

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(status_code=200, json={"data": [{"id": training.MODEL, "max_model_len": 8192}]})
        assert str(request.url) == f"{training.SERVER_URL}/inference/v1/generate"
        payload = json.loads(request.content)
        requests.append(payload)
        if mode == "provider-error":
            return httpx.Response(status_code=400, json={"error": {"message": "invalid model"}})
        prompt = tokenizer.decode(payload["token_ids"])
        completion_ids = tool_ids if "<tool_response>" not in prompt or mode == "tool-error" else answer_ids  # (C,)
        return httpx.Response(
            status_code=200,
            json={
                "request_id": f"completion-{len(requests)}",
                "choices": [
                    {
                        "index": 0,
                        "token_ids": completion_ids,
                        "finish_reason": "stop",
                        "logprobs": {
                            "content": [{"token": f"token_id:{token}", "logprob": -0.2} for token in completion_ids]
                        },
                    }
                ],
            },
        )

    clients: list[httpx.AsyncClient] = []

    def build_client(config: vf.TrainClientConfig) -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        clients.append(client)
        return client

    monkeypatch.setattr(client_base, "build_async_httpx", build_client)
    if mode == "provider-error":
        with pytest.raises(RuntimeError, match="Rollout failed"):
            actor.collect_group(actor.tasks[0])
    else:
        group = actor.collect_group(actor.tasks[0])
        assert len(group.rollouts) == actor.group_size
        assert len(requests) == actor.group_size * (2 if mode == "success" else 3)
        expected_samples = tool_ids + answer_ids if mode == "success" else tool_ids * 3  # (S,)
        for rollout in group.rollouts:
            assert rollout.num_turns == (2 if mode == "success" else 3)
            assert rollout.num_output_tokens == len(expected_samples)
            assert rollout.rollout_seconds > 0
            assert rollout.reward == (1.0 if mode == "success" else 0.0)
            assert rollout.advantage == 0.0
            sequence = rollout.sequence
            assert len(sequence.token_ids) == len(sequence.loss_mask) == len(sequence.old_log_probs)
            assert rollout.num_input_tokens == len(sequence.token_ids) - sum(sequence.loss_mask)
            assert [
                token for token, train in zip(sequence.token_ids, sequence.loss_mask, strict=True) if train
            ] == expected_samples
            assert sequence.old_log_probs == [-0.2 if train else 0.0 for train in sequence.loss_mask]
            if mode == "success":
                assert rollout.completion == "42"
                assert rollout.stop_condition == "agent_completed"
            else:
                assert rollout.stop_condition == "max_turns"
        prompts = [tokenizer.decode(request["token_ids"]) for request in requests]
        if mode == "success":
            assert any("<tool_response>\n42" in prompt for prompt in prompts)
        else:
            observations = [
                prompt.split("<tool_response>\n", 1)[1] for prompt in prompts if "<tool_response>\n" in prompt
            ]
            assert observations and all(
                "required" in observation.lower() and "b" in observation for observation in observations
            )
        assert pickle.loads(pickle.dumps(group)) == group
    assert clients and all(client.is_closed for client in clients)
    assert requests[0]["model"] == training.MODEL
    assert requests[0]["sampling_params"]["max_tokens"] == 512
    assert requests[0]["sampling_params"]["logprobs"] == 1
    prompt = tokenizer.decode(requests[0]["token_ids"])
    assert '"name": "add"' in prompt
    assert prompt.endswith("<think>\n\n</think>\n\n")
