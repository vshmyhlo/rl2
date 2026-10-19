import asyncio
import io
import json
import pickle
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock
from urllib.error import URLError

import httpx
import numpy as np
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
    monkeypatch.setattr(training, "serve_vllm", Mock(return_value=MagicMock()))
    monkeypatch.setattr(training, "wait_for_server", Mock())
    actor = training.Actor.__ray_metadata__.modified_class()
    monkeypatch.setattr(training.random, "choice", Mock(return_value=actor.tasks[0]))
    return actor


@pytest.mark.parametrize("startup_fails", [False, True])
def test_actor_server_lifetime(startup_fails: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    server = MagicMock()
    serve = Mock(return_value=server)
    monkeypatch.setattr(training, "serve_vllm", serve)
    ready = Mock(side_effect=TimeoutError("not ready") if startup_fails else None)
    monkeypatch.setattr(training, "wait_for_server", ready)
    group = training.RolloutGroup(task_id="0", prompt="test", rollouts=[])

    if startup_fails:
        with pytest.raises(TimeoutError, match="not ready"):
            training.Actor.__ray_metadata__.modified_class()
    else:
        actor = training.Actor.__ray_metadata__.modified_class()
        actor.collect_group = AsyncMock(return_value=group)
        monkeypatch.setattr(training.random, "choice", Mock(return_value=actor.tasks[0]))
        for _ in range(2):
            assert asyncio.run(actor.fetch_next_group(group_size=2)) is group
        actor.collect_group.assert_awaited_with(actor.tasks[0], group_size=2)
        assert actor.collect_group.await_count == 2
        server.__exit__.assert_not_called()
        actor.__ray_shutdown__()
    serve.assert_called_once_with()
    ready.assert_called_once_with()
    server.__enter__.assert_called_once()
    server.__exit__.assert_called_once()


@pytest.mark.parametrize(("num_groups", "fails"), [(1, False), (2, False), (2, True)])
def test_learner_pulls_groups_through_batcher(num_groups: int, fails: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    actor = Mock()
    groups = [training.RolloutGroup(task_id=str(i), prompt="test", rollouts=[]) for i in range(num_groups)]
    actor.fetch_next_group.remote = AsyncMock(
        side_effect=[groups[0], RuntimeError("rollout failed")] if fails else groups
    )
    batcher = training.Batcher.__ray_metadata__.modified_class(actor=actor)
    batcher_handle = Mock()
    batcher_handle.fetch_next_batch.remote.side_effect = batcher.fetch_next_batch
    learner = training.Learner.__ray_metadata__.modified_class(batcher=batcher_handle)

    def get(result: Any) -> Any:
        return asyncio.run(result)

    monkeypatch.setattr(training.ray, "get", get)

    if fails:
        with pytest.raises(RuntimeError, match="rollout failed"):
            learner.fetch_next_batch(num_groups=num_groups, group_size=3)
    else:
        assert learner.fetch_next_batch(num_groups=num_groups, group_size=3) == groups
    batcher_handle.fetch_next_batch.remote.assert_called_once_with(num_groups=num_groups, group_size=3)
    assert actor.fetch_next_group.remote.call_count == num_groups
    assert all(call.kwargs == {"group_size": 3} for call in actor.fetch_next_group.remote.call_args_list)


def test_batch_requires_groups() -> None:
    actor = Mock()
    batcher = training.Batcher.__ray_metadata__.modified_class(actor=actor)
    with pytest.raises(ValueError, match="num_groups"):
        asyncio.run(batcher.fetch_next_batch(num_groups=0, group_size=2))
    actor.fetch_next_group.remote.assert_not_called()


def test_learner_requests_batch_dimensions() -> None:
    learner = training.Learner.__ray_metadata__.modified_class(batcher=Mock())
    batch = [training.RolloutGroup(task_id="0", prompt="test", rollouts=[])]
    learner.fetch_next_batch = Mock(side_effect=[batch, RuntimeError("done")])
    learner.optimize = Mock()
    learner.publish_weights = Mock()
    with pytest.raises(RuntimeError, match="done"):
        learner.start()
    learner.fetch_next_batch.assert_called_with(num_groups=1, group_size=4)
    learner.optimize.assert_called_once_with(batch)
    learner.publish_weights.assert_called_once_with()


@pytest.mark.parametrize("fails", [False, True])
def test_main_gracefully_terminates_actor(fails: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    actor, batcher, learner = Mock(), Mock(), Mock()
    actor.__ray_terminate__ = Mock()
    monkeypatch.setattr(training, "Actor", Mock(remote=Mock(return_value=actor)))
    monkeypatch.setattr(training, "Batcher", Mock(remote=Mock(return_value=batcher)))
    monkeypatch.setattr(training, "Learner", Mock(remote=Mock(return_value=learner)))
    monkeypatch.setattr(training.ray, "init", MagicMock())
    get = Mock()
    monkeypatch.setattr(training.ray, "get", get)
    wait = Mock()
    monkeypatch.setattr(training.ray, "wait", wait)

    def start_learner() -> Any:
        training.Actor.remote.assert_called_once_with()
        if fails:
            raise RuntimeError("training failed")
        return learner.start.remote.return_value

    learner.start.remote.side_effect = start_learner
    if fails:
        with pytest.raises(RuntimeError, match="training failed"):
            training.main()
    else:
        training.main()
    training.Actor.remote.assert_called_once_with()
    training.Batcher.remote.assert_called_once_with(actor=actor)
    training.Learner.remote.assert_called_once_with(batcher=batcher)
    actor.start.remote.assert_not_called()
    actor.__ray_terminate__.remote.assert_called_once_with()
    wait.assert_called_once_with([actor.__ray_terminate__.remote.return_value])
    if fails:
        get.assert_not_called()
    else:
        get.assert_called_once_with(learner.start.remote.return_value)


def test_environment_examples(actor: Any) -> None:
    assert len(actor.tasks) == 5
    assert [task.data.idx for task in actor.tasks] == list(range(5))
    assert [task.data.answer for task in actor.tasks] == [42, 21, 9, 3, 43]
    task = actor.tasks[0]
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
def test_group_requires_multiple_rollouts(size: int, actor: Any) -> None:
    with pytest.raises(ValueError, match="group_size"):
        asyncio.run(actor.fetch_next_group(group_size=size))


@pytest.mark.parametrize("failure", [None, "rollout", "score", "tokens"])
def test_actor_returns_only_complete_groups(failure: str | None, actor: Any, monkeypatch: pytest.MonkeyPatch) -> None:
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
    if failure:
        with pytest.raises(RuntimeError if failure == "rollout" else ValueError):
            asyncio.run(actor.fetch_next_group(group_size=2))
    else:
        group = asyncio.run(actor.fetch_next_group(group_size=2))
    assert len(seen) == 2
    assert seen[0] is not seen[1] and all(sample is not task for sample in seen)
    assert all(sample.data == task.data and sample.scoring_deferred for sample in seen)
    assert not task.scoring_deferred
    env.__aexit__.assert_awaited_once()
    if not failure:
        assert group.task_id == task.data.id
        assert [rollout.reward for rollout in group.rollouts] == [1.0, 0.0]
        assert [rollout.advantage for rollout in group.rollouts] == pytest.approx([1.0, -1.0], abs=3e-4)
        assert group.prompt == task.data.prompt_text
        assert [rollout.completion for rollout in group.rollouts] == ["42", "41"]
        np.testing.assert_array_equal([rollout.sequence.token_ids for rollout in group.rollouts], [[42], [41]])
        np.testing.assert_equal(asdict(pickle.loads(pickle.dumps(group))), asdict(group))


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


@pytest.mark.parametrize("turns", [1, 2], ids=["single-turn", "multi-turn"])
def test_training_sequence_preserves_tokens_and_masks_context(token_trace: vf.Trace, turns: int) -> None:
    token_trace.nodes = token_trace.nodes[: 2 * turns]
    length = 2 if turns == 1 else 5
    sequence = training.training_sequence(token_trace)
    np.testing.assert_array_equal(sequence.token_ids, [10, 20, 30, 40, 41][:length])
    np.testing.assert_array_equal(sequence.loss_mask, [False, True, False, True, True][:length])
    np.testing.assert_allclose(sequence.old_log_probs, [0.0, -0.2, 0.0, -0.4, -0.5][:length])
    assert sequence.token_ids.shape == sequence.loss_mask.shape == sequence.old_log_probs.shape == (length,)
    assert sequence.token_ids.dtype == np.int32
    assert sequence.loss_mask.dtype == np.bool_
    assert sequence.old_log_probs.dtype == np.float32
    np.testing.assert_equal(asdict(pickle.loads(pickle.dumps(sequence))), asdict(sequence))


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


def test_batcher_collects_groups_concurrently(
    actor: Any, token_trace: vf.Trace, monkeypatch: pytest.MonkeyPatch
) -> None:
    all_started = asyncio.Event()
    started = 0
    env = AsyncMock()
    env.__aenter__.return_value = env

    async def run(task: training.AdditionTask) -> vf.Trace:
        nonlocal started
        started += 1
        if started == 4:
            all_started.set()
        # A serialized actor would wait forever for the second group's rollouts.
        await all_started.wait()
        return token_trace.model_copy(deep=True, update={"ok": True})

    env.run.side_effect = run
    monkeypatch.setattr(training, "create_agent", Mock(return_value=env))
    monkeypatch.setattr(training.random, "choice", Mock(side_effect=actor.tasks[:2]))
    handle = Mock()
    handle.fetch_next_group.remote.side_effect = actor.fetch_next_group
    batcher = training.Batcher.__ray_metadata__.modified_class(actor=handle)

    async def collect() -> list[training.RolloutGroup]:
        async with asyncio.timeout(5):
            return await batcher.fetch_next_batch(num_groups=2, group_size=2)

    groups = asyncio.run(collect())
    assert started == 4
    assert [group.task_id for group in groups] == [task.data.id for task in actor.tasks[:2]]
    assert [len(group.rollouts) for group in groups] == [2, 2]
    assert env.__aexit__.await_count == 2


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
        asyncio.run(actor.collect_group(actor.tasks[0], group_size=2))
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
            asyncio.run(actor.collect_group(actor.tasks[0], group_size=2))
    else:
        group = asyncio.run(actor.collect_group(actor.tasks[0], group_size=2))
        assert len(group.rollouts) == 2
        assert len(requests) == 2 * (2 if mode == "success" else 3)
        expected_samples = tool_ids + answer_ids if mode == "success" else tool_ids * 3  # (S,)
        for rollout in group.rollouts:
            assert rollout.num_turns == (2 if mode == "success" else 3)
            assert rollout.num_output_tokens == len(expected_samples)
            assert rollout.rollout_seconds > 0
            assert rollout.reward == (1.0 if mode == "success" else 0.0)
            assert rollout.advantage == 0.0
            sequence = rollout.sequence
            assert sequence.token_ids.shape == sequence.loss_mask.shape == sequence.old_log_probs.shape
            assert rollout.num_input_tokens == len(sequence.token_ids) - sequence.loss_mask.sum()
            np.testing.assert_array_equal(sequence.token_ids[sequence.loss_mask], expected_samples)
            np.testing.assert_allclose(sequence.old_log_probs, np.where(sequence.loss_mask, -0.2, 0.0))
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
        np.testing.assert_equal(asdict(pickle.loads(pickle.dumps(group))), asdict(group))
    assert clients and all(client.is_closed for client in clients)
    assert requests[0]["model"] == training.MODEL
    assert requests[0]["sampling_params"]["max_tokens"] == 512
    assert requests[0]["sampling_params"]["logprobs"] == 1
    prompt = tokenizer.decode(requests[0]["token_ids"])
    assert '"name": "add"' in prompt
    assert prompt.endswith("<think>\n\n</think>\n\n")
