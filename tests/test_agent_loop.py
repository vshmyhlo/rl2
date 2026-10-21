import asyncio
import json
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from rl2 import agent_loop as loop
from rl2.alfworld_env import ALFWorldTask, EpisodeResult, ToolCall, tool_schemas


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(
        models.BPE(vocab={c: i for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}, merges=[])
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>", "<|endoftext|>"],
        chat_template=(
            "{% if tools %}{{ tools | tojson }}{% endif %}"
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
            "{% endfor %}{{ '<|im_start|>assistant\n' if add_generation_prompt else '' }}"
        ),
    )


@pytest.fixture
def environment(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    session = AsyncMock()
    session.__aenter__.return_value = session
    session.request.side_effect = [
        EpisodeResult(observation="Goal: put mug on desk. Admissible commands: look", reward=0.0, done=False),
        EpisodeResult(observation="At shelf. Admissible commands: put mug 1 in/on desk 1", reward=0.0, done=False),
        EpisodeResult(observation="Task complete", reward=1.0, done=True),
    ]
    monkeypatch.setattr(loop, "ALFWorldSession", Mock(return_value=session))
    return session


def response_for(tokenizer: PreTrainedTokenizerFast, text: str, finish_reason: str = "stop") -> dict[str, Any]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return {
        "choices": [
            {
                "token_ids": ids,
                "logprobs": {"content": [{"token": f"token_id:{token}", "logprob": -0.25} for token in ids]},
                "finish_reason": finish_reason,
            }
        ],
    }


def make_agent(tokenizer: PreTrainedTokenizerFast, client: httpx.AsyncClient) -> loop.Agent:
    return loop.Agent(
        client=client,
        tokenizer=tokenizer,
        model="Qwen/test",
        context_length=8192,
        max_tokens=128,
        max_turns=3,
        temperature=0.7,
        rollout_timeout=10,
        environment_timeout=2,
    )


def test_multiturn_exact_training_sequence(tokenizer: PreTrainedTokenizerFast, environment: AsyncMock) -> None:
    replies = [
        '<think>inspect</think>\n<tool_call>{"name":"look","arguments":{}}</tool_call><|im_end|>',
        '<tool_call>{"name":"move","arguments":{"object":"mug 1","location":"desk 1"}}</tool_call><|im_end|>',
    ]
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/inference/v1/generate"
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json=response_for(tokenizer, replies[len(requests) - 1]))

    async def run() -> loop.Rollout:
        async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
            return await make_agent(tokenizer, client).run(ALFWorldTask(id="a", config_file="config.yaml"))

    rollout = asyncio.run(run())
    first, second = requests
    sampled = [tokenizer.encode(reply, add_special_tokens=False) for reply in replies]
    prefix = first["token_ids"] + sampled[0]
    assert second["token_ids"][: len(prefix)] == prefix
    assert "<think>inspect</think>" in tokenizer.decode(second["token_ids"])
    expected_ids = second["token_ids"] + sampled[1]
    mask = np.zeros(len(expected_ids), dtype=np.bool_)  # (T,)
    mask[len(first["token_ids"]) : len(prefix)] = True
    mask[len(second["token_ids"]) :] = True
    sequence = rollout.sequence
    assert sequence.token_ids.shape == sequence.loss_mask.shape == sequence.old_log_probs.shape == (len(expected_ids),)
    assert sequence.token_ids.dtype == np.int32
    assert sequence.loss_mask.dtype == np.bool_
    assert sequence.old_log_probs.dtype == np.float32
    np.testing.assert_array_equal(sequence.token_ids, expected_ids)
    np.testing.assert_array_equal(sequence.loss_mask, mask)
    np.testing.assert_array_equal(sequence.old_log_probs, np.where(mask, -0.25, 0.0))
    assert [call.args for call in environment.request.await_args_list] == [
        (),
        (ToolCall(name="look", arguments={}),),
        (ToolCall(name="move", arguments={"object": "mug 1", "location": "desk 1"}),),
    ]
    initial_text = tokenizer.decode(first["token_ids"])
    assert loop.TASK_PROMPT in initial_text
    assert '"name": "move"' in initial_text and '"name": "inventory"' in initial_text
    tool_context = tokenizer.decode(second["token_ids"][len(prefix) :])
    assert tool_context.startswith("\n<|im_start|>user\n<tool_response>\n")
    assert tool_context.endswith("\n</tool_response><|im_end|>\n<|im_start|>assistant\n")
    assert '"reward": 0.0' in tokenizer.decode(second["token_ids"])
    assert rollout.reward == 1 and rollout.stop_condition == "alfworld_done"
    assert rollout.num_turns == 2
    assert rollout.num_output_tokens == sum(map(len, sampled))
    assert rollout.num_input_tokens == len(expected_ids) - rollout.num_output_tokens
    assert rollout.completion == replies[-1].removesuffix("<|im_end|>")
    assert "Task complete" in rollout.interaction
    assert first["sampling_params"] == {
        "max_tokens": 128,
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": -1,
        "logprobs": 0,
        "stop_token_ids": [tokenizer.eos_token_id],
        "skip_special_tokens": False,
    }
    environment.__aexit__.assert_awaited_once()


@pytest.mark.parametrize(
    "boundary", ["max_turns", "max_tokens", "context_generation", "context_observation", "no_room"]
)
def test_rollout_limits(boundary: str, tokenizer: PreTrainedTokenizerFast, environment: AsyncMock) -> None:
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=response_for(
                tokenizer,
                "x"
                if boundary in {"max_tokens", "context_generation"}
                else '<tool_call>{"name":"look","arguments":{}}</tool_call><|im_end|>',
                "length" if boundary in {"max_tokens", "context_generation"} else "stop",
            ),
        )

    async def run() -> loop.Rollout:
        async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
            agent = make_agent(tokenizer, client)
            initial = environment.request.side_effect
            # Reset the iterator after reading the initial observation for the boundary calculation.
            results = list(initial)
            environment.request.side_effect = results
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": loop.SYSTEM_PROMPT},
                    {"role": "user", "content": loop.TASK_PROMPT},
                ],
                tools=tool_schemas(),
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
            )
            if boundary == "max_turns":
                agent = replace(agent, max_turns=1)
            elif boundary == "max_tokens":
                agent = replace(agent, max_tokens=1)
            else:
                room = {"context_generation": 1, "context_observation": 100, "no_room": 0}[boundary]
                agent = replace(agent, context_length=len(prompt) + room)
            return await agent.run(ALFWorldTask(id="a", config_file="config.yaml"))

    if boundary == "no_room":
        with pytest.raises(ValueError, match="no trainable tokens"):
            asyncio.run(run())
        assert not requests
    else:
        rollout = asyncio.run(run())
        assert rollout.stop_condition == ("context_length" if boundary.startswith("context") else boundary)
        assert len(requests) == 1
        assert environment.request.await_count == (1 if boundary in {"max_tokens", "context_generation"} else 2)
        assert rollout.num_turns == 1
        if boundary == "context_generation":
            assert requests[0]["sampling_params"]["max_tokens"] == 1
        assert len(rollout.sequence.token_ids) <= 8192
    environment.__aexit__.assert_awaited_once()


@pytest.mark.parametrize(
    "failure", ["http", "empty", "logprobs", "alignment", "missing_probability", "abort", "timeout", "cancel"]
)
def test_failures_cleanup(failure: str, tokenizer: PreTrainedTokenizerFast, environment: AsyncMock) -> None:
    entered = asyncio.Event()

    async def respond(request: httpx.Request) -> httpx.Response:
        entered.set()
        if failure in {"timeout", "cancel"}:
            await asyncio.Future()
        if failure == "http":
            return httpx.Response(500)
        data = response_for(tokenizer, "look<|im_end|>")
        choice = data["choices"][0]
        if failure == "empty":
            choice["token_ids"] = []
            choice["logprobs"]["content"] = []
        elif failure == "logprobs":
            choice["logprobs"]["content"].pop()
        elif failure == "alignment":
            choice["logprobs"]["content"][0]["token"] = "token_id:999"
        elif failure == "missing_probability":
            choice["logprobs"]["content"][0]["logprob"] = -9999.0
        elif failure == "abort":
            choice["finish_reason"] = "abort"
        return httpx.Response(200, json=data)

    async def run() -> None:
        async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
            agent = make_agent(tokenizer, client)
            if failure == "timeout":
                agent = replace(agent, rollout_timeout=0.05)
            task = asyncio.create_task(agent.run(ALFWorldTask(id="a", config_file="config.yaml")))
            if failure == "cancel":
                await entered.wait()
                task.cancel()
            await task

    error = {
        "http": httpx.HTTPStatusError,
        "abort": RuntimeError,
        "timeout": TimeoutError,
        "cancel": asyncio.CancelledError,
    }.get(failure, ValueError)
    with pytest.raises(error):
        asyncio.run(run())
    environment.__aexit__.assert_awaited_once()
    assert environment.request.await_count == 1


@pytest.mark.parametrize(
    "text",
    [
        '<tool_call>{"name":"unknown","arguments":{}}</tool_call>',
        '<tool_call>{"name":"take","arguments":{"object":"mug 1"}}</tool_call>',
        '<tool_call>{"name":"look","arguments":{"extra":"x"}}</tool_call>',
        '<tool_call>{"name":"use","arguments":{"object":1}}</tool_call>',
        '<tool_call>{"name":"look","arguments":"{}"}</tool_call>',
        "<tool_call>not json</tool_call>",
        '<tool_call>{"name":"look","arguments":{}}',
    ],
    ids=["unknown", "missing-argument", "extra-argument", "argument-type", "arguments-type", "json", "unclosed"],
)
def test_invalid_tool_calls(text: str) -> None:
    with pytest.raises(ValueError):
        loop.parse_tool_calls(text)


def test_tool_calls_ignore_thinking() -> None:
    call = '<tool_call>{"name":"look","arguments":{}}</tool_call>'
    assert loop.parse_tool_calls(f"<think>{call}</think>Finished") == []
    assert loop.parse_tool_calls(f"<think>{call}") == []
    assert loop.parse_tool_calls(f"<think>{call}</think>{call}") == [ToolCall(name="look", arguments={})]


@pytest.mark.parametrize("mode", ["text", "invalid", "multiple", "terminal"])
def test_tool_dispatch(mode: str, tokenizer: PreTrainedTokenizerFast, environment: AsyncMock) -> None:
    look = '<tool_call>{"name":"look","arguments":{}}</tool_call>'
    inventory = '<tool_call>{"name":"inventory","arguments":{}}</tool_call>'
    replies = {
        "text": ["look"],
        "invalid": ['<tool_call>{"name":"unknown","arguments":{}}</tool_call>', "Finished"],
        "multiple": [look + inventory],
        "terminal": [look + inventory],
    }[mode]
    if mode == "terminal":
        environment.request.side_effect = [
            EpisodeResult(observation="Goal", reward=0.0, done=False),
            EpisodeResult(observation="Won", reward=1.0, done=True),
        ]
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response_for(tokenizer, replies[len(requests) - 1] + "<|im_end|>"))

    async def run() -> loop.Rollout:
        async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
            return await replace(make_agent(tokenizer, client), max_tokens=256).run(
                ALFWorldTask(id="a", config_file="config.yaml")
            )

    rollout = asyncio.run(run())
    assert rollout.stop_condition == ("alfworld_done" if mode in {"multiple", "terminal"} else "agent_completed")
    assert environment.request.await_count == {"text": 1, "invalid": 1, "multiple": 3, "terminal": 2}[mode]
    if mode == "invalid":
        assert "<tool_response>" in tokenizer.decode(requests[1]["token_ids"])
        assert "Unknown ALFWorld tool" in rollout.interaction
    if mode == "multiple":
        assert [call.args[0].name for call in environment.request.await_args_list[1:]] == ["look", "inventory"]
        assert '"observation": "At shelf.' in rollout.interaction
        assert '"observation": "Task complete"' in rollout.interaction
