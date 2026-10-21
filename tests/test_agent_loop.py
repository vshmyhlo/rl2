import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any, ClassVar
from unittest.mock import AsyncMock, Mock

import httpx
import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from rl2 import agent_loop as loop
from rl2 import alfworld_env
from rl2.agent_task import Environment, Task, ToolSchema
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
    results = iter(
        [
            EpisodeResult(observation="At shelf. Admissible commands: put mug 1 in/on desk 1", reward=0.0, done=False),
            EpisodeResult(observation="Task complete", reward=1.0, done=True),
        ]
    )

    async def step(action: ToolCall) -> EpisodeResult:
        alfworld_env.validate_tool_call(action)
        return next(results)

    session.reset.return_value = EpisodeResult(
        observation="Goal: put mug on desk. Admissible commands: look", reward=0.0, done=False
    )
    session.step.side_effect = step
    monkeypatch.setattr(alfworld_env, "ALFWorldSession", Mock(return_value=session))
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
    environment.reset.assert_awaited_once_with()
    assert [call.args for call in environment.step.await_args_list] == [
        (ToolCall(name="look", arguments={}),),
        (ToolCall(name="move", arguments={"object": "mug 1", "location": "desk 1"}),),
    ]
    initial_text = tokenizer.decode(first["token_ids"])
    assert ALFWorldTask.prompt in initial_text
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
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": ALFWorldTask.system_prompt},
                    {"role": "user", "content": ALFWorldTask.prompt},
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
        assert environment.step.await_count == (0 if boundary in {"max_tokens", "context_generation"} else 1)
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
    environment.reset.assert_awaited_once_with()
    environment.step.assert_not_awaited()


@pytest.mark.parametrize(
    "text",
    [
        '<tool_call>{"name":"look","arguments":"{}"}</tool_call>',
        "<tool_call>not json</tool_call>",
        '<tool_call>{"name":"look","arguments":{}}',
        "</tool_call>",
        '<tool_call>{"name":"look","arguments":{}} {"name":"look","arguments":{}}</tool_call>',
        '<tool_call><tool_call>{"name":"look","arguments":{}}</tool_call></tool_call>',
    ],
    ids=["arguments-type", "json", "unclosed", "unopened", "multiple-objects", "nested-blocks"],
)
def test_invalid_tool_calls(text: str) -> None:
    with pytest.raises(ValueError):
        loop.parse_tool_calls(text)


def test_tool_calls_ignore_thinking() -> None:
    call = '<tool_call>{"name":"look","arguments":{}}</tool_call>'
    assert loop.parse_tool_calls(f"<think>{call}</think>Finished") == []
    assert loop.parse_tool_calls(f"<think>{call}") == []
    assert loop.parse_tool_calls(f"<think>{call}</think>{call}") == [ToolCall(name="look", arguments={})]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>unfinished", ""),
        ("<think>\n\n</think>\n\nFinished<|im_end|>", "Finished"),
        ("Reasoning without opening tag</think>Finished<|im_end|>", "Finished"),
        ("Finished<|endoftext|>", "Finished"),
    ],
    ids=["unfinished-thinking", "disabled-thinking", "implicit-thinking", "endoftext"],
)
def test_completion_text(text: str, expected: str) -> None:
    assert loop.completion_text(text) == expected


def test_tool_calls_preserve_json_strings() -> None:
    calls = [
        ToolCall(name="write", arguments={"text": '<tool_call></tool_call><think></think> "雪"', "nested": [1, None]}),
        ToolCall(name="look", arguments={}),
    ]
    text = "\n".join(
        f"<tool_call>\n{json.dumps(call.model_dump(), ensure_ascii=False, indent=2)}\n</tool_call>" for call in calls
    )
    assert loop.parse_tool_calls(text + "<|im_end|>") == calls


def test_completion_tokens_preserve_qwen_tokens() -> None:
    ids = [151667, 198, 151668, 151657, 151658, 151645]
    probabilities = [-0.1, -0.2, -0.3, -0.4, -0.5, 0.0]
    choice = {
        "token_ids": ids,
        "logprobs": {
            "content": [
                {"token": f"token_id:{token}", "logprob": probability, "top_logprobs": []}
                for token, probability in zip(ids, probabilities, strict=True)
            ]
        },
    }
    assert loop.completion_tokens(choice) == (ids, probabilities)


@pytest.mark.parametrize(
    "logprob",
    [-9999.0, 0.1, float("nan"), None, "-0.25"],
    ids=["clipped", "positive", "nan", "missing", "string"],
)
def test_completion_tokens_invalid_probability(logprob: Any) -> None:
    with pytest.raises(ValueError):
        loop.completion_tokens(
            {"token_ids": [151645], "logprobs": {"content": [{"token": "token_id:151645", "logprob": logprob}]}}
        )


@pytest.mark.parametrize("logprobs", [None, {}, {"content": None}, {"content": [{}]}])
def test_completion_tokens_missing_evidence(logprobs: Any) -> None:
    with pytest.raises(ValueError):
        loop.completion_tokens({"token_ids": [151645], "logprobs": logprobs})


def test_completion_preserves_literal_thinking_tags(tokenizer: PreTrainedTokenizerFast, environment: AsyncMock) -> None:
    answer = '<tool_call>{"name":"use","arguments":{"object":"<think></think>"}}</tool_call>'

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_for(tokenizer, "<think>Act.</think>" + answer + "<|im_end|>"))

    async def run() -> loop.Rollout:
        async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
            return await replace(make_agent(tokenizer, client), max_turns=1).run(
                ALFWorldTask(id="a", config_file="config.yaml")
            )

    rollout = asyncio.run(run())
    assert rollout.completion == answer
    environment.step.assert_awaited_with(ToolCall(name="use", arguments={"object": "<think></think>"}))
    assert '"error"' not in rollout.interaction


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
        environment.step.side_effect = [EpisodeResult(observation="Won", reward=1.0, done=True)]
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
    environment.reset.assert_awaited_once_with()
    assert environment.step.await_count == {"text": 0, "invalid": 1, "multiple": 2, "terminal": 1}[mode]
    if mode == "invalid":
        assert "<tool_response>" in tokenizer.decode(requests[1]["token_ids"])
        assert "Unknown ALFWorld tool" in rollout.interaction
    if mode == "multiple":
        assert [call.args[0].name for call in environment.step.await_args_list] == ["look", "inventory"]
        assert '"observation": "At shelf.' in rollout.interaction
        assert '"observation": "Task complete"' in rollout.interaction


@pytest.mark.parametrize("mode", ["success", "setup_error", "step_error", "cancel"])
def test_generic_task_environment_lifetime(mode: str, tokenizer: PreTrainedTokenizerFast) -> None:
    events: list[str] = []
    stepping = asyncio.Event()

    class ExampleEnvironment(Environment):
        async def reset(self) -> EpisodeResult:
            return EpisodeResult(observation="Ready", reward=0.0, done=False)

        async def step(self, action: ToolCall) -> EpisodeResult:
            assert action.name == "count" and action.arguments == {"amount": 2}
            stepping.set()
            if mode == "step_error":
                raise RuntimeError("Environment failed")
            if mode == "cancel":
                await asyncio.Future()
            return EpisodeResult(observation="Counted two", reward=0.5, done=True)

    class ExampleTask(Task):
        id = "one"
        name = "example"
        system_prompt = "Count using tools."
        prompt = "Count two."
        tools: ClassVar[list[ToolSchema]] = [
            {
                "type": "function",
                "function": {
                    "name": "count",
                    "parameters": {
                        "type": "object",
                        "properties": {"amount": {"type": "integer"}},
                        "required": ["amount"],
                    },
                },
            }
        ]

        @asynccontextmanager
        async def init_env(self, *, max_steps: int, timeout: float) -> AsyncGenerator[Environment]:
            assert max_steps == 3 and timeout == 2
            events.append("enter")
            try:
                if mode == "setup_error":
                    raise RuntimeError("Setup failed")
                yield ExampleEnvironment()
            finally:
                events.append("exit")

    def respond(request: httpx.Request) -> httpx.Response:
        prompt = tokenizer.decode(json.loads(request.content)["token_ids"])
        assert "Count using tools." in prompt and "Count two." in prompt
        assert '"name": "count"' in prompt and "alfworld" not in prompt
        return httpx.Response(
            200,
            json=response_for(
                tokenizer,
                '<tool_call>{"name":"count","arguments":{"amount":2}}</tool_call><|im_end|>',
            ),
        )

    async def run() -> loop.Rollout:
        async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
            task: Task = ExampleTask()
            running = asyncio.create_task(make_agent(tokenizer, client).run(task))
            if mode == "cancel":
                await stepping.wait()
                running.cancel()
            return await running

    if mode == "success":
        result = asyncio.run(run())
        assert result.reward == 0.5 and result.stop_condition == "example_done"
        assert "Counted two" in result.interaction
    else:
        with pytest.raises(asyncio.CancelledError if mode == "cancel" else RuntimeError):
            asyncio.run(run())
    assert events == ["enter", "exit"]
