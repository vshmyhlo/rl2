import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import numpy as np
import pytest
from openai import APIStatusError, AsyncOpenAI
from openai.types import CompletionChoice
from openai.types.completion_choice import Logprobs
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from rl2 import agent_loop as loop
from rl2 import alfworld_env
from rl2.agent_task import Environment, ResetResult, Task, ToolSchema
from rl2.alfworld_env import StepResult, ToolCall, tool_schemas
from rl2.grpo_v2 import reasoning_gym
from rl2.sandbox.gvisor import PythonResult


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
        # Qwen/Qwen3-0.6B tokenizer_config.json, revision c1899de289a04d12100db370d81485cdf75e47ca.
        chat_template=(Path(__file__).parent / "data/qwen3_chat_template.jinja").read_text(),
    )


@pytest.fixture
def environment() -> AsyncMock:
    session = AsyncMock()
    session.__aenter__.return_value = session
    results = iter(
        [
            StepResult(observation="At shelf. Admissible commands: put mug 1 in/on desk 1", reward=0.0, done=False),
            StepResult(observation="Task complete", reward=1.0, done=True),
        ]
    )

    async def step(action: ToolCall, *, timeout: float | None = None) -> StepResult:
        alfworld_env.validate_tool_call(action)
        return next(results)

    session.reset.return_value = ResetResult(
        tools=tool_schemas(),
        system_prompt="Use one action tool per turn.",
        prompt="Goal: put mug on desk. Admissible commands: go to shelf 1",
    )
    session.step.side_effect = step
    return session


@pytest.fixture
def task(environment: AsyncMock) -> Task:
    return Mock(
        spec=Task,
        init_env=Mock(return_value=environment),
    )


def response_for(tokenizer: PreTrainedTokenizerFast, text: str, finish_reason: str = "stop") -> dict[str, Any]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return {
        "id": "completion-test",
        "object": "text_completion",
        "created": 0,
        "model": "Qwen/test",
        "choices": [
            {
                "index": 0,
                "text": text,
                "token_ids": ids,
                "logprobs": {
                    "tokens": [f"token_id:{token}" for token in ids],
                    "token_logprobs": [-0.25] * len(ids),
                },
                "finish_reason": finish_reason,
            }
        ],
    }


def make_client(respond: Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]]) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url="http://test/v1",
        api_key="EMPTY",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )


def make_agent(tokenizer: PreTrainedTokenizerFast, client: AsyncOpenAI) -> loop.Agent:
    return loop.Agent(
        client=client,
        tokenizer=tokenizer,
        model="Qwen/test",
        context_length=8192,
        max_tokens=8192,
        max_turns=3,
        temperature=0.7,
        rollout_timeout=10,
    )


@pytest.mark.parametrize("mode", ["submitted", "calculator", "python", "plain_text", "truncated"])
def test_reasoning_gym_rollout(mode: str, tokenizer: PreTrainedTokenizerFast, monkeypatch: pytest.MonkeyPatch) -> None:
    task = reasoning_gym.ReasoningGymTask(config=reasoning_gym.ReasoningGymConfig(), seed=1)
    worker = reasoning_gym.ReasoningGymWorker.__ray_metadata__.modified_class(task=task)
    answer = reasoning_gym.create_dataset(task.config, task.seed)[0]["answer"]
    actor = Mock()
    actor.reset.remote = AsyncMock(side_effect=worker.reset)
    actor.step.remote = AsyncMock(side_effect=worker.step)
    monkeypatch.setattr(reasoning_gym.ReasoningGymWorker, "remote", Mock(return_value=actor))
    monkeypatch.setattr(reasoning_gym.ray, "kill", Mock())
    sandbox = AsyncMock(spec=reasoning_gym.PythonSession)
    sandbox.__aenter__.return_value = sandbox
    sandbox.execute.return_value = PythonResult(output=f"{answer}\n", exit_code=0, output_truncated=False)
    python_constructor = Mock(return_value=sandbox)
    monkeypatch.setattr(reasoning_gym, "PythonSession", python_constructor)
    submission = ToolCall(name="submit_answer", arguments={"answer": answer})
    call = submission.model_dump_json()
    text = answer if mode == "plain_text" else f"<think>solve</think><tool_call>{call}</tool_call><|im_end|>"
    calculation = ToolCall(name="calculator", arguments={"expression": f"{answer} + 0"})
    replies = [text]
    if mode == "calculator":
        replies.insert(0, f"<think>calculate</think><tool_call>{calculation.model_dump_json()}</tool_call><|im_end|>")
    elif mode == "python":
        code = ToolCall(name="python", arguments={"code": f"print({answer})"})
        replies.insert(0, f"<think>compute</think><tool_call>{code.model_dump_json()}</tool_call><|im_end|>")
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=response_for(
                tokenizer, replies[len(requests) - 1], finish_reason="length" if mode == "truncated" else "stop"
            ),
        )

    async def run() -> loop.Rollout:
        async with make_client(respond) as client, task.init_env() as session:
            return await replace(make_agent(tokenizer, client), max_turns=len(replies)).run(env=session)

    rollout = asyncio.run(run())
    assert rollout.num_turns == len(replies)
    assert rollout.reward == (1.0 if mode in {"submitted", "calculator", "python"} else 0.0)
    assert (
        rollout.stop_condition
        == {
            "submitted": "environment_done",
            "calculator": "environment_done",
            "python": "environment_done",
            "plain_text": "agent_completed",
            "truncated": "context_length",
        }[mode]
    )
    if mode == "submitted":
        actor.step.remote.assert_awaited_once_with(submission)
    elif mode == "calculator":
        assert [call.args[0] for call in actor.step.remote.await_args_list] == [calculation, submission]
        assert f"<tool_response>\n{answer}\n</tool_response>" in tokenizer.decode(requests[1]["prompt"])
    elif mode == "python":
        actor.step.remote.assert_awaited_once_with(submission)
        sandbox.execute.assert_awaited_once_with(f"print({answer})")
        sandbox.__aexit__.assert_awaited_once()
        output = json.dumps({"output": f"{answer}\n", "exit_code": 0, "output_truncated": False})
        assert f"<tool_response>\n{output}\n</tool_response>" in tokenizer.decode(requests[1]["prompt"])
    else:
        actor.step.remote.assert_not_called()
    if mode != "python":
        python_constructor.assert_not_called()
    sequence = rollout.sequence
    assert sequence.token_ids.shape == sequence.loss_mask.shape == sequence.old_log_probs.shape
    assert sequence.loss_mask.sum() == sum(len(tokenizer.encode(reply, add_special_tokens=False)) for reply in replies)


def test_multiturn_exact_training_sequence(
    tokenizer: PreTrainedTokenizerFast, environment: AsyncMock, task: Task
) -> None:
    environment.step.side_effect = [
        StepResult(observation="At shelf", reward=-0.25, done=False),
        StepResult(observation="Task complete", reward=1.0, done=True),
    ]
    replies = [
        '<think>inspect</think>\n<tool_call>{"name":"go_to","arguments":{"location":"shelf 1"}}</tool_call><|im_end|>',
        '<tool_call>{"name":"move","arguments":{"object":"mug 1","location":"desk 1"}}</tool_call><|im_end|>',
    ]
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/completions"
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json=response_for(tokenizer, replies[len(requests) - 1]))

    async def run() -> loop.Rollout:
        async with make_client(respond) as client, task.init_env() as env:
            return await replace(make_agent(tokenizer, client), environment_timeout=2).run(env=env)

    rollout = asyncio.run(run())
    first, second = requests
    sampled = [tokenizer.encode(reply, add_special_tokens=False) for reply in replies]
    prefix = first["prompt"] + sampled[0]
    assert second["max_tokens"] == 8192 - len(second["prompt"])
    assert second["max_tokens"] < first["max_tokens"]
    assert second["prompt"][: len(prefix)] == prefix
    assert "<think>inspect</think>" in tokenizer.decode(second["prompt"])
    expected_ids = second["prompt"] + sampled[1]
    mask = np.zeros(len(expected_ids), dtype=np.bool_)  # (T,)
    mask[len(first["prompt"]) : len(prefix)] = True
    mask[len(second["prompt"]) :] = True
    sequence = rollout.sequence
    assert sequence.token_ids.shape == sequence.loss_mask.shape == sequence.old_log_probs.shape == (len(expected_ids),)
    assert sequence.token_ids.dtype == np.int32
    assert sequence.loss_mask.dtype == np.bool_
    assert sequence.old_log_probs.dtype == np.float32
    np.testing.assert_array_equal(sequence.token_ids, expected_ids)
    np.testing.assert_array_equal(sequence.loss_mask, mask)
    np.testing.assert_array_equal(sequence.old_log_probs, np.where(mask, -0.25, 0.0))
    environment.reset.assert_awaited_once_with(timeout=2)
    assert [call.kwargs for call in environment.step.await_args_list] == [{"timeout": 2}, {"timeout": 2}]
    assert [call.args for call in environment.step.await_args_list] == [
        (ToolCall(name="go_to", arguments={"location": "shelf 1"}),),
        (ToolCall(name="move", arguments={"object": "mug 1", "location": "desk 1"}),),
    ]
    initial_text = tokenizer.decode(first["prompt"])
    assert environment.reset.return_value.prompt in initial_text
    assert environment.reset.return_value.system_prompt in initial_text
    assert rollout.prompt == environment.reset.return_value.prompt
    assert '"name": "move"' in initial_text and '"name": "inventory"' in initial_text
    tool_context = tokenizer.decode(second["prompt"][len(prefix) :])
    assert tool_context.startswith("\n<|im_start|>user\n<tool_response>\n")
    assert tool_context.endswith("\n</tool_response><|im_end|>\n<|im_start|>assistant\n")
    assert tool_context == (
        "\n<|im_start|>user\n<tool_response>\nAt shelf\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
    )
    assert rollout.reward == 0.75 and rollout.stop_condition == "environment_done"
    assert rollout.num_turns == 2
    assert rollout.num_output_tokens == sum(map(len, sampled))
    assert rollout.num_input_tokens == len(expected_ids) - rollout.num_output_tokens
    assert rollout.completion == replies[-1].removesuffix("<|im_end|>")
    assert rollout.interaction == tokenizer.decode(
        expected_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    assert rollout.interaction.startswith("<|im_start|>system\n" + environment.reset.return_value.system_prompt)
    assert "Task complete" not in rollout.interaction
    assert {key: value for key, value in first.items() if key not in {"model", "prompt"}} == {
        "max_tokens": 8192 - len(first["prompt"]),
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": -1,
        "logprobs": 0,
        "stop_token_ids": tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]),
        "skip_special_tokens": False,
        "add_special_tokens": False,
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
    }
    environment.__aenter__.assert_awaited_once()
    environment.__aexit__.assert_awaited_once()


@pytest.mark.parametrize("terminal", [True, False], ids=["terminal", "turn-limit"])
def test_return_sums_steps_within_turn(
    terminal: bool, tokenizer: PreTrainedTokenizerFast, environment: AsyncMock
) -> None:
    environment.step.side_effect = [
        StepResult(observation="Progress", reward=0.5, done=False),
        alfworld_env.ToolCallError("Invalid action"),
        StepResult(observation="Penalty", reward=-0.25, done=terminal),
    ]
    reply = (
        "".join(
            f'<tool_call>{{"name":"{name}","arguments":{{}}}}</tool_call>' for name in ("look", "unknown", "inventory")
        )
        + "<|im_end|>"
    )

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_for(tokenizer, reply))

    async def run() -> loop.Rollout:
        async with make_client(respond) as client:
            return await replace(make_agent(tokenizer, client), max_turns=1).run(env=environment)

    rollout = asyncio.run(run())
    assert rollout.reward == 0.25
    assert rollout.stop_condition == ("environment_done" if terminal else "max_turns")
    assert environment.step.await_count == 3
    assert rollout.interaction.endswith(reply)
    assert "Invalid action" not in rollout.interaction
    assert '"reward": 0.5' not in rollout.interaction
    assert '"reward": -0.25' not in rollout.interaction


@pytest.mark.parametrize("extra_tokens", [-1, 0, 1], ids=["below-limit", "at-limit", "over-limit"])
def test_total_token_budget_across_turns(
    extra_tokens: int, tokenizer: PreTrainedTokenizerFast, environment: AsyncMock, task: Task
) -> None:
    requests: list[dict[str, Any]] = []
    token_limit = 8191  # Below the model context length.

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        assert body["max_tokens"] == token_limit - len(body["prompt"])
        text = (
            '<tool_call>{"name":"look","arguments":{}}</tool_call><|im_end|>'
            if len(requests) == 1
            else "x" * (body["max_tokens"] + extra_tokens)
        )
        return httpx.Response(200, json=response_for(tokenizer, text, "stop" if len(requests) == 1 else "length"))

    async def run() -> loop.Rollout:
        async with make_client(respond) as client, task.init_env() as env:
            return await replace(make_agent(tokenizer, client), max_tokens=token_limit).run(env=env)

    if extra_tokens > 0:
        with pytest.raises(ValueError, match="Completion exceeds the requested token budget"):
            asyncio.run(run())
        return

    rollout = asyncio.run(run())
    sequence = rollout.sequence
    expected_length = token_limit + extra_tokens
    assert sequence.token_ids.shape == sequence.loss_mask.shape == sequence.old_log_probs.shape == (expected_length,)
    assert rollout.num_input_tokens + rollout.num_output_tokens == expected_length
    assert rollout.stop_condition == "max_tokens"
    assert len(requests) == rollout.num_turns == 2
    assert requests[1]["max_tokens"] < requests[0]["max_tokens"]
    environment.step.assert_awaited_once()


@pytest.mark.parametrize(
    "boundary",
    [
        "max_turns",
        "max_tokens",
        "max_observation",
        "context_generation",
        "context_observation",
        "no_room",
        "prompt_full",
        "prompt_overlong",
    ],
)
def test_rollout_limits(boundary: str, tokenizer: PreTrainedTokenizerFast, environment: AsyncMock, task: Task) -> None:
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
        async with make_client(respond) as client, task.init_env() as env:
            agent = make_agent(tokenizer, client)
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": environment.reset.return_value.system_prompt},
                    {"role": "user", "content": environment.reset.return_value.prompt},
                ],
                tools=tool_schemas(),
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
            )
            if boundary == "max_turns":
                agent = replace(agent, max_turns=1)
            elif boundary == "max_tokens":
                agent = replace(agent, max_tokens=len(prompt) + 1)
            elif boundary in {"max_observation", "prompt_full", "prompt_overlong"}:
                room = {"max_observation": 100, "prompt_full": 0, "prompt_overlong": -1}[boundary]
                agent = replace(agent, max_tokens=len(prompt) + room)
            else:
                room = {"context_generation": 1, "context_observation": 100, "no_room": 0}[boundary]
                agent = replace(agent, context_length=len(prompt) + room)
            return await agent.run(env=env)

    if boundary in {"no_room", "prompt_full", "prompt_overlong"}:
        with pytest.raises(ValueError, match="Initial prompt.*must leave room"):
            asyncio.run(run())
        assert not requests
    else:
        rollout = asyncio.run(run())
        assert rollout.stop_condition == (
            "context_length"
            if boundary.startswith("context")
            else "max_tokens"
            if boundary == "max_observation"
            else boundary
        )
        assert len(requests) == 1
        assert environment.step.await_count == (0 if boundary in {"max_tokens", "context_generation"} else 1)
        assert rollout.num_turns == 1
        if boundary in {"max_tokens", "context_generation"}:
            assert requests[0]["max_tokens"] == 1
        assert len(rollout.sequence.token_ids) <= 8192
        assert rollout.interaction == tokenizer.decode(
            rollout.sequence.token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
    environment.__aenter__.assert_awaited_once()
    environment.__aexit__.assert_awaited_once()


@pytest.mark.parametrize(
    "failure",
    ["http", "empty", "logprobs", "alignment", "missing_probability", "abort", "abort_empty", "timeout", "cancel"],
)
def test_failures_close_environment(
    failure: str, tokenizer: PreTrainedTokenizerFast, environment: AsyncMock, task: Task
) -> None:
    entered = asyncio.Event()
    requests = 0

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        entered.set()
        if failure in {"timeout", "cancel"}:
            await asyncio.Future()
        if failure == "http":
            return httpx.Response(500)
        data = response_for(tokenizer, "look<|im_end|>")
        choice = data["choices"][0]
        if failure in {"empty", "abort_empty"}:
            choice["token_ids"] = []
            choice["logprobs"]["tokens"] = []
            choice["logprobs"]["token_logprobs"] = []
        elif failure == "logprobs":
            choice["logprobs"]["token_logprobs"].pop()
        elif failure == "alignment":
            choice["logprobs"]["tokens"][0] = "token_id:999"
        elif failure == "missing_probability":
            choice["logprobs"]["token_logprobs"][0] = -9999.0
        if failure in {"abort", "abort_empty"}:
            choice["finish_reason"] = "abort"
        return httpx.Response(200, json=data)

    async def run() -> None:
        async with make_client(respond) as client, task.init_env() as env:
            agent = make_agent(tokenizer, client)
            if failure == "timeout":
                agent = replace(agent, rollout_timeout=0.05)
            running = asyncio.create_task(agent.run(env=env))
            if failure == "cancel":
                await entered.wait()
                running.cancel()
            await running

    error = {
        "http": APIStatusError,
        "abort": RuntimeError,
        "abort_empty": RuntimeError,
        "timeout": TimeoutError,
        "cancel": asyncio.CancelledError,
    }.get(failure, ValueError)
    with pytest.raises(error):
        asyncio.run(run())
    # The rollout timeout also covers prompt tokenization before the first request.
    if failure != "timeout":
        assert requests == 1
    environment.__aenter__.assert_awaited_once()
    environment.__aexit__.assert_awaited_once()
    assert issubclass(environment.__aexit__.await_args.args[0], error)
    environment.reset.assert_awaited_once_with(timeout=None)
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


def test_tool_calls_reject_non_json_constants() -> None:
    with pytest.raises(ValueError, match="Invalid JSON constant"):
        loop.parse_tool_calls('<tool_call>{"name":"count","arguments":{"amount":NaN}}</tool_call>')


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>unfinished", ""),
        ("<think>\n\n</think>\n\nFinished<|im_end|>", "Finished"),
        ("Reasoning without opening tag</think>Finished<|im_end|>", "Finished"),
        ("Finished<|endoftext|>", "Finished"),
        ("Finished<|im_end|>\n", "Finished"),
        ("Use <think> and </think> tags.", "Use <think> and </think> tags."),
    ],
    ids=[
        "unfinished-thinking",
        "disabled-thinking",
        "implicit-thinking",
        "endoftext",
        "trailing-space",
        "literal-tags",
    ],
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


@pytest.mark.parametrize(
    ("ending", "eos_token"),
    [
        ("<|im_end|>", "<|im_end|>"),
        ("<|endoftext|>", "<|im_end|>"),
        ("", "<|im_end|>"),
        ("<|im_end|>", "<|endoftext|>"),
        ("<|endoftext|>", "<|endoftext|>"),
    ],
    ids=["im-end", "endoftext", "omitted-stop", "im-end-with-alternate-eos", "endoftext-with-alternate-eos"],
)
def test_tool_response_renderer(
    ending: str, eos_token: str, tokenizer: PreTrainedTokenizerFast, environment: AsyncMock, task: Task
) -> None:
    tokenizer.eos_token = eos_token
    calls = [ToolCall(name="look", arguments={}), ToolCall(name="inventory", arguments={})]
    results = [
        StepResult(observation='A shelf\nWith a "mug" and a café sign.', reward=0.0, done=False),
        StepResult(observation="A mug", reward=0.0, done=False),
    ]
    environment.step.side_effect = results
    assistant = {"role": "assistant", "content": "", "reasoning_content": "Inspect.", "tool_calls": calls}
    messages = [
        {"role": "system", "content": environment.reset.return_value.system_prompt},
        {"role": "user", "content": environment.reset.return_value.prompt},
    ]
    initial = tokenizer.apply_chat_template(
        messages, tools=tool_schemas(), tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    rendered = tokenizer.apply_chat_template(messages + [assistant], tools=tool_schemas(), tokenize=False)
    reply = rendered[len(initial) :].removesuffix("<|im_end|>\n") + ending
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response_for(tokenizer, reply if len(requests) == 1 else "Finished<|im_end|>"))

    async def run() -> loop.Rollout:
        async with make_client(respond) as client, task.init_env() as env:
            return await make_agent(tokenizer, client).run(env=env)

    rollout = asyncio.run(run())
    tool_messages = [{"role": "tool", "content": result.observation} for result in results]
    expected = tokenizer.apply_chat_template(
        messages + [assistant] + tool_messages,
        tools=tool_schemas(),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if ending == "<|endoftext|>":
        # Preserve the sampled alternative EOS before inserting the chat boundary.
        expected = expected.replace(rendered, rendered.removesuffix("<|im_end|>\n") + ending + "<|im_end|>\n", 1)
    assert tokenizer.decode(requests[1]["prompt"]) == expected
    assert loop.parse_tool_calls(reply) == calls
    assert rollout.stop_condition == "agent_completed"
    sampled = tokenizer.encode(reply, add_special_tokens=False)
    start = len(requests[0]["prompt"])
    np.testing.assert_array_equal(rollout.sequence.token_ids[start : start + len(sampled)], sampled)
    assert rollout.sequence.loss_mask[start : start + len(sampled)].all()
    assert not rollout.sequence.loss_mask[start + len(sampled) : len(requests[1]["prompt"])].any()


def test_completion_tokens_preserve_qwen_tokens() -> None:
    ids = [151667, 198, 151668, 151657, 151658, 151645]
    probabilities = [-0.1, -0.2, -0.3, -0.4, -0.5, 0.0]
    choice = CompletionChoice.model_construct(
        token_ids=ids,
        logprobs=Logprobs(tokens=[f"token_id:{token}" for token in ids], token_logprobs=probabilities),
    )
    assert loop.completion_tokens(choice) == (ids, probabilities)


@pytest.mark.parametrize(
    "logprob",
    [-9999.0, 0.1, float("nan"), None, "invalid"],
    ids=["clipped", "positive", "nan", "missing", "nonnumeric"],
)
def test_completion_tokens_invalid_probability(logprob: Any) -> None:
    with pytest.raises(ValueError):
        loop.completion_tokens(
            CompletionChoice.model_construct(
                token_ids=[151645],
                logprobs=Logprobs.model_construct(tokens=["token_id:151645"], token_logprobs=[logprob]),
            )
        )


@pytest.mark.parametrize("logprobs", [None, {}, {"tokens": ["token_id:151645"]}, {"token_logprobs": [-0.25]}])
def test_completion_tokens_missing_evidence(logprobs: Any) -> None:
    with pytest.raises(ValueError):
        loop.completion_tokens(
            CompletionChoice.model_construct(
                token_ids=[151645], logprobs=None if logprobs is None else Logprobs.model_construct(**logprobs)
            )
        )


@pytest.mark.parametrize("ids", [None, [-1], ["151645"]], ids=["missing", "negative", "string"])
def test_completion_tokens_invalid_ids(ids: Any) -> None:
    with pytest.raises(ValueError):
        loop.completion_tokens(
            CompletionChoice.model_construct(
                token_ids=ids, logprobs=Logprobs(tokens=["token_id:151645"], token_logprobs=[-0.25])
            )
        )


def test_completion_preserves_literal_thinking_tags(
    tokenizer: PreTrainedTokenizerFast, environment: AsyncMock, task: Task
) -> None:
    answer = '<tool_call>{"name":"use","arguments":{"object":"<think></think>"}}</tool_call>'

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_for(tokenizer, "<think>Act.</think>" + answer + "<|im_end|>"))

    async def run() -> loop.Rollout:
        async with make_client(respond) as client, task.init_env() as env:
            return await replace(make_agent(tokenizer, client), max_turns=1).run(env=env)

    rollout = asyncio.run(run())
    assert rollout.completion == answer
    environment.step.assert_awaited_with(ToolCall(name="use", arguments={"object": "<think></think>"}), timeout=None)
    assert '"error"' not in rollout.interaction


@pytest.mark.parametrize("mode", ["text", "invalid", "malformed", "multiple", "terminal"])
def test_tool_dispatch(mode: str, tokenizer: PreTrainedTokenizerFast, environment: AsyncMock, task: Task) -> None:
    look = '<tool_call>{"name":"look","arguments":{}}</tool_call>'
    inventory = '<tool_call>{"name":"inventory","arguments":{}}</tool_call>'
    replies = {
        "text": ["look"],
        "invalid": ['<tool_call>{"name":"unknown","arguments":{}}</tool_call>', "Finished"],
        "malformed": [look + "<tool_call>invalid</tool_call>", "Finished"],
        "multiple": [look + inventory],
        "terminal": [look + inventory],
    }[mode]
    if mode == "terminal":
        environment.step.side_effect = [StepResult(observation="Won", reward=1.0, done=True)]
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response_for(tokenizer, replies[len(requests) - 1] + "<|im_end|>"))

    async def run() -> loop.Rollout:
        async with make_client(respond) as client, task.init_env() as env:
            return await make_agent(tokenizer, client).run(env=env)

    rollout = asyncio.run(run())
    assert rollout.stop_condition == ("environment_done" if mode in {"multiple", "terminal"} else "agent_completed")
    environment.reset.assert_awaited_once_with(timeout=None)
    assert environment.step.await_count == {"text": 0, "invalid": 1, "malformed": 0, "multiple": 2, "terminal": 1}[mode]
    assert rollout.reward == (1.0 if mode in {"multiple", "terminal"} else 0.0)
    if mode == "invalid":
        assert "<tool_response>" in tokenizer.decode(requests[1]["prompt"])
        assert "Unknown ALFWorld tool" in rollout.interaction
    if mode == "malformed":
        assert '<tool_response>\n{"error":' in tokenizer.decode(requests[1]["prompt"])
    if mode == "multiple":
        assert [call.args[0].name for call in environment.step.await_args_list] == ["look", "inventory"]
        assert "At shelf." not in rollout.interaction
        assert "Task complete" not in rollout.interaction


@pytest.mark.parametrize("mode", ["success", "setup_error", "setup_timeout", "step_error", "cancel"])
def test_generic_task_environment_lifetime(mode: str, tokenizer: PreTrainedTokenizerFast) -> None:
    events: list[str] = []
    stepping = asyncio.Event()

    class ExampleEnvironment(Environment):
        async def reset(self, *, timeout: float | None = None) -> ResetResult:
            tools: list[ToolSchema] = [
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

            return ResetResult(tools=tools, system_prompt="Count using tools.", prompt="Count two.")

        async def step(self, action: ToolCall, *, timeout: float | None = None) -> StepResult:
            assert action.name == "count" and action.arguments == {"amount": 2}
            stepping.set()
            if mode == "step_error":
                raise RuntimeError("Environment failed")
            if mode == "cancel":
                await asyncio.Future()
            return StepResult(observation="Counted two", reward=0.5, done=True)

    class ExampleTask(Task):
        id = "one"
        name = "example"

        @asynccontextmanager
        async def init_env(self) -> AsyncGenerator[Environment]:
            events.append("enter")
            try:
                if mode == "setup_error":
                    raise RuntimeError("Setup failed")
                if mode == "setup_timeout":
                    await asyncio.Future()
                yield ExampleEnvironment()
            finally:
                events.append("exit")

    def respond(request: httpx.Request) -> httpx.Response:
        prompt = tokenizer.decode(json.loads(request.content)["prompt"])
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
        task: Task = ExampleTask()
        async with (
            make_client(respond) as client,
            asyncio.timeout(0.01 if mode == "setup_timeout" else 5),
            task.init_env() as env,
        ):
            agent = make_agent(tokenizer, client)
            running = asyncio.create_task(agent.run(env=env))
            if mode == "cancel":
                await stepping.wait()
                running.cancel()
            return await running

    if mode == "success":
        result = asyncio.run(run())
        assert result.reward == 0.5 and result.stop_condition == "environment_done"
        assert "Counted two" not in result.interaction
        assert result.prompt == "Count two."
    else:
        error = {"cancel": asyncio.CancelledError, "setup_timeout": TimeoutError}.get(mode, RuntimeError)
        with pytest.raises(error):
            asyncio.run(run())
    assert events == ["enter", "exit"]
