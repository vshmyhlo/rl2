import asyncio
import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
from datasets import Dataset
from datasets import config as datasets_config

from rl2.agent_task import ToolCall, ToolCallError
from rl2.lm_envs.codegym import env
from rl2.lm_envs.codegym.tools import extract_tools

TOOLS = extract_tools((Path(__file__).parent / "fixtures/codegym.py").read_text(), "SumEnv")

ABILITY = 'codegym_v1@Example__SumEnv@{"values": [1, 2], "label": "a@b"}'


def row(**updates: Any) -> dict[str, Any]:
    return {
        "ability": ABILITY,
        "prompt": [
            {"role": "system", "content": "Functions: Observe(), Done(answer)", "name": "functions"},
            {"role": "user", "content": "Use <|FunctionCallBegin|>...<|FunctionCallEnd|>.\nQuestion:\nAdd the values"},
        ],
        "plugin_call_max_round": 3,
        "answer": "Hidden reference",
    } | updates


def task(**updates: Any) -> env.CodeGymTask:
    return env.CodeGymTask(id="0", config=env.CodeGymConfig(), row=env.CodeGymRow.model_validate(row(**updates)))


def action(name: str, **parameters: Any) -> ToolCall:
    return ToolCall(name=name, arguments=parameters)


@dataclass
class FakeServer:
    paths: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    clients: list[httpx.AsyncClient] = field(default_factory=list)
    failure: str = ""
    initial_reward: float = 0
    initial_done: bool = False
    step_timeout: float = 10

    def respond(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.paths.append(path)
        if path == self.failure:
            return httpx.Response(500, json={"detail": "backend failed"})
        if path.startswith("/tools/"):
            return httpx.Response(200, json={"tools": TOOLS})
        if path == "/get_instance":
            return httpx.Response(200, json={"uid": "worker", "port": 10000})
        if path == "/release_instance":
            return httpx.Response(200, json={"status": "released"})
        assert request.url.port == 10000
        body = json.loads(request.content) if request.content else dict(request.url.params)
        session_id = body["session_id"]
        if path == "/start":
            assert body["env_name"] == "codegym_v1"
            assert body["env_str"] == ABILITY.split("@", 1)[1]
            self.sessions[session_id] = {"done": self.initial_done, "reward": self.initial_reward}
            return httpx.Response(200, json={"observation": "Initial state"})
        if path == "/stop":
            self.sessions.pop(session_id, None)
            return httpx.Response(200, json={})
        state = self.sessions[session_id]
        if path == "/step":
            call = json.loads(body["action"])
            self.actions.append(call)
            assert body["timeout"] == self.step_timeout
            assert request.extensions["timeout"]["read"] > self.step_timeout
            if call["name"] in {"Done", "Earn"}:
                state["reward"] = 2 if call["parameters"].get("answer") == 3 else 0
                state["done"] = call["name"] == "Done"
            return httpx.Response(200, json={"status": True, "observation": "Function result"})
        assert path in {"/is_done", "/reward"}
        return httpx.Response(200, json=state)


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> FakeServer:
    server = FakeServer()
    original_client = httpx.AsyncClient

    def client(**kwargs: Any) -> httpx.AsyncClient:
        instance = original_client(transport=httpx.MockTransport(server.respond), **kwargs)
        server.clients.append(instance)
        return instance

    monkeypatch.setattr(env.httpx, "AsyncClient", client)
    return server


@pytest.mark.parametrize("suffix", ["jsonl", "parquet"])
def test_load_dataset(suffix: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", tmp_path / "cache")
    path = tmp_path / f"train.{suffix}"
    if suffix == "parquet":
        Dataset.from_list([row()]).to_parquet(path)
    else:
        path.write_text(json.dumps(row()) + "\n")
    config = env.CodeGymConfig(dataset_file=str(path))
    tasks = env.stream_codegym_tasks(config, Random(0))
    sampled = next(tasks)
    assert sampled.id == "0"
    assert sampled.row.ability == ABILITY
    assert sampled.row.question == "Add the values"
    assert sampled.row.plugin_call_max_round == 3
    assert sampled.config == config
    assert next(tasks) == sampled
    assert pickle.loads(pickle.dumps(sampled)) == sampled


def test_default_dataset_and_rng_restoration(monkeypatch: pytest.MonkeyPatch) -> None:
    load = Mock(return_value=[row(), row(plugin_call_max_round=2)])
    monkeypatch.setattr(env, "load_dataset", load)
    rng = Random(3)
    config = env.CodeGymConfig()
    stream = env.stream_codegym_tasks(config, rng)
    load.assert_called_once_with("parquet", data_files={"train": env.CODEGYM_TRAIN_FILE}, split="train")
    next(stream)
    restored_rng = Random()
    restored_rng.setstate(rng.getstate())
    restored = env.stream_codegym_tasks(config, restored_rng)
    assert [next(stream).id for _ in range(8)] == [next(restored).id for _ in range(8)]


def test_load_all_training_shards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", tmp_path / "cache")
    for name in ["train-00000", "train-00001", "test-00000"]:
        Dataset.from_list([row()]).to_parquet(tmp_path / f"{name}.parquet")
    rng = Mock(spec=Random)
    rng.randrange.side_effect = [0, 1]
    stream = env.stream_codegym_tasks(env.CodeGymConfig(dataset_file=str(tmp_path / "train-*.parquet")), rng)
    assert [next(stream).id, next(stream).id] == ["0", "1"]
    assert rng.randrange.call_args_list == [((2,),), ((2,),)]


@pytest.mark.parametrize(
    "update",
    [
        {"prompt": []},
        {"ability": "other@Env@{}"},
        {"ability": "codegym_v1@@{}"},
        {"ability": "codegym_v1@Env@bad-json"},
        {"ability": "codegym_v1@Env@[]"},
        {"plugin_call_max_round": 0},
        {"prompt": [{"role": "system", "content": "Docs"}, {"role": "user", "content": "No question marker"}]},
    ],
    ids=[
        "empty-prompt",
        "wrong-environment",
        "missing-name",
        "invalid-config",
        "non-object-config",
        "zero-steps",
        "question",
    ],
)
def test_invalid_rows(update: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "load_dataset", Mock(return_value=[row(**update)]))
    with pytest.raises(ValueError, match="Invalid CodeGym row 0"):
        next(env.stream_codegym_tasks(env.CodeGymConfig(), Random(0)))


def test_empty_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "load_dataset", Mock(return_value=[]))
    with pytest.raises(ValueError, match="empty"):
        env.stream_codegym_tasks(env.CodeGymConfig(), Random(0))


def test_episode_and_independent_attempts(server: FakeServer) -> None:
    async def run() -> None:
        sampled = task()
        async with sampled.init_env() as first, sampled.init_env() as second:
            initial = await first.reset()
            await second.reset()
            assert first.session.session_id != second.session.session_id
            assert "call_function" not in initial.system_prompt
            assert "Initial state" in initial.prompt
            assert initial.single_tool_call
            assert (
                "If the initial observation lacks the task data, use an available observation tool "
                "to inspect the current instance before acting."
            ) in initial.system_prompt
            assert "illustrative, not the current instance" in initial.system_prompt
            assert "Add the values" in initial.prompt
            assert "FunctionCallBegin" not in initial.prompt + initial.system_prompt
            assert "Hidden reference" not in initial.prompt + initial.system_prompt
            assert initial.tools == TOOLS
            observed = await first.step(action("Observe", nested=[1, {"a": True}]))
            assert not observed.done and observed.reward == 0
            finished = await first.step(action("Done", answer=3))
            assert finished.done and finished.reward == 1
            independent = await second.step(action("Observe"))
            assert not independent.done and independent.reward == 0
            assert observed.observation == "Function result"

    asyncio.run(run())
    assert server.actions[0] == {"name": "Observe", "parameters": {"nested": [1, {"a": True}]}}
    assert not server.sessions
    assert server.paths.count("/release_instance") == 2
    assert all(client.is_closed for client in server.clients)


def test_cumulative_reward_and_step_limit(server: FakeServer) -> None:
    async def run() -> None:
        async with task(plugin_call_max_round=2).init_env() as environment:
            with pytest.raises(RuntimeError, match="Reset"):
                await environment.step(action("Observe"))
            assert not server.actions
            await environment.reset()
            first = await environment.step(action("Earn", answer=3))
            second = await environment.step(action("Observe"))
            assert first.reward == 1 and not first.done
            assert second.reward == 0 and second.done
            with pytest.raises(RuntimeError, match="Reset"):
                await environment.step(action("Observe"))
            assert len(server.actions) == 2
            await environment.reset()
            assert len(server.sessions) == 1
            assert environment.score == 0 and environment.steps == 0
            failed = await environment.step(action("Done", answer=0))
            assert failed.done and failed.reward == 0
            with pytest.raises(RuntimeError, match="Reset"):
                await environment.step(action("Observe"))
            assert len(server.actions) == 3

    asyncio.run(run())
    assert server.paths.count("/release_instance") == 2


@pytest.mark.parametrize("state", ["reward", "negative_reward", "done"])
def test_reset_contract(state: str, server: FakeServer) -> None:
    server.initial_reward = 1 if state == "reward" else -1 if state == "negative_reward" else 0
    server.initial_done = state == "done"

    async def run() -> None:
        async with task().init_env() as environment:
            with pytest.raises(AssertionError, match="CodeGym reset"):
                await environment.reset()

    asyncio.run(run())
    assert not server.sessions


@pytest.mark.parametrize(
    "call",
    [
        ToolCall(name="python", arguments={}),
        ToolCall(name="Done", arguments={}),
        ToolCall(name="Done", arguments={"answer": 3, "extra": 1}),
    ],
    ids=["unknown-tool", "missing-argument", "extra-argument"],
)
def test_invalid_tool_calls(call: ToolCall, server: FakeServer) -> None:
    async def run() -> None:
        async with task().init_env() as environment:
            await environment.reset()
            with pytest.raises(ToolCallError):
                await environment.step(call)
            assert environment.steps == 0

    asyncio.run(run())
    assert not server.actions


@pytest.mark.parametrize("failure", ["/start", "/step", "/reward", "/stop", "cancel"])
def test_failures_release_sessions(failure: str, server: FakeServer) -> None:
    server.failure = failure

    async def run() -> None:
        async with task().init_env() as environment:
            await environment.reset()
            if failure == "cancel":
                raise asyncio.CancelledError()
            await environment.step(action("Observe"))

    with pytest.raises(asyncio.CancelledError if failure == "cancel" else httpx.HTTPStatusError):
        asyncio.run(run())
    assert server.paths[-2:] == ["/stop", "/release_instance"]
    assert all(client.is_closed for client in server.clients)


def test_step_deadline_releases_session(server: FakeServer, monkeypatch: pytest.MonkeyPatch) -> None:
    async def blocked_request(client: httpx.AsyncClient, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if path == "/step":
            await asyncio.Event().wait()
        return await original_request(client, method, path, **kwargs)

    original_request = env.request_json
    monkeypatch.setattr(env, "request_json", blocked_request)

    async def run() -> None:
        with pytest.raises(TimeoutError):
            async with task().init_env() as environment:
                await environment.reset()
                await environment.step(action("Observe"), timeout=0.01)

    asyncio.run(run())
    assert server.paths[-2:] == ["/stop", "/release_instance"]
    assert not server.sessions


@pytest.mark.skipif(
    not os.environ.get("CODEGYM_MANAGER_URL"), reason="Set CODEGYM_MANAGER_URL for the live server test"
)
@pytest.mark.parametrize(
    ("ability", "answer"),
    [
        pytest.param('codegym_v1@Code_Contests_19816_I__NextClosestTimeEnv@{"time":"19:34"}', "19:39", id="clock"),
        pytest.param(
            'codegym_v1@Codeforces_5537_I__MinPathCostEnv@{"grid":[[0,0,0],[0,0,0],[0,0,0]]}',
            0,
            id="repaired-upstream-syntax",
        ),
    ],
)
def test_live_cluster_episode(ability: str, answer: str | int) -> None:
    sampled = env.CodeGymTask(
        id="smoke",
        config=env.CodeGymConfig(manager_url=os.environ["CODEGYM_MANAGER_URL"]),
        row=env.CodeGymRow.model_validate(row(ability=ability)),
    )

    async def run() -> None:
        async with sampled.init_env() as first, sampled.init_env() as second:
            await first.reset(timeout=30)
            await second.reset(timeout=30)
            observation = await first.step(action("Observe"), timeout=30)
            assert observation.observation and not observation.done
            result = await first.step(action("Done", answer=answer), timeout=30)
            assert result.reward == 1 and result.done
            other = await second.step(action("Observe"), timeout=30)
            assert other.reward == 0 and not other.done

    asyncio.run(run())


@pytest.mark.parametrize("tools", [[], [TOOLS[0], TOOLS[0]]], ids=["empty", "duplicate"])
def test_invalid_schemas_do_not_allocate_worker(
    tools: list[dict], server: FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(globals(), "TOOLS", tools)

    async def run() -> None:
        async with task().init_env() as environment:
            with pytest.raises(ValueError, match="tool definitions"):
                await environment.reset()

    asyncio.run(run())
    assert server.paths == ["/tools/Example__SumEnv"]
    assert all(client.is_closed for client in server.clients)


@pytest.mark.skipif(
    not os.environ.get("CODEGYM_MANAGER_URL"), reason="Set CODEGYM_MANAGER_URL for the live server test"
)
def test_live_serialized_and_array_arguments() -> None:
    sampled = env.CodeGymTask(
        id="expression",
        config=env.CodeGymConfig(manager_url=os.environ["CODEGYM_MANAGER_URL"]),
        row=env.CodeGymRow.model_validate(
            row(ability='codegym_v1@Code_Contests_11321_I__ExpressionEvaluationEnv@{"expression":"1+2"}')
        ),
    )

    async def run() -> None:
        async with sampled.init_env() as environment:
            reset = await environment.reset(timeout=30)
            functions = {tool["function"]["name"]: tool["function"] for tool in reset.tools}
            assert functions["GetTokenAtPosition"]["parameters"]["properties"]["tokens"]["type"] == "string"
            assert functions["ApplyOperator"]["parameters"]["properties"]["values"]["type"] == "array"
            token = await environment.step(action("GetTokenAtPosition", tokens='[1,"+",2]', position=0), timeout=30)
            assert token.observation == "1" and not token.done
            result = await environment.step(action("ApplyOperator", operator="+", values=[1, 2]), timeout=30)
            assert result.observation == "3" and not result.done
            done = await environment.step(action("Done", answer=3), timeout=30)
            assert done.reward == 1 and done.done

    asyncio.run(run())


def test_backend_step_timeout_fits_http_deadline(server: FakeServer) -> None:
    server.step_timeout = 180
    sampled = env.CodeGymTask(
        id="slow-step",
        config=env.CodeGymConfig(step_timeout=server.step_timeout),
        row=env.CodeGymRow.model_validate(row()),
    )

    async def run() -> None:
        async with sampled.init_env() as environment:
            await environment.reset()
            result = await environment.step(action("Observe"))
            assert not result.done and result.reward == 0

    asyncio.run(run())
