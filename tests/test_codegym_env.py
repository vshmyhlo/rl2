import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
import verifiers.v1 as vf
from datasets import Dataset
from datasets import config as datasets_config

from rl2 import codegym_env as env

ABILITY = 'codegym_v1@Example__SumEnv@{"values": [1, 2], "label": "a@b"}'


def call(body: str) -> str:
    return f"<|FunctionCallBegin|>{body}<|FunctionCallEnd|>"


def test_parse_action() -> None:
    action = {"name": "Done", "parameters": {"answer": [3, {"nested": True}]}}
    assert json.loads(env.parse_action("Brief reasoning.\n" + call(json.dumps([action])))) == action


@pytest.mark.parametrize(
    "reply",
    [
        "no function call",
        call("not JSON"),
        call("[]"),
        call('[{"name":"A","parameters":{}},{"name":"B","parameters":{}}]'),
        call('[{"name":"A","arguments":{}}]'),
        call('[{"name":"A","parameters":[]}]'),
        call('[{"name":"A","parameters":{}}]') * 2,
    ],
    ids=["missing", "bad-json", "empty", "multiple-actions", "wrong-schema", "bad-parameters", "multiple-blocks"],
)
def test_invalid_actions(reply: str) -> None:
    with pytest.raises(ValueError):
        env.parse_action(reply)


@pytest.mark.parametrize("suffix", ["jsonl", "parquet"])
def test_load_dataset(suffix: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", tmp_path / "cache")
    row = {
        "ability": ABILITY,
        "prompt": [
            {"role": "system", "content": "Functions: Done", "name": "functions"},
            {"role": "user", "content": "Add the values"},
        ],
        "plugin_call_max_round": 2,
        "answer": "Hidden reference",
    }
    path = tmp_path / f"train.{suffix}"
    if suffix == "parquet":
        Dataset.from_list([row]).to_parquet(path)
    else:
        path.write_text(json.dumps(row) + "\n")
    config = env.CodeGymTasksetConfig(data_file=str(path), task=vf.TaskConfig())
    tasks = list(env.CodeGymTaskset(config=config))
    assert len(tasks) == 1
    task = tasks[0]
    assert task.data.ability == ABILITY
    assert task.data.id == "0"
    assert task.data.idx == 0
    assert task.data.max_steps == 2
    assert task.data.prompt_text == "Functions: Done\n\nAdd the values"
    assert "Hidden reference" not in task.data.prompt_text
    assert task.config == config.task


@pytest.mark.parametrize(
    "update",
    [
        {"prompt": []},
        {"ability": "other@Env@{}"},
        {"ability": "codegym_v1@@{}"},
        {"ability": "codegym_v1@Env@bad-json"},
        {"ability": "codegym_v1@Env@[]"},
        {"plugin_call_max_round": 0},
    ],
    ids=["empty-prompt", "wrong-environment", "missing-name", "invalid-config", "non-object-config", "zero-steps"],
)
def test_invalid_rows(update: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    row = {"prompt": [{"role": "user", "content": "Solve"}], "ability": ABILITY} | update
    monkeypatch.setattr(env, "load_dataset", Mock(return_value=[row]))
    with pytest.raises(ValueError, match="Invalid CodeGym row 0"):
        list(env.CodeGymTaskset(config=env.CodeGymTasksetConfig()))


def test_empty_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "load_dataset", Mock(return_value=[]))
    with pytest.raises(ValueError, match="empty"):
        list(env.CodeGymTaskset(config=env.CodeGymTasksetConfig()))


@pytest.mark.parametrize("failure", [None, "start", "rollout", "stop", "cancel"])
def test_session_cleanup(failure: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    paths: list[str] = []
    clients: list[httpx.AsyncClient] = []
    original_client = httpx.AsyncClient

    def respond(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        paths.append(path)
        if path == "/get_instance":
            assert request.url.port == 8000
            return httpx.Response(200, json={"uid": "worker-1", "port": 10000})
        if path == "/release_instance":
            assert request.url.params["uid"] == "worker-1"
        else:
            assert request.url.host == "::1"
            assert request.url.port == 10000
        if path == "/start":
            payload = json.loads(request.content)
            assert payload["env_name"] == "codegym_v1"
            assert payload["env_str"] == ABILITY.split("@", 1)[1]
        return httpx.Response(500 if path == f"/{failure}" else 200, json={"observation": "Initial state"})

    def client(**kwargs: Any) -> httpx.AsyncClient:
        instance = original_client(transport=httpx.MockTransport(respond), **kwargs)
        clients.append(instance)
        return instance

    monkeypatch.setattr(env.httpx, "AsyncClient", client)

    async def run() -> None:
        async with env.codegym_session("http://[::1]:8000", ABILITY) as session:
            assert session.observation == "Initial state"
            assert session.session_id
            if failure == "rollout":
                raise RuntimeError("rollout failed")
            if failure == "cancel":
                raise asyncio.CancelledError()

    if failure is None:
        asyncio.run(run())
    else:
        error = (
            asyncio.CancelledError
            if failure == "cancel"
            else RuntimeError
            if failure == "rollout"
            else httpx.HTTPStatusError
        )
        with pytest.raises(error):
            asyncio.run(run())
    assert paths == ["/get_instance", "/start", "/stop", "/release_instance"]
    assert all(client.is_closed for client in clients)
