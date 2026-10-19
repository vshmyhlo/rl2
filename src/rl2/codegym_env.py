import json
import re
from collections.abc import AsyncIterator, Iterable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Self
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import verifiers.v1 as vf
from datasets import load_dataset
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, model_validator

CODEGYM_TRAIN_FILE = (
    "https://raw.githubusercontent.com/StigLidu/CodeGym/"
    "e85a439761ae2df813d5997a81bbd9f9e2ed27b7/example/training_instance.jsonl"
)


class CodeGymRow(BaseModel):
    prompt: vf.Messages = Field(min_length=1)
    ability: str
    plugin_call_max_round: int = Field(default=256, ge=1)

    @model_validator(mode="after")
    def validate_ability(self) -> Self:
        parts = self.ability.split("@", 2)
        if len(parts) != 3 or parts[0] != "codegym_v1" or not parts[1]:
            raise ValueError("Expected codegym_v1@environment_name@JSON_config ability")
        TypeAdapter(dict[str, JsonValue]).validate_json(parts[2])
        return self


class CodeGymAction(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    name: str = Field(min_length=1)
    parameters: dict[str, JsonValue]


def parse_action(reply: str) -> str:
    blocks = re.findall(r"<\|FunctionCallBegin\|>(.*?)<\|FunctionCallEnd\|>", reply, re.DOTALL)
    if len(blocks) != 1:
        raise ValueError("Return exactly one <|FunctionCallBegin|> JSON list <|FunctionCallEnd|> block")
    try:
        calls = json.loads(blocks[0])
        if not isinstance(calls, list) or len(calls) != 1:
            raise ValueError("Call exactly one function per step")
        return CodeGymAction.model_validate(calls[0]).model_dump_json()
    except RecursionError as error:
        raise ValueError("Function call exceeds the JSON nesting limit") from error


async def request_json(client: httpx.AsyncClient, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    response = await client.request(method, path, **kwargs)
    response.raise_for_status()
    return response.json()


@dataclass(frozen=True)
class CodeGymSession:
    client: httpx.AsyncClient
    session_id: str
    observation: str


@asynccontextmanager
async def codegym_session(manager_url: str, ability: str) -> AsyncIterator[CodeGymSession]:
    async with AsyncExitStack() as resources:
        manager = await resources.enter_async_context(httpx.AsyncClient(base_url=manager_url, timeout=120))
        instance = await request_json(manager, "GET", "/get_instance")
        resources.push_async_callback(
            request_json, manager, "GET", "/release_instance", params={"uid": instance["uid"]}
        )
        worker_url = httpx.URL(manager_url).copy_with(port=int(instance["port"]))
        worker = await resources.enter_async_context(httpx.AsyncClient(base_url=worker_url, timeout=120))
        session_id = str(uuid4())
        # Stop before releasing the worker, including when /start fails partway through.
        resources.push_async_callback(request_json, worker, "POST", "/stop", json={"session_id": session_id})
        env_name, env_str = ability.split("@", 1)
        started = await request_json(
            worker, "POST", "/start", json={"session_id": session_id, "env_name": env_name, "env_str": env_str}
        )
        yield CodeGymSession(client=worker, session_id=session_id, observation=started["observation"])


class CodeGymData(vf.TaskData):
    prompt: vf.Messages
    ability: str
    max_steps: int


class CodeGymTask(vf.Task[CodeGymData]):
    async def run(self, agent: vf.Agent, manager_url: str) -> vf.Trace:
        async with codegym_session(manager_url, self.data.ability) as session:
            # Open the original messages plus the server's initial observation as one prompt.
            prompt = [*self.data.prompt, vf.UserMessage(content=session.observation)]
            task = self.with_data(prompt=prompt).defer_scoring()
            async with agent.interaction(task) as interaction:
                segment = await interaction.turn()
                for _ in range(self.data.max_steps):
                    if segment.terminated:
                        break
                    try:
                        action = parse_action(segment.last_reply)
                    except ValueError as error:
                        observation = f"Invalid function call: {error}"
                    else:
                        result = await request_json(
                            session.client,
                            "POST",
                            "/step",
                            json={"session_id": session.session_id, "action": action, "timeout": 10},
                        )
                        observation = result["observation"]
                        done = await request_json(
                            session.client, "GET", "/is_done", params={"session_id": session.session_id}
                        )
                        if done["done"]:
                            interaction.trace.stop("codegym_done")
                            break
                    if interaction.trace.num_turns >= self.data.max_steps:
                        interaction.trace.stop("codegym_max_steps")
                        break
                    segment = await interaction.turn(observation)
                result = await request_json(session.client, "GET", "/reward", params={"session_id": session.session_id})
                # Match upstream OnlineGymEnv.reward: positive environment reward means success.
                interaction.trace.record_reward("codegym_reward", float(result["reward"] > 0))
            return interaction.trace


class CodeGymTasksetConfig(vf.TasksetConfig):
    data_file: str = CODEGYM_TRAIN_FILE


class CodeGymTaskset(vf.Taskset[CodeGymTask, CodeGymTasksetConfig]):
    def load(self) -> Iterable[CodeGymTask]:
        file_type = "parquet" if urlsplit(self.config.data_file).path.endswith(".parquet") else "json"
        dataset = load_dataset(file_type, data_files={"train": self.config.data_file}, split="train")
        if not len(dataset):
            raise ValueError("CodeGym dataset is empty")
        for index, row in enumerate(dataset):
            try:
                parsed = CodeGymRow.model_validate(row)
            except ValueError as error:
                raise ValueError(f"Invalid CodeGym row {index}: {error}") from error
            yield CodeGymTask(
                data=CodeGymData(
                    id=str(index), prompt=parsed.prompt, ability=parsed.ability, max_steps=parsed.plugin_call_max_round
                ),
                config=self.config.task,
            )
