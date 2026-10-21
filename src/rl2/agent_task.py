"""Task and environment contracts for the tool-calling agent loop."""

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict


class ToolParameter(TypedDict):
    type: Literal["string", "integer", "number", "boolean"]
    description: NotRequired[str]


class ToolParameters(TypedDict):
    type: Literal["object"]
    properties: dict[str, ToolParameter]
    required: list[str]
    additionalProperties: NotRequired[bool]


class ToolFunction(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: ToolParameters


class ToolSchema(TypedDict):
    type: Literal["function"]
    function: ToolFunction


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str
    arguments: dict[str, Any]


class ToolCallError(ValueError):
    """Invalid model-supplied action; returned to the model as tool feedback."""


@dataclass(frozen=True)
class EpisodeResult:
    observation: str
    reward: float
    done: bool


class Environment(ABC):
    @abstractmethod
    async def reset(self) -> EpisodeResult: ...

    @abstractmethod
    async def step(self, action: ToolCall) -> EpisodeResult: ...


class Task(ABC):
    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def system_prompt(self) -> str: ...

    @property
    @abstractmethod
    def prompt(self) -> str: ...

    @property
    @abstractmethod
    def tools(self) -> list[ToolSchema]: ...

    @abstractmethod
    def init_env(self, *, max_steps: int, timeout: float) -> AbstractAsyncContextManager[Environment]: ...
