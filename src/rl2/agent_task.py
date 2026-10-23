"""Task and environment contracts for the tool-calling agent loop.

Reset starts an active episode with no reward and at least one valid step available.
Adapters must assert that upstream reset rewards are absent or zero and terminal
flags are absent or false. Rewards and termination are reported only by step().
Each step reports its own reward; the episode return is the sum of step rewards,
whether rewards arrive throughout the episode or only at termination.
"""

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
class StepResult:
    """Outcome of one step; reward is for this step, not a cumulative score."""

    observation: str
    reward: float
    done: bool


@dataclass(frozen=True)
class ResetResult:
    tools: list[ToolSchema]
    system_prompt: str
    prompt: str


class Environment(ABC):
    @abstractmethod
    async def reset(self, *, timeout: float | None = None) -> ResetResult: ...

    @abstractmethod
    async def step(self, action: ToolCall, *, timeout: float | None = None) -> StepResult: ...


class Task(ABC):
    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def init_env(self) -> AbstractAsyncContextManager[Environment]: ...
