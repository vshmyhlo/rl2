"""Task and environment contracts for the tool-calling agent loop.

Reset starts an active episode with no reward and at least one valid step available.
Adapters must assert that upstream reset rewards are absent or zero and terminal
flags are absent or false. Rewards and termination are reported only by step().
Each step reports its own reward; the episode return is the sum of step rewards,
whether rewards arrive throughout the episode or only at termination.
After termination, reset is required before another step. Model-supplied formatting,
parsing, and tool-schema errors are recoverable: raise ToolCallError with actionable
feedback before advancing the environment, awarding reward, or ending the episode.
The agent loop returns that feedback to the model so it can correct and retry within
its turn/token budget. Native tool execution errors may instead be returned as
nonterminal observations when execution has already occurred. Well-formed incorrect
answers receive normal task scoring; do not expose reference answers in error feedback
or disguise infrastructure failures and internal bugs as model formatting errors.
Each rollout owns its resources through Task.init_env(), including cleanup on
errors and cancellation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict


class ToolParameter(TypedDict, total=False):
    type: Literal["string", "integer", "number", "boolean", "object", "array", "null"]
    description: str
    additionalProperties: bool | ToolParameter
    items: ToolParameter
    prefixItems: list[ToolParameter]
    minItems: int
    maxItems: int
    anyOf: list[ToolParameter]


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
    """Recoverable model input error; give correction guidance and leave the episode active."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.feedback = message  # Preserved when remote transports wrap the exception.


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
    single_tool_call: bool = False


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
