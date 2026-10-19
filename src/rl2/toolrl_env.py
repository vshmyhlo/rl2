import json
import re
from collections import Counter
from collections.abc import Iterable

import verifiers.v1 as vf
from datasets import load_dataset
from pydantic import BaseModel, ConfigDict, JsonValue
from renderers.base import ParsedResponse, ToolSpec
from renderers.configs import Qwen3RendererConfig
from renderers.qwen3 import Qwen3Renderer

TOOLRL_TRAIN_FILE = (
    "https://raw.githubusercontent.com/qiancheng0/ToolRL/"
    "8cee13ec0ca72f0461da372a93a6fd8140dbb840/dataset/rlla_4k/train.parquet"
)


class ToolRLRenderer(Qwen3Renderer):
    """Keep ToolRL's output tags as text for grading, without executing calls."""

    config_class = Qwen3RendererConfig

    def parse_response(
        self,
        token_ids: list[int],  # (T,) Sampled completion tokens.
        *,
        tools: list[ToolSpec] | None = None,
        prompt_ids: list[int] | None = None,  # (P,)
    ) -> ParsedResponse:
        end = len(token_ids)
        stop_ids = self.get_stop_token_ids()
        while end and token_ids[end - 1] in stop_ids:
            end -= 1
        return ParsedResponse(content=self._tokenizer.decode(token_ids[:end], skip_special_tokens=False))


class ToolCall(BaseModel):
    model_config = ConfigDict(strict=True)

    name: str
    parameters: dict[str, JsonValue]


def parse_tool_calls(text: str) -> list[ToolCall]:
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if match is None:
        raise ValueError("Missing tool_call block")
    return [ToolCall.model_validate(json.loads(line)) for line in match.group(1).strip().split("\n")]


def format_reward(completion: str, answer: str) -> float:
    tags = [tag for tag in ("tool_call", "response") if f"<{tag}>" in answer]
    pattern = r"<think>.*?</think>"
    for tag in tags:
        if completion.count(f"<{tag}>") != 1 or completion.count(f"</{tag}>") != 1:
            return 0.0
        body = r"\n.*?\n" if tag == "tool_call" else r".*?"
        pattern += rf"\n<{tag}>{body}</{tag}>"
    return float(re.fullmatch(pattern, completion.strip(), re.DOTALL) is not None)


def match_score(expected: list[str], predicted: list[str]) -> float:
    if expected == predicted:
        return 1.0
    counts, predictions = Counter(expected), Counter(predicted)
    overlap = sum((counts & predictions).values())
    return overlap / (len(expected) + len(predicted) - overlap)


def tool_correctness(completion: str, expected: list[ToolCall]) -> float:
    """ToolRL's default partial-credit reward, scaled to [-3, 3]."""
    try:
        predicted = parse_tool_calls(completion)
    except ValueError:
        return -3.0
    if expected == predicted:
        return 3.0
    score = match_score([tool.name for tool in expected], [tool.name for tool in predicted])
    maximum = 1 + sum(1 + len(tool.parameters) for tool in expected)
    used: set[int] = set()
    for tool in expected:
        best_index, best_score = -1, 0.0
        for index, candidate in enumerate(predicted):
            if index in used or candidate.name != tool.name:
                continue
            value = match_score(list(tool.parameters), list(candidate.parameters))
            value += sum(
                key in candidate.parameters and candidate.parameters[key] == val for key, val in tool.parameters.items()
            )
            if value > best_score:
                best_index, best_score = index, value
        if best_index >= 0:
            used.add(best_index)
            score += best_score
    return 6 * score / maximum - 3


class ToolRLData(vf.TaskData):
    answer: str
    expected_calls: list[ToolCall]


class ToolRLTask(vf.Task[ToolRLData]):
    @vf.reward
    async def format(self, trace: vf.Trace) -> float:
        return format_reward(trace.last_reply, self.data.answer)

    @vf.reward
    async def correctness(self, trace: vf.Trace) -> float:
        # Upstream assigns no correctness reward to response-only references.
        return tool_correctness(trace.last_reply, self.data.expected_calls) if self.data.expected_calls else 0.0


class RewardReference(BaseModel):
    ground_truth: str


class ToolRLRow(BaseModel):
    prompt: vf.Messages
    reward_model: RewardReference


class ToolRLTasksetConfig(vf.TasksetConfig):
    data_file: str = TOOLRL_TRAIN_FILE


class ToolRLTaskset(vf.Taskset[ToolRLTask, ToolRLTasksetConfig]):
    def load(self) -> Iterable[ToolRLTask]:
        dataset = load_dataset("parquet", data_files={"train": self.config.data_file}, split="train")
        if not len(dataset):
            raise ValueError("ToolRL dataset is empty")
        for index, row in enumerate(dataset):
            try:
                parsed = ToolRLRow.model_validate(row)
                answer = parsed.reward_model.ground_truth
                calls = parse_tool_calls(answer) if "<tool_call>" in answer else []
            except ValueError as error:
                raise ValueError(f"Invalid ToolRL row {index}: {error}") from error
            yield ToolRLTask(
                data=ToolRLData(id=str(index), prompt=parsed.prompt, answer=answer, expected_calls=calls),
                config=self.config.task,
            )
