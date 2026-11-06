"""Append-only Qwen3 rollouts for tool-calling tasks."""

import asyncio
import json
import re
import time
from dataclasses import dataclass
from math import exp
from typing import Literal

import httpx
import numpy as np
from numpy.typing import NDArray
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerBase

from rl2.agent_task import Environment, ToolCall, ToolCallError
from rl2.tracing import span

StopCondition = Literal[
    "agent_completed",  # The model finished a response without tool calls.
    "environment_done",  # A tool step reported that the environment episode ended.
    "max_turns",  # The rollout reached its maximum number of model responses.
    "max_tokens",  # The total transcript token budget prevented further generation.
    "context_length",  # The model context window prevented further generation (wins ties with max_tokens).
    "unspecified",  # No stop reason was recorded.
]


def completion_text(text: str) -> str:
    # Strip leading reasoning, without interpreting tags inside tool arguments.
    text = text.strip()
    marker = re.search(r"<think>|</think>|<tool_call>", text)
    if marker is not None and (marker.group() == "</think>" or text.startswith("<think>")):
        text = text[marker.end() :]
        if marker.group() == "<think>":
            _, separator, text = text.partition("</think>")
            if not separator:
                return ""
    return text.removesuffix("<|im_end|>").removesuffix("<|endoftext|>").strip()


def parse_tool_calls(text: str) -> list[ToolCall]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Invalid JSON constant: {value}")

    text = completion_text(text)
    calls: list[ToolCall] = []
    decoder = json.JSONDecoder(parse_constant=reject_constant)
    while match := re.search(r"</?tool_call>", text):
        if match.group() != "<tool_call>":
            raise ValueError("Tool calls must use complete <tool_call>...</tool_call> blocks")
        text = text[match.end() :].lstrip()
        try:
            value, end = decoder.raw_decode(text)
        except RecursionError as error:
            raise ValueError("Tool calls exceed the JSON nesting limit") from error
        text = text[end:].lstrip()
        if not text.startswith("</tool_call>"):
            raise ValueError("Tool calls must use complete <tool_call>...</tool_call> blocks")
        calls.append(ToolCall.model_validate(value))
        text = text[len("</tool_call>") :]
    return calls


class WeightVersionSpan(BaseModel):
    version: str
    start: int
    end: int


class AlternativeLogProb(BaseModel):
    token_id: int
    logprob: float


class TokenLogProb(AlternativeLogProb):
    top_logprobs: list[AlternativeLogProb] = Field(default_factory=list)


class CompletionLogProbs(BaseModel):
    content: list[TokenLogProb]


class CompletionChoice(BaseModel):
    token_ids: list[int]
    logprobs: CompletionLogProbs
    weight_versions: list[WeightVersionSpan]
    finish_reason: str


class CompletionResponse(BaseModel):
    choices: list[CompletionChoice]


@dataclass(frozen=True)
class TrainingSequence:
    token_ids: NDArray[np.int32]  # (T,) exact context and generated tokens.
    loss_mask: NDArray[np.bool_]  # (T,) True only for sampled tokens.
    old_log_probs: NDArray[np.float32]  # (T,) zero for context tokens.
    weight_versions: NDArray[np.int32]  # (T,) publication counters; -1 for context tokens.
    sampler_topk_ids: NDArray[np.int32] | None = None  # (T, K) when score centering is enabled.
    sampler_topk_log_probs: NDArray[np.float32] | None = None  # (T, K); zero for context.


@dataclass(frozen=True)
class Rollout:
    sequence: TrainingSequence
    prompt: str
    reward: float
    advantage: float
    completion: str
    interaction: str  # Full decoded sequence.token_ids, with special tokens preserved.
    num_turns: int
    num_input_tokens: int
    num_output_tokens: int
    rollout_seconds: float
    stop_condition: StopCondition


def completion_tokens(choice: CompletionChoice) -> tuple[list[int], list[float]]:
    """Validate sampled tokens, scores, and version coverage at the vLLM boundary."""
    ids = choice.token_ids
    log_probs = [token.logprob for token in choice.logprobs.content]
    if not ids or len(log_probs) != len(ids):
        raise ValueError("Each sampled token must have a log-probability")
    for token_id, token in zip(ids, choice.logprobs.content, strict=True):
        if token_id < 0 or token.token_id != token_id:
            raise ValueError("Sampled token and log-probability IDs differ")
    if any(not -9999.0 < value <= 0 for value in log_probs):
        raise ValueError("Missing or invalid sampling log-probability")
    end = 0
    for version in choice.weight_versions:
        if (
            not version.version.isascii()
            or not version.version.isdecimal()
            or int(version.version) > np.iinfo(np.int32).max
        ):
            raise ValueError("Weight versions must be nonnegative int32 publication counters")
        if version.start != end or not version.start < version.end <= len(ids):
            raise ValueError("Weight version spans must cover each sampled token exactly once")
        end = version.end
    if end != len(ids):
        raise ValueError("Weight version spans must cover each sampled token exactly once")
    return ids, log_probs


def completion_topk(choice: CompletionChoice, top_k: int) -> tuple[list[list[int]], list[list[float]]]:
    """Validate and retain K distinct sampler alternatives per generated token."""
    ids: list[list[int]] = []
    log_probs: list[list[float]] = []
    for token in choice.logprobs.content:
        # vLLM may include the sampled token in addition to its top K alternatives.
        alternatives = sorted(token.top_logprobs, key=lambda entry: entry.logprob, reverse=True)[:top_k]
        if len(alternatives) != top_k or len({entry.token_id for entry in alternatives}) != top_k:
            raise ValueError("Score centering requires K distinct sampler top-logprobs per token")
        if any(entry.token_id < 0 or not -9999.0 < entry.logprob <= 0 for entry in alternatives):
            raise ValueError("Invalid sampler top-logprob")
        values = [entry.logprob for entry in alternatives]
        if sum(exp(value) for value in values) > 1.00001:
            raise ValueError("Sampler top-logprob probability mass exceeds one")
        ids.append([entry.token_id for entry in alternatives])
        log_probs.append(values)
    return ids, log_probs


@dataclass(frozen=True)
class Agent:
    client: AsyncOpenAI
    tokenizer: PreTrainedTokenizerBase
    model: str
    context_length: int
    max_tokens: int  # Total transcript budget, including context and sampled tokens.
    max_turns: int
    temperature: float
    rollout_timeout: float
    environment_timeout: float | None = None
    sampler_top_k: int = 0

    async def run(self, env: Environment) -> Rollout:
        started = time.monotonic()
        token_limit = min(self.max_tokens, self.context_length)
        limit_condition: StopCondition = "max_tokens" if self.max_tokens < self.context_length else "context_length"
        async with asyncio.timeout(self.rollout_timeout):
            with span("actor.rollout.reset"):
                initial = await env.reset(timeout=self.environment_timeout)
            reward = 0.0
            token_ids = await asyncio.to_thread(
                self.tokenizer.apply_chat_template,
                [
                    {"role": "system", "content": initial.system_prompt},
                    {"role": "user", "content": initial.prompt},
                ],
                tools=initial.tools,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            if len(token_ids) >= token_limit:
                raise ValueError(
                    f"Initial prompt has {len(token_ids)} tokens; {limit_condition}={token_limit} "
                    "must leave room for at least one sampled token"
                )
            loss_mask = [False] * len(token_ids)
            log_probs = [0.0] * len(token_ids)
            weight_versions = [-1] * len(token_ids)
            topk_ids = [[0] * self.sampler_top_k for _ in token_ids] if self.sampler_top_k else []
            topk_log_probs = [[0.0] * self.sampler_top_k for _ in token_ids] if self.sampler_top_k else []
            num_turns = 0
            completion = ""
            stop_condition: StopCondition = "max_turns"
            stop_token_ids = self.tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
            for _ in range(self.max_turns):
                remaining = token_limit - len(token_ids)
                with span("actor.rollout.model", {"input_tokens": len(token_ids)}) as model_span:
                    response = await self.client.post(
                        str(self.client.base_url.copy_with(path="/inference/v1/generate")),
                        cast_to=httpx.Response,
                        body={
                            "model": self.model,
                            "token_ids": token_ids,
                            "output_mode": "tokens",
                            "sampling_params": {
                                "max_tokens": remaining,
                                "temperature": self.temperature,
                                "top_p": 1.0,
                                "top_k": -1,
                                # vLLM includes the sampled token before truncating this list.
                                "logprobs": self.sampler_top_k + 1 if self.sampler_top_k else 0,
                                "stop_token_ids": stop_token_ids,
                                "detokenize": False,
                            },
                        },
                    )
                    choices = CompletionResponse.model_validate(response.json(), strict=True).choices
                    if len(choices) != 1:
                        raise ValueError("Expected one completion per model request")
                    choice = choices[0]
                    if choice.finish_reason not in {"stop", "length"}:
                        raise RuntimeError(f"Generation failed: {choice.finish_reason}")
                    ids, sampled_log_probs = completion_tokens(choice)
                    if self.sampler_top_k:
                        head_ids, head_log_probs = completion_topk(choice, self.sampler_top_k)
                        topk_ids.extend(head_ids)
                        topk_log_probs.extend(head_log_probs)
                    if len(ids) > remaining:
                        raise ValueError("Completion exceeds the requested token budget")
                    model_span.set_attribute("output_tokens", len(ids))
                for version in choice.weight_versions:
                    weight_versions.extend([int(version.version)] * (version.end - version.start))
                token_ids.extend(ids)
                loss_mask.extend([True] * len(ids))
                log_probs.extend(sampled_log_probs)
                num_turns += 1
                text = await asyncio.to_thread(
                    self.tokenizer.decode, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                completion = completion_text(text)
                if choice.finish_reason == "length":
                    stop_condition = limit_condition
                    break
                pending: list[str] = []
                done = False
                try:
                    calls = parse_tool_calls(text)
                    if initial.single_tool_call and len(calls) > 1:
                        raise ToolCallError(
                            "Call exactly one function per response and wait for its result. "
                            "None of these calls were executed. Retry with a single tool call."
                        )
                except ValueError as error:
                    pending.append(json.dumps({"error": str(error)}))
                else:
                    if not calls:
                        stop_condition = "agent_completed"
                        break
                    for call in calls:
                        with span("actor.rollout.step", {"tool": call.name}):
                            try:
                                result = await env.step(call, timeout=self.environment_timeout)
                            except ToolCallError as error:
                                pending.append(json.dumps({"error": error.feedback}))
                                continue
                        reward += result.reward
                        done = result.done
                        pending.append(result.observation)
                        if done:
                            break
                if done:
                    stop_condition = "environment_done"
                    break
                if num_turns == self.max_turns:
                    break
                # Qwen3 groups tool results in one user message; sampled history stays untouched.
                responses = "\n".join(f"<tool_response>\n{content}\n</tool_response>" for content in pending)
                suffix = ("" if ids[-1] == stop_token_ids[0] else "<|im_end|>") + (
                    f"\n<|im_start|>user\n{responses}<|im_end|>\n<|im_start|>assistant\n"
                )
                context = await asyncio.to_thread(self.tokenizer.encode, suffix, add_special_tokens=False)
                if len(token_ids) + len(context) >= token_limit:
                    stop_condition = limit_condition
                    break
                token_ids.extend(context)
                loss_mask.extend([False] * len(context))
                log_probs.extend([0.0] * len(context))
                weight_versions.extend([-1] * len(context))
                if self.sampler_top_k:
                    topk_ids.extend([[0] * self.sampler_top_k for _ in context])
                    topk_log_probs.extend([[0.0] * self.sampler_top_k for _ in context])
            num_output_tokens = sum(loss_mask)
            if num_output_tokens == 0:
                raise ValueError("Rollout has no trainable tokens")
            interaction = await asyncio.to_thread(
                self.tokenizer.decode, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            return Rollout(
                sequence=TrainingSequence(
                    token_ids=np.asarray(token_ids, dtype=np.int32),  # (T,)
                    loss_mask=np.asarray(loss_mask, dtype=np.bool_),  # (T,)
                    old_log_probs=np.asarray(log_probs, dtype=np.float32),  # (T,)
                    weight_versions=np.asarray(weight_versions, dtype=np.int32),  # (T,)
                    sampler_topk_ids=(np.asarray(topk_ids, dtype=np.int32) if self.sampler_top_k else None),  # (T, K)
                    sampler_topk_log_probs=(
                        np.asarray(topk_log_probs, dtype=np.float32) if self.sampler_top_k else None
                    ),  # (T, K)
                ),
                prompt=initial.prompt,
                reward=reward,
                advantage=0.0,
                completion=completion,
                interaction=interaction,
                num_turns=num_turns,
                num_input_tokens=len(token_ids) - num_output_tokens,
                num_output_tokens=num_output_tokens,
                rollout_seconds=time.monotonic() - started,
                stop_condition=stop_condition,
            )
