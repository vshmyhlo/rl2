"""Append-only Qwen3 rollouts for tool-calling tasks."""

import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx
import numpy as np
from numpy.typing import NDArray
from pydantic import TypeAdapter
from transformers import PreTrainedTokenizerBase

from rl2.agent_task import Task, ToolCall, ToolCallError
from rl2.tracing import span


def completion_text(text: str) -> str:
    # Strip leading reasoning, without interpreting tags inside tool arguments.
    marker = re.search(r"<think>|</think>|<tool_call>", text)
    if marker is not None and marker.group() != "<tool_call>":
        text = text[marker.end() :]
        if marker.group() == "<think>":
            _, separator, text = text.partition("</think>")
            if not separator:
                return ""
    return text.removesuffix("<|im_end|>").removesuffix("<|endoftext|>").strip()


def parse_tool_calls(text: str) -> list[ToolCall]:
    text = completion_text(text)
    calls: list[ToolCall] = []
    decoder = json.JSONDecoder()
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


@dataclass(frozen=True)
class TrainingSequence:
    token_ids: NDArray[np.int32]  # (T,) exact context and generated tokens.
    loss_mask: NDArray[np.bool_]  # (T,) True only for sampled tokens.
    old_log_probs: NDArray[np.float32]  # (T,) zero for context tokens.


@dataclass(frozen=True)
class Rollout:
    sequence: TrainingSequence
    reward: float
    advantage: float
    completion: str
    interaction: str
    num_turns: int
    num_input_tokens: int
    num_output_tokens: int
    rollout_seconds: float
    stop_condition: str | None


def completion_tokens(choice: dict[str, Any]) -> tuple[list[int], list[float]]:
    """Validate the token/log-probability alignment at the vLLM boundary."""
    try:
        ids = TypeAdapter(list[int]).validate_python(choice["token_ids"], strict=True)
        entries = TypeAdapter(list[dict[str, Any]]).validate_python(choice["logprobs"]["content"], strict=True)
        if not ids or len(entries) != len(ids):
            raise ValueError("Each sampled token must have a log-probability")
        for token_id, entry in zip(ids, entries, strict=True):
            if token_id < 0 or entry["token"] != f"token_id:{token_id}":
                raise ValueError("Sampled token and log-probability IDs differ")
        log_probs = TypeAdapter(list[float]).validate_python([entry["logprob"] for entry in entries], strict=True)
    except (KeyError, TypeError) as error:
        raise ValueError("Missing token/log-probability data") from error
    # vLLM 0.27 uses token_id:N strings and clips unavailable/very low logprobs to -9999.
    if any(not -9999.0 < value <= 0 for value in log_probs):
        raise ValueError("Missing or invalid sampling log-probability")
    return ids, log_probs


@dataclass(frozen=True)
class Agent:
    client: httpx.AsyncClient
    tokenizer: PreTrainedTokenizerBase
    model: str
    context_length: int
    max_tokens: int
    max_turns: int
    temperature: float
    rollout_timeout: float
    environment_timeout: float

    async def run(self, task: Task) -> Rollout:
        started = time.monotonic()
        async with (
            asyncio.timeout(self.rollout_timeout),
            task.init_env(max_steps=self.max_turns, timeout=self.environment_timeout) as environment,
        ):
            with span("actor.rollout.reset"):
                result = await environment.reset()
            token_ids = await asyncio.to_thread(
                self.tokenizer.apply_chat_template,
                [{"role": "system", "content": task.system_prompt}, {"role": "user", "content": task.prompt}],
                tools=task.tools,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            loss_mask = [False] * len(token_ids)
            log_probs = [0.0] * len(token_ids)
            num_turns = 0
            completion = ""
            stop_condition = "max_turns"
            pending: list[str] = []
            for _ in range(self.max_turns):
                if result.done:
                    stop_condition = f"{task.name}_done"
                    break
                remaining = self.context_length - len(token_ids)
                if remaining <= 0:
                    stop_condition = "context_length"
                    break
                with span("actor.rollout.model", {"input_tokens": len(token_ids)}) as model_span:
                    response = await self.client.post(
                        "/inference/v1/generate",
                        json={
                            "model": self.model,
                            "token_ids": token_ids,
                            "sampling_params": {
                                "max_tokens": min(self.max_tokens, remaining),
                                "temperature": self.temperature,
                                "top_p": 1.0,
                                "top_k": -1,
                                "logprobs": 0,
                                "stop_token_ids": [self.tokenizer.eos_token_id],
                                "skip_special_tokens": False,
                            },
                        },
                    )
                    response.raise_for_status()
                    choices = response.json()["choices"]
                    if len(choices) != 1:
                        raise ValueError("Expected one completion per model request")
                    choice = choices[0]
                    ids, sampled_log_probs = completion_tokens(choice)
                    if len(ids) > min(self.max_tokens, remaining):
                        raise ValueError("Completion exceeds the requested token budget")
                    if choice["finish_reason"] not in {"stop", "length"}:
                        raise RuntimeError(f"Generation failed: {choice['finish_reason']}")
                    model_span.set_attribute("output_tokens", len(ids))
                token_ids.extend(ids)
                loss_mask.extend([True] * len(ids))
                log_probs.extend(sampled_log_probs)
                num_turns += 1
                text = await asyncio.to_thread(
                    self.tokenizer.decode, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                completion = completion_text(text)
                if choice["finish_reason"] == "length":
                    stop_condition = "context_length" if len(token_ids) == self.context_length else "max_tokens"
                    break
                pending = []
                try:
                    calls = parse_tool_calls(text)
                except ValueError as error:
                    pending.append(json.dumps({"error": str(error)}))
                else:
                    if not calls:
                        stop_condition = "agent_completed"
                        break
                    for call in calls:
                        with span("actor.rollout.step", {"tool": call.name}):
                            try:
                                result = await environment.step(call)
                            except ToolCallError as error:
                                pending.append(json.dumps({"error": str(error)}))
                                continue
                        pending.append(json.dumps(asdict(result)))
                        if result.done:
                            break
                if result.done:
                    stop_condition = f"{task.name}_done"
                    break
                if num_turns == self.max_turns:
                    break
                # Qwen3 groups tool results in one user message; sampled history stays untouched.
                responses = "\n".join(f"<tool_response>\n{content}\n</tool_response>" for content in pending)
                suffix = ("" if ids[-1] == self.tokenizer.eos_token_id else "<|im_end|>") + (
                    f"\n<|im_start|>user\n{responses}<|im_end|>\n<|im_start|>assistant\n"
                )
                context = await asyncio.to_thread(self.tokenizer.encode, suffix, add_special_tokens=False)
                if len(token_ids) + len(context) >= self.context_length:
                    stop_condition = "context_length"
                    break
                token_ids.extend(context)
                loss_mask.extend([False] * len(context))
                log_probs.extend([0.0] * len(context))
                pending = []
            if not any(loss_mask):
                raise ValueError("Rollout has no trainable tokens")
            interaction = await asyncio.to_thread(
                self.tokenizer.decode, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            if pending:
                interaction += "\n\nTool results without model tokens:\n" + "\n".join(pending)
            return Rollout(
                sequence=TrainingSequence(
                    token_ids=np.asarray(token_ids, dtype=np.int32),  # (T,)
                    loss_mask=np.asarray(loss_mask, dtype=np.bool_),  # (T,)
                    old_log_probs=np.asarray(log_probs, dtype=np.float32),  # (T,)
                ),
                reward=result.reward,
                advantage=0.0,
                completion=completion,
                interaction=interaction,
                num_turns=num_turns,
                num_input_tokens=len(token_ids) - sum(loss_mask),
                num_output_tokens=sum(loss_mask),
                rollout_seconds=time.monotonic() - started,
                stop_condition=stop_condition,
            )
