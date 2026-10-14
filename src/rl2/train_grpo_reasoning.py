"""Fine-tune Gemma 3 270M with GRPO and optional calculator access on Reasoning Gym."""

from __future__ import annotations

import argparse
import re
from collections import deque
from collections.abc import Sequence
from contextlib import closing
from dataclasses import asdict, field, replace
from functools import partial
from pathlib import Path
from time import monotonic
from typing import Annotated, Any, Literal, NamedTuple, Protocol

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import reasoning_gym
import yaml
from flax import serialization
from flax.training.train_state import TrainState
from gemma import gm
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.utils import extract_answer

from rl2.calculator import calculate
from rl2.configuration import load_settings, resolve_settings
from rl2.gemma3 import (
    Gemma3Carry,
    Gemma3LM,
    Gemma3Tokenizer,
    Parameters,
    load_gemma3_checkpoint,
    save_gemma3_checkpoint,
)
from rl2.jax_cache import configure_compilation_cache
from rl2.mlflow_logging import MLflowLogger
from rl2.utils import read_optional, write_bytes

type Entry = dict[str, Any]
type Metrics = dict[str, jax.Array]


class Tokenizer(Protocol):
    def bos_id(self) -> int: ...
    def eos_id(self) -> int: ...
    def stop_ids(self) -> tuple[int, ...]: ...
    def vocab_size(self) -> int: ...
    def encode(self, text: str, *, out_type: type[int], add_bos: bool, add_eos: bool) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class PromptExample:
    question: Annotated[str, Field(strict=True, pattern=r"\S")]
    completion: Annotated[str, Field(strict=True, pattern=r"\S")]

    def __post_init__(self) -> None:
        if extract_answer(self.completion) is None:
            raise ValueError("Each prompt example completion must contain <answer>...</answer>")


@dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class Config:
    checkpoint: str = "gs://gemma-data/checkpoints/gemma3-270m-it"
    tokenizer: str | None = None
    dtype: Literal["float32", "bfloat16"] = "float32"
    task: str = "leg_counting"
    task_config: dict[str, Any] = field(default_factory=dict)
    prompt_examples: list[PromptExample] = field(default_factory=list)
    seed: Annotated[int, Field(ge=0, lt=2**32)] = 1
    total_updates: Annotated[int, Field(gt=0)] = 1000
    num_tasks: Annotated[int, Field(gt=0)] = 2
    group_size: Annotated[int, Field(ge=2)] = 4
    num_minibatches: Annotated[int, Field(gt=0)] = 8
    update_epochs: Annotated[int, Field(gt=0)] = 1
    max_prompt_tokens: Annotated[int, Field(gt=0)] = 256
    max_new_tokens: Annotated[int, Field(gt=0)] = 128
    calculator_max_calls: Annotated[int, Field(ge=0)] = 0
    temperature: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1.0
    learning_rate: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1e-6
    clip_coef: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)] = 0.2
    beta: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.01
    duration_penalty_coef: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    max_grad_norm: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1.0
    ckpt_dir: str = "runs/grpo_reasoning/${run_id}"
    run_id: str | None = None
    log_interval: Annotated[int, Field(gt=0)] = 10
    log_completion_count: Annotated[int, Field(ge=2, le=4)] = 4
    checkpoint_interval_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 600.0

    def __post_init__(self) -> None:
        if self.num_tasks * self.group_size % self.num_minibatches:
            raise ValueError("num_minibatches must divide num_tasks * group_size")
        if self.max_prompt_tokens + self.max_new_tokens > 32768:
            raise ValueError("Prompt + completion budget must fit the 32768-token context")
        if {"seed", "size"} & self.task_config.keys():
            raise ValueError("task_config cannot override seed or size")
        if self.run_id is not None and (
            not self.run_id.strip() or self.run_id in (".", "..") or any(c in self.run_id for c in "/\\")
        ):
            raise ValueError("run_id must be a nonempty directory name without slashes or traversal")


def resolve_config(config: Config) -> Config:
    settings = resolve_settings(asdict(config), run_id_prefix="grpo_reasoning")
    return replace(config, run_id=settings["run_id"], ckpt_dir=settings["ckpt_dir"])


def load_config(path: str | Path) -> Config:
    return Config(**load_settings(path, asdict(Config()), run_id_prefix="grpo_reasoning"))


class GRPOBatch(NamedTuple):
    prompts: jax.Array  # [B,P], right padded; repeated within each task group.
    prompt_lengths: jax.Array  # [B]
    completions: jax.Array  # [B,C], model tokens and inserted tool results, then right padding.
    mask: jax.Array  # [B,C], model tokens including EOS; excludes tool results and padding.
    old_log_probs: jax.Array  # [B,C], sampled behavior policy.
    ref_log_probs: jax.Array  # [B,C], frozen initial policy (unused when beta=0).
    advantages: jax.Array  # [B], normalized before shuffling/minibatching.


def group_advantages(rewards: jax.Array) -> jax.Array:
    centered = rewards - rewards.mean(axis=1, keepdims=True)
    result = centered / (rewards.std(axis=1, keepdims=True) + 1e-4)
    return result


def policy_log_probs(logits: jax.Array, tokens: jax.Array, temperature: float) -> jax.Array:
    result = jnp.take_along_axis(jax.nn.log_softmax(logits / temperature), tokens[..., None], axis=-1)[..., 0]
    return result


def build_prompt(question: str, examples: Sequence[PromptExample] = (), calculator_max_calls: int = 0) -> str:
    """Use the same full prompt for tokenization and sample logging."""

    def format_question(text: str) -> str:
        # Remove only the known redundant leg-counting preamble. Other tasks may
        # contain meaningful layout, so preserve their internal whitespace.
        text = text.strip("\n")
        prefix = (
            "Your task is to count how many legs there are in total when given a list of animals.\n\n"
            "Now, how many legs are there in total if you have "
        )
        if text.startswith(prefix):
            return "How many legs are there in total if you have " + text[len(prefix) :]
        return text

    instruction = "Put only your final answer inside <answer>...</answer>. End your turn immediately after </answer>.\n"
    if calculator_max_calls:
        instruction += (
            f"You may use the calculator up to {calculator_max_calls} times per question. "
            "Write <calculator>expression</calculator>. Pause there without ending your turn.\n"
            "The trainer inserts <result>...</result>; do not write the result yourself. Then continue to your answer.\n"
            "Expressions support numbers, parentheses, and + - * / // % **.\n"
        )
    turns: list[str] = []
    for example in examples:
        content = (
            f"{instruction}\n{format_question(example.question)}" if instruction else format_question(example.question)
        )
        completion = example.completion.rstrip("\n")
        turns.append(f"<start_of_turn>user\n{content}<end_of_turn>\n")
        turns.append(f"<start_of_turn>model\n{completion}<end_of_turn>\n")
        instruction = ""
    content = f"{instruction}\n{format_question(question)}" if instruction else format_question(question)
    turns.append(f"<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n")
    return "".join(turns)


def format_group_samples(prompt: str, completions: Sequence[str], rewards: Sequence[float]) -> str:
    """Render one shared prompt and numbered completions as literal Markdown blocks."""

    def literal(text: str) -> str:
        return "    " + text.replace("\n", "\n    ")

    sections = [f"### Shared prompt\n\n{literal(prompt)}"]
    for index, (completion, reward) in enumerate(zip(completions, rewards, strict=True), start=1):
        sections.append(f"### Completion {index}\n\n{literal(completion)}\n\n**Reward:** {reward:.3f}")
    return "\n\n---\n\n".join(sections)


def encode_prompts(
    entries: list[Entry],
    tokenizer: Tokenizer,
    max_tokens: int,
    examples: Sequence[PromptExample] = (),
    calculator_max_calls: int = 0,
) -> tuple[jax.Array, jax.Array]:  # (N, P), (N,)
    prompts = np.zeros((len(entries), max_tokens), np.int32)  # (N, P)
    lengths = np.empty(len(entries), np.int32)  # (N,)
    for i, entry in enumerate(entries):
        text = build_prompt(entry["question"], examples, calculator_max_calls)
        ids = tokenizer.encode(text, out_type=int, add_bos=tokenizer.bos_id() >= 0, add_eos=False)
        if not ids or len(ids) > max_tokens:
            raise ValueError(f"Prompt has {len(ids)} tokens; expected 1..{max_tokens}. Increase max_prompt_tokens.")
        if any(token < 0 or token >= tokenizer.vocab_size() for token in ids):
            raise ValueError("Prompt token ID outside tokenizer vocabulary")
        prompts[i, : len(ids)] = ids
        lengths[i] = len(ids)
    return jnp.asarray(prompts), jnp.asarray(lengths)


class DecodeCarry(NamedTuple):
    cache: Gemma3Carry  # leaves: (B, ...)
    logits: jax.Array  # (B, V)
    active: jax.Array  # (B,)
    key: jax.Array  # ()


class DecodeOutput(NamedTuple):
    tokens: jax.Array  # (B,)
    log_probs: jax.Array  # (B,)
    mask: jax.Array  # (B,)


class Generation(NamedTuple):
    tokens: jax.Array  # (B, C) Full transcript, including inserted results.
    log_probs: jax.Array  # (B, C) Zero at tool results/padding.
    mask: jax.Array  # (B, C) Model-generated tokens only.
    key: jax.Array  # () PRNG key.


@partial(jax.jit, static_argnames=("model", "group_size"))
def prefill_generation(
    model: Gemma3LM,
    params: Parameters,
    prompts: jax.Array,  # (N, P)
    lengths: jax.Array,  # (N,)
    group_size: int,
) -> tuple[Gemma3Carry, jax.Array]:  # carry leaves: (B, ...), logits: (B, V); B=N*group_size
    carry, logits = model.prefill(params, prompts, lengths)  # logits: (N, P, V)
    logits = jnp.repeat(logits[jnp.arange(prompts.shape[0]), lengths - 1], group_size, axis=0)  # (B, V)

    def repeat_carry(value: jax.Array) -> jax.Array:  # (N, ...) -> (B, ...)
        return jnp.repeat(value, group_size, axis=0)  # (B, ...)

    carry = jax.tree.map(repeat_carry, carry)  # leaves: (B, ...)
    return carry, logits  # logits: (B, V)


@partial(jax.jit, static_argnames=("model", "group_size", "max_new_tokens", "temperature", "stop_token_ids"))
def generate(
    model: Gemma3LM,
    params: Parameters,
    prompts: jax.Array,  # (N, P)
    lengths: jax.Array,  # (N,)
    key: jax.Array,  # ()
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    stop_token_ids: tuple[int, ...],
) -> Generation:
    """Prefill unique prompts once; independently sample each repeated model state."""
    if prompts.shape[1] + max_new_tokens > model.cache_length:
        raise ValueError("Prompt + completion budget exceeds Gemma cache length")
    carry, logits = prefill_generation(model, params, prompts, lengths, group_size)  # logits: (B, V)
    batch_size = prompts.shape[0] * group_size

    def step(previous: DecodeCarry, _: None) -> tuple[DecodeCarry, DecodeOutput]:
        carry, logits, active, key = previous  # logits: (B, V); active: (B,); key: ()
        result = decode_step(
            model,
            params,
            carry,
            logits,
            jnp.full((batch_size,), -1, jnp.int32),
            active,
            key,
            temperature,
            stop_token_ids,
        )
        output = DecodeOutput(tokens=result.tokens, log_probs=result.log_probs, mask=active)
        active = active & ~jnp.isin(result.tokens, jnp.asarray(stop_token_ids))  # (B,)
        return DecodeCarry(cache=result.carry, logits=result.logits, active=active, key=result.key), output

    (_, _, _, key), (tokens, log_probs, mask) = jax.lax.scan(
        step,
        DecodeCarry(cache=carry, logits=logits, active=jnp.ones(batch_size, jnp.bool_), key=key),
        None,
        length=max_new_tokens,
    )
    tokens, log_probs, mask = jax.tree.map(jnp.transpose, (tokens, log_probs, mask))  # (B, C) each
    return Generation(tokens=tokens, log_probs=log_probs, mask=mask, key=key)


class DecodeStep(NamedTuple):
    carry: Gemma3Carry  # leaves: (B, ...)
    logits: jax.Array  # (B, V)
    tokens: jax.Array  # (B,)
    log_probs: jax.Array  # (B,)
    key: jax.Array  # ()


@partial(jax.jit, static_argnames=("model", "temperature", "stop_token_ids"))
def decode_step(
    model: Gemma3LM,
    params: Parameters,
    carry: Gemma3Carry,  # leaves: (B, ...)
    logits: jax.Array,  # (B, V)
    forced: jax.Array,  # (B,) Tool-result IDs; -1 requests sampling.
    active: jax.Array,  # (B,)
    key: jax.Array,  # ()
    temperature: float,
    stop_token_ids: tuple[int, ...],
) -> DecodeStep:
    key, sample_key = jax.random.split(key)  # (), ()

    def sample(_: None) -> tuple[jax.Array, jax.Array]:  # (B,), (B,)
        sampled = gm.text.RandomSampling(temperature=temperature).get_next_tokens(logits, sample_key)  # (B,)
        log_probs = policy_log_probs(logits[:, None], sampled[:, None], temperature)[:, 0]  # (B,)
        return sampled, log_probs  # (B,), (B,)

    def skip(_: None) -> tuple[jax.Array, jax.Array]:  # (B,), (B,)
        return jnp.zeros_like(forced), jnp.zeros_like(forced, dtype=jnp.float32)  # (B,), (B,)

    sampled, log_probs = jax.lax.cond(jnp.any(active & (forced < 0)), sample, skip, None)  # (B,), (B,)
    tokens = jnp.where(active, jnp.where(forced >= 0, forced, sampled), 0)  # (B,)
    log_probs = jnp.where(active & (forced < 0), log_probs, 0.0)  # (B,)
    active = active & ~jnp.isin(tokens, jnp.asarray(stop_token_ids))  # (B,)
    carry, logits = model.step(params, tokens, active, carry)  # logits: (B, V)
    return DecodeStep(carry=carry, logits=logits, tokens=tokens, log_probs=log_probs, key=key)


def generate_with_calculator(
    model: Gemma3LM,
    params: Parameters,
    prompts: jax.Array,  # (N, P)
    lengths: jax.Array,  # (N,)
    key: jax.Array,  # ()
    *,
    tokenizer: Tokenizer,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    max_calls: int,
) -> Generation:
    """Decode with a KV cache, pausing for host-side arithmetic at complete calculator tags."""
    if prompts.shape[1] + max_new_tokens > model.cache_length:
        raise ValueError("Prompt + completion budget exceeds Gemma cache length")
    carry, logits = prefill_generation(model, params, prompts, lengths, group_size)  # logits: (B, V)
    batch_size = prompts.shape[0] * group_size
    outputs: list[DecodeOutput] = []  # Each leaf: (B,)
    active = [True] * batch_size
    pending: list[deque[int]] = [deque() for _ in active]
    segments: list[list[int]] = [[] for _ in active]
    calls = [0] * batch_size
    stop_ids = tokenizer.stop_ids()
    for position in range(max_new_tokens):
        forced_ids = [queue.popleft() if queue else -1 for queue in pending]
        is_sampled = [enabled and token < 0 for enabled, token in zip(active, forced_ids, strict=True)]
        forced = jnp.asarray(forced_ids, jnp.int32)  # (B,)
        sampled = jnp.asarray(is_sampled)  # (B,)
        step = decode_step(model, params, carry, logits, forced, jnp.asarray(active), key, temperature, stop_ids)
        carry, logits, key = step.carry, step.logits, step.key  # logits: (B, V); key: ()
        outputs.append(DecodeOutput(tokens=step.tokens, log_probs=step.log_probs, mask=sampled))
        if not any(is_sampled):
            continue
        host_tokens = jax.device_get(step.tokens)  # (B,)
        for row, (token, sampled_row) in enumerate(zip(host_tokens, is_sampled, strict=True)):
            if not sampled_row:
                continue
            if token in stop_ids:
                active[row] = False
                continue
            segments[row].append(int(token))
            match = re.search(r"<calculator>(.*?)</calculator>", tokenizer.decode(segments[row]), re.DOTALL)
            if match is None:
                continue
            result = calculate(match[1]) if calls[row] < max_calls else "Error: calculator call limit reached"
            calls[row] += 1
            result_ids = tokenizer.encode(f"\n<result>{result}</result>\n", out_type=int, add_bos=False, add_eos=False)
            # Insert only complete results, leaving at least one slot for the model to continue.
            if len(result_ids) >= max_new_tokens - position - 1:
                active[row] = False
                continue
            pending[row].extend(result_ids)
            segments[row].clear()
        if not any(active):
            break
    padding = DecodeOutput(
        tokens=jnp.zeros(batch_size, jnp.int32),  # (B,)
        log_probs=jnp.zeros(batch_size, jnp.float32),  # (B,)
        mask=jnp.zeros(batch_size, jnp.bool_),  # (B,)
    )
    outputs.extend([padding] * (max_new_tokens - len(outputs)))

    def stack(*values: jax.Array) -> jax.Array:  # (B,) each -> (B, C)
        return jnp.stack(values, axis=1)  # (B, C)

    tokens, log_probs, mask = jax.tree.map(stack, *outputs)  # (B, C) each
    return Generation(tokens=tokens, log_probs=log_probs, mask=mask, key=key)


def transcript_mask(mask: jax.Array) -> jax.Array:  # (B, C) -> (B, C)
    """Include tool results before the last model token, excluding trailing padding."""
    positions = jnp.arange(mask.shape[1])[None, :]  # (1, C)
    last = jnp.max(jnp.where(mask, positions, -1), axis=1, keepdims=True)  # (B, 1)
    return positions <= last  # (B, C)


@partial(jax.jit, static_argnames=("model", "temperature"))
def completion_log_probs(
    model: Gemma3LM,
    params: Parameters,
    batch: GRPOBatch,
    temperature: float,
) -> jax.Array:  # (B, C)
    """Pack prompt+completion without padding gaps; backpropagate through the prompt."""
    b, p = batch.prompts.shape
    c = batch.completions.shape[1]
    positions = jnp.arange(p + c - 1)[None, :]  # (1, P+C-1)
    completion_positions = positions - batch.prompt_lengths[:, None]  # (B, P+C-1)
    prompts = jnp.take_along_axis(batch.prompts, jnp.minimum(positions, p - 1), axis=1)  # (B, P+C-1)
    completions = jnp.take_along_axis(batch.completions, jnp.clip(completion_positions, 0, c - 1), axis=1)  # (B, P+C-1)
    tokens = jnp.where(completion_positions < 0, prompts, completions)  # (B, P+C-1)
    lengths = batch.prompt_lengths + transcript_mask(batch.mask).sum(axis=1, dtype=jnp.int32) - 1  # (B,)
    # Rematerialize model activations during backward instead of retaining every layer.
    _, logits = jax.checkpoint(model.apply)({"params": params}, tokens, lengths)  # (B, P+C-1, V)
    indices = batch.prompt_lengths[:, None] - 1 + jnp.arange(c)[None, :]  # (B, C)
    logits = logits[jnp.arange(b)[:, None], indices]  # (B, C, V)
    result = policy_log_probs(logits, batch.completions, temperature)  # (B, C)
    return jnp.where(batch.mask, result, 0.0)  # (B, C)


def objective(log_probs: jax.Array, batch: GRPOBatch, clip_coef: float, beta: float) -> tuple[jax.Array, Metrics]:
    """Clipped token ratios and sampled reference KL, averaged equally per completion."""

    def average(values: jax.Array) -> jax.Array:
        return (jnp.where(batch.mask, values, 0).sum(axis=1) / jnp.maximum(batch.mask.sum(axis=1), 1)).mean()

    log_ratio = jnp.where(batch.mask, log_probs - jax.lax.stop_gradient(batch.old_log_probs), 0.0)
    ratio = jnp.exp(log_ratio)
    advantages = jax.lax.stop_gradient(batch.advantages)[:, None]
    policy_loss = -average(jnp.minimum(ratio * advantages, jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef) * advantages))
    if beta:
        ref_ratio = jnp.where(batch.mask, jax.lax.stop_gradient(batch.ref_log_probs) - log_probs, 0.0)
        ref_kl = average(jnp.expm1(ref_ratio) - ref_ratio)
    else:
        ref_kl = jnp.asarray(0.0, jnp.float32)
    loss = policy_loss + beta * ref_kl
    metrics = {
        "losses/total": loss,
        "losses/policy": policy_loss,
        "policy/ref_kl": ref_kl,
        "policy/approx_kl": average(jnp.expm1(log_ratio) - log_ratio),
        "policy/clip_fraction": average((jnp.abs(ratio - 1) > clip_coef).astype(jnp.float32)),
    }
    return loss, metrics


def create_state(model: Gemma3LM, params: Parameters, config: Config) -> TrainState:
    return TrainState.create(
        apply_fn=model.apply,
        params=jax.device_put(params),
        tx=optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate)),
    )


@partial(jax.jit, static_argnames=("model", "temperature", "clip_coef", "beta"))
def update(
    state: TrainState,
    model: Gemma3LM,
    batch: GRPOBatch,
    *,
    temperature: float,
    clip_coef: float,
    beta: float,
) -> tuple[TrainState, Metrics]:
    def loss_fn(params: Parameters) -> tuple[jax.Array, Metrics]:
        return objective(completion_log_probs(model, params, batch, temperature), batch, clip_coef, beta)

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    metrics["policy/grad_norm"] = optax.global_norm(grads)
    return state.apply_gradients(grads=grads), metrics


def score_completions(
    dataset: ProceduralDataset,
    entries: list[Entry],
    texts: list[str],
    group_size: int,
) -> jax.Array:
    """Score the last answer tag; missing tags receive zero reward."""
    if len(texts) != len(entries) * group_size:
        raise ValueError("Expected group_size completions per entry")
    rewards = np.zeros(len(texts), np.float32)
    for i, text in enumerate(texts):
        answer = extract_answer(text)
        if answer is not None:
            score = dataset.score_answer(answer=answer, entry=entries[i // group_size])
            if not 0 <= score <= 1:
                raise ValueError(f"Task scorer returned invalid reward: {score}")
            rewards[i] = score
    return jnp.asarray(rewards)


def apply_duration_penalty(
    task_rewards: jax.Array,
    mask: jax.Array,
    coefficient: float,
) -> tuple[jax.Array, jax.Array]:
    """Deduct a token-budget-normalized cost, counting EOS but excluding padding."""
    chex.assert_scalar_positive(mask.shape[1])
    penalties = coefficient * mask.sum(axis=1, dtype=jnp.float32) / mask.shape[1]
    rewards = task_rewards - penalties
    return rewards, penalties


def collect_rollout(
    state: TrainState,
    model: Gemma3LM,
    ref_params: Parameters,
    dataset: ProceduralDataset,
    tokenizer: Tokenizer,
    key: jax.Array,
    iteration: int,
    config: Config,
) -> tuple[GRPOBatch, dict[str, float], str, jax.Array]:
    entries = [dataset[iteration * config.num_tasks + i] for i in range(config.num_tasks)]
    prompts, lengths = encode_prompts(
        entries, tokenizer, config.max_prompt_tokens, config.prompt_examples, config.calculator_max_calls
    )  # (N, P), (N,)
    generation_start = monotonic()
    if config.calculator_max_calls:
        tokens, log_probs, mask, key = generate_with_calculator(
            model,
            state.params,
            prompts,
            lengths,
            key,
            tokenizer=tokenizer,
            group_size=config.group_size,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            max_calls=config.calculator_max_calls,
        )  # (B, C), (B, C), (B, C), ()
    else:
        tokens, log_probs, mask, key = generate(
            model,
            state.params,
            prompts,
            lengths,
            key,
            group_size=config.group_size,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            stop_token_ids=tokenizer.stop_ids(),
        )  # (B, C), (B, C), (B, C), ()
    host_tokens, host_mask = jax.device_get((tokens, transcript_mask(mask)))  # (B, C), (B, C)
    generation_seconds = monotonic() - generation_start
    decoding_start = monotonic()
    texts = [
        tokenizer.decode(ids[valid & ~np.isin(ids, tokenizer.stop_ids())].tolist())
        for ids, valid in zip(host_tokens, host_mask, strict=True)
    ]
    decoding_seconds = monotonic() - decoding_start
    scoring_start = monotonic()
    task_rewards = score_completions(dataset, entries, texts, config.group_size)
    scoring_seconds = monotonic() - scoring_start
    processing_start = monotonic()
    rewards, duration_penalties = apply_duration_penalty(task_rewards, mask, config.duration_penalty_coef)
    grouped = rewards.reshape(config.num_tasks, config.group_size)
    batch = GRPOBatch(
        prompts=jnp.repeat(prompts, config.group_size, axis=0),  # (B, P)
        prompt_lengths=jnp.repeat(lengths, config.group_size),  # (B,)
        completions=tokens,  # (B, C)
        mask=mask,  # (B, C)
        old_log_probs=log_probs,  # (B, C)
        ref_log_probs=jnp.zeros_like(log_probs),  # (B, C)
        advantages=group_advantages(grouped).reshape(-1),  # (B,)
    )
    if config.beta:
        # Bound reference replay memory to the same batch size as optimization.
        ref = []
        for indices in np.split(np.arange(tokens.shape[0]), config.num_minibatches):
            minibatch = GRPOBatch(*(x[indices] for x in batch))
            ref.append(completion_log_probs(model, ref_params, minibatch, config.temperature))
        batch = batch._replace(ref_log_probs=jnp.concatenate(ref))
    diagnostics = {
        "charts/reward_mean": float(rewards.mean()),
        "charts/task_reward_mean": float(task_rewards.mean()),
        "charts/duration_penalty_mean": float(duration_penalties.mean()),
        "charts/success_rate": float((task_rewards == 1).mean()),
        "charts/informative_group_fraction": float((jnp.ptp(grouped, axis=1) > 0).mean()),
        "charts/completion_length_mean": float(mask.sum(axis=1).mean()),
        "charts/truncation_rate": float(
            (~jnp.any(mask & jnp.isin(tokens, jnp.asarray(tokenizer.stop_ids())), axis=1)).mean()
        ),
    }
    processing_seconds = monotonic() - processing_start
    diagnostics.update(
        {
            "time/generation_seconds": generation_seconds,
            "time/decoding_seconds": decoding_seconds,
            "time/scoring_seconds": scoring_seconds,
            "time/rollout_processing_seconds": processing_seconds,
            "time/jax_seconds": generation_seconds + processing_seconds,
        }
    )
    count = min(config.log_completion_count, config.group_size)
    group_rewards = np.asarray(rewards[: config.group_size])
    # Stable sorting preserves sampling order for ties; only the displayed group changes.
    indices = np.argsort(-group_rewards, kind="stable")[:count]
    samples = format_group_samples(
        build_prompt(entries[0]["question"], config.prompt_examples, config.calculator_max_calls),
        [texts[index] for index in indices],
        group_rewards[indices].tolist(),
    )
    return batch, diagnostics, samples, key


class TrainStepResult(NamedTuple):
    state: TrainState  # Parameter/optimizer pytrees; scalar step.
    key: jax.Array  # () typed PRNG key
    metrics: list[Metrics]  # One dict of scalar arrays per minibatch update.
    diagnostics: dict[str, float]
    samples: str


def train_step(
    state: TrainState,
    model: Gemma3LM,
    ref_params: Parameters,  # Model parameter pytree.
    dataset: ProceduralDataset,
    tokenizer: Tokenizer,
    key: jax.Array,  # () typed PRNG key
    iteration: int,
    config: Config,
    *,
    log_startup: bool = False,
) -> TrainStepResult:
    """Collect and score one rollout, then dispatch its minibatch updates without waiting."""
    if log_startup:
        print("[startup] Collecting first rollout; generation may trigger JAX compilation", flush=True)
    batch, diagnostics, samples, key = collect_rollout(
        state, model, ref_params, dataset, tokenizer, key, iteration, config
    )  # batch leaves: (B, ...); key: (); B = num_tasks * group_size
    if log_startup:
        print("[startup] First rollout collected; starting optimization (may trigger JAX compilation)", flush=True)
    metrics: list[Metrics] = []  # Scalar leaves per update.
    batch_size = config.num_tasks * config.group_size
    update_start = monotonic()
    for _ in range(config.update_epochs):
        key, shuffle_key = jax.random.split(key)  # each () typed PRNG key
        indices = jax.random.permutation(shuffle_key, batch_size).reshape(config.num_minibatches, -1)  # (M, B/M)
        for minibatch_indices in indices:  # (B/M,)
            minibatch = jax.tree.map(partial(jnp.take, indices=minibatch_indices, axis=0), batch)  # leaves: (B/M, ...)
            state, metric = update(
                state,
                model,
                minibatch,
                temperature=config.temperature,
                clip_coef=config.clip_coef,
                beta=config.beta,
            )  # metric leaves: ()
            metrics.append(metric)
    diagnostics["time/update_dispatch_seconds"] = monotonic() - update_start
    diagnostics["time/jax_seconds"] += diagnostics["time/update_dispatch_seconds"]
    return TrainStepResult(state=state, key=key, metrics=metrics, diagnostics=diagnostics, samples=samples)


class TrainingProgress(NamedTuple):
    state: TrainState  # Model/optimizer pytree; step: ().
    key: jax.Array  # () typed PRNG key
    iteration: int
    prompts_seen: int


def checkpoint_manager(run_dir: str) -> ocp.CheckpointManager:
    directory = f"{run_dir.rstrip('/')}/checkpoints"
    if not directory.startswith("gs://"):
        directory = str(Path(directory).resolve())
    return ocp.CheckpointManager(
        directory,
        item_handlers={"state": ocp.StandardCheckpointHandler(), "metadata": ocp.JsonCheckpointHandler()},
        options=ocp.CheckpointManagerOptions(max_to_keep=2),
    )


def save_training_checkpoint(
    manager: ocp.CheckpointManager,
    state: TrainState,
    key: jax.Array,  # () typed PRNG key
    iteration: int,
    config: Config,
    *,
    prompts_seen: int,
) -> None:
    manager.save(
        iteration,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave(
                {"train_state": serialization.to_state_dict(state), "key": jax.random.key_data(key)}
            ),
            metadata=ocp.args.JsonSave(
                {
                    "version": 1,
                    "iteration": iteration,
                    "prompts_seen": prompts_seen,
                    "key_impl": str(jax.random.key_impl(key)),
                    "config": asdict(config),  # Provenance only.
                }
            ),
        ),
        force=True,
    )
    manager.wait_until_finished()


def restore_training_checkpoint(manager: ocp.CheckpointManager, state: TrainState) -> TrainingProgress | None:
    """Restore the latest checkpoint using the current model/optimizer functions."""
    iteration = manager.latest_step()
    if iteration is None:
        return None
    metadata = manager.restore(iteration, args=ocp.args.Composite(metadata=ocp.args.JsonRestore())).metadata
    if metadata["version"] != 1:
        raise ValueError("Unsupported reasoning GRPO checkpoint version")
    if type(metadata["iteration"]) is not int or metadata["iteration"] != iteration or iteration < 0:
        raise ValueError("Invalid checkpoint iteration")
    prompts_seen = metadata["prompts_seen"]
    if type(prompts_seen) is not int or prompts_seen < 0:
        raise ValueError("Invalid checkpoint prompts_seen")
    target = serialization.to_state_dict(state)
    saved = manager.metadata(iteration).item_metadata["state"].tree["train_state"]
    try:
        # A restore target can hide extra keys or cast stored dtypes; check metadata first.
        chex.assert_trees_all_equal_shapes_and_dtypes(
            (target["params"], target["opt_state"]), (saved["params"], saved["opt_state"])
        )
    except (AssertionError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            "Checkpoint model/optimizer structure, shapes, or dtypes are incompatible with the current state"
        ) from error
    key = jax.random.key(0, impl=metadata["key_impl"])  # () typed PRNG key
    restored = manager.restore(
        iteration,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore({"train_state": target, "key": jax.random.key_data(key)})
        ),
    ).state
    return TrainingProgress(
        state=serialization.from_state_dict(state, restored["train_state"]),
        key=jax.random.wrap_key_data(restored["key"], impl=metadata["key_impl"]),  # () typed PRNG key
        iteration=iteration,
        prompts_seen=prompts_seen,
    )


def restore_legacy_training_checkpoint(data: bytes, state: TrainState) -> TrainingProgress:
    """Restore compatible state while retaining the current model/optimizer functions."""
    payload = serialization.msgpack_restore(data)
    if payload["version"] != 1:
        raise ValueError("Unsupported reasoning GRPO checkpoint version")
    try:
        # Compare serialized trees before restoration: Flax can ignore extra dict
        # keys, so checking only the reconstructed state would miss those changes.
        chex.assert_trees_all_equal_shapes_and_dtypes(
            (serialization.to_state_dict(state.params), serialization.to_state_dict(state.opt_state)),
            (payload["state"]["params"], payload["state"]["opt_state"]),
        )
        restored = serialization.from_state_dict(state, payload["state"])
    except (AssertionError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            "Checkpoint model/optimizer structure, shapes, or dtypes are incompatible with the current state"
        ) from error
    iteration = payload["iteration"]
    if type(iteration) is not int or iteration < 0:
        raise ValueError("Invalid checkpoint iteration")
    if "prompts_seen" in payload:
        prompts_seen = payload["prompts_seen"]
    else:
        # Legacy checkpoints did not store this counter; use their own task batch size.
        num_tasks = payload["config"]["num_tasks"]
        if type(num_tasks) is not int or num_tasks < 1:
            raise ValueError("Invalid checkpoint num_tasks")
        prompts_seen = iteration * num_tasks
    if type(prompts_seen) is not int or prompts_seen < 0:
        raise ValueError("Invalid checkpoint prompts_seen")
    key = jax.random.wrap_key_data(jnp.asarray(payload["key"]), impl=payload["key_impl"])  # () typed PRNG key
    return TrainingProgress(state=restored, key=key, iteration=iteration, prompts_seen=prompts_seen)


def train(config: Config) -> TrainState:
    init_start = monotonic()

    def log_init(message: str) -> None:
        print(f"[init +{monotonic() - init_start:.1f}s] {message}", flush=True)

    log_init("Configuring compilation cache")
    configure_compilation_cache()
    log_init("Initializing JAX devices")
    devices = jax.devices()
    log_init(f"JAX devices ready: {devices}")
    log_init(f"Creating {config.task} dataset ({config.total_updates * config.num_tasks:,} questions)")
    dataset = reasoning_gym.create_dataset(
        config.task,
        seed=config.seed,
        size=config.total_updates * config.num_tasks,
        **config.task_config,
    )
    log_init(f"Loading tokenizer: {config.tokenizer or 'Gemma 3 default'}")
    tokenizer = Gemma3Tokenizer(config.tokenizer)
    log_init(f"Tokenizer ready: vocabulary size {tokenizer.vocab_size():,}")
    log_init(f"Loading pretrained checkpoint: {config.checkpoint} (dtype={config.dtype})")
    model, variables = load_gemma3_checkpoint(
        config.checkpoint,
        cache_length=config.max_prompt_tokens + config.max_new_tokens,
        dtype=config.dtype,
    )
    parameter_count = sum(leaf.size for leaf in jax.tree.leaves(variables["params"]))
    log_init(f"Pretrained checkpoint loaded: {parameter_count:,} parameters")
    if tokenizer.vocab_size() != model.vocab_size:
        raise ValueError("Tokenizer vocabulary size does not match checkpoint vocabulary size")
    if not 0 <= tokenizer.eos_id() < model.vocab_size:
        raise ValueError("Tokenizer must define an EOS token within the model vocabulary")
    log_init("Transferring weights to device and initializing Adam optimizer")
    ref_params = jax.device_put(variables["params"])
    state = create_state(model, ref_params, config)
    jax.block_until_ready(state)
    log_init("Model and optimizer ready")
    if not config.beta:
        ref_params = {}
    del variables
    config = resolve_config(config)
    run_dir = config.ckpt_dir.rstrip("/")
    log_init(f"Checking for saved training state: {run_dir}/checkpoints")
    with closing(checkpoint_manager(run_dir)) as manager:
        saved = restore_training_checkpoint(manager, state)
        if saved is None:
            legacy = read_optional(f"{run_dir}/checkpoint.msgpack")
            if legacy is not None:
                saved = restore_legacy_training_checkpoint(legacy, state)
            del legacy
        key = jax.random.key(config.seed)  # () typed PRNG key
        start_iteration, prompts_seen = 0, 0
        if saved is not None:
            state, key, start_iteration, prompts_seen = saved
            log_init(f"Restored training checkpoint at rollout {start_iteration}, optimizer step {int(state.step)}")
        else:
            log_init("No training checkpoint found; starting from pretrained weights")
        log_init(f"Writing run configuration: {run_dir}/config.yaml")
        write_bytes(f"{run_dir}/config.yaml", yaml.safe_dump(asdict(config)).encode())
        last_checkpoint = monotonic()
        first_log_iteration = (start_iteration // config.log_interval + 1) * config.log_interval
        log_init(f"Opening MLflow logger: {config.run_id}")
        with MLflowLogger(experiment="grpo_reasoning", run_name=config.run_id) as writer:
            del saved
            print(f"Run: {run_dir}; devices: {devices}; starting at rollout {start_iteration}", flush=True)
            print(f"MLflow run: {writer.run_id}", flush=True)
            writer.log_text(yaml.safe_dump(asdict(config)), f"config/{prompts_seen}.yaml")
            writer.log_params(asdict(config))
            writer.log_text(str(devices), f"devices/{prompts_seen}.txt")
            writer.log_metrics({"model/params_millions": parameter_count / 1_000_000}, prompts_seen)
            writer.log_text(str(parameter_count), f"model/parameter_count/{prompts_seen}.txt")
            log_init("Initialization complete")
            window_timings: dict[str, float] = {}
            window_rollouts = 0
            window_start = monotonic()
            for iteration in range(start_iteration, config.total_updates):
                result = train_step(
                    state,
                    model,
                    ref_params,
                    dataset,
                    tokenizer,
                    key,
                    iteration,
                    config,
                    log_startup=iteration == start_iteration,
                )
                state, key = result.state, result.key  # key: () typed PRNG key
                metrics = result.metrics  # Scalar leaves per update.
                diagnostics, samples = result.diagnostics, result.samples
                window_rollouts += 1
                for name, value in diagnostics.items():
                    if name.startswith("time/"):
                        window_timings[name] = window_timings.get(name, 0.0) + value
                completed = iteration + 1
                prompts_seen += config.num_tasks
                summary = f"rollout_batches={completed} reward={diagnostics['charts/reward_mean']:.3f}"
                log_interval_reached = completed % config.log_interval == 0 or completed == config.total_updates
                if log_interval_reached:
                    # Updates stay asynchronous inside the window. Charge the final
                    # outstanding work here; generation already waits for CPU scoring.
                    wait_start = monotonic()
                    jax.block_until_ready((state, metrics))
                    wait_seconds = monotonic() - wait_start
                    window_timings["time/jax_wait_seconds"] = wait_seconds
                    window_timings["time/jax_seconds"] += wait_seconds
                    window_timings["time/window_seconds"] = monotonic() - window_start
                    window_timings["time/iteration_seconds"] = window_timings["time/window_seconds"] / window_rollouts
                    means = {name: float(np.mean([m[name] for m in jax.device_get(metrics)])) for name in metrics[0]}
                    writer.log_metrics(
                        {
                            **diagnostics,
                            **means,
                            **window_timings,
                            "time/window_rollouts": window_rollouts,
                        },
                        prompts_seen,
                    )
                    writer.log_text(samples, f"samples/completions/{prompts_seen}.md")
                    summary += (
                        f" loss={means['losses/total']:.4f} window_rollouts={window_rollouts}"
                        f" jax={window_timings['time/jax_seconds']:.3f}s"
                        f" scoring={window_timings['time/scoring_seconds']:.3f}s"
                        f" decode={window_timings['time/decoding_seconds']:.3f}s"
                        f" window={window_timings['time/window_seconds']:.3f}s"
                    )
                    window_timings = {}
                    window_rollouts = 0
                    window_start = monotonic()
                if completed <= first_log_iteration or log_interval_reached:
                    print(summary, flush=True)
                if (
                    monotonic() - last_checkpoint >= config.checkpoint_interval_seconds
                    or completed == config.total_updates
                ):
                    save_training_checkpoint(manager, state, key, completed, config, prompts_seen=prompts_seen)
                    last_checkpoint = monotonic()
                    print(f"Saved {run_dir}/checkpoints at rollout {completed}", flush=True)
            # Export in the source model format; inference exports are local-only.
            if not run_dir.startswith("gs://"):
                export = Path(run_dir) / f"model-{max(start_iteration, config.total_updates)}"
                if not export.exists():
                    save_gemma3_checkpoint(export, jax.device_get(state.params))
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grpo_reasoning/gemma3.yaml", help="Path to YAML config")
    args = parser.parse_args()
    print(f"[init] Loading configuration: {args.config}", flush=True)
    train(load_config(args.config))


if __name__ == "__main__":
    main()
