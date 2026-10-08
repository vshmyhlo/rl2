"""Fine-tune GDN2 or Gemma 3 270M with GRPO on Reasoning Gym.

Run with ``uv run --extra gdn2 python -m rl2.train_grpo_reasoning --config
configs/grpo_reasoning.yaml``. Sampling and replay use the same full-vocabulary,
temperature-scaled policy. Only completion tokens (including EOS) incur loss.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import monotonic
from typing import Any, Literal, NamedTuple, Protocol

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reasoning_gym
import yaml
from flax import serialization
from flax.training.train_state import TrainState
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.utils import extract_answer
from tensorboardX import SummaryWriter

from rl2.gdn2.checkpoints import Parameters, load_checkpoint, save_checkpoint
from rl2.gdn2.generate import DEFAULT_CHECKPOINT, load_tokenizer
from rl2.gdn2.model import GatedDeltaNet2Backend, GatedDeltaNet2LM, GatedDeltaNet2StackCarry
from rl2.gemma3 import Gemma3Carry, Gemma3LM, Gemma3Tokenizer, load_gemma3_checkpoint, save_gemma3_checkpoint
from rl2.jax_cache import configure_compilation_cache
from rl2.shape_checker import ShapeChecker
from rl2.utils import read_bytes, read_optional, write_bytes

type Entry = dict[str, Any]
type Metrics = dict[str, jax.Array]
type LanguageModel = GatedDeltaNet2LM | Gemma3LM


class Tokenizer(Protocol):
    def bos_id(self) -> int: ...
    def eos_id(self) -> int: ...
    def vocab_size(self) -> int: ...
    def encode(self, text: str, *, out_type: type[int], add_bos: bool, add_eos: bool) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


@dataclass(frozen=True)
class PromptExample:
    question: str
    completion: str


@dataclass(frozen=True)
class Config:
    model_type: Literal["gdn2", "gemma3"] = "gdn2"
    checkpoint: str = str(DEFAULT_CHECKPOINT)
    tokenizer: str | None = None
    backend: GatedDeltaNet2Backend = "jax"
    dtype: Literal["float32", "bfloat16"] = "float32"
    task: str = "leg_counting"
    task_config: dict[str, Any] = field(default_factory=dict)
    prompt_examples: list[PromptExample] = field(default_factory=list)
    seed: int = 1
    total_updates: int = 1000
    num_tasks: int = 2
    group_size: int = 4
    num_minibatches: int = 8
    update_epochs: int = 1
    max_prompt_tokens: int = 256
    max_new_tokens: int = 128
    temperature: float = 1.0
    learning_rate: float = 1e-6
    clip_coef: float = 0.2
    beta: float = 0.01
    duration_penalty_coef: float = 0.0
    max_grad_norm: float = 1.0
    log_dir: str = "runs"
    run_id: str | None = None
    log_interval: int = 10
    log_completion_count: int = 4
    checkpoint_interval_seconds: float = 600.0

    def __post_init__(self) -> None:
        if self.model_type not in ("gdn2", "gemma3"):
            raise ValueError("model_type must be gdn2/gemma3")
        if self.model_type == "gemma3" and self.backend != "jax":
            raise ValueError("Gemma 3 requires backend: jax (also on GPUs)")
        for name in (
            "total_updates",
            "num_tasks",
            "group_size",
            "num_minibatches",
            "update_epochs",
            "max_prompt_tokens",
            "max_new_tokens",
            "log_interval",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.group_size < 2:
            raise ValueError("GRPO requires group_size >= 2")
        if not 2 <= self.log_completion_count <= 4:
            raise ValueError("log_completion_count must be between 2 and 4")
        if self.num_tasks * self.group_size % self.num_minibatches:
            raise ValueError("num_minibatches must divide num_tasks * group_size")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be between 0 and 2**32 - 1")
        for name in ("temperature", "learning_rate", "max_grad_norm", "checkpoint_interval_seconds"):
            if not 0 < getattr(self, name) < math.inf:
                raise ValueError(f"{name} must be positive and finite")
        if not 0 < self.clip_coef < 1:
            raise ValueError("clip_coef must be between 0 and 1")
        if not 0 <= self.beta < math.inf:
            raise ValueError("beta must be finite and nonnegative")
        if not 0 <= self.duration_penalty_coef < math.inf:
            raise ValueError("duration_penalty_coef must be finite and nonnegative")
        if self.backend not in ("jax", "triton") or self.dtype not in ("float32", "bfloat16"):
            raise ValueError("backend must be jax/triton and dtype must be float32/bfloat16")
        if {"seed", "size"} & self.task_config.keys():
            raise ValueError("task_config cannot override seed or size")
        if self.run_id is not None and (
            not self.run_id.strip() or self.run_id in (".", "..") or any(c in self.run_id for c in "/\\")
        ):
            raise ValueError("run_id must be a nonempty directory name without slashes or traversal")


def load_config(path: str | Path) -> Config:
    """Validate untyped YAML values at the configuration boundary."""
    values = yaml.safe_load(read_bytes(str(path)))
    if not isinstance(values, dict):
        raise TypeError("Config must be a YAML mapping")
    defaults = asdict(Config())
    for name, value in values.items():
        if name not in defaults:
            raise ValueError(f"Unknown config option: {name}")
        default = defaults[name]
        if name in ("tokenizer", "run_id"):
            valid = value is None or isinstance(value, str)
        elif isinstance(default, float):
            valid = type(value) in (int, float)
        else:
            valid = type(value) is type(default)
        if not valid:
            raise ValueError(f"Invalid type for {name}")
    if "prompt_examples" in values:
        examples = values["prompt_examples"]
        for example in examples:
            if (
                not isinstance(example, dict)
                or set(example) != {"question", "completion"}
                or any(not isinstance(text, str) or not text.strip() for text in example.values())
            ):
                raise ValueError("Each prompt example must contain nonempty question and completion strings")
            if extract_answer(example["completion"]) is None:
                raise ValueError("Each prompt example completion must contain <answer>...</answer>")
        values["prompt_examples"] = [PromptExample(**example) for example in examples]
    return Config(**values)


class GRPOBatch(NamedTuple):
    prompts: jax.Array  # [B,P], right padded; repeated within each task group.
    prompt_lengths: jax.Array  # [B]
    completions: jax.Array  # [B,C], right padded after EOS.
    mask: jax.Array  # [B,C], includes the first EOS.
    old_log_probs: jax.Array  # [B,C], sampled behavior policy.
    ref_log_probs: jax.Array  # [B,C], frozen initial policy (unused when beta=0).
    advantages: jax.Array  # [B], normalized before shuffling/minibatching.

    def validate(self) -> None:
        sc = ShapeChecker()
        sc.check(self.prompts, "BP", jnp.int32)
        sc.check(self.prompt_lengths, "B", jnp.int32)
        sc.check(self.completions, "BC", jnp.int32)
        sc.check(self.mask, "BC", jnp.bool_)
        sc.check([self.old_log_probs, self.ref_log_probs], "BC", jnp.float32)
        sc.check(self.advantages, "B", jnp.float32)


def group_advantages(rewards: jax.Array) -> jax.Array:
    sc = ShapeChecker()
    sc.check(rewards, "NG", jnp.float32)
    centered = rewards - rewards.mean(axis=1, keepdims=True)
    result = centered / (rewards.std(axis=1, keepdims=True) + 1e-4)
    sc.check(result, "NG", jnp.float32)
    return result


def policy_log_probs(logits: jax.Array, tokens: jax.Array, temperature: float) -> jax.Array:
    sc = ShapeChecker()
    sc.check(logits, "BCV", jnp.float32)
    sc.check(tokens, "BC", jnp.int32)
    result = jnp.take_along_axis(jax.nn.log_softmax(logits / temperature), tokens[..., None], axis=-1)[..., 0]
    sc.check(result, "BC", jnp.float32)
    return result


def build_prompt(question: str, examples: Sequence[PromptExample] = ()) -> str:
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

    instruction = (
        "Answer only the last question.\n"
        "Give brief reasoning, then put only your final answer inside <answer>...</answer>.\n"
        "Finish immediately after </answer>.\n\n"
    )
    demonstrations = "".join(
        f"Question: {format_question(example.question)}\n\nAnswer: {example.completion.rstrip('\n')}\n\n"
        for example in examples
    )
    return instruction + demonstrations + f"Question: {format_question(question)}\n\nAnswer:"


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
) -> tuple[jax.Array, jax.Array]:
    prompts = np.zeros((len(entries), max_tokens), np.int32)
    lengths = np.empty(len(entries), np.int32)
    for i, entry in enumerate(entries):
        text = build_prompt(entry["question"], examples)
        ids = tokenizer.encode(text, out_type=int, add_bos=tokenizer.bos_id() >= 0, add_eos=False)
        if not ids or len(ids) > max_tokens:
            raise ValueError(f"Prompt has {len(ids)} tokens; expected 1..{max_tokens}. Increase max_prompt_tokens.")
        if any(token < 0 or token >= tokenizer.vocab_size() for token in ids):
            raise ValueError("Prompt token ID outside tokenizer vocabulary")
        prompts[i, : len(ids)] = ids
        lengths[i] = len(ids)
    sc = ShapeChecker()
    sc.check(prompts, "BP", np.int32)
    sc.check(lengths, "B", np.int32)
    return jnp.asarray(prompts), jnp.asarray(lengths)


type DecodeCarry = tuple[GatedDeltaNet2StackCarry | Gemma3Carry, jax.Array, jax.Array, jax.Array]
type DecodeOutput = tuple[jax.Array, jax.Array, jax.Array]


@partial(jax.jit, static_argnames=("model", "group_size", "max_new_tokens", "temperature", "eos_token_id"))
def generate(
    model: LanguageModel,
    params: Parameters,
    prompts: jax.Array,
    lengths: jax.Array,
    key: jax.Array,
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    eos_token_id: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Prefill unique prompts once; independently sample each repeated model state."""
    sc = ShapeChecker(V=model.vocab_size)
    sc.check(prompts, "NP", jnp.int32)
    sc.check(lengths, "N", jnp.int32)
    sc.check(jax.random.key_data(key), "R", jnp.uint32)
    if isinstance(model, Gemma3LM):
        if prompts.shape[1] + max_new_tokens > model.cache_length:
            raise ValueError("Prompt + completion budget exceeds Gemma cache length")
        carry, logits = model.prefill(params, prompts, lengths)
    else:
        carry, logits = model.apply({"params": params}, prompts, lengths)
    sc.check(logits, "NPV", jnp.float32)
    logits = jnp.repeat(logits[jnp.arange(prompts.shape[0]), lengths - 1], group_size, axis=0)

    def repeat_carry(x: jax.Array) -> jax.Array:
        return jnp.repeat(x, group_size, axis=0)

    carry = jax.tree.map(repeat_carry, carry)
    batch_size = prompts.shape[0] * group_size

    def step(previous: DecodeCarry, _: None) -> tuple[DecodeCarry, DecodeOutput]:
        carry, logits, active, key = previous
        sc = ShapeChecker(B=batch_size, V=model.vocab_size)
        sc.check(logits, "BV", jnp.float32)
        sc.check(active, "B", jnp.bool_)
        key, sample_key = jax.random.split(key)
        tokens = jax.random.categorical(sample_key, logits / temperature).astype(jnp.int32)
        tokens = jnp.where(active, tokens, 0)
        log_probs = policy_log_probs(logits[:, None], tokens[:, None], temperature)[:, 0]
        output = (tokens, jnp.where(active, log_probs, 0.0), active)
        active = active & (tokens != eos_token_id)
        if isinstance(model, Gemma3LM):
            carry, logits = model.step(params, tokens, active, carry)
        else:
            carry, logits = model.apply({"params": params}, tokens, active, carry, method=model.step)
        return (carry, logits, active, key), output

    (_, _, _, key), (tokens, log_probs, mask) = jax.lax.scan(
        step,
        (carry, logits, jnp.ones(batch_size, jnp.bool_), key),
        None,
        length=max_new_tokens,
    )
    output_sc = ShapeChecker(B=batch_size, C=max_new_tokens)
    tokens, log_probs, mask = tokens.T, log_probs.T, mask.T
    output_sc.check(tokens, "BC", jnp.int32)
    output_sc.check(log_probs, "BC", jnp.float32)
    output_sc.check(mask, "BC", jnp.bool_)
    return tokens, log_probs, mask, key


@partial(jax.jit, static_argnames=("model", "temperature"))
def completion_log_probs(
    model: LanguageModel,
    params: Parameters,
    batch: GRPOBatch,
    temperature: float,
) -> jax.Array:
    """Pack prompt+completion without padding gaps; backpropagate through the prompt."""
    batch.validate()
    b, p = batch.prompts.shape
    c = batch.completions.shape[1]
    positions = jnp.arange(p + c - 1)[None, :]
    completion_positions = positions - batch.prompt_lengths[:, None]
    prompts = jnp.take_along_axis(batch.prompts, jnp.minimum(positions, p - 1), axis=1)
    completions = jnp.take_along_axis(batch.completions, jnp.clip(completion_positions, 0, c - 1), axis=1)
    tokens = jnp.where(completion_positions < 0, prompts, completions)
    lengths = batch.prompt_lengths + batch.mask.sum(axis=1, dtype=jnp.int32) - 1
    # Rematerialize model activations during backward instead of retaining every layer.
    _, logits = jax.checkpoint(model.apply)({"params": params}, tokens, lengths)
    sc = ShapeChecker(B=b, T=p + c - 1, V=model.vocab_size, C=c)
    sc.check(tokens, "BT", jnp.int32)
    sc.check(logits, "BTV", jnp.float32)
    indices = batch.prompt_lengths[:, None] - 1 + jnp.arange(c)[None, :]
    logits = logits[jnp.arange(b)[:, None], indices]
    result = policy_log_probs(logits, batch.completions, temperature)
    sc.check(result, "BC", jnp.float32)
    return jnp.where(batch.mask, result, 0.0)


def objective(log_probs: jax.Array, batch: GRPOBatch, clip_coef: float, beta: float) -> tuple[jax.Array, Metrics]:
    """Clipped token ratios and sampled reference KL, averaged equally per completion."""
    batch.validate()
    sc = ShapeChecker()
    sc.check(batch.completions, "BC", jnp.int32)
    sc.check(log_probs, "BC", jnp.float32)

    def average(values: jax.Array) -> jax.Array:
        sc = ShapeChecker(B=batch.mask.shape[0], C=batch.mask.shape[1])
        sc.check(values, "BC", jnp.float32)
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
    sc.check(list(metrics.values()), "", jnp.float32)
    return loss, metrics


def create_state(model: LanguageModel, params: Parameters, config: Config) -> TrainState:
    return TrainState.create(
        apply_fn=model.apply,
        params=jax.device_put(params),
        tx=optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate)),
    )


@partial(jax.jit, static_argnames=("model", "temperature", "clip_coef", "beta"))
def update(
    state: TrainState,
    model: LanguageModel,
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
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Task scorer returned invalid reward: {score}")
            rewards[i] = score
    sc = ShapeChecker(B=len(texts))
    sc.check(rewards, "B", np.float32)
    return jnp.asarray(rewards)


def apply_duration_penalty(
    task_rewards: jax.Array,
    mask: jax.Array,
    coefficient: float,
) -> tuple[jax.Array, jax.Array]:
    """Deduct a token-budget-normalized cost, counting EOS but excluding padding."""
    sc = ShapeChecker()
    sc.check(task_rewards, "B", jnp.float32)
    sc.check(mask, "BT", jnp.bool_)
    chex.assert_scalar_positive(mask.shape[1])
    penalties = coefficient * mask.sum(axis=1, dtype=jnp.float32) / mask.shape[1]
    rewards = task_rewards - penalties
    sc.check([penalties, rewards], "B", jnp.float32)
    return rewards, penalties


def collect_rollout(
    state: TrainState,
    model: LanguageModel,
    ref_params: Parameters,
    dataset: ProceduralDataset,
    tokenizer: Tokenizer,
    key: jax.Array,
    iteration: int,
    config: Config,
) -> tuple[GRPOBatch, dict[str, float], str, jax.Array]:
    entries = [dataset[iteration * config.num_tasks + i] for i in range(config.num_tasks)]
    prompts, lengths = encode_prompts(entries, tokenizer, config.max_prompt_tokens, config.prompt_examples)
    generation_start = monotonic()
    tokens, log_probs, mask, key = generate(
        model,
        state.params,
        prompts,
        lengths,
        key,
        group_size=config.group_size,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        eos_token_id=tokenizer.eos_id(),
    )
    host_tokens, host_mask = jax.device_get((tokens, mask))
    generation_seconds = monotonic() - generation_start
    decoding_start = monotonic()
    texts = [
        tokenizer.decode(ids[valid & (ids != tokenizer.eos_id())].tolist())
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
        jnp.repeat(prompts, config.group_size, axis=0),
        jnp.repeat(lengths, config.group_size),
        tokens,
        mask,
        log_probs,
        jnp.zeros_like(log_probs),
        group_advantages(grouped).reshape(-1),
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
        "charts/truncation_rate": float((~jnp.any(mask & (tokens == tokenizer.eos_id()), axis=1)).mean()),
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
    sc = ShapeChecker(G=config.group_size)
    sc.check(group_rewards, "G", np.float32)
    # Stable sorting preserves sampling order for ties; only the displayed group changes.
    indices = np.argsort(-group_rewards, kind="stable")[:count]
    samples = format_group_samples(
        build_prompt(entries[0]["question"], config.prompt_examples),
        [texts[index] for index in indices],
        group_rewards[indices].tolist(),
    )
    return batch, diagnostics, samples, key


def save_training_checkpoint(
    run_dir: str,
    state: TrainState,
    key: jax.Array,
    iteration: int,
    config: Config,
    *,
    prompts_seen: int,
) -> None:
    sc = ShapeChecker()
    sc.check(jax.random.key_data(key), "R", jnp.uint32)
    payload = {
        "version": 1,
        "state": serialization.to_state_dict(state),
        "iteration": iteration,
        "prompts_seen": prompts_seen,
        "key": np.asarray(jax.random.key_data(key)),
        "key_impl": str(jax.random.key_impl(key)),
        "config": asdict(config),  # Provenance only; compatibility is determined by the saved state.
    }
    write_bytes(f"{run_dir}/checkpoint.msgpack", serialization.msgpack_serialize(payload))


def restore_training_checkpoint(data: bytes, state: TrainState) -> tuple[TrainState, jax.Array, int, int]:
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
    sc = ShapeChecker()
    sc.check(payload["key"], "R", np.uint32)
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
    key = jax.random.wrap_key_data(jnp.asarray(payload["key"]), impl=payload["key_impl"])
    return restored, key, iteration, prompts_seen


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
    tokenizer_source = config.tokenizer or (
        "Gemma 3 default" if config.model_type == "gemma3" else "cached TinyLlama default"
    )
    log_init(f"Loading tokenizer: {tokenizer_source}")
    tokenizer = (
        Gemma3Tokenizer(config.tokenizer)
        if config.model_type == "gemma3"
        else load_tokenizer(Path(config.tokenizer) if config.tokenizer is not None else None)
    )
    log_init(f"Tokenizer ready: vocabulary size {tokenizer.vocab_size():,}")
    log_init(f"Loading pretrained checkpoint: {config.checkpoint} (backend={config.backend}, dtype={config.dtype})")
    if config.model_type == "gemma3":
        model, variables = load_gemma3_checkpoint(
            config.checkpoint,
            cache_length=config.max_prompt_tokens + config.max_new_tokens,
            dtype=config.dtype,
        )
    else:
        model, variables = load_checkpoint(
            Path(config.checkpoint), dtype=jnp.dtype(config.dtype), backend=config.backend
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
    config = replace(config, run_id=config.run_id or f"grpo_reasoning_{datetime.now(UTC):%Y%m%d-%H%M%S-%f}")
    run_dir = f"{config.log_dir.rstrip('/')}/{config.run_id}"
    log_init(f"Checking for saved training state: {run_dir}/checkpoint.msgpack (reading if present)")
    saved = read_optional(f"{run_dir}/checkpoint.msgpack")
    key, start_iteration, prompts_seen = jax.random.key(config.seed), 0, 0
    if saved is not None:
        log_init(f"Training checkpoint read ({len(saved) / 1_000_000:.1f} MB); restoring model, optimizer, and RNG")
        state, key, start_iteration, prompts_seen = restore_training_checkpoint(saved, state)
        print(
            f"Restored {run_dir}/checkpoint.msgpack at rollout {start_iteration}, optimizer step {int(state.step)}",
            flush=True,
        )
    else:
        log_init("No training checkpoint found; starting from pretrained weights")
    log_init(f"Writing run configuration: {run_dir}/config.yaml")
    write_bytes(f"{run_dir}/config.yaml", yaml.safe_dump(asdict(config)).encode())
    last_checkpoint = monotonic()
    batch_size = config.num_tasks * config.group_size
    first_log_iteration = (start_iteration // config.log_interval + 1) * config.log_interval
    print(f"Run: {run_dir}; devices: {devices}; starting at rollout {start_iteration}", flush=True)
    log_init(f"Opening TensorBoard writer: {run_dir}")
    with SummaryWriter(logdir=run_dir, purge_step=prompts_seen + 1 if saved else None) as writer:
        writer.add_text("config", f"```yaml\n{yaml.safe_dump(asdict(config))}```", prompts_seen)
        log_init("Initialization complete")
        window_timings: dict[str, float] = {}
        window_rollouts = 0
        window_start = monotonic()
        for iteration in range(start_iteration, config.total_updates):
            if iteration == start_iteration:
                print("[startup] Collecting first rollout; generation may trigger JAX compilation", flush=True)
            batch, diagnostics, samples, key = collect_rollout(
                state,
                model,
                ref_params,
                dataset,
                tokenizer,
                key,
                iteration,
                config,
            )
            if iteration == start_iteration:
                print(
                    "[startup] First rollout collected; starting optimization (may trigger JAX compilation)", flush=True
                )
            metrics: list[Metrics] = []
            update_start = monotonic()
            for _ in range(config.update_epochs):
                key, shuffle_key = jax.random.split(key)
                for indices in np.split(
                    np.asarray(jax.random.permutation(shuffle_key, batch_size)), config.num_minibatches
                ):
                    minibatch = GRPOBatch(*(x[indices] for x in batch))
                    state, metric = update(
                        state,
                        model,
                        minibatch,
                        temperature=config.temperature,
                        clip_coef=config.clip_coef,
                        beta=config.beta,
                    )
                    metrics.append(metric)
            diagnostics["time/update_dispatch_seconds"] = monotonic() - update_start
            diagnostics["time/jax_seconds"] += diagnostics["time/update_dispatch_seconds"]
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
                for name, value in {
                    **diagnostics,
                    **means,
                    **window_timings,
                    "time/window_rollouts": window_rollouts,
                }.items():
                    writer.add_scalar(name, value, prompts_seen)
                writer.add_text("samples/completions", samples, prompts_seen)
                writer.flush()
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
            if monotonic() - last_checkpoint >= config.checkpoint_interval_seconds or completed == config.total_updates:
                save_training_checkpoint(run_dir, state, key, completed, config, prompts_seen=prompts_seen)
                last_checkpoint = monotonic()
                print(f"Saved {run_dir}/checkpoint.msgpack at rollout {completed}", flush=True)
    # Export in the source model format; inference exports are local-only.
    if not run_dir.startswith("gs://"):
        export = Path(run_dir) / f"model-{max(start_iteration, config.total_updates)}"
        if not export.exists():
            if isinstance(model, Gemma3LM):
                save_gemma3_checkpoint(export, jax.device_get(state.params))
            else:
                save_checkpoint(export, model, jax.device_get(state.params), {"task": config.task})
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grpo_reasoning.yaml", help="Path to YAML config")
    args = parser.parse_args()
    print(f"[init] Loading configuration: {args.config}", flush=True)
    train(load_config(args.config))


if __name__ == "__main__":
    main()
