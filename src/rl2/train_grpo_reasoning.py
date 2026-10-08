"""Fine-tune a converted GDN2 checkpoint with GRPO on Reasoning Gym.

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
from rl2.jax_cache import configure_compilation_cache
from rl2.shape_checker import ShapeChecker
from rl2.utils import read_bytes, read_optional, write_bytes

type Entry = dict[str, Any]
type Metrics = dict[str, jax.Array]


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
    max_grad_norm: float = 1.0
    log_dir: str = "runs"
    run_id: str | None = None
    log_interval: int = 10
    checkpoint_interval_seconds: float = 600.0

    def __post_init__(self) -> None:
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
    instruction = (
        "Solve the following problem. You may reason before answering. "
        "Put only your final answer inside <answer>...</answer>.\n\n"
    )
    demonstrations = "".join(f"Question: {example.question}\n\nAnswer: {example.completion}\n\n" for example in examples)
    return instruction + demonstrations + f"Question: {question}\n\nAnswer:"


def format_sample(prompt: str, completion: str, reward: float) -> str:
    """Render labeled, literal text blocks in TensorBoard's Markdown viewer."""

    def literal(text: str) -> str:
        return "    " + text.replace("\n", "\n    ")

    return f"### Prompt\n\n{literal(prompt)}\n\n### Completion\n\n{literal(completion)}\n\n**Reward:** {reward:.3f}"


def encode_prompts(
    entries: list[Entry], tokenizer: Tokenizer, max_tokens: int, examples: Sequence[PromptExample] = (),
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


type DecodeCarry = tuple[GatedDeltaNet2StackCarry, jax.Array, jax.Array, jax.Array]
type DecodeOutput = tuple[jax.Array, jax.Array, jax.Array]


@partial(jax.jit, static_argnames=("model", "group_size", "max_new_tokens", "temperature", "eos_token_id"))
def generate(
    model: GatedDeltaNet2LM,
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
    """Prefill unique prompts once; independently sample each repeated recurrent state."""
    sc = ShapeChecker(V=model.vocab_size)
    sc.check(prompts, "NP", jnp.int32)
    sc.check(lengths, "N", jnp.int32)
    sc.check(jax.random.key_data(key), "R", jnp.uint32)
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
    model: GatedDeltaNet2LM,
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


def create_state(model: GatedDeltaNet2LM, params: Parameters, config: Config) -> TrainState:
    return TrainState.create(
        apply_fn=model.apply,
        params=jax.device_put(params),
        tx=optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate)),
    )


@partial(jax.jit, static_argnames=("model", "temperature", "clip_coef", "beta"))
def update(
    state: TrainState,
    model: GatedDeltaNet2LM,
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


def collect_rollout(
    state: TrainState,
    model: GatedDeltaNet2LM,
    ref_params: Parameters,
    dataset: ProceduralDataset,
    tokenizer: Tokenizer,
    key: jax.Array,
    iteration: int,
    config: Config,
) -> tuple[GRPOBatch, dict[str, float], list[str], jax.Array]:
    entries = [dataset[iteration * config.num_tasks + i] for i in range(config.num_tasks)]
    prompts, lengths = encode_prompts(entries, tokenizer, config.max_prompt_tokens, config.prompt_examples)
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
    texts = [
        tokenizer.decode(ids[valid & (ids != tokenizer.eos_id())].tolist())
        for ids, valid in zip(host_tokens, host_mask, strict=True)
    ]
    rewards = score_completions(dataset, entries, texts, config.group_size)
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
        "charts/success_rate": float((rewards == 1).mean()),
        "charts/informative_group_fraction": float((jnp.ptp(grouped, axis=1) > 0).mean()),
        "charts/completion_length_mean": float(mask.sum(axis=1).mean()),
        "charts/truncation_rate": float((~jnp.any(mask & (tokens == tokenizer.eos_id()), axis=1)).mean()),
    }
    samples = [
        format_sample(
            build_prompt(entries[i // config.group_size]["question"], config.prompt_examples), text, float(rewards[i])
        )
        for i, text in enumerate(texts[: config.group_size])
    ]
    return batch, diagnostics, samples, key


def _resume_config(config: Config) -> dict[str, Any]:
    # Extending a run and changing reporting frequency do not change its policy/data stream.
    ignored = {"total_updates", "log_interval", "checkpoint_interval_seconds", "log_dir", "run_id"}
    return {name: value for name, value in asdict(config).items() if name not in ignored}


def save_training_checkpoint(run_dir: str, state: TrainState, key: jax.Array, iteration: int, config: Config) -> None:
    sc = ShapeChecker()
    sc.check(jax.random.key_data(key), "R", jnp.uint32)
    payload = {
        "version": 1,
        "state": serialization.to_state_dict(state),
        "iteration": iteration,
        "key": np.asarray(jax.random.key_data(key)),
        "key_impl": str(jax.random.key_impl(key)),
        "config": _resume_config(config),
    }
    write_bytes(f"{run_dir}/checkpoint.msgpack", serialization.msgpack_serialize(payload))


def restore_training_checkpoint(data: bytes, state: TrainState, config: Config) -> tuple[TrainState, jax.Array, int]:
    payload = serialization.msgpack_restore(data)
    if payload["version"] != 1:
        raise ValueError("Unsupported reasoning GRPO checkpoint version")
    if payload["config"] != _resume_config(config):
        raise ValueError("Training configuration differs from checkpoint; use a new run_id")
    restored = serialization.from_state_dict(state, payload["state"])
    chex.assert_trees_all_equal_shapes_and_dtypes(
        (state.params, state.opt_state), (restored.params, restored.opt_state)
    )
    sc = ShapeChecker()
    sc.check(payload["key"], "R", np.uint32)
    iteration = payload["iteration"]
    if type(iteration) is not int or iteration < 0:
        raise ValueError("Invalid checkpoint iteration")
    key = jax.random.wrap_key_data(jnp.asarray(payload["key"]), impl=payload["key_impl"])
    return restored, key, iteration


def train(config: Config) -> TrainState:
    configure_compilation_cache()
    dataset = reasoning_gym.create_dataset(
        config.task,
        seed=config.seed,
        size=config.total_updates * config.num_tasks,
        **config.task_config,
    )
    tokenizer = load_tokenizer(Path(config.tokenizer) if config.tokenizer is not None else None)
    model, variables = load_checkpoint(Path(config.checkpoint), dtype=jnp.dtype(config.dtype), backend=config.backend)
    if tokenizer.vocab_size() != model.vocab_size:
        raise ValueError("Tokenizer vocabulary size does not match checkpoint vocabulary size")
    if not 0 <= tokenizer.eos_id() < model.vocab_size:
        raise ValueError("Tokenizer must define an EOS token within the model vocabulary")
    ref_params = jax.device_put(variables["params"])
    state = create_state(model, ref_params, config)
    if not config.beta:
        ref_params = {}
    del variables
    config = replace(config, run_id=config.run_id or f"grpo_reasoning_{datetime.now(UTC):%Y%m%d-%H%M%S-%f}")
    run_dir = f"{config.log_dir.rstrip('/')}/{config.run_id}"
    saved = read_optional(f"{run_dir}/checkpoint.msgpack")
    key, start_iteration = jax.random.key(config.seed), 0
    if saved is not None:
        state, key, start_iteration = restore_training_checkpoint(saved, state, config)
        print(
            f"Restored {run_dir}/checkpoint.msgpack at rollout {start_iteration}, optimizer step {int(state.step)}",
            flush=True,
        )
    write_bytes(f"{run_dir}/config.yaml", yaml.safe_dump(asdict(config)).encode())
    last_checkpoint = monotonic()
    batch_size = config.num_tasks * config.group_size
    print(f"Run: {run_dir}; devices: {jax.devices()}; starting at rollout {start_iteration}", flush=True)
    with SummaryWriter(logdir=run_dir, purge_step=start_iteration * batch_size + 1 if saved else None) as writer:
        writer.add_text("config", f"```yaml\n{yaml.safe_dump(asdict(config))}```", start_iteration * batch_size)
        for iteration in range(start_iteration, config.total_updates):
            start = monotonic()
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
            metrics: list[Metrics] = []
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
            completed = iteration + 1
            if completed % config.log_interval == 0 or completed == config.total_updates:
                jax.block_until_ready(state)
                means = {name: float(np.mean([m[name] for m in jax.device_get(metrics)])) for name in metrics[0]}
                for name, value in {**diagnostics, **means, "time/iteration_seconds": monotonic() - start}.items():
                    writer.add_scalar(name, value, completed * batch_size)
                writer.add_text("samples/completions", "\n\n---\n\n".join(samples), completed * batch_size)
                writer.flush()
                print(
                    f"iteration={completed} reward={diagnostics['charts/reward_mean']:.3f} loss={means['losses/total']:.4f}",
                    flush=True,
                )
            if monotonic() - last_checkpoint >= config.checkpoint_interval_seconds or completed == config.total_updates:
                save_training_checkpoint(run_dir, state, key, completed, config)
                last_checkpoint = monotonic()
                print(f"Saved {run_dir}/checkpoint.msgpack at rollout {completed}", flush=True)
    # Export a standard converted checkpoint usable by rl2.gdn2.generate.
    # Training state can also live on GCS; converted inference exports are local-only.
    if not run_dir.startswith("gs://"):
        export = Path(run_dir) / f"model-{max(start_iteration, config.total_updates)}"
        if not export.exists():
            save_checkpoint(export, model, jax.device_get(state.params), {"task": config.task})
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grpo_reasoning.yaml", help="Path to YAML config")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
