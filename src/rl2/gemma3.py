"""GRPO adapter for Google DeepMind's Gemma model, tokenizer, and checkpoints.

Gemma owns model computation and KV-cache updates; this module adapts
padding, precision, and interfaces for the trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import chex
import jax
import jax.numpy as jnp
from gemma import gm

type Parameters = dict[str, Any]

# B: batch, T: input tokens, C: cache slots, V: vocabulary.
# Cache leaves: k/v (B, C, K, H), positions (B, C), end_index (B,).
# K: KV heads, H: head dimension.
type Cache = dict[str, dict[str, jax.Array]]
type ComputeDtype = Literal["float32", "bfloat16"]


class Gemma3Carry(NamedTuple):
    cache: Cache
    lengths: jax.Array  # (B,) Next logical position, excluding padding.
    valid: jax.Array  # (B, C) Valid physical cache slots; padding gaps stay masked.


@dataclass(frozen=True)
class Gemma3LM:
    model: gm.nn.Transformer
    cache_length: int
    dtype: ComputeDtype = "float32"

    def __post_init__(self) -> None:
        if not 0 < self.cache_length <= 32768:
            raise ValueError("Gemma 3 270M prompt + completion budget must fit its 32768-token context")

    @property
    def vocab_size(self) -> int:
        return self.model.config.num_embed

    def _variables(self, params: Parameters) -> Parameters:
        # Gemma's model dtype controls initialization. Cast for computation here
        # while retaining float32 master parameters and differentiable casts.
        def to_compute_dtype(value: jax.Array) -> jax.Array:  # (...) -> (...)
            return value.astype(self.dtype)  # (...)

        return {"params": jax.tree.map(to_compute_dtype, params)}

    def apply(
        self,
        variables: Parameters,
        tokens: jax.Array,  # (B, T)
        lengths: jax.Array,  # (B,)
    ) -> tuple[None, jax.Array]:
        """Return (B, T, V) float32 logits for right-padded sequences."""
        positions = jnp.broadcast_to(jnp.arange(tokens.shape[1], dtype=jnp.int32), tokens.shape)  # (B, T)
        keys = jnp.arange(tokens.shape[1])[None, None, :]  # (1, 1, T)
        mask = (keys <= positions[..., None]) & (keys < lengths[:, None, None])  # (B, T, T)
        output = self.model.apply(
            self._variables(variables["params"]),
            tokens=tokens,
            positions=positions,
            attention_mask=mask,
            return_last_only=False,
        )
        logits = output.logits.astype(jnp.float32)  # (B, T, V)
        return None, logits  # (B, T, V)

    def prefill(
        self,
        params: Parameters,
        tokens: jax.Array,  # (B, T)
        lengths: jax.Array,  # (B,)
    ) -> tuple[Gemma3Carry, jax.Array]:
        """Return a padded cache and (B, T, V) float32 prompt logits."""
        if not 0 < tokens.shape[1] <= self.cache_length:
            raise ValueError("Prompt width must be positive and fit Gemma cache length")
        # Prefill attends only to prompt slots; reserve completion slots afterward.
        cache = self.model.init_cache(
            batch_size=tokens.shape[0], dtype=jnp.dtype(self.dtype), cache_length=tokens.shape[1]
        )
        positions = jnp.broadcast_to(jnp.arange(tokens.shape[1], dtype=jnp.int32), tokens.shape)  # (B, T)
        slots = jnp.arange(tokens.shape[1])  # (T,)
        mask = (slots[None, None, :] < lengths[:, None, None]) & (
            slots[None, None, :] <= positions[..., None]
        )  # (B, T, T)
        output = self.model.apply(
            self._variables(params),
            tokens=tokens,
            positions=positions,
            attention_mask=mask,
            cache=cache,
            return_last_only=False,
        )

        def pad_cache(value: jax.Array) -> jax.Array:  # (B, T, ...) or (B,)
            if value.ndim == 1:
                return value  # (B,) end_index
            padding = [(0, 0), (0, self.cache_length - tokens.shape[1])] + [(0, 0)] * (value.ndim - 2)
            return jnp.pad(value, padding)  # (B, C, ...)

        cache = jax.tree.map(pad_cache, cast(Cache, output.cache))
        valid = jnp.arange(self.cache_length)[None, :] < lengths[:, None]  # (B, C)
        carry = Gemma3Carry(cache=cache, lengths=lengths, valid=valid)
        logits = output.logits.astype(jnp.float32)  # (B, T, V)
        return carry, logits  # logits: (B, T, V)

    def step(
        self,
        params: Parameters,
        tokens: jax.Array,  # (B,)
        active: jax.Array,  # (B,)
        carry: Gemma3Carry,
    ) -> tuple[Gemma3Carry, jax.Array]:
        """Append one physical slot; callers must stay within the cache budget."""
        # Upstream writes every row at the same physical offset. Keep right-padded
        # prompts in place and pass logical positions independently of that offset.
        index = carry.cache["layer_0"]["end_index"][0]  # ()
        valid = carry.valid.at[:, index].set(active)  # (B, C)
        mask = valid[:, None, :]  # (B, 1, C)
        output = self.model.apply(
            self._variables(params),
            tokens=tokens[:, None],
            positions=carry.lengths[:, None],
            attention_mask=mask,
            cache=carry.cache,
            return_last_only=False,
        )
        carry = Gemma3Carry(
            cache=cast(Cache, output.cache),
            lengths=carry.lengths + active.astype(jnp.int32),  # (B,)
            valid=valid,  # (B, C)
        )
        logits = output.logits[:, 0].astype(jnp.float32)  # (B, V)
        return carry, logits  # logits: (B, V)


def load_gemma3_checkpoint(source: str, *, cache_length: int, dtype: ComputeDtype) -> tuple[Gemma3LM, Parameters]:
    model = Gemma3LM(model=gm.nn.Gemma3_270M(dtype=jnp.float32), cache_length=cache_length, dtype=dtype)
    params = gm.ckpts.load_params(source)
    # Check the external checkpoint against an abstract initialization: this does
    # not allocate or randomly initialize a second 270M-parameter model.
    template = jax.eval_shape(
        model.model.init,
        jax.random.key(0),
        tokens=jnp.ones((1, 1), jnp.int32),
    )["params"]
    if jax.tree.structure(params) != jax.tree.structure(template):
        raise ValueError("Checkpoint parameter structure does not match Gemma 3 270M")
    for value, expected in zip(jax.tree.leaves(params), jax.tree.leaves(template), strict=True):
        chex.assert_shape(value, expected.shape)
        if value.dtype not in (jnp.float32, jnp.bfloat16, jnp.float16):
            raise ValueError(f"Unsupported Gemma weight dtype: {value.dtype}")

    def to_float32(value: jax.Array) -> jax.Array:  # (...) -> (...)
        return value.astype(jnp.float32)  # (...)

    return model, {"params": jax.tree.map(to_float32, params)}


class Gemma3Tokenizer:
    def __init__(self, source: str | None = None) -> None:
        self.tokenizer = gm.text.Gemma3Tokenizer() if source is None else gm.text.Gemma3Tokenizer(path=source)

    def bos_id(self) -> int:
        return int(self.tokenizer.special_tokens.BOS)

    def eos_id(self) -> int:
        return int(self.tokenizer.special_tokens.EOS)

    def stop_ids(self) -> tuple[int, ...]:
        return self.eos_id(), int(self.tokenizer.special_tokens.END_OF_TURN)

    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def encode(self, text: str, *, out_type: type[int], add_bos: bool, add_eos: bool) -> list[int]:
        return self.tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)


def save_gemma3_checkpoint(path: Path, params: Parameters) -> None:
    gm.ckpts.save_params(params, path.resolve(), wait_until_finished=True)
