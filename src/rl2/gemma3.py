"""GRPO adapter for Google DeepMind's Gemma model, tokenizer, and checkpoints.

Install the optional ``gemma3`` extra. Imports stay lazy so GDN2 training does not
require Gemma's dependencies. Model computation and KV-cache updates are owned by
Gemma; this module adapts padding, precision, and interfaces for the trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np

from rl2.gdn2.checkpoints import Parameters
from rl2.shape_checker import ShapeChecker

if TYPE_CHECKING:
    from gemma import gm

# Every cache leaf has a leading batch dimension, including end_index.
type Cache = dict[str, dict[str, jax.Array]]
type ComputeDtype = Literal["float32", "bfloat16"]


class Gemma3Carry(NamedTuple):
    cache: Cache
    lengths: jax.Array  # Next logical position, excluding padding.
    valid: jax.Array  # Valid physical cache slots; padding gaps stay masked.


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
        def to_compute_dtype(value: jax.Array) -> jax.Array:
            return value.astype(self.dtype)

        return {"params": jax.tree.map(to_compute_dtype, params)}

    def apply(self, variables: Parameters, tokens: jax.Array, lengths: jax.Array) -> tuple[None, jax.Array]:
        sc = ShapeChecker(V=self.vocab_size)
        sc.check(tokens, "BT", jnp.int32)
        sc.check(lengths, "B", jnp.int32)
        positions = jnp.broadcast_to(jnp.arange(tokens.shape[1], dtype=jnp.int32), tokens.shape)
        keys = jnp.arange(tokens.shape[1])[None, None, :]
        mask = (keys <= positions[..., None]) & (keys < lengths[:, None, None])
        sc.check(positions, "BT", jnp.int32)
        sc.check(mask, "BTT", jnp.bool_)
        output = self.model.apply(
            self._variables(variables["params"]),
            tokens=tokens,
            positions=positions,
            attention_mask=mask,
            return_last_only=False,
        )
        logits = output.logits.astype(jnp.float32)
        sc.check(logits, "BTV", jnp.float32)
        return None, logits

    def prefill(self, params: Parameters, tokens: jax.Array, lengths: jax.Array) -> tuple[Gemma3Carry, jax.Array]:
        sc = ShapeChecker(C=self.cache_length, V=self.vocab_size)
        sc.check(tokens, "BT", jnp.int32)
        sc.check(lengths, "B", jnp.int32)
        if tokens.shape[1] > self.cache_length:
            raise ValueError("Prompt exceeds Gemma cache length")
        cache = self.model.init_cache(
            batch_size=tokens.shape[0], dtype=jnp.dtype(self.dtype), cache_length=self.cache_length
        )
        positions = jnp.broadcast_to(jnp.arange(tokens.shape[1], dtype=jnp.int32), tokens.shape)
        slots = jnp.arange(self.cache_length)
        valid = slots[None, :] < lengths[:, None]
        mask = valid[:, None, :] & (slots[None, None, :] <= positions[..., None])
        sc.check(positions, "BT", jnp.int32)
        sc.check(valid, "BC", jnp.bool_)
        sc.check(mask, "BTC", jnp.bool_)
        output = self.model.apply(
            self._variables(params),
            tokens=tokens,
            positions=positions,
            attention_mask=mask,
            cache=cache,
            return_last_only=False,
        )
        carry = Gemma3Carry(cast(Cache, output.cache), lengths, valid)
        self._check_carry(carry)
        logits = output.logits.astype(jnp.float32)
        sc.check(logits, "BTV", jnp.float32)
        return carry, logits

    def step(
        self,
        params: Parameters,
        tokens: jax.Array,
        active: jax.Array,
        carry: Gemma3Carry,
    ) -> tuple[Gemma3Carry, jax.Array]:
        self._check_carry(carry)
        sc = ShapeChecker(B=carry.lengths.shape[0], C=self.cache_length, V=self.vocab_size)
        sc.check(tokens, "B", jnp.int32)
        sc.check(active, "B", jnp.bool_)
        # Upstream writes every row at the same physical offset. Keep right-padded
        # prompts in place and pass logical positions independently of that offset.
        index = carry.cache["layer_0"]["end_index"][0]
        valid = carry.valid.at[:, index].set(active)
        mask = valid[:, None, :]
        sc.check(mask, "BSC", jnp.bool_)
        output = self.model.apply(
            self._variables(params),
            tokens=tokens[:, None],
            positions=carry.lengths[:, None],
            attention_mask=mask,
            cache=carry.cache,
            return_last_only=False,
        )
        carry = Gemma3Carry(cast(Cache, output.cache), carry.lengths + active.astype(jnp.int32), valid)
        self._check_carry(carry)
        logits = output.logits[:, 0].astype(jnp.float32)
        sc.check(logits, "BV", jnp.float32)
        return carry, logits

    def _check_carry(self, carry: Gemma3Carry) -> None:
        sc = ShapeChecker(C=self.cache_length, K=self.model.config.num_kv_heads, H=self.model.config.head_dim)
        sc.check(carry.lengths, "B", jnp.int32)
        sc.check(carry.valid, "BC", jnp.bool_)
        for layer in carry.cache.values():
            sc.check([layer["k"], layer["v"]], "BCKH", jnp.dtype(self.dtype))
            sc.check(layer["end_index"], "B", jnp.int32)
            sc.check(layer["positions"], "BC", jnp.int32)


def load_gemma3_checkpoint(source: str, *, cache_length: int, dtype: ComputeDtype) -> tuple[Gemma3LM, Parameters]:
    from gemma import gm

    model = Gemma3LM(gm.nn.Gemma3_270M(dtype=jnp.float32), cache_length, dtype)
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
        names = "ABCD"[: value.ndim]
        sc = ShapeChecker(**dict(zip(names, expected.shape, strict=True)))
        sc.check(value, names)
        if value.dtype not in (jnp.float32, jnp.bfloat16, jnp.float16):
            raise ValueError(f"Unsupported Gemma weight dtype: {value.dtype}")
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError("Non-finite Gemma checkpoint weight")

    def to_float32(value: jax.Array) -> jax.Array:
        return value.astype(jnp.float32)

    return model, {"params": jax.tree.map(to_float32, params)}


class Gemma3Tokenizer:
    def __init__(self, source: str | None = None) -> None:
        from gemma import gm

        self.tokenizer = gm.text.Gemma3Tokenizer() if source is None else gm.text.Gemma3Tokenizer(path=source)

    def bos_id(self) -> int:
        return int(self.tokenizer.special_tokens.BOS)

    def eos_id(self) -> int:
        return int(self.tokenizer.special_tokens.EOS)

    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def encode(self, text: str, *, out_type: type[int], add_bos: bool, add_eos: bool) -> list[int]:
        return self.tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)


def save_gemma3_checkpoint(path: Path, params: Parameters) -> None:
    from gemma import gm

    gm.ckpts.save_params(params, path.resolve(), wait_until_finished=True)
