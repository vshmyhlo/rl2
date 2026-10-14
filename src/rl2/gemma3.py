"""GRPO adapter for Google DeepMind's Gemma model, tokenizer, and checkpoints.

This module adapts padding, precision, attention, and trainer interfaces.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import chex
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import linen as nn
from flax.linen.module import InterceptorContext
from gemma import gm

from rl2.attention import AttentionType

type Parameters = dict[str, Any]

# B: batch, T: input tokens, C: cache slots, V: vocabulary.
# Cache leaves: k/v (B, C, K, H), positions (B, C), end_index (B,).
# K: KV heads, N: query heads, H: head dimension, D: embedding dimension.
# S: attended key slots (prompt width or cache length).
type Cache = dict[str, dict[str, jax.Array]]
type ComputeDtype = Literal["float32", "bfloat16"]


def _cudnn_attention(
    module: gm.nn.Attention,
    x: jax.Array,  # (B, T, D)
    segment_pos: jax.Array,  # (B, T)
    cache: dict[str, jax.Array] | None,
    attn_mask: jax.Array,  # (B, T, S)
) -> tuple[dict[str, jax.Array] | None, jax.Array]:
    """Reuse Gemma projections and RoPE; return cache and (B, T, D) output."""
    if module.use_qkv_einsum:
        query, key, value = module.qkv_einsum("BTD,SNDH->SBTNH", x)  # each: (B, T, N, H)
    else:
        query = module.q_einsum("BTD,NDH->BTNH", x)  # (B, T, N, H)
        key, value = module.kv_einsum("BTD,CKDH->CBTKH", x)  # each: (B, T, K, H)
    if module.use_qk_norm:
        query = module._query_norm(query)  # (B, T, N, H)
        key = module._key_norm(key)  # (B, T, K, H)

    def rotate(projected: jax.Array) -> jax.Array:  # (B, T, heads, H)
        return gm.math.apply_rope(
            projected,
            segment_pos,
            base_frequency=module.rope_base_frequency,
            scale_factor=module.rope_scale_factor,
        )  # (B, T, heads, H)

    query, key = jax.tree.map(rotate, (query, key))  # (B, T, N, H), (B, T, K, H)
    query = query * module.query_pre_attn_scalar  # (B, T, N, H)
    cache_positions = segment_pos  # (B, S=T)
    if cache is not None:
        index = cache["end_index"][0] % cache["v"].shape[1]  # ()

        def update(old: jax.Array, new: jax.Array) -> jax.Array:  # (B, S, ...), (B, T, ...)
            return jax.lax.dynamic_update_slice(old, new, (0, index) + (0,) * (old.ndim - 2))  # (B, S, ...)

        key, value, cache_positions = jax.tree.map(
            update, (cache["k"], cache["v"], cache["positions"]), (key, value, segment_pos)
        )  # (B, S, K, H), (B, S, K, H), (B, S)
        cache = {
            "k": key,  # (B, S, K, H)
            "v": value,  # (B, S, K, H)
            "positions": cache_positions,  # (B, S)
            "end_index": cache["end_index"] + x.shape[1],  # (B,)
        }
    if module.attn_type == gm.nn.AttentionType.LOCAL_SLIDING:
        window = module.sliding_window_size
        if window is None:
            raise ValueError("Local attention requires sliding_window_size")
        distance = cache_positions[:, None, :] - segment_pos[:, :, None]  # (B, T, S)
        attn_mask = attn_mask & (distance > -window) & (distance < window)  # (B, T, S)

    # Gemma's fully masked rows average all values. Avoid empty fused softmaxes.
    visible = jnp.any(attn_mask, axis=-1)  # (B, T)
    empty_output = jnp.repeat(value.mean(axis=1), module.num_heads // module.num_kv_heads, axis=1)  # (B, N, H)
    mask = (attn_mask | ~visible[..., None])[:, None]  # (B, 1, T, S)
    # Masked cuDNN backward requires even query and key lengths.
    if query.shape[1] % 2:
        query = jnp.pad(query, ((0, 0), (0, 1), (0, 0), (0, 0)))  # (B, T+1, N, H)
        mask = jnp.concatenate((mask, mask[:, :, -1:]), axis=2)  # (B, 1, T+1, S)
    if key.shape[1] % 2:

        def pad(projected: jax.Array) -> jax.Array:  # (B, S, K, H)
            return jnp.pad(projected, ((0, 0), (0, 1), (0, 0), (0, 0)))  # (B, S+1, K, H)

        key, value = jax.tree.map(pad, (key, value))  # each: (B, S+1, K, H)
        mask = jnp.pad(mask, ((0, 0), (0, 0), (0, 0), (0, 1)))  # (B, 1, padded T, S+1)
    encoded = jax.nn.dot_product_attention(query, key, value, mask=mask, scale=1.0, implementation="cudnn")[
        :, : x.shape[1]
    ]  # (B, T, N, H)
    encoded = jnp.where(visible[..., None, None], encoded, empty_output[:, None])  # (B, T, N, H)
    output = module.attn_vec_einsum("BTNH,NHD->BTD", encoded)  # (B, T, D)
    return cache, output  # output: (B, T, D)


def _intercept_attention(
    next_fun: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any], context: InterceptorContext
) -> Any:
    if isinstance(context.module, gm.nn.Attention) and context.method_name == "__call__":
        return _cudnn_attention(context.module, *args, **kwargs)
    return next_fun(*args, **kwargs)


class Gemma3Carry(NamedTuple):
    cache: Cache
    lengths: jax.Array  # (B,) Next logical position, excluding padding.
    valid: jax.Array  # (B, C) Valid physical cache slots; padding gaps stay masked.


@dataclass(frozen=True)
class Gemma3LM:
    model: gm.nn.Transformer
    cache_length: int
    dtype: ComputeDtype = "float32"
    attention_backend: AttentionType = "xla"

    def __post_init__(self) -> None:
        if not 0 < self.cache_length <= 32768:
            raise ValueError("Gemma 3 270M prompt + completion budget must fit its 32768-token context")
        if self.attention_backend == "cudnn":
            if self.dtype != "bfloat16":
                raise ValueError("cuDNN attention requires dtype=bfloat16")
            if self.model.config.head_dim % 8:
                raise ValueError("cuDNN attention requires head_dim divisible by 8")
            if self.model.config.attn_logits_soft_cap is not None:
                raise ValueError("cuDNN attention does not support attention logit soft caps")

    def _apply(self, variables: Parameters, **kwargs: Any) -> gm.nn.Output:
        if self.attention_backend == "xla":
            return self.model.apply(variables, **kwargs)
        # Scoped to this call (and JAX tracing), without patching Gemma globally.
        with nn.intercept_methods(_intercept_attention):
            return self.model.apply(variables, **kwargs)

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
        output = self._apply(
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
        output = self._apply(
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
        output = self._apply(
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


def load_gemma3_checkpoint(
    source: str, *, cache_length: int, dtype: ComputeDtype, attention_backend: AttentionType = "xla"
) -> tuple[Gemma3LM, Parameters]:
    model = Gemma3LM(
        model=gm.nn.Gemma3_270M(dtype=jnp.float32),
        cache_length=cache_length,
        dtype=dtype,
        attention_backend=attention_backend,
    )
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
    # Keep Gemma's native format without Orbax's asynchronous signaling path.
    with ocp.Checkpointer(ocp.StandardCheckpointHandler()) as checkpointer:
        checkpointer.save(path.resolve(), args=ocp.args.StandardSave(params))
