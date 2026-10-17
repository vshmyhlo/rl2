"""Flax Linen Qwen3 causal language model, with Hugging Face weight names."""

import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import tokamax
from flax import linen as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from tokenizers import Tokenizer

type Parameters = dict[str, jax.Array]
type Variables = dict[str, Parameters]
type AttentionBackend = Literal["xla", "tokamax", "cudnn"]
# B: batch, T: input tokens, S: key slots, C: cache capacity, L: layers,
# D: hidden size, H: head dimension, N: query heads, K: KV heads, V: vocabulary.


@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 40960
    tie_word_embeddings: bool = True
    eos_token_id: int = 151645
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        dimensions = (
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.max_position_embeddings,
        )
        if any(size <= 0 for size in dimensions):
            raise ValueError("Model dimensions must be positive")
        if self.num_attention_heads % self.num_key_value_heads or self.head_dim % 2:
            raise ValueError("Query heads must be divisible by KV heads, and head_dim must be even")
        if not self.rms_norm_eps > 0 or not self.rope_theta > 0:
            raise ValueError("rms_norm_eps and rope_theta must be positive")
        if not 0 <= self.eos_token_id < self.vocab_size:
            raise ValueError("eos_token_id must be in the vocabulary")
        if not self.initializer_range > 0:
            raise ValueError("initializer_range must be positive")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Qwen3Config":
        if values.get("model_type") != "qwen3":
            raise ValueError("Expected a dense Qwen3 checkpoint")
        if values.get("hidden_act", "silu") != "silu" or values.get("attention_bias", False):
            raise ValueError("Only SiLU and bias-free Qwen3 are supported")
        if values.get("rope_scaling") is not None or values.get("use_sliding_window", False):
            raise ValueError("Scaled RoPE and sliding-window attention are not supported")
        if any(kind != "full_attention" for kind in values.get("layer_types", [])):
            raise ValueError("Only full-attention layers are supported")
        return cls(**{field.name: values[field.name] for field in fields(cls) if field.name in values})

    def parameter_shapes(self) -> dict[str, tuple[int, ...]]:
        d, h = self.hidden_size, self.head_dim
        shapes = {"model.embed_tokens.weight": (self.vocab_size, d), "model.norm.weight": (d,)}
        if not self.tie_word_embeddings:
            shapes["lm_head.weight"] = (self.vocab_size, d)
        for layer in range(self.num_hidden_layers):
            prefix = f"model.layers.{layer}."
            shapes.update(
                {
                    prefix + "input_layernorm.weight": (d,),
                    prefix + "post_attention_layernorm.weight": (d,),
                    prefix + "self_attn.q_norm.weight": (h,),
                    prefix + "self_attn.k_norm.weight": (h,),
                    prefix + "self_attn.q_proj.weight": (self.num_attention_heads * h, d),
                    prefix + "self_attn.k_proj.weight": (self.num_key_value_heads * h, d),
                    prefix + "self_attn.v_proj.weight": (self.num_key_value_heads * h, d),
                    prefix + "self_attn.o_proj.weight": (d, self.num_attention_heads * h),
                    prefix + "mlp.gate_proj.weight": (self.intermediate_size, d),
                    prefix + "mlp.up_proj.weight": (self.intermediate_size, d),
                    prefix + "mlp.down_proj.weight": (d, self.intermediate_size),
                }
            )
        return shapes


class Qwen3Cache(NamedTuple):
    key: jax.Array  # (L, B, C, K, H)
    value: jax.Array  # (L, B, C, K, H)
    valid: jax.Array  # (B, C)
    lengths: jax.Array  # (B,) Logical positions, excluding padding.
    index: jax.Array  # () Next physical slot; callers must not exceed capacity.


class Qwen3Output(NamedTuple):
    logits: jax.Array  # (B, U, V) float32; U is the retained input width.
    cache: Qwen3Cache | None


def _rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:  # (..., D), (D,)
    normalized = x.astype(jnp.float32)  # (..., D)
    normalized *= jax.lax.rsqrt(jnp.mean(jnp.square(normalized), axis=-1, keepdims=True) + eps)  # (..., D)
    return normalized.astype(x.dtype) * weight.astype(x.dtype)  # (..., D)


def _rope(x: jax.Array, positions: jax.Array, theta: float) -> jax.Array:  # (B, T, heads, H), (B, T)
    frequencies = theta ** (-jnp.arange(0, x.shape[-1], 2, dtype=jnp.float32) / x.shape[-1])  # (H/2,)
    angles = positions[..., None].astype(jnp.float32) * frequencies  # (B, T, H/2)
    angles = jnp.concatenate((angles, angles), axis=-1)[:, :, None, :]  # (B, T, 1, H)
    rotated = jnp.concatenate((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), axis=-1)  # (B, T, heads, H)
    return x * jnp.cos(angles).astype(x.dtype) + rotated * jnp.sin(angles).astype(x.dtype)  # (B, T, heads, H)


def _attention(
    query: jax.Array,  # (B, T, N, H)
    key: jax.Array,  # (B, S, K, H)
    value: jax.Array,  # (B, S, K, H)
    mask: jax.Array,  # (B, 1, T, S)
    backend: AttentionBackend,
) -> jax.Array:
    if backend == "xla":
        with jax.default_matmul_precision("highest"):
            return jax.nn.dot_product_attention(query, key, value, mask=mask, implementation="xla")  # (B, T, N, H)

    width = query.shape[1]
    # Avoid empty fused softmaxes; preserve XLA's mean-value output for fully masked queries.
    visible = jnp.any(mask, axis=-1)  # (B, 1, T)
    empty = jnp.repeat(value.mean(axis=1), query.shape[2] // key.shape[2], axis=1)  # (B, N, H)
    mask = mask | ~visible[..., None]  # (B, 1, T, S)
    # Masked cuDNN backward requires even query and key lengths.
    if backend == "cudnn" and width % 2:
        query = jnp.pad(query, ((0, 0), (0, 1), (0, 0), (0, 0)))  # (B, T+1, N, H)
        mask = jnp.concatenate((mask, mask[:, :, -1:]), axis=2)  # (B, 1, T+1, S)
    if backend == "cudnn" and key.shape[1] % 2:

        def pad(projected: jax.Array) -> jax.Array:  # (B, S, K, H)
            return jnp.pad(projected, ((0, 0), (0, 1), (0, 0), (0, 0)))  # (B, S+1, K, H)

        key, value = jax.tree.map(pad, (key, value))  # Each (B, S+1, K, H).
        mask = jnp.pad(mask, ((0, 0), (0, 0), (0, 0), (0, 1)))  # (B, 1, padded T, S+1)
    if backend == "tokamax":
        head_dim = query.shape[-1]
        # Triton's tensor-core dot needs at least 16 contraction elements.
        if head_dim < 16:

            def pad_head(projected: jax.Array) -> jax.Array:  # (B, T/S, N/K, H)
                return jnp.pad(projected, ((0, 0), (0, 0), (0, 0), (0, 16 - head_dim)))  # (B, T/S, N/K, 16)

            query, key, value = jax.tree.map(pad_head, (query, key, value))
        attended = tokamax.dot_product_attention(
            query, key, value, mask=mask, scale=head_dim**-0.5, precision=jax.lax.Precision.HIGHEST
        )[..., :head_dim]  # (B, T, N, H)
    else:
        attended = jax.nn.dot_product_attention(query, key, value, mask=mask, implementation="cudnn")[
            :, :width
        ]  # (B, T, N, H)
    return jnp.where(visible[:, 0, :, None, None], attended, empty[:, None])  # (B, T, N, H)


class Qwen3LM(nn.Module):
    """Linen module with flat Hugging Face parameter names and explicit KV cache."""

    config: Qwen3Config = Qwen3Config()
    dtype: str = "bfloat16"
    param_dtype: jax.typing.DTypeLike = jnp.float32
    attention_backend: AttentionBackend = "xla"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("dtype must be float32 or bfloat16")
        if self.attention_backend == "cudnn":
            if self.dtype != "bfloat16":
                raise ValueError("cuDNN attention requires dtype=bfloat16")
            if self.config.head_dim % 8:
                raise ValueError("cuDNN attention requires head_dim divisible by 8")

    @nn.nowrap
    def init_cache(self, batch_size: int, capacity: int) -> Qwen3Cache:
        if batch_size <= 0 or not 0 < capacity <= self.config.max_position_embeddings:
            raise ValueError("Cache batch size must be positive and capacity must fit the model context")
        shape = (
            self.config.num_hidden_layers,
            batch_size,
            capacity,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        return Qwen3Cache(
            key=jnp.zeros(shape, self.dtype),  # (L, B, C, K, H)
            value=jnp.zeros(shape, self.dtype),  # (L, B, C, K, H)
            valid=jnp.zeros((batch_size, capacity), jnp.bool_),  # (B, C)
            lengths=jnp.zeros((batch_size,), jnp.int32),  # (B,)
            index=jnp.array(0, jnp.int32),  # ()
        )

    @nn.compact
    def __call__(
        self,
        tokens: jax.Array,  # (B, T)
        attention_mask: jax.Array | None = None,  # (B, T)
        position_ids: jax.Array | None = None,  # (B, T)
        cache: Qwen3Cache | None = None,
        logits_to_keep: int = 0,
    ) -> Qwen3Output:
        """Return logits and optional updated cache; compatible with jit and grad.

        tokens: (B, T) integer token IDs.
        attention_mask: (B, T) valid input tokens, including during cached calls.
        position_ids: (B, T) RoPE positions; defaults to cumulative non-padding positions.
        logits_to_keep: Static trailing output width; zero returns all input logits.
        Cache calls append T physical slots; total appended width must fit capacity.
        """
        config = self.config
        batch, width = tokens.shape
        if not 0 <= logits_to_keep <= width:
            raise ValueError("logits_to_keep must be between zero and the input width")
        capacity = config.max_position_embeddings if cache is None else cache.valid.shape[1]
        if not 0 < width <= capacity:
            raise ValueError("Input width must be positive and fit the context/cache capacity")
        params = {
            name: self.param(
                name,
                nn.initializers.ones_init() if "norm" in name else nn.initializers.normal(config.initializer_range),
                shape,
                self.param_dtype,
            )
            for name, shape in config.parameter_shapes().items()
        }  # Each leaf has its shape from parameter_shapes().
        if attention_mask is None:
            attention_mask = jnp.ones(tokens.shape, jnp.bool_)  # (B, T)
        attention_mask = attention_mask.astype(jnp.bool_)  # (B, T)
        offset = jnp.array(0, jnp.int32) if cache is None else cache.index  # ()
        lengths = jnp.zeros((batch,), jnp.int32) if cache is None else cache.lengths  # (B,)
        if position_ids is None:
            position_ids = jnp.maximum(lengths[:, None] + jnp.cumsum(attention_mask, axis=1) - 1, 0)  # (B, T)
        valid = attention_mask  # (B, S=T)
        if cache is not None:
            valid = jax.lax.dynamic_update_slice(cache.valid, attention_mask, (0, offset))  # (B, S=C)
        mask = (
            valid[:, None, None, :]
            & (jnp.arange(valid.shape[1])[None, :] <= offset + jnp.arange(width)[:, None])[None, None, :, :]
        )  # (B, 1, T, S)
        x = params["model.embed_tokens.weight"][tokens].astype(self.dtype)  # (B, T, D)

        def linear(inputs: jax.Array, name: str) -> jax.Array:  # (..., in)
            return jnp.matmul(
                inputs, params[name + ".weight"].astype(self.dtype).T, precision=jax.lax.Precision.HIGHEST
            )  # (..., out)

        def project(inputs: jax.Array, name: str) -> jax.Array:  # (B, T, D)
            return linear(inputs, name).reshape(batch, width, -1, config.head_dim)  # (B, T, N/K, H)

        def rotate(projected: jax.Array) -> jax.Array:  # (B, T, N/K, H)
            return _rope(projected, position_ids, config.rope_theta)  # (B, T, N/K, H)

        def update(old: jax.Array, new: jax.Array) -> jax.Array:  # (L, B, C, K, H), (B, T, K, H)
            return jax.lax.dynamic_update_slice(old, new[None], (layer, 0, offset, 0, 0))  # (L, B, C, K, H)

        for layer in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer}."
            normalized = _rms_norm(x, params[prefix + "input_layernorm.weight"], config.rms_norm_eps)  # (B, T, D)
            query, key, value = jax.tree.map(
                project,
                (normalized,) * 3,
                tuple(prefix + f"self_attn.{name}_proj" for name in ("q", "k", "v")),
            )  # (B, T, N, H), (B, T, K, H), (B, T, K, H).
            query = _rms_norm(query, params[prefix + "self_attn.q_norm.weight"], config.rms_norm_eps)  # (B, T, N, H)
            key = _rms_norm(key, params[prefix + "self_attn.k_norm.weight"], config.rms_norm_eps)  # (B, T, K, H)
            query, key = jax.tree.map(rotate, (query, key))  # (B, T, N/K, H)
            if cache is not None:
                updated_key, updated_value = jax.tree.map(
                    update,
                    (cache.key, cache.value),
                    (key, value),
                )  # Each (L, B, C, K, H).
                cache = cache._replace(key=updated_key, value=updated_value)
                key, value = cache.key[layer], cache.value[layer]  # Each (B, C, K, H).
            attended = _attention(query, key, value, mask, self.attention_backend)  # (B, T, N, H)
            x = x + linear(attended.reshape(batch, width, -1), prefix + "self_attn.o_proj")  # (B, T, D)
            normalized = _rms_norm(
                x, params[prefix + "post_attention_layernorm.weight"], config.rms_norm_eps
            )  # (B, T, D)
            hidden = jax.nn.silu(linear(normalized, prefix + "mlp.gate_proj")) * linear(
                normalized, prefix + "mlp.up_proj"
            )  # (B, T, intermediate_size)
            x = x + linear(hidden, prefix + "mlp.down_proj")  # (B, T, D)
        x = _rms_norm(x, params["model.norm.weight"], config.rms_norm_eps)  # (B, T, D)
        head = "model.embed_tokens" if config.tie_word_embeddings else "lm_head"
        logits = linear(x[:, -logits_to_keep:], head).astype(jnp.float32)  # (B, U, V)
        if cache is not None:
            cache = cache._replace(
                valid=valid,
                lengths=lengths + attention_mask.sum(axis=1, dtype=jnp.int32),  # (B,)
                index=offset + width,  # ()
            )
        return Qwen3Output(logits=logits, cache=cache)


def checkpoint_directory(source: str, revision: str | None = None) -> Path:
    path = Path(source).expanduser()
    if path.is_dir():
        return path
    return Path(
        snapshot_download(
            source,
            revision=revision,
            allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json", "tokenizer.json"],
        )
    )


def load_qwen3_checkpoint(
    source: str = "Qwen/Qwen3-0.6B",
    *,
    revision: str | None = None,
    dtype: str = "bfloat16",
    attention_backend: AttentionBackend = "xla",
) -> tuple[Qwen3LM, Variables]:
    """Return a Linen module and variables loaded directly from HF safetensors."""
    directory = checkpoint_directory(source, revision)
    config = Qwen3Config.from_dict(json.loads((directory / "config.json").read_text()))
    model = Qwen3LM(config=config, dtype=dtype, param_dtype=jnp.dtype(dtype), attention_backend=attention_backend)
    shapes = config.parameter_shapes()
    if config.tie_word_embeddings:
        shapes["lm_head.weight"] = shapes["model.embed_tokens.weight"]
    params: Parameters = {}
    locations: dict[str, str] = {}
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors weights in {directory}")
    for file in files:
        with safe_open(file, framework="flax") as tensors:
            for name in tensors.keys():  # noqa: SIM118 -- safe_open is not iterable.
                if name not in shapes or name in params:
                    raise ValueError(f"Unexpected or duplicate checkpoint weight: {name}")
                weight = tensors.get_tensor(name)  # Shape from parameter_shapes().
                if weight.shape != shapes[name] or weight.dtype not in (jnp.float32, jnp.float16, jnp.bfloat16):
                    raise ValueError(
                        f"Invalid shape or dtype for {name}: {weight.shape}, {weight.dtype}; expected {shapes[name]}"
                    )
                params[name] = weight  # Same shape and dtype as checkpoint tensor.
                locations[name] = file.name
    index = directory / "model.safetensors.index.json"
    if index.exists() and json.loads(index.read_text()).get("weight_map") != locations:
        raise ValueError("Checkpoint index does not match stored weight locations")
    if config.tie_word_embeddings:
        del shapes["lm_head.weight"]
    if missing := shapes.keys() - params.keys():
        raise ValueError(f"Missing checkpoint weights: {sorted(missing)}")
    if config.tie_word_embeddings and "lm_head.weight" in params:
        head = params.pop("lm_head.weight")  # (V, D)
        embedding = params["model.embed_tokens.weight"]  # (V, D)
        if head.dtype != embedding.dtype or not bool(jnp.array_equal(head, embedding)):
            raise ValueError("Tied lm_head.weight must equal model.embed_tokens.weight in storage dtype")
    params = jax.tree.map(lambda weight: weight.astype(dtype), params)  # Checkpoint parameter shapes.
    return model, {"params": params}


def generate_tokens(
    model: Qwen3LM,
    variables: Variables,
    prompt_ids: list[int],
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 20,
    seed: int = 0,
) -> list[int]:
    """Generate one continuation with a fixed-size KV cache, stopping at EOS."""
    if not temperature >= 0 or not 0 <= top_k <= model.config.vocab_size:
        raise ValueError("temperature must be nonnegative and top_k must fit the vocabulary")
    if max_new_tokens < 0 or not 0 <= seed < 2**32:
        raise ValueError("max_new_tokens must be nonnegative and seed must fit uint32")
    if not prompt_ids or any(token < 0 or token >= model.config.vocab_size for token in prompt_ids):
        raise ValueError("Prompt must contain valid vocabulary IDs")
    if len(prompt_ids) + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Prompt plus completion exceeds the model context")
    if max_new_tokens == 0:
        return []
    cache = model.init_cache(batch_size=1, capacity=len(prompt_ids) + max_new_tokens)
    forward = jax.jit(model.apply, static_argnames=("logits_to_keep",), donate_argnames=("cache",))
    output = forward(variables, jnp.array([prompt_ids], jnp.int32), cache=cache, logits_to_keep=1)

    @jax.jit
    def sample(logits: jax.Array, key: jax.Array) -> jax.Array:  # (V,), ()
        if temperature == 0:
            return jnp.argmax(logits).astype(jnp.int32)  # ()
        logits = (logits - logits.max()) / temperature  # (V,)
        if top_k:
            scores, indices = jax.lax.top_k(logits, top_k)  # Each (top_k,).
            return indices[jax.random.categorical(key, scores)]  # ()
        return jax.random.categorical(key, logits).astype(jnp.int32)  # ()

    key = jax.random.key(seed)  # ()
    generated: list[int] = []
    for index in range(max_new_tokens):
        key, sample_key = jax.random.split(key)  # Each ().
        token = int(sample(output.logits[0, -1], sample_key))
        generated.append(token)
        if token == model.config.eos_token_id or index + 1 == max_new_tokens:
            break
        output = forward(variables, jnp.array([[token]], jnp.int32), cache=output.cache, logits_to_keep=1)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3 text generation with JAX/Flax and a JIT-compiled KV cache.")
    parser.add_argument(
        "--model", default="Qwen/Qwen3-0.6B", help="Hugging Face model ID or local checkpoint directory"
    )
    parser.add_argument("--revision", help="Hugging Face checkpoint revision or commit")
    parser.add_argument("--prompt", required=True, help="User message, or literal model input with --raw")
    parser.add_argument("--raw", action="store_true", help="Use a raw completion prompt instead of single-turn chat")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum completion length (default: 128)")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature; 0 selects greedy decoding"
    )
    parser.add_argument(
        "--top-k", type=int, default=20, help="Sample from the top k tokens; 0 uses the full vocabulary"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--attention-backend", choices=("xla", "tokamax", "cudnn"), default="xla")
    args = parser.parse_args()
    if args.max_new_tokens < 0 or not args.temperature >= 0 or args.top_k < 0:
        parser.error("--max-new-tokens, --temperature, and --top-k must be nonnegative")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must fit uint32")
    if args.attention_backend == "cudnn" and args.dtype != "bfloat16":
        parser.error("--attention-backend cudnn requires --dtype bfloat16")
    if args.raw and not args.prompt:
        parser.error("--raw requires a nonempty --prompt")
    directory = checkpoint_directory(args.model, args.revision)
    model, variables = load_qwen3_checkpoint(str(directory), dtype=args.dtype, attention_backend=args.attention_backend)
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    # Official single-user chat template with enable_thinking=False.
    prompt = (
        args.prompt
        if args.raw
        else f"<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    generated = generate_tokens(
        model,
        variables,
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
