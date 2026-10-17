# Qwen3 in JAX

`rl2.qwen3` implements the dense [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
model with grouped-query attention, Q/K RMSNorm, RoPE, SwiGLU, and tied embeddings.
It loads Hugging Face safetensors directly into JAX. PyTorch is not needed at runtime.

```bash
uv run python -m rl2.qwen3 --prompt "Explain reinforcement learning in one sentence."
```

The first run downloads weights and `tokenizer.json` to the Hugging Face cache.
Use `--model /path/to/checkpoint` for local files, `--revision COMMIT` to pin a Hub
revision, or `--dtype float32` for float32 computation. The default is bfloat16.
The CLI formats a single user message with thinking disabled. `--raw` accepts an
already formatted prompt. Sampling supports temperature and top-k; temperature
zero selects greedy decoding. Generation stops at EOS or `--max-new-tokens`.

Select Tokamax attention with `--attention-backend tokamax`, or pass
`attention_backend="tokamax"` to `load_qwen3_checkpoint` or `Qwen3LM`.
Tokamax automatically selects an available kernel and supports full forward passes,
cached decoding, and gradients. The default backend remains `xla`.

Use `--attention-backend cudnn` (Python: `attention_backend="cudnn"`) for JAX's
built-in cuDNN attention on a compatible NVIDIA GPU. This backend requires
`dtype="bfloat16"` and a head dimension divisible by 8; Qwen3-0.6B satisfies both
with the defaults. Odd sequence/cache lengths are padded internally for cuDNN
gradients, without changing cache capacity or returned shapes.

```python
import jax
import jax.numpy as jnp

from rl2.qwen3 import load_qwen3_checkpoint

model, params = load_qwen3_checkpoint()
tokens = jnp.array([[9707, 0]], dtype=jnp.int32)  # (B=1, T=2)
output = jax.jit(model)(params, tokens)
print(output.logits.shape)  # (B, T, V) = (1, 2, 151936)

cache = model.init_cache(batch_size=1, capacity=128)
output = jax.jit(model)(params, tokens, cache=cache)
output = jax.jit(model)(params, jnp.array([[40]], jnp.int32), cache=output.cache)
```

Parameters retain their Hugging Face names and matrix layouts. The forward pass
supports `jax.grad`; request float32 weights when using float32 master parameters.
`attention_mask` is a `(B, T)` validity mask for the current input chunk. Default
RoPE positions count non-padding tokens; supply `(B, T)` `position_ids` to override.
Cached calls append the whole physical chunk, including padding. Direct callers
must keep the total appended width within cache capacity; `generate_tokens`
validates its prompt/completion budget. Padding-query logits are unspecified.

Supported scope: dense, bias-free, full-attention Qwen3 with standard RoPE and
SiLU, including tied or separate output embeddings. Scaled RoPE, sliding attention,
MoE, and quantized checkpoints are unsupported.
This module is standalone; it does not add a GRPO trainer configuration.

```bash
uv run pytest tests/test_qwen3.py
```

Tests include shapes, gradients, cache/padding equivalence, loading, and generation.
The checked-in tiny reference fixture comes from Hugging Face Transformers 4.57.1
with eager attention. To regenerate it in an environment with CPU PyTorch and
Transformers installed, run `python tests/qwen3_reference.py`.
The reference architecture is documented in
[Transformers' Qwen3 implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3/modeling_qwen3.py).
