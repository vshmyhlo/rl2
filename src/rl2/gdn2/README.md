# Gated DeltaNet-2 in JAX/Flax

This package implements the recurrent GDN-2 architecture from
[NVlabs/GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2/tree/a5552fe3c67e0ebc7ef1220df68ae8896ec62d56).
The reference revision is `a5552fe3c67e0ebc7ef1220df68ae8896ec62d56`, specifically
`lit_gpt/gdn2.py`, `lit_gpt/gdn2_ops/fused_recurrent_gdn2.py`, and the recurrent
blocks/configuration in `lit_gpt/model.py` and `lit_gpt/config.py`.
The adaptation retains the upstream [NVIDIA Source Code License-NC](LICENSE),
which limits use to noncommercial research or evaluation.

## Interfaces

- `GatedDeltaNet2Config`: frozen configuration for mixer dimensions and precision.
- `GatedDeltaNet2`: token mixer with independent channel-wise erase/write gates,
  channel-wise decay, causal depthwise q/k/v convolutions with SiLU, q/k L2
  normalization, grouped value heads, and SiLU-gated output RMS normalization.
- `GatedDeltaNet2Stack`: pre-RMSNorm residual mixer/SwiGLU blocks and a final RMSNorm.
- `GatedDeltaNet2LM`: the stack with token embeddings and an untied language-model head.
- `delta_rule_step` / `gated_delta_rule`: float32 recurrence primitives for a single
  step or a sequence. These accept already normalized/scaled queries and keys.

All sequence interfaces are **time-major**, matching the rest of `rl2`:
features `[time, batch, hidden_size]`, tokens `[time, batch]`. Each model returns
`(final_carry, output)`. Mixer/stack outputs have the input feature shape; LM
logits are `[time, batch, vocab_size]` in float32.

```python
import jax
import jax.numpy as jnp
from rl2.gdn2 import GatedDeltaNet2, GatedDeltaNet2Config, GatedDeltaNet2LM

config = GatedDeltaNet2Config(hidden_size=64, head_dim=16, num_heads=4)
mixer = GatedDeltaNet2(config)
x = jnp.ones((8, 2, 64))
variables = mixer.init(jax.random.key(0), x)
carry, y = jax.jit(mixer.apply)(variables, x)
carry, next_y = mixer.apply(variables, x[0], carry, method=mixer.step)

lm = GatedDeltaNet2LM(config, num_layers=2, intermediate_size=128, vocab_size=256)
tokens = jnp.zeros((8, 2), dtype=jnp.int32)
variables = lm.init(jax.random.key(1), tokens)
carry, logits = jax.jit(lm.apply)(variables, tokens)
```

Call `initial_carry(batch_size)` without parameter initialization to allocate
empty history, or omit carry on the first call. Carry is an ordinary JAX pytree;
save it with Flax serialization or pass it between arbitrary chunks. Empty
sequences return empty outputs and preserve carry. Gradients flow through carry;
use `jax.tree.map(jax.lax.stop_gradient, carry)` for truncated backpropagation.

Boolean `episode_starts[time,batch]` resets an example's recurrent and convolution
history before processing that token. Boolean `mask[time,batch]` skips padding,
preserves every carry leaf, and returns zero output. Padding takes precedence
over a simultaneous reset. The `step` methods accept the corresponding inputs
with the time axis removed. Validate external token IDs against the vocabulary
before passing them to the LM; masked token IDs are ignored.

## Recurrence and precision

With state `S[B,H,K,V]`, the update is

```text
S_decay = diag(exp(g)) @ S_previous
S       = (I - outer(k, b * k)) @ S_decay + outer(k, w * v)
output  = q @ S
```

The mixer normalizes q/k using `sqrt(sum(x**2) + 1e-6)` and scales q by
`head_dim**-0.5`. It computes `g = -exp(A_log) * softplus(f(x) + dt_bias)`.
Erase and write use separate sigmoid projections. `allow_neg_eigval=True`
doubles only the erase gate. Setting both gates to the same scalar recovers
the KDA rule; scalar decay additionally recovers Gated DeltaNet.

Parameters, convolution history, recurrent accumulation, and normalization
statistics use float32. Set `dtype=jnp.bfloat16` (or `jnp.float16`) in the
configuration for reduced-precision projections and mixer/stack outputs.
Exclude `A_log` and `dt_bias` from optimizer weight decay, as in upstream.
Mixer initialization follows the upstream Xavier gain, convolution bounds,
log-rate, and inverse-softplus step-size distributions. The stack/LM follows
the recurrent recipe's MLP residual scaling and embedding initialization;
random samples naturally differ between JAX and PyTorch.

The upstream recurrent 1.3B architecture can be configured with
`hidden_size=2304`, `head_dim=128`, `num_heads=16`, `num_layers=18`,
`intermediate_size=6208`, and `vocab_size=32000`. The upstream GPT configuration's
18 attention heads do not override its GDN-2 mixer's default 16 heads.

## Scope and verification

This is a portable `jax.lax.scan` implementation supporting JIT and autodiff on
JAX backends. It does not implement the upstream Triton chunkwise WY algorithm,
custom fused backward, hybrid sliding-window attention, checkpoint conversion,
or the pretraining data pipeline. Training retains scan intermediates; it is
intended as a usable reference implementation, not a throughput match for the
upstream GPU kernels. Float32 internal activations also differ from upstream
mixed-precision intermediate rounding.

`tests/test_gdn2.py` checks the update against independent NumPy dense transition
matrices, numerical gradients, NumPy reproduction of the mixer wiring, gate
limits, streaming/chunk equivalence, resets, padding, causality, and a tiny
bfloat16 language-model gradient/update. The tests do not execute the upstream
Triton kernels.

```bash
JAX_PLATFORMS=cpu uv run pytest tests/test_gdn2.py
```
