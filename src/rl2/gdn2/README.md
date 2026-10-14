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
- `GatedDeltaNet2Recurrent`: time-major stack adapter implementing
  `RecurrentSequenceModel` with per-example episode resets.
- `GatedDeltaNet2LM`: the stack with token embeddings and an untied language-model head.
- `delta_rule_step` / `gated_delta_rule`: float32 recurrence primitives for a single
  step or a sequence. These accept already normalized/scaled queries and keys.

The mixer and stack implement `ARSequenceModel` with **batch-first** features
`[batch, time, hidden_size]`. Call `model(x, x_len, carry=None)` with required
int32 prefix lengths `[batch]`, or `model.step(x, x_active, carry=None)` with
features `[batch, hidden_size]` and a required boolean active mask `[batch]`.
Both return `(final_carry, output)` with the input feature shape. Lengths must
lie in `[0, time]`; padding produces zero output and preserves carry, including
convolution histories. A zero length leaves that example's carry unchanged.

The autoregressive sequence interfaces use **batch-first** layouts: features are BTD
`[batch, time, dim]`, and headed activations are BTHD
`[batch, time, heads, dim]`. The LM accepts tokens `[batch, time]` and returns
float32 logits `[batch, time, vocab_size]`. The low-level sequence recurrence
and Triton wrappers use `[batch, time, heads, key_dim/value_dim]`; recurrent
state remains `[batch, heads, key_dim, value_dim]`. Single-step inputs omit
the time axis. The LM uses the same length/active-mask convention, but accepts
integer token IDs rather than inheriting the floating-input `ARSequenceModel` contract.

```python
import jax
import jax.numpy as jnp
from rl2.gdn2 import GatedDeltaNet2, GatedDeltaNet2Config, GatedDeltaNet2LM

config = GatedDeltaNet2Config(hidden_size=64, head_dim=16, num_heads=4)
mixer = GatedDeltaNet2(config)
x = jnp.ones((2, 8, 64))
x_len = jnp.array([8, 5], dtype=jnp.int32)
variables = mixer.init(jax.random.key(0), x, x_len)
carry, y = jax.jit(mixer.apply)(variables, x, x_len)
x_active = jnp.array([True, False])
carry, next_y = mixer.apply(variables, x[:, 0], x_active, carry, method=mixer.step)

lm = GatedDeltaNet2LM(config, num_layers=2, intermediate_size=128, vocab_size=256)
tokens = jnp.zeros((2, 8), dtype=jnp.int32)
variables = lm.init(jax.random.key(1), tokens, x_len)
carry, logits = jax.jit(lm.apply)(variables, tokens, x_len)
carry, next_logits = lm.apply(variables, tokens[:, 0], x_active, carry, method=lm.step)
```

Call `initial_carry(batch_size)` without parameter initialization to allocate
empty history, or omit carry on the first call. Carry is an ordinary JAX pytree;
save it with Flax serialization or pass it between arbitrary chunks. Empty
sequences return empty outputs and preserve carry. Gradients flow through carry;
use `jax.tree.map(jax.lax.stop_gradient, carry)` for truncated backpropagation.

For the LM, required int32 `x_len[batch]` selects each example's left-aligned valid
prefix. Its `step` method takes required boolean `x_active[batch]`. Padding and
inactive steps preserve every carry leaf and return zero logits. Zero lengths
preserve supplied carry or return fresh initial carry when omitted. Validate
external token IDs against the vocabulary before passing them to the LM;
padded and inactive token IDs are ignored.

For episode-based workloads, construct
`GatedDeltaNet2Recurrent(config, num_layers, intermediate_size)`. Its
`model(x, carry, episode_starts)` accepts **time-major** features
`[time, batch, hidden_size]` and boolean resets `[time, batch]`; its
`step(x, carry, episode_starts)` omits the time axis. Both return
`(updated_carry, output)` and require incoming carry, initialized with
`model.initial_carry(num_envs)`. A true reset clears every layer's state and
convolution history before processing that input. Time and batch dimensions
must be nonempty. Sequence calls scan the stack's step operation with shared
parameters, preserving gradients through carry except across episode resets.

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

## Paper-matched 370M checkpoint

`gdn2_370m()` constructs the architecture used by
[LLM-OS-Models2/gdn2-370m-fineweb-edu-30b-paper-matched](https://huggingface.co/LLM-OS-Models2/gdn2-370m-fineweb-edu-30b-paper-matched/tree/6f3458a5d7979ed3d85622b1e71e8eee25895742).
Its actual tensors have 380,603,648 parameters: hidden size 1024, **16 mixer
heads** of width 128, 16 layers, SwiGLU width 2048, and vocabulary size 32000.
The mixer's head count is independent of the GPT configuration's `n_head`.

The converted artifact is in `checkpoints/gdn2/paper-matched-30b/` at the
repository root. It contains float32 parameters in `params.npz` and a
`manifest.json` with architecture, source revision/checksum, training metadata,
and output checksum. Optimizer state is not included. Load it without PyTorch:

```python
from pathlib import Path
import jax
import jax.numpy as jnp
from rl2.gdn2.checkpoints import load_checkpoint

model, variables = load_checkpoint(Path("checkpoints/gdn2/paper-matched-30b"))
tokens = jnp.array([[1, 42, 100]], dtype=jnp.int32)  # batch, time
x_len = jnp.array([3], dtype=jnp.int32)
carry, logits = jax.jit(model.apply)(variables, tokens, x_len)
```

For reduced-precision inference, pass `dtype=jnp.bfloat16` to `load_checkpoint`;
stored parameters remain float32. The checkpoint does not include a tokenizer.
Loading verifies checksums, names, shapes, dtypes, and finite values without
initializing a second set of model weights. The returned variables contain NumPy
leaves that JAX transfers to the selected device when used.

To reproduce the conversion, install CPU PyTorch separately (only needed to read
the original `.pth`) and download the pinned source:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.6'
curl -L --fail 'https://huggingface.co/LLM-OS-Models2/gdn2-370m-fineweb-edu-30b-paper-matched/resolve/6f3458a5d7979ed3d85622b1e71e8eee25895742/model.pth' -o /tmp/gdn2-paper-matched.pth
uv run --no-sync python -m rl2.gdn2.checkpoints \
  --input /tmp/gdn2-paper-matched.pth \
  --output checkpoints/gdn2/paper-matched-30b
```

The CLI verifies the pinned source SHA256 and refuses to overwrite an existing
destination. It uses `torch.load(weights_only=True, mmap=True)` on CPU, without
executing downloaded Python code. Conversion transposes linear matrices and
permutes depthwise convolution axes; it does not quantize or retrain weights.
`convert_litgpt_state_dict(state, model)` exposes the generic NumPy mapping for
other explicitly configured models using the same LitGPT parameter layout.

## Generate from a prompt

From the repository root:

```bash
uv run --extra gdn2 python -m rl2.gdn2.generate \
  --prompt "The capital of France is" \
  --max-new-tokens 100
```

The default checkpoint is `checkpoints/gdn2/paper-matched-30b`. The optional
`gdn2` dependency installs SentencePiece; PyTorch is not needed for generation.
The script uses the 32K TinyLlama/Llama tokenizer from
[TinyLlama_v1.1 revision ff3c701f](https://huggingface.co/TinyLlama/TinyLlama_v1.1/tree/ff3c701f2424c7625fdefb9dd470f45ef18b02d6).
It caches `tokenizer.model` in `checkpoints/gdn2/`, downloading it on first use
and checking its pinned SHA256. Subsequent runs can operate offline.

Pass `--temperature 0` for greedy decoding. Sampling defaults to temperature
0.8, top-k 40, and seed 0; use `--top-k 0` to sample the full vocabulary and
`--seed` to choose a reproducible random sequence. Generation stops at EOS or
`--max-new-tokens`, whichever comes first. The prompt is encoded with BOS and
without EOS; `--no-bos` disables the prefix. An empty prompt generates from BOS.
These are plain-text completions, without a chat template.

`--checkpoint PATH` selects another converted checkpoint, `--tokenizer PATH`
selects a local SentencePiece model, and `--dtype bfloat16` enables reduced
precision. Tokenizer and checkpoint vocabulary sizes must match. The script
prints the decoded prompt and continuation to stdout; loading progress goes to
stderr. It transfers weights once, processes the prompt once, and then reuses
the recurrent state for each new token. The first run includes JAX compilation.
To select CPU explicitly, prefix the command with `JAX_PLATFORMS=cpu`.

## Optional Triton GPU backend

Install the CUDA and kernel extras, then select `triton` explicitly:

```bash
uv sync --extra cuda12 --extra gdn2 --extra gdn2-triton
uv run --no-sync python -m rl2.gdn2.generate \
  --backend triton --prompt "The capital of France is" \
  --temperature 0 --max-new-tokens 100
```

In Python, use `load_checkpoint(path, backend="triton")`,
`gdn2_370m(backend="triton")`, or pass `backend="triton"` to the mixer,
stack, or LM constructor. An existing module can use
`model.clone(backend="triton")`; parameters, carries, and checkpoint files
are identical between backends. The default `backend="jax"` remains portable
and does not import Triton or require the optional extra.

The GPU path provides:

- Fused recurrent state update and output for one-token decoding.
- The upstream 64-token chunked WY algorithm for multi-token calls, with custom
  reverse-mode gradients for queries, keys, values, decay, erase/write gates,
  and the incoming state. Gradients also flow between successive calls.
- Fused causal short convolution/SiLU and gated RMSNorm, with backward passes.

Prefix lengths and active masks preserve state for padding. The fused convolution
returns each example's history at its valid prefix length and zeros right-padded
outputs. Decoding can also be differentiated, using the chunked backward.

The kernels retain float32 internal activations and state, including when dense
projections use bfloat16. Float32 matrix products use `tf32x3`, rather than
single-pass TF32. JAX's dense projections still follow its matrix-precision
setting. For strict float32 comparisons, set
`JAX_DEFAULT_MATMUL_PRECISION=highest` before launching Python (or use
`jax.default_matmul_precision("highest")` around tracing). The real 370M
checkpoint matched the portable backend within `1.3e-5` on logits for a
65-token prefill plus matching greedy predictions and three cached decoding
steps on the A100. With JAX's default matrix precision, the same prefill had
maximum logit difference `0.008`, with all greedy predictions matching.

A small synchronized benchmark with that checkpoint, batch size 1, float32,
and `highest` precision measured median prefill times of 50.5 ms (JAX) versus
14.8 ms (Triton), after compilation. Decode times were 8.5 ms versus 9.0 ms;
this integration does not promise faster batch-1 decoding. Timing used three
prefill and five decode repetitions, so treat it as a spot check, not a broad
performance characterization.

Supported key dimensions are at most 256. This backend
supports first-order reverse-mode `jax.grad`/`jax.value_and_grad` and JIT on
NVIDIA GPUs; higher-order derivatives, JVP, extra `vmap` axes, and distributed
sharding have not been validated. Use the model's batch axis for batching.
Kernel sources, licenses, and adaptations are documented in
[_triton/README.md](_triton/README.md). PyTorch and the FLA Python package are
not runtime dependencies.

## Scope and verification

The portable backend uses `jax.lax.scan` and supports JIT and autodiff on JAX
backends. The optional Triton backend accelerates the recurrent architecture;
neither backend implements hybrid sliding-window attention or the pretraining
data pipeline. Float32 internal activations differ from upstream mixed-precision
intermediate rounding.

`tests/test_gdn2.py` checks the update against independent NumPy dense transition
matrices, numerical gradients, NumPy reproduction of the mixer wiring, gate
limits, streaming/chunk equivalence, prefix padding, causality, and a tiny
bfloat16 language-model gradient/update.

`tests/test_gdn2_triton.py` checks GPU outputs and gradients against the portable
backend, including chunk boundaries, partial chunks, incoming state gradients,
fused decoding, convolution histories, gated normalization, grouped value heads,
prefix padding, and float32/bfloat16 LM parameter gradients. It skips without
a GPU or the optional kernel dependency. Run it on an NVIDIA GPU:

```bash
JAX_PLATFORMS=cuda uv run --no-sync pytest tests/test_gdn2_triton.py
```

`tests/test_gdn2_checkpoints.py` covers strict conversion, checkpoint integrity,
save/load behavior, and converted-model logits against an independent PyTorch
CPU implementation. PyTorch-specific tests require the optional PyTorch install.

```bash
JAX_PLATFORMS=cpu uv run pytest tests/test_gdn2.py
```
