# Qwen3 in JAX

`rl2.qwen3` implements the dense [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
as a Flax Linen module with grouped-query attention, Q/K RMSNorm, RoPE, SwiGLU, and tied embeddings.
It loads Hugging Face safetensors directly into JAX. PyTorch is not needed at runtime.

## Inference

```bash
uv run python scripts/infer_qwen3.py --prompt "Explain reinforcement learning in one sentence."

# Greedy inference with fused attention on an NVIDIA GPU:
JAX_PLATFORMS=cuda uv run python scripts/infer_qwen3.py \
  --attention-backend cudnn --temperature 0 --max-new-tokens 32 \
  --prompt "What is the capital of France? Answer in one word."
# Paris
```

The equivalent module entry point is `uv run python -m rl2.qwen3` with the same arguments.

The first run downloads weights and `tokenizer.json` to the Hugging Face cache.
Use `--model /path/to/checkpoint` for local files, `--revision COMMIT` to pin a Hub
revision, or `--dtype float32` for float32 computation. The default is bfloat16.
The CLI formats a single user message with thinking disabled. `--raw` accepts an
already formatted prompt. Sampling supports temperature and top-k; temperature
zero selects greedy decoding. Generation stops at EOS or `--max-new-tokens`.
Only the decoded completion is printed to stdout. The first call includes JIT
compilation; subsequent token steps reuse the compiled decode function.
Run `uv run python scripts/infer_qwen3.py --help` for all options.

For a two-turn thinking and tool-calling example, run:

```bash
uv run python scripts/infer_qwen3.py --demo
```

The demo asks for `17 * 23`, then asks to multiply that result by `3`. It prints
thinking, calls a local `multiply` tool, and feeds the result back for an answer.
It uses Qwen3's [chat/tool format](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer_config.json),
retaining thinking within each tool round and removing it from earlier user turns.
Each response re-prefills the conversation; `--max-new-tokens` defaults to 1024
per response. Truncated responses and more than four tool rounds stop the demo.
Use `--demo --help` for model, attention, and sampling options.

Qwen3, its checkpoint loader and CLIs, and GRPO v2 default to BF16 computation
with Tokamax attention. Tokamax explicitly uses its Triton GPU kernel for full
forward passes, cached decoding, and gradients. Unsupported configurations fail without falling back
to XLA. Select `--attention-backend xla` (Python: `attention_backend="xla"`)
for CPU use or the portable reference implementation.

Use `--attention-backend cudnn` (Python: `attention_backend="cudnn"`) for JAX's
built-in cuDNN attention on a compatible NVIDIA GPU. This backend requires
`dtype="bfloat16"` and a head dimension divisible by 8; Qwen3-0.6B satisfies both
with the defaults. Odd sequence/cache lengths are padded internally for cuDNN
gradients, without changing cache capacity or returned shapes.

## BFCL evaluation

For tool-use evaluation of the Hugging Face checkpoint with vLLM, see the
[pinned BFCL V4 workflow](bfcl.md). It includes a GPU smoke check
and full single-turn/multi-turn evaluation through `rl2-bfcl`.

## Python API

```python
import jax
import jax.numpy as jnp

from rl2.qwen3 import load_qwen3_checkpoint

model, variables = load_qwen3_checkpoint()
tokens = jnp.array([[9707, 0]], dtype=jnp.int32)  # (B=1, T=2)
forward = jax.jit(model.apply)
output = forward(variables, tokens)
print(output.logits.shape)  # (B, T, V) = (1, 2, 151936)

cache = model.init_cache(batch_size=1, capacity=128)
output = forward(variables, tokens, cache=cache)
output = forward(variables, jnp.array([[40]], jnp.int32), cache=output.cache)
```

`Qwen3LM` uses standard Linen `init`/`apply`: `model.init(key, tokens)` creates
variables, and `model.apply(variables, tokens)` runs the model. Weights are registered
under `variables["params"]`, retaining their flat Hugging Face names and matrix
layouts. The loader returns this variables dictionary without initializing another
copy of the model. Fresh initialization uses float32 parameters by default;
`param_dtype` controls initialization, while `dtype` controls computation. Loaded
weights use the requested `dtype`. For float32 master parameters with bfloat16
computation, load with `dtype="float32"` then use `model.clone(dtype="bfloat16")`.
The module supports `jax.grad`, Flax `TrainState`, composition, and serialization.
Layers run through `jax.lax.scan` by default, including cached decoding, while
preserving the flat parameter layout. Pass `scan_layers=False` to `Qwen3LM` or
`load_qwen3_checkpoint` to unroll layers instead. Scanning also works with the
default per-layer gradient checkpointing.
`attention_mask` is a `(B, T)` validity mask for the current input chunk. Default
RoPE positions count non-padding tokens; supply `(B, T)` `position_ids` to override.
Cached calls append the whole physical chunk, including padding. Direct callers
must keep the total appended width within cache capacity; `generate_tokens`
validates its prompt/completion budget. Padding-query logits are unspecified.

For inference, pass `logits_to_keep=1` to project only the last token to vocabulary
logits; mark it static with `jax.jit(model.apply, static_argnames=("logits_to_keep",))`.
The default `0` returns all logits. `generate_tokens` uses this optimization and
donates its internal cache buffers between calls.

For data parallelism, replicate parameters and shard the batch using
`jax.jit(jax.shard_map(...))`; see [the HLO audit script](../scripts/audit_qwen3_hlo.py).
Tokamax 0.0.15 requires `check_vma=False` in this wrapper. Automatic `jit` sharding
alone introduces unnecessary attention all-gathers with this Tokamax version.

## Supported scope

Supported scope: dense, bias-free, full-attention Qwen3 with standard RoPE and
SiLU, including tied or separate output embeddings. Scaled RoPE, sliding attention,
MoE, and quantized checkpoints are unsupported.
This module is standalone; it does not add a GRPO trainer configuration.

## Tests

```bash
uv run pytest tests/test_qwen3.py
```

The checked-in tiny reference fixture comes from Hugging Face Transformers 4.57.1
with eager attention. To regenerate it in an environment with CPU PyTorch and
Transformers installed, run `python tests/qwen3_reference.py`.

### BF16 parity with the GRPO v2 vLLM server

`tests/test_qwen3_vllm.py` uses two GPUs and the actual `serve_vllm`,
`Learner.publish_weights`, and `Actor.update_weights` methods. Only Ray scheduling
is replaced with local threads. Weights travel through DLPack and packed NCCL;
the server runs its normal compiled BF16 inference and processed log probabilities.
Context length, concurrency, and memory allocation are reduced for the test.

```bash
CUDA_VISIBLE_DEVICES=0 QWEN3_VLLM_DEVICE=1 \
QWEN3_VLLM_TESTS=1 \
QWEN3_CHECKPOINT=/path/to/local/Qwen3-0.6B/snapshot \
QWEN3_VLLM_REPORT_DIR=/tmp/qwen3-vllm \
uv run --env-file nvida.env pytest --confcutdir=tests tests/test_qwen3_vllm.py -q -s
```

The checkpoint must already exist locally. Without `QWEN3_CHECKPOINT`, only the
small tied/untied models and operation tests run. Without `QWEN3_VLLM_TESTS=1`,
the suite skips. On hosts configured for an unavailable NCCL network plugin,
select `NCCL_NET=Socket` for this local test. TorchInductor needs a C++ compiler;
set `CXX` to its installed executable if necessary.

Coverage includes FP32 master weights rounded to BF16, two successive publications,
prefix-cache invalidation, unequal prompt lengths, two-turn transcripts with masked
user messages, batched generation, temperatures 1.0/0.7, and training's shifted
`token_log_probs`. Transcripts use the actual `Batcher.prepare_batch`, with the
production 256-token padding multiple for the checkpoint. Small models compare
every vocabulary probability; the real checkpoint compares sampled tokens and
the top ten alternatives. The same rollout
is replayed through JAX XLA/cuDNN and Tokamax Triton/cuDNN/XLA/chunked-XLA.
Tokamax Mosaic is included on SM90/SM100; A100 does not support it.

Reports retain package versions, server logs, exact rollout IDs/masks/probabilities,
per-backend errors, sampled GRPO approximate KL, and false clipping at a 0.2 clip
coefficient. A vLLM full-transcript replay measures its own prefill/decode variation.
`QWEN3_VLLM_EAGER=1` is available for diagnosis; the default tests compiled serving.
Reports also record optimization level, matmul precision, and relevant runtime
settings. The environment preset disables automatic NCCL profiler loading: the host's profiler
aborted when a second weight-transfer communicator initialized. Profiling can be
enabled explicitly with `NCCL_PROFILER_PLUGIN` when needed.

The operation tests compile vLLM's native normalization, SwiGLU, and RoPE operations,
including fused Q/K normalization plus rotation. They caught erased BF16 rounding
of the stored RoPE sine/cosine cache and a premature rounding path through BF16
rotation slicing. JAX now rounds stored serving values explicitly and widens before
rotation. Forcing rounding at every activation boundary instead matches eager CUDA
kernels, whose arithmetic differs from production TorchInductor fusion.

The real-checkpoint regression bounds are 0.35 maximum / 0.03 mean sampled log-probability
error, 0.10 maximum / 0.0025 mean alternative-token probability error, sampled
approximate KL below 0.0025, and false clipping at most 5%. Tokamax/Triton, the
training default, additionally requires **zero false clipping** on these rollouts.
Small-model limits are tighter. These are bounded BF16 agreement checks: kernel
reduction order and prefill/decode shapes still change rounding. All raw errors remain in the report,
including log-probability errors for very unlikely alternatives.

On A100 with Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`,
JAX 0.11.2, vLLM 0.27.0, Torch 2.13.0+cu130, and Tokamax 0.0.15, the
production-padding checks found alternative-token probability differences up to
0.086. A provisional 0.065 bound failed; the 0.10 bound above records this
remaining limitation rather than claiming close agreement for every candidate.
The workload has 128 sampled tokens across two publications and two temperatures;
passing it does not establish parity for arbitrary rollout lengths or checkpoints.

The final preset run on 2026-10-23 passed all eight integration/operation tests.
The table gives the worst result across the four rollout batches:

| Attention implementation | Max sampled log-probability error | Max batch mean error | False clipping |
| --- | ---: | ---: | ---: |
| XLA / Tokamax XLA | 0.189 | 0.0259 | 0% |
| cuDNN / Tokamax cuDNN | 0.293 | 0.0287 | 3.125% |
| Tokamax Triton | 0.189 | 0.0248 | 0% |
| Tokamax chunked XLA | 0.194 | 0.0191 | 0% |

Tokamax/Triton's largest alternative-token probability error was 0.073 and its
sampled approximate KL peaked at 0.00129. vLLM's own full-transcript replay
had up to 0.201 sampled log-probability error relative to its generation scores.
All five compiled-operation comparisons were exact for the tested inputs;
the 88 Qwen3 and GRPO update regression tests also passed on GPU.

Environment experiments loaded `nvida.env`, including `JAX_OPTIMIZATION_LEVEL=O1`,
90% preallocation, and the persistent compilation/autotuning caches. Fixed-rollout
replays and compiled HLO inspection showed sensitivity to GEMM selection and
fusion. Disabling `--xla_gpu_enable_triton_gemm` reduced the largest candidate
probability error in the integration run from 0.073 to 0.063, but increased
chunked-XLA's worst mean sampled log-probability error from 0.019 to 0.031, failing
its bound. It is not enabled in the preset. Independent vLLM server runs also
varied, so these integration differences cannot all be attributed to `O1`.
The model explicitly requests highest matmul precision; the preset's commented
TF32 option is not a BF16 parity fix.

## GPU translation review

See [the current audit](jax_torch_parity.md) for fresh GPU checkpoint checks,
full parameter-gradient comparisons, benchmarks, and backward HLO inspection.
The measurements below describe the earlier implementation review and explain
the existing numerical tolerances; current results are in the linked reports.

Reviewed on 2026-10-17 using an NVIDIA A100-SXM4-40GB. **37 focused unit tests
and 6 real-checkpoint integration tests passed on GPU with JIT enabled.**
Generated token IDs match PyTorch on the tested prompts across all supported
attention/dtype combinations. Logits are numerically close in FP32, not bitwise
identical; BF16 has appreciable rounding differences, quantified below.

### Reference and coverage

- Checkpoint: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca),
  revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- PyTorch `2.9.1+cu128`, Transformers `4.57.1`, CUDA eager attention, TF32 disabled,
  evaluation mode. [Original implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3/modeling_qwen3.py).
- JAX/jaxlib `0.11.2`, Flax `0.12.10`, Tokamax `0.0.15`, safetensors `0.8.0`.
- Real checkpoint weights, real tokenizer, non-thinking single-user chat template.
  Batch two, token lengths 24/25, both left and right padding, explicit matching
  logical RoPE positions, prefill plus three cached logit comparisons and eight
  greedy tokens. Padding-query logits are excluded as documented by the API.
- Unit tests additionally cover tied/untied heads, PyTorch gradients, Flax
  initialization/composition/serialization, fully masked rows, cache chunking,
  causality, shape boundaries, loading, sampling, and cuDNN backward on GPU.
- Reviewed Q/K RMSNorm, split-half RoPE, GQA grouping/scaling, residuals, SwiGLU,
  weight layouts, tied embeddings, and padding/cache positions against HF.
  Model execution is JAX/Flax; Tokamax uses JAX Pallas/Triton and cuDNN uses JAX's
  native primitive. PyTorch exists only in reference-generation test scripts.
  No Torch calls, host callbacks, or host array conversions occur in the forward.

Both prompts use
`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`.

| Prompt | PyTorch and all five JAX configurations |
| --- | --- |
| What is the capital of France? Answer in one word. | `Paris` then EOS; IDs `[59604, 151645]` |
| Compute 7 * 8. Answer with just the number. | `7 × 8 = 56`; IDs `[22, 24768, 220, 23, 284, 220, 20, 21]` |

### Measured logit errors against PyTorch eager

Each cell is the worst value across padding sides; decode also takes the worst
of three steps. MAE is mean absolute error over all vocabulary logits. Prefill
argmax agreement includes every non-padding prompt position, not just the final
one. Every measured decode step and generated continuation matched greedy IDs.

| JAX backend | Dtype | Prefill max abs | Prefill MAE | Decode max abs | Decode MAE | Prefill argmax |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| XLA | FP32 | 0.000068665 | 0.000005444 | 0.000059605 | 0.000008195 | 100% |
| Tokamax | FP32 | 0.000080109 | 0.000006115 | 0.000056982 | 0.000009122 | 100% |
| XLA | BF16 | 1.062500 | 0.077328 | 1.015625 | 0.157487 | 93.88% |
| Tokamax | BF16 | 1.156250 | 0.076656 | 1.000000 | 0.137006 | 100% |
| cuDNN | BF16 | 0.937500 | 0.078331 | 1.265625 | 0.166432 | 97.96% |

Worst prefill/decode RMS errors, respectively: XLA FP32 `7.688e-6 / 1.050e-5`,
Tokamax FP32 `8.759e-6 / 1.194e-5`, XLA BF16 `0.111965 / 0.206400`,
Tokamax BF16 `0.111723 / 0.166475`, cuDNN BF16 `0.110755 / 0.226989`.
cuDNN FP32 is unsupported by this module and is intentionally not tested.

BF16 eager and fused operations round intermediates differently. For context,
**PyTorch BF16 eager versus its own BF16 SDPA** reached max error `1.402344`
and MAE `0.215118`; PyTorch BF16 eager versus FP32 reached max `0.941542`
and MAE `0.133292`. Both controls produced the same generated tokens. These
controls support interpreting the BF16 discrepancies as numerical variation,
not exact-logit parity. They do not guarantee identical generation on other inputs.

Regression limits per measured call are max/MAE `< 2e-4 / 2e-5` for FP32 and
`< 1.5 / 0.2` for BF16, with exact generated-token assertions. Prefill argmax
agreement must be 100% for FP32 and at least 90% for BF16. Initial provisional
BF16 limits `0.5 / 0.05` failed; the final limits reflect these measured controls.
Per-padding/per-step errors and reference metadata are emitted as JSON by pytest.

### Corrections and efficiency findings

1. **FP32 attention precision:** explicitly request highest precision in XLA and
   Tokamax. Before this fix, XLA's real-checkpoint maximum error was `0.039366`;
   it is now `0.000068665`. Linear projections already requested highest precision.
2. **Tokamax small-head GPU bug:** tensor-core attention with head width 4/8
   produced large errors (tiny-model logit error up to `0.5`), hidden by CPU tests.
   Padding heads below 16 with zeros, retaining the original scale and trimming
   the result, fixes the regression. Qwen3-0.6B's width 128 needs no padding.
   The existing masked-gradient, cache, and checkpoint-loading tests now pass on GPU.
3. **Vocabulary projection:** `logits_to_keep=1` slices hidden states before the
   output projection. Generation uses it and donates its internal KV cache.
   At 128 prompt tokens, output memory falls from `74.1875 MiB` to `0.5796 MiB`
   per replica and the vocabulary projection computes one row instead of 128.
   Updating cache slots directly instead of stacking updated layers cuts compiled
   decode temporary memory from `110.04 MiB` to `4.06 MiB` (BF16 cuDNN, batch one,
   capacity 1,024). Donation additionally aliases `112 MiB` of cache storage.
4. **Data parallelism:** automatic sharding around Tokamax introduced **28
   all-gathers**, gathering Q/K/V/masks and redundantly computing global attention.
   Explicit `jax.shard_map` removes all of them. The audit script defaults to
   this setup; ordinary auto-sharded `jit` alone still has this upstream limitation.
   Tokamax 0.0.15 needs `check_vma=False` because its Pallas outputs lack manual-axis
   annotations. XLA and cuDNN had zero collectives with either approach.

### Compiled data-parallel HLO

Compiled the full 28-layer architecture for **two A100 targets**, replicated BF16
parameters, global batch two, batch-sharded inputs/outputs. Only **one physical
A100** was attached: two-device compilation used its target configuration and
compile-only devices. This verifies SPMD lowering, not multi-GPU execution,
communication latency, or scaling. Parameters are runtime arguments, not embedded
checkpoint constants. The script saves StableHLO, optimized HLO, and memory JSON.

| Backend, explicit sharding | Attention calls | Collectives | Temp MiB, 1,024 tokens | Output MiB, last token |
| --- | --- | ---: | ---: | ---: |
| XLA | Native dot/softmax fusions | 0 | 106.25 | 0.5796 |
| Tokamax | 28 `__gpu$xla.gpu.triton` | 0 | 31.2510 | 0.5796 |
| cuDNN | 28 `__cudnn$fmhaScaleBiasSoftmax` | 0 | 32.2576 | 0.5796 |

Memory figures are compiler estimates per device, excluding allocator overhead;
replicated parameter/input arguments add approximately 1.11 GiB per device.
No all-reduce, all-gather, reduce-scatter, all-to-all, collective-permute, host
callbacks, or standalone copy instructions remain in the explicit-sharding HLO.
Expected vocabulary gather, reductions for RMSNorm/softmax, and layout operations
remain. XLA has unfused quadratic attention intermediates; use a fused backend
for longer sequences. RoPE trigonometry is shared in XLA HLO but duplicated inside
Q/K producer fusions for fused attention; this is a possible profiling target.

Remaining limits: the model still builds a dense causal mask; decoding attends
over fixed cache capacity, including unused slots. Generation synchronizes once
per token for sampling/EOS and is not a throughput-optimized serving engine.
Long-context accuracy, multi-GPU runtime behavior, and real-checkpoint training
were not established by this review. Tokamax emits a Pallas Triton deprecation
warning on JAX 0.11.2; rerun the GPU checks when upgrading either dependency.

### Reproduce

Keep PyTorch reference dependencies separate from the JAX runtime:

```bash
uv venv /tmp/qwen3-torch
uv pip install --python /tmp/qwen3-torch/bin/python torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python /tmp/qwen3-torch/bin/python transformers==4.57.1 numpy
export QWEN3_CHECKPOINT=/path/to/c1899de289a04d12100db370d81485cdf75e47ca
export QWEN3_TORCH_PYTHON=/tmp/qwen3-torch/bin/python
export QWEN3_AUDIT_DIR=/tmp/qwen3-audit
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run pytest tests/test_qwen3.py tests/test_qwen3_checkpoint.py -q -s

for backend in xla tokamax cudnn; do
  uv run python scripts/audit_qwen3_hlo.py \
    --checkpoint "$QWEN3_CHECKPOINT" --output /tmp/qwen3-hlo \
    --attention-backend "$backend" --devices 2 --batch 2 --length 1024 --logits-to-keep 1
done
```

Without `QWEN3_CHECKPOINT`, expensive checkpoint tests skip and perform no network
access. With it, GPU and enabled JIT are mandatory. The reference script refuses
to fall back to CPU. Use a fresh audit directory for a new checkpoint/reference
version; saved reference metadata must match the checkpoint path and dtype.
To reproduce the Tokamax collective regression, add `--sharding auto`; the audit
writes the HLO then fails on detected collectives. For the BF16 control, run
`tests/qwen3_checkpoint_reference.py` with `--attention-backend sdpa` in the Torch
environment and compare its NPZ with the default eager reference.

Run artifacts from this review are in `/tmp/qwen3-audit`; large checkpoints, NPZ
references, and HLO files are intentionally not checked into the repository.
