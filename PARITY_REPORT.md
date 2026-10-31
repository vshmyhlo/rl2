# Qwen3 GPU parity audit — 2026-10-17

Fresh GPU measurements for `src/rl2/qwen3.py`; no CPU execution or CPU fallback.
Related reports: [HLO review](HLO_REVIEW.md), [performance](PERFORMANCE_REPORT.md).

**FP32 forward and all-parameter gradients pass. BF16 forward passes, but full
gradient parity fails the existing acceptance threshold on all three backends.**

## Environment and reference

One NVIDIA A100-SXM4-40GB; driver 580.178.04 (CUDA compatibility 13.0).
GPU access works outside the sandbox; both frameworks allocated and multiplied
GPU tensors before the audit.

| Component | Version |
| --- | --- |
| JAX / jaxlib / CUDA plugin | 0.11.2 |
| Python / Optax | 3.12.14 / 0.2.8 |
| Flax / Tokamax / safetensors | 0.12.10 / 0.0.15 / 0.8.0 |
| JAX CUDA runtime / cuDNN | 12.9.79 / 9.27.0.42 |
| PyTorch / Transformers | 2.9.1+cu128 / 4.57.1 |
| PyTorch CUDA / cuDNN | 12.8 / 9.10.2 |

Reference: locally inspected Transformers `Qwen3ForCausalLM`, eager attention,
checkpoint `Qwen/Qwen3-0.6B` at
`c1899de289a04d12100db370d81485cdf75e47ca`.
Architecture: 28 layers, hidden width 1024, MLP width 3072, 16 query/8 KV heads,
head width 128, vocabulary 151936, tied embeddings, split-half RoPE and Q/K RMSNorm.

Primary precision is BF16 checkpoint storage and computation. RMSNorm statistics,
RoPE angles, softmax and next-token loss use FP32; normalization casts back before
weight multiplication. JAX widens BF16 vocabulary logits to FP32 at its public API.
This audit disabled Torch autocast and TF32 and requested highest JAX matmul
precision. Current JAX operations use backend defaults; those overrides were removed.
Torch permits reduced-precision BF16 GEMM reductions by default. SiLU now uses
FP32 internal arithmetic, matching PyTorch, then returns the activation dtype.
FP32 rows are separate diagnostics. Forward uses evaluation mode; backward uses
training mode with the checkpoint's zero dropout. There is no Qwen3-specific
trainer in this repository, so backward uses mean next-token cross entropy.

## Forward results

Two real chat prompts ask for France's capital and `7 * 8`, with thinking disabled.
Batch 2, token lengths 24/25, left/right padding, three cached logit comparisons
and eight greedy tokens. All vocabulary logits at valid positions are compared;
padding-query outputs are unspecified by the API. JAX executes under `jax.jit`.
Each table entry is the worst metric across prefill and matching-history decode calls.

| Backend | Dtype | Max abs | Mean abs | RMSE | Normalized RMSE | Max relative | Mean relative |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XLA | FP32 | 6.67572e-5 | 8.33659e-6 | 1.15410e-5 | 2.88519e-6 | 0.0365973 | 1.73085e-5 |
| Tokamax Triton | FP32 | 8.58307e-5 | 7.53546e-6 | 9.85509e-6 | 2.33834e-6 | 0.0411305 | 1.22154e-5 |
| XLA | BF16 | 1.000000 | 0.131310 | 0.163015 | 0.040285 | 481.361 | 0.206938 |
| Tokamax Triton | BF16 | 1.125000 | 0.117707 | 0.151941 | 0.037548 | 590.359 | 0.212834 |
| cuDNN | BF16 | 0.968750 | 0.142780 | 0.184714 | 0.045647 | 602.455 | 0.260522 |

All five cases pass the existing regression envelope, with zero NaN/Inf values.
All greedy continuations match: `Paris` then EOS, and `7 × 8 = 56`.
This is approximate BF16 agreement, not bitwise equivalence or a guarantee for
other prompts. Relative errors divide by `max(abs(reference), 1e-3)`; normalized
RMSE divides by `max(reference RMS, 1e-3)`. Near-zero logits magnify relative error.

Unchanged forward limits: FP32 max/mean abs `< 2e-4 / 2e-5`; BF16 `< 1.5 / 0.2`,
exact generated tokens, and per-call argmax agreement ≥100% / ≥90%.
The BF16 limits are an established eager-versus-fused regression envelope,
motivated by the controls in [the earlier review](docs/qwen3.md#gpu-translation-review).
They are not strict elementwise all-close tolerances. JSON also records absolute
percentiles and the fraction exceeding an absolute diagnostic threshold.

## Checkpoint and backward results

The mapping test checks every stored value, shape and dtype against safetensors:
311 BF16 tensors, including the redundant tied head, become 310 unique parameters
with 596,049,920 elements. Matrix layouts remain `(out, in)`; no transposition is
performed during loading. Tied embeddings are shared, not duplicated. Existing
unit tests cover missing/duplicate weights, invalid shapes/dtypes, incorrect shard
indexes, and tied-weight mismatches before conversion.

Backward uses the 24-token capital-of-France prompt, batch 1, mean next-token
cross entropy on FP32 logits, and `jax.jit(jax.value_and_grad(...))`. Every unique
parameter is differentiated; the previous test covered only three tensors.
All 310 gradient shapes match, every gradient tensor is nonzero, and NaN/Inf
counts are zero. Model input IDs are integers, so input gradients are inapplicable.

| Backend / dtype | Max abs | Mean abs | RMSE | Global relative L2 | Max relative | Mean relative | Failing tensors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XLA FP32 | 3.38554e-5 | 7.76686e-9 | 2.67014e-8 | 4.28227e-6 | 0.003734 | 3.77518e-6 | 0 |
| Tokamax FP32 | 4.81606e-5 | 8.88513e-9 | 2.96223e-8 | 4.75071e-6 | 0.005874 | 4.35919e-6 | 0 |
| XLA BF16 | 0.593750 | 1.10518e-4 | 3.79067e-4 | 0.061408 | 62.9883 | 0.053401 | **10** |
| Tokamax BF16 | 0.531250 | 1.15674e-4 | 4.00565e-4 | 0.064890 | 70.1904 | 0.056880 | **42** |
| cuDNN BF16 | 0.500000 | 1.13290e-4 | 3.81127e-4 | 0.061741 | 55.6857 | 0.056021 | **44** |

Global relative L2 is `||JAX − Torch||₂ / max(||Torch||₂, 1e-3)` over all
parameters; means and RMSE weight every element equally. Per-tensor normalized
RMSE must remain below `1e-4` for FP32 and `0.1` for BF16. Loss differences must
remain below `1e-4 / 0.1`; these loss checks pass. No threshold was loosened.
Torch losses are 5.16977167 (FP32) and 5.15962362 (BF16). JAX BF16 losses are
5.17136335 / 5.15488195 / 5.15678263 for XLA / Tokamax / cuDNN.

| Configuration | Worst parameter | Normalized RMSE |
| --- | --- | ---: |
| XLA FP32 | layer 7 Q norm | 1.40127e-5 |
| Tokamax FP32 | layer 22 K norm | 1.21542e-5 |
| XLA BF16 | layer 22 K norm | 0.192014 |
| Tokamax BF16 | layer 24 K norm | 0.186338 |
| cuDNN BF16 | layer 22 K norm | 0.225539 |

Each `jax-grad-*.json` records all 310 per-parameter metrics and the ten worst
tensors. The failures remain active in the opt-in test suite; BF16 training
equivalence is **not established**.

## Discrepancy investigation and changes

- **SiLU precision corrected.** PyTorch's installed SiLU forward/backward
  definitions use FP32 internal arithmetic for BF16. On 36,864 real first-layer
  activations, the previous JAX expression had maximum forward/gradient errors
  `0.0078125 / 0.00390625`; the corrected operation matches both exactly. A tiny
  GPU regression fails before the fix and passes afterward. Weights and stored
  activations remain BF16. Full-model forward MAE decreases on every backend;
  full-gradient acceptance still fails.
- **Tokamax selection made explicit.** `implementation="triton"` prevents its
  automatic backend fallback. cuDNN FP32 remains explicitly unsupported.
- **Normalization investigated, candidate rejected.** A BF16 fusion barrier
  made an isolated RMSNorm exact on real inputs, with unchanged exact local input
  and weight gradients. However, full-model Tokamax relative gradient error rose
  from 6.49% to 7.14%, and its continuation changed from `×` to `*`. The barrier
  was removed; candidate measurements remain in `norm-barrier-candidate/`.
- **Within-PyTorch control.** BF16 SDPA versus eager, with identical real weights,
  input and loss, has global relative gradient error 6.17766%; 33 tensors exceed
  10%, and the worst reaches 19.3992%. This demonstrates substantial kernel
  sensitivity even within one framework. It does not excuse or convert the JAX
  failures into passes. Remaining differences involve fusion/rounding, attention
  accumulation and reductions; no single remaining backward operation is proven
  to explain all discrepancies.

The probe compares inputs/outputs at all 113 RMSNorm boundaries and separately
evaluates normalization on identical Torch inputs. Capture can change fusion,
so uninstrumented full-model tests remain the acceptance checks. Per-boundary
metrics are in `boundaries-*.json`; the intermediate regression checks pass.

| Normalization output | XLA FP32 normalized RMSE | XLA BF16 | Tokamax BF16 | cuDNN BF16 |
| --- | ---: | ---: | ---: | ---: |
| Layer 0 input norm | 6.06509e-8 | 0.002468 | 0.002468 | 0.002468 |
| Layer 0 Q norm | 8.13289e-8 | 0.002789 | 0.002789 | 0.002789 |
| Layer 0 post-attention norm | 6.12139e-7 | 0.010476 | 0.010501 | 0.010483 |
| Layer 14 input norm | 1.31495e-6 | 0.015267 | 0.019785 | 0.016696 |
| Layer 27 input norm | 1.39938e-6 | 0.023921 | 0.024377 | 0.023910 |
| Final norm | 1.63693e-6 | 0.027224 | 0.027050 | 0.027126 |

## Checks and limits

- Initial targeted GPU suite: **69 passed, 3 failed**; all failures were BF16
  full-gradient comparisons.
- Selected SiLU implementation: **10 affected unit tests passed**, then
  **7 integration tests passed, 3 failed** (the same three gradient configurations).
- Final intermediate refresh: **3 passed**. Error-metric unit test: **1 passed**.
- Rejected normalization experiment: 11 unit checks passed, but integration
  had 6 passes and 4 failures, including Tokamax generation. These are not final
  implementation results.
- Ruff formatting/checks pass. No full repository suite or end-to-end training
  was run. Numerical execution used GPU throughout; no configured audit cases
  were skipped. Unselected tests were deselected explicitly.
- Only one physical GPU is available. Multi-device execution, communication
  efficiency, long-context numerical behavior and broader prompt coverage remain
  unvalidated. This audit does not certify BF16 training parity.

## Reproduction

Use a CUDA PyTorch/Transformers reference environment and a fresh artifact directory.
`QWEN3_TORCH_PYTHONPATH` optionally points to isolated packages; the intermediate
probe requires those packages to be importable by the project's Python interpreter.

```bash
export QWEN3_CHECKPOINT=/path/to/pinned/snapshot
export QWEN3_TORCH_PYTHONPATH=/path/to/cuda-torch:/path/to/transformers
export QWEN3_AUDIT_DIR=/tmp/qwen3-new-audit
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run pytest tests/test_qwen3.py tests/test_qwen3_checkpoint.py -q
```

Current raw artifacts: `/tmp/qwen3-audit-current`; test log:
`/tmp/qwen3-audit-current-tests.log`. These are local temporary artifacts, not
checked-in checkpoint weights or portable reference fixtures.
