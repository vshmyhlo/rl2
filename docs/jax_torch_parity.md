# Qwen3 translation verification — 2026-10-17

**GPU access is available outside the execution sandbox.** Inside the sandbox,
`nvidia-smi` fails and JAX reports `CUDA_ERROR_NO_DEVICE`; elevated execution
detects an A100 and enables CUDA in both frameworks. Fresh GPU validation is in
progress. The CPU results below remain supplemental diagnostics.

## Environment and reference

- Checkpoint: `Qwen/Qwen3-0.6B`, revision
  `c1899de289a04d12100db370d81485cdf75e47ca`, already cached locally.
- Dense 28-layer model: hidden width 1,024, MLP width 3,072, 16 query / 8 KV heads,
  head dimension 128, vocabulary 151,936, tied embeddings, standard split-half
  RoPE, Q/K RMSNorm, SiLU, causal attention, dropout zero.
- Hardware: NVIDIA A100-SXM4-40GB (one physical GPU), driver 580.178.04;
  Intel Xeon CPU @ 2.20 GHz, 12 logical CPUs. GPU access requires elevated execution.
  JAX/jaxlib 0.11.2, Flax 0.12.10, Tokamax 0.0.15, safetensors 0.8.0;
  CPU reference PyTorch 2.14.1+cpu, Transformers 4.57.1. GPU reference PyTorch
  2.9.1+cu128 with cuDNN 9.10.2; JAX CUDA runtime 12.9.79 / cuDNN 9.27.0.42.
- Reference implementation inspected locally:
  `transformers/models/qwen3/modeling_qwen3.py`, including RMSNorm, RoPE,
  eager attention, residuals, and MLP. Reference execution uses evaluation mode,
  no autocast, no stochastic operations, and disables Torch TF32.
- Separate **earlier GPU artifacts** in `/tmp/qwen3-audit` identify one NVIDIA
  A100-SXM4-40GB, PyTorch 2.9.1+cu128, CUDA 12.8, and Transformers 4.57.1.
  Their measurements below were inspected, not rerun in this environment.

## Checkpoint mapping and precision

The real snapshot contains **311 BF16 tensors**, including a redundant tied head.
The loader verifies the alias and returns **310 unique parameters / 596,049,920
elements**, all BF16 by default; every shape matches the configured architecture.
There are no missing or unexpected weights. Linear matrices retain HF's
`(out, in)` layout and are transposed at multiplication; embeddings retain `(V, D)`.
RoPE frequencies are regenerated nonpersistent state.

Previously the loader skipped `lm_head.weight` before validating anything about
it. It now validates shape, dtype, duplicates, and exact equality with the tied
embedding **before conversion**, and checks any safetensors shard index against
actual tensor locations. Regression cases include mismatches that disappear on
BF16 conversion. Explicit `dtype="float32"` remains a diagnostic conversion.

BF16 is the checkpoint's declared and stored precision. Runtime projections,
residuals and MLP use BF16; RMSNorm statistics and RoPE angles use FP32. Both
frameworks cast normalized values back before multiplying normalization weights.
Torch eager rounds attention scores/scaling and probabilities to the activation
dtype around its FP32 softmax; native JAX/fused kernels have different fusion and
accumulation boundaries. Matmuls request JAX's highest precision. The JAX API
widens projected logits to FP32; HF inference returns BF16 logits. Widening does
not restore precision lost in the projection. Gradient losses use FP32 logits.
FP16 execution remains unsupported by this module; it was not added or tested.

## Forward parity

Two real chat prompts ask for France's capital and `7 * 8`, with thinking disabled.
Token lengths are 24/25. Tests cover batch two, left/right padding, cached
single-token decoding, and eight greedy tokens; gradient/probe runs use batch one,
length 24. Padding-query logits are excluded. All JAX comparisons use JIT.

**New CPU measurements**, worst prefill metric across padding sides:

| Configuration | Dtype | Max absolute | Mean absolute | RMSE | Relative RMSE | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Torch CPU vs JAX XLA CPU | FP32 | 9.3460e-5 | 7.3678e-6 | 1.0594e-5 | 2.7875e-6 | Pass, including decode/generation |
| Torch CPU vs JAX XLA CPU | BF16 | 1.44238 | 0.078641 | 0.117818 | 0.030943 | **Fail: different generated token** |

BF16 JAX produces `7 × 8 = 56`; CPU Torch produces `7 * 8 = 56` (second token
24768 versus 353). Both produce `Paris` then EOS on the first prompt. Logits are
compared during decoding only while token histories agree; differing continuations
still fail the test, and diagnostics are written before assertions.

| CPU prefill | Absolute p50 / p95 / p99 | Max relative | Fraction outside element tolerance |
| --- | --- | ---: | ---: |
| FP32 | 4.7684e-6 / 2.3365e-5 / 3.5226e-5 | 0.050604 | 0% at 2e-4 |
| BF16 | 0.0546875 / 0.25 / 0.421875 | 780.64 | 0.47113% at 0.5 |

Relative error divides by `max(abs(reference), 1e-3)`; normalized RMSE divides by
`max(reference RMS, 1e-3)`. Near-zero logits explain the large maximum relative
errors. Percentiles and exceedance rates are diagnostic, not relaxed assertions.
Existing per-call limits remain max/mean `< 2e-4 / 2e-5` for FP32 and `< 1.5 / 0.2`
for BF16, with exact generated-token matching and prefill argmax agreement
≥100% / ≥90%. The earlier BF16 limits were motivated by Torch eager/SDPA controls
documented in [the original GPU review](qwen3.md#gpu-translation-review); they
describe a regression envelope, not strict BF16 equivalence.

**Earlier GPU artifacts**, worst metrics across prefill and matching decode calls:

| Configuration | Dtype | Max absolute | Mean absolute | RMSE | Relative error | Recorded result |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Torch vs JAX XLA | FP32 | 6.8665e-5 | 8.1955e-6 | 1.0496e-5 | Not recorded | Pass |
| Torch vs JAX Tokamax | FP32 | 8.0109e-5 | 9.1218e-6 | 1.1938e-5 | Not recorded | Pass |
| Torch vs JAX XLA | BF16 | 1.0625 | 0.157487 | 0.206400 | Not recorded | Pass with rounding differences |
| Torch vs JAX Tokamax | BF16 | 1.15625 | 0.137006 | 0.166475 | Not recorded | Pass with rounding differences |
| Torch vs JAX cuDNN | BF16 | 1.265625 | 0.166432 | 0.226989 | Not recorded | Pass with rounding differences |

The current suite requires GPU by default. CPU diagnostics require an explicit
override and skip fused GPU comparisons; they never silently substitute XLA for
Tokamax/cuDNN. cuDNN FP32 remains unsupported.

## Gradients and discrepancy investigation

New JIT comparisons differentiate mean next-token cross entropy through the real
checkpoint, selecting the first Q normalization, last MLP down projection and final
normalization. Other weights stay runtime arguments. These cover both early and
late backward paths, but do not establish every parameter's gradient parity.

| CPU gradient | Dtype | Max absolute | Mean absolute | RMSE | Relative RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| First Q norm | BF16 | 0.21875 | 0.002540 | 0.019470 | 0.065165 |
| Last MLP down projection | BF16 | 0.056152 | 7.5457e-5 | 3.0326e-4 | 0.038230 |
| Final norm | BF16 | 0.001709 | 1.2709e-4 | 1.7503e-4 | 0.044990 |
| First Q norm | FP32 | 1.6928e-5 | 2.6345e-7 | 1.5685e-6 | 5.0452e-6 |
| Last MLP down projection | FP32 | 5.0485e-5 | 3.7378e-8 | 2.3081e-7 | 2.9114e-5 |
| Final norm | FP32 | 8.7358e-6 | 1.1396e-7 | 4.4438e-7 | **1.1326e-4** |

BF16 loss is 5.176787 (JAX) versus 5.140510 (Torch); FP32 is 5.169775 versus
5.169755. BF16 passes the diagnostic 10% normalized-error and 0.1 loss limits;
these are deliberately labelled diagnostics rather than precise training parity.
FP32 **fails** the chosen 1e-4 normalized-gradient limit on final normalization.
No thresholds were increased after these measurements.

The reproducible normalization-boundary probe found:

- BF16 embedding/input RMSNorm agree exactly. The first query projection differs
  by max 0.001953125 before Q normalization; Q-norm output differs by 0.03125.
  After the first attention residual, max error is 0.01953125. At final norm
  output, relative RMS error is 2.77%. Differences accumulate through layers.
- FP32 starts with a 9.54e-7 input-normalization difference; final normalized
  activations have relative RMS error 2.42e-6. This supports accumulation/reduction
  ordering as a source of the small remaining gradient discrepancy, but does not
  prove a single backward operation is responsible.
- Capturing intermediates can alter compiler fusion. The probe is diagnostic;
  the full-model tests above remain the acceptance checks. No precision promotion,
  numerical workaround, or architectural change was made to force agreement.

## Compilation, memory and performance

Compiled full 28-layer BF16 StableHLO and optimized HLO with an explicit one-device
`data` mesh, replicated weights, batch-sharded inputs, `shard_map`, batch 1 and
length 24. Forward and full backward executed with real weights; a functional SGD
step compiled for the full model and executed in the tiny automated audit test.

| CPU computation | All-reduces | Output MiB | Temporary MiB | Median latency | Input tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forward | 0 | 13.910 | 7.002 | 153.63 ms | 156.22 |
| Loss + full backward | 2 | 1136.877 | 3675.129 | 2632.40 ms | 9.12 |
| Functional SGD step | 2 | 1136.877 | 3675.129 | Not measured | Not measured |

Arguments occupy 1136.875 MiB. Memory values are compiler estimates, not allocator
or peak GPU measurements. Timings use two warmups, five synchronized calls and
exclude compilation; input repeats a real sentence to the selected length. They
are CPU observations, not GPU performance claims or a comparison with Torch.

The backward all-reduces aggregate replicated-weight gradients; their groups have
one member here, so no physical inter-device transfer occurs. No all-gathers or
host callbacks were found. CPU forward/backward HLO contains 168/898 copies and
1610/5447 conversions, including layout work and BF16/FP32 transitions. Backward
gradient intermediates are expensive; optimizing these CPU-specific graphs was
not justified as a GPU improvement. The dense attention mask and XLA's quadratic
attention remain longer-context concerns. The earlier GPU review found Tokamax
auto-sharding all-gathers removed by explicit `shard_map`; this wrapper is retained.

## Reproduction and artifacts

Use a fresh audit directory when changing checkpoints or reference versions.
Install Torch/Transformers in a separate reference environment as described in
[qwen3.md](qwen3.md#reproduce). GPU defaults remain mandatory unless overridden.

```bash
export QWEN3_CHECKPOINT="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
export QWEN3_TORCH_PYTHON=/tmp/qwen3-torch/bin/python
export QWEN3_AUDIT_DIR=/tmp/qwen3-new-gpu-audit
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/tmp/qwen3-review-cache
unset QWEN3_TEST_PLATFORM
uv run pytest tests/test_qwen3.py tests/test_qwen3_checkpoint.py -q -s

for backend in xla tokamax cudnn; do
  for mode in forward backward train; do
    uv run python scripts/audit_qwen3_hlo.py --checkpoint "$QWEN3_CHECKPOINT" \
      --output /tmp/qwen3-new-gpu-hlo --attention-backend "$backend" \
      --devices 1 --batch 1 --length 128 --mode "$mode" --execute
  done
done
# With two physical GPUs, use --devices 2 --batch 2.
```

Exact current-environment CPU diagnostics (expected: two passing and two failing
real-checkpoint XLA cases):

```bash
export UV_CACHE_DIR=/tmp/rl2-uv-cache UV_OFFLINE=1
export JAX_PLATFORMS=cpu QWEN3_TEST_PLATFORM=cpu
unset QWEN3_TORCH_PYTHON
export QWEN3_TORCH_PYTHONPATH=/tmp/qwen3-reference
export QWEN3_AUDIT_DIR=/tmp/qwen3-review-cpu
uv run pytest tests/test_qwen3_checkpoint.py -k xla -q -s
PYTHONPATH=/tmp/qwen3-reference USE_TF=0 uv run python tests/qwen3_precision_probe.py \
  --checkpoint "$QWEN3_CHECKPOINT" --device cpu \
  --output /tmp/qwen3-review-cpu/norm-boundaries.json
# Add --dtype float32 for the FP32 probe.
for mode in forward backward; do
  uv run python scripts/audit_qwen3_hlo.py --checkpoint "$QWEN3_CHECKPOINT" \
    --output /tmp/qwen3-review-benchmark --platform cpu --devices 1 --batch 1 \
    --length 24 --mode "$mode" --execute --warmup 2 --runs 5
done
uv run python scripts/audit_qwen3_hlo.py --checkpoint "$QWEN3_CHECKPOINT" \
  --output /tmp/qwen3-review-hlo --platform cpu --devices 1 --batch 1 --length 24 --mode train
```

Current NPZ/JSON and probe results: `/tmp/qwen3-review-cpu`; compiler dumps:
`/tmp/qwen3-review-hlo`; benchmarks: `/tmp/qwen3-review-benchmark`. Large artifacts
are deliberately outside the repository. The initial focused run had 49 passing
and 7 skipped tests; six added audit execution/validation tests subsequently
passed. The real-checkpoint diagnostic run had two passes and the two numerical
failures above. Ruff formatting and checks cover all changed Python files.

**Status:** strict real-checkpoint loading, CPU FP32 forward parity, CPU JIT,
full forward/backward compilation and CPU execution are verified. BF16 forward
and FP32 gradient discrepancies remain. Current GPU backend/gradient validation,
GPU latency/peak memory and actual multi-GPU execution are blocked, not passing.
