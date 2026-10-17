# Qwen3 optimized HLO review — 2026-10-17

Inspected `jax.jit(...).lower(...).compile().as_text()` for the full real model.
StableHLO is saved only as a supplementary artifact. See
[parity](PARITY_REPORT.md) and [performance](PERFORMANCE_REPORT.md).

## Compilation and sharding

BF16, 28 layers, batch 1, sequence 128, full vocabulary output. Forward, mean
next-token loss plus all gradients, and a functional SGD step were compiled and
executed on the A100. Parameters are replicated; tokens/masks use `P("data", None)`
in an explicit mesh and `shard_map`. Tokamax uses `check_vma=False` because the
installed Pallas path lacks the required manual-axis annotations.

An additional two-device target topology compiled Tokamax forward and training
with global batch 2, one example per shard. **This was compile-only, not physical
multi-GPU execution.** There is only one GPU; collective latency and scaling are
unvalidated.

## Findings

- Every one-GPU graph has zero collectives. Two-device forward also has zero.
- Two-device training has **36 all-reduces**: two scalars for loss/token count and
  34 BF16 gradient buckets. The buckets contain exactly **596,049,920 elements**,
  covering each unique parameter gradient once. These are required synchronization,
  not redundant forward communication. No all-gathers, reduce-scatters or
  collective-permutes were found. The HLO marks these reductions synchronous;
  overlap/bucket tuning requires actual multi-GPU profiling.
- No host callbacks or HLO send/recv transfers appear. Numerical model execution
  has no PyTorch dependency or host array conversion.
- XLA fuses Q/K/V projections into `(128, 4096)` GEMMs and gate/up projections into
  `(128, 6144)` GEMMs. Projection weights retain checkpoint layouts; transposes
  are generally handled in fused kernels or bitcasts. No manual packing was needed.
- Native XLA attention materializes FP32 scores `(8, 256, 128)` (grouped query
  heads flattened into the query dimension), followed by separate softmax/value
  operations. This remains quadratic in sequence length.
- Tokamax emits **28 `__gpu$xla.gpu.triton` calls** in forward and **56** in
  backward/training. cuDNN emits 28 `__cudnn$fmhaScaleBiasSoftmax` calls and adds
  28 `__cudnn$fmhaScaleBiasSoftmaxBackward` calls for gradients. These fused paths
  avoid global attention-score matrices, although the explicit Boolean mask
  still scales quadratically.
- Forward contains 56 sine/cosine pairs inside fused Q/K rotations; backward and
  training share one pair. Materializing shared rotary values is a profiling
  opportunity for longer contexts, not an established performance improvement.
- FP32 SiLU arithmetic stays within fused GPU computation. A normalization fusion
  barrier was evaluated and rejected because full-model numerical agreement
  regressed; it is absent from the selected implementation.

## Memory and layout

Per-device arguments occupy 1136.876 MiB. Forward output is 74.188 MiB; returned
full gradients or updated weights occupy 1136.877 MiB. Input/output aliasing is
zero in this functional benchmark. A production training caller could evaluate
parameter donation; the benchmark deliberately preserves its fixed input weights.

| Backend | Graph | Compiler temporary MiB | Explicit copies | Transposes | Conversions |
| --- | --- | ---: | ---: | ---: | ---: |
| xla | forward | 3.501 | 0 | 112 | 1130 |
| xla | backward | 97.072 | 0 | 392 | 3527 |
| xla | train | 97.072 | 0 | 392 | 4993 |
| tokamax | forward | 3.516 | 0 | 0 | 1158 |
| tokamax | backward | 99.830 | 0 | 0 | 3636 |
| tokamax | train | 99.830 | 0 | 0 | 5102 |
| cudnn | forward | 3.539 | 0 | 0 | 1158 |
| cudnn | backward | 99.815 | 0 | 0 | 3356 |
| cudnn | train | 99.815 | 0 | 0 | 4822 |

These are optimized-HLO operation counts, including fusion bodies, not GPU kernel
launch counts. Zero explicit copies does not mean zero data movement. The XLA
attention layouts account for its extra transposes; precision conversions include
normalization, softmax, SiLU and gradient accumulation. No broad removal of casts
is justified while BF16 gradient parity remains unresolved.

The two-device training target estimates 123.329 MiB temporary storage per device,
including collective packing. This is a compiler estimate, not measured device
memory or proof of efficient communication. Allocator peaks are reported separately.

Full optimized dumps, supplementary StableHLO and JSON summaries are saved in
`/tmp/qwen3-audit-current/hlo/`, named
`gpu-{mode}-bfloat16-{backend}-manual-dp{devices}-t128-keep0.*`.
These are temporary local artifacts. Reproduction commands are in
[the audit guide](docs/jax_torch_parity.md). Cached generation and long-context
HLO were not benchmarked in this run.
