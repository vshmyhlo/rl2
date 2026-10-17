# Qwen3 GPU performance — 2026-10-17

One NVIDIA A100-SXM4-40GB; pinned Qwen3-0.6B checkpoint and versions from
[the parity report](PARITY_REPORT.md). **These timings do not establish BF16
gradient parity; the numerical acceptance failures remain.**

## Setup

BF16 parameters/computation, FP32 logits and next-token cross entropy, batch 1,
sequence 128, no KV cache, all vocabulary logits. Input repeats the real-tokenized
sentence “Explain reinforcement learning in one sentence.” to the requested length.
Backward differentiates all 310 parameters. The training graph performs one
functional SGD update; it is not an end-to-end training run.

Every result is the median of seven synchronized calls after compilation and two
warmups. Throughput counts input tokens, not autoregressive generation tokens.
JAX uses compiled execution; PyTorch is uncompiled eager execution with either
eager attention or SDPA. Thus the comparison includes PyTorch dispatch overhead;
`torch.compile` was not benchmarked. TF32/autocast are disabled in PyTorch.

An unrelated GPU job overlapped the first Tokamax measurements. Those timings
were excluded, and all three Tokamax modes were rerun after that job exited.

## Steady-state measurements

| Implementation | Forward ms | Loss + backward ms | Forward tokens/s | Backward tokens/s | Peak allocated MiB, forward / backward |
| --- | ---: | ---: | ---: | ---: | ---: |
| JAX xla | 4.522 | 16.547 | 28309 | 7735 | 1623.2 / 2382.1 |
| JAX tokamax | 4.438 | 16.522 | 28839 | 7747 | 1623.2 / 2384.8 |
| JAX cudnn | 4.390 | 17.276 | 29158 | 7409 | 1623.2 / 2384.8 |
| PyTorch eager | 46.029 | 151.195 | 2781 | 847 | 1257.2 / 2957.7 |
| PyTorch sdpa | 35.781 | 128.813 | 3577 | 994 | 1257.2 / 2957.7 |

JAX peaks are process-lifetime `peak_bytes_in_use`, including compilation/loading;
PyTorch peaks are `max_memory_allocated`, reset before each mode's warmups.
They measure framework allocations, not total driver/NVML memory, and are not
strictly comparable per-call peaks. JAX preallocation was disabled. Raw repeated
samples, compiler estimates and complete allocator statistics are retained in JSON.

## Compilation and training step

Seconds include tracing, lowering and compilation, separately from steady-state
execution. Existing installed compiler/kernel caches may be reused; these are
observed compilation times, not a cold-install comparison.

| JAX backend | Forward compile s | Backward compile s | SGD compile s | SGD step ms |
| --- | ---: | ---: | ---: | ---: |
| xla | 15.16 | 34.72 | 35.93 | 18.250 |
| tokamax | 14.76 | 34.64 | 36.37 | 18.329 |
| cudnn | 11.59 | 25.71 | 27.27 | 18.880 |

## Changes and limits

The retained SiLU correction matches PyTorch's internal precision without a
meaningful observed latency regression. On the cuDNN control, before/after
forward is **4.452 / 4.390 ms**, and loss plus backward is **17.177 / 17.276 ms**.
These small changes are within ordinary run variation; no speedup is claimed.
Explicit Tokamax Triton selection prevents fallback and preserves fused attention.
The compiler already fuses projection work effectively; no speculative custom
kernels or manual parameter repacking were introduced.

Only one short-context workload and one GPU were timed. Longer contexts, cached
decoding, model-parallel execution and real multi-GPU scaling remain unmeasured.
Native XLA's score matrices and the common dense mask remain longer-context memory
concerns. Numerical correctness takes priority over the performance comparison.

Artifacts: `/tmp/qwen3-audit-current/hlo/`, `torch-benchmark-{eager,sdpa}.json`, and
`before-silu-performance/`. Run `scripts/audit_qwen3_hlo.py --execute` for JAX and
`tests/benchmark_qwen3_torch.py` for the matching PyTorch workload, using the same
checkpoint, batch, length, dtype, warmup and run count.
