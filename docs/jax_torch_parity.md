# Qwen3 GPU audit

The current audit uses the real Qwen3-0.6B checkpoint on an NVIDIA A100,
with JAX and PyTorch CUDA execution outside the sandbox. It supersedes the
previous CPU diagnostic report.

- [Numerical parity, checkpoint integrity, and full gradients](../PARITY_REPORT.md)
- [Optimized forward/backward/training HLO](../HLO_REVIEW.md)
- [GPU latency, compilation time, and memory](../PERFORMANCE_REPORT.md)

The implementation now matches PyTorch’s FP32 internal SiLU arithmetic while
preserving BF16 storage, and explicitly selects Tokamax’s Triton kernel.
The opt-in suite compares every unique checkpoint parameter gradient and
normalization boundaries. Reports distinguish passing checks from remaining
BF16 discrepancies; passing forward tests alone does not certify training parity.

See the parity report for test setup. Compile and benchmark each backend with:

```bash
for backend in xla tokamax cudnn; do
  for mode in forward backward train; do
    uv run python scripts/audit_qwen3_hlo.py \
      --checkpoint "$QWEN3_CHECKPOINT" --output /tmp/qwen3-hlo \
      --attention-backend "$backend" --devices 1 --batch 1 --length 128 \
      --mode "$mode" --execute --warmup 2 --runs 7
  done
done
```

Set `JAX_PLATFORMS=cuda` and `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
For actual two-GPU validation, use `--devices 2 --batch 2 --execute` on a
host with two physical GPUs. Compile-only topology results do not establish
multi-device execution or collective efficiency.
