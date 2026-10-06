# A100 BF16 attention benchmark

Measured on 6 October 2026 using an NVIDIA A100-SXM4-40GB. Tokamax Triton
was fastest in the initial run. Its batch-8 forward-plus-backward median
increased from 401.5 µs to 514.0 µs on repetition, so the initial advantage
over cuDNN should not be treated as a stable speedup.

## Configuration and method

| Setting | Value |
| --- | --- |
| JAX / jaxlib | 0.11.2 / 0.11.2 |
| Tokamax | 0.0.15 |
| Input, output, and gradient dtype | bfloat16 |
| Sequence length | 1024 |
| Batch sizes | 1 and 8 |
| Query / key-value heads | 8 / 8 |
| Head dimension | 64 |
| Attention | Causal, no additional mask or bias |
| Timing | Median of 20 CUPTI device-execution samples per case |
| Kernel configuration | Defaults; no Tokamax autotuning |
| XLA_FLAGS | Unset or empty |

The benchmark measures attention kernels only; projections, RoPE, and KV-cache
updates are excluded. Forward-plus-backward returns the attention output and
gradients for all three of Q, K, and V, using a random output cotangent.
Inputs use random seed 0 and are identical across implementations for each batch
size. Compilation and correctness checks are outside the timing window; CUPTI
timing excludes Python dispatch overhead but adds profiling overhead.

The existing training process was paused during each timing window, with a
two-second delay to let submitted GPU work drain, and resumed afterward.
Its allocated GPU memory remained resident. GPU clocks were not fixed, and
the cause of the difference between the initial and repeat timings was not
established.

## Initial results

All times are **microseconds (µs)**, rounded to one decimal place. Lower is better.
Forward-plus-backward includes both passes, not backward alone.

| Implementation | B=1 forward | B=1 forward + backward | B=8 forward | B=8 forward + backward |
| --- | ---: | ---: | ---: | ---: |
| JAX XLA | 103.5 | 313.7 | 1003.6 | 2939.6 |
| JAX cuDNN | 67.0 | 179.1 | 137.7 | 532.8 |
| Tokamax XLA | 118.7 | 370.0 | 1133.2 | 3087.6 |
| Tokamax chunked XLA | 157.1 | 767.9 | 1237.6 | 4826.7 |
| Tokamax cuDNN | 51.4 | 179.1 | 137.7 | 532.8 |
| Tokamax Triton | 35.4 | 117.5 | 128.8 | 401.5 |
| Tokamax automatic selection | 35.6 | 117.6 | 129.5 | 514.9 |
| Tokamax Mosaic | Unsupported | Unsupported | Unsupported | Unsupported |

Tokamax Mosaic rejected all four cases with
`NotImplementedError: Only supported for sm90 and sm100 GPUs.`

## Batch-8 repeat measurements

Automatic selection and explicit Triton were repeated in reverse order to
investigate their initial backward-timing difference. Each row below contains
20 new samples. These results are separate from the initial run, not pooled.

| Implementation | Mode | Median (µs) | Minimum (µs) | Maximum (µs) |
| --- | --- | ---: | ---: | ---: |
| Tokamax automatic selection | Forward | 129.4 | 124.8 | 135.7 |
| Tokamax Triton | Forward | 127.8 | 126.4 | 134.6 |
| Tokamax automatic selection | Forward + backward | 513.4 | 507.8 | 519.0 |
| Tokamax Triton | Forward + backward | 514.0 | 508.6 | 521.3 |

At batch 1, explicit Triton took about 1.52× less time than cuDNN for
forward-plus-backward in the initial run. At batch 8, the repeat Triton
forward-plus-backward median was close to automatic selection and only about
4% below the initial cuDNN measurement. These are kernel measurements for this
shape, not end-to-end training speedups.

## Correctness and validation

All 28 supported initial cases and all four repeat cases passed correctness
checks against JAX XLA attention evaluated in float32 on the same BF16 inputs
cast to float32. Outputs and gradients had to be finite and satisfy a relative
L2 error threshold of 2%. The largest observed relative L2 error was **0.343%**.

The benchmark's six targeted pytest cases passed, along with Ruff formatting
and lint checks. The full test suite was not run.

## Reproduction and artifacts

The [benchmark script](../src/rl2/benchmark_attention.py) uses Tokamax from the
project's `cuda12` extra. On an idle GPU, reproduce the initial comparison with:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --extra cuda12 python -m rl2.benchmark_attention \
  --seq-len 1024 --batches 1 8 --heads 8 --kv-heads 8 --head-dim 64 \
  --iterations 20 --output runs/attention_a100_bf16_s1024.json
```

For the repeat, use `--batches 8 --backends tokamax_auto tokamax_triton` and
`--output runs/attention_a100_bf16_s1024_recheck.json`. The recorded runs also
used `--pause-pid` with the training process's PID, with its owner's permission;
the script resumes that process after each timing window.

Local raw artifacts retain every timing sample, correctness metric, and backend error:

- [Initial results](../runs/attention_a100_bf16_s1024.json)
- [Repeat results](../runs/attention_a100_bf16_s1024_recheck.json)
