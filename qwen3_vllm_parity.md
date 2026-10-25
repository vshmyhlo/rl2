# Qwen3 / vLLM numerical parity

Audit: 2026-10-25. **BF16 parity is not established:** unchanged weights can
push importance ratios beyond GRPO's clipping bounds.

## Setup

- Two A100 40 GB GPUs; Qwen3-0.6B, revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- JAX 0.11.2, vLLM 0.27.0, PyTorch 2.13.0+cu130, Tokamax 0.0.15.
- Loaded `nvida.env` (the repository's spelling): CUDA, optimization `O1`,
  90% memory preallocation, default XLA flags. No FP32 model switch.
- BF16 projections/attention, FP32 logits; Tokamax Triton versus compiled vLLM
  FlashAttention. Two unequal prompts, two turns, 32 generated tokens per turn,
  context 4096, temperatures 1.0 and 0.7, before/after an optimizer update.

## Confirmed fix

`Precision.HIGHEST` did not enforce FP32 accumulation for BF16 projections.
Identical rows produced different results when changing the GEMM row count.
`qwen3.py` now explicitly requests `BF16_BF16_F32` for BF16 outputs. The FP32
head retains its original arithmetic, including FP32 backward cotangents.

The CUDA regression's mean absolute error against a BF16-rounded FP32 reference
fell from `3.88e-4` to `3.82e-6`. It fails before the fix and passes afterward.
Regression coverage also checks that FP32 head cotangents are not rounded to BF16.

## Final audit

Each row compares 128 sampled tokens against the same published weights.

| Publication / temperature | Max log-probability error | Mean error | Ratio range | Outside [0.8, 1.2] |
| --- | ---: | ---: | --- | ---: |
| Initial / 1.0 | 0.143965 | 0.017781 | 0.86592–1.08355 | 0/128 |
| Initial / 0.7 | 0.267176 | 0.015147 | 0.76554–1.12398 | 1/128 |
| Updated / 1.0 | 0.106094 | 0.008871 | 0.89934–1.09567 | 0/128 |
| Updated / 0.7 | 0.187259 | 0.006902 | 0.94581–1.20594 | 1/128 |

**15 focused checks pass; the real-checkpoint audit fails its zero-clipping
assertion.** Published BF16 weight hashes match, and the optimizer reproduces
standalone JAX loss/KL/clipping. Temperature 1.0 passes the sampled bounds;
these synthetic transcripts do not establish parity for full ALFWorld episodes.

## Remaining mismatch

At the captured first-layer token, input normalization and Q projection match
exactly. Attention is the first observed cross-framework divergence: Tokamax
rounds scaled queries to BF16 before the dot product;
[FlashAttention](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/softmax.h)
scales FP32 scores. Softmax/reduction order and prefill/decode kernels also differ.

Moving scaling after the dot improves that token's first-layer attention mean
error from `5.71e-5` to `1.17e-5`, but the saved rollout's worst log-probability
error rises to `0.30151`, with 2/128 ratios outside the bounds. cuDNN reaches
`0.29812`, with 1/128 outside. Neither attention experiment is applied.

The original failing token has vLLM log-probability `-0.55777` versus JAX
`-0.82495`: ratio `0.76554`, despite identical weights. The projection fix does
not eliminate this full-batch outlier. Matching attention arithmetic and validating
both prefill and decode remain necessary before claiming parity.

Whether an out-of-range ratio clips the loss also depends on the advantage sign.
vLLM's own prefill replay differs from generation by up to `0.13527` log-probability.

## Reproduce

Set `QWEN3_CHECKPOINT` to the local checkpoint directory, then run:

```sh
CUDA_VISIBLE_DEVICES=0 QWEN3_VLLM_DEVICE=1 QWEN3_VLLM_TESTS=1 \
QWEN3_VLLM_CONTEXT_LENGTH=4096 QWEN3_VLLM_BACKENDS=tokamax-triton \
QWEN3_VLLM_REPORT_DIR=/tmp/qwen3-parity NCCL_NET=Socket CXX=/usr/bin/g++-12 \
uv run --env-file nvida.env pytest --confcutdir=tests tests/test_qwen3_vllm.py -k checkpoint -q -s
```

Final artifacts on the test host: `/tmp/rl2-parity-final-accumulation/checkpoint/`.
