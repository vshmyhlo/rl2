# Qwen3 / vLLM numerical parity

Audit: 2026-10-28. **BF16 parity is not established.** Identical published
weights still produce some ratios outside GRPO's `[0.8, 1.2]` clipping range.
The regression remains failing; its thresholds have not been relaxed.

## Fixes in this review

- Honor modern Hugging Face `rope_parameters.rope_theta`; previously JAX silently
  used the top-level/default theta. Reject unsupported scaled, partial, or
  per-layer RoPE instead of silently evaluating a different model.
- Construct static RoPE powers accurately before FP32 rounding and reciprocation.
  XLA's previous constant-folded power introduced frequency error that grew with
  position. Across the full 40,960-position BF16 cache, mean absolute error against
  vLLM CUDA fell from `2.20206e-5` to `3.64508e-7` (60×). The existing sampled
  RoPE and QK-normalization/RoPE operation comparisons now match exactly.
- Use FP32 `log_softmax` before token selection, matching vLLM's shifted arithmetic.
  Selecting a logit and subtracting `logsumexp` lost precision with large common
  offsets. A compiled value-and-gradient regression fails before this fix.
- Let the publication audit select its actual learner backend through
  `QWEN3_VLLM_LEARNER_BACKEND`. Previously alternative forward backends were
  compared while optimizer checks always used Tokamax. Require that the selected
  backend's forward metrics agree with its differentiated training graph.

The existing BF16 projection accumulation and native LoRA rounding are retained.
Frozen base-weight hashes match vLLM after both publications; temperature scaling,
exact token IDs, neutral top-k/top-p, and processed sampling log-probabilities agree.

## Real-checkpoint results

Two A100 40 GB GPUs; Qwen3-0.6B from `/tmp/rl2-review-qwen3`; JAX 0.11.2,
vLLM 0.32.0, PyTorch 2.13.0+cu130, Tokamax 0.0.15. BF16 transformer/LoRA,
FP32 logits, XLA optimization O1, 90% learner memory reservation. Two unequal
prompts, two turns, 32 tokens per turn, padded context 4,096. Each row compares
128 sampled tokens against the same published adapter; “updated” follows a real
Adam step and adapter reload. Separate backend runs have different updated policies.

| Learner | Publication / temperature | Max log-probability error | Mean error | Ratio range | Outside clip range |
| --- | --- | ---: | ---: | --- | ---: |
| Tokamax | Initial / 1.0 | 0.215575 | 0.021594 | 0.83679–1.24058 | 1/128 |
| Tokamax | Initial / 0.7 | 0.228113 | 0.012725 | 0.79603–1.15022 | 1/128 |
| Tokamax | Updated / 1.0 | 0.104729 | 0.014350 | 0.93805–1.11041 | 0/128 |
| Tokamax | Updated / 0.7 | 0.080618 | 0.006955 | 0.92255–1.07671 | 0/128 |
| cuDNN | Initial / 1.0 | 0.130131 | 0.016960 | 0.87798–1.10586 | 0/128 |
| cuDNN | Initial / 0.7 | 0.153815 | 0.012387 | 0.85743–1.09332 | 0/128 |
| cuDNN | Updated / 1.0 | 0.190114 | 0.015136 | 0.87815–1.20939 | 1/128 |
| cuDNN | Updated / 0.7 | 0.119212 | 0.007909 | 0.88762–1.11729 | 0/128 |

Both full-checkpoint audits fail: Tokamax has unchanged-policy false clipping;
cuDNN also exceeds the candidate-probability limit (`0.06709` versus `0.06`).
Both optimizer graphs agree with their corresponding standalone forward metrics.
vLLM's own prefill replay differs from its generation scores by up to `0.15681`.

A controlled replay of identical initial tokens compared old/new RoPE with
Tokamax, cuDNN, and XLA attention. No backend eliminated out-of-range ratios
across the full audit. More accurate RoPE reduces local error but does not
monotonically reduce every final-token error: Tokamax's temperature-1 maximum
fell from `0.239763` to `0.215575`, while its temperature-0.7 maximum rose from
`0.173155` to `0.228113`. A subsequent stability-focused defaults update selects
cuDNN for GRPO v2 and ALFWorld because its worst sampled-token error was lower,
and disables rollout prefetching to avoid policy staleness. This does not resolve
the parity failures above; the standalone Qwen3 default remains Tokamax.
At the time of this audit, the environment set JAX matmul precision to `highest`
and disabled TF32. Those overrides have since been removed; current operations
use backend defaults.
A tiny uncached CUDA GRPO update and FP32-head gradient check pass with these
settings. The existing small cached-attention cuDNN test fails to build an
execution plan even with both new precision settings removed; that path is not
used by the learner. Configuration/prefetch checks pass (52). The full-checkpoint
audit has not been repeated with the new environment settings.

Attention remains a known arithmetic mismatch: Tokamax 0.0.15 rounds scaled
queries to BF16 before QK multiplication; vLLM FlashAttention scales FP32 scores.
Reduction order, online softmax, and prefill/decode execution also differ. Earlier
score-scaling experiments did not consistently improve whole-model parity.
Disabling PyTorch BF16 reduced-precision reductions made no difference in this
review's representative Qwen3 projection-shape probe.

## Validation and reproduction

CPU model/LoRA tests pass (63 passed, 4 GPU-only skips); focused configuration and
checkpoint checks pass (25). The optimizer suite had 41 passing checks and 3
multi-device skips before the final log-softmax implementation; the 6 affected
log-probability/update/learner checks pass with the final code. Sixteen focused CUDA
operation/model checks pass, including the new full-context RoPE regression.
Both tiny-model CUDA publication tests pass (tied and untied heads), including
adapter reloads and optimizer updates. Ruff formatting and lint checks pass.
The new configuration, RoPE, and log-probability regressions failed before their
respective fixes. An initial real-checkpoint attempt without the documented O1
settings exhausted memory during optimization; both O1 audits completed.
No full test suite or environment training was run.

```sh
CUDA_VISIBLE_DEVICES=0 QWEN3_VLLM_DEVICE=1 JAX_PLATFORMS=cuda \
JAX_OPTIMIZATION_LEVEL=O1 XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.90 \
QWEN3_VLLM_TESTS=1 QWEN3_CHECKPOINT=/path/to/checkpoint QWEN3_VLLM_CONTEXT_LENGTH=4096 \
QWEN3_VLLM_BACKENDS=tokamax-triton QWEN3_VLLM_LEARNER_BACKEND=tokamax \
QWEN3_VLLM_REPORT_DIR=/tmp/qwen3-parity NCCL_NET=Socket CXX=/usr/bin/g++-12 \
uv run pytest --confcutdir=tests tests/test_qwen3_vllm.py -k checkpoint -q -s
```

For cuDNN, set both backend variables to `cudnn`. Local artifacts:
`/tmp/rl2-parity-review-fixed/checkpoint/`, `/tmp/rl2-parity-review-cudnn/checkpoint/`,
`/tmp/rl2-parity-review-tiny/`, `/tmp/rl2-parity-review-replay.json`, and
`/tmp/rl2-parity-review-kernels.log`.
These synthetic transcripts do not certify arbitrary prompts, longer contexts,
other hardware, or training stability. Numerical error must not be hidden by
replacing the actor's sampling log-probabilities with learner scores.
