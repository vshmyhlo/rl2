# vLLM settings for GRPO v2 mix

Measured on 3 November 2026 with Qwen3-0.6B and vLLM 0.32.0 on NVIDIA
A100-SXM4-40GB GPUs. The existing A100 serving defaults were already competitive.
The selected settings are explicit in `configs/grpo_v2/mix.yaml`:

```yaml
vllm:
  max_model_len: 8192
  max_num_seqs: 256
  max_num_batched_tokens: 2048
  gpu_memory_utilization: 0.90
  enable_prefix_caching: true
  enable_chunked_prefill: true
  async_scheduling: true
```

The existing `weight_update_pause_mode: keep` is retained. This benchmark did
not measure weight publication or learner updates.

## Serving results

Each row changes only the listed setting from the selected configuration,
except that disabling chunked prefill also requires the token budget to be at
least the 8192-token context limit. Token-budget rows use two workload samples;
other comparisons use the first sample. Differences around 1–2% are not treated
as established improvements.

| Setting | Generated tokens/s | Mean request latency (s) |
| --- | ---: | ---: |
| Selected baseline, 2048 batched tokens | 3,405 | 10.74 |
| Selected configuration, final isolated repeat | 3,397 | 10.73 |
| 4096 batched tokens | 3,386 | 10.81 |
| 8192 batched tokens | 3,337 | 10.92 |
| Baseline on second GPU | 3,418 | 10.69 |
| 128 sequences | 3,443 | 10.93 |
| 64 sequences | 3,331 | 14.33 |
| 95% GPU memory | 3,440 | 10.65 |
| 40,960-token context | 3,377 | 10.81 |
| Async scheduling disabled | 2,826 | 13.16 |
| Prefix caching disabled | 2,338 | 16.45 |
| Chunked prefill disabled, 8192 batched tokens | 3,446 | 10.65 |

Keep prefix caching and async scheduling: disabling them reduced throughput
by approximately 32% and 18%, respectively. The baseline prefix-cache hit rate
was about 96%. None of the 16 measured samples required KV-cache preemption.
The final isolated samples measured 3,386 and 3,407 generated tokens/s, confirming
the baseline. All samples together generated 3,686,400 tokens; every completion
returned the requested token count with aligned sampled-token log-probabilities.

Keep 256 sequence slots. Lowering the limit to 128 did not establish a throughput
gain and increased mean time to first token from 0.17 to 0.67 seconds. At 64
slots, mean time to first token rose to 5.89 seconds. A 95% memory budget did
not establish a gain; 90% leaves more allocation headroom.

The 8192-token context cap matches the training transcript budget; its benefit
is an explicit workload limit, not an established decoding speedup. Disabling
chunked prefill did not improve on the selected baseline meaningfully, and
vLLM warned that disabling it is unsupported for this model and could produce
incorrect outputs. Keep it enabled.

## Method and limits

- One serving GPU per run, BF16 weights, FP32 output head, processed sampled-token
  log-probabilities, CUDA graphs, and the NCCL weight-transfer backend enabled.
- 160 concurrent episodes: 20 groups of eight samples, matching the maximum
  episode pool for four groups, group size eight, and four prefetched batches.
  This is a finite burst, not a continuously replenished training pool.
- Three sequential model calls per episode. Fixed response lengths of 128, 256,
  512, or 1024 tokens prevent early stopping from changing the amount of work.
  Each sample produces exactly 230,400 tokens through 480 HTTP completions.
- Temperature 1, top-p 1, top-k disabled, `logprobs=0`, token IDs returned,
  and EOS ignored for the controlled response lengths.
- Seeded questions from the 23 configured Reasoning Gym tasksets, with their
  tool schemas. One quarter of groups include synthetic history to reach 4096
  prompt tokens. Initial prompts span 478–4096 tokens. Subsequent prompts append
  the actual generated tokens and a fixed 34-token simulated tool response.
- Identical workload definitions and seeds across settings; generated text can
  differ with scheduling. The second sample uses different seeded questions.
- Server startup, compilation, and warmup are excluded. Prefix caches are cleared
  before every sample. Throughput is total generated tokens divided by elapsed
  client wall time; latency includes HTTP handling, queueing, prefill, and decode.
- Broad comparisons ran on the two GPUs concurrently. The first baseline and
  final two-sample confirmation ran alone. GPU clocks were not fixed.

These are inference measurements using the local base-model snapshot at
`/tmp/rl2-review-qwen3`, not end-to-end training or reward evaluations. Real
CodeGym/ALFWorld execution, environment delays, policy changes, and mid-generation
cache invalidation are excluded. Their timing and context distributions can
change the best serving settings.

Raw JSON measurements, server logs, workload tokens, profiles, and the benchmark
driver are in `/tmp/rl2-vllm-token-budget-20261103/` on the benchmark host. To repeat
the selected setup with a free GPU:

```bash
RL2_BENCHMARK_PORT=18080 uv run --no-sync python \
  /tmp/rl2-vllm-token-budget-20261103/benchmark.py \
  --gpu 0 --trials 2 \
  --profiles /tmp/rl2-vllm-token-budget-20261103/profiles-final.json
```

This command overwrites that profile's previous measurements; copy the artifact
directory first if retaining them. The driver requires the local model snapshot.
