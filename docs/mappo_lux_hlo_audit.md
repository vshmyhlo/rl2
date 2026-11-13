# Lux MAPPO two-GPU HLO audit

## Built-in cuDNN attention: 2026-11-13

Lux now uses `jax.nn.dot_product_attention`, with `cudnn` as its default
backend and `xla` for CPU/float32 execution. The training configuration selects
`cudnn` and `bf16: true`; float32 cuDNN configurations are rejected early.
Parameters and recurrent memory remain float32. cuDNN head widths are asserted
to be divisible by eight. Query/key lengths are padded to
even values for JAX's cuDNN backward path; padded keys are masked, extra query
outputs are discarded, and the attention scale uses the original head width.

The current 32×32 configuration compiles on two A100-SXM4-40GB GPUs with JAX
0.11.2: eight environments, 512 steps, four minibatches, one update epoch.
Optimized HLO confirms:

- All six attention calls use cuDNN forward/backward kernels. Rollout attention
  processes eight actors per GPU; learner attention processes 1,024 observations
  per GPU. **No attention all-gathers remain.**
- Two static all-gather sites remain: minibatch indices and spatial-gather
  gradients. Rollout has no all-gathers; autoreset retains its boolean reduction.
- Estimated total executable storage is **12.129 GiB per GPU**, including
  12.057 GiB of temporaries. Parameters/optimizer storage aliases 73.429 MiB.
  No host callbacks occur. The sap-head intermediate remains outside this fix.

Artifacts: `runs/mappo_hlo/lux_cudnn_32_20261113/`. The earlier baseline below
used a 16×16 map and float32; its memory estimate is not a like-for-like comparison.
Compilation still encountered convolution autotuning workspace shortages while
another job occupied the GPUs; no full-training throughput benchmark was run.

cuDNN's runtime compiler needs the project's CUDA 13 NVRTC libraries. Selecting
their directory resolved the host's `No valid execution plans built` failures,
including for tiny unmasked attention. The training Dockerfile now sets this
library path; the image has not been rebuilt during this change. For host runs:

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python \
  scripts/audit_mappo_hlo.py --config configs/mappo/lux.yaml --step train \
  --output runs/mappo_hlo/lux_cudnn_32_20261113
```

Validation: 61 targeted cases passed across attention, configuration, recurrent
replay, sharded policy gradients, curriculum wiring, self-play configuration,
and evaluation wiring. GPU tests compare cuDNN outputs/parameter gradients with
XLA and assert no attention all-gathers, covering masked and unmasked attention
and odd key lengths. Ruff formatting/checks and `git diff --check` passed.
Tests ran from `tests/` with `--rootdir=. --confcutdir=.` because repository-root
collection encountered an unreadable `cluster-runs` directory. The curriculum
pool wiring test requires one CPU device; sharding tests used two devices.

## Triton baseline: 2026-11-13

Compiled `train_step_donated` with JAX 0.11.2 on two A100-SXM4-40GB GPUs,
using the saved `configs/mappo/lux.yaml` snapshot: 8 environments, 512 steps,
4 minibatches, 1 update epoch, float32, and Triton attention. The model has
two attention blocks, eight heads of width 32, and 64 convolution channels.
Each GPU receives four environments, eight rollout actors, and two recurrent
sequences per minibatch. One iteration collects 4,096 environment steps and
attempts at most four updates. The snapshot's YAML comment claiming 16
environments per GPU was stale and is now corrected.

**Compilation succeeds, but this execution is not efficient enough to call
optimal.** The primary issue is replicated Triton attention; the sap head also
materializes a large intermediate.

| Compiler diagnostic | Current Triton configuration |
| --- | ---: |
| Compilation, including lowering | 113.65 seconds |
| Temporary buffers per GPU | 11.865 GiB |
| Arguments + outputs + temporaries − aliases per GPU | 11.937 GiB |
| Donated/aliased storage per GPU | 73.429 MiB |
| Static all-gather sites, counting asynchronous start/done once | 63 |
| Host callbacks | None |

Findings, in priority order:

1. **Attention loses batch sharding.** `model/lux.py` calls Tokamax without
   `q_sharding`. Optimized HLO gathers Q/K/V before all six attention calls
   per policy evaluation. Rollout kernels process 16 actors on each GPU
   instead of eight; learner kernels process 2,048 observations instead of
   1,024. Backward attention is replicated too. Six gathers per rollout step
   receive approximately 4.767 MiB per GPU, or 2.383 GiB over 512 steps.
   Learner forward/backward gathers receive approximately 1.437 GiB per
   accepted update. Including the final value evaluation, all-gather peer
   receives total approximately **8.136 GiB per GPU per iteration** when all
   four updates run; this excludes all-reduce/reduce-scatter traffic and
   protocol overhead. Tokamax exposes explicit `q_sharding`; integrating and
   validating batch sharding is the first optimization to pursue.
2. **Sap scoring materializes 900 MiB.** The forward fusion produces
   `f32[1024,16,225,64]` in `LuxHeads._sap_logits`: gathered spatial keys plus
   relative-offset embeddings, retained for differentiation. Consider scoring
   the 16×16 map against each ship query first, then gathering scalar scores
   and adding separate relative-offset scores. The corresponding map-score
   tensor would be 16 MiB. Numerical/gradient equivalence and actual compiler
   memory savings need validation before changing this path.
3. **Recurrence has little batch parallelism.** Each block retains 512-step
   forward/backward loops with only two sequences per GPU. Input projections
   are already outside these loops. Increasing environments or reducing
   minibatches may improve utilization, but changes training batch semantics
   and increases activation memory; benchmark after fixing attention sharding.
4. **Some communication is expected.** Global sequence shuffling uses an
   index gather and three grouped minibatch reduce-scatters. Accepted updates
   combine gradients into one approximately 24.61 MiB all-reduce. Autoreset
   retains one boolean all-reduce per environment step, without gathering
   environment state. A further reduce-scatter and feature/index gather occur
   in spatial-gather gradients. Reset generation remains conditional; the
   always-false Lux timeout removes the timeout-bootstrap policy evaluation.

The executable uses cuBLASLt matmuls, cuDNN convolutions, and Triton attention;
model/optimizer state donation is effective. Static copies and fusion counts
alone do not establish runtime cost.

An otherwise identical diagnostic configuration using `attention_implementation:
xla` also compiles. It removes every attention all-gather, leaving two static
sites (minibatch indices and spatial-gather gradients), and no rollout gathers.
However, its estimated total storage rises to **18.160 GiB per GPU**, including
18.088 GiB of temporaries. This confirms the attention communication is
avoidable, but does not establish XLA attention as the preferred replacement.
Explicitly sharded Triton should be evaluated first. The comparison config,
HLO, summary, and compilation log are saved alongside the baseline artifacts.

Artifacts are under `runs/mappo_hlo/lux_2gpu_20261113/`: optimized HLO,
StableHLO, resolved configuration, compiler summary, source/configuration
snapshots and hashes, and `gather_analysis.json`. Communication estimates use
half each gathered output's bytes as the two-device peer receive payload,
weighted by loop execution counts; KL stopping reduces learner traffic.

Reproduce the baseline on two visible GPUs:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python \
  scripts/audit_mappo_hlo.py --config configs/mappo/lux.yaml --step train \
  --output runs/mappo_hlo/lux_2gpu_20261113
```

No training was executed; training source and configuration remain unchanged.
Another job held approximately 37 GB on each GPU; compilation
completed despite workspace allocation failures during convolution autotuning.
Consequently, kernel choices and throughput must be rechecked on idle GPUs.
Memory figures exclude runtime/library overhead. No pytest run was needed for
this compilation and documentation-only audit.

## Historical audits

Compared on 2026-11-10 with JAX 0.11.2 and two NVIDIA A100-SXM4-40GB GPUs:
16 environments, 256 rollout steps, four minibatches, four update epochs, and
float32 computation. Each GPU holds eight environments, 16 actor sequences,
and four sequences per minibatch. Source snapshots accompany the artifacts;
the comparison holds the policy and trainer source fixed.

## Larger rollout configuration

A previous `configs/mappo/lux.yaml` used 128 environments and a 512-step horizon,
with four minibatches and four update epochs. The horizon exceeds the 505-step
game length. On two GPUs this gives:

- 64 environments and 128 actor sequences per GPU during rollout.
- 32 recurrent sequences × 512 steps = 16,384 agent observations per GPU per
  minibatch, versus 4 × 256 = 1,024 previously.
- 65,536 environment steps and 131,072 agent observations per iteration.
- 152 iterations from scratch within the 10M-step budget: 9,961,472 environment
  steps and at most 2,432 parameter updates; KL stopping can reduce updates.
- Logging every iteration; evaluation and checkpoints every eight iterations
  (524,288 environment steps, approximately 5% of the budget).

Reset, rollout, and training compile on both A100s. The training executable
requires an estimated **3.49 GiB per GPU**, including 3.46 GiB of temporary
buffers. No environment-state all-gathers reappear. Larger minibatches use five
reduce-scatter sites instead of three; the remaining all-gather carries indices.
Artifacts are in `runs/mappo_hlo/lux_128x512/`. This increases work per update;
throughput and learning efficiency have not been benchmarked.

## Autoreset change

`sequential_vmap` was replaced by a custom batching rule with a scalar
`any(done)` guard. No completed games means no reset generation. Otherwise,
reset runs in parallel across the batch and fresh state is selected only for
finished games. Final observations, parameters, and per-environment RNG behavior
are preserved. Single-environment stepping retains its original conditional.

| Autoreset communication | Before | After |
| --- | --- | --- |
| Per environment step | 16 all-gathers of 51 buffers | One boolean all-reduce |
| Per 256-step iteration | 4,096 state all-gathers | 256 boolean all-reduces |
| State-gather peer payload per GPU | Approximately 0.95 GiB | None |

The old gather result contained 499,104 bytes and lived inside the global
16-environment sequential loop. The new HLO has no environment-state all-gather
or sequential autoreset loop; reset generation remains in a conditional branch.
The tradeoff is generating candidate resets for unfinished games when another
game finishes. Those candidates are discarded. The scalar reduction still
synchronizes devices each step.

## Other findings

- Scalar reset/step and batched reset retain the same compiler memory estimates
  and have no collectives. Batched step replaces its state gather with the
  boolean all-reduce; temporary storage decreases from 0.603 to 0.576 MiB.
- Rollout HLO copy sites decrease from 235 to 134; training copy sites decrease
  from 252 to 151. These are static instruction counts, not runtime copy counts.
- Rollout temporary storage decreases from 44.646 to 44.471 MiB. Training
  temporary storage is effectively unchanged: 235.841 versus 235.815 MiB.
  Training aliases 27.795 MiB; combined input/output/temporary storage, excluding
  aliases, is 264.120 MiB per GPU after the change.
- Learner communication is unchanged: one combined 9.27 MiB gradient
  all-reduce per accepted update, small global statistics reductions, index
  gathers, and three reduce-scatters for selected minibatch data. The remaining
  training all-gather carries minibatch indices, not environment state.
- Episode-statistic reductions remain outside the rollout loop. No host
  callbacks were found. Recurrent scans, layout copies, and small matmuls remain;
  HLO alone does not establish optimal runtime utilization.

## Artifacts and checks

Local, Git-ignored artifacts are under `runs/mappo_hlo/autoreset_before/` and
`runs/mappo_hlo/autoreset_after/`. They include optimized scalar reset/step,
batched reset/step, rollout, and donated training HLO. Training reports include
resolved configuration and source snapshots with hashes.

Recompile the training paths with two allocated GPUs:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python \
  scripts/audit_mappo_hlo.py --config configs/mappo/lux.yaml \
  --output runs/mappo_hlo/lux_128x512
```

All CUDA paths compile with the default partitioner; no `shard_map` workaround
is needed. Sixteen targeted CPU tests passed, including Lux rollout/update/
playback and no-reset, mixed-reset, and all-reset cases. The new two-device HLO
regression fails with the old sequential batching rule and passes with the new
rule. It also passes on both A100s with `NCCL_NET=Socket`; the inherited `gIB`
transport failed plugin initialization. Ruff formatting and checks passed.

Memory and payload figures are compiler-derived estimates. No training
throughput benchmark was run; the GPUs were occupied by another training job.

## Nested observations (CPU)

Compared scalar and eight-environment reset/step executables on CPU with 16
ships and randomized parameters. CUDA was unavailable in this session. Native
observation dtypes reduce output storage; conversion now happens in the encoder.

| Path | Output bytes, before → after | Temporary bytes, before → after |
| --- | --- | --- |
| Scalar reset | 31,577 → 25,245 | 65,856 → 63,744 |
| Scalar step | 63,217 → 50,553 | 102,984 → 111,856 |
| Batched reset | 249,872 → 198,488 | 525,952 → 502,784 |
| Batched step | 499,912 → 397,144 | 747,288 → 964,736 |

Loop counts and conditional reset guards are unchanged. More native output
leaves introduce layout copies: batched-step copy sites rise from 1,174 to 1,211
and transpose sites from 47 to 60; conversion sites fall from 399 to 362.
No new environment collectives or host callbacks were found.

Reset, rollout, and donated training also compile on two CPU devices with two
environments, two ships, three steps, and width-four encoder/LSTM. Rollout has
no all-gathers; training gathers shuffled minibatch leaves. This verifies nested
storage and partitioning, not GPU efficiency or throughput. Artifacts are under
`/tmp/rl2-nested-obs-hlo/{before,after,training}/`.

## Self/opponent observations (CPU)

Compared the preceding nested layout against explicit `self`/`opponent` team
fields without `team_id`, using the same scalar and eight-environment paths.
Shared world fields are stored once per observer; opponent fields retain that
observer's visibility. Loop counts and reset guards are unchanged, with no
environment all-gathers or host callbacks. Extra split-output copies remain:
batched-step copy sites increase from 1,211 to 1,228. Batched-step output storage
falls by 192 bytes and temporary storage rises by 96 bytes; reset temporary
storage is unchanged for the batch and falls by 64 bytes for a single environment.

Two-device CPU rollout and training compile at the small shapes above. Rollout
still has no all-gathers; learner gathers now cover the split minibatch leaves.
Eighteen targeted tests passed, including ownership/visibility, two-device
autoreset, encoding, rollout, update, and playback. CUDA efficiency remains
unverified. Artifacts are under `/tmp/rl2-nested-obs-hlo/self_opponent_{before,after,training}/`.

Grouping world fields under `LuxWorldObservation` preserves leaf order. Scalar
and eight-environment reset/step, plus two-device reset/rollout/training HLO,
are identical after removing source metadata and locations; operation counts
and compiler memory estimates are unchanged. Seven targeted tests passed.
Artifacts are under `/tmp/rl2-nested-obs-hlo/{world_after,world_training}/`.
This comparison remains CPU-only.

## Remove unused spawn observation fields (CPU)

Compared optimized single-environment and two-environment `reset`/`step` HLO
on 24×24 maps with two ship slots, randomized parameters, and mixed game autoreset.
Removing `init_unit_energy` and `spawn_rate` from policy observations preserves
all 408 retained output leaves exactly, including simulator parameters/state.
Loop counts, reset guards, and largest array intermediates are unchanged;
no operations were added. Redundant observation broadcasts/selects disappear,
and batched-step copies fall from 1,202 to 1,196. Batched-step output/temporary
storage falls from 72,050/124,080 to 71,954/123,936 bytes. No all-gathers appear.
Artifacts: `/tmp/rl2-global-only-hlo/{before,after}/`. CUDA was unavailable;
GPU fusion and throughput remain unverified.
