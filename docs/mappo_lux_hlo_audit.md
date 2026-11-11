# Lux MAPPO two-GPU HLO audit

Compared on 2026-11-10 with JAX 0.11.2 and two NVIDIA A100-SXM4-40GB GPUs:
16 environments, 256 rollout steps, four minibatches, four update epochs, and
float32 computation. Each GPU holds eight environments, 16 actor sequences,
and four sequences per minibatch. Source snapshots accompany the artifacts;
the comparison holds the policy and trainer source fixed.

## Larger rollout configuration

`configs/mappo/lux.yaml` now uses 128 environments and a 512-step horizon,
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
