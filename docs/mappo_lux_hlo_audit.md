# Lux MAPPO two-GPU HLO audit

Compared on 2026-11-10 with JAX 0.11.2 and two NVIDIA A100-SXM4-40GB GPUs:
16 environments, 256 rollout steps, four minibatches, four update epochs, and
float32 computation. Each GPU holds eight environments, 16 actor sequences,
and four sequences per minibatch. Source snapshots accompany the artifacts;
the comparison holds the policy and trainer source fixed.

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
  --output runs/mappo_hlo/autoreset_after
```

All CUDA paths compile with the default partitioner; no `shard_map` workaround
is needed. Sixteen targeted CPU tests passed, including Lux rollout/update/
playback and no-reset, mixed-reset, and all-reset cases. The new two-device HLO
regression fails with the old sequential batching rule and passes with the new
rule. It also passes on both A100s with `NCCL_NET=Socket`; the inherited `gIB`
transport failed plugin initialization. Ruff formatting and checks passed.

Memory and payload figures are compiler-derived estimates. No training
throughput benchmark was run; the GPUs were occupied by another training job.
