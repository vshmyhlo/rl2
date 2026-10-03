# Karel synthesis reference

Current implementation, recorded 2026-10-03. Given an initial and target grid,
generate one program that transforms the initial state into the target.
The interpreter and task generator are local implementations.

## Environment and programs

[karel.py](../src/rl2/karel.py) samples a random world and grammar-generated
program, executes it, and rejects failed or unchanged outcomes. This gives a
reachable target and a known solution. Sampling is with replacement; uniqueness
across resets is not guaranteed. `reference_program` exposes the solution for
debugging or supervision, but it is not part of the policy observation.

- `reset(seed=...)` returns `KarelPair(initial, target)`, without an info wrapper.
- Each state is `int32[H, W, 6]`: four robot-heading channels, walls, marker counts.
- `step(token_id)` returns `(None, reward, terminated, truncated, info)`.
- `m)` submits and terminates the program. There is no separate EOS token.
- The vocabulary has 50 program tokens plus PAD. PAD is only for batching and is
  rejected by the environment and excluded from the policy distribution.
- Intermediate rewards are zero. `info["success"]` means exact target equality;
  `info["error"]` distinguishes syntax, runtime, execution-limit, and token-limit failures.

The DSL supports movement, turns, marker pickup/placement, `IF`, `IFELSE`,
`WHILE`, and `REPEAT R=0..R=19`. Blocks must be nonempty. Delimiters are
`m(` / `m)`, `i(` / `i)`, `e(` / `e)`, `w(` / `w)`, and `r(` / `r)`;
conditions use `c(` / `c)` and allow one negation. The interpreter allows nesting to depth 64.
The shortest valid program has five tokens: `DEF run m( turnLeft m)`.

## Terminal rewards

```text
D(s,t) = position_weight * shortest_path(robot_s, robot_t)
       + orientation_weight * int(heading_s != heading_t)
       + marker_weight * sum_cells(abs(markers_s - markers_t))
```

All weights default to 1 and must be positive and finite. Position distance
counts orthogonal moves through free cells; marker errors include every cell. Initial and target differ,
so the normalization denominator is positive. A BFS distance map from the target
robot is cached once per reset and reused for both initial and final/partial
states. Unreachable positions raise `ValueError`; sampled tasks and executions
always stay in the target's connected component.

The reward sums syntax, runtime, and distance terms, each in `[0, 1]`, plus a
**+1 exact-success bonus** for normal completion at the exact target. Total: `[0, 4]`.

```text
progress(s) = clip(1 - 0.5 * D(s,target) / D(initial,target), 0, 1)
```

| Outcome | Syntax | Runtime | Distance | Total |
| --- | --- | --- | --- | --- |
| Invalid syntax or token limit | `1/(1+d)` | `0` | `0` | `1/(1+d)` |
| Runtime error or execution limit | `1` | `progress(last_valid_state)` | `0` | `1 + progress(last_valid_state)` |
| Completes without exact match | `1` | `progress(final_state)` | `progress(final_state)` | `1 + 2 * progress(final_state)` |
| Completes at exact target | `1` | `1` | `1` | `4` (includes +1 bonus) |

Here `d` is the minimum token edits to any syntactically valid program,
independently of the target or reference solution. One edit earns 0.5; four edits
earn 0.2. Repairs are never executed. Runtime progress uses the last valid state
before failure, including updated position, heading, and markers. It does not
use the best state visited. Execution exceptions expose `partial_state`.

Progress maps the original `[-1, 1]` score to `[0, 1]`: unchanged distance
scores **0.5**, a 50% increase scores **0.25**, and doubled distance or worse
scores **0**. Completed exact solutions earn **4**, completed half-distance
solutions **2.5**, and completed programs with unchanged distance **2**.
A runtime failure after halving the distance earns **1.75**. Failure at the target earns **2** but
is not an exact success. Only normal completion unlocks the distance term.
A terminal `m)` on the last allowed token still evaluates normally.

[karel_syntax.py](../src/rl2/karel_syntax.py) computes exact edit distance with a
minimum-cost parsing chart. A cheap bound on repair length avoids expanding the
grammar to enforce depth 64 unless needed. Hypothetical repairs are not limited
by the episode's token budget or the sampler's depth setting.

## Model and training

[karel_model.py](../src/rl2/karel_model.py) concatenates the pair into 12 channels,
normalizes marker counts, and applies a CNN, flattening, projection, and LayerNorm
to create one context token. `backbone_type` selects a
[transformer](../src/rl2/transformer.py) or [Mamba3 stack](../src/rl2/mamba3.py)
to generate program tokens autoregressively. The YAML selects the transformer;
Python defaults retain Mamba3 compatibility. Transformer settings are `num_heads`
and `num_kv_heads` (null = full multi-head attention); Mamba uses `d_state` and
`headdim`. GRPO sets the transformer cache length to `env.max_program_tokens`,
covering the context prefix plus all L−1 inputs needed to predict L tokens.
Backbone parameters and carry states are not interchangeable. For a complete program `p`, feed `p[:-1]` and
predict `p`: the context predicts the first token, and `m)` remains a training label.

The example config enables `bf16: true` and `attention_implementation: cudnn`.
CNN, embedding, backbone, and head computations use BF16, as do transformer KV
caches. Parameters, optimizer state, normalization statistics, and loss/log-probability
arithmetic remain float32; the model casts output logits to float32.
cuDNN requires a supported NVIDIA GPU and does not silently fall back to XLA.
For CPU runs, set `attention_implementation: xla` (BF16 can stay enabled).
When selecting Mamba, also use `attention_implementation: xla`.

The output head initializes to zero, giving each program token probability 2%
after PAD masking. The encoder and backbone initialize randomly; gradients reach
them once the head becomes nonzero. `prefill()` starts fresh recurrent state;
`step()` advances it.

[grpo.py](../src/rl2/grpo.py) samples several programs per pair, normalizes rewards
within each group, and applies a clipped token objective. It averages over nonpadding
tokens within each program, then over programs. Whole programs stay together in
minibatches. PAD is excluded from losses, lengths, entropy, and all policy
probabilities; terminal `m)` is included. There is no value head.

Generated programs are printed to the console and saved under TensorBoard Text
`samples/generated_programs`. By default, logging occurs on the first rollout
and every 10th rollout, showing the first 8 programs from one task group in
sampling order, with rewards and token counts. Programs use indented blocks,
one action per line, inline conditions, and fenced code blocks in TensorBoard.
Malformed programs are formatted best-effort without changing their tokens. Groups rotate between logging
events; each displayed block shares an initial/target pair. PAD is omitted and
incomplete programs are preserved. Configure `log_program_interval` and
`log_program_count`; set the count to 0 to disable samples. Logging uses the
existing rollout and does not consume training randomness or generate extra programs.

TensorBoard logs `charts/reward_{syntax,runtime,distance,success}_mean` separately.
Each averages over all programs, including zero terms; their sum equals
`charts/reward_mean`. Distance is final progress only on completed executions;
runtime always measures progress from the final or last valid state. The success-bonus mean
equals `charts/success_rate`. Terminal `info` exposes `reward_syntax`,
`reward_runtime`, `reward_distance`, and `reward_success`.
`charts/informative_group_fraction` measures groups with differing rewards;
`charts/group_success_rate` measures groups containing at least one exact solution.

Identical rewards produce exactly zero advantages, including fractional rewards.
Different syntax distances can provide learning signal even in all-invalid groups.
The optional reference KL uses a frozen copy of the initial model; old-policy KL
early stopping is a separate setting.

## Configuration and verification

[grpo_karel.yaml](../configs/grpo_karel.yaml) is the source of truth. At this snapshot:

- 8×8 grids; 128 generated tokens; 256 execution steps, counting statements and loop checks.
- Reference sampler: depth 2, up to 3 statements per block, at most 10 markers per cell.
- Transformer: width 320, 7 layers, 5 attention/KV heads of dimension 64,
  CNN channels 32/64/64; 10,296,403 total parameters (about 10.3M).
- 16 tasks × 16 samples = 256 programs per rollout; 4 minibatches of 64, over 2 epochs.
- 100,000 rollout updates; learning rate 0.00025 with linear decay; clipping 0.2;
  target KL 0.02; gradient norm limit 0.5; entropy and reference KL coefficients zero.

```bash
uv run python -m rl2.grpo --config configs/grpo_karel.yaml
JAX_PLATFORMS=cpu uv run pytest tests/test_karel.py tests/test_karel_syntax.py tests/test_karel_model.py tests/test_grpo.py
```

Tests cover independent repair-search checks,
reward boundaries and partial execution states, PAD masking, recurrent/teacher-forcing
agreement, gradient flow, and a training smoke test. These are implementation checks, not
successful learning. No supervised warmup or grammar masking is implemented.

Monitor mean reward, exact success, informative-group fraction, syntax errors,
and truncations separately. Syntax error rate below 1 does not imply valid
programs: the remaining attempts may have hit the token limit.

Exact syntax scoring is a CPU cost: a measured 128-token case took about 0.25 s;
128 variable-length uniform samples took about 6.8 s in total. These are local
measurements, not throughput guarantees; the expanded grammar can cost more.
