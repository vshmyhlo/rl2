# Lux Design S3 environment

[Lux AI Season 3](https://github.com/Lux-AI-Challenge/Lux-Design-S3) is a
two-team strategy game implemented in JAX. With the default game settings:

- **Setup:** Two teams play on a symmetric 24 × 24 map. Each controls up to 16
  ships, which spawn periodically at the team's starting corner.
- **Game length:** A game contains five matches, each with 100 action-bearing
  steps. Ships and match points reset between matches; map progression and game
  parameters persist.
- **Turns:** Both teams act simultaneously. Each ship can stay still, move one
  tile up/right/down/left, or sap energy from enemy ships around a target tile.
- **Visibility:** Ships reveal nearby tiles through fog of war. Teams must
  explore to discover opponents, relics, and scoring locations. Competition game
  parameters are randomized once per game, and many must be inferred through play.
- **Terrain and energy:** Asteroids block movement. Nebulae reduce vision and can
  drain energy. Tile energy fields recharge or drain ships; moving and sapping
  cost energy, so ships need enough energy to perform their chosen action.
- **Combat:** Sapping and adjacent enemies can drain ship energy. When opposing
  ships occupy the same tile, the team with more combined energy there survives;
  a tie removes both sides. Ships with negative energy are removed.
- **Scoring:** Occupying a hidden scoring tile near a relic earns one point per
  step. Multiple friendly ships on the same tile still earn only one point.
  Discover these tiles by exploring and observing score changes.
- **Winning:** The team with more points at match end wins that match. Ties use
  total ship energy, then a random winner. Win the most matches to win the game.
- **Strategy:** Balance early exploration with holding scoring tiles and denying
  them to the opponent. Retain knowledge across matches to improve later decisions.

## Core JAX API

```python
import jax
import jax.numpy as jnp
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams

params = EnvParams()
env = LuxAIS3Env(auto_reset=False, fixed_env_params=params)
reset_key, step_key = jax.random.split(jax.random.key(0))  # Two scalar PRNG keys.
obs, state = env.reset(reset_key, params=params)  # Pytrees; shapes below.
actions = {
    player: jnp.zeros((params.max_units, 3), dtype=jnp.int16)  # (U, 3)
    for player in ("player_0", "player_1")
}
obs, state, rewards, terminated, truncated, info = env.step(
    step_key, state, actions, params=params
)  # Rewards and done flags: dictionaries of scalar arrays, one per player.
```

`reset` returns `(obs, state)`; `step` returns six values as shown above.
Both methods are JIT compiled. `jax.vmap` can batch keys, states, and actions;
keep `params` shared with `in_axes=None`, or batch its dynamic leaves when
randomizing games. `fixed_env_params` controls static dimensions and must agree
with the corresponding dimensions in runtime parameters.

Actions are dictionaries keyed by `player_0` and `player_1`, each holding an
integer array of shape `(U, 3)`. Each row is `[action_type, dx, dy]`: 0 stays,
1 moves up, 2 right, 3 down, 4 left, and 5 saps. Only sap uses the relative target
offsets; both absolute offsets must be at most `unit_sap_range`. Actions for both
teams are submitted in the same environment step.

Each player's observation is an `EnvObs` pytree. With defaults, `T=2` teams,
`U=16` unit slots per team, `W=H=24` map dimensions, and `R=6` relic slots:

| Field | Shape | Meaning |
|---|---|---|
| `units.position` | `(T, U, 2)` | Unit coordinates `(x, y)` |
| `units.energy` | `(T, U)` | Observed energy; state energy instead has shape `(T, U, 1)` |
| `units_mask` | `(T, U)` | Present and observable units |
| `sensor_mask` | `(W, H)` | Visible tiles |
| `map_features.energy` | `(W, H)` | Tile energy |
| `map_features.tile_type` | `(W, H)` | Empty 0, nebula 1, asteroid 2 |
| `relic_nodes` | `(R, 2)` | Visible relic coordinates |
| `relic_nodes_mask` | `(R,)` | Visible relic slots |
| `team_points`, `team_wins` | `(T,)` each | Current match points and cumulative match wins |
| `steps`, `match_steps` | `()` each | Game and match step counters |

Hidden fields use `-1`; use the masks, since tile energy can also be negative.
Team indices remain absolute in both players' observations. The full `EnvState`
contains hidden information and must not be used directly as the policy input.
Upstream's `observation_space` and `state_space` are placeholders, so derive
model shapes from the actual pytrees.

## Rewards, resets, and randomization

- `rewards[player]` is the player's **cumulative match wins**, not a per-step
  increment or the current relic-point score. For incremental match rewards,
  subtract the previous state's `team_wins`.
- A match has 100 action-bearing steps plus a boundary frame. The default game
  ends after 505 calls to `step`, with `truncated[player] = True` for both players;
  `terminated[player]` remains false. It runs all five matches.
- Match boundaries reset units and match points while retaining map progression
  and game parameters. Preserve agent memory across matches for adaptation.
- With `auto_reset=False`, explicitly reset after game truncation. With
  `auto_reset=True`, returned state/observations reset at game end;
  `info["final_state"]` and `info["final_observation"]` retain the transition result.
- Core `reset` randomizes the map but uses the supplied parameter values.
  `luxai_s3.wrappers.LuxAIS3GymEnv` samples game parameters from
  `env_params_ranges` on reset; its host-side wrapper is separate from the pure
  JAX API. Its reset info includes privileged `full_params` and `state` fields.

## rl2 integration

The dependency is available, but no rl2 environment adapter or training config is
provided. The AlphaZero trainer currently constructs PGX environments. Lux needs
an adapter and policy/search design for simultaneous joint actions, hidden
information, per-unit actions, reward conversion, and memory across matches.

Upstream references: [game rules](https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/c6d1e665d12a467b299fb586b3f876e47847746b/docs/specs.md),
[environment implementation](https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/c6d1e665d12a467b299fb586b3f876e47847746b/src/luxai_s3/env.py),
[parameters](https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/c6d1e665d12a467b299fb586b3f876e47847746b/src/luxai_s3/params.py),
and [Gym wrapper](https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/c6d1e665d12a467b299fb586b3f876e47847746b/src/luxai_s3/wrappers.py).
