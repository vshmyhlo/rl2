# Historical self-play opponents

MAPPO's opponent pool is opt-in. Omit `selfplay` or set it to `null` to retain
ordinary shared-policy training. To enable it:

```yaml
selfplay:
  pool_size: 8
  snapshot_interval: 100
  current_policy_fraction: 0.5
```

Only environments reporting `supports_selfplay=True` with two controlled agents
are supported, including Lux's two team agents. Cooperative environments and
multi-agent team layouts are rejected.

- At each full-game reset, sample the learner side and opponent. With the configured
  probability, use the current policy; otherwise sample a populated pool slot uniformly.
- Preserve assignments and separate recurrent memories across matches and iterations.
  Train only on the learner side, even in current-policy games. Consequently,
  `num_envs` must be divisible by `num_minibatches * device_count`.
- Snapshot every `snapshot_interval` completed iterations. Keep the initial policy
  permanently and replace the oldest slot unused by active games. Skip insertion
  when no replaceable slot is available. Capacity includes the initial policy.
- Checkpoint the pool. Resume with the current capacity, retaining the initial policy
  and newest history; restart games and assignments. Enabling the pool on an existing
  checkpoint initializes it from the restored learner. Disabling it ignores saved history.

Training returns measure the learner's raw rewards; observation counters count both
agents. Evaluation is unchanged and its reference snapshots remain separate.
Historical inference adds work and currently maps over environments individually
to avoid gathering a full parameter copy for every environment.
