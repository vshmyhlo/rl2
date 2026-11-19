# VisFly evaluation manifests

The VisFly preset uses `/runs/visfly/eval_manifest.json` on the shared
`training-runs` volume. Submit generation before training:

```bash
./submit.py --visfly --gpus 1 --memory-gib 32 --cpus 5 -- \
  uv run python -m rl2.envs.visfly.generate_manifest --config configs/ppo_gym/visfly.yaml --workers 4
```

Wait for the generation Job to succeed before submitting training. Every invocation
regenerates all scenes; the final manifest is replaced only after success.

For host generation, set `env.evaluation.manifest_path` to a writable path such as
`data/visfly/eval_manifest.json` and run the Python module in a VisFly runtime.
The ignored `*.scenes/` directories contain per-scene generation records.
