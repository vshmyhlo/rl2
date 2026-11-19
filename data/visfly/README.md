# VisFly evaluation manifests

`eval_manifest.json` is generated offline and used by `configs/ppo_gym/visfly.yaml`.
Final manifests belong in the repository and are bundled into training images.
The ignored `*.scenes/` directories contain resumable per-scene generation records.

With a local VisFly runtime, run from the repository root:

```bash
scripts/generate_visfly_manifest.sh --workers 4
```

On the k3s host, use an existing VisFly image and an idle GPU:

```bash
VISFLY_IMAGE='rl2.local/train:<image-tag>' VISFLY_GPU=0 scripts/generate_visfly_manifest.sh --workers 4
```

Rerun to resume; an existing final manifest is reused. Use a different
`env.evaluation.manifest_path` to generate a new evaluation set.
