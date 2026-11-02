# CodeGym environment server

One CPU-only pod runs the pinned upstream manager and 40 workers. English
`envs_en` environments are downloaded at image build time; runtime needs no
internet or credentials. The server has no GPU or service-account token.
Preparation fixes two malformed display f-strings in the pinned dataset's
`Codeforces_5537_I__MinPathCostEnv` and compile-checks every environment.

From the repository root:

```bash
sudo buildctl --addr unix:///run/rl2-buildkit/buildkitd.sock build \
  --frontend dockerfile.v0 --local context=. --local dockerfile=infra/k3s/codegym \
  --output type=image,name=rl2.local/codegym:3,unpack=true
sudo k3s ctr -n k8s.io images label rl2.local/codegym:3 io.cri-containerd.pinned=pinned
sudo k3s kubectl apply -f infra/k3s/codegym/codegym.yaml
sudo k3s kubectl -n codegym rollout status deployment/codegym --timeout=300s
```

Training pods use `http://codegym.codegym.svc.cluster.local:8000`. This headless
Service resolves to the single server pod; clients reach the manager on 8000
and allocated workers on 10000–10039. Keep one replica: independent managers
cannot share allocations. NetworkPolicy permits ingress from `training` and
`codegym` only and denies outbound connections. Deployment updates interrupt
active episodes; use a new image tag for updates and deploy between runs.

Worker count must cover all concurrent rollouts across jobs, including
prefetched batches. The CodeGym preset matches `mix.yaml`: 32 rollouts per batch
and four prefetched batches require up to 160 concurrent workers. Before using
that preset, increase the deployed 40-worker pool and NetworkPolicy port range,
or set `prefetch_batches: 0` to fit the existing pool.

```bash
sudo k3s kubectl -n codegym get pods
sudo k3s kubectl -n codegym logs deployment/codegym --tail=50
sudo k3s kubectl -n codegym exec deployment/codegym -- tail -n 50 logs/env_10000.log
```

`/ready` checks every worker; `/health` lists worker ports. Worker logs are
bounded by upstream rotation and live on an ephemeral volume. Episodes are
in-memory and do not survive pod replacement. Upstream reclaims abandoned
allocations after approximately three hours; normal clients stop and release
sessions on exit.

For host-side diagnostics on the k3s node, use the pod IP as the manager host
(the cluster DNS name is for pods). A manager-only port-forward is insufficient:
worker ports must also be reachable. The server is not exposed publicly.

Run adapter unit tests with `uv run pytest tests/test_codegym_env.py`.
Set `CODEGYM_MANAGER_URL=http://<pod-ip>:8000` to include live checks of
session isolation, rewards, and the repaired upstream environment.
