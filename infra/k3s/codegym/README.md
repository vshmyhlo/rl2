# CodeGym environment server

One CPU-only pod runs the pinned upstream manager and 160 workers, with an
8 GiB memory request and 16 GiB limit. English
`envs_en` environments are downloaded at image build time; runtime needs no
internet or credentials. The server has no GPU or service-account token.
Preparation fixes two malformed display f-strings in the pinned dataset's
`Codeforces_5537_I__MinPathCostEnv` and compile-checks every environment. It also generates public tool schemas
from each environment’s `func_mapping`, dispatch code, signatures, and docstrings.
`GET /tools/{environment_name}` serves those schemas to the training adapter.
Private helper methods and example-output sections are excluded. Missing parameter
annotations fall back to documented types for matching wire keys; unknown types stay unrestricted.
Some upstream annotations disagree with their implementations; schemas preserve
those hints, while the adapter validates names and argument keys, not value types.
The individual-tool adapter requires an image built with this schema endpoint.
The adapter requires one tool call per response and tells the model to inspect the
current instance when the initial observation lacks task data. Multi-call responses
are rejected without execution.

From the repository root:

```bash
sudo buildctl --addr unix:///run/rl2-buildkit/buildkitd.sock build \
  --frontend dockerfile.v0 --local context=. --local dockerfile=infra/k3s/codegym \
  --output type=image,name=rl2.local/codegym:5,unpack=true
sudo k3s ctr -n k8s.io images label rl2.local/codegym:5 io.cri-containerd.pinned=pinned
sudo k3s kubectl apply -f infra/k3s/codegym/codegym.yaml
sudo k3s kubectl -n codegym rollout status deployment/codegym --timeout=300s
```

Training pods use `http://codegym.codegym.svc.cluster.local:8000`. This headless
Service resolves to the single server pod; clients reach the manager on 8000
and allocated workers on 10000–10159. Keep one replica: independent managers
cannot share allocations. NetworkPolicy permits ingress from `training` and
`codegym` only and denies outbound connections. Deployment updates interrupt
active episodes; use a new image tag for updates and deploy between runs.

Worker count must cover all concurrent rollouts across jobs, including
prefetched batches. The pool supports one `configs/grpo_v2/codegym.yaml` run:
4 groups × 8 rollouts × (1 current + 4 prefetched batches) = 160 workers.
Concurrent jobs need additional capacity or reduced rollout concurrency.

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
