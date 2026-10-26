# Local k3s development cluster

One server/worker on `vlad-dev-gpu-1`, using its two NVIDIA GPUs.
K3s bundles containerd and kubectl; Docker is not required. Traefik and
ServiceLB are disabled. CoreDNS, metrics-server, and local-path storage retain
their defaults. The admin kubeconfig is root-only and Secrets are encrypted at rest.

## Install

Run from the repository root on a fresh Ubuntu host with systemd and sudo:

```bash
curl --fail --silent --show-error --location https://get.k3s.io -o /tmp/rl2-install-k3s.sh
# Review the installer before running it as root.
sudo install -d -m 0755 /etc/rancher/k3s
sudo install -m 0600 infra/k3s/config.yaml /etc/rancher/k3s/config.yaml
sudo env INSTALL_K3S_VERSION=v1.36.5+k3s1 sh /tmp/rl2-install-k3s.sh server
```

This installs and enables the `k3s` systemd service, including startup at boot.
The version was resolved from the stable channel on 2026-10-14. Do not overwrite
an existing cluster's configuration with these fresh-install commands.

## Check and use

```bash
sudo k3s kubectl wait --for=condition=Ready nodes --all --timeout=180s
sudo k3s kubectl get nodes -o wide
sudo k3s kubectl get pods -A
sudo k3s kubectl get storageclasses
sudo k3s kubectl apply -f infra/k3s/smoke-job.yaml
sudo k3s kubectl wait --for=condition=Complete job/k3s-smoke --timeout=150s
sudo k3s kubectl logs job/k3s-smoke
sudo k3s kubectl delete job k3s-smoke
```

The smoke Job checks image pulling, scheduling, container execution, and cluster
DNS. It should print `k3s-smoke-ok`; it does not test GPU execution. Completed
smoke Jobs also expire after five minutes.

Use `sudo k3s kubectl` to access this cluster without copying admin credentials
or changing another kubectl context. Configuration lives at
`/etc/rancher/k3s/config.yaml`; state and local volumes live under
`/var/lib/rancher/k3s`. Local-path volumes are tied to this machine; keep durable
training checkpoints backed up separately.

```bash
sudo systemctl status k3s --no-pager
sudo journalctl -u k3s -n 100 --no-pager
sudo k3s kubectl get events -A --sort-by=.metadata.creationTimestamp
```

## Persistent dashboard access

Install localhost port-forwards that start at boot and retry after disconnects:

```bash
sudo bash infra/k3s/port-forwards/install.sh
```

This enables `rl2-forward-{headlamp,mlflow,grafana,kueue-frontend,kueue-backend,ray-timelines}`
systemd services on ports **8080–8085**, respectively. Pass a service name to
install only that forward, for example `ray-timelines`. They bind only to
`127.0.0.1`. When browsing from another machine, also forward these ports through
your IDE or SSH; Kueue needs both 8083 and 8084. Skip the manual port-forward
commands below while these services are running. Rerun the installer to update.

```bash
sudo systemctl status 'rl2-forward-*' --no-pager
sudo journalctl -u rl2-forward-headlamp -n 30 --no-pager
```

To stop one forward and disable its boot startup, run
`sudo systemctl disable --now rl2-forward-headlamp` (substitute its service name).

## Headlamp dashboard

Headlamp runs in the `headlamp` namespace, managed by k3s's built-in Helm
controller with the official chart pinned to `0.45.0`. The Kueue plugin is
pinned to `0.1.0-alpha` and downloaded with checksum verification by an init
container on each Pod start. Install or reconcile it:

```bash
sudo k3s kubectl apply -f infra/k3s/headlamp.yaml
sudo k3s kubectl wait --for=create deployment/headlamp -n headlamp --timeout=180s
sudo k3s kubectl rollout status deployment/headlamp -n headlamp --timeout=180s
```

Open a localhost port-forward and leave it running:

```bash
sudo k3s kubectl -n headlamp port-forward --address=127.0.0.1 service/headlamp 8080:80
```

Visit <http://localhost:8080>. If your browser runs on another machine, forward
port 8080 through your IDE's Ports panel or SSH. The service is ClusterIP only;
there is no ingress or public endpoint.

Generate a 30-day login token in another terminal and paste it into Headlamp:

```bash
sudo k3s kubectl -n headlamp create token headlamp-viewer --duration=720h
```

The chart passes `-session-ttl=2592000` through `config.extraArgs` (30 days),
matching the token lifetime. After upgrading, log out and paste a new token once;
existing tokens and browser cookies keep their old expiry. Use the same browser
hostname each time (`localhost` and `127.0.0.1` have separate cookies).

The login account has Kubernetes's built-in `view` role across namespaces, plus
`get`, `list`, and `watch` access to nodes through `headlamp-node-viewer` (the
built-in `view` role excludes nodes) and Kueue resources through
`headlamp-kueue-viewer`. It can inspect nodes, workloads, queues, and logs.
The training installer also grants Job deletion and Pod eviction in `training`;
other writes and access to Secrets remain disallowed. Some cluster administration views are
unavailable. The Headlamp pod uses a
separate service account without a cluster role binding. Tokens are generated
on demand and are not stored in this repository. Generate another after expiry.

To enable Job deletion and Pod eviction on an existing cluster, after creating the `training` namespace:

```bash
sudo k3s kubectl apply -f infra/k3s/training/headlamp-access.yaml
```

In Headlamp, open **Workloads → Jobs**, select namespace **training**, open the
Job, and choose **Delete**. Delete the Job to cancel it and remove it from the
queue; evicting its Pod can cause a replacement to start. Saved `/runs`
checkpoints remain. Refresh Headlamp after changing permissions.

Verify login persistence and permissions with the localhost port-forward running,
without deleting or evicting workloads:

```bash
RL2_K3S_TESTS=1 uv run pytest infra/k3s/tests/test_headlamp_access.py -q
```

Open **Kueue → Workloads** for admission status, priority, requests, and queue
conditions. **Kueue → LocalQueues → training/gpu** and **ClusterQueues → gpu**
show queue usage and related workloads. These alpha plugin views are read-only;
use **Workloads → Jobs** for logs and deleting Jobs. Refresh the browser after
installing the plugin. Existing login tokens retain the updated permissions.

To diagnose installation or startup:

```bash
sudo k3s kubectl -n kube-system get helmchart headlamp
sudo k3s kubectl -n headlamp get pods,services
sudo k3s kubectl -n headlamp logs deployment/headlamp
```

See the [Headlamp installation guide](https://headlamp.dev/docs/latest/installation/in-cluster/).

## MLflow experiment tracking

MLflow 3.17.1 runs in namespace `mlflow`, with SQLite metadata at
`/mlflow/mlflow.db` and artifacts at `/mlflow/artifacts` on the `mlflow-data`
PVC. One replica and one server worker are sufficient for this local
setup. Updates use `Recreate` to avoid overlapping server instances. The
server requests 100m CPU and 512 MiB RAM, has a 2 GiB RAM limit, and uses no GPU.

```bash
sudo k3s kubectl apply -f infra/k3s/mlflow.yaml
sudo k3s kubectl -n mlflow rollout status deployment/mlflow --timeout=300s
sudo k3s kubectl -n mlflow port-forward --address=127.0.0.1 service/mlflow 8081:5000
```

Leave the port-forward running and open <http://localhost:8081>. Forward port
8081 through your IDE if your browser is on another machine. Restart the
port-forward after replacing the server Pod. This local development instance
has no login or license activation. It is reachable through the ClusterIP
service within the cluster and the localhost port-forward on the host.

Log an experiment from the GPU host:

```bash
MLFLOW_TRACKING_URI=http://localhost:8081 uv run --no-project --with mlflow-skinny==3.17.1 python - <<'PYTHON'
import mlflow

mlflow.set_experiment("rl2-smoke")
with mlflow.start_run(run_name="connection-check"):
    mlflow.log_param("purpose", "connection-check")
    mlflow.log_metric("test_metric", 1.0, step=0)
    mlflow.log_text("MLflow artifact upload works.\n", "check.txt")
PYTHON
```

For training Pods, set
`MLFLOW_TRACKING_URI=http://mlflow.mlflow.svc.cluster.local:5000`. Artifacts are
proxied through the tracking server, so both host and Pod clients can upload
and download them without mounting the server's volume. Allowed Host headers
cover localhost and these internal service names.

PPO and MAPPO log metrics, configuration, evaluation reports, and gameplay
videos to MLflow. `submit.py` supplies the tracking URL automatically.

The PVC requests 50 GiB. Local-path storage does not enforce this as a disk
quota. Monitor disk usage and back up the database and artifacts together
while the server is stopped. Deleting the PVC can delete the experiments.

```bash
sudo k3s kubectl -n mlflow get pods,pvc
sudo k3s kubectl -n mlflow logs deployment/mlflow --tail=100
```

W&B's Deployment and Service were removed when switching to MLflow. Its
`wandb/wandb-data` PVC and namespace remain solely to preserve any previous
data; no W&B server is running.

See [MLflow tracking server documentation](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/).

## GPU, host, and job monitoring

Grafana shows the A100's utilization, memory bandwidth utilization, used/total
VRAM, temperatures, power draw, SM/memory clocks, and uncorrectable ECC errors.
Node Exporter adds host CPU utilization and RAM utilization/used/total history.
CPU usage averages non-idle time across logical CPUs; used RAM is total minus
available memory, excluding reclaimable cache. NVIDIA DCGM Exporter collects
every 10 seconds; Prometheus scrapes both exporters every 15 seconds
and retains up to seven days or 4 GB of metrics, whichever limit is reached
first. Grafana and Prometheus have separate persistent volumes.

```bash
sudo k3s kubectl apply -k infra/k3s/monitoring
sudo k3s kubectl -n monitoring wait --for=create daemonset/dcgm-exporter daemonset/node-exporter deployment/gpu-prometheus deployment/gpu-grafana --timeout=180s
sudo k3s kubectl -n monitoring rollout status daemonset/dcgm-exporter --timeout=180s
sudo k3s kubectl -n monitoring rollout status daemonset/node-exporter --timeout=180s
sudo k3s kubectl -n monitoring rollout status deployment/gpu-prometheus --timeout=180s
sudo k3s kubectl -n monitoring rollout status deployment/gpu-grafana --timeout=180s
sudo k3s kubectl -n monitoring rollout status deployment/training-kube-state-metrics --timeout=180s
sudo k3s kubectl -n monitoring port-forward --address=127.0.0.1 service/gpu-grafana 8082:80
```

Open <http://localhost:8082/d/gpu-overview> and forward port 8082 in your IDE
if needed. Anonymous users have Editor access, including Explore, all data
sources, and dashboard editing, without signing in. The dashboard refreshes every
15 seconds; use its time picker to inspect earlier training runs. No GPU
resource is reserved by the exporter, so the A100 remains available for training.

The chart-generated administrator credentials are available locally if you
need to edit Grafana settings. Username: `admin`; retrieve the password with:

```bash
sudo k3s kubectl -n monitoring get secret gpu-grafana -o jsonpath='{.data.admin-password}' | base64 --decode
```

Dashboards are provisioned from `monitoring/*-dashboard.json`. Edit those
files and reapply the Kustomization to change them. All services use ClusterIP;
the localhost port-forward provides browser access. Chart versions are pinned
in `monitoring/monitoring.yaml` and k3s's Helm controller manages installation.

This dashboard reports physical GPU telemetry, including work launched directly
on the host. It does not report per-process utilization or tensor-core profiling.
Queries group by GPU UUID and stable device labels so changing training Pods
does not create duplicate GPU entries or split their history.
The exporter reads the kubelet's pod-resource socket for GPU assignment labels;
those labels identify GPU assignments for training Pods scheduled through the
NVIDIA device plugin. It uses the existing NVIDIA runtime and basic DCGM counters
without additional Linux capabilities.
Node Exporter collects only CPU and memory metrics from the host's read-only
`/proc` and `/sys` mounts, without host networking or a root filesystem mount.
Its static scrape target is pinned to `vlad-dev-gpu-1`, matching its node selector.

Open [Training job resources](http://localhost:8082/d/training-job) and select a
Job to see CPU cores used/requested/limited, RAM working set/RSS/requests/limits,
CPU throttling, network traffic, disk I/O, container restarts, Pod phases, and
assigned-GPU utilization/VRAM. Collection is automatic for Jobs in `training`,
including `submit.py` jobs; no trainer flags or resubmission are needed.
Prometheus scrapes kubelet cAdvisor through the authenticated API server proxy
and kube-state-metrics every 15 seconds. The latter can list/watch only training
Pods; Prometheus can also list/watch training Pods and GET the proxy for
`vlad-dev-gpu-1` only. Only selected
container metrics from `training` are stored.

Queries join Pod ownership at each historical timestamp, so retries and deleted
Pods remain visible within retention. Select a time range covering the run to
find completed/deleted jobs. Metrics begin after installation and are retained
for up to seven days or 4 GB together with host/GPU metrics; there is no backfill.
Queued Jobs have no usage, very short runs may fall between scrapes, and rates
need at least two samples. Disk counters depend on kubelet support. Assigned-GPU
panels show device telemetry while assigned, not per-process GPU usage.

Open [vLLM performance](http://localhost:8082/d/training-vllm-performance) or
[vLLM queries](http://localhost:8082/d/training-vllm-query), then select a Job
and optionally a model. Both are also linked from **Training job resources**,
preserving the selected Job and time range. These adapt the official
[vLLM dashboards](https://docs.vllm.ai/en/stable/examples/observability/dashboards/)
from [v0.27.0](https://github.com/vllm-project/vllm/tree/v0.27.0/examples/observability/dashboards/grafana)
(Apache-2.0), covering request rates, token throughput, token sizes, E2E latency,
time to first token, and inter-token latency. Both dashboards also show prefix
cache hit rate (weighted by queried tokens) and KV cache utilization per Pod
and engine. The hit-rate graph has gaps when no cache queries occur.

Prometheus discovers running Job Pods in `training` with a container port named
`vllm` and scrapes `/metrics` every 15 seconds. It stores only `vllm:*` metrics,
with namespace, Job, and Pod labels attached at scrape time. `submit.py` declares
port 8000 and sets `RL2_VLLM_HOST=0.0.0.0`; GRPO v2 uses that bind address while
local runs default to loopback. The vLLM API is reachable on the Pod network;
no host port or public Service is added. Resubmit existing GRPO v2 jobs to pick
up these changes. Custom launchers must bind to the Pod network and declare
their metrics port as `vllm`. Jobs without a vLLM server have no vLLM data and
their scrape target stays down. Historical samples share the retention above.

```bash
sudo k3s kubectl -n monitoring get pods,pvc
sudo k3s kubectl -n monitoring logs daemonset/dcgm-exporter --tail=50
sudo k3s kubectl -n monitoring logs deployment/gpu-prometheus -c prometheus-server --tail=50
```

See [NVIDIA DCGM Exporter documentation](https://docs.nvidia.com/datacenter/dcgm/latest/installation/install-dcgm-exporter.html).

## OpenTelemetry traces with Tempo

The monitoring Kustomization also deploys Tempo `3.1.0` and provisions the
**Tempo** data source in Grafana. It runs as one process with a `10Gi`
local-path PVC and 14-day trace retention. This is a single-node development
setup; the PVC's requested size is not a hard disk quota.

```bash
sudo k3s kubectl apply -k infra/k3s/monitoring
sudo k3s kubectl -n monitoring rollout status deployment/tempo --timeout=180s
sudo k3s kubectl -n monitoring rollout status deployment/gpu-grafana --timeout=180s
```

Open [Grafana Explore](http://localhost:8082/explore) and select **Tempo**.
No login is required.
Choose **Search** to filter by service, span name, duration, or attributes,
or use TraceQL `{}` to list traces in the selected time range. Open a trace
to inspect its span waterfall, attributes, and errors.

Applications inside k3s can export OTLP to:

| Protocol | Endpoint |
|---|---|
| gRPC | `http://tempo.monitoring.svc.cluster.local:4317` |
| HTTP/protobuf | `http://tempo.monitoring.svc.cluster.local:4318/v1/traces` |

For an instrumented process on the host, forward the receiver:

```bash
sudo k3s kubectl -n monitoring port-forward --address=127.0.0.1 service/tempo 4317:4317 4318:4318
```

Use `http://127.0.0.1:4317` for gRPC or
`http://127.0.0.1:4318/v1/traces` for HTTP traces. Tempo's query API is internal
on port `3200`; Grafana proxies queries to it. No receiver is exposed publicly.
GRPO v2 jobs export traces when their config sets `enable_tracing: true`
(disabled by default). `submit.py` supplies the Tempo endpoint automatically.
See [GRPO tracing](../../src/rl2/train_grpo_v2.md) for local configuration.
Ray timeline files remain separate from OpenTelemetry traces.

Configuration lives in `monitoring/tempo-config.yaml`; reapply the Kustomization
to roll out changes. See [Tempo deployment documentation](https://grafana.com/docs/tempo/latest/set-up-for-tracing/setup-tempo/).

## GPU jobs with Kueue

Kueue `0.20.1` manages labeled Kubernetes batch Jobs in the `training`
namespace. NVIDIA's device plugin `0.20.1` advertises two units of
`nvidia.com/gpu`; GPU sharing and MIG are not enabled.

The `gpu` LocalQueue maps to the `gpu` ClusterQueue, with quota for two GPUs,
10 CPUs, and 64 GiB RAM on `vlad-dev-gpu-1`. This leaves resources for cluster
services. `BestEffortFIFO` admits eligible Jobs in queue order within a
priority level, allowing a Job that fits to bypass one that cannot fit.
Each Job gets exclusive Kubernetes allocation of its requested GPUs. With
`withinClusterQueue: LowerPriority`, a higher-priority Job can evict a running
lower-priority Job; otherwise waiting Jobs remain suspended without a Pod.

Install controllers first, then queue resources:

```bash
sudo k3s kubectl apply -f infra/k3s/kueue/controllers.yaml
sudo k3s kubectl -n kueue-system wait --for=create deployment/kueue-controller-manager --timeout=180s
sudo k3s kubectl -n kueue-system rollout status deployment/kueue-controller-manager --timeout=180s
sudo k3s kubectl -n kube-system wait --for=create daemonset/nvidia-device-plugin --timeout=180s
sudo k3s kubectl -n kube-system rollout status daemonset/nvidia-device-plugin --timeout=180s
sudo k3s kubectl apply -f infra/k3s/kueue/queues.yaml
sudo k3s kubectl wait --for=condition=Active clusterqueue/gpu --timeout=60s
```

Submit a short GPU-access check (use `create`, because each submission gets a
new generated name):

```bash
sudo k3s kubectl create -f infra/k3s/kueue/gpu-smoke-job.yaml
sudo k3s kubectl -n training get jobs,workloads
sudo k3s kubectl get clusterqueue gpu -o wide
sudo k3s kubectl -n training get localqueue gpu -o wide
```

The sample runs `nvidia-smi` and prints `gpu-queue-ok`; it verifies device
access, not CUDA computation. Finished sample Jobs expire after ten minutes.
Inspect a Job by substituting its generated name:

```bash
sudo k3s kubectl -n training logs job/<job-name>
sudo k3s kubectl -n training describe job <job-name>
sudo k3s kubectl -n training describe workload <workload-name>
sudo k3s kubectl -n training delete job <job-name>
```

For a training Job, copy the sample and replace its image and command. Keep:

- Namespace `training` and label `kueue.x-k8s.io/queue-name: gpu` on the Job.
- `spec.suspend: true` so Kueue controls admission.
- `runtimeClassName: nvidia` in the Pod spec and `limits: {nvidia.com/gpu: 1}`
  on the training container.
- Explicit CPU and memory requests within the queue's quota. Increase the
  sample's memory limit and remove or extend its 120-second active deadline.

The template includes the internal MLflow tracking URL. Use `submit.py` below
to build and queue PPO/MAPPO training jobs. Kueue does not control processes
launched directly on the host or Jobs without its queue
label. Its quota tracks declared requests, not actual usage.

The host keeps its existing NVIDIA driver and Container Toolkit. The device
plugin uses the `nvidia` runtime; no GPU Operator or driver replacement is
needed. See [Kueue's quick start](https://kueue.sigs.k8s.io/docs/getting-started/quick-start/).

### Build and queue training jobs

Run `submit.py` on this k3s host to build the current working tree and submit a
GPU Job to `training/gpu` (one GPU by default):

```bash
./submit.py --prio high -- uv run mappo --config configs/mappo/jaxnav.yaml
./submit.py --prio low -- uv run ppo --config configs/ppo/kinetix.yaml
./submit.py --prio low -- uv run python path/to/train.py
./submit.py --gpus 2 -- uv run python path/to/multi_gpu_train.py
```

`--gpus` accepts `1` or `2` per Job, including each trial in a sweep. The NVIDIA
device plugin and runtime expose only the allocated GPUs inside the container;
leave `NVIDIA_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` to the runtime and trainer.
The training command must itself support multiple GPUs to use both.
After upgrading from one GPU, apply the updated queue quota:

```bash
sudo k3s kubectl apply -f infra/k3s/kueue/queues.yaml
```

The submitter uses the project's installed Python environment through `uv run
--no-sync`; run it from the repository root after installing dependencies.

For [BFCL evaluation](../../docs/bfcl.md#kubernetes-job), include its dependencies
with `--extra bfcl` and run the launcher in a single container:

```bash
./submit.py --extra bfcl --memory-gib 32 -- \
  python -m rl2.bfcl.job --run-dir /runs/bfcl/qwen3-0.6b
```

The launcher starts vLLM, waits for readiness, generates responses, stops vLLM,
and scores the results. Logs and results persist under `/runs`. Add `--smoke`
after the run directory for a four-case check in a separate directory.

To submit a parameter grid, define a sweep YAML:

```yaml
name: foragax_square_wave
config: configs/ppo/foragax/square_wave_two_biome.yaml
ckpt_dir: /runs/ppo/${run_id}
parameters:
  learning_rate: [0.0001, 0.00025, 0.001]
  seed: [1, 2, 3]
  env.period: [512, 1024]
```

`config` is relative to the repository root. `name` defaults to the sweep file's
stem. Parameter keys use dotted paths, and each list supplies one grid axis.
Duplicate configurations produce only one trial. Preview this 18-trial example,
then submit it by removing `--dry-run`:

```bash
./submit.py --dry-run --sweep configs/sweeps/foragax_square_wave.yaml -- uv run ppo
./submit.py --sweep configs/sweeps/foragax_square_wave.yaml -- uv run ppo
```

Omit `--config` from the training command: the submitter supplies a generated
config for each trial. It builds one image for the entire sweep, validates every
Job manifest before building, and queues separate Jobs with `--gpus` GPUs each.
The queue admits up to two one-GPU Jobs or one two-GPU Job at a time, subject to
the shared CPU and memory quotas. `--dry-run` only expands the plan; it does
not contact Kubernetes or validate the trainer's configuration schema.

Sweep run IDs are `<name>_<hash>` of the resolved, supplied training settings,
excluding `run_id` and `ckpt_dir`. Repeating a submission resumes those runs;
reordering or extending the grid preserves existing trial IDs. Changing a trial's
settings changes its hash. Sweep IDs replace the base config's `run_id`.
Optional `ckpt_dir` overrides the base config's path; if both omit it, sweeps use
`/runs/${run_id}`. Generated YAML preserves interpolation for the trainer to resolve.
Paths are used exactly as configured; include `${run_id}` for separate trial directories.
Regular runs keep explicit IDs, or derive stable IDs when none is supplied.
Resumption uses the latest saved checkpoint, or starts training if none exists.

Submission records under `runs/sweeps/<name>/` contain the generated configs and
Job manifests, including attempted jobs if submission fails partway through.
Already queued jobs remain queued on failure. Inspect a sweep with:

```bash
sudo k3s kubectl -n training get jobs -l rl2.local/sweep=foragax_square_wave
```

One-time host setup (installs BuildKit `v0.34.0`, enables the root
`rl2-buildkit` service and image-pin cleanup timer, creates two Kueue priority
classes, and provisions run storage; rerun to update an existing installation):

```bash
sudo bash infra/k3s/training/install.sh
```

BuildKit uses a root-only Unix socket and k3s's containerd image store. Each
submission gets a unique local image tag; no registry is needed. Source edits
and untracked files are included, subject to `.dockerignore`, which excludes
virtual environments, generated outputs, and common credential paths. Dependency
layers are cached using `uv.lock`. JAX and PyTorch/vLLM share CUDA 13.0;
`cuda-toolkit[cccl,nvcc]` pins matching compiler, headers, and runtime packages.
The image exposes that toolkit through `CUDA_HOME` and checks compiler/header
compatibility during the build. Inside the image, `uv run` uses the built
environment without syncing dependencies again.

Before creating a Job, the submitter pins its image in containerd so kubelet
cleanup cannot remove it while queued or preempted. The `rl2-image-pins.timer`
reconciles pins every minute, releasing them after Jobs finish or are deleted.
If the Kubernetes API is unavailable, pins are preserved. Unpinning allows
normal kubelet cleanup; it does not immediately delete images or BuildKit cache.
Submission requires the timer to be active. Check it with:

```bash
sudo systemctl status rl2-image-pins.timer --no-pager
sudo journalctl -u rl2-image-pins.service -n 20 --no-pager
```

Verify image pinning and release with a tiny CPU-only Job:

```bash
RL2_K3S_TESTS=1 uv run pytest infra/k3s/tests/test_image_lifecycle.py -q
```

The non-secret `nvida.env` configuration is included in the image and loaded by
`uv run` automatically through `UV_ENV_FILE=/app/nvida.env`. Use `uv run
--no-env-file` to disable it. Other `.env` files remain excluded. Never
put credentials in `nvida.env`; supply secrets at runtime. Resubmit after changing
this file so the job uses a newly built image.

Defaults are `--prio low --gpus 1 --cpus 4 --memory-gib 32`. High priority moves ahead of
waiting low-priority Jobs and preempts running low-priority Jobs when quota is
full. Kueue suspends the low-priority Job and deletes its Pod, then readmits the
same Job with a new Pod when higher-priority work finishes. Equal-priority Jobs
do not preempt each other. Apply the policy to an existing cluster:

```bash
sudo k3s kubectl apply -f infra/k3s/kueue/queues.yaml -f infra/k3s/training/priorities.yaml
```

Readmission restarts the command; it does not preserve process or GPU memory.
PPO/MAPPO restore the latest saved checkpoint from the same `ckpt_dir`. Use durable
storage (`/runs` or the configured GCS path): container-local files disappear on
eviction, and progress since the last checkpoint is lost. Preemption does not
force a new checkpoint. Keep different runs' checkpoint directories distinct.

The command after `--` runs from `/app` with its arguments preserved, plus the
internal MLflow URL.
Use `-- sh -c '...'` explicitly for shell pipelines. Submission returns after
queueing and prints status/log commands. Failed Jobs are not retried, and
finished Jobs expire after seven days.

Only the image is a repository snapshot; local credentials are not mounted.
Outputs outside `/runs` are ephemeral unless written to external storage.
Builds and images consume host disk; inspect cache usage with
`sudo buildctl --addr unix:///run/rl2-buildkit/buildkitd.sock du`.

Remove all unused build cache (including dependency download caches and frontend
references) without deleting container images or Jobs:

```bash
sudo bash infra/k3s/training/cleanup-build-cache.sh
```

The script forwards `buildctl prune` options; add `--keep-duration 24h` to retain
recent cache. Subsequent builds may take longer while rebuilding pruned layers.

To also remove incomplete `*.orbax-checkpoint-tmp` directories on the training
volume and the invoking user's Hugging Face download caches:

```bash
bash infra/k3s/training/cleanup-disk.sh          # Preview only
bash infra/k3s/training/cleanup-disk.sh --apply
```

Stop local training/downloads and new job submissions first. The script refuses
cleanup while training Job pods are nonterminal; this check does not lock out
new writers. By default, completed checkpoints and Hugging Face tokens are preserved.
`HF_HOME` overrides the default Hugging Face cache location. Downloads and build
cache will be recreated when needed.

Optionally remove whole GRPO v2 runs, including completed checkpoints, after
**7 days without modifications**:

```bash
bash infra/k3s/training/cleanup-disk.sh --run-retention-days 7
bash infra/k3s/training/cleanup-disk.sh --run-retention-days 7 --apply
```

The age is configurable and uses the newest file or directory modification time
within each run under `/runs/grpo_v2`. Run deletion is disabled without the flag.
Queued or suspended training Jobs also block this cleanup because they may need
checkpoints to resume. This runs only when invoked; it does not install a schedule.

To check GPU visibility and preemption on an idle queue (creates and cleans up
two short Jobs; the low-priority Job requests the full GPU quota):

```bash
RL2_K3S_TESTS=1 uv run pytest infra/k3s/tests/test_preemption.py -q -s
```

### Shared on-disk checkpoints

Every newly submitted Job mounts the `training/training-runs` PVC at `/runs`.
It uses this node's disk through `local-path`, with a 50 GiB storage request
(not an enforced filesystem quota). Install it on an existing cluster:

```bash
sudo k3s kubectl apply -f infra/k3s/training/storage.yaml
```

Configs under `configs/ppo/` and `configs/mappo/` set `ckpt_dir` to
`/runs/ppo/${run_id}` and `/runs/mappo/${run_id}`, respectively. Choose a separate
`run_id` in YAML for each experiment; reuse it to resume:

```bash
./submit.py --prio low -- uv run mappo --config configs/mappo/jaxnav.yaml
./submit.py --prio high -- uv run ppo --config configs/ppo/minatar/breakout.yaml
```

Both trainers write to `<ckpt_dir>/checkpoints` and restore the latest completed
checkpoint automatically. To change the directory, edit `ckpt_dir` in YAML.
Files survive Pod replacement, preemption, and Job deletion. Do not delete the
PVC: its storage class deletes the backing directory. This is shared storage on
one machine, not a backup. Existing Jobs need resubmission to receive the mount.

Open [Training jobs](http://localhost:8085/) to list running and queued jobs.
The list refreshes every 10 seconds. Click **Ray dashboard** to open a running
job's dashboard, **Timeline** to load its saved trace, **Job metrics** for its
Grafana resource history since creation, or **Tempo traces** to search its `rl2`
traces over the last hour. Grafana links also appear in the timeline view;
the metrics link there covers the last seven days, even without a saved snapshot.
Forward Grafana port **8082** as well. Install or update the viewer:

```bash
sudo bash infra/k3s/ray-timelines/install.sh
```

Forward **8085** through your IDE or SSH for timelines and Ray dashboards. Dashboards are proxied at
`/dashboard/<job-name>/`; no per-job port-forward is needed. The viewer's service
account can list Jobs and Pods only in `training`. Dashboard targets are resolved
from running Pods owned by those Jobs. Submitted GRPO v2 jobs bind Ray to the Pod
network; dashboards stop when Ray exits. Set `enable_ray_dashboard: true` in the
training YAML to enable the dashboard (disabled by default).

GRPO v2 sets `RAY_AUTH_MODE=disabled` before starting Ray, so dashboards open without
a login token. Existing jobs need resubmission for this change.

With `enable_ray_timeline: true`, submitted GRPO v2 jobs save Ray timeline snapshots to
`/runs/ray-timelines/<job-name>.json` every minute and on exit.
`submit.py` prints the timeline link; the driver logs it when profiling is enabled.

Forward **8085** through your IDE or SSH and open the printed
`http://localhost:8085/?job=<job-name>` link. The page waits for the first snapshot,
then opens it in Perfetto. Use **Load latest** to refresh. Links remain usable after
job completion/deletion while their files remain on the PVC. The service mounts
the PVC read-only, serves only the dedicated trace directory, and uses browser
`postMessage` to load traces into Perfetto without public artifact uploads.
Rerun the installer after changing the viewer. Existing jobs need resubmission
to receive the path/link environment variables; the viewer does not enable
profiling for trainers other than GRPO v2.

Locate the backing directory on this host:

```bash
run_pv=$(sudo k3s kubectl -n training get pvc training-runs -o jsonpath='{.spec.volumeName}')
sudo k3s kubectl get pv "$run_pv" -o jsonpath='{.spec.local.path}{.spec.hostPath.path}{"\n"}'
```

The volume binds on first use. Verify persistence using two small CPU-only Pods:

```bash
RL2_K3S_TESTS=1 uv run pytest infra/k3s/tests/test_storage.py -q
```

### Kueue UI

The controller chart also installs KueueViz `0.20.1`, with read-only access to
queues, workloads, resource flavors, and node information. It has no login in
this local development setup; both services remain ClusterIP without ingress.

Start these port-forwards in separate terminals:

```bash
sudo k3s kubectl -n kueue-system port-forward --address 127.0.0.1 svc/kueue-kueueviz-frontend 8083:8080
sudo k3s kubectl -n kueue-system port-forward --address 127.0.0.1 svc/kueue-kueueviz-backend 8084:8080
```

Open <http://localhost:8083>. With a remote IDE or SSH session, forward **both
8083 and 8084** to the same ports on your laptop: the browser connects directly
to the backend at `ws://localhost:8084`. Keep these forwards bound to localhost.
Select ClusterQueues → `gpu` or LocalQueues → `training/gpu` to inspect quota
and admitted/pending workloads. GPU utilization remains in Grafana on port 8082.

See [KueueViz documentation](https://kueue.sigs.k8s.io/v0.20/docs/tasks/manage/enable_kueueviz/).

References: [K3s configuration](https://docs.k3s.io/installation/configuration),
[NVIDIA runtime integration](https://docs.k3s.io/advanced#nvidia-container-runtime),
[network requirements](https://docs.k3s.io/installation/requirements#networking).
Cluster ports should remain restricted to this development environment.
