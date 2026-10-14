# Local k3s development cluster

One server/worker on `vlad-dev-gpu-1`, using its single A100 40 GB GPU.
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

## Headlamp dashboard

Headlamp runs in the `headlamp` namespace, managed by k3s's built-in Helm
controller with the official chart pinned to `0.45.0`. Install or reconcile it:

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

Generate a one-hour login token in another terminal and paste it into Headlamp:

```bash
sudo k3s kubectl -n headlamp create token headlamp-viewer --duration=1h
```

The login account has Kubernetes's built-in `view` role across namespaces, plus
`get`, `list`, and `watch` access to nodes through `headlamp-node-viewer` (the
built-in `view` role excludes nodes). It can inspect nodes, workloads, and logs,
but cannot edit resources or read Secrets;
some cluster administration views are unavailable. The Headlamp pod uses a
separate service account without a cluster role binding. Tokens are generated
on demand and are not stored in this repository. Generate another after expiry.

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
PVC. One replica and one server worker are sufficient for this single-GPU
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

The existing PPO and MAPPO trainers still use TensorBoard; they need MLflow
logging integration before their runs appear here. This deployment tracks
experiments; submit GPU jobs through the Kueue configuration below.

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

## GPU monitoring

Grafana shows the A100's utilization, memory bandwidth utilization, used/total
VRAM, temperatures, power draw, SM/memory clocks, and uncorrectable ECC errors. NVIDIA
DCGM Exporter collects every 10 seconds; Prometheus scrapes every 15 seconds
and retains up to seven days or 4 GB of metrics, whichever limit is reached
first. Grafana and Prometheus have separate persistent volumes.

```bash
sudo k3s kubectl apply -k infra/k3s/monitoring
sudo k3s kubectl -n monitoring rollout status daemonset/dcgm-exporter --timeout=180s
sudo k3s kubectl -n monitoring rollout status deployment/gpu-prometheus --timeout=180s
sudo k3s kubectl -n monitoring rollout status deployment/gpu-grafana --timeout=180s
sudo k3s kubectl -n monitoring port-forward --address=127.0.0.1 service/gpu-grafana 8082:80
```

Open <http://localhost:8082/d/gpu-overview> and forward port 8082 in your IDE
if needed. Anonymous access is read-only. The dashboard refreshes every
15 seconds; use its time picker to inspect earlier training runs. No GPU
resource is reserved by the exporter, so the A100 remains available for training.

The chart-generated administrator credentials are available locally if you
need to edit Grafana settings. Username: `admin`; retrieve the password with:

```bash
sudo k3s kubectl -n monitoring get secret gpu-grafana -o jsonpath='{.data.admin-password}' | base64 --decode
```

The dashboard is provisioned from `monitoring/gpu-dashboard.json`. Edit that
file and reapply the Kustomization to change it. All services use ClusterIP;
the localhost port-forward provides browser access. Chart versions are pinned
in `monitoring/monitoring.yaml` and k3s's Helm controller manages installation.

This dashboard reports physical GPU telemetry, including work launched directly
on the host. It does not report per-process utilization or tensor-core profiling.
The exporter reads the kubelet's pod-resource socket for GPU assignment labels;
those labels identify GPU assignments for training Pods scheduled through the
NVIDIA device plugin. It uses the existing NVIDIA runtime and basic DCGM counters
without additional Linux capabilities.

```bash
sudo k3s kubectl -n monitoring get pods,pvc
sudo k3s kubectl -n monitoring logs daemonset/dcgm-exporter --tail=50
sudo k3s kubectl -n monitoring logs deployment/gpu-prometheus -c prometheus-server --tail=50
```

See [NVIDIA DCGM Exporter documentation](https://docs.nvidia.com/datacenter/dcgm/latest/installation/install-dcgm-exporter.html).

## GPU jobs with Kueue

Kueue `0.20.1` manages labeled Kubernetes batch Jobs in the `training`
namespace. NVIDIA's device plugin `0.20.1` advertises the A100 as one
`nvidia.com/gpu`; GPU sharing and MIG are not enabled.

The `gpu` LocalQueue maps to the `gpu` ClusterQueue, with quota for one GPU,
10 CPUs, and 64 GiB RAM on `vlad-dev-gpu-1`. This leaves resources for cluster
services. `BestEffortFIFO` admits eligible Jobs in queue order within a
priority level, allowing a Job that fits to bypass one that cannot fit.
Each Job requesting one GPU gets exclusive Kubernetes allocation until it
finishes; the next Job remains suspended without creating a Pod.

Install controllers first, then queue resources:

```bash
sudo k3s kubectl apply -f infra/k3s/kueue/controllers.yaml
sudo k3s kubectl -n kueue-system wait --for=create deployment/kueue-controller-manager --timeout=180s
sudo k3s kubectl -n kueue-system rollout status deployment/kueue-controller-manager --timeout=180s
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

The template includes the internal MLflow tracking URL. Building the training
image and adding MLflow calls to PPO/MAPPO are still separate work. Kueue does
not control processes launched directly on the host or Jobs without its queue
label. Its quota tracks declared requests, not actual usage.

The host keeps its existing NVIDIA driver and Container Toolkit. The device
plugin uses the `nvidia` runtime; no GPU Operator or driver replacement is
needed. See [Kueue's quick start](https://kueue.sigs.k8s.io/docs/getting-started/quick-start/).

### Build and queue training jobs

Run `submit.py` on this k3s host to build the current working tree and submit a
one-GPU Job to `training/gpu`:

```bash
./submit.py --prio high -- uv run mappo --config configs/mappo/jaxnav.yaml
./submit.py --prio low -- uv run ppo --config configs/ppo/kinetix.yaml
./submit.py --prio low -- uv run python path/to/train.py
```

One-time host setup (installs BuildKit `v0.34.0`, enables the root
`rl2-buildkit` service, and creates two Kueue priority classes):

```bash
sudo bash infra/k3s/training/install.sh
```

BuildKit uses a root-only Unix socket and k3s's containerd image store. Each
submission gets a unique local image tag; no registry is needed. Source edits
and untracked files are included, subject to `.dockerignore`, which excludes
virtual environments, generated outputs, and common credential paths. Dependency
layers are cached using `uv.lock`; CUDA12 and the environment packages are
regular project dependencies. Inside the image, `uv run` uses the
built environment without syncing dependencies again.

The non-secret `nvida.env` configuration is included in the image and loaded by
`uv run` automatically through `UV_ENV_FILE=/app/nvida.env`. Use `uv run
--no-env-file` to disable it. Other `.env` files remain excluded. Never
put credentials in `nvida.env`; supply secrets at runtime. Resubmit after changing
this file so the job uses a newly built image.

Defaults are `--prio low --cpus 4 --memory-gib 16`. High priority moves ahead of
waiting low-priority Jobs; running Jobs are not preempted. The command after
`--` runs from `/app` with its arguments preserved, plus the internal MLflow URL.
Use `-- sh -c '...'` explicitly for shell pipelines. Submission returns after
queueing and prints status/log commands. Failed Jobs are not retried, and
finished Jobs expire after seven days.

Only the image is a repository snapshot: local credentials and host files are
not mounted. Container-local outputs are ephemeral; use a durable `log_dir`
(the existing configs use GCS) and provide access to that storage at runtime.
Builds and images consume host disk; inspect cache usage with
`sudo buildctl --addr unix:///run/rl2-buildkit/buildkitd.sock du`.

### Submit jobs with kjobs

[kjobs](https://github.com/kubernetes-sigs/kjob) `v0.1.0` provides the
`kubectl kjob` CLI and cluster-side templates. Install on this Linux amd64 host:

```bash
curl -fsSL https://github.com/kubernetes-sigs/kjob/releases/download/v0.1.0/kubectl-kjob-linux-amd64.tar.gz -o /tmp/rl2-kjob.tar.gz
printf '%s\n' '34fd7e04cacfd49dd361453819ab460101ddef98f94b5fb4a76430ccb1e3a92a  /tmp/rl2-kjob.tar.gz' | sha256sum --check
mkdir -p /tmp/rl2-kjob-bin
tar -xzf /tmp/rl2-kjob.tar.gz -C /tmp/rl2-kjob-bin ./kubectl-kjob
sudo install -m 0755 /tmp/rl2-kjob-bin/kubectl-kjob /usr/local/bin/kubectl-kjob
sudo k3s kubectl apply --server-side -f https://github.com/kubernetes-sigs/kjob/releases/download/v0.1.0/manifests.yaml
sudo k3s kubectl wait --for=condition=Established crd/applicationprofiles.kjobctl.x-k8s.io crd/jobtemplates.kjobctl.x-k8s.io --timeout=60s
sudo k3s kubectl apply -f infra/k3s/kjobs/gpu-profile.yaml
```

The `gpu-smoke` profile requests one GPU and uses the NVIDIA runtime. Submit:

```bash
sudo kubectl kjob create job --kubeconfig /etc/rancher/k3s/k3s.yaml \
  -n training --profile gpu-smoke --localqueue gpu \
  --skip-localqueue-validation --cmd 'nvidia-smi'
sudo kubectl kjob list job --kubeconfig /etc/rancher/k3s/k3s.yaml -n training
sudo k3s kubectl -n training logs job/<generated-job-name>
```

Add `--dry-run=server -o yaml` to preview a submission. The explicit kubeconfig
is needed because this plugin does not inherit k3s's default config path.
`--skip-localqueue-validation` skips kjobs v0.1.0's lookup through the retired
Kueue `v1beta1` API; the queue label and Kueue admission still apply. Specify the
existing `gpu` queue; an incorrect name leaves the Job suspended.

This profile uses a CUDA base image for smoke checks, with a 120-second runtime
limit and ten-minute cleanup. For training, create a profile with your training
image, command, resource requests, and runtime limit. It already supplies the
internal `MLFLOW_TRACKING_URI`; kjobs does not upload local code or dependencies.

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
