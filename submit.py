#!/usr/bin/env -S uv run --no-sync python
"""Build this working tree once and submit GPU Jobs to the local k3s queue."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException
from yaml import YAMLError

from rl2.sweeps import load_sweep

ROOT = Path(__file__).resolve().parent
BUILD_ADDRESS = "unix:///run/rl2-buildkit/buildkitd.sock"
IMAGE_MANAGER = "/usr/local/lib/rl2/image_pins.py"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, usage="%(prog)s [options] -- COMMAND [ARG ...]")
    parser.add_argument("--prio", choices=("low", "high"), default="low")
    parser.add_argument("--gpus", type=int, choices=range(1, 3), default=1, help="GPUs per job (default: 1)")
    parser.add_argument("--cpus", type=int, choices=range(1, 11), default=10, metavar="1..10")
    parser.add_argument("--memory-gib", type=int, choices=range(1, 65), default=32, metavar="1..64")
    parser.add_argument("--extra", choices=("bfcl",), help="Include an optional dependency group in the image")
    parser.add_argument(
        "--python-sandbox", action="store_true", help="Allow this job to manage offline Python sandbox pods"
    )
    parser.add_argument("--sweep", type=Path, help="Sweep YAML; omit --config from COMMAND")
    parser.add_argument("--dry-run", action="store_true", help="Print the submission plan without building or queuing")
    if "--" not in argv:
        parser.parse_args(argv)
        parser.error("separate the container command with --")
    separator = argv.index("--")
    args = parser.parse_args(argv[:separator])
    args.command = argv[separator + 1 :]
    if not args.command or not args.command[0]:
        parser.error("provide a command after --")
    if args.sweep and any(arg == "--config" or arg.startswith("--config=") for arg in args.command):
        parser.error("--sweep supplies --config; omit it from COMMAND")
    return args


def job_manifest(args: argparse.Namespace, name: str, image: str) -> dict[str, Any]:
    manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": "training",
            "labels": {
                "kueue.x-k8s.io/queue-name": "gpu",
                "kueue.x-k8s.io/priority-class": f"train-{args.prio}",
            },
        },
        "spec": {
            "suspend": True,
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 604800,
            "template": {
                "spec": {
                    "runtimeClassName": "nvidia",
                    "restartPolicy": "Never",
                    "automountServiceAccountToken": False,
                    "containers": [
                        {
                            "name": "train",
                            "image": image,
                            "imagePullPolicy": "Never",
                            "workingDir": "/app",
                            # Kubelet expands dollar references even without a shell.
                            "command": [arg.replace("$", "$$") for arg in args.command],
                            "ports": [
                                {"name": "ray-dashboard", "containerPort": 8265},
                                {"name": "vllm", "containerPort": 8000},
                            ],
                            "securityContext": {"capabilities": {"add": ["SYS_PTRACE"]}},
                            "env": [
                                {"name": "MLFLOW_TRACKING_URI", "value": "http://mlflow.mlflow.svc.cluster.local:5000"},
                                {
                                    "name": "OTEL_EXPORTER_OTLP_ENDPOINT",
                                    "value": "http://tempo.monitoring.svc.cluster.local:4317",
                                },
                                {"name": "OTEL_RESOURCE_ATTRIBUTES", "value": f"k8s.job.name={name}"},
                                {"name": "JAX_PLATFORMS", "value": "cuda,cpu"},
                                {"name": "RL2_RAY_DASHBOARD_HOST", "value": "0.0.0.0"},
                                {"name": "RL2_VLLM_HOST", "value": "0.0.0.0"},
                                {"name": "RL2_RAY_TIMELINE_PATH", "value": f"/runs/ray-timelines/{name}.json"},
                                {"name": "RL2_RAY_TIMELINE_URL", "value": f"http://localhost:8085/?job={name}"},
                                {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "compute,utility,graphics"},
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": str(args.cpus),
                                    "memory": f"{args.memory_gib}Gi",
                                    "nvidia.com/gpu": str(args.gpus),
                                },
                                "limits": {"memory": f"{args.memory_gib}Gi", "nvidia.com/gpu": str(args.gpus)},
                            },
                            "volumeMounts": [
                                {"name": "shm", "mountPath": "/dev/shm"},
                                {"name": "runs", "mountPath": "/runs"},
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}},
                        {"name": "runs", "persistentVolumeClaim": {"claimName": "training-runs"}},
                    ],
                },
            },
        },
    }
    if args.python_sandbox:
        pod = manifest["spec"]["template"]["spec"]
        pod["serviceAccountName"] = "python-sandbox"
        pod["automountServiceAccountToken"] = True
    return manifest


def submit(args: argparse.Namespace) -> list[str]:
    sweep = load_sweep(path=args.sweep, root=ROOT) if args.sweep else None
    image = f"rl2.local/train:train-{uuid4().hex[:16]}"
    configs = {trial.run_id: OmegaConf.to_container(trial, resolve=False) for trial in sweep.trials} if sweep else {}
    if args.dry_run:
        print(
            json.dumps(
                {
                    "command": args.command,
                    "extra": args.extra,
                    "gpus": args.gpus,
                    "python_sandbox": args.python_sandbox,
                    "sweep": sweep.name if sweep else None,
                    "trials": configs,
                },
                indent=2,
            )
        )
        return []
    # The image retains these files after the temporary host directory is removed.
    with TemporaryDirectory(prefix=".rl2-sweep-", dir=ROOT) as directory:
        manifests = []
        for run_id in configs if sweep else [None]:
            command = args.command.copy()
            if run_id is not None:
                config_path = Path(directory) / f"{run_id}.yaml"
                OmegaConf.save(config=configs[run_id], f=config_path)
                command.extend(["--config", config_path.relative_to(ROOT).as_posix()])
            job_args = argparse.Namespace(**{**vars(args), "command": command})
            manifest = job_manifest(args=job_args, name=f"train-{uuid4().hex[:16]}", image=image)
            if sweep:
                manifest["metadata"]["labels"]["rl2.local/sweep"] = sweep.name
                manifest["metadata"]["annotations"] = {"rl2.local/run-id": run_id}
            manifests.append(manifest)
        if sweep:
            record = ROOT / "runs" / "sweeps" / sweep.name / f"{image.rsplit(':', 1)[1]}.json"
            record.parent.mkdir(parents=True, exist_ok=True)
            record.write_text(json.dumps({"configs": configs, "jobs": manifests}, indent=2) + "\n")
            print(f"Sweep {sweep.name}: {len(manifests)} trials; submission record: {record}", flush=True)
        return submit_jobs(args=args, manifests=manifests, image=image)


def submit_jobs(args: argparse.Namespace, manifests: list[dict[str, Any]], image: str) -> list[str]:
    privilege = [] if os.geteuid() == 0 else ["sudo", "-n"]
    kubectl = [*privilege, "k3s", "kubectl"]
    payloads = [json.dumps(manifest) for manifest in manifests]
    # Fail before building if the queue, priority, or Job specification is invalid.
    subprocess.run([*kubectl, "-n", "training", "get", "localqueue", "gpu"], check=True)
    subprocess.run([*kubectl, "get", "workloadpriorityclass", f"train-{args.prio}"], check=True)
    subprocess.run([*kubectl, "-n", "training", "get", "pvc", "training-runs"], check=True)
    if args.python_sandbox:
        subprocess.run([*kubectl, "-n", "training", "get", "serviceaccount", "python-sandbox"], check=True)
        subprocess.run([*kubectl, "get", "runtimeclass", "runsc-no-network"], check=True)
    for payload in payloads:
        subprocess.run([*kubectl, "create", "--dry-run=server", "-f", "-"], input=payload, text=True, check=True)
    subprocess.run([*privilege, "systemctl", "is-active", "--quiet", "rl2-image-pins.timer"], check=True)
    print(f"Building {image} from {ROOT}", flush=True)
    subprocess.run(
        [
            *privilege,
            "buildctl",
            "--addr",
            BUILD_ADDRESS,
            "build",
            "--frontend",
            "dockerfile.v0",
            "--local",
            f"context={ROOT}",
            "--local",
            f"dockerfile={ROOT}",
            "--opt",
            "filename=infra/k3s/training/Dockerfile",
            *(["--opt", f"build-arg:UV_EXTRA={args.extra}"] if args.extra else []),
            "--output",
            f"type=image,name={image},unpack=true,compression=zstd,compression-level=1,oci-mediatypes=true",
            "--progress",
            "plain",
        ],
        check=True,
    )
    names = []
    for manifest, payload in zip(manifests, payloads, strict=True):
        subprocess.run([*privilege, "/usr/bin/python3", IMAGE_MANAGER, "submit"], input=payload, text=True, check=True)
        name = manifest["metadata"]["name"]
        names.append(name)
        run_id = manifest["metadata"].get("annotations", {}).get("rl2.local/run-id")
        print(
            f"Queued {name} (priority={args.prio}, GPU={args.gpus})" + (f" run_id={run_id}" if run_id else ""),
            flush=True,
        )
        print(f"Status: sudo k3s kubectl -n training get job {name}")
        print(f"Logs:   sudo k3s kubectl -n training logs -f job/{name}")
        print("Training jobs: http://localhost:8085/")
        print(f"Ray dashboard (when enabled): http://localhost:8085/dashboard/{name}/")
        print(f"Ray timeline (when recorded): http://localhost:8085/?job={name}")
    return names


def main() -> int:
    args = parse_args(sys.argv[1:])
    try:
        submit(args)
    except (subprocess.CalledProcessError, OSError, TypeError, ValueError, OmegaConfBaseException, YAMLError) as error:
        print(f"Submission failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
