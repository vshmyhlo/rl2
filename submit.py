#!/usr/bin/env python3
"""Build this working tree and submit a GPU Job to the local k3s queue."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
BUILD_ADDRESS = "unix:///run/rl2-buildkit/buildkitd.sock"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, usage="%(prog)s [options] -- COMMAND [ARG ...]")
    parser.add_argument("--prio", choices=("low", "high"), default="low")
    parser.add_argument("--cpus", type=int, choices=range(1, 11), default=4, metavar="1..10")
    parser.add_argument("--memory-gib", type=int, choices=range(1, 65), default=16, metavar="1..64")
    if "--" not in argv:
        parser.parse_args(argv)
        parser.error("separate the container command with --")
    separator = argv.index("--")
    args = parser.parse_args(argv[:separator])
    args.command = argv[separator + 1 :]
    if not args.command or not args.command[0]:
        parser.error("provide a command after --")
    return args


def job_manifest(args: argparse.Namespace, name: str, image: str) -> dict[str, Any]:
    return {
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
                            "command": args.command,
                            "env": [
                                {"name": "MLFLOW_TRACKING_URI", "value": "http://mlflow.mlflow.svc.cluster.local:5000"},
                                {"name": "JAX_PLATFORMS", "value": "cuda"},
                                {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "compute,utility"},
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": str(args.cpus),
                                    "memory": f"{args.memory_gib}Gi",
                                    "nvidia.com/gpu": "1",
                                },
                                "limits": {"memory": f"{args.memory_gib}Gi", "nvidia.com/gpu": "1"},
                            },
                            "volumeMounts": [{"name": "shm", "mountPath": "/dev/shm"}],
                        }
                    ],
                    "volumes": [{"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}}],
                },
            },
        },
    }


def submit(args: argparse.Namespace) -> str:
    privilege = [] if os.geteuid() == 0 else ["sudo", "-n"]
    kubectl = [*privilege, "k3s", "kubectl"]
    name = f"train-{uuid4().hex[:16]}"
    image = f"rl2.local/train:{name}"
    manifest = json.dumps(job_manifest(args=args, name=name, image=image))
    # Fail before building if the queue, priority, or Job specification is invalid.
    subprocess.run([*kubectl, "-n", "training", "get", "localqueue", "gpu"], check=True)
    subprocess.run([*kubectl, "get", "workloadpriorityclass", f"train-{args.prio}"], check=True)
    subprocess.run([*kubectl, "create", "--dry-run=server", "-f", "-"], input=manifest, text=True, check=True)
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
            "--output",
            f"type=image,name={image},unpack=true",
            "--progress",
            "plain",
        ],
        check=True,
    )
    subprocess.run([*kubectl, "create", "-f", "-"], input=manifest, text=True, check=True)
    print(f"Queued {name} (priority={args.prio}, GPU=1)")
    print(f"Status: sudo k3s kubectl -n training get job {name}")
    print(f"Logs:   sudo k3s kubectl -n training logs -f job/{name}")
    return name


def main() -> int:
    args = parse_args(sys.argv[1:])
    try:
        submit(args)
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        print(f"Submission failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
