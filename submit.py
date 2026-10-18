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
    parser.add_argument("--cpus", type=int, choices=range(1, 11), default=4, metavar="1..10")
    parser.add_argument("--memory-gib", type=int, choices=range(1, 65), default=16, metavar="1..64")
    parser.add_argument("--extra", choices=("bfcl",), help="Include an optional dependency group in the image")
    parser.add_argument(
        "--vllm-sidecar",
        action="store_true",
        help="Serve the pinned BFCL model in a native sidecar; includes --extra bfcl",
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
    if args.vllm_sidecar:
        args.extra = "bfcl"
        if args.cpus < 2 or args.memory_gib < 8:
            parser.error("--vllm-sidecar requires at least 2 CPUs and 8 GiB total memory")
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
                            "env": [
                                {"name": "MLFLOW_TRACKING_URI", "value": "http://mlflow.mlflow.svc.cluster.local:5000"},
                                {"name": "JAX_PLATFORMS", "value": "cuda,cpu"},
                                {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "compute,utility,graphics"},
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": str(args.cpus),
                                    "memory": f"{args.memory_gib}Gi",
                                    "nvidia.com/gpu": "1",
                                },
                                "limits": {"memory": f"{args.memory_gib}Gi", "nvidia.com/gpu": "1"},
                            },
                            "volumeMounts": [
                                {"name": "shm", "mountPath": "/dev/shm"},
                                {"name": "runs", "mountPath": "/runs"},
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}},
                        {"name": "runs", "persistentVolumeClaim": {"claimName": "training-runs"}},
                    ],
                },
            },
        },
    }
    if args.vllm_sidecar:
        add_vllm_sidecar(manifest["spec"]["template"]["spec"], args)
    return manifest


def add_vllm_sidecar(pod: dict[str, Any], args: argparse.Namespace) -> None:
    main = pod["containers"][0]
    main["name"] = "bfcl"
    main["env"].extend(
        [
            {"name": "HF_HOME", "value": "/runs/bfcl/.huggingface"},
            {"name": "NVIDIA_VISIBLE_DEVICES", "value": "void"},
        ]
    )
    main["resources"] = {
        "requests": {"cpu": "1", "memory": "4Gi"},
        "limits": {"memory": "4Gi"},
    }
    # An exec probe reaches the loopback-only server; HTTP probes target the Pod IP.
    probe = {
        "exec": {
            "command": [
                "python",
                "-c",
                "from urllib.request import urlopen; urlopen('http://127.0.0.1:1053/health', timeout=2).close()",
            ]
        },
        "periodSeconds": 5,
        "timeoutSeconds": 5,
    }
    pod["initContainers"] = [
        {
            "name": "vllm",
            "image": main["image"],
            "imagePullPolicy": main["imagePullPolicy"],
            "workingDir": "/app",
            "restartPolicy": "Always",
            "command": ["sh", "-ec", "python -m rl2.bfcl download\nexec python -m rl2.bfcl serve"],
            "env": [{"name": "HF_HOME", "value": "/runs/bfcl/.huggingface"}],
            "startupProbe": {**probe, "failureThreshold": 180},
            "readinessProbe": {**probe, "failureThreshold": 3},
            "resources": {
                "requests": {"cpu": str(args.cpus - 1), "memory": f"{args.memory_gib - 4}Gi", "nvidia.com/gpu": "1"},
                "limits": {"memory": f"{args.memory_gib - 4}Gi", "nvidia.com/gpu": "1"},
            },
            "volumeMounts": main["volumeMounts"],
        }
    ]


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
                    "sweep": sweep.name if sweep else None,
                    "trials": configs,
                    "job": job_manifest(args, "preview", image) if args.vllm_sidecar else None,
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
            f"type=image,name={image},unpack=true",
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
        print(f"Queued {name} (priority={args.prio}, GPU=1)" + (f" run_id={run_id}" if run_id else ""), flush=True)
        print(f"Status: sudo k3s kubectl -n training get job {name}")
        print(f"Logs:   sudo k3s kubectl -n training logs -f job/{name}")
        if args.vllm_sidecar:
            print(f"vLLM:   sudo k3s kubectl -n training logs -f job/{name} -c vllm")
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
