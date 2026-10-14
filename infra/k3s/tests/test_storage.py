"""Opt-in persistence check: RL2_K3S_TESTS=1 uv run pytest <this-file>."""

import json
import os
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RL2_K3S_TESTS") != "1", reason="Requires local k3s storage")
submit = SimpleNamespace(**runpy.run_path(str(Path(__file__).resolve().parents[3] / "submit.py")))


def kubectl(*args: str, manifest: dict[str, Any] | None = None) -> str:
    return subprocess.run(
        ["sudo", "-n", "k3s", "kubectl", "-n", "training", *args],
        input=json.dumps(manifest) if manifest is not None else None,
        capture_output=True,
        text=True,
        check=True,
        timeout=150,
    ).stdout


def test_shared_files_survive_pod_replacement() -> None:
    token = uuid4().hex[:12]
    filename = f"/runs/.storage-test-{token}"
    commands = [
        f"printf persisted > {filename}; sync",
        f'test "$(cat {filename})" = persisted; rm {filename}; echo persistence-ok',
    ]
    created: list[str] = []
    try:
        for index, command in enumerate(commands):
            name = f"storage-check-{token}-{index}"
            args = submit.parse_args(["--", "sh", "-ec", command])
            job = submit.job_manifest(args=args, name=name, image="busybox:1.37.0")
            spec = job["spec"]["template"]["spec"]
            spec.pop("runtimeClassName")
            spec["activeDeadlineSeconds"] = 120
            container = spec["containers"][0]
            container["imagePullPolicy"] = "IfNotPresent"
            container["workingDir"] = "/tmp"
            container["resources"] = {"requests": {"cpu": "10m", "memory": "16Mi"}, "limits": {"memory": "64Mi"}}
            pod = {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": name, "namespace": "training"}, "spec": spec}
            kubectl("create", "-f", "-", manifest=pod)
            created.append(name)
            kubectl("wait", "--for=jsonpath={.status.phase}=Succeeded", f"pod/{name}", "--timeout=120s")
            if index == 1:
                assert kubectl("logs", name).strip() == "persistence-ok"
            # Fully remove the writer before the reader mounts the same volume.
            kubectl("delete", "pod", name, "--wait=true")
    finally:
        for name in created:
            kubectl("delete", "pod", name, "--ignore-not-found", "--wait=true")
