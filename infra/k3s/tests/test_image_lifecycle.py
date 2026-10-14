"""Opt-in CRI image-pin check using a small CPU-only Job."""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RL2_K3S_TESTS") != "1", reason="Requires local k3s and image helper")
HELPER = ["/usr/bin/python3", "/usr/local/lib/rl2/image_pins.py"]


def run(*args: str, manifest: dict[str, Any] | None = None) -> str:
    return subprocess.run(
        ["sudo", "-n", *args],
        input=json.dumps(manifest) if manifest is not None else None,
        text=True,
        capture_output=True,
        check=True,
        timeout=150,
    ).stdout


def wait_for_pin(image: str, expected: bool) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if json.loads(run("k3s", "crictl", "inspecti", image))["status"].get("pinned", False) == expected:
            return
        time.sleep(0.2)
    pytest.fail(f"CRI image pin did not become {expected}")


def test_image_pinned_while_queued_and_released_after_completion(tmp_path: Path) -> None:
    name = f"train-pin-check-{uuid4().hex[:12]}"
    image = f"rl2.local/train:{name}"
    # Use the same exporter as submission, with a unique config digest and a tiny image.
    (tmp_path / "Dockerfile").write_text(f"FROM busybox:1.37.0\nLABEL rl2.pin-test={name}\n")
    run(
        "buildctl",
        "--addr",
        "unix:///run/rl2-buildkit/buildkitd.sock",
        "build",
        "--frontend",
        "dockerfile.v0",
        "--local",
        f"context={tmp_path}",
        "--local",
        f"dockerfile={tmp_path}",
        "--output",
        f"type=image,name={image},unpack=true",
    )
    manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "namespace": "training"},
        "spec": {
            "suspend": True,
            "backoffLimit": 0,
            "activeDeadlineSeconds": 60,
            "ttlSecondsAfterFinished": 300,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "automountServiceAccountToken": False,
                    "containers": [{"name": "check", "image": image, "imagePullPolicy": "Never", "command": ["true"]}],
                }
            },
        },
    }
    try:
        run(*HELPER, "submit", manifest=manifest)
        run(*HELPER, "reconcile")
        wait_for_pin(image, expected=True)
        run(
            "k3s", "kubectl", "-n", "training", "patch", "job", name, "--type=merge", "-p", '{"spec":{"suspend":false}}'
        )
        run("k3s", "kubectl", "-n", "training", "wait", "--for=condition=Complete", f"job/{name}", "--timeout=90s")
        run(*HELPER, "reconcile")
        wait_for_pin(image, expected=False)
    finally:
        run("k3s", "kubectl", "-n", "training", "delete", "job", name, "--ignore-not-found", "--wait=true")
        run("k3s", "ctr", "-n", "k8s.io", "images", "rm", image)
