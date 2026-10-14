"""Opt-in live GPU queue check: RL2_K3S_TESTS=1 uv run pytest <this-file>."""

import json
import os
import runpy
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RL2_K3S_TESTS") != "1", reason="Requires an idle local GPU queue")
submit = SimpleNamespace(**runpy.run_path(str(Path(__file__).resolve().parents[3] / "submit.py")))


def kubectl(*args: str, manifest: dict[str, Any] | None = None) -> str:
    return subprocess.run(
        ["sudo", "-n", "k3s", "kubectl", "-n", "training", *args],
        input=json.dumps(manifest) if manifest is not None else None,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout


def get(kind: str, name: str) -> dict[str, Any]:
    return json.loads(kubectl("get", kind, name, "-o", "json"))


def wait_for(check: Callable[[], bool], description: str) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(1)
    pytest.fail(f"Timed out waiting for {description}")


def running_pods(job: str) -> list[dict[str, Any]]:
    pods = json.loads(kubectl("get", "pods", "-l", f"batch.kubernetes.io/job-name={job}", "-o", "json"))
    return [
        p for p in pods["items"] if p["status"]["phase"] == "Running" and not p["metadata"].get("deletionTimestamp")
    ]


def complete(job: str) -> bool:
    return any(
        c["type"] == "Complete" and c["status"] == "True" for c in get("job", job)["status"].get("conditions", [])
    )


def test_high_preempts_low_and_low_returns() -> None:
    queue = get("clusterqueue", "gpu")
    if queue["status"].get("reservingWorkloads", 0) or queue["status"].get("pendingWorkloads", 0):
        pytest.skip("GPU queue is busy")
    assert queue["spec"]["preemption"]["withinClusterQueue"] == "LowerPriority"
    names = {priority: f"preemption-check-{priority}-{uuid4().hex[:8]}" for priority in ("low", "high")}
    created: list[str] = []
    try:
        for priority in ("low", "high"):
            args = submit.parse_args(
                [
                    "--prio",
                    priority,
                    "--",
                    "sh",
                    "-ec",
                    "nvidia-smi -L; trap 'exit 143' TERM; while [ ! -f /tmp/release ]; do sleep 1; done; echo finished",
                ]
            )
            job = submit.job_manifest(args=args, name=names[priority], image="nvidia/cuda:12.6.3-base-ubuntu22.04")
            # Keep the real submitter's suspension/backoff settings; only shrink test resources.
            pod_spec = job["spec"]["template"]["spec"]
            pod_spec["terminationGracePeriodSeconds"] = 5
            container = pod_spec["containers"][0]
            container["workingDir"] = "/tmp"
            container["resources"]["requests"].update(cpu="100m", memory="128Mi")
            container["resources"]["limits"]["memory"] = "256Mi"
            job["spec"]["activeDeadlineSeconds"] = 180
            job["spec"]["ttlSecondsAfterFinished"] = 600
            kubectl("create", "-f", "-", manifest=job)
            created.append(names[priority])
            wait_for(lambda name=names[priority]: bool(running_pods(name)), f"{priority} job running")
            if priority == "low":
                original_pod_uid = running_pods(names["low"])[0]["metadata"]["uid"]

        low, high = names["low"], names["high"]
        wait_for(lambda: get("job", low)["spec"]["suspend"] and not running_pods(low), "low job requeued")
        assert get("job", low)["status"].get("failed", 0) == 0
        workloads = json.loads(kubectl("get", "workloads", "-o", "json"))["items"]
        low_workload = next(w for w in workloads if any(r["name"] == low for r in w["metadata"]["ownerReferences"]))
        assert any(c["type"] == "Preempted" and c["status"] == "True" for c in low_workload["status"]["conditions"])
        assert "A100" in kubectl("logs", f"job/{high}")
        print("High job running; low job suspended and preempted", flush=True)

        kubectl("exec", f"job/{high}", "--", "touch", "/tmp/release")
        wait_for(lambda: complete(high), "high job complete")
        wait_for(lambda: bool(running_pods(low)), "low job readmitted")
        assert running_pods(low)[0]["metadata"]["uid"] != original_pod_uid
        assert not get("job", low)["spec"]["suspend"]
        kubectl("exec", f"job/{low}", "--", "touch", "/tmp/release")
        wait_for(lambda: complete(low), "low job complete")
        assert "A100" in kubectl("logs", f"job/{low}")
        print("High completed; same low Job restarted in a new Pod and completed", flush=True)
    finally:
        for name in reversed(created):
            kubectl("delete", "job", name, "--ignore-not-found", "--wait=true")
