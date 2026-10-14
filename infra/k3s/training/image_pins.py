"""Protect local training images until their Kubernetes Jobs finish."""

import argparse
import fcntl
import json
import subprocess
import sys
from typing import Any

IMAGE_PREFIX = "rl2.local/train:train-"
PIN_LABEL = "io.cri-containerd.pinned"


def run(args: list[str], manifest: dict[str, Any] | None = None) -> str:
    return subprocess.run(
        args,
        input=json.dumps(manifest) if manifest is not None else None,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    ).stdout


def set_pin(image: str, pinned: bool) -> None:
    value = "pinned" if pinned else ""
    run(["k3s", "ctr", "-n", "k8s.io", "images", "label", image, f"{PIN_LABEL}={value}"])


def submit_job(manifest: dict[str, Any]) -> None:
    image = manifest["spec"]["template"]["spec"]["containers"][0]["image"]
    if manifest["metadata"]["namespace"] != "training" or not image.startswith(IMAGE_PREFIX):
        raise ValueError("Expected a training Job with a local training image")
    set_pin(image, pinned=True)
    # Keep the pin if the response is lost: reconciliation checks whether creation succeeded.
    print(run(["k3s", "kubectl", "create", "--request-timeout=30s", "-f", "-"], manifest=manifest), end="")


def reconcile() -> None:
    # Read successfully before changing pins; an unavailable API must preserve them.
    jobs = json.loads(run(["k3s", "kubectl", "-n", "training", "get", "jobs", "-o", "json"]))["items"]
    active_images = {
        container["image"]
        for job in jobs
        if not any(
            condition["type"] in ("Complete", "Failed") and condition["status"] == "True"
            for condition in job.get("status", {}).get("conditions", [])
        )
        for container in job["spec"]["template"]["spec"]["containers"]
    }
    images = run(["k3s", "ctr", "-n", "k8s.io", "images", "list", "-q", f"name~={IMAGE_PREFIX}"])
    for image in images.splitlines():
        if image.startswith(IMAGE_PREFIX):
            set_pin(image, pinned=image in active_images)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "reconcile"))
    args = parser.parse_args()
    with open("/run/lock/rl2-image-pins.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.action == "submit":
            submit_job(json.load(sys.stdin))
        else:
            reconcile()


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.stderr or str(error))
