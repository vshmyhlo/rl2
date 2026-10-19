"""Opt-in Headlamp RBAC regression check: RL2_K3S_TESTS=1 uv run pytest <this-file>."""

import os
import subprocess

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RL2_K3S_TESTS") != "1", reason="Requires local k3s RBAC")


@pytest.mark.parametrize(
    ("namespace", "verb", "resource", "subresource", "allowed"),
    [
        ("training", "delete", "jobs.batch", "", True),
        ("training", "create", "pods", "eviction", True),
        ("default", "create", "pods", "eviction", False),
        ("training", "get", "secrets", "", False),
    ],
)
def test_headlamp_permissions(namespace: str, verb: str, resource: str, subresource: str, allowed: bool) -> None:
    result = subprocess.run(
        [
            "sudo",
            "-n",
            "k3s",
            "kubectl",
            "auth",
            "can-i",
            verb,
            resource,
            "--subresource",
            subresource,
            "--namespace",
            namespace,
            "--as=system:serviceaccount:headlamp:headlamp-viewer",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.stdout.strip() == ("yes" if allowed else "no"), result.stdout + result.stderr
    assert result.returncode == (0 if allowed else 1), result.stderr
