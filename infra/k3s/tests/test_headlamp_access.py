"""Opt-in login/RBAC checks; requires k3s and the localhost:8080 port-forward."""

import base64
import json
import os
import subprocess

import httpx
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RL2_K3S_TESTS") != "1", reason="Requires local k3s RBAC")


def test_headlamp_login_lasts_thirty_days() -> None:
    result = subprocess.run(
        ["sudo", "-n", "k3s", "kubectl", "-n", "headlamp", "create", "token", "headlamp-viewer", "--duration=720h"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    token = result.stdout.strip()
    payload = token.split(".")[1]
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    assert claims["exp"] - claims["iat"] == 30 * 24 * 60 * 60

    with httpx.Client(base_url="http://127.0.0.1:8080", trust_env=False, timeout=30) as client:
        response = client.get("/clusters/main/api/v1/nodes")
        assert response.status_code in (401, 403)
        response = client.post("/clusters/main/set-token", json={"token": token})
        assert response.status_code == 200
        cookies = [cookie for cookie in client.cookies.jar if cookie.name.startswith("headlamp-auth-main.")]
        assert cookies
        assert all(cookie.has_nonstandard_attr("HttpOnly") for cookie in cookies)
        assert all("Max-Age=2592000" in header for header in response.headers.get_list("set-cookie"))
        response = client.get("/clusters/main/api/v1/nodes")
        assert response.status_code == 200
        assert response.json()["kind"] == "NodeList"


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
