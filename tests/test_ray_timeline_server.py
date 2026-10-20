import json
import runpy
from collections.abc import Iterator
from contextlib import ExitStack
from functools import partial
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
viewer = SimpleNamespace(**runpy.run_path(str(ROOT / "infra/k3s/ray-timelines/server.py")))


@pytest.fixture
def client(tmp_path: Path) -> Iterator[httpx.Client]:
    with (
        ThreadingHTTPServer(
            ("127.0.0.1", 0), partial(viewer.TimelineHandler, root=tmp_path, cluster=viewer.Cluster())
        ) as server,
        ExitStack() as resources,
        httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}", trust_env=False) as client,
    ):
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        resources.callback(thread.join)
        resources.callback(server.shutdown)
        yield client


def test_viewer_and_latest_snapshot(client: httpx.Client, tmp_path: Path) -> None:
    assert client.get("/health").text == "ok\n"
    page = client.get("/?job=train-test")
    assert page.status_code == 200
    assert "https://ui.perfetto.dev" in page.text
    assert client.get("/traces/train-test.json").status_code == 404
    path = tmp_path / "train-test.json"
    path.write_text('[{"name":"first"}]')
    response = client.get("/traces/train-test.json")
    assert response.json() == [{"name": "first"}]
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-type"] == "application/json"
    assert response.headers["last-modified"]
    replacement = tmp_path / "next.json"
    replacement.write_text('[{"name":"latest"}]')
    replacement.replace(path)
    assert client.get("/traces/train-test.json").json() == [{"name": "latest"}]


@pytest.mark.parametrize("path", ["/traces/", "/traces/%2e%2e/secret.json", "/traces/train-test.txt"])
def test_only_trace_files_are_served(path: str, client: httpx.Client) -> None:
    assert client.get(path).status_code == 404


def test_symlinks_and_directories_are_not_served(client: httpx.Client, tmp_path: Path) -> None:
    (tmp_path / "private.txt").write_text("private")
    (tmp_path / "link.json").symlink_to(tmp_path / "private.txt")
    (tmp_path / "directory.json").mkdir()
    assert client.get("/traces/link.json").status_code == 403
    assert client.get("/traces/directory.json").status_code == 403


def test_viewer_mounts_submission_storage_read_only() -> None:
    submit = SimpleNamespace(**runpy.run_path(str(ROOT / "submit.py")))
    args = submit.parse_args(["--gpus", "2", "--", "python", "-m", "rl2.train_grpo_v2"])
    job = submit.job_manifest(args=args, name="train-test", image="test")
    container = job["spec"]["template"]["spec"]["containers"][0]
    env = {entry["name"]: entry["value"] for entry in container["env"]}
    deployment, service, account, role, binding = yaml.safe_load_all(
        (ROOT / "infra/k3s/ray-timelines/service.yaml").read_text()
    )
    pod = deployment["spec"]["template"]["spec"]
    mount = next(mount for mount in pod["containers"][0]["volumeMounts"] if mount["name"] == "runs")
    assert mount == {"name": "runs", "mountPath": "/runs", "readOnly": True}
    volume = next(volume for volume in pod["volumes"] if volume["name"] == "runs")
    assert volume["persistentVolumeClaim"]["claimName"] == "training-runs"
    assert env["RL2_RAY_TIMELINE_PATH"] == "/runs/ray-timelines/train-test.json"
    assert env["RL2_RAY_TIMELINE_URL"] == "http://localhost:8085/?job=train-test"
    assert service["spec"]["type"] == "ClusterIP"
    assert pod["serviceAccountName"] == account["metadata"]["name"] == binding["subjects"][0]["name"]
    assert role["rules"] == [
        {"apiGroups": ["batch"], "resources": ["jobs"], "verbs": ["list"]},
        {"apiGroups": [""], "resources": ["pods"], "verbs": ["list"]},
    ]
    assert binding["roleRef"]["name"] == role["metadata"]["name"]


def test_job_discovery_tracks_current_pods_and_omits_finished_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = [
        {
            "metadata": {"name": name, "uid": name, "creationTimestamp": "2026-10-20T12:00:00Z"},
            "spec": {"suspend": name == "queued"},
            "status": {"conditions": [{"type": name, "status": "True"}]} if name in {"Complete", "Failed"} else {},
        }
        for name in ["running", "queued", "starting", "Complete", "Failed"]
    ]
    pods = [
        {
            "metadata": {
                "ownerReferences": [{"kind": "Job", "uid": owner}],
                "creationTimestamp": created,
                **({"deletionTimestamp": created} if deleting else {}),
            },
            "status": {"phase": phase, "podIP": address},
        }
        for owner, created, phase, address, deleting in [
            ("running", "1", "Failed", "10.0.0.1", False),
            ("running", "2", "Running", "10.0.0.2", False),
            ("running", "3", "Running", "10.0.0.3", True),
            ("deleted-job-uid", "4", "Running", "10.0.0.4", False),
            ("starting", "1", "Pending", None, False),
        ]
    ]
    cluster = viewer.Cluster()

    def list_resources(resource: str) -> list[dict[str, Any]]:
        return jobs if resource == "jobs" else pods

    monkeypatch.setattr(cluster, "list_resources", list_resources)
    result = {job["name"]: job for job in cluster.jobs()}
    assert set(result) == {"running", "queued", "starting"}
    assert result["running"]["pod_ip"] == "10.0.0.2"
    assert result["running"]["status"] == "Running"
    assert result["queued"]["status"] == "Queued"
    assert result["starting"]["status"] == "Starting"
    assert result["queued"]["pod_ip"] is None


def test_job_listing_and_unavailable_dashboards(client: httpx.Client, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = Mock(return_value=[{"name": "queued", "status": "Queued", "pod_ip": None}])
    monkeypatch.setattr(viewer.Cluster, "jobs", jobs)
    assert client.get("/api/jobs").json() == [{"name": "queued", "status": "Queued"}]
    assert client.get("/dashboard/queued/").status_code == 503
    assert client.get("/dashboard/finished/").status_code == 404
    assert client.get("/dashboard/bad_name/").status_code == 404
    assert client.post("/dashboard/queued/api/jobs/").status_code == 405
    jobs.side_effect = OSError("unavailable")
    assert client.get("/api/jobs").status_code == 503
    assert client.get("/dashboard/queued/").status_code == 503
    assert client.get("/health").status_code == 200


def test_dashboard_proxy_login_assets_and_cookie_isolation(
    client: httpx.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[dict[str, Any]] = []

    class Dashboard(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append({"path": self.path, "cookie": self.headers.get("Cookie")})
            self.send_response(302 if self.path == "/redirect" else 200)
            if self.path == "/redirect":
                self.send_header("Location", "/logs?node=1")
            if self.path == "/stream":
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.write(b"5\r\nfirst\r\n6\r\nsecond\r\n0\r\n\r\n")
                return
            self.end_headers()
            self.wfile.write(b"dashboard asset")

        def do_HEAD(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", "15")
            self.end_headers()

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append({"path": self.path, "auth": self.headers["Authorization"], "body": json.loads(body)})
            self.send_response(200)
            self.send_header("Set-Cookie", "ray-authentication-token=test-token; Path=/; HttpOnly; SameSite=Strict")
            self.end_headers()

    with ThreadingHTTPServer(("127.0.0.1", 0), Dashboard) as upstream, ExitStack() as resources:
        thread = Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        resources.callback(thread.join)
        resources.callback(upstream.shutdown)

        def connect(host: str, port: int, timeout: float) -> HTTPConnection:
            assert host == "127.0.0.1" and port == 8265
            return HTTPConnection(host, upstream.server_port, timeout=timeout)

        monkeypatch.setitem(viewer.TimelineHandler.proxy_dashboard.__globals__, "HTTPConnection", connect)
        monkeypatch.setattr(
            viewer.Cluster,
            "jobs",
            Mock(
                return_value=[
                    {"name": "train-one", "pod_ip": "127.0.0.1"},
                    {"name": "train-two", "pod_ip": "127.0.0.1"},
                ]
            ),
        )
        redirect = client.get("/dashboard/train-one?view=1")
        assert redirect.status_code == 307
        assert redirect.headers["location"] == "/dashboard/train-one/?view=1"
        response = client.get("/dashboard/train-one/static/app.js?version=1")
        assert response.content == b"dashboard asset"
        assert requests[-1]["path"] == "/static/app.js?version=1"
        assert client.get("/dashboard/train-one/stream").content == b"firstsecond"
        response = client.head("/dashboard/train-one/api/v0/platform_events")
        assert response.status_code == 200 and response.content == b""
        assert response.headers["content-length"] == "15"
        response = client.post(
            "/dashboard/train-one/api/authenticate", headers={"Authorization": "Bearer test-token"}, json={}
        )
        assert response.status_code == 200
        assert requests[-1] == {"path": "/api/authenticate", "auth": "Bearer test-token", "body": {}}
        assert "Path=/dashboard/train-one/" in response.headers["set-cookie"]
        assert "HttpOnly" in response.headers["set-cookie"]
        client.get("/dashboard/train-one/api/version")
        assert requests[-1]["cookie"] == "ray-authentication-token=test-token"
        client.get("/dashboard/train-two/api/version")
        assert requests[-1]["cookie"] is None
        response = client.get("/dashboard/train-one/redirect")
        assert response.headers["location"] == "/dashboard/train-one/logs?node=1"
        monkeypatch.setitem(
            viewer.TimelineHandler.proxy_dashboard.__globals__,
            "HTTPConnection",
            Mock(side_effect=ConnectionRefusedError),
        )
        assert client.get("/dashboard/train-one/").status_code == 502
