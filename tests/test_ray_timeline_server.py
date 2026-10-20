import runpy
from collections.abc import Iterator
from contextlib import ExitStack
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import httpx
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
viewer = SimpleNamespace(**runpy.run_path(str(ROOT / "infra/k3s/ray-timelines/server.py")))


@pytest.fixture
def client(tmp_path: Path) -> Iterator[httpx.Client]:
    with (
        ThreadingHTTPServer(("127.0.0.1", 0), partial(viewer.TimelineHandler, root=tmp_path)) as server,
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
    deployment, service = yaml.safe_load_all((ROOT / "infra/k3s/ray-timelines/service.yaml").read_text())
    pod = deployment["spec"]["template"]["spec"]
    mount = next(mount for mount in pod["containers"][0]["volumeMounts"] if mount["name"] == "runs")
    assert mount == {"name": "runs", "mountPath": "/runs", "readOnly": True}
    volume = next(volume for volume in pod["volumes"] if volume["name"] == "runs")
    assert volume["persistentVolumeClaim"]["claimName"] == "training-runs"
    assert env["RL2_RAY_TIMELINE_PATH"] == "/runs/ray-timelines/train-test.json"
    assert env["RL2_RAY_TIMELINE_URL"] == "http://localhost:8085/?job=train-test"
    assert service["spec"]["type"] == "ClusterIP"
