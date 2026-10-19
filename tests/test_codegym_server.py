import io
import os
import runpy
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock

import httpx
import pytest

from rl2 import codegym_server as server
from rl2 import codegym_setup as setup


@pytest.mark.parametrize("failure", [None, "startup", "rollout"])
def test_server_lifecycle(failure: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    (tmp_path / "env_instance_manager.py").touch()
    process = MagicMock(pid=123)
    process.__enter__.return_value = process
    popen = Mock(return_value=process)
    kill = Mock()
    wait = Mock(side_effect=TimeoutError("startup") if failure == "startup" else None)
    monkeypatch.setattr(server.subprocess, "Popen", popen)
    monkeypatch.setattr(server.os, "killpg", kill)
    monkeypatch.setattr(server, "wait_for_codegym", wait)
    config = server.CodeGymServerConfig(root=str(tmp_path), workers=2)

    def run() -> None:
        with server.serve_codegym(config, "http://127.0.0.1:8001"):
            kill.assert_not_called()
            if failure == "rollout":
                raise RuntimeError("rollout")

    if failure is None:
        run()
    else:
        with pytest.raises(TimeoutError if failure == "startup" else RuntimeError, match=failure):
            run()
    assert kill.call_args_list == [((123, signal.SIGTERM),), ((123, signal.SIGKILL),)]
    args, kwargs = popen.call_args
    assert args[0][0] == str(python)
    assert args[0][2:] == ["--port", "8001", "--workers", "2"]
    assert kwargs["cwd"] == tmp_path
    assert kwargs["env"]["PYTHONPATH"] == str(tmp_path)
    assert kwargs["env"]["PATH"].split(os.pathsep)[0] == str(python.parent)
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert kwargs["start_new_session"]
    wait.assert_called_once_with(process, manager_url="http://127.0.0.1:8001", workers=2, timeout=120)


def test_server_requires_setup(tmp_path: Path) -> None:
    with (
        pytest.raises(FileNotFoundError, match="rl2.codegym_setup"),
        server.serve_codegym(server.CodeGymServerConfig(root=str(tmp_path)), "http://127.0.0.1:8001"),
    ):
        pytest.fail("Unprepared server started")


@pytest.mark.parametrize("state", ["hung", "exited"])
def test_cleanup_escalates_and_tolerates_exited_process(state: str, monkeypatch: pytest.MonkeyPatch) -> None:
    process = Mock(pid=123)
    if state == "hung":
        process.wait.side_effect = [subprocess.TimeoutExpired(cmd="codegym", timeout=5), 0]
    kill = Mock(side_effect=ProcessLookupError if state == "exited" else None)
    monkeypatch.setattr(server.os, "killpg", kill)
    server.stop_server(process)
    assert kill.call_count == 2
    assert process.wait.call_count == 2


def test_readiness_waits_for_every_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    process = Mock(pid=123, poll=Mock(return_value=None))
    requests: list[int] = []

    def respond(request: httpx.Request) -> httpx.Response:
        port = request.url.port
        requests.append(port)
        if port == 8001:
            assert request.url.path == "/health"
            return httpx.Response(200, json={"pid": 123, "ports": [10000, 10001]})
        assert request.url.path == "/openapi.json"
        return httpx.Response(503 if port == 10001 and requests.count(port) == 1 else 200)

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(server.httpx, "Client", Mock(return_value=client))
    monkeypatch.setattr(server.time, "sleep", Mock())
    server.wait_for_codegym(process, "http://127.0.0.1:8001", workers=2, timeout=1)
    assert requests == [8001, 10000, 10001, 8001, 10001]
    assert client.is_closed


@pytest.mark.parametrize("failure", ["exit", "timeout", "occupied"])
def test_readiness_failures(failure: str, monkeypatch: pytest.MonkeyPatch) -> None:
    process = Mock(pid=123, poll=Mock(return_value=1 if failure == "exit" else None))
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"pid": 456})))
    monkeypatch.setattr(server.httpx, "Client", Mock(return_value=client))
    if failure == "timeout":
        monkeypatch.setattr(server.time, "monotonic", Mock(side_effect=[0, 2]))
    with pytest.raises(TimeoutError if failure == "timeout" else RuntimeError):
        server.wait_for_codegym(process, "http://127.0.0.1:8001", workers=2, timeout=1)
    assert client.is_closed


def test_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    app = MagicMock()
    app.get.return_value = lambda function: function
    upstream = SimpleNamespace(
        app=app, EnvInstanceManager=SimpleNamespace(_all_instances={"worker": SimpleNamespace(port=10000)})
    )
    uvicorn = SimpleNamespace(run=Mock())
    monkeypatch.setitem(sys.modules, "env_instance_manager", upstream)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.setattr(sys, "argv", ["codegym_launcher", "--port", "8001", "--workers", "2"])
    namespace = runpy.run_path(str(Path(server.__file__).with_name("codegym_launcher.py")), run_name="__main__")
    assert app.state.workers == 2
    uvicorn.run.assert_called_once_with(app, host="127.0.0.1", port=8001, access_log=False)
    assert namespace["health"]() == {"pid": os.getpid(), "ports": [10000]}


@pytest.mark.parametrize("failure", [None, "filename", "empty", "install"])
def test_setup(failure: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    downloads: list[str] = []

    def download(url: str, **kwargs: Any) -> io.BytesIO:
        downloads.append(url)
        return io.BytesIO(b'ARGS = ["--host", "::"]' if url.endswith("env_instance_manager.py") else b"upstream")

    row = {
        "source": "../outside" if failure == "filename" else "Example",
        "env_name": "SumEnv",
        "env_code": "class SumEnv: pass\n",
    }
    load = Mock(return_value=[] if failure == "empty" else [row])
    run = Mock(side_effect=subprocess.CalledProcessError(1, "uv") if failure == "install" else None)
    monkeypatch.setattr(setup, "urlopen", download)
    monkeypatch.setattr(setup, "load_dataset", load)
    monkeypatch.setattr(setup.subprocess, "run", run)
    if failure is not None:
        with pytest.raises(subprocess.CalledProcessError if failure == "install" else ValueError):
            setup.prepare_codegym(tmp_path)
        assert not (tmp_path / ".ready").exists()
        return
    setup.prepare_codegym(tmp_path)
    assert (tmp_path / "envs/codegym_v1/Example__SumEnv.py").read_text() == row["env_code"]
    assert (tmp_path / "LICENSE").read_text() == "upstream"
    assert (tmp_path / "env_instance_manager.py").read_text() == 'ARGS = ["--host", "127.0.0.1"]'
    assert all(setup.SERVER_REVISION in url for url in downloads)
    load.assert_called_once_with("VanishD/CodeGym", "envs_en", revision=setup.DATASET_REVISION, split="train")
    assert run.call_count == 2
    assert run.call_args.args[0][-2:] == setup.ENV_REQUIREMENTS
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    setup.prepare_codegym(tmp_path)
    assert len(downloads) == 5
    assert run.call_count == 2
