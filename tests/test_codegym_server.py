import io
import json
import os
import runpy
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
from fastapi import HTTPException

from rl2.lm_envs.codegym import setup


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
def test_launcher(host: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = MagicMock()
    app.get.return_value = lambda function: function
    worker = SimpleNamespace(port=10000, health_check=Mock(return_value=True))
    upstream = SimpleNamespace(app=app, EnvInstanceManager=SimpleNamespace(_all_instances={"worker": worker}))
    uvicorn = SimpleNamespace(run=Mock())
    monkeypatch.setitem(sys.modules, "env_instance_manager", upstream)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.setattr(
        sys, "argv", ["rl2.lm_envs.codegym.launcher", "--port", "8001", "--workers", "1", "--host", host]
    )
    namespace = runpy.run_module("rl2.lm_envs.codegym.launcher", run_name="__main__")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tools").mkdir()
    tools_path = tmp_path / "tools/Example__SumEnv.json"
    tools_path.write_text('{"tools": []}')
    assert namespace["tools"]("Example__SumEnv").path == Path("tools/Example__SumEnv.json")
    for name in ["missing", "../outside", "..\\outside"]:
        with pytest.raises(HTTPException) as raised:
            namespace["tools"](name)
        assert raised.value.status_code == 404
    assert app.state.workers == 1
    uvicorn.run.assert_called_once_with(app, host=host, port=8001, access_log=False)
    assert namespace["health"]() == {"pid": os.getpid(), "ports": [10000]}
    connect = Mock(return_value=MagicMock())
    monkeypatch.setattr(socket, "create_connection", connect)
    assert namespace["ready"]() == {"workers": 1}
    connect.assert_called_once_with(("127.0.0.1", 10000), timeout=1)
    worker.health_check.return_value = False
    connect.reset_mock()
    with pytest.raises(HTTPException) as raised:
        namespace["ready"]()
    assert raised.value.status_code == 503
    connect.assert_not_called()
    worker.health_check.return_value = True
    connect.side_effect = OSError()
    with pytest.raises(HTTPException) as raised:
        namespace["ready"]()
    assert raised.value.status_code == 503
    app.state.workers = 2
    with pytest.raises(HTTPException, match="Workers are starting"):
        namespace["ready"]()


@pytest.mark.parametrize("arguments", [["--workers", "0"], ["--port", "0"], ["--port", "65536"]])
def test_launcher_rejects_invalid_arguments(arguments: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    app = MagicMock()
    upstream = SimpleNamespace(app=app, EnvInstanceManager=SimpleNamespace())
    uvicorn = SimpleNamespace(run=Mock())
    monkeypatch.setitem(sys.modules, "env_instance_manager", upstream)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.setattr(sys, "argv", ["rl2.lm_envs.codegym.launcher", "--port", "8000", "--workers", "1", *arguments])
    with pytest.raises(SystemExit) as raised:
        runpy.run_module("rl2.lm_envs.codegym.launcher", run_name="__main__")
    assert raised.value.code == 2
    uvicorn.run.assert_not_called()


@pytest.mark.parametrize("failure", [None, "filename", "empty", "install"])
def test_setup(failure: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    downloads: list[str] = []

    def download(url: str, **kwargs: Any) -> io.BytesIO:
        downloads.append(url)
        return io.BytesIO(b'ARGS = ["--host", "::"]' if url.endswith("env_instance_manager.py") else b"upstream")

    row = {
        "source": "../outside" if failure == "filename" else "Example",
        "env_name": "SumEnv",
        "env_code": (Path(__file__).parent / "fixtures/codegym.py").read_text(),
    }
    load = Mock(
        return_value=[]
        if failure == "empty"
        else [row, row | {"env_name": "OtherEnv", "env_code": row["env_code"].replace("SumEnv", "OtherEnv")}]
    )
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
    tools_file = tmp_path / "tools/Example__SumEnv.json"
    assert json.loads(tools_file.read_text())["tools"][0]["function"]["name"] == "Observe"
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
    setup.prepare_codegym(tmp_path, worker_host="0.0.0.0")
    assert (tmp_path / "env_instance_manager.py").read_text() == 'ARGS = ["--host", "0.0.0.0"]'
    assert len(downloads) == 10
    tools_file.unlink()
    setup.prepare_codegym(tmp_path, worker_host="0.0.0.0")
    assert tools_file.is_file()
    (tmp_path / "env_server.py").unlink()
    setup.prepare_codegym(tmp_path, worker_host="0.0.0.0")
    assert (tmp_path / "env_server.py").is_file()
    (tmp_path / "envs/codegym_v1/Example__SumEnv.py").unlink()
    setup.prepare_codegym(tmp_path, worker_host="0.0.0.0")
    assert (tmp_path / "envs/codegym_v1/Example__SumEnv.py").is_file()


def test_setup_repairs_upstream_fstrings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code = (Path(__file__).parent / "fixtures/codegym.py").read_text().replace("SumEnv", "MinPathCostEnv")
    code += '\nresult = f"First column: [{self.dp[i][0] for i in range(self.n)}]"'
    with pytest.raises(SyntaxError):
        compile(code, "upstream.py", "exec")
    monkeypatch.setattr(setup, "urlopen", Mock(side_effect=lambda *args, **kwargs: io.BytesIO(b"upstream")))
    monkeypatch.setattr(setup.subprocess, "run", Mock())
    monkeypatch.setattr(
        setup,
        "load_dataset",
        Mock(return_value=[{"source": "Codeforces_5537_I", "env_name": "MinPathCostEnv", "env_code": code}]),
    )
    setup.prepare_codegym(tmp_path)
    namespace = runpy.run_path(
        str(tmp_path / "envs/codegym_v1/Codeforces_5537_I__MinPathCostEnv.py"),
        init_globals={"self": SimpleNamespace(dp=[[1], [2]], n=2)},
    )
    assert namespace["result"] == "First column: [1, 2]"


def test_setup_rejects_invalid_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup, "urlopen", Mock(side_effect=lambda *args, **kwargs: io.BytesIO(b"upstream")))
    monkeypatch.setattr(setup.subprocess, "run", Mock())
    monkeypatch.setattr(
        setup,
        "load_dataset",
        Mock(return_value=[{"source": "Example", "env_name": "BrokenEnv", "env_code": "def broken("}]),
    )
    with pytest.raises(SyntaxError):
        setup.prepare_codegym(tmp_path)
    assert not (tmp_path / ".ready").exists()
