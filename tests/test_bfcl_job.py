import fcntl
import io
import os
import signal
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock
from urllib.error import URLError

import pytest

from rl2 import bfcl_job as job


@pytest.mark.parametrize("failure", [None, "download", "ready", "generate", "server", "cancel", "evaluate"])
def test_job_lifecycle(failure: str | None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events = []
    commands = {}
    environments = []

    @contextmanager
    def fake_process(command: list[str], log_path: Path, env: dict[str, str]) -> Iterator[Mock]:
        stage = log_path.stem
        events.append(f"start:{stage}")
        commands[stage] = command
        environments.append(env)
        child = Mock(args=command, stage=stage)
        try:
            yield child
        finally:
            events.append(f"stop:{stage}")

    def ready(server: Mock, port: int, timeout: int) -> None:
        events.append("ready")
        assert port == 1053 and timeout == 30
        if failure == "ready":
            raise TimeoutError("startup")

    def wait(child: Mock, server: Mock | None = None) -> None:
        if failure == "cancel" and child.stage == "generate":
            job.terminate(signal.SIGTERM, None)
        if failure == "server" and child.stage == "generate":
            assert server is not None
            raise RuntimeError("server died")
        if failure == child.stage:
            raise subprocess.CalledProcessError(returncode=1, cmd=child.args)

    monkeypatch.setenv("HF_HOME", "/shared/model-cache")
    monkeypatch.setattr(job, "process", fake_process)
    monkeypatch.setattr(job, "wait_ready", ready)
    monkeypatch.setattr(job, "wait_success", wait)
    run_dir = tmp_path / "run"
    if failure:
        with pytest.raises((TimeoutError, RuntimeError, subprocess.CalledProcessError, SystemExit)):
            job.run_job(run_dir, port=1053, smoke=True, startup_timeout=30)
    else:
        job.run_job(run_dir, port=1053, smoke=True, startup_timeout=30)
        assert events == [
            "start:download",
            "stop:download",
            "start:serve",
            "ready",
            "start:generate",
            "stop:generate",
            "stop:serve",
            "start:evaluate",
            "stop:evaluate",
        ]
        assert commands["generate"][4:] == ["--run-dir", str(run_dir), "--port", "1053", "--smoke"]
        assert commands["evaluate"][4:] == commands["generate"][4:]
    assert all(f"stop:{stage}" in events for stage in commands)
    assert ("evaluate" in commands) == (failure in (None, "evaluate"))
    assert all(env["HF_HOME"] == "/shared/model-cache" for env in environments)
    # The run lock is released on both success and failure.
    with (run_dir / ".job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_concurrent_run_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    launch = Mock()
    monkeypatch.setattr(job, "process", launch)
    with (tmp_path / ".job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            job.run_job(tmp_path, port=1053, smoke=False, startup_timeout=30)
    launch.assert_not_called()


def test_readiness_retries_until_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    request = Mock(side_effect=[URLError("not listening"), io.BytesIO(b"ok")])
    monkeypatch.setattr(job, "urlopen", request)
    monkeypatch.setattr(job.time, "sleep", Mock())
    job.wait_ready(Mock(poll=Mock(return_value=None)), port=1053, timeout=30)
    assert request.call_count == 2


@pytest.mark.parametrize("exited", [False, True], ids=["startup-timeout", "server-exited"])
def test_readiness_failure(exited: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    request = Mock(side_effect=URLError("not listening"))
    monkeypatch.setattr(job, "urlopen", request)
    monkeypatch.setattr(job.time, "sleep", Mock())
    monkeypatch.setattr(job.time, "monotonic", Mock(side_effect=[0, 0, 2]))
    server = Mock(poll=Mock(return_value=1 if exited else None), returncode=1)
    with pytest.raises(RuntimeError if exited else TimeoutError):
        job.wait_ready(server, port=1053, timeout=1)


def test_generation_detects_server_failure() -> None:
    child = Mock(wait=Mock(side_effect=subprocess.TimeoutExpired(cmd="generate", timeout=0.5)))
    server = Mock(poll=Mock(side_effect=[None, 1]), returncode=1)
    with pytest.raises(RuntimeError, match="during generation"):
        job.wait_success(child, server)
    child.wait.assert_called_once()


def test_scoring_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    request = Mock()
    monkeypatch.setattr(job, "urlopen", request)
    with pytest.raises(subprocess.CalledProcessError) as error:
        job.wait_success(
            Mock(wait=Mock(side_effect=[subprocess.TimeoutExpired(cmd="evaluate", timeout=5), 7]), args=["evaluate"])
        )
    assert error.value.returncode == 7
    request.assert_not_called()


def test_process_cleanup_on_cancellation(tmp_path: Path) -> None:
    with (
        pytest.raises(SystemExit) as error,
        job.process(
            [sys.executable, "-c", "import time; time.sleep(60)"], tmp_path / "child.log", dict(os.environ)
        ) as child,
    ):
        assert os.getpgid(child.pid) == child.pid
        job.terminate(signal.SIGTERM, None)
    assert error.value.code == 143
    assert child.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(child.pid, 0)


def test_process_forces_cleanup_of_remaining_workers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    child = Mock(pid=123, wait=Mock(side_effect=[subprocess.TimeoutExpired(cmd="serve", timeout=10), 0]))
    popen = Mock()
    popen.__enter__ = Mock(return_value=child)
    popen.__exit__ = Mock(return_value=False)
    kill = Mock()
    monkeypatch.setattr(job.subprocess, "Popen", Mock(return_value=popen))
    monkeypatch.setattr(job.os, "killpg", kill)
    with job.process(["serve"], tmp_path / "serve.log", {}):
        pass
    assert [call.args for call in kill.call_args_list] == [(123, signal.SIGTERM), (123, signal.SIGKILL)]
