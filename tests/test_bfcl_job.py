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


@pytest.mark.parametrize("failure", [None, "generate", "server", "cancel", "evaluate"])
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

    def wait(child: Mock, port: int | None = None) -> None:
        if failure == "cancel" and child.stage == "generate":
            job.terminate(signal.SIGTERM, None)
        if failure == "server" and child.stage == "generate":
            assert port == 1053
            raise RuntimeError("server died")
        if failure == child.stage:
            raise subprocess.CalledProcessError(returncode=1, cmd=child.args)

    monkeypatch.setenv("HF_HOME", "/shared/model-cache")
    monkeypatch.setattr(job, "process", fake_process)
    monkeypatch.setattr(job, "wait_success", wait)
    run_dir = tmp_path / "run"
    if failure:
        with pytest.raises((TimeoutError, RuntimeError, subprocess.CalledProcessError, SystemExit)):
            job.run_job(run_dir, port=1053, smoke=True)
    else:
        job.run_job(run_dir, port=1053, smoke=True)
        assert events == [
            "start:generate",
            "stop:generate",
            "start:evaluate",
            "stop:evaluate",
        ]
        assert commands["generate"][4:] == ["--run-dir", str(run_dir), "--port", "1053", "--smoke"]
        assert commands["evaluate"][4:] == commands["generate"][4:]
    assert all(f"stop:{stage}" in events for stage in commands)
    assert ("evaluate" in commands) == (failure in (None, "evaluate"))
    assert set(commands) <= {"generate", "evaluate"}
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
            job.run_job(tmp_path, port=1053, smoke=False)
    launch.assert_not_called()


def test_generation_tolerates_temporary_server_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    request = Mock(side_effect=[URLError("not listening"), io.BytesIO(b"ok")])
    monkeypatch.setattr(job, "urlopen", request)
    monkeypatch.setattr(job.time, "monotonic", Mock(side_effect=[0, 5, 10]))
    waiting = subprocess.TimeoutExpired(cmd="generate", timeout=5)
    job.wait_success(Mock(wait=Mock(side_effect=[waiting, waiting, 0])), port=1053)
    assert request.call_count == 2


def test_generation_detects_server_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    request = Mock(side_effect=URLError("not listening"))
    monkeypatch.setattr(job, "urlopen", request)
    monkeypatch.setattr(job.time, "monotonic", Mock(side_effect=[0, 5, 30]))
    child = Mock(wait=Mock(side_effect=subprocess.TimeoutExpired(cmd="generate", timeout=5)))
    with pytest.raises(RuntimeError, match="during generation"):
        job.wait_success(child, port=1053)
    assert child.wait.call_count == 2


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
