import fcntl
import io
import os
import signal
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import URLError

import pytest
import ray
from ray.exceptions import GetTimeoutError, RayTaskError, TaskCancelledError

from rl2.bfcl import job


@pytest.mark.parametrize("failure", [None, "download", "ready", "generate", "cancel", "score"])
def test_job_lifecycle(failure: str | None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events = []
    initialize = Mock(return_value=nullcontext())
    monkeypatch.setattr(job.ray, "init", initialize)
    monkeypatch.setattr(job.ray, "cluster_resources", Mock(return_value={"GPU": 1}))
    monkeypatch.setenv("HF_HOME", "/shared/model-cache")
    for stage in ("download", "generate", "score"):
        monkeypatch.setattr(job, stage, SimpleNamespace(remote=Mock(return_value=stage)))

    def wait(task: str, server: str | None = None) -> None:
        events.append(task)
        if task == "generate":
            assert server == "server"
            if failure == "cancel":
                job.terminate(signal.SIGTERM, None)
        if failure == task:
            raise RuntimeError(task)

    @contextmanager
    def serving(run: job.Run, startup_timeout: int) -> Iterator[str]:
        events.append("start")
        try:
            if failure == "ready":
                raise TimeoutError("startup")
            yield "server"
        finally:
            events.append("stop")

    monkeypatch.setattr(job, "wait_task", wait)
    monkeypatch.setattr(job, "serving", serving)
    if failure:
        with pytest.raises((TimeoutError, RuntimeError, SystemExit)):
            job.run_job(tmp_path, port=1053, smoke=True, startup_timeout=30)
    else:
        job.run_job(tmp_path, port=1053, smoke=True, startup_timeout=30)
        assert events == ["download", "start", "generate", "stop", "score"]
    assert ("score" in events) == (failure in (None, "score"))
    assert ("stop" in events) == ("start" in events)
    assert initialize.call_args.kwargs["address"] == "local"
    assert initialize.call_args.kwargs["log_to_driver"] is True
    assert initialize.call_args.kwargs["runtime_env"]["env_vars"]["HF_HOME"] == "/shared/model-cache"
    with (tmp_path / ".job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_concurrent_run_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    initialize = Mock()
    monkeypatch.setattr(job.ray, "init", initialize)
    with (tmp_path / ".job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            job.run_job(tmp_path, port=1053, smoke=False, startup_timeout=30)
    initialize.assert_not_called()


def test_missing_gpu_fails_before_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(job.ray, "init", Mock(return_value=nullcontext()))
    monkeypatch.setattr(job.ray, "cluster_resources", Mock(return_value={}))
    download = Mock()
    monkeypatch.setattr(job, "download", download)
    with pytest.raises(RuntimeError, match="one visible GPU"):
        job.run_job(tmp_path, port=1053, smoke=True, startup_timeout=30)
    download.remote.assert_not_called()


def test_command_preserves_protocol_options(tmp_path: Path) -> None:
    run = job.Run(directory=tmp_path, port=1054, smoke=True)
    assert run.command("generate") == [
        sys.executable,
        "-m",
        "rl2.bfcl",
        "generate",
        "--run-dir",
        str(tmp_path),
        "--port",
        "1054",
        "--smoke",
    ]


def test_actor_readiness_and_stop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    child = Mock(poll=Mock(return_value=None), returncode=1)
    stopped = Mock()

    @contextmanager
    def process(command: list[str], log_path: Path) -> Iterator[Mock]:
        try:
            yield child
        finally:
            stopped()

    monkeypatch.setattr(job, "process", process)
    monkeypatch.setattr(job, "urlopen", Mock(side_effect=[URLError("starting"), io.BytesIO(b"ok")]))
    actor = job.VLLMServer(job.Run(directory=tmp_path, port=1053, smoke=False))
    assert not actor.ready()
    assert actor.ready()
    child.poll.return_value = 1
    with pytest.raises(RuntimeError, match="vLLM exited"):
        actor.check()
    actor.stop()
    actor.stop()
    stopped.assert_called_once()


def test_startup_timeout_stops_actor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    actor = Mock()
    monkeypatch.setattr(job, "ServerActor", Mock(remote=Mock(return_value=actor)))
    monkeypatch.setattr(job.ray, "get", Mock(return_value=None))
    kill = Mock()
    monkeypatch.setattr(job.ray, "kill", kill)
    monkeypatch.setattr(job.time, "monotonic", Mock(side_effect=[0, 2]))
    with (
        pytest.raises(TimeoutError),
        job.serving(job.Run(directory=tmp_path, port=1053, smoke=False), startup_timeout=1),
    ):
        pytest.fail("Server was not ready")
    actor.stop.remote.assert_called_once()
    job.ray.get.assert_called_with(actor.stop.remote.return_value)
    kill.assert_called_once_with(actor)


def test_actor_startup_waits_for_scheduling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    actor = Mock()
    monkeypatch.setattr(job, "ServerActor", Mock(remote=Mock(return_value=actor)))
    monkeypatch.setattr(job.ray, "get", Mock(side_effect=[GetTimeoutError(), True, None]))
    monkeypatch.setattr(job.ray, "kill", Mock())
    monkeypatch.setattr(job.time, "monotonic", Mock(side_effect=[0, 0, 5]))
    with job.serving(job.Run(directory=tmp_path, port=1053, smoke=False), startup_timeout=30) as server:
        assert server is actor
    actor.ready.remote.assert_called_once()
    actor.stop.remote.assert_called_once()
    job.ray.get.assert_called_with(actor.stop.remote.return_value)


def test_server_failure_cancels_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    task, server = Mock(), Mock()
    monkeypatch.setattr(job.ray, "get", Mock(side_effect=[GetTimeoutError(), RuntimeError("server died"), None]))
    cancel = Mock()
    monkeypatch.setattr(job.ray, "cancel", cancel)
    with pytest.raises(RuntimeError, match="server died"):
        job.wait_task(task, server)
    cancel.assert_called_once_with(task, force=False)
    job.ray.get.assert_called_with(task)


def test_process_cleanup_on_cancellation(tmp_path: Path) -> None:
    with (
        pytest.raises(SystemExit),
        job.process([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path / "child.log") as child,
    ):
        assert os.getpgid(child.pid) == child.pid
        job.terminate(signal.SIGTERM, None)
    assert child.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(child.pid, 0)


def test_process_streams_output_before_exit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    release = tmp_path / "release"
    log = tmp_path / "generate.log"
    log.write_text("previous run\n")
    command = [
        sys.executable,
        "-c",
        (
            "import os, subprocess, sys, time\n"
            "from pathlib import Path\n"
            "print('stdout-output')\n"
            "print('stderr-output', file=sys.stderr)\n"
            "subprocess.run([sys.executable, '-c', \"print('worker-output')\"], check=True)\n"
            "while not Path(sys.argv[1]).exists(): time.sleep(0.01)\n"
            "os.write(1, b'trailing-\\xff')\n"
        ),
        str(release),
    ]
    with job.process(command, log) as child:
        captured = ""
        deadline = time.monotonic() + 5
        while "[generate] worker-output" not in captured and time.monotonic() < deadline:
            captured += capsys.readouterr().out
            time.sleep(0.01)
        assert child.poll() is None
        for message in ("stdout-output", "stderr-output", "worker-output"):
            assert f"[generate] {message}" in captured
            assert message in log.read_text()
        release.touch()
        assert child.wait(timeout=5) == 0
    assert "[generate] trailing-\ufffd\n" in capsys.readouterr().out
    assert log.read_text() == "previous run\nstdout-output\nstderr-output\nworker-output\ntrailing-\ufffd"


def test_ray_actor_and_task_processes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Exercise real Ray workers with tiny CPU-only subprocesses, never a model.
    class Server(job.VLLMServer):
        def __init__(self, run: job.Run) -> None:
            command = [sys.executable, "-c", "import time; print('server-output'); time.sleep(60)"]
            with patch.object(job.Run, "command", return_value=command):
                super().__init__(run)

        def pid(self) -> int:
            return self.child.pid

    @ray.remote(max_retries=0)
    def stage(run: job.Run, command: list[str], action: str = "generate") -> None:
        with patch.object(job.Run, "command", return_value=command):
            job.run_stage(action, run)

    run = job.Run(directory=tmp_path, port=1053, smoke=False)
    with ray.init(address="local", num_cpus=1, num_gpus=0, include_dashboard=False, object_store_memory=80 * 1024**2):
        actor = ray.remote(num_cpus=0)(Server).remote(run)
        try:
            server_pid = ray.get(actor.pid.remote(), timeout=30)
            ray.get(actor.check.remote(), timeout=5)
            ray.get(stage.remote(run, [sys.executable, "-c", "print('task-output')"]), timeout=30)
            assert "task-output" in (tmp_path / "generate.log").read_text()
            ray.get(stage.remote(run, [sys.executable, "-c", "print('score-output')"], "evaluate"), timeout=15)
            captured = ""
            messages = ("[serve] server-output", "[generate] task-output", "[evaluate] score-output")
            deadline = time.monotonic() + 10
            while not all(message in captured for message in messages) and time.monotonic() < deadline:
                captured += capsys.readouterr().out
                time.sleep(0.05)
            for message in messages:
                assert message in captured
            with pytest.raises(RayTaskError):
                ray.get(stage.remote(run, [sys.executable, "-c", "raise SystemExit(7)"]), timeout=15)
            pid_file = tmp_path / "task.pid"
            command = [
                sys.executable,
                "-c",
                "import os, sys, time; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)",
                str(pid_file),
            ]
            task = stage.remote(run, command)
            try:
                deadline = time.monotonic() + 15
                while not pid_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                assert pid_file.exists()
            finally:
                ray.cancel(task, force=False)
                with pytest.raises(TaskCancelledError):
                    ray.get(task, timeout=15)
            task_pid = int(pid_file.read_text())
            with pytest.raises(ProcessLookupError):
                os.killpg(task_pid, 0)
            ray.get(actor.stop.remote(), timeout=15)
            with pytest.raises(ProcessLookupError):
                os.killpg(server_pid, 0)
        finally:
            ray.get(actor.stop.remote(), timeout=15)
            ray.kill(actor)
