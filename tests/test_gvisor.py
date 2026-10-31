import asyncio
import os
import sys
from typing import Any
from unittest.mock import AsyncMock

import pytest

from rl2.sandbox import gvisor


@pytest.mark.parametrize("exit_code", [0, 1], ids=["success", "python-error"])
def test_python_execution_and_isolation_options(exit_code: int, monkeypatch: pytest.MonkeyPatch) -> None:
    result = gvisor.PythonResult(output="result or traceback", exit_code=exit_code, output_truncated=False)
    success = gvisor.PythonResult(output="", exit_code=0, output_truncated=False)
    docker = AsyncMock(side_effect=[success, result, success, success, result, success])
    monkeypatch.setattr(gvisor, "_docker", docker)
    code = "print('shell syntax is just source: $(id)')"

    async def run() -> None:
        assert await gvisor.run_python(code) == result
        assert await gvisor.run_python(code) == result

    asyncio.run(run())
    create, start, remove, second_create, _, _ = docker.call_args_list
    assert create.args[0] == "create"
    assert {
        "--runtime=runsc",
        "--network=none",
        "--ipc=none",
        "--read-only",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=16m",
        "--user=65534:65534",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--memory=256m",
        "--memory-swap=256m",
        "--pids-limit=32",
        "--cpus=1",
        "--log-driver=none",
        "--no-healthcheck",
        "--pull=never",
        "--entrypoint=python",
    } <= set(create.args)
    assert not any(argument.startswith(("--volume", "--mount", "--env")) for argument in create.args)
    assert create.args[-6:] == ("python:3.12-slim", "-I", "-B", "-u", "-c", code)
    name = create.args[1].removeprefix("--name=")
    assert start.args == ("start", "--attach", name)
    assert remove.args == ("rm", "--force", "--volumes", name)
    assert second_create.args[1] != create.args[1]


@pytest.mark.parametrize("mode", ["timeout", "cancel", "start-error", "create-cancel"])
def test_execution_failure_removes_container(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    success = gvisor.PythonResult(output="", exit_code=0, output_truncated=False)

    async def docker(*arguments: str, **kwargs: object) -> gvisor.PythonResult:
        if arguments[0] == ("create" if mode == "create-cancel" else "start"):
            started.set()
            if mode == "start-error":
                raise OSError("connection lost")
            await asyncio.Future()
        return success

    command = AsyncMock(side_effect=docker)
    monkeypatch.setattr(gvisor, "_docker", command)

    async def run() -> None:
        if mode in {"cancel", "create-cancel"}:
            async with asyncio.TaskGroup() as tg:
                task = tg.create_task(gvisor.run_python("pass"))
                await started.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        else:
            with pytest.raises(TimeoutError if mode == "timeout" else OSError):
                await gvisor.run_python("pass", timeout=0.001)

    asyncio.run(run())
    expected = ["create", "rm"] if mode == "create-cancel" else ["create", "start", "rm"]
    assert [call.args[0] for call in command.call_args_list] == expected


@pytest.mark.parametrize("operation", ["create", "rm"])
def test_docker_failure_is_reported(operation: str, monkeypatch: pytest.MonkeyPatch) -> None:
    async def docker(*arguments: str, **kwargs: object) -> gvisor.PythonResult:
        return gvisor.PythonResult(
            output="runtime unavailable", exit_code=int(arguments[0] == operation), output_truncated=False
        )

    command = AsyncMock(side_effect=docker)
    monkeypatch.setattr(gvisor, "_docker", command)
    with pytest.raises(RuntimeError, match="runtime unavailable"):
        asyncio.run(gvisor.run_python("pass"))
    if operation == "create":
        assert [call.args[0] for call in command.call_args_list] == ["create", "rm"]


@pytest.mark.parametrize(
    "settings",
    [
        {"code": "é" * 8193},
        {"code": "\0"},
        {"image": "--privileged"},
        {"timeout": 0},
        {"timeout": float("inf")},
        {"max_output_bytes": 0},
    ],
    ids=["code-bytes", "nul", "image-option", "zero-timeout", "unbounded-timeout", "output-limit"],
)
def test_invalid_requests_do_not_launch(settings: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    docker = AsyncMock()
    monkeypatch.setattr(gvisor, "_docker", docker)
    with pytest.raises(ValueError):
        asyncio.run(gvisor.run_python(**({"code": "pass"} | settings)))
    docker.assert_not_called()


@pytest.mark.parametrize("extra_bytes", [0, 100000], ids=["exact-limit", "output-flood"])
def test_bounded_subprocess_output(extra_bytes: int, monkeypatch: pytest.MonkeyPatch) -> None:
    launch = asyncio.create_subprocess_exec

    async def start(*arguments: str, **kwargs: Any) -> asyncio.subprocess.Process:
        assert arguments == ("docker", "test-command")
        # A local fixture exercises transport only, never model-generated code.
        return await launch(
            sys.executable,
            "-c",
            f"import os; os.write(1, b'42'); os.write(2, b'!'); os.write(1, b'x' * {extra_bytes}); raise SystemExit(7)",
            **kwargs,
        )

    monkeypatch.setattr(gvisor.asyncio, "create_subprocess_exec", start)
    result = asyncio.run(gvisor._docker("test-command", max_output_bytes=3))
    assert result == gvisor.PythonResult(output="42!", exit_code=7, output_truncated=bool(extra_bytes))


def test_cancel_kills_and_reaps_docker_client(monkeypatch: pytest.MonkeyPatch) -> None:
    launch = asyncio.create_subprocess_exec
    processes: list[asyncio.subprocess.Process] = []
    started = asyncio.Event()

    async def start(*arguments: str, **kwargs: Any) -> asyncio.subprocess.Process:
        process = await launch(sys.executable, "-c", "import signal; signal.pause()", **kwargs)
        processes.append(process)
        started.set()
        return process

    monkeypatch.setattr(gvisor.asyncio, "create_subprocess_exec", start)

    async def run() -> None:
        async with asyncio.TaskGroup() as tg:
            task = tg.create_task(gvisor._docker("test-command"))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert processes[0].returncode is not None

    asyncio.run(run())


@pytest.mark.skipif(
    os.environ.get("RL2_TEST_GVISOR") != "1", reason="requires Docker, runsc, and a pre-pulled Python image"
)
def test_live_gvisor_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RL2_SANDBOX_SECRET", "must-not-be-inherited")
    code = """
import os
import socket
from pathlib import Path

assert os.getuid() == 65534
assert 'RL2_SANDBOX_SECRET' not in os.environ
assert not Path('/var/run/docker.sock').exists()
assert {name for _, name in socket.if_nameindex()} <= {'lo'}
try:
    socket.create_connection(('192.0.2.1', 80), timeout=0.1)
except OSError:
    pass
else:
    raise AssertionError('network is available')
try:
    Path('/root-write-test').write_text('blocked')
except OSError:
    pass
else:
    raise AssertionError('root filesystem is writable')
Path('/tmp/example').write_text('temporary')
print(6 * 7)
"""
    result = asyncio.run(gvisor.run_python(code, timeout=30))
    assert result == gvisor.PythonResult(output="42\n", exit_code=0, output_truncated=False)
