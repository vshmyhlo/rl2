import asyncio
import json
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from rl2.sandbox import _python_worker as worker
from rl2.sandbox import kubernetes as sandbox


@pytest.fixture
def commands(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Exercise the real protocol with local fixture code, without Kubernetes."""
    command = AsyncMock()
    monkeypatch.setattr(sandbox.PythonSession, "_command", command)
    original = sandbox._process

    @asynccontextmanager
    async def local_process(arguments: tuple[str, ...], *, limit: int) -> AsyncGenerator[asyncio.subprocess.Process]:
        assert "exec" in arguments and "-i" in arguments
        # Only the cluster/network/resource setup is faked; the interpreter loop is real.
        source = """import resource
import socket
def interfaces() -> list[tuple[int, str]]:
    return [(1, 'lo')]
def setrlimit(kind: int, limits: tuple[int, int]) -> None:
    pass
socket.if_nameindex = interfaces
resource.setrlimit = setrlimit
""" + arguments[-2]
        async with original((sys.executable, "-I", "-B", "-u", "-c", source, arguments[-1]), limit=limit) as process:
            yield process

    monkeypatch.setattr(sandbox, "_process", local_process)
    return command


def test_persistent_state_errors_and_separate_sessions(commands: AsyncMock) -> None:
    async def run() -> None:
        async with sandbox.PythonSession() as first, sandbox.PythonSession() as second:
            assert (await first.execute("import math\nnumbers = [12, 18, 24]")).exit_code == 0
            assert (await first.execute("print(math.gcd(*numbers))")).output == "6\n"
            assert "NameError" in (await second.execute("print(numbers)")).output
            failed = await first.execute("value = 40\nraise ValueError('example')")
            assert failed.exit_code == 1 and "ValueError: example" in failed.output
            assert (await first.execute("print(value + 2)")).output == "42\n"
            syntax_error = await first.execute("if")
            assert syntax_error.exit_code == 1 and "SyntaxError" in syntax_error.output
            assert (await first.execute("print(value)")).output == "40\n"
            assert (await first.execute("raise SystemExit(3)")).exit_code == 1
            assert (await first.execute("print('still running')")).output == "still running\n"
        await first.aclose()
        with pytest.raises(RuntimeError, match="not active"):
            await first.execute("pass")
        with pytest.raises(RuntimeError, match="only be entered once"):
            await first.__aenter__()

    asyncio.run(run())
    operations = [call.args[0] for call in commands.call_args_list]
    assert operations.count("create") == operations.count("wait") == operations.count("delete") == 2
    creates = [json.loads(call.kwargs["payload"]) for call in commands.call_args_list if call.args[0] == "create"]
    assert creates[0]["metadata"]["name"] != creates[1]["metadata"]["name"]
    spec = creates[0]["spec"]
    assert spec["runtimeClassName"] == "runsc-no-network"
    assert not spec["automountServiceAccountToken"] and not spec["enableServiceLinks"]
    assert spec["activeDeadlineSeconds"] == 1800 and spec["restartPolicy"] == "Never"
    assert spec["securityContext"]["runAsNonRoot"]
    container = spec["containers"][0]
    assert container["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "readOnlyRootFilesystem": True,
        "capabilities": {"drop": ["ALL"]},
    }
    assert container["resources"]["limits"] == {"cpu": "1", "memory": "256Mi"}
    assert spec["volumes"] == [{"name": "tmp", "emptyDir": {"medium": "Memory", "sizeLimit": "16Mi"}}]


def test_output_capture_and_limit_reset(commands: AsyncMock) -> None:
    async def run() -> None:
        async with sandbox.PythonSession(max_output_bytes=16) as session:
            result = await session.execute(
                "import os, subprocess, sys\n"
                "print('A')\nos.write(2, b'B\\n')\n"
                "subprocess.run([sys.executable, '-c', 'print(42)'], check=True)"
            )
            assert result == sandbox.PythonResult(output="A\nB\n42\n", exit_code=0, output_truncated=False)
            exact = await session.execute("os.write(1, b'x' * 16)")
            assert exact.output == "x" * 16 and not exact.output_truncated
            flood = await session.execute("os.write(2, b'x' * 100000)")
            assert flood == sandbox.PythonResult(output="x" * 16, exit_code=0, output_truncated=True)
            assert (await session.execute("print('next')")).output == "next\n"
            assert (await session.execute("os.write(1, b'\\xff')")).output == "\ufffd"
            assert (await session.execute("print(input())")).exit_code == 1

    asyncio.run(run())


def test_concurrent_calls_are_serialized(commands: AsyncMock) -> None:
    async def run() -> None:
        async with sandbox.PythonSession() as session:
            await session.execute("value = 0")
            async with asyncio.TaskGroup() as tg:
                first = tg.create_task(session.execute("value += 1\nprint(value)"))
                second = tg.create_task(session.execute("value += 1\nprint(value)"))
            assert {first.result().output, second.result().output} == {"1\n", "2\n"}
            assert (await session.execute("print(value)")).output == "2\n"

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["timeout", "cancel", "exit"])
def test_interrupted_execution_closes_session(mode: str, commands: AsyncMock) -> None:
    async def run() -> None:
        async with sandbox.PythonSession() as session:
            if mode == "cancel":
                async with asyncio.TaskGroup() as tg:
                    task = tg.create_task(session.execute("while True: pass"))
                    await asyncio.sleep(0)
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
            elif mode == "timeout":
                with pytest.raises(TimeoutError):
                    await session.execute("while True: pass", timeout=0.02)
            else:
                with pytest.raises(RuntimeError, match="exited without a response"):
                    await session.execute("import os; os._exit(0)")
            with pytest.raises(RuntimeError, match="not active"):
                await session.execute("print(42)")

    asyncio.run(run())
    assert [call.args[0] for call in commands.call_args_list].count("delete") == 1


def test_invalid_code_does_not_break_session(commands: AsyncMock) -> None:
    async def run() -> None:
        async with sandbox.PythonSession() as session:
            for code in ("é" * 8193, "\0"):
                with pytest.raises(ValueError, match="16384"):
                    await session.execute(code)
            with pytest.raises(ValueError, match="timeout"):
                await session.execute("pass", timeout=0)
            assert (await session.execute("#" + "x" * 16383)).exit_code == 0
            assert (await session.execute("print(42)")).output == "42\n"

    asyncio.run(run())


@pytest.mark.parametrize("operation", ["create", "wait"])
def test_startup_failure_attempts_cleanup(operation: str, commands: AsyncMock) -> None:
    async def command(*arguments: str, payload: bytes = b"") -> None:
        if arguments[0] == operation:
            raise RuntimeError("cluster failure")

    commands.side_effect = command
    with pytest.raises(RuntimeError, match="cluster failure"):
        asyncio.run(sandbox.PythonSession().__aenter__())
    assert commands.call_args.args[0] == "delete"


def test_cleanup_failure_is_reported(commands: AsyncMock) -> None:
    async def command(*arguments: str, payload: bytes = b"") -> None:
        if arguments[0] == "delete":
            raise RuntimeError("delete failed")

    commands.side_effect = command

    async def run() -> None:
        with pytest.raises(RuntimeError, match="delete failed"):
            async with sandbox.PythonSession() as session:
                assert (await session.execute("pass")).exit_code == 0

    asyncio.run(run())


def test_cancelled_startup_attempts_cleanup(commands: AsyncMock) -> None:
    creating = asyncio.Event()

    async def command(*arguments: str, payload: bytes = b"") -> None:
        if arguments[0] == "create":
            creating.set()
            await asyncio.Future()

    commands.side_effect = command

    async def run() -> None:
        async with asyncio.TaskGroup() as tg:
            task = tg.create_task(sandbox.PythonSession().__aenter__())
            await creating.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
    assert [call.args[0] for call in commands.call_args_list] == ["create", "delete"]


def test_worker_rejects_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def interfaces() -> list[tuple[int, str]]:
        return [(1, "lo"), (2, "eth0")]

    monkeypatch.setattr(sys, "argv", ["worker", "16"])
    monkeypatch.setattr(worker.socket, "if_nameindex", interfaces)
    with pytest.raises(RuntimeError, match="external network"):
        worker.main()


@pytest.mark.parametrize("exit_code", [0, 7], ids=["success", "failure-with-bounded-diagnostics"])
def test_kubectl_command_transport(exit_code: int) -> None:
    code = (
        "import sys; assert sys.stdin.read() == 'payload'; "
        f"print('x' * 20000, file=sys.stderr); raise SystemExit({exit_code})"
    )
    session = sandbox.PythonSession(kubectl=(sys.executable, "-c", code))

    async def run() -> None:
        async with asyncio.timeout(5):
            if exit_code:
                with pytest.raises(RuntimeError, match="kubectl test failed") as error:
                    await session._command("test", payload=b"payload")
                assert len(str(error.value)) < 16500
            else:
                await session._command("test", payload=b"payload")

    asyncio.run(run())


@pytest.mark.parametrize(
    "response",
    [
        b"not json\n",
        b'{"id":999,"output":"","exit_code":0,"output_truncated":false}\n',
        json.dumps({"id": 1, "output": "x" * 49, "exit_code": 0, "output_truncated": False}).encode() + b"\n",
    ],
    ids=["invalid-json", "wrong-request", "excessive-output"],
)
def test_corrupt_protocol_closes_session(response: bytes, commands: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        async with sandbox.PythonSession(max_output_bytes=16) as session:
            assert session._process is not None and session._process.stdout is not None
            reader = session._process.stdout
            original = reader.readline
            monkeypatch.setattr(reader, "readline", AsyncMock(return_value=response))
            with pytest.raises((ValueError, RuntimeError)):
                await session.execute("pass")
            monkeypatch.setattr(reader, "readline", original)
            with pytest.raises(RuntimeError, match="not active"):
                await session.execute("pass")

    asyncio.run(run())


@pytest.mark.skipif(
    os.environ.get("RL2_TEST_K3S_SANDBOX") != "1", reason="requires offline runsc RuntimeClass and sandbox namespace"
)
def test_live_k3s_session() -> None:
    async def run() -> None:
        async with sandbox.PythonSession(kubectl=("sudo", "-n", "k3s", "kubectl")) as session:
            assert (await session.execute("value = 40")).exit_code == 0
            assert (await session.execute("print(value + 2)")).output == "42\n"
            result = await session.execute(
                "import os, socket\n"
                "assert os.getuid() == 65534\n"
                "assert all(name == 'lo' for _, name in socket.if_nameindex())\n"
                "assert not os.path.exists('/var/run/secrets/kubernetes.io/serviceaccount/token')\n"
                "open('/tmp/example', 'w').write('temporary')"
            )
            assert result.exit_code == 0, result.output

    asyncio.run(run())
