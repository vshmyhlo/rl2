"""One offline gVisor pod and persistent Python interpreter per async session."""

import asyncio
import json
import os
import signal
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from rl2.sandbox.gvisor import PythonResult


class _Response(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: int
    output: str
    exit_code: Literal[0, 1]
    output_truncated: bool


@asynccontextmanager
async def _process(command: tuple[str, ...], *, limit: int) -> AsyncGenerator[asyncio.subprocess.Process]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        limit=limit,
        start_new_session=True,
    )
    try:
        yield process
    finally:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        async with asyncio.timeout(10):
            assert process.stdout is not None
            while await process.stdout.read(4096):
                pass
            await process.wait()


class PythonSession:
    """Own a single-use Kubernetes sandbox; execute calls share globals and files.

    Requires kubectl credentials in the trusted caller and a RuntimeClass backed
    by runsc with network=none. Startup, calls, and cleanup have separate deadlines.
    Exceptions in submitted Python return exit_code=1; transport failures, timeout,
    or cancellation close the session. Concurrent calls are serialized.
    """

    def __init__(
        self,
        *,
        namespace: str = "sandbox",
        image: str = "python:3.12-slim-bookworm",
        runtime_class: str = "runsc-no-network",
        kubectl: tuple[str, ...] = ("kubectl",),
        startup_timeout: float = 90,
        execution_timeout: float = 5,
        cleanup_timeout: float = 30,
        lifetime_seconds: int = 1800,
        max_output_bytes: int = 16384,
    ) -> None:
        if not kubectl or not all((namespace, image, runtime_class)):
            raise ValueError("kubectl, namespace, image, and runtime_class must be nonempty")
        if not all(0 < value <= 3600 for value in (startup_timeout, execution_timeout, cleanup_timeout)):
            raise ValueError("timeouts must be in (0, 3600] seconds")
        if lifetime_seconds <= 0 or not 0 < max_output_bytes <= 1048576:
            raise ValueError("lifetime must be positive and output limit must be in [1, 1048576] bytes")
        self.namespace = namespace
        self.image = image
        self.runtime_class = runtime_class
        self.kubectl = (*kubectl, "--namespace", namespace)
        self.startup_timeout = startup_timeout
        self.execution_timeout = execution_timeout
        self.cleanup_timeout = cleanup_timeout
        self.lifetime_seconds = lifetime_seconds
        self.max_output_bytes = max_output_bytes
        self.pod_name = f"rl2-python-{uuid4().hex}"
        self._resources = AsyncExitStack()
        self._lock = asyncio.Lock()
        self._state = "new"
        self._request_id = 0
        self._process: asyncio.subprocess.Process | None = None

    def _manifest(self) -> dict[str, Any]:
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": self.pod_name, "namespace": self.namespace, "labels": {"app": "rl2-python"}},
            "spec": {
                "runtimeClassName": self.runtime_class,
                "restartPolicy": "Never",
                "automountServiceAccountToken": False,
                "enableServiceLinks": False,
                "activeDeadlineSeconds": self.lifetime_seconds,
                "terminationGracePeriodSeconds": 0,
                "securityContext": {
                    "runAsNonRoot": True,
                    "runAsUser": 65534,
                    "runAsGroup": 65534,
                    "fsGroup": 65534,
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
                "containers": [
                    {
                        "name": "python",
                        "image": self.image,
                        "imagePullPolicy": "IfNotPresent",
                        "command": ["sleep", "infinity"],
                        "workingDir": "/tmp",
                        "securityContext": {
                            "allowPrivilegeEscalation": False,
                            "readOnlyRootFilesystem": True,
                            "capabilities": {"drop": ["ALL"]},
                        },
                        "resources": {
                            "requests": {"cpu": "100m", "memory": "128Mi"},
                            "limits": {"cpu": "1", "memory": "256Mi"},
                        },
                        "volumeMounts": [{"name": "tmp", "mountPath": "/tmp"}],
                    }
                ],
                "volumes": [{"name": "tmp", "emptyDir": {"medium": "Memory", "sizeLimit": "16Mi"}}],
            },
        }

    async def _command(self, *arguments: str, payload: bytes = b"") -> None:
        async with _process((*self.kubectl, *arguments), limit=16384) as process:
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(payload)
            await process.stdin.drain()
            process.stdin.close()
            output = bytearray()
            while chunk := await process.stdout.read(4096):
                output.extend(chunk[: 16384 - len(output)])
            if await process.wait():
                raise RuntimeError(f"kubectl {arguments[0]} failed: {output.decode(errors='replace')}")

    async def _delete_pod(self) -> None:
        async with asyncio.timeout(self.cleanup_timeout):
            await self._command(
                "delete",
                "pod",
                self.pod_name,
                "--ignore-not-found",
                "--wait=true",
                "--grace-period=0",
                f"--timeout={self.cleanup_timeout}s",
            )

    async def __aenter__(self) -> Self:
        async with self._lock:
            if self._state != "new":
                raise RuntimeError("PythonSession can only be entered once")
            self._state = "starting"
            self._resources.push_async_callback(self._delete_pod)
            try:
                async with asyncio.timeout(self.startup_timeout):
                    source = await asyncio.to_thread(Path(__file__).with_name("_python_worker.py").read_text)
                    await self._command("create", "-f", "-", payload=json.dumps(self._manifest()).encode())
                    await self._command(
                        "wait", "--for=condition=Ready", f"pod/{self.pod_name}", f"--timeout={self.startup_timeout}s"
                    )
                    self._process = await self._resources.enter_async_context(
                        _process(
                            (
                                *self.kubectl,
                                "exec",
                                "-i",
                                self.pod_name,
                                "--container=python",
                                "--",
                                "python",
                                "-I",
                                "-B",
                                "-u",
                                "-c",
                                source,
                                str(self.max_output_bytes),
                            ),
                            limit=self.max_output_bytes * 6 + 4096,
                        )
                    )
                    assert self._process.stdout is not None
                    ready = await self._process.stdout.readline()
                    if ready != b'{"ready": true}\n':
                        raise RuntimeError(f"Python sandbox did not become ready: {ready[:1024]!r}")
                self._state = "ready"
                return self
            except BaseException:
                await self._close()
                raise

    async def execute(self, code: str, *, timeout: float | None = None) -> PythonResult:
        """Execute statements in the session namespace; print values to return output.

        A call timeout starts after acquiring the session's execution lock.
        Ordinary Python exceptions preserve state, including assignments before the error.
        """
        if len(code.encode()) > 16384 or "\0" in code:
            raise ValueError("code must be at most 16384 UTF-8 bytes and contain no NUL characters")
        deadline = self.execution_timeout if timeout is None else timeout
        if not 0 < deadline <= 3600:
            raise ValueError("timeout must be in (0, 3600] seconds")
        async with self._lock:
            if self._state != "ready":
                raise RuntimeError("PythonSession is not active")
            assert self._process is not None and self._process.stdin is not None and self._process.stdout is not None
            try:
                async with asyncio.timeout(deadline):
                    self._request_id += 1
                    request = json.dumps({"id": self._request_id, "code": code}).encode() + b"\n"
                    self._process.stdin.write(request)
                    await self._process.stdin.drain()
                    line = await self._process.stdout.readline()
                    if not line:
                        raise RuntimeError("Python sandbox exited without a response")
                    response = _Response.model_validate_json(line)
                    if response.id != self._request_id or len(response.output.encode()) > self.max_output_bytes * 3:
                        raise RuntimeError("Invalid Python sandbox response")
                    return PythonResult(
                        output=response.output, exit_code=response.exit_code, output_truncated=response.output_truncated
                    )
            except BaseException:
                await self._close()
                raise

    async def _close(self) -> None:
        self._state = "closed"
        # Keep cleanup alive if the owner is cancelled while awaiting it.
        task = asyncio.create_task(self._resources.aclose())
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    async def aclose(self) -> None:
        async with self._lock:
            await self._close()

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        await self.aclose()
