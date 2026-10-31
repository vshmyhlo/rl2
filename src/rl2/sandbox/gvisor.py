"""Stateless Python execution through Docker's gVisor (runsc) runtime.

Requires a trusted Docker daemon configured with runsc and a locally available
Python image: https://gvisor.dev/docs/user_guide/quick_start/docker/
Pre-pull python:3.12-slim, or supply a trusted image pinned by digest. Images must
contain `python`; do not use training images containing code, data, or secrets.

Usage: result = await run_python("print(6 * 7)")
The execution deadline includes container startup; creation and removal each
have a separate 30-second deadline. TimeoutError and cancellation propagate
after attempting removal. Daemon failure can prevent cleanup; this runner does
not provide crash recovery or a global concurrency limit.
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class PythonResult:
    output: str
    exit_code: int
    output_truncated: bool


async def _docker(*arguments: str, max_output_bytes: int = 16384) -> PythonResult:
    process = await asyncio.create_subprocess_exec(
        "docker",
        *arguments,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output = bytearray()
    truncated = False
    try:
        assert process.stdout is not None
        while chunk := await process.stdout.read(4096):
            remaining = max_output_bytes - len(output)
            output.extend(chunk[:remaining])
            truncated |= len(chunk) > remaining
        return PythonResult(
            output=output.decode("utf-8", errors="replace"),
            exit_code=await process.wait(),
            output_truncated=truncated,
        )
    finally:
        if process.returncode is None:
            process.kill()
        # Drain after killing so a full pipe cannot prevent subprocess cleanup.
        if process.stdout is not None:
            while await process.stdout.read(4096):
                pass
        await process.wait()


@asynccontextmanager
async def _container(code: str, image: str) -> AsyncGenerator[str]:
    name = f"rl2-python-{uuid4().hex}"
    created = False
    try:
        async with asyncio.timeout(30):
            result = await _docker(
                "create",
                f"--name={name}",
                "--runtime=runsc",
                "--network=none",
                "--ipc=none",
                "--read-only",
                "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=16m",
                "--user=65534:65534",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--pids-limit=32",
                "--cpus=1",
                "--memory=256m",
                "--memory-swap=256m",
                "--ulimit=nofile=64:64",
                "--ulimit=core=0:0",
                "--log-driver=none",
                "--no-healthcheck",
                "--workdir=/tmp",
                "--entrypoint=python",
                "--pull=never",
                image,
                "-I",
                "-B",
                "-u",
                "-c",
                code,
            )
        if result.exit_code:
            raise RuntimeError(f"Could not create gVisor container: {result.output}")
        created = True
        yield name
    finally:
        async with asyncio.timeout(30):
            removed = await _docker("rm", "--force", "--volumes", name)
        if removed.exit_code and created:
            raise RuntimeError(f"Could not remove gVisor container {name}: {removed.output}")


async def run_python(
    code: str,
    *,
    image: str = "python:3.12-slim",
    timeout: float = 5.0,
    max_output_bytes: int = 16384,
) -> PythonResult:
    """Run code with no external network or host mounts; return combined stdout/stderr.

    Only code is model-controlled; image and limits are trusted caller settings.
    Each call gets a fresh container. Output beyond the byte limit is discarded.
    Python exceptions return a nonzero exit code; execution timeouts raise TimeoutError.
    """
    if len(code.encode("utf-8")) > 16384 or "\0" in code:
        raise ValueError("code must be at most 16384 UTF-8 bytes and contain no NUL characters")
    if not image or image.startswith("-") or "\0" in image:
        raise ValueError("image must be a nonempty Docker image reference")
    if not 0 < timeout <= 300:
        raise ValueError("timeout must be in (0, 300] seconds")
    if max_output_bytes <= 0:
        raise ValueError("max_output_bytes must be positive")
    async with _container(code=code, image=image) as name, asyncio.timeout(timeout):
        return await _docker("start", "--attach", name, max_output_bytes=max_output_bytes)
