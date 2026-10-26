"""Actor-owned CodeGym manager and worker processes."""

import os
import signal
import subprocess
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path
from typing import Annotated

import httpx
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class CodeGymServerConfig:
    root: str
    workers: Annotated[int, Field(ge=1)] = 40
    startup_timeout: Annotated[float, Field(gt=0)] = 120


def stop_server(process: subprocess.Popen[bytes]) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait()
    finally:
        # The manager may exit before its workers or step subprocesses do.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def wait_for_codegym(process: subprocess.Popen[bytes], manager_url: str, workers: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    ready: set[int] = set()
    with httpx.Client(timeout=1, trust_env=False) as client:
        while time.monotonic() < deadline:
            if (returncode := process.poll()) is not None:
                raise RuntimeError(f"CodeGym exited with code {returncode} before becoming ready; see server logs")
            try:
                response = client.get(f"{manager_url}/health")
                response.raise_for_status()
                status = response.json()
                if status["pid"] != process.pid:
                    raise RuntimeError(f"CodeGym manager address is already in use: {manager_url}")
                for port in status["ports"]:
                    if time.monotonic() >= deadline:
                        break
                    if port not in ready:
                        worker_url = httpx.URL(manager_url).copy_with(port=port, path="/openapi.json")
                        response = client.get(worker_url)
                        response.raise_for_status()
                        ready.add(port)
                if len(ready) == workers:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
    raise TimeoutError(f"CodeGym did not become ready within {timeout}s ({len(ready)}/{workers} workers ready)")


@contextmanager
def serve_codegym(config: CodeGymServerConfig, manager_url: str) -> Iterator[None]:
    root = Path(config.root).resolve()
    python = root / ".venv/bin/python"
    if not python.is_file() or not (root / "env_instance_manager.py").is_file():
        raise FileNotFoundError(f"Prepare CodeGym first: uv run python -m rl2.codegym_setup --root {root}")
    launcher = Path(__file__).with_name("codegym_launcher.py")
    env = {
        **os.environ,
        "PATH": f"{python.parent}{os.pathsep}{os.environ.get('PATH', '')}",
        "PYTHONPATH": str(root),
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    with ExitStack() as resources:
        process = resources.enter_context(
            subprocess.Popen(
                [
                    str(python),
                    str(launcher),
                    "--port",
                    str(httpx.URL(manager_url).port),
                    "--workers",
                    str(config.workers),
                ],
                cwd=root,
                env=env,
                start_new_session=True,
            )
        )
        resources.callback(stop_server, process)
        wait_for_codegym(process, manager_url=manager_url, workers=config.workers, timeout=config.startup_timeout)
        yield
