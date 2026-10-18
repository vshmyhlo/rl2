"""Run download, serving, generation, and scoring in one GPU Job container."""

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from types import FrameType
from urllib.error import URLError
from urllib.request import urlopen


@contextmanager
def process(command: list[str], log_path: Path, env: dict[str, str]) -> Iterator[subprocess.Popen]:
    print(f"Starting {log_path.stem}; log: {log_path}", flush=True)
    with (
        log_path.open("a") as log,
        subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True) as child,
    ):
        try:
            yield child
        finally:
            # Include engine workers and BFCL's child process, even if their parent exited.
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def wait_ready(server: subprocess.Popen, port: int, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {server.returncode}; see serve.log")
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return
        except (URLError, TimeoutError, ConnectionError):
            time.sleep(0.5)
    raise TimeoutError(f"vLLM was not ready within {timeout}s; see serve.log")


def wait_success(child: subprocess.Popen, server: subprocess.Popen | None = None) -> None:
    while True:
        if server is not None and server.poll() is not None:
            raise RuntimeError(f"vLLM exited during generation with code {server.returncode}; see serve.log")
        try:
            code = child.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            continue
        if code != 0:
            raise subprocess.CalledProcessError(returncode=code, cmd=child.args)
        return


def run_job(run_dir: Path, port: int, smoke: bool, startup_timeout: int) -> None:
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    env.setdefault("HF_HOME", str(run_dir.parent / ".huggingface"))
    command = [sys.executable, "-m", "rl2.bfcl"]
    options = ["--run-dir", str(run_dir), "--port", str(port), *(["--smoke"] if smoke else [])]
    with (run_dir / ".job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with process([*command, "download"], run_dir / "download.log", env) as download:
            wait_success(download)
        with process([*command, "serve", "--port", str(port)], run_dir / "serve.log", env) as server:
            wait_ready(server, port, startup_timeout)
            with process([*command, "generate", *options], run_dir / "generate.log", env) as generation:
                wait_success(generation, server)
        print("vLLM stopped; starting scoring", flush=True)
        with process([*command, "evaluate", *options], run_dir / "evaluate.log", env) as evaluation:
            wait_success(evaluation)
    print(f"BFCL evaluation complete: {run_dir / 'score'}", flush=True)


def terminate(signum: int, frame: FrameType | None) -> None:
    raise SystemExit(128 + signum)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="Use /runs/bfcl/<run-name> for persistent Job output"
    )
    parser.add_argument("--port", type=int, default=1053)
    parser.add_argument("--smoke", action="store_true", help="Run only four fixed pipeline checks")
    parser.add_argument("--startup-timeout", type=int, default=900, help="Seconds to wait for vLLM readiness")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.startup_timeout <= 0:
        parser.error("--port must be 1..65535 and --startup-timeout must be positive")
    previous = {sig: signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        run_job(args.run_dir, args.port, args.smoke, args.startup_timeout)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"BFCL job failed: {error}; logs: {args.run_dir}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
