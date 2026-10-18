"""Generate and score BFCL results against the Pod's vLLM sidecar."""

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


def wait_success(child: subprocess.Popen, port: int | None = None) -> None:
    healthy_at = time.monotonic()
    while True:
        try:
            code = child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if port is not None:
                try:
                    with urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                        healthy_at = time.monotonic()
                except (URLError, TimeoutError, ConnectionError) as error:
                    if time.monotonic() - healthy_at >= 30:
                        raise RuntimeError(
                            "vLLM unavailable for 30s during generation; inspect the vllm container logs"
                        ) from error
            continue
        if code != 0:
            raise subprocess.CalledProcessError(returncode=code, cmd=child.args)
        return


def run_job(run_dir: Path, port: int, smoke: bool) -> None:
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    command = [sys.executable, "-m", "rl2.bfcl"]
    options = ["--run-dir", str(run_dir), "--port", str(port), *(["--smoke"] if smoke else [])]
    with (run_dir / ".job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with process([*command, "generate", *options], run_dir / "generate.log", env) as generation:
            wait_success(generation, port)
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
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be 1..65535")
    previous = {sig: signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        run_job(args.run_dir, args.port, args.smoke)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"BFCL job failed: {error}; logs: {args.run_dir}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
