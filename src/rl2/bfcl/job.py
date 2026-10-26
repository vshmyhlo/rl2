"""Run BFCL with a local Ray server actor and generation/scoring tasks."""

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from urllib.error import URLError
from urllib.request import urlopen

import ray
from ray.actor import ActorHandle
from ray.exceptions import GetTimeoutError, RayError


@dataclass(frozen=True)
class Run:
    directory: Path
    port: int
    smoke: bool

    def command(self, action: str) -> list[str]:
        return [
            sys.executable,
            "-m",
            "rl2.bfcl",
            action,
            "--run-dir",
            str(self.directory),
            "--port",
            str(self.port),
            *(["--smoke"] if self.smoke else []),
        ]


@contextmanager
def process(command: list[str], log_path: Path) -> Generator[subprocess.Popen[str]]:
    print(f"Starting {log_path.stem}; log: {log_path}", flush=True)
    with (
        log_path.open("a", encoding="utf-8", buffering=1) as log,
        ThreadPoolExecutor(max_workers=1) as output,
        subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,
        ) as child,
    ):

        def stream() -> None:
            assert child.stdout is not None
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(f"[{log_path.stem}] {line.removesuffix('\n')}", flush=True)

        # Drain concurrently so a long-running server cannot fill its output pipe.
        streaming = output.submit(stream)
        try:
            yield child
        finally:
            # Stop the entire subprocess tree, including vLLM workers.
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            child.wait()
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            streaming.result()


class VLLMServer:
    def __init__(self, run: Run) -> None:
        self.resources = ExitStack()
        self.port = run.port
        self.child = self.resources.enter_context(process(run.command("serve"), run.directory / "serve.log"))

    def check(self) -> None:
        if self.child.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {self.child.returncode}; see serve.log")

    def ready(self) -> bool:
        self.check()
        try:
            with urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2):
                return True
        except (URLError, TimeoutError, ConnectionError):
            return False

    def stop(self) -> None:
        self.resources.close()


ServerActor = ray.remote(num_cpus=0, num_gpus=1, max_restarts=0, max_task_retries=0)(VLLMServer)


def run_stage(action: str, run: Run) -> None:
    with process(run.command(action), run.directory / f"{action}.log") as child:
        while True:
            try:
                code = child.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                continue
            if code != 0:
                raise RuntimeError(f"{action} exited with code {code}; see {action}.log")
            return


@ray.remote(num_cpus=1, max_retries=0)
def download(run: Run) -> None:
    run_stage("download", run)


@ray.remote(num_cpus=1, max_retries=0)
def generate(run: Run) -> None:
    run_stage("generate", run)


@ray.remote(num_cpus=1, max_retries=0)
def score(run: Run) -> None:
    run_stage("evaluate", run)


def wait_task(task: ray.ObjectRef, server: ActorHandle | None = None) -> None:
    try:
        while True:
            try:
                ray.get(task, timeout=1)
                return
            except GetTimeoutError:
                if server is not None:
                    ray.get(server.check.remote(), timeout=5)
    except BaseException:
        # Cooperative cancellation lets the task close its subprocess group.
        ray.cancel(task, force=False)
        with suppress(RayError):
            ray.get(task)
        raise


@contextmanager
def serving(run: Run, startup_timeout: int) -> Generator[ActorHandle]:
    server = ServerActor.remote(run)
    try:
        deadline = time.monotonic() + startup_timeout
        ready = server.ready.remote()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"vLLM was not ready within {startup_timeout}s; see serve.log")
            try:
                if ray.get(ready, timeout=min(5, remaining)):
                    break
            except GetTimeoutError:
                continue
            time.sleep(0.5)
            ready = server.ready.remote()
        yield server
    finally:
        try:
            ray.get(server.stop.remote())
        finally:
            ray.kill(server)


def run_job(run_dir: Path, port: int, smoke: bool, startup_timeout: int) -> None:
    run = Run(directory=run_dir.resolve(), port=port, smoke=smoke)
    run.directory.mkdir(parents=True, exist_ok=True)
    print(f"Starting BFCL job: {run.directory}", flush=True)
    with (run.directory / ".job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ray.init(
            address="local",
            num_cpus=2,
            include_dashboard=False,
            log_to_driver=True,
            object_store_memory=128 * 1024**2,
            runtime_env={
                "env_vars": {
                    "HF_HOME": os.environ.get("HF_HOME", str(run.directory.parent / ".huggingface")),
                    "PYTHONUNBUFFERED": "1",
                }
            },
        ):
            if ray.cluster_resources().get("GPU", 0) < 1:
                raise RuntimeError("BFCL serving requires one visible GPU")
            wait_task(download.remote(run))
            with serving(run, startup_timeout) as server:
                wait_task(generate.remote(run), server)
            print("vLLM stopped; starting scoring", flush=True)
            wait_task(score.remote(run))
    print(f"BFCL evaluation complete: {run.directory / 'score'}", flush=True)


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
    except (OSError, RuntimeError, RayError) as error:
        print(f"BFCL job failed: {error}; logs: {args.run_dir}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
