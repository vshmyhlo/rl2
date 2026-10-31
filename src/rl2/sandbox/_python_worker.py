"""Standalone interpreter loop shipped into the sandbox through kubectl exec."""

import json
import os
import resource
import socket
import sys
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Thread
from types import ModuleType
from typing import BinaryIO


@dataclass
class Output:
    data: bytearray = field(default_factory=bytearray)
    truncated: bool = False


def collect(stream: BinaryIO, output: Output, limit: int) -> None:
    with stream:
        while chunk := stream.read(4096):
            remaining = limit - len(output.data)
            output.data.extend(chunk[:remaining])
            output.truncated |= len(chunk) > remaining


@contextmanager
def capture(limit: int) -> Generator[Output]:
    output = Output()
    read_fd, write_fd = os.pipe()
    reader = Thread(target=collect, args=(os.fdopen(read_fd, "rb", buffering=0), output, limit), daemon=True)
    reader.start()
    with os.fdopen(os.dup(1), "wb") as stdout, os.fdopen(os.dup(2), "wb") as stderr:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        os.dup2(write_fd, 1)
        os.dup2(write_fd, 2)
        os.close(write_fd)
        try:
            yield output
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout.fileno(), 1)
            os.dup2(stderr.fileno(), 2)
            reader.join()


def main() -> None:
    limit = int(sys.argv[1])
    if any(name != "lo" for _, name in socket.if_nameindex()):
        raise RuntimeError("Sandbox has external network interfaces; configure runsc with network=none")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    module = ModuleType("__main__")
    sys.modules["__main__"] = module
    namespace = module.__dict__
    with os.fdopen(os.dup(0), "r") as requests, open(os.devnull, "rb") as null:
        os.dup2(null.fileno(), 0)
        print(json.dumps({"ready": True}), flush=True)
        for line in requests:
            request = json.loads(line)
            with capture(limit) as output:
                exit_code = 0
                try:
                    exec(compile(request["code"], "<sandbox>", "exec"), namespace)  # noqa: S102 -- gVisor is the boundary.
                except BaseException:  # noqa: BLE001 -- SystemExit and KeyboardInterrupt are cell errors too.
                    exit_code = 1
                    traceback.print_exc()
            print(
                json.dumps(
                    {
                        "id": request["id"],
                        "output": output.data.decode("utf-8", errors="replace"),
                        "exit_code": exit_code,
                        "output_truncated": output.truncated,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
