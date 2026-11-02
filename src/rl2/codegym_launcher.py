"""Run with the prepared CodeGym interpreter and its root on PYTHONPATH."""

import argparse
import os
import socket

import uvicorn
from env_instance_manager import EnvInstanceManager, app
from fastapi import HTTPException


@app.get("/health")
def health() -> dict[str, int | list[int]]:
    return {"pid": os.getpid(), "ports": [instance.port for instance in EnvInstanceManager._all_instances.values()]}


@app.get("/ready")
def ready() -> dict[str, int]:
    instances = list(EnvInstanceManager._all_instances.values())
    if len(instances) != app.state.workers:
        raise HTTPException(status_code=503, detail="Workers are starting")
    for instance in instances:
        try:
            # Upstream steps block the worker event loop; probe its listener without interrupting busy episodes.
            with socket.create_connection(("127.0.0.1", instance.port), timeout=1):
                pass
        except OSError as error:
            raise HTTPException(status_code=503, detail="Worker unavailable") from error
    return {"workers": len(instances)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    app.state.workers = args.workers
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
