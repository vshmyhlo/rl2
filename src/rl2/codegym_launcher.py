"""Run with the prepared CodeGym interpreter and its root on PYTHONPATH."""

import argparse
import os

import uvicorn
from env_instance_manager import EnvInstanceManager, app


@app.get("/health")
def health() -> dict[str, int | list[int]]:
    return {"pid": os.getpid(), "ports": [instance.port for instance in EnvInstanceManager._all_instances.values()]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    app.state.workers = args.workers
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
