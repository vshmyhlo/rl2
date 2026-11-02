"""Prepare a pinned CodeGym server and English environments without a Git checkout."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal
from urllib.request import urlopen

from datasets import load_dataset

from rl2.lm_envs.codegym.tools import extract_tools

SERVER_REVISION = "e85a439761ae2df813d5997a81bbd9f9e2ed27b7"
DATASET_REVISION = "85286359a342f7a288aea74273772b69b9b784c2"
ENV_REQUIREMENTS = ["gymnasium==1.3.0", "numpy==2.3.5"]


def prepare_codegym(root: Path, *, worker_host: Literal["127.0.0.1", "0.0.0.0"] = "127.0.0.1") -> None:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".ready"
    version = f"tools-v1\n{worker_host}\n{SERVER_REVISION}\n{DATASET_REVISION}\n{' '.join(ENV_REQUIREMENTS)}\n"
    files = {
        "env_instance_manager.py": "online_server/online_server/env_instance_manager.py",
        "env_server.py": "online_server/online_server/env_server.py",
        "server_config.py": "online_server/online_server/server_config.py",
        "requirements.txt": "online_server/requirements.txt",
        "LICENSE": "LICENSE",
    }
    destination = root / "envs/codegym_v1"
    schemas = root / "tools"
    inventory = "".join(f"{path.name}\n" for path in sorted(destination.glob("*.py")) if path.is_file())
    if (
        marker.is_file()
        and inventory
        and marker.read_text() == version + inventory
        and all((root / name).is_file() for name in [".venv/bin/python", *files])
        and all((schemas / f"{Path(name).stem}.json").is_file() for name in inventory.splitlines())
    ):
        return
    marker.unlink(missing_ok=True)
    base_url = f"https://raw.githubusercontent.com/StigLidu/CodeGym/{SERVER_REVISION}"
    for name, source in files.items():
        with urlopen(f"{base_url}/{source}", timeout=60) as response, (root / name).open("wb") as output:
            shutil.copyfileobj(response, output)
    # Bind workers to loopback locally or the pod network for cluster clients.
    manager = root / "env_instance_manager.py"
    manager.write_text(manager.read_text().replace('"--host", "::"', f'"--host", "{worker_host}"'))
    subprocess.run(["uv", "venv", "--allow-existing", "--python", sys.executable, str(root / ".venv")], check=True)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(root / ".venv/bin/python"),
            "-r",
            str(root / "requirements.txt"),
            *ENV_REQUIREMENTS,
        ],
        check=True,
    )
    environments = load_dataset("VanishD/CodeGym", "envs_en", revision=DATASET_REVISION, split="train")
    destination.mkdir(parents=True, exist_ok=True)
    schemas.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for row in environments:
        name = f"{row['source']}__{row['env_name']}.py"
        if Path(name).name != name or "\\" in name:
            raise ValueError(f"Invalid CodeGym environment filename: {name!r}")
        code = row["env_code"]
        if name == "Codeforces_5537_I__MinPathCostEnv.py":
            # The pinned dataset has two invalid list-comprehension f-strings in this environment.
            code = code.replace("[{self.dp[i][0] for i in range(self.n)}]", "{[self.dp[i][0] for i in range(self.n)]}")
        compile(code, name, "exec")
        tools = extract_tools(code, row["env_name"])
        (destination / name).write_text(code, encoding="utf-8")
        (schemas / f"{Path(name).stem}.json").write_text(json.dumps({"tools": tools}), encoding="utf-8")
        names.append(name)
    if not len(environments):
        raise ValueError("CodeGym environment dataset is empty")
    marker.write_text(version + "".join(f"{name}\n" for name in sorted(names)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/runs/codegym"))
    parser.add_argument("--worker-host", choices=["127.0.0.1", "0.0.0.0"], default="127.0.0.1")
    args = parser.parse_args()
    prepare_codegym(args.root, worker_host=args.worker_host)
    print(f"CodeGym ready at {args.root.resolve()}")


if __name__ == "__main__":
    main()
