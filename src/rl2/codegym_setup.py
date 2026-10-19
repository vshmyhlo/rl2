"""Prepare a pinned CodeGym server and English environments without a Git checkout."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

from datasets import load_dataset

SERVER_REVISION = "e85a439761ae2df813d5997a81bbd9f9e2ed27b7"
DATASET_REVISION = "85286359a342f7a288aea74273772b69b9b784c2"
ENV_REQUIREMENTS = ["gymnasium==1.3.0", "numpy==2.3.5"]


def prepare_codegym(root: Path) -> None:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".ready"
    version = f"ipv4-v1\n{SERVER_REVISION}\n{DATASET_REVISION}\n{' '.join(ENV_REQUIREMENTS)}\n"
    if marker.is_file() and marker.read_text() == version and (root / ".venv/bin/python").is_file():
        return
    marker.unlink(missing_ok=True)
    base_url = f"https://raw.githubusercontent.com/StigLidu/CodeGym/{SERVER_REVISION}"
    files = {
        "env_instance_manager.py": "online_server/online_server/env_instance_manager.py",
        "env_server.py": "online_server/online_server/env_server.py",
        "server_config.py": "online_server/online_server/server_config.py",
        "requirements.txt": "online_server/requirements.txt",
        "LICENSE": "LICENSE",
    }
    for name, source in files.items():
        with urlopen(f"{base_url}/{source}", timeout=60) as response, (root / name).open("wb") as output:
            shutil.copyfileobj(response, output)
    # Upstream workers bind IPv6-only; our local manager and client use IPv4.
    manager = root / "env_instance_manager.py"
    manager.write_text(manager.read_text().replace('"--host", "::"', '"--host", "127.0.0.1"'))
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
    destination = root / "envs/codegym_v1"
    destination.mkdir(parents=True, exist_ok=True)
    for row in environments:
        name = f"{row['source']}__{row['env_name']}.py"
        if Path(name).name != name or "\\" in name:
            raise ValueError(f"Invalid CodeGym environment filename: {name!r}")
        (destination / name).write_text(row["env_code"], encoding="utf-8")
    if not len(environments):
        raise ValueError("CodeGym environment dataset is empty")
    marker.write_text(version)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/runs/codegym"))
    args = parser.parse_args()
    prepare_codegym(args.root)
    print(f"CodeGym ready at {args.root.resolve()}")


if __name__ == "__main__":
    main()
