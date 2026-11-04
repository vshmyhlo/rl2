"""Download Panda or Go1 assets at Playground's pinned Menagerie revision over HTTPS."""

import argparse
from pathlib import Path
from typing import Literal
from urllib.request import urlopen
from xml.etree import ElementTree

from mujoco_playground._src import mjx_env


def download_assets(robot: Literal["panda", "go1"] = "panda") -> None:
    name = "franka_emika_panda" if robot == "panda" else "unitree_go1"
    directory = Path(mjx_env.MENAGERIE_PATH) / name
    base_url = (
        f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{mjx_env.MENAGERIE_COMMIT_SHA}/{name}"
    )

    def download(relative_path: str) -> Path:
        path = directory / relative_path
        if not path.exists():
            print(f"Downloading {robot} asset: {relative_path}", flush=True)
            with urlopen(f"{base_url}/{relative_path}", timeout=60) as response:
                data = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return path

    model_path = (
        download("mjx_panda.xml")
        if robot == "panda"
        else Path(mjx_env.ROOT_PATH) / "locomotion/go1/xmls/go1_mjx_feetonly.xml"
    )
    model = ElementTree.parse(model_path)
    download("LICENSE")
    for mesh in model.findall("asset/mesh"):
        download(f"assets/{Path(mesh.attrib['file']).name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("panda", "go1"), default="panda")
    download_assets(robot=parser.parse_args().robot)
