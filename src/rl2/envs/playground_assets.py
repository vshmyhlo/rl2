"""Download Panda assets at Playground's pinned Menagerie revision over HTTPS."""

from pathlib import Path
from urllib.request import urlopen
from xml.etree import ElementTree

from mujoco_playground._src import mjx_env


def download_panda_assets() -> None:
    directory = Path(mjx_env.MENAGERIE_PATH) / "franka_emika_panda"
    base_url = (
        "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/"
        f"{mjx_env.MENAGERIE_COMMIT_SHA}/franka_emika_panda"
    )

    def download(relative_path: str) -> Path:
        path = directory / relative_path
        if not path.exists():
            print(f"Downloading Panda asset: {relative_path}", flush=True)
            with urlopen(f"{base_url}/{relative_path}", timeout=60) as response:
                data = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return path

    model = ElementTree.parse(download("mjx_panda.xml"))
    download("LICENSE")
    for mesh in model.findall("asset/mesh"):
        download(f"assets/{Path(mesh.attrib['file']).name}")


if __name__ == "__main__":
    download_panda_assets()
