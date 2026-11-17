"""Register the articulated assets omitted from the pinned VisFly dataset."""

import json
import sys
from pathlib import Path


def prepare_assets(dataset: Path) -> None:
    config_path = dataset / "visfly-beta.scene_dataset_config.json"
    config = json.loads(config_path.read_text())
    config["articulated_objects"] = {
        "paths": {".urdf": sorted({path.parent.relative_to(dataset).as_posix() for path in dataset.rglob("*.urdf")})}
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    prepare_assets(Path(sys.argv[1]))
