"""Register object templates omitted from the pinned VisFly dataset."""

import json
import sys
from pathlib import Path


def prepare_assets(dataset: Path) -> None:
    pillar_dir = dataset / "configs/objects/pillar"
    pillar = json.loads((pillar_dir / "square_pillar.object_config.json").read_text())
    # These scene handles have bundled meshes but no object templates.
    for handle, mesh in {
        "round_pillar": "round_pillar",
        "suqare_pillar": "square_pillar",
        "debug_wall": "wall",
    }.items():
        collision = f"convex/{mesh}" if mesh != "wall" else mesh
        template = {
            **pillar,
            "render_asset": f"../../../self_define_objects/{mesh}.glb",
            "collision_asset": f"../../../self_define_objects/{collision}.glb",
        }
        (pillar_dir / f"{handle}.object_config.json").write_text(json.dumps(template, indent=2) + "\n")
    config_path = dataset / "visfly-beta.scene_dataset_config.json"
    config = json.loads(config_path.read_text())
    config["articulated_objects"] = {
        "paths": {".urdf": sorted({path.parent.relative_to(dataset).as_posix() for path in dataset.rglob("*.urdf")})}
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    prepare_assets(Path(sys.argv[1]))
