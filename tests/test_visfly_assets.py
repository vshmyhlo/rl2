"""VisFly dataset preparation preserves scenes and registers bundled assets."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

assets = SimpleNamespace(
    **runpy.run_path(str(Path(__file__).resolve().parents[1] / "infra/k3s/training/visfly/prepare_assets.py"))
)


def test_register_bundled_assets(tmp_path: Path) -> None:
    config = {
        "stages": {"paths": {".json": ["configs/stages"]}},
        "scene_instances": {"paths": {".json": ["scenes"]}},
        "objects": {"paths": {".json": ["configs/objects/pillar"]}},
    }
    config_path = tmp_path / "visfly-beta.scene_dataset_config.json"
    config_path.write_text(json.dumps(config))
    pillar_dir = tmp_path / "configs/objects/pillar"
    pillar_dir.mkdir(parents=True)
    pillar_config = {
        "render_asset": "../../../self_define_objects/square_pillar.glb",
        "collision_asset": "../../../self_define_objects/convex/square_pillar.glb",
        "COM": [0, 0, 0],
        "join_collision_meshes": False,
        "semantic_id": 3,
    }
    square_path = pillar_dir / "square_pillar.object_config.json"
    square_path.write_text(json.dumps(pillar_config))
    for name in (
        "round_pillar.glb",
        "convex/round_pillar.glb",
        "square_pillar.glb",
        "convex/square_pillar.glb",
        "wall.glb",
    ):
        mesh_path = tmp_path / "self_define_objects" / name
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        mesh_path.touch()
    for name in ("fridge/fridge", "fridge/fridge_dynamic", "doors/door2"):
        path = tmp_path / "urdf" / f"{name}.urdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<robot/>")
    assets.prepare_assets(tmp_path)
    expected = {**config, "articulated_objects": {"paths": {".urdf": ["urdf/doors", "urdf/fridge"]}}}
    assert json.loads(config_path.read_text()) == expected
    assets.prepare_assets(tmp_path)
    assert json.loads(config_path.read_text()) == expected
    expected_assets = {
        "round_pillar": ("round_pillar.glb", "convex/round_pillar.glb"),
        "suqare_pillar": ("square_pillar.glb", "convex/square_pillar.glb"),
        "debug_wall": ("wall.glb", "wall.glb"),
    }
    for handle, (render, collision) in expected_assets.items():
        template = json.loads((pillar_dir / f"{handle}.object_config.json").read_text())
        assert template == {
            **pillar_config,
            "render_asset": f"../../../self_define_objects/{render}",
            "collision_asset": f"../../../self_define_objects/{collision}",
        }
        for key in ("render_asset", "collision_asset"):
            assert (pillar_dir / template[key]).is_file()
    assert json.loads(square_path.read_text()) == pillar_config
