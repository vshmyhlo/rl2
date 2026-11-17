"""VisFly dataset preparation preserves scenes and registers bundled URDFs."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

assets = SimpleNamespace(
    **runpy.run_path(str(Path(__file__).resolve().parents[1] / "infra/k3s/training/visfly/prepare_assets.py"))
)


def test_register_articulated_assets(tmp_path: Path) -> None:
    config = {"stages": {"paths": {".json": ["configs/stages"]}}, "scene_instances": {"paths": {".json": ["scenes"]}}}
    config_path = tmp_path / "visfly-beta.scene_dataset_config.json"
    config_path.write_text(json.dumps(config))
    for name in ("fridge/fridge", "fridge/fridge_dynamic", "doors/door2"):
        path = tmp_path / "urdf" / f"{name}.urdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<robot/>")
    assets.prepare_assets(tmp_path)
    expected = {**config, "articulated_objects": {"paths": {".urdf": ["urdf/doors", "urdf/fridge"]}}}
    assert json.loads(config_path.read_text()) == expected
    assets.prepare_assets(tmp_path)
    assert json.loads(config_path.read_text()) == expected
