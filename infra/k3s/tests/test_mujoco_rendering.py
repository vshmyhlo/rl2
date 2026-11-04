"""Opt-in NVIDIA EGL check using RL2_K3S_TESTS=1 and RL2_TRAINING_IMAGE."""

import json
import os
import runpy
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RL2_K3S_TESTS") != "1" or not os.environ.get("RL2_TRAINING_IMAGE"),
    reason="Requires local k3s and an existing training image",
)
submit = SimpleNamespace(**runpy.run_path(str(Path(__file__).resolve().parents[3] / "submit.py")))

# Import the trainer first: TensorFlow's exported LLVM symbols can crash Mesa.
RENDER_CHECK = """
from rl2 import ppo
import mujoco
import numpy as np
from OpenGL import GL

model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom type="sphere" size="0.1" rgba="1 0 0 1"/></worldbody></mujoco>'
)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
for _ in range(2):
    with mujoco.Renderer(model, height=24, width=32) as renderer:
        assert GL.glGetString(GL.GL_VENDOR) == b'NVIDIA Corporation'
        for _ in range(2):
            renderer.update_scene(data)
            frame = renderer.render()  # (24, 32, 3)
            assert frame.shape == (24, 32, 3) and frame.dtype == np.uint8
            assert frame.max() > frame.min()
print('nvidia-egl-render-ok', flush=True)
"""


def kubectl(*args: str) -> str:
    return subprocess.run(
        ["sudo", "-n", "k3s", "kubectl", "-n", "training", *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout


def test_training_image_renders_with_nvidia_egl() -> None:
    name = f"train-render-check-{uuid4().hex[:12]}"
    args = submit.parse_args(
        ["--cpus", "2", "--memory-gib", "8", "--", "uv", "run", "python", "-X", "faulthandler", "-c", RENDER_CHECK]
    )
    manifest: dict[str, Any] = submit.job_manifest(args=args, name=name, image=os.environ["RL2_TRAINING_IMAGE"])
    manifest["spec"]["activeDeadlineSeconds"] = 120
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    # Reserve a GPU for EGL, but avoid CUDA allocations just to import the trainer.
    next(env for env in container["env"] if env["name"] == "JAX_PLATFORMS")["value"] = "cpu"
    try:
        subprocess.run(
            ["sudo", "-n", "/usr/bin/python3", "/usr/local/lib/rl2/image_pins.py", "submit"],
            input=json.dumps(manifest),
            text=True,
            check=True,
            timeout=60,
        )
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            job = json.loads(kubectl("get", "job", name, "-o", "json"))
            conditions = {c["type"] for c in job.get("status", {}).get("conditions", []) if c["status"] == "True"}
            if conditions & {"Complete", "Failed"}:
                logs = kubectl("logs", f"job/{name}")
                assert "Complete" in conditions, logs
                assert "nvidia-egl-render-ok" in logs
                return
            time.sleep(2)
        pytest.fail(f"Rendering check timed out: {job}")
    finally:
        kubectl("delete", "job", name, "--ignore-not-found", "--wait=true")
