"""Shared sensor ordering, fallback, and scene lifecycle without OpenGL."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.typing import NDArray

from rl2.envs.visfly.sensors import SharedRgbdRenderer


class RendererStub:
    Flags = SimpleNamespace(RGB_DEPTH=32, NONE=0)

    def __init__(self) -> None:
        self.bind_render_target = Mock()


class LegacyRenderer(RendererStub):
    Flags = SimpleNamespace(NONE=0)


def sensor(kind: str, value: int) -> SimpleNamespace:
    pixels = np.arange(6, dtype=np.float32).reshape(2, 3) + value  # (H, W)

    def read_depth(view: NDArray[np.float32]) -> None:  # (H, W)
        view[:] = pixels

    def no_noise(image: NDArray[np.float32]) -> NDArray[np.float32]:  # (H, W)
        return image  # (H, W)

    buffer = np.zeros((2, 3), dtype=np.float32)  # (H, W)
    return SimpleNamespace(
        _spec=SimpleNamespace(
            sensor_type=SimpleNamespace(name=kind),
            sensor_subtype=SimpleNamespace(name="PINHOLE"),
            gpu2gpu_transfer=False,
            noise_model="None",
            resolution=[2, 3],
            position=[0, 0, -0.2],
            orientation=[0, 0, 0],
            near=0.01,
            far=1000.0,
            hfov=90.0,
        ),
        _sensor_object=SimpleNamespace(
            node=SimpleNamespace(transformation="aligned"),
            render_target=SimpleNamespace(read_frame_depth=read_depth),
        ),
        _buffer=buffer,  # (H, W)
        view=buffer,  # (H, W)
        _noise_model=no_noise,
        draw_observation=Mock(),
        get_observation=Mock(return_value=np.full((2, 3, 4), value, dtype=np.uint8)),  # (H, W, 4)
    )


def scene(offset: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        renderer=RendererStub(),
        config=SimpleNamespace(enable_batch_renderer=False, sim_cfg=SimpleNamespace(enable_hbao=False)),
        _Simulator__sensors=[
            {"color": sensor("COLOR", offset + i), "depth": sensor("DEPTH", offset + i)} for i in range(2)
        ],
    )


def manager() -> SimpleNamespace:
    return SimpleNamespace(
        scenes=[scene(), scene(2)],
        num_agent_per_scene=2,
        get_observation=Mock(return_value=[{"fallback": np.zeros((1, 1))}]),  # leaves: (H, W)
        render=Mock(return_value="video"),
    )


@pytest.mark.parametrize("indices", [None, [3, 0, 3], []], ids=["full_batch", "ordered_subset_with_duplicate", "empty"])
def test_shared_sensor_values_and_selection(indices: list[int] | None) -> None:
    backend = manager()
    SharedRgbdRenderer(backend)
    images = backend.get_observation(indices)
    selected = list(range(4)) if indices is None else indices
    assert len(images) == len(selected)
    for image, index in zip(images, selected, strict=True):
        assert image["color"].shape == (2, 3, 4) and image["color"].dtype == np.uint8
        assert image["depth"].shape == (2, 3) and image["depth"].dtype == np.float32
        np.testing.assert_array_equal(image["color"], index)
        np.testing.assert_array_equal(image["depth"], (np.arange(6).reshape(2, 3) + index)[::-1])
    for loaded in backend.scenes:
        for suite in loaded._Simulator__sensors:
            suite["depth"].draw_observation.assert_not_called()
            suite["depth"].get_observation.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("resolution", [3, 2]),
        ("position", [0, 0, 0]),
        ("orientation", [1, 0, 0]),
        ("near", 0.1),
        ("far", 100.0),
        ("hfov", 60.0),
        ("sensor_subtype", SimpleNamespace(name="ORTHOGRAPHIC")),
        ("sensor_type", SimpleNamespace(name="COLOR")),
        ("gpu2gpu_transfer", True),
        ("noise_model", "GaussianNoiseModel"),
    ],
)
def test_incompatible_sensors_preserve_original_call(field: str, value: Any) -> None:
    backend = manager()
    original = backend.get_observation
    setattr(backend.scenes[0]._Simulator__sensors[0]["depth"]._spec, field, value)
    SharedRgbdRenderer(backend)
    assert backend.get_observation(indices=[2, 0]) is original.return_value
    original.assert_called_once_with(indices=[2, 0])
    assert all(not loaded.renderer.bind_render_target.called for loaded in backend.scenes)


@pytest.mark.parametrize("reason", ["legacy", "hbao", "batch", "rgb_only", "local_transform", "reversed_sensors"])
def test_unsupported_renderer_or_rig_falls_back(reason: str) -> None:
    backend = manager()
    original = backend.get_observation
    loaded = backend.scenes[0]
    if reason == "legacy":
        loaded.renderer = LegacyRenderer()
    elif reason == "hbao":
        loaded.config.sim_cfg.enable_hbao = True
    elif reason == "batch":
        loaded.config.enable_batch_renderer = True
        loaded.renderer = None
    elif reason == "rgb_only":
        del loaded._Simulator__sensors[0]["depth"]
    elif reason == "local_transform":
        loaded._Simulator__sensors[0]["depth"]._sensor_object.node.transformation = "different"
    else:
        loaded._Simulator__sensors[0] = dict(reversed(loaded._Simulator__sensors[0].items()))
    SharedRgbdRenderer(backend)
    assert backend.get_observation() is original.return_value


def test_reload_rebinds_only_live_scenes_and_video_restores_fallback() -> None:
    backend = manager()
    original = backend.get_observation
    renderer = SharedRgbdRenderer(backend)
    backend.get_observation()
    old = backend.scenes[0]
    old.renderer.bind_render_target.reset_mock()
    backend.scenes[0] = scene(8)
    images = backend.get_observation()
    np.testing.assert_array_equal(images[0]["color"], 8)
    old.renderer.bind_render_target.assert_not_called()
    assert backend.scenes[0].renderer.bind_render_target.call_count == 2
    assert backend.render() == "video"
    assert backend.get_observation() is original.return_value
    assert renderer._groups == [] and renderer._scenes == ()
    for loaded in backend.scenes:
        assert loaded.renderer.bind_render_target.call_args.args[-1] == RendererStub.Flags.NONE


def test_reload_to_incompatible_scene_restores_remaining_targets() -> None:
    backend = manager()
    original = backend.get_observation
    SharedRgbdRenderer(backend)
    backend.get_observation()
    backend.scenes[0] = scene(9)
    backend.scenes[0].config.sim_cfg.enable_hbao = True
    assert backend.get_observation() is original.return_value
    assert backend.scenes[1].renderer.bind_render_target.call_args.args[-1] == RendererStub.Flags.NONE
