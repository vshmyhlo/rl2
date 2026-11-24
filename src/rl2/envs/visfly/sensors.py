"""Shared RGB-D sensing for fixed, aligned Habitat pinhole cameras."""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

SensorImages = dict[str, NDArray[Any]]  # values: (H, W, C) color or (H, W) depth


def compatible_sensors(scene: Any, suite: dict[str, Any]) -> bool:
    """Restrict sharing to aligned CPU cameras without noise or postprocessing."""
    if suite.keys() != {"color", "depth"} or scene.config.enable_batch_renderer or scene.config.sim_cfg.enable_hbao:
        return False
    color, depth = suite["color"], suite["depth"]
    a, b = color._spec, depth._spec
    if a.sensor_type.name != "COLOR" or b.sensor_type.name != "DEPTH":
        return False
    if any(
        spec.sensor_subtype.name != "PINHOLE" or spec.gpu2gpu_transfer or spec.noise_model != "None" for spec in (a, b)
    ):
        return False
    return (
        all(np.array_equal(getattr(a, key), getattr(b, key)) for key in ("resolution", "position", "orientation"))
        and all(getattr(a, key) == getattr(b, key) for key in ("near", "far", "hfov"))
        and color._sensor_object.node.transformation == depth._sensor_object.node.transformation
    )


class SharedRgbdRenderer:
    """Cache fixed camera pairs per loaded scene; preserve ordinary sensing as fallback.

    Camera configuration stays fixed until a scene reload. Video rendering disables
    sharing for this manager because diagnostic overlays can modify RGB depth.
    """

    def __init__(self, manager: Any) -> None:
        self.manager = manager
        self._original: Callable[..., list[SensorImages]] = manager.get_observation
        self._render: Callable[..., Any] = manager.render
        self._scenes: tuple[Any, ...] = ()
        self._groups: list[list[dict[str, Any]]] = []
        self._enabled = True
        manager.get_observation = self
        manager.render = self.render

    def _prepare(self) -> None:
        self._restore_targets()
        self._scenes = tuple(self.manager.scenes)
        self._groups = []
        groups = [
            [scene._Simulator__sensors[index] for index in range(self.manager.num_agent_per_scene)]
            for scene in self._scenes
        ]
        if not all(
            getattr(getattr(type(scene.renderer), "Flags", None), "RGB_DEPTH", None) is not None
            and all(compatible_sensors(scene, suite) for suite in group)
            for scene, group in zip(self._scenes, groups, strict=True)
        ):
            return
        for scene, group in zip(self._scenes, groups, strict=True):
            for suite in group:
                scene.renderer.bind_render_target(suite["color"]._sensor_object, type(scene.renderer).Flags.RGB_DEPTH)
        self._groups = groups

    def __call__(self, indices: Any = None) -> list[SensorImages]:  # indices: (M,) selected agents, or None
        if indices is not None and len(indices) == 0:
            return []
        if not self._enabled:
            return self._original(indices=indices)
        if len(self._scenes) != len(self.manager.scenes) or any(
            previous is not current for previous, current in zip(self._scenes, self.manager.scenes)
        ):
            self._prepare()
        if not self._groups:
            return self._original(indices=indices)
        groups = (
            self._groups
            if indices is None
            else [
                [
                    self._groups[int(index) // self.manager.num_agent_per_scene][
                        int(index) % self.manager.num_agent_per_scene
                    ]
                ]
                for index in indices
            ]
        )
        observations = []
        for group in groups:
            for suite in group:
                suite["color"].draw_observation()
            for suite in group:
                color, depth = suite["color"], suite["depth"]
                rgb = color.get_observation()  # (H, W, 4)
                color._sensor_object.render_target.read_frame_depth(depth.view)
                distances = depth._noise_model(np.flip(depth._buffer, axis=0))  # (H, W)
                observations.append({"color": rgb, "depth": distances})
        return observations  # length B or M; leaves: (H, W, C) or (H, W)

    def _restore_targets(self) -> None:
        for scene, group in zip(self._scenes, self._groups):
            if any(scene is current for current in self.manager.scenes):
                for suite in group:
                    scene.renderer.bind_render_target(suite["color"]._sensor_object, type(scene.renderer).Flags.NONE)

    def render(self, *args: Any, **kwargs: Any) -> Any:
        """Restore separate targets before visualization can add depth-writing lines."""
        if self._enabled:
            self._restore_targets()
            self._enabled = False
            self._scenes = ()
            self._groups = []
        return self._render(*args, **kwargs)
