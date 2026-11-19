"""Precision and geometry-boundary contracts for host collision queries."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from rl2.envs.visfly.collision import collision_points


def test_collision_coordinates_threshold_and_out_of_bounds() -> None:
    points = np.array([[1, 2, 3]] * 3, dtype=np.float32)  # (3, 3)
    habitat = np.array([-2, 3, -1], dtype=np.float64)  # (3,)
    scene = Mock()
    scene.get_closest_collision_point.side_effect = [
        SimpleNamespace(hit_pos=habitat + [0.25, 0, 0], is_out_bound=False),
        SimpleNamespace(hit_pos=habitat + [0.5, 0, 0], is_out_bound=False),
        SimpleNamespace(hit_pos=habitat + [1, 0, 0], is_out_bound=True),
    ]
    result = collision_points(scene, points, radius=0.5)  # (3,)
    assert result.shape == (3,) and result.dtype == np.bool_
    assert result.tolist() == [True, False, True]
    for call in scene.get_closest_collision_point.call_args_list:
        np.testing.assert_array_equal(call.kwargs["pt"], habitat[:, None])
        assert call.kwargs["max_search_radius"] == 0.5
    scene.reset_mock()
    assert collision_points(scene, points[:0], radius=0.5).shape == (0,)
    scene.get_closest_collision_point.assert_not_called()
