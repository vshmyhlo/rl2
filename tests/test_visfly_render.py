"""Navigation geometry and diagnostic panel content without a simulator."""

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from rl2.envs.gym_style_env import StopReason
from rl2.envs.visfly.render import (
    RenderContext,
    depth_panel,
    diagnostics_panel,
    fit_panel,
    navigation_diagnostics,
    top_down_panel,
)


@pytest.mark.parametrize("shape", [(2, 8), (8, 2)], ids=["wide", "tall"])
def test_fit_panel_preserves_aspect_ratio_and_centers(shape: tuple[int, int]) -> None:
    source = np.full((*shape, 3), 180, dtype=np.uint8)  # (H_source, W_source, 3)
    result = fit_panel(source, height=8, width=8)  # (H, W, 3)
    assert result.shape == (8, 8, 3)
    assert result.dtype == np.uint8
    if shape[0] == 2:
        np.testing.assert_array_equal(result[3:5], 180)
        np.testing.assert_array_equal(result[[0, 7], 0], [[20, 25, 34]] * 2)
    else:
        np.testing.assert_array_equal(result[:, 3:5], 180)
        np.testing.assert_array_equal(result[0, [0, 7]], [[20, 25, 34]] * 2)
    np.testing.assert_array_equal(source, 180)


def test_top_down_trail_orientation_scale_and_markers() -> None:
    trajectory = np.array([[0, 0, 1], [0, 4, 1], [3, 4, 2]], dtype=np.float32)  # (T, 3)
    target = np.array([6, 0, 3], dtype=np.float32)  # (3,)
    with (
        patch("rl2.envs.visfly.render.cv2.polylines", wraps=cv2.polylines) as lines,
        patch("rl2.envs.visfly.render.cv2.circle", wraps=cv2.circle) as circles,
        patch("rl2.envs.visfly.render.cv2.putText", wraps=cv2.putText) as text,
    ):
        panel = top_down_panel(trajectory, target, height=320)  # (H, H, 3)
    assert panel.shape == (320, 320, 3)
    assert panel.dtype == np.uint8
    points = lines.call_args.args[1][0]  # (T, 2)
    assert points.shape == (3, 2)
    assert points[0, 0] == points[1, 0] < points[2, 0]  # +X right.
    assert points[0, 1] > points[1, 1] == points[2, 1]  # +Y up.
    assert (points[0, 1] - points[1, 1]) / 4 == pytest.approx((points[2, 0] - points[1, 0]) / 3, abs=0.5)
    goal, agent = [call.args[1] for call in circles.call_args_list]
    assert goal[0] > agent[0] and goal[1] > agent[1]
    assert agent == tuple(points[-1])
    assert "Altitude 2.0 m | target 3.0 m" in [call.args[1] for call in text.call_args_list]
    translated = top_down_panel(trajectory + [100, -50, 0], target + [100, -50, 0], height=320)  # (H, H, 3)
    np.testing.assert_array_equal(panel, translated)
    np.testing.assert_array_equal(trajectory, [[0, 0, 1], [0, 4, 1], [3, 4, 2]])
    np.testing.assert_array_equal(target, [6, 0, 3])


def test_top_down_single_position_at_target() -> None:
    panel = top_down_panel(np.zeros((1, 3)), np.zeros(3), height=64)  # (H, H, 3)
    assert panel.shape == (64, 64, 3)
    assert panel.dtype == np.uint8
    assert panel.any()


def test_depth_panel_fixed_scale_clipping_and_aspect_ratio() -> None:
    depth = np.tile(np.array([-1, 0, 10, 20, 30], dtype=np.float32)[None, :, None], (2, 1, 1))  # (H_depth, W_depth, 1)
    before = depth.copy()  # (H_depth, W_depth, 1)
    with patch("rl2.envs.visfly.render.cv2.putText", wraps=cv2.putText) as put_text:
        panel = depth_panel(depth, height=64)  # (H, W, 3)
    assert panel.shape == (64, 160, 3)
    assert panel.dtype == np.uint8
    assert "Depth: 0-20 m" in [call.args[1] for call in put_text.call_args_list]
    np.testing.assert_array_equal(depth, before)
    expected = cv2.cvtColor(
        cv2.applyColorMap(np.array([[0, 0, 128, 255, 255]], dtype=np.uint8), cv2.COLORMAP_INFERNO),
        cv2.COLOR_BGR2RGB,
    )  # (1, W_depth, 3)
    np.testing.assert_array_equal(panel[-1, ::32], expected[0])
    depth[:] = 10
    constant = depth_panel(depth, height=64)  # (H, W, 3)
    np.testing.assert_array_equal(constant[-1, 0], panel[-1, 64])  # Same metres retain the same colour.


def test_quaternion_diagnostics_rotate_goal_and_project_velocity() -> None:
    state = np.zeros(13)  # (S,)
    state[:3] = [1, 2, 3]
    state[3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]  # (4,) yaw +90 degrees, wxyz
    state[-6:-3] = [0, 2, 0]
    result = navigation_diagnostics(state, np.array([1, 6, 3]))
    assert result.body_goal.shape == (3,)
    np.testing.assert_allclose(result.body_goal, [4, 0, 0], atol=1e-12)
    assert result.distance == 4
    assert result.closing_speed == result.speed == 2
    assert result.tilt_degrees == pytest.approx(0)
    state[-6:-3] *= -1
    assert navigation_diagnostics(state, np.array([1, 6, 3])).closing_speed == -2


def test_euler_diagnostics_at_goal_with_tilt() -> None:
    state = np.zeros(12)  # (S,)
    state[3] = np.pi / 3
    state[-6] = 3
    result = navigation_diagnostics(state, np.zeros(3))
    assert result.distance == result.closing_speed == 0
    assert result.speed == 3
    assert result.tilt_degrees == pytest.approx(60)
    np.testing.assert_array_equal(result.body_goal, np.zeros(3))


def test_ongoing_stop_reason_keeps_flight_title() -> None:
    state = np.zeros(13)  # (S,)
    state[3] = 1
    with patch("rl2.envs.visfly.render.cv2.putText", wraps=cv2.putText) as put_text:
        panel = diagnostics_panel(
            state,
            np.zeros(3),  # (3,)
            height=256,
            action_low=-np.ones(4),  # (A,)
            action_high=np.ones(4),  # (A,)
            action_type="bodyrate",
            context=RenderContext(step=1, episode_return=0, stop_reason=StopReason.ONGOING),
        )  # (H, W_debug, 3)
    labels = [call.args[1] for call in put_text.call_args_list]
    assert panel.shape == (256, 384, 3)
    assert "FLIGHT DIAGNOSTICS" in labels
    assert "Step 1" in labels
    assert not any("last recorded" in label for label in labels)


@pytest.mark.parametrize("height", [256, 128], ids=["native", "scaled"])
def test_panel_shape_content_and_missing_transition(height: int) -> None:
    state = np.zeros(13)  # (S,)
    state[3] = 1
    kwargs = {
        "height": height,
        "action_low": -np.ones(4),  # (A,)
        "action_high": np.ones(4),  # (A,)
        "action_type": "bodyrate",
        "max_episode_steps": 500,
    }
    with patch("rl2.envs.visfly.render.cv2.putText", wraps=cv2.putText) as put_text:
        initial = diagnostics_panel(state, np.zeros(3), **kwargs)  # (H, W_debug, 3)
    labels = [call.args[1] for call in put_text.call_args_list]
    assert not any("reward" in label or "Return" in label or "Last action" in label for label in labels)
    context = RenderContext(
        step=10,
        episode_return=0,
        reward=0,
        action=np.array([-1.0, 0, 0.5, 1.0]),  # (A,)
        stop_reason=StopReason.COLLISION,
    )
    with patch("rl2.envs.visfly.render.cv2.putText", wraps=cv2.putText) as put_text:
        final = diagnostics_panel(state, np.zeros(3), context=context, **kwargs)  # (H, W_debug, 3)
    labels = [call.args[1] for call in put_text.call_args_list]
    assert "COLLISION" in labels
    assert "Step 10 / 500 | views: last recorded" in labels
    assert "Prev reward +0.000 | Return +0.000" in labels
    assert all(label in labels for label in ("Thrust", "Rate X", "Rate Y", "Rate Z"))
    assert initial.shape == final.shape == (height, round(height * 1.5), 3)
    assert initial.dtype == final.dtype == np.uint8
    assert not np.array_equal(initial, final)
