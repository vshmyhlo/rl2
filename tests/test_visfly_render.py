"""Navigation geometry and diagnostic panel content without a simulator."""

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from rl2.envs.gym_style_env import StopReason
from rl2.envs.visfly.render import RenderContext, diagnostics_panel, navigation_diagnostics


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
