"""Shape, rejection, and termination contracts for scene position sampling."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from rl2.envs.visfly.sampling import sample_positions


def test_sampling_retries_only_rejected_positions() -> None:
    collision = Mock(side_effect=[np.array([False, True]), np.array([False])])
    candidates = [torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]), torch.tensor([[0.8, 0.8, 0.8]])]  # (P, 3)
    with patch("torch.rand", side_effect=candidates):
        positions = sample_positions(torch.zeros(3), torch.ones(3), count=2, collision=collision)  # (N, 3)
    assert positions.shape == (2, 3)
    torch.testing.assert_close(positions, torch.tensor([[0.2, 0.3, 0.4], [0.8, 0.8, 0.8]]))
    assert [call.args[0].shape for call in collision.call_args_list] == [(2, 3), (1, 3)]


def test_goals_respect_distance_and_scene_bounds() -> None:
    reference = torch.tensor([[0.0, 0.0, 0.0]])  # (N, 3)
    collision = Mock(return_value=np.array([False]))
    # First candidate is too close; second is within the requested spherical shell.
    with patch("torch.rand", side_effect=[torch.zeros((1, 3)), torch.tensor([[0.75, 0.0, 0.0]])]):
        positions = sample_positions(
            lower=torch.zeros(3),  # (3,)
            upper=torch.full((3,), 10.0),  # (3,)
            count=1,
            collision=collision,
            reference=reference,
            distance=(1.0, 2.0),
        )  # (N, 3)
    assert positions.shape == (1, 3)
    torch.testing.assert_close(positions, torch.tensor([[1.5, 0.0, 0.0]]))
    assert collision.call_count == 2


def test_sampling_fails_after_bounded_attempts() -> None:
    collision = Mock(return_value=np.array([True]))
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        sample_positions(torch.zeros(3), torch.ones(3), count=1, collision=collision, max_attempts=3)
    assert collision.call_count == 3


def test_sampling_rejects_empty_volume_before_collision_queries() -> None:
    collision = Mock()
    with pytest.raises(ValueError, match="no sampling volume"):
        sample_positions(torch.zeros(3), torch.tensor([1.0, 0.0, 1.0]), count=1, collision=collision)
    collision.assert_not_called()


def test_goal_sampling_rejects_disjoint_reference_bounds() -> None:
    collision = Mock(return_value=np.array([False]))
    with (
        patch("torch.rand", return_value=torch.zeros((1, 3))),
        pytest.raises(ValueError, match="no sampling volume"),
    ):
        sample_positions(
            lower=torch.zeros(3),  # (3,)
            upper=torch.ones(3),  # (3,)
            count=1,
            collision=collision,
            reference=torch.tensor([[4.0, 0.0, 0.0]]),  # (N, 3)
            distance=(1.0, 2.0),
        )
    collision.assert_not_called()


def test_sampling_empty_batch_skips_collision_queries() -> None:
    collision = Mock(side_effect=AssertionError("No positions to query"))
    positions = sample_positions(torch.zeros(3), torch.ones(3), count=0, collision=collision)  # (0, 3)
    assert positions.shape == (0, 3)
    assert positions.dtype == torch.float32
    collision.assert_not_called()
