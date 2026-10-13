"""Episode statistics shared by training and evaluation."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingScores:
    """Raw episode statistics with reported means over all training.

    These are the training metrics from Section 6.4 of
    https://arxiv.org/abs/1707.06347, separate from frozen-policy evaluation.
    A missing return_sum (e.g. an older checkpoint) leaves the all-training
    mean unavailable.
    Matching these metrics does not match Atari protocols.
    length_sum tracks completed episode lengths in steps;
    None leaves the length mean unavailable.
    """

    episode_count: int
    return_sum: float | None
    length_sum: int | None = None

    def __post_init__(self) -> None:
        if self.episode_count < 0:
            raise ValueError("episode_count must be nonnegative")
        if self.return_sum is not None and not math.isfinite(self.return_sum):
            raise ValueError("training return_sum must be finite or unavailable")
        if self.length_sum is not None and self.length_sum < 0:
            raise ValueError("training length_sum must be nonnegative or unavailable")

    def as_report(self) -> dict[str, int | float | None]:
        return {
            "total_episodes": self.episode_count,
            "total_return_mean": self.return_sum / self.episode_count
            if self.episode_count and self.return_sum is not None
            else None,
            "total_episode_length_mean": self.length_sum / self.episode_count
            if self.episode_count and self.length_sum is not None
            else None,
        }
