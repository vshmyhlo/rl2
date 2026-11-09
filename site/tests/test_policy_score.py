"""Keep the interactive exploration bonus aligned with the training implementation."""

import json
import shutil
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2.alphazero.train import compute_policy_score


def test_policy_score_matches_training_and_exploration_behavior() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Site tests require Node >=24.22.2")
    # Zero prior, fresh node, unvisited edge, repeated edge, another edge, and visit limit.
    cases = [(0.0, 32, 8), (0.6, 1, 0), (0.6, 32, 0), (0.6, 32, 8), (0.6, 32, 9), (0.6, 33, 8), (1.0, 1000, 999)]
    result = subprocess.run(
        [
            node,
            "--input-type=module",
            "-e",
            (
                "import { computePolicyScore } from './src/components/policyScore.ts';"
                f"console.log(JSON.stringify({json.dumps(cases)}.map(args => computePolicyScore(...args))));"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    scores = json.loads(result.stdout)
    probabilities, node_visits, action_visits = jnp.array(cases).T  # Each: (C,), C = distinct cases.
    expected = jax.vmap(compute_policy_score)(probabilities, node_visits, action_visits)  # (C,)
    np.testing.assert_allclose(scores, expected, rtol=1e-6)
    assert scores[0] == 0
    assert scores[1] > 0
    assert scores[2] > scores[3] > scores[4]
    assert scores[5] > scores[3]
