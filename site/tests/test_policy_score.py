"""Check the interactive PUCT bonus and final selection scores."""

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


def test_policy_score_matches_formula_and_exploration_behavior() -> None:
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
    expected = [
        math.sqrt(node_visits)
        * (1.25 + math.log((node_visits + 19652.0 + 1.0) / 19652.0))
        * probability
        / (action_visits + 1)
        for probability, node_visits, action_visits in cases
    ]
    np.testing.assert_allclose(scores, expected, rtol=1e-6)
    assert scores[0] == 0
    assert scores[1] > 0
    assert scores[2] > scores[3] > scores[4]
    assert scores[5] > scores[3]


def test_final_selection_score_combines_terms_and_masks_actions() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Site tests require Node >=24.22.2")
    result = subprocess.run(
        [
            node,
            "--input-type=module",
            "-e",
            """
            import { computePolicyScore, computeSelectionScore, selectionNoise } from './src/components/policyScore.ts';
            console.log(JSON.stringify({
                bonus: computePolicyScore(0.6, 32, 8),
                visited: computeSelectionScore(0.6, 32, 8, 0.75, 'legal'),
                unvisitedBonus: computePolicyScore(0.6, 32, 0),
                unvisited: computeSelectionScore(0.6, 32, 0, 0.75, 'legal'),
                zeroPrior: computeSelectionScore(0, 32, 8, 0.75, 'legal'),
                zeroTerms: computeSelectionScore(0, 1, 0, 0.75, 'legal'),
                masked: computeSelectionScore(0.6, 32, 8, 0.75, 'illegal') === -Infinity,
                fallback: computeSelectionScore(0.6, 32, 8, 0.75, 'none_legal'),
                noise: selectionNoise,
            }));
            """,
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    scores = json.loads(result.stdout)
    assert 0 < scores["noise"] < 1e-7
    assert scores["visited"] == pytest.approx(0.75 + scores["bonus"] + scores["noise"])
    assert scores["unvisited"] == pytest.approx(scores["unvisitedBonus"] + scores["noise"])
    assert scores["zeroPrior"] == pytest.approx(0.75 + scores["noise"])
    assert scores["zeroTerms"] == scores["noise"]
    assert scores["masked"]
    assert scores["fallback"] == scores["visited"]
