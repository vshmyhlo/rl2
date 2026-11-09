"""Check direct Gamma sampling and the browser's prepare_logits behavior."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest


SITE = Path(__file__).resolve().parents[1]


def run_node(script: str) -> Any:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Site tests require Node >=24.22.2")
    result = subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=SITE,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_policy_distributions_masking_and_floor() -> None:
    data = run_node("""
        import { policyDistributions as distribution } from './src/components/dirichletNoise.ts';
        const logits = [2, 1, 0, -1];
        const legal = [true, true, false, true];
        const logNoise = [0, 1, 5, -2];
        console.log(JSON.stringify({
            base: distribution(logits, legal, logNoise, 0.25),
            zero: distribution(logits, legal, logNoise, 0),
            one: distribution(logits, legal, logNoise, 1),
            single: distribution(logits, [false, true, false, false], logNoise, 0.5),
            none: distribution(logits, [false, false, false, false], logNoise, 0.5),
            tiny: distribution([0, -1000], [true, true], [0, -1000], 1),
        }));
    """)
    base = data["base"]
    assert base["raw"] == pytest.approx([0.64391426, 0.23688282, 0.08714432, 0.03205860])
    assert base["masked"] == pytest.approx([0.70538451, 0.25949646, 0, 0.03511903])
    for name in ("raw", "masked", "noise", "mixed"):
        assert len(base[name]) == 4
        assert sum(base[name]) == pytest.approx(1)
        assert all(0 <= value <= 1 for value in base[name])
    for name in ("masked", "noise", "mixed"):
        assert base[name][2] == 0
        assert data["single"][name] == [0, 1, 0, 0]
        assert data["none"][name] == [0.25] * 4
    assert base["mixed"] == pytest.approx(
        [0.75 * policy + 0.25 * noise for policy, noise in zip(base["masked"], base["noise"], strict=True)]
    )
    assert data["zero"]["mixed"] == base["masked"]
    assert data["one"]["mixed"] == pytest.approx(base["noise"])
    assert len(data["tiny"]["mixed"]) == 2
    assert data["tiny"]["mixed"][1] == pytest.approx(2**-126, rel=1e-6, abs=0)


def test_direct_sampler_distribution_and_reproducibility() -> None:
    data = run_node("""
        import { sampleLogGamma, policyDistributions } from './src/components/dirichletNoise.ts';
        const draw = (alpha, seed) => {
            const logNoise = sampleLogGamma(alpha, 6, seed);
            return policyDistributions(Array(6).fill(0), Array(6).fill(true), logNoise, 1).noise;
        };
        const moments = [0.05, 1, 3].map(alpha => {
            const draws = Array.from({length: 2000}, (_, i) => draw(alpha, i * 7919 + 7));
            return {
                means: Array.from({length: 6}, (_, i) => draws.reduce((sum, row) => sum + row[i], 0) / draws.length),
                second: draws.reduce((sum, row) => sum + row[0] ** 2, 0) / draws.length,
                normalized: draws.every(row => row.length === 6 && row.every(p => p >= 0 && p <= 1)
                    && Math.abs(row.reduce((sum, p) => sum + p, 0) - 1) < 1e-12),
            };
        });
        console.log(JSON.stringify({
            sample: draw(0.3, 7),
            repeated: draw(0.3, 7),
            resampled: draw(0.3, 8),
            single: policyDistributions([0], [true], sampleLogGamma(0.05, 1, 7), 1).noise,
            moments,
        }));
    """)
    assert data["sample"] == data["repeated"]
    assert data["sample"] != data["resampled"]
    assert data["single"] == [1]
    # Small concentration, the branch at one, and the slider maximum.
    for alpha, moments in zip((0.05, 1, 3), data["moments"], strict=True):
        assert moments["normalized"]
        assert moments["means"] == pytest.approx([1 / 6] * 6, abs=0.02)
        assert moments["second"] == pytest.approx((alpha + 1) / (6 * (6 * alpha + 1)), abs=0.02)
