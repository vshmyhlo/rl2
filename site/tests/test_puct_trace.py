"""Check the website's deterministic PUCT walkthrough using Node's TypeScript support."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="module")
def demo() -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Site tests require Node >=24.22.2")
    script = """
        import { buildTrace, actionWeights, samplingWeights } from './src/components/puctTrace.ts';
        const weights = actionWeights([3, 5, 0, 0], [true, true, true, false]);
        console.log(JSON.stringify({
            frames: buildTrace(),
            zero: buildTrace(0),
            weights,
            legalFallback: actionWeights([0, 0, 0], [true, false, true]),
            noLegalFallback: actionWeights([0, 0], [false, false]),
            cold: samplingWeights(weights, 0),
            normal: samplingWeights(weights, 1),
            warm: samplingWeights(weights, 2),
            tie: samplingWeights([0.5, 0.5, 0], 0),
        }));
    """
    result = subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_trace_expands_one_node_and_backs_up_each_path(demo: dict[str, Any]) -> None:
    frames = demo["frames"]
    assert len(frames) == 26
    assert len(frames[0]["nodes"]) == 1
    assert frames[0]["nodes"][0]["visits"] == 1
    for start in range(1, 25, 3):
        select, expand, backup = frames[start : start + 3]
        simulation = select["simulation"]
        assert [select["phase"], expand["phase"], backup["phase"]] == ["select", "expand", "backup"]
        assert len(select["nodes"]) == simulation
        assert len(expand["nodes"]) == len(backup["nodes"]) == simulation + 1
        assert sum(expand["nodes"][0]["counts"]) == simulation - 1
        assert sum(backup["nodes"][0]["counts"]) == simulation
        assert backup["nodes"][0]["visits"] == simulation + 1
        for before, after in zip(expand["nodes"], backup["nodes"], strict=True):
            traversed = before["id"] in backup["path"][:-1]
            assert after["visits"] == before["visits"] + traversed
        for child, parent in zip(backup["returns"], backup["returns"][1:]):
            assert parent["value"] == pytest.approx(-child["value"])
    assert frames[-1]["nodes"][0]["counts"] == [3, 5, 0, 0]
    assert any(len(frame["path"]) > 2 for frame in frames)


def test_first_selection_and_player_perspective(demo: dict[str, Any]) -> None:
    select, expand, backup = demo["frames"][1:4]
    assert select["action"] == 0
    assert [score["normalized"] for score in select["scores"]] == [0, 0, 0, 0]
    assert expand["nodes"][1]["prediction"] == pytest.approx(-0.2)
    assert backup["nodes"][0]["value"] == pytest.approx(0.15)
    assert backup["nodes"][1]["value"] == pytest.approx(-0.2)


def test_fallbacks_and_temperature_keep_targets_separate(demo: dict[str, Any]) -> None:
    assert [frame["phase"] for frame in demo["zero"]] == ["prepare", "output"]
    assert demo["legalFallback"] == [0.5, 0, 0.5]
    assert demo["noLegalFallback"] == [0.5, 0.5]
    assert demo["weights"] == [0.375, 0.625, 0, 0]
    assert demo["normal"] == pytest.approx(demo["weights"])
    assert demo["cold"] == [0, 1, 0, 0]
    assert demo["tie"] == [0.5, 0.5, 0]
    assert 0.5 < demo["warm"][1] < 0.625
    assert sum(demo["warm"]) == pytest.approx(1)
    assert demo["warm"][2:] == [0, 0]
