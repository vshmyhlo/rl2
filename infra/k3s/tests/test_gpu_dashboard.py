"""Opt-in PromQL regression check using the installed Prometheus promtool."""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.skipif(os.environ.get("RL2_K3S_TESTS") != "1", reason="Requires local Prometheus")


def test_dashboard_keeps_gpu_identity_across_pod_changes() -> None:
    dashboard = json.loads((Path(__file__).parents[1] / "monitoring/gpu-dashboard.json").read_text())
    expressions = {target["expr"] for panel in dashboard["panels"] for target in panel["targets"]}
    metrics = sorted(set(re.findall(r"DCGM_FI_DEV_\w+", " ".join(expressions))))
    first_gpu = 'UUID="GPU-a",hostname="node",gpu="0",modelName="A100"'
    second_gpu = 'UUID="GPU-b",hostname="node",gpu="1",modelName="A100"'
    inputs = [
        {"series": f"{metric}{{{labels}}}", "values": values}
        for metric in metrics
        for labels, values in [
            (first_gpu + ',pod="old",namespace="training",container="train"', "20 20 stale"),
            (first_gpu + ',pod="new",namespace="training",container="train"', "_ 40 40"),
            (second_gpu, "60 60 60"),
        ]
    ]
    checks = []
    for expression in sorted(expressions):
        scale = (1024**2 if "1024" in expression else 1) * (2 if "DCGM_FI_DEV_FB_FREE" in expression else 1)
        # Cover the old Pod, overlapping labels, and the replacement after staleness.
        for minute, value in [(0, 20), (1, 40), (2, 40)]:
            checks.append(
                {
                    "expr": expression,
                    "eval_time": f"{minute}m",
                    "exp_samples": [
                        {"labels": "{" + first_gpu + "}", "value": value * scale},
                        {"labels": "{" + second_gpu + "}", "value": 60 * scale},
                    ],
                }
            )
    suite = {
        "evaluation_interval": "1m",
        "tests": [{"interval": "1m", "input_series": inputs, "promql_expr_test": checks}],
    }
    result = subprocess.run(
        [
            "sudo",
            "-n",
            "k3s",
            "kubectl",
            "-n",
            "monitoring",
            "exec",
            "-i",
            "deployment/gpu-prometheus",
            "-c",
            "prometheus-server",
            "--",
            "promtool",
            "test",
            "rules",
            "/dev/stdin",
        ],
        input=yaml.safe_dump(suite),
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
