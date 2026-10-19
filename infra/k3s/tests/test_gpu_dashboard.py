"""Opt-in PromQL regression check using the installed Prometheus promtool."""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.skipif(os.environ.get("RL2_K3S_TESTS") != "1", reason="Requires local Prometheus")


def test_dashboard_keeps_gpu_identity_across_pod_changes() -> None:
    dashboard = json.loads((Path(__file__).parents[1] / "monitoring/gpu-dashboard.json").read_text())
    expressions = {
        target["expr"]
        for panel in dashboard["panels"]
        for target in panel["targets"]
        if "DCGM_FI_DEV_" in target["expr"]
    }
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
    _check_promql(suite)


def test_dashboard_host_cpu_and_ram() -> None:
    dashboard = json.loads((Path(__file__).parents[1] / "monitoring/gpu-dashboard.json").read_text())
    panels = {panel["title"]: panel for panel in dashboard["panels"]}
    inputs = []
    # Unequal per-core idle time checks averaging; separate hosts check isolation.
    for node, idle in [("busy", ["0+15x4", "0+45x4"]), ("idle", ["0+60x4"]), ("full", ["0+0x4"])]:
        for cpu, values in enumerate(idle):
            inputs.append(
                {
                    "series": f'node_cpu_seconds_total{{job="node",instance="{node}",cpu="{cpu}",mode="idle"}}',
                    "values": values,
                }
            )
        inputs.append(
            {
                "series": f'node_cpu_seconds_total{{job="node",instance="{node}",cpu="0",mode="user"}}',
                "values": "0+30x4",
            }
        )
        for metric, value in [("MemTotal", 1024), ("MemAvailable", {"busy": 256, "idle": 1024, "full": 0}[node])]:
            inputs.append(
                {"series": f'node_memory_{metric}_bytes{{job="node",instance="{node}"}}', "values": f"{value}+0x4"}
            )
    checks = []
    for title, expected_targets in [
        ("Host CPU utilization", [{"busy": 50, "idle": 0, "full": 100}]),
        ("Host CPU usage history", [{"busy": 50, "idle": 0, "full": 100}]),
        ("Host RAM utilization", [{"busy": 75, "idle": 0, "full": 100}]),
        ("Host RAM usage", [{"busy": 768, "idle": 0, "full": 1024}, {"busy": 1024, "idle": 1024, "full": 1024}]),
    ]:
        panel = panels[title]
        assert panel["fieldConfig"]["defaults"]["unit"] == ("bytes" if title == "Host RAM usage" else "percent")
        assert len(panel["targets"]) == len(expected_targets)
        for target, expected in zip(panel["targets"], expected_targets, strict=True):
            expression = target["expr"].replace("$__rate_interval", "2m")
            labels = "" if "CPU" in title else 'job="node",'
            if expression == 'node_memory_MemTotal_bytes{job="node"}':
                labels += '__name__="node_memory_MemTotal_bytes",'
            checks.append(
                {
                    "expr": expression,
                    "eval_time": "4m",
                    "exp_samples": [
                        {"labels": "{" + labels + f'instance="{node}"' + "}", "value": value}
                        for node, value in expected.items()
                    ],
                }
            )
    _check_promql(
        {"evaluation_interval": "1m", "tests": [{"interval": "1m", "input_series": inputs, "promql_expr_test": checks}]}
    )


def _check_promql(suite: dict[str, Any]) -> None:
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
