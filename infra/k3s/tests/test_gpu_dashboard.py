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


def test_job_dashboard_isolates_jobs_and_tracks_replaced_pods() -> None:
    dashboard = json.loads((Path(__file__).parents[1] / "monitoring/job-dashboard.json").read_text())
    inputs: list[dict[str, str]] = []

    def series(metric: str, labels: str, values: str) -> None:
        inputs.append({"series": f"{metric}{{{labels}}}", "values": values})

    for pod, namespace, owner_kind, owner_name in [
        ("old", "training", "Job", "train-test"),
        ("new", "training", "Job", "train-test"),
        ("other", "training", "Job", "train-other"),
        ("new", "other", "Job", "train-test"),
        ("viewer", "training", "ReplicaSet", "train-test"),
    ]:
        labels = f'namespace="{namespace}",pod="{pod}"'
        owner_values = "1 1 1 stale _" if pod == "old" else "_ _ 1 1 1"
        series(
            "kube_pod_owner",
            f'{labels},job="training-pods",owner_kind="{owner_kind}",owner_name="{owner_name}",owner_is_controller="true"',
            owner_values,
        )
        # Duplicate exporter series must not multiply usage or cause a many-to-many join.
        if pod == "old":
            series(
                "kube_pod_owner",
                f'{labels},job="training-pods",owner_kind="Job",owner_name="train-test",owner_is_controller="true",instance="duplicate"',
                owner_values,
            )
        for phase, values in [
            ("Running", "1 1 0 stale _" if pod == "old" else "0 0 1 1 1"),
            ("Failed", "0 0 1 stale _" if pod == "old" else "0+0x4"),
        ]:
            series("kube_pod_status_phase", f'{labels},job="training-pods",phase="{phase}"', values)
        for container, cores, memory, requested, throttled, periods in [
            (
                "train",
                2 if pod == "old" else 4,
                100 if pod == "old" else 200,
                4 if pod == "old" else 8,
                5 if pod == "old" else 2,
                10 if pod == "old" else 20,
            ),
            ("sidecar", 1, 50, 1, 0, 10),
            ("", 100, 10000, 0, 0, 0),
            ("POD", 100, 10000, 0, 0, 0),
        ]:
            if container == "sidecar" and pod != "old":
                continue
            cadvisor = f'{labels},job="training-cadvisor",container="{container}"'
            for metric, increment in [
                ("container_cpu_usage_seconds_total", cores * 60),
                ("container_cpu_cfs_throttled_periods_total", throttled * 60),
                ("container_cpu_cfs_periods_total", periods * 60),
                ("container_fs_reads_bytes_total", cores * 30 * 60),
                ("container_fs_writes_bytes_total", cores * 60 * 60),
            ]:
                series(metric, cadvisor, f"0+{increment}x4")
            series("container_memory_working_set_bytes", cadvisor, f"{memory}+0x4")
            series("container_memory_rss", cadvisor, f"{memory // 2}+0x4")
            if container not in {"", "POD"}:
                kube = f'{labels},job="training-pods",container="{container}"'
                for kind, scale in [("requests", 1), ("limits", 2)]:
                    for resource, multiplier in [("cpu", 1), ("memory", 100)]:
                        series(
                            f"kube_pod_container_resource_{kind}",
                            f'{kube},resource="{resource}"',
                            f"{requested * scale * multiplier}+0x4",
                        )
                series("kube_pod_container_status_restarts_total", kube, "1+0x4" if pod == "old" else "0+0x4")
        for direction, speed in [("receive", 120 if pod == "old" else 240), ("transmit", 60 if pod == "old" else 180)]:
            for interface in ["eth0", "lo"]:
                for container in ["", "POD"]:
                    series(
                        f"container_network_{direction}_bytes_total",
                        f'{labels},job="training-cadvisor",interface="{interface}",container="{container}"',
                        f"0+{speed * 60}x4",
                    )
        gpu = "0" if pod == "old" else "1"
        for metric in ["DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_DEV_FB_USED"]:
            series(metric, f'{labels},job="nvidia-gpu",UUID="GPU-{gpu}",gpu="{gpu}"', "40+0x4")
    expected = {
        "CPU usage": [7, 8, 16],
        "RAM usage": [350, 175, 800, 1600],
        "CPU throttling": [17.5],
        "Network traffic": [360, 240],
        "Disk I/O": [210, 420],
        "Container restarts": [{'{pod="old"}': 2, '{pod="new"}': 0}],
        "Assigned GPU utilization": [{'{UUID="GPU-0",gpu="0"}': 40, '{UUID="GPU-1",gpu="1"}': 40}],
        "Assigned GPU memory": [{'{UUID="GPU-0",gpu="0"}': 40 * 1024**2, '{UUID="GPU-1",gpu="1"}': 40 * 1024**2}],
        "Pod phases": [{'{phase="Failed"}': 1, '{phase="Running"}': 1}],
    }
    checks = []
    for panel in dashboard["panels"]:
        for target, value in zip(panel["targets"], expected[panel["title"]], strict=True):
            expression = (
                target["expr"]
                .replace("$namespace", "training")
                .replace("$job_name", "train-test")
                .replace("$__rate_interval", "2m")
            )
            samples = value if isinstance(value, dict) else {"{}": value}
            checks.append(
                {
                    "expr": expression,
                    "eval_time": "2m",
                    "exp_samples": [{"labels": labels, "value": sample} for labels, sample in samples.items()],
                }
            )
            # A selected Job with no Pods must not display unrelated usage or fake zeroes.
            checks.append(
                {
                    "expr": expression.replace('owner_name="train-test"', 'owner_name="queued"'),
                    "eval_time": "2m",
                    "exp_samples": [],
                }
            )
            if panel["title"] == "CPU usage" and target["refId"] == "A":
                checks.append({"expr": expression, "eval_time": "4m", "exp_samples": [{"labels": "{}", "value": 4}]})
    _check_promql(
        {"evaluation_interval": "1m", "tests": [{"interval": "1m", "input_series": inputs, "promql_expr_test": checks}]}
    )


def test_vllm_dashboards_isolate_jobs_and_combine_replacement_pods() -> None:
    root = Path(__file__).parents[1] / "monitoring"
    dashboards = [json.loads((root / f"vllm-{name}-dashboard.json").read_text()) for name in ["performance", "query"]]
    expressions = {
        target["expr"]
        for dashboard in dashboards
        for panel in dashboard["panels"]
        for target in panel.get("targets", [])
    }
    metrics = set(re.findall(r"vllm:\w+", " ".join(expressions)))
    inputs = []
    for namespace, job, pod in [
        ("training", "selected", "old"),
        ("training", "selected", "replacement"),
        ("training", "other", "unrelated"),
        ("other", "selected", "unrelated"),
    ]:
        labels = f'job="training-vllm",namespace="{namespace}",job_name="{job}",pod="{pod}",model_name="model"'
        for metric in metrics:
            if metric.endswith("_bucket"):
                for boundary, increment in [("1", 60), ("4", 120), ("+Inf", 120)]:
                    inputs.append({"series": f'{metric}{{{labels},le="{boundary}"}}', "values": f"0+{increment}x4"})
            else:
                increment = 360 if metric.endswith("_sum") else 120
                inputs.append({"series": f"{metric}{{{labels}}}", "values": f"0+{increment}x4"})

    def render(expression: str, job: str) -> str:
        return (
            expression.replace("$namespace", "training")
            .replace("$job_name", job)
            .replace("$model_name", "model")
            .replace("$__rate_interval", "2m")
            .replace("$__range", "2m")
        )

    checks = [
        {"expr": render(expression, "absent"), "eval_time": "4m", "exp_samples": []}
        for expression in sorted(expressions)
    ]
    expected = {
        "E2E Latency over Time": [3],
        "TTFT Over Time": [3],
        "E2E Latency (P50)": [1],
        "E2E Latency (P99)": [3.94],
        "TPS (Tokens Per Second) Over Time": [4, 4, 12],
    }
    for panel in dashboards[0]["panels"]:
        if panel["title"] not in expected:
            continue
        for target, value in zip(panel["targets"], expected[panel["title"]], strict=True):
            checks.append(
                {
                    "expr": render(target["expr"], "selected"),
                    "eval_time": "4m",
                    "exp_samples": [{"labels": "{}", "value": value}],
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
