import json
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).parents[1]


def test_job_dashboard_metrics_are_collected_and_provisioned() -> None:
    manifests = list(yaml.safe_load_all((ROOT / "monitoring/monitoring.yaml").read_text()))
    charts = {
        item["metadata"]["name"]: yaml.safe_load(item["spec"]["valuesContent"])
        for item in manifests
        if item["kind"] == "HelmChart"
    }
    prometheus = charts["gpu-prometheus"]
    exporters = {item["job_name"]: item for item in yaml.safe_load(prometheus["extraScrapeConfigs"])}
    cadvisor = exporters["training-cadvisor"]
    keep = next(rule["regex"] for rule in cadvisor["metric_relabel_configs"] if rule["source_labels"] == ["__name__"])
    kube = prometheus["kube-state-metrics"]
    assert kube["enabled"] and kube["namespaces"] == ["training"]
    assert kube["collectors"] == ["pods"]
    assert prometheus["server"]["automountServiceAccountToken"]
    assert cadvisor["scheme"] == "https" and cadvisor["tls_config"]["ca_file"]
    assert cadvisor["authorization"]["credentials_file"]
    dashboard = json.loads((ROOT / "monitoring/job-dashboard.json").read_text())
    for panel in dashboard["panels"]:
        for target in panel["targets"]:
            for metric in re.findall(r"\b(?:container_|kube_pod_|DCGM_FI_DEV_)\w+", target["expr"]):
                if metric.startswith("container_"):
                    assert re.fullmatch(keep, metric), metric
                elif metric.startswith("kube_pod_"):
                    assert metric in kube["metricAllowlist"], metric
                else:
                    assert metric in charts["dcgm-exporter"]["customMetrics"], metric
    kustomization = yaml.safe_load((ROOT / "monitoring/kustomization.yaml").read_text())
    configmaps: dict[str, Any] = {item["name"]: item for item in kustomization["configMapGenerator"]}
    mounted = charts["gpu-grafana"]["dashboardsConfigMaps"]["gpu"]
    assert "job-dashboard.json" in configmaps[mounted]["files"]
    page = (ROOT / "ray-timelines/index.html").read_text()
    assert f"/d/{dashboard['uid']}?" in page
    variables = {item["name"] for item in dashboard["templating"]["list"]}
    assert set(re.findall(r'"var-([^"]+)"', page)) <= variables


def test_monitoring_permissions_are_scoped_to_training_and_local_node() -> None:
    manifests = list(yaml.safe_load_all((ROOT / "monitoring/monitoring.yaml").read_text()))
    role = next(item for item in manifests if item["kind"] == "Role")
    assert role["metadata"]["namespace"] == "training"
    assert role["rules"] == [{"apiGroups": [""], "resources": ["pods"], "verbs": ["list", "watch"]}]
    cluster_role = next(item for item in manifests if item["kind"] == "ClusterRole")
    assert cluster_role["rules"] == [
        {"apiGroups": [""], "resources": ["nodes/proxy"], "resourceNames": ["vlad-dev-gpu-1"], "verbs": ["get"]}
    ]
