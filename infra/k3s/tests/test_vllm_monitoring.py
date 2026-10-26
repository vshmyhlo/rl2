import json
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).parents[1]


def test_vllm_discovery_and_dashboard_provisioning() -> None:
    manifests = list(yaml.safe_load_all((ROOT / "monitoring/monitoring.yaml").read_text()))
    charts = {
        item["metadata"]["name"]: yaml.safe_load(item["spec"]["valuesContent"])
        for item in manifests
        if item["kind"] == "HelmChart"
    }
    scrapes = yaml.safe_load(charts["gpu-prometheus"]["extraScrapeConfigs"])
    scrape = next(item for item in scrapes if item["job_name"] == "training-vllm")
    assert scrape["kubernetes_sd_configs"] == [{"role": "pod", "namespaces": {"names": ["training"]}}]
    rules = {tuple(rule["source_labels"]): rule for rule in scrape["relabel_configs"]}
    for label, value in [("controller_kind", "Job"), ("phase", "Running"), ("container_port_name", "vllm")]:
        rule = rules[(f"__meta_kubernetes_pod_{label}",)]
        assert rule["action"] == "keep" and rule["regex"] == value
    for source, target in [
        ("__meta_kubernetes_namespace", "namespace"),
        ("__meta_kubernetes_pod_controller_name", "job_name"),
        ("__meta_kubernetes_pod_name", "pod"),
    ]:
        assert rules[(source,)]["target_label"] == target
    assert scrape["metric_relabel_configs"] == [{"source_labels": ["__name__"], "regex": "vllm:.*", "action": "keep"}]
    binding = next(item for item in manifests if item["kind"] == "RoleBinding")
    assert binding["metadata"]["namespace"] == "training"
    assert {"kind": "ServiceAccount", "name": "gpu-prometheus", "namespace": "monitoring"} in binding["subjects"]
    kustomization = yaml.safe_load((ROOT / "monitoring/kustomization.yaml").read_text())
    mounted = charts["gpu-grafana"]["dashboardsConfigMaps"]["gpu"]
    configmap = next(item for item in kustomization["configMapGenerator"] if item["name"] == mounted)
    resources = json.loads((ROOT / "monitoring/job-dashboard.json").read_text())
    for name in ["performance", "query"]:
        filename = f"vllm-{name}-dashboard.json"
        assert filename in configmap["files"]
        dashboard: dict[str, Any] = json.loads((ROOT / "monitoring" / filename).read_text())
        assert any(link["url"] == f"/d/{dashboard['uid']}" and link["includeVars"] for link in resources["links"])
        variables = {item["name"]: item for item in dashboard["templating"]["list"]}
        assert variables["job_name"]["refresh"] == 2
        assert not variables["job_name"]["multi"] and not variables["job_name"]["includeAll"]
        assert 'job_name="$job_name"' in variables["model_name"]["query"]["query"]
        for panel in dashboard["panels"]:
            if panel["type"] == "row":
                continue
            assert panel["datasource"]["uid"] == "gpu-prometheus"
            if "Latency" in panel["title"]:
                assert panel["fieldConfig"]["defaults"]["unit"] == "s"
            if panel["title"].startswith("Input Token Size") and panel["title"] != "Input Token Size Distribution":
                assert panel["fieldConfig"]["defaults"]["unit"] == "short"
            for target in panel["targets"]:
                expression = target["expr"]
                assert "$__interval" not in expression
                metrics = re.findall(r"vllm:\w+(\{[^}]*\})?", expression)
                assert metrics
                for selector in metrics:
                    for label in [
                        'job="training-vllm"',
                        'namespace="$namespace"',
                        'job_name="$job_name"',
                        'model_name=~"$model_name"',
                    ]:
                        assert label in selector
                assert "vllm:iteration_tokens_total_count" not in expression
