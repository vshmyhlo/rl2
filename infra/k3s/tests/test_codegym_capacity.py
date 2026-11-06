from pathlib import Path

import yaml


def test_network_covers_worker_pool() -> None:
    root = Path(__file__).resolve().parents[3]
    resources = {
        resource["kind"]: resource
        for resource in yaml.safe_load_all((root / "infra/k3s/codegym/codegym.yaml").read_text())
    }
    deployment = resources["Deployment"]["spec"]
    assert deployment["replicas"] == 1
    args = deployment["template"]["spec"]["containers"][0]["args"]
    workers = int(args[args.index("--workers") + 1])
    ports = resources["NetworkPolicy"]["spec"]["ingress"][0]["ports"]
    worker_ports = next(port for port in ports if port["port"] == 10000)
    assert worker_ports["endPort"] >= 10000 + workers - 1
