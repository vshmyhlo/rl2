#!/usr/bin/env bash
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo: sudo bash infra/k3s/ray-timelines/install.sh" >&2
  exit 1
fi

viewer_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
k3s kubectl -n training create configmap ray-timeline-viewer \
  --from-file="server.py=$viewer_dir/server.py" \
  --from-file="index.html=$viewer_dir/index.html" \
  --dry-run=client -o yaml | k3s kubectl apply -f -
k3s kubectl apply -f "$viewer_dir/service.yaml"
k3s kubectl -n training rollout restart deployment/ray-timelines
k3s kubectl -n training rollout status deployment/ray-timelines --timeout=120s
bash "$viewer_dir/../port-forwards/install.sh" ray-timelines
