#!/usr/bin/env bash
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo: sudo bash infra/k3s/port-forwards/install.sh" >&2
  exit 1
fi

forward_tmp=$(mktemp -d)
trap 'rm -rf -- "$forward_tmp"' EXIT
units=()
while read -r name namespace service ports; do
  if [[ $# -gt 0 && " $* " != *" $name "* ]]; then
    continue
  fi
  unit="rl2-forward-$name.service"
  units+=("$unit")
  cat > "$forward_tmp/$unit" <<EOF
[Unit]
Description=Localhost port-forward for $name
Wants=k3s.service
After=k3s.service
StartLimitIntervalSec=0

[Service]
ExecStart=/usr/local/bin/k3s kubectl -n $namespace port-forward --address=127.0.0.1 service/$service $ports
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
done <<'EOF'
headlamp headlamp headlamp 8080:80
mlflow mlflow mlflow 8081:5000
grafana monitoring gpu-grafana 8082:80
kueue-frontend kueue-system kueue-kueueviz-frontend 8083:8080
kueue-backend kueue-system kueue-kueueviz-backend 8084:8080
ray-timelines training ray-timelines 8085:8085
EOF

if [[ ${#units[@]} -eq 0 ]]; then
  echo "No matching port-forwards" >&2
  exit 1
fi

systemd-analyze verify "$forward_tmp/"*.service
# Unload temporary systemd-run units before installing their replacements.
for unit in "${units[@]}"; do
  if [[ $(systemctl show "$unit" -p Transient --value) == yes ]]; then
    systemctl stop "$unit"
  fi
done
install -m 0644 "$forward_tmp/"*.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable "${units[@]}"
systemctl restart "${units[@]}"
