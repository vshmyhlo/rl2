#!/usr/bin/env bash
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo: sudo bash infra/k3s/training/install.sh" >&2
  exit 1
fi
if [[ $(uname -m) != x86_64 ]]; then
  echo "This installer targets the local Linux amd64 k3s host." >&2
  exit 1
fi

training_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
buildkit_tmp=$(mktemp -d)
trap 'rm -rf -- "$buildkit_tmp"' EXIT
curl -fsSL https://github.com/moby/buildkit/releases/download/v0.34.0/buildkit-v0.34.0.linux-amd64.tar.gz \
  -o "$buildkit_tmp/buildkit.tar.gz"
printf '14813b1fb2a40082114e6d386c6fc5e7dd5298176ce7ee1a18bc33e73e109ac3  %s\n' \
  "$buildkit_tmp/buildkit.tar.gz" | sha256sum --check
tar -xzf "$buildkit_tmp/buildkit.tar.gz" -C "$buildkit_tmp" bin/buildctl bin/buildkitd
install -m 0755 "$buildkit_tmp/bin/buildctl" "$buildkit_tmp/bin/buildkitd" /usr/local/bin/
install -m 0644 "$training_dir/buildkitd.toml" /etc/rl2-buildkit.toml
install -m 0644 "$training_dir/rl2-buildkit.service" /etc/systemd/system/rl2-buildkit.service
systemctl daemon-reload
systemctl enable rl2-buildkit
systemctl restart rl2-buildkit
k3s kubectl apply -f "$training_dir/priorities.yaml"
k3s kubectl apply -f "$training_dir/storage.yaml"
