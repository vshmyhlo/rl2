#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

# Use the local runtime, or an existing VisFly image on this k3s host.
if [[ -z ${VISFLY_IMAGE:-} ]]; then
  exec uv run python -m rl2.envs.visfly.generate_manifest "$@"
fi

mkdir -p data/visfly
exec sudo k3s ctr -n k8s.io run --rm \
  --runc-binary /usr/bin/nvidia-container-runtime \
  --user "$(id -u):$(id -g)" \
  --env "NVIDIA_VISIBLE_DEVICES=${VISFLY_GPU:-0}" \
  --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  --env PYTHONPATH=/app/src:/opt \
  --env JAX_PLATFORMS=cuda,cpu \
  --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  --env XDG_CACHE_HOME=/tmp/rl2-cache \
  --env XDG_CONFIG_HOME=/tmp/rl2-config \
  --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --env HABITAT_SIM_LOG=quiet --env MAGNUM_LOG=quiet \
  --mount "type=bind,src=$repo_root/src,dst=/app/src,options=rbind:ro" \
  --mount "type=bind,src=$repo_root/configs,dst=/app/configs,options=rbind:ro" \
  --mount "type=bind,src=$repo_root/data/visfly,dst=/app/data/visfly,options=rbind:rw" \
  "$VISFLY_IMAGE" "visfly-manifest-$$" \
  /app/.venv/bin/python -m rl2.envs.visfly.generate_manifest "$@"
