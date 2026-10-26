#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash infra/k3s/training/cleanup-disk.sh [--dry-run | --apply]"
  echo "Defaults to --dry-run. Stop local training/downloads and job submissions before --apply."
}

mode=${1:---dry-run}
if [[ $# -gt 1 ]]; then
  usage >&2
  exit 1
fi
case "$mode" in
  --dry-run|--apply) ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 1 ;;
esac

privileged=()
if [[ $EUID -ne 0 ]]; then
  privileged=(sudo)
fi
training_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
user_home=$HOME
if [[ -n ${SUDO_USER:-} ]]; then
  user_home=$(getent passwd "$SUDO_USER" | cut -d: -f6)
  [[ -n $user_home ]] || exit 1
fi
hf_home=${HF_HOME:-${XDG_CACHE_HOME:-$user_home/.cache}/huggingface}
if [[ $hf_home != /* || $hf_home == / ]]; then
  echo "Hugging Face cache must have an absolute, non-root path: $hf_home" >&2
  exit 1
fi

pv=$("${privileged[@]}" k3s kubectl -n training get pvc training-runs -o jsonpath='{.spec.volumeName}')
[[ -n $pv ]] || { echo "training-runs PVC is not bound" >&2; exit 1; }
runs=$("${privileged[@]}" k3s kubectl get pv "$pv" -o jsonpath='{.spec.local.path}{.spec.hostPath.path}')
case "$runs" in
  /*/pvc-*_training_training-runs) ;;
  *) echo "Unexpected training volume path: $runs" >&2; exit 1 ;;
esac
"${privileged[@]}" test -d "$runs"

# Terminal pods cannot write checkpoints; Pending/Unknown pods are also blockers.
active_pods=$("${privileged[@]}" k3s kubectl -n training get pods \
  --field-selector='status.phase!=Succeeded,status.phase!=Failed' \
  -l 'job-name' -o name)
if [[ -n $active_pods ]]; then
  printf 'Training pods are still active; cleanup refused:\n%s\n' "$active_pods" >&2
  exit 1
fi

printf 'Mode: %s\nUnused BuildKit cache:\n' "$mode"
if [[ $mode == --apply ]]; then
  "${privileged[@]}" bash "$training_dir/cleanup-build-cache.sh"
else
  "${privileged[@]}" buildctl --addr unix:///run/rl2-buildkit/buildkitd.sock du
fi

echo "Incomplete checkpoints under $runs:"
find_action=(-print)
if [[ $mode == --apply ]]; then
  find_action+=(-exec rm -rf --one-file-system -- '{}' +)
fi
"${privileged[@]}" find "$runs" -xdev -type d -path '*/checkpoints/*.orbax-checkpoint-tmp' \
  -prune "${find_action[@]}"

# Keep tokens and other authentication files stored directly in HF_HOME.
for cache in "$hf_home/hub" "$hf_home/xet" "$hf_home/assets"; do
  if [[ -e $cache || -L $cache ]]; then
    du -sh -- "$cache"
    if [[ $mode == --apply ]]; then
      "${privileged[@]}" rm -rf --one-file-system -- "$cache"
    fi
  fi
done
