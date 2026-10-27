#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash infra/k3s/training/cleanup-disk.sh [--dry-run | --apply] [--run-retention-days DAYS]"
  echo "Defaults to --dry-run. Stop local training/downloads and job submissions before --apply."
  echo "Removes unused BuildKit cache and unpinned, unused Kubernetes images."
  echo "Run deletion is opt-in: --run-retention-days 7 removes GRPO v2 runs inactive for over 7 days."
}

mode=--dry-run
retention_days=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|--apply) mode=$1; shift ;;
    --run-retention-days)
      if [[ ! ${2:-} =~ ^[1-9][0-9]*$ ]]; then
        echo "--run-retention-days requires a positive integer" >&2
        exit 1
      fi
      retention_days=$2
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 1 ;;
  esac
done

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

if [[ -n $retention_days ]]; then
  cutoff=$(date -u -d "$retention_days days ago" '+%Y-%m-%dT%H:%M:%S.%NZ')
  # Queued and suspended Jobs may still need their saved checkpoints to resume.
  unfinished_jobs=$("${privileged[@]}" k3s kubectl -n training get jobs -o json | python3 -c '
import json
import sys

for job in json.load(sys.stdin)["items"]:
    if not any(c["type"] in ("Complete", "Failed") and c["status"] == "True"
               for c in job.get("status", {}).get("conditions", [])):
        print(job["metadata"]["name"])
')
  if [[ -n $unfinished_jobs ]]; then
    printf 'Unfinished training Jobs exist; run retention refused:\n%s\n' "$unfinished_jobs" >&2
    exit 1
  fi
fi

printf 'Mode: %s\nUnused BuildKit cache:\n' "$mode"
if [[ $mode == --apply ]]; then
  "${privileged[@]}" bash "$training_dir/cleanup-build-cache.sh"
else
  "${privileged[@]}" buildctl --addr unix:///run/rl2-buildkit/buildkitd.sock du
fi

if [[ $mode == --apply ]]; then
  echo "Refreshing training image pins and removing unused Kubernetes images:"
  "${privileged[@]}" systemctl start rl2-image-pins.service
  "${privileged[@]}" k3s crictl rmi --prune
else
  echo "Kubernetes image inventory (apply refreshes Job pins, then prunes unused, unpinned images):"
  "${privileged[@]}" k3s crictl images --pinned
fi

# Check age before partial-checkpoint deletion changes directory timestamps.
if [[ -n $retention_days ]] && "${privileged[@]}" test -d "$runs/grpo_v2"; then
  echo "GRPO v2 runs with no modifications since $cutoff (including completed checkpoints):"
  "${privileged[@]}" find "$runs/grpo_v2" -mindepth 1 -maxdepth 1 -type d -exec bash -c '
    set -euo pipefail
    cutoff=$1
    mode=$2
    shift 2
    for run in "$@"; do
      recent=$(find "$run" -xdev -newermt "$cutoff" -print -quit)
      if [[ -z $recent ]]; then
        du -sh -- "$run"
        if [[ $mode == --apply ]]; then
          rm -rf --one-file-system -- "$run"
        fi
      fi
    done
  ' bash "$cutoff" "$mode" '{}' +
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
