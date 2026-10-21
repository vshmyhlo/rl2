#!/usr/bin/env bash
set -euo pipefail

exec buildctl --addr unix:///run/rl2-buildkit/buildkitd.sock prune --all "$@"
