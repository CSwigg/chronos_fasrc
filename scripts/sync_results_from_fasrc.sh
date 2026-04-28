#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="${LOCAL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REMOTE="${REMOTE:-fasrc:~/chronos_fasrc/runs/current/}"

mkdir -p "$LOCAL_DIR/runs/current"
rsync -az --progress \
  --include "*/" \
  --include "*.csv" \
  --include "*.json" \
  --include "*.txt" \
  --include "*.out" \
  --include "*.err" \
  --include "*.png" \
  --exclude "*" \
  "$REMOTE" "$LOCAL_DIR/runs/current/"
