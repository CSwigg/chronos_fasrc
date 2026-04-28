#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="${LOCAL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REMOTE="${REMOTE:-fasrc:~/chronos_fasrc/}"

rsync -az --progress \
  --exclude ".git/" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude "inputs/" \
  --exclude "runs/" \
  --include "logs/" \
  --include "logs/.gitkeep" \
  --exclude "logs/*" \
  --exclude "envs/" \
  "$LOCAL_DIR/" "$REMOTE"
