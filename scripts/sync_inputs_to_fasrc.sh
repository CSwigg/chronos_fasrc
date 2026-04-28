#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="${LOCAL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REMOTE="${REMOTE:-fasrc:~/chronos_fasrc/inputs/}"

rsync -azL --progress "$LOCAL_DIR/inputs/" "$REMOTE"
