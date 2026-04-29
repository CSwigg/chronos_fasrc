#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-configs/paths.toml}"
RUN_DIRNAME="${RUN_DIRNAME:-hunt_lt150_mf_fit_all_models_96w_2000b_10000s_20kpost_1000mass}"

cd "$PROJECT_DIR"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/chronos-mpl-${USER:-user}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/chronos-xdg-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

python -m workflows.plot_chronos_fit_outputs \
  --config "$CONFIG_PATH" \
  --run-dirname "$RUN_DIRNAME" \
  "$@"
