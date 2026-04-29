#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-configs/paths.toml}"
N_PROCESSES="${N_PROCESSES:-1}"
OUTPUT_DIRNAME="${OUTPUT_DIRNAME:-hunt_lt150_mf_fit_all_models_test50_96w_2000b_10000s_20kpost_1000mass}"
SAMPLE_SEED="${SAMPLE_SEED:-20260428}"
PYTHON_CMD="${PYTHON_CMD:-python3}"

cd "$PROJECT_DIR"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/chronos-mpl-${USER:-user}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/chronos-xdg-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

read -r -a PYTHON_ARGS <<< "$PYTHON_CMD"

time "${PYTHON_ARGS[@]}" -m workflows.run_chronos_hunt_lt150_mf_fit \
  --config "$CONFIG_PATH" \
  --n-processes "$N_PROCESSES" \
  --output-dirname "$OUTPUT_DIRNAME" \
  --sample-n-clusters 50 \
  --sample-seed "$SAMPLE_SEED" \
  "$@"
