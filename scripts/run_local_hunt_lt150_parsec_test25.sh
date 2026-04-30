#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-configs/paths.toml}"
SAMPLE_N_CLUSTERS="${SAMPLE_N_CLUSTERS:-25}"
SAMPLE_SEED="${SAMPLE_SEED:-20260430}"
MODELS="${MODELS:-parsec}"
OUTPUT_DIRNAME="${OUTPUT_DIRNAME:-hunt_lt150_mf_fit_parsec_test25_96w_2000b_10000s_20kpost_1000mass}"
PYTHON_CMD="${PYTHON_CMD:-python3}"

if [[ -z "${N_PROCESSES:-}" ]]; then
  if command -v sysctl >/dev/null 2>&1; then
    N_PROCESSES="$(sysctl -n hw.logicalcpu 2>/dev/null || true)"
  fi
  if [[ -z "${N_PROCESSES:-}" ]] && command -v nproc >/dev/null 2>&1; then
    N_PROCESSES="$(nproc)"
  fi
  if [[ -z "${N_PROCESSES:-}" ]] && command -v getconf >/dev/null 2>&1; then
    N_PROCESSES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
  fi
  if [[ -z "${N_PROCESSES:-}" ]]; then
    N_PROCESSES=1
  fi
fi

cd "$PROJECT_DIR"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/chronos-mpl-${USER:-user}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/chronos-xdg-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" logs/local_runs

read -r -a PYTHON_ARGS <<< "$PYTHON_CMD"
read -r -a MODEL_ARGS <<< "$MODELS"

LOG_PATH="logs/local_runs/${OUTPUT_DIRNAME}_$(date +%Y%m%d_%H%M%S).log"
echo "Launching local Chronos timing test | clusters=${SAMPLE_N_CLUSTERS} | models=${MODELS} | n_processes=${N_PROCESSES} | output=${OUTPUT_DIRNAME}"
echo "Writing local run log: ${LOG_PATH}"

SECONDS=0
set +e
"${PYTHON_ARGS[@]}" -m workflows.run_chronos_hunt_lt150_mf_fit \
  --config "$CONFIG_PATH" \
  --n-processes "$N_PROCESSES" \
  --models "${MODEL_ARGS[@]}" \
  --output-dirname "$OUTPUT_DIRNAME" \
  --sample-n-clusters "$SAMPLE_N_CLUSTERS" \
  --sample-seed "$SAMPLE_SEED" \
  "$@" 2>&1 | tee "$LOG_PATH"
status="${PIPESTATUS[0]}"
set -e

elapsed_seconds="$SECONDS"
printf "Local Chronos timing test exited with status %s after %02d:%02d:%02d\n" \
  "$status" \
  "$((elapsed_seconds / 3600))" \
  "$(((elapsed_seconds % 3600) / 60))" \
  "$((elapsed_seconds % 60))" | tee -a "$LOG_PATH"
exit "$status"
