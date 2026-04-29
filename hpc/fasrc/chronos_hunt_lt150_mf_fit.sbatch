#!/bin/bash
#SBATCH -J chronos_mf
#SBATCH -p shared
#SBATCH -c 8
#SBATCH -t 2-00:00:00
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
ENV_PREFIX="${ENV_PREFIX:-$HOME/chronos_fasrc/envs/chronos}"
CONFIG_PATH="${CONFIG_PATH:-configs/paths.toml}"
N_PROCESSES="${N_PROCESSES:-${SLURM_CPUS_PER_TASK:-8}}"
OUTPUT_DIRNAME="${OUTPUT_DIRNAME:-}"
SAMPLE_N_CLUSTERS="${SAMPLE_N_CLUSTERS:-}"
SAMPLE_SEED="${SAMPLE_SEED:-}"

module purge
module load python
source activate "$ENV_PREFIX"

cd "$PROJECT_DIR"
mkdir -p logs runs/current/chronos

export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mpl-$USER-${SLURM_JOB_ID:-manual}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/xdg-$USER-${SLURM_JOB_ID:-manual}}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY_ARGS=(
  --config "$CONFIG_PATH"
  --n-processes "$N_PROCESSES"
)

if [[ -n "$OUTPUT_DIRNAME" ]]; then
  PY_ARGS+=(--output-dirname "$OUTPUT_DIRNAME")
fi
if [[ -n "$SAMPLE_N_CLUSTERS" ]]; then
  PY_ARGS+=(--sample-n-clusters "$SAMPLE_N_CLUSTERS")
fi
if [[ -n "$SAMPLE_SEED" ]]; then
  PY_ARGS+=(--sample-seed "$SAMPLE_SEED")
fi

python -m workflows.run_chronos_hunt_lt150_mf_fit "${PY_ARGS[@]}" "$@"
