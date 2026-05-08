#!/bin/bash
#SBATCH -J chronos_finalize
#SBATCH -p shared
#SBATCH -c 1
#SBATCH -t 0-01:00:00
#SBATCH --mem=8G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
ENV_PREFIX="${ENV_PREFIX:-$HOME/chronos_fasrc/envs/chronos}"
CONFIG_PATH="${CONFIG_PATH:-configs/paths.toml}"
OUTPUT_DIRNAME="${OUTPUT_DIRNAME:?OUTPUT_DIRNAME is required}"

module purge
module load python
source activate "$ENV_PREFIX"

cd "$PROJECT_DIR"
mkdir -p logs

export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mpl-$USER-${SLURM_JOB_ID:-manual}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/xdg-$USER-${SLURM_JOB_ID:-manual}}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -m workflows.finalize_chronos_run \
  --config "$CONFIG_PATH" \
  --output-dirname "$OUTPUT_DIRNAME"
