#!/bin/bash
#SBATCH -J chronos_full_mf
#SBATCH -p sapphire
#SBATCH -c 96
#SBATCH -t 0-09:00:00
#SBATCH --mem=512G
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
ENV_PREFIX="${ENV_PREFIX:-$HOME/chronos_fasrc/envs/chronos}"
CONFIG_PATH="${CONFIG_PATH:-configs/paths.toml}"
N_PROCESSES="${N_PROCESSES:-${SLURM_CPUS_PER_TASK:-96}}"
OUTPUT_DIRNAME="${OUTPUT_DIRNAME:-full_catalog_mf_fit_parsec_48shards_96w_1000b_10000s_20kpost_1000mass}"
SAMPLE_N_CLUSTERS="${SAMPLE_N_CLUSTERS:-}"
SAMPLE_SEED="${SAMPLE_SEED:-}"
CLUSTER_SHARD_COUNT="${CLUSTER_SHARD_COUNT:-}"
CLUSTER_SHARD_INDEX="${CLUSTER_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"

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
  --output-dirname "$OUTPUT_DIRNAME"
)

if [[ -n "$SAMPLE_N_CLUSTERS" ]]; then
  PY_ARGS+=(--sample-n-clusters "$SAMPLE_N_CLUSTERS")
fi
if [[ -n "$SAMPLE_SEED" ]]; then
  PY_ARGS+=(--sample-seed "$SAMPLE_SEED")
fi
if [[ -n "$CLUSTER_SHARD_COUNT" ]]; then
  PY_ARGS+=(--cluster-shard-count "$CLUSTER_SHARD_COUNT")
fi
if [[ -n "$CLUSTER_SHARD_INDEX" ]]; then
  PY_ARGS+=(--cluster-shard-index "$CLUSTER_SHARD_INDEX")
fi

python -m workflows.run_chronos_full_catalog_mf_fit "${PY_ARGS[@]}" "$@"
