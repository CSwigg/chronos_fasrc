#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_PREFIX="${ENV_PREFIX:-$HOME/chronos_fasrc/envs/chronos}"

module purge
module load python

mkdir -p "$(dirname "$ENV_PREFIX")"
cd "$PROJECT_DIR"

if [[ ! -d "$ENV_PREFIX" ]]; then
  mamba env create -p "$ENV_PREFIX" -f environment.yml
fi

source activate "$ENV_PREFIX"
python -m pip install -e .
python - <<'PY'
import chronos
import workflows.run_chronos_hunt_lt150_mf_fit
print("Chronos FASRC environment import check passed.")
PY
