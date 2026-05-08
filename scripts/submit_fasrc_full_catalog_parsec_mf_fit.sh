#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
RUN_NAME="${RUN_NAME:-}"
SHARD_COUNT="${SHARD_COUNT:-48}"
PARTITION="${PARTITION:-sapphire}"
CPUS_PER_TASK="${CPUS_PER_TASK:-96}"
MEM="${MEM:-512G}"
TIME_LIMIT="${TIME_LIMIT:-0-09:00:00}"
ACCOUNT="${ACCOUNT:-}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-}"
DRY_RUN="${DRY_RUN:-0}"

NWALKERS="${NWALKERS:-96}"
AGE_MIN_MYR="${AGE_MIN_MYR:-1}"
AGE_MAX_MYR="${AGE_MAX_MYR:-1000}"
AGE_PRIOR="${AGE_PRIOR:-linear}"
BURNIN="${BURNIN:-1000}"
NSTEPS="${NSTEPS:-10000}"
MASS_DRAWS="${MASS_DRAWS:-1000}"
N_IMFS="${N_IMFS:-1000}"
POSTERIOR_SAMPLE_SIZE="${POSTERIOR_SAMPLE_SIZE:-20000}"

if [[ -z "$RUN_NAME" ]]; then
  posterior_label="$POSTERIOR_SAMPLE_SIZE"
  if [[ "$POSTERIOR_SAMPLE_SIZE" == "20000" ]]; then
    posterior_label="20k"
  fi
  RUN_NAME="full_catalog_mf_fit_parsec_${SHARD_COUNT}shards_${NWALKERS}w_${BURNIN}b_${NSTEPS}s_${posterior_label}post_${MASS_DRAWS}mass_${AGE_PRIOR}age"
fi

if [[ ! -f "$PROJECT_DIR/hpc/fasrc/chronos_full_catalog_mf_fit.sbatch" ]]; then
  echo "Could not find hpc/fasrc/chronos_full_catalog_mf_fit.sbatch under PROJECT_DIR=$PROJECT_DIR" >&2
  exit 1
fi

if [[ "$SHARD_COUNT" -le 0 ]]; then
  echo "SHARD_COUNT must be positive; got $SHARD_COUNT" >&2
  exit 1
fi

array_spec="0-$((SHARD_COUNT - 1))"
if [[ -n "$ARRAY_CONCURRENCY" ]]; then
  array_spec="${array_spec}%${ARRAY_CONCURRENCY}"
fi

account_args=()
if [[ -n "$ACCOUNT" ]]; then
  account_args=(-A "$ACCOUNT")
fi

array_cmd=(
  sbatch
  --parsable
  "${account_args[@]}"
  -p "$PARTITION"
  -c "$CPUS_PER_TASK"
  --mem "$MEM"
  -t "$TIME_LIMIT"
  --array "$array_spec"
  --export "ALL,PROJECT_DIR=$PROJECT_DIR,OUTPUT_DIRNAME=$RUN_NAME,CLUSTER_SHARD_COUNT=$SHARD_COUNT,N_PROCESSES=$CPUS_PER_TASK"
  "$PROJECT_DIR/hpc/fasrc/chronos_full_catalog_mf_fit.sbatch"
  --models parsec
  --age-min-myr "$AGE_MIN_MYR"
  --age-max-myr "$AGE_MAX_MYR"
  --age-prior "$AGE_PRIOR"
  --nwalkers "$NWALKERS"
  --burnin "$BURNIN"
  --nsteps "$NSTEPS"
  --mass-draws "$MASS_DRAWS"
  --n-imfs "$N_IMFS"
  --posterior-sample-size "$POSTERIOR_SAMPLE_SIZE"
)

echo "Submitting Chronos full-catalog PARSEC array:"
printf ' %q' "${array_cmd[@]}"
echo

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1; not submitting jobs."
  echo
  echo "Finalize command template, after replacing <array_job_id>:"
  finalize_template=(
    sbatch
    --parsable
    "${account_args[@]}"
    -p shared
    -c 1
    --mem 8G
    -t 0-01:00:00
    --dependency "afterany:<array_job_id>"
    --export "ALL,PROJECT_DIR=$PROJECT_DIR,OUTPUT_DIRNAME=$RUN_NAME"
    "$PROJECT_DIR/hpc/fasrc/chronos_finalize_run.sbatch"
  )
  printf ' %q' "${finalize_template[@]}"
  echo
  exit 0
fi

array_job_id="$("${array_cmd[@]}")"
echo "Array job: $array_job_id"

finalize_cmd=(
  sbatch
  --parsable
  "${account_args[@]}"
  -p shared
  -c 1
  --mem 8G
  -t 0-01:00:00
  --dependency "afterany:$array_job_id"
  --export "ALL,PROJECT_DIR=$PROJECT_DIR,OUTPUT_DIRNAME=$RUN_NAME"
  "$PROJECT_DIR/hpc/fasrc/chronos_finalize_run.sbatch"
)

echo "Submitting finalize job:"
printf ' %q' "${finalize_cmd[@]}"
echo

finalize_job_id="$("${finalize_cmd[@]}")"
echo "Finalize job: $finalize_job_id"
echo "Run directory: runs/current/chronos/$RUN_NAME"
