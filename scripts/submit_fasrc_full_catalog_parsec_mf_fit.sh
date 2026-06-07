#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
RUN_NAME="${RUN_NAME:-}"
SHARD_COUNT="${SHARD_COUNT:-1}"
PARTITION="${PARTITION:-sapphire}"
CPUS_PER_TASK="${CPUS_PER_TASK:-96}"
MEM="${MEM:-512G}"
TIME_LIMIT="${TIME_LIMIT:-0-12:00:00}"
ACCOUNT="${ACCOUNT:-}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-}"
DRY_RUN="${DRY_RUN:-0}"
CATALOG_ORDER="${CATALOG_ORDER:-hunt_young_solar_box}"
CLUSTER_SHARD_STRATEGY="${CLUSTER_SHARD_STRATEGY:-contiguous}"
PRIORITY_HUNT_AGE_MAX_MYR="${PRIORITY_HUNT_AGE_MAX_MYR:-200}"
PRIORITY_BOX_HALF_WIDTH_PC="${PRIORITY_BOX_HALF_WIDTH_PC:-1000}"
FILTER_HUNT_AGE_MAX_MYR="${FILTER_HUNT_AGE_MAX_MYR:-}"
FILTER_XY_HALF_WIDTH_PC="${FILTER_XY_HALF_WIDTH_PC:-}"
MODELS="${MODELS:-parsec}"

NWALKERS="${NWALKERS:-46}"
AGE_MIN_MYR="${AGE_MIN_MYR:-1}"
AGE_MAX_MYR="${AGE_MAX_MYR:-12000}"
AGE_PRIOR="${AGE_PRIOR:-linear}"
AV_PRIOR="${AV_PRIOR:-dust}"
AV_MIN_MAG="${AV_MIN_MAG:-0}"
AV_MAX_MAG="${AV_MAX_MAG:-}"
BURNIN="${BURNIN:-100}"
NSTEPS="${NSTEPS:-1000}"
MASS_DRAWS="${MASS_DRAWS:-1000}"
N_IMFS="${N_IMFS:-1000}"
POSTERIOR_SAMPLE_SIZE="${POSTERIOR_SAMPLE_SIZE:-20000}"

if [[ -z "$RUN_NAME" ]]; then
  model_label="${MODELS// /_}"
  posterior_label="$POSTERIOR_SAMPLE_SIZE"
  if [[ "$POSTERIOR_SAMPLE_SIZE" == "20000" ]]; then
    posterior_label="20k"
  fi
  age_max_label="${AGE_MAX_MYR//./p}"
  av_label="$AV_PRIOR"
  if [[ -n "$AV_MAX_MAG" ]]; then
    av_min_label="${AV_MIN_MAG//./p}"
    av_max_label="${AV_MAX_MAG//./p}"
    av_label="${AV_PRIOR}av${av_min_label}to${av_max_label}"
  fi
  order_label="$CATALOG_ORDER"
  if [[ "$CATALOG_ORDER" == "hunt_young_solar_box" ]]; then
    priority_age_label="${PRIORITY_HUNT_AGE_MAX_MYR//./p}"
    box_half_label="${PRIORITY_BOX_HALF_WIDTH_PC//./p}"
    order_label="huntlt${priority_age_label}myr_boxhalf${box_half_label}pcfirst"
  fi
  filter_label="allclusters"
  if [[ -n "$FILTER_HUNT_AGE_MAX_MYR" || -n "$FILTER_XY_HALF_WIDTH_PC" ]]; then
    filter_label="filtered"
    if [[ -n "$FILTER_HUNT_AGE_MAX_MYR" ]]; then
      filter_age_label="${FILTER_HUNT_AGE_MAX_MYR//./p}"
      filter_label="${filter_label}_huntlt${filter_age_label}myr"
    fi
    if [[ -n "$FILTER_XY_HALF_WIDTH_PC" ]]; then
      filter_xy_label="${FILTER_XY_HALF_WIDTH_PC//./p}"
      filter_label="${filter_label}_xyhalf${filter_xy_label}pc"
    fi
  fi
  if [[ "$SHARD_COUNT" -eq 1 ]]; then
    shard_label="unsharded"
  else
    shard_label="${SHARD_COUNT}shards_${CLUSTER_SHARD_STRATEGY}shards"
  fi
  RUN_NAME="full_catalog_mf_fit_${model_label}_${filter_label}_${shard_label}_${NWALKERS}w_${BURNIN}b_${NSTEPS}s_${posterior_label}post_${MASS_DRAWS}mass_${AGE_PRIOR}age_agemax${age_max_label}myr_${av_label}_${order_label}"
fi

if [[ "$AV_PRIOR" != "dust" && "$AV_PRIOR" != "flat" ]]; then
  echo "AV_PRIOR must be 'dust' or 'flat'; got '$AV_PRIOR'" >&2
  exit 1
fi
if [[ "$AV_PRIOR" == "flat" && -z "$AV_MAX_MAG" ]]; then
  echo "AV_PRIOR=flat requires AV_MAX_MAG, e.g. AV_MAX_MAG=5" >&2
  exit 1
fi
if [[ "$RUN_NAME" == *flatav* && "$AV_PRIOR" != "flat" ]]; then
  echo "RUN_NAME contains 'flatav' but AV_PRIOR=$AV_PRIOR. Set AV_PRIOR=flat AV_MIN_MAG=0 AV_MAX_MAG=5." >&2
  exit 1
fi

if [[ ! -f "$PROJECT_DIR/hpc/fasrc/chronos_full_catalog_mf_fit.sbatch" ]]; then
  echo "Could not find hpc/fasrc/chronos_full_catalog_mf_fit.sbatch under PROJECT_DIR=$PROJECT_DIR" >&2
  exit 1
fi

if [[ "$SHARD_COUNT" -le 0 ]]; then
  echo "SHARD_COUNT must be positive; got $SHARD_COUNT" >&2
  exit 1
fi

account_args=()
if [[ -n "$ACCOUNT" ]]; then
  account_args=(-A "$ACCOUNT")
fi

shard_export="N_PROCESSES=$CPUS_PER_TASK"
if [[ "$SHARD_COUNT" -gt 1 ]]; then
  array_spec="0-$((SHARD_COUNT - 1))"
  if [[ -n "$ARRAY_CONCURRENCY" ]]; then
    array_spec="${array_spec}%${ARRAY_CONCURRENCY}"
  fi
  shard_export="CLUSTER_SHARD_COUNT=$SHARD_COUNT,CLUSTER_SHARD_STRATEGY=$CLUSTER_SHARD_STRATEGY,$shard_export"
fi

read -r -a model_args <<< "$MODELS"

array_cmd=(
  sbatch
  --parsable
  "${account_args[@]}"
  -p "$PARTITION"
  -c "$CPUS_PER_TASK"
  --mem "$MEM"
  -t "$TIME_LIMIT"
)
if [[ "$SHARD_COUNT" -gt 1 ]]; then
  array_cmd+=(--array "$array_spec")
fi
array_cmd+=(
  --export "ALL,PROJECT_DIR=$PROJECT_DIR,OUTPUT_DIRNAME=$RUN_NAME,$shard_export,CATALOG_ORDER=$CATALOG_ORDER,PRIORITY_HUNT_AGE_MAX_MYR=$PRIORITY_HUNT_AGE_MAX_MYR,PRIORITY_BOX_HALF_WIDTH_PC=$PRIORITY_BOX_HALF_WIDTH_PC"
  "$PROJECT_DIR/hpc/fasrc/chronos_full_catalog_mf_fit.sbatch"
  --models "${model_args[@]}"
  --age-min-myr "$AGE_MIN_MYR"
  --age-max-myr "$AGE_MAX_MYR"
  --age-prior "$AGE_PRIOR"
  --av-prior "$AV_PRIOR"
  --av-min-mag "$AV_MIN_MAG"
  --nwalkers "$NWALKERS"
  --burnin "$BURNIN"
  --nsteps "$NSTEPS"
  --mass-draws "$MASS_DRAWS"
  --n-imfs "$N_IMFS"
  --posterior-sample-size "$POSTERIOR_SAMPLE_SIZE"
)
if [[ -n "$AV_MAX_MAG" ]]; then
  array_cmd+=(--av-max-mag "$AV_MAX_MAG")
fi
if [[ -n "$FILTER_HUNT_AGE_MAX_MYR" ]]; then
  array_cmd+=(--filter-hunt-age-max-myr "$FILTER_HUNT_AGE_MAX_MYR")
fi
if [[ -n "$FILTER_XY_HALF_WIDTH_PC" ]]; then
  array_cmd+=(--filter-xy-half-width-pc "$FILTER_XY_HALF_WIDTH_PC")
fi

if [[ "$SHARD_COUNT" -eq 1 ]]; then
  echo "Submitting Chronos full-catalog PARSEC job:"
else
  echo "Submitting Chronos full-catalog PARSEC array:"
fi
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
if [[ "$SHARD_COUNT" -eq 1 ]]; then
  echo "Chronos job: $array_job_id"
else
  echo "Array job: $array_job_id"
fi

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
