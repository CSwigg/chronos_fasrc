# AGENTS.md

This directory is a Chronos-only extraction from `supernovae_map` for FASRC runs.

## Scope

- Keep changes focused on Chronos age/mass fitting and FASRC execution.
- Do not reintroduce map, paper, orbit integration, visualizer, or velocity-pipeline code unless explicitly asked.
- Large private inputs belong in `inputs/`; generated products belong in `runs/` and `logs/`.
- Preserve headless plotting with `MPLBACKEND=Agg`.
- For Slurm jobs, keep worker count tied to `SLURM_CPUS_PER_TASK` and set BLAS/OpenMP thread env vars to 1.

## Current Goal

Run Chronos on the full cluster catalog on Harvard FASRC/Cannon, using PARSEC only, then bring saved fit products back to the local machine for analysis and plotting.

The important output is not FASRC-side figures. The important output is:

- `cluster_results.csv`
- per-cluster checkpoints
- per-cluster downsampled posterior `.npz` files
- sampler diagnostics JSON
- mass-function-fit draw CSVs
- dust-map provenance columns

Plotting and scientific inspection should happen locally after syncing results back.

## Next Full-Catalog Run Ordering

For the next Chronos full-catalog PARSEC run, prioritize scientifically urgent clusters before exhausting the whole catalog:

- first: Hunt `age_myr < 200` Myr and inside a Sun-centered `2 x 2 x 2` kpc cube, implemented as `|x|, |y|, |z| <= 1000 pc`
- within that priority set: younger Hunt ages first
- after that: remaining Hunt `age_myr < 200` Myr clusters, still younger first, with location metadata retained
- then: older finite-Hunt-age clusters, younger first
- finally: clusters without finite Hunt ages

Use `CATALOG_ORDER=hunt_young_solar_box` and `CLUSTER_SHARD_STRATEGY=contiguous` for this run. Contiguous shards matter when the Slurm array is throttled, because the early array tasks then cover the earliest priority-ranked chunks instead of round-robin sampling the whole catalog.

## Production Scientific Settings

For the current full-catalog PARSEC run:

- clusters: all clusters in `data/clusters/hunt_partII_partIII_merged.csv`
- model: `parsec` only
- no Hunt age cut
- no RV cut
- no `x/y/z` box cut
- age range: `1e6` to `1.2e10` yr, passed as `--age-min-myr 1 --age-max-myr 12000`
- age prior: flat in linear age, passed as `--age-prior linear`
- walkers: `46`
- burn-in: `100`
- production steps: `1000`
- saved posterior samples: `20000` randomly selected flattened samples per cluster/model
- mass posterior draws: `1000`
- IMF draws: `1000`

Do not casually change these settings; they define the comparable production run.

## Dust Prior

The `A_V` prior is dust-map informed and bounded by the Chronos sampler to `0 <= A_V <= 5`.

Dust-map fallback order:

1. Edenhofer 2023, using `inputs/mean_and_std_healpix.fits`
2. Bayestar 2019, using `inputs/dustmaps/bayestar/bayestar2019.h5`
3. DECaPS mean map, using `inputs/dustmaps/decaps/decaps_mean.h5`

For each member star, the code uses the first map in that list that returns a finite value at the star's position and distance. The cluster-level prior is then built from member-star `A_V` values:

- if all member stars have valid values: Gaussian around the median member `A_V`, with `sigma_Av = 0.10`
- if some are outside map support: lower-limit style prior using valid/floor extinction estimates

The results must preserve dust provenance through:

- `prior_map_name`
- `prior_map_counts`
- `prior_floor_map_counts`
- `prior_mode`
- `prior_center_av`
- `prior_floor_av`
- `prior_sigma_av`

## FASRC Data Requirements

The FASRC clone needs the large private inputs under `~/chronos_fasrc/inputs/`.

Required for the current run:

```text
inputs/members-2.csv
inputs/mean_and_std_healpix.fits
inputs/dustmaps/bayestar/bayestar2019.h5
inputs/dustmaps/decaps/decaps_mean.h5
inputs/parsec_isochrones_hybrid_0p1myr_to13gyr/*.dat
```

The intended PARSEC grid is generated with `scripts/download_parsec_isochrones.py`.
It uses CMD 3.7 / PARSEC v1.2S / Gaia EDR3. As of May 2026 the live CMD 3.7
service only accepts Gaia EDR3 with `photsys_version=odfnew`; the script records
this in `manifest.json`.

- one anchor isochrone at `0.1` Myr
- linear `1` Myr spacing from `1` to `300` Myr
- log spacing with `0.02` dex steps after `300` Myr
- an exact `13` Gyr endpoint
- `382` actual unique ages across `21` CMD output files, about `333 MB`
- current [M/H] groups matching the previous Chronos grid:
  `[-1.5, -1.3, -1.1, -0.9, -0.5, -0.3, -0.1, 0.1, 0.3]`

Do not download a PARSEC extinction grid by default. Chronos loads unreddened
PARSEC tables and applies `A_V` continuously at runtime. The desired `0.1` mag
extinction spacing is a reference scale for checks/diagnostics; precomputing
`A_V=0..5` in 0.1 mag CMD files would multiply the grid by 51 and would require
a separate 4D isochrone interpolator.

Expected dust-map sizes:

```text
bayestar2019.h5  ~694M
decaps_mean.h5   ~8.0G
```

The direct Dataverse download on FASRC previously failed checksum, so copying these files from the local machine with resumable `rsync` may be necessary.

## Current FASRC Run

As of 2026-05-12, the submitted production run is:

```text
array job:    12580507
finalize job: 12580508
run dirname:  full_catalog_mf_fit_parsec_144shards_96w_1000b_10000s_20kpost_1000mass_linearage
```

Submitted with:

```bash
ACCOUNT=itc_lab SHARD_COUNT=144 ARRAY_CONCURRENCY=48 TIME_LIMIT=0-09:00:00 \
  ./scripts/submit_fasrc_full_catalog_parsec_mf_fit.sh
```

This means:

- `144` total shards
- up to `48` shards running at once
- each shard requests `96` CPU cores and `512G` memory
- each shard gets up to `9` hours
- total live request at full throttle is `4608` cores and `24T` requested memory

This is intended to finish the full catalog in fewer waves while still avoiding an all-at-once 144-shard request.

## FASRC Monitoring

Check queue state:

```bash
squeue -u cswiggum
squeue -j 12580507
```

Check output progress:

```bash
RUN=full_catalog_mf_fit_parsec_144shards_96w_1000b_10000s_20kpost_1000mass_linearage

find ~/chronos_fasrc/runs/current/chronos/$RUN/checkpoints -name '*.json' | wc -l
find ~/chronos_fasrc/runs/current/chronos/$RUN/parsec/posterior_samples -name '*.npz' | wc -l
find ~/chronos_fasrc/runs/current/chronos/$RUN/parsec/sampler_diagnostics -name '*.json' | wc -l
```

Check disk usage:

```bash
df -h ~
du -sh ~/chronos_fasrc
du -sh ~/chronos_fasrc/inputs
du -sh ~/chronos_fasrc/runs
```

## Local Validation Already Done

A local 5-cluster full-MCMC test was run with the production settings:

```text
run dirname: local_fullmcmc_parsec_5clusters_linearage_dustmaps
runtime:     15:48 local
status:      5/5 clusters succeeded
```

It produced:

- 5 posterior sample files
- 5 sampler diagnostics files
- 5 posterior plots
- 5 isochrone fit plots

The test exercised all dust-map paths:

```text
HSC_2329   DECaPS
HSC_2909   DECaPS
NGC_2483   Bayestar 2019
UBC_1103   Bayestar 2019
UPK_31     Edenhofer 2023
```

## Main Commands

```bash
python -m workflows.run_chronos_hunt_lt150_mf_fit --config configs/paths.toml --n-processes 8
python -m workflows.run_chronos_full_catalog_mf_fit --config configs/paths.toml --n-processes 8
DRY_RUN=1 ACCOUNT=itc_lab ./scripts/submit_fasrc_full_catalog_parsec_mf_fit.sh
```
