# chronos_fasrc

This is a Chronos-only copy extracted from `supernovae_map` for running
the age and mass components on Harvard FASRC. It intentionally does not
include the map, paper, visualizer, orbit integration, or velocity
pipeline code.

## What Is Included

- `chronos/`: vendored Chronos fitting code and bundled PARSEC/Baraffe data.
- `workflows/run_chronos_*.py`: Chronos age and mass entry points.
- `mapper/sampling.py`: the one IMF helper needed by Chronos mass fitting.
- `data/clusters/` and `data/support/`: small CSV inputs needed by the Chronos workflows.
- `configs/paths.toml`: Chronos-only paths, with large private inputs expected in `inputs/`.
- `hpc/fasrc/`: Slurm scripts for FASRC.
- `scripts/`: rsync and environment helpers.

## Large Inputs Not Copied

Put these files in `inputs/` before running:

```bash
inputs/members-2.csv
inputs/mean_and_std_healpix.fits
```

Local symlinks are fine if you do not want duplicate 3 GB files on your laptop:

```bash
ln -s ~/Downloads/members-2.csv inputs/members-2.csv
ln -s ~/Downloads/mean_and_std_healpix.fits inputs/mean_and_std_healpix.fits
```

The input sync script follows symlinks when uploading to FASRC.

If you run MIST variants, also put MIST Gaia CMD files under:

```bash
inputs/mist_isochrones/
```

## Local Smoke Test

From this directory:

```bash
python -m workflows.run_chronos_masses --config configs/paths.toml --help
python -m workflows.run_chronos_hunt_lt150_mf_fit --config configs/paths.toml --help
```

## FASRC Setup

Sync this directory to your FASRC home:

```bash
./scripts/sync_code_to_fasrc.sh
./scripts/sync_inputs_to_fasrc.sh
```

On FASRC, create the environment from an interactive allocation:

```bash
salloc --partition test --nodes=1 --cpus-per-task=2 --mem=4GB --time=0-02:00:00
cd ~/chronos_fasrc
./scripts/create_fasrc_env.sh
```

Submit the recommended Chronos age-plus-mass run:

```bash
cd ~/chronos_fasrc
sbatch hpc/fasrc/chronos_hunt_lt150_mf_fit.sbatch
```

Monitor:

```bash
squeue -u "$USER"
tail -f logs/chronos_mf_*.out
```

Pull light results back locally:

```bash
./scripts/sync_results_from_fasrc.sh
```

## Notes

- Heavy Chronos work should run through Slurm, not on a login node.
- The Slurm script sets BLAS/OpenMP thread counts to 1 and matches Chronos workers to `SLURM_CPUS_PER_TASK`.
- `inputs/`, `runs/`, `logs/`, and `envs/` are ignored by Git so large data and results stay out of commits.
