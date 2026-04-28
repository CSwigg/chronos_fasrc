# AGENTS.md

This directory is a Chronos-only extraction from `supernovae_map` for FASRC runs.

## Scope

- Keep changes focused on Chronos age/mass fitting and FASRC execution.
- Do not reintroduce map, paper, orbit integration, visualizer, or velocity-pipeline code unless explicitly asked.
- Large private inputs belong in `inputs/`; generated products belong in `runs/` and `logs/`.
- Preserve headless plotting with `MPLBACKEND=Agg`.
- For Slurm jobs, keep worker count tied to `SLURM_CPUS_PER_TASK` and set BLAS/OpenMP thread env vars to 1.

## Main Commands

```bash
python -m workflows.run_chronos_hunt_lt150_mf_fit --config configs/paths.toml --n-processes 8
python -m workflows.run_chronos_dual_model_with_masses --config configs/paths.toml --n-processes 8
python -m workflows.run_chronos_masses --config configs/paths.toml
```
