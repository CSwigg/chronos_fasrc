#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _set_cache_env() -> None:
    tmp = Path(os.environ.get("TMPDIR", "/private/tmp"))
    user = os.environ.get("USER", "user")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(tmp / f"chronos-mpl-{user}"))
    os.environ.setdefault("XDG_CACHE_HOME", str(tmp / f"chronos-xdg-{user}"))


def main() -> None:
    _set_cache_env()

    from chronos.run_chronos.dual_model import DualModelRunConfig, run_dual_model_refit
    from chronos.run_chronos.pipeline import ChronosFitConfig

    parser = argparse.ArgumentParser(
        description="Run a 10-cluster local Chronos comparison with aggressive MCMC settings."
    )
    parser.add_argument("--config", default="configs/paths.toml")
    parser.add_argument("--n-processes", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--output-dirname", default="local_mcmc46_100b_1000s_compare10_to_heavy")
    args = parser.parse_args()

    clusters = [
        "NGC_2422",
        "Collinder_135",
        "Theia_35",
        "HSC_729",
        "Theia_177",
        "HSC_2630",
        "HSC_2810",
        "FSR_0398",
        "Theia_652",
        "UPK_422",
    ]
    run_config = DualModelRunConfig(
        fit_config=ChronosFitConfig(
            models="parsec",
            age_range_myr=(1.0, 1000.0),
            nwalkers=46,
            burnin=100,
            nsteps=1000,
        ),
        age_prior="linear",
        include_swiggum_masses=True,
        mass_n_draws=1000,
        mass_n_imfs=1000,
        mass_output_prefix="mass_mf_fit",
        save_mass_draws=True,
        save_mass_diagnostic_plots=False,
        save_fit_plots=False,
        save_posterior_samples=True,
        posterior_sample_size=20000,
        quiet_worker_output=True,
        print_cluster_updates=False,
        model_names=("parsec",),
        output_dirname=args.output_dirname,
    )
    print(
        "Running local MCMC comparison"
        f" | clusters={len(clusters)}"
        " | nwalkers=46 | burnin=100 | nsteps=1000"
        f" | n_processes={args.n_processes}"
        f" | output={args.output_dirname}",
        flush=True,
    )
    output_path = run_dual_model_refit(
        config_path=args.config,
        n_processes=args.n_processes,
        force=True,
        clusters=clusters,
        run_config=run_config,
    )
    print(f"Finished: {output_path}", flush=True)


if __name__ == "__main__":
    main()
