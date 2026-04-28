from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from chronos.run_chronos.dual_model import DualModelRunConfig, run_dual_model_refit
from chronos.run_chronos.pipeline import ChronosFitConfig
from workflows.config import load_runtime_paths


DEFAULT_OUTPUT_DIRNAME = "hunt_lt150_mf_fit_parsec_64w_1000b_2000s_1000mass"


def select_hunt_lt150_clean_map_clusters(
    *,
    config_path: str | Path | None = None,
    hunt_age_max_myr: float = 150.0,
    box_half_width_pc: float = 1500.0,
    min_n_rvs_2026: int = 3,
) -> pd.DataFrame:
    """Select the clean Hunt starting sample for Chronos reruns."""
    paths = load_runtime_paths(config_path)
    cols = [
        "name",
        "age_myr",
        "mass_all",
        "mass_all_error",
        "x_2026",
        "y_2026",
        "z_2026",
        "n_rvs_2026",
    ]
    clusters = pd.read_csv(paths.inputs.velocity_catalog_csv, usecols=cols).copy()
    for column in cols:
        if column != "name":
            clusters[column] = pd.to_numeric(clusters[column], errors="coerce")

    clean = clusters.loc[
        np.isfinite(clusters["age_myr"])
        & (clusters["age_myr"] > 0.0)
        & (clusters["age_myr"] < float(hunt_age_max_myr))
        & np.isfinite(clusters["mass_all"])
        & (clusters["mass_all"] > 0.0)
        & np.isfinite(clusters["mass_all_error"])
        & (clusters["mass_all_error"] >= 0.0)
        & clusters["x_2026"].between(-float(box_half_width_pc), float(box_half_width_pc))
        & clusters["y_2026"].between(-float(box_half_width_pc), float(box_half_width_pc))
        & clusters["z_2026"].between(-float(box_half_width_pc), float(box_half_width_pc))
        & (clusters["n_rvs_2026"] >= int(min_n_rvs_2026))
    ].copy()
    return clean.sort_values(["age_myr", "name"]).reset_index(drop=True)


def _write_selection(
    selected: pd.DataFrame,
    *,
    output_root: Path,
    selection: Mapping[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    selected_path = output_root / "selected_hunt_lt150_clean_map_cut.csv"
    names_path = output_root / "selected_hunt_lt150_clean_map_cut_names.txt"
    summary_path = output_root / "selected_hunt_lt150_clean_map_cut_summary.json"
    selected.to_csv(selected_path, index=False)
    names_path.write_text("\n".join(selected["name"].astype(str)) + "\n", encoding="utf-8")

    summary = {
        "n_clusters": int(len(selected)),
        "selection": dict(selection),
        "selected_csv": str(selected_path),
        "selected_names_txt": str(names_path),
        "hunt_age_min_myr": float(selected["age_myr"].min()) if len(selected) else None,
        "hunt_age_median_myr": float(selected["age_myr"].median()) if len(selected) else None,
        "hunt_age_max_myr": float(selected["age_myr"].max()) if len(selected) else None,
    }
    import json

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run(
    *,
    config_path: str | Path | None = None,
    n_processes: int | None = None,
    force: bool = False,
    models: tuple[str, ...] = ("parsec",),
    output_dirname: str = DEFAULT_OUTPUT_DIRNAME,
    mist_isochrone_dir: str | Path | None = None,
    hunt_age_max_myr: float = 150.0,
    box_half_width_pc: float = 1500.0,
    min_n_rvs_2026: int = 3,
    nwalkers: int = 64,
    burnin: int = 1000,
    nsteps: int = 2000,
    mass_draws: int = 1000,
    n_imfs: int = 1000,
) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("MPLBACKEND", "Agg")

    paths = load_runtime_paths(config_path)
    selected = select_hunt_lt150_clean_map_clusters(
        config_path=config_path,
        hunt_age_max_myr=hunt_age_max_myr,
        box_half_width_pc=box_half_width_pc,
        min_n_rvs_2026=min_n_rvs_2026,
    )
    output_root = paths.outputs.chronos_dir / output_dirname
    selection = {
        "hunt_age_max_myr": float(hunt_age_max_myr),
        "box_half_width_pc": float(box_half_width_pc),
        "min_n_rvs_2026": int(min_n_rvs_2026),
        "requires_finite_positive_hunt_age": True,
        "requires_finite_positive_hunt_mass_all": True,
        "requires_finite_nonnegative_hunt_mass_all_error": True,
    }
    _write_selection(selected, output_root=output_root, selection=selection)

    isochrone_dirs: dict[str, str] = {}
    if mist_isochrone_dir is not None:
        isochrone_dirs["mist"] = str(Path(mist_isochrone_dir).expanduser().resolve())

    fit_config = ChronosFitConfig(
        age_range_myr=(1.0, 500.0),
        nwalkers=int(nwalkers),
        burnin=int(burnin),
        nsteps=int(nsteps),
        isochrone_dirs=isochrone_dirs or None,
    )
    run_config = DualModelRunConfig(
        fit_config=fit_config,
        include_swiggum_masses=True,
        mass_n_draws=int(mass_draws),
        mass_n_imfs=int(n_imfs),
        mass_output_prefix="mass_mf_fit",
        save_mass_draws=True,
        save_mass_diagnostic_plots=True,
        model_names=tuple(models),
        output_dirname=output_dirname,
    )
    if paths.inputs.mist_isochrone_dir is not None and "mist" not in isochrone_dirs:
        run_config = replace(
            run_config,
            fit_config=replace(
                run_config.fit_config,
                isochrone_dirs={"mist": str(paths.inputs.mist_isochrone_dir)},
            ),
        )

    print(
        "Launching Hunt<150 clean Chronos mass-function-fit run"
        f" | clusters={len(selected)}"
        f" | models={','.join(models)}"
        f" | nwalkers={nwalkers}"
        f" | burnin={burnin}"
        f" | nsteps={nsteps}"
        f" | mass_draws={mass_draws}"
        f" | n_imfs={n_imfs}"
        f" | output={output_root}",
        flush=True,
    )

    return run_dual_model_refit(
        config_path=config_path,
        n_processes=n_processes,
        force=force,
        clusters=selected["name"].astype(str).tolist(),
        run_config=run_config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Chronos for the clean Hunt <150 Myr map-cut sample with mass-function-fit masses."
    )
    parser.add_argument("--config", type=str, default="configs/paths.toml")
    parser.add_argument("--n-processes", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--models", nargs="+", default=("parsec",))
    parser.add_argument("--output-dirname", type=str, default=DEFAULT_OUTPUT_DIRNAME)
    parser.add_argument("--mist-isochrone-dir", type=str, default=None)
    parser.add_argument("--hunt-age-max-myr", type=float, default=150.0)
    parser.add_argument("--box-half-width-pc", type=float, default=1500.0)
    parser.add_argument("--min-n-rvs-2026", type=int, default=3)
    parser.add_argument("--nwalkers", type=int, default=64)
    parser.add_argument("--burnin", type=int, default=1000)
    parser.add_argument("--nsteps", type=int, default=2000)
    parser.add_argument("--mass-draws", type=int, default=1000)
    parser.add_argument("--n-imfs", type=int, default=1000)
    args = parser.parse_args()
    output_path = run(
        config_path=args.config,
        n_processes=args.n_processes,
        force=args.force,
        models=tuple(args.models),
        output_dirname=args.output_dirname,
        mist_isochrone_dir=args.mist_isochrone_dir,
        hunt_age_max_myr=args.hunt_age_max_myr,
        box_half_width_pc=args.box_half_width_pc,
        min_n_rvs_2026=args.min_n_rvs_2026,
        nwalkers=args.nwalkers,
        burnin=args.burnin,
        nsteps=args.nsteps,
        mass_draws=args.mass_draws,
        n_imfs=args.n_imfs,
    )
    print(f"Chronos mass-function-fit results: {output_path}", flush=True)


if __name__ == "__main__":
    main()
