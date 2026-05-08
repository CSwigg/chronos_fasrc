from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

_CACHE_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))
_CACHE_USER = os.environ.get("USER", "user")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / f"chronos-mpl-{_CACHE_USER}"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / f"chronos-xdg-{_CACHE_USER}"))

from chronos.run_chronos.dual_model import DualModelRunConfig, run_dual_model_refit
from chronos.run_chronos.pipeline import ChronosFitConfig
from workflows.config import load_runtime_paths
from workflows.run_chronos_hunt_lt150_mf_fit import (
    _apply_cluster_shard,
    _require_input_file,
    _resolve_mist_dir_for_models,
)


DEFAULT_OUTPUT_DIRNAME = "full_catalog_mf_fit_parsec_48shards_96w_1000b_10000s_20kpost_1000mass"


def select_full_catalog_clusters(*, config_path: str | Path | None = None) -> pd.DataFrame:
    """Select every named cluster in the configured Chronos cluster catalog."""
    paths = load_runtime_paths(config_path)
    clusters = pd.read_csv(paths.inputs.cluster_catalog_csv).copy()
    if "name" not in clusters.columns:
        raise ValueError(f"Cluster catalog is missing required 'name' column: {paths.inputs.cluster_catalog_csv}")
    clusters["name"] = clusters["name"].astype(str)
    clusters = clusters.loc[clusters["name"].str.strip() != ""].copy()
    return clusters.sort_values("name").reset_index(drop=True)


def _selection_stem(shard_selection: Mapping[str, int | None]) -> str:
    shard_count = shard_selection.get("cluster_shard_count")
    shard_index = shard_selection.get("cluster_shard_index")
    if shard_count is None or shard_index is None:
        return "selected_full_catalog"
    return f"selected_full_catalog_shard_{int(shard_index):03d}_of_{int(shard_count):03d}"


def _write_selection(
    selected: pd.DataFrame,
    *,
    output_root: Path,
    selection: Mapping[str, Any],
    shard_selection: Mapping[str, int | None],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    stem = _selection_stem(shard_selection)
    selected_path = output_root / f"{stem}.csv"
    names_path = output_root / f"{stem}_names.txt"
    summary_path = output_root / f"{stem}_summary.json"

    selected.to_csv(selected_path, index=False)
    names_path.write_text("\n".join(selected["name"].astype(str)) + "\n", encoding="utf-8")
    summary = {
        "n_clusters": int(len(selected)),
        "selection": dict(selection),
        "selected_csv": str(selected_path),
        "selected_names_txt": str(names_path),
    }
    if "age_myr" in selected.columns:
        ages = pd.to_numeric(selected["age_myr"], errors="coerce")
        finite_ages = ages[np.isfinite(ages)]
        summary.update(
            {
                "catalog_age_min_myr": float(finite_ages.min()) if len(finite_ages) else None,
                "catalog_age_median_myr": float(finite_ages.median()) if len(finite_ages) else None,
                "catalog_age_max_myr": float(finite_ages.max()) if len(finite_ages) else None,
            }
        )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run(
    *,
    config_path: str | Path | None = None,
    n_processes: int | None = None,
    force: bool = False,
    models: tuple[str, ...] = ("parsec",),
    output_dirname: str = DEFAULT_OUTPUT_DIRNAME,
    mist_isochrone_dir: str | Path | None = None,
    sample_n_clusters: int | None = None,
    sample_seed: int = 20260508,
    cluster_shard_count: int | None = None,
    cluster_shard_index: int | None = None,
    age_min_myr: float = 1.0,
    age_max_myr: float = 1000.0,
    age_prior: str = "linear",
    nwalkers: int = 96,
    burnin: int = 1000,
    nsteps: int = 10000,
    mass_draws: int = 1000,
    n_imfs: int = 1000,
    posterior_sample_size: int = 20000,
    save_fit_plots: bool = False,
    save_mass_diagnostic_plots: bool = False,
    quiet_worker_output: bool = True,
    print_cluster_updates: bool = False,
) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("MPLBACKEND", "Agg")

    paths = load_runtime_paths(config_path)
    _require_input_file(paths.inputs.member_catalog_csv, label="member catalog")
    _require_input_file(paths.inputs.extinction_healpix_fits, label="Edenhofer HEALPix extinction map")
    mist_dir = _resolve_mist_dir_for_models(
        models=tuple(models),
        configured_mist_dir=paths.inputs.mist_isochrone_dir,
        mist_isochrone_dir=mist_isochrone_dir,
    )

    selected = select_full_catalog_clusters(config_path=config_path)
    if sample_n_clusters is not None:
        sample_n_clusters = int(sample_n_clusters)
        if sample_n_clusters <= 0:
            raise ValueError("--sample-n-clusters must be positive when provided.")
        if sample_n_clusters < len(selected):
            rng = np.random.default_rng(int(sample_seed))
            sampled_indices = np.sort(rng.choice(selected.index.to_numpy(), size=sample_n_clusters, replace=False))
            selected = selected.loc[sampled_indices].sort_values("name").reset_index(drop=True)

    selected, shard_selection = _apply_cluster_shard(
        selected,
        cluster_shard_count=cluster_shard_count,
        cluster_shard_index=cluster_shard_index,
    )

    output_root = paths.outputs.chronos_dir / output_dirname
    selection = {
        "catalog": "full",
        "sample_n_clusters": int(sample_n_clusters) if sample_n_clusters is not None else None,
        "sample_seed": int(sample_seed) if sample_n_clusters is not None else None,
        **shard_selection,
        "requires_finite_positive_hunt_age": False,
        "rv_cut_enabled": False,
        "map_box_cut_enabled": False,
        "age_prior": str(age_prior),
    }
    _write_selection(
        selected,
        output_root=output_root,
        selection=selection,
        shard_selection=shard_selection,
    )

    isochrone_dirs: dict[str, str] = {}
    if mist_dir is not None:
        isochrone_dirs["mist"] = str(mist_dir)

    fit_config = ChronosFitConfig(
        age_range_myr=(float(age_min_myr), float(age_max_myr)),
        nwalkers=int(nwalkers),
        burnin=int(burnin),
        nsteps=int(nsteps),
        isochrone_dirs=isochrone_dirs or None,
    )
    run_config = DualModelRunConfig(
        fit_config=fit_config,
        age_prior=str(age_prior),
        include_swiggum_masses=True,
        mass_n_draws=int(mass_draws),
        mass_n_imfs=int(n_imfs),
        mass_output_prefix="mass_mf_fit",
        save_mass_draws=True,
        save_mass_diagnostic_plots=bool(save_mass_diagnostic_plots),
        save_fit_plots=bool(save_fit_plots),
        save_posterior_samples=True,
        posterior_sample_size=int(posterior_sample_size),
        quiet_worker_output=bool(quiet_worker_output),
        print_cluster_updates=bool(print_cluster_updates),
        model_names=tuple(models),
        output_dirname=output_dirname,
    )
    if paths.inputs.mist_isochrone_dir is not None and "mist" in {model.lower() for model in models} and "mist" not in isochrone_dirs:
        run_config = replace(
            run_config,
            fit_config=replace(
                run_config.fit_config,
                isochrone_dirs={"mist": str(paths.inputs.mist_isochrone_dir)},
            ),
        )

    print(
        "Launching full-catalog Chronos mass-function-fit run"
        f" | clusters={len(selected)}"
        f" | models={','.join(models)}"
        f" | shard={shard_selection['cluster_shard_index']}/{shard_selection['cluster_shard_count']}"
        f" | age_range_myr={age_min_myr}-{age_max_myr}"
        f" | age_prior={age_prior}"
        f" | nwalkers={nwalkers}"
        f" | burnin={burnin}"
        f" | nsteps={nsteps}"
        f" | posterior_sample_size={posterior_sample_size}"
        f" | mass_draws={mass_draws}"
        f" | n_imfs={n_imfs}"
        f" | save_fit_plots={save_fit_plots}"
        f" | save_mass_diagnostic_plots={save_mass_diagnostic_plots}"
        f" | quiet_worker_output={quiet_worker_output}"
        f" | print_cluster_updates={print_cluster_updates}"
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
        description="Run Chronos for the full cluster catalog with mass-function-fit masses."
    )
    parser.add_argument("--config", type=str, default="configs/paths.toml")
    parser.add_argument("--n-processes", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--models", nargs="+", default=("parsec",))
    parser.add_argument("--output-dirname", type=str, default=DEFAULT_OUTPUT_DIRNAME)
    parser.add_argument("--mist-isochrone-dir", type=str, default=None)
    parser.add_argument("--sample-n-clusters", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=20260508)
    parser.add_argument("--cluster-shard-count", type=int, default=None)
    parser.add_argument("--cluster-shard-index", type=int, default=None)
    parser.add_argument("--age-min-myr", type=float, default=1.0)
    parser.add_argument("--age-max-myr", type=float, default=1000.0)
    parser.add_argument("--age-prior", choices=("linear", "log"), default="linear")
    parser.add_argument("--nwalkers", type=int, default=96)
    parser.add_argument("--burnin", type=int, default=1000)
    parser.add_argument("--nsteps", type=int, default=10000)
    parser.add_argument("--mass-draws", type=int, default=1000)
    parser.add_argument("--n-imfs", type=int, default=1000)
    parser.add_argument("--posterior-sample-size", type=int, default=20000)
    parser.add_argument("--save-fit-plots", action="store_true")
    parser.add_argument("--save-mass-diagnostic-plots", action="store_true")
    parser.add_argument("--show-worker-output", action="store_true")
    parser.add_argument("--print-cluster-updates", action="store_true")
    args = parser.parse_args()
    output_path = run(
        config_path=args.config,
        n_processes=args.n_processes,
        force=args.force,
        models=tuple(args.models),
        output_dirname=args.output_dirname,
        mist_isochrone_dir=args.mist_isochrone_dir,
        sample_n_clusters=args.sample_n_clusters,
        sample_seed=args.sample_seed,
        cluster_shard_count=args.cluster_shard_count,
        cluster_shard_index=args.cluster_shard_index,
        age_min_myr=args.age_min_myr,
        age_max_myr=args.age_max_myr,
        age_prior=args.age_prior,
        nwalkers=args.nwalkers,
        burnin=args.burnin,
        nsteps=args.nsteps,
        mass_draws=args.mass_draws,
        n_imfs=args.n_imfs,
        posterior_sample_size=args.posterior_sample_size,
        save_fit_plots=args.save_fit_plots,
        save_mass_diagnostic_plots=args.save_mass_diagnostic_plots,
        quiet_worker_output=not args.show_worker_output,
        print_cluster_updates=args.print_cluster_updates,
    )
    print(f"Chronos full-catalog mass-function-fit results: {output_path}", flush=True)


if __name__ == "__main__":
    main()
