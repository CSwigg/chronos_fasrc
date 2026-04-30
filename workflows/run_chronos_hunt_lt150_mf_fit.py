from __future__ import annotations

import argparse
from dataclasses import replace
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


DEFAULT_OUTPUT_DIRNAME = "hunt_lt150_mf_fit_all_models_96w_2000b_10000s_20kpost_1000mass"


def select_hunt_lt150_clean_map_clusters(
    *,
    config_path: str | Path | None = None,
    hunt_age_max_myr: float = 150.0,
    box_half_width_pc: float = 1500.0,
    min_n_rvs_2026: int | None = 3,
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

    mask = (
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
    )
    if min_n_rvs_2026 is not None:
        mask &= clusters["n_rvs_2026"] >= int(min_n_rvs_2026)

    clean = clusters.loc[mask].copy()
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


def _apply_cluster_shard(
    selected: pd.DataFrame,
    *,
    cluster_shard_count: int | None,
    cluster_shard_index: int | None,
) -> tuple[pd.DataFrame, dict[str, int | None]]:
    if cluster_shard_count is None and cluster_shard_index is None:
        return selected, {
            "cluster_shard_count": None,
            "cluster_shard_index": None,
            "n_clusters_before_shard": int(len(selected)),
            "n_clusters_after_shard": int(len(selected)),
        }

    shard_count = 1 if cluster_shard_count is None else int(cluster_shard_count)
    shard_index = 0 if cluster_shard_index is None else int(cluster_shard_index)
    if shard_count <= 0:
        raise ValueError("--cluster-shard-count must be positive when provided.")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            "--cluster-shard-index must satisfy "
            f"0 <= index < count; got index={shard_index}, count={shard_count}."
        )

    before_count = int(len(selected))
    shard_mask = (np.arange(before_count) % shard_count) == shard_index
    sharded = selected.loc[shard_mask].reset_index(drop=True)
    return sharded, {
        "cluster_shard_count": int(shard_count),
        "cluster_shard_index": int(shard_index),
        "n_clusters_before_shard": before_count,
        "n_clusters_after_shard": int(len(sharded)),
    }


def _require_input_file(path: Path, *, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _resolve_mist_dir_for_models(
    *,
    models: tuple[str, ...],
    configured_mist_dir: Path | None,
    mist_isochrone_dir: str | Path | None,
) -> Path | None:
    if "mist" not in {model.lower() for model in models}:
        return None
    mist_dir = Path(mist_isochrone_dir).expanduser().resolve() if mist_isochrone_dir is not None else configured_mist_dir
    if mist_dir is None or not mist_dir.exists():
        raise FileNotFoundError(
            "MIST was requested, but the MIST Gaia CMD directory is missing. "
            f"Expected: {mist_dir or 'inputs/mist_isochrones'}"
        )
    if not (any(mist_dir.glob("*.cmd")) or any(mist_dir.glob("*.iso.cmd"))):
        raise FileNotFoundError(
            f"MIST was requested, but no Gaia CMD files were found in {mist_dir}. "
            "Expected files ending in .cmd or .iso.cmd."
        )
    return mist_dir


def run(
    *,
    config_path: str | Path | None = None,
    n_processes: int | None = None,
    force: bool = False,
    models: tuple[str, ...] = ("parsec", "mist", "baraffe"),
    output_dirname: str = DEFAULT_OUTPUT_DIRNAME,
    mist_isochrone_dir: str | Path | None = None,
    hunt_age_max_myr: float = 150.0,
    box_half_width_pc: float = 1500.0,
    min_n_rvs_2026: int | None = 3,
    sample_n_clusters: int | None = None,
    sample_seed: int = 20260428,
    cluster_shard_count: int | None = None,
    cluster_shard_index: int | None = None,
    nwalkers: int = 96,
    burnin: int = 2000,
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
    selected = select_hunt_lt150_clean_map_clusters(
        config_path=config_path,
        hunt_age_max_myr=hunt_age_max_myr,
        box_half_width_pc=box_half_width_pc,
        min_n_rvs_2026=min_n_rvs_2026,
    )
    if sample_n_clusters is not None:
        sample_n_clusters = int(sample_n_clusters)
        if sample_n_clusters <= 0:
            raise ValueError("--sample-n-clusters must be positive when provided.")
        if sample_n_clusters < len(selected):
            rng = np.random.default_rng(int(sample_seed))
            sampled_indices = np.sort(rng.choice(selected.index.to_numpy(), size=sample_n_clusters, replace=False))
            selected = selected.loc[sampled_indices].sort_values(["age_myr", "name"]).reset_index(drop=True)
    selected, shard_selection = _apply_cluster_shard(
        selected,
        cluster_shard_count=cluster_shard_count,
        cluster_shard_index=cluster_shard_index,
    )

    output_root = paths.outputs.chronos_dir / output_dirname
    selection = {
        "hunt_age_max_myr": float(hunt_age_max_myr),
        "box_half_width_pc": float(box_half_width_pc),
        "min_n_rvs_2026": int(min_n_rvs_2026) if min_n_rvs_2026 is not None else None,
        "rv_cut_enabled": min_n_rvs_2026 is not None,
        "sample_n_clusters": int(sample_n_clusters) if sample_n_clusters is not None else None,
        "sample_seed": int(sample_seed) if sample_n_clusters is not None else None,
        **shard_selection,
        "requires_finite_positive_hunt_age": True,
        "requires_finite_positive_hunt_mass_all": True,
        "requires_finite_nonnegative_hunt_mass_all_error": True,
    }
    _write_selection(selected, output_root=output_root, selection=selection)

    isochrone_dirs: dict[str, str] = {}
    if mist_dir is not None:
        isochrone_dirs["mist"] = str(mist_dir)

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
        "Launching Hunt<150 clean Chronos mass-function-fit run"
        f" | clusters={len(selected)}"
        f" | models={','.join(models)}"
        f" | shard={shard_selection['cluster_shard_index']}/{shard_selection['cluster_shard_count']}"
        f" | min_n_rvs_2026={min_n_rvs_2026}"
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
        description="Run Chronos for the clean Hunt <150 Myr map-cut sample with mass-function-fit masses."
    )
    parser.add_argument("--config", type=str, default="configs/paths.toml")
    parser.add_argument("--n-processes", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--models", nargs="+", default=("parsec", "mist", "baraffe"))
    parser.add_argument("--output-dirname", type=str, default=DEFAULT_OUTPUT_DIRNAME)
    parser.add_argument("--mist-isochrone-dir", type=str, default=None)
    parser.add_argument("--hunt-age-max-myr", type=float, default=150.0)
    parser.add_argument("--box-half-width-pc", type=float, default=1500.0)
    parser.add_argument("--min-n-rvs-2026", type=int, default=3)
    parser.add_argument("--no-rv-cut", action="store_true")
    parser.add_argument("--sample-n-clusters", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=20260428)
    parser.add_argument("--cluster-shard-count", type=int, default=None)
    parser.add_argument("--cluster-shard-index", type=int, default=None)
    parser.add_argument("--nwalkers", type=int, default=96)
    parser.add_argument("--burnin", type=int, default=2000)
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
        hunt_age_max_myr=args.hunt_age_max_myr,
        box_half_width_pc=args.box_half_width_pc,
        min_n_rvs_2026=None if args.no_rv_cut else args.min_n_rvs_2026,
        sample_n_clusters=args.sample_n_clusters,
        sample_seed=args.sample_seed,
        cluster_shard_count=args.cluster_shard_count,
        cluster_shard_index=args.cluster_shard_index,
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
    print(f"Chronos mass-function-fit results: {output_path}", flush=True)


if __name__ == "__main__":
    main()
