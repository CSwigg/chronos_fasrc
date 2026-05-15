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
    _require_input_file,
    _resolve_mist_dir_for_models,
)


DEFAULT_OUTPUT_DIRNAME = "full_catalog_mf_fit_parsec_unsharded_46w_100b_1000s_20kpost_1000mass_linearage_agemax12000myr"
DEFAULT_CATALOG_ORDER = "name"
DEFAULT_PRIORITY_HUNT_AGE_MAX_MYR = 200.0
DEFAULT_PRIORITY_BOX_HALF_WIDTH_PC = 1000.0
DEFAULT_AGE_MAX_MYR = 12000.0


def _coordinate_columns(clusters: pd.DataFrame) -> tuple[str, str, str] | None:
    for columns in (("x_2026", "y_2026", "z_2026"), ("x", "y", "z"), ("x_new", "y_new", "z_new")):
        if all(column in clusters.columns for column in columns):
            return columns
    return None


def _add_velocity_catalog_sort_columns(
    clusters: pd.DataFrame,
    *,
    config_path: str | Path | None,
) -> pd.DataFrame:
    paths = load_runtime_paths(config_path)
    if not paths.inputs.velocity_catalog_csv.exists():
        return clusters

    velocity_header = pd.read_csv(paths.inputs.velocity_catalog_csv, nrows=0)
    velocity_cols = ["name"]
    for column in ("age_myr", "x_2026", "y_2026", "z_2026", "x", "y", "z"):
        if column in velocity_header.columns:
            velocity_cols.append(column)
    if len(velocity_cols) == 1:
        return clusters

    velocity = pd.read_csv(paths.inputs.velocity_catalog_csv, usecols=velocity_cols)
    velocity["name"] = velocity["name"].astype(str)
    rename_map = {
        column: f"{column}_velocity_catalog"
        for column in velocity.columns
        if column != "name" and column in clusters.columns
    }
    velocity = velocity.rename(columns=rename_map)
    return clusters.merge(velocity, on="name", how="left")


def _sort_full_catalog_clusters(
    clusters: pd.DataFrame,
    *,
    catalog_order: str,
    priority_hunt_age_max_myr: float,
    priority_box_half_width_pc: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ordered = clusters.copy()
    ordered["__sort_name"] = ordered["name"].astype(str)

    if catalog_order == "name":
        ordered = ordered.sort_values("__sort_name").drop(columns=["__sort_name"]).reset_index(drop=True)
        return ordered, {
            "catalog_order": "name",
            "priority_hunt_age_max_myr": None,
            "priority_box_half_width_pc": None,
            "priority_young_in_box_count": None,
            "priority_young_total_count": None,
        }

    if catalog_order not in {"hunt_age", "hunt_young_solar_box"}:
        raise ValueError(f"Unsupported catalog_order={catalog_order!r}")

    age_source = "age_myr"
    if age_source not in ordered.columns and "age_myr_velocity_catalog" in ordered.columns:
        age_source = "age_myr_velocity_catalog"
    ordered["priority_hunt_age_myr"] = pd.to_numeric(ordered.get(age_source), errors="coerce")
    finite_age = np.isfinite(ordered["priority_hunt_age_myr"])
    ordered["__finite_age_sort"] = np.where(finite_age, 0, 1)

    if catalog_order == "hunt_age":
        ordered["__age_sort"] = ordered["priority_hunt_age_myr"].fillna(np.inf)
        ordered = (
            ordered.sort_values(["__finite_age_sort", "__age_sort", "__sort_name"])
            .drop(columns=["__sort_name", "__finite_age_sort", "__age_sort"])
            .reset_index(drop=True)
        )
        return ordered, {
            "catalog_order": "hunt_age",
            "priority_hunt_age_max_myr": None,
            "priority_box_half_width_pc": None,
            "priority_young_in_box_count": None,
            "priority_young_total_count": None,
        }

    coord_cols = _coordinate_columns(ordered)
    if coord_cols is None:
        raise ValueError(
            "Catalog priority ordering requires position columns. Expected one of "
            "(x_2026,y_2026,z_2026), (x,y,z), or (x_new,y_new,z_new)."
        )
    x_col, y_col, z_col = coord_cols
    for column, output_column in ((x_col, "priority_x_pc"), (y_col, "priority_y_pc"), (z_col, "priority_z_pc")):
        ordered[output_column] = pd.to_numeric(ordered[column], errors="coerce")

    half_width = float(priority_box_half_width_pc)
    age_max = float(priority_hunt_age_max_myr)
    abs_xyz = ordered[["priority_x_pc", "priority_y_pc", "priority_z_pc"]].abs()
    finite_xyz = np.isfinite(abs_xyz).all(axis=1)
    outside_delta = (abs_xyz - half_width).clip(lower=0.0)
    ordered["priority_outside_box_distance_pc"] = np.sqrt((outside_delta**2.0).sum(axis=1))
    ordered["priority_radius_pc"] = np.sqrt((ordered[["priority_x_pc", "priority_y_pc", "priority_z_pc"]] ** 2.0).sum(axis=1))
    ordered["priority_in_2kpc_box"] = finite_xyz & (abs_xyz <= half_width).all(axis=1)
    ordered["priority_hunt_age_lt_200myr"] = finite_age & (ordered["priority_hunt_age_myr"] > 0.0) & (
        ordered["priority_hunt_age_myr"] < age_max
    )
    ordered["priority_young_in_2kpc_box"] = ordered["priority_hunt_age_lt_200myr"] & ordered["priority_in_2kpc_box"]
    ordered["__priority_bucket"] = np.select(
        [
            ordered["priority_young_in_2kpc_box"],
            ordered["priority_hunt_age_lt_200myr"],
            finite_age,
        ],
        [0, 1, 2],
        default=3,
    )
    ordered["__distance_sort"] = ordered["priority_outside_box_distance_pc"].fillna(np.inf)
    ordered["__radius_sort"] = ordered["priority_radius_pc"].fillna(np.inf)
    ordered["__age_sort"] = ordered["priority_hunt_age_myr"].fillna(np.inf)
    ordered = (
        ordered.sort_values(
            [
                "__priority_bucket",
                "__age_sort",
                "__distance_sort",
                "__radius_sort",
                "__sort_name",
            ]
        )
        .drop(
            columns=[
                "__sort_name",
                "__finite_age_sort",
                "__priority_bucket",
                "__distance_sort",
                "__radius_sort",
                "__age_sort",
            ]
        )
        .reset_index(drop=True)
    )
    return ordered, {
        "catalog_order": "hunt_young_solar_box",
        "priority_hunt_age_max_myr": age_max,
        "priority_box_half_width_pc": half_width,
        "priority_position_columns": list(coord_cols),
        "priority_young_in_box_count": int(ordered["priority_young_in_2kpc_box"].sum()),
        "priority_young_total_count": int(ordered["priority_hunt_age_lt_200myr"].sum()),
        "priority_in_box_total_count": int(ordered["priority_in_2kpc_box"].sum()),
        "priority_finite_age_count": int(finite_age.sum()),
        "priority_finite_xyz_count": int(finite_xyz.sum()),
    }


def _apply_full_catalog_shard(
    selected: pd.DataFrame,
    *,
    cluster_shard_count: int | None,
    cluster_shard_index: int | None,
    cluster_shard_strategy: str,
) -> tuple[pd.DataFrame, dict[str, int | str | None]]:
    if cluster_shard_count is None and cluster_shard_index is None:
        return selected, {
            "cluster_shard_count": None,
            "cluster_shard_index": None,
            "cluster_shard_strategy": None,
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
    if cluster_shard_strategy not in {"round_robin", "contiguous"}:
        raise ValueError("--cluster-shard-strategy must be one of: round_robin, contiguous.")

    before_count = int(len(selected))
    if cluster_shard_strategy == "round_robin":
        shard_mask = (np.arange(before_count) % shard_count) == shard_index
        sharded = selected.loc[shard_mask].reset_index(drop=True)
    else:
        index_chunks = np.array_split(np.arange(before_count), shard_count)
        shard_indices = index_chunks[shard_index]
        sharded = selected.iloc[shard_indices].reset_index(drop=True)

    return sharded, {
        "cluster_shard_count": int(shard_count),
        "cluster_shard_index": int(shard_index),
        "cluster_shard_strategy": str(cluster_shard_strategy),
        "n_clusters_before_shard": before_count,
        "n_clusters_after_shard": int(len(sharded)),
    }


def _bundled_parsec_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "chronos" / "data" / "parsec_files"


def _parsec_log_age_bounds(parsec_dir: Path) -> tuple[float, float]:
    parsec_files = sorted(Path(parsec_dir).glob("*.dat"))
    if not parsec_files:
        raise FileNotFoundError(f"No PARSEC .dat files found under {parsec_dir}")

    min_log_age = np.inf
    max_log_age = -np.inf
    for parsec_file in parsec_files:
        with parsec_file.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    log_age = float(parts[2])
                except ValueError:
                    continue
                min_log_age = min(min_log_age, log_age)
                max_log_age = max(max_log_age, log_age)

    if not np.isfinite(min_log_age) or not np.isfinite(max_log_age):
        raise ValueError(f"Could not read PARSEC logAge values from {parsec_dir}")
    return float(min_log_age), float(max_log_age)


def _validate_parsec_age_coverage(*, parsec_dir: Path, age_min_myr: float, age_max_myr: float) -> None:
    min_log_age, max_log_age = _parsec_log_age_bounds(parsec_dir)
    requested_min_log_age = float(np.log10(float(age_min_myr) * 1e6))
    requested_max_log_age = float(np.log10(float(age_max_myr) * 1e6))
    if requested_min_log_age < min_log_age or requested_max_log_age > max_log_age:
        available_min_myr = 10**min_log_age / 1e6
        available_max_myr = 10**max_log_age / 1e6
        raise ValueError(
            "Requested PARSEC age prior is outside the configured PARSEC isochrone grid. "
            f"requested={age_min_myr:g}-{age_max_myr:g} Myr "
            f"(logAge={requested_min_log_age:.3f}-{requested_max_log_age:.3f}); "
            f"available={available_min_myr:g}-{available_max_myr:g} Myr "
            f"(logAge={min_log_age:.3f}-{max_log_age:.3f}) from {parsec_dir}. "
            "Install/configure PARSEC isochrones through 12 Gyr and set "
            "inputs.parsec_isochrone_dir in configs/paths.toml before running."
        )


def select_full_catalog_clusters(
    *,
    config_path: str | Path | None = None,
    catalog_order: str = DEFAULT_CATALOG_ORDER,
    priority_hunt_age_max_myr: float = DEFAULT_PRIORITY_HUNT_AGE_MAX_MYR,
    priority_box_half_width_pc: float = DEFAULT_PRIORITY_BOX_HALF_WIDTH_PC,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select every named cluster in the configured Chronos cluster catalog."""
    paths = load_runtime_paths(config_path)
    clusters = pd.read_csv(paths.inputs.cluster_catalog_csv).copy()
    if "name" not in clusters.columns:
        raise ValueError(f"Cluster catalog is missing required 'name' column: {paths.inputs.cluster_catalog_csv}")
    clusters["name"] = clusters["name"].astype(str)
    clusters = clusters.loc[clusters["name"].str.strip() != ""].copy()
    clusters = _add_velocity_catalog_sort_columns(clusters, config_path=config_path)
    return _sort_full_catalog_clusters(
        clusters,
        catalog_order=catalog_order,
        priority_hunt_age_max_myr=priority_hunt_age_max_myr,
        priority_box_half_width_pc=priority_box_half_width_pc,
    )


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


def _apply_selection_filters(
    selected: pd.DataFrame,
    *,
    filter_hunt_age_max_myr: float | None,
    filter_xy_half_width_pc: float | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    before_count = int(len(selected))
    filtered = selected.copy()
    filter_summary: dict[str, Any] = {
        "filter_hunt_age_max_myr": None,
        "filter_xy_half_width_pc": None,
        "n_clusters_before_filters": before_count,
    }

    mask = pd.Series(True, index=filtered.index)
    if filter_hunt_age_max_myr is not None:
        age_max = float(filter_hunt_age_max_myr)
        ages = pd.to_numeric(filtered.get("priority_hunt_age_myr", filtered.get("age_myr")), errors="coerce")
        age_mask = np.isfinite(ages) & (ages > 0.0) & (ages < age_max)
        mask &= age_mask
        filter_summary["filter_hunt_age_max_myr"] = age_max

    if filter_xy_half_width_pc is not None:
        half_width = float(filter_xy_half_width_pc)
        if {"priority_x_pc", "priority_y_pc"}.issubset(filtered.columns):
            x_values = pd.to_numeric(filtered["priority_x_pc"], errors="coerce")
            y_values = pd.to_numeric(filtered["priority_y_pc"], errors="coerce")
        else:
            coord_cols = _coordinate_columns(filtered)
            if coord_cols is None:
                raise ValueError(
                    "XY filtering requires position columns. Expected one of "
                    "(x_2026,y_2026,z_2026), (x,y,z), or (x_new,y_new,z_new)."
                )
            x_values = pd.to_numeric(filtered[coord_cols[0]], errors="coerce")
            y_values = pd.to_numeric(filtered[coord_cols[1]], errors="coerce")
        xy_mask = (
            np.isfinite(x_values)
            & np.isfinite(y_values)
            & (x_values > -half_width)
            & (x_values < half_width)
            & (y_values > -half_width)
            & (y_values < half_width)
        )
        mask &= xy_mask
        filter_summary["filter_xy_half_width_pc"] = half_width

    filtered = filtered.loc[mask].reset_index(drop=True)
    filter_summary["n_clusters_after_filters"] = int(len(filtered))
    return filtered, filter_summary


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
    cluster_shard_strategy: str = "round_robin",
    catalog_order: str = DEFAULT_CATALOG_ORDER,
    priority_hunt_age_max_myr: float = DEFAULT_PRIORITY_HUNT_AGE_MAX_MYR,
    priority_box_half_width_pc: float = DEFAULT_PRIORITY_BOX_HALF_WIDTH_PC,
    filter_hunt_age_max_myr: float | None = None,
    filter_xy_half_width_pc: float | None = None,
    age_min_myr: float = 1.0,
    age_max_myr: float = DEFAULT_AGE_MAX_MYR,
    age_prior: str = "linear",
    nwalkers: int = 46,
    burnin: int = 100,
    nsteps: int = 1000,
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

    selected, ordering_summary = select_full_catalog_clusters(
        config_path=config_path,
        catalog_order=catalog_order,
        priority_hunt_age_max_myr=priority_hunt_age_max_myr,
        priority_box_half_width_pc=priority_box_half_width_pc,
    )
    selected, filter_summary = _apply_selection_filters(
        selected,
        filter_hunt_age_max_myr=filter_hunt_age_max_myr,
        filter_xy_half_width_pc=filter_xy_half_width_pc,
    )
    if sample_n_clusters is not None:
        sample_n_clusters = int(sample_n_clusters)
        if sample_n_clusters <= 0:
            raise ValueError("--sample-n-clusters must be positive when provided.")
        if sample_n_clusters < len(selected):
            rng = np.random.default_rng(int(sample_seed))
            sampled_indices = np.sort(rng.choice(selected.index.to_numpy(), size=sample_n_clusters, replace=False))
            selected = selected.loc[sampled_indices].reset_index(drop=True)

    selected, shard_selection = _apply_full_catalog_shard(
        selected,
        cluster_shard_count=cluster_shard_count,
        cluster_shard_index=cluster_shard_index,
        cluster_shard_strategy=cluster_shard_strategy,
    )

    output_root = paths.outputs.chronos_dir / output_dirname
    selection = {
        "catalog": "full",
        "sample_n_clusters": int(sample_n_clusters) if sample_n_clusters is not None else None,
        "sample_seed": int(sample_seed) if sample_n_clusters is not None else None,
        **ordering_summary,
        **filter_summary,
        **shard_selection,
        "requires_finite_positive_hunt_age": filter_hunt_age_max_myr is not None,
        "rv_cut_enabled": False,
        "map_box_cut_enabled": filter_xy_half_width_pc is not None,
        "age_prior": str(age_prior),
    }
    _write_selection(
        selected,
        output_root=output_root,
        selection=selection,
        shard_selection=shard_selection,
    )

    isochrone_dirs: dict[str, str] = {}
    if paths.inputs.parsec_isochrone_dir is not None:
        isochrone_dirs["parsec"] = str(paths.inputs.parsec_isochrone_dir)
    if mist_dir is not None:
        isochrone_dirs["mist"] = str(mist_dir)

    if "parsec" in {model.lower() for model in models}:
        parsec_dir = Path(isochrone_dirs.get("parsec", str(_bundled_parsec_dir())))
        _validate_parsec_age_coverage(
            parsec_dir=parsec_dir,
            age_min_myr=float(age_min_myr),
            age_max_myr=float(age_max_myr),
        )

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
        f" | shard_strategy={shard_selection['cluster_shard_strategy']}"
        f" | catalog_order={ordering_summary['catalog_order']}"
        f" | filter_hunt_age_max_myr={filter_summary['filter_hunt_age_max_myr']}"
        f" | filter_xy_half_width_pc={filter_summary['filter_xy_half_width_pc']}"
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
    parser.add_argument("--cluster-shard-strategy", choices=("round_robin", "contiguous"), default="round_robin")
    parser.add_argument(
        "--catalog-order",
        choices=("name", "hunt_age", "hunt_young_solar_box"),
        default=DEFAULT_CATALOG_ORDER,
    )
    parser.add_argument("--priority-hunt-age-max-myr", type=float, default=DEFAULT_PRIORITY_HUNT_AGE_MAX_MYR)
    parser.add_argument("--priority-box-half-width-pc", type=float, default=DEFAULT_PRIORITY_BOX_HALF_WIDTH_PC)
    parser.add_argument("--filter-hunt-age-max-myr", type=float, default=None)
    parser.add_argument("--filter-xy-half-width-pc", type=float, default=None)
    parser.add_argument("--age-min-myr", type=float, default=1.0)
    parser.add_argument("--age-max-myr", type=float, default=DEFAULT_AGE_MAX_MYR)
    parser.add_argument("--age-prior", choices=("linear", "log"), default="linear")
    parser.add_argument("--nwalkers", type=int, default=46)
    parser.add_argument("--burnin", type=int, default=100)
    parser.add_argument("--nsteps", type=int, default=1000)
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
        cluster_shard_strategy=args.cluster_shard_strategy,
        catalog_order=args.catalog_order,
        priority_hunt_age_max_myr=args.priority_hunt_age_max_myr,
        priority_box_half_width_pc=args.priority_box_half_width_pc,
        filter_hunt_age_max_myr=args.filter_hunt_age_max_myr,
        filter_xy_half_width_pc=args.filter_xy_half_width_pc,
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
