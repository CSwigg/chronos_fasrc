from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from chronos.run_chronos.dual_model import DualModelRunConfig, run_dual_model_refit
from chronos.run_chronos.pipeline import ChronosFitConfig
from workflows.config import RuntimePaths, load_runtime_paths


DEFAULT_OUTPUT_DIRNAME = "mist_prev_chronos_lt100_1p5kpc_with_masses"
POSITION_COLUMN_SETS = (
    ("x_2026", "y_2026", "z_2026"),
    ("x", "y", "z"),
    ("x_new", "y_new", "z_new"),
)


def _position_columns(table: pd.DataFrame) -> tuple[str, str, str]:
    for columns in POSITION_COLUMN_SETS:
        if all(column in table.columns for column in columns):
            return columns
    expected = " or ".join("/".join(columns) for columns in POSITION_COLUMN_SETS)
    raise KeyError(f"Velocity catalog needs one position column set: {expected}")


def select_previous_chronos_young_nearby_clusters(
    *,
    config_path: str | Path | None = None,
    age_max_myr: float = 100.0,
    box_half_width_pc: float = 1500.0,
    age_column: str = "age_chronos_mode",
) -> pd.DataFrame:
    paths = load_runtime_paths(config_path)
    ages = pd.read_csv(paths.inputs.chronos_ages_csv).copy()
    if age_column not in ages.columns:
        raise KeyError(f"Previous Chronos table is missing {age_column!r}: {paths.inputs.chronos_ages_csv}")
    ages = ages[["name", age_column]].drop_duplicates("name").copy()
    ages[age_column] = pd.to_numeric(ages[age_column], errors="coerce")

    velocities = pd.read_csv(paths.inputs.velocity_catalog_csv).copy()
    x_col, y_col, z_col = _position_columns(velocities)
    velocities = velocities[["name", x_col, y_col, z_col]].drop_duplicates("name").copy()
    for column in (x_col, y_col, z_col):
        velocities[column] = pd.to_numeric(velocities[column], errors="coerce")

    merged = ages.merge(velocities, on="name", how="inner")
    position_ok = (
        np.isfinite(merged[x_col])
        & np.isfinite(merged[y_col])
        & np.isfinite(merged[z_col])
        & merged[x_col].between(-box_half_width_pc, box_half_width_pc)
        & merged[y_col].between(-box_half_width_pc, box_half_width_pc)
        & merged[z_col].between(-box_half_width_pc, box_half_width_pc)
    )
    age_ok = np.isfinite(merged[age_column]) & (merged[age_column] < float(age_max_myr))
    selected = merged.loc[age_ok & position_ok].copy()
    selected = selected.rename(
        columns={
            age_column: "previous_chronos_age_myr",
            x_col: "box_x_pc",
            y_col: "box_y_pc",
            z_col: "box_z_pc",
        }
    )
    return selected.sort_values(["previous_chronos_age_myr", "name"]).reset_index(drop=True)


def _resolve_mist_isochrone_dir(paths: RuntimePaths, mist_isochrone_dir: str | Path | None) -> Path:
    if mist_isochrone_dir is not None:
        path = Path(mist_isochrone_dir).expanduser()
    elif paths.inputs.mist_isochrone_dir is not None:
        path = paths.inputs.mist_isochrone_dir
    else:
        path = Path(__file__).resolve().parents[1] / "chronos" / "data" / "mist_files"
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"MIST isochrone directory does not exist: {path}")
    has_grid = any(path.glob("*.cmd")) or any(path.glob("*.iso.cmd"))
    if not has_grid:
        raise FileNotFoundError(
            f"No MIST Gaia DR3 CMD files found in {path}. "
            "Expected files ending in .cmd or .iso.cmd."
        )
    return path


def run(
    *,
    config_path: str | Path | None = None,
    mist_isochrone_dir: str | Path | None = None,
    n_processes: int | None = None,
    force: bool = False,
    age_max_myr: float = 100.0,
    box_half_width_pc: float = 1500.0,
    mass_draws: int = 100,
    n_imfs: int = 1000,
    output_dirname: str = DEFAULT_OUTPUT_DIRNAME,
) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("MPLBACKEND", "Agg")

    paths = load_runtime_paths(config_path)
    selected = select_previous_chronos_young_nearby_clusters(
        config_path=config_path,
        age_max_myr=age_max_myr,
        box_half_width_pc=box_half_width_pc,
    )
    if selected.empty:
        raise RuntimeError(
            "No clusters matched previous-Chronos age and nearby-box cuts "
            f"(age < {age_max_myr} Myr, |x/y/z| <= {box_half_width_pc} pc)."
        )

    output_root = paths.outputs.chronos_dir / output_dirname
    output_root.mkdir(parents=True, exist_ok=True)
    selected_path = output_root / "selected_previous_chronos_lt100_nearby_1p5kpc.csv"
    selected.to_csv(selected_path, index=False)

    mist_dir = _resolve_mist_isochrone_dir(paths, mist_isochrone_dir)
    summary = {
        "selection": {
            "n_clusters": int(len(selected)),
            "previous_chronos_age_column": "age_chronos_mode",
            "age_max_myr": float(age_max_myr),
            "box_half_width_pc": float(box_half_width_pc),
            "selected_clusters_csv": str(selected_path),
        },
        "mist_isochrone_dir": str(mist_dir),
        "output_dir": str(output_root),
    }
    summary_path = output_root / "mist_subset_run_setup.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cluster_names = selected["name"].astype(str).tolist()
    print(
        "Launching MIST-only Chronos run"
        f" | selected_clusters={len(cluster_names)}"
        f" | mist_isochrone_dir={mist_dir}"
        f" | selected_csv={selected_path}",
        flush=True,
    )
    return run_dual_model_refit(
        config_path=config_path,
        n_processes=n_processes,
        force=force,
        clusters=cluster_names,
        run_config=DualModelRunConfig(
            fit_config=ChronosFitConfig(isochrone_dirs={"mist": str(mist_dir)}),
            include_swiggum_masses=True,
            mass_n_draws=int(mass_draws),
            mass_n_imfs=int(n_imfs),
            model_names=("mist",),
            output_dirname=output_dirname,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run MIST Chronos ages and masses for clusters with previous Chronos age "
            "< 100 Myr inside the nearby +/-1.5 kpc box."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Path to workflow TOML config.")
    parser.add_argument(
        "--mist-isochrone-dir",
        type=str,
        default=None,
        help="Directory containing near-solar MIST Gaia DR3 CMD files.",
    )
    parser.add_argument("--n-processes", type=int, default=None, help="Number of worker processes.")
    parser.add_argument("--force", action="store_true", help="Rerun completed MIST checkpoints.")
    parser.add_argument("--age-max-myr", type=float, default=100.0, help="Previous Chronos age cut.")
    parser.add_argument(
        "--box-half-width-pc",
        type=float,
        default=1500.0,
        help="Nearby box half-width applied to the active velocity catalog positions.",
    )
    parser.add_argument("--mass-draws", type=int, default=100)
    parser.add_argument("--n-imfs", type=int, default=1000)
    parser.add_argument(
        "--output-dirname",
        type=str,
        default=DEFAULT_OUTPUT_DIRNAME,
        help="Subdirectory under outputs/chronos for MIST products.",
    )
    args = parser.parse_args()
    output_path = run(
        config_path=args.config,
        mist_isochrone_dir=args.mist_isochrone_dir,
        n_processes=args.n_processes,
        force=args.force,
        age_max_myr=args.age_max_myr,
        box_half_width_pc=args.box_half_width_pc,
        mass_draws=args.mass_draws,
        n_imfs=args.n_imfs,
        output_dirname=args.output_dirname,
    )
    print(f"MIST subset Chronos results: {output_path}", flush=True)


if __name__ == "__main__":
    main()
