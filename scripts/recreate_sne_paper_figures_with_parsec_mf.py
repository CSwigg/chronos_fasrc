#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = (
    Path(__file__).resolve().parents[1]
    / "runs/current/chronos/full_catalog_mf_fit_parsec_144shards_96w_1000b_10000s_20kpost_1000mass_linearage"
)
DEFAULT_SUPERNOVAE_MAP_DIR = Path(__file__).resolve().parents[2] / "supernovae_map"
DEFAULT_VARIANT_TAG = "galpy_parsec_mf_linearage_partial_almeida2024_radius_total_uniform_sphere"


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _write_compat_cluster_results(cluster_results: Path, output_path: Path) -> dict:
    results = pd.read_csv(cluster_results)
    required = [
        "name",
        "parsec_status",
        "parsec_age_mode",
        "parsec_age_lo",
        "parsec_age_hi",
        "parsec_av_mode",
        "parsec_av_lo",
        "parsec_av_hi",
        "parsec_mass_mf_fit_16",
        "parsec_mass_mf_fit_50",
        "parsec_mass_mf_fit_84",
        "parsec_mass_mf_fit_status",
    ]
    missing = [column for column in required if column not in results.columns]
    if missing:
        raise KeyError(f"Chronos cluster_results.csv is missing required columns: {missing}")

    compat = results.copy()
    mass_ok = compat["parsec_mass_mf_fit_status"].astype(str).eq("success")
    for src, dst in [
        ("parsec_mass_mf_fit_16", "parsec_mass_swiggum_16"),
        ("parsec_mass_mf_fit_50", "parsec_mass_swiggum_50"),
        ("parsec_mass_mf_fit_84", "parsec_mass_swiggum_84"),
    ]:
        values = _numeric(compat, src)
        compat[dst] = values.where(mass_ok & np.isfinite(values) & (values > 0.0), np.nan)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    compat.to_csv(output_path, index=False)
    finite = np.isfinite(pd.to_numeric(compat["parsec_mass_swiggum_50"], errors="coerce"))
    return {
        "cluster_results": str(cluster_results),
        "compat_cluster_results": str(output_path),
        "input_rows": int(len(results)),
        "parsec_success_rows": int(compat["parsec_status"].astype(str).eq("success").sum()),
        "mass_function_success_rows": int(mass_ok.sum()),
        "finite_compat_mass_rows": int(finite.sum()),
    }


def _copy_with_pdf(source: Path, destination: Path) -> dict[str, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    out = {destination.name: str(destination)}
    source_pdf = source.with_suffix(".pdf")
    if source_pdf.exists():
        pdf_dest = destination.with_suffix(".pdf")
        shutil.copy2(source_pdf, pdf_dest)
        out[pdf_dest.name] = str(pdf_dest)
    return out


def run(
    *,
    run_dir: Path,
    supernovae_map_dir: Path,
    output_root: Path,
    variant_tag: str,
    n_jobs: int,
    force_orbits: bool,
) -> Path:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(output_root / "mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output_root / "xdg_cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    if str(supernovae_map_dir) not in sys.path:
        sys.path.insert(0, str(supernovae_map_dir))

    from workflows.build_almeida2024_dissolution_variant import run as build_variant
    import workflows.build_almeida2024_dissolution_variant as dissolution_variant
    import workflows.build_almeida2024_local500_stats as local500_stats
    from workflows.build_almeida2024_local500_stats import run as build_local500_stats
    from workflows.build_figure2_raw_vs_tracked_box_preview import run as build_raw_vs_tracked
    from workflows.build_parsec_paper_comparison import (
        _prepare_parsec_input_tables,
        _write_variant_config,
    )
    from workflows.build_xy_nn_with_cluster_clustering_preview import run as build_xy_clustering
    from workflows.config import load_runtime_paths
    from paper.other.supernova_traceback_style_figure3 import make_figure as make_traceback_figure

    output_root.mkdir(parents=True, exist_ok=True)
    base_config = supernovae_map_dir / "configs" / "paths.toml"
    if not base_config.exists():
        raise FileNotFoundError(f"Missing supernovae-map config: {base_config}")

    compat_results = output_root / "inputs" / "parsec_mf_compat_cluster_results.csv"
    compat_summary = _write_compat_cluster_results(run_dir / "cluster_results.csv", compat_results)

    base_paths = load_runtime_paths(base_config)
    input_summary = _prepare_parsec_input_tables(
        dual_model_results_path=compat_results,
        base_velocity_catalog_path=base_paths.inputs.velocity_catalog_csv,
        age_table_path=output_root / "inputs" / "parsec_mf_chronos_ages.csv",
        velocity_table_path=output_root / "inputs" / "parsec_mf_velocity_catalog.csv",
        mass_source="parsec_swiggum",
    )
    variant_config = _write_variant_config(
        base_config_path=base_paths.config_path,
        age_table_path=Path(input_summary["age_table"]),
        velocity_table_path=Path(input_summary["velocity_table"]),
        output_root=output_root,
    )

    orbit_cache = output_root / "cache" / "cluster_orbits_current.csv.gz"
    dissolution_variant.ORBIT_CACHE = orbit_cache
    local500_stats.ORBIT_CACHE = orbit_cache

    variant_dir = output_root / "plots" / "variants" / variant_tag
    core_outputs = [
        variant_dir / "cluster_input_variant.csv.gz",
        variant_dir / "solar_encounter_catalog.csv.gz",
        variant_dir / "Figure_1.jpeg",
        variant_dir / "Figure_2.jpeg",
        variant_dir / "run_metadata.json",
        orbit_cache,
    ]
    if force_orbits or any(not path.exists() for path in core_outputs):
        build_variant(
            config_path=str(variant_config),
            output_tag=variant_tag,
            imf_samples=100,
            n_jobs=int(n_jobs),
            random_seed=24680,
            position_radius_col="radius_total_pc",
            position_radius_distribution="uniform_sphere",
            force_orbits=force_orbits,
        )

    local500_outputs = [
        variant_dir / "local_seed500_stats_summary.json",
        variant_dir / "local_seed500_fulltrace_sfr_sn_nn.png",
        variant_dir / "local_seed500_fulltrace_clustering_evolution.png",
    ]
    if any(not path.exists() for path in local500_outputs):
        build_local500_stats(config_path=str(variant_config), variant_tag=variant_tag)

    figures: dict[str, str] = {}
    figures.update(_copy_with_pdf(variant_dir / "Figure_1.jpeg", output_root / "Figure_1.jpeg"))
    figures.update(_copy_with_pdf(variant_dir / "Figure_2.jpeg", output_root / "Figure_2_time_panels.jpeg"))
    figures.update(
        _copy_with_pdf(
            variant_dir / "local_seed500_fulltrace_sfr_sn_nn.png",
            output_root / "Figure_3_local500_sfr_sn_nn.png",
        )
    )
    figures.update(
        _copy_with_pdf(
            variant_dir / "local_seed500_fulltrace_clustering_evolution.png",
            output_root / "Figure_4_local500_clustering_evolution.png",
        )
    )

    fig2 = build_raw_vs_tracked(
        config_path=str(variant_config),
        variant_tag=variant_tag,
        output_path=output_root / "Figure_2_local_history_raw_vs_tracked_box.png",
        metadata_path=output_root / "Figure_2_local_history_raw_vs_tracked_box.metadata.json",
        initial_half_width_pc=500.0,
        initial_selection_dimensions="xy",
        tracked_xy_half_width_pc=500.0,
        tracked_z_half_width_pc=None,
    )
    figures["Figure_2_local_history_raw_vs_tracked_box.png"] = fig2["figure2_png"]
    figures["Figure_2_local_history_raw_vs_tracked_box.pdf"] = fig2["figure2_pdf"]

    fig3 = build_xy_clustering(
        config_path=str(variant_config),
        variant_tag=variant_tag,
        output_path=output_root / "Figure_3_local_clustering_sne.png",
        metadata_path=output_root / "Figure_3_local_clustering_sne.metadata.json",
        box_half_width_pc=1000.0,
        clustering_ymax=None,
        clustering_ymin=1.0,
        show_legend=False,
        initial_selection_dimensions="xy",
        initial_half_width_pc=1000.0,
        restrict_to_time_box=True,
        time_kernel="gaussian_kde",
        kde_bandwidth_myr=3.0,
        kde_grid_step_myr=0.25,
        include_family_series=False,
        plot_metric="enhancement",
        future_max_myr=0.0,
        random_support_mode="data_support",
    )
    figures["Figure_3_local_clustering_sne.png"] = fig3["figure_png"]
    figures["Figure_3_local_clustering_sne.pdf"] = fig3["figure_pdf"]

    traceback_png = output_root / "Figure_4_traceback_style.png"
    traceback_pdf = output_root / "Figure_4_traceback_style.pdf"
    make_traceback_figure(
        output_path_png=traceback_png,
        output_path_pdf=traceback_pdf,
        config_path=str(variant_config),
        variant_tag=variant_tag,
    )
    figures[traceback_png.name] = str(traceback_png)
    figures[traceback_pdf.name] = str(traceback_pdf)

    paper_filename_png = output_root / "Figure_3_traceback_style.png"
    paper_filename_pdf = output_root / "Figure_3_traceback_style.pdf"
    shutil.copy2(traceback_png, paper_filename_png)
    shutil.copy2(traceback_pdf, paper_filename_pdf)
    figures[paper_filename_png.name] = str(paper_filename_png)
    figures[paper_filename_pdf.name] = str(paper_filename_pdf)

    metadata = {
        "note": (
            "Recreated paper figures using the synced partial full-catalog PARSEC mass-function "
            "Chronos outputs. This script writes only under the Chronos run directory."
        ),
        "run_dir": str(run_dir),
        "supernovae_map_dir": str(supernovae_map_dir),
        "variant_tag": variant_tag,
        "variant_config": str(variant_config),
        "variant_dir": str(variant_dir),
        "orbit_cache": str(orbit_cache),
        "compat_summary": compat_summary,
        "input_summary": input_summary,
        "figures": figures,
    }
    summary_path = output_root / "recreation_summary.json"
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "summary": str(summary_path), "figures": figures}, indent=2))
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recreate supernova-map paper figures from Chronos PARSEC mass-function outputs."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--supernovae-map-dir", type=Path, default=DEFAULT_SUPERNOVAE_MAP_DIR)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--variant-tag", type=str, default=DEFAULT_VARIANT_TAG)
    parser.add_argument("--n-jobs", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--force-orbits", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else run_dir / "paper_figure_recreation"
    )
    run(
        run_dir=run_dir,
        supernovae_map_dir=args.supernovae_map_dir.expanduser().resolve(),
        output_root=output_root,
        variant_tag=args.variant_tag,
        n_jobs=int(args.n_jobs),
        force_orbits=bool(args.force_orbits),
    )


if __name__ == "__main__":
    main()
