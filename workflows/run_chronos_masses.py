from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chronos.run_chronos.mass_estimation import (
    IntervalSummary,
    age_summary_from_row,
    estimate_cluster_mass,
    resolve_age_summary_row,
    write_json,
    write_markdown_summary,
)
from chronos.run_chronos.pipeline import prepare_member_photometry
from workflows.config import load_runtime_paths
from workflows.run_chronos_ages import run as run_chronos_ages


DEFAULT_METHODS: tuple[str, ...] = ("swiggum", "hunt")


def _paper_age_interval(row: pd.Series) -> IntervalSummary:
    return IntervalSummary(
        p16=float(10 ** (float(row["log_age_16"]) - 6.0)),
        p50=float(10 ** (float(row["log_age_50"]) - 6.0)),
        p84=float(10 ** (float(row["log_age_84"]) - 6.0)),
    )


def _paper_mass_interval(row: pd.Series) -> IntervalSummary:
    return IntervalSummary(
        p16=float(row["mass_16"]),
        p50=float(row["mass_50"]),
        p84=float(row["mass_84"]),
    )


def _parse_methods(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_METHODS
    methods = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    invalid = sorted(set(methods) - {"swiggum", "hunt"})
    if invalid:
        raise ValueError(f"Unsupported mass methods: {invalid}")
    return methods or DEFAULT_METHODS


def _load_member_catalog_subset(
    member_catalog_path: Path,
    *,
    cluster_names: set[str],
    chunksize: int = 250_000,
) -> pd.DataFrame:
    requested_columns = [
        "name",
        "source_id",
        "ra",
        "dec",
        "parallax",
        "parallax_error",
        "pmra",
        "pmra_error",
        "pmdec",
        "pmdec_error",
        "phot_g_mean_mag",
        "phot_bp_mean_mag",
        "phot_rp_mean_mag",
        "bp_rp",
        "g_rp",
        "distance_50",
        "ruwe",
        "non_single_star",
        "rybizki_fidelity_v1",
        "astrometric_params_solved",
    ]
    header = pd.read_csv(member_catalog_path, nrows=0)
    usecols = [column for column in requested_columns if column in header.columns]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(member_catalog_path, usecols=usecols, chunksize=chunksize):
        subset = chunk.loc[chunk["name"].astype(str).isin(cluster_names)].copy()
        if not subset.empty:
            chunks.append(subset)
    if not chunks:
        return pd.DataFrame(columns=usecols)
    return pd.concat(chunks, ignore_index=True)


def _empty_method_row(prefix: str, *, status: str, n_members_total: int, skip_reason: str | None) -> dict[str, object]:
    return {
        f"{prefix}_status": status,
        f"{prefix}_16": np.nan,
        f"{prefix}_50": np.nan,
        f"{prefix}_84": np.nan,
        f"{prefix}_n_draws_succeeded": 0,
        f"{prefix}_n_members_total": int(n_members_total),
        f"{prefix}_n_members_used": 0,
        f"{prefix}_skip_reason": skip_reason,
    }


def _ensure_age_catalog(
    *,
    config_path: str | None,
    ages_csv: str | None,
) -> Path:
    paths = load_runtime_paths(config_path)
    if ages_csv is not None:
        requested = Path(ages_csv).expanduser().resolve()
        if not requested.exists():
            raise FileNotFoundError(f"Requested ages CSV does not exist: {requested}")
        return requested
    if paths.inputs.chronos_ages_csv.exists():
        return paths.inputs.chronos_ages_csv
    run_chronos_ages(config_path=config_path)
    if not paths.inputs.chronos_ages_csv.exists():
        raise FileNotFoundError(f"Chronos ages workflow did not produce: {paths.inputs.chronos_ages_csv}")
    return paths.inputs.chronos_ages_csv


def run(
    *,
    config_path: str | None = None,
    ages_csv: str | None = None,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    output_dir: str | None = None,
    n_draws: int = 100,
    n_imfs: int = 1000,
    seed: int = 20260420,
    clusters: tuple[str, ...] | None = None,
) -> dict[str, str]:
    paths = load_runtime_paths(config_path)
    if paths.inputs.cluster_family_csv is None:
        raise FileNotFoundError("This workflow requires `inputs.cluster_family_csv` for the 272-cluster table.")

    ages_path = _ensure_age_catalog(config_path=config_path, ages_csv=ages_csv)
    output_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (paths.outputs.chronos_dir / "mass_methods")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    sample_catalog = pd.read_csv(paths.inputs.cluster_family_csv).copy()
    if clusters:
        cluster_set = {str(name) for name in clusters}
        sample_catalog = sample_catalog.loc[sample_catalog["name"].astype(str).isin(cluster_set)].copy()

    cluster_catalog = pd.read_csv(paths.inputs.cluster_catalog_csv).copy()
    age_catalog = pd.read_csv(ages_path).copy()
    member_catalog = _load_member_catalog_subset(
        paths.inputs.member_catalog_csv,
        cluster_names=set(sample_catalog["name"].astype(str)),
    )

    fit_data = prepare_member_photometry(member_catalog, sample_catalog)
    grouped_fit_data = {
        str(cluster_name): group.reset_index(drop=True)
        for cluster_name, group in fit_data.groupby("label", sort=False)
    }
    cluster_lookup = {
        str(row["name"]): row
        for _, row in cluster_catalog.iterrows()
    }

    mass_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    skipped_clusters: list[str] = []

    for index, (_, sample_row) in enumerate(sample_catalog.iterrows()):
        cluster_name = str(sample_row["name"])
        paper_age = _paper_age_interval(sample_row)
        paper_mass = _paper_mass_interval(sample_row)
        cluster_row = cluster_lookup.get(cluster_name, sample_row)
        age_row = resolve_age_summary_row(age_catalog, cluster_name)

        if age_row is None:
            age_status = "missing_age_summary"
            age_mode = np.nan
            age_lo = np.nan
            age_hi = np.nan
            av_mode = np.nan
            av_lo = np.nan
            av_hi = np.nan
            age_summary = None
        else:
            age_summary = age_summary_from_row(age_row)
            age_status = age_summary.status
            age_mode = age_summary.age_myr.p50
            age_lo = age_summary.age_myr.p16
            age_hi = age_summary.age_myr.p84
            av_mode = age_summary.av_mag.p50
            av_lo = age_summary.av_mag.p16
            av_hi = age_summary.av_mag.p84

        df_group = grouped_fit_data.get(cluster_name)
        swiggum_estimate = _empty_method_row(
            "mass_swiggum",
            status="not_run",
            n_members_total=int(len(df_group)) if df_group is not None else 0,
            skip_reason="not_requested" if "swiggum" not in methods else None,
        )
        hunt_estimate = _empty_method_row(
            "mass_hunt",
            status="not_run",
            n_members_total=int(len(df_group)) if df_group is not None else 0,
            skip_reason="not_requested" if "hunt" not in methods else None,
        )
        if df_group is None or df_group.empty:
            if "swiggum" in methods:
                swiggum_estimate = _empty_method_row(
                    "mass_swiggum",
                    status="missing_membership_catalog",
                    n_members_total=0,
                    skip_reason="missing_membership_catalog",
                )
            if "hunt" in methods:
                hunt_estimate = _empty_method_row(
                    "mass_hunt",
                    status="missing_membership_catalog",
                    n_members_total=0,
                    skip_reason="missing_membership_catalog",
                )
            skipped_clusters.append(cluster_name)
        else:
            if age_summary is None:
                if "swiggum" in methods:
                    swiggum_estimate = _empty_method_row(
                        "mass_swiggum",
                        status="age_summary_unavailable",
                        n_members_total=int(len(df_group)),
                        skip_reason="missing_age_summary",
                    )
                if "hunt" in methods:
                    hunt_estimate = _empty_method_row(
                        "mass_hunt",
                        status="age_summary_unavailable",
                        n_members_total=int(len(df_group)),
                        skip_reason="missing_age_summary",
                    )
            else:
                if "swiggum" in methods:
                    swiggum_result = estimate_cluster_mass(
                        cluster_name=cluster_name,
                        df_group=df_group,
                        cluster_row=cluster_row,
                        age_summary=age_summary,
                        mass_method="swiggum",
                        rng_seed=seed + index,
                        n_draws=n_draws,
                        swiggum_n_imf_draws=n_imfs,
                    )
                    swiggum_estimate = swiggum_result.as_row("mass_swiggum")
                if "hunt" in methods:
                    hunt_result = estimate_cluster_mass(
                        cluster_name=cluster_name,
                        df_group=df_group,
                        cluster_row=cluster_row,
                        age_summary=age_summary,
                        mass_method="hunt",
                        rng_seed=seed + 10_000 + index,
                        n_draws=n_draws,
                        swiggum_n_imf_draws=n_imfs,
                        gaia_neighborhood_cache_dir=paths.inputs.gaia_neighborhood_cache_dir,
                        selection_cache_dir=paths.inputs.hunt_selection_cache_dir,
                    )
                    hunt_estimate = hunt_result.as_row("mass_hunt")

        base_row: dict[str, object] = {
            "name": cluster_name,
            "family": sample_row.get("family"),
            "paper_age_16_myr": paper_age.p16,
            "paper_age_50_myr": paper_age.p50,
            "paper_age_84_myr": paper_age.p84,
            "paper_mass_16_msun": paper_mass.p16,
            "paper_mass_50_msun": paper_mass.p50,
            "paper_mass_84_msun": paper_mass.p84,
            "chronos_age_status": age_status,
            "chronos_age_16_myr": age_lo,
            "chronos_age_50_myr": age_mode,
            "chronos_age_84_myr": age_hi,
            "chronos_av_16_mag": av_lo,
            "chronos_av_50_mag": av_mode,
            "chronos_av_84_mag": av_hi,
        }

        mass_row = {**base_row, **swiggum_estimate, **hunt_estimate}
        mass_rows.append(mass_row)

        comparison_row = dict(mass_row)
        if np.isfinite(pd.to_numeric(comparison_row.get("mass_swiggum_50"), errors="coerce")):
            comparison_row["mass_swiggum_delta_msun"] = (
                float(comparison_row["mass_swiggum_50"]) - paper_mass.p50
            )
            comparison_row["mass_swiggum_ratio"] = float(comparison_row["mass_swiggum_50"]) / paper_mass.p50
        else:
            comparison_row["mass_swiggum_delta_msun"] = np.nan
            comparison_row["mass_swiggum_ratio"] = np.nan
        if np.isfinite(pd.to_numeric(comparison_row.get("mass_hunt_50"), errors="coerce")):
            comparison_row["mass_hunt_delta_msun"] = float(comparison_row["mass_hunt_50"]) - paper_mass.p50
            comparison_row["mass_hunt_ratio"] = float(comparison_row["mass_hunt_50"]) / paper_mass.p50
        else:
            comparison_row["mass_hunt_delta_msun"] = np.nan
            comparison_row["mass_hunt_ratio"] = np.nan
        comparison_rows.append(comparison_row)

    mass_df = pd.DataFrame(mass_rows)
    comparison_df = pd.DataFrame(comparison_rows)

    mass_catalog_path = output_root / "chronos_mass_catalog.csv"
    comparison_catalog_path = output_root / "chronos_mass_comparison.csv"
    summary_path = output_root / "chronos_mass_summary.md"
    json_path = output_root / "chronos_mass_summary.json"

    mass_df.to_csv(mass_catalog_path, index=False)
    comparison_df.to_csv(comparison_catalog_path, index=False)

    summary: dict[str, object] = {
        "config_path": str(paths.config_path),
        "ages_csv": str(ages_path),
        "member_catalog": str(paths.inputs.member_catalog_csv),
        "n_rows": int(len(comparison_df)),
        "n_swiggum_success": int((comparison_df["mass_swiggum_status"] == "success").sum()),
        "n_hunt_success": int((comparison_df["mass_hunt_status"] == "success").sum()),
        "n_missing_membership": int(
            ((comparison_df["mass_swiggum_status"] == "missing_membership_catalog")
            | (comparison_df["mass_hunt_status"] == "missing_membership_catalog")).sum()
        ),
        "skipped_clusters": sorted(set(skipped_clusters)),
    }
    swiggum_delta = pd.to_numeric(comparison_df.get("mass_swiggum_delta_msun"), errors="coerce")
    hunt_delta = pd.to_numeric(comparison_df.get("mass_hunt_delta_msun"), errors="coerce")
    if swiggum_delta.notna().any():
        summary["swiggum_median_abs_delta"] = float(np.nanmedian(np.abs(swiggum_delta.to_numpy(dtype=float))))
    if hunt_delta.notna().any():
        summary["hunt_median_abs_delta"] = float(np.nanmedian(np.abs(hunt_delta.to_numpy(dtype=float))))

    write_markdown_summary(summary, output_path=summary_path)
    write_json(json_path, payload=summary)

    return {
        "output_root": str(output_root),
        "mass_catalog_csv": str(mass_catalog_path),
        "comparison_catalog_csv": str(comparison_catalog_path),
        "summary_md": str(summary_path),
        "summary_json": str(json_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Chronos cluster masses from fitted ages using the `swiggum` and/or `hunt` "
            "mass methods, and write a full comparison table for the Swiggum et al. (2024) sample."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Path to workflow TOML config.")
    parser.add_argument(
        "--ages-csv",
        type=str,
        default=None,
        help="Optional existing Chronos age-summary CSV. When omitted, reuse the configured file or run ages first.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="swiggum,hunt",
        help="Comma-separated mass methods to run. Valid values: swiggum,hunt",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory override.")
    parser.add_argument("--n-draws", type=int, default=100, help="Number of age/A_V Monte Carlo draws per cluster.")
    parser.add_argument(
        "--n-imfs",
        type=int,
        default=1000,
        help="Number of stochastic IMF candidates per Swiggum draw.",
    )
    parser.add_argument("--seed", type=int, default=20260420, help="Random seed for Monte Carlo sampling.")
    parser.add_argument(
        "--clusters",
        type=str,
        default=None,
        help="Optional comma-separated subset of cluster names to process.",
    )
    args = parser.parse_args()
    outputs = run(
        config_path=args.config,
        ages_csv=args.ages_csv,
        methods=_parse_methods(args.methods),
        output_dir=args.output_dir,
        n_draws=args.n_draws,
        n_imfs=args.n_imfs,
        seed=args.seed,
        clusters=tuple(part.strip() for part in args.clusters.split(",") if part.strip()) if args.clusters else None,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
