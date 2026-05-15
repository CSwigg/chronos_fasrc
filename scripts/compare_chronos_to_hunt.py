#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

_CACHE_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))
_CACHE_USER = os.environ.get("USER", "user")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / f"chronos-mpl-{_CACHE_USER}"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / f"chronos-xdg-{_CACHE_USER}"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _finite_mask(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        mask &= np.isfinite(values)
    return mask


def _log10(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype=float)
    mask = values > 0
    out.loc[mask] = np.log10(values.loc[mask])
    return out


def _dex_summary(values: pd.Series) -> dict[str, float | int | None]:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite)]
    if finite.empty:
        return {"n": 0}
    q16, q50, q84 = finite.quantile([0.16, 0.50, 0.84]).tolist()
    return {
        "n": int(finite.size),
        "median_dex": float(q50),
        "p16_dex": float(q16),
        "p84_dex": float(q84),
        "median_ratio": float(10**q50),
        "p16_ratio": float(10**q16),
        "p84_ratio": float(10**q84),
    }


def _correlation_summary(df: pd.DataFrame, x: str, y: str) -> dict[str, float | int | None]:
    sub = df.loc[_finite_mask(df, [x, y]), [x, y]]
    if len(sub) < 3:
        return {"n": int(len(sub)), "pearson": None, "spearman": None}
    return {
        "n": int(len(sub)),
        "pearson": float(sub[x].corr(sub[y], method="pearson")),
        "spearman": float(sub[x].corr(sub[y], method="spearman")),
    }


def _scatter_loglog(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    clean_mask: pd.Series,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    sub = df.loc[_finite_mask(df, [x, y]) & (df[x] > 0) & (df[y] > 0)].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        return
    clean = clean_mask.reindex(sub.index, fill_value=False)
    flagged = sub.loc[~clean]
    clean_sub = sub.loc[clean]
    ax.scatter(flagged[x], flagged[y], s=8, alpha=0.20, color="#b05b35", label="flagged/review")
    ax.scatter(clean_sub[x], clean_sub[y], s=8, alpha=0.35, color="#1f6f8b", label="clean")
    lower = min(sub[x].min(), sub[y].min())
    upper = max(sub[x].max(), sub[y].max())
    ax.plot([lower, upper], [lower, upper], color="black", linewidth=1, alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)


def _delta_plot(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    delta: str,
    *,
    clean_mask: pd.Series,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    sub = df.loc[_finite_mask(df, [x, delta]) & (df[x] > 0)].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        return
    clean = clean_mask.reindex(sub.index, fill_value=False)
    ax.scatter(sub.loc[~clean, x], sub.loc[~clean, delta], s=8, alpha=0.20, color="#b05b35")
    ax.scatter(sub.loc[clean, x], sub.loc[clean, delta], s=8, alpha=0.35, color="#1f6f8b")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def compare(run_dir: Path, catalog_csv: Path, model: str) -> Path:
    run_dir = run_dir.expanduser().resolve()
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    quality = pd.read_csv(analysis_dir / "quality_table.csv")
    catalog_columns = [
        "name",
        "age_myr",
        "log_age_50",
        "mass_all",
        "mass_all_error",
        "mass_jacobi",
        "mass_jacobi_error",
        "n_stars",
        "n_stars_new",
        "kind_hunt_part_III",
    ]
    catalog = pd.read_csv(catalog_csv, usecols=lambda col: col in set(catalog_columns))
    df = quality.merge(catalog, on="name", how="left", validate="one_to_one", suffixes=("", "_hunt"))

    df["hunt_age_myr"] = pd.to_numeric(df["age_myr"], errors="coerce")
    df["chronos_age_myr"] = pd.to_numeric(df[f"{model}_age_mode"], errors="coerce")
    df["chronos_mass_msun"] = pd.to_numeric(df[f"{model}_mass_mf_fit_50"], errors="coerce")
    df["log_hunt_age"] = _log10(df["hunt_age_myr"])
    df["log_chronos_age"] = _log10(df["chronos_age_myr"])
    df["delta_log_age_chronos_minus_hunt"] = df["log_chronos_age"] - df["log_hunt_age"]
    df["age_ratio_chronos_over_hunt"] = 10 ** df["delta_log_age_chronos_minus_hunt"]

    for hunt_mass in ("mass_all", "mass_jacobi"):
        df[f"log_hunt_{hunt_mass}"] = _log10(df[hunt_mass])
        df[f"log_chronos_mass_vs_{hunt_mass}"] = _log10(df["chronos_mass_msun"])
        df[f"delta_log_mass_chronos_minus_{hunt_mass}"] = (
            df[f"log_chronos_mass_vs_{hunt_mass}"] - df[f"log_hunt_{hunt_mass}"]
        )
        df[f"mass_ratio_chronos_over_{hunt_mass}"] = 10 ** df[f"delta_log_mass_chronos_minus_{hunt_mass}"]

    no_age_boundary = ~df["quality_flags"].fillna("").str.contains("age_at_upper_bound|age_at_lower_bound", regex=True)
    clean = df["quality_bucket"].eq("clean")
    young_hunt = df["hunt_age_myr"].le(1000)
    mass_success = df[f"{model}_mass_mf_fit_status"].eq("success")

    subsets = {
        "all_completed": pd.Series(True, index=df.index),
        "clean": clean,
        "clean_no_age_boundary": clean & no_age_boundary,
        "hunt_age_le_1gyr": young_hunt,
        "hunt_age_le_1gyr_clean_no_age_boundary": young_hunt & clean & no_age_boundary,
        "mass_success_clean": mass_success & clean,
    }

    summary: dict[str, object] = {
        "run_dir": str(run_dir),
        "catalog_csv": str(catalog_csv),
        "n_rows": int(len(df)),
        "n_with_hunt_age": int(_finite_mask(df, ["hunt_age_myr"]).sum()),
        "n_with_chronos_age": int(_finite_mask(df, ["chronos_age_myr"]).sum()),
        "n_with_chronos_mass": int(_finite_mask(df, ["chronos_mass_msun"]).sum()),
        "n_with_hunt_mass_all": int(_finite_mask(df, ["mass_all"]).sum()),
        "n_with_hunt_mass_jacobi": int(_finite_mask(df, ["mass_jacobi"]).sum()),
        "subsets": {},
    }

    for label, mask in subsets.items():
        sub = df.loc[mask].copy()
        age_sub = sub.loc[_finite_mask(sub, ["delta_log_age_chronos_minus_hunt"])]
        mass_all_sub = sub.loc[_finite_mask(sub, ["delta_log_mass_chronos_minus_mass_all"])]
        mass_jacobi_sub = sub.loc[_finite_mask(sub, ["delta_log_mass_chronos_minus_mass_jacobi"])]
        summary["subsets"][label] = {
            "n": int(len(sub)),
            "age_delta": _dex_summary(age_sub["delta_log_age_chronos_minus_hunt"]),
            "age_log_correlation": _correlation_summary(age_sub, "log_hunt_age", "log_chronos_age"),
            "mass_all_delta": _dex_summary(mass_all_sub["delta_log_mass_chronos_minus_mass_all"]),
            "mass_all_log_correlation": _correlation_summary(mass_all_sub, "log_hunt_mass_all", "log_chronos_mass_vs_mass_all"),
            "mass_jacobi_delta": _dex_summary(mass_jacobi_sub["delta_log_mass_chronos_minus_mass_jacobi"]),
            "mass_jacobi_log_correlation": _correlation_summary(
                mass_jacobi_sub, "log_hunt_mass_jacobi", "log_chronos_mass_vs_mass_jacobi"
            ),
        }

    merged_path = analysis_dir / "chronos_vs_hunt_comparison_table.csv"
    df.to_csv(merged_path, index=False)

    clean_age_mask = clean & no_age_boundary
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    _scatter_loglog(
        axes[0, 0],
        df,
        "hunt_age_myr",
        "chronos_age_myr",
        clean_mask=clean_age_mask,
        xlabel="Hunt age [Myr]",
        ylabel="Chronos PARSEC age [Myr]",
        title="Age comparison",
    )
    _delta_plot(
        axes[0, 1],
        df,
        "hunt_age_myr",
        "delta_log_age_chronos_minus_hunt",
        clean_mask=clean_age_mask,
        xlabel="Hunt age [Myr]",
        ylabel="log10 Chronos/Hunt age",
        title="Age offset",
    )
    _scatter_loglog(
        axes[1, 0],
        df.loc[mass_success],
        "mass_all",
        "chronos_mass_msun",
        clean_mask=clean,
        xlabel="Hunt mass_all [M_sun]",
        ylabel="Chronos MF mass [M_sun]",
        title="Mass comparison: mass_all",
    )
    _scatter_loglog(
        axes[1, 1],
        df.loc[mass_success],
        "mass_jacobi",
        "chronos_mass_msun",
        clean_mask=clean,
        xlabel="Hunt mass_jacobi [M_sun]",
        ylabel="Chronos MF mass [M_sun]",
        title="Mass comparison: mass_jacobi",
    )
    comparison_plot = analysis_dir / "chronos_vs_hunt_comparison.png"
    fig.savefig(comparison_plot, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, column, title in [
        (axes[0], "delta_log_age_chronos_minus_hunt", "Age ratio"),
        (axes[1], "delta_log_mass_chronos_minus_mass_all", "Mass ratio vs mass_all"),
        (axes[2], "delta_log_mass_chronos_minus_mass_jacobi", "Mass ratio vs mass_jacobi"),
    ]:
        values = pd.to_numeric(df[column], errors="coerce")
        values = values[np.isfinite(values)]
        ax.hist(values, bins=60, color="#376f8f", alpha=0.85)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("log10 Chronos/Hunt")
        ax.set_ylabel("clusters")
    ratio_plot = analysis_dir / "chronos_vs_hunt_ratio_histograms.png"
    fig.savefig(ratio_plot, dpi=180)
    plt.close(fig)

    summary["comparison_table_csv"] = str(merged_path)
    summary["plots"] = {
        "comparison": str(comparison_plot),
        "ratio_histograms": str(ratio_plot),
    }
    summary_path = analysis_dir / "chronos_vs_hunt_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Chronos PARSEC results with Hunt ages and masses.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--catalog", type=Path, default=Path("data/clusters/hunt_partII_partIII_merged.csv"))
    parser.add_argument("--model", default="parsec")
    args = parser.parse_args()
    compare(args.run_dir, args.catalog, args.model)


if __name__ == "__main__":
    main()
