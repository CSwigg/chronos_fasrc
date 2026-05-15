#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def _log10(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype=float)
    mask = values > 0
    out.loc[mask] = np.log10(values.loc[mask])
    return out


def _finite(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        mask &= np.isfinite(values)
    return mask


def _ratio_summary(values: pd.Series) -> dict[str, float | int]:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite)]
    if finite.empty:
        return {"n": 0}
    q16, q50, q84 = finite.quantile([0.16, 0.50, 0.84]).tolist()
    return {
        "n": int(len(finite)),
        "median_log10_ratio": float(q50),
        "p16_log10_ratio": float(q16),
        "p84_log10_ratio": float(q84),
        "median_ratio": float(10**q50),
        "p16_ratio": float(10**q16),
        "p84_ratio": float(10**q84),
    }


def _corr(df: pd.DataFrame, x: str, y: str) -> dict[str, float | int | None]:
    sub = df.loc[_finite(df, [x, y]), [x, y]]
    if len(sub) < 3:
        return {"n": int(len(sub)), "pearson": None, "spearman": None}
    return {
        "n": int(len(sub)),
        "pearson": float(sub[x].corr(sub[y], method="pearson")),
        "spearman": float(sub[x].corr(sub[y], method="spearman")),
    }


def _scatter_age(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    sub = df.loc[_finite(df, ["old_chronos_age_myr", "new_chronos_age_myr"])].copy()
    sub = sub.loc[(sub["old_chronos_age_myr"] > 0) & (sub["new_chronos_age_myr"] > 0)]
    if sub.empty:
        ax.text(0.5, 0.5, "no overlap", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    clean = sub["quality_bucket"].eq("clean")
    ax.scatter(
        sub.loc[~clean, "old_chronos_age_myr"],
        sub.loc[~clean, "new_chronos_age_myr"],
        s=10,
        alpha=0.22,
        color="#b05b35",
        label="flagged/review",
    )
    ax.scatter(
        sub.loc[clean, "old_chronos_age_myr"],
        sub.loc[clean, "new_chronos_age_myr"],
        s=10,
        alpha=0.42,
        color="#1f6f8b",
        label="clean",
    )
    lo = min(sub["old_chronos_age_myr"].min(), sub["new_chronos_age_myr"].min())
    hi = max(sub["old_chronos_age_myr"].max(), sub["new_chronos_age_myr"].max())
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("previous Chronos age [Myr]")
    ax.set_ylabel("new Chronos PARSEC age [Myr]")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)


def _scatter_mass(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    sub = df.loc[_finite(df, ["old_sne_mass_50", "new_chronos_mass_msun"])].copy()
    sub = sub.loc[(sub["old_sne_mass_50"] > 0) & (sub["new_chronos_mass_msun"] > 0)]
    if sub.empty:
        ax.text(0.5, 0.5, "no overlap", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    clean = sub["quality_bucket"].eq("clean")
    ax.scatter(
        sub.loc[~clean, "old_sne_mass_50"],
        sub.loc[~clean, "new_chronos_mass_msun"],
        s=20,
        alpha=0.35,
        color="#b05b35",
        label="flagged/review",
    )
    ax.scatter(
        sub.loc[clean, "old_sne_mass_50"],
        sub.loc[clean, "new_chronos_mass_msun"],
        s=20,
        alpha=0.55,
        color="#1f6f8b",
        label="clean",
    )
    lo = min(sub["old_sne_mass_50"].min(), sub["new_chronos_mass_msun"].min())
    hi = max(sub["old_sne_mass_50"].max(), sub["new_chronos_mass_msun"].max())
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("previous SNe-map mass_50 [M_sun]")
    ax.set_ylabel("new Chronos MF mass [M_sun]")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)


def compare(
    *,
    run_dir: Path,
    old_chronos_csv: Path,
    old_sne_sample_csv: Path,
    model: str,
) -> Path:
    run_dir = run_dir.expanduser().resolve()
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    new = pd.read_csv(analysis_dir / "quality_table.csv")
    old = pd.read_csv(old_chronos_csv)
    sne = pd.read_csv(old_sne_sample_csv)

    new_cols = [
        "name",
        "quality_bucket",
        "quality_flags",
        f"{model}_age_mode",
        f"{model}_age_lo",
        f"{model}_age_hi",
        f"{model}_av_mode",
        f"{model}_mass_mf_fit_50",
        f"{model}_mass_mf_fit_16",
        f"{model}_mass_mf_fit_84",
        f"{model}_mass_mf_fit_status",
        "acceptance_mean",
        "autocorr_max",
    ]
    new = new[new_cols].rename(
        columns={
            f"{model}_age_mode": "new_chronos_age_myr",
            f"{model}_age_lo": "new_chronos_age_lo_myr",
            f"{model}_age_hi": "new_chronos_age_hi_myr",
            f"{model}_av_mode": "new_chronos_av",
            f"{model}_mass_mf_fit_50": "new_chronos_mass_msun",
            f"{model}_mass_mf_fit_16": "new_chronos_mass_16_msun",
            f"{model}_mass_mf_fit_84": "new_chronos_mass_84_msun",
            f"{model}_mass_mf_fit_status": "new_chronos_mass_status",
        }
    )

    old = old[
        [
            "name",
            "age_chronos_mode",
            "age_chronos_lo",
            "age_chronos_hi",
            "av_chronos_mode",
            "av_chronos_lo",
            "av_chronos_hi",
            "age_myr",
            "mass_all",
            "mass_jacobi",
        ]
    ].rename(
        columns={
            "age_chronos_mode": "old_chronos_age_myr",
            "age_chronos_lo": "old_chronos_age_lo_myr",
            "age_chronos_hi": "old_chronos_age_hi_myr",
            "av_chronos_mode": "old_chronos_av",
            "av_chronos_lo": "old_chronos_av_lo",
            "av_chronos_hi": "old_chronos_av_hi",
            "age_myr": "hunt_age_myr",
            "mass_all": "hunt_mass_all",
            "mass_jacobi": "hunt_mass_jacobi",
        }
    )

    all_overlap = new.merge(old, on="name", how="inner", validate="one_to_one")
    all_overlap["delta_log_age_new_minus_old"] = _log10(all_overlap["new_chronos_age_myr"]) - _log10(
        all_overlap["old_chronos_age_myr"]
    )
    all_overlap["age_ratio_new_over_old"] = 10 ** all_overlap["delta_log_age_new_minus_old"]
    all_overlap["delta_av_new_minus_old"] = all_overlap["new_chronos_av"] - all_overlap["old_chronos_av"]

    sne_keep = [
        "name",
        "family",
        "data_source",
        "age_myr",
        "mass_16",
        "mass_50",
        "mass_84",
        "n_sne_16",
        "n_sne_50",
        "n_sne_84",
        "n_stars_greater_8M_16",
        "n_stars_greater_8M_50",
        "n_stars_greater_8M_84",
        "weight_score",
    ]
    sne = sne[[column for column in sne_keep if column in sne.columns]].rename(
        columns={
            "age_myr": "old_sne_age_myr",
            "mass_16": "old_sne_mass_16",
            "mass_50": "old_sne_mass_50",
            "mass_84": "old_sne_mass_84",
        }
    )
    sne_overlap = all_overlap.merge(sne, on="name", how="inner", validate="one_to_one")
    sne_overlap["delta_log_mass_new_minus_old_sne"] = _log10(sne_overlap["new_chronos_mass_msun"]) - _log10(
        sne_overlap["old_sne_mass_50"]
    )
    sne_overlap["mass_ratio_new_over_old_sne"] = 10 ** sne_overlap["delta_log_mass_new_minus_old_sne"]

    all_path = analysis_dir / "chronos_vs_previous_chronos_table.csv"
    sne_path = analysis_dir / "chronos_vs_previous_sne_sample_table.csv"
    all_overlap.to_csv(all_path, index=False)
    sne_overlap.to_csv(sne_path, index=False)

    clean = all_overlap["quality_bucket"].eq("clean")
    no_age_bound = ~all_overlap["quality_flags"].fillna("").str.contains("age_at_upper_bound|age_at_lower_bound", regex=True)
    sne_clean = sne_overlap["quality_bucket"].eq("clean")
    sne_no_age_bound = ~sne_overlap["quality_flags"].fillna("").str.contains(
        "age_at_upper_bound|age_at_lower_bound", regex=True
    )

    subsets = {
        "all_overlap": all_overlap,
        "clean_no_age_boundary": all_overlap.loc[clean & no_age_bound],
        "previous_sne_sample_overlap": sne_overlap,
        "previous_sne_sample_clean_no_age_boundary": sne_overlap.loc[sne_clean & sne_no_age_bound],
    }

    summary: dict[str, object] = {
        "run_dir": str(run_dir),
        "old_chronos_csv": str(old_chronos_csv),
        "old_sne_sample_csv": str(old_sne_sample_csv),
        "new_completed_clusters": int(len(new)),
        "old_chronos_rows": int(len(old)),
        "old_sne_sample_rows": int(len(sne)),
        "overlap_new_old_chronos": int(len(all_overlap)),
        "overlap_new_old_sne_sample": int(len(sne_overlap)),
        "subsets": {},
    }

    for label, frame in subsets.items():
        entry = {
            "n": int(len(frame)),
            "age_new_over_old": _ratio_summary(frame.get("delta_log_age_new_minus_old", pd.Series(dtype=float))),
            "age_log_correlation": _corr(
                frame.assign(
                    log_old_age=_log10(frame.get("old_chronos_age_myr", pd.Series(dtype=float))),
                    log_new_age=_log10(frame.get("new_chronos_age_myr", pd.Series(dtype=float))),
                ),
                "log_old_age",
                "log_new_age",
            ),
            "av_delta_new_minus_old": {
                "n": int(_finite(frame, ["delta_av_new_minus_old"]).sum()) if len(frame) else 0,
                "median": float(frame["delta_av_new_minus_old"].median()) if "delta_av_new_minus_old" in frame else None,
                "p16": float(frame["delta_av_new_minus_old"].quantile(0.16)) if "delta_av_new_minus_old" in frame else None,
                "p84": float(frame["delta_av_new_minus_old"].quantile(0.84)) if "delta_av_new_minus_old" in frame else None,
            },
        }
        if "delta_log_mass_new_minus_old_sne" in frame:
            entry["mass_new_over_old_sne_mass_50"] = _ratio_summary(frame["delta_log_mass_new_minus_old_sne"])
            entry["mass_log_correlation"] = _corr(
                frame.assign(
                    log_old_mass=_log10(frame.get("old_sne_mass_50", pd.Series(dtype=float))),
                    log_new_mass=_log10(frame.get("new_chronos_mass_msun", pd.Series(dtype=float))),
                ),
                "log_old_mass",
                "log_new_mass",
            )
        summary["subsets"][label] = entry

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    _scatter_age(axes[0, 0], all_overlap, "All overlap: previous vs new Chronos age")
    _scatter_age(axes[0, 1], sne_overlap, "SNe-map sample overlap: previous vs new age")
    _scatter_mass(axes[1, 0], sne_overlap, "SNe-map sample: previous mass_50 vs new MF mass")
    values = pd.to_numeric(all_overlap["delta_log_age_new_minus_old"], errors="coerce")
    values = values[np.isfinite(values)]
    axes[1, 1].hist(values, bins=70, color="#376f8f", alpha=0.85, label="all overlap")
    sne_values = pd.to_numeric(sne_overlap["delta_log_age_new_minus_old"], errors="coerce")
    sne_values = sne_values[np.isfinite(sne_values)]
    axes[1, 1].hist(sne_values, bins=40, color="#b05b35", alpha=0.45, label="SNe-map sample")
    axes[1, 1].axvline(0, color="black", linewidth=1)
    axes[1, 1].set_xlabel("log10 new/previous Chronos age")
    axes[1, 1].set_ylabel("clusters")
    axes[1, 1].set_title("Age-ratio distributions")
    axes[1, 1].legend(frameon=False)
    comparison_plot = analysis_dir / "chronos_vs_previous_sne_run_comparison.png"
    fig.savefig(comparison_plot, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    mass_values = pd.to_numeric(sne_overlap["delta_log_mass_new_minus_old_sne"], errors="coerce")
    mass_values = mass_values[np.isfinite(mass_values)]
    axes[0].hist(mass_values, bins=45, color="#376f8f", alpha=0.85)
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_xlabel("log10 new MF mass / previous SNe mass_50")
    axes[0].set_ylabel("clusters")
    axes[0].set_title("SNe-map sample mass ratio")

    av_values = pd.to_numeric(all_overlap["delta_av_new_minus_old"], errors="coerce")
    av_values = av_values[np.isfinite(av_values)]
    axes[1].hist(av_values, bins=70, color="#4d6f3f", alpha=0.85, label="all overlap")
    sne_av = pd.to_numeric(sne_overlap["delta_av_new_minus_old"], errors="coerce")
    sne_av = sne_av[np.isfinite(sne_av)]
    axes[1].hist(sne_av, bins=35, color="#b05b35", alpha=0.45, label="SNe-map sample")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_xlabel("new A_V - previous A_V")
    axes[1].set_ylabel("clusters")
    axes[1].set_title("Extinction shift")
    axes[1].legend(frameon=False)
    ratio_plot = analysis_dir / "chronos_vs_previous_sne_run_ratio_histograms.png"
    fig.savefig(ratio_plot, dpi=180)
    plt.close(fig)

    summary["tables"] = {
        "all_overlap": str(all_path),
        "sne_sample_overlap": str(sne_path),
    }
    summary["plots"] = {
        "comparison": str(comparison_plot),
        "ratio_histograms": str(ratio_plot),
    }
    summary_path = analysis_dir / "chronos_vs_previous_sne_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare new Chronos PARSEC results to the previous SNe-map Chronos inputs.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--old-chronos",
        type=Path,
        default=Path("data/clusters/hunt_sample_chronos_ages_multiprocessing_feb_2026.csv"),
    )
    parser.add_argument("--old-sne-sample", type=Path, default=Path("data/clusters/cluster_sample_data.csv"))
    parser.add_argument("--model", default="parsec")
    args = parser.parse_args()
    compare(
        run_dir=args.run_dir,
        old_chronos_csv=args.old_chronos,
        old_sne_sample_csv=args.old_sne_sample,
        model=args.model,
    )


if __name__ == "__main__":
    main()
