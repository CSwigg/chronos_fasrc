#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
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


def _as_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_load_error": str(exc)}


def _diagnostics_frame(run_dir: Path, model: str) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted((run_dir / model / "sampler_diagnostics").glob("*.json")):
        payload = _load_json(path)
        acc = np.asarray(payload.get("acceptance_fraction_by_walker", []), dtype=float)
        tau = np.asarray(payload.get("autocorr_time", []), dtype=float)
        acc = acc[np.isfinite(acc)]
        tau = tau[np.isfinite(tau)]
        rows.append(
            {
                "name": payload.get("cluster") or path.name.removesuffix(f"_{model}_diagnostics.json"),
                "diagnostics_path": str(path),
                "diagnostics_load_error": payload.get("_load_error"),
                "acceptance_mean": _as_float(payload.get("acceptance_fraction_mean")),
                "acceptance_min": _as_float(payload.get("acceptance_fraction_min")),
                "acceptance_max": _as_float(payload.get("acceptance_fraction_max")),
                "stuck_walker_fraction": float(np.mean(acc < 0.02)) if acc.size else math.nan,
                "autocorr_mean": float(np.mean(tau)) if tau.size else math.nan,
                "autocorr_max": float(np.max(tau)) if tau.size else math.nan,
                "autocorr_min": float(np.min(tau)) if tau.size else math.nan,
                "autocorr_error": payload.get("autocorr_time_error"),
                "nwalkers": payload.get("nwalkers"),
                "nsteps": payload.get("nsteps"),
                "burnin": payload.get("burnin"),
                "finite_flat_samples": payload.get("finite_flat_samples"),
                "best_log_prob": _as_float(payload.get("best_log_prob")),
            }
        )
    return pd.DataFrame(rows)


def _parse_counter(raw: object) -> Counter:
    if not isinstance(raw, str) or not raw.strip():
        return Counter()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return Counter()
    if not isinstance(payload, dict):
        return Counter()
    return Counter({str(key): int(value) for key, value in payload.items()})


def _quality_flags(row: pd.Series, model: str) -> list[str]:
    flags: list[str] = []
    if str(row.get("cluster_status", "")).strip() != "success":
        flags.append("cluster_status_not_success")
    if str(row.get(f"{model}_status", "")).strip() != "success":
        flags.append("fit_status_not_success")
    if str(row.get(f"{model}_mass_mf_fit_status", "")).strip() != "success":
        flags.append("mass_fit_not_success")

    acc = _as_float(row.get("acceptance_mean"))
    if math.isfinite(acc):
        if acc < 0.15:
            flags.append("low_acceptance")
        elif acc > 0.65:
            flags.append("high_acceptance")
    else:
        flags.append("missing_acceptance")

    stuck = _as_float(row.get("stuck_walker_fraction"))
    if math.isfinite(stuck) and stuck >= 0.10:
        flags.append("stuck_walkers")

    tau = _as_float(row.get("autocorr_max"))
    nsteps = _as_float(row.get("nsteps"))
    if not math.isfinite(tau):
        flags.append("missing_autocorr")
    elif math.isfinite(nsteps) and tau > 0 and nsteps / tau < 50:
        flags.append("short_vs_autocorr")

    age = _as_float(row.get(f"{model}_age_mode"))
    if math.isfinite(age):
        if age <= 1.05:
            flags.append("age_at_lower_bound")
        if age >= 950:
            flags.append("age_at_upper_bound")

    av = _as_float(row.get(f"{model}_av_mode"))
    if math.isfinite(av):
        if av <= 0.02:
            flags.append("av_at_lower_bound")
        if av >= 4.98:
            flags.append("av_at_upper_bound")

    valid_fraction = _as_float(row.get("prior_valid_fraction"))
    if math.isfinite(valid_fraction) and valid_fraction < 0.5:
        flags.append("low_dust_valid_fraction")

    return flags


def _safe_hist(ax, data: pd.Series, *, bins: int, title: str, xlabel: str, logx: bool = False) -> None:
    values = pd.to_numeric(data, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        ax.text(0.5, 0.5, "no finite values", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    if logx:
        values = values[values > 0]
        if values.size:
            ax.hist(values, bins=np.logspace(np.log10(values.min()), np.log10(values.max()), bins))
            ax.set_xscale("log")
    else:
        ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")


def _save_overview_plots(df: pd.DataFrame, output_dir: Path, model: str, total_catalog: int) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    _safe_hist(axes[0, 0], df[f"{model}_age_mode"], bins=50, title="PARSEC age modes", xlabel="age [Myr]", logx=True)
    _safe_hist(axes[0, 1], df[f"{model}_av_mode"], bins=50, title="A_V modes", xlabel="A_V")
    _safe_hist(axes[0, 2], df[f"{model}_mass_mf_fit_50"], bins=50, title="Mass median", xlabel="M_sun", logx=True)
    _safe_hist(axes[1, 0], df["acceptance_mean"], bins=50, title="Mean acceptance fraction", xlabel="acceptance")
    _safe_hist(axes[1, 1], df["autocorr_max"], bins=50, title="Max autocorr time", xlabel="steps", logx=True)
    _safe_hist(axes[1, 2], df[f"{model}_runtime_sec"] / 60.0, bins=50, title="Runtime per cluster", xlabel="minutes")
    path = output_dir / "overview_histograms.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["overview_histograms"] = str(path)

    flag_counts = Counter(flag for flags in df["quality_flags"].str.split(";") for flag in flags if flag)
    common_flags = flag_counts.most_common(16)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    completed = int(len(df))
    remaining = max(int(total_catalog) - completed, 0)
    axes[0].bar(["complete", "not yet synced/finished"], [completed, remaining], color=["#246b8f", "#b8b8b8"])
    axes[0].set_title("Catalog progress")
    axes[0].set_ylabel("clusters")
    if common_flags:
        labels, counts = zip(*common_flags)
        y = np.arange(len(labels))
        axes[1].barh(y, counts, color="#8f3f24")
        axes[1].set_yticks(y, labels)
        axes[1].invert_yaxis()
        axes[1].set_title("Most common quality flags")
    else:
        axes[1].text(0.5, 0.5, "no flags", ha="center", va="center", transform=axes[1].transAxes)
    path = output_dir / "progress_and_quality_flags.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["progress_and_quality_flags"] = str(path)

    map_counts = Counter()
    for raw in df["prior_map_counts"]:
        map_counts.update(_parse_counter(raw))
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if map_counts:
        labels, counts = zip(*map_counts.most_common())
        ax.bar(labels, counts, color="#4d6f3f")
        ax.set_title("Dust map usage by member-star lookup")
        ax.set_ylabel("member-star lookups")
        ax.tick_params(axis="x", rotation=25)
    else:
        ax.text(0.5, 0.5, "no dust map counts", ha="center", va="center", transform=ax.transAxes)
    path = output_dir / "dust_map_usage.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["dust_map_usage"] = str(path)

    return paths


def _choose_clusters(df: pd.DataFrame, model: str, max_clusters: int, seed: int) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    pieces.append(df.sort_values("acceptance_mean", ascending=True).head(6))
    pieces.append(df.sort_values("autocorr_max", ascending=False).head(6))
    pieces.append(df.loc[df["quality_flags"].str.contains("age_at_|av_at_", regex=True)].head(6))
    pieces.append(df.loc[df["quality_flags"].eq("")].sample(n=min(8, int((df["quality_flags"].eq("")).sum())), random_state=seed))
    pieces.append(df.sample(n=min(8, len(df)), random_state=seed + 1))
    selected = pd.concat(pieces, ignore_index=True).drop_duplicates("name")
    selected = selected.head(max_clusters).copy()
    return selected[["name", "quality_flags", f"{model}_age_mode", f"{model}_av_mode", f"{model}_mass_mf_fit_50", "acceptance_mean", "autocorr_max"]]


def analyze_run(run_dir: Path, model: str, total_catalog: int, seed: int, selected_count: int) -> Path:
    run_dir = run_dir.expanduser().resolve()
    output_dir = run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(run_dir / "cluster_results.csv")
    diagnostics = _diagnostics_frame(run_dir, model)
    df = results.merge(diagnostics, on="name", how="left", validate="one_to_one")
    df["quality_flags"] = df.apply(lambda row: ";".join(_quality_flags(row, model)), axis=1)
    df["quality_flag_count"] = df["quality_flags"].map(lambda raw: len([flag for flag in raw.split(";") if flag]))
    df["quality_bucket"] = np.select(
        [df["quality_flag_count"].eq(0), df["quality_flag_count"].le(2)],
        ["clean", "review"],
        default="problem",
    )

    all_path = output_dir / "quality_table.csv"
    df.to_csv(all_path, index=False)

    selected = _choose_clusters(df, model=model, max_clusters=selected_count, seed=seed)
    selected_path = output_dir / "selected_clusters_for_visual_review.txt"
    selected_path.write_text("\n".join(selected["name"].astype(str)) + "\n", encoding="utf-8")
    selected.to_csv(output_dir / "selected_clusters_for_visual_review.csv", index=False)

    plot_paths = _save_overview_plots(df, output_dir, model=model, total_catalog=total_catalog)

    flag_counts = Counter(flag for flags in df["quality_flags"].str.split(";") for flag in flags if flag)
    summary = {
        "run_dir": str(run_dir),
        "model": model,
        "total_catalog_clusters": int(total_catalog),
        "completed_clusters_in_results": int(len(df)),
        "completion_fraction": float(len(df) / total_catalog) if total_catalog else None,
        "posterior_files": len(list((run_dir / model / "posterior_samples").glob("*.npz"))),
        "diagnostic_files": len(list((run_dir / model / "sampler_diagnostics").glob("*.json"))),
        "mass_draw_files": len(list((run_dir / model / "mass_mf_fit_draws").glob("*.csv"))),
        "status_counts": df[f"{model}_status"].value_counts(dropna=False).to_dict(),
        "mass_status_counts": df[f"{model}_mass_mf_fit_status"].value_counts(dropna=False).to_dict(),
        "quality_bucket_counts": df["quality_bucket"].value_counts(dropna=False).to_dict(),
        "quality_flag_counts": dict(flag_counts.most_common()),
        "age_myr_quantiles": df[f"{model}_age_mode"].quantile([0.05, 0.16, 0.50, 0.84, 0.95]).to_dict(),
        "av_quantiles": df[f"{model}_av_mode"].quantile([0.05, 0.16, 0.50, 0.84, 0.95]).to_dict(),
        "mass_msun_quantiles": df[f"{model}_mass_mf_fit_50"].quantile([0.05, 0.16, 0.50, 0.84, 0.95]).to_dict(),
        "acceptance_mean_quantiles": df["acceptance_mean"].quantile([0.05, 0.16, 0.50, 0.84, 0.95]).to_dict(),
        "autocorr_max_quantiles": df["autocorr_max"].quantile([0.05, 0.16, 0.50, 0.84, 0.95]).to_dict(),
        "runtime_minutes_quantiles": (df[f"{model}_runtime_sec"] / 60.0).quantile([0.05, 0.16, 0.50, 0.84, 0.95]).to_dict(),
        "plot_paths": plot_paths,
        "quality_table_csv": str(all_path),
        "selected_clusters_csv": str(output_dir / "selected_clusters_for_visual_review.csv"),
        "selected_clusters_txt": str(selected_path),
    }
    summary_path = output_dir / "quality_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a saved Chronos PARSEC run for fit quality.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", default="parsec")
    parser.add_argument("--total-catalog", type=int, default=7167)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--selected-count", type=int, default=24)
    args = parser.parse_args()
    analyze_run(
        run_dir=args.run_dir,
        model=args.model,
        total_catalog=args.total_catalog,
        seed=args.seed,
        selected_count=args.selected_count,
    )


if __name__ == "__main__":
    main()
