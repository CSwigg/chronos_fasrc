from __future__ import annotations

import argparse
from collections.abc import Iterable
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

_CACHE_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))
_CACHE_USER = os.environ.get("USER", "user")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / f"chronos-mpl-{_CACHE_USER}"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / f"chronos-xdg-{_CACHE_USER}"))

from chronos.run_chronos.dual_model import _save_isochrone_plot, _save_posterior_plot, _slugify
from chronos.run_chronos.pipeline import ChronosFitConfig, configure_cluster_fitter, prepare_member_photometry
from workflows.config import load_runtime_paths
from workflows.run_chronos_hunt_lt150_mf_fit import DEFAULT_OUTPUT_DIRNAME


def _split_models(raw: object) -> tuple[str, ...]:
    if isinstance(raw, str):
        return tuple(model.strip() for model in raw.split(",") if model.strip())
    if isinstance(raw, Iterable):
        return tuple(str(model).strip() for model in raw if str(model).strip())
    return ()


def _load_chronos_samples(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as payload:
        posterior = np.asarray(payload["posterior"], dtype=float)
        columns = [str(column) for column in payload["columns"]]
    column_index = {column: index for index, column in enumerate(columns)}
    required = ["log_age", "feh", "av", "skewness", "scale"]
    missing = [column for column in required if column not in column_index]
    if missing:
        raise KeyError(f"Posterior file {npz_path} is missing columns: {missing}")
    return np.column_stack([posterior[:, column_index[column]] for column in required])


def _resolve_artifact_path(raw_value: object, fallback: Path) -> Path:
    if raw_value is not None and str(raw_value).strip() and str(raw_value).strip().lower() != "nan":
        path = Path(str(raw_value)).expanduser()
        if path.exists():
            return path
    return fallback


def _float_from_row(row: pd.Series, key: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
    if not np.isfinite(value):
        raise ValueError(f"Missing finite {key!r} in cluster_results.csv for {row.get('name')}")
    return float(value)


def _isochrone_dirs_for(paths) -> dict[str, str] | None:
    if paths.inputs.mist_isochrone_dir is None:
        return None
    return {"mist": str(paths.inputs.mist_isochrone_dir)}


def plot_run(
    *,
    config_path: str | Path | None = None,
    run_dirname: str = DEFAULT_OUTPUT_DIRNAME,
    run_root: str | Path | None = None,
    models: tuple[str, ...] | None = None,
    clusters: tuple[str, ...] | None = None,
    max_clusters: int | None = None,
    make_posterior_plots: bool = True,
    make_isochrone_plots: bool = True,
    overwrite: bool = False,
) -> Path:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    paths = load_runtime_paths(config_path)
    output_root = Path(run_root).expanduser().resolve() if run_root is not None else paths.outputs.chronos_dir / run_dirname
    results_path = output_root / "cluster_results.csv"
    results = pd.read_csv(results_path)
    if clusters:
        wanted = {str(cluster) for cluster in clusters}
        results = results.loc[results["name"].astype(str).isin(wanted)].copy()
    if max_clusters is not None:
        results = results.head(int(max_clusters)).copy()

    grouped_data: dict[str, pd.DataFrame] = {}
    if make_isochrone_plots:
        df_stars = pd.read_csv(paths.inputs.member_catalog_csv)
        df_clusters = pd.read_csv(paths.inputs.cluster_catalog_csv)
        fit_data = prepare_member_photometry(df_stars, df_clusters)
        grouped_data = {
            str(cluster_name): group.reset_index(drop=True)
            for cluster_name, group in fit_data.groupby("label", sort=False)
        }

    isochrone_dirs = _isochrone_dirs_for(paths)
    completed = 0
    skipped = 0
    errors: list[str] = []
    iterator = tqdm(
        list(results.iterrows()),
        desc="Plot Chronos outputs",
        unit="cluster",
        dynamic_ncols=True,
    )
    for _, row in iterator:
        cluster_name = str(row["name"])
        slug = _slugify(cluster_name)
        row_models = models or _split_models(row.get("model_names"))
        for model_name in row_models:
            model_root = output_root / model_name
            status = str(row.get(f"{model_name}_status", "")).strip()
            if status != "success":
                skipped += 1
                continue

            if make_posterior_plots:
                posterior_output = model_root / "posterior_plots" / f"{slug}_posterior.png"
                if overwrite or not posterior_output.exists():
                    sample_path = _resolve_artifact_path(
                        row.get(f"{model_name}_posterior_samples_npz"),
                        model_root / "posterior_samples" / f"{slug}_{model_name}_posterior.npz",
                    )
                    try:
                        samples = _load_chronos_samples(sample_path)
                        _save_posterior_plot(samples, posterior_output, hdi_prob=0.64)
                        completed += 1
                    except Exception as exc:
                        errors.append(f"{cluster_name} {model_name} posterior: {exc}")
                else:
                    skipped += 1

            if make_isochrone_plots:
                isochrone_output = model_root / "isochrone_plots" / f"{slug}_isochrone.png"
                if overwrite or not isochrone_output.exists():
                    df_group = grouped_data.get(cluster_name)
                    if df_group is None:
                        errors.append(f"{cluster_name} {model_name} isochrone: missing member photometry")
                        continue
                    try:
                        fitter = configure_cluster_fitter(
                            df_group=df_group,
                            fit_config=ChronosFitConfig(models=model_name, isochrone_dirs=isochrone_dirs),
                        )
                        _save_isochrone_plot(
                            fitter,
                            age_mode=_float_from_row(row, f"{model_name}_age_mode"),
                            age_lo=_float_from_row(row, f"{model_name}_age_lo"),
                            age_hi=_float_from_row(row, f"{model_name}_age_hi"),
                            av_mode=_float_from_row(row, f"{model_name}_av_mode"),
                            av_lo=_float_from_row(row, f"{model_name}_av_lo"),
                            av_hi=_float_from_row(row, f"{model_name}_av_hi"),
                            output_path=isochrone_output,
                        )
                        completed += 1
                    except Exception as exc:
                        errors.append(f"{cluster_name} {model_name} isochrone: {exc}")
                else:
                    skipped += 1

    summary_path = output_root / "plot_outputs_summary.json"
    import json

    summary_path.write_text(
        json.dumps(
            {
                "run_root": str(output_root),
                "cluster_results_csv": str(results_path),
                "n_cluster_rows": int(len(results)),
                "plot_products_written": int(completed),
                "plot_products_skipped": int(skipped),
                "n_errors": int(len(errors)),
                "errors": errors[:200],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if errors:
        print(f"Completed with {len(errors)} plotting errors. See {summary_path}", flush=True)
    else:
        print(f"Plot products complete. Summary: {summary_path}", flush=True)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create posterior and isochrone plots from saved Chronos fit outputs."
    )
    parser.add_argument("--config", type=str, default="configs/paths.toml")
    parser.add_argument("--run-dirname", type=str, default=DEFAULT_OUTPUT_DIRNAME)
    parser.add_argument("--run-root", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--clusters", nargs="*", default=None)
    parser.add_argument("--max-clusters", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=("posterior", "isochrone"),
        default=("posterior", "isochrone"),
    )
    args = parser.parse_args()
    plot_run(
        config_path=args.config,
        run_dirname=args.run_dirname,
        run_root=args.run_root,
        models=tuple(args.models) if args.models else None,
        clusters=tuple(args.clusters) if args.clusters else None,
        max_clusters=args.max_clusters,
        make_posterior_plots="posterior" in args.plots,
        make_isochrone_plots="isochrone" in args.plots,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
