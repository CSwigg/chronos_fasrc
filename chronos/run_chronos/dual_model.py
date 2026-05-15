from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import math
import multiprocessing as mp
import os
import re
import sys
import tempfile
import time
import warnings
from typing import Any

warnings.filterwarnings("ignore", message=r"\s*ArviZ is undergoing a major refactor.*", category=FutureWarning)
warnings.filterwarnings("ignore", message="Configuration file not found:.*", module="dustmaps.config")
warnings.filterwarnings("ignore", message="Overriding default configuration file with.*", module="dustmaps.config")

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from chronos.bayes_fitting.ChronosSkewedCauchy_bayes import ChronosSkewCauchyBayes
from chronos.run_chronos.mass_estimation import (
    ChronosAgeSummary,
    ClusterMassEstimate,
    IntervalSummary,
    estimate_cluster_mass,
    estimate_legacy_imf_masses,
)
from chronos.run_chronos.pipeline import (
    ChronosFitConfig,
    mode_reals,
    prepare_member_photometry,
    summarize_skew_cauchy_samples,
)
from chronos.utils.ExtinctionPrior import ExtinctionPrior
from workflows.config import load_runtime_paths


matplotlib.use("Agg")
plt.ioff()


@dataclass(frozen=True)
class ExtinctionPriorDescriptor:
    mode: str
    center_av: float | None
    floor_av: float | None
    sigma_av: float
    valid_fraction: float
    valid_count: int
    total_count: int
    map_name: str
    map_counts: dict[str, int]
    floor_map_counts: dict[str, int]


@dataclass(frozen=True)
class DualModelRunConfig:
    fit_config: ChronosFitConfig = ChronosFitConfig()
    age_prior: str = "log"
    sigma_av: float = 0.10
    binary_mass_scale: float = 1.25
    include_swiggum_masses: bool = False
    mass_n_draws: int = 100
    mass_n_imfs: int = 1000
    mass_output_prefix: str = "mass_swiggum"
    save_mass_draws: bool = False
    save_mass_diagnostic_plots: bool = False
    save_fit_plots: bool = True
    save_posterior_samples: bool = False
    posterior_sample_size: int = 20_000
    quiet_worker_output: bool = True
    print_cluster_updates: bool = False
    model_names: tuple[str, ...] = ("parsec", "baraffe")
    output_dirname: str = "dual_model_refit"


_WORKER_CLUSTER_DATA: dict[str, pd.DataFrame] = {}
_WORKER_CLUSTER_METADATA: dict[str, dict[str, Any]] = {}
_WORKER_EXTINCTION_PRIOR: ExtinctionPrior | None = None
_WORKER_RUN_CONFIG: DualModelRunConfig | None = None
_WORKER_OUTPUT_ROOT: Path | None = None


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    return slug or "cluster"


def _stable_seed(*parts: str) -> int:
    token = "::".join(str(part) for part in parts)
    seed = 0
    for char in token:
        seed = (seed * 131 + ord(char)) % (2**32 - 1)
    return int(seed or 1)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def _atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8", newline="") as handle:
        df.to_csv(handle.name, index=False)
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def _atomic_save_npz_compressed(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".npz", dir=path.parent, delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _clean_float(value: Any) -> float | None:
    try:
        value = float(value)
    except Exception:
        return None
    if math.isfinite(value):
        return value
    return None


def _build_extinction_descriptor(details: pd.DataFrame, sigma_av: float) -> ExtinctionPriorDescriptor:
    valid_av = pd.to_numeric(details.loc[details["is_valid"], "av"], errors="coerce").to_numpy(dtype=float)
    fallback_floor = pd.to_numeric(details["floor_av"], errors="coerce").to_numpy(dtype=float)
    valid_fraction = float(np.mean(details["is_valid"].to_numpy(dtype=bool))) if len(details) else 0.0
    valid_map_names = (
        details.loc[details["is_valid"], "map_name"].replace("", np.nan).dropna()
        if "map_name" in details.columns
        else pd.Series(dtype=object)
    )
    floor_map_names = (
        details["floor_map_name"].replace("", np.nan).dropna()
        if "floor_map_name" in details.columns
        else pd.Series(dtype=object)
    )
    map_counts = {
        str(map_name): int(count)
        for map_name, count in valid_map_names.value_counts().sort_index().items()
    }
    floor_map_counts = {
        str(map_name): int(count)
        for map_name, count in floor_map_names.value_counts().sort_index().items()
    }
    map_name = "mixed" if len(map_counts) > 1 else next(iter(map_counts), "none")
    if len(details) and np.all(details["is_valid"].to_numpy(dtype=bool)):
        center = float(np.nanmedian(valid_av))
        return ExtinctionPriorDescriptor(
            mode="gaussian",
            center_av=center,
            floor_av=None,
            sigma_av=sigma_av,
            valid_fraction=valid_fraction,
            valid_count=int(np.sum(np.isfinite(valid_av))),
            total_count=int(len(details)),
            map_name=map_name,
            map_counts=map_counts,
            floor_map_counts=floor_map_counts,
        )

    floor_candidates = np.concatenate(
        [
            valid_av[np.isfinite(valid_av)],
            fallback_floor[np.isfinite(fallback_floor)],
        ]
    )
    floor_value = float(np.nanmedian(floor_candidates)) if floor_candidates.size else 0.0
    return ExtinctionPriorDescriptor(
        mode="lower_limit",
        center_av=None,
        floor_av=floor_value,
        sigma_av=sigma_av,
        valid_fraction=valid_fraction,
        valid_count=int(np.sum(np.isfinite(valid_av))),
        total_count=int(len(details)),
        map_name=map_name if map_counts else "lower_limit",
        map_counts=map_counts,
        floor_map_counts=floor_map_counts,
    )


class ChronosSkewCauchyBayesAVPrior(ChronosSkewCauchyBayes):
    def __init__(self, *args, extinction_prior: ExtinctionPriorDescriptor, age_prior: str = "log", **kwargs):
        super().__init__(*args, **kwargs)
        self.extinction_prior = extinction_prior
        self.age_prior = str(age_prior).strip().lower()

    def log_prior(self, theta):
        base = super().log_prior(theta)
        if not np.isfinite(base):
            return -np.inf

        log_age, _, av, _, _ = theta
        if self.age_prior in {"linear", "age", "linear_age"}:
            base += float(log_age) * math.log(10.0)
        elif self.age_prior not in {"log", "logage", "log_age"}:
            raise ValueError(f"Unsupported age_prior={self.age_prior!r}")

        sigma = max(float(self.extinction_prior.sigma_av), 1e-6)
        if self.extinction_prior.mode == "gaussian":
            center = float(self.extinction_prior.center_av or 0.0)
            return base - 0.5 * ((av - center) / sigma) ** 2

        floor = float(self.extinction_prior.floor_av or 0.0)
        if av >= floor:
            return base
        return base - 0.5 * ((av - floor) / sigma) ** 2


def _configure_cluster_fitter(
    df_group: pd.DataFrame,
    fit_config: ChronosFitConfig,
    extinction_prior: ExtinctionPriorDescriptor,
    age_prior: str,
) -> ChronosSkewCauchyBayesAVPrior:
    fitter = ChronosSkewCauchyBayesAVPrior(
        **fit_config.chronos_kwargs(data=df_group),
        extinction_prior=extinction_prior,
        age_prior=age_prior,
    )
    fitter.set_fitting_kwargs(**fit_config.fitting_kwargs())
    fitter.set_bounds(**fit_config.bayes_bounds())
    return fitter


def _summarize_mass_outputs(
    masses: np.ndarray,
    *,
    binary_mass_scale: float,
) -> dict[str, float | None]:
    return estimate_legacy_imf_masses(
        masses=np.asarray(masses, dtype=float),
        binary_mass_scale=binary_mass_scale,
    )


def _posterior_age_summary(
    *,
    cluster_name: str,
    summary,
) -> ChronosAgeSummary:
    return ChronosAgeSummary(
        name=str(cluster_name),
        age_myr=IntervalSummary(
            p16=float(summary.age_lo),
            p50=float(summary.age_mode),
            p84=float(summary.age_hi),
        ),
        av_mag=IntervalSummary(
            p16=float(summary.av_lo),
            p50=float(summary.av_mode),
            p84=float(summary.av_hi),
        ),
        status="success",
        age_samples_myr=np.asarray(summary.age_samples_myr, dtype=float),
        av_samples_mag=np.asarray(summary.av_samples, dtype=float),
    )


def _empty_swiggum_mass_outputs(
    *,
    n_members_total: int,
    status: str,
    skip_reason: str | None,
    prefix: str = "mass_swiggum",
) -> dict[str, Any]:
    return ClusterMassEstimate(
        method="swiggum",
        status=status,
        total_mass_msun=None,
        n_draws_succeeded=0,
        n_members_total=int(n_members_total),
        n_members_used=0,
        skip_reason=skip_reason,
        details={},
    ).as_row(prefix)


def _save_posterior_plot(
    samples: np.ndarray,
    output_path: Path,
    *,
    hdi_prob: float,
) -> None:
    log_age, _, av, skewness, scale = np.asarray(samples).T
    plot_series = [10**log_age / 1e6, av, skewness, scale]
    labels = ["Age (Myr)", "AV (mag)", "Skewness", "Scale"]

    fig, axes = plt.subplots(1, len(labels), figsize=(len(labels) * 4, 5))
    for label, values, axis in zip(labels, plot_series, axes):
        axis.hist(values, bins=50, histtype="step", color="k")
        axis.hist(values, bins=50, histtype="stepfilled", color="k", alpha=0.25)
        lo, hi = az.hdi(values, hdi_prob=hdi_prob)
        axis.axvline(mode_reals(values, bins=100), c="k", alpha=0.5)
        axis.axvline(lo, c="k", alpha=0.5)
        axis.axvline(hi, c="k", alpha=0.5)
        axis.set_xlabel(label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _save_isochrone_plot(
    fitter: ChronosSkewCauchyBayesAVPrior,
    *,
    age_mode: float,
    age_lo: float,
    age_hi: float,
    av_mode: float,
    av_lo: float,
    av_hi: float,
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(6, 9))
    axis.scatter(*fitter.distance_handler.fit_data["hrd"].T, s=40, c="tab:purple", alpha=0.85)
    axis.set_ylim(14, -4)
    axis.set_xlim(-1, 5)
    axis.set_xlabel(r"$G_{BP} - G_{RP}$")
    axis.set_ylabel(r"$M_G$")

    isochrone = fitter.isochrone_handler.model(
        logAge=np.log10(age_mode * 1e6),
        feh=0.0,
        A_V=av_mode,
        g_rp=fitter.use_grp,
    )
    axis.plot(*isochrone.T, c="k", alpha=0.7, zorder=0)
    axis.annotate(
        (
            r"${{{:.1f}}}^{{+{:.1f}}}_{{{:.1f}}}$ Myr"
            "\n"
            r"$A_V={{{:.2f}}}^{{+{:.2f}}}_{{{:.2f}}}$ mag"
        ).format(
            age_mode,
            age_hi - age_mode,
            age_lo - age_mode,
            av_mode,
            av_hi - av_mode,
            av_lo - av_mode,
        ),
        (0.98, 0.98),
        xycoords="axes fraction",
        ha="right",
        va="top",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _best_fit_payload(best_fit: np.ndarray, best_log_prob: float | None) -> dict[str, float | None]:
    best_fit = np.asarray(best_fit, dtype=float)
    keys = ["log_age", "feh", "av", "skewness", "scale"]
    payload = {f"best_fit_{key}": None for key in keys}
    payload.update({f"best_fit_{key}": _clean_float(value) for key, value in zip(keys, best_fit, strict=False)})
    if best_fit.size:
        payload["best_fit_age_myr"] = _clean_float(10 ** float(best_fit[0]) / 1e6)
    else:
        payload["best_fit_age_myr"] = None
    payload["best_log_prob"] = _clean_float(best_log_prob)
    return payload


def _save_posterior_samples_and_diagnostics(
    *,
    sampler,
    samples: np.ndarray,
    best_fit: np.ndarray,
    output_path: Path,
    diagnostics_path: Path,
    fit_config: ChronosFitConfig,
    cluster_name: str,
    model_name: str,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    samples = np.asarray(samples, dtype=float)
    log_prob = np.asarray(sampler.get_log_prob(flat=True), dtype=float)
    if samples.shape[0] != log_prob.shape[0]:
        raise ValueError(
            f"Sample/log-probability length mismatch for {cluster_name} {model_name}: "
            f"{samples.shape[0]} != {log_prob.shape[0]}"
        )

    finite = np.isfinite(log_prob) & np.all(np.isfinite(samples), axis=1)
    candidate_indices = np.flatnonzero(finite)
    rng = np.random.default_rng(int(seed))
    n_save = min(int(sample_size), int(candidate_indices.size))
    if n_save > 0 and n_save < candidate_indices.size:
        selected_indices = np.sort(rng.choice(candidate_indices, size=n_save, replace=False))
    else:
        selected_indices = candidate_indices

    posterior_columns = np.array(
        ["log_age", "age_myr", "feh", "av", "skewness", "scale", "log_prob"],
        dtype="U16",
    )
    selected_samples = samples[selected_indices]
    selected_log_prob = log_prob[selected_indices]
    age_myr = 10 ** selected_samples[:, 0] / 1e6 if selected_samples.size else np.array([], dtype=float)
    posterior = np.column_stack(
        [
            selected_samples[:, 0],
            age_myr,
            selected_samples[:, 1],
            selected_samples[:, 2],
            selected_samples[:, 3],
            selected_samples[:, 4],
            selected_log_prob,
        ]
    ) if selected_samples.size else np.empty((0, len(posterior_columns)), dtype=float)
    _atomic_save_npz_compressed(
        output_path,
        posterior=posterior,
        columns=posterior_columns,
        selected_flat_indices=selected_indices.astype(np.int64),
    )

    best_index = int(np.nanargmax(log_prob)) if log_prob.size and np.any(np.isfinite(log_prob)) else None
    best_log_prob = float(log_prob[best_index]) if best_index is not None else None
    autocorr_time = None
    autocorr_error = None
    try:
        autocorr_time = np.asarray(sampler.get_autocorr_time(tol=0), dtype=float).tolist()
    except Exception as exc:
        autocorr_error = str(exc)

    acceptance_fraction = np.asarray(getattr(sampler, "acceptance_fraction", []), dtype=float)
    diagnostics = {
        "cluster": str(cluster_name),
        "model": str(model_name),
        "posterior_samples_npz": str(output_path),
        "posterior_columns": posterior_columns.tolist(),
        "nwalkers": int(fit_config.nwalkers),
        "burnin": int(fit_config.burnin),
        "nsteps": int(fit_config.nsteps),
        "ndim": int(samples.shape[1]) if samples.ndim == 2 else None,
        "total_flat_samples": int(samples.shape[0]),
        "finite_flat_samples": int(candidate_indices.size),
        "saved_flat_samples": int(len(selected_indices)),
        "posterior_sample_size_requested": int(sample_size),
        "posterior_sample_seed": int(seed),
        "best_flat_index": best_index,
        **_best_fit_payload(np.asarray(best_fit, dtype=float), best_log_prob),
        "acceptance_fraction_mean": _clean_float(np.nanmean(acceptance_fraction)) if acceptance_fraction.size else None,
        "acceptance_fraction_min": _clean_float(np.nanmin(acceptance_fraction)) if acceptance_fraction.size else None,
        "acceptance_fraction_max": _clean_float(np.nanmax(acceptance_fraction)) if acceptance_fraction.size else None,
        "acceptance_fraction_by_walker": acceptance_fraction.tolist(),
        "autocorr_time": autocorr_time,
        "autocorr_time_error": autocorr_error,
    }
    _atomic_write_json(diagnostics_path, diagnostics)
    return {
        "posterior_samples_npz": str(output_path),
        "sampler_diagnostics_json": str(diagnostics_path),
        "posterior_saved_samples": int(len(selected_indices)),
        **_best_fit_payload(np.asarray(best_fit, dtype=float), best_log_prob),
    }


def _fit_single_model(
    cluster_name: str,
    df_group: pd.DataFrame,
    *,
    cluster_row: pd.Series,
    model_name: str,
    descriptor: ExtinctionPriorDescriptor,
    output_root: Path,
    run_config: DualModelRunConfig,
) -> dict[str, Any]:
    started = time.time()
    fit_config = ChronosFitConfig(**{**asdict(run_config.fit_config), "models": model_name})
    plot_slug = _slugify(cluster_name)
    model_output_root = output_root / model_name
    posterior_path = model_output_root / "posterior_plots" / f"{plot_slug}_posterior.png"
    isochrone_path = model_output_root / "isochrone_plots" / f"{plot_slug}_isochrone.png"
    posterior_samples_path = model_output_root / "posterior_samples" / f"{plot_slug}_{model_name}_posterior.npz"
    diagnostics_path = model_output_root / "sampler_diagnostics" / f"{plot_slug}_{model_name}_diagnostics.json"

    try:
        fitter = _configure_cluster_fitter(
            df_group=df_group,
            fit_config=fit_config,
            extinction_prior=descriptor,
            age_prior=run_config.age_prior,
        )
        sampler, best_fit, samples = fitter.fit_bayesian(**fit_config.sampler_kwargs())
        summary = summarize_skew_cauchy_samples(samples, hdi_prob=fit_config.summary_hdi_prob)
        _distances, masses, keep_mask = fitter.compute_fit_info(
            logAge=np.log10(summary.age_mode * 1e6),
            feh=0.0,
            A_V=summary.av_mode,
            g_rp=fitter.use_grp,
            signed_distance=True,
        )
        mass_outputs = _summarize_mass_outputs(
            masses[keep_mask],
            binary_mass_scale=run_config.binary_mass_scale,
        )
        swiggum_outputs: dict[str, Any] = {}
        if run_config.include_swiggum_masses:
            age_summary = _posterior_age_summary(
                cluster_name=cluster_name,
                summary=summary,
            )
            draws_output_path = None
            diagnostic_plot_path = None
            mass_prefix = str(run_config.mass_output_prefix or "mass_swiggum")
            if run_config.save_mass_draws:
                draws_output_path = (
                    model_output_root
                    / f"{mass_prefix}_draws"
                    / f"{plot_slug}_{mass_prefix}_draws.csv"
                )
            if run_config.save_mass_diagnostic_plots:
                diagnostic_plot_path = (
                    model_output_root
                    / f"{mass_prefix}_plots"
                    / f"{plot_slug}_{mass_prefix}_diagnostic.png"
                )
            swiggum_estimate = estimate_cluster_mass(
                cluster_name=cluster_name,
                df_group=df_group,
                cluster_row=cluster_row,
                age_summary=age_summary,
                mass_method="swiggum",
                models=model_name,
                isochrone_dirs=fit_config.isochrone_dirs,
                rng_seed=_stable_seed(cluster_name, model_name, "swiggum"),
                n_draws=int(run_config.mass_n_draws),
                swiggum_n_imf_draws=int(run_config.mass_n_imfs),
                draws_output_path=draws_output_path,
                diagnostic_plot_path=diagnostic_plot_path,
                diagnostic_title=f"{cluster_name} {model_name.upper()} mass-function fit",
            )
            swiggum_outputs = swiggum_estimate.as_row(mass_prefix)

        posterior_outputs: dict[str, Any] = {}
        if run_config.save_posterior_samples:
            posterior_outputs = _save_posterior_samples_and_diagnostics(
                sampler=sampler,
                samples=samples,
                best_fit=best_fit,
                output_path=posterior_samples_path,
                diagnostics_path=diagnostics_path,
                fit_config=fit_config,
                cluster_name=cluster_name,
                model_name=model_name,
                sample_size=int(run_config.posterior_sample_size),
                seed=_stable_seed(cluster_name, model_name, "posterior"),
            )

        if run_config.save_fit_plots:
            _save_posterior_plot(
                samples=samples,
                output_path=posterior_path,
                hdi_prob=fit_config.posterior_plot_hdi_prob,
            )
            _save_isochrone_plot(
                fitter,
                age_mode=summary.age_mode,
                age_lo=summary.age_lo,
                age_hi=summary.age_hi,
                av_mode=summary.av_mode,
                av_lo=summary.av_lo,
                av_hi=summary.av_hi,
                output_path=isochrone_path,
            )
        return {
            "status": "success",
            "runtime_sec": float(time.time() - started),
            "age_mode": summary.age_mode,
            "age_lo": summary.age_lo,
            "age_hi": summary.age_hi,
            "av_mode": summary.av_mode,
            "av_lo": summary.av_lo,
            "av_hi": summary.av_hi,
            "posterior_plot": str(posterior_path) if run_config.save_fit_plots else None,
            "isochrone_plot": str(isochrone_path) if run_config.save_fit_plots else None,
            **mass_outputs,
            **swiggum_outputs,
            **posterior_outputs,
        }
    except Exception as exc:
        swiggum_outputs: dict[str, Any] = {}
        if run_config.include_swiggum_masses:
            swiggum_outputs = _empty_swiggum_mass_outputs(
                n_members_total=int(len(df_group)),
                status="not_run",
                skip_reason="model_fit_failed",
                prefix=str(run_config.mass_output_prefix or "mass_swiggum"),
            )
        return {
            "status": f"error: {exc}",
            "runtime_sec": float(time.time() - started),
            "age_mode": None,
            "age_lo": None,
            "age_hi": None,
            "av_mode": None,
            "av_lo": None,
            "av_hi": None,
            "posterior_plot": str(posterior_path) if run_config.save_fit_plots else None,
            "isochrone_plot": str(isochrone_path) if run_config.save_fit_plots else None,
            "posterior_samples_npz": str(posterior_samples_path) if run_config.save_posterior_samples else None,
            "sampler_diagnostics_json": str(diagnostics_path) if run_config.save_posterior_samples else None,
            "posterior_saved_samples": None,
            **_best_fit_payload(np.array([], dtype=float), None),
            "mass_members_observed": None,
            "mass_cluster_imf_corrected": None,
            "mass_cluster_imf_binary_corrected": None,
            **swiggum_outputs,
        }


def _init_worker(
    cluster_data: dict[str, pd.DataFrame],
    cluster_metadata: dict[str, dict[str, Any]],
    extinction_map_path: str,
    bayestar2019_map_path: str | None,
    decaps_map_path: str | None,
    output_root: str,
    run_config: DualModelRunConfig,
) -> None:
    global _WORKER_CLUSTER_DATA, _WORKER_CLUSTER_METADATA, _WORKER_EXTINCTION_PRIOR, _WORKER_RUN_CONFIG, _WORKER_OUTPUT_ROOT
    _WORKER_CLUSTER_DATA = cluster_data
    _WORKER_CLUSTER_METADATA = cluster_metadata
    _WORKER_RUN_CONFIG = run_config
    _WORKER_OUTPUT_ROOT = Path(output_root)
    if run_config.quiet_worker_output:
        init_log = _WORKER_OUTPUT_ROOT / "worker_logs" / f"worker_init_{os.getpid()}.log"
        init_log.parent.mkdir(parents=True, exist_ok=True)
        with init_log.open("a", encoding="utf-8", buffering=1) as log_handle:
            log_handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | worker init pid={os.getpid()}\n")
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                with warnings.catch_warnings():
                    warnings.simplefilter("default")
                    _WORKER_EXTINCTION_PRIOR = ExtinctionPrior(
                        extinction_map_path,
                        bayestar2019_map_fname=bayestar2019_map_path,
                        decaps_map_fname=decaps_map_path,
                    )
    else:
        _WORKER_EXTINCTION_PRIOR = ExtinctionPrior(
            extinction_map_path,
            bayestar2019_map_fname=bayestar2019_map_path,
            decaps_map_fname=decaps_map_path,
        )


def _process_cluster_uncaptured(cluster_name: str) -> dict[str, Any]:
    if _WORKER_EXTINCTION_PRIOR is None or _WORKER_RUN_CONFIG is None or _WORKER_OUTPUT_ROOT is None:
        raise RuntimeError("worker globals not initialized")

    df_group = _WORKER_CLUSTER_DATA[cluster_name]
    cluster_row = pd.Series(_WORKER_CLUSTER_METADATA.get(cluster_name, {"name": cluster_name}))
    details = _WORKER_EXTINCTION_PRIOR.compute_prior_details(
        ra=df_group["ra"],
        dec=df_group["dec"],
        distance=df_group["distance_50"],
    )
    descriptor = _build_extinction_descriptor(details=details, sigma_av=_WORKER_RUN_CONFIG.sigma_av)

    model_results = {
        model_name: _fit_single_model(
            cluster_name=cluster_name,
            df_group=df_group,
            cluster_row=cluster_row,
            model_name=model_name,
            descriptor=descriptor,
            output_root=_WORKER_OUTPUT_ROOT,
            run_config=_WORKER_RUN_CONFIG,
        )
        for model_name in _WORKER_RUN_CONFIG.model_names
    }
    all_success = all(result["status"] == "success" for result in model_results.values())
    payload = {
        "name": cluster_name,
        "completed": all_success,
        "cluster_status": "success" if all_success else "partial",
        "model_names": list(_WORKER_RUN_CONFIG.model_names),
        "prior_mode": descriptor.mode,
        "prior_center_av": descriptor.center_av,
        "prior_floor_av": descriptor.floor_av,
        "prior_sigma_av": descriptor.sigma_av,
        "prior_valid_fraction": descriptor.valid_fraction,
        "prior_valid_count": descriptor.valid_count,
        "prior_total_count": descriptor.total_count,
        "prior_map_name": descriptor.map_name,
        "prior_map_counts": descriptor.map_counts,
        "prior_floor_map_counts": descriptor.floor_map_counts,
        "age_prior": _WORKER_RUN_CONFIG.age_prior,
    }
    payload.update(model_results)
    return payload


def _process_cluster(cluster_name: str) -> dict[str, Any]:
    if _WORKER_RUN_CONFIG is None or _WORKER_OUTPUT_ROOT is None:
        raise RuntimeError("worker globals not initialized")
    if not _WORKER_RUN_CONFIG.quiet_worker_output:
        return _process_cluster_uncaptured(cluster_name)

    worker_log = _WORKER_OUTPUT_ROOT / "worker_logs" / f"{_slugify(cluster_name)}.log"
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    with worker_log.open("a", encoding="utf-8", buffering=1) as log_handle:
        log_handle.write(
            "\n"
            + "=" * 78
            + "\n"
            + f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {cluster_name}\n"
        )
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
            with warnings.catch_warnings():
                warnings.simplefilter("default")
                payload = _process_cluster_uncaptured(cluster_name)
    payload["worker_log"] = str(worker_log)
    return payload


def _checkpoint_path(output_root: Path, cluster_name: str) -> Path:
    return output_root / "checkpoints" / f"{_slugify(cluster_name)}.json"


def _swiggum_checkpoint_finished(status: Any) -> bool:
    if status is None:
        return False
    status_str = str(status).strip()
    return bool(status_str) and status_str != "not_run"


def _payload_model_names(
    payload: dict[str, Any],
    fallback: tuple[str, ...] = ("parsec", "baraffe"),
) -> tuple[str, ...]:
    raw = payload.get("model_names")
    if isinstance(raw, str):
        names = tuple(name.strip() for name in raw.split(",") if name.strip())
    elif isinstance(raw, (list, tuple)):
        names = tuple(str(name).strip() for name in raw if str(name).strip())
    else:
        names = ()
    return names or fallback


def _cluster_is_complete(
    payload: dict[str, Any],
    *,
    require_swiggum: bool = False,
    model_names: tuple[str, ...] | None = None,
    mass_output_prefix: str = "mass_swiggum",
) -> bool:
    required_models = model_names or _payload_model_names(payload)
    models_complete = bool(payload.get("completed")) and all(
        payload.get(model_name, {}).get("status") == "success"
        for model_name in required_models
    )
    if not models_complete:
        return False
    if not require_swiggum:
        return True
    status_key = f"{mass_output_prefix}_status"
    return all(
        _swiggum_checkpoint_finished(payload.get(model_name, {}).get(status_key))
        for model_name in required_models
    )


def _load_existing_results(output_root: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    checkpoint_dir = output_root / "checkpoints"
    if not checkpoint_dir.exists():
        return results
    for checkpoint in checkpoint_dir.glob("*.json"):
        try:
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        except Exception:
            continue
        cluster_name = str(payload.get("name", "")).strip()
        if cluster_name:
            results[cluster_name] = payload
    return results


def _flatten_result(payload: dict[str, Any]) -> dict[str, Any]:
    row = {
        "name": payload["name"],
        "completed": payload.get("completed"),
        "cluster_status": payload.get("cluster_status"),
        "model_names": ",".join(_payload_model_names(payload)),
        "prior_mode": payload.get("prior_mode"),
        "prior_center_av": payload.get("prior_center_av"),
        "prior_floor_av": payload.get("prior_floor_av"),
        "prior_sigma_av": payload.get("prior_sigma_av"),
        "prior_valid_fraction": payload.get("prior_valid_fraction"),
        "prior_valid_count": payload.get("prior_valid_count"),
        "prior_total_count": payload.get("prior_total_count"),
        "prior_map_name": payload.get("prior_map_name"),
        "prior_map_counts": json.dumps(payload.get("prior_map_counts") or {}, sort_keys=True),
        "prior_floor_map_counts": json.dumps(payload.get("prior_floor_map_counts") or {}, sort_keys=True),
        "age_prior": payload.get("age_prior"),
        "worker_log": payload.get("worker_log"),
    }
    for model_name in _payload_model_names(payload):
        model_payload = payload.get(model_name, {})
        for key, value in model_payload.items():
            row[f"{model_name}_{key}"] = value
    return row


def _write_summary_csv(output_root: Path, results: dict[str, dict[str, Any]]) -> None:
    rows = [_flatten_result(results[name]) for name in sorted(results)]
    summary_path = output_root / "cluster_results.csv"
    _atomic_write_csv(summary_path, pd.DataFrame(rows))


def _count_swiggum_successes(
    results: dict[str, dict[str, Any]],
    model_names: tuple[str, ...],
    mass_output_prefix: str = "mass_swiggum",
) -> dict[str, int]:
    successes = {model_name: 0 for model_name in model_names}
    status_key = f"{mass_output_prefix}_status"
    for payload in results.values():
        for model_name in model_names:
            model_payload = payload.get(model_name, {})
            successes[model_name] += int(model_payload.get(status_key) == "success")
    return successes


def _format_progress_postfix(
    *,
    done: int,
    total: int,
    success: int,
    partial: int,
    swiggum_successes: dict[str, int] | None,
) -> str:
    parts = [
        f"done={done}/{total}",
        f"ok={success}",
        f"partial={partial}",
    ]
    if swiggum_successes is not None:
        parts.append(
            "swg="
            + ",".join(f"{model_name}:{count}" for model_name, count in swiggum_successes.items())
        )
    return " | ".join(parts)


def _print_terminal_banner(
    *,
    output_root: Path,
    run_config: DualModelRunConfig,
    total_clusters: int,
    pending_clusters: int,
    completed_at_start: int,
    n_processes: int,
    chunksize: int,
) -> None:
    lines = [
        "=" * 78,
        "Chronos Dual-Model Refit",
        f"output: {output_root}",
        f"models: {', '.join(run_config.model_names)}",
        f"age_prior: {run_config.age_prior}",
        f"clusters: total={total_clusters} pending={pending_clusters} already_complete={completed_at_start}",
        f"workers: n_processes={n_processes} chunksize={chunksize}",
    ]
    if run_config.include_swiggum_masses:
        lines.append(
            "masses: legacy + mass-function fit"
            f" | posterior_mass_draws={run_config.mass_n_draws}"
            f" | imf_draws={run_config.mass_n_imfs}"
            f" | output_prefix={run_config.mass_output_prefix}"
            f" | mass_plots={run_config.save_mass_diagnostic_plots}"
        )
    else:
        lines.append("masses: legacy only")
    lines.append(
        "outputs:"
        f" posterior_samples={run_config.save_posterior_samples}"
        f" | posterior_sample_size={run_config.posterior_sample_size}"
        f" | fit_plots={run_config.save_fit_plots}"
    )
    lines.append(
        "terminal:"
        f" quiet_worker_output={run_config.quiet_worker_output}"
        f" | print_cluster_updates={run_config.print_cluster_updates}"
        f" | worker_logs={output_root / 'worker_logs'}"
    )
    lines.append("=" * 78)
    print("\n".join(lines), flush=True)


def _print_stage(message: str) -> None:
    print(f"[setup] {message}", flush=True)


def _print_cluster_update(
    payload: dict[str, Any],
    *,
    done: int,
    total: int,
    include_swiggum_masses: bool,
    mass_output_prefix: str = "mass_swiggum",
) -> None:
    model_names = _payload_model_names(payload)
    parts = [
        f"[{done:>4}/{total:<4}]",
        str(payload.get("name", "unknown")),
        f"cluster={payload.get('cluster_status', 'unknown')}",
    ]
    for model_name in model_names:
        parts.append(f"{model_name}={payload.get(model_name, {}).get('status', 'unknown')}")
    if include_swiggum_masses:
        status_key = f"{mass_output_prefix}_status"
        parts.append(
            "mf="
            + ",".join(
                f"{model_name}:{payload.get(model_name, {}).get(status_key, 'not_run')}"
                for model_name in model_names
            )
        )
    runtime = max(
        float(payload.get(model_name, {}).get("runtime_sec") or 0.0)
        for model_name in model_names
    )
    parts.append(f"t={runtime:.1f}s")
    tqdm.write(" | ".join(parts), file=sys.stdout)


def run_dual_model_refit(
    *,
    config_path: str | Path | None = None,
    n_processes: int | None = None,
    force: bool = False,
    clusters: list[str] | None = None,
    run_config: DualModelRunConfig | None = None,
) -> Path:
    paths = load_runtime_paths(config_path)
    run_config = run_config or DualModelRunConfig()
    if paths.inputs.mist_isochrone_dir is not None:
        isochrone_dirs = dict(run_config.fit_config.isochrone_dirs or {})
        isochrone_dirs.setdefault("mist", str(paths.inputs.mist_isochrone_dir))
        run_config = replace(
            run_config,
            fit_config=replace(run_config.fit_config, isochrone_dirs=isochrone_dirs),
        )
    output_root = paths.outputs.chronos_dir / run_config.output_dirname
    output_root.mkdir(parents=True, exist_ok=True)

    _print_stage(f"using config: {paths.config_path}")
    _print_stage(f"writing outputs under: {output_root}")
    _print_stage(f"loading member catalog: {paths.inputs.member_catalog_csv}")
    df_stars = pd.read_csv(paths.inputs.member_catalog_csv)
    _print_stage(f"loaded {len(df_stars):,} member rows")
    _print_stage(f"loading cluster catalog: {paths.inputs.cluster_catalog_csv}")
    df_clusters = pd.read_csv(paths.inputs.cluster_catalog_csv)
    _print_stage(f"loaded {len(df_clusters):,} cluster rows")
    if clusters is not None:
        cluster_order = {str(name): index for index, name in enumerate(clusters)}
        cluster_set = set(cluster_order)
        df_clusters = df_clusters.loc[df_clusters["name"].astype(str).isin(cluster_set)].copy()
        df_clusters["__chronos_run_order"] = df_clusters["name"].astype(str).map(cluster_order)
        df_clusters = df_clusters.sort_values("__chronos_run_order").drop(columns=["__chronos_run_order"])
        _print_stage(f"applied cluster subset: {len(df_clusters):,} clusters remain")

    _print_stage("preparing Chronos photometry table")
    fit_data = prepare_member_photometry(df_stars, df_clusters)
    _print_stage(f"prepared {len(fit_data):,} stellar photometry rows for fitting")
    grouped_data = {
        str(cluster_name): group.reset_index(drop=True)
        for cluster_name, group in fit_data.groupby("label", sort=False)
    }
    _print_stage(f"assembled fit groups for {len(grouped_data):,} clusters")
    cluster_metadata = {
        str(row["name"]): row.to_dict()
        for _, row in df_clusters.iterrows()
    }

    _print_stage("loading any existing checkpoints")
    existing_results = _load_existing_results(output_root)
    if clusters is not None:
        all_cluster_names = [str(name) for name in clusters if str(name) in grouped_data]
    else:
        all_cluster_names = list(grouped_data.keys())
    if force:
        pending_clusters = all_cluster_names
    else:
        pending_clusters = [
            name
            for name in all_cluster_names
            if not _cluster_is_complete(
                existing_results.get(name, {}),
                require_swiggum=run_config.include_swiggum_masses,
                model_names=run_config.model_names,
                mass_output_prefix=run_config.mass_output_prefix,
            )
        ]
    _print_stage(
        f"checkpoint scan complete: {len(existing_results):,} checkpoint payloads found, "
        f"{len(pending_clusters):,} clusters still pending"
    )

    n_processes = int(n_processes or mp.cpu_count())
    n_processes = max(1, n_processes)
    chunksize = max(1, min(8, len(pending_clusters) // max(n_processes * 4, 1) or 1))

    completed_at_start = len(all_cluster_names) - len(pending_clusters)
    success_count = sum(
        1
        for payload in existing_results.values()
        if _cluster_is_complete(
            payload,
            require_swiggum=run_config.include_swiggum_masses,
            model_names=run_config.model_names,
            mass_output_prefix=run_config.mass_output_prefix,
        )
    )
    partial_count = sum(
        1
        for payload in existing_results.values()
        if payload
        and not _cluster_is_complete(
            payload,
            require_swiggum=run_config.include_swiggum_masses,
            model_names=run_config.model_names,
            mass_output_prefix=run_config.mass_output_prefix,
        )
    )
    swiggum_successes = _count_swiggum_successes(
        existing_results,
        run_config.model_names,
        mass_output_prefix=run_config.mass_output_prefix,
    )

    _print_terminal_banner(
        output_root=output_root,
        run_config=run_config,
        total_clusters=len(all_cluster_names),
        pending_clusters=len(pending_clusters),
        completed_at_start=completed_at_start,
        n_processes=n_processes,
        chunksize=chunksize,
    )

    if not pending_clusters:
        _write_summary_csv(output_root, existing_results)
        print(f"Nothing to run. Existing summary is at {output_root / 'cluster_results.csv'}", flush=True)
        return output_root / "cluster_results.csv"

    ctx = mp.get_context("spawn")
    try:
        _print_stage("starting worker pool")
        with ctx.Pool(
            processes=n_processes,
            initializer=_init_worker,
            initargs=(
                grouped_data,
                cluster_metadata,
                str(paths.inputs.extinction_healpix_fits),
                str(paths.inputs.bayestar2019_h5) if paths.inputs.bayestar2019_h5 is not None else None,
                str(paths.inputs.decaps_h5) if paths.inputs.decaps_h5 is not None else None,
                str(output_root),
                run_config,
            ),
        ) as pool:
            _print_stage("worker pool ready; entering fit loop")
            with tqdm(
                total=len(all_cluster_names),
                initial=completed_at_start,
                desc="Chronos",
                unit="cluster",
                dynamic_ncols=True,
                file=sys.stdout,
                mininterval=0.5,
                leave=True,
                colour="cyan",
                bar_format=(
                    "{desc}: {percentage:3.0f}%|{bar:40}| "
                    "{n_fmt}/{total_fmt} clusters "
                    "[{elapsed} elapsed, {remaining} left, {rate_fmt}] {postfix}"
                ),
            ) as progress:
                progress.set_postfix_str(
                    _format_progress_postfix(
                        done=completed_at_start,
                        total=len(all_cluster_names),
                        success=success_count,
                        partial=partial_count,
                        swiggum_successes=swiggum_successes if run_config.include_swiggum_masses else None,
                    )
                )
                progress.refresh()
                for payload in pool.imap_unordered(_process_cluster, pending_clusters, chunksize=chunksize):
                    existing_results[payload["name"]] = payload
                    _atomic_write_json(_checkpoint_path(output_root, payload["name"]), payload)
                    _write_summary_csv(output_root, existing_results)

                    success_count = sum(
                        1
                        for result in existing_results.values()
                        if _cluster_is_complete(
                            result,
                            require_swiggum=run_config.include_swiggum_masses,
                            model_names=run_config.model_names,
                            mass_output_prefix=run_config.mass_output_prefix,
                        )
                    )
                    partial_count = sum(
                        1
                        for result in existing_results.values()
                        if result
                        and not _cluster_is_complete(
                            result,
                            require_swiggum=run_config.include_swiggum_masses,
                            model_names=run_config.model_names,
                            mass_output_prefix=run_config.mass_output_prefix,
                        )
                    )
                    swiggum_successes = _count_swiggum_successes(
                        existing_results,
                        run_config.model_names,
                        mass_output_prefix=run_config.mass_output_prefix,
                    )
                    progress.update(1)
                    progress.set_postfix_str(
                        _format_progress_postfix(
                            done=progress.n,
                            total=len(all_cluster_names),
                            success=success_count,
                            partial=partial_count,
                            swiggum_successes=swiggum_successes if run_config.include_swiggum_masses else None,
                        )
                    )
                    if run_config.print_cluster_updates:
                        _print_cluster_update(
                            payload,
                            done=progress.n,
                            total=len(all_cluster_names),
                            include_swiggum_masses=run_config.include_swiggum_masses,
                            mass_output_prefix=run_config.mass_output_prefix,
                        )
    except KeyboardInterrupt:
        _write_summary_csv(output_root, existing_results)
        raise

    print(f"Finished. Summary CSV: {output_root / 'cluster_results.csv'}", flush=True)
    return output_root / "cluster_results.csv"
