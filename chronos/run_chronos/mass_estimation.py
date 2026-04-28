from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
from typing import Any, Literal, Mapping

from astropy import units as u
from astropy.coordinates import SkyCoord
import imf
import matplotlib
import numpy as np
import pandas as pd
from scipy.integrate import quad

from chronos.run_chronos.pipeline import ChronosFitConfig, configure_cluster_fitter
from chronos.utils.ClusterMass import MassFitter
from mapper.sampling import make_cluster_compat


matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.ioff()


MassMethod = Literal["swiggum", "hunt"]

PAPER_G_MAG_COMPLETE_RANGE: tuple[float, float] = (12.0, 17.0)
SWIGGUM_TOTAL_MASS_GRID_MSUN = np.round(np.arange(10.0, 5000.0, 0.1), 1)
HUNT_G_MAG_RANGE: tuple[float, float] = (2.0, 21.0)
HUNT_G_BIN_WIDTH = 0.2
HUNT_BINARY_ANGULAR_RESOLUTION_ARCSEC = 0.4
HUNT_MIN_SELECTION_PROBABILITY = 1e-3
HUNT_REQUIRED_GAIA_COLUMNS = {
    "ra",
    "dec",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "pmra",
    "pmdec",
    "parallax",
    "pmra_error",
    "pmdec_error",
    "parallax_error",
    "astrometric_params_solved",
    "rybizki_fidelity_v1",
}


@dataclass(frozen=True)
class IntervalSummary:
    p16: float
    p50: float
    p84: float


@dataclass(frozen=True)
class ChronosAgeSummary:
    name: str
    age_myr: IntervalSummary
    av_mag: IntervalSummary
    status: str
    age_samples_myr: np.ndarray | None = None
    av_samples_mag: np.ndarray | None = None


@dataclass(frozen=True)
class ClusterMassEstimate:
    method: MassMethod
    status: str
    total_mass_msun: IntervalSummary | None
    n_draws_succeeded: int
    n_members_total: int
    n_members_used: int
    skip_reason: str | None
    details: dict[str, Any]

    def as_row(self, prefix: str) -> dict[str, Any]:
        row: dict[str, Any] = {
            f"{prefix}_status": self.status,
            f"{prefix}_n_draws_succeeded": self.n_draws_succeeded,
            f"{prefix}_n_members_total": self.n_members_total,
            f"{prefix}_n_members_used": self.n_members_used,
            f"{prefix}_skip_reason": self.skip_reason,
        }
        if self.total_mass_msun is None:
            row[f"{prefix}_16"] = np.nan
            row[f"{prefix}_50"] = np.nan
            row[f"{prefix}_84"] = np.nan
        else:
            row[f"{prefix}_16"] = self.total_mass_msun.p16
            row[f"{prefix}_50"] = self.total_mass_msun.p50
            row[f"{prefix}_84"] = self.total_mass_msun.p84
        for key, value in self.details.items():
            row[f"{prefix}_{key}"] = value
        return row


@dataclass(frozen=True)
class SwiggumFitResult:
    total_mass_msun: float
    model_complete_mass_msun: float
    complete_mass_abs_delta_msun: float
    mass_function_score: float
    observed_complete_masses_msun: np.ndarray | None = None
    model_complete_masses_msun: np.ndarray | None = None
    bin_edges_msun: np.ndarray | None = None
    complete_mass_range_msun: tuple[float, float] | None = None


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    return slug or "cluster"


def _clean_float(value: Any) -> float | None:
    try:
        candidate = float(value)
    except Exception:
        return None
    if math.isfinite(candidate):
        return candidate
    return None


def _normalize_source_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True)


def _quantile_interval(values: np.ndarray) -> IntervalSummary:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("Cannot summarize an empty sample.")
    q16, q50, q84 = np.quantile(arr, [0.16, 0.50, 0.84])
    return IntervalSummary(p16=float(q16), p50=float(q50), p84=float(q84))


def _distance_modulus(distance_pc: float) -> float:
    return float(5.0 * np.log10(distance_pc) - 5.0)


def resolve_age_summary_row(age_catalog: pd.DataFrame, cluster_name: str) -> pd.Series | None:
    exact = age_catalog.loc[age_catalog["name"].astype(str) == cluster_name]
    if not exact.empty:
        return exact.iloc[0]

    if "name_all" in age_catalog.columns:
        aliases = cluster_name.replace("_", " ")
        mask = age_catalog["name_all"].astype(str).str.contains(aliases, case=False, na=False)
        if mask.any():
            return age_catalog.loc[mask].iloc[0]
        mask = age_catalog["name_all"].astype(str).str.contains(cluster_name, case=False, na=False)
        if mask.any():
            return age_catalog.loc[mask].iloc[0]
    return None


def age_summary_from_row(row: pd.Series) -> ChronosAgeSummary:
    status = row.get("status")
    if status is None or (isinstance(status, float) and not np.isfinite(status)):
        finite_fields = [
            _clean_float(row.get("age_chronos_mode")),
            _clean_float(row.get("age_chronos_lo")),
            _clean_float(row.get("age_chronos_hi")),
            _clean_float(row.get("av_chronos_mode")),
            _clean_float(row.get("av_chronos_lo")),
            _clean_float(row.get("av_chronos_hi")),
        ]
        status = "success" if all(value is not None for value in finite_fields) else "unknown"
    return ChronosAgeSummary(
        name=str(row["name"]),
        age_myr=IntervalSummary(
            p16=float(row["age_chronos_lo"]),
            p50=float(row["age_chronos_mode"]),
            p84=float(row["age_chronos_hi"]),
        ),
        av_mag=IntervalSummary(
            p16=float(row["av_chronos_lo"]),
            p50=float(row["av_chronos_mode"]),
            p84=float(row["av_chronos_hi"]),
        ),
        status=str(status),
    )


def sample_age_av_draws(
    summary: ChronosAgeSummary,
    *,
    n_draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if summary.age_samples_myr is not None and summary.av_samples_mag is not None:
        age_samples = np.asarray(summary.age_samples_myr, dtype=float)
        av_samples = np.asarray(summary.av_samples_mag, dtype=float)
        if age_samples.size and age_samples.size == av_samples.size:
            indices = rng.integers(0, age_samples.size, size=int(n_draws))
            return age_samples[indices], av_samples[indices]

    age_draws = _draw_split_normal(
        center=summary.age_myr.p50,
        lo=summary.age_myr.p16,
        hi=summary.age_myr.p84,
        n_draws=n_draws,
        rng=rng,
        floor=1e-3,
    )
    av_draws = _draw_split_normal(
        center=summary.av_mag.p50,
        lo=summary.av_mag.p16,
        hi=summary.av_mag.p84,
        n_draws=n_draws,
        rng=rng,
        floor=0.0,
    )
    return age_draws, av_draws


def _draw_split_normal(
    *,
    center: float,
    lo: float,
    hi: float,
    n_draws: int,
    rng: np.random.Generator,
    floor: float,
) -> np.ndarray:
    sigma_lo = max(float(center - lo), 1e-6)
    sigma_hi = max(float(hi - center), 1e-6)
    if sigma_lo <= 1e-6 and sigma_hi <= 1e-6:
        return np.full(int(n_draws), max(float(center), floor), dtype=float)

    side_probability = sigma_lo / (sigma_lo + sigma_hi)
    draw_left = rng.random(int(n_draws)) < side_probability
    draws = np.empty(int(n_draws), dtype=float)
    draws[draw_left] = float(center) - np.abs(rng.normal(loc=0.0, scale=sigma_lo, size=int(np.sum(draw_left))))
    draws[~draw_left] = float(center) + np.abs(
        rng.normal(loc=0.0, scale=sigma_hi, size=int(np.sum(~draw_left)))
    )
    return np.clip(draws, floor, None)


def _member_mass_frame(
    fitter,
    df_group: pd.DataFrame,
    *,
    age_myr: float,
    av_mag: float,
) -> pd.DataFrame:
    _, masses, keep_mask = fitter.compute_fit_info(
        logAge=np.log10(float(age_myr) * 1e6),
        feh=0.0,
        A_V=float(av_mag),
        g_rp=fitter.use_grp,
        signed_distance=True,
    )
    base = df_group.loc[fitter.distance_handler.is_not_nan].copy().reset_index(drop=True)
    base["mass_msun"] = np.asarray(masses, dtype=float)
    base["fit_keep"] = np.asarray(keep_mask, dtype=bool)
    base = base.loc[
        base["fit_keep"]
        & np.isfinite(base["mass_msun"])
        & (pd.to_numeric(base["mass_msun"], errors="coerce") > 0.0)
    ].copy()
    base["mass_msun"] = pd.to_numeric(base["mass_msun"], errors="coerce")
    if "source_id" in base.columns:
        base["source_id"] = _normalize_source_id(base["source_id"])
    base.reset_index(drop=True, inplace=True)
    return base


def compute_complete_mass_range(
    fitter,
    *,
    age_myr: float,
    av_mag: float,
    distance_pc: float,
    apparent_g_range: tuple[float, float] = PAPER_G_MAG_COMPLETE_RANGE,
) -> tuple[tuple[float, float], tuple[float, float]]:
    log_age = float(np.log10(age_myr * 1e6))
    distance_modulus = _distance_modulus(distance_pc)
    abs_g_limits = tuple(sorted(float(g_mag - distance_modulus) for g_mag in apparent_g_range))

    iso_coords = fitter.isochrone_handler.model(logAge=log_age, feh=0.0, A_V=av_mag, g_rp=fitter.use_grp)
    iso_masses = fitter.isochrone_handler.compute_mass(
        iso_coords,
        logAge=log_age,
        feh=0.0,
        A_V=av_mag,
        g_rp=fitter.use_grp,
    )
    iso_abs_g = np.asarray(iso_coords[:, 1], dtype=float)
    iso_masses = np.asarray(iso_masses, dtype=float)
    complete_mask = (
        np.isfinite(iso_abs_g)
        & np.isfinite(iso_masses)
        & (iso_masses > 0.0)
        & (iso_abs_g >= abs_g_limits[0])
        & (iso_abs_g <= abs_g_limits[1])
    )
    if not np.any(complete_mask):
        raise ValueError("No isochrone points fell inside the Gaia-complete G-magnitude interval.")
    complete_masses = iso_masses[complete_mask]
    mass_limits = (float(np.min(complete_masses)), float(np.max(complete_masses)))
    return mass_limits, abs_g_limits


def _select_complete_masses(
    masses: np.ndarray,
    *,
    mass_limits: tuple[float, float],
) -> np.ndarray:
    arr = np.asarray(masses, dtype=float)
    valid = (
        np.isfinite(arr)
        & (arr > 0.0)
        & (arr >= float(mass_limits[0]))
        & (arr <= float(mass_limits[1]))
    )
    return arr[valid]


def _swiggum_histogram_edges(
    observed_complete_masses_msun: np.ndarray,
    *,
    complete_mass_range_msun: tuple[float, float],
) -> np.ndarray:
    lo = max(float(complete_mass_range_msun[0]), 1e-3)
    hi = max(float(complete_mass_range_msun[1]), lo * 1.01)
    n_complete = int(np.asarray(observed_complete_masses_msun, dtype=float).size)
    n_bins = int(np.clip(np.sqrt(max(n_complete, 1)) + 2, 5, 12))
    edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    edges[0] = lo
    edges[-1] = hi
    return edges


def _swiggum_mass_function_score(
    observed_complete_masses_msun: np.ndarray,
    model_complete_masses_msun: np.ndarray,
    *,
    bin_edges: np.ndarray,
) -> float:
    observed_counts, _ = np.histogram(observed_complete_masses_msun, bins=bin_edges)
    model_counts, _ = np.histogram(model_complete_masses_msun, bins=bin_edges)
    valid = (observed_counts > 0) | (model_counts > 0)
    if not np.any(valid):
        return math.inf
    variance = np.clip(observed_counts[valid] + model_counts[valid], 1.0, None).astype(float)
    residual = observed_counts[valid].astype(float) - model_counts[valid].astype(float)
    return float(np.sum((residual**2) / variance))


def _fit_swiggum_total_mass_once(
    observed_member_masses_msun: np.ndarray,
    *,
    complete_mass_range_msun: tuple[float, float],
    rng: np.random.Generator,
    total_mass_grid_msun: np.ndarray,
    n_imf_draws: int,
) -> SwiggumFitResult:
    observed_complete_masses = _select_complete_masses(
        observed_member_masses_msun,
        mass_limits=complete_mass_range_msun,
    )
    if observed_complete_masses.size == 0:
        raise ValueError("No observed masses fell inside the complete mass interval.")

    observed_complete_mass = float(np.sum(observed_complete_masses))
    bin_edges = _swiggum_histogram_edges(
        observed_complete_masses,
        complete_mass_range_msun=complete_mass_range_msun,
    )
    draw_count = min(int(n_imf_draws), int(total_mass_grid_msun.size))
    candidate_total_masses = rng.choice(
        total_mass_grid_msun,
        size=draw_count,
        replace=draw_count > total_mass_grid_msun.size,
    )

    best_result: SwiggumFitResult | None = None
    for total_mass in candidate_total_masses:
        model_masses = make_cluster_compat(
            float(total_mass),
            massfunc="kroupa",
            mmin=0.03,
            mmax=120.0,
        )
        model_complete_masses = _select_complete_masses(
            model_masses,
            mass_limits=complete_mass_range_msun,
        )
        model_complete_mass = float(np.sum(model_complete_masses))
        candidate = SwiggumFitResult(
            total_mass_msun=float(total_mass),
            model_complete_mass_msun=model_complete_mass,
            complete_mass_abs_delta_msun=float(abs(model_complete_mass - observed_complete_mass)),
            mass_function_score=_swiggum_mass_function_score(
                observed_complete_masses,
                model_complete_masses,
                bin_edges=bin_edges,
            ),
            observed_complete_masses_msun=np.asarray(observed_complete_masses, dtype=float),
            model_complete_masses_msun=np.asarray(model_complete_masses, dtype=float),
            bin_edges_msun=np.asarray(bin_edges, dtype=float),
            complete_mass_range_msun=tuple(float(value) for value in complete_mass_range_msun),
        )
        if best_result is None or (
            candidate.mass_function_score < best_result.mass_function_score
            or (
                math.isclose(candidate.mass_function_score, best_result.mass_function_score, rel_tol=0.0, abs_tol=1e-12)
                and candidate.complete_mass_abs_delta_msun < best_result.complete_mass_abs_delta_msun
            )
        ):
            best_result = candidate

    if best_result is None or not math.isfinite(best_result.total_mass_msun):
        raise RuntimeError("Failed to identify a Swiggum-model cluster mass.")
    return best_result


def _write_mass_draws(draw_records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(draw_records).to_csv(output_path, index=False)


def _save_mass_function_fit_plot(
    *,
    observed_member_masses_msun: np.ndarray,
    fit_result: SwiggumFitResult,
    complete_mass_range_msun: tuple[float, float],
    complete_abs_g_range: tuple[float, float],
    distance_pc: float,
    apparent_g_range: tuple[float, float],
    age_myr: float,
    av_mag: float,
    output_path: Path,
    title: str | None = None,
) -> None:
    all_observed = np.asarray(observed_member_masses_msun, dtype=float)
    all_observed = all_observed[np.isfinite(all_observed) & (all_observed > 0.0)]
    observed_complete = (
        np.asarray(fit_result.observed_complete_masses_msun, dtype=float)
        if fit_result.observed_complete_masses_msun is not None
        else np.asarray([], dtype=float)
    )
    model_complete = (
        np.asarray(fit_result.model_complete_masses_msun, dtype=float)
        if fit_result.model_complete_masses_msun is not None
        else np.asarray([], dtype=float)
    )
    observed_complete = observed_complete[np.isfinite(observed_complete) & (observed_complete > 0.0)]
    model_complete = model_complete[np.isfinite(model_complete) & (model_complete > 0.0)]

    mass_lo, mass_hi = tuple(float(value) for value in complete_mass_range_msun)
    plot_values = np.concatenate([all_observed, observed_complete, model_complete, np.array([mass_lo, mass_hi])])
    plot_values = plot_values[np.isfinite(plot_values) & (plot_values > 0.0)]
    if plot_values.size == 0:
        raise ValueError("No finite stellar masses available for mass-function diagnostic plot.")

    x_lo = 10 ** np.floor(np.log10(np.nanmin(plot_values)))
    x_hi = 10 ** np.ceil(np.log10(np.nanmax(plot_values)))
    if x_hi <= x_lo:
        x_hi = x_lo * 10.0

    bin_edges = (
        np.asarray(fit_result.bin_edges_msun, dtype=float)
        if fit_result.bin_edges_msun is not None
        else _swiggum_histogram_edges(observed_complete, complete_mass_range_msun=complete_mass_range_msun)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True, facecolor="white")
    if all_observed.size:
        all_bins = np.logspace(np.log10(x_lo), np.log10(x_hi), 24)
        axes[0].hist(all_observed, bins=all_bins, color="#4C72B0", alpha=0.72, edgecolor="white")
    axes[0].axvspan(mass_lo, mass_hi, color="#55A868", alpha=0.20, label="fit regime")
    axes[0].axvline(mass_lo, color="#2F6B3F", linestyle="--", linewidth=1.1)
    axes[0].axvline(mass_hi, color="#2F6B3F", linestyle="--", linewidth=1.1)
    axes[0].set_xscale("log")
    axes[0].set_xlim(x_lo, x_hi)
    axes[0].set_xlabel("Inferred member mass (Msun)")
    axes[0].set_ylabel("Observed members")
    axes[0].set_title("Observed Member Masses")
    axes[0].legend(frameon=False, loc="upper right")

    if observed_complete.size:
        axes[1].hist(
            observed_complete,
            bins=bin_edges,
            histtype="step",
            linewidth=2.1,
            color="#2F6BFF",
            label="observed complete",
        )
    if model_complete.size:
        axes[1].hist(
            model_complete,
            bins=bin_edges,
            histtype="stepfilled",
            alpha=0.28,
            color="#D96B27",
            label=f"Kroupa draw, M={fit_result.total_mass_msun:.0f} Msun",
        )
        axes[1].hist(
            model_complete,
            bins=bin_edges,
            histtype="step",
            linewidth=1.8,
            color="#D96B27",
        )
    axes[1].set_xscale("log")
    axes[1].set_xlim(max(mass_lo * 0.95, x_lo), min(mass_hi * 1.05, x_hi))
    axes[1].set_xlabel("Mass inside completeness interval (Msun)")
    axes[1].set_ylabel("Stars per bin")
    axes[1].set_title("Mass-Function Fit")
    axes[1].legend(frameon=False, loc="upper right")

    for axis in axes:
        axis.grid(True, which="both", color="0.90", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    subtitle = (
        f"distance={distance_pc:.0f} pc | G={apparent_g_range[0]:.1f}-{apparent_g_range[1]:.1f} "
        f"=> M_G={complete_abs_g_range[0]:.2f}-{complete_abs_g_range[1]:.2f} | "
        f"complete mass={mass_lo:.2f}-{mass_hi:.2f} Msun | "
        f"age={age_myr:.1f} Myr, A_V={av_mag:.2f}, score={fit_result.mass_function_score:.2f}"
    )
    fig.suptitle((title or "Mass-Function Fit Diagnostic") + "\n" + subtitle, fontsize=11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _member_selection_cache_path(cache_dir: Path, cluster_name: str) -> Path:
    return cache_dir / f"{_slugify(cluster_name)}.selection.csv"


def _load_gaia_neighborhood_cache(cluster_name: str, cache_dir: Path | None) -> pd.DataFrame:
    if cache_dir is None:
        raise FileNotFoundError("No Gaia-neighborhood cache directory configured.")
    if not cache_dir.exists():
        raise FileNotFoundError(f"Gaia-neighborhood cache directory does not exist: {cache_dir}")

    candidates = [
        cache_dir / f"{cluster_name}.csv",
        cache_dir / f"{_slugify(cluster_name)}.csv",
        cache_dir / f"{cluster_name}.parquet",
        cache_dir / f"{_slugify(cluster_name)}.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            if candidate.suffix == ".parquet":
                data = pd.read_parquet(candidate)
            else:
                data = pd.read_csv(candidate)
            return _normalize_gaia_cache_columns(data)
    raise FileNotFoundError(f"Missing Gaia-neighborhood cache file for cluster {cluster_name!r}.")


def _normalize_gaia_cache_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    aliases = {
        "source_id": ("source_id",),
        "ra": ("ra",),
        "dec": ("dec",),
        "phot_g_mean_mag": ("phot_g_mean_mag", "g_mag", "G", "g"),
        "phot_bp_mean_mag": ("phot_bp_mean_mag", "bp_mag", "BP", "bp"),
        "phot_rp_mean_mag": ("phot_rp_mean_mag", "rp_mag", "RP", "rp"),
        "pmra": ("pmra",),
        "pmdec": ("pmdec",),
        "parallax": ("parallax",),
        "pmra_error": ("pmra_error",),
        "pmdec_error": ("pmdec_error",),
        "parallax_error": ("parallax_error",),
        "astrometric_params_solved": ("astrometric_params_solved",),
        "rybizki_fidelity_v1": ("rybizki_fidelity_v1", "fidelity_v1"),
    }
    for target, options in aliases.items():
        if target in renamed.columns:
            continue
        for option in options:
            if option in renamed.columns:
                renamed = renamed.rename(columns={option: target})
                break

    missing = sorted(HUNT_REQUIRED_GAIA_COLUMNS - set(renamed.columns))
    if missing:
        raise ValueError(f"Gaia-neighborhood cache is missing required columns: {missing}")
    if "source_id" in renamed.columns:
        renamed["source_id"] = _normalize_source_id(renamed["source_id"])
    return renamed


def _gaia_subsample_mask(df: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_numeric(df["astrometric_params_solved"], errors="coerce").to_numpy(dtype=float) >= 31.0
    ) & np.isfinite(pd.to_numeric(df["phot_bp_mean_mag"], errors="coerce").to_numpy(dtype=float)) & np.isfinite(
        pd.to_numeric(df["phot_rp_mean_mag"], errors="coerce").to_numpy(dtype=float)
    ) & (
        pd.to_numeric(df["rybizki_fidelity_v1"], errors="coerce").to_numpy(dtype=float) > 0.5
    )


def _merged_bin_edges(values: np.ndarray, *, width: float, min_count: int) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.asarray([0.0, width], dtype=float)

    edges = np.arange(float(np.min(finite)), float(np.max(finite)) + width, width, dtype=float)
    if edges.size < 2:
        edges = np.asarray([float(np.min(finite)), float(np.min(finite)) + width], dtype=float)

    counts, _ = np.histogram(finite, bins=edges)
    merged = [float(edges[0])]
    running = 0
    for index, count in enumerate(counts, start=1):
        running += int(count)
        if running >= int(min_count):
            merged.append(float(edges[index]))
            running = 0
    if merged[-1] < float(edges[-1]):
        merged.append(float(edges[-1]))
    if len(merged) < 2:
        merged = [float(edges[0]), float(edges[-1])]
    return np.asarray(merged, dtype=float)


def _histogram_probability(
    reference_values: np.ndarray,
    success_mask: np.ndarray,
    query_values: np.ndarray,
    *,
    width: float,
    min_count: int,
) -> np.ndarray:
    edges = _merged_bin_edges(reference_values, width=width, min_count=min_count)
    total_counts, _ = np.histogram(reference_values, bins=edges)
    success_counts, _ = np.histogram(reference_values[success_mask], bins=edges)
    probabilities = success_counts / np.clip(total_counts, 1, None)
    bin_index = np.clip(np.digitize(query_values, edges) - 1, 0, len(probabilities) - 1)
    return np.clip(probabilities[bin_index], HUNT_MIN_SELECTION_PROBABILITY, 1.0)


def _histogram_mean_probability(
    reference_values: np.ndarray,
    probabilities: np.ndarray,
    query_values: np.ndarray,
    *,
    width: float,
    min_count: int,
) -> np.ndarray:
    edges = _merged_bin_edges(reference_values, width=width, min_count=min_count)
    total_counts, _ = np.histogram(reference_values, bins=edges)
    weighted_counts, _ = np.histogram(reference_values, bins=edges, weights=probabilities)
    mean_probability = weighted_counts / np.clip(total_counts, 1, None)
    bin_index = np.clip(np.digitize(query_values, edges) - 1, 0, len(mean_probability) - 1)
    return np.clip(mean_probability[bin_index], HUNT_MIN_SELECTION_PROBABILITY, 1.0)


def _gaia_selection_probabilities(members: pd.DataFrame) -> np.ndarray:
    try:
        from gaiaunlimited.selectionfunctions import DR3SelectionFunctionTCG
    except Exception as exc:  # pragma: no cover - exercised in failure-mode tests
        raise RuntimeError("gaiaunlimited is required for the Hunt mass method.") from exc

    coords = SkyCoord(
        ra=pd.to_numeric(members["ra"], errors="coerce").to_numpy(dtype=float) * u.deg,
        dec=pd.to_numeric(members["dec"], errors="coerce").to_numpy(dtype=float) * u.deg,
        frame="icrs",
    )
    g_mag = pd.to_numeric(members["phot_g_mean_mag"], errors="coerce").to_numpy(dtype=float)
    selection_function = DR3SelectionFunctionTCG("multi")
    probabilities = np.asarray(selection_function.query(coords, g_mag), dtype=float)
    return np.clip(probabilities, HUNT_MIN_SELECTION_PROBABILITY, 1.0)


def _subsample_selection_probabilities(
    members: pd.DataFrame,
    gaia_neighborhood: pd.DataFrame,
) -> np.ndarray:
    reference_g = pd.to_numeric(gaia_neighborhood["phot_g_mean_mag"], errors="coerce").to_numpy(dtype=float)
    query_g = pd.to_numeric(members["phot_g_mean_mag"], errors="coerce").to_numpy(dtype=float)
    return _histogram_probability(
        reference_values=reference_g,
        success_mask=_gaia_subsample_mask(gaia_neighborhood),
        query_values=query_g,
        width=HUNT_G_BIN_WIDTH,
        min_count=10,
    )


def _fit_member_ellipsoid(member_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    astrometry = member_frame[["pmra", "pmdec", "parallax"]].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=float
    )
    center = np.nanmedian(astrometry, axis=0)
    covariance = np.cov(astrometry.T)
    covariance = np.asarray(covariance, dtype=float)
    covariance += np.eye(3, dtype=float) * 1e-6
    inverse = np.linalg.inv(covariance)
    delta = astrometry - center
    d2 = np.einsum("...i,ij,...j->...", delta, inverse, delta)
    threshold = float(np.quantile(d2[np.isfinite(d2)], 0.95))
    return center, inverse, threshold


def _nearest_error_rows(
    g_values: np.ndarray,
    gaia_neighborhood: pd.DataFrame,
) -> pd.DataFrame:
    reference = gaia_neighborhood.loc[
        np.isfinite(pd.to_numeric(gaia_neighborhood["phot_g_mean_mag"], errors="coerce"))
        & np.isfinite(pd.to_numeric(gaia_neighborhood["pmra_error"], errors="coerce"))
        & np.isfinite(pd.to_numeric(gaia_neighborhood["pmdec_error"], errors="coerce"))
        & np.isfinite(pd.to_numeric(gaia_neighborhood["parallax_error"], errors="coerce"))
    ].copy()
    if reference.empty:
        raise ValueError("Gaia-neighborhood cache does not include finite astrometric errors.")

    reference.sort_values("phot_g_mean_mag", inplace=True)
    reference_g = pd.to_numeric(reference["phot_g_mean_mag"], errors="coerce").to_numpy(dtype=float)
    indices = np.searchsorted(reference_g, g_values, side="left")
    indices = np.clip(indices, 0, len(reference_g) - 1)
    lower = np.clip(indices - 1, 0, len(reference_g) - 1)
    upper = indices
    choose_lower = np.abs(reference_g[lower] - g_values) <= np.abs(reference_g[upper] - g_values)
    final_index = np.where(choose_lower, lower, upper)
    return reference.iloc[final_index].reset_index(drop=True)


def _algorithm_selection_probabilities(
    members: pd.DataFrame,
    gaia_neighborhood: pd.DataFrame,
    *,
    rng: np.random.Generator,
    n_simulated: int,
    n_perturbations: int,
) -> np.ndarray:
    center, inverse, threshold = _fit_member_ellipsoid(members)
    sim_g = rng.uniform(HUNT_G_MAG_RANGE[0], HUNT_G_MAG_RANGE[1], size=int(n_simulated))
    matched_errors = _nearest_error_rows(sim_g, gaia_neighborhood)
    pmra_error = pd.to_numeric(matched_errors["pmra_error"], errors="coerce").to_numpy(dtype=float)
    pmdec_error = pd.to_numeric(matched_errors["pmdec_error"], errors="coerce").to_numpy(dtype=float)
    parallax_error = pd.to_numeric(matched_errors["parallax_error"], errors="coerce").to_numpy(dtype=float)

    member_astrometry = members[["pmra", "pmdec", "parallax"]].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=float
    )
    draw_indices = rng.integers(0, member_astrometry.shape[0], size=int(n_simulated))
    true_astrometry = member_astrometry[draw_indices].copy()

    inside_counts = np.zeros(int(n_simulated), dtype=float)
    error_cube = np.column_stack([pmra_error, pmdec_error, parallax_error])
    for _ in range(int(n_perturbations)):
        noisy = true_astrometry + rng.normal(loc=0.0, scale=error_cube)
        delta = noisy - center
        d2 = np.einsum("...i,ij,...j->...", delta, inverse, delta)
        inside_counts += d2 <= threshold
    simulated_probabilities = inside_counts / float(n_perturbations)

    query_g = pd.to_numeric(members["phot_g_mean_mag"], errors="coerce").to_numpy(dtype=float)
    return _histogram_mean_probability(
        reference_values=sim_g,
        probabilities=simulated_probabilities,
        query_values=query_g,
        width=HUNT_G_BIN_WIDTH,
        min_count=10,
    )


def _binary_population_parameters(primary_mass: float) -> tuple[float, float, float, float, float]:
    if primary_mass < 0.8:
        return 0.26, 0.33, 0.4, 5.0, 2.3
    if primary_mass < 1.5:
        return 0.41, 0.50, 0.3, 4.8, 2.3
    if primary_mass < 5.0:
        return 0.50, 0.75, -0.5, 4.0, 2.0
    return 0.70, 1.30, -0.1, 3.5, 2.0


def _sample_mass_ratio(rng: np.random.Generator, gamma: float) -> float:
    qmin = 0.1
    qmax = 1.0
    if abs(gamma + 1.0) < 1e-8:
        u_value = rng.random()
        return float(qmin * (qmax / qmin) ** u_value)
    exponent = gamma + 1.0
    u_value = rng.random()
    return float((u_value * (qmax**exponent - qmin**exponent) + qmin**exponent) ** (1.0 / exponent))


def _binary_correction_factors(
    masses: np.ndarray,
    *,
    distance_pc: float,
    rng: np.random.Generator,
) -> np.ndarray:
    factors = np.ones_like(masses, dtype=float)
    for index, primary_mass in enumerate(np.asarray(masses, dtype=float)):
        multiplicity_fraction, companion_frequency, gamma, log_period_mean, log_period_sigma = (
            _binary_population_parameters(float(primary_mass))
        )
        if rng.random() >= multiplicity_fraction:
            continue

        companions = 1 + rng.poisson(max(companion_frequency / max(multiplicity_fraction, 1e-6) - 1.0, 0.0))
        hidden_mass = 0.0
        for _ in range(int(companions)):
            q_value = _sample_mass_ratio(rng, gamma)
            companion_mass = float(primary_mass) * q_value
            period_days = 10 ** rng.normal(log_period_mean, log_period_sigma)
            eccentricity = np.clip(rng.beta(1.5, 4.0), 0.0, 0.95)
            period_years = period_days / 365.25
            semi_major_axis_au = ((primary_mass + companion_mass) * period_years**2) ** (1.0 / 3.0)
            mean_separation_au = semi_major_axis_au * (1.0 + 0.5 * eccentricity**2)
            angular_separation_arcsec = mean_separation_au / max(distance_pc, 1e-6)
            if angular_separation_arcsec < HUNT_BINARY_ANGULAR_RESOLUTION_ARCSEC:
                hidden_mass += companion_mass
        factors[index] = 1.0 + hidden_mass / max(float(primary_mass), 1e-6)
    return factors


def _fit_kroupa_amplitude(
    masses: np.ndarray,
    *,
    weights: np.ndarray,
) -> float:
    masses = np.asarray(masses, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(masses) & (masses > 0.0) & np.isfinite(weights) & (weights > 0.0)
    masses = masses[valid]
    weights = weights[valid]
    if masses.size < 3:
        raise ValueError("Need at least three weighted stellar masses to fit the Hunt IMF normalization.")

    n_bins = int(np.clip(np.sqrt(masses.size), 5, 12))
    bin_edges = np.logspace(np.log10(max(0.03, np.min(masses))), np.log10(np.max(masses)), n_bins + 1)
    counts, _ = np.histogram(masses, bins=bin_edges, weights=weights)
    widths = np.diff(bin_edges)
    observed_density = counts / widths

    mass_function = imf.get_massfunc("kroupa", mmin=0.03, mmax=float(np.max(masses)))
    model_density = []
    for lo, hi, width in zip(bin_edges[:-1], bin_edges[1:], widths):
        integral = quad(lambda mass: float(mass_function.distr.pdf(mass)), float(lo), float(hi))[0]
        model_density.append(float(integral / width))
    model_density_arr = np.asarray(model_density, dtype=float)
    amplitude = float(np.sum(observed_density * model_density_arr) / np.sum(model_density_arr**2))
    return max(amplitude, 0.0)


def _integrated_kroupa_mass(amplitude: float, *, max_mass: float) -> float:
    mass_function = imf.get_massfunc("kroupa", mmin=0.03, mmax=float(max_mass))
    total_mass = quad(
        lambda mass: float(mass) * float(mass_function.distr.pdf(mass)),
        float(mass_function.mmin),
        float(mass_function.mmax),
    )[0]
    return float(amplitude * total_mass)


def _load_or_build_hunt_selection_table(
    cluster_name: str,
    members: pd.DataFrame,
    *,
    gaia_neighborhood_cache_dir: Path | None,
    selection_cache_dir: Path | None,
    rng: np.random.Generator,
    n_algorithm_simulated_stars: int,
    n_algorithm_perturbations: int,
) -> pd.DataFrame:
    if "source_id" in members.columns:
        members = members.copy()
        members["source_id"] = _normalize_source_id(members["source_id"])
    if selection_cache_dir is not None:
        selection_cache_dir.mkdir(parents=True, exist_ok=True)
        selection_cache_path = _member_selection_cache_path(selection_cache_dir, cluster_name)
        if selection_cache_path.exists():
            cached = pd.read_csv(selection_cache_path)
            if set(cached.columns) >= {"source_id", "s_gaia", "s_subsample", "s_algorithm"}:
                merged = members.merge(cached, on="source_id", how="left")
                if merged[["s_gaia", "s_subsample", "s_algorithm"]].notna().all().all():
                    return merged

    gaia_neighborhood = _load_gaia_neighborhood_cache(cluster_name, gaia_neighborhood_cache_dir)
    selection_table = members.copy()
    selection_table["s_gaia"] = _gaia_selection_probabilities(selection_table)
    selection_table["s_subsample"] = _subsample_selection_probabilities(selection_table, gaia_neighborhood)
    selection_table["s_algorithm"] = _algorithm_selection_probabilities(
        selection_table,
        gaia_neighborhood,
        rng=rng,
        n_simulated=n_algorithm_simulated_stars,
        n_perturbations=n_algorithm_perturbations,
    )

    if selection_cache_dir is not None:
        cache_columns = ["source_id", "s_gaia", "s_subsample", "s_algorithm"]
        selection_table[cache_columns].to_csv(
            _member_selection_cache_path(selection_cache_dir, cluster_name),
            index=False,
        )
    return selection_table


def estimate_cluster_mass(
    *,
    cluster_name: str,
    df_group: pd.DataFrame,
    cluster_row: pd.Series,
    age_summary: ChronosAgeSummary,
    mass_method: MassMethod,
    models: str = "parsec",
    isochrone_dirs: Mapping[str, str] | None = None,
    rng_seed: int,
    n_draws: int = 100,
    swiggum_n_imf_draws: int = 1000,
    swiggum_total_mass_grid_msun: np.ndarray = SWIGGUM_TOTAL_MASS_GRID_MSUN,
    draws_output_path: Path | None = None,
    diagnostic_plot_path: Path | None = None,
    diagnostic_title: str | None = None,
    gaia_neighborhood_cache_dir: Path | None = None,
    selection_cache_dir: Path | None = None,
    hunt_algorithm_simulated_stars: int = 100_000,
    hunt_algorithm_perturbations: int = 10,
) -> ClusterMassEstimate:
    if str(age_summary.status).strip().lower() != "success":
        return ClusterMassEstimate(
            method=mass_method,
            status="age_summary_unavailable",
            total_mass_msun=None,
            n_draws_succeeded=0,
            n_members_total=int(len(df_group)),
            n_members_used=0,
            skip_reason=str(age_summary.status),
            details={},
        )
    if df_group.empty:
        return ClusterMassEstimate(
            method=mass_method,
            status="missing_membership_catalog",
            total_mass_msun=None,
            n_draws_succeeded=0,
            n_members_total=0,
            n_members_used=0,
            skip_reason="missing_membership_catalog",
            details={},
        )

    rng = np.random.default_rng(int(rng_seed))
    distance_pc = _clean_float(cluster_row.get("distance_50"))
    if distance_pc is None or distance_pc <= 0.0:
        distance_pc = _clean_float(pd.to_numeric(df_group.get("distance_50"), errors="coerce").median())
    if distance_pc is None or distance_pc <= 0.0:
        return ClusterMassEstimate(
            method=mass_method,
            status="missing_distance",
            total_mass_msun=None,
            n_draws_succeeded=0,
            n_members_total=int(len(df_group)),
            n_members_used=0,
            skip_reason="missing_distance",
            details={},
        )

    fitter = configure_cluster_fitter(
        df_group.reset_index(drop=True),
        ChronosFitConfig(models=str(models), isochrone_dirs=isochrone_dirs),
    )
    age_draws, av_draws = sample_age_av_draws(age_summary, n_draws=n_draws, rng=rng)
    total_masses: list[float] = []
    members_used: list[int] = []
    completeness_lo: list[float] = []
    completeness_hi: list[float] = []
    draw_records: list[dict[str, Any]] = []
    selection_table: pd.DataFrame | None = None
    selection_members = df_group.loc[fitter.distance_handler.is_not_nan].copy().reset_index(drop=True)
    if "source_id" in selection_members.columns:
        selection_members["source_id"] = _normalize_source_id(selection_members["source_id"])

    try:
        for index, (age_myr, av_mag) in enumerate(zip(age_draws, av_draws, strict=False)):
            member_frame = _member_mass_frame(fitter, df_group, age_myr=float(age_myr), av_mag=float(av_mag))
            if member_frame.empty:
                continue

            if mass_method == "swiggum":
                complete_mass_range_msun, _complete_abs_g = compute_complete_mass_range(
                    fitter,
                    age_myr=float(age_myr),
                    av_mag=float(av_mag),
                    distance_pc=float(distance_pc),
                )
                fit_result = _fit_swiggum_total_mass_once(
                    member_frame["mass_msun"].to_numpy(dtype=float),
                    complete_mass_range_msun=complete_mass_range_msun,
                    rng=np.random.default_rng(int(rng_seed) + 10_000 + index),
                    total_mass_grid_msun=swiggum_total_mass_grid_msun,
                    n_imf_draws=swiggum_n_imf_draws,
                )
                total_masses.append(float(fit_result.total_mass_msun))
                members_used.append(int(len(member_frame)))
                completeness_lo.append(float(complete_mass_range_msun[0]))
                completeness_hi.append(float(complete_mass_range_msun[1]))
                draw_records.append(
                    {
                        "draw_index": int(index),
                        "age_myr": float(age_myr),
                        "av_mag": float(av_mag),
                        "total_mass_msun": float(fit_result.total_mass_msun),
                        "model_complete_mass_msun": float(fit_result.model_complete_mass_msun),
                        "complete_mass_abs_delta_msun": float(fit_result.complete_mass_abs_delta_msun),
                        "mass_function_score": float(fit_result.mass_function_score),
                        "n_members_used": int(len(member_frame)),
                        "n_observed_complete": int(
                            len(fit_result.observed_complete_masses_msun)
                            if fit_result.observed_complete_masses_msun is not None
                            else 0
                        ),
                        "n_model_complete": int(
                            len(fit_result.model_complete_masses_msun)
                            if fit_result.model_complete_masses_msun is not None
                            else 0
                        ),
                        "complete_mass_min_msun": float(complete_mass_range_msun[0]),
                        "complete_mass_max_msun": float(complete_mass_range_msun[1]),
                        "complete_abs_g_min": float(_complete_abs_g[0]),
                        "complete_abs_g_max": float(_complete_abs_g[1]),
                        "distance_pc": float(distance_pc),
                    }
                )
                continue

            if selection_table is None:
                selection_table = _load_or_build_hunt_selection_table(
                    cluster_name,
                    selection_members,
                    gaia_neighborhood_cache_dir=gaia_neighborhood_cache_dir,
                    selection_cache_dir=selection_cache_dir,
                    rng=np.random.default_rng(int(rng_seed) + 20_000),
                    n_algorithm_simulated_stars=hunt_algorithm_simulated_stars,
                    n_algorithm_perturbations=hunt_algorithm_perturbations,
                )

            member_frame = member_frame.merge(
                selection_table[["source_id", "s_gaia", "s_subsample", "s_algorithm"]],
                on="source_id",
                how="left",
            )
            if member_frame[["s_gaia", "s_subsample", "s_algorithm"]].isna().any().any():
                raise ValueError("Failed to align Hunt selection probabilities to member masses.")

            selection_weight = 1.0 / np.clip(
                member_frame[["s_gaia", "s_subsample", "s_algorithm"]].prod(axis=1).to_numpy(dtype=float),
                HUNT_MIN_SELECTION_PROBABILITY,
                None,
            )
            binary_factor = _binary_correction_factors(
                member_frame["mass_msun"].to_numpy(dtype=float),
                distance_pc=float(distance_pc),
                rng=np.random.default_rng(int(rng_seed) + 30_000 + index),
            )
            corrected_weights = selection_weight * binary_factor
            amplitude = _fit_kroupa_amplitude(
                member_frame["mass_msun"].to_numpy(dtype=float),
                weights=corrected_weights,
            )
            total_mass = _integrated_kroupa_mass(
                amplitude,
                max_mass=float(np.max(member_frame["mass_msun"].to_numpy(dtype=float))),
            )
            total_masses.append(float(total_mass))
            members_used.append(int(len(member_frame)))
    except FileNotFoundError:
        return ClusterMassEstimate(
            method=mass_method,
            status="missing_gaia_neighborhood_cache",
            total_mass_msun=None,
            n_draws_succeeded=0,
            n_members_total=int(len(df_group)),
            n_members_used=0,
            skip_reason="missing_gaia_neighborhood_cache",
            details={},
        )
    except RuntimeError as exc:
        if "gaiaunlimited" in str(exc).lower():
            return ClusterMassEstimate(
                method=mass_method,
                status="missing_gaiaunlimited",
                total_mass_msun=None,
                n_draws_succeeded=0,
                n_members_total=int(len(df_group)),
                n_members_used=0,
                skip_reason="missing_gaiaunlimited",
                details={},
            )
        return ClusterMassEstimate(
            method=mass_method,
            status=f"error: {exc}",
            total_mass_msun=None,
            n_draws_succeeded=0,
            n_members_total=int(len(df_group)),
            n_members_used=0,
            skip_reason=str(exc),
            details={},
        )
    except Exception as exc:
        return ClusterMassEstimate(
            method=mass_method,
            status=f"error: {exc}",
            total_mass_msun=None,
            n_draws_succeeded=0,
            n_members_total=int(len(df_group)),
            n_members_used=0,
            skip_reason=str(exc),
            details={},
        )

    if not total_masses:
        return ClusterMassEstimate(
            method=mass_method,
            status="no_usable_members",
            total_mass_msun=None,
            n_draws_succeeded=0,
            n_members_total=int(len(df_group)),
            n_members_used=0,
            skip_reason="no_usable_members",
            details={},
        )

    details: dict[str, Any] = {}
    if mass_method == "swiggum" and completeness_lo and completeness_hi:
        details["complete_mass_min_msun_p50"] = float(np.median(completeness_lo))
        details["complete_mass_max_msun_p50"] = float(np.median(completeness_hi))
        if draws_output_path is not None and draw_records:
            _write_mass_draws(draw_records, Path(draws_output_path))
            details["draws_csv"] = str(Path(draws_output_path))
        if diagnostic_plot_path is not None and draw_records:
            try:
                median_mass = float(np.median(total_masses))
                representative = min(
                    draw_records,
                    key=lambda record: abs(float(record["total_mass_msun"]) - median_mass),
                )
                representative_index = int(representative["draw_index"])
                representative_age = float(representative["age_myr"])
                representative_av = float(representative["av_mag"])
                member_frame = _member_mass_frame(
                    fitter,
                    df_group,
                    age_myr=representative_age,
                    av_mag=representative_av,
                )
                complete_mass_range_msun, complete_abs_g = compute_complete_mass_range(
                    fitter,
                    age_myr=representative_age,
                    av_mag=representative_av,
                    distance_pc=float(distance_pc),
                )
                fit_result = _fit_swiggum_total_mass_once(
                    member_frame["mass_msun"].to_numpy(dtype=float),
                    complete_mass_range_msun=complete_mass_range_msun,
                    rng=np.random.default_rng(int(rng_seed) + 10_000 + representative_index),
                    total_mass_grid_msun=swiggum_total_mass_grid_msun,
                    n_imf_draws=swiggum_n_imf_draws,
                )
                _save_mass_function_fit_plot(
                    observed_member_masses_msun=member_frame["mass_msun"].to_numpy(dtype=float),
                    fit_result=fit_result,
                    complete_mass_range_msun=complete_mass_range_msun,
                    complete_abs_g_range=complete_abs_g,
                    distance_pc=float(distance_pc),
                    apparent_g_range=PAPER_G_MAG_COMPLETE_RANGE,
                    age_myr=representative_age,
                    av_mag=representative_av,
                    output_path=Path(diagnostic_plot_path),
                    title=diagnostic_title,
                )
                details["diagnostic_plot"] = str(Path(diagnostic_plot_path))
            except Exception as exc:
                details["diagnostic_plot_error"] = str(exc)

    return ClusterMassEstimate(
        method=mass_method,
        status="success",
        total_mass_msun=_quantile_interval(np.asarray(total_masses, dtype=float)),
        n_draws_succeeded=int(len(total_masses)),
        n_members_total=int(len(df_group)),
        n_members_used=int(round(float(np.median(members_used)))),
        skip_reason=None,
        details=details,
    )


def estimate_legacy_imf_masses(
    masses: np.ndarray,
    *,
    binary_mass_scale: float,
) -> dict[str, float | None]:
    masses = np.asarray(masses, dtype=float)
    masses = masses[np.isfinite(masses) & (masses > 0.0)]
    observed_mass = float(np.sum(masses)) if masses.size else None
    imf_corrected = None
    binary_corrected = None

    def _imf_fraction_fallback(observed_masses: np.ndarray) -> float | None:
        min_mass = float(np.nanmin(observed_masses))
        min_mass = max(min_mass, 0.05)
        mfc = imf.get_massfunc("kroupa", mmin=0.03, mmax=120.0)
        total_mass = quad(
            lambda mass: float(mass) * float(mfc.distr.pdf(mass)),
            float(mfc.mmin),
            float(mfc.mmax),
        )[0]
        observed_fraction_mass = quad(
            lambda mass: float(mass) * float(mfc.distr.pdf(mass)),
            min_mass,
            float(mfc.mmax),
        )[0]
        if not np.isfinite(total_mass) or not np.isfinite(observed_fraction_mass) or observed_fraction_mass <= 0.0:
            return None
        return float(total_mass / observed_fraction_mass)

    if masses.size >= 2:
        try:
            mass_fitter = MassFitter(observed_masses=masses, n_draws=100)
            imf_corrected = _clean_float(mass_fitter.fit())
        except Exception:
            imf_corrected = None
    if imf_corrected is None and observed_mass is not None:
        try:
            correction_factor = _imf_fraction_fallback(masses)
        except Exception:
            correction_factor = None
        if correction_factor is not None:
            imf_corrected = float(observed_mass * correction_factor)
    if imf_corrected is not None:
        if observed_mass is not None:
            imf_corrected = max(float(imf_corrected), float(observed_mass))
        binary_corrected = float(imf_corrected * float(binary_mass_scale))
    return {
        "mass_members_observed": observed_mass,
        "mass_cluster_imf_corrected": imf_corrected,
        "mass_cluster_imf_binary_corrected": binary_corrected,
    }


def write_markdown_summary(
    summary: dict[str, Any],
    *,
    output_path: Path,
) -> None:
    lines = [
        "# Chronos Mass-Method Summary",
        "",
        f"- Output rows: `{summary['n_rows']}`",
        f"- Successful `swiggum` rows: `{summary['n_swiggum_success']}`",
        f"- Successful `hunt` rows: `{summary['n_hunt_success']}`",
        f"- Missing-membership rows: `{summary['n_missing_membership']}`",
    ]
    if "swiggum_median_abs_delta" in summary:
        lines.append(f"- Median |Swiggum - paper| mass offset: `{summary['swiggum_median_abs_delta']:.1f} Msun`")
    if "hunt_median_abs_delta" in summary:
        lines.append(f"- Median |Hunt - paper| mass offset: `{summary['hunt_median_abs_delta']:.1f} Msun`")
    if summary.get("skipped_clusters"):
        lines.extend(["", "## Skipped clusters", ""])
        for cluster_name in summary["skipped_clusters"]:
            lines.append(f"- `{cluster_name}`")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "ChronosAgeSummary",
    "ClusterMassEstimate",
    "IntervalSummary",
    "MassMethod",
    "PAPER_G_MAG_COMPLETE_RANGE",
    "SWIGGUM_TOTAL_MASS_GRID_MSUN",
    "age_summary_from_row",
    "compute_complete_mass_range",
    "estimate_cluster_mass",
    "estimate_legacy_imf_masses",
    "resolve_age_summary_row",
    "sample_age_av_draws",
    "write_json",
    "write_markdown_summary",
]
