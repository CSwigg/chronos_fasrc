from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import arviz as az
import numpy as np
import pandas as pd


def mode_reals(array: np.ndarray, bins: int = 100) -> float:
    """Approximate the mode of a real-valued sample using a histogram."""
    counts, bin_edges = np.histogram(array, bins=bins)
    bins_left_edges = bin_edges[:-1]
    return float(bins_left_edges[np.argmax(counts)])


@dataclass(frozen=True)
class ChronosFitConfig:
    """Production configuration for the vendored Chronos age fitter."""

    max_input_age_myr: float = 200.0
    use_grp: bool = False
    models: str = "parsec"
    isochrone_dirs: Mapping[str, str] | None = None
    abs_gmag_column: str = "g_abs_mag"
    color_bprp_column: str = "bp-rp"
    color_grp_column: str = "g-rp"
    age_range_myr: tuple[float, float] = (1.0, 500.0)
    feh_range: tuple[float, float] = (-1e-5, 1e-5)
    av_range: tuple[float, float] = (0.0, np.inf)
    skewness_range: tuple[float, float] = (0.5, 0.99)
    scale_range: tuple[float, float] = (0.001, 0.1)
    fit_range: tuple[float, float] = (-np.inf, 10.0)
    nwalkers: int = 40
    nsteps: int = 400
    burnin: int = 100
    posterior_plot_hdi_prob: float = 0.64
    summary_hdi_prob: float = 0.68

    def chronos_kwargs(self, data: pd.DataFrame) -> dict[str, object]:
        return {
            "data": data,
            "use_grp": self.use_grp,
            "models": self.models,
            "isochrone_dirs": self.isochrone_dirs,
            "abs_Gmag_name": self.abs_gmag_column,
            "color_bprp_name": self.color_bprp_column,
            "color_grp_name": self.color_grp_column,
        }

    def bayes_bounds(self) -> dict[str, tuple[float, float]]:
        age_lo, age_hi = self.age_range_myr
        return {
            "logAge_range": (np.log10(age_lo * 1e6), np.log10(age_hi * 1e6)),
            "feh_range": self.feh_range,
            "av_range": self.av_range,
            "skewness_range": self.skewness_range,
            "scale_range": self.scale_range,
        }

    def fitting_kwargs(self) -> dict[str, object]:
        return {
            "fit_range": self.fit_range,
            "do_mass_normalize": False,
            "weights": None,
        }

    def sampler_kwargs(self) -> dict[str, int]:
        return {
            "nwalkers": self.nwalkers,
            "nsteps": self.nsteps,
            "burnin": self.burnin,
        }


@dataclass(frozen=True)
class ChronosPosteriorSummary:
    age_samples_myr: np.ndarray
    av_samples: np.ndarray
    skewness_samples: np.ndarray
    scale_samples: np.ndarray
    age_mode: float
    age_lo: float
    age_hi: float
    age_median_lo: float
    age_median: float
    age_median_hi: float
    av_mode: float
    av_lo: float
    av_hi: float


def select_clusters_for_chronos(
    df_clusters: pd.DataFrame,
    *,
    max_input_age_myr: float,
) -> pd.DataFrame:
    """Apply the production Chronos pre-fit cluster selection."""
    age_myr = pd.to_numeric(df_clusters["age_myr"], errors="coerce")
    selected = df_clusters.loc[np.isfinite(age_myr) & (age_myr < max_input_age_myr)].copy()
    selected["age_myr"] = pd.to_numeric(selected["age_myr"], errors="coerce")
    return selected.sort_values(by="age_myr", ascending=True).reset_index(drop=True)


def prepare_member_photometry(
    df_stars: pd.DataFrame,
    df_clusters: pd.DataFrame,
) -> pd.DataFrame:
    """Build the stellar photometry table that Chronos fits cluster-by-cluster."""
    cluster_names = df_clusters["name"].astype(str)
    stars = df_stars.loc[df_stars["name"].astype(str).isin(cluster_names)].copy()

    parallax = pd.to_numeric(stars["parallax"], errors="coerce")
    stars["g_abs_mag"] = pd.to_numeric(stars["phot_g_mean_mag"], errors="coerce") + 5.0 - 5.0 * np.log10(
        1000.0 / parallax
    )

    optional_columns = [
        "parallax_error",
        "phot_g_mean_flux",
        "phot_g_mean_flux_error",
        "phot_bp_mean_flux",
        "phot_bp_mean_flux_error",
        "phot_rp_mean_flux",
        "phot_rp_mean_flux_error",
        "distance_50",
    ]
    fit_columns = [
        "name",
        "ra",
        "dec",
        "parallax",
        "phot_g_mean_mag",
        "phot_bp_mean_mag",
        "phot_rp_mean_mag",
        "g_abs_mag",
        "bp_rp",
        "g_rp",
    ]
    fit_columns.extend(column for column in optional_columns if column in stars.columns)
    fit_data = stars[fit_columns].copy()
    fit_data = fit_data.rename(columns={"bp_rp": "bp-rp", "g_rp": "g-rp"})

    if "distance_50" in fit_data.columns:
        fit_data["distance_50"] = pd.to_numeric(stars["distance_50"], errors="coerce")
    else:
        fit_data["distance_50"] = 1000.0 / pd.to_numeric(fit_data["parallax"], errors="coerce")

    fit_data["label"] = fit_data["name"]
    fit_data = pd.merge(fit_data, df_clusters[["name", "age_myr"]], on="name", how="left")
    return fit_data


def configure_cluster_fitter(
    df_group: pd.DataFrame,
    fit_config: ChronosFitConfig,
):
    """Instantiate and configure the production Chronos skew-Cauchy fitter."""
    from chronos.bayes_fitting.ChronosSkewedCauchy_bayes import ChronosSkewCauchyBayes

    cbayes = ChronosSkewCauchyBayes(**fit_config.chronos_kwargs(data=df_group))
    cbayes.set_fitting_kwargs(**fit_config.fitting_kwargs())
    cbayes.set_bounds(**fit_config.bayes_bounds())
    return cbayes


def summarize_skew_cauchy_samples(
    samples: np.ndarray,
    *,
    hdi_prob: float,
) -> ChronosPosteriorSummary:
    """Summarize the production Chronos posterior samples into reportable ages and extinctions."""
    log_age, _, av, skewness, scale = np.asarray(samples).T
    age_samples_myr = 10**log_age / 1e6
    age_lo, age_hi = az.hdi(age_samples_myr, hdi_prob=hdi_prob)
    age_median_lo, age_median, age_median_hi = np.nanpercentile(age_samples_myr, [16, 50, 84])
    av_lo, av_hi = az.hdi(av, hdi_prob=hdi_prob)
    return ChronosPosteriorSummary(
        age_samples_myr=age_samples_myr,
        av_samples=av,
        skewness_samples=skewness,
        scale_samples=scale,
        age_mode=mode_reals(age_samples_myr, bins=100),
        age_lo=float(age_lo),
        age_hi=float(age_hi),
        age_median_lo=float(age_median_lo),
        age_median=float(age_median),
        age_median_hi=float(age_median_hi),
        av_mode=mode_reals(av, bins=100),
        av_lo=float(av_lo),
        av_hi=float(av_hi),
    )
