from __future__ import annotations

import numpy as np
import pandas as pd
import imf
from astropy import units as u
from scipy.integrate import quad


_EXPECTED_MASS_CACHE: dict[tuple[str, float, float], float] = {}


def make_cluster_compat(
    cluster_mass: float,
    massfunc: str = "kroupa",
    sampling: str = "random",
    stop_criterion: str = "nearest",
    mmin: float | None = None,
    mmax: float | None = None,
) -> np.ndarray:
    """Compatibility wrapper for keflavich/imf across releases.

    Newer `initial_mass_function` releases require numerical integration for the
    broken-power-law Kroupa IMF. The package's public `make_cluster()` helper
    still attempts the analytic path, so we reproduce the relevant random
    sampling logic here while forcing `numerical=True` in the expected-mass
    calculation.
    """
    if sampling != "random":
        raise ValueError(f"Unsupported sampling mode for compatibility wrapper: {sampling!r}")

    cluster_mass = u.Quantity(cluster_mass, u.M_sun).value
    mfc = imf.get_massfunc(massfunc, mmin=mmin, mmax=mmax)
    cache_key = (str(massfunc), float(mfc.mmin), float(mfc.mmax))
    expected_mass = _EXPECTED_MASS_CACHE.get(cache_key)
    if expected_mass is None:
        expected_mass = float(
            quad(lambda mass: float(mass) * float(mfc.distr.pdf(mass)), float(mfc.mmin), float(mfc.mmax))[0]
        )
        _EXPECTED_MASS_CACHE[cache_key] = expected_mass
    if not np.isfinite(expected_mass) or expected_mass <= 0.0:
        return np.array([], dtype=float)

    mtot = 0.0
    masses = np.array([], dtype=float)
    while mtot < cluster_mass:
        nsamp = int(np.ceil((cluster_mass - mtot) / expected_mass))
        nsamp = max(nsamp, 1)
        new_masses = np.asarray(mfc.distr.rvs(nsamp), dtype=float)
        masses = np.concatenate([masses, new_masses])
        mtot = float(np.sum(masses))
        if mtot < cluster_mass:
            continue

        cumulative = np.cumsum(masses)
        if stop_criterion == "nearest":
            last_index = int(np.argmin(np.abs(cumulative - cluster_mass))) + 1
        elif stop_criterion == "before":
            last_index = int(np.argmax(cumulative > cluster_mass))
        elif stop_criterion == "after":
            last_index = int(np.argmax(cumulative > cluster_mass)) + 1
        elif stop_criterion == "sorted":
            masses = np.sort(masses)
            if np.abs(np.sum(masses[:-1]) - cluster_mass) < np.abs(np.sum(masses) - cluster_mass):
                last_index = len(masses) - 1
            else:
                last_index = len(masses)
        else:
            raise ValueError(f"Unsupported stop_criterion={stop_criterion!r}")
        masses = masses[:last_index]
        break

    return np.asarray(masses, dtype=float)


def sample_cluster_age_myr(
    age_myr: float,
    age_myr_error: float,
    rng: np.random.Generator | None = None,
) -> float:
    """Sample a cluster age in Myr (clipped to >= 0)."""
    try:
        mu = float(age_myr)
    except Exception:
        return np.nan

    try:
        sigma = float(age_myr_error)
    except Exception:
        sigma = np.nan

    if not np.isfinite(mu):
        return np.nan
    if (not np.isfinite(sigma)) or sigma <= 0:
        return mu

    if rng is None:
        sampled = float(np.random.normal(mu, sigma))
    else:
        sampled = float(rng.normal(mu, sigma))
    if not np.isfinite(sampled):
        return mu
    return max(sampled, 0.0)


def sample_cluster_masses(
    cluster_mass: float,
    cluster_mass_err: float,
    n_samples: int,
    mass_scaling_factor: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample cluster masses with uncertainty, applying a scaling factor."""
    if not np.isfinite(cluster_mass) or not np.isfinite(cluster_mass_err):
        return np.array([], dtype=float)

    m = float(cluster_mass) * float(mass_scaling_factor)
    s = float(cluster_mass_err) * float(mass_scaling_factor)
    if m <= 0:
        return np.array([], dtype=float)

    if s <= 0 or not np.isfinite(s):
        return np.full(int(n_samples), m, dtype=float)

    if rng is None:
        return np.random.normal(m, s, int(n_samples))

    return rng.normal(m, s, int(n_samples))


def cluster_mass_dependent_mmax_msun(
    mass_sample: float,
    relation: str = "none",
) -> float | None:
    """Return the cluster-dependent stellar upper-mass limit, if configured."""
    if relation == "none":
        return None

    mass_sample = float(mass_sample)
    if not np.isfinite(mass_sample) or mass_sample <= 0.0:
        return None

    if relation == "weidner_2013":
        if mass_sample > 2.5e5:
            return 150.0
        if mass_sample < 3.0:
            return None
        x = np.log10(mass_sample)
        y = -0.66 + 1.08 * x - 0.150 * x * x + 0.0084 * x * x * x
        return min(float(10.0**y), 150.0)

    raise ValueError(f"Unsupported cluster_mass_max_relation={relation!r}")


def sample_imf_cluster(
    mass_sample: float,
    massive_star_threshold_msun: float = 8.0,
    imf_sampling: str = "random",
    cluster_mass_max_relation: str = "none",
) -> np.ndarray:
    """Sample an IMF for a single cluster mass and return massive stars."""
    if not np.isfinite(mass_sample) or mass_sample <= 0:
        return np.array([], dtype=float)

    mmax = cluster_mass_dependent_mmax_msun(mass_sample, relation=cluster_mass_max_relation)
    imf_kwargs = {
        "massfunc": "kroupa",
        "sampling": str(imf_sampling),
    }
    if mmax is not None and np.isfinite(mmax):
        imf_kwargs["mmax"] = float(mmax)

    imf_stars = make_cluster_compat(mass_sample, **imf_kwargs)
    massive = imf_stars[imf_stars >= float(massive_star_threshold_msun)]
    return np.asarray(massive, dtype=float)


__all__ = [
    "cluster_mass_dependent_mmax_msun",
    "make_cluster_compat",
    "sample_cluster_age_myr",
    "sample_cluster_masses",
    "sample_imf_cluster",
]
