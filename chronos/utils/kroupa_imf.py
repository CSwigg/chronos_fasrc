from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np


KROUPA_DEFAULT_MMIN = 0.03
KROUPA_DEFAULT_MMAX = 120.0
KROUPA_BREAKS = (0.08, 0.5)
KROUPA_POWERS = (0.3, 1.3, 2.3)


def _resolve_bounds(mmin: float | None, mmax: float | None) -> tuple[float, float]:
    lo = KROUPA_DEFAULT_MMIN if mmin is None else float(mmin)
    hi = KROUPA_DEFAULT_MMAX if mmax is None else float(mmax)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0.0 or hi <= lo:
        raise ValueError(f"Invalid Kroupa IMF bounds: mmin={mmin!r}, mmax={mmax!r}")
    return lo, hi


def _primitive_power(lo: float, hi: float, alpha: float) -> float:
    if hi <= lo:
        return 0.0
    if np.isclose(alpha, 1.0):
        return float(np.log(hi / lo))
    return float((hi ** (1.0 - alpha) - lo ** (1.0 - alpha)) / (1.0 - alpha))


def _primitive_mass_power(lo: float, hi: float, alpha: float) -> float:
    if hi <= lo:
        return 0.0
    if np.isclose(alpha, 2.0):
        return float(np.log(hi / lo))
    return float((hi ** (2.0 - alpha) - lo ** (2.0 - alpha)) / (2.0 - alpha))


def _segment_coefficients() -> tuple[float, float, float]:
    b1, b2 = KROUPA_BREAKS
    a1, a2, a3 = KROUPA_POWERS
    c1 = 1.0
    c2 = c1 * b1 ** (a2 - a1)
    c3 = c2 * b2 ** (a3 - a2)
    return float(c1), float(c2), float(c3)


def _segments(mmin: float, mmax: float) -> list[tuple[float, float, float, float]]:
    b1, b2 = KROUPA_BREAKS
    bounds = (mmin, min(b1, mmax), min(b2, mmax), mmax)
    coeffs = _segment_coefficients()
    segments: list[tuple[float, float, float, float]] = []
    for lo, hi, alpha, coeff in zip(bounds[:-1], bounds[1:], KROUPA_POWERS, coeffs):
        lo = max(float(lo), float(mmin))
        hi = min(float(hi), float(mmax))
        if hi > lo:
            segments.append((lo, hi, float(alpha), float(coeff)))
    return segments


@lru_cache(maxsize=128)
def _normalization(mmin: float, mmax: float) -> float:
    total = 0.0
    for lo, hi, alpha, coeff in _segments(mmin, mmax):
        total += coeff * _primitive_power(lo, hi, alpha)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"Cannot normalize Kroupa IMF for bounds {(mmin, mmax)!r}")
    return float(total)


def kroupa_pdf(
    masses_msun: np.ndarray | float,
    *,
    mmin: float | None = None,
    mmax: float | None = None,
) -> np.ndarray:
    """Return the normalized Kroupa number-density PDF."""
    lo_bound, hi_bound = _resolve_bounds(mmin, mmax)
    masses = np.asarray(masses_msun, dtype=float)
    pdf = np.zeros_like(masses, dtype=float)
    norm = _normalization(lo_bound, hi_bound)
    for lo, hi, alpha, coeff in _segments(lo_bound, hi_bound):
        mask = (masses >= lo) & (masses <= hi)
        pdf[mask] = coeff * np.power(masses[mask], -alpha) / norm
    return pdf


def kroupa_number_integral(
    lo: float,
    hi: float,
    *,
    mmin: float | None = None,
    mmax: float | None = None,
) -> float:
    """Integrate the normalized Kroupa number-density PDF over mass."""
    lo_bound, hi_bound = _resolve_bounds(mmin, mmax)
    lo = max(float(lo), lo_bound)
    hi = min(float(hi), hi_bound)
    if hi <= lo:
        return 0.0
    norm = _normalization(lo_bound, hi_bound)
    total = 0.0
    for seg_lo, seg_hi, alpha, coeff in _segments(lo_bound, hi_bound):
        part_lo = max(lo, seg_lo)
        part_hi = min(hi, seg_hi)
        if part_hi > part_lo:
            total += coeff * _primitive_power(part_lo, part_hi, alpha)
    return float(total / norm)


def kroupa_mass_integral(
    lo: float | None = None,
    hi: float | None = None,
    *,
    mmin: float | None = None,
    mmax: float | None = None,
) -> float:
    """Integrate mass times the normalized Kroupa number-density PDF."""
    lo_bound, hi_bound = _resolve_bounds(mmin, mmax)
    lo = lo_bound if lo is None else max(float(lo), lo_bound)
    hi = hi_bound if hi is None else min(float(hi), hi_bound)
    if hi <= lo:
        return 0.0
    norm = _normalization(lo_bound, hi_bound)
    total = 0.0
    for seg_lo, seg_hi, alpha, coeff in _segments(lo_bound, hi_bound):
        part_lo = max(lo, seg_lo)
        part_hi = min(hi, seg_hi)
        if part_hi > part_lo:
            total += coeff * _primitive_mass_power(part_lo, part_hi, alpha)
    return float(total / norm)


def sample_kroupa_masses(
    size: int,
    *,
    mmin: float | None = None,
    mmax: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw stellar masses from the normalized Kroupa IMF."""
    lo_bound, hi_bound = _resolve_bounds(mmin, mmax)
    size = int(size)
    if size <= 0:
        return np.array([], dtype=float)

    rng_uniform = np.random.random(size) if rng is None else rng.random(size)
    segment_weights = np.array(
        [
            coeff * _primitive_power(lo, hi, alpha)
            for lo, hi, alpha, coeff in _segments(lo_bound, hi_bound)
        ],
        dtype=float,
    )
    segment_prob = segment_weights / np.sum(segment_weights)
    segment_cdf = np.cumsum(segment_prob)
    segment_indices = np.searchsorted(segment_cdf, rng_uniform, side="right")
    segment_indices = np.clip(segment_indices, 0, len(segment_prob) - 1)

    prior_cdf = np.concatenate([[0.0], segment_cdf[:-1]])
    local_u = (rng_uniform - prior_cdf[segment_indices]) / segment_prob[segment_indices]
    local_u = np.clip(local_u, 0.0, np.nextafter(1.0, 0.0))

    draws = np.empty(size, dtype=float)
    segments = _segments(lo_bound, hi_bound)
    for index, (lo, hi, alpha, _) in enumerate(segments):
        mask = segment_indices == index
        if not np.any(mask):
            continue
        u = local_u[mask]
        if np.isclose(alpha, 1.0):
            draws[mask] = lo * np.exp(u * np.log(hi / lo))
        else:
            exponent = 1.0 - alpha
            draws[mask] = (lo**exponent + u * (hi**exponent - lo**exponent)) ** (1.0 / exponent)
    return draws


def make_kroupa_cluster(
    cluster_mass_msun: float,
    *,
    stop_criterion: Literal["nearest", "before", "after", "sorted"] = "nearest",
    mmin: float | None = None,
    mmax: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a Kroupa IMF until the stellar masses approximate a cluster mass."""
    cluster_mass = float(cluster_mass_msun)
    if not np.isfinite(cluster_mass) or cluster_mass <= 0.0:
        return np.array([], dtype=float)

    lo_bound, hi_bound = _resolve_bounds(mmin, mmax)
    expected_mass = kroupa_mass_integral(mmin=lo_bound, mmax=hi_bound)
    if not np.isfinite(expected_mass) or expected_mass <= 0.0:
        return np.array([], dtype=float)

    total = 0.0
    masses = np.array([], dtype=float)
    while total < cluster_mass:
        n_draw = max(int(np.ceil((cluster_mass - total) / expected_mass)), 1)
        masses = np.concatenate(
            [
                masses,
                sample_kroupa_masses(n_draw, mmin=lo_bound, mmax=hi_bound, rng=rng),
            ]
        )
        total = float(np.sum(masses))

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
    return np.asarray(masses[:last_index], dtype=float)


__all__ = [
    "KROUPA_DEFAULT_MMAX",
    "KROUPA_DEFAULT_MMIN",
    "kroupa_mass_integral",
    "kroupa_number_integral",
    "kroupa_pdf",
    "make_kroupa_cluster",
    "sample_kroupa_masses",
]
