import numpy as np
from mapper.sampling import make_cluster_compat


class MassFitter:
    def __init__(self, observed_masses, massfunc='kroupa', completeness_range=(0.05, 10), n_bins=20, n_draws=500,
                 mass_range=(10, 5_000)):
        observed = np.asarray(observed_masses, dtype=float)
        self.observed_masses = observed[np.isfinite(observed) & (observed > 0.0)]
        self.massfunc = massfunc
        self.completeness_range = completeness_range
        self.n_bins = n_bins
        self.bins = np.logspace(np.log10(completeness_range[0]), np.log10(completeness_range[1]), n_bins)
        # Number of draws to smooth the mass distribution
        self.n_draws = n_draws
        self.mass_range = mass_range

    def compute_observed_hist(self, bootstrap=False):
        if bootstrap:
            n_samples = len(self.observed_masses)
            idx = np.random.choice(range(n_samples), size=n_samples, replace=True)
            hist, edges = np.histogram(self.observed_masses[idx], bins=self.bins)
        else:
            hist, edges = np.histogram(self.observed_masses, bins=self.bins)
        return hist

    def set_observed_masses(self, observed_masses):
        observed = np.asarray(observed_masses, dtype=float)
        self.observed_masses = observed[np.isfinite(observed) & (observed > 0.0)]

    def set_bins(self, completeness_range, n_bins):
        self.completeness_range = completeness_range
        self.n_bins = n_bins
        self.bins = np.logspace(np.log10(completeness_range[0]), np.log10(completeness_range[1]), n_bins)

    def set_draws(self, n_draws):
        self.n_draws = n_draws

    def model_masses(self, cluster_mass):
        masses = []
        edges = None
        for i in range(self.n_draws):
            m_samples = make_cluster_compat(cluster_mass, massfunc=self.massfunc)
            hist, edges = np.histogram(m_samples, bins=self.bins)
            masses.append(hist)
        hist = np.mean(masses, axis=0)
        return hist, edges

    def chi2_cluster_mass(self, cluster_mass, observed_masses_binned):
        if isinstance(cluster_mass, np.ndarray):
            cluster_mass = cluster_mass[0]
        hist, _ = self.model_masses(cluster_mass)
        valid = (hist > 0.0) | (observed_masses_binned > 0.0)
        if not np.any(valid):
            return np.inf
        variance = np.clip(hist[valid], 1.0, None)
        return float(np.sum((observed_masses_binned[valid] - hist[valid]) ** 2 / variance))

    def grid_fitter(self, min_mass, max_mass, n_grid_pts, observed_masses_binned):
        """Implement grid fitter due to significant time savings"""
        mass_grid = np.linspace(min_mass, max_mass, n_grid_pts)
        chi2_res = [self.chi2_cluster_mass(mass_i, observed_masses_binned) for mass_i in mass_grid]
        return mass_grid[np.argmin(chi2_res)], np.diff(mass_grid)[0]

    def fit(self, n_iter=5, n_grid_pts=5, bootstrap=False):
        if self.observed_masses.size == 0:
            return np.nan
        observed_masses_binned = self.compute_observed_hist(bootstrap=bootstrap)
        observed_sum = float(np.sum(self.observed_masses))
        mass_lo = max(float(self.mass_range[0]), observed_sum)
        mass_hi = max(float(self.mass_range[1]), observed_sum * 4.0, mass_lo * 1.5)
        for _ in range(n_iter):
            best_mass_estimate, step_size = self.grid_fitter(
                min_mass=max((mass_lo, self.mass_range[0])),
                max_mass=mass_hi,
                n_grid_pts=n_grid_pts,
                observed_masses_binned=observed_masses_binned
            )
            if not np.isfinite(best_mass_estimate):
                return np.nan
            mass_lo = max(observed_sum, best_mass_estimate - step_size)
            mass_hi = max(mass_lo * 1.05, best_mass_estimate + step_size)
        return best_mass_estimate
