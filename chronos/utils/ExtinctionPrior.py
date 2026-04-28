import numpy as np
import pandas as pd
from dustmaps.edenhofer2023 import Edenhofer2023Query
from astropy import units as u
from astropy.coordinates import SkyCoord


class ExtinctionPrior:
    def __init__(self, fname_edenhofer2023):
        self.edh_dustmap = Edenhofer2023Query(
            map_fname=fname_edenhofer2023,
            load_samples=False,
            integrated=True,
            flavor='main',
            seed=None
        )

    def check_input(self, coord_input):
        if isinstance(coord_input, (pd.DataFrame, pd.Series)):
            return coord_input.values
        elif isinstance(coord_input, (list, tuple)):
            return np.array(coord_input)
        elif isinstance(coord_input, np.ndarray):
            return coord_input
        else:
            raise ValueError('Input should be a DataFrame, Series, list, or numpy array')

    def _query_av(self, ra, dec, distance):
        ra_arr = np.asarray(ra, dtype=float)
        dec_arr = np.asarray(dec, dtype=float)
        distance_arr = np.asarray(distance, dtype=float)
        av = np.full(distance_arr.shape, np.nan, dtype=float)
        valid_mask = np.isfinite(ra_arr) & np.isfinite(dec_arr) & np.isfinite(distance_arr) & (distance_arr > 0.0)
        if not np.any(valid_mask):
            return av

        c = SkyCoord(
            ra=ra_arr[valid_mask] * u.deg,
            dec=dec_arr[valid_mask] * u.deg,
            distance=distance_arr[valid_mask] * u.pc,
            frame='icrs'
        )
        E = self.edh_dustmap.query(c)
        av[valid_mask] = np.asarray(E, dtype=float) * 2.8
        return av

    def compute_prior(self, ra, dec, plx=None, distance=None):
        ra = self.check_input(ra)
        dec = self.check_input(dec)
        if plx is None:
            distance = self.check_input(distance)
        else:
            distance = 1000/self.check_input(plx)
        return self._query_av(ra=ra, dec=dec, distance=distance)

    def compute_prior_details(
        self,
        ra,
        dec,
        plx=None,
        distance=None,
        *,
        min_distance_pc: float = 1.0,
        max_iter: int = 12,
    ) -> pd.DataFrame:
        ra = self.check_input(ra)
        dec = self.check_input(dec)
        if plx is None:
            target_distance = self.check_input(distance)
        else:
            target_distance = 1000 / self.check_input(plx)

        av = self._query_av(ra=ra, dec=dec, distance=target_distance)
        valid_mask = np.isfinite(av)
        floor_av = np.full_like(av, np.nan, dtype=float)
        floor_distance_pc = np.full_like(av, np.nan, dtype=float)

        for idx in np.where(~valid_mask)[0]:
            distance_i = float(target_distance[idx])
            if not np.isfinite(distance_i) or distance_i <= min_distance_pc:
                continue

            lo = min_distance_pc
            hi = distance_i
            lo_av = self._query_av(
                ra=np.array([ra[idx]], dtype=float),
                dec=np.array([dec[idx]], dtype=float),
                distance=np.array([lo], dtype=float),
            )[0]
            if not np.isfinite(lo_av):
                continue

            best_distance = lo
            best_av = float(lo_av)
            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                mid_av = self._query_av(
                    ra=np.array([ra[idx]], dtype=float),
                    dec=np.array([dec[idx]], dtype=float),
                    distance=np.array([mid], dtype=float),
                )[0]
                if np.isfinite(mid_av):
                    lo = mid
                    best_distance = mid
                    best_av = float(mid_av)
                else:
                    hi = mid

            floor_distance_pc[idx] = best_distance
            floor_av[idx] = best_av

        return pd.DataFrame(
            {
                "av": av,
                "target_distance_pc": np.asarray(target_distance, dtype=float),
                "is_valid": valid_mask,
                "floor_av": floor_av,
                "floor_distance_pc": floor_distance_pc,
            }
        )
