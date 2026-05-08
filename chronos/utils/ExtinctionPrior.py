from pathlib import Path
import warnings

from astropy import units as u
from astropy.coordinates import SkyCoord
from dustmaps.edenhofer2023 import Edenhofer2023Query
import numpy as np
import pandas as pd


class ExtinctionPrior:
    EDENHOFER_TO_AV = 2.8
    BAYESTAR2019_TO_AV = 2.742
    DECAPS_EBV_TO_AV = 3.1

    def __init__(
        self,
        fname_edenhofer2023,
        *,
        bayestar2019_map_fname=None,
        decaps_map_fname=None,
    ):
        self.edh_dustmap = Edenhofer2023Query(
            map_fname=fname_edenhofer2023,
            load_samples=False,
            integrated=True,
            flavor="main",
            seed=None,
        )
        self.bayestar2019_map_fname = self._existing_path_or_none(bayestar2019_map_fname)
        self.decaps_map_fname = self._existing_path_or_none(decaps_map_fname)
        self._bayestar2019 = None
        self._decaps = None

    @staticmethod
    def _existing_path_or_none(path):
        if path is None:
            return None
        candidate = Path(path).expanduser()
        return str(candidate) if candidate.exists() else None

    def check_input(self, coord_input):
        if isinstance(coord_input, (pd.DataFrame, pd.Series)):
            return coord_input.values
        elif isinstance(coord_input, (list, tuple)):
            return np.array(coord_input)
        elif isinstance(coord_input, np.ndarray):
            return coord_input
        else:
            raise ValueError("Input should be a DataFrame, Series, list, or numpy array")

    def _skycoord(self, ra, dec, distance):
        ra_arr = np.asarray(ra, dtype=float)
        dec_arr = np.asarray(dec, dtype=float)
        distance_arr = np.asarray(distance, dtype=float)
        valid_mask = np.isfinite(ra_arr) & np.isfinite(dec_arr) & np.isfinite(distance_arr) & (distance_arr > 0.0)
        if not np.any(valid_mask):
            return None, valid_mask, distance_arr

        coords = SkyCoord(
            ra=ra_arr[valid_mask] * u.deg,
            dec=dec_arr[valid_mask] * u.deg,
            distance=distance_arr[valid_mask] * u.pc,
            frame="icrs",
        )
        return coords, valid_mask, distance_arr

    def _query_av_from_map(self, map_name, ra, dec, distance):
        coords, valid_mask, distance_arr = self._skycoord(ra, dec, distance)
        av = np.full(distance_arr.shape, np.nan, dtype=float)
        if coords is None:
            return av

        try:
            if map_name == "edenhofer2023":
                reddening = self.edh_dustmap.query(coords)
                av[valid_mask] = np.asarray(reddening, dtype=float) * self.EDENHOFER_TO_AV
            elif map_name == "bayestar2019":
                if self.bayestar2019_map_fname is None:
                    return av
                if self._bayestar2019 is None:
                    from dustmaps.bayestar import BayestarQuery

                    self._bayestar2019 = BayestarQuery(
                        map_fname=self.bayestar2019_map_fname,
                        max_samples=1,
                        version="bayestar2019",
                    )
                reddening = self._bayestar2019.query(coords, mode="best")
                av[valid_mask] = np.asarray(reddening, dtype=float) * self.BAYESTAR2019_TO_AV
            elif map_name == "decaps":
                if self.decaps_map_fname is None:
                    return av
                if self._decaps is None:
                    from dustmaps.decaps import DECaPSQueryLite

                    self._decaps = DECaPSQueryLite(
                        map_fname=self.decaps_map_fname,
                        mean_only=True,
                    )
                reddening = self._decaps.query(coords, mode="mean")
                av[valid_mask] = np.asarray(reddening, dtype=float) * self.DECAPS_EBV_TO_AV
            else:
                raise ValueError(f"Unsupported dust map: {map_name!r}")
        except Exception as exc:
            warnings.warn(f"{map_name} dust-map query failed: {exc}", RuntimeWarning)
        return av

    def _query_av(self, ra, dec, distance):
        return self._query_av_from_map("edenhofer2023", ra, dec, distance)

    def _query_av_stack(self, ra, dec, distance):
        ra_arr = np.asarray(ra, dtype=float)
        dec_arr = np.asarray(dec, dtype=float)
        distance_arr = np.asarray(distance, dtype=float)
        av = np.full(distance_arr.shape, np.nan, dtype=float)
        map_name = np.full(distance_arr.shape, "", dtype=object)
        remaining_mask = np.ones(distance_arr.shape, dtype=bool)
        for candidate in ("edenhofer2023", "bayestar2019", "decaps"):
            if not np.any(remaining_mask):
                break
            remaining_indices = np.where(remaining_mask)[0]
            candidate_av = self._query_av_from_map(
                candidate,
                ra=ra_arr[remaining_mask],
                dec=dec_arr[remaining_mask],
                distance=distance_arr[remaining_mask],
            )
            fill_mask = np.isfinite(candidate_av)
            if np.any(fill_mask):
                fill_indices = remaining_indices[fill_mask]
                av[fill_indices] = candidate_av[fill_mask]
                map_name[fill_indices] = candidate
                remaining_mask[fill_indices] = False
        return av, map_name

    def compute_prior(self, ra, dec, plx=None, distance=None):
        ra = self.check_input(ra)
        dec = self.check_input(dec)
        if plx is None:
            distance = self.check_input(distance)
        else:
            distance = 1000 / self.check_input(plx)
        return self._query_av_stack(ra=ra, dec=dec, distance=distance)[0]

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

        av, map_name = self._query_av_stack(ra=ra, dec=dec, distance=target_distance)
        valid_mask = np.isfinite(av)
        floor_av = np.full_like(av, np.nan, dtype=float)
        floor_distance_pc = np.full_like(av, np.nan, dtype=float)
        floor_map_name = np.full_like(map_name, "", dtype=object)

        for idx in np.where(~valid_mask)[0]:
            distance_i = float(target_distance[idx])
            if not np.isfinite(distance_i) or distance_i <= min_distance_pc:
                continue

            for candidate in ("edenhofer2023", "bayestar2019", "decaps"):
                lo = min_distance_pc
                hi = distance_i
                lo_av = self._query_av_from_map(
                    map_name=candidate,
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
                    mid_av = self._query_av_from_map(
                        map_name=candidate,
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
                floor_map_name[idx] = candidate
                break

        return pd.DataFrame(
            {
                "av": av,
                "map_name": map_name,
                "target_distance_pc": np.asarray(target_distance, dtype=float),
                "is_valid": valid_mask,
                "floor_av": floor_av,
                "floor_map_name": floor_map_name,
                "floor_distance_pc": floor_distance_pc,
            }
        )
