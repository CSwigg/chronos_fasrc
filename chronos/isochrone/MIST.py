from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from chronos.isochrone.ICBase import ICBase


class MIST(ICBase):
    """Handling MIST CMD isochrones in Gaia DR3-like passbands."""

    _CACHE = {}

    _AGE_COLUMNS = (
        "log10_isochrone_age_yr",
        "logAge",
        "log_age",
        "log_age_yr",
    )
    _MASS_COLUMNS = (
        "initial_mass",
        "star_mass",
        "mass",
        "M_ini",
        "Mini",
    )
    _METAL_COLUMNS = (
        "[Fe/H]_init",
        "[Fe/H]",
        "FeH",
        "MH",
        "feh",
    )
    _G_COLUMNS = (
        "Gaia_G_DR3",
        "GaiaDR3_G",
        "Gaia_G_EDR3",
        "GaiaEDR3_G",
        "Gaia_G",
        "G_DR3",
        "G",
        "Gmag",
    )
    _BP_COLUMNS = (
        "Gaia_BP_DR3",
        "GaiaDR3_BP",
        "Gaia_BP_EDR3",
        "GaiaEDR3_BP",
        "Gaia_BP",
        "BP_DR3",
        "BP",
        "G_BP",
        "G_BP_DR3",
        "G_BPmag",
    )
    _RP_COLUMNS = (
        "Gaia_RP_DR3",
        "GaiaDR3_RP",
        "Gaia_RP_EDR3",
        "GaiaEDR3_RP",
        "Gaia_RP",
        "RP_DR3",
        "RP",
        "G_RP",
        "G_RP_DR3",
        "G_RPmag",
    )

    def __init__(self, dir_path, file_ending="cmd", nb_interpolated=400):
        super().__init__(nb_interpolated)
        self.comment = "#"
        self.dir_path = str(dir_path)
        self.colnames = {
            "mass": "initial_mass",
            "age": "log10_isochrone_age_yr",
            "metal": "[Fe/H]_init",
            "gmag": "Gaia_G_DR3",
            "bp": "Gaia_BP_DR3",
            "rp": "Gaia_RP_DR3",
        }
        endings = (file_ending,) if isinstance(file_ending, str) else tuple(file_ending)
        self.flist_all: list[str] = []
        for ending in endings:
            self.flist_all.extend(glob.glob(os.path.join(self.dir_path, f"*.{ending}")))
        self.flist_all = sorted(set(self.flist_all))
        if not self.flist_all:
            raise FileNotFoundError(
                "No MIST isochrone files found in "
                f"{Path(self.dir_path).expanduser()} with endings {', '.join(endings)}. "
                "Expected MIST Gaia DR3 CMD files, e.g. '*.cmd' or '*.iso.cmd'."
            )
        cache_key = (
            str(Path(self.dir_path).expanduser().resolve()),
            tuple(self.flist_all),
            int(nb_interpolated),
        )
        cached = self._CACHE.get(cache_key)
        if cached is None:
            self.data = self.read_files(self.flist_all)
            self.process_isochrone_infos()
            self._CACHE[cache_key] = {
                "data": self.data,
                "unique_ages": self.unique_ages,
                "age_mask": self.age_mask,
                "unique_metallicity": self.unique_metallicity,
                "metallicity_mask": self.metallicity_mask,
                "rgi": self.rgi,
                "nndi": self.nndi,
            }
        else:
            self.data = cached["data"]
            self.unique_ages = cached["unique_ages"]
            self.age_mask = cached["age_mask"]
            self.unique_metallicity = cached["unique_metallicity"]
            self.metallicity_mask = cached["metallicity_mask"]
            self.rgi = cached["rgi"]
            self.nndi = cached["nndi"]

    @staticmethod
    def _select_column(
        columns: list[str],
        candidates: tuple[str, ...],
        *,
        label: str,
        fname: str,
    ) -> str:
        for candidate in candidates:
            if candidate in columns:
                return candidate
        raise ValueError(
            f"Could not find {label} column in MIST file {fname}. "
            f"Tried: {', '.join(candidates)}"
        )

    @classmethod
    def _header_columns(cls, fname: str) -> list[str]:
        header: list[str] | None = None
        with open(fname, encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped.startswith("#"):
                    continue
                tokens = stripped.lstrip("#").strip().split()
                if not tokens:
                    continue
                has_age = any(column in tokens for column in cls._AGE_COLUMNS)
                has_g = any(column in tokens for column in cls._G_COLUMNS)
                has_bp = any(column in tokens for column in cls._BP_COLUMNS)
                has_rp = any(column in tokens for column in cls._RP_COLUMNS)
                if has_age and has_g and has_bp and has_rp:
                    header = tokens
        if header is None:
            raise ValueError(
                f"Could not identify a MIST CMD header in {fname}. "
                "Expected a commented column line with age and Gaia DR3 G/BP/RP columns."
            )
        return header

    def read_files(self, flist: list[str]) -> pd.DataFrame:
        frames = [self.read(fname) for fname in sorted(flist)]
        data = pd.concat(frames, ignore_index=True)
        return self._ensure_metallicity_grid(data)

    def read(self, fname: str) -> pd.DataFrame:
        columns = self._header_columns(fname)
        df_iso = pd.read_csv(fname, sep=r"\s+", comment=self.comment, header=None)
        if len(columns) != df_iso.shape[1]:
            raise ValueError(
                f"MIST header/data column mismatch in {fname}: "
                f"header has {len(columns)} columns, data has {df_iso.shape[1]}."
            )
        df_iso.columns = columns

        age_col = self._select_column(columns, self._AGE_COLUMNS, label="age", fname=fname)
        mass_col = self._select_column(columns, self._MASS_COLUMNS, label="mass", fname=fname)
        g_col = self._select_column(columns, self._G_COLUMNS, label="Gaia DR3 G", fname=fname)
        bp_col = self._select_column(columns, self._BP_COLUMNS, label="Gaia DR3 BP", fname=fname)
        rp_col = self._select_column(columns, self._RP_COLUMNS, label="Gaia DR3 RP", fname=fname)
        metal_col = next((column for column in self._METAL_COLUMNS if column in columns), None)

        data = pd.DataFrame(
            {
                self.colnames["age"]: pd.to_numeric(df_iso[age_col], errors="coerce"),
                self.colnames["mass"]: pd.to_numeric(df_iso[mass_col], errors="coerce"),
                self.colnames["gmag"]: pd.to_numeric(df_iso[g_col], errors="coerce"),
                self.colnames["bp"]: pd.to_numeric(df_iso[bp_col], errors="coerce"),
                self.colnames["rp"]: pd.to_numeric(df_iso[rp_col], errors="coerce"),
            }
        )
        if metal_col is None:
            data[self.colnames["metal"]] = 0.0
        else:
            data[self.colnames["metal"]] = pd.to_numeric(df_iso[metal_col], errors="coerce")
        data[self.g_rp] = data[self.colnames["gmag"]] - data[self.colnames["rp"]]
        data[self.bp_rp] = data[self.colnames["bp"]] - data[self.colnames["rp"]]
        keep_cols = [
            self.colnames["age"],
            self.colnames["mass"],
            self.colnames["metal"],
            self.colnames["gmag"],
            self.colnames["bp"],
            self.colnames["rp"],
            self.g_rp,
            self.bp_rp,
        ]
        return data[keep_cols].replace([np.inf, -np.inf], np.nan).dropna()

    def _ensure_metallicity_grid(self, data: pd.DataFrame) -> pd.DataFrame:
        metals = np.sort(data[self.colnames["metal"]].dropna().unique())
        if len(metals) != 1:
            return data
        metal = float(metals[0])
        delta = 1.0e-5
        lower = data.copy()
        upper = data.copy()
        lower[self.colnames["metal"]] = metal - delta
        upper[self.colnames["metal"]] = metal + delta
        return pd.concat([lower, upper], ignore_index=True)
