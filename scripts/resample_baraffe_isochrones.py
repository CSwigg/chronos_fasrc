#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SOURCE = Path("chronos/data/baraffe_files/BHAC15_iso.GAIA")
DEFAULT_PARSEC_MANIFEST = Path("inputs/parsec_isochrones_hybrid_0p1myr_to13gyr/manifest.json")
DEFAULT_OUTPUT_DIR = Path("inputs/baraffe_isochrones_bhac15_parsec_like")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_baraffe_file(path: Path) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    columns: list[str] | None = None
    current_age_gyr: float | None = None
    rows: list[list[float]] = []

    def flush() -> None:
        nonlocal rows, current_age_gyr
        if columns is None or current_age_gyr is None or not rows:
            rows = []
            return
        frame = pd.DataFrame(rows, columns=columns)
        frame["age_gyr"] = float(current_age_gyr)
        frame["logAge"] = float(np.log10(current_age_gyr * 1.0e9))
        frames.append(frame)
        rows = []

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("!  t (Gyr) ="):
                flush()
                match = re.search(r"=\s*([0-9.Ee+-]+)", stripped)
                if match is None:
                    raise ValueError(f"Could not parse age from line: {line!r}")
                current_age_gyr = float(match.group(1))
                continue
            if stripped.startswith("! M/Ms"):
                columns = stripped.lstrip("!").split()
                continue
            if not stripped or stripped.startswith("!"):
                continue
            if columns is None or current_age_gyr is None:
                continue
            values = [float(value) for value in stripped.split()]
            if len(values) == len(columns):
                rows.append(values)
    flush()
    if not frames or columns is None:
        raise ValueError(f"No Baraffe isochrone blocks found in {path}")
    return pd.concat(frames, ignore_index=True), columns


def _build_target_ages_from_parsec_manifest(manifest_path: Path, *, source_min_yr: float, source_max_yr: float) -> np.ndarray:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    values: list[float] = []
    for segment in manifest["age_grid"]["segments"]:
        count = int(segment["count"])
        low = float(segment["low"])
        step = float(segment["step"])
        if count <= 0:
            continue
        if bool(segment["is_log_age"]):
            values.extend(10.0 ** (low + index * step) for index in range(count))
        elif count == 1 or step == 0.0:
            values.append(low)
        else:
            values.extend(low + index * step for index in range(count))
    ages = np.array(sorted(set(round(float(value), 6) for value in values)), dtype=float)
    ages = ages[(ages >= source_min_yr) & (ages <= source_max_yr)]
    ages = np.array(sorted(set(ages.tolist() + [round(float(source_max_yr), 6)])), dtype=float)
    return ages


def _resample(frame: pd.DataFrame, columns: list[str], target_ages_yr: np.ndarray) -> pd.DataFrame:
    source_log_age = np.sort(frame["logAge"].unique())
    target_log_age = np.log10(target_ages_yr)
    numeric_columns = [column for column in columns if column != "M/Ms"]
    output_frames: list[pd.DataFrame] = []
    by_age = {
        float(age): group.set_index("M/Ms").sort_index()
        for age, group in frame.groupby("logAge", sort=True)
    }
    for age_yr, log_age in zip(target_ages_yr, target_log_age, strict=True):
        upper_index = int(np.searchsorted(source_log_age, log_age, side="left"))
        if upper_index == 0:
            lo_age = hi_age = float(source_log_age[0])
        elif upper_index >= len(source_log_age):
            lo_age = hi_age = float(source_log_age[-1])
        elif np.isclose(source_log_age[upper_index], log_age, rtol=0.0, atol=1e-12):
            lo_age = hi_age = float(source_log_age[upper_index])
        else:
            lo_age = float(source_log_age[upper_index - 1])
            hi_age = float(source_log_age[upper_index])

        lo = by_age[lo_age]
        hi = by_age[hi_age]
        if lo_age == hi_age:
            block = lo.reset_index()[columns].copy()
        else:
            common_masses = lo.index.intersection(hi.index)
            weight = float((log_age - lo_age) / (hi_age - lo_age))
            block = pd.DataFrame({"M/Ms": common_masses.to_numpy(dtype=float)})
            for column in numeric_columns:
                lo_values = pd.to_numeric(lo.loc[common_masses, column], errors="coerce").to_numpy(dtype=float)
                hi_values = pd.to_numeric(hi.loc[common_masses, column], errors="coerce").to_numpy(dtype=float)
                block[column] = lo_values + weight * (hi_values - lo_values)
        block["age_gyr"] = float(age_yr / 1.0e9)
        block["logAge"] = float(log_age)
        output_frames.append(block)
    resampled = pd.concat(output_frames, ignore_index=True)
    return resampled.replace([np.inf, -np.inf], np.nan).dropna()


def _write_baraffe_file(frame: pd.DataFrame, columns: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for age_gyr, block in frame.groupby("age_gyr", sort=True):
            handle.write(f"!  t (Gyr) =   {float(age_gyr):.10g}\n")
            handle.write("!-----------------------------------------------------------------------------------------------\n")
            handle.write("! " + " ".join(columns) + "\n")
            handle.write("!-----------------------------------------------------------------------------------------------\n")
            block = block.sort_values("M/Ms")
            for _, row in block.iterrows():
                handle.write(" ".join(f"{float(row[column]):.10g}" for column in columns) + "\n")
            handle.write("!-----------------------------------------------------------------------------------------------\n\n\n")


def build_grid(*, source: Path, parsec_manifest: Path, output_dir: Path, overwrite: bool) -> dict[str, object]:
    output_path = output_dir / "BHAC15_iso_parsec_like.GAIA"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it.")
    source_frame, columns = _parse_baraffe_file(source)
    source_ages_yr = 10.0 ** source_frame["logAge"].to_numpy(dtype=float)
    target_ages_yr = _build_target_ages_from_parsec_manifest(
        parsec_manifest,
        source_min_yr=float(np.min(source_ages_yr)),
        source_max_yr=float(np.max(source_ages_yr)),
    )
    resampled = _resample(source_frame, columns, target_ages_yr)
    _write_baraffe_file(resampled, columns, output_path)
    manifest = {
        "source": str(source),
        "output_file": str(output_path),
        "parsec_manifest": str(parsec_manifest),
        "method": "linear interpolation in logAge at fixed BHAC15 mass grid",
        "important_limitations": [
            "BHAC15 is low-mass only; this grid remains limited to the source mass range.",
            "The source BHAC15 file only spans 0.5 Myr to 10 Gyr, so target ages outside that range are omitted.",
            "This is a resampled Chronos grid, not a new stellar evolution calculation.",
        ],
        "actual_counts": {
            "rows": int(len(resampled)),
            "ages": int(resampled["age_gyr"].nunique()),
            "age_min_myr": float(resampled["age_gyr"].min() * 1000.0),
            "age_max_myr": float(resampled["age_gyr"].max() * 1000.0),
            "masses": int(resampled["M/Ms"].nunique()),
            "mass_min_msun": float(resampled["M/Ms"].min()),
            "mass_max_msun": float(resampled["M/Ms"].max()),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resample the bundled BHAC15/Baraffe Gaia grid onto the PARSEC-like age grid.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parsec-manifest", type=Path, default=DEFAULT_PARSEC_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = repo_root()
    source = args.source if args.source.is_absolute() else base / args.source
    parsec_manifest = args.parsec_manifest if args.parsec_manifest.is_absolute() else base / args.parsec_manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else base / args.output_dir
    manifest = build_grid(
        source=source,
        parsec_manifest=parsec_manifest,
        output_dir=output_dir,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(manifest["actual_counts"], indent=2))


if __name__ == "__main__":
    main()
