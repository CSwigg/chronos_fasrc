#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import re
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests


MIST_BASE_URL = "https://mist.science/"
MIST_FORM_URL = urljoin(MIST_BASE_URL, "iso_form.php")
DEFAULT_PARSEC_MANIFEST = Path("inputs/parsec_isochrones_hybrid_0p1myr_to13gyr/manifest.json")
DEFAULT_OUTPUT_DIR = Path("inputs/mist_isochrones_parsec_like_0p1myr_to13gyr")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _metallicity_label(value: float) -> str:
    prefix = "m" if value < 0 else "p"
    return prefix + str(abs(float(value))).replace(".", "p")


def _build_age_grid_from_parsec_manifest(manifest_path: Path, *, min_age_myr: float) -> np.ndarray:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    age_grid = manifest["age_grid"]
    values: list[float] = []
    for segment in age_grid["segments"]:
        count = int(segment["count"])
        low = float(segment["low"])
        step = float(segment["step"])
        if count <= 0:
            continue
        if bool(segment["is_log_age"]):
            values.extend((10.0 ** (low + index * step)) for index in range(count))
        elif count == 1 or step == 0.0:
            values.append(low)
        else:
            values.extend((low + index * step) for index in range(count))
    # The PARSEC downloader can request an exact endpoint that rounds onto an
    # existing sampled age. Keep the unique ages stable at sub-year precision.
    rounded = np.array([round(float(value), 6) for value in values], dtype=float)
    ages = np.array(sorted(set(rounded.tolist())), dtype=float)
    return ages[ages >= float(min_age_myr) * 1.0e6]


def _chunked(values: np.ndarray, chunk_size: int) -> list[np.ndarray]:
    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


def _request_mist_zip(
    *,
    ages_yr: np.ndarray,
    feh: float,
    version: str,
    rotation: str,
    output: str,
    timeout: int,
) -> bytes:
    payload = {
        "version": version,
        "v_div_vcrit": rotation,
        "age_scale": "linear",
        "age_type": "list",
        "age_list": " ".join(f"{float(age):.8g}" for age in ages_yr),
        "FeH_value": f"{float(feh):.5g}",
        "alpha_value": "p0",
        "output_option": "photometry",
        "output": output,
        "Av_value": "0",
    }
    response = requests.post(MIST_FORM_URL, data=payload, timeout=timeout)
    response.raise_for_status()
    match = re.search(r'href="([^"]+\.zip)"', response.text)
    if match is None:
        snippet = response.text[:500].replace("\n", " ")
        raise RuntimeError(f"MIST response did not include a zip link: {snippet}")
    zip_url = urljoin(MIST_BASE_URL, match.group(1))
    zip_response = requests.get(zip_url, timeout=timeout)
    zip_response.raise_for_status()
    return zip_response.content


def _header_columns(text: str) -> list[str]:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        tokens = stripped.lstrip("#").strip().split()
        if {"isochrone_age_yr", "initial_mass", "Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3"}.issubset(tokens):
            return tokens
        if {"log10_isochrone_age_yr", "initial_mass", "Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3"}.issubset(tokens):
            return tokens
    raise ValueError("Could not identify the MIST synthetic-photometry column header.")


def _normalize_mist_photometry(zip_bytes: bytes, *, feh: float) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = [name for name in archive.namelist() if name.endswith(".UBVRIplus")]
        if not names:
            raise ValueError("MIST zip did not contain an .UBVRIplus synthetic-photometry file.")
        text = archive.read(names[0]).decode("utf-8")
    columns = _header_columns(text)
    data = pd.read_csv(io.StringIO(text), sep=r"\s+", comment="#", header=None)
    if len(columns) != data.shape[1]:
        raise ValueError(f"MIST header/data column mismatch: {len(columns)} columns vs {data.shape[1]} data columns.")
    data.columns = columns

    if "log10_isochrone_age_yr" in data.columns:
        log_age = pd.to_numeric(data["log10_isochrone_age_yr"], errors="coerce")
    else:
        age_raw = pd.to_numeric(data["isochrone_age_yr"], errors="coerce")
        # MIST v1.2 UBVRIplus output labels this column as isochrone_age_yr,
        # but the values are log10(age/yr). Handle both conventions.
        finite = age_raw[np.isfinite(age_raw)]
        log_age = age_raw if len(finite) and float(finite.median()) < 100.0 else np.log10(age_raw)

    normalized = pd.DataFrame(
        {
            "log10_isochrone_age_yr": log_age,
            "initial_mass": pd.to_numeric(data["initial_mass"], errors="coerce"),
            "star_mass": pd.to_numeric(data.get("star_mass", data["initial_mass"]), errors="coerce"),
            "[Fe/H]_init": pd.to_numeric(data.get("[Fe/H]_init", feh), errors="coerce"),
            "Gaia_G_EDR3": pd.to_numeric(data["Gaia_G_EDR3"], errors="coerce"),
            "Gaia_BP_EDR3": pd.to_numeric(data["Gaia_BP_EDR3"], errors="coerce"),
            "Gaia_RP_EDR3": pd.to_numeric(data["Gaia_RP_EDR3"], errors="coerce"),
        }
    )
    normalized["[Fe/H]_init"] = normalized["[Fe/H]_init"].fillna(float(feh))
    normalized = normalized.replace([np.inf, -np.inf], np.nan).dropna()
    return normalized.sort_values(["log10_isochrone_age_yr", "initial_mass"]).reset_index(drop=True)


def _write_cmd(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Chronos-normalized MIST synthetic photometry\n")
        handle.write("# log10_isochrone_age_yr initial_mass star_mass [Fe/H]_init Gaia_G_EDR3 Gaia_BP_EDR3 Gaia_RP_EDR3\n")
        frame.to_csv(handle, sep=" ", header=False, index=False, float_format="%.10g")


def build_grid(
    *,
    output_dir: Path,
    parsec_manifest: Path,
    metallicities: list[float],
    min_age_myr: float,
    max_ages_per_request: int,
    version: str,
    rotation: str,
    output: str,
    timeout: int,
    sleep_seconds: float,
    overwrite: bool,
    dry_run: bool,
    limit_requests: int | None,
) -> dict[str, object]:
    ages_yr = _build_age_grid_from_parsec_manifest(parsec_manifest, min_age_myr=float(min_age_myr))
    requests_plan: list[tuple[float, int, np.ndarray, Path]] = []
    version_label = "v12" if version == "MIST1" else "v25"
    for feh in metallicities:
        for part_index, chunk in enumerate(_chunked(ages_yr, max_ages_per_request), start=1):
            out_name = (
                f"mist_{version_label}_{rotation}_{output.lower()}_"
                f"age_part{part_index:03d}_feh_{_metallicity_label(feh)}.cmd"
            )
            requests_plan.append((float(feh), part_index, chunk, output_dir / out_name))
    if limit_requests is not None:
        requests_plan = requests_plan[: int(limit_requests)]

    manifest: dict[str, object] = {
        "source": MIST_FORM_URL,
        "output_dir": str(output_dir),
        "parsec_manifest": str(parsec_manifest),
        "version": version,
        "rotation": rotation,
        "photometric_output": output,
        "age_grid": {
            "ages": int(len(ages_yr)),
            "age_min_myr": float(np.min(ages_yr) / 1.0e6),
            "age_max_myr": float(np.max(ages_yr) / 1.0e6),
            "requested_parsec_like_min_age_myr": float(min_age_myr),
        },
        "metallicity_values": [float(value) for value in metallicities],
        "max_ages_per_request": int(max_ages_per_request),
        "request_count": int(len(requests_plan)),
        "files": [],
    }

    if dry_run:
        print(json.dumps(manifest, indent=2))
        return manifest

    output_dir.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, object]] = []
    for request_index, (feh, part_index, chunk, output_path) in enumerate(requests_plan, start=1):
        if output_path.exists() and not overwrite:
            print(f"[{request_index}/{len(requests_plan)}] exists: {output_path.name}")
            completed.append({"path": str(output_path), "feh": feh, "part": part_index, "ages": int(len(chunk)), "status": "exists"})
            continue
        print(
            f"[{request_index}/{len(requests_plan)}] MIST feh={feh:+.2f} "
            f"ages={len(chunk)} -> {output_path.name}",
            flush=True,
        )
        zip_bytes = _request_mist_zip(
            ages_yr=chunk,
            feh=feh,
            version=version,
            rotation=rotation,
            output=output,
            timeout=timeout,
        )
        frame = _normalize_mist_photometry(zip_bytes, feh=feh)
        _write_cmd(frame, output_path)
        completed.append(
            {
                "path": str(output_path),
                "feh": feh,
                "part": part_index,
                "ages": int(len(np.unique(frame["log10_isochrone_age_yr"]))),
                "rows": int(len(frame)),
                "status": "downloaded",
            }
        )
        if sleep_seconds > 0 and request_index < len(requests_plan):
            time.sleep(float(sleep_seconds))

    manifest["files"] = completed
    manifest["actual_counts"] = {
        "files": int(len(completed)),
        "rows": int(sum(int(entry.get("rows", 0)) for entry in completed)),
        "ages_requested": int(len(ages_yr)),
        "metallicities": int(len(metallicities)),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a PARSEC-like MIST v1.2 Gaia/UBVRIplus isochrone grid.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--parsec-manifest", type=Path, default=DEFAULT_PARSEC_MANIFEST)
    parser.add_argument("--metallicity", type=float, action="append", default=None)
    parser.add_argument(
        "--min-age-myr",
        type=float,
        default=1.0,
        help=(
            "Minimum age to request. The MIST web interpolator failed for the 0.1-0.9 Myr "
            "custom list requests during validation, so the default is 1 Myr."
        ),
    )
    parser.add_argument("--max-ages-per-request", type=int, default=200)
    parser.add_argument("--version", choices=("MIST1", "MIST2"), default="MIST1")
    parser.add_argument("--rotation", choices=("vvcrit0.0", "vvcrit0.4"), default="vvcrit0.0")
    parser.add_argument("--output", default="UBVRIplus")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-requests", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = repo_root()
    output_dir = args.output_dir if args.output_dir.is_absolute() else base / args.output_dir
    parsec_manifest = args.parsec_manifest if args.parsec_manifest.is_absolute() else base / args.parsec_manifest
    manifest = json.loads(parsec_manifest.read_text(encoding="utf-8"))
    metallicities = args.metallicity or [float(value) for value in manifest["actual_counts"]["metallicity_values"]]
    build_grid(
        output_dir=output_dir,
        parsec_manifest=parsec_manifest,
        metallicities=metallicities,
        min_age_myr=float(args.min_age_myr),
        max_ages_per_request=int(args.max_ages_per_request),
        version=str(args.version),
        rotation=str(args.rotation),
        output=str(args.output),
        timeout=int(args.timeout),
        sleep_seconds=float(args.sleep_seconds),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        limit_requests=args.limit_requests,
    )


if __name__ == "__main__":
    main()
