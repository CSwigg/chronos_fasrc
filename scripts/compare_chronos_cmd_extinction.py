#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chronos.isochrone.ICBase import ICBase, corr_BPmag, corr_Gmag, corr_RPmag
from chronos.isochrone.PARSEC import PARSEC
from scripts.download_parsec_isochrones import (
    DEFAULT_PHOTSYS_FILE,
    DEFAULT_PHOTSYS_VERSION,
    AgeSegment,
    MetallicityGroup,
    _base_cmd_params,
    _download_dat,
    _request_params,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_cmd_dat(path: Path) -> pd.DataFrame:
    header: list[str] | None = None
    with path.open() as handle:
        for line in handle:
            if line.startswith("# Zini"):
                header = line.replace("#", " ", 1).split()
                break
    if header is None:
        raise ValueError(f"Could not find CMD header in {path}")
    frame = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    frame.columns = header
    return frame


def select_isochrone(frame: pd.DataFrame, *, log_age: float, mh: float) -> pd.DataFrame:
    mask = np.isclose(frame["logAge"], log_age, atol=2e-5) & np.isclose(frame["MH"], mh, atol=2e-5)
    selected = frame.loc[mask].copy()
    if selected.empty:
        raise ValueError(f"No isochrone found at logAge={log_age}, MH={mh}")
    return selected.sort_values("Mini").reset_index(drop=True)


def download_cmd_extincted(
    *,
    output_path: Path,
    age_yr: float,
    mh: float,
    av_mag: float,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    params = _request_params(
        _base_cmd_params(session),
        age_segment=AgeSegment("single", False, age_yr, None, 0.0, 1),
        metallicity_group=MetallicityGroup("single", mh, mh, 0.0, 1),
        photsys_file=DEFAULT_PHOTSYS_FILE,
        photsys_version=DEFAULT_PHOTSYS_VERSION,
    )
    params["extinction_av"] = f"{av_mag:.8g}"
    _download_dat(session, params=params, output_path=output_path, retries=4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Chronos runtime extinction to CMD extincted PARSEC output.")
    parser.add_argument("--parsec-dir", type=Path, default=repo_root() / "inputs/parsec_isochrones_hybrid_0p1myr_to13gyr")
    parser.add_argument("--age-myr", type=float, default=100.0)
    parser.add_argument("--mh", type=float, default=0.1)
    parser.add_argument("--av-mag", type=float, default=2.0)
    parser.add_argument("--overwrite-download", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "runs/current/analysis/extinction_validation",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    age_yr = float(args.age_myr) * 1e6
    log_age = float(np.log10(age_yr))
    mh = float(args.mh)
    av_mag = float(args.av_mag)

    local_file = args.parsec_dir / "parsec_cmd37_young_linear_1to300myr_1myr_mh_p0p1.dat"
    local_unreddened = select_isochrone(read_cmd_dat(local_file), log_age=log_age, mh=mh)

    cmd_path = output_dir / f"cmd37_parsec_age{args.age_myr:g}myr_mh{mh:+.1f}_av{av_mag:g}.dat"
    download_cmd_extincted(
        output_path=cmd_path,
        age_yr=age_yr,
        mh=mh,
        av_mag=av_mag,
        overwrite=args.overwrite_download,
    )
    cmd_extincted = select_isochrone(read_cmd_dat(cmd_path), log_age=log_age, mh=mh)

    helper = ICBase()
    local_color = local_unreddened["G_BPmag"].to_numpy() - local_unreddened["G_RPmag"].to_numpy()
    local_g = local_unreddened["Gmag"].to_numpy()
    chronos_color, chronos_g = helper.apply_extinction_by_color(
        local_g,
        local_color,
        av_mag,
        helper.bp_rp,
    )

    cmd_color = cmd_extincted["G_BPmag"].to_numpy() - cmd_extincted["G_RPmag"].to_numpy()
    cmd_g = cmd_extincted["Gmag"].to_numpy()
    local_mini = local_unreddened["Mini"].to_numpy()
    cmd_mini = cmd_extincted["Mini"].to_numpy()
    row_aligned = (
        len(local_unreddened) == len(cmd_extincted)
        and np.allclose(local_mini, cmd_mini, rtol=0.0, atol=1e-10)
        and np.allclose(local_unreddened["Mass"].to_numpy(), cmd_extincted["Mass"].to_numpy(), rtol=0.0, atol=1e-10)
    )
    if row_aligned:
        cmd_color_compare = cmd_color
        cmd_g_compare = cmd_g
    else:
        order = np.argsort(cmd_mini)
        cmd_color_compare = np.interp(local_mini, cmd_mini[order], cmd_color[order])
        cmd_g_compare = np.interp(local_mini, cmd_mini[order], cmd_g[order])
    delta_color = chronos_color - cmd_color_compare
    delta_g = chronos_g - cmd_g_compare
    cmd_delta_color = cmd_color_compare - local_color
    cmd_delta_g = cmd_g_compare - local_g

    parsec = PARSEC(str(args.parsec_dir), file_ending="dat")
    chronos_model = parsec.model(logAge=log_age, feh=mh, A_V=av_mag, bp_rp=True)

    summary = {
        "age_myr": args.age_myr,
        "log_age": log_age,
        "mh": mh,
        "av_mag": av_mag,
        "n_local_points": int(len(local_unreddened)),
        "n_cmd_points": int(len(cmd_extincted)),
        "row_aligned": bool(row_aligned),
        "chronos_coefficients": {
            "A_G_over_Av": corr_Gmag,
            "A_BP_over_Av": corr_BPmag,
            "A_RP_over_Av": corr_RPmag,
            "E_BP_RP_over_Av": corr_BPmag - corr_RPmag,
        },
        "cmd_effective_coefficients": {
            "median_A_G_over_Av": float(np.nanmedian(cmd_delta_g) / av_mag),
            "median_E_BP_RP_over_Av": float(np.nanmedian(cmd_delta_color) / av_mag),
        },
        "median_delta_bp_rp_mag": float(np.nanmedian(delta_color)),
        "median_delta_g_mag": float(np.nanmedian(delta_g)),
        "max_abs_delta_bp_rp_mag": float(np.nanmax(np.abs(delta_color))),
        "max_abs_delta_g_mag": float(np.nanmax(np.abs(delta_g))),
    }
    summary_path = output_dir / "chronos_vs_cmd_extinction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    plot_path = output_dir / "chronos_vs_cmd_extinction_av2.png"
    fig, (ax_cmd, ax_resid) = plt.subplots(
        1,
        2,
        figsize=(11.0, 5.0),
        gridspec_kw={"width_ratios": [1.1, 1.0]},
        constrained_layout=True,
    )
    ax_cmd.scatter(cmd_color, cmd_g, s=12, color="#d55e00", alpha=0.65, label="CMD website, A_V=2")
    ax_cmd.plot(chronos_color, chronos_g, color="black", lw=1.5, label="Chronos coeff shift, raw points")
    ax_cmd.plot(
        chronos_model[:, 0],
        chronos_model[:, 1],
        color="#0072b2",
        lw=1.0,
        alpha=0.8,
        label="Chronos interpolated model",
    )
    ax_cmd.set_title(f"PARSEC {args.age_myr:g} Myr, [M/H]={mh:+.1f}, A_V={av_mag:g}")
    ax_cmd.set_xlabel("BP - RP")
    ax_cmd.set_ylabel("G")
    ax_cmd.invert_yaxis()
    ax_cmd.legend(loc="best", fontsize=8)

    ax_resid.axhline(0.0, color="0.2", lw=1)
    ax_resid.plot(local_mini, delta_color, color="#009e73", lw=1.2, label="Delta BP-RP")
    ax_resid.plot(local_mini, delta_g, color="#cc79a7", lw=1.2, label="Delta G")
    ax_resid.set_xscale("log")
    ax_resid.set_xlabel("Initial mass / Msun")
    ax_resid.set_ylabel("Chronos shifted - CMD extincted (mag)")
    ax_resid.set_title("Residuals at matching initial mass")
    ax_resid.legend(loc="best", fontsize=8)
    ax_resid.grid(alpha=0.25)
    fig.savefig(plot_path, dpi=180)
    print(json.dumps({"summary": str(summary_path), "plot": str(plot_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
