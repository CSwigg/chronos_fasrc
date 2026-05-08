from __future__ import annotations

import argparse
import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def configure_dustmaps(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DUSTMAPS_CONFIG_FNAME", str(data_dir / "dustmapsrc.json"))

    from dustmaps.config import config

    config["data_dir"] = str(data_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download optional fallback dust maps for Chronos.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root() / "inputs" / "dustmaps",
        help="Dustmaps data directory. Defaults to inputs/dustmaps under the repo.",
    )
    parser.add_argument("--skip-bayestar", action="store_true")
    parser.add_argument("--skip-decaps", action="store_true")
    parser.add_argument(
        "--decaps-samples",
        action="store_true",
        help="Download DECaPS mean+samples file. Default downloads the 7 GB mean-only file.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    configure_dustmaps(data_dir)

    if not args.skip_bayestar:
        from dustmaps.bayestar import fetch as fetch_bayestar

        fetch_bayestar(version="bayestar2019")

    if not args.skip_decaps:
        from dustmaps.decaps import fetch as fetch_decaps

        fetch_decaps(mean_only=not args.decaps_samples, silence_warnings=True)

    print(f"Dustmaps data directory: {data_dir}", flush=True)
    print(f"Bayestar 2019: {data_dir / 'bayestar' / 'bayestar2019.h5'}", flush=True)
    print(
        "DECaPS: "
        + str(data_dir / "decaps" / ("decaps_mean_and_samples.h5" if args.decaps_samples else "decaps_mean.h5")),
        flush=True,
    )


if __name__ == "__main__":
    main()
