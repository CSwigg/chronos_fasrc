from __future__ import annotations

import argparse
import os


def run(config_path: str | None = None) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("MPLBACKEND", "Agg")
    from chronos.run_chronos.run_chronos_multiprocessing import main as run_chronos_main

    run_chronos_main(config_path=config_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chronos age fitting with the shared supernova-map config.")
    parser.add_argument("--config", type=str, default=None, help="Path to workflow TOML config.")
    args = parser.parse_args()
    run(config_path=args.config)


if __name__ == "__main__":
    main()
