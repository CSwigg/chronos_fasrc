from __future__ import annotations

import argparse
import os

from chronos.run_chronos.dual_model import DualModelRunConfig, run_dual_model_refit


def run(
    *,
    config_path: str | None = None,
    n_processes: int | None = None,
    force: bool = False,
    clusters: list[str] | None = None,
    models: tuple[str, ...] = ("parsec", "baraffe"),
    output_dirname: str = "dual_model_refit",
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("MPLBACKEND", "Agg")
    output_path = run_dual_model_refit(
        config_path=config_path,
        n_processes=n_processes,
        force=force,
        clusters=clusters,
        run_config=DualModelRunConfig(
            model_names=tuple(models),
            output_dirname=output_dirname,
        ),
    )
    print(f"Dual-model Chronos results: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run dual-model Chronos age fitting for Hunt clusters with resumable checkpoints."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to workflow TOML config.")
    parser.add_argument(
        "--n-processes",
        type=int,
        default=None,
        help="Number of worker processes. Defaults to all logical CPU cores.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore completed checkpoints and rerun all selected clusters.",
    )
    parser.add_argument(
        "--clusters",
        nargs="*",
        default=None,
        help="Optional subset of cluster names to run.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=("parsec", "baraffe"),
        help="Isochrone models to fit, e.g. parsec baraffe mist.",
    )
    parser.add_argument(
        "--output-dirname",
        type=str,
        default="dual_model_refit",
        help="Subdirectory under outputs/chronos for this run.",
    )
    args = parser.parse_args()
    run(
        config_path=args.config,
        n_processes=args.n_processes,
        force=args.force,
        clusters=args.clusters,
        models=tuple(args.models),
        output_dirname=args.output_dirname,
    )


if __name__ == "__main__":
    main()
