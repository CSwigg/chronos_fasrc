from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chronos.run_chronos.dual_model import _flatten_result, _load_existing_results
from workflows.config import load_runtime_paths


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def finalize_run(*, config_path: str | Path | None, output_dirname: str) -> Path:
    paths = load_runtime_paths(config_path)
    output_root = paths.outputs.chronos_dir / output_dirname
    results = _load_existing_results(output_root)
    rows = [_flatten_result(results[name]) for name in sorted(results)]

    import pandas as pd

    summary_csv = output_root / "cluster_results.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = summary_csv.with_suffix(summary_csv.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp_csv, index=False)
    tmp_csv.replace(summary_csv)

    model_names = sorted(
        {
            model_name
            for payload in results.values()
            for model_name in payload.get("model_names", [])
        }
    )
    status_counts: dict[str, dict[str, int]] = {}
    for model_name in model_names:
        counts: dict[str, int] = {}
        for payload in results.values():
            status = str(payload.get(model_name, {}).get("status", "missing"))
            counts[status] = counts.get(status, 0) + 1
        status_counts[model_name] = counts

    payload: dict[str, Any] = {
        "output_dirname": output_dirname,
        "output_root": str(output_root),
        "checkpoint_count": int(len(results)),
        "cluster_results_csv": str(summary_csv),
        "model_status_counts": status_counts,
    }
    _atomic_write_text(output_root / "finalize_summary.json", json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    return summary_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild Chronos cluster_results.csv from checkpoint JSON files.")
    parser.add_argument("--config", type=str, default="configs/paths.toml")
    parser.add_argument("--output-dirname", required=True)
    args = parser.parse_args()
    finalize_run(config_path=args.config, output_dirname=args.output_dirname)


if __name__ == "__main__":
    main()
