"""Command line entry points. See configs/ for the seeds actually used."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wdbc_calib.cross_dataset import run_cross_dataset_experiment
from wdbc_calib.experiments import (
    compare_to_preserved,
    run_calibration_experiment,
    run_corruption_experiment,
    run_more_splits,
    run_threshold_experiment,
)
from wdbc_calib.make_figures import render_all


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="WDBC calibration and noise experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("recalibration")
    rec.add_argument("--config", default="configs/recalibration.json")

    more = sub.add_parser("more-splits")
    more.add_argument("--config", default="configs/more_splits.json")

    corr = sub.add_parser("corruptions")
    corr.add_argument("--config", default="configs/corruptions.json")

    thr = sub.add_parser("thresholds")
    thr.add_argument("--config", default="configs/thresholds.json")

    cross = sub.add_parser("cross-dataset")
    cross.add_argument("--config", default="configs/cross_dataset.json")

    sub.add_parser("figures")

    comp = sub.add_parser("compare-preserved")
    comp.add_argument("--summary", default="results/recalibration/summary_draws.csv")
    comp.add_argument("--output", default="results/recalibration/comparison_vs_preserved.csv")

    args = parser.parse_args()
    root = Path.cwd()
    if args.command == "recalibration":
        config = _load(args.config)
        run_calibration_experiment(config, root / config["output_dir"])
    elif args.command == "more-splits":
        config = _load(args.config)
        run_more_splits(config, root / config["output_dir"])
    elif args.command == "corruptions":
        config = _load(args.config)
        run_corruption_experiment(config, root / config["output_dir"])
    elif args.command == "thresholds":
        config = _load(args.config)
        run_threshold_experiment(config, root / config["output_dir"])
    elif args.command == "cross-dataset":
        config = _load(args.config)
        run_cross_dataset_experiment(config, root / config["output_dir"])
    elif args.command == "figures":
        written = render_all(root)
        print("\n".join(str(path) for path in written))
    elif args.command == "compare-preserved":
        import pandas as pd

        summary = pd.read_csv(root / args.summary)
        compare_to_preserved(
            summary,
            root / "preserved/results/results_summary.csv",
            root / "preserved/results/noise_draw_summary.csv",
            root / args.output,
        )
    else:
        raise SystemExit(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
