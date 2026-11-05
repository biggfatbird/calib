"""The archived scripts regenerate the archived tables. They are not rewritten to do so."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "preserved" / "original_source"
PRESERVED = ROOT / "preserved" / "results"


def _run(tmp_path: Path) -> None:
    subprocess.check_call(
        [sys.executable, "reproduce_study.py", "--out", str(tmp_path)],
        cwd=SOURCE,
    )
    subprocess.check_call(
        [sys.executable, "noise_draws.py", "--draws", "50", "--out", str(tmp_path)],
        cwd=SOURCE,
    )


def test_original_scripts_match_preserved_numbers(tmp_path: Path):
    _run(tmp_path)
    specs = {
        "results_by_split.csv": ["split", "sigma", "model"],
        "results_summary.csv": ["model", "sigma"],
        "noise_draws.csv": ["split", "sigma", "draw", "model"],
        "noise_draw_summary.csv": ["model", "sigma", "metric"],
        "paired_differences.csv": ["sigma", "a", "b", "metric"],
    }
    for name, keys in specs.items():
        archived = pd.read_csv(PRESERVED / name)
        fresh = pd.read_csv(tmp_path / name)
        merged = archived.merge(fresh, on=keys, suffixes=("_archived", "_fresh"))
        assert len(merged) == len(archived) == len(fresh)
        for column in archived.columns:
            if column in keys or not np.issubdtype(archived[column].dtype, np.number):
                continue
            delta = (merged[f"{column}_archived"] - merged[f"{column}_fresh"]).abs()
            assert float(delta.max()) < 1e-12, f"{name}.{column} max abs {float(delta.max())}"


def test_one_draw_seed_is_not_draw_zero_of_the_fifty():
    single = pd.read_csv(PRESERVED / "results_by_split.csv")
    draws = pd.read_csv(PRESERVED / "noise_draws.csv")
    noisy = single[single.sigma > 0]
    merged = noisy.merge(draws, on=["split", "sigma", "model"], suffixes=("_one", "_draw"))
    gap = (merged["brier_one"] - merged["brier_draw"]).abs()
    assert float(gap.min()) > 1e-8
