"""Invariants of the archived CSVs. These files are not regenerated."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRESERVED = ROOT / "preserved" / "results"
GREATER_TOLERANCE = 1e-10


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preserved_checksums_unchanged():
    sums = (ROOT / "preserved" / "SHA256SUMS").read_text().strip().splitlines()
    for line in sums:
        digest, rel = line.split("  ", 1)
        assert _sha(ROOT / "preserved" / rel) == digest


def test_noise_draw_shape_and_sigma0_constant():
    df = pd.read_csv(PRESERVED / "noise_draws.csv")
    assert len(df) == 5 * 4 * 50 * 3
    assert sorted(df["split"].unique()) == [1, 2, 3, 4, 5]
    assert df["draw"].nunique() == 50
    assert set(np.round(df["sigma"].unique(), 2)) == {0.0, 0.25, 0.5, 1.0}
    assert set(df["model"]) == {"Logistic regression", "Random forest", "RBF SVM"}
    sigma0 = df[df.sigma == 0]
    nunique = sigma0.groupby(["split", "model"])[["roc_auc", "brier", "ece10", "sensitivity", "f1"]].nunique()
    assert int(nunique.max().max()) == 1


def test_summary_matches_by_split_with_sample_sd():
    by_split = pd.read_csv(PRESERVED / "results_by_split.csv")
    summary = pd.read_csv(PRESERVED / "results_summary.csv")
    assert len(by_split) == 60
    for metric in ["roc_auc", "brier", "ece10", "sensitivity", "f1", "accuracy"]:
        agg = by_split.groupby(["model", "sigma"])[metric].agg(["mean", "std"]).reset_index()
        merged = summary.merge(agg, on=["model", "sigma"])
        assert np.allclose(merged[f"{metric}_mean"], merged["mean"])
        assert np.allclose(merged[f"{metric}_std"], merged["std"])


def test_draw_summary_separates_draw_percentiles_and_split_sd():
    draws = pd.read_csv(PRESERVED / "noise_draws.csv")
    summary = pd.read_csv(PRESERVED / "noise_draw_summary.csv")
    per_draw = draws.groupby(["model", "sigma", "draw"])["roc_auc"].mean().reset_index()
    per_split = draws.groupby(["model", "sigma", "split"])["roc_auc"].mean().reset_index()
    for _, row in summary[summary.metric == "roc_auc"].iterrows():
        values = per_draw[(per_draw.model == row.model) & np.isclose(per_draw.sigma, row.sigma)]["roc_auc"]
        splits = per_split[(per_split.model == row.model) & np.isclose(per_split.sigma, row.sigma)]["roc_auc"]
        assert abs(values.mean() - row["mean"]) < 1e-12
        assert abs(values.std(ddof=1) - row.draw_sd) < 1e-12
        assert abs(np.quantile(values, 0.025) - row.draw_lo) < 1e-12
        assert abs(np.quantile(values, 0.975) - row.draw_hi) < 1e-12
        assert abs(splits.std(ddof=1) - row.split_sd) < 1e-12


def test_paired_differences_recomputed():
    draws = pd.read_csv(PRESERVED / "noise_draws.csv")
    paired = pd.read_csv(PRESERVED / "paired_differences.csv")
    metrics = ["roc_auc", "brier", "ece10", "sensitivity", "f1"]
    per_draw = draws.groupby(["sigma", "draw", "model"])[metrics].mean().reset_index()
    pairs = [
        ("RBF SVM", "Logistic regression"),
        ("Random forest", "Logistic regression"),
        ("RBF SVM", "Random forest"),
    ]
    for sigma, group in per_draw.groupby("sigma"):
        wide = {model: part.set_index("draw") for model, part in group.groupby("model")}
        for a, b in pairs:
            for metric in metrics:
                diff = wide[a][metric] - wide[b][metric]
                row = paired[
                    np.isclose(paired.sigma, sigma)
                    & (paired.a == a)
                    & (paired.b == b)
                    & (paired.metric == metric)
                ].iloc[0]
                assert abs(diff.mean() - row.mean_diff) < 1e-12
                assert abs(np.quantile(diff, 0.025) - row.lo) < 1e-12
                assert abs(np.quantile(diff, 0.975) - row.hi) < 1e-12
                frac = float(np.mean(diff.to_numpy() > GREATER_TOLERANCE))
                assert abs(frac - row.frac_a_greater) < 1e-12
