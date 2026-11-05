"""Locks on the committed experiment outputs. These files are produced by wdbc_calib, not by hand."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_recalibration_shape_and_sigma0():
    df = pd.read_csv(ROOT / "results/recalibration/metrics_long.csv")
    assert len(df) == 5 * 4 * 50 * 3 * 3
    clean = df[df.sigma == 0]
    nunique = clean.groupby(["split", "model", "calibration"])["brier"].nunique()
    assert int(nunique.max()) == 1


def test_isotropic_corruption_matches_primary_baseline():
    primary = pd.read_csv(ROOT / "results/recalibration/summary_draws.csv")
    corr = pd.read_csv(ROOT / "results/corruptions/summary_draws.csv")
    left = primary[(primary.calibration == "baseline") & (primary.metric == "brier")][
        ["model", "sigma", "mean"]
    ]
    right = corr[(corr.corruption == "isotropic_gaussian") & (corr.metric == "brier")][
        ["model", "level", "mean"]
    ]
    merged = left.merge(right, left_on=["model", "sigma"], right_on=["model", "level"])
    assert len(merged) == 12
    assert np.allclose(merged.mean_x, merged.mean_y)


def test_extension_shapes_and_no_corruption_failures():
    assert len(pd.read_csv(ROOT / "results/more_splits/metrics_long.csv")) == 30 * 4 * 20 * 3
    assert len(pd.read_csv(ROOT / "results/corruptions/metrics_long.csv")) == 5 * 50 * 19 * 3
    assert len(pd.read_csv(ROOT / "results/thresholds/metrics_long.csv")) == 5 * 4 * 50 * 3 * 3
    failures = pd.read_csv(ROOT / "results/corruptions/failures.csv")
    assert len(failures) == 0
    chosen = pd.read_csv(ROOT / "results/thresholds/thresholds_train.csv")
    assert set(chosen.selection) == {"out-of-fold training predictions only"}


def test_cross_dataset_matches_wdbc_and_does_not_keep_its_signs():
    frame = pd.read_csv(ROOT / "results/cross_dataset/metrics_long.csv")
    assert len(frame) == 8 * 5 * 4 * 50 * 3 * 3
    agreement = pd.read_csv(ROOT / "results/cross_dataset/wdbc_agreement.csv")
    assert float(agreement["max_abs_diff"].max()) < 1e-12
    paired = pd.read_csv(ROOT / "results/cross_dataset/paired_calibration.csv")

    def row(dataset, model, arm, metric, sigma):
        hit = paired[
            (paired.dataset == dataset)
            & (paired.model == model)
            & (paired.arm_a == arm)
            & (paired.arm_b == "baseline")
            & (paired.metric == metric)
            & (paired.sigma.sub(sigma).abs() < 1e-9)
        ]
        assert len(hit) == 1
        return hit.iloc[0]

    wdbc_iso = row("wdbc", "Logistic regression", "isotonic", "brier", 1.0)
    assert wdbc_iso.mean_diff > 0 and wdbc_iso.frac_a_greater == 1
    sonar_iso = row("sonar", "Logistic regression", "isotonic", "brier", 1.0)
    assert sonar_iso.mean_diff < 0 and sonar_iso.frac_a_less == 1
    bank_rf = row("banknote", "Random forest", "sigmoid", "brier", 1.0)
    assert bank_rf.mean_diff > 0 and bank_rf.frac_a_greater == 1
    wdbc_rf = row("wdbc", "Random forest", "sigmoid", "brier", 1.0)
    assert wdbc_rf.mean_diff < 0 and wdbc_rf.frac_a_less == 1


def test_forest_sigmoid_brier_direction_is_the_saved_one():
    summary = pd.read_csv(ROOT / "results/recalibration/summary_draws.csv")
    row = summary[
        (summary.model == "Random forest")
        & (summary.calibration == "sigmoid")
        & (summary.metric == "brier")
        & (summary.sigma.sub(1).abs() < 1e-9)
    ].iloc[0]
    base = summary[
        (summary.model == "Random forest")
        & (summary.calibration == "baseline")
        & (summary.metric == "brier")
        & (summary.sigma.sub(1).abs() < 1e-9)
    ].iloc[0]
    assert row["mean"] < base["mean"]
    assert 0.09 < row["mean"] < 0.10
