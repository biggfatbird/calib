"""Hand-computed metric checks."""

import numpy as np

from wdbc_calib.metrics import classification_metrics, expected_calibration_error, expected_cost


def test_equal_width_ece_two_bins():
    y = np.array([0, 0, 1, 1], dtype=float)
    p = np.array([0.1, 0.2, 0.8, 0.9])
    ece, rows = expected_calibration_error(y, p, n_bins=2, strategy="equal_width")
    assert rows[0]["count"] == 2
    assert rows[1]["count"] == 2
    assert abs(ece - 0.15) < 1e-12


def test_equal_width_and_equal_mass_differ():
    y = np.array([0, 0, 1, 1], dtype=float)
    p = np.array([0.1, 0.2, 0.3, 0.9])
    width, _ = expected_calibration_error(y, p, n_bins=2, strategy="equal_width")
    mass, _ = expected_calibration_error(y, p, n_bins=2, strategy="equal_mass")
    # width: (3/4)*|1/3 - 0.2| + (1/4)*|1 - 0.9| = 0.125
    assert abs(width - 0.125) < 1e-12
    # mass splits after the median 0.25: gaps 0.15 and 0.40
    assert abs(mass - 0.275) < 1e-12
    assert abs(width - mass) > 0.1


def test_confusion_and_brier():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.6, 0.4, 0.1])
    m = classification_metrics(y, p, threshold=0.5)
    assert m["tp"] == 2 and m["tn"] == 2 and m["fp"] == 0 and m["fn"] == 0
    assert m["sensitivity"] == 1
    assert m["specificity"] == 1
    assert abs(m["brier"] - np.mean((p - y) ** 2)) < 1e-15
    assert m["roc_auc"] == 1


def test_threshold_changes_sensitivity_only():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.8, 0.7, 0.4, 0.2])
    low = classification_metrics(y, p, threshold=0.5)
    high = classification_metrics(y, p, threshold=0.75)
    assert low["tp"] == 1 and low["fp"] == 1
    assert high["tp"] == 1 and high["fp"] == 0
    assert low["roc_auc"] == high["roc_auc"]
    assert abs(expected_cost(y, p, 0.75, cost_fn=5, cost_fp=1) - (5 * 1 + 1 * 0) / 4) < 1e-12
