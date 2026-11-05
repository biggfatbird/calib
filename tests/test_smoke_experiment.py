"""A tiny end-to-end run. This is not the study result."""

import numpy as np
import pandas as pd

from wdbc_calib.experiments import run_calibration_experiment


def test_smoke_sigma0_identical_and_deterministic(tmp_path):
    config = {
        "master_seed": 20260311,
        "n_splits": 1,
        "test_size": 0.25,
        "sigmas": [0.0, 1.0],
        "n_draws": 3,
        "models": ["logistic_regression", "rbf_svm"],
        "calibrations": ["baseline", "sigmoid", "isotonic"],
        "output_dir": "unused",
    }
    first = tmp_path / "a"
    second = tmp_path / "b"
    run_calibration_experiment(config, first)
    run_calibration_experiment(config, second)
    left = pd.read_csv(first / "metrics_long.csv")
    right = pd.read_csv(second / "metrics_long.csv")
    assert np.allclose(left.select_dtypes(include="number"), right.select_dtypes(include="number"))
    clean = left[left.sigma == 0]
    for (_, _), part in clean.groupby(["model", "calibration"]):
        for column in ["roc_auc", "brier", "ece10", "sensitivity"]:
            assert part[column].nunique() == 1
    # Shared-noise consequence: both models were scored on the same draw index.
    assert set(left.draw) == {0, 1, 2}
    assert (first / "manifest.json").exists()
    assert "preserved" not in str(first)
