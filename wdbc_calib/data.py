"""Wisconsin Diagnostic Breast Cancer data, malignant coded as the positive class."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer

N_ROWS = 569
N_FEATURES = 30
N_MALIGNANT = 212
N_BENIGN = 357


def load_wdbc() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X, y, feature names.

    scikit-learn codes malignant as 0 and benign as 1. This study codes
    malignant as 1 so sensitivity is sensitivity for malignancy.
    """
    bunch = load_breast_cancer()
    if bunch.data.shape != (N_ROWS, N_FEATURES):
        raise RuntimeError(
            f"Unexpected WDBC shape {bunch.data.shape}; expected {(N_ROWS, N_FEATURES)}."
        )
    if list(bunch.target_names) != ["malignant", "benign"]:
        raise RuntimeError(f"Unexpected target names: {bunch.target_names}")
    y = (bunch.target == 0).astype(np.int8)
    if int(y.sum()) != N_MALIGNANT:
        raise RuntimeError(f"Expected {N_MALIGNANT} malignant rows, found {int(y.sum())}.")
    return bunch.data.astype(np.float64), y, np.asarray(bunch.feature_names)


def worst_feature_mask(feature_names: np.ndarray) -> np.ndarray:
    """Boolean mask for the ten 'worst' nucleus features."""
    mask = np.array([str(name).startswith("worst ") for name in feature_names])
    if int(mask.sum()) != 10:
        raise RuntimeError(f"Expected 10 worst features, found {int(mask.sum())}.")
    return mask
