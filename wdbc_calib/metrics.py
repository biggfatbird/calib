"""Discrimination, operating-point, and calibration metrics.

ECE here is the binned L1 calibration error. Equal-width bins use edges
``linspace(0, 1, n_bins + 1)``. Equal-mass bins use empirical quantiles of the
predicted probabilities. Empty bins contribute nothing. These are descriptive
binning choices, not confidence intervals.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "equal_width",
) -> tuple[float, list[dict]]:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape or y_true.ndim != 1:
        raise ValueError("y_true and y_prob must be 1-D and the same length.")
    if n_bins < 1:
        raise ValueError("n_bins must be positive.")
    if np.any((y_prob < -1e-12) | (y_prob > 1 + 1e-12)):
        raise ValueError("Probabilities must lie in [0, 1].")
    y_prob = np.clip(y_prob, 0.0, 1.0)
    n = int(y_true.size)
    if n == 0:
        raise ValueError("Empty inputs.")

    if strategy == "equal_width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "equal_mass":
        edges = np.quantile(y_prob, np.linspace(0.0, 1.0, n_bins + 1))
        edges = np.asarray(edges, dtype=float)
        # Quantile ties would create empty or inverted bins. Nudge duplicates
        # forward so digitize has non-decreasing edges; leftover ties stay empty.
        for i in range(1, edges.size):
            if edges[i] <= edges[i - 1]:
                edges[i] = np.nextafter(edges[i - 1], np.inf)
        edges[0] = min(float(edges[0]), float(y_prob.min()))
        edges[-1] = max(float(edges[-1]), float(y_prob.max()))
    else:
        raise ValueError(f"Unknown ECE strategy: {strategy}")

    bin_ids = np.digitize(y_prob, edges[1:-1], right=False)
    rows: list[dict] = []
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin": b,
                    "count": 0,
                    "mean_predicted": np.nan,
                    "observed_frequency": np.nan,
                    "gap": np.nan,
                    "edge_left": float(edges[b]),
                    "edge_right": float(edges[b + 1]),
                }
            )
            continue
        mean_predicted = float(y_prob[mask].mean())
        observed = float(y_true[mask].mean())
        gap = abs(observed - mean_predicted)
        ece += (count / n) * gap
        rows.append(
            {
                "bin": b,
                "count": count,
                "mean_predicted": mean_predicted,
                "observed_frequency": observed,
                "gap": gap,
                "edge_left": float(edges[b]),
                "edge_right": float(edges[b + 1]),
            }
        )
    return float(ece), rows


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Metrics for malignant = 1. The threshold is applied to P(malignant)."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape or y_true.ndim != 1:
        raise ValueError("y_true and y_prob must be 1-D and the same length.")
    y_hat = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_hat == 1) & (y_true == 1)))
    tn = int(np.sum((y_hat == 0) & (y_true == 0)))
    fp = int(np.sum((y_hat == 1) & (y_true == 0)))
    fn = int(np.sum((y_hat == 0) & (y_true == 1)))
    n = int(y_true.size)
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    if precision + sensitivity == 0 or np.isnan(sensitivity):
        f1 = 0.0 if not np.isnan(sensitivity) else float("nan")
    else:
        f1 = 2 * precision * sensitivity / (precision + sensitivity)
    if len(np.unique(y_true)) < 2:
        roc_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    ece10, _ = expected_calibration_error(y_true, y_prob, n_bins=10, strategy="equal_width")
    ece15, _ = expected_calibration_error(y_true, y_prob, n_bins=15, strategy="equal_width")
    ece_mass, _ = expected_calibration_error(y_true, y_prob, n_bins=10, strategy="equal_mass")
    return {
        "accuracy": (tp + tn) / n,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "roc_auc": roc_auc,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "brier": float(np.mean((y_prob - y_true) ** 2)),
        "ece10": ece10,
        "ece15": ece15,
        "ece_equal_mass10": ece_mass,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def expected_cost(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, cost_fn: float, cost_fp: float) -> float:
    """Mean per-row cost. False negatives and false positives use the supplied costs."""
    y_true = np.asarray(y_true, dtype=int)
    y_hat = (np.asarray(y_prob) >= threshold).astype(int)
    fn = (y_hat == 0) & (y_true == 1)
    fp = (y_hat == 1) & (y_true == 0)
    return float(np.mean(cost_fn * fn + cost_fp * fp))
