"""Training-only operating points.

``select_cost_threshold`` sees out-of-fold training predictions only.
Evaluation on a test set is a separate function and is not allowed to choose
the threshold.
"""

from __future__ import annotations

import numpy as np


def select_cost_threshold(
    y_oof: np.ndarray,
    p_oof: np.ndarray,
    cost_fn: float,
    cost_fp: float,
) -> tuple[float, float]:
    """Pick a threshold from out-of-fold training labels and probabilities.

    Cost is ``cost_fn * FN + cost_fp * FP`` counted on the out-of-fold rows.
    Ties break toward the threshold closer to 0.5, then toward the smaller
    threshold. There is no parameter for test labels; passing them is a
    programming error at the call site, and the unit test locks the signature.
    """
    if cost_fn < 0 or cost_fp < 0:
        raise ValueError("Costs must be non-negative.")
    y_oof = np.asarray(y_oof, dtype=int)
    p_oof = np.asarray(p_oof, dtype=float)
    if y_oof.shape != p_oof.shape or y_oof.ndim != 1:
        raise ValueError("y_oof and p_oof must be 1-D and the same length.")
    if y_oof.size == 0:
        raise ValueError("Empty out-of-fold predictions.")

    candidates = np.unique(np.concatenate([p_oof, np.array([0.0, 0.5, 1.0])]))
    best_key: tuple[float, float, float] | None = None
    best_t = 0.5
    best_cost = float("inf")
    for t in candidates:
        y_hat = (p_oof >= t).astype(int)
        fn = int(np.sum((y_hat == 0) & (y_oof == 1)))
        fp = int(np.sum((y_hat == 1) & (y_oof == 0)))
        cost = float(cost_fn * fn + cost_fp * fp)
        key = (cost, abs(float(t) - 0.5), float(t))
        if best_key is None or key < best_key:
            best_key = key
            best_t = float(t)
            best_cost = cost
    return best_t, best_cost
