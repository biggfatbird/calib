"""Summaries that keep noise-draw variation and split variation separate.

Percentile intervals are empirical percentiles of a named unit (a draw, after
averaging splits, or a split). They are not bootstrap confidence intervals and
they are not intervals for a new patient drawn from another dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

GREATER_TOLERANCE = 1e-10
METRIC_COLUMNS = (
    "roc_auc",
    "brier",
    "ece10",
    "ece15",
    "ece_equal_mass10",
    "sensitivity",
    "specificity",
    "f1",
    "accuracy",
    "balanced_accuracy",
)


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q, method="linear"))


def summarize_draws(df: pd.DataFrame, group_cols: list[str], metrics: tuple[str, ...] = METRIC_COLUMNS) -> pd.DataFrame:
    """One row per group.

    ``mean`` is the mean over draws of the within-draw split-average.
    ``draw_sd`` is the sample SD over draws (ddof=1).
    ``draw_lo`` / ``draw_hi`` are the 2.5 and 97.5 percent linear quantiles over draws.
    ``split_sd`` is the sample SD, over splits, of each split's mean over draws.
    """
    present = [m for m in metrics if m in df.columns]
    rows = []
    for keys, group in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        per_draw = group.groupby("draw", sort=True)[present].mean()
        per_split = group.groupby("split", sort=True)[present].mean()
        for metric in present:
            draw_values = per_draw[metric].to_numpy()
            split_values = per_split[metric].to_numpy()
            record = dict(zip(group_cols, keys))
            record.update(
                {
                    "metric": metric,
                    "n_draws": int(per_draw.shape[0]),
                    "n_splits": int(per_split.shape[0]),
                    "mean": float(np.mean(draw_values)),
                    "draw_sd": float(np.std(draw_values, ddof=1)) if len(draw_values) > 1 else 0.0,
                    "draw_lo": _quantile(draw_values, 0.025),
                    "draw_hi": _quantile(draw_values, 0.975),
                    "split_sd": float(np.std(split_values, ddof=1)) if len(split_values) > 1 else 0.0,
                }
            )
            rows.append(record)
    return pd.DataFrame(rows)


def paired_draw_summary(
    df: pd.DataFrame,
    arm_col: str,
    arm_a: str,
    arm_b: str,
    group_cols: list[str],
    metrics: tuple[str, ...] = METRIC_COLUMNS,
) -> pd.DataFrame:
    """Paired difference arm_a minus arm_b.

    Within each draw the metric is averaged over splits, the arms are
    subtracted, and that draw-level difference is summarized. ``frac_a_greater``
    ignores absolute differences below 1e-10 so binary-metric ties are not
    broken by floating-point dust.
    """
    present = [m for m in metrics if m in df.columns]
    rows = []
    identity = group_cols
    left = df[df[arm_col] == arm_a]
    right = df[df[arm_col] == arm_b]
    keys = ["split", "draw", *identity]
    merged = left.merge(right, on=keys, suffixes=("_a", "_b"))
    if merged.empty:
        return pd.DataFrame()
    for metric in present:
        merged[f"diff_{metric}"] = merged[f"{metric}_a"] - merged[f"{metric}_b"]
    for keys_t, group in merged.groupby(identity, sort=True):
        if not isinstance(keys_t, tuple):
            keys_t = (keys_t,)
        for metric in present:
            per_draw = group.groupby("draw")[f"diff_{metric}"].mean()
            values = per_draw.to_numpy()
            record = dict(zip(identity, keys_t))
            record.update(
                {
                    "arm_a": arm_a,
                    "arm_b": arm_b,
                    "metric": metric,
                    "mean_diff": float(np.mean(values)),
                    "draw_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "draw_lo": _quantile(values, 0.025),
                    "draw_hi": _quantile(values, 0.975),
                    "frac_a_greater": float(np.mean(values > GREATER_TOLERANCE)),
                    "frac_a_less": float(np.mean(values < -GREATER_TOLERANCE)),
                    "n_draws": int(values.size),
                }
            )
            rows.append(record)
    return pd.DataFrame(rows)


def nadeau_bengio_split_test(
    split_differences: np.ndarray,
    n_train: int,
    n_test: int,
) -> dict[str, float]:
    """Corrected resampled t-test on split-level paired differences.

    Variance estimate from Nadeau and Bengio (2003):
    ``(1/n + n_test/n_train) * s^2`` with ``s^2`` the sample variance of the
    split-level differences. With overlapping repeated holdouts and a handful
    of splits this is an approximate description, not a licence for a clinical
    claim. Multiplicity is not adjusted here.
    """
    values = np.asarray(split_differences, dtype=float)
    n = int(values.size)
    if n < 2:
        raise ValueError("Need at least two splits.")
    mean = float(np.mean(values))
    variance = float(np.var(values, ddof=1))
    corrected = (1.0 / n + (n_test / n_train)) * variance
    se = float(np.sqrt(corrected))
    if se == 0:
        t_stat = float("inf") if mean > 0 else float("-inf") if mean < 0 else 0.0
        p_value = 1.0 if mean == 0 else 0.0
    else:
        t_stat = mean / se
        p_value = float(2 * scipy_stats.t.sf(abs(t_stat), df=n - 1))
    return {
        "n_splits": n,
        "mean_diff": mean,
        "split_sd": float(np.sqrt(variance)),
        "corrected_se": se,
        "t": float(t_stat),
        "df": n - 1,
        "two_sided_p": p_value,
        "n_train": int(n_train),
        "n_test": int(n_test),
    }


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values, in the original order.

    For ordered raw p-values p_(1) <= ... <= p_(m), the adjusted value at rank
    i (starting at 1) is (m - i + 1) * p_(i), capped at 1, then forced to be
    nondecreasing in the sorted order. The unit of each p-value is whatever
    comparison produced it. This function does not change that unit.
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be one-dimensional.")
    if np.any((p < 0) | (p > 1) | ~np.isfinite(p)):
        raise ValueError("p-values must lie in [0, 1].")
    m = int(p.size)
    adjusted = np.empty(m, dtype=float)
    if m == 0:
        return adjusted
    order = np.argsort(p, kind="mergesort")
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * float(p[idx]))
        running = max(running, value)
        adjusted[idx] = running
    return adjusted
