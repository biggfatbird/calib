"""Dataset-level description of when recalibration changes Brier score.

The unit is the dataset. Eight datasets are the whole sample. Noise draws are
used only to build the within-dataset percentile interval of a Brier difference
that has already been averaged over splits. They are not eight times fifty
observations, and they are not resampled in the Spearman bootstrap.

No regression with more than one predictor is fit. Spearman correlations are
exploratory. Their permutation p-values are Holm-adjusted inside the family
declared in ``FAMILY_NOTE``. A confidence interval that includes both signs is
not a discovered predictor.

Holm adjustment of the existing split-level tests is a separate family. It does
not use the Spearman tests, and the Spearman tests do not use the split-level
p-values.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from wdbc_calib.stats import holm_adjust

FAMILY_NOTE = (
    "Spearman family: 6 existing Brier contrasts (sigmoid and isotonic minus baseline, "
    "for logistic regression, random forest, and RBF SVM) crossed with 6 characteristics. "
    "36 two-sided permutation tests. Holm is applied inside this family only. "
    "The dataset is the unit (n=8)."
)
SPLIT_FAMILY_NOTE = (
    "Two Holm families, adjusted separately. "
    "Cross-dataset: the 64 pre-specified Nadeau-Bengio tests "
    "(logistic isotonic Brier, logistic isotonic ROC-AUC, logistic sigmoid Brier, "
    "forest sigmoid Brier; sigma 0 and 1; eight datasets). "
    "WDBC: the 48 Brier and 10-bin ECE tests "
    "(sigmoid and isotonic minus baseline; three models; four sigmas)."
)

CHARACTERISTICS = (
    ("n_rows", "Rows", "table design"),
    ("n_features", "Features", "table design"),
    ("minority_rate", "Smaller-class rate", "table design"),
    ("baseline_ece10_sigma0", "Baseline ECE-10 at sigma 0", "estimated on the same splits"),
    ("baseline_roc_auc_sigma0", "Baseline ROC-AUC at sigma 0", "estimated on the same splits"),
    ("noise_brier_increase", "Baseline Brier increase, sigma 1 minus 0", "co-outcome of the same run"),
)

BRIER_CONTRASTS = (
    ("Logistic regression", "sigmoid"),
    ("Logistic regression", "isotonic"),
    ("Random forest", "sigmoid"),
    ("Random forest", "isotonic"),
    ("RBF SVM", "sigmoid"),
    ("RBF SVM", "isotonic"),
)

CROSS_SPLIT_CONTRASTS = (
    ("brier", "Logistic regression", "isotonic"),
    ("roc_auc", "Logistic regression", "isotonic"),
    ("brier", "Random forest", "sigmoid"),
    ("brier", "Logistic regression", "sigmoid"),
)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    rho = spearmanr(x, y).statistic
    return float(rho)


def dataset_bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    """Percentile interval from resampling datasets with replacement."""
    n = int(x.size)
    rhos = []
    for _ in range(n_boot):
        take = rng.integers(0, n, size=n)
        rho = _spearman(x[take], y[take])
        if np.isfinite(rho):
            rhos.append(rho)
    if len(rhos) < 2:
        return float("nan"), float("nan"), len(rhos)
    return (
        float(np.quantile(rhos, 0.025)),
        float(np.quantile(rhos, 0.975)),
        len(rhos),
    )


def permutation_spearman_p(
    x: np.ndarray,
    y: np.ndarray,
    observed: float,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    """Two-sided permutation p with the add-one correction. Datasets are permuted."""
    if not np.isfinite(observed):
        return float("nan")
    n = int(y.size)
    hits = 0
    for _ in range(n_perm):
        rho = _spearman(x, rng.permutation(y))
        if np.isfinite(rho) and abs(rho) >= abs(observed) - 1e-15:
            hits += 1
    return float((1 + hits) / (n_perm + 1))


def leave_one_out_range(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rhos = []
    for i in range(x.size):
        mask = np.ones(x.size, dtype=bool)
        mask[i] = False
        rho = _spearman(x[mask], y[mask])
        if np.isfinite(rho):
            rhos.append(rho)
    if not rhos:
        return float("nan"), float("nan")
    return float(min(rhos)), float(max(rhos))


def build_characteristic_table(card: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, info in card.set_index("dataset").iterrows():
        minority = float(min(info.positive_rate, 1.0 - info.positive_rate))
        for model in sorted(summary.model.unique()):
            base = summary[
                (summary.dataset == dataset)
                & (summary.model == model)
                & (summary.calibration == "baseline")
            ]
            def _mean(metric: str, sigma: float) -> float:
                hit = base[(base.metric == metric) & (base.sigma.sub(sigma).abs() < 1e-9)]
                if len(hit) != 1:
                    raise RuntimeError(f"Missing baseline {dataset} {model} {metric} {sigma}")
                return float(hit.iloc[0]["mean"])

            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "n_rows": int(info.n_rows),
                    "n_features": int(info.n_features),
                    "n_positive": int(info.n_positive),
                    "positive_rate": float(info.positive_rate),
                    "minority_rate": minority,
                    "baseline_ece10_sigma0": _mean("ece10", 0.0),
                    "baseline_roc_auc_sigma0": _mean("roc_auc", 0.0),
                    "baseline_brier_sigma0": _mean("brier", 0.0),
                    "baseline_brier_sigma1": _mean("brier", 1.0),
                    "noise_brier_increase": _mean("brier", 1.0) - _mean("brier", 0.0),
                }
            )
    return pd.DataFrame(rows)


def association_table(
    paired: pd.DataFrame,
    characteristics: pd.DataFrame,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for model, arm in BRIER_CONTRASTS:
        part = paired[
            (paired.model == model)
            & (paired.arm_a == arm)
            & (paired.arm_b == "baseline")
            & (paired.metric == "brier")
            & (paired.sigma.sub(1.0).abs() < 1e-9)
        ].copy()
        if part.dataset.nunique() != 8 or len(part) != 8:
            raise RuntimeError(f"Expected 8 dataset rows for {model} {arm}, found {len(part)}.")
        merged = part.merge(characteristics[characteristics.model == model], on=["dataset", "model"])
        if len(merged) != 8:
            raise RuntimeError(f"Characteristic merge failed for {model}.")
        y = merged["mean_diff"].to_numpy(float)
        for key, label, role in CHARACTERISTICS:
            x = merged[key].to_numpy(float)
            rho = _spearman(x, y)
            lo, hi, n_used = dataset_bootstrap_spearman(x, y, n_boot, rng)
            p_value = permutation_spearman_p(x, y, rho, n_perm, rng)
            loo_min, loo_max = leave_one_out_range(x, y)
            rows.append(
                {
                    "model": model,
                    "arm_a": arm,
                    "arm_b": "baseline",
                    "sigma": 1.0,
                    "metric": "brier",
                    "characteristic": key,
                    "characteristic_label": label,
                    "characteristic_role": role,
                    "n_datasets": 8,
                    "spearman_rho": rho,
                    "bootstrap_lo": lo,
                    "bootstrap_hi": hi,
                    "n_bootstrap_finite": n_used,
                    "permutation_p": p_value,
                    "loo_rho_min": loo_min,
                    "loo_rho_max": loo_max,
                    "interval_covers_zero": bool(lo <= 0 <= hi) if np.isfinite(lo) and np.isfinite(hi) else True,
                    "draws_used_as_sample_size": False,
                }
            )
    frame = pd.DataFrame(rows)
    frame["holm_p"] = holm_adjust(frame["permutation_p"].to_numpy(float))
    frame["family"] = FAMILY_NOTE
    frame["n_family"] = len(frame)
    return frame


def _cross_split_family(tests: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for metric, model, arm in CROSS_SPLIT_CONTRASTS:
        pieces.append(
            tests[
                (tests.metric == metric)
                & (tests.model == model)
                & (tests.arm_a == arm)
                & (tests.arm_b == "baseline")
                & (tests.sigma.isin([0.0, 1.0]))
            ]
        )
    family = pd.concat(pieces, ignore_index=True)
    if len(family) != 64:
        raise RuntimeError(f"Cross-dataset split family has {len(family)} rows, expected 64.")
    return family


def _wdbc_split_family(tests: pd.DataFrame) -> pd.DataFrame:
    family = tests[
        tests.metric.isin(["brier", "ece10"])
        & tests.arm_a.isin(["sigmoid", "isotonic"])
        & (tests.arm_b == "baseline")
    ].copy()
    if len(family) != 48:
        raise RuntimeError(f"WDBC split family has {len(family)} rows, expected 48.")
    return family


def adjust_split_families(cross_tests: pd.DataFrame, wdbc_tests: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for name, family in (
        ("cross_dataset_64", _cross_split_family(cross_tests)),
        ("wdbc_brier_ece_48", _wdbc_split_family(wdbc_tests)),
    ):
        out = family.copy()
        out.insert(0, "family", name)
        out["holm_p"] = holm_adjust(out["two_sided_p"].to_numpy(float))
        out["raw_p_below_0.05"] = out["two_sided_p"] < 0.05
        out["holm_p_below_0.05"] = out["holm_p"] < 0.05
        pieces.append(out)
    return pd.concat(pieces, ignore_index=True)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def figure_effect_sizes(paired: pd.DataFrame, output: Path) -> None:
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4), sharex=True)
    for ax, (model, arm) in zip(axes.ravel(), BRIER_CONTRASTS):
        part = paired[
            (paired.model == model)
            & (paired.arm_a == arm)
            & (paired.arm_b == "baseline")
            & (paired.metric == "brier")
            & (paired.sigma.sub(1.0).abs() < 1e-9)
        ].sort_values("mean_diff")
        y = np.arange(len(part))
        ax.errorbar(
            part["mean_diff"],
            y,
            xerr=[part["mean_diff"] - part["draw_lo"], part["draw_hi"] - part["mean_diff"]],
            fmt="o",
            color="#0072B2",
            markersize=4,
            capsize=2,
        )
        ax.axvline(0, color="0.4", lw=0.8)
        ax.set_yticks(y, part["dataset"])
        ax.set_title(f"{model}, {arm} − baseline")
    for ax in axes[1]:
        ax.set_xlabel("Brier difference at sigma = 1")
    fig.suptitle(
        "Eight datasets. Bars are 2.5–97.5 percent intervals across 50 draws, not confidence intervals for a new dataset.",
        fontsize=9,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def figure_spearman(associations: pd.DataFrame, output: Path) -> None:
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.2), sharex=True)
    contrast_labels = [f"{model.split()[0]} {arm}" for model, arm in BRIER_CONTRASTS]
    for ax, (key, label, _role) in zip(axes.ravel(), CHARACTERISTICS):
        part = associations[associations.characteristic == key].copy()
        part["_order"] = part.apply(lambda row: BRIER_CONTRASTS.index((row.model, row.arm_a)), axis=1)
        part = part.sort_values("_order")
        y = np.arange(len(part))
        ax.errorbar(
            part["spearman_rho"],
            y,
            xerr=[part["spearman_rho"] - part["bootstrap_lo"], part["bootstrap_hi"] - part["spearman_rho"]],
            fmt="o",
            color="#D55E00",
            markersize=4,
            capsize=2,
        )
        ax.axvline(0, color="0.4", lw=0.8)
        ax.set_yticks(y, contrast_labels)
        ax.set_title(label, fontsize=8)
        ax.set_xlim(-1.05, 1.05)
    for ax in axes[1]:
        ax.set_xlabel("Spearman rho")
    fig.suptitle(
        "Association across 8 datasets. Bars resample datasets, not the 50 draws. An interval covering 0 is not a predictor.",
        fontsize=9,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def figure_scatters(paired: pd.DataFrame, characteristics: pd.DataFrame, output: Path) -> None:
    """All 36 exploratory scatters. Eight points each. No fitted line."""
    _style()
    fig, axes = plt.subplots(6, 6, figsize=(12.5, 12.2))
    for row, (model, arm) in enumerate(BRIER_CONTRASTS):
        part = paired[
            (paired.model == model)
            & (paired.arm_a == arm)
            & (paired.arm_b == "baseline")
            & (paired.metric == "brier")
            & (paired.sigma.sub(1.0).abs() < 1e-9)
        ]
        merged = part.merge(characteristics[characteristics.model == model], on=["dataset", "model"])
        for col, (key, label, _role) in enumerate(CHARACTERISTICS):
            ax = axes[row, col]
            ax.scatter(merged[key], merged["mean_diff"], s=16, color="#0072B2")
            ax.axhline(0, color="0.5", lw=0.6)
            if row == 0:
                ax.set_title(label, fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{model.split()[0]} {arm}", fontsize=7)
    fig.suptitle(
        "Each panel is 8 datasets. Vertical axis: Brier difference at sigma = 1 (recalibrated minus baseline). No line is fit.",
        fontsize=9,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def run_association_analysis(root: Path, n_boot: int = 4000, n_perm: int = 4999, seed: int = 20260311) -> None:
    root = Path(root)
    output = root / "results" / "association"
    if "preserved" in output.parts:
        raise RuntimeError("Refusing to write into preserved/.")
    output.mkdir(parents=True, exist_ok=True)
    card = pd.read_csv(root / "results/cross_dataset/dataset_card.csv")
    summary = pd.read_csv(root / "results/cross_dataset/summary_draws.csv")
    paired = pd.read_csv(root / "results/cross_dataset/paired_calibration.csv")
    cross_tests = pd.read_csv(root / "results/cross_dataset/split_level_tests.csv")
    wdbc_tests = pd.read_csv(root / "results/recalibration/split_level_tests.csv")

    characteristics = build_characteristic_table(card, summary)
    characteristics.to_csv(output / "characteristics.csv", index=False)
    associations = association_table(paired, characteristics, n_boot, n_perm, seed)
    associations.to_csv(output / "spearman.csv", index=False)
    adjusted = adjust_split_families(cross_tests, wdbc_tests)
    adjusted.to_csv(output / "holm_split_tests.csv", index=False)

    counts = []
    for family, group in adjusted.groupby("family", sort=True):
        counts.append(
            {
                "family": family,
                "n_tests": int(len(group)),
                "n_raw_p_below_0.05": int(group["raw_p_below_0.05"].sum()),
                "n_holm_p_below_0.05": int(group["holm_p_below_0.05"].sum()),
            }
        )
    pd.DataFrame(counts).to_csv(output / "holm_counts.csv", index=False)
    n_cover = int(associations["interval_covers_zero"].sum())
    n_holm_assoc = int((associations["holm_p"] < 0.05).sum())
    summary_row = {
        "n_datasets": 8,
        "n_spearman_tests": int(len(associations)),
        "n_bootstrap": n_boot,
        "n_permutations": n_perm,
        "seed": seed,
        "n_spearman_intervals_covering_zero": n_cover,
        "n_spearman_holm_p_below_0.05": n_holm_assoc,
        "draws_are_not_independent_datasets": True,
        "spearman_family": FAMILY_NOTE,
        "split_families": SPLIT_FAMILY_NOTE,
    }
    (output / "summary.json").write_text(json.dumps(summary_row, indent=2) + "\n")
    (output / "METHOD.txt").write_text(
        "Unit of analysis: one dataset. n = 8.\n"
        "Outcome: mean paired Brier difference at sigma = 1, recalibrated arm minus baseline. "
        "The mean is the mean over 50 draws of the five-split average. "
        "The draw interval in the effect-size figure is the 2.5 and 97.5 percent linear quantile of those 50 numbers. "
        "It is not a confidence interval for a new dataset, and the 50 draws are not the sample size of the Spearman correlation.\n"
        "Characteristics: rows, features, smaller-class rate, baseline 10-bin ECE at sigma 0, "
        "baseline ROC-AUC at sigma 0, and the increase in baseline Brier from sigma 0 to sigma 1. "
        "The last three are computed from the same experiment as the outcome. "
        "They are descriptions, not exogenous causes, and not a prediction model.\n"
        "Association: Spearman rho. Uncertainty: percentile interval from 4000 bootstrap resamples of the 8 datasets. "
        "Leave-one-out rho range is stored beside each correlation. No line is fit to the scatters.\n"
        f"{FAMILY_NOTE}\n"
        "Permutation p-values permute the Brier difference across the 8 datasets, "
        "4999 times, two-sided, with the add-one correction (1 + hits) / (4999 + 1). "
        "Holm adjustment uses those 36 p-values only.\n"
        f"{SPLIT_FAMILY_NOTE}\n"
        "Holm is the step-down procedure: sort p-values, multiply rank i by (m - i + 1), cap at 1, "
        "then enforce nondecreasing adjusted values. Raw files under results/recalibration/ and "
        "results/cross_dataset/ are not rewritten.\n"
        "This analysis does not claim that a characteristic predicts recalibration benefit.\n"
    )
    figure_effect_sizes(paired, root / "figures" / "brier_effect_by_dataset.png")
    figure_spearman(associations, root / "figures" / "spearman_associations.png")
    figure_scatters(paired, characteristics, root / "figures" / "association_scatters.png")
