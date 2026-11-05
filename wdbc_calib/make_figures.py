"""Figures drawn only from result CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {
    "Logistic regression": "#0072B2",
    "Random forest": "#E69F00",
    "RBF SVM": "#009E73",
}
CAL_COLORS = {
    "baseline": "#0072B2",
    "sigmoid": "#D55E00",
    "isotonic": "#CC79A7",
}
CAL_LABELS = {
    "baseline": "Baseline probability",
    "sigmoid": "Sigmoid recalibration",
    "isotonic": "Isotonic recalibration",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def figure_baseline_bands(summary_csv: Path, output: Path) -> None:
    summary = pd.read_csv(summary_csv)
    summary = summary[summary.calibration == "baseline"]
    metrics = [
        ("roc_auc", "ROC-AUC"),
        ("sensitivity", "Sensitivity (malignant)"),
        ("brier", "Brier score (lower better)"),
        ("ece10", "ECE, 10 equal-width bins (lower better)"),
    ]
    _style()
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.4))
    for ax, (metric, title) in zip(axes, metrics):
        for model, color in COLORS.items():
            part = summary[(summary.model == model) & (summary.metric == metric)].sort_values("sigma")
            if part.empty:
                continue
            ax.plot(part["sigma"], part["mean"], marker="o", color=color, label=model)
            ax.fill_between(part["sigma"], part["draw_lo"], part["draw_hi"], color=color, alpha=0.18)
        ax.set_title(title)
        ax.set_xlabel("Noise sigma (× training-feature SD)")
        ax.set_xticks([0, 0.25, 0.5, 1.0])
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Explicit protocol. Mean over splits; band = 2.5–97.5 percentile across noise draws\n"
        "(noise-seed variation with the splits held fixed; not a bootstrap confidence interval)",
        fontsize=10,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def figure_recalibration(summary_csv: Path, output: Path) -> None:
    summary = pd.read_csv(summary_csv)
    metrics = [
        ("brier", "Brier score"),
        ("ece10", "ECE (10 equal-width bins)"),
        ("roc_auc", "ROC-AUC"),
    ]
    models = ["Logistic regression", "Random forest", "RBF SVM"]
    _style()
    fig, axes = plt.subplots(len(metrics), len(models), figsize=(11.2, 7.6), sharex=True)
    for i, (metric, metric_title) in enumerate(metrics):
        for j, model in enumerate(models):
            ax = axes[i, j]
            for cal, color in CAL_COLORS.items():
                part = summary[
                    (summary.model == model) & (summary.metric == metric) & (summary.calibration == cal)
                ].sort_values("sigma")
                if part.empty:
                    continue
                ax.plot(part["sigma"], part["mean"], marker="o", color=color, label=CAL_LABELS[cal])
                ax.fill_between(part["sigma"], part["draw_lo"], part["draw_hi"], color=color, alpha=0.15)
            if i == 0:
                ax.set_title(model)
            if j == 0:
                ax.set_ylabel(metric_title)
            if i == len(metrics) - 1:
                ax.set_xlabel("sigma")
                ax.set_xticks([0, 0.25, 0.5, 1.0])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Recalibration fit on clean training folds, evaluated under isotropic feature noise\n"
        "Bands: 2.5–97.5 percentile across draws of the split-average. Not a patient-level confidence interval.",
        fontsize=10,
        y=1.12,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def figure_reliability(reliability_csv: Path, output: Path) -> None:
    rel = pd.read_csv(reliability_csv)
    models = ["Logistic regression", "Random forest", "RBF SVM"]
    cals = ["baseline", "sigmoid", "isotonic"]
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), sharex=True, sharey=True)
    for col, model in enumerate(models):
        for row, sigma in enumerate((0.0, 1.0)):
            ax = axes[row, col]
            ax.plot([0, 1], [0, 1], color="0.5", lw=1, zorder=0)
            for cal in cals:
                part = rel[
                    (rel.model == model)
                    & (rel.calibration == cal)
                    & (rel["sigma"].sub(sigma).abs() < 1e-9)
                ]
                part = part[part.count_prediction_rows > 0].sort_values("bin")
                ax.plot(
                    part.mean_predicted,
                    part.observed_frequency,
                    marker="o",
                    color=CAL_COLORS[cal],
                    label=CAL_LABELS[cal],
                )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            if row == 0:
                ax.set_title(model)
            if col == 0:
                ax.set_ylabel(f"Observed frequency\nsigma = {sigma:g}")
            if row == 1:
                ax.set_xlabel("Mean predicted P(malignant)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Reliability diagrams, 10 equal-width bins, pooled over splits and draws\n"
        "Counts are repeated predictions of overlapping test rows, not distinct patients.",
        fontsize=10,
        y=1.12,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def figure_corruptions(summary_csv: Path, output: Path) -> None:
    summary = pd.read_csv(summary_csv)
    summary = summary[summary.metric == "brier"]
    corruptions = [
        ("isotropic_gaussian", "Isotropic Gaussian"),
        ("correlated_gaussian", "Correlated Gaussian"),
        ("multiplicative_gaussian", "Multiplicative Gaussian"),
        ("worst10_isotropic_gaussian", "Noise on 10 worst features"),
        ("train_mean_dropout", "Train-mean feature dropout"),
    ]
    _style()
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.3), sharey=True)
    for ax, (name, title) in zip(axes, corruptions):
        for model, color in COLORS.items():
            part = summary[(summary.corruption == name) & (summary.model == model)].sort_values("level")
            if part.empty:
                continue
            ax.plot(part["level"], part["mean"], marker="o", color=color, label=model)
            ax.fill_between(part["level"], part["draw_lo"], part["draw_hi"], color=color, alpha=0.15)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("level")
    axes[0].set_ylabel("Brier score")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Alternative synthetic corruptions, baseline models. Bands are draw percentiles, not confidence intervals.",
        fontsize=10,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def figure_thresholds(summary_csv: Path, output: Path) -> None:
    summary = pd.read_csv(summary_csv)
    summary = summary[summary.metric == "mean_cost"]
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    for ax, model in zip(axes, ["Logistic regression", "Random forest", "RBF SVM"]):
        part = summary[summary.model == model]
        for (fn, fp), color in [((1, 1), "#0072B2"), ((5, 1), "#D55E00"), ((10, 1), "#CC79A7")]:
            sub = part[(part.cost_fn == fn) & (part.cost_fp == fp)].sort_values("sigma")
            ax.plot(sub["sigma"], sub["mean"], marker="o", color=color, label=f"FN:FP = {fn}:{fp}")
            ax.fill_between(sub["sigma"], sub["draw_lo"], sub["draw_hi"], color=color, alpha=0.15)
        ax.set_title(model)
        ax.set_xlabel("sigma")
        ax.set_xticks([0, 0.25, 0.5, 1])
    axes[0].set_ylabel("Mean cost per test row")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Thresholds chosen on training out-of-fold predictions, then frozen and applied to the test fold.",
        fontsize=10,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def render_all(root: Path) -> list[Path]:
    root = Path(root)
    written: list[Path] = []
    summary = root / "results/recalibration/summary_draws.csv"
    if summary.exists():
        for dest, fn in (
            (root / "figures/recalibration_bands.png", figure_recalibration),
            (root / "figures/baseline_noise_bands.png", figure_baseline_bands),
        ):
            fn(summary, dest)
            written.append(dest)
    reliability = root / "results/recalibration/reliability_bins.csv"
    if reliability.exists():
        dest = root / "figures/reliability_diagrams.png"
        figure_reliability(reliability, dest)
        written.append(dest)
    corruptions = root / "results/corruptions/summary_draws.csv"
    if corruptions.exists():
        dest = root / "figures/corruptions_brier.png"
        figure_corruptions(corruptions, dest)
        written.append(dest)
    thresholds = root / "results/thresholds/summary_draws.csv"
    if thresholds.exists():
        dest = root / "figures/threshold_costs.png"
        figure_thresholds(thresholds, dest)
        written.append(dest)
    return written
