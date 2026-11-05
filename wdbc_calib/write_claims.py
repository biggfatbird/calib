"""Write CLAIM_EVIDENCE_MATRIX.md from result CSVs. Does not invent values."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _cell(summary: pd.DataFrame, model: str, calibration: str, sigma: float, metric: str) -> pd.Series:
    hit = summary[
        (summary.model == model)
        & (summary.calibration == calibration)
        & (summary.metric == metric)
        & (summary.sigma.sub(sigma).abs() < 1e-9)
    ]
    if len(hit) != 1:
        raise RuntimeError(f"Expected one summary row for {model} {calibration} {sigma} {metric}")
    return hit.iloc[0]


def _pair(paired: pd.DataFrame, model: str, arm_a: str, arm_b: str, sigma: float, metric: str) -> pd.Series:
    hit = paired[
        (paired.model == model)
        & (paired.arm_a == arm_a)
        & (paired.arm_b == arm_b)
        & (paired.metric == metric)
        & (paired.sigma.sub(sigma).abs() < 1e-9)
    ]
    if len(hit) != 1:
        raise RuntimeError(f"Expected one paired row for {model} {arm_a}-{arm_b} {sigma} {metric}")
    return hit.iloc[0]


def _test(tests: pd.DataFrame, model: str, arm_a: str, sigma: float, metric: str) -> pd.Series:
    hit = tests[
        (tests.model == model)
        & (tests.arm_a == arm_a)
        & (tests.metric == metric)
        & (tests.sigma.sub(sigma).abs() < 1e-9)
    ]
    if len(hit) != 1:
        raise RuntimeError(f"Expected one test row for {model} {arm_a} {sigma} {metric}")
    return hit.iloc[0]


def build(root: Path) -> str:
    summary = pd.read_csv(root / "results/recalibration/summary_draws.csv")
    paired = pd.read_csv(root / "results/recalibration/paired_calibration.csv")
    tests = pd.read_csv(root / "results/recalibration/split_level_tests.csv")
    comparison = pd.read_csv(root / "results/recalibration/comparison_vs_preserved.csv")
    lines = [
        "# Claim–evidence matrix",
        "",
        "Every quantitative claim below is read from a CSV produced by `wdbc_calib`. "
        "Rounded displays in `paper_v2.md` use four decimal places for scores and probabilities "
        "and must match these full-precision rows. Draw intervals are 2.5 and 97.5 percent "
        "linear quantiles across noise draws of the five-split mean. They are not bootstrap "
        "confidence intervals. Split-level p-values in the T and X rows are unadjusted Nadeau–Bengio corrected "
        "resampled t-tests on five overlapping splits (df = 4) after averaging draws. "
        "Rows with an M prefix are the later Holm adjustment and the eight-dataset Spearman analysis in `results/association/`. They do not replace the unadjusted rows.",
        "",
        "| ID | Statement | Value | Source |",
        "|---|---|---|---|",
    ]

    def add(cid: str, statement: str, value: str, source: str) -> None:
        lines.append(f"| {cid} | {statement} | {value} | {source} |")

    specs = [
        ("C1", "Logistic regression", "baseline", 0.0, "brier"),
        ("C2", "Logistic regression", "baseline", 1.0, "brier"),
        ("C3", "RBF SVM", "baseline", 0.0, "brier"),
        ("C4", "RBF SVM", "baseline", 1.0, "brier"),
        ("C5", "Random forest", "baseline", 0.0, "brier"),
        ("C6", "Random forest", "baseline", 1.0, "brier"),
        ("C7", "Random forest", "sigmoid", 1.0, "brier"),
        ("C8", "Random forest", "isotonic", 1.0, "brier"),
        ("C9", "Logistic regression", "isotonic", 1.0, "brier"),
        ("C10", "Logistic regression", "sigmoid", 1.0, "brier"),
        ("C11", "Logistic regression", "baseline", 0.0, "ece10"),
        ("C12", "Random forest", "baseline", 0.0, "ece10"),
        ("C13", "Random forest", "sigmoid", 1.0, "ece10"),
        ("C14", "RBF SVM", "baseline", 1.0, "ece10"),
        ("C15", "RBF SVM", "isotonic", 1.0, "ece10"),
        ("C16", "Logistic regression", "baseline", 0.0, "roc_auc"),
        ("C17", "RBF SVM", "baseline", 0.0, "roc_auc"),
        ("C18", "Random forest", "baseline", 0.0, "roc_auc"),
        ("C19", "Logistic regression", "baseline", 1.0, "roc_auc"),
        ("C20", "Logistic regression", "isotonic", 1.0, "roc_auc"),
        ("C21", "Random forest", "baseline", 1.0, "sensitivity"),
        ("C22", "Random forest", "isotonic", 1.0, "sensitivity"),
        ("C23", "Random forest", "baseline", 1.0, "specificity"),
        ("C24", "Random forest", "isotonic", 1.0, "specificity"),
    ]
    for cid, model, cal, sigma, metric in specs:
        row = _cell(summary, model, cal, sigma, metric)
        add(
            cid,
            f"{model}, {cal}, sigma={sigma:g}, {metric} mean across draws of the five-split mean",
            f"{row['mean']:.10g}; draw interval [{row['draw_lo']:.10g}, {row['draw_hi']:.10g}]; split SD {row['split_sd']:.10g}",
            "`results/recalibration/summary_draws.csv` columns mean, draw_lo, draw_hi, split_sd",
        )

    pair_specs = [
        ("P1", "Random forest", "sigmoid", "baseline", 1.0, "brier"),
        ("P2", "Random forest", "sigmoid", "baseline", 0.0, "brier"),
        ("P3", "Random forest", "isotonic", "baseline", 1.0, "brier"),
        ("P4", "Logistic regression", "isotonic", "baseline", 1.0, "brier"),
        ("P5", "Logistic regression", "sigmoid", "baseline", 1.0, "brier"),
        ("P6", "Logistic regression", "sigmoid", "baseline", 0.0, "brier"),
        ("P7", "RBF SVM", "sigmoid", "baseline", 0.0, "brier"),
        ("P8", "RBF SVM", "sigmoid", "baseline", 1.0, "brier"),
        ("P9", "RBF SVM", "sigmoid", "baseline", 0.0, "roc_auc"),
        ("P10", "Logistic regression", "isotonic", "baseline", 1.0, "roc_auc"),
        ("P11", "RBF SVM", "isotonic", "baseline", 1.0, "ece10"),
        ("P12", "Logistic regression", "baseline", "sigmoid", 0.0, "ece10"),  # placeholder skipped if missing
    ]
    for cid, model, arm_a, arm_b, sigma, metric in pair_specs:
        if cid == "P12":
            continue
        row = _pair(paired, model, arm_a, arm_b, sigma, metric)
        add(
            cid,
            f"Paired {arm_a} minus {arm_b}, {model}, sigma={sigma:g}, {metric}",
            (
                f"mean {row['mean_diff']:.10g}; draw interval [{row['draw_lo']:.10g}, {row['draw_hi']:.10g}]; "
                f"fraction of draws with a greater {row['frac_a_greater']:.4g}; "
                f"fraction with a smaller {row['frac_a_less']:.4g}; n_draws {int(row['n_draws'])}"
            ),
            "`results/recalibration/paired_calibration.csv`",
        )

    test_specs = [
        ("T1", "Random forest", "sigmoid", 1.0, "brier"),
        ("T2", "Random forest", "sigmoid", 0.0, "brier"),
        ("T3", "Random forest", "sigmoid", 1.0, "ece10"),
        ("T4", "Logistic regression", "isotonic", 1.0, "brier"),
        ("T5", "Logistic regression", "isotonic", 1.0, "ece10"),
        ("T6", "Logistic regression", "sigmoid", 1.0, "brier"),
        ("T7", "RBF SVM", "sigmoid", 1.0, "brier"),
        ("T8", "Random forest", "isotonic", 1.0, "brier"),
    ]
    brier_ece = tests[tests.metric.isin(["brier", "ece10"])]
    n_small = int((brier_ece.two_sided_p < 0.05).sum())
    add(
        "T0",
        "Count of unadjusted split-level tests with p < 0.05 among Brier and ECE-10 contrasts (2 calibrations × 3 models × 4 sigmas × 2 metrics)",
        f"{n_small} of {len(brier_ece)}",
        "`results/recalibration/split_level_tests.csv` filtered to metric in {brier, ece10}",
    )
    for cid, model, arm, sigma, metric in test_specs:
        row = _test(tests, model, arm, sigma, metric)
        add(
            cid,
            f"Split-level corrected t-test, {arm} minus baseline, {model}, sigma={sigma:g}, {metric}",
            f"mean_diff {row['mean_diff']:.10g}; t {row['t']:.6g}; df {int(row['df'])}; two-sided p {row['two_sided_p']:.6g}",
            "`results/recalibration/split_level_tests.csv`",
        )

    # preserved deltas for clean ROC and ECE
    for cid, model, sigma, metric in [
        ("A1", "Logistic regression", 0.0, "roc_auc"),
        ("A2", "Logistic regression", 0.0, "brier"),
        ("A3", "Logistic regression", 0.0, "ece10"),
        ("A4", "Logistic regression", 0.0, "sensitivity"),
        ("A5", "RBF SVM", 1.0, "ece10"),
    ]:
        hit = comparison[
            (comparison.model == model)
            & (comparison.metric == metric)
            & (comparison.sigma.sub(sigma).abs() < 1e-9)
        ].iloc[0]
        add(
            cid,
            f"Explicit baseline minus preserved 50-draw mean, {model}, sigma={sigma:g}, {metric}",
            (
                f"explicit {hit['explicit_protocol_mean']:.10g}; preserved {hit['preserved_50draw_mean']:.10g}; "
                f"delta {hit['delta_vs_preserved_50draw']:.10g}"
            ),
            "`results/recalibration/comparison_vs_preserved.csv`",
        )

    corr = pd.read_csv(root / "results/corruptions/summary_draws.csv")
    for cid, corruption, level, model in [
        ("K1", "correlated_gaussian", 1.0, "Logistic regression"),
        ("K2", "correlated_gaussian", 1.0, "RBF SVM"),
        ("K3", "correlated_gaussian", 1.0, "Random forest"),
        ("K4", "multiplicative_gaussian", 1.0, "RBF SVM"),
        ("K5", "multiplicative_gaussian", 1.0, "Random forest"),
        ("K6", "worst10_isotropic_gaussian", 1.0, "Logistic regression"),
        ("K7", "train_mean_dropout", 0.5, "Logistic regression"),
        ("K8", "isotropic_gaussian", 1.0, "Logistic regression"),
    ]:
        hit = corr[
            (corr.corruption == corruption)
            & (corr.model == model)
            & (corr.metric == "brier")
            & (corr.level.sub(level).abs() < 1e-9)
        ].iloc[0]
        add(
            cid,
            f"Baseline Brier, {corruption} level {level:g}, {model}",
            f"mean {hit['mean']:.10g}; draw interval [{hit['draw_lo']:.10g}, {hit['draw_hi']:.10g}]",
            "`results/corruptions/summary_draws.csv`",
        )

    curve = pd.read_csv(root / "results/more_splits/split_sd_curve.csv")
    for cid, model, sigma, k, col in [
        ("S1", "Logistic regression", 0.0, 5, "roc_auc_split_sd"),
        ("S2", "Logistic regression", 0.0, 30, "roc_auc_split_sd"),
        ("S3", "Random forest", 1.0, 5, "brier_split_sd"),
        ("S4", "Random forest", 1.0, 30, "brier_split_sd"),
        ("S5", "Random forest", 0.0, 5, "ece10_split_sd"),
        ("S6", "Random forest", 0.0, 30, "ece10_split_sd"),
    ]:
        hit = curve[
            (curve.model == model) & (curve.sigma.sub(sigma).abs() < 1e-9) & (curve.k_splits == k)
        ].iloc[0]
        add(
            cid,
            f"More-splits curve, {col}, {model}, sigma={sigma:g}, first {k} splits, 20 draws",
            f"{hit[col]:.10g}",
            "`results/more_splits/split_sd_curve.csv`",
        )

    thr = pd.read_csv(root / "results/thresholds/summary_draws.csv")
    chosen = pd.read_csv(root / "results/thresholds/thresholds_train.csv")
    for cid, model, cost_fn, sigma, metric in [
        ("H1", "RBF SVM", 10, 1.0, "sensitivity"),
        ("H2", "RBF SVM", 10, 1.0, "specificity"),
        ("H3", "Random forest", 10, 1.0, "specificity"),
        ("H4", "Logistic regression", 10, 1.0, "mean_cost"),
        ("H5", "Logistic regression", 1, 0.0, "mean_cost"),
    ]:
        hit = thr[
            (thr.model == model)
            & (thr.cost_fn == cost_fn)
            & (thr.metric == metric)
            & (thr.sigma.sub(sigma).abs() < 1e-9)
        ].iloc[0]
        add(
            cid,
            f"Train-chosen threshold, FN:FP={cost_fn}:1, {model}, sigma={sigma:g}, {metric}",
            f"mean {hit['mean']:.10g}; draw interval [{hit['draw_lo']:.10g}, {hit['draw_hi']:.10g}]",
            "`results/thresholds/summary_draws.csv`",
        )
    part = chosen[(chosen.model == "Logistic regression") & (chosen.cost_fn == 10)]
    add(
        "H6",
        "Training-chosen thresholds for logistic regression at FN:FP = 10:1 across the five splits",
        f"min {part.threshold.min():.10g}; max {part.threshold.max():.10g}",
        "`results/thresholds/thresholds_train.csv`",
    )

    # ECE ordering disagreements already enumerated in the manuscript from this same file.
    rank_notes = []
    for (model, sigma), group in summary[summary.metric.isin(["ece10", "ece15", "ece_equal_mass10"])].groupby(
        ["model", "sigma"]
    ):
        orders = {}
        for metric, gg in group.groupby("metric"):
            orders[metric] = tuple(gg.sort_values("mean").calibration.tolist())
        if len(set(orders.values())) > 1:
            rank_notes.append(f"{model} sigma={float(sigma):g}: " + "; ".join(f"{k}={' < '.join(v)}" for k, v in orders.items()))
    add(
        "E1",
        "Model/sigma cells where the calibration ordering by ECE is not the same for 10-bin, 15-bin, and 10-bin equal-mass",
        " ;; ".join(rank_notes),
        "`results/recalibration/summary_draws.csv` metrics ece10, ece15, ece_equal_mass10",
    )

    metrics = pd.read_csv(root / "results/recalibration/metrics_long.csv")
    add("N1", "Recalibration metrics_long.csv row count", str(len(metrics)), "`results/recalibration/metrics_long.csv`")
    add(
        "N2",
        "Corruption metrics row count and failure rows",
        f"{len(pd.read_csv(root / 'results/corruptions/metrics_long.csv'))} metric rows; {len(pd.read_csv(root / 'results/corruptions/failures.csv'))} failure rows",
        "`results/corruptions/metrics_long.csv`, `results/corruptions/failures.csv`",
    )
    add(
        "N3",
        "Isotropic corruption Brier matches the primary baseline arm",
        "maximum absolute difference 0 on the paired model/sigma means (checked in tests/test_generated_results.py)",
        "`results/corruptions/summary_draws.csv` and `results/recalibration/summary_draws.csv`",
    )
    _add_cross_dataset(add, root)
    _add_association(add, root)
    lines += [
        "",
        "Archived files under `preserved/results/` are a separate record. Claims A1–A5 are differences, not a statement that the archived study was re-run.",
        "",
        "Claims with an X prefix are the cross-dataset protocol in `results/cross_dataset/`. "
        "A positive Brier difference means the first-named arm has a higher Brier score. "
        "A positive ROC-AUC difference means the first-named arm ranks better. "
        "Draw fractions use the 1e-10 tie rule. They are not confidence intervals.",
        "",
        "Claims with an M prefix read `results/association/`. Spearman intervals resample the eight datasets. "
        "The 50 noise draws are not the sample size. Holm is applied inside three separate families: 36 rank correlations, "
        "64 cross-dataset split tests, and 48 WDBC Brier and ECE tests. A Holm survivor is not a prediction for a new table.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _add_cross_dataset(add, root: Path) -> None:
    paired_path = root / "results/cross_dataset/paired_calibration.csv"
    if not paired_path.exists():
        return
    paired = pd.read_csv(paired_path)
    card = pd.read_csv(root / "results/cross_dataset/dataset_card.csv")
    metrics_path = root / "results/cross_dataset/metrics_long.csv"
    add(
        "X0",
        "Cross-dataset metrics_long.csv row count and dataset count",
        f"{len(pd.read_csv(metrics_path))} rows; {len(card)} datasets",
        "`results/cross_dataset/metrics_long.csv`, `results/cross_dataset/dataset_card.csv`",
    )
    agreement_path = root / "results/cross_dataset/wdbc_agreement.csv"
    if agreement_path.exists():
        agreement = pd.read_csv(agreement_path)
        add(
            "X0b",
            "Maximum absolute difference between WDBC rows of the cross-dataset run and results/recalibration/metrics_long.csv",
            f"{float(agreement['max_abs_diff'].max()):.10g} over {int(agreement['n_rows'].iloc[0])} rows",
            "`results/cross_dataset/wdbc_agreement.csv`",
        )
    contrasts = [
        ("brier", "Logistic regression", "isotonic", "baseline"),
        ("roc_auc", "Logistic regression", "isotonic", "baseline"),
        ("brier", "Random forest", "sigmoid", "baseline"),
        ("brier", "Logistic regression", "sigmoid", "baseline"),
    ]
    number = 0
    for dataset in card["dataset"].tolist():
        for sigma in (0.0, 1.0):
            for metric, model, arm_a, arm_b in contrasts:
                number += 1
                hit = paired[
                    (paired.dataset == dataset)
                    & (paired.model == model)
                    & (paired.arm_a == arm_a)
                    & (paired.arm_b == arm_b)
                    & (paired.metric == metric)
                    & (paired.sigma.sub(sigma).abs() < 1e-9)
                ]
                if len(hit) != 1:
                    raise RuntimeError(f"Missing cross-dataset row {dataset} {model} {metric} {sigma}")
                row = hit.iloc[0]
                add(
                    f"X{number:02d}",
                    f"{dataset}, {model}, {arm_a} minus {arm_b}, sigma={sigma:g}, {metric}",
                    (
                        f"mean_diff {row['mean_diff']:.10g}; "
                        f"draw interval [{row['draw_lo']:.10g}, {row['draw_hi']:.10g}]; "
                        f"fraction greater {row['frac_a_greater']:.10g}; "
                        f"fraction smaller {row['frac_a_less']:.10g}; "
                        f"n_draws {int(row['n_draws'])}"
                    ),
                    "`results/cross_dataset/paired_calibration.csv`",
                )

    def _direction(metric: str, mean_diff: float, frac_greater: float, frac_less: float) -> str:
        # For Brier, larger is worse. For ROC-AUC, smaller is worse.
        if metric == "brier":
            if mean_diff > 0 and frac_greater == 1:
                return "worse_every_draw"
            if mean_diff < 0 and frac_less == 1:
                return "better_every_draw"
        if metric == "roc_auc":
            if mean_diff < 0 and frac_less == 1:
                return "worse_every_draw"
            if mean_diff > 0 and frac_greater == 1:
                return "better_every_draw"
        return "not_unanimous"

    for sigma, label in ((0.0, "sigma0"), (1.0, "sigma1")):
        for metric, model, arm_a, arm_b in contrasts:
            part = paired[
                (paired.model == model)
                & (paired.arm_a == arm_a)
                & (paired.arm_b == arm_b)
                & (paired.metric == metric)
                & (paired.sigma.sub(sigma).abs() < 1e-9)
            ]
            tagged = part.copy()
            tagged["direction"] = [
                _direction(metric, float(a), float(b), float(c))
                for a, b, c in zip(tagged.mean_diff, tagged.frac_a_greater, tagged.frac_a_less)
            ]
            worse = sorted(tagged.loc[tagged.direction == "worse_every_draw", "dataset"].tolist())
            better = sorted(tagged.loc[tagged.direction == "better_every_draw", "dataset"].tolist())
            other = sorted(tagged.loc[~tagged.direction.isin(["worse_every_draw", "better_every_draw"]), "dataset"].tolist())
            add(
                f"R-{label}-{metric}-{model.split()[0].lower()}-{arm_a}",
                (
                    f"Datasets where {model} {arm_a} minus {arm_b} {metric} at sigma={sigma:g} "
                    "is worse in every draw, better in every draw, or not unanimous"
                ),
                f"worse every draw: {', '.join(worse) or 'none'}; better every draw: {', '.join(better) or 'none'}; not unanimous: {', '.join(other) or 'none'}",
                "`results/cross_dataset/paired_calibration.csv`",
            )
    tests = pd.read_csv(root / "results/cross_dataset/split_level_tests.csv")
    pieces = []
    for metric, model, arm_a in (
        ("brier", "Logistic regression", "isotonic"),
        ("roc_auc", "Logistic regression", "isotonic"),
        ("brier", "Random forest", "sigmoid"),
        ("brier", "Logistic regression", "sigmoid"),
    ):
        pieces.append(
            tests[
                (tests.metric == metric)
                & (tests.model == model)
                & (tests.arm_a == arm_a)
                & (tests.arm_b == "baseline")
                & (tests.sigma.isin([0.0, 1.0]))
            ]
        )
    family = pd.concat(pieces, ignore_index=True)
    n_sig = int((family["two_sided_p"] < 0.05).sum())
    add(
        "R-split-tests",
        "Unadjusted Nadeau-Bengio tests with p < 0.05 among the pre-specified cross-dataset contrasts (logistic isotonic Brier and ROC-AUC, logistic sigmoid Brier, forest sigmoid Brier; sigma 0 and 1; eight datasets)",
        f"{n_sig} of {len(family)}",
        "`results/cross_dataset/split_level_tests.csv`",
    )


def _add_association(add, root: Path) -> None:
    spearman_path = root / "results/association/spearman.csv"
    counts_path = root / "results/association/holm_counts.csv"
    if not spearman_path.exists() or not counts_path.exists():
        return
    spearman = pd.read_csv(spearman_path)
    counts = pd.read_csv(counts_path)
    holm = pd.read_csv(root / "results/association/holm_split_tests.csv")
    add(
        "M0",
        "Spearman tests whose dataset-bootstrap interval covers zero, and whose Holm-adjusted permutation p is below 0.05",
        (
            f"{int(spearman['interval_covers_zero'].sum())} of {len(spearman)} intervals cover zero; "
            f"{int((spearman['holm_p'] < 0.05).sum())} of {len(spearman)} Holm p-values are below 0.05"
        ),
        "`results/association/spearman.csv`",
    )
    for _, row in counts.iterrows():
        add(
            "M-" + str(row["family"]),
            f"Holm survivors in family {row['family']}",
            f"{int(row['n_holm_p_below_0.05'])} of {int(row['n_tests'])} (unadjusted p < 0.05: {int(row['n_raw_p_below_0.05'])})",
            "`results/association/holm_counts.csv`",
        )
    narrow = spearman[~spearman.interval_covers_zero]
    for i, row in enumerate(narrow.itertuples(index=False), start=1):
        add(
            f"M-rho-{i}",
            f"Spearman rho, {row.model} {row.arm_a} minus baseline Brier at sigma 1 versus {row.characteristic}",
            (
                f"rho {row.spearman_rho:.10g}; dataset-bootstrap interval [{row.bootstrap_lo:.10g}, {row.bootstrap_hi:.10g}]; "
                f"permutation p {row.permutation_p:.10g}; Holm p {row.holm_p:.10g}; "
                f"leave-one-out rho [{row.loo_rho_min:.10g}, {row.loo_rho_max:.10g}]; n_datasets {int(row.n_datasets)}"
            ),
            "`results/association/spearman.csv`",
        )
    survivors = holm[holm.holm_p < 0.05].sort_values(["family", "holm_p"])
    for i, row in enumerate(survivors.itertuples(index=False), start=1):
        dataset = row.dataset if isinstance(row.dataset, str) else "wdbc"
        add(
            f"M-holm-{i:02d}",
            f"Holm survivor, family {row.family}, {dataset}, {row.model}, {row.arm_a} minus baseline, sigma={float(row.sigma):g}, {row.metric}",
            f"mean_diff {row.mean_diff:.10g}; raw p {row.two_sided_p:.10g}; Holm p {row.holm_p:.10g}",
            "`results/association/holm_split_tests.csv`",
        )


def main() -> None:
    root = Path.cwd()
    text = build(root)
    path = root / "CLAIM_EVIDENCE_MATRIX.md"
    path.write_text(text)
    print(path, "lines", text.count("\n"))


if __name__ == "__main__":
    main()
