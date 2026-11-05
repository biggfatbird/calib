"""Same calibration-under-noise protocol on several public binary tables.

The seed formula, estimators, and noise generator are the ones used for WDBC.
The dataset name is not part of the seed. On WDBC the rows must match
``results/recalibration/`` when the sigma grid, draw count, and master seed match.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from wdbc_calib.experiments import METRIC_KEYS, _fit_arm, _write_manifest
from wdbc_calib.stats import nadeau_bengio_split_test
from wdbc_calib.metrics import classification_metrics
from wdbc_calib.models import CALIBRATIONS, DISPLAY_NAMES, MODEL_KEYS, positive_probability
from wdbc_calib.noise import apply_isotropic, noise_seed, population_sd, standard_normal
from wdbc_calib.splits import repeated_stratified_holdout
from wdbc_calib.stats import paired_draw_summary, summarize_draws
from wdbc_calib.tabular_datasets import BinaryTable, load_study_tables


def run_cross_dataset_experiment(
    config: dict,
    output_dir: Path,
    tables: list[BinaryTable] | None = None,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    if "preserved" in output_dir.parts:
        raise RuntimeError("Refusing to write into preserved/.")
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = list(tables) if tables is not None else load_study_tables()
    if not tables:
        raise RuntimeError("No tables to run.")
    n_splits = int(config["n_splits"])
    test_size = float(config["test_size"])
    master_seed = int(config["master_seed"])
    sigmas = [float(s) for s in config["sigmas"]]
    n_draws = int(config["n_draws"])
    model_keys = list(config.get("models", MODEL_KEYS))
    calibrations = list(config.get("calibrations", CALIBRATIONS))

    card = pd.DataFrame([table.card_row() for table in tables])
    card.to_csv(output_dir / "dataset_card.csv", index=False)

    records: list[dict] = []
    warning_lines: list[str] = []
    for table in tables:
        print(f"cross-dataset {table.key}: {table.n_rows} rows, {table.n_positive} positive", flush=True)
        X, y = table.X, table.y
        if len(np.unique(y)) < 2:
            raise RuntimeError(f"{table.key} is not binary.")
        splits = repeated_stratified_holdout(y, n_splits, test_size, master_seed)
        for split_index, (train_idx, test_idx) in enumerate(splits):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                raise RuntimeError(f"{table.key} split {split_index} lost a class.")
            train_sd = population_sd(X_train)
            fitted = {}
            for model_key in model_keys:
                for calibration in calibrations:
                    estimator, messages = _fit_arm(
                        model_key,
                        calibration,
                        random_state=master_seed + 100 + MODEL_KEYS.index(model_key),
                        cv_seed=master_seed + 200,
                        X_train=X_train,
                        y_train=y_train,
                    )
                    fitted[(model_key, calibration)] = estimator
                    for message in messages:
                        warning_lines.append(
                            f"dataset={table.key} split={split_index} model={model_key} calibration={calibration}: {message}"
                        )
            reference: dict[tuple, np.ndarray] = {}
            for sigma_index, sigma in enumerate(sigmas):
                for draw in range(n_draws):
                    seed = noise_seed(master_seed, stream=0, split_index=split_index, level_index=sigma_index, draw=draw)
                    Z = standard_normal(X_test.shape, seed)
                    X_noisy = apply_isotropic(X_test, train_sd, sigma, Z)
                    for model_key in model_keys:
                        for calibration in calibrations:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", FutureWarning)
                                probability = positive_probability(fitted[(model_key, calibration)], X_noisy)
                            if sigma == 0 and draw == 0:
                                reference[(model_key, calibration)] = probability
                            if sigma == 0 and draw > 0:
                                if not np.allclose(probability, reference[(model_key, calibration)], atol=1e-12):
                                    raise RuntimeError(
                                        f"sigma=0 predictions changed for {table.key} {model_key}/{calibration}."
                                    )
                            metrics = classification_metrics(y_test, probability, threshold=0.5)
                            record = {
                                "dataset": table.key,
                                "split": split_index,
                                "sigma": sigma,
                                "draw": draw,
                                "model": DISPLAY_NAMES[model_key],
                                "model_key": model_key,
                                "calibration": calibration,
                                "n_train": int(len(train_idx)),
                                "n_test": int(len(test_idx)),
                                "threshold": 0.5,
                            }
                            record.update({key: metrics[key] for key in METRIC_KEYS})
                            records.append(record)

    frame = pd.DataFrame.from_records(records)
    frame.to_csv(output_dir / "metrics_long.csv", index=False)
    summary = summarize_draws(frame, ["dataset", "model", "model_key", "calibration", "sigma"])
    summary.to_csv(output_dir / "summary_draws.csv", index=False)
    paired_frames = []
    for left, right in (("sigmoid", "baseline"), ("isotonic", "baseline"), ("isotonic", "sigmoid")):
        paired_frames.append(
            paired_draw_summary(
                frame,
                "calibration",
                left,
                right,
                ["dataset", "model", "model_key", "sigma"],
            )
        )
    paired = pd.concat(paired_frames, ignore_index=True)
    paired.to_csv(output_dir / "paired_calibration.csv", index=False)
    _write_dataset_split_tests(frame, output_dir / "split_level_tests.csv")
    (output_dir / "warnings.txt").write_text("\n".join(warning_lines) + ("\n" if warning_lines else ""))
    _write_manifest(
        output_dir,
        config,
        {
            "n_rows_metrics": int(len(frame)),
            "datasets": [table.key for table in tables],
            "question": (
                "Calibrators are fit on clean training rows only and scored under isotropic "
                "Gaussian feature noise. The pre-specified contrasts are isotonic minus baseline "
                "for logistic regression (Brier and ROC-AUC) and sigmoid minus baseline for the "
                "random forest and for logistic regression (Brier), at sigma 0 and sigma 1."
            ),
            "seed": "Identical to the WDBC recalibration seed. The dataset name is not an input.",
            "wdbc_positive_class": "malignant = 1",
        },
    )
    if config.get("compare_wdbc_to"):
        _compare_wdbc(frame, Path(config["compare_wdbc_to"]), output_dir / "wdbc_agreement.csv")
    return frame


def _write_dataset_split_tests(frame: pd.DataFrame, path: Path) -> None:
    """Nadeau–Bengio tests within one dataset. Datasets are not pooled."""
    rows = []
    metrics = ["brier", "ece10", "roc_auc", "sensitivity", "f1"]
    contrasts = (("sigmoid", "baseline"), ("isotonic", "baseline"))
    group_cols = ["dataset", "model", "sigma"]
    for keys, group in frame.groupby(group_cols, sort=True):
        dataset, model, sigma = keys
        n_train = int(group["n_train"].iloc[0])
        n_test = int(group["n_test"].iloc[0])
        for arm_a, arm_b in contrasts:
            left = group[group.calibration == arm_a]
            right = group[group.calibration == arm_b]
            if left.empty or right.empty:
                continue
            merged = left.merge(right, on=["split", "draw"], suffixes=("_a", "_b"))
            for metric in metrics:
                per_split = (
                    merged.assign(_diff=merged[f"{metric}_a"] - merged[f"{metric}_b"]).groupby("split")["_diff"].mean()
                )
                if len(per_split) < 2:
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "sigma": float(sigma),
                            "arm_a": arm_a,
                            "arm_b": arm_b,
                            "metric": metric,
                            "n_splits": int(len(per_split)),
                            "note": "Not computed: the corrected resampled t-test needs at least two splits.",
                        }
                    )
                    continue
                result = nadeau_bengio_split_test(per_split.to_numpy(), n_train, n_test)
                result.update(
                    {
                        "dataset": dataset,
                        "model": model,
                        "sigma": float(sigma),
                        "arm_a": arm_a,
                        "arm_b": arm_b,
                        "metric": metric,
                        "note": "Exploratory corrected resampled t-test within one dataset. Not multiplicity-adjusted.",
                    }
                )
                rows.append(result)
    pd.DataFrame(rows).to_csv(path, index=False)


def _compare_wdbc(frame: pd.DataFrame, primary_metrics: Path, output_path: Path) -> None:
    """WDBC rows from this run against the primary recalibration file, on shared keys."""
    if not primary_metrics.exists():
        pd.DataFrame([{"note": f"missing {primary_metrics}"}]).to_csv(output_path, index=False)
        return
    primary = pd.read_csv(primary_metrics)
    ours = frame[frame.dataset == "wdbc"]
    keys = ["split", "sigma", "draw", "model", "calibration"]
    metrics = ["brier", "roc_auc", "ece10", "sensitivity"]
    merged = ours.merge(primary, on=keys, suffixes=("_cross", "_primary"))
    rows = []
    for metric in metrics:
        delta = (merged[f"{metric}_cross"] - merged[f"{metric}_primary"]).abs()
        rows.append(
            {
                "metric": metric,
                "n_rows": int(len(merged)),
                "max_abs_diff": float(delta.max()) if len(delta) else np.nan,
            }
        )
    report = pd.DataFrame(rows)
    report.to_csv(output_path, index=False)
    if len(merged) == 0 or float(report["max_abs_diff"].max()) > 1e-12:
        raise RuntimeError(f"WDBC cross-dataset run does not match {primary_metrics}.")
