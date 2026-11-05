"""Experiment runners.

Outputs go to the directory requested by the caller. This module never writes
into ``preserved/``.
"""

from __future__ import annotations

import json
import platform
import traceback
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from wdbc_calib.data import load_wdbc, worst_feature_mask
from wdbc_calib.metrics import classification_metrics, expected_cost
from wdbc_calib.models import CALIBRATIONS, DISPLAY_NAMES, MODEL_KEYS, make_estimator, positive_probability
from wdbc_calib.noise import (
    apply_correlated,
    apply_group_isotropic,
    apply_isotropic,
    apply_mean_dropout,
    apply_multiplicative,
    noise_seed,
    population_sd,
    standard_normal,
)
from wdbc_calib.splits import repeated_stratified_holdout
from wdbc_calib.stats import nadeau_bengio_split_test, paired_draw_summary, summarize_draws
from wdbc_calib.thresholds import select_cost_threshold

IST = timezone(timedelta(hours=5, minutes=30))
METRIC_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "sensitivity",
    "specificity",
    "f1",
    "brier",
    "ece10",
    "ece15",
    "ece_equal_mass10",
)


def _now() -> str:
    moment = datetime.now(timezone.utc)
    return moment.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S %Z") + " / " + moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _fit_arm(model_key: str, calibration: str, random_state: int, cv_seed: int, X_train, y_train):
    estimator = make_estimator(model_key, calibration, random_state, cv_seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(X_train, y_train)
    messages = []
    for item in caught:
        if issubclass(item.category, FutureWarning) and "probability" in str(item.message):
            messages.append(str(item.message))
    return estimator, messages


def run_calibration_experiment(config: dict, output_dir: Path) -> pd.DataFrame:
    """Baseline, sigmoid, and isotonic arms on shared splits and shared noise."""
    output_dir = Path(output_dir)
    if "preserved" in output_dir.parts:
        raise RuntimeError("Refusing to write into preserved/.")
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y, _feature_names = load_wdbc()
    n_splits = int(config["n_splits"])
    test_size = float(config["test_size"])
    master_seed = int(config["master_seed"])
    sigmas = [float(s) for s in config["sigmas"]]
    n_draws = int(config["n_draws"])
    model_keys = list(config.get("models", MODEL_KEYS))
    calibrations = list(config.get("calibrations", CALIBRATIONS))
    splits = repeated_stratified_holdout(y, n_splits, test_size, master_seed)
    split_rows = []
    for split_index, (train_idx, test_idx) in enumerate(splits):
        for role, idx in (("train", train_idx), ("test", test_idx)):
            for row_i in idx:
                split_rows.append({"split": split_index, "role": role, "row": int(row_i)})
    pd.DataFrame(split_rows).to_csv(output_dir / "split_indices.csv", index=False)

    warning_lines: list[str] = []
    records: list[dict] = []
    # probabilities: split, sigma, draw, arm -> vector. Stored after the loop via npz of stacked arrays.
    prob_store: dict[tuple, np.ndarray] = {}
    label_store: dict[int, np.ndarray] = {}

    for split_index, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        label_store[split_index] = y_test
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
                        f"split={split_index} model={model_key} calibration={calibration}: {message}"
                    )
        reference_probs: dict[tuple, np.ndarray] = {}
        for sigma_index, sigma in enumerate(sigmas):
            for draw in range(n_draws):
                seed = noise_seed(master_seed, stream=0, split_index=split_index, level_index=sigma_index, draw=draw)
                Z = standard_normal(X_test.shape, seed)
                X_noisy = apply_isotropic(X_test, train_sd, sigma, Z)
                for model_key in model_keys:
                    for calibration in calibrations:
                        p = positive_probability(fitted[(model_key, calibration)], X_noisy)
                        key = (split_index, sigma_index, draw, model_key, calibration)
                        if sigma == 0 and draw == 0:
                            reference_probs[(model_key, calibration)] = p
                        if sigma == 0 and draw > 0:
                            if not np.allclose(p, reference_probs[(model_key, calibration)], atol=1e-12):
                                raise RuntimeError(
                                    f"sigma=0 predictions changed across draws for {model_key}/{calibration} split {split_index}."
                                )
                        metrics = classification_metrics(y_test, p, threshold=0.5)
                        record = {
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
                        record.update({k: metrics[k] for k in METRIC_KEYS})
                        records.append(record)
                        prob_store[key] = p

    frame = pd.DataFrame.from_records(records)
    frame.to_csv(output_dir / "metrics_long.csv", index=False)
    summary = summarize_draws(frame, ["model", "model_key", "calibration", "sigma"])
    summary.to_csv(output_dir / "summary_draws.csv", index=False)
    paired_frames = []
    for left, right in (("sigmoid", "baseline"), ("isotonic", "baseline"), ("isotonic", "sigmoid")):
        paired_frames.append(
            paired_draw_summary(frame, "calibration", left, right, ["model", "model_key", "sigma"])
        )
    paired = pd.concat(paired_frames, ignore_index=True) if paired_frames else pd.DataFrame()
    paired.to_csv(output_dir / "paired_calibration.csv", index=False)
    _write_split_tests(frame, output_dir / "split_level_tests.csv")
    _write_reliability(prob_store, label_store, sigmas, model_keys, calibrations, output_dir / "reliability_bins.csv")
    _write_manifest(
        output_dir,
        config,
        {
            "n_rows_metrics": int(len(frame)),
            "svc_probability_warning": (
                "SVC(probability=True) raises FutureWarning in scikit-learn 1.9: "
                "the flag is deprecated and internally Platt-scales decision values. "
                "The baseline arm keeps it. Sigmoid and isotonic arms use "
                "CalibratedClassifierCV(..., ensemble=False) on an uncalibrated base."
            ),
            "noise": "Isotropic Gaussian in original feature units, sigma times training population SD. One Z shared by all models and calibrations.",
            "threshold": 0.5,
            "positive_class": "malignant = 1",
        },
    )
    (output_dir / "warnings.txt").write_text("\n".join(warning_lines) + ("\n" if warning_lines else ""))
    return frame


def _write_split_tests(frame: pd.DataFrame, path: Path) -> None:
    """Nadeau–Bengio tests on split-level means (draws averaged first). Exploratory."""
    rows = []
    metrics = ["brier", "ece10", "ece15", "ece_equal_mass10", "roc_auc", "sensitivity", "f1"]
    contrasts = (("sigmoid", "baseline"), ("isotonic", "baseline"))
    for (model, sigma), group in frame.groupby(["model", "sigma"], sort=True):
        n_train = int(group["n_train"].iloc[0])
        n_test = int(group["n_test"].iloc[0])
        for arm_a, arm_b in contrasts:
            left = group[group.calibration == arm_a]
            right = group[group.calibration == arm_b]
            if left.empty or right.empty:
                continue
            merged = left.merge(right, on=["split", "draw"], suffixes=("_a", "_b"))
            for metric in metrics:
                per_split = merged.assign(_diff=merged[f"{metric}_a"] - merged[f"{metric}_b"]).groupby("split")["_diff"].mean()
                if len(per_split) < 2:
                    rows.append(
                        {
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
                        "model": model,
                        "sigma": float(sigma),
                        "arm_a": arm_a,
                        "arm_b": arm_b,
                        "metric": metric,
                        "note": "Exploratory corrected resampled t-test. Not multiplicity-adjusted. Unit is the split after averaging draws.",
                    }
                )
                rows.append(result)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_reliability(prob_store, label_store, sigmas, model_keys, calibrations, path: Path) -> None:
    """Equal-width reliability, pooled over splits and draws.

    Counts are prediction-rows, not distinct patients. Splits overlap and draws
    repeat the same test rows.
    """
    rows = []
    n_bins = 10
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for sigma_index, sigma in enumerate(sigmas):
        for model_key in model_keys:
            for calibration in calibrations:
                chunks_p = []
                chunks_y = []
                for (split_index, s_index, _draw, m_key, cal), prob in prob_store.items():
                    if s_index == sigma_index and m_key == model_key and cal == calibration:
                        chunks_p.append(prob)
                        chunks_y.append(label_store[split_index])
                if not chunks_p:
                    continue
                p = np.concatenate(chunks_p)
                yy = np.concatenate(chunks_y)
                bin_ids = np.digitize(p, edges[1:-1], right=False)
                for b in range(n_bins):
                    mask = bin_ids == b
                    count = int(mask.sum())
                    rows.append(
                        {
                            "sigma": sigma,
                            "model": DISPLAY_NAMES[model_key],
                            "model_key": model_key,
                            "calibration": calibration,
                            "bin": b,
                            "edge_left": float(edges[b]),
                            "edge_right": float(edges[b + 1]),
                            "count_prediction_rows": count,
                            "mean_predicted": float(p[mask].mean()) if count else np.nan,
                            "observed_frequency": float(yy[mask].mean()) if count else np.nan,
                            "pooling": "splits x draws x test rows; not independent patients",
                        }
                    )
    pd.DataFrame(rows).to_csv(path, index=False)


def run_corruption_experiment(config: dict, output_dir: Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    if "preserved" in output_dir.parts:
        raise RuntimeError("Refusing to write into preserved/.")
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y, feature_names = load_wdbc()
    mask = worst_feature_mask(feature_names)
    master_seed = int(config["master_seed"])
    splits = repeated_stratified_holdout(y, int(config["n_splits"]), float(config["test_size"]), master_seed)
    model_keys = list(config.get("models", MODEL_KEYS))
    n_draws = int(config["n_draws"])
    records = []
    failures = []
    specs = []
    for sigma in config["gaussian_sigmas"]:
        specs.append(("isotropic_gaussian", float(sigma), 0))
        specs.append(("correlated_gaussian", float(sigma), 1))
        specs.append(("multiplicative_gaussian", float(sigma), 2))
        specs.append(("worst10_isotropic_gaussian", float(sigma), 4))
    for rate in config["dropout_rates"]:
        specs.append(("train_mean_dropout", float(rate), 3))

    for split_index, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        train_sd = population_sd(X_train)
        train_mean = X_train.mean(axis=0)
        fitted = {}
        for model_key in model_keys:
            estimator, _messages = _fit_arm(
                model_key,
                "baseline",
                random_state=master_seed + 100 + MODEL_KEYS.index(model_key),
                cv_seed=master_seed + 200,
                X_train=X_train,
                y_train=y_train,
            )
            fitted[model_key] = estimator
        # level index is the position within its corruption family, not a global index
        family_levels: dict[str, list[float]] = {}
        for name, level, _stream in specs:
            family_levels.setdefault(name, [])
            if level not in family_levels[name]:
                family_levels[name].append(level)
        for name, level, stream in specs:
            level_index = family_levels[name].index(level)
            for draw in range(n_draws):
                seed = noise_seed(master_seed, stream, split_index, level_index, draw)
                try:
                    Z = standard_normal(X_test.shape, seed)
                    if name == "isotropic_gaussian":
                        X_noisy = apply_isotropic(X_test, train_sd, level, Z)
                    elif name == "correlated_gaussian":
                        X_noisy = apply_correlated(X_test, X_train, level, Z)
                    elif name == "multiplicative_gaussian":
                        X_noisy = apply_multiplicative(X_test, level, Z)
                    elif name == "train_mean_dropout":
                        X_noisy = apply_mean_dropout(X_test, train_mean, level, seed)
                    elif name == "worst10_isotropic_gaussian":
                        X_noisy = apply_group_isotropic(X_test, train_sd, level, Z, mask)
                    else:
                        raise ValueError(name)
                except Exception as exc:  # numerical failure of one condition is recorded, not dropped silently
                    failures.append(
                        {
                            "split": split_index,
                            "draw": draw,
                            "corruption": name,
                            "level": level,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    )
                    continue
                for model_key in model_keys:
                    p = positive_probability(fitted[model_key], X_noisy)
                    metrics = classification_metrics(y_test, p, threshold=0.5)
                    record = {
                        "split": split_index,
                        "draw": draw,
                        "corruption": name,
                        "level": level,
                        "model": DISPLAY_NAMES[model_key],
                        "model_key": model_key,
                        "n_test": int(len(test_idx)),
                    }
                    record.update({k: metrics[k] for k in METRIC_KEYS})
                    records.append(record)
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(output_dir / "metrics_long.csv", index=False)
    if len(frame):
        summarize_draws(frame, ["corruption", "level", "model", "model_key"]).to_csv(
            output_dir / "summary_draws.csv", index=False
        )
    pd.DataFrame(failures).to_csv(output_dir / "failures.csv", index=False)
    _write_manifest(
        output_dir,
        config,
        {
            "n_rows_metrics": int(len(frame)),
            "n_failures": int(len(failures)),
            "models": "baseline probability arms only",
            "correlated_noise": "population covariance of the training fold, plus a 1e-8 relative ridge before Cholesky",
            "multiplicative_noise": "X * (1 + level * Z); negative values are kept",
            "dropout": "each feature dropped independently and replaced by the training-fold mean",
            "worst_group": "isotropic noise on the 10 features whose names start with 'worst '",
        },
    )
    return frame


def run_threshold_experiment(config: dict, output_dir: Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    if "preserved" in output_dir.parts:
        raise RuntimeError("Refusing to write into preserved/.")
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y, _names = load_wdbc()
    master_seed = int(config["master_seed"])
    splits = repeated_stratified_holdout(y, int(config["n_splits"]), float(config["test_size"]), master_seed)
    model_keys = list(config.get("models", MODEL_KEYS))
    sigmas = [float(s) for s in config["sigmas"]]
    n_draws = int(config["n_draws"])
    ratios = [tuple(item) for item in config["fn_fp_cost_ratios"]]
    records = []
    chosen_rows = []
    for split_index, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        train_sd = population_sd(X_train)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=master_seed + 200)
        for model_key in model_keys:
            estimator = make_estimator(
                model_key,
                "baseline",
                random_state=master_seed + 100 + MODEL_KEYS.index(model_key),
                cv_seed=master_seed + 200,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                oof = cross_val_predict(
                    clone(estimator),
                    X_train,
                    y_train,
                    cv=cv,
                    method="predict_proba",
                    n_jobs=1,
                )
            # Column order follows sorted class labels. Malignant is 1.
            if oof.shape[1] != 2:
                raise RuntimeError("Expected binary probabilities.")
            p_oof = oof[:, 1]
            fitted, _messages = _fit_arm(
                model_key,
                "baseline",
                random_state=master_seed + 100 + MODEL_KEYS.index(model_key),
                cv_seed=master_seed + 200,
                X_train=X_train,
                y_train=y_train,
            )
            thresholds = {}
            for cost_fn, cost_fp in ratios:
                threshold, train_cost = select_cost_threshold(y_train, p_oof, float(cost_fn), float(cost_fp))
                thresholds[(int(cost_fn), int(cost_fp))] = threshold
                chosen_rows.append(
                    {
                        "split": split_index,
                        "model": DISPLAY_NAMES[model_key],
                        "model_key": model_key,
                        "cost_fn": int(cost_fn),
                        "cost_fp": int(cost_fp),
                        "threshold": threshold,
                        "oof_total_cost": train_cost,
                        "n_train": int(len(train_idx)),
                        "selection": "out-of-fold training predictions only",
                    }
                )
            for sigma_index, sigma in enumerate(sigmas):
                for draw in range(n_draws):
                    seed = noise_seed(master_seed, 0, split_index, sigma_index, draw)
                    Z = standard_normal(X_test.shape, seed)
                    X_noisy = apply_isotropic(X_test, train_sd, sigma, Z)
                    p_test = positive_probability(fitted, X_noisy)
                    for (cost_fn, cost_fp), threshold in thresholds.items():
                        metrics = classification_metrics(y_test, p_test, threshold=threshold)
                        cost = expected_cost(y_test, p_test, threshold, cost_fn, cost_fp)
                        fixed = classification_metrics(y_test, p_test, threshold=0.5)
                        records.append(
                            {
                                "split": split_index,
                                "sigma": sigma,
                                "draw": draw,
                                "model": DISPLAY_NAMES[model_key],
                                "model_key": model_key,
                                "cost_fn": cost_fn,
                                "cost_fp": cost_fp,
                                "threshold": threshold,
                                "mean_cost": cost,
                                "sensitivity": metrics["sensitivity"],
                                "specificity": metrics["specificity"],
                                "f1": metrics["f1"],
                                "fn": metrics["fn"],
                                "fp": metrics["fp"],
                                "sensitivity_at_0.5": fixed["sensitivity"],
                                "specificity_at_0.5": fixed["specificity"],
                                "n_test": int(len(test_idx)),
                            }
                        )
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(output_dir / "metrics_long.csv", index=False)
    pd.DataFrame(chosen_rows).to_csv(output_dir / "thresholds_train.csv", index=False)
    if len(frame):
        summarize_draws(
            frame,
            ["model", "model_key", "sigma", "cost_fn", "cost_fp"],
            metrics=("mean_cost", "sensitivity", "specificity", "f1"),
        ).to_csv(output_dir / "summary_draws.csv", index=False)
    _write_manifest(
        output_dir,
        config,
        {
            "n_rows_metrics": int(len(frame)),
            "threshold_rule": "minimize cost_fn*FN + cost_fp*FP on training out-of-fold predictions; ties -> closer to 0.5, then smaller threshold",
            "test_labels_used_for_selection": False,
        },
    )
    return frame


def run_more_splits(config: dict, output_dir: Path) -> pd.DataFrame:
    """Baseline models on a longer repeated-holdout sequence. The first splits match the primary seed."""
    narrowed = dict(config)
    narrowed["calibrations"] = ["baseline"]
    frame = run_calibration_experiment(narrowed, output_dir)
    # Curve of split-sample SD using the first k splits, at each sigma, clean of draw noise by averaging draws.
    rows = []
    for (model, sigma), group in frame.groupby(["model", "sigma"], sort=True):
        per_split = group.groupby("split")["roc_auc"].mean().sort_index()
        brier = group.groupby("split")["brier"].mean().sort_index()
        ece = group.groupby("split")["ece10"].mean().sort_index()
        for k in range(2, len(per_split) + 1):
            rows.append(
                {
                    "model": model,
                    "sigma": float(sigma),
                    "k_splits": k,
                    "roc_auc_mean": float(per_split.iloc[:k].mean()),
                    "roc_auc_split_sd": float(per_split.iloc[:k].std(ddof=1)),
                    "brier_mean": float(brier.iloc[:k].mean()),
                    "brier_split_sd": float(brier.iloc[:k].std(ddof=1)),
                    "ece10_mean": float(ece.iloc[:k].mean()),
                    "ece10_split_sd": float(ece.iloc[:k].std(ddof=1)),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "split_sd_curve.csv", index=False)
    return frame


def compare_to_preserved(summary: pd.DataFrame, preserved_summary: Path, preserved_draws: Path, output_path: Path) -> pd.DataFrame:
    """Differences between this protocol's baseline arm and the archived CSVs. Not a claim of recovery."""
    archived_single = pd.read_csv(preserved_summary)
    archived_draws = pd.read_csv(preserved_draws)
    ours = summary[(summary.calibration == "baseline")].copy()
    rows = []
    for _, row in ours.iterrows():
        metric = row["metric"]
        if metric not in archived_single.columns and f"{metric}_mean" not in archived_single.columns:
            # archived single-draw file uses metric_mean column names
            pass
        single = archived_single[
            (archived_single.model == row["model"]) & (np.isclose(archived_single.sigma, row["sigma"]))
        ]
        draws = archived_draws[
            (archived_draws.model == row["model"])
            & (np.isclose(archived_draws.sigma, row["sigma"]))
            & (archived_draws.metric == metric)
        ]
        record = {
            "model": row["model"],
            "sigma": row["sigma"],
            "metric": metric,
            "explicit_protocol_mean": row["mean"],
            "explicit_protocol_draw_lo": row["draw_lo"],
            "explicit_protocol_draw_hi": row["draw_hi"],
        }
        mean_col = f"{metric}_mean"
        if len(single) and mean_col in single.columns:
            record["preserved_single_draw_split_mean"] = float(single.iloc[0][mean_col])
            record["delta_vs_preserved_single_draw"] = record["explicit_protocol_mean"] - record["preserved_single_draw_split_mean"]
        else:
            record["preserved_single_draw_split_mean"] = np.nan
            record["delta_vs_preserved_single_draw"] = np.nan
        if len(draws):
            record["preserved_50draw_mean"] = float(draws.iloc[0]["mean"])
            record["delta_vs_preserved_50draw"] = record["explicit_protocol_mean"] - record["preserved_50draw_mean"]
        else:
            record["preserved_50draw_mean"] = np.nan
            record["delta_vs_preserved_50draw"] = np.nan
        rows.append(record)
    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    return out


def _write_manifest(output_dir: Path, config: dict, notes: dict) -> None:
    payload = {
        "written_at_ist_and_utc": _now(),
        "versions": library_versions(),
        "config": config,
        "notes": notes,
        "package_version": "0.2.0",
    }
    _write_json(output_dir / "manifest.json", payload)
    (output_dir / "RUN_LOG.txt").write_text(
        f"completed {_now()}\nversions {library_versions()}\n"
    )
