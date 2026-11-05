"""Estimators.

Baseline RBF SVM uses ``SVC(probability=True)``. In scikit-learn 1.9 that flag
is deprecated because it already runs an internal sigmoid (Platt) calibration
on out-of-fold decision values, with ``ensemble=False`` semantics. The baseline
keeps the flag so the arm stays comparable to that historical API.

Sigmoid and isotonic arms wrap a base estimator in ``CalibratedClassifierCV``
with ``ensemble=False`` and a stratified 5-fold split of the training fold
only. The SVM base in those arms uses ``probability=False``, so the decision
function is calibrated once. Wrapping ``probability=True`` would calibrate
an already calibrated score and is not done.

``ensemble=False`` refits the base estimator on the whole training fold and
fits the calibrator on out-of-fold predictions. That separates recalibration
from the extra averaging that ``ensemble=True`` would introduce.
"""

from __future__ import annotations

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_KEYS = ("logistic_regression", "random_forest", "rbf_svm")
DISPLAY_NAMES = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "rbf_svm": "RBF SVM",
}
CALIBRATIONS = ("baseline", "sigmoid", "isotonic")


def _logistic() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, solver="lbfgs")),
        ]
    )


def _forest(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=1,
    )


def _svm(random_state: int, probability: bool) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(kernel="rbf", probability=probability, random_state=random_state),
            ),
        ]
    )


def make_estimator(model_key: str, calibration: str, random_state: int, cv_seed: int):
    if model_key not in MODEL_KEYS:
        raise ValueError(f"Unknown model {model_key}")
    if calibration not in CALIBRATIONS:
        raise ValueError(f"Unknown calibration {calibration}")

    if calibration == "baseline":
        if model_key == "logistic_regression":
            return _logistic()
        if model_key == "random_forest":
            return _forest(random_state)
        return _svm(random_state, probability=True)

    if model_key == "logistic_regression":
        base = _logistic()
    elif model_key == "random_forest":
        base = _forest(random_state)
    else:
        base = _svm(random_state, probability=False)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed)
    return CalibratedClassifierCV(
        estimator=base,
        method=calibration,
        cv=cv,
        ensemble=False,
    )


def positive_probability(estimator, X, positive_label: int = 1):
    """Column of predict_proba that corresponds to malignant = 1."""
    proba = estimator.predict_proba(X)
    classes = list(estimator.classes_)
    if positive_label not in classes:
        raise RuntimeError(f"Positive label {positive_label} not in classes_ {classes}.")
    return proba[:, classes.index(positive_label)]


def fresh(estimator):
    return clone(estimator)
