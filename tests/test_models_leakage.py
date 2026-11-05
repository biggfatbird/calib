"""Training-only fitting and malignant-positive probabilities."""

import warnings

import numpy as np
from sklearn.model_selection import train_test_split

from wdbc_calib.data import N_MALIGNANT, load_wdbc
from wdbc_calib.models import make_estimator, positive_probability


def _split():
    X, y, _names = load_wdbc()
    assert int(y.sum()) == N_MALIGNANT
    idx = np.arange(len(y))
    train, test = train_test_split(idx, test_size=0.25, stratify=y, random_state=0)
    return X, y, train, test


def test_scaler_uses_training_rows_only_for_baseline_and_sigmoid():
    X, y, train, test = _split()
    X_poison = X.copy()
    X_poison[test, 0] = 10_000
    for calibration in ("baseline", "sigmoid"):
        estimator = make_estimator("logistic_regression", calibration, random_state=0, cv_seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            estimator.fit(X_poison[train], y[train])
        if calibration == "baseline":
            mean0 = estimator.named_steps["scaler"].mean_[0]
        else:
            mean0 = estimator.calibrated_classifiers_[0].estimator.named_steps["scaler"].mean_[0]
        assert abs(mean0 - X_poison[train, 0].mean()) < 1e-8
        assert abs(mean0 - X_poison[:, 0].mean()) > 1


def test_positive_probability_is_the_malignant_column():
    X, y, train, test = _split()
    estimator = make_estimator("logistic_regression", "baseline", random_state=0, cv_seed=1)
    estimator.fit(X[train], y[train])
    assert list(estimator.classes_) == [0, 1]
    p = positive_probability(estimator, X[test])
    assert np.allclose(p, estimator.predict_proba(X[test])[:, 1])
    # A row with a very large malignant-direction score is not required; the column
    # index itself is the contract. Ranking against labels should exceed chance.
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y[test], p) > 0.9


def test_baseline_svm_keeps_internal_probability_and_warns():
    X, y, train, _test = _split()
    estimator = make_estimator("rbf_svm", "baseline", random_state=0, cv_seed=1)
    assert estimator.named_steps["clf"].probability is True
    sigmoid = make_estimator("rbf_svm", "sigmoid", random_state=0, cv_seed=1)
    assert sigmoid.estimator.named_steps["clf"].probability is False
    assert sigmoid.ensemble is False
    assert sigmoid.method == "sigmoid"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(X[train], y[train])
    assert any(issubclass(item.category, FutureWarning) and "probability" in str(item.message) for item in caught)
