"""Reproduce 'When Accuracy Is Not Enough' (M. Kanwar, 4 Nov 2026): calibration and noise robustness on WDBC.

Protocol (paper Sec. 4 / Appendix A): StratifiedShuffleSplit(5, test 0.25, rs=2026); malignant = positive class 1;
LR(C=1, max_iter=3000)+scaler, RF(200, min_samples_leaf=2), SVC(rbf, C=1, probability=True)+scaler; Gaussian test noise
sigma in {0,.25,.5,1} x training-feature SD, one perturbation matrix per (split, sigma) shared by all models; threshold 0.5.
The paper does not state the RNG seeding of the noise or the model random_state values; those are assumptions below.
"""
from pathlib import Path
import argparse
import numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SIGMAS = (0.0, 0.25, 0.5, 1.0)
SEED = 2026


def load():
    d = load_breast_cancer()
    return d.data, (1 - d.target).astype(int)          # sklearn: 0=malignant -> recode malignant=1


def models(rs=SEED):
    return {"Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)),
            "Random forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=rs),
            "RBF SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=rs))}


def ece(y, p, bins=10):
    """Paper formula: sum_b (n_b/n)|freq_b - conf_b| over 10 equal-width bins of p (freq=observed positive rate, conf=mean p)."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def metrics(y, p):
    yh = (p >= 0.5).astype(int); tp = ((yh == 1) & (y == 1)).sum(); fn = ((yh == 0) & (y == 1)).sum()
    tn = ((yh == 0) & (y == 0)).sum(); fp = ((yh == 1) & (y == 0)).sum()
    return dict(accuracy=accuracy_score(y, yh), balanced_accuracy=balanced_accuracy_score(y, yh), roc_auc=roc_auc_score(y, p),
                sensitivity=tp / (tp + fn), specificity=tn / (tn + fp), f1=f1_score(y, yh), brier=brier_score_loss(y, p), ece10=ece(y, p))


def run(n_splits=5, seed=SEED):
    X, y = load(); rows = []
    for s, (tr, te) in enumerate(StratifiedShuffleSplit(n_splits, test_size=0.25, random_state=seed).split(X, y), 1):
        fitted = {n: m.fit(X[tr], y[tr]) for n, m in models(seed).items()}
        sd = X[tr].std(axis=0, ddof=0)
        for sig in SIGMAS:
            rng = np.random.default_rng([seed, s, int(sig * 100)])      # ASSUMPTION: paper does not give the noise seeding
            Xn = X[te] + sig * sd * rng.standard_normal(X[te].shape)
            for n, m in fitted.items():
                rows.append(dict(split=s, sigma=sig, model=n, **metrics(y[te], m.predict_proba(Xn)[:, 1])))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); ap.add_argument("--splits", type=int, default=5)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True)
    df = run(a.splits); df.to_csv(out / "results_by_split.csv", index=False)
    g = df.groupby(["model", "sigma"]).agg(["mean", "std"]).drop(columns="split"); g.columns = ["_".join(c) for c in g.columns]
    g.to_csv(out / "results_summary.csv"); pd.set_option("display.width", 200)
    print(df.groupby(["model", "sigma"])[["accuracy", "roc_auc", "sensitivity", "f1", "brier", "ece10"]].mean().round(3))
