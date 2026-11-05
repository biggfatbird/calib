"""Task 1 (EXTENSION to the paper): K independent noise draws per (split, sigma), shared across models within a draw.

Same splits, models, sigma grid and metrics as reproduce_study.py. Fits each model once per split (noise only affects the
test matrix). Outputs results/noise_draws.csv (K*splits*sigmas*models rows) and results/noise_draw_summary.csv, and
results/paired_differences.csv. Uncertainty reported:
  * draw interval: 2.5-97.5 percentile, across noise draws, of the metric averaged over splits (noise-seed uncertainty)
  * split SD: SD across the 5 splits of the draw-averaged metric (split uncertainty; only 5 splits, so crude)
"""
import argparse, platform
from pathlib import Path
import numpy as np, pandas as pd
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
import reproduce_study as R

MODELS = ["Logistic regression", "Random forest", "RBF SVM"]
METRICS = ["roc_auc", "brier", "ece10", "sensitivity", "specificity", "f1", "accuracy"]


def run(K=50, n_splits=5, seed=R.SEED):
    X, y = R.load(); rows = []
    for s, (tr, te) in enumerate(StratifiedShuffleSplit(n_splits, test_size=0.25, random_state=seed).split(X, y), 1):
        fitted = {n: m.fit(X[tr], y[tr]) for n, m in R.models(seed).items()}
        sd = X[tr].std(axis=0, ddof=0)
        for sig in R.SIGMAS:
            for k in range(K):
                rng = np.random.default_rng([seed, s, k, int(round(sig * 100))])   # one matrix per (split, sigma, draw)
                Xn = X[te] + sig * sd * rng.standard_normal(X[te].shape)
                for n, m in fitted.items():
                    rows.append(dict(split=s, sigma=sig, draw=k, model=n, **R.metrics(y[te], m.predict_proba(Xn)[:, 1])))
    return pd.DataFrame(rows)


def summarize(df):
    per_draw = df.groupby(["model", "sigma", "draw"])[METRICS].mean()          # mean over splits, per draw
    per_split = df.groupby(["model", "sigma", "split"])[METRICS].mean()        # mean over draws, per split
    out = []
    for (m, sig), g in per_draw.groupby(level=[0, 1]):
        for met in METRICS:
            v = g[met].values; sp = per_split.loc[(m, sig), met].values
            out.append(dict(model=m, sigma=sig, metric=met, mean=v.mean(), draw_sd=v.std(ddof=1),
                            draw_lo=np.percentile(v, 2.5), draw_hi=np.percentile(v, 97.5), split_sd=sp.std(ddof=1)))
    return pd.DataFrame(out)


def paired(df, pairs=(("RBF SVM", "Logistic regression"), ("Random forest", "Logistic regression"), ("RBF SVM", "Random forest"))):
    per_draw = df.groupby(["model", "sigma", "draw"])[METRICS].mean()
    out = []
    for sig in R.SIGMAS:
        for a, b in pairs:
            for met in ("roc_auc", "brier", "ece10", "sensitivity", "f1"):
                d = per_draw.loc[(a, sig)][met].values - per_draw.loc[(b, sig)][met].values
                out.append(dict(sigma=sig, a=a, b=b, metric=met, mean_diff=d.mean(), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                                frac_a_greater=(d > 0).mean()))
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=50); ap.add_argument("--out", default="results")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True)
    df = run(a.draws); df.to_csv(out / "noise_draws.csv", index=False)
    summarize(df).to_csv(out / "noise_draw_summary.csv", index=False); paired(df).to_csv(out / "paired_differences.csv", index=False)
    (out / "noise_draws_env.txt").write_text(f"python {platform.python_version()}\nsklearn {sklearn.__version__}\nnumpy {np.__version__}\ndraws {a.draws}\n")
    print(len(df), "rows")
