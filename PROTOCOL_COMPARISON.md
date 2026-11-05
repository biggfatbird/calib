# Comparison of the original scripts with `wdbc_calib`

Read and executed on 5 November 2026, 22:04 IST (16:34 UTC).

The scripts compared are the bytes added in commit `636d281` (`Add files via upload`, authored 5 November 2026, 21:59 IST) and then moved, without an edit, to `preserved/original_source/`. SHA-256:

- `reproduce_study.py` `7fabe0430624ec7c0117f1c846ab06564c82f0ef8fee9caf44ec1b12a6934e46`
- `noise_draws.py` `0e2ddeb6cf1fd9ffad5e6f27ba0a198138a37f0740bd81aa7376b5731de7dae7`
- `make_figures_task1.py` `81aa7b47eeed5f994ac4a4707dfe8c3eeb92f67fcb8459aa2dbfe773b3cd7f5e`

`wdbc_calib/` was not edited to match them. `preserved/results/` was not rewritten.

## Which numbers are a direct reproduction

`reproduce_study.py` and `noise_draws.py` were run from `preserved/original_source/` with output directed to a temporary directory (5 splits, 50 draws). Parsed values were compared with `preserved/results/`. The largest absolute difference in any numeric column is 2.1×10⁻¹⁶ (`noise_draws.csv`, column `ece10`). No cell differs by more than 10⁻¹². The per-file maxima are in `results/original_source_check/numeric_agreement.csv`.

This workspace is Python 3.12.3, scikit-learn 1.9.1, NumPy 2.2.6. The archived stamp `preserved/results/noise_draws_env.txt` is Python 3.13.18, scikit-learn 1.9.1, NumPy 2.5.3. The tables match numerically across that version gap. The CSV text does not: a few floats differ in the last written digit, and the environment file records the interpreter that wrote it. The archived bytes stay the archive.

`results_by_split.csv` is the output of `reproduce_study.py`. `noise_draws.csv`, `noise_draw_summary.csv`, and `paired_differences.csv` are the output of `noise_draws.py`. `results_summary.csv` is the split mean and sample standard deviation written by `reproduce_study.py`.

Those files are not outputs of `wdbc_calib/`. Everything under `results/recalibration/`, `results/corruptions/`, `results/thresholds/`, and `results/more_splits/` belongs to the later protocol (master seed `20260311`).

`make_figures_task1.py` reads `results/noise_draw_summary.csv` and writes a four-panel figure whose title, axes, and three series match `preserved/figures/uploaded_noise_panel.png`. A fresh run on this machine is 2400×576 RGB (figsize 15×3.6 inches at 160 dpi). The archived PNG is 2000×480 RGBA. It was not replaced and is not claimed as a byte copy of this run.

At sigma > 0, no row of the one-draw file matches a row of the 50-draw file. The smallest absolute Brier difference between a one-draw row and the 50-draw rows with the same split, sigma, and model is about 8.9×10⁻⁸. The scripts use different noise seeds, quoted below.

## Same choices

- Data: `sklearn.datasets.load_breast_cancer`. Malignant is coded 1. The original uses `1 - target`. `wdbc_calib.data.load_wdbc` uses `target == 0`. For this dataset those are the same 212 rows.
- Splitter: `StratifiedShuffleSplit`, 5 splits, `test_size=0.25` (143 test rows).
- Logistic regression and the RBF SVM are `StandardScaler` pipelines fit on the training rows of that split only. The forest is fit on the unscaled training rows.
- Noise scale is the training-fold population standard deviation (`std(axis=0, ddof=0)`), in original feature units: `X_test + sigma * s_train * Z`.
- Sigma grid: 0, 0.25, 0.5, 1.
- One normal matrix is drawn per (split, sigma, draw) and reused for every model in that draw. The model name is not an input to the seed in either protocol.
- Models are fit once per split. Noise is applied only at prediction.
- Hard labels use threshold 0.5 on P(malignant). The original scripts do not choose that threshold from labels.
- Metrics at that threshold: accuracy, balanced accuracy, ROC-AUC, sensitivity, specificity, F1, Brier score, 10-bin ECE.
- The 10-bin ECE implementations agree on probabilities in [0, 1]. The original bins with `(p * 10)` clipped to 9. `wdbc_calib.metrics.expected_calibration_error` uses `linspace(0, 1, 11)` and `digitize`. On 160 random samples that included the endpoints 0 and 1, the two functions differed by 0.
- Draw summaries: average the five splits inside a draw, then take the mean, the sample standard deviation (divisor n−1), and the 2.5 and 97.5 percent linear quantiles across draws. `split_sd` is the sample standard deviation, across splits, of each split's mean over draws. That is what `noise_draws.summarize` does and what `wdbc_calib.stats.summarize_draws` does.
- At sigma = 0 the added noise is zero, so predictions do not depend on the draw.

## Differences that change the experiment

### Splits and seeds

| Choice | Original scripts | `wdbc_calib` |
|---|---|---|
| Split `random_state` | `2026` | `20260311` |
| Split index stored in outputs | 1, 2, 3, 4, 5 (`enumerate(..., 1)`) | 0, 1, 2, 3, 4 |
| Forest and SVM `random_state` | `2026` on every split | `20260311 + 100 + model index` on every split |
| Logistic regression `random_state` | unset | unset |
| Inner calibration-split seed | none, except the seed inside `SVC(probability=True)` | `20260311 + 200` for `StratifiedKFold` and for out-of-fold threshold scores |
| One-draw noise seed | `numpy.random.default_rng([2026, split, int(sigma * 100)])` | not used |
| 50-draw noise seed | `default_rng([2026, split, draw, int(round(sigma * 100))])` | `SeedSequence([20260311, stream, split, level_index, draw])` reduced to one `uint32`, then `default_rng` |

The original module docstring says the paper does not state the noise seed or the model `random_state`, and that the values above are assumptions. Those assumptions are what the preserved tables reproduce. They are not a seed recovered from an external PDF.

`int(sigma * 100)` and `int(round(sigma * 100))` are equal on {0, 0.25, 0.5, 1}. The one-draw and 50-draw seeds still differ because only the 50-draw seed includes the draw index.

### Preprocessing

Both protocols scale inside the training fold. Two text differences do not change these fits:

- Logistic `C` is set to 1.0 in the original and left at the scikit-learn default 1.0 in `wdbc_calib`.
- `gamma="scale"` is set in the original SVM and left at the default `"scale"` in `wdbc_calib`.
- `solver="lbfgs"` is explicit only in `wdbc_calib`. It is the default used by the original call.

`wdbc_calib.noise.population_sd` replaces a zero training standard deviation with 1. The original script does not. No WDBC training fold in these splits has a zero-variance column, so this guard does not move the archived numbers. It is still a different rule.

`LogisticRegression(max_iter=3000)` versus `max_iter=5000` does not change the fitted coefficients on these two split seeds. Counted iterations on all five original splits and all five new-protocol splits were between 15 and 20.

### Training

| Estimator | Original | `wdbc_calib` baseline |
|---|---|---|
| Logistic regression | `C=1.0`, `max_iter=3000`, default solver, inside `StandardScaler` | `max_iter=5000`, `solver="lbfgs"`, default `C=1`, inside `StandardScaler` |
| Random forest | 200 trees, `min_samples_leaf=2`, `random_state=2026`, default `n_jobs` | 100 trees, default `min_samples_leaf=1`, `n_jobs=1`, `random_state=20260311+100+1` |
| RBF SVM | `C=1`, `gamma="scale"`, `probability=True`, `random_state=2026`, inside `StandardScaler` | `C=1` (default), `gamma="scale"` (default), `probability=True`, different `random_state`, inside `StandardScaler` |

The forest is a different model. The split seed is a different partition. Either one is enough to make `results/` a new experiment. The clean-data gaps versus the archive (logistic ROC-AUC −0.0003, Brier −0.0023, 10-bin ECE −0.0152, sensitivity +0.0189) are gaps between these protocols. They are not a failure of the original script to regenerate its own table.

### Probability calibration

The original scripts do not fit a sigmoid or isotonic map after the base model. The only calibration in them is the internal Platt scaling that `SVC(probability=True)` runs on the training fold. Logistic regression uses its own probabilities. The forest uses vote fractions.

`wdbc_calib` adds `CalibratedClassifierCV(..., ensemble=False)` with sigmoid and isotonic maps, inner `StratifiedKFold(5, shuffle=True)`, fit on the training fold only. The SVM base in those arms uses `probability=False`, so the decision function is calibrated once. That block is new. It is not a silent rewrite of `reproduce_study.py`.

### Noise generation

Shared isotropic Gaussian noise on the test matrix is in both. Only `wdbc_calib` adds correlated Gaussian noise, multiplicative noise, training-mean dropout, and isotropic noise restricted to the ten features whose names start with `"worst "`.

### Metrics and tests

The original paired file contrasts models (SVM minus logistic regression, forest minus logistic regression, SVM minus forest) on ROC-AUC, Brier, ECE-10, sensitivity, and F1. `frac_a_greater` is `(difference > 0).mean()` on the full-precision draw differences. Re-aggregating the saved CSV turns one exact tie into about 1.1×10⁻¹⁶ (forest minus logistic regression, sensitivity, sigma 0.25, draw 31), which is why a later reader gets 0.10 instead of the stored 0.08. The script itself wrote 0.08. `wdbc_calib` uses a 10⁻¹⁰ tolerance so that CSV round-trip does not order a tie. Its paired file contrasts calibration arms, not the original model pairs.

`wdbc_calib` also reports 15-bin ECE, equal-mass ECE, a Nadeau–Bengio corrected resampled t-test, cost-minimising thresholds, and a 30-split dispersion curve. None of those are computed by the original scripts.

## Leakage reading of the original scripts

From the source, not from a guess about a missing file:

- The split is drawn before any fit.
- `StandardScaler` sits inside the pipeline and is fit on `X[train]` only.
- The noise standard deviation is computed on `X[train]` only and added to `X[test]` only.
- The threshold is the constant 0.5. Test labels are used in `metrics`, after prediction.
- One noise matrix is shared across the three models.

No test in the original scripts poisons the test fold. The structural reading above is the check that this repository can now make. It does not extend to a manuscript PDF that is not in the repository. The script header says it reproduces a 4 November 2026 methods note and that two seeds are assumptions.

## What this comparison does not say

It does not say the later recalibration-under-noise experiment is the same study as the preserved tables. It does not say either protocol is a clinical validation. It does not replace the Phopase overlap note in `ORIGINALITY.md`.
