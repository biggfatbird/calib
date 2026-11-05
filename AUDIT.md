# Audit

Date of inspection: 5 November 2026, Asia/Calcutta.
Repository: `https://github.com/biggfatbird/calib`.
Inspected remote commit: `ebb50c3e58b33ece227753da2c053d6fd81b33d1` (`Add Cloud Agent install script for workspace verification`).
Earlier commit on `main`: `864ca53` (`Initial commit`), README only.

This audit records what was actually present. It does not reconstruct missing code or treat later reimplementations as the original study.

## Severity summary

| ID | Severity | Finding |
|---|---|---|
| A1 | Critical | The connected GitHub repository does not contain the research project. |
| A2 | Resolved on 5 November 2026, 22:04 IST | The three scripts were absent at the first inspection. Commit `636d281` added them. They are archived unchanged in `preserved/original_source/`. Leakage and seeds are read from that source in the follow-up below. |
| A3 | High | `CURSOR_TASKS.md` calls draw-wise percentiles "95% percentile bootstrap CIs". The uploaded figure and `TASK1_RESULTS.md` do not. Those percentiles are not bootstrap confidence intervals. |
| A4 | High | The one-draw noise seed is an assumption written in `reproduce_study.py`, not a seed stated as part of an external paper. It is a different stream from the 50-draw seed. Sigma = 1 orderings still move across draws. |
| A5 | High | Inference is from five overlapping 75/25 splits of one 569-row dataset. Draw intervals hold the splits fixed. They are not uncertainty for a new patient sample. |
| A6 | Medium | One paired fraction is 0.08 in the CSV and 0.10 if that CSV is re-aggregated with a strict `> 0` test. The script's own full-precision comparison wrote 0.08. The extra 1e-16 appears after the values are saved. |
| A7 | Resolved on 5 November 2026, 22:04 IST | The original scripts fit the scaler, the models, and the training-fold noise scale on the training split only. `SVC(probability=True)` is the SVM probability. Threshold 0.5 is fixed. ECE is the 10-bin function in `reproduce_study.py`. |
| A8 | Low | The environment stamp is Python 3.13.18, scikit-learn 1.9.1, NumPy 2.5.3. This workspace is Python 3.12 with scikit-learn 1.9.1 and a different NumPy build. |

## A1 and A2 — repository contents

`gh api repos/biggfatbird/calib/git/trees/main?recursive=1` returned:

- `.cursor/install.sh` (checks that `README.md` exists and contains "calib")
- `README.md` (the eight characters `# calib`)

Branches on the remote at inspection: `main` and `cursor/cloud-agent-environment-b17d`, both at `ebb50c3`. No other commits contain study code.

Local upload set (now copied to `preserved/`):

- three markdown instruction/result files
- five result CSVs and a four-line environment stamp
- one PNG figure

Not present at that inspection: `reproduce_study.py`, `noise_draws.py`, a test suite, a manuscript, dependency pins, or figure-generation scripts.

That paragraph is the state of `main` at `ebb50c3`. It is not the state of this branch after commit `636d281`. The follow-up dated 22:04 IST is the current reading.

## Checks that did run on the preserved CSVs

Computed in this audit (and locked in `tests/test_preserved_csvs.py`):

- `noise_draws.csv` has 3000 data rows: splits `{1,2,3,4,5}`, sigmas `{0, 0.25, 0.5, 1}`, draws `0..49`, models `{Logistic regression, Random forest, RBF SVM}`.
- At sigma = 0, ROC-AUC, Brier, ECE-10, sensitivity, and F1 are constant across the 50 draws within each split and model (`nunique = 1`).
- `results_summary.csv` mean and sample SD (divisor n−1) match `results_by_split.csv` (60 rows = 5 × 4 × 3).
- `noise_draw_summary.csv` ROC-AUC mean matches the mean across draws of the five-split mean. `draw_sd` matches NumPy/pandas sample SD (ddof = 1) across draws. `draw_lo` / `draw_hi` match the 2.5 and 97.5 percent linear quantiles across draws. `split_sd` matches the sample SD, across the five splits, of each split's mean over draws.
- Paired differences match a − b on the five-split mean, then the mean and 2.5/97.5 quantiles across draws, to numerical noise (max absolute error about 1e-15), **except** the fraction below.

Test-set size implied by the metrics: sensitivity 49/53 = 0.924528… and specificity 89/90 = 0.9888… on split 1, logistic regression, sigma = 0. That is 143 test rows and a malignant-positive coding (53 malignant, 90 benign), which is what a stratified 25% holdout of the 569-row Wisconsin Diagnostic Breast Cancer set produces (212 malignant, 357 benign). This is evidence about the label and the split size, not about the random seed.

## A3 — intervals are not bootstrap confidence intervals

`preserved/docs/CURSOR_TASKS.md` says Task 1 should "compute mean, SD and 95% percentile bootstrap CIs (over splits × draws)".

`preserved/docs/TASK1_RESULTS.md` says the interval is the "2.5–97.5 percentile across draws (noise-seed uncertainty only)" and that split-to-split SD is a separate column. The uploaded figure title says "bands = 95% interval across 50 independent noise draws".

The CSV matches the second description. Pooling splits and draws into one bootstrap would mix two different sources of variation and would still not be a patient-level confidence interval. This repository keeps those sources separate and does not rename percentiles as bootstrap confidence intervals.

## A4 — single draw versus 50 draws

At sigma = 0 the single-draw summary and the 50-draw mean agree, as they must if no noise is added.

At sigma = 1 the preserved files disagree with each other because they answer different questions. Logistic regression ROC-AUC is 0.9556 in `results_summary.csv` (one draw, then the mean of five splits) and 0.9616 in `noise_draw_summary.csv` (mean over 50 draws). `TASK1_RESULTS.md` already states that a single sigma = 1 ordering is inside draw-to-draw variation. The single-draw file remains preserved; it is not overwritten or "corrected".

## A5 — dependence and one dataset

Each test fold is about 25% of the same 569 rows, so the five test sets overlap. The 50 draws repeat the same patients. Effective information is closer to one small cohort than to 5 × 50 independent samples. Draw percentiles describe how the chosen noise generator moves the five-split mean when the partition is held fixed. They do not license a claim about other hospitals, other assays, or clinical performance.

## A6 — one tied paired fraction

For random forest minus logistic regression, sensitivity, sigma = 0.25, re-aggregating `noise_draws.csv` with `(difference > 0)` gives 0.10 (5/50). The stored `frac_a_greater` is 0.08 (4/50). Draw 31 is about `1.1e-16` in the CSV. Sensitivity is a multiple of 1/53, so that value is a tie that the CSV rounding disturbed.

`noise_draws.paired` computes `(d > 0).mean()` on the full-precision draw differences and then writes the CSV. A fresh run of that function writes 0.08, matching the archive. The 10⁻¹⁰ tolerance used later in `wdbc_calib` is there so a reader of the CSV gets the same 0.08. The archived cell was not rewritten.

## A7 — procedures now read from the original scripts

Commit `636d281` supplies the source. For the preserved numbers the scripts do the following:

- `StandardScaler` is inside the logistic and SVM pipelines and is fit on the training split only.
- Malignant is `1 - sklearn_target`, which is class 1. Metrics use `predict_proba` column 1.
- The hard-label threshold is the constant 0.5.
- One noise matrix is drawn and then reused for all three models.
- Split `random_state` is 2026. Forest and SVM `random_state` are 2026. The module docstring calls the noise seed and the model `random_state` assumptions, because it says the paper does not state them.
- ECE is the 10-bin function in `reproduce_study.ece`.
- Noise is `sigma * training population SD (ddof=0) * Z` on the test matrix only.

An earlier seed search, run before this source was in the workspace, did not guess seed 2026 and did not reproduce the five-split Brier vector. That negative search is historical. The reproduction that matches is the follow-up below, which runs the script rather than searching for it.

## A8 — environment

`preserved/results/noise_draws_env.txt` records Python 3.13.18, scikit-learn 1.9.1, NumPy 2.5.3. New runs record their own versions beside each output directory. A later rerun on Python 3.12.3 and NumPy 2.2.6 matched the parsed numbers and did not match the CSV bytes. Bitwise equality is not claimed.

## Follow-up on 5 November 2026, 21:20 IST

The study author stated that `reproduce_study.py`, `noise_draws.py`, `make_figures_task1.py`, `AGENTS.md`, `CURSOR_TASKS.md`, and `TASK1_RESULTS.md` exist outside this GitHub checkout and would be made available in the workspace.

What was searched, and what was found:

- Workspace, home directory, `/tmp`, `/opt`, and `/cursor`: the three Python names were not present. The markdown names exist only as the copies already under `preserved/docs/`, taken from the earlier upload.
- `biggfatbird/calib` trees on `main` and `cursor/research-integrity-3411`: no `reproduce_study.py`.
- Public repositories listed for `github.com/manishakanwar`: `example` and `anything` only. Neither is this study.

No line-by-line comparison was performed. No experimental CSV was regenerated. No archived or new result was reclassified as a direct reproduction of unseen source. Until those scripts are readable, every number under `results/` remains an output of `wdbc_calib/`, and every number under `preserved/results/` remains an uploaded table whose generating code was not inspected.

## Follow-up on 5 November 2026, 22:04 IST

Commit `636d281` added `reproduce_study.py`, `noise_draws.py`, and `make_figures_task1.py` at the repository root. Those bytes were moved to `preserved/original_source/` and the checksum file was extended. The scripts were not edited.

They were executed from that directory on Python 3.12.3, scikit-learn 1.9.1, and NumPy 2.2.6, with output kept out of `preserved/`. Every numeric column of the five archived result CSVs matches that run to a maximum absolute difference of 2.1×10⁻¹⁶. Detail is in `results/original_source_check/numeric_agreement.csv` and `PROTOCOL_COMPARISON.md`. The CSV files are not byte-identical, and the environment stamp still records Python 3.13.18 and NumPy 2.5.3. The archived files were left in place.

Classification:

- `preserved/results/results_by_split.csv` and `results_summary.csv` are a direct reproduction of `reproduce_study.py`.
- `preserved/results/noise_draws.csv`, `noise_draw_summary.csv`, and `paired_differences.csv` are a direct reproduction of `noise_draws.py`.
- The one-draw noise seed is `default_rng([2026, split, int(sigma*100)])`. The 50-draw seed is `default_rng([2026, split, draw, int(round(sigma*100))])`. At sigma > 0 the one-draw Brier scores do not match any of the 50 draws (minimum absolute gap about 8.9×10⁻⁸).
- `results/recalibration/`, `results/corruptions/`, `results/thresholds/`, and `results/more_splits/` are the later protocol. Split seed 20260311, a 100-tree forest with `min_samples_leaf=1`, and sigmoid/isotonic arms are not in the original scripts. Those directories are not a reproduction of the archive.
- The uploaded figure has the same panels as `make_figures_task1.py`. Its file is 2000×480 RGBA. A fresh run of the script here is 2400×576 RGB. The PNG was not replaced.

The original scripts fit on the training split only, share one noise matrix across models, and use a fixed threshold of 0.5. They do not fit a post-hoc sigmoid or isotonic calibrator. The docstring states that the noise seed and the model `random_state` are assumptions.

## What this audit does not claim

- It does not claim the original scripts never existed. The 21:20 IST section records that they were not in the places searched then. The 22:04 IST section records that they were subsequently uploaded.
- It does not claim the preserved study leaked labels. The uploaded scripts fit the scaler and the models on the training split only.
- It does not claim the preserved study is clinically valid, peer reviewed, or publishable as a clinical result.
- It does not treat `TASK1_RESULTS.md` verdicts as new findings of this repository. They are the uploaded write-up of `noise_draws.py`. The arithmetic was recomputed from the CSV, and the CSV was then regenerated from that script.
- It does not treat `wdbc_calib/` as a line-by-line port. The differences that change the experiment are in `PROTOCOL_COMPARISON.md`.
