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
| A2 | Critical | `reproduce_study.py`, `noise_draws.py`, and `make_figures_task1.py` were not in this checkout or the upload. Leakage and seeds in the original run still cannot be audited from code. This does not mean those files never existed outside the checkout. |
| A3 | High | `CURSOR_TASKS.md` calls draw-wise percentiles "95% percentile bootstrap CIs". The uploaded figure and `TASK1_RESULTS.md` do not. Those percentiles are not bootstrap confidence intervals. |
| A4 | High | Noisy single-draw results (`results_by_split.csv`, sigma > 0) use an unspecified seed. Task 1 already reports that sigma = 1 model orderings move across draws. |
| A5 | High | Inference is from five overlapping 75/25 splits of one 569-row dataset. Draw intervals hold the splits fixed. They are not uncertainty for a new patient sample. |
| A6 | Medium | One published paired fraction matches the CSV only after treating a 1e-16 difference as a tie. |
| A7 | Medium | Whether preprocessing was fit on training data only, and whether SVC probabilities were the internally Platt-scaled kind, cannot be checked without source. |
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

Not present anywhere inspected: `reproduce_study.py`, `noise_draws.py`, a test suite, a manuscript, dependency pins, or figure-generation scripts.

Consequence: preserved CSVs are an archived numerical record, not a result this repository can regenerate. Later code under `wdbc_calib/` is a new, explicitly seeded protocol. Agreement or disagreement with `preserved/results/` is reported as a comparison, not as proof that the original script was recovered.

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

For random forest minus logistic regression, sensitivity, sigma = 0.25, `(difference > 0).mean()` over draws is 0.10 (5/50). The preserved `frac_a_greater` is 0.08 (4/50). Draw 31 has a difference of about `1.1e-16`, which is floating-point dust around a tie (sensitivity is a multiple of 1/53). Counting differences above `1e-10` as strictly greater reproduces 0.08. All other preserved paired fractions match a strict `> 0` comparison. The audit treats 0.08 as the tie-aware value and does not rewrite the CSV.

## A7 — procedures that could not be verified

Without source, the following stay unverified for the preserved numbers:

- scaler fit on the training fold only, versus fit on all rows before splitting
- probability column aligned with malignant = 1, versus a benign-positive score passed through a flipped metric
- threshold fixed at 0.5 versus chosen using test labels
- one shared noise matrix per draw versus independent noise per model
- random_state of the split, the forest, and `SVC(probability=True)`
- ECE bin edges (equal-width 10-bin is a common default and is consistent with the column name `ece10`, but the binning code was not recovered)
- whether noise was added in original units using the training feature SD, and whether that SD used divisor n or n−1

A partial search did **not** recover a seed. It is not claimed to be exhaustive. What was run: scikit-learn 1.9.1 logistic regression, several scalers and solvers, `train_test_split` seeds 0–399 and `StratifiedShuffleSplit` seeds in a smaller range, compared with the preserved clean-data Brier and ROC-AUC. No setting reproduced the five-split Brier vector exactly. The closest clean logistic Brier gap observed under training-only `StandardScaler` was about 2e-6, with ROC-AUC one or two rank-pairs away (1/4770 ≈ 0.00021). An interrupted search with scaling fit on all rows before splitting also did not produce an exact Brier match in the seeds it finished. The preserved numbers are therefore kept as an archive. The new protocol specifies its own seeds.

## A8 — environment

`preserved/results/noise_draws_env.txt` records Python 3.13.18, scikit-learn 1.9.1, NumPy 2.5.3. New runs record their own versions beside each output directory. Numerical disagreement across NumPy/Python builds is a reason not to expect bitwise equality even if the source were recovered.

## Follow-up on 5 November 2026, 21:20 IST

The study author stated that `reproduce_study.py`, `noise_draws.py`, `make_figures_task1.py`, `AGENTS.md`, `CURSOR_TASKS.md`, and `TASK1_RESULTS.md` exist outside this GitHub checkout and would be made available in the workspace.

What was searched, and what was found:

- Workspace, home directory, `/tmp`, `/opt`, and `/cursor`: the three Python names were not present. The markdown names exist only as the copies already under `preserved/docs/`, taken from the earlier upload.
- `biggfatbird/calib` trees on `main` and `cursor/research-integrity-3411`: no `reproduce_study.py`.
- Public repositories listed for `github.com/manishakanwar`: `example` and `anything` only. Neither is this study.

No line-by-line comparison was performed. No experimental CSV was regenerated. No archived or new result was reclassified as a direct reproduction of unseen source. Until those scripts are readable, every number under `results/` remains an output of `wdbc_calib/`, and every number under `preserved/results/` remains an uploaded table whose generating code was not inspected.

## What this audit does not claim

- It does not claim the original scripts never existed. It claims they were not in the places inspected.
- It does not claim the preserved study leaked labels. It claims the leak checks could not be performed on that run.
- It does not claim the preserved study is clinically valid, peer reviewed, or publishable as a clinical result.
- It does not treat `TASK1_RESULTS.md` verdicts as new findings of this repository. They are quoted as the uploaded write-up, and the underlying arithmetic was recomputed as above.
