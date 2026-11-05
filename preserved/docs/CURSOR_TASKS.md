# Cursor task pack (do in order; paste one task at a time into Cursor Agent; commit after each)

Context: `reproduce_study.py` reproduces the paper's clean-data numbers (verified: ROC-AUC/Brier within 0.003).
Noisy (sigma>0) numbers use ONE noise draw per split with an unspecified seed; I measured 30 alternative draws:
at sigma=1 mean ROC-AUC varies ~0.949-0.972 and Brier ~0.065-0.106 across draws, so the paper's single-draw sigma=1
orderings (e.g. SVM AUC 0.973 > LR 0.967) are inside noise-seed variation. Task 1 fixes this properly.

## Task 1 - [DONE by Claude, see TASK1_RESULTS.md] Multi-draw noise + confidence intervals (highest value)
Add `noise_draws.py`: for each split and sigma, draw K=50 independent noise matrices (shared across models within a draw),
store per-draw metrics in `results/noise_draws.csv`. Compute mean, SD and 95% percentile bootstrap CIs (over splits x draws)
per model/sigma/metric. Regenerate Figures 2-3 with CI bands. Acceptance: CSV has K*5*4*3 rows; a test asserts shape and that
sigma=0 results are identical across draws; a short markdown table states which paper claims remain true with CIs.

## Task 2 - More splits
Add `--splits 50` option (repeated stratified splits). Report that split-to-split SD shrinks/stabilises. Keep paper's 5-split
output byte-identical when run with defaults. Acceptance: test that default output hash equals the stored baseline hash.

## Task 3 - Recalibration experiment (new contribution; label it as such)
Add `recalibrate.py`: Platt (sigmoid) and isotonic via `CalibratedClassifierCV` fitted on training data ONLY (inner 5-fold),
then evaluate Brier/ECE under the same noise grid. Question: does recalibration on clean data survive noise? Use the same
shared noise matrices as Task 1. Acceptance: results CSV + one figure; explicit text that recalibration was fit on clean
training data and tested under shift; no claim beyond the data.

## Task 4 - Better calibration metrics
Add adaptive (equal-mass) binning ECE, 15-bin ECE, and a reliability diagram aggregated over splits with bin counts shown.
Show whether ECE ranking changes with binning (paper acknowledges ECE sensitivity with n=143). Acceptance: table of ECE vs
bin scheme; test for each ECE implementation on hand-computed toy examples.

## Task 5 - Alternative corruption models
Add `corruptions.py`: (a) correlated Gaussian noise using the training covariance, (b) multiplicative noise, (c) random
feature dropout imputed with the training mean, (d) a single group of features (e.g. the 10 "worst" features) corrupted.
Compare to isotropic noise. Acceptance: one summary figure, one CSV, a sentence per corruption stating limits (synthetic).

## Task 6 - Decision-threshold and cost analysis
For cost ratios FN:FP in {1:1, 5:1, 10:1}, choose the threshold on TRAINING-fold predictions (cross-val predictions), apply it
to clean and noisy test sets, and report expected cost + sensitivity/specificity. Acceptance: no threshold chosen on test data
(add a test that checks the function never receives test labels).

## Task 7 - Engineering hygiene
GitHub Actions workflow (Python 3.11-3.13, run tests), `requirements.txt` with pinned ranges, type hints, docstrings, a Makefile
(`make reproduce`, `make test`, `make figures`), suppress/handle the sklearn 1.9 `SVC(probability=True)` FutureWarning by
documenting it (do not silently switch to CalibratedClassifierCV in the baseline). Acceptance: CI green, warnings documented.

## Task 8 - Paper v2 (only after Tasks 1-4 ran)
Update the manuscript text using ONLY numbers from the new CSVs: replace single-draw claims with mean +/- CI, add the new
sections as "Extension" with their own limitations, keep the declarations ("course-level, not peer reviewed, no clinical use").
Output `paper_v2.md` plus a change log listing every number that changed and why.

## Task 9 - README + figure polish
README with: question, method, headline results (with CIs), limitations, reproduction commands, citations. Consistent
colour-blind-safe palette, labelled axes, n per panel. Acceptance: README commands run from a clean clone.

### End-of-day checklist
- `python -m pytest -q tests` passes; `python reproduce_study.py` still matches the paper's clean-data numbers.
- Every number in README / paper_v2 appears in a results CSV.
- Commit, push to a private GitHub repo first; make it public only after you re-read every claim and can explain the code.
