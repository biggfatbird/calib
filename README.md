# When accuracy is not enough

Calibration and synthetic-noise checks for three classifiers on the Wisconsin Diagnostic Breast Cancer (WDBC) data shipped with scikit-learn.

This repository is an educational methods study. It is **not** a clinical validation, not a medical device, and not peer reviewed. Nothing here should be used to interpret a patient's risk.

## What was already in GitHub

As of commit `ebb50c3` on `main`, this repository contained `README.md` and `.cursor/install.sh` only. The study code was not there. Uploaded result tables, the task notes, and one figure were copied byte-for-byte into `preserved/` and are covered by `preserved/SHA256SUMS`. See `AUDIT.md` and `preserved/PROVENANCE.md`.

Commit `636d281` (5 November 2026, 21:59 IST) added `reproduce_study.py`, `noise_draws.py`, and `make_figures_task1.py`. They are stored unchanged in `preserved/original_source/`. Running them reproduces the archived numeric tables to within 10⁻¹². See `PROTOCOL_COMPARISON.md`. New runs use the explicit protocol in `wdbc_calib/` and write to `results/`. That protocol is a different split seed, a different forest, and a recalibration experiment. `results/recalibration/comparison_vs_preserved.csv` is the numerical gap between the two. It is not a failed reproduction of the original script.

## Question for the new experiments

Does probability recalibration fitted on clean training data improve or preserve calibration when the same models are scored under synthetic feature corruption?

The baseline RBF SVM uses `SVC(probability=True)`. In scikit-learn 1.9 that option already fits an internal sigmoid calibration and is deprecated. Sigmoid and isotonic arms use `CalibratedClassifierCV(..., ensemble=False)` on a base estimator that has not already been calibrated. The SVM comparison is therefore "internal Platt probabilities versus an explicit calibrator on the decision function", not "raw margins versus calibration".

## Protocol (new work)

- Data: `load_breast_cancer`, 569 rows, 30 features. Malignant is coded as class 1 (212 rows). Benign is class 0.
- Splits: `StratifiedShuffleSplit`, test size 0.25 (143 rows, 53 malignant), `random_state` 20260311. Primary analysis uses 5 splits. A 30-split run extends that same sequence.
- Preprocessing: `StandardScaler` inside the logistic and SVM pipelines, fit on the training fold only (and, for recalibration, inside each training fold of the inner split). The forest is fit on the original features.
- Noise: isotropic Gaussian in original units, `sigma * training population SD`, shared across models and calibration arms within a draw. Sigma grid: 0, 0.25, 0.5, 1. Primary draw count: 50.
- Operating point for the primary metrics: probability threshold 0.5. A separate experiment chooses thresholds from training out-of-fold predictions only.
- Uncertainty: the 2.5–97.5 percentile across noise draws, with the splits held fixed, is a noise-seed interval. Split-to-split sample SDs are stored separately. Neither one is a bootstrap confidence interval for a new patient population.

## Commands

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python -m wdbc_calib recalibration
python -m wdbc_calib more-splits
python -m wdbc_calib corruptions
python -m wdbc_calib thresholds
python -m wdbc_calib cross-dataset
python -m wdbc_calib compare-preserved
python -m wdbc_calib figures
```

`make test`, `make recalibrate`, `make more-splits`, `make corruptions`, `make thresholds`, `make cross-dataset`, `make compare`, and `make figures` call the same commands. `cross-dataset` downloads eight public tables through scikit-learn and OpenML.

Pinned versions that this environment used are in `requirements.txt`. The archived noise run reported Python 3.13.18, scikit-learn 1.9.1, and NumPy 2.5.3. This environment is not that interpreter. Each new output directory records its own versions in `manifest.json`.

## Tests

`tests/test_preserved_csvs.py` recomputes the archived summaries and refuses checksum changes under `preserved/`. Other tests cover a hand-computed ECE, malignant-positive probabilities, training-only scaling, shared noise seeds, sigma = 0 invariance, and a threshold function whose signature cannot accept test labels.

## What the explicit runs found

Details, rounding, and claim ids are in `paper_v2.md` and `CLAIM_EVIDENCE_MATRIX.md`. Short version, from `results/recalibration/` (Python 3.12.3, scikit-learn 1.9.1, NumPy 2.2.6, five splits, 50 shared draws):

- The original scripts do reproduce the archived numbers. The explicit protocol does not, and the reason is now the code difference in `PROTOCOL_COMPARISON.md`: split seed 2026 versus 20260311, a 200-tree forest with `min_samples_leaf=2` versus a 100-tree forest, and no post-hoc calibrator in the original. Clean logistic-regression ROC-AUC is within 0.0004 of the archive; 10-bin ECE differs by about 0.015. See `results/recalibration/comparison_vs_preserved.csv`.
- On WDBC, sigmoid recalibration of the random forest lowers Brier score at sigma = 1 in all 50 draws (paired mean about −0.023), and isotonic recalibration of logistic regression raises Brier score and lowers ROC-AUC in every draw. The same protocol on eight public numeric tables does not keep those signs. Sonar reverses the logistic isotonic Brier result. Banknote and ionosphere reverse the forest sigmoid result. Dataset size, class balance, clean-data calibration, clean-data discrimination, and the baseline's noise damage do not line up with that Brier change: 34 of 36 dataset-bootstrap intervals cover zero, and Holm leaves none of the 36 rank tests. See `paper_v2.md` Sections 7 and 14.
- ECE orderings change between 10-bin, 15-bin, and equal-mass binning.
- Split-to-split standard deviations did not uniformly shrink when the holdout count went from 5 to 30.
- Correlated and multiplicative noise reorder models relative to isotropic noise. No corruption condition failed numerically; `results/corruptions/failures.csv` has a header and no data rows.
- Thresholds were chosen on training out-of-fold predictions only. At a 10:1 false-negative cost they vary widely across splits, and specificity at sigma = 1 falls hard.

Draw percentiles are noise-seed intervals with the splits held fixed. They are not bootstrap confidence intervals and not patient-level intervals. Thirteen of 48 unadjusted split-level Brier/ECE tests had p < 0.05. That count is exploratory.

## Further reading

- `AUDIT.md` — GitHub contents, archived CSV checks, and what could not be verified
- `LIMITATIONS.md` — dataset, noise, dependence, multiplicity
- `REPRODUCIBILITY.md` — seeds, commands, and what cannot be regenerated
- `paper_v2.md` — methods note using only the generated CSVs and the checked citations
- `CLAIM_EVIDENCE_MATRIX.md` — regenerated by `python -m wdbc_calib.write_claims`
