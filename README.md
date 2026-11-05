# When accuracy is not enough

Calibration and synthetic-noise checks for three classifiers on the Wisconsin Diagnostic Breast Cancer (WDBC) data shipped with scikit-learn.

This repository is an educational methods study. It is **not** a clinical validation, not a medical device, and not peer reviewed. Nothing here should be used to interpret a patient's risk.

## What was already in GitHub

As of commit `ebb50c3` on `main`, this repository contained `README.md` and `.cursor/install.sh` only. The study code was not there. Uploaded result tables, the task notes, and one figure were copied byte-for-byte into `preserved/` and are covered by `preserved/SHA256SUMS`. See `AUDIT.md` and `preserved/PROVENANCE.md`.

Those archived numbers are not regenerated. New runs use the explicit protocol in `wdbc_calib/` and write to `results/`. A comparison file, when produced, is a difference between the two protocols, not evidence that the missing script was recovered.

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
python -m wdbc_calib compare-preserved
python -m wdbc_calib figures
```

`make test`, `make recalibrate`, `make more-splits`, `make corruptions`, `make thresholds`, `make compare`, and `make figures` call the same commands.

Pinned versions that this environment used are in `requirements.txt`. The archived noise run reported Python 3.13.18, scikit-learn 1.9.1, and NumPy 2.5.3. This environment is not that interpreter. Each new output directory records its own versions in `manifest.json`.

## Tests

`tests/test_preserved_csvs.py` recomputes the archived summaries and refuses checksum changes under `preserved/`. Other tests cover a hand-computed ECE, malignant-positive probabilities, training-only scaling, shared noise seeds, sigma = 0 invariance, and a threshold function whose signature cannot accept test labels.

## Further reading in this repo

- `AUDIT.md` — what was missing and what was checked on the archived CSVs
- `LIMITATIONS.md` — written with the experimental report
- `REPRODUCIBILITY.md` — seeds, commands, and what is not reproducible
- `paper_v2.md` and `CLAIM_EVIDENCE_MATRIX.md` — written only after the corresponding CSVs exist
