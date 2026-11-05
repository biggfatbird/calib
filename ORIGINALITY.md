# Originality and overlap

Written 5 November 2026, 21:20 IST. This is an assessment of documents that were actually read. It is not a clearance that the work is novel.

## Status of the original study code

`reproduce_study.py`, `noise_draws.py`, and `make_figures_task1.py` were not in this workspace at the time of writing. The user states that they exist outside the GitHub checkout. Public listings checked the same evening (`biggfatbird/calib`, `github.com/manishakanwar`) did not contain them.

Consequence: no line of `wdbc_calib/` has been compared with those scripts. No result under `results/` is a verified reproduction of them. No result under `preserved/results/` has been re-derived from them. The numerical disagreement in `results/recalibration/comparison_vs_preserved.csv` remains a difference between an uploaded table and a new protocol, not a diagnosed code diff.

When the scripts are readable, copy them unchanged into `preserved/original_source/` and only then edit the manuscript's protocol section.

## Phopase preprint

Source read: Research Square HTML for doi:10.21203/rs.3.rs-11491081/v1, posted 27 October 2026. The page names Kartik Phopase (Sanjivani University) and says the manuscript is not peer reviewed. Author string on the fetched page: "kartik phopase".

### What overlaps the explicit protocol in this repository

- Dataset: scikit-learn WDBC, 569 rows, 212 malignant (37.3%), malignant coded 1.
- Models in common: L2 logistic regression with C=1, RBF SVM with C=1, random forest. Both note that scikit-learn SVM probabilities are already internal Platt scaling.
- Both standardise inside the training fold and do not tune hyperparameters.
- Both fit sigmoid and isotonic calibration on training data only.
- Both report Brier score and 10-bin equal-width ECE.
- Both use a Nadeau–Bengio corrected resampled t-test, with different designs (theirs: 50 folds of 5×10 CV and factor `1/k + 1/4`; this repository: 5 repeated holdouts and factor `1/J + n_test/n_train`).
- Both choose a threshold from training out-of-fold scores and then apply it once. Their rule is "largest threshold with sensitivity at least 0.95". This repository minimises costs 1:1, 5:1, and 10:1.
- Both warn that the work is not a clinical validation.
- Direction on clean data: an extra sigmoid can worsen logistic regression and the SVM; isotonic calibration reduces forest ECE. Their Table IV (scenario B, one 114-row test): logistic regression raw 0.021 / 0.030, sigmoid 0.031 / 0.070, isotonic 0.017 / 0.021; SVM raw 0.022 / 0.020; forest raw 0.031 / 0.059, isotonic 0.029 / 0.032. Those cells are theirs. They were not reused as inputs. This repository's clean five-split means are different numbers (logistic regression Brier 0.0235, ECE 0.0275).

### What does not overlap

- No synthetic feature noise, no shared noise draws, no correlated or multiplicative corruption, no feature dropout, no "worst-10" block noise.
- No repeated 75/25 holdout. Their split is one 455/114 cut, seed 42, scikit-learn 1.8.0. This repository uses seed 20260311, test size 0.25, scikit-learn 1.9.1.
- Their forest has 500 trees. The forest here has 100.
- They include gradient boosting, a majority baseline, a 10-mean-feature scenario, and permutation importance. This repository does not.
- The negative result that isotonic calibration fitted on clean rows worsens logistic-regression Brier score and ROC-AUC at isotropic sigma = 1, in every draw, is not in that preprint.

### Attribution

There is no evidence in this repository that code or tables were copied from that preprint. The shared clean-data moral is also the one in the scikit-learn 1.9.1 calibration guide and in Niculescu-Mizil and Caruana (as cited by that guide). Calling that moral a contribution of this repository would be wrong. Calling the noise experiment a copy of the preprint would also be wrong: the preprint does not run it.

A separate 2026 medRxiv preprint (doi:10.64898/2026.06.17.26355900) studies a demographic calibration gap on WDBC plus MIMIC-IV. Only its abstract was read. It is a different question. It is not a source of the noise results.

## What would still be required before an originality claim

1. The original `reproduce_study.py` and `noise_draws.py`, compared line by line.
2. A literature search that is not limited to one Research Square preprint, the scikit-learn guide, and Nadeau and Bengio.
3. Removal of any sentence that presents the clean-data sigmoid/isotonic pattern as new.
