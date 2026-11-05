# Originality and overlap

Updated 5 November 2026, 22:04 IST, after the original scripts were read and rerun. This is an assessment of documents that were actually read. It is not a clearance that the work is novel.

## What the original scripts are

`preserved/original_source/` holds the bytes of commit `636d281`. The header of `reproduce_study.py` says it reproduces a 4 November 2026 note by M. Kanwar and that the noise seed and the model `random_state` are assumptions.

What those scripts contain:

- Five stratified 75/25 splits, `random_state=2026`, malignant coded 1.
- Logistic regression (`C=1`, `max_iter=3000`) and an RBF SVM (`probability=True`) with a training-fold `StandardScaler`. A random forest with 200 trees and `min_samples_leaf=2` on the raw features.
- Isotropic Gaussian noise on the test matrix only, sigma in {0, 0.25, 0.5, 1}, one shared draw per (split, sigma) or, in `noise_draws.py`, 50 draws.
- Threshold 0.5. No sigmoid or isotonic calibrator wrapped around the base models. No Nadeau–Bengio test. No cost-sensitive threshold. No correlated, multiplicative, dropout, or grouped noise.

A rerun reproduced `preserved/results/` numerically (maximum absolute difference 2.1×10⁻¹⁶). That table is the original study. It is not a finding produced by `wdbc_calib/`.

## What `wdbc_calib` adds, and what it repeats

Repeats, with different seeds and a different forest: the isotropic shared-noise design, the same sigma grid, the same three model families, and the same draw-versus-split summary.

Does not repeat the original hyperparameters. Split seed `20260311`. Forest of 100 trees, `min_samples_leaf` left at 1. Noise seeds go through `SeedSequence`.

Adds, and these additions are absent from the three scripts: sigmoid and isotonic `CalibratedClassifierCV(..., ensemble=False)` fit on clean training rows and scored on noisy test rows; 15-bin and equal-mass ECE; a Nadeau–Bengio test on five splits; correlated noise, multiplicative noise, training-mean dropout, and noise on the ten "worst" features; thresholds minimising declared false-negative costs on training out-of-fold scores; a 30-split dispersion curve.

The numerical gap in `results/recalibration/comparison_vs_preserved.csv` is this protocol difference. It is not evidence that the original script was lost.

## Phopase preprint

Source read: Research Square HTML for doi:10.21203/rs.3.rs-11491081/v1, posted 27 October 2026. The page names Kartik Phopase (Sanjivani University) and says the manuscript is not peer reviewed. Author string on the fetched page: "kartik phopase".

The preprint is earlier than the 4 November 2026 date in the original script header. The original scripts do not cite it. They also do not run its calibration experiment.

### What overlaps the explicit protocol in this repository

- Dataset: scikit-learn WDBC, 569 rows, 212 malignant (37.3%), malignant coded 1.
- Models in common: L2 logistic regression with C=1, RBF SVM with C=1, random forest. Both note that scikit-learn SVM probabilities are already internal Platt scaling.
- Both standardise inside the training fold and do not tune hyperparameters.
- Both fit sigmoid and isotonic calibration on training data only. The original three scripts do not.
- Both report Brier score and 10-bin equal-width ECE.
- Both use a Nadeau–Bengio corrected resampled t-test, with different designs (theirs: 50 folds of 5×10 CV and factor `1/k + 1/4`; this repository: 5 repeated holdouts and factor `1/J + n_test/n_train`). The original scripts do not compute that test.
- Both choose a threshold from training out-of-fold scores and then apply it once. Their rule is "largest threshold with sensitivity at least 0.95". This repository minimises costs 1:1, 5:1, and 10:1. The original scripts use a fixed 0.5.
- Both warn that the work is not a clinical validation.
- Direction on clean data: an extra sigmoid can worsen logistic regression and the SVM; isotonic calibration reduces forest ECE. Their Table IV (scenario B, one 114-row test): logistic regression raw 0.021 / 0.030, sigmoid 0.031 / 0.070, isotonic 0.017 / 0.021; SVM raw 0.022 / 0.020; forest raw 0.031 / 0.059, isotonic 0.029 / 0.032. Those cells are theirs. They were not reused as inputs. This repository's clean five-split means are different numbers (logistic regression Brier 0.0235, ECE 0.0275).

### What does not overlap

- No synthetic feature noise, no shared noise draws, no correlated or multiplicative corruption, no feature dropout, no "worst-10" block noise. The original scripts do contain isotropic shared noise. The preprint does not.
- No repeated 75/25 holdout. Their split is one 455/114 cut, seed 42, scikit-learn 1.8.0. The original scripts use seed 2026. `wdbc_calib` uses seed 20260311. Both use test size 0.25 and, in this workspace, scikit-learn 1.9.1.
- Their forest has 500 trees. The original forest has 200. The forest in `wdbc_calib` has 100.
- They include gradient boosting, a majority baseline, a 10-mean-feature scenario, and permutation importance. Neither protocol here does.
- The negative result that isotonic calibration fitted on clean rows worsens logistic-regression Brier score and ROC-AUC at isotropic sigma = 1, in every draw, is not in that preprint and is not in the original scripts.

### Attribution

There is no evidence in this repository that code or tables were copied from that preprint. The shared clean-data moral is also the one in the scikit-learn 1.9.1 calibration guide and in Niculescu-Mizil and Caruana (as cited by that guide). Calling that moral a contribution of this repository would be wrong. Calling the noise experiment a copy of the preprint would also be wrong: the preprint does not run it. Calling the isotropic noise table a new result of `wdbc_calib` would also be wrong: it is the archived script, and it reproduces.

A separate 2026 medRxiv preprint (doi:10.64898/2026.06.17.26355900) studies a demographic calibration gap on WDBC plus MIMIC-IV. Only its abstract was read. It is a different question. It is not a source of the noise results.

## What is distinct, and what is not claimed

Distinct from the Phopase preprint: synthetic feature noise. That distinction belongs first to the original scripts.

Distinct from those scripts: recalibration fitted on clean training rows and evaluated under noise, plus the other corruption mechanisms and the cost thresholds. That is the `wdbc_calib` protocol.

Not claimed: that the clean-data sigmoid/isotonic pattern is new. Manokhin and Grønhaug (PMLR 329:378–381, 2026; arXiv:2601.19944) already show that isotonic regression and Platt scaling can fail to improve proper scores on i.i.d. tabular tasks, and they say they omitted distributional shift. Ovadia and colleagues (NeurIPS 2019) already show that temperature scaling fit on clean validation data can fail under dataset shift for deep networks.

The cross-dataset run in `results/cross_dataset/` asks whether two WDBC signs, isotonic logistic Brier getting worse at sigma = 1 and sigmoid forest Brier getting better, hold on seven other public numeric tables under the same protocol. They do not. That negative transport result is evidence against publishing the WDBC pattern as a general finding. It is not a claim that the eight-table check is itself a new theorem, and it is not a publication-readiness clearance. Beta calibration and Venn–Abers were not run. A literature search beyond the sources named in `paper_v2.md` Section 2 was not done.
