# Calibration under synthetic feature noise on the Wisconsin Diagnostic Breast Cancer data

A methods note. Not a clinical validation. Not peer reviewed. Not a medical device.

Numbers below are rounded to four decimal places from `CLAIM_EVIDENCE_MATRIX.md`, which is generated from the CSVs by `python -m wdbc_calib.write_claims`. Full precision stays in the CSVs. Claim ids in brackets point at that matrix.

## Abstract

Three standard classifiers were fit to the scikit-learn Wisconsin Diagnostic Breast Cancer table (569 rows, malignant coded positive) and scored under shared synthetic feature noise. Logistic regression, a 100-tree random forest, and an RBF support-vector machine were compared with the same five stratified 75/25 splits and the same 50 noise draws. Two recalibrators, sigmoid and isotonic, were fit by `CalibratedClassifierCV` with `ensemble=False` on clean training folds only and then applied to noisy test folds.

Discrimination on clean data is high (baseline ROC-AUC 0.9929, 0.9951, and 0.9875). Isotropic noise at one training-feature standard deviation lowers ROC-AUC into the mid-0.95s and raises Brier scores. Recalibration does not do one thing. Sigmoid recalibration of the forest lowers Brier score at sigma = 1 by about 0.023, and that sign is the same in all 50 draws; a split-level corrected t-test on five splits also separates it, but that test is one of 48 unadjusted Brier and ECE contrasts. The same sigmoid map barely changes logistic regression or the SVM. The SVM baseline is already an internal Platt scaling, and an explicit sigmoid fit leaves ROC-AUC unchanged on clean data. Isotonic recalibration of logistic regression worsens Brier score and ECE at sigma = 1 and lowers ROC-AUC in every draw. ECE orderings change with the binning rule. Correlated and multiplicative noise hurt more than isotropic noise of the same nominal scale. None of this is a statement about clinical benefit.

## 1. Introduction

Accuracy and ROC-AUC on this table are easy to push near the ceiling. A probability that will be thresholded, or read as a risk, can still be miscalibrated, and a ranking that is stable on clean features can move once features are corrupted. The question asked here is narrow:

Does probability recalibration fitted only on clean training rows improve or preserve calibration when the same fitted models are evaluated under synthetic feature corruption?

The answer is allowed to be no. Deterioration and near-ties are reported with the improvements.

This file is not a revision of a recovered manuscript. The GitHub repository, at the commit audited in `AUDIT.md`, contained a README and an install check. Uploaded tables and a figure were archived under `preserved/` because the script that produced them was not in the repository. Section 3 separates that archive from the explicit protocol that was actually run.

## 2. Related work that was checked

The following sources were opened and are the only ones cited.

scikit-learn 1.9.1, User Guide section 1.16, "Probability calibration" (https://scikit-learn.org/stable/modules/calibration.html), states that a calibrated binary classifier should have an observed frequency near 0.8 among rows scored near 0.8; that the Brier score mixes calibration and discrimination (Murphy, 1973, as cited there); that sigmoid calibration is Platt's logistic map; that isotonic calibration is more flexible and overfits more easily on small samples; that isotonic maps can change ROC-AUC by introducing ties while a sigmoid map is strictly monotone; and that random forests tend to pull probabilities away from 0 and 1. The same page cites Niculescu-Mizil and Caruana, "Predicting good probabilities with supervised learning," ICML 2005, and Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods," 1999. Those two papers are cited here only as the documents named by that user-guide page. They were not re-read in full.

Nadeau and Bengio, "Inference for the generalization error," *Machine Learning* 52:239–281, 2003, doi:10.1023/A:1024068626366. The split-level tests below use the corrected resampled t-test variance described in that paper, `(1/J + n_test/n_train) s^2`, with `J = 5`. The paper's own simulations suggested more repeats than five. The p-values are therefore a rough check on split noise, not a confirmatory test.

Phopase, "Calibration and threshold selection of machine learning classifiers for breast tumour diagnosis," Research Square preprint, 27 October 2026, doi:10.21203/rs.3.rs-11491081/v1. That preprint also uses the scikit-learn WDBC table, logistic regression, an RBF SVM, and a random forest, and it also reports that an extra sigmoid can fail to help models that are already reasonably calibrated while isotonic calibration moves the forest. Its design is different: a 455/114 split, repeated stratified cross-validation, no synthetic feature noise, and an added gradient-boosting model. It is not a replication target for the noise experiment, it is not peer reviewed, and none of its numerical results are copied into the tables here.

No claim is made that the noise-plus-recalibration design is new. A search broad enough to support a novelty claim was not done.

## 3. What was archived, and what was run

### 3.1 Archive

`preserved/results/` holds a five-split, three-model noise study: one draw in `results_by_split.csv`, and 50 draws in `noise_draws.csv`. `TASK1_RESULTS.md` in that archive treats the 2.5–97.5 percentile across draws as noise-seed uncertainty and says several single-draw model orderings at sigma = 1 sit inside that uncertainty. Those percentiles were recomputed from the CSV (`tests/test_preserved_csvs.py`). They are not bootstrap confidence intervals, despite one sentence in the archived task list that calls them that.

The explicit protocol does not reproduce that archive. On clean data, logistic-regression ROC-AUC differs by −0.0003 [A1], Brier score by −0.0023 [A2], 10-bin ECE by −0.0152 [A3], and sensitivity by +0.0189 [A4], which is one malignant case in 53. At sigma = 1 the SVM's 10-bin ECE differs by +0.0115 [A5]. Possible reasons, none of which can be checked without the missing script, are the split seed, the scaler, the SVM probability seed, and the ECE bin edges. The archive is left untouched.

### 3.2 Explicit protocol

- Data: `sklearn.datasets.load_breast_cancer`. Malignant is class 1 (212 rows). Benign is class 0 (357 rows).
- Primary splits: `StratifiedShuffleSplit`, 5 splits, test size 0.25 (143 rows, 53 malignant), `random_state=20260311`. Indices are in `results/recalibration/split_indices.csv`.
- Models, untuned: `StandardScaler` + `LogisticRegression(C=1)`; `RandomForestClassifier(n_estimators=100)` on the raw features; `StandardScaler` + `SVC(kernel="rbf", probability=True)` for the baseline arm.
- Recalibration: `CalibratedClassifierCV(method="sigmoid"` or `"isotonic"`, `cv=StratifiedKFold(5, shuffle=True)`, `ensemble=False`) around a base that has not already been calibrated. For the SVM that base uses `probability=False`. The scaler sits inside the pipeline, so each training fit, including each inner fold, sees only its own training rows. A unit test poisons the test fold and checks that the stored training mean does not move.
- Noise: for draw `d`, split `s`, and sigma index `k`, one standard normal matrix is drawn from a seed that does not depend on the model. Test rows become `X + sigma * s_train * Z`, where `s_train` is the training-fold population standard deviation. Sigma is 0, 0.25, 0.5, or 1. Fifty draws. At sigma = 0 the matrix is ignored and predictions match across draws (tested).
- Primary hard-label threshold: 0.5 on `P(malignant)`.
- Metrics: ROC-AUC, Brier score (mean squared error), sensitivity, specificity, F1, and ECE with 10 equal-width bins, 15 equal-width bins, and 10 equal-mass bins.

The run that wrote `results/recalibration/` used Python 3.12.3, scikit-learn 1.9.1, and NumPy 2.2.6. It has 9000 metric rows [N1].

## 4. How uncertainty is reported

Two sources are kept apart.

1. **Noise draws, splits fixed.** For each draw, average the metric over the five splits. The reported interval is the 2.5 and 97.5 percent linear quantile of those 50 numbers. At sigma = 0 the interval collapses to a point. This is not a bootstrap confidence interval and not an interval for a new cohort.
2. **Splits, draws averaged.** Sample standard deviation of the per-split means, and, separately, a Nadeau–Bengio corrected resampled t-test on those five paired differences. With `n_test/n_train ≈ 143/426` and `J = 5`, the corrected standard error is larger than the ordinary standard error of five numbers. Overlapping splits violate the independence story the ordinary error would assume. The correction is an approximation. Forty-eight such tests were computed for Brier score and 10-bin ECE. Thirteen had unadjusted two-sided p < 0.05 [T0]. No multiplicity adjustment was applied. A small p-value is not a clinical finding, and a large p-value is not equivalence.

Paired differences use the same split and the same noise matrix. A difference smaller than `1e-10` in absolute value is treated as a tie so that binary metrics are not ordered by floating-point dust. That rule is what makes the archived paired fractions recompute; see `AUDIT.md`.

## 5. Baseline behaviour under isotropic noise

Clean baseline Brier scores are 0.0235 (logistic regression), 0.0218 (RBF SVM), and 0.0354 (random forest) [C1, C3, C5]. Clean 10-bin ECE is 0.0275, with the forest higher at 0.0443 [C11, C12]. Clean ROC-AUC is 0.9929, 0.9951, and 0.9875 [C16–C18].

At sigma = 1, mean Brier scores are 0.0803, 0.0963, and 0.1158 [C2, C4, C6]. The logistic-regression draw interval for Brier score is about [0.0658, 0.0913], so the noise seed moves the five-split mean by more than the gap between some clean-data model differences. Mean ROC-AUC for logistic regression falls to 0.9628, with draw interval about [0.9558, 0.9754] [C19]. The same pattern, noise hurting every model and hurting the forest's Brier score and ECE most under isotropic noise, is visible in `figures/baseline_noise_bands.png`. Split standard deviations are often comparable to the draw intervals, so a ranking from one split would be unstable even before the noise seed is considered.

## 6. Recalibration fit on clean training data

`figures/recalibration_bands.png` shows the three arms. Reliability diagrams that pool splits and draws are in `figures/reliability_diagrams.png`. Those diagrams repeat the same test rows and must not be read as counts of distinct patients.

### 6.1 Random forest

This is the arm where recalibration moves the scores. At sigma = 1, sigmoid Brier score is 0.0929 against a baseline of 0.1158 [C7, C6]. The paired difference (sigmoid minus baseline) averages −0.0229, the draw interval is about [−0.0271, −0.0198], and the sigmoid arm is better in all 50 draws [P1]. The split-level corrected test gives t ≈ −16.5 on 4 degrees of freedom, unadjusted p ≈ 7.8×10⁻⁵ [T1]. The paired 10-bin ECE difference (sigmoid minus baseline) averages −0.0699, with unadjusted split-level p ≈ 3.8×10⁻⁶ [T3]. The sigmoid mean itself is 0.1121 [C13]. Isotonic calibration also lowers Brier score in all 50 draws, by about 0.0081, with draw interval about [−0.0155, −0.0022] [P3], but the split-level test does not separate that contrast (p ≈ 0.51) [T8]. The draw interval and the split test answer different questions. The first says the noise seed does not flip the sign on these five splits. The second says five overlapping splits do not pin down the mean difference once the Nadeau–Bengio correction is applied.

On clean data the sigmoid Brier improvement is only about 0.0008 [P2], and the split-level test does not separate it (p ≈ 0.79) [T2]. So the forest result that survives both descriptions is the improvement under noise, not a precise clean-data ECE claim.

Isotonic calibration also changes the 0.5 operating point. At sigma = 1, forest sensitivity rises from 0.9061 to 0.9457 [C21, C22] while specificity falls from 0.8534 to 0.7450 [C23, C24]. That is not a free improvement in detection.

Sigmoid calibration does not change forest ROC-AUC, as expected for a strictly monotone map. Isotonic calibration lowers it in every draw (the paired file records fraction smaller = 1 at each sigma).

### 6.2 Logistic regression

Sigmoid recalibration slightly worsens clean Brier score, by about 0.0006 [P6]. At sigma = 1 it improves Brier score by about 0.0024, in all 50 draws, draw interval about [−0.0033, −0.0016] [P5]. The split-level test also separates that contrast (unadjusted p ≈ 0.0025) [T6]. The change is small next to the noise-driven rise from 0.0235 to 0.0803.

Isotonic recalibration goes the other way under strong noise. At sigma = 1, Brier score rises by about 0.0099, draw interval about [0.0059, 0.0142], worse in all 50 draws [P4, C9]. Ten-bin ECE moves with it (unadjusted split-level p ≈ 0.0048) [T5]. ROC-AUC falls by about 0.0277, and the draw interval for that drop stays below zero [P10, C20]. Isotonic calibration fitted on clean training rows does not preserve logistic-regression performance once features are heavily noised. That is a negative result, not a failed run.

### 6.3 RBF SVM

`SVC(probability=True)` already fits a sigmoid on out-of-fold decision values. The explicit sigmoid arm fits the same kind of map on an SVM with `probability=False`, using the same `ensemble=False` design scikit-learn documents as the implementation behind that flag. The two fits need not match numerically, because the internal cross-validation seeds differ.

On clean data the ROC-AUC difference is 0 [P9]. The Brier difference is about −0.00007 [P7]. At sigma = 1 the explicit sigmoid is slightly worse on Brier score in all 50 draws, by about 0.0012, draw interval about [0.0009, 0.0015] [P8]. The split-level test does not separate that gap (p ≈ 0.38) [T7]. Both sentences are true: the noise seed does not flip the sign, and the gap is small relative to split variability. This is not an equivalence claim, and it is not evidence that a second calibration helps.

Isotonic calibration lowers SVM 10-bin ECE at sigma = 1 by about 0.0217, with the draw interval below zero [P11, C14, C15]. It also lowers ROC-AUC relative to the baseline in every draw at sigma = 1 (`paired_calibration.csv`). The ECE gain and the AUC loss are reported together.

### 6.4 Binning

ECE orderings are not stable [E1]. On clean logistic regression, 10-bin and 15-bin equal-width ECE rank isotonic best, while 10-bin equal-mass ECE ranks the baseline best. At sigma = 0.5 the three binning rules produce three different logistic-regression orderings. With 143 test rows, several equal-width bins are sparse. An ECE ranking on this sample is a property of the binning rule as well as of the model.

## 7. Other corruption mechanisms

Baseline models only. Same five splits. Fifty draws. Zero recorded numerical failures [N2]. Isotropic Brier means match the primary baseline arm exactly [N3, K8], which is a check that the shared seed stream was reused rather than a new scientific result.

At level 1, correlated Gaussian noise (population training covariance) raises logistic-regression Brier score to 0.2420, the SVM to 0.2794, and the forest to 0.1925 [K1–K3]. Multiplicative noise `X * (1 + Z)` raises the SVM's Brier score to 0.3728 and the forest's to 0.1669 [K4, K5]. Under these two mechanisms the forest's Brier score is lower than the SVM's, which is the reverse of the isotropic ordering at sigma = 1 (forest 0.1158, SVM 0.0963). The ordering is mechanism-dependent. Noise restricted to the ten features whose names start with "worst" is milder: logistic-regression Brier score 0.0537 at level 1 [K6], against 0.0803 for isotropic noise on all 30 features. Replacing each test feature independently with the training mean at probability 0.5 gives logistic-regression Brier score 0.0525 [K7].

These are synthetic corruptions. A lower Brier score under one of them is not a claim of robustness to clinical measurement error. `figures/corruptions_brier.png` shows the curves. Draw bands are percentiles across draws.

## 8. More splits

`configs/more_splits.json` asks for 30 stratified holdouts from the same master seed, so the first five splits are the primary five, and for 20 draws rather than 50. Split-to-split standard deviations do not shrink in general. Clean logistic-regression ROC-AUC split SD is 0.0047 using the first five splits and 0.0048 using all thirty [S1, S2]. Forest Brier-score split SD at sigma = 1 goes from 0.0062 to 0.0079 [S3, S4]. Forest clean ECE split SD does fall, from 0.0164 to 0.0120 [S5, S6]. With only five splits the sample SD is itself noisy. The honest summary is that moving from 5 to 30 splits did not produce a uniform reduction, not that the variance "stabilised" at a smaller value.

## 9. Thresholds and false-negative costs

Thresholds for false-negative:false-positive costs 1:1, 5:1, and 10:1 were chosen by minimising `cost_fn * FN + cost_fp * FP` on stratified out-of-fold predictions from the training fold only, then frozen and applied to the test fold. Ties prefer the threshold closer to 0.5, then the smaller one. The test labels are not an argument of `select_cost_threshold`. The saved threshold file records that selection rule on every row.

Higher false-negative cost lowers the threshold and buys sensitivity by spending specificity. At sigma = 1 and cost 10:1, SVM specificity averages 0.5324 while sensitivity averages 0.9833 [H1, H2]. Forest specificity averages 0.2949 [H3]. Logistic-regression mean cost per test row at that setting is 0.3623, with a wide draw interval about [0.2619, 0.4298] [H4], against 0.0266 on clean data at cost 1:1 [H5]. The training-chosen logistic threshold at cost 10:1 already ranges from 0.117 to 0.375 across the five training folds [H6]. Thresholds selected on this training size are unstable, and the test costs above are not utilities for a clinic.

## 10. Discussion

On this table, with these defaults, clean-training recalibration is not uniformly helpful once features are noised.

- The forest, whose raw probabilities are known to shrink toward the centre, is where sigmoid recalibration reduces Brier score and 10-bin ECE under isotropic noise, including at sigma = 1, in every draw of this design. The clean-data Brier gap is small and is not separated by the split-level test.
- Logistic regression does not need a second sigmoid on clean data. Under heavy isotropic noise the sigmoid shift is a small Brier improvement and the isotonic shift is a clear deterioration, with a loss of ROC-AUC.
- The SVM baseline is already Platt-scaled. Explicit sigmoid recalibration is nearly the same procedure with a different internal split seed. Isotonic recalibration can lower ECE and also lower AUC.
- Which corruption you add changes which model looks worse. That is a reason not to treat the isotropic ranking as the robustness ranking.
- ECE binning changes orderings at this test size. Reporting one ECE and ranking models by it is not supported here.

Nothing in these experiments estimates harm, benefit, or net benefit for a patient. The costs in Section 9 are multipliers chosen in the config file.

## 11. Threats to validity

- One dataset, constructed features, no site shift.
- Overlapping resamples and `J = 5` for the primary test. The corrected t-test is anti-conservative or conservative depending on whether the assumed correlation matches the real one (Nadeau and Bengio, 2003). With 4 degrees of freedom it is easy to over-read.
- Forty-eight related contrasts, unadjusted.
- Untuned hyperparameters and one feature scaling convention.
- Synthetic noise, including a covariance ridge and unclipped multiplicative noise.
- The archived study could not be audited for leakage because its code is missing. The new code's leakage checks do not retroactively clear the archive.
- Library versions differ from the archive stamp.

## 12. Conclusion

Recalibration fitted on clean WDBC training folds does not reliably survive, or improve, probability quality under synthetic feature noise. It helps the random forest's Brier score and 10-bin ECE under isotropic noise in this design. It slightly helps logistic regression's Brier score at sigma = 1 when the map is sigmoid, and it hurts logistic regression when the map is isotonic. It is nearly redundant for an SVM that was already producing Platt probabilities, and a second sigmoid is slightly worse on Brier score across noise draws without being separated from split noise. ECE rankings depend on the bins. These are descriptive results on 569 rows.

## 13. Reproducibility statement

Commands, seeds, and the interpreter that wrote the CSVs are in `REPRODUCIBILITY.md`. Tests that recompute the archive and lock the new row counts are in `tests/`. The archive itself cannot be regenerated from this repository. No clinical use is intended.

## References

1. scikit-learn developers. Section 1.16, Probability calibration. scikit-learn 1.9.1 User Guide. https://scikit-learn.org/stable/modules/calibration.html (page fetched 5 November 2026).
2. Nadeau C, Bengio Y. Inference for the generalization error. *Machine Learning*. 2003;52:239–281. doi:10.1023/A:1024068626366.
3. Phopase K. Calibration and threshold selection of machine learning classifiers for breast tumour diagnosis: a decision-support study for AI-driven clinical screening. Research Square preprint. Posted 27 October 2026. doi:10.21203/rs.3.rs-11491081/v1. Different protocol; not peer reviewed; not a source of the numbers in this file.
4. Platt JC. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. 1999. Cited from the scikit-learn 1.9.1 user guide, reference [4] of section 1.16. Not re-read in full for this note.
5. Niculescu-Mizil A, Caruana R. Predicting good probabilities with supervised learning. ICML 2005. Cited from the same user-guide section, reference [3]. Not re-read in full for this note.
6. Murphy AH. A new vector partition of the probability score. *Journal of Applied Meteorology and Climatology*. 1973;12(4):595–600. Cited from the same user-guide section, reference [1], for the Brier-score decomposition. Not re-read in full for this note.
