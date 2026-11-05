# Calibration fitted on clean rows and scored under synthetic feature noise

A methods note. Not a clinical validation. Not peer reviewed. Not a medical device.

Numbers below are rounded to four decimal places from `CLAIM_EVIDENCE_MATRIX.md`, which is generated from the CSVs by `python -m wdbc_calib.write_claims`. Full precision stays in the CSVs. Claim ids in brackets point at that matrix.

## Abstract

Three standard classifiers were fit on clean training rows and scored under shared isotropic feature noise. On the Wisconsin Diagnostic Breast Cancer table, sigmoid recalibration lowers the random forest's Brier score at sigma = 1 by about 0.023 in every one of 50 draws [P1], and isotonic recalibration raises logistic regression's Brier score by about 0.010 and lowers its ROC-AUC in every draw [P4, P10]. The same protocol on eight public numeric binary tables, including that one, does not keep those signs. At sigma = 1, isotonic recalibration of logistic regression raises Brier score in every draw on four tables and lowers it in every draw on sonar (about −0.033) [R-sigma1-brier-logistic-isotonic, X45]. Sigmoid recalibration of the forest lowers Brier score in every draw on four tables and raises it in every draw on banknote (about +0.026) and ionosphere (about +0.038) [R-sigma1-brier-random-sigmoid, X39, X23]. This note does not call either WDBC pattern a general result, and it is not a publication-ready claim of a new calibration law. Dataset size, feature count, class balance, clean-data calibration, clean-data discrimination, and the baseline's Brier increase under noise were then compared with those Brier changes across the eight tables. Thirty-four of 36 dataset-bootstrap intervals cover zero, and none of the 36 rank correlations survives a Holm adjustment [M0]. None of it is a statement about clinical benefit.

## 1. Introduction

Accuracy and ROC-AUC on this table are easy to push near the ceiling. A probability that will be thresholded, or read as a risk, can still be miscalibrated, and a ranking that is stable on clean features can move once features are corrupted. The question asked here is narrow:

Does probability recalibration fitted only on clean training rows improve or preserve calibration when the same fitted models are evaluated under synthetic feature corruption?

The answer is allowed to be no. Deterioration and near-ties are reported with the improvements.

This file is not a revision of a recovered manuscript. At the commit audited in `AUDIT.md`, this GitHub repository contained a README and an install check. Uploaded tables and a figure were archived under `preserved/`. The scripts that produced them were added later, in commit `636d281` (5 November 2026, 21:59 IST), and stored unchanged at `preserved/original_source/`. Section 3 separates that reproduced table from the explicit protocol that was actually run for the recalibration question. The line-by-line differences are in `PROTOCOL_COMPARISON.md`.

## 2. Related work that was checked

The following sources were opened and are the only ones cited. Counts in this section and in Section 2.2 are quotations from those pages. They are not rows of `results/`.

scikit-learn 1.9.1, User Guide section 1.16, "Probability calibration" (https://scikit-learn.org/stable/modules/calibration.html), states that a calibrated binary classifier should have an observed frequency near 0.8 among rows scored near 0.8; that the Brier score mixes calibration and discrimination (Murphy, 1973, as cited there); that sigmoid calibration is Platt's logistic map; that isotonic calibration is more flexible and overfits more easily on small samples; that isotonic maps can change ROC-AUC by introducing ties while a sigmoid map is strictly monotone; and that random forests tend to pull probabilities away from 0 and 1. The same page cites Niculescu-Mizil and Caruana, "Predicting good probabilities with supervised learning," ICML 2005, and Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods," 1999. Those two papers are cited here only as the documents named by that user-guide page. They were not re-read in full.

Nadeau and Bengio, "Inference for the generalization error," *Machine Learning* 52:239–281, 2003, doi:10.1023/A:1024068626366. The split-level tests below use the corrected resampled t-test variance described in that paper, `(1/J + n_test/n_train) s^2`, with `J = 5`. The paper's own simulations suggested more repeats than five. The p-values are therefore a rough check on split noise, not a confirmatory test.

Phopase, "Calibration and threshold selection of machine learning classifiers for breast tumour diagnosis," Research Square preprint, posted 27 October 2026, doi:10.21203/rs.3.rs-11491081/v1. The author is named there as Kartik Phopase (Sanjivani University). The page states that the preprint has not been peer reviewed. The full text fetched for this note is the Research Square HTML, not a journal version. Overlap with the explicit protocol in this repository, and what is not shared, is in Section 2.1. None of that preprint's table entries are copied into the results here.

No claim is made that the noise-plus-recalibration design is new. A search broad enough to support a novelty claim was not done.

## 2.1 Overlap with the Phopase preprint

Read from the Research Square HTML for doi:10.21203/rs.3.rs-11491081/v1. This is an overlap assessment, not an attribution of copying in either direction.

Shared with the explicit protocol here: the scikit-learn WDBC table (569 rows, 212 malignant); malignant coded as the positive class; logistic regression, an RBF SVM, and a random forest; standardisation inside the training fold; no hyperparameter search; the remark that `SVC` probabilities are already an internal Platt scaling; sigmoid and isotonic maps fit on training data only; Brier score and 10-bin ECE; a Nadeau–Bengio corrected resampled t-test; a threshold chosen from training out-of-fold probabilities and then frozen; and a closing statement that the work is not a clinical validation.

Not shared: that preprint uses one stratified 455/114 split (42 malignant and 72 benign in the test piece; random seed 42; scikit-learn 1.8.0), then 5×10-fold cross-validation on the training piece. This repository uses repeated 75/25 holdouts (143 test rows, 53 malignant) with master seed 20260311 and scikit-learn 1.9.1. Their forest has 500 trees; the forest here has 100. They add gradient boosting, a majority-class baseline, and a 10-feature "mean only" scenario. They do not add synthetic feature noise. Their threshold is the largest score with out-of-fold sensitivity at least 0.95. The thresholds here minimise declared false-negative to false-positive cost ratios. Their test-set intervals are percentile bootstraps of one holdout. The intervals here are percentiles across noise draws, kept separate from the split-level test.

Their Table IV, scenario B, raw logistic regression is Brier 0.021 and ECE 0.030 on 114 test rows. The explicit protocol's clean logistic regression is Brier 0.0235 and 10-bin ECE 0.0275, averaged over five larger holdouts [C1, C11]. The figures are in the same neighbourhood and come from different splits. They were not used as inputs.

Their discussion already says that an extra sigmoid can make logistic regression and the SVM worse, and that isotonic calibration helps the forest, on clean data. That clean-data pattern is therefore not a finding this repository can call its own. A second 2026 preprint, on a demographic calibration gap using WDBC and MIMIC-IV (reference 11), asks a different question and is not a source for these noise results. Its text was not used beyond the abstract.

## 2.2 What is already established, what was only a WDBC pattern, and what is still open

The sources in this section were opened for this assessment. They are not a systematic review.

**Already established, and not claimed here.**

Manokhin and Grønhaug, "When Post-Hoc Calibration Hurts," Proceedings of Machine Learning Research 329:378–381, 2026 (COPA, 2–4 September 2026), report 3,150 runs per calibrator: 21 classifiers, 30 binary TabArena-v0.1 tasks, 5-fold cross-validation. Isotonic regression has a mean relative log-loss change of +3.22% and improves log-loss in 43.7% of runs. Platt scaling improves log-loss in 49.8%. Mean AUC-ROC changes by under 1%, and mean accuracy is essentially unchanged except under Pearsonify. The same headline counts appear in their longer preprint, arXiv:2601.19944 (submitted 19 January 2026). The fetched HTML of that preprint says distributional shifts were omitted in order to keep an i.i.d. assumption. Their COPA table, as extracted from the proceedings PDF, gives isotonic regression a mean relative Brier change of −3.74% with an improvement rate of 52.5%, so log-loss and Brier do not even share a sign in that aggregate. The established statement is that post-hoc calibration can hurt on clean i.i.d. tabular data, and that the hurt is model-dependent. It is not a statement about feature noise.

Ovadia and colleagues, "Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift" (NeurIPS 2019; arXiv:1906.02530 HTML read here), find that temperature scaling fit on an i.i.d. validation set does not stay well calibrated as image corruptions grow, and that on the Criteo click table a temperature-scaled model has worse Brier score than the uncalibrated network when categorical features are randomized. That is post-hoc calibration under shift. The models are deep networks, the calibrator is temperature scaling, and the corruptions are not the isotropic Gaussian feature noise used below.

The scikit-learn 1.9.1 calibration guide, already cited, says isotonic regression overfits small samples and can change ROC-AUC by introducing ties, while a sigmoid map is strictly monotone. On the eight tables in Section 7, isotonic recalibration of logistic regression lowers clean-data ROC-AUC in every draw on seven tables and ties the eighth [R-sigma0-roc_auc-logistic-isotonic]. That is the known tie effect, measured again. It is not a noise result.

Phopase's clean-data remark that an extra sigmoid need not help logistic regression or an already Platt-scaled SVM is also measured again. At sigma = 0, logistic sigmoid-minus-baseline Brier is positive on five of the eight tables and negative on three, and every one of those eight signs is unanimous across draws [R-sigma0-brier-logistic-sigmoid].

**WDBC patterns that the cross-dataset run does not carry.**

Two descriptions were true on WDBC and were not in the sources above: isotonic recalibration fit on clean rows raises logistic-regression Brier score at isotropic sigma = 1 in every draw, and sigmoid recalibration lowers the forest's Brier score at that same sigma in every draw. Section 7 applies the same seed, estimators, and noise generator to seven further public numeric tables. Neither description holds on all eight. Sonar reverses the logistic isotonic Brier sign. Banknote and ionosphere reverse the forest sigmoid sign.

**Still required before a general claim.**

A wider dataset sample than these eight convenient numeric tables. A shift that is not isotropic Gaussian noise in the original feature units. The beta and Venn–Abers calibrators that Manokhin and Grønhaug found more reliable on i.i.d. data, which this repository does not run. Section 7.1 does apply Holm to the existing split-level tests and to the rank correlations. That adjustment was documented after the unadjusted counts, not locked before the cross-dataset run. The file is still not publication-ready evidence of a new calibration rule.

Virdee (reference 10) was opened. The preprint reports isotonic Brier sign changes across four conditions of one corpus and states that no external transport is claimed. The fetched page does not identify those conditions with isotropic feature noise, so none of its figures are used.

## 3. What was archived, and what was run

### 3.1 Archive

`preserved/results/` holds a five-split, three-model noise study: one draw in `results_by_split.csv`, and 50 draws in `noise_draws.csv`. `TASK1_RESULTS.md` in that archive treats the 2.5–97.5 percentile across draws as noise-seed uncertainty and says several single-draw model orderings at sigma = 1 sit inside that uncertainty. Those percentiles were recomputed from the CSV (`tests/test_preserved_csvs.py`). They are not bootstrap confidence intervals, despite one sentence in the archived task list that calls them that.

On 5 November 2026 at 22:04 IST `reproduce_study.py` and `noise_draws.py` were run from `preserved/original_source/` on Python 3.12.3, scikit-learn 1.9.1, and NumPy 2.2.6. Every numeric column in the five archived result CSVs matches that run. The largest value in column `max_abs_diff` of `results/original_source_check/numeric_agreement.csv` is 2.1×10⁻¹⁶. The archived environment stamp still says Python 3.13.18 and NumPy 2.5.3, and the CSV text is not byte-identical. The files under `preserved/` were not replaced. `results/original_source_check/numeric_agreement.csv` records the comparison. Those preserved tables are a direct reproduction of the original scripts.

They are not a direct reproduction of `wdbc_calib/`. The original split seed is 2026. The forest has 200 trees and `min_samples_leaf=2`. There is no sigmoid or isotonic arm after the base model. The one-draw noise seed is `default_rng([2026, split, int(sigma * 100)])`, and the 50-draw seed is `default_rng([2026, split, draw, int(round(sigma * 100))])`. The module docstring calls those seeds assumptions. The one-draw file is therefore not draw 0 of the 50-draw file.

The explicit protocol uses split seed 20260311, a 100-tree forest with the default leaf size, and `CalibratedClassifierCV(..., ensemble=False)` for sigmoid and isotonic maps. On clean data, its logistic-regression ROC-AUC differs from the archive by −0.0003 [A1], Brier score by −0.0023 [A2], 10-bin ECE by −0.0152 [A3], and sensitivity by +0.0189 [A4], which is one malignant case in 53. At sigma = 1 the SVM's 10-bin ECE differs by +0.0115 [A5]. Those deltas are the two protocols, not a failed rerun of the original script. The 10-bin ECE formulas agree on probabilities in [0, 1], so the ECE gap is not a bin-edge artifact. Nothing under `results/recalibration/`, `results/corruptions/`, `results/thresholds/`, or `results/more_splits/` is labelled a direct reproduction of `reproduce_study.py`.

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

Three interval types are named separately. "Draw interval" means the first. "Dataset-bootstrap interval" means the third. A split-level p-value is the second, and it is unadjusted unless the sentence says Holm.

1. **Noise draws, splits fixed.** For each draw, average the metric over the five splits. The reported interval is the 2.5 and 97.5 percent linear quantile of those 50 numbers. At sigma = 0 the interval collapses to a point. This is not a bootstrap confidence interval and not an interval for a new cohort.
2. **Splits, draws averaged.** Sample standard deviation of the per-split means, and, separately, a Nadeau–Bengio corrected resampled t-test on those five paired differences. Every primary split has 143 test rows and 426 training rows (`results/recalibration/split_indices.csv`), and `J = 5`. The corrected standard error is larger than the ordinary standard error of five numbers. Overlapping splits violate the independence story the ordinary error would assume. The correction is an approximation. Forty-eight such tests were computed for Brier score and 10-bin ECE. Thirteen had unadjusted two-sided p < 0.05 [T0]. Section 7.1 applies Holm to this same family afterwards: 5 of 48 remain [M-wdbc_brier_ece_48]. The unadjusted count is left in place. A small p-value is not a clinical finding, and a large p-value is not equivalence.
3. **Eight datasets, one correlation.** Spearman intervals in Section 7.1 are 2.5 and 97.5 percent percentiles from 4,000 bootstrap resamples of the eight datasets. They are not intervals across the 50 noise draws, and an endpoint at −1 or +1 is a small-sample resample, not a slope. The permutation p-values that enter Holm for that family also treat the dataset as the unit. "Draw interval" always means the first of these three objects. "Dataset-bootstrap interval" means the third. Neither phrase is a confidence interval for a new patient or a new table.

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

Sigmoid calibration does not change forest ROC-AUC, as expected for a strictly monotone map. Isotonic calibration lowers it in every draw. In `results/recalibration/paired_calibration.csv`, the forest isotonic-minus-baseline ROC-AUC rows have `frac_a_less` = 1 and `frac_a_greater` = 0 at sigma 0, 0.25, 0.5, and 1.

### 6.2 Logistic regression

Sigmoid recalibration slightly worsens clean Brier score, by about 0.0006 [P6]. At sigma = 1 it improves Brier score by about 0.0024, in all 50 draws, draw interval about [−0.0033, −0.0016] [P5]. The split-level test also separates that contrast (unadjusted p ≈ 0.0025) [T6]. The change is small next to the noise-driven rise from 0.0235 to 0.0803.

Isotonic recalibration goes the other way under strong noise. At sigma = 1, Brier score rises by about 0.0099, draw interval about [0.0059, 0.0142], worse in all 50 draws [P4, C9]. Ten-bin ECE moves with it (unadjusted split-level p ≈ 0.0048) [T5]. ROC-AUC falls by about 0.0277, and the draw interval for that drop stays below zero [P10, C20]. On this table, isotonic calibration fitted on clean training rows does not preserve logistic-regression Brier score or ROC-AUC once features are heavily noised. That is a negative result on WDBC, not a failed run. Section 7 asks whether the sign travels.

### 6.3 RBF SVM

`SVC(probability=True)` already fits a sigmoid on out-of-fold decision values. The explicit sigmoid arm fits the same kind of map on an SVM with `probability=False`, using the same `ensemble=False` design scikit-learn documents as the implementation behind that flag. The two fits need not match numerically, because the internal cross-validation seeds differ.

On clean data the ROC-AUC difference is 0 [P9]. The Brier difference is about −0.00007 [P7]. At sigma = 1 the explicit sigmoid is slightly worse on Brier score in all 50 draws, by about 0.0012, draw interval about [0.0009, 0.0015] [P8]. The split-level test does not separate that gap (p ≈ 0.38) [T7]. Both sentences are true: the noise seed does not flip the sign, and the gap is small relative to split variability. This is not an equivalence claim, and it is not evidence that a second calibration helps.

Isotonic calibration lowers SVM 10-bin ECE at sigma = 1 by about 0.0217, with the draw interval below zero [P11, C14, C15]. It also lowers ROC-AUC relative to the baseline in every draw at sigma = 1 (`paired_calibration.csv`). The ECE gain and the AUC loss are reported together.

### 6.4 Binning

ECE orderings are not stable [E1]. On clean logistic regression, 10-bin and 15-bin equal-width ECE rank isotonic best, while 10-bin equal-mass ECE ranks the baseline best. At sigma = 0.5 the three binning rules produce three different logistic-regression orderings. With 143 test rows, several equal-width bins are sparse. An ECE ranking on this sample is a property of the binning rule as well as of the model.

## 7. The same protocol on eight tables

`configs/cross_dataset.json` keeps the WDBC master seed, the five stratified 75/25 splits, the 50 draws, the sigma grid, and the three calibration arms. The dataset name is not part of the seed. WDBC rows from this run match `results/recalibration/metrics_long.csv` to a maximum absolute difference of 1.1×10⁻¹⁶ on 9,000 rows [X0b]. The other tables are Pima Indians diabetes (OpenML 37, positive = tested_positive), ionosphere (OpenML 59, positive = bad return), blood transfusion (OpenML 1464, positive = the 178 March 2007 donations), banknote authentication (OpenML 1462, positive = the 610 rows matching the UCI forged count), sonar (OpenML 40, positive = mine), QSAR biodegradation (OpenML 1494, positive = the 356 ready-biodegradable chemicals), and spambase (OpenML 44, positive = spam). All eight are numeric and have no missing values. Row counts, feature counts, and positive-class counts in the previous sentence are the columns of `results/cross_dataset/dataset_card.csv`. The run has 72,000 metric rows [X0]. It finished at 22:33 IST on 5 November 2026 (17:03 UTC), on Python 3.12.3, scikit-learn 1.9.1, and NumPy 2.2.6.

`figures/cross_dataset_brier.png` shows the pre-specified Brier contrasts. A positive value means the recalibrated arm has a higher Brier score than the baseline.

At sigma = 1, logistic isotonic minus baseline Brier is worse in every draw on banknote, blood transfusion, spambase, and WDBC, better in every draw on sonar (mean difference −0.0334, draw interval about [−0.0439, −0.0244]) [X45], and not unanimous on ionosphere, Pima, and QSAR [R-sigma1-brier-logistic-isotonic]. The ROC-AUC version of that contrast is worse in every draw on five tables, including WDBC, and not unanimous on blood transfusion, Pima, and sonar [R-sigma1-roc_auc-logistic-isotonic]. The clean-data AUC drop is already unanimous on seven tables [R-sigma0-roc_auc-logistic-isotonic], so a lower AUC after isotonic regression is not, by itself, news about noise.

At sigma = 1, forest sigmoid minus baseline Brier is better in every draw on blood transfusion, QSAR, sonar, and WDBC (WDBC mean −0.0229) [X07], worse in every draw on banknote (mean +0.0258) [X39] and ionosphere (mean +0.0380) [X23], and not unanimous on Pima and spambase [R-sigma1-brier-random-sigmoid].

At sigma = 1, logistic sigmoid minus baseline Brier is better in every draw on seven tables and worse in every draw on banknote [R-sigma1-brier-logistic-sigmoid]. At sigma = 0 the same contrast is already worse on five tables and better on three [R-sigma0-brier-logistic-sigmoid].

Of the 64 pre-specified split-level corrected t-tests (four contrasts, two sigmas, eight datasets), 27 have unadjusted two-sided p < 0.05 [R-split-tests]. With five overlapping splits those p-values are a rough screen. Section 7.1 applies a Holm adjustment to that family and to the separate WDBC family of 48 Brier and ECE tests.

### 7.1 Do dataset characteristics explain the Brier change?

The question is whether something measurable about a table lines up with the sign of clean-trained recalibration under this noise. The characteristics are the number of rows, the number of features, the smaller-class rate, the baseline model's 10-bin ECE at sigma = 0, its ROC-AUC at sigma = 0, and the increase in its Brier score from sigma = 0 to sigma = 1. The outcome is the paired Brier difference at sigma = 1, recalibrated arm minus baseline, for the six contrasts that the experiment already computed (sigmoid and isotonic, for each of the three models). Each correlation uses eight datasets. The 50 draws are the within-dataset percentile bar in `figures/brier_effect_by_dataset.png`. They are not 400 independent tables, and the Spearman bootstrap resamples datasets, not draws.

No line was fit. Spearman rho is the only association statistic. Its uncertainty is a 2.5–97.5 percent interval from 4,000 bootstrap resamples of the eight datasets, plus the range of the correlation when each dataset is left out once. Thirty-six permutation tests, two-sided and add-one corrected, were Holm-adjusted as one family. The procedure is in `results/association/METHOD.txt`.

Thirty-four of the 36 bootstrap intervals include zero. None of the 36 Holm-adjusted permutation p-values is below 0.05 [M0]. The two intervals that exclude zero are both logistic regression against feature count. Sigmoid minus baseline has rho −0.8743, interval [−1, −0.2895], permutation p 0.0078, Holm p 0.2808, and leave-one-out rhos from −0.9190 to −0.8108 [M-rho-1]. Isotonic minus baseline has rho −0.8024, interval [−1, −0.1358], permutation p 0.0236, Holm p 0.8260, and leave-one-out rhos from −0.9550 to −0.7027 [M-rho-2]. An interval that reaches −1 is what happens when eight points are resampled. It is not a slope that can be used to choose a calibrator. Rows, class balance, clean ECE, clean ROC-AUC, and the baseline's own Brier increase under noise all have intervals that include positive and negative associations (`figures/spearman_associations.png`). The scatters are `figures/association_scatters.png`. They are eight points and no fitted line.

Holm on the existing split-level tests changes which within-table contrasts can be talked about, and it does not remove the disagreement across tables. In the 64-test cross-dataset family, 27 unadjusted p-values were below 0.05 and 8 remain after Holm [M-cross_dataset_64]. All eight are at sigma = 1. They do not share a sign: banknote logistic isotonic ROC-AUC −0.0633 (Holm p 1.26×10⁻⁵) [M-holm-01], banknote logistic sigmoid Brier +0.0151 [M-holm-02], banknote logistic isotonic Brier +0.0240 [M-holm-03], QSAR logistic sigmoid Brier −0.0097 [M-holm-04], banknote forest sigmoid Brier +0.0258 [M-holm-05], WDBC forest sigmoid Brier −0.0229 [M-holm-06], sonar logistic sigmoid Brier −0.0484 [M-holm-07], and spambase logistic sigmoid Brier −0.0047 [M-holm-08]. The WDBC logistic isotonic Brier increase of 0.0099 has unadjusted p 0.0020 and Holm p 0.1077 in this family, so it is outside the eight. In the separate 48-test WDBC family, 13 unadjusted Brier or ECE tests were below 0.05 and 5 remain [M-wdbc_brier_ece_48]. All five are forest sigmoid minus baseline, at sigma 0.25, 0.5, or 1, for Brier or 10-bin ECE [M-holm-09–M-holm-13]. None is a clean-data contrast. A within-WDBC forest result that survives Holm is still one table. Banknote's opposite forest sign survived the other family.

The characteristics examined here do not explain when recalibration helps. That is a statement about eight datasets and six numbers per model, not a proof that no such characteristic exists. The Holm step and the 36-correlation family were written down for this analysis after the unadjusted cross-dataset counts already existed. They were not a rule stored before those p-values were seen.

## 8. Other corruption mechanisms

Baseline models only. Same five splits. Fifty draws. Zero recorded numerical failures [N2]. Isotropic Brier means match the primary baseline arm exactly [N3, K8], which is a check that the shared seed stream was reused rather than a new scientific result.

At level 1, correlated Gaussian noise (population training covariance) raises logistic-regression Brier score to 0.2420, the SVM to 0.2794, and the forest to 0.1925 [K1–K3]. Multiplicative noise `X * (1 + Z)` raises the SVM's Brier score to 0.3728 and the forest's to 0.1669 [K4, K5]. Under these two mechanisms the forest's Brier score is lower than the SVM's, which is the reverse of the isotropic ordering at sigma = 1 (forest 0.1158, SVM 0.0963). The ordering is mechanism-dependent. Noise restricted to the ten features whose names start with "worst" is milder: logistic-regression Brier score 0.0537 at level 1 [K6], against 0.0803 for isotropic noise on all 30 features. Replacing each test feature independently with the training mean at probability 0.5 gives logistic-regression Brier score 0.0525 [K7].

These are synthetic corruptions. A lower Brier score under one of them is not a claim of robustness to clinical measurement error. `figures/corruptions_brier.png` shows the curves. Draw bands are percentiles across draws.

## 9. More splits

`configs/more_splits.json` asks for 30 stratified holdouts from the same master seed, so the first five splits are the primary five, and for 20 draws rather than 50. Split-to-split standard deviations do not shrink in general. Clean logistic-regression ROC-AUC split SD is 0.0047 using the first five splits and 0.0048 using all thirty [S1, S2]. Forest Brier-score split SD at sigma = 1 goes from 0.0062 to 0.0079 [S3, S4]. Forest clean ECE split SD does fall, from 0.0164 to 0.0120 [S5, S6]. With only five splits the sample SD is itself noisy. The honest summary is that moving from 5 to 30 splits did not produce a uniform reduction, not that the variance "stabilised" at a smaller value.

## 10. Thresholds and false-negative costs

Thresholds for false-negative:false-positive costs 1:1, 5:1, and 10:1 were chosen by minimising `cost_fn * FN + cost_fp * FP` on stratified out-of-fold predictions from the training fold only, then frozen and applied to the test fold. Ties prefer the threshold closer to 0.5, then the smaller one. The test labels are not an argument of `select_cost_threshold`. The saved threshold file records that selection rule on every row.

Higher false-negative cost lowers the threshold and buys sensitivity by spending specificity. At sigma = 1 and cost 10:1, SVM specificity averages 0.5324 while sensitivity averages 0.9833 [H1, H2]. Forest specificity averages 0.2949 [H3]. Logistic-regression mean cost per test row at that setting is 0.3623, with a wide draw interval about [0.2619, 0.4298] [H4], against 0.0266 on clean data at cost 1:1 [H5]. The training-chosen logistic threshold at cost 10:1 already ranges from 0.117 to 0.375 across the five training folds [H6]. Thresholds selected on this training size are unstable, and the test costs above are not utilities for a clinic.

## 11. Discussion

On WDBC, with these defaults, clean-training recalibration is not uniformly helpful once features are noised. That paragraph does not survive contact with the other seven tables.

- The forest result that looked stable on WDBC, a sigmoid Brier improvement at sigma = 1 in every draw, reverses on banknote and ionosphere, also in every draw [X39, X23]. A known tendency of forests to shrink probabilities does not decide the sign under this noise.
- The logistic isotonic deterioration at sigma = 1 is a WDBC fact [X05] and a sonar fact in the opposite direction [X45]. Pima and QSAR sit on the fence: the draw interval includes both signs.
- Logistic sigmoid recalibration at sigma = 1 lowers Brier score in every draw on seven of the eight tables and raises it in every draw on banknote [R-sigma1-brier-logistic-sigmoid]. The clean-data signs were already mixed [R-sigma0-brier-logistic-sigmoid], which is the Manokhin and Grønhaug observation in miniature. The noise did not turn that into a universal improvement.
- The SVM baseline is already Platt-scaled. A second sigmoid remains a small Brier gap on these tables, with the sign depending on the table. That is consistent with two fits of the same kind of map, not with a new robustness method.
- Which corruption you add on WDBC changes which model looks worse. Which table you add changes which calibrator looks worse. Neither ranking is the robustness ranking.
- ECE binning changes orderings at the WDBC test size. Reporting one ECE and ranking models by it is not supported here.

Nothing in these experiments estimates harm, benefit, or net benefit for a patient. The costs in Section 10 are multipliers chosen in the config file. The package is not publication-ready as a general result. Section 7.1 is the reason the dataset characteristics do not rescue a general rule. The missing pieces are listed in Section 2.2, and the hypotheses that would need a new sample are in Section 14.

## 12. Threats to validity

- The WDBC sections are one dataset. The cross-dataset section is eight public numeric tables chosen because they load from scikit-learn or OpenML without categorical encoding. That is not a random sample of tabular problems, and it is not a site shift.
- Overlapping resamples and `J = 5`. The corrected t-test is anti-conservative or conservative depending on whether the assumed correlation matches the real one (Nadeau and Bengio, 2003). With 4 degrees of freedom it is easy to over-read. Sixty-four cross-dataset tests had 27 unadjusted p-values below 0.05 and 8 Holm survivors [R-split-tests, M-cross_dataset_64]. The separate WDBC Brier/ECE family goes from 13 to 5 [M-wdbc_brier_ece_48]. Holm was applied after those unadjusted counts. It does not make the surviving signs agree across tables.
- The association sample is the same eight tables. Resampling them is not a new sample. Feature count is the only characteristic whose logistic-regression intervals excluded zero, and those two tests do not survive Holm.
- Untuned hyperparameters and one feature scaling convention.
- Synthetic noise, including a covariance ridge and unclipped multiplicative noise.
- The original scripts fit the scaler and the models on the training split only and use a fixed threshold of 0.5. Their noise seed is an assumption written in the script. The leakage tests in `tests/` cover `wdbc_calib/`. They were not retrofitted onto the archived functions.
- Library versions differ from the archive stamp. The numeric tables still matched to 2.1×10⁻¹⁶ on this machine.

## 13. Conclusion

On WDBC alone, sigmoid recalibration lowers the forest's Brier score at sigma = 1 in every draw of this design, and isotonic recalibration raises logistic regression's Brier score and lowers its ROC-AUC in every draw. Those sentences are about 569 rows and one noise generator. The same protocol on eight public numeric tables does not keep either sign [R-sigma1-brier-logistic-isotonic, R-sigma1-brier-random-sigmoid]. Sonar reverses the logistic isotonic Brier result. Banknote and ionosphere reverse the forest sigmoid result. Logistic sigmoid-minus-baseline Brier at sigma = 1 is negative on seven tables and positive on banknote, in every draw on each of them [R-sigma1-brier-logistic-sigmoid]. That is still one counterexample, not a rule.

The clean-data pattern that post-hoc calibration can hurt, and that isotonic regression can move ROC-AUC, is already in Manokhin and Grønhaug (2026) and in the scikit-learn calibration guide. Temperature scaling fit on clean validation data failing under shift is already in Ovadia and colleagues (2019), for deep networks and a different corruption. This file does not add a general law on top of those results. It records that two WDBC signs do not survive a small cross-dataset check, and that rows, features, class balance, clean-data calibration, clean-data discrimination, and baseline noise damage do not explain the Brier change on these eight tables [M0]. That is evidence against treating the WDBC write-up as ready for publication as a general finding. It is not itself a finished robustness theorem: eight numeric tables, isotropic Gaussian noise, three untuned classical models, and no beta or Venn–Abers arm.

No clinical use is intended.

## 14. Hypotheses that need a sample which does not include these eight tables

The patterns below were noticed in this run. They are not findings. A test of any of them has to use tables that were not used to write the sentence.

1. For logistic regression, the Brier difference at isotropic sigma = 1 (clean-trained sigmoid or isotonic minus baseline) is more negative on tables with more features. The intervals in this file excluded zero [M-rho-1, M-rho-2]. The Holm p-values did not. Leave-one-out correlations on these eight points stayed negative. That is the whole of the support.
2. Row count, smaller-class rate, clean-data 10-bin ECE, clean-data ROC-AUC, and the increase in baseline Brier from sigma 0 to sigma 1 do not order the six recalibration contrasts well enough to choose a calibrator. Every one of those bootstrap intervals included both signs in this sample.
3. The sign of the forest sigmoid Brier difference at sigma = 1 is not determined by those six characteristics. After Holm, WDBC remains negative and banknote remains positive [M-holm-06, M-holm-05].
4. A forest sigmoid improvement on one table, including one that survives a Holm adjustment of that table's own Brier and ECE tests, does not travel to the next table. The WDBC-only survivors [M-holm-09–M-holm-13] and the banknote reversal are the motivation, not the confirmation.

Until a later sample addresses these sentences, the supported conclusion is the negative one already stated: these characteristics, on these eight tables, do not explain the Brier change.

## 15. Reproducibility statement

Commands, seeds, and the interpreter that wrote the CSVs are in `REPRODUCIBILITY.md`. Tests that recompute the archive from `preserved/original_source/` and lock the new row counts are in `tests/`. The archived CSV bytes and the uploaded figure are kept as uploaded. The numeric tables can be regenerated from the archived scripts to within 10⁻¹². No clinical use is intended.

## Figure captions

There is no separate results table in this note. The quantitative table is `CLAIM_EVIDENCE_MATRIX.md`. `MANUSCRIPT_DRAFT.md` reprints two short displays of claims that already appear above. Figure files are drawn from the CSVs named here. Bands and error bars are the interval type named in the caption. They are not confidence intervals.

**Figure 1.** `figures/baseline_noise_bands.png`, from `results/recalibration/summary_draws.csv`. Baseline ROC-AUC, sensitivity, Brier score, and 10-bin ECE against isotropic sigma on WDBC. The line is the mean over five splits. The band is the 2.5–97.5 percent draw interval with those splits held fixed.

**Figure 2.** `figures/recalibration_bands.png`, from the same summary. Baseline, sigmoid, and isotonic arms for Brier score and 10-bin ECE. Same draw-interval band.

**Figure 3.** `figures/reliability_diagrams.png`, from `results/recalibration/reliability_bins.csv`. Reliability curves that pool splits and draws. Repeated test rows are not distinct patients. This figure is a display, not a sample size.

**Figure 4.** `figures/cross_dataset_brier.png`, from `results/cross_dataset/paired_calibration.csv`. Pre-specified Brier differences at sigma 0 and sigma 1. A positive value means the recalibrated arm has a higher Brier score. Bars are draw percentiles within each dataset.

**Figure 5.** `figures/brier_effect_by_dataset.png`, from the same paired file, sigma = 1, six contrasts. Eight datasets. Bars are draw percentiles, not the sample size of the Spearman analysis.

**Figure 6.** `figures/spearman_associations.png`, from `results/association/spearman.csv`. Spearman rho for each characteristic against each of the six Brier contrasts. Bars are dataset-bootstrap intervals. An interval that covers zero is not a predictor.

**Figure 7.** `figures/association_scatters.png`, from `results/association/characteristics.csv` joined to the paired Brier differences. Thirty-six panels, eight points each, no fitted line.

**Figure 8.** `figures/corruptions_brier.png`, from `results/corruptions/summary_draws.csv`. Baseline Brier score under the other synthetic corruptions. Bands are draw percentiles.

**Figure 9.** `figures/threshold_costs.png`, from `results/thresholds/summary_draws.csv`. Mean cost per test row after a threshold chosen on training out-of-fold predictions. Bands are draw percentiles. The costs are the multipliers in the config file.

The uploaded archive figure `preserved/figures/uploaded_noise_panel.png` is not a figure from this protocol.

## References

1. scikit-learn developers. Section 1.16, Probability calibration. scikit-learn 1.9.1 User Guide. https://scikit-learn.org/stable/modules/calibration.html (page fetched 5 November 2026).
2. Nadeau C, Bengio Y. Inference for the generalization error. *Machine Learning*. 2003;52:239–281. doi:10.1023/A:1024068626366.
3. Phopase K. Calibration and threshold selection of machine learning classifiers for breast tumour diagnosis: a decision-support study for AI-driven clinical screening. Research Square preprint. Posted 27 October 2026. doi:10.21203/rs.3.rs-11491081/v1. Different protocol; not peer reviewed; not a source of the numbers in this file.
4. Platt JC. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. 1999. Cited from the scikit-learn 1.9.1 user guide, reference [4] of section 1.16. Not re-read in full for this note.
5. Niculescu-Mizil A, Caruana R. Predicting good probabilities with supervised learning. ICML 2005. Cited from the same user-guide section, reference [3]. Not re-read in full for this note.
6. Murphy AH. A new vector partition of the probability score. *Journal of Applied Meteorology*. 1973;12(4):595–600. Cited from the same user-guide section, reference [1], for the Brier-score decomposition. Not re-read in full for this note. The 1973 journal title is used. The later series name, *Journal of Applied Meteorology and Climatology*, is not the title on the 1973 issue.
7. Manokhin V, Grønhaug D. When post-hoc calibration hurts: Platt scaling and isotonic regression on modern tabular models. *Proceedings of Machine Learning Research*. 2026;329:378–381. COPA 2026. https://proceedings.mlr.press/v329/manokhin26a.html. The four-page proceedings version. The i.i.d. restriction is quoted from the longer preprint below, not from this abstract.
8. Manokhin V, Grønhaug D. Classifier calibration at scale: an empirical study of model-agnostic post-hoc methods. arXiv:2601.19944. Submitted 19 January 2026. HTML fetched 5 November 2026. States that distributional shift was omitted.
9. Ovadia Y, Fertig E, Ren J, Nado Z, Sculley D, Nowozin S, Dillon JV, Lakshminarayanan B, Snoek J. Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. NeurIPS 2019. arXiv:1906.02530 HTML fetched 5 November 2026. Deep networks and temperature scaling, not the protocol in Section 7.
10. Virdee GS. Condition-stratified robustness analysis of post-hoc calibration methods for probabilistic classifiers. arXiv:2607.11542. HTML fetched 5 November 2026. Temperature scaling versus isotonic regression across four conditions of one corpus. The page does not identify those conditions with isotropic feature noise, and it states that no external transport is claimed. No number from it is used below.
11. Demographic calibration gaps in breast cancer risk prediction: introducing the Demographic Calibration Gap Score. medRxiv preprint, doi:10.64898/2026.06.17.26355900. Not peer reviewed. Only the abstract was read. It studies subgroup calibration on WDBC and MIMIC-IV. It is not a source for the noise results, and the author list is not copied here because the full text was not used.
