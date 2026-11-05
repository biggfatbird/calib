# Limitations

This is a methods study on one public dataset. It is not a clinical validation and not a medical device.

## Data

- The WDBC sections use one cohort: 569 rows, 30 numeric features, 212 malignant and 357 benign. There is no external hospital, no assay shift, and no prospective sample.
- Section 7 of `paper_v2.md` repeats the isotropic-noise protocol on seven further public numeric tables. The WDBC signs do not all travel. Those seven tables are still a convenience sample of OpenML binary problems with numeric columns, not a survey of covariate shift.
- Repeated 75/25 splits reuse the same people. A draw interval holds those splits fixed. It describes the noise generator, not a new patient population.
- Five splits are too few for a stable test of a mean difference. The corrected resampled t-test uses 4 degrees of freedom. Thirty splits were added as a description of dispersion, not as a second confirmatory sample.

## Noise

- Isotropic Gaussian noise, correlated Gaussian noise, multiplicative noise, training-mean dropout, and noise on the ten "worst" features are synthetic. None of them is a measured scanner or staining artifact.
- Sigma is a fraction of the training-fold population standard deviation. A value of 1 is a large perturbation relative to these features, not a calibrated clinical dose.
- Correlated noise uses the training covariance plus a small ridge before the Cholesky factor. Multiplicative noise is allowed to make features negative. Those choices are declared in `configs/corruptions.json` and in each `manifest.json`.

## Models and calibration

- Hyperparameters are scikit-learn defaults (logistic `C=1`, forest of 100 trees, RBF SVM with `gamma="scale"` and `C=1`). They were not tuned. A tuned model could change the calibration gaps.
- The baseline SVM uses `SVC(probability=True)`, which already Platt-scales out-of-fold decision values. The sigmoid arm calibrates an SVM with `probability=False`. A near-tie is expected. It is not evidence about raw margins.
- `CalibratedClassifierCV(..., ensemble=False)` fits the calibrator on out-of-fold training predictions and refits the base model on the whole training fold. That avoids test labels. It does not use a separate calibration set held out from training.
- Isotonic regression creates ties, so ROC-AUC can fall even when the score was only remapped. A lower AUC after isotonic calibration is a property of that map, not by itself a claim that discrimination of the underlying score got worse in every ranking sense.
- Threshold 0.5 is an arbitrary operating point. Sensitivity changes after isotonic calibration can be operating-point shifts. The cost experiment is the place those shifts are measured, and it still uses synthetic costs, not estimated clinical utilities.

## Statistics

- Forty-eight Brier and ECE contrasts were computed (two calibrators, three models, four noise levels, two metrics). Thirteen had an unadjusted split-level p-value below 0.05. Those p-values are exploratory. They are not a family-wise or false-discovery claim.
- A draw-percentile interval that excludes zero means the sign was stable across noise seeds on these splits. It can coexist with a split-level test that does not separate the same contrast, because the two calculations use different units. Neither one is equivalence, and neither one is a patient-level confidence interval.
- An interval that crosses zero is not a demonstration that two methods are equivalent.

## Reproducibility of the archived numbers

- The original scripts are now in `preserved/original_source/` and regenerate the archived numeric tables. The explicit protocol is a different seed and a different forest, so it does not recover the archived ECE or sensitivity. See `PROTOCOL_COMPARISON.md` and claims A1–A5. Archived tables stay in `preserved/` and are not rewritten.
- The original noise seed and model `random_state` are labelled as assumptions in `reproduce_study.py`. Matching the CSV confirms the script, not an external paper's unstated seed.

## What was not run

- No external dataset, no label noise, no adversarial corruption, no temperature scaling, no hyperparameter search, and no patient-level bootstrap that refits the models.
- The 30-split extension uses 20 noise draws, not 50. That count is in `configs/more_splits.json`. It was chosen before looking at the outcome and is not a silent deletion of a failed run.
