# Task 1 result: which paper claims survive 50 noise draws?

Setup: same splits/models as the paper; 50 independent Gaussian noise matrices per split and sigma, shared by the three models within a draw; metrics averaged over the 5 splits per draw; interval = 2.5-97.5 percentile across draws (noise-seed uncertainty only; split-to-split SD is in `noise_draw_summary.csv`). EXTENSION to the paper, not part of the original study.

Paired differences at sigma = 1.0 (a minus b): mean [95% interval across draws], share of draws with a > b.

| Claim (a vs b) | a - b | a > b in | Verdict |
|---|---|---|---|
| SVM ROC-AUC vs logistic regression (paper: 0.973 vs 0.967) | -0.000 [-0.006, +0.005] | 46% | NOT supported: difference is zero within noise |
| SVM Brier vs logistic regression (higher = worse) | +0.013 [+0.007, +0.020] | 100% | Supported: SVM worse in every draw |
| SVM ECE vs logistic regression (higher = worse) | +0.052 [+0.030, +0.067] | 100% | Supported: SVM worse in every draw |
| SVM sensitivity vs logistic regression (paper: 0.932 vs 0.887) | +0.061 [+0.042, +0.082] | 100% | Supported: SVM higher in every draw, ~0.06 |
| SVM Brier vs random forest | -0.022 [-0.031, -0.015] | 0% | Supported: forest has the worst Brier in every draw |
| SVM ECE vs random forest | -0.044 [-0.059, -0.030] | 0% | Supported: forest has the worst ECE in every draw |
| SVM F1 vs logistic regression (paper: 0.849 vs 0.878) | -0.014 [-0.035, +0.007] | 12% | Direction holds in 88% of draws; interval includes 0, so weak |

Caveats: 5 splits of one 569-row dataset; intervals describe noise-draw variability, not generalisation to other data. sigma=0 results do not depend on the draw (tested).
Figure: `figures/noise_draws_ci.png`. Data: `results/noise_draws.csv`, `noise_draw_summary.csv`, `paired_differences.csv`.
