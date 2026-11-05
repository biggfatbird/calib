# Project rules for AI coding agents (Cursor reads this file)

Project: reproducible calibration + noise-robustness study on the Wisconsin Diagnostic Breast Cancer dataset
(scikit-learn `load_breast_cancer`). Author: Manisha Kanwar. Educational study; NOT a clinical tool, NOT peer reviewed.

## Non-negotiable rules
1. **Never invent results.** Every number in a table, figure, README or paper text must be produced by code in this repo
   and traceable to a CSV in `results/`. If you did not run it, do not state it.
2. **No test-set tuning.** Hyper-parameters and thresholds are chosen on training data only (nested CV / inner split).
3. **Report everything you ran**, including runs that did not help. Do not cherry-pick seeds.
4. **Do not change the original protocol** in `reproduce_study.py` (splits, models, sigma grid). New work goes in new files
   or behind explicit flags so the baseline stays reproducible. Keep `tests/` green (`python -m pytest -q tests`).
5. No medical claims. Keep the wording "not a clinical validation". Cite sources already in the paper; do not add
   references you cannot verify.
6. Fix random seeds, print library versions into every results file, write outputs to `results/` and `figures/`.

## Commands
- Baseline: `python reproduce_study.py`
- Tests: `python -m pytest -q tests`
