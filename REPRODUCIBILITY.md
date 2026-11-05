# Reproducibility

## What can be regenerated

From a clean environment with the pins in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python -m wdbc_calib recalibration
python -m wdbc_calib compare-preserved
python -m wdbc_calib more-splits
python -m wdbc_calib corruptions
python -m wdbc_calib thresholds
python -m wdbc_calib cross-dataset
python -m wdbc_calib figures
python -m wdbc_calib.write_claims
```

The same commands are `make test`, `make recalibrate`, `make compare`, `make more-splits`, `make corruptions`, `make thresholds`, and `make figures`.

Seeds, split counts, and draw counts are in `configs/`. Each output directory gets `manifest.json` (library versions and the config), `RUN_LOG.txt`, and `split_indices.csv` where splits are used. Noise seeds are `numpy.random.SeedSequence([master, stream, split, level, draw])` reduced to one `uint32`, then `Generator.standard_normal`. The model name is not part of the seed, so every model and calibration arm in a draw sees the same matrix.

Primary master seed: `20260311`.

Positive class: malignant = 1. scikit-learn's bundled coding is the opposite (`target == 0` is malignant). `wdbc_calib.data.load_wdbc` flips it and checks that 212 rows are positive.

## Environment that produced `results/` and `figures/`

Recorded in `results/recalibration/manifest.json` on 5 November 2026, 20:56 IST (15:26 UTC):

- Python 3.12.3
- NumPy 2.2.6
- scikit-learn 1.9.1
- pandas 2.2.3

The full command sequence in this workspace started at 20:55:29 IST and finished at 20:58:18 IST (15:25:29Z to 15:28:18Z), including the 30-split run, the corruption grid, and the threshold run. That timing is machine-specific.

The archived noise file `preserved/results/noise_draws_env.txt` says Python 3.13.18, scikit-learn 1.9.1, NumPy 2.5.3. A rerun on 5 November 2026 at 22:04 IST used Python 3.12.3, scikit-learn 1.9.1, and NumPy 2.2.6. Parsed numeric columns still matched to a maximum absolute difference of 2.1×10⁻¹⁶. The CSV text was not byte-identical. `preserved/` was not overwritten.

## Original scripts

```bash
python preserved/original_source/reproduce_study.py --out /tmp/wdbc_original
python preserved/original_source/noise_draws.py --draws 50 --out /tmp/wdbc_original
```

Run them from `preserved/original_source/` so `noise_draws.py` can import `reproduce_study`. Do not point `--out` at `preserved/` or at `results/`. The original seed is 2026. The `wdbc_calib` master seed is 20260311. `tests/test_original_source.py` runs this check into a temporary directory.

## What stays an archive

- `preserved/` is a byte-level archive (`preserved/SHA256SUMS`). Tests fail if those bytes change. Numerical agreement is not an invitation to replace the CSVs with a newly formatted write.
- `preserved/figures/uploaded_noise_panel.png` is the uploaded figure (2000×480). A fresh run of `make_figures_task1.py` here writes 2400×576. `figures/*.png` at the repository root are plots from the explicit protocol.

## Invariants the tests lock

- Training-fold `StandardScaler` means do not move when the test fold is poisoned.
- Probabilities used for malignant metrics are column `classes_ == 1`.
- Sigma 0 predictions do not depend on the draw.
- The threshold function's signature is `y_oof, p_oof, cost_fn, cost_fp`.
- Archived paired differences, draw percentiles, and split SDs recompute from `preserved/results/noise_draws.csv`.
- `tests/test_original_source.py` reruns the archived scripts and requires the archived numeric columns to agree within 10⁻¹².
- The isotropic corruption arm matches the primary baseline Brier means exactly (same seed stream).
- Committed result shapes: 9000 recalibration rows, 7200 more-split rows, 14250 corruption rows, 9000 threshold rows, 0 corruption failures, 72000 cross-dataset rows.
- WDBC rows inside `results/cross_dataset/metrics_long.csv` match `results/recalibration/metrics_long.csv` within 10⁻¹². The cross-dataset file also records that the WDBC sign pattern is not shared by all eight tables.

## SVC warning

scikit-learn 1.9 warns that `SVC(probability=...)` is deprecated and points users to `CalibratedClassifierCV(SVC(), ensemble=False)`. The baseline arm keeps `probability=True` on purpose and records the warning in `results/recalibration/warnings.txt`. The sigmoid and isotonic SVM arms set `probability=False` and still emit the same warning because the constructor accepts the `probability` argument. They are not a second Platt scaling of an already calibrated SVM.

## Continuous integration

`.github/workflows/tests.yml` installs `requirements.txt` and runs `pytest` on Python 3.11, 3.12, and 3.13. This workspace executed the tests on Python 3.12.3 only. The other two interpreters were not run here. If a wheel pin fails on 3.11 or 3.13, that CI job is the place it will show; it is not claimed to be green from this machine.
