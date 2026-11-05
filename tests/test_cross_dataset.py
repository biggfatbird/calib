"""The cross-dataset runner stays on the training fold and off the archive."""

from pathlib import Path

import numpy as np
import pandas as pd

from wdbc_calib.cross_dataset import run_cross_dataset_experiment
from wdbc_calib.tabular_datasets import BinaryTable, encode_binary


def test_encode_binary_rejects_an_unexpected_label():
    labels = pd.Series(["a", "b", "a"])
    y = encode_binary(labels, "b", {"a", "b"})
    assert list(y) == [0, 1, 0]
    try:
        encode_binary(labels, "b", {"a", "c"})
    except RuntimeError:
        return
    raise AssertionError("unexpected labels were accepted")


def test_cross_dataset_smoke_does_not_write_preserved(tmp_path: Path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4))
    y = np.array([0, 1] * 40, dtype=np.int8)
    table = BinaryTable(
        key="synthetic",
        title="synthetic",
        source="unit test",
        positive_class="1",
        X=X,
        y=y,
    )
    config = {
        "master_seed": 20260311,
        "n_splits": 2,
        "test_size": 0.25,
        "sigmas": [0.0, 1.0],
        "n_draws": 2,
        "models": ["logistic_regression"],
        "calibrations": ["baseline", "sigmoid"],
        "output_dir": str(tmp_path / "out"),
    }
    frame = run_cross_dataset_experiment(config, tmp_path / "out", tables=[table])
    assert len(frame) == 2 * 2 * 2 * 1 * 2
    assert "preserved" not in str(tmp_path / "out")
    clean = frame[frame.sigma == 0]
    assert int(clean.groupby(["split", "calibration"])["brier"].nunique().max()) == 1
