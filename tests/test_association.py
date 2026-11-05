"""Holm adjustment and the eight-dataset association summary."""

import numpy as np
import pandas as pd

from wdbc_calib.stats import holm_adjust


def test_holm_matches_the_worked_example():
    adjusted = holm_adjust(np.array([0.01, 0.04, 0.03, 0.20]))
    assert np.allclose(adjusted, [0.04, 0.09, 0.09, 0.20])


def test_holm_never_makes_a_p_value_smaller_and_caps_at_one():
    raw = np.array([0.001, 0.02, 0.04, 0.2, 0.9])
    adjusted = holm_adjust(raw)
    assert np.all(adjusted + 1e-15 >= raw)
    assert np.all(adjusted <= 1)
    order = np.argsort(raw)
    assert np.all(np.diff(adjusted[order]) >= -1e-15)


def test_saved_association_uses_eight_datasets_and_holm():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spearman = pd.read_csv(root / "results/association/spearman.csv")
    counts = pd.read_csv(root / "results/association/holm_counts.csv")
    holm = pd.read_csv(root / "results/association/holm_split_tests.csv")
    assert set(spearman.n_datasets) == {8}
    assert len(spearman) == 36
    assert bool(spearman.draws_used_as_sample_size.any()) is False
    assert np.all(spearman.holm_p + 1e-12 >= spearman.permutation_p)
    cross = holm[holm.family == "cross_dataset_64"]
    wdbc = holm[holm.family == "wdbc_brier_ece_48"]
    assert len(cross) == 64
    assert len(wdbc) == 48
    assert int(counts.set_index("family").loc["cross_dataset_64", "n_raw_p_below_0.05"]) == int((cross.two_sided_p < 0.05).sum())
    assert int((cross.holm_p < 0.05).sum()) <= int((cross.two_sided_p < 0.05).sum())
    assert int((wdbc.holm_p < 0.05).sum()) <= int((wdbc.two_sided_p < 0.05).sum())
