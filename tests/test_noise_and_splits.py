"""Seeds, shared noise, and split identity."""

import numpy as np

from wdbc_calib.data import load_wdbc
from wdbc_calib.noise import apply_isotropic, noise_seed, population_sd, standard_normal
from wdbc_calib.splits import repeated_stratified_holdout


def test_noise_seed_ignores_nothing_about_models_and_is_stable():
    a = noise_seed(20260311, 0, 1, 2, 3)
    b = noise_seed(20260311, 0, 1, 2, 3)
    c = noise_seed(20260311, 0, 1, 2, 4)
    assert a == b
    assert a != c
    Z1 = standard_normal((8, 3), a)
    Z2 = standard_normal((8, 3), a)
    assert np.array_equal(Z1, Z2)


def test_sigma_zero_returns_the_clean_matrix_even_if_Z_is_huge():
    X = np.arange(12, dtype=float).reshape(4, 3)
    Z = np.ones_like(X) * 50
    out = apply_isotropic(X, population_sd(X), sigma=0, Z=Z)
    assert np.array_equal(out, X)
    noisy = apply_isotropic(X, np.array([1.0, 2.0, 4.0]), sigma=0.5, Z=Z)
    assert np.allclose(noisy[:, 0], X[:, 0] + 0.5 * 1 * 50)


def test_longer_splitter_extends_the_short_one():
    _X, y, _names = load_wdbc()
    short = repeated_stratified_holdout(y, n_splits=5, test_size=0.25, random_state=20260311)
    long = repeated_stratified_holdout(y, n_splits=30, test_size=0.25, random_state=20260311)
    for (tr_s, te_s), (tr_l, te_l) in zip(short, long):
        assert np.array_equal(tr_s, tr_l)
        assert np.array_equal(te_s, te_l)
    assert len(te_s) == 143
    assert int(y[te_s].sum()) == 53
