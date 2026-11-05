"""Repeated stratified holdouts. Indices are saved so later code does not depend on re-deriving them."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def repeated_stratified_holdout(
    y: np.ndarray,
    n_splits: int,
    test_size: float,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )
    X_dummy = np.zeros((len(y), 1))
    return [(train, test) for train, test in splitter.split(X_dummy, y)]
