"""Synthetic feature corruption. Every generator is a function of an explicit seed.

Isotropic noise is ``sigma * s_j * Z_ij`` with ``Z_ij`` standard normal and
``s_j`` the training-fold population SD (divisor n), the same SD used by
``StandardScaler``. A single ``Z`` is drawn per (split, level, draw, stream)
and reused for every model and calibration arm.
"""

from __future__ import annotations

import numpy as np


def noise_seed(
    master_seed: int,
    stream: int,
    split_index: int,
    level_index: int,
    draw: int,
) -> int:
    """Deterministic 32-bit seed. The model name is intentionally not an input."""
    seq = np.random.SeedSequence([int(master_seed), int(stream), int(split_index), int(level_index), int(draw)])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def standard_normal(shape: tuple[int, ...], seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(size=shape)


def population_sd(X_train: np.ndarray) -> np.ndarray:
    sd = np.asarray(X_train, dtype=float).std(axis=0, ddof=0)
    sd = sd.copy()
    sd[sd == 0] = 1.0
    return sd


def apply_isotropic(X_test: np.ndarray, train_sd: np.ndarray, sigma: float, Z: np.ndarray) -> np.ndarray:
    if Z.shape != X_test.shape:
        raise ValueError("Noise matrix shape does not match the test matrix.")
    if sigma == 0:
        return np.array(X_test, dtype=float, copy=True)
    return np.asarray(X_test, dtype=float) + (float(sigma) * np.asarray(train_sd, dtype=float)) * Z


def apply_correlated(X_test: np.ndarray, X_train: np.ndarray, sigma: float, Z: np.ndarray) -> np.ndarray:
    """Gaussian noise with covariance ``sigma**2 * Cov_train`` (population covariance).

    Marginal SD of feature j equals ``sigma * s_j`` when the covariance is PSD,
    so the scale is comparable to isotropic noise. A ridge of ``1e-8`` times
    the mean variance is added before the Cholesky factor.
    """
    if sigma == 0:
        return np.array(X_test, dtype=float, copy=True)
    if Z.shape != X_test.shape:
        raise ValueError("Noise matrix shape does not match the test matrix.")
    X_train = np.asarray(X_train, dtype=float)
    centered = X_train - X_train.mean(axis=0)
    n = X_train.shape[0]
    cov = (centered.T @ centered) / n
    jitter = 1e-8 * float(np.mean(np.diag(cov)))
    cov = cov + jitter * np.eye(cov.shape[0])
    chol = np.linalg.cholesky(cov)
    noise = float(sigma) * (Z @ chol.T)
    return np.asarray(X_test, dtype=float) + noise


def apply_multiplicative(X_test: np.ndarray, sigma: float, Z: np.ndarray) -> np.ndarray:
    """``X * (1 + sigma * Z)``. Entries may become negative. That is reported, not clipped."""
    if Z.shape != X_test.shape:
        raise ValueError("Noise matrix shape does not match the test matrix.")
    if sigma == 0:
        return np.array(X_test, dtype=float, copy=True)
    return np.asarray(X_test, dtype=float) * (1.0 + float(sigma) * Z)


def apply_mean_dropout(
    X_test: np.ndarray,
    train_mean: np.ndarray,
    drop_probability: float,
    seed: int,
) -> np.ndarray:
    """Replace each test feature independently with the training mean."""
    if not 0.0 <= drop_probability <= 1.0:
        raise ValueError("drop_probability must lie in [0, 1].")
    rng = np.random.default_rng(seed)
    X_out = np.array(X_test, dtype=float, copy=True)
    if drop_probability == 0:
        return X_out
    mask = rng.random(X_out.shape) < drop_probability
    X_out[mask] = np.asarray(train_mean, dtype=float)[np.where(mask)[1]]
    return X_out


def apply_group_isotropic(
    X_test: np.ndarray,
    train_sd: np.ndarray,
    sigma: float,
    Z: np.ndarray,
    column_mask: np.ndarray,
) -> np.ndarray:
    """Isotropic Gaussian noise restricted to columns where ``column_mask`` is true."""
    if sigma == 0:
        return np.array(X_test, dtype=float, copy=True)
    X_out = np.array(X_test, dtype=float, copy=True)
    idx = np.where(np.asarray(column_mask, dtype=bool))[0]
    X_out[:, idx] = X_out[:, idx] + (float(sigma) * np.asarray(train_sd, dtype=float)[idx]) * Z[:, idx]
    return X_out
