from __future__ import annotations

from typing import Optional

import numpy as np


def ellipse_collapse_mask(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Collapse indicator for the 2-bus analytic boundary: P^2 + 4Q - 4 >= 0."""

    return (P ** 2 + 4.0 * Q - 4.0) >= 0.0


def sample_gaussian(mu: np.ndarray, Sigma: np.ndarray, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Draw samples from N(mu, Sigma).

    Sigma may be a diagonal vector or full covariance matrix.
    """

    rng = np.random.default_rng() if rng is None else rng
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    m = mu.size

    if Sigma.ndim == 1:
        if Sigma.shape[0] != m:
            raise ValueError("Diagonal Sigma must match dimension of mu.")
        Z = rng.standard_normal(size=(n, m))
        return mu + Z * np.sqrt(Sigma)

    if Sigma.shape != (m, m):
        raise ValueError("Sigma must be diagonal vector or full (m x m) covariance matrix.")

    L = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal(size=(n, m))
    return mu + Z @ L.T
