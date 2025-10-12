from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


def _smallest_singular_pair_dense(jac: csr_matrix) -> tuple[float, np.ndarray, np.ndarray]:
    u, s, vt = np.linalg.svd(jac.toarray(), full_matrices=False)
    idx = int(np.argmin(s))
    return float(s[idx]), u[:, idx], vt[idx, :].conj()


def _smallest_singular_pair_sparse(jac: csr_matrix) -> tuple[float, np.ndarray, np.ndarray]:
    u, s, vt = svds(jac, k=1, which="SM")
    # svds returns column vectors in descending order of |s|
    sigma = float(s[0])
    left = u[:, 0]
    right = vt[0, :].conj()
    return sigma, left, right


def smallest_singular_pair(jac: csr_matrix) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the smallest singular value and its left/right singular vectors."""

    m, n = jac.shape
    if m == 0 or n == 0:
        raise ValueError("Jacobian has zero dimension.")

    try:
        if min(m, n) <= 3:
            return _smallest_singular_pair_dense(jac)
        return _smallest_singular_pair_sparse(jac)
    except Exception:
        return _smallest_singular_pair_dense(jac)


def smallest_left_singular_vector(jac: csr_matrix) -> tuple[float, np.ndarray]:
    """Return smallest singular value and corresponding left singular vector."""

    sigma, left, _ = smallest_singular_pair(jac)
    return sigma, left


def normalize_left_vector(w: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Scale `w` so that w^T c = 1."""

    dot = float(w @ c)
    if abs(dot) < 1e-12:
        raise ValueError("Normalization vector is nearly orthogonal to the null vector.")
    return w / dot


def normalize_singular_pair(w: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale (w, v) so that w^T v = 1."""

    scale = float(w @ v)
    if abs(scale) < 1e-12:
        raise ValueError("Left/right singular vectors are nearly orthogonal; cannot normalize.")
    return w / scale, v
