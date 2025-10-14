from __future__ import annotations

from typing import Callable, Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

Array = np.ndarray
Mat = Union[np.ndarray, sparse.spmatrix, LinearOperator, list, tuple]


def as_linear_op_Sigma_inv(Sigma_inv: Mat, m: int) -> LinearOperator:
    """Wrap Sigma^{-1} as a scipy.sparse.linalg.LinearOperator of shape (m, m).

    Accepts:
      - 1D array of length m (interpreted as diagonal),
      - dense (m x m) ndarray,
      - sparse CSR/CSC matrix,
      - LinearOperator.
    """

    if isinstance(Sigma_inv, LinearOperator):
        return Sigma_inv

    if isinstance(Sigma_inv, (np.ndarray, list, tuple)) and np.ndim(Sigma_inv) == 1:
        d = np.asarray(Sigma_inv, dtype=float).ravel()

        def mv(v: Array) -> Array:
            return d * v

        def rmv(v: Array) -> Array:
            return d * v

        return LinearOperator((m, m), matvec=mv, rmatvec=rmv)

    if sparse.issparse(Sigma_inv):
        Sigma_inv_csr = Sigma_inv.tocsr()
        return LinearOperator(
            (m, m),
            matvec=lambda v: Sigma_inv_csr @ v,
            rmatvec=lambda v: Sigma_inv_csr.transpose() @ v,
        )

    Sigma_inv_arr = np.asarray(Sigma_inv, dtype=float)
    return LinearOperator(
        (m, m),
        matvec=lambda v: Sigma_inv_arr @ v,
        rmatvec=lambda v: Sigma_inv_arr.T @ v,
    )


def grad_I(lambda_vec: Array, mu: Array, Sigma_inv_op: LinearOperator) -> Array:
    """Return ∇_λ I = Σ^{-1} (λ - μ)."""

    return Sigma_inv_op @ (lambda_vec - mu)


def rate_I(lambda_vec: Array, mu: Array, Sigma_inv_op: LinearOperator) -> float:
    """Compute I(λ) = 1/2 (λ - μ)^T Σ^{-1} (λ - μ)."""

    delta = lambda_vec - mu
    return 0.5 * float(delta @ (Sigma_inv_op @ delta))


def first_order_ldt_prob(beta: float) -> float:
    """Evaluate first-order Gaussian LDT probability approximation.

    P_1st ≈ (1 / (sqrt(2π) * β)) * exp(-β^2 / 2). Guard β <= 0.
    """

    if beta <= 0:
        return np.inf
    return (1.0 / (np.sqrt(2.0 * np.pi) * beta)) * np.exp(-0.5 * beta * beta)


def oriented_unit_normal(normal: Array, reference: Array, *, tol: float = 1e-12) -> Array:
    """Return a unit-length normal oriented to align with a reference vector.

    Parameters
    ----------
    normal:
        Candidate normal vector (will be normalized).
    reference:
        Vector providing orientation; resulting unit normal will satisfy
        ``unit_normal · reference >= 0`` within the tolerance.
    tol:
        Absolute tolerance used for guarding degenerate vectors.
    """

    normal_arr = np.asarray(normal, dtype=float).ravel()
    if normal_arr.size == 0:
        raise ValueError("Normal vector must not be empty.")
    if not np.isfinite(normal_arr).all():
        raise ValueError("Normal vector must contain only finite values.")

    norm = np.linalg.norm(normal_arr)
    if norm <= tol:
        raise ValueError("Normal vector norm is too small to normalize.")

    unit = normal_arr / norm

    reference_arr = np.asarray(reference, dtype=float).ravel()
    if reference_arr.shape[0] != unit.shape[0]:
        raise ValueError("Reference vector must have the same dimension as normal.")
    if not np.isfinite(reference_arr).all():
        raise ValueError("Reference vector must contain only finite values.")

    ref_norm = np.linalg.norm(reference_arr)
    if ref_norm <= tol:
        raise ValueError("Reference vector norm is too small to determine orientation.")

    dot = float(unit @ reference_arr)
    if np.isnan(dot):
        raise ValueError("Dot product with reference vector is not finite.")
    if dot < 0.0:
        unit = -unit

    return unit
