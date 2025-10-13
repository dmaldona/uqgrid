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
