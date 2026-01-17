from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from scipy.sparse import csc_matrix, spmatrix
from scipy.sparse.linalg import factorized, spsolve

SparseSolverName = Literal["scipy", "klu"]
SUPPORTED_SPARSE_SOLVERS = ("scipy", "klu")

try:
    from PyKLU import Klu
    _KLU_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    Klu = None
    _KLU_AVAILABLE = False


def klu_available() -> bool:
    return _KLU_AVAILABLE


def validate_sparse_solver(solver: str) -> str:
    if solver not in SUPPORTED_SPARSE_SOLVERS:
        raise ValueError(
            f"Unknown sparse solver '{solver}'. "
            f"Expected one of {', '.join(SUPPORTED_SPARSE_SOLVERS)}."
        )
    return solver


def _require_klu() -> None:
    if not _KLU_AVAILABLE:
        raise ImportError(
            "PyKLU is required for the KLU solver. Install via `pip install -e \".[klu]\"`."
        )


def _ensure_csc(matrix: spmatrix) -> csc_matrix:
    if isinstance(matrix, csc_matrix):
        return matrix
    return matrix.tocsc(copy=True)


def solve_sparse_system(
    matrix: spmatrix,
    rhs: np.ndarray,
    solver: SparseSolverName,
) -> np.ndarray:
    solver = validate_sparse_solver(solver)
    if solver == "klu":
        _require_klu()
        return Klu(_ensure_csc(matrix)).solve(rhs)
    return spsolve(matrix, rhs)


def factorize_sparse_system(
    matrix: spmatrix,
    solver: SparseSolverName,
) -> Callable[[np.ndarray], np.ndarray]:
    solver = validate_sparse_solver(solver)
    if solver == "klu":
        _require_klu()
        return Klu(_ensure_csc(matrix)).solve
    return factorized(_ensure_csc(matrix))
