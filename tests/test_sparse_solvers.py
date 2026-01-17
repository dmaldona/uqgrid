import numpy as np
import pytest
from scipy.sparse import csc_matrix

from uqgrid.simulation.sparse_solvers import (
    factorize_sparse_system,
    klu_available,
    solve_sparse_system,
)


@pytest.mark.skipif(not klu_available(), reason="PyKLU not installed")
def test_klu_solver_roundtrip():
    matrix = csc_matrix([[3.0, 1.0], [1.0, 2.0]])
    rhs = np.array([9.0, 8.0])

    sol = solve_sparse_system(matrix, rhs, "klu")
    assert np.allclose(matrix.dot(sol), rhs)

    factor = factorize_sparse_system(matrix, "klu")
    sol_factor = factor(rhs)
    assert np.allclose(matrix.dot(sol_factor), rhs)
