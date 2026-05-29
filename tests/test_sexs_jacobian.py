import numpy as np
from scipy.sparse import csr_matrix

from uqgrid.models import ExcSEXS


def _sexs_jacobian_fixture():
    exc = ExcSEXS("1", TA_TB=0.1, TB=10.0, K=100.0, TE=0.05, Emin=-1.0, Emax=1.0)
    exc.set_bus(0)
    exc.vref = 2.0

    jac_idxs = np.array([0, 0, 2], dtype=np.int64)
    coords = exc.preallocate_jacobian(jac_idxs, None, power_injection=False)
    rows = []
    cols = []
    for row, row_cols in coords:
        rows.extend([row] * len(row_cols))
        cols.extend(row_cols)

    data = np.zeros(len(rows), dtype=np.float64)
    J = csr_matrix((data, (rows, cols)), shape=(4, 4), dtype=np.float64)
    theta = np.array(
        [
            exc.TA_TB,
            exc.TB,
            exc.K,
            exc.TE,
            exc.Emin,
            exc.Emax,
            float(exc.enable_limits),
            exc.vref,
        ],
        dtype=np.float64,
    )
    residual_idxs = np.array([0, 0, 0], dtype=np.int64)
    return exc, residual_idxs, jac_idxs, theta, J


def test_sexs_ignores_output_limits_in_residual():
    exc, residual_idxs, _, theta, _ = _sexs_jacobian_fixture()
    z = np.array([1.0, 1.1], dtype=np.float64)
    v = np.array([1.0, 0.0], dtype=np.float64)
    F = np.zeros(2, dtype=np.float64)

    exc.residual_diff(F, z, v, theta, residual_idxs, False)

    y1 = z[0] + exc.TA_TB * (exc.vref - v[0])
    expected_dedt = (-z[1] + exc.K * y1) / exc.TE
    assert z[1] > exc.Emax
    assert expected_dedt > 0.0
    assert F[1] == expected_dedt


def test_sexs_efd_row_matches_unconstrained_finite_difference_outside_limits():
    exc, residual_idxs, jac_idxs, theta, J = _sexs_jacobian_fixture()
    z = np.array([1.0, 1.1], dtype=np.float64)
    v = np.array([1.0, 0.0], dtype=np.float64)

    exc.residual_jac(J, z, v, theta, jac_idxs, power_injection=False)
    analytical = J.toarray()[1, [0, 1, 2, 3]]

    eps = 1e-6
    finite_difference = []
    for col in [0, 1, 2, 3]:
        z_p = z.copy()
        z_m = z.copy()
        v_p = v.copy()
        v_m = v.copy()
        if col < 2:
            z_p[col] += eps
            z_m[col] -= eps
        else:
            v_p[col - 2] += eps
            v_m[col - 2] -= eps

        f_p = np.zeros(2, dtype=np.float64)
        f_m = np.zeros(2, dtype=np.float64)
        exc.residual_diff(f_p, z_p, v_p, theta, residual_idxs, False)
        exc.residual_diff(f_m, z_m, v_m, theta, residual_idxs, False)
        finite_difference.append((f_p[1] - f_m[1]) / (2.0 * eps))

    assert z[1] > exc.Emax
    np.testing.assert_allclose(analytical, finite_difference, rtol=1e-8, atol=1e-8)


def test_sexs_limits_are_stored_but_disabled_by_default():
    exc, _, _, theta, _ = _sexs_jacobian_fixture()

    assert exc.Emin == -1.0
    assert exc.Emax == 1.0
    assert exc.enable_limits is False
    assert theta[6] == 0.0
