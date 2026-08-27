import numpy as np
import pytest
from scipy.sparse import csr_matrix

from uqgrid.models.gast_imp import GovGAST


def _model(**overrides):
    values = {
        "R": 0.05, "T1": 0.4, "T2": 0.1, "T3": 3.0, "AT": 1.0,
        "KT": 2.0, "VMAX": 1.2, "VMIN": 0.0, "DT": 0.15,
    }
    values.update(overrides)
    gov = GovGAST("1", **values)
    gov.set_pointers(2, 0, 0, 0)
    gov.w_idx = 0
    return gov


def _theta(gov):
    theta = np.zeros(gov.par_dim)
    gov.initialize_theta(theta)
    return theta


def _residual(gov, z, theta):
    F = np.zeros_like(z)
    idxs = np.array([2, 5, 0, 0], dtype=np.int64)
    gov.residual_diff(F, z, np.empty(0), theta, idxs, False)
    return F


def _jacobian(gov, z, theta):
    idxs = np.array([2, 5, 0, 0], dtype=np.int64)
    coords = gov.preallocate_jacobian(idxs, None, False)
    rows = []
    cols = []
    for row, row_cols in coords:
        rows.extend([row] * len(row_cols))
        cols.extend(row_cols)
    J = csr_matrix(
        (np.zeros(len(rows)), (rows, cols)), shape=(len(z), len(z)), dtype=float
    )
    gov.residual_jac(J, z, np.empty(0), theta, idxs, False)
    return J


def test_gast_dimensions_initialization_and_zero_residual():
    gov = _model()
    gov.p_m0 = 0.6
    x = np.zeros(5)
    y = np.zeros(1)

    gov.initialize(1.0, 0.0, 0.0, 0.0, x, y, None)
    theta = _theta(gov)
    z = np.concatenate((x, y))

    assert gov.getdim() == (3, 1, 11)
    np.testing.assert_allclose(x[2:5], 0.6)
    assert y[0] == 0.6
    assert gov.pref == pytest.approx(0.03)
    np.testing.assert_allclose(_residual(gov, z, theta), 0.0, atol=1e-14)


def test_gast_initialization_accepts_temperature_branch_and_rejects_nonstationary():
    temperature_limited = _model(AT=0.4, KT=1.0)
    temperature_limited.p_m0 = 0.4
    temperature_limited.initialize(1.0, 0.0, 0.0, 0.0, np.zeros(5), np.zeros(1), None)

    invalid = _model(AT=0.3, KT=1.0)
    invalid.p_m0 = 0.4
    with pytest.raises(ValueError, match="no stationary selector branch"):
        invalid.initialize(1.0, 0.0, 0.0, 0.0, np.zeros(5), np.zeros(1), None)


@pytest.mark.parametrize(
    "x3,w,pref,expected_u,expected_x1_derivatives",
    [
        (0.4, 0.01, 0.03, 0.4, (-1.0 / 0.4, 0.0, -1.0 / (0.05 * 0.4))),
        (1.3, 0.0, 0.03, 0.4, (-1.0 / 0.4, -2.0 / 0.4, 0.0)),
        (1.25, 0.0, 0.025, 0.5, (-1.0 / 0.4, -2.0 / 0.4, 0.0)),
    ],
    ids=["demand", "temperature", "equality-uses-temperature"],
)
def test_gast_selector_branches_and_structural_zeros(
    x3, w, pref, expected_u, expected_x1_derivatives
):
    gov = _model()
    gov.pref = pref
    theta = _theta(gov)
    z = np.array([w, 9.0, 0.5, 0.45, x3, 0.42])

    J = _jacobian(gov, z, theta)
    row = J.toarray()[2]

    assert _residual(gov, z, theta)[2] == pytest.approx((expected_u - z[2]) / gov.T1)
    np.testing.assert_allclose(row[[2, 4, 0]], expected_x1_derivatives)
    assert set(J.indices[J.indptr[2]:J.indptr[3]]) == {0, 2, 4}


@pytest.mark.parametrize("x3,w", [(0.4, 0.01), (1.3, 0.0)])
def test_gast_jacobian_matches_finite_difference_away_from_switch(x3, w):
    gov = _model()
    gov.pref = 0.03
    theta = _theta(gov)
    z = np.array([w, 9.0, 0.5, 0.45, x3, 0.42])
    analytical = _jacobian(gov, z, theta).toarray()

    finite_difference = np.zeros_like(analytical)
    eps = 1e-7
    for col in range(len(z)):
        z_p = z.copy()
        z_m = z.copy()
        z_p[col] += eps
        z_m[col] -= eps
        finite_difference[:, col] = (
            _residual(gov, z_p, theta) - _residual(gov, z_m, theta)
        ) / (2.0 * eps)

    np.testing.assert_allclose(analytical, finite_difference, rtol=1e-8, atol=1e-8)


def test_gast_bounded_state_metadata_and_hessians():
    metadata = GovGAST.bounded_state_metadata

    assert len(metadata) == 1
    assert metadata[0].state_name == "x1"
    assert metadata[0].state_offset == 0
    assert metadata[0].lower_parameter_offset == 7
    assert metadata[0].upper_parameter_offset == 6
    assert metadata[0].enabled_parameter_offset == 9
    assert metadata[0].device_type == "GAST"

    gov = _model()
    with pytest.raises(NotImplementedError):
        gov.preallocate_hessian(0, np.array([0, 0, 0]), None)
    with pytest.raises(NotImplementedError):
        gov.residual_hess(None, None, None, None, None)


def test_gast_runtime_uses_theta_parameters():
    gov = _model()
    gov.pref = 0.03
    theta = _theta(gov)
    z = np.array([0.0, 9.0, 0.5, 0.45, 0.4, 0.42])
    baseline = _residual(gov, z, theta)

    theta[10] += 0.01
    changed = _residual(gov, z, theta)

    assert changed[2] != baseline[2]


@pytest.mark.parametrize(
    "overrides,message",
    [({"T1": 0.0}, "time constants must be positive"), ({"VMIN": 1.2}, "VMIN")],
)
def test_gast_rejects_invalid_parameters(overrides, message):
    with pytest.raises(ValueError, match=message):
        _model(**overrides)
