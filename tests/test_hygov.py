import numpy as np
import pytest
from scipy.sparse import csr_matrix

from uqgrid.models.hygov_imp import GovHYGOV


def _governor(**overrides):
    values = {
        "R": 0.05,
        "r": 0.4,
        "Tr": 5.0,
        "Tf": 0.2,
        "Tg": 0.5,
        "VELM": 0.3,
        "GMAX": 1.2,
        "GMIN": 0.0,
        "Tw": 1.5,
        "At": 1.1,
        "DT": 0.08,
        "qNL": 0.1,
        "g_floor": 1e-4,
        "enable_limits": True,
        "adjust_initial_limits": False,
    }
    values.update(overrides)
    gov = GovHYGOV("1", **values)
    gov.set_pointers(0, 0, 0, 0)
    gov.w_idx = 0
    gov.pref = 0.4
    return gov


def _theta(gov):
    theta = np.zeros(gov.par_dim)
    gov.initialize_theta(theta)
    return theta


def _residual(gov, z):
    F = np.zeros_like(z)
    gov.residual_diff(F, z, np.empty(0), _theta(gov), np.array([1, 5, 0]), False)
    return F


def _jacobian(gov, z):
    idxs = np.array([1, 5, 0, 0], dtype=np.int64)
    coords = gov.preallocate_jacobian(idxs, None, False)
    rows = []
    cols = []
    for row, row_cols in coords:
        rows.extend([row] * len(row_cols))
        cols.extend(row_cols)
    J = csr_matrix((np.zeros(len(rows)), (rows, cols)), shape=(6, 6))
    gov.residual_jac(J, z, np.empty(0), _theta(gov), idxs, False)
    return J.toarray()


@pytest.mark.parametrize(
    "LG, expected, expected_derivative",
    [
        (0.05, 0.05, 1.0),
        (0.2, -0.2, -2.5),
        (-0.2, 0.2, -2.5),
        (0.3 / 3.5, 0.3 / 3.5, -2.5),
        (-0.3 / 3.5, -0.3 / 3.5, -2.5),
    ],
)
def test_rate_limit_branches_and_equality(LG, expected, expected_derivative):
    gov = _governor()
    z = np.array([0.0, LG, 0.5, 0.7, 0.8, 0.4])

    F = _residual(gov, z)
    J = _jacobian(gov, z)

    assert F[2] == pytest.approx(expected)
    assert J[2, 1] == pytest.approx(expected_derivative)


@pytest.mark.parametrize("g", [5e-5, 1e-4])
def test_floor_branch_is_finite_and_has_branch_consistent_derivative(g):
    gov = _governor(g_floor=1e-4)
    z = np.array([0.01, 0.02, 0.5, g, 0.2, 0.4])

    F = _residual(gov, z)
    J = _jacobian(gov, z)

    assert np.all(np.isfinite(F))
    assert np.all(np.isfinite(J))
    assert J[4, 3] == 0.0
    assert J[5, 3] == pytest.approx(-gov.DT * z[0])


@pytest.mark.parametrize(
    "z",
    [
        np.array([0.03, -0.1, 0.4, 0.6, 0.75, 0.2]),
        np.array([-0.02, 0.2, 0.4, 0.8, 0.35, 0.2]),
        np.array([0.01, -0.2, 0.4, 5e-5, 0.002, 0.2]),
    ],
)
def test_analytical_jacobian_matches_finite_difference(z):
    gov = _governor()
    analytical = _jacobian(gov, z)
    finite_difference = np.zeros_like(analytical)
    eps = 1e-7

    for col in range(z.size):
        z_plus = z.copy()
        z_minus = z.copy()
        z_plus[col] += eps
        z_minus[col] -= eps
        finite_difference[:, col] = (
            _residual(gov, z_plus) - _residual(gov, z_minus)
        ) / (2.0 * eps)

    np.testing.assert_allclose(analytical, finite_difference, rtol=2e-7, atol=2e-7)


def test_closed_form_initialization_has_zero_residual():
    gov = _governor()
    gov.p_m0 = 0.66
    x = np.zeros(4)
    y = np.zeros(1)

    gov.initialize(1.0, 0.0, 0.0, 0.0, x, y, None)
    z = np.concatenate(([0.0], x, y))

    q0 = gov.p_m0 / gov.At + gov.qNL
    np.testing.assert_allclose(x, [0.0, q0, q0, q0])
    assert y[0] == gov.p_m0
    assert gov.pref == pytest.approx(gov.R * q0)
    np.testing.assert_allclose(_residual(gov, z), 0.0, atol=1e-14)


@pytest.mark.parametrize(
    "p_m0, expected_min, expected_max",
    [(-0.22, -0.1, 1.2), (1.65, 0.0, 1.6)],
)
def test_explicit_policy_adjusts_effective_bounds_and_retains_originals(
    p_m0, expected_min, expected_max
):
    gov = _governor(adjust_initial_limits=True)
    gov.p_m0 = p_m0
    x = np.zeros(4)
    y = np.zeros(1)

    gov.initialize(1.0, 0.0, 0.0, 0.0, x, y, None)
    theta = _theta(gov)

    assert (gov.GMIN_original, gov.GMAX_original) == (0.0, 1.2)
    assert gov.GMIN == pytest.approx(expected_min)
    assert gov.GMAX == pytest.approx(expected_max)
    assert theta[7] == pytest.approx(expected_min)
    assert theta[6] == pytest.approx(expected_max)


def test_enabled_bounds_reject_out_of_range_initial_gate_by_default():
    gov = _governor(GMAX=0.5)
    gov.p_m0 = 0.66

    with pytest.raises(ValueError, match="outside enabled GMIN/GMAX bounds"):
        gov.initialize(1.0, 0.0, 0.0, 0.0, np.zeros(4), np.zeros(1), None)


def test_dimensions_parameters_and_bounded_state_metadata():
    gov = _governor()
    metadata = gov.bounded_state_metadata[0]

    assert gov.getdim() == (4, 1, 15)
    assert gov.state_list == ["LG", "gtpos", "g", "q", "p_m"]
    assert gov.p_m_idx == 0
    assert metadata.state_name == "gtpos"
    assert metadata.state_offset == 1
    assert metadata.lower_parameter_offset == 7
    assert metadata.upper_parameter_offset == 6
    assert metadata.enabled_parameter_offset == 13
    assert metadata.device_type == "HYGOV"


def test_enabled_and_effective_bounds_are_written_to_theta():
    gov = _governor(enable_limits=True)
    gov.pref = 0.4
    theta = _theta(gov)

    assert theta[6] == gov.GMAX
    assert theta[7] == gov.GMIN
    assert theta[12] == gov.g_floor
    assert theta[13] == 1.0
    assert theta[14] == gov.pref


@pytest.mark.parametrize("g_floor", [0.0, -1e-4])
def test_g_floor_must_be_positive(g_floor):
    with pytest.raises(ValueError, match="g_floor must be positive"):
        _governor(g_floor=g_floor)


def test_velocity_limit_must_be_non_negative():
    with pytest.raises(ValueError, match="VELM must be non-negative"):
        _governor(VELM=-0.1)


def test_disabled_limits_leave_gate_rate_unclipped():
    gov = _governor(enable_limits=False)
    z = np.array([0.0, 0.2, 0.5, 0.7, 0.8, 0.4])

    assert _residual(gov, z)[2] == pytest.approx(0.2)
    assert _jacobian(gov, z)[2, 1] == pytest.approx(1.0)


def test_hessian_operations_raise():
    gov = _governor()

    with pytest.raises(NotImplementedError, match="HYGOV"):
        gov.preallocate_hessian(0, np.array([1, 5, 0]), None)
    with pytest.raises(NotImplementedError, match="HYGOV"):
        gov.residual_hess(None, np.zeros(6), np.empty(0), np.zeros(15), np.array([1, 5, 0]))


def test_hygov_runtime_uses_theta_parameters():
    gov = _governor()
    z = np.array([0.0, 0.05, 0.5, 0.7, 0.8, 0.4])
    theta = _theta(gov)
    baseline = np.zeros_like(z)
    gov.residual_diff(baseline, z, np.empty(0), theta, np.array([1, 5, 0]), False)

    theta[14] += 0.1
    changed = np.zeros_like(z)
    gov.residual_diff(changed, z, np.empty(0), theta, np.array([1, 5, 0]), False)

    assert changed[1] != baseline[1]
