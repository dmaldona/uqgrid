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


def _state_for_raw_gate_rate(gov, raw_gate_rate):
    filter_state = 0.02
    desired_gate = 0.5
    actual_gate = 0.7
    flow = 0.8
    filter_rate = raw_gate_rate * gov.r - filter_state / gov.Tr
    speed = (
        gov.pref
        - gov.R * desired_gate
        - filter_state
        - gov.Tf * filter_rate
    )
    return np.array([speed, filter_state, desired_gate, actual_gate, flow, 0.4])


@pytest.mark.parametrize(
    "raw_gate_rate, expected, clipped",
    [
        (-0.5, -0.3, True),
        (-0.1, -0.1, False),
        (0.1, 0.1, False),
        (0.5, 0.3, True),
    ],
)
def test_rate_limit_branches_and_equality(raw_gate_rate, expected, clipped):
    gov = _governor()
    z = _state_for_raw_gate_rate(gov, raw_gate_rate)

    F = _residual(gov, z)
    J = _jacobian(gov, z)

    assert F[2] == pytest.approx(expected)
    expected_dfilter = 0.0 if clipped else (1.0 / gov.Tr - 1.0 / gov.Tf) / gov.r
    assert J[2, 1] == pytest.approx(expected_dfilter)
    assert bool(J[2, 0] == 0.0) is clipped
    assert bool(J[2, 2] == 0.0) is clipped


def test_residual_matches_independent_hygov_block_equations():
    gov = _governor(VELM=10.0)
    z = np.array([0.012, 0.03, 0.52, 0.49, 0.44, 0.61])

    filter_rate = (gov.pref - z[0] - gov.R * z[2] - z[1]) / gov.Tf
    gate_rate = (z[1] + gov.Tr * filter_rate) / (gov.r * gov.Tr)
    head = z[4] ** 2 / z[3] ** 2
    expected = np.array(
        [
            0.0,
            filter_rate,
            gate_rate,
            (z[2] - z[3]) / gov.Tg,
            (1.0 - head) / gov.Tw,
            gov.At * head * (z[4] - gov.qNL) - gov.DT * z[0] * z[3] - z[5],
        ]
    )

    np.testing.assert_allclose(_residual(gov, z), expected, atol=1e-14)


def test_temporary_droop_time_constant_changes_gate_rate():
    state = np.array([0.0, 0.04, 0.5, 0.5, 0.5, 0.4])
    short = _governor(Tr=2.0, VELM=10.0, enable_limits=False)
    long = _governor(Tr=8.0, VELM=10.0, enable_limits=False)

    assert _residual(short, state)[2] != pytest.approx(_residual(long, state)[2])


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
    z = _state_for_raw_gate_rate(gov, 0.8)

    assert _residual(gov, z)[2] == pytest.approx(0.8)
    assert _jacobian(gov, z)[2, 1] == pytest.approx(
        (1.0 / gov.Tr - 1.0 / gov.Tf) / gov.r
    )


@pytest.mark.parametrize(
    "parameter,value,message",
    [
        ("r", np.nan, "r must be positive"),
        ("Tr", np.inf, "Tr must be positive"),
        ("Tr", 0.0, "Tr must be positive"),
        ("Tf", 0.0, "Tf, Tg, and Tw must be positive"),
        ("Tg", -0.1, "Tf, Tg, and Tw must be positive"),
        ("Tw", np.nan, "Tf, Tg, and Tw must be positive"),
        ("Tw", 0.0, "Tf, Tg, and Tw must be positive"),
        ("At", 0.0, "At must be positive"),
        ("At", np.inf, "At must be positive"),
    ],
)
def test_required_time_constants_and_turbine_gain_are_positive(
    parameter, value, message
):
    with pytest.raises(ValueError, match=message):
        _governor(**{parameter: value})


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"GMIN": 1.0, "GMAX": 1.0}, "GMIN must be less than GMAX"),
        ({"GMIN": np.nan}, "GMIN must be less than GMAX"),
        ({"VELM": np.inf}, "VELM must be non-negative"),
        ({"g_floor": np.nan}, "g_floor must be positive"),
    ],
)
def test_invalid_limits_and_denominator_floor_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        _governor(**overrides)


@pytest.mark.parametrize(
    "side,pref_delta,expected_sign",
    [("upper", -0.2, -1.0), ("lower", 0.2, 1.0)],
)
def test_gate_at_position_bound_retains_inward_raw_velocity(
    side, pref_delta, expected_sign
):
    gov = _governor(VELM=10.0)
    gate = gov.GMAX if side == "upper" else gov.GMIN
    state = np.array([0.0, 0.0, gate, gate, max(gate, 0.1), 0.0])
    gov.pref = gov.R * gate + pref_delta

    assert np.sign(_residual(gov, state)[2]) == expected_sign


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
