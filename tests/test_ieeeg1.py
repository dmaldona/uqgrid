import itertools

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from uqgrid.models.ieeeg1_imp import GovIEEEG1


def _governor(**overrides):
    values = {
        "BUS2": 0, "ID2": 0, "K": 20.0, "T1": 0.2, "T2": 0.05,
        "T3": 0.1, "UO": 0.3, "UC": -0.2, "PMAX": 1.2, "PMIN": 0.0,
        "T4": 0.4, "K1": 2.0, "T5": 1.0, "K3": 1.0,
        "T6": 0.5, "K5": 1.0, "T7": 0.2, "K7": 0.0,
        "K2": 0.0, "K4": 0.0, "K6": 0.0, "K8": 0.0,
    }
    values.update(overrides)
    governor = GovIEEEG1("1", **values)
    governor.set_pointers(0, 0, 0, 0)
    governor.w_idx = 7
    return governor


def _initialized(governor, mechanical_power=0.8):
    governor.p_m0 = mechanical_power
    x = np.zeros(8)
    y = np.zeros(governor.alg_dim)
    governor.initialize(1.0, 0.0, 0.0, 0.0, x, y, None)
    z = np.concatenate((x[:6], y, x[7:8]))
    governor.w_idx = 6 + governor.alg_dim
    idxs = np.array([0, 6, 0, 0], dtype=np.int64)
    theta = np.zeros(governor.par_dim)
    governor.initialize_theta(theta)
    return z, theta, idxs


def _jacobian(governor, z, theta, idxs):
    coords = governor.preallocate_jacobian(idxs, None, False)
    rows = []
    columns = []
    for row, row_columns in coords:
        rows.extend([row] * len(row_columns))
        columns.extend(row_columns)
    matrix = csr_matrix(
        (np.zeros(len(rows)), (rows, columns)), shape=(len(z), len(z))
    )
    governor.residual_jac(matrix, z, np.empty(0), theta, idxs, False)
    return matrix.toarray()


@pytest.mark.parametrize("zero_times", tuple(itertools.product((False, True), repeat=4)))
@pytest.mark.parametrize("zero_t1", (False, True))
def test_zero_time_constant_combinations_are_finite(zero_times, zero_t1):
    governor = _governor(
        T1=0.0 if zero_t1 else 0.2,
        **{
            name: 0.0 if zero else value
            for name, zero, value in zip(("T4", "T5", "T6", "T7"), zero_times, (0.4, 1.0, 0.5, 0.2))
        },
    )
    z, theta, idxs = _initialized(governor)
    residual = np.zeros_like(z)
    governor.residual_diff(residual, z, np.empty(0), theta, idxs, False)

    assert np.all(np.isfinite(residual))
    assert np.linalg.norm(residual, np.inf) < 1e-12

    inactive_offsets = [
        offset
        for offset, zero in zip(range(2, 6), zero_times)
        if zero
    ]
    if zero_t1:
        inactive_offsets.append(0)
    perturbed = z.copy()
    perturbed[inactive_offsets] += 10.0
    perturbed_residual = np.zeros_like(z)
    governor.residual_diff(
        perturbed_residual, perturbed, np.empty(0), theta, idxs, False
    )
    np.testing.assert_array_equal(perturbed_residual[inactive_offsets], 0.0)
    np.testing.assert_allclose(perturbed_residual, residual)


def test_initialization_uses_effective_coefficients_and_has_zero_residual():
    governor = _governor()
    z, theta, idxs = _initialized(governor)
    residual = np.zeros_like(z)
    governor.residual_diff(residual, z, np.empty(0), theta, idxs, False)

    assert sum(governor.normalized_K) == pytest.approx(1.0)
    assert governor.K1n == pytest.approx(0.5)
    assert np.linalg.norm(residual, np.inf) < 1e-12


def test_coefficients_below_one_are_preserved_by_shaft():
    governor = _governor(K1=0.4, K3=0.2, K5=0.1)

    assert governor.source_K[0::2] == pytest.approx((0.4, 0.2, 0.1, 0.0))
    assert governor.normalized_K[0::2] == pytest.approx((0.4, 0.2, 0.1, 0.0))
    assert governor.coefficient_adjustment_diagnostics["coefficients_adjusted"] is False


def test_coefficients_at_one_are_preserved_by_shaft():
    governor = _governor(K1=0.6, K3=0.3, K5=0.1)

    assert governor.source_K[0::2] == pytest.approx((0.6, 0.3, 0.1, 0.0))
    assert governor.normalized_K[0::2] == pytest.approx((0.6, 0.3, 0.1, 0.0))
    assert governor.coefficient_adjustment_diagnostics["coefficients_adjusted"] is False


def test_coefficients_above_one_are_normalized_within_hp_shaft():
    governor = _governor(K1=1.0, K3=0.5, K5=0.0)

    assert governor.normalized_K[0::2] == pytest.approx((2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0))
    diagnostics = governor.coefficient_adjustment_diagnostics
    assert diagnostics["hp_coefficients_adjusted"] is True
    assert diagnostics["lp_coefficients_adjusted"] is False
    assert diagnostics["effective_hp_coefficient_sum"] == pytest.approx(1.0)


def test_dual_output_coefficients_are_normalized_by_shaft():
    governor = _governor(
        BUS2=2,
        ID2="1",
        K1=0.4,
        K3=0.2,
        K5=0.1,
        K2=0.8,
        K4=0.4,
    )

    assert governor.normalized_K[0::2] == pytest.approx((0.4, 0.2, 0.1, 0.0))
    assert governor.normalized_K[1::2] == pytest.approx((2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0))
    diagnostics = governor.coefficient_adjustment_diagnostics
    assert diagnostics["hp_coefficients_adjusted"] is False
    assert diagnostics["lp_coefficients_adjusted"] is True


def test_dual_output_coefficients_at_one_are_preserved_by_shaft():
    governor = _governor(
        BUS2=2,
        ID2="1",
        K1=0.6,
        K3=0.4,
        K5=0.0,
        K2=0.75,
        K4=0.25,
    )

    assert governor.normalized_K[0::2] == pytest.approx((0.6, 0.4, 0.0, 0.0))
    assert governor.normalized_K[1::2] == pytest.approx((0.75, 0.25, 0.0, 0.0))
    assert governor.coefficient_adjustment_diagnostics["coefficients_adjusted"] is False


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"K1": -0.1}, "must be non-negative"),
        ({"K1": np.nan}, "must be finite"),
        ({"K1": 0.0, "K3": 0.0, "K5": 0.0}, "must be positive"),
        ({"K2": 0.1}, "must be zero without a secondary output"),
        (
            {"BUS2": 2, "ID2": "1", "K2": 0.0, "K4": 0.0, "K6": 0.0, "K8": 0.0},
            "must be positive for a secondary output",
        ),
    ],
)
def test_invalid_shaft_coefficients_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        _governor(**overrides)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"K": np.nan}, "K must be finite"),
        ({"T3": np.nan}, "T3 must be positive"),
        ({"T2": -0.1}, "time constants must be non-negative"),
        ({"T4": np.inf}, "time constants must be non-negative"),
        ({"UO": np.inf}, "UC <= 0 <= UO"),
        ({"PMIN": np.nan}, "PMIN must be less than PMAX"),
        ({"PMIN": 1.2, "PMAX": 1.2}, "PMIN must be less than PMAX"),
    ],
)
def test_invalid_gains_time_constants_and_limits_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        _governor(**overrides)


@pytest.mark.parametrize(
    "speed, expected_rate, clipped",
    [(0.0002, -0.04, False), (-0.1, 0.3, True), (0.1, -0.2, True)],
)
def test_valve_rate_branches(speed, expected_rate, clipped):
    governor = _governor(T1=0.0, T2=0.0)
    z, theta, idxs = _initialized(governor)
    z[governor.w_idx] = speed
    residual = np.zeros_like(z)
    governor.residual_diff(residual, z, np.empty(0), theta, idxs, False)
    jacobian = _jacobian(governor, z, theta, idxs)

    assert residual[1] == pytest.approx(expected_rate)
    assert bool(jacobian[1, governor.w_idx] == 0.0) is clipped


@pytest.mark.parametrize(
    "overrides,speed",
    [({}, 0.001), ({"T1": 0.0, "T5": 0.0, "T6": 0.0, "T7": 0.0}, 0.0005),
     ({"T4": 0.0, "T5": 0.0, "T6": 0.0, "T7": 0.0}, -0.1)],
)
def test_jacobian_matches_finite_difference(overrides, speed):
    governor = _governor(**overrides)
    z, theta, idxs = _initialized(governor)
    z[governor.w_idx] = speed
    analytical = _jacobian(governor, z, theta, idxs)
    numerical = np.zeros_like(analytical)
    epsilon = 1e-6
    for column in range(len(z)):
        plus = z.copy()
        minus = z.copy()
        plus[column] += epsilon
        minus[column] -= epsilon
        f_plus = np.zeros_like(z)
        f_minus = np.zeros_like(z)
        governor.residual_diff(f_plus, plus, np.empty(0), theta, idxs, False)
        governor.residual_diff(f_minus, minus, np.empty(0), theta, idxs, False)
        numerical[:, column] = (f_plus - f_minus) / (2.0 * epsilon)

    np.testing.assert_allclose(analytical, numerical, rtol=1e-7, atol=1e-8)


def test_enabled_bounds_reject_out_of_range_initial_valve_by_default():
    governor = _governor(PMAX=0.5)

    with pytest.raises(ValueError, match="outside enabled PMIN/PMAX bounds"):
        _initialized(governor, mechanical_power=0.8)

    assert governor.PMAX == 0.5
    assert governor.effective_PMAX == 0.5


def test_explicit_policy_adjusts_effective_bounds_without_changing_source():
    governor = _governor(PMAX=0.5, adjust_initial_limits=True)
    _, theta, _ = _initialized(governor, mechanical_power=0.8)
    metadata = governor.bounded_state_metadata[0]

    assert governor.PMAX == 0.5
    assert governor.effective_PMAX == pytest.approx(0.8)
    assert theta[metadata.upper_parameter_offset] == pytest.approx(0.8)
    assert theta[metadata.lower_parameter_offset] == 0.0
    assert theta[metadata.enabled_parameter_offset] == 1.0
    assert governor.limit_initialization_diagnostics["bounds_adjusted"] is True
    assert governor.limit_initialization_diagnostics["adjust_initial_limits"] is True
    assert governor.limit_initialization_diagnostics["coefficients_adjusted"] is True
    assert metadata.state_offset == 1
    assert metadata.device_type == "IEEEG1"


def test_disabled_limits_do_not_adjust_or_reject_initial_valve():
    governor = _governor(PMAX=0.5, enable_limits=False)
    _, theta, _ = _initialized(governor, mechanical_power=0.8)
    metadata = governor.bounded_state_metadata[0]

    assert governor.effective_PMAX == 0.5
    assert theta[metadata.upper_parameter_offset] == 0.5
    assert theta[metadata.enabled_parameter_offset] == 0.0
    assert governor.limit_initialization_diagnostics["bounds_adjusted"] is False


def test_disabled_limits_leave_valve_rate_unclipped():
    governor = _governor(
        T1=0.0, T2=0.0, UO=0.01, UC=-0.01, enable_limits=False
    )
    z, theta, idxs = _initialized(governor)
    z[governor.w_idx] = -0.1
    residual = np.zeros_like(z)
    governor.residual_diff(residual, z, np.empty(0), theta, idxs, False)
    jacobian = _jacobian(governor, z, theta, idxs)

    assert residual[1] > governor.UO
    assert jacobian[1, governor.w_idx] == pytest.approx(-governor.K / governor.T3)


@pytest.mark.parametrize(
    "side,pref_delta,expected_sign",
    [("upper", -0.2, -1.0), ("lower", 0.2, 1.0)],
)
def test_valve_at_position_bound_retains_inward_raw_velocity(
    side, pref_delta, expected_sign
):
    governor = _governor(UO=10.0, UC=-10.0)
    z, theta, idxs = _initialized(governor)
    z[1] = governor.PMAX if side == "upper" else governor.PMIN
    theta[28] = z[1] + pref_delta
    residual = np.zeros_like(z)

    governor.residual_diff(residual, z, np.empty(0), theta, idxs, False)

    assert np.sign(residual[1]) == expected_sign


def test_runtime_residual_and_jacobian_read_theta_values():
    governor = _governor(T1=0.0, T2=0.0)
    z, theta, idxs = _initialized(governor)
    z[governor.w_idx] = -0.001
    theta[0] = 10.0
    theta[4] = 0.08
    theta[28] = 0.79
    residual = np.zeros_like(z)
    governor.residual_diff(residual, z, np.empty(0), theta, idxs, False)
    jacobian = _jacobian(governor, z, theta, idxs)

    assert residual[1] == pytest.approx(0.0)
    assert jacobian[1, governor.w_idx] == pytest.approx(-100.0)

    theta[28] = 1.0
    governor.residual_diff(residual, z, np.empty(0), theta, idxs, False)
    jacobian = _jacobian(governor, z, theta, idxs)
    assert residual[1] == pytest.approx(0.08)
    assert jacobian[1, governor.w_idx] == 0.0


def test_optional_secondary_output_uses_algebraic_offset_one():
    governor = _governor(
        BUS2=2,
        ID2="1",
        K1=0.5,
        K2=0.25,
        K3=0.0,
        K5=0.0,
    )
    governor.p_m0_secondary = 0.2
    z, theta, idxs = _initialized(governor, mechanical_power=0.4)
    residual = np.zeros_like(z)
    governor.residual_diff(residual, z, np.empty(0), theta, idxs, False)
    jacobian = _jacobian(governor, z, theta, idxs)

    assert governor.alg_dim == 2
    assert governor.secondary_output_offset == 1
    assert np.linalg.norm(residual, np.inf) < 1e-12
    assert jacobian[7, 7] == -1.0


def test_hessian_is_explicitly_unsupported():
    governor = _governor()
    with pytest.raises(NotImplementedError):
        governor.preallocate_hessian(0, np.zeros(3, dtype=int), None)
    with pytest.raises(NotImplementedError):
        governor.residual_hess(None, None, None, None, None)


@pytest.mark.parametrize("limits", [(0.1, 0.2), (-0.2, -0.1), (0.2, -0.2)])
def test_rate_limits_must_contain_zero(limits):
    with pytest.raises(ValueError, match="UC <= 0 <= UO"):
        _governor(UC=limits[0], UO=limits[1])
