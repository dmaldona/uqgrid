import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from uqgrid.io.parse import add_dyr, load_psse
from uqgrid.simulation import dynamics
from uqgrid.simulation.config import IntegrationConfig, IntegrationCtx
from uqgrid.simulation.dynamic_limits import (
    DynamicLimitError,
    DynamicLimitMode,
    LimitedStateDescriptor,
    collect_limited_state_descriptors,
    initialize_dynamic_limit_modes,
)
from uqgrid.simulation.dynamics import (
    initialize_system,
    integrate_system,
    preallocate_jacobian,
)
from uqgrid.simulation.jacobian import residual_jacobian
from uqgrid.simulation.pflow import runpf
from uqgrid.simulation.residual import residual_function


@pytest.fixture
def data_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )


class _FakeVector:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float).copy()

    def __getitem__(self, key):
        return self.values[key]

    def setArray(self, values):
        self.values = np.asarray(values, dtype=float).copy()

    def assemble(self):
        return None


class _FakeMatrix:
    def __init__(self):
        self.value = None

    def setValuesCSR(self, indptr, indices, data):
        self.value = csr_matrix(
            (
                np.array(data, copy=True),
                np.array(indices, copy=True),
                np.array(indptr, copy=True),
            )
        )

    def assemble(self):
        return None


def _build_two_bus(data_dir, *, fault=False):
    psys = load_psse(raw_filename=os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, os.path.join(data_dir, "2bus_SEXS.dyr"))
    if fault:
        psys.add_busfault(1, 1e-4)
    psys.createYbusComplex()
    psys.power_injection = False
    return psys


def _initialize_context(psys):
    solution = runpf(psys, verbose=False)
    z0, theta = initialize_system(psys, solution)
    ctx = IntegrationCtx()
    ctx.set_initial_conditions(z0.copy())
    ctx.set_theta(theta.copy())
    return z0, theta, ctx


def _set_limits(theta, exciter, lower, upper, *, vref_offset=0.0):
    ptr = exciter.par_ptr
    theta[ptr + 4] = lower
    theta[ptr + 5] = upper
    theta[ptr + 6] = 1.0
    theta[ptr + 7] += vref_offset


def _biased_two_bus(data_dir, side, *, fault=False, width=0.01):
    psys = _build_two_bus(data_dir, fault=fault)
    z0, theta, ctx = _initialize_context(psys)
    exciter = psys.exc[0]
    state_index = exciter.dif_ptr + exciter.efd_idx
    initial = float(z0[state_index])
    lower = initial - width
    upper = initial + width
    _set_limits(
        theta,
        exciter,
        lower,
        upper,
        vref_offset=0.05 if side == "upper" else -0.05,
    )
    ctx.set_theta(theta.copy())
    config = IntegrationConfig(
        method="cn",
        petsc=True,
        steps=4,
        dt=0.005,
        ton=10.0,
        toff=11.0,
        newton_tol=1e-10,
        newton_max_iter=100,
    )
    return psys, ctx, config, state_index, lower, upper


def _descriptor(index, lower=0.0, upper=1.0):
    return LimitedStateDescriptor(
        state_index=index,
        lower_bound=lower,
        upper_bound=upper,
        device_type="SEXS",
        bus=index + 1,
        device_id=str(index + 1),
        enabled=True,
    )


def test_petsc_cn_callbacks_match_trapezoidal_rows_for_active_limits(data_dir):
    psys = load_psse(raw_filename=os.path.join(data_dir, "ieee9_v33.raw"))
    add_dyr(psys, os.path.join(data_dir, "ieee9bus_SEXS.dyr"))
    psys.createYbusComplex()
    psys.power_injection = False
    z0, theta, _ = _initialize_context(psys)
    descriptors = collect_limited_state_descriptors(psys, theta)
    modes = initialize_dynamic_limit_modes(descriptors)
    modes[descriptors[0].state_index] = DynamicLimitMode.UPPER_ACTIVE
    modes[descriptors[1].state_index] = DynamicLimitMode.LOWER_ACTIVE
    endpoint = z0.copy()
    endpoint[descriptors[0].state_index] = descriptors[0].upper_bound
    endpoint[descriptors[1].state_index] = descriptors[1].lower_bound
    h = 0.005
    start_derivative = dynamics._effective_cn_start_derivative(
        z0,
        theta,
        psys,
        descriptors,
        modes,
        tolerance=1e-8,
    )

    ordinary_residual = np.zeros_like(endpoint)
    dynamics._assemble_cn_residual(
        ordinary_residual,
        endpoint,
        z0,
        h,
        start_derivative,
        psys,
        theta,
    )
    ordinary_jacobian = preallocate_jacobian(psys)
    residual_jacobian(ordinary_jacobian, endpoint, theta, psys)
    dynamics.jacobian_beuler(
        ordinary_jacobian, psys.num_dof_dif, 0.5 * h
    )

    callback_residual = np.zeros_like(endpoint)
    callback_jacobian = preallocate_jacobian(psys)
    problem = dynamics._PETScCNActiveSetProblem(
        psys, theta, callback_residual, callback_jacobian
    )
    problem.set_interval(
        z0,
        h,
        descriptors,
        modes,
        start_derivative=start_derivative,
    )
    x = _FakeVector(endpoint)
    f = _FakeVector(np.zeros_like(endpoint))
    matrix = _FakeMatrix()
    problem.function(None, x, f)
    problem.jacobian_function(None, x, matrix, matrix)

    active_indices = {
        descriptors[0].state_index,
        descriptors[1].state_index,
    }
    for state_index in active_indices:
        expected = np.zeros(endpoint.size)
        expected[state_index] = 1.0
        np.testing.assert_allclose(matrix.value.toarray()[state_index], expected)
    inactive = [
        index for index in range(endpoint.size) if index not in active_indices
    ]
    np.testing.assert_allclose(
        matrix.value.toarray()[inactive], ordinary_jacobian.toarray()[inactive]
    )
    np.testing.assert_allclose(f.values[inactive], ordinary_residual[inactive])


def test_cn_start_derivative_blocks_only_inherited_outward_modes(monkeypatch):
    raw = np.asarray([2.0, -3.0, 4.0])

    def fake_residual(output, state, theta, psys):
        output[:] = raw

    monkeypatch.setattr(dynamics, "residual_function", fake_residual)
    descriptors = [_descriptor(0), _descriptor(1), _descriptor(2)]
    modes = {
        0: DynamicLimitMode.UPPER_ACTIVE,
        1: DynamicLimitMode.LOWER_ACTIVE,
        2: DynamicLimitMode.FREE,
    }
    state = np.asarray([1.0, 0.0, 1.0])

    effective = dynamics._effective_cn_start_derivative(
        state,
        np.asarray([]),
        SimpleNamespace(num_dof_dif=3),
        descriptors,
        modes,
        tolerance=1e-8,
    )
    np.testing.assert_allclose(effective, [0.0, 0.0, 4.0])

    raw[:] = [-2.0, 3.0, 4.0]
    effective = dynamics._effective_cn_start_derivative(
        state,
        np.asarray([]),
        SimpleNamespace(num_dof_dif=3),
        descriptors,
        modes,
        tolerance=1e-8,
    )
    np.testing.assert_allclose(effective, raw)


def test_cn_start_derivative_is_fixed_across_active_set_retries(monkeypatch):
    descriptor = _descriptor(0)
    captured = []

    class RetryWorkspace:
        def solve_fixed_active_set(
            self, zold, h, descriptors, modes, *, start_derivative
        ):
            captured.append(start_derivative)
            if len(captured) == 1:
                return np.asarray([1.2]), 1, 0.0, True, {}
            return np.asarray([1.0]), 1, 0.0, True, {}

    fixed_start = np.asarray([0.25])
    monkeypatch.setattr(
        dynamics,
        "_effective_cn_start_derivative",
        lambda *args, **kwargs: fixed_start.copy(),
    )

    def fake_assemble(output, endpoint, *args, **kwargs):
        output[:] = -0.1

    monkeypatch.setattr(dynamics, "_assemble_cn_residual", fake_assemble)
    endpoint, modes, events = dynamics._integrate_petsc_cn_with_dynamic_limits(
        np.asarray([0.5]),
        np.asarray([]),
        0.1,
        SimpleNamespace(num_dof_dif=1),
        np.zeros(1),
        csr_matrix(np.eye(1)),
        workspace=RetryWorkspace(),
        descriptors=[descriptor],
        modes={0: DynamicLimitMode.FREE},
        state_tolerance=1e-8,
        release_tolerance=1e-10,
        max_active_set_iterations=20,
        endpoint_time=0.1,
        newton_tol=1e-10,
        newton_max_iter=10,
    )

    assert len(captured) == 2
    assert captured[0] is captured[1]
    assert captured[0].flags.writeable is False
    assert endpoint[0] == pytest.approx(1.0)
    assert modes[0] == DynamicLimitMode.UPPER_ACTIVE
    assert [event["action"] for event in events] == ["activate"]


def test_cn_nonfinite_start_derivative_has_runtime_context(
    data_dir, monkeypatch
):
    psys = _build_two_bus(data_dir)
    z0, theta, _ = _initialize_context(psys)
    state_index = psys.exc[0].dif_ptr + psys.exc[0].efd_idx
    descriptors = collect_limited_state_descriptors(psys, theta)
    modes = initialize_dynamic_limit_modes(descriptors)
    modes[state_index] = DynamicLimitMode.UPPER_ACTIVE
    original_residual = dynamics.residual_function

    def non_finite_efd_derivative(F, z, theta, psys):
        original_residual(F, z, theta, psys)
        F[state_index] = np.nan

    monkeypatch.setattr(
        dynamics, "residual_function", non_finite_efd_derivative
    )

    with pytest.raises(DynamicLimitError) as exc_info:
        dynamics._integrate_petsc_cn_with_dynamic_limits(
            z0,
            theta,
            0.005,
            psys,
            np.zeros_like(z0),
            preallocate_jacobian(psys),
            workspace=SimpleNamespace(),
            descriptors=descriptors,
            modes=modes,
            state_tolerance=1e-8,
            release_tolerance=1e-10,
            max_active_set_iterations=20,
            endpoint_time=0.005,
            newton_tol=1e-10,
            newton_max_iter=100,
        )

    diagnostics = exc_info.value.diagnostics
    assert diagnostics["phase"] == "runtime"
    assert diagnostics["method"] == "cn"
    assert diagnostics["backend"] == "petsc"
    assert diagnostics["operation"] == "derivative_projection"
    assert diagnostics["failure_reasons"] == ["non_finite_derivative"]
    assert diagnostics["time"] == pytest.approx(0.0)
    assert diagnostics["stage_or_endpoint"] == "start"
    json.dumps(diagnostics, allow_nan=False)


@pytest.mark.parametrize("side", ["upper", "lower"])
def test_limited_petsc_cn_activates_without_bound_overshoot(data_dir, side):
    pytest.importorskip("petsc4py")
    psys, ctx, config, state_index, lower, upper = _biased_two_bus(
        data_dir, side
    )

    result = integrate_system(psys, config, ctx)

    values = result["history"][state_index]
    assert np.min(values) >= lower - config.dynamic_limit_tolerance
    assert np.max(values) <= upper + config.dynamic_limit_tolerance
    events = result["dynamic_limit_diagnostics"]["events"]
    assert any(
        event["action"] == "activate" and event["side"] == side
        for event in events
    )


@pytest.mark.parametrize("side", ["upper", "lower"])
def test_limited_petsc_cn_releases_immediately_under_inward_drive(data_dir, side):
    pytest.importorskip("petsc4py")
    psys = _build_two_bus(data_dir)
    z0, theta, _ = _initialize_context(psys)
    exciter = psys.exc[0]
    state_index = exciter.dif_ptr + exciter.efd_idx
    ptr = exciter.par_ptr
    initial = float(z0[state_index])
    base_vref = float(theta[ptr + 7])
    if side == "upper":
        _set_limits(theta, exciter, initial - 0.1, initial, vref_offset=0.05)
    else:
        _set_limits(theta, exciter, initial, initial + 0.1, vref_offset=-0.05)
    descriptors = collect_limited_state_descriptors(psys, theta)
    modes = initialize_dynamic_limit_modes(descriptors)
    initial_modes = dict(modes)
    residual = np.zeros_like(z0)
    jacobian = preallocate_jacobian(psys)
    PETSc = dynamics._get_petsc_for_config(
        IntegrationConfig(petsc=True, method="cn")
    )
    workspace = dynamics._PETScCNActiveSetWorkspace(
        PETSc,
        psys,
        theta,
        residual,
        jacobian,
        newton_tol=1e-10,
        newton_max_iter=100,
    )
    assert workspace.snes_type == "newtonls"
    try:
        pinned, modes, first_events = (
            dynamics._integrate_petsc_cn_with_dynamic_limits(
                z0,
                theta,
                0.005,
                psys,
                residual,
                jacobian,
                workspace=workspace,
                descriptors=descriptors,
                modes=modes,
                state_tolerance=1e-8,
                release_tolerance=1e-10,
                max_active_set_iterations=20,
                endpoint_time=0.005,
                newton_tol=1e-10,
                newton_max_iter=100,
            )
        )
        first_start = dynamics._effective_cn_start_derivative(
            z0,
            theta,
            psys,
            descriptors,
            initial_modes,
            tolerance=1e-8,
        )
        first_free_residual = np.zeros_like(z0)
        dynamics._assemble_cn_residual(
            first_free_residual,
            pinned,
            z0,
            0.005,
            first_start,
            psys,
            theta,
        )
        pinned_modes = dict(modes)
        theta[ptr + 7] = (
            base_vref - 0.05 if side == "upper" else base_vref + 0.05
        )
        released, modes, second_events = (
            dynamics._integrate_petsc_cn_with_dynamic_limits(
                pinned,
                theta,
                0.005,
                psys,
                residual,
                jacobian,
                workspace=workspace,
                descriptors=descriptors,
                modes=modes,
                state_tolerance=1e-8,
                release_tolerance=1e-10,
                max_active_set_iterations=20,
                endpoint_time=0.01,
                newton_tol=1e-10,
                newton_max_iter=100,
            )
        )
        second_start = dynamics._effective_cn_start_derivative(
            pinned,
            theta,
            psys,
            descriptors,
            pinned_modes,
            tolerance=1e-8,
        )
        second_free_residual = np.zeros_like(z0)
        dynamics._assemble_cn_residual(
            second_free_residual,
            released,
            pinned,
            0.005,
            second_start,
            psys,
            theta,
        )
    finally:
        workspace.destroy()

    assert pinned[state_index] == pytest.approx(initial)
    assert modes[state_index] == DynamicLimitMode.FREE
    if side == "upper":
        assert released[state_index] < pinned[state_index]
        assert first_free_residual[state_index] <= 1e-10
    else:
        assert released[state_index] > pinned[state_index]
        assert first_free_residual[state_index] >= -1e-10
    assert (
        np.linalg.norm(first_free_residual[psys.num_dof_dif :], np.inf)
        < 1e-8
    )
    assert np.linalg.norm(second_free_residual, np.inf) < 1e-8
    assert [event["action"] for event in first_events + second_events] == [
        "activate",
        "release",
    ]


def test_limited_petsc_cn_handles_multiple_sexs_limits(data_dir):
    pytest.importorskip("petsc4py")
    psys = load_psse(raw_filename=os.path.join(data_dir, "ieee9_v33.raw"))
    add_dyr(psys, os.path.join(data_dir, "ieee9bus_SEXS.dyr"))
    psys.createYbusComplex()
    psys.power_injection = False
    z0, theta, ctx = _initialize_context(psys)
    limited = []
    for index, exciter in enumerate(psys.exc):
        state_index = exciter.dif_ptr + exciter.efd_idx
        initial = float(z0[state_index])
        _set_limits(
            theta,
            exciter,
            initial - 0.01,
            initial + 0.01,
            vref_offset=0.05 if index % 2 == 0 else -0.05,
        )
        limited.append((state_index, initial - 0.01, initial + 0.01))
    ctx.set_theta(theta.copy())

    result = integrate_system(
        psys,
        IntegrationConfig(
            method="cn",
            petsc=True,
            steps=3,
            dt=0.005,
            ton=10.0,
            toff=11.0,
        ),
        ctx,
    )

    for state_index, lower, upper in limited:
        values = result["history"][state_index]
        assert np.min(values) >= lower - 1e-8
        assert np.max(values) <= upper + 1e-8


def test_limited_petsc_cn_fault_projection_uses_shared_grid(data_dir, monkeypatch):
    pytest.importorskip("petsc4py")
    psys, ctx, config, _, _, _ = _biased_two_bus(
        data_dir, "upper", fault=True
    )
    config.ton = 0.0075
    config.toff = 0.0135
    projections = []
    original_integrate = dynamics.integrate

    def recording_integrate(zold, theta, h, psys, *args, **kwargs):
        result = original_integrate(zold, theta, h, psys, *args, **kwargs)
        if h == 0.0:
            projections.append((np.array(zold, copy=True), result[0].copy()))
        return result

    monkeypatch.setattr(dynamics, "integrate", recording_integrate)
    result = integrate_system(psys, config, ctx)

    np.testing.assert_allclose(
        result["tvec"],
        [0.0, 0.005, 0.0075, 0.01, 0.0135, 0.015, 0.02],
    )
    assert len(projections) == 2
    for before, after in projections:
        np.testing.assert_allclose(
            after[: psys.num_dof_dif], before[: psys.num_dof_dif]
        )
    fault = psys.fault_events[0]
    residual = np.zeros(result["history"].shape[0])
    fault.apply()
    residual_function(residual, result["history"][:, 2], ctx.theta_user, psys)
    assert np.linalg.norm(residual[psys.num_dof_dif :], np.inf) < 1e-8
    fault.remove()
    residual_function(residual, result["history"][:, 4], ctx.theta_user, psys)
    assert np.linalg.norm(residual[psys.num_dof_dif :], np.inf) < 1e-8
    assert fault.active is False


def test_limited_petsc_cn_bypasses_ts(data_dir, monkeypatch):
    pytest.importorskip("petsc4py")
    psys, ctx, config, _, _, _ = _biased_two_bus(data_dir, "upper")
    config.steps = 1
    monkeypatch.setattr(
        dynamics,
        "_petsc_ts_type",
        lambda *args: pytest.fail("limited PETSc CN must not configure TS"),
    )

    result = integrate_system(psys, config, ctx)

    assert result["history"].shape[1] == 2


def test_disabled_limits_keep_existing_petsc_cn_ts_path(data_dir, monkeypatch):
    pytest.importorskip("petsc4py")
    psys = _build_two_bus(data_dir)
    monkeypatch.setattr(
        dynamics,
        "_integrate_system_petsc_cn_limits",
        lambda *args: pytest.fail("disabled limits must keep TSCN"),
    )

    result = integrate_system(
        psys,
        IntegrationConfig(
            petsc=True,
            method="cn",
            enforce_dynamic_limits=False,
            steps=1,
            dt=0.005,
            ton=10.0,
            toff=11.0,
        ),
    )

    assert result["history"].shape[1] == 2


def test_zero_limited_states_keep_existing_petsc_cn_ts_path(
    data_dir, monkeypatch
):
    pytest.importorskip("petsc4py")
    psys = load_psse(raw_filename=os.path.join(data_dir, "ieee9_v33.raw"))
    add_dyr(psys, os.path.join(data_dir, "ieee9bus.dyr"))
    psys.createYbusComplex()
    monkeypatch.setattr(
        dynamics,
        "_integrate_system_petsc_cn_limits",
        lambda *args: pytest.fail("zero limited states must keep TSCN"),
    )

    result = integrate_system(
        psys,
        IntegrationConfig(
            petsc=True,
            method="cn",
            steps=1,
            dt=0.005,
            ton=10.0,
            toff=11.0,
        ),
    )

    assert result["dynamic_limit_diagnostics"]["enabled_state_count"] == 0
    assert result["history"].shape[1] == 2


def test_petsc_cn_snes_nonconvergence_has_method_diagnostics(monkeypatch):
    descriptor = _descriptor(0)
    monkeypatch.setattr(
        dynamics,
        "_effective_cn_start_derivative",
        lambda *args, **kwargs: np.asarray([0.0]),
    )

    class FailedWorkspace:
        def solve_fixed_active_set(self, *args, **kwargs):
            return (
                np.asarray([0.5]),
                4,
                np.inf,
                False,
                {
                    "failure_reason": "snes_nonconvergence",
                    "snes_type": "newtonls",
                    "snes_converged_reason": -5,
                    "snes_converged_reason_name": "diverged_max_it",
                    "snes_iterations": 4,
                    "snes_function_norm": None,
                },
            )

    with pytest.raises(DynamicLimitError) as exc_info:
        dynamics._integrate_petsc_cn_with_dynamic_limits(
            np.asarray([0.5]),
            np.asarray([]),
            0.1,
            SimpleNamespace(num_dof_dif=1),
            np.zeros(1),
            csr_matrix(np.eye(1)),
            workspace=FailedWorkspace(),
            descriptors=[descriptor],
            modes={0: DynamicLimitMode.FREE},
            state_tolerance=1e-8,
            release_tolerance=1e-10,
            max_active_set_iterations=20,
            endpoint_time=0.1,
            newton_tol=1e-10,
            newton_max_iter=4,
        )

    diagnostics = exc_info.value.diagnostics
    assert diagnostics["method"] == "cn"
    assert diagnostics["operation"] == "cn_active_set"
    assert diagnostics["failure_reasons"] == ["snes_nonconvergence"]
    assert diagnostics["snes_converged_reason"] == -5
    json.dumps(diagnostics, allow_nan=False)


def _run_scalar_cn(monkeypatch, *, rate, lower, upper, z0, step_sizes):
    pytest.importorskip("petsc4py")

    def scalar_residual(output, state, theta, psys):
        output[0] = rate(state[0])

    def scalar_jacobian(jacobian, state, theta, psys):
        epsilon = 1e-7
        derivative = (
            rate(state[0] + epsilon) - rate(state[0] - epsilon)
        ) / (2.0 * epsilon)
        jacobian.data[:] = derivative

    monkeypatch.setattr(dynamics, "residual_function", scalar_residual)
    monkeypatch.setattr(dynamics, "residual_jacobian", scalar_jacobian)
    psys = SimpleNamespace(num_dof_dif=1)
    descriptor = _descriptor(0, lower=lower, upper=upper)
    residual = np.zeros(1)
    jacobian = csr_matrix(np.asarray([[1.0]]))
    PETSc = dynamics._get_petsc_for_config(
        IntegrationConfig(petsc=True, method="cn")
    )
    workspace = dynamics._PETScCNActiveSetWorkspace(
        PETSc,
        psys,
        np.asarray([]),
        residual,
        jacobian,
        newton_tol=1e-12,
        newton_max_iter=50,
    )
    state = np.asarray([z0], dtype=float)
    modes = {0: DynamicLimitMode.FREE}
    events = []
    time = 0.0
    try:
        for h in step_sizes:
            time += h
            state, modes, interval_events = (
                dynamics._integrate_petsc_cn_with_dynamic_limits(
                    state,
                    np.asarray([]),
                    h,
                    psys,
                    residual,
                    jacobian,
                    workspace=workspace,
                    descriptors=[descriptor],
                    modes=modes,
                    state_tolerance=1e-12,
                    release_tolerance=1e-12,
                    max_active_set_iterations=20,
                    endpoint_time=time,
                    newton_tol=1e-12,
                    newton_max_iter=50,
                    prior_events=events,
                )
            )
            events.extend(interval_events)
    finally:
        workspace.destroy()
    return state[0], events


def test_limited_petsc_cn_is_second_order_away_from_switching(monkeypatch):
    errors = []
    for h in (0.2, 0.1, 0.05):
        endpoint, events = _run_scalar_cn(
            monkeypatch,
            rate=lambda value: -value,
            lower=-10.0,
            upper=10.0,
            z0=1.0,
            step_sizes=[h] * round(1.0 / h),
        )
        assert events == []
        errors.append(abs(endpoint - math.exp(-1.0)))

    assert errors[0] / errors[1] > 3.5
    assert errors[1] / errors[2] > 3.5


def test_limited_petsc_cn_switch_time_error_is_first_order(monkeypatch):
    crossing_time = 0.1
    errors = []
    step_sizes = []
    for intervals_before_half_step in (4, 8, 16):
        h = crossing_time / (intervals_before_half_step + 0.5)
        endpoint, events = _run_scalar_cn(
            monkeypatch,
            rate=lambda value: 1.0,
            lower=-1.0,
            upper=crossing_time,
            z0=0.0,
            step_sizes=[h] * (intervals_before_half_step + 1),
        )
        assert endpoint == pytest.approx(crossing_time)
        activation_time = next(
            event["time"]
            for event in events
            if event["action"] == "activate"
        )
        step_sizes.append(h)
        errors.append(activation_time - crossing_time)

    np.testing.assert_allclose(
        np.asarray(errors) / np.asarray(step_sizes),
        0.5,
        atol=1e-10,
    )
