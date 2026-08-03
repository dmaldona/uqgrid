import json
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
from uqgrid.simulation.pflow import runpf
from uqgrid.simulation.residual import residual_function


@pytest.fixture
def data_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )


def _descriptor(state_index=0, device_id="1"):
    return LimitedStateDescriptor(
        state_index=state_index,
        lower_bound=0.0,
        upper_bound=1.0,
        device_type="SEXS",
        bus=101 + state_index,
        device_id=device_id,
        enabled=True,
    )


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
    return psys, ctx, state_index, lower, upper


def _config(**overrides):
    values = {
        "method": "beuler",
        "petsc": False,
        "steps": 4,
        "dt": 0.005,
        "ton": 10.0,
        "toff": 11.0,
    }
    values.update(overrides)
    return IntegrationConfig(**values)


def test_fixed_active_rows_replace_residual_and_sparse_jacobian_rows():
    descriptors = [_descriptor(0, "upper"), _descriptor(1, "lower")]
    modes = {
        0: DynamicLimitMode.UPPER_ACTIVE,
        1: DynamicLimitMode.LOWER_ACTIVE,
    }
    state = np.asarray([1.25, -0.25, 7.0])
    residual = np.asarray([9.0, 8.0, 7.0])
    jacobian = csr_matrix(np.full((3, 3), 2.0))

    dynamics._apply_beuler_active_rows(
        residual, jacobian, state, descriptors, modes
    )

    np.testing.assert_allclose(residual, [0.25, -0.25, 7.0])
    np.testing.assert_allclose(jacobian.toarray()[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(jacobian.toarray()[1], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(jacobian.toarray()[2], [2.0, 2.0, 2.0])


@pytest.mark.parametrize("side", ["upper", "lower"])
def test_native_be_crossing_activates_bound_without_overshoot(data_dir, side):
    psys, ctx, state_index, lower, upper = _biased_two_bus(data_dir, side)

    result = integrate_system(psys, _config(), ctx)

    values = result["history"][state_index]
    assert np.min(values) >= lower - 1e-12
    assert np.max(values) <= upper + 1e-12
    bound = upper if side == "upper" else lower
    assert np.any(np.isclose(values, bound, atol=1e-10))
    events = result["dynamic_limit_diagnostics"]["events"]
    activations = [
        event
        for event in events
        if event["action"] == "activate" and event["side"] == side
    ]
    assert len(activations) == 1
    assert activations[0]["state_after"] == pytest.approx(bound)
    assert activations[0]["active_set_iterations"] == 1
    json.dumps(events, allow_nan=False)


@pytest.mark.parametrize("side", ["upper", "lower"])
def test_native_be_pins_outward_and_releases_for_inward_drive(data_dir, side):
    psys = _build_two_bus(data_dir)
    z0, theta, _ = _initialize_context(psys)
    exciter = psys.exc[0]
    state_index = exciter.dif_ptr + exciter.efd_idx
    ptr = exciter.par_ptr
    initial = float(z0[state_index])
    base_vref = float(theta[ptr + 7])
    if side == "upper":
        _set_limits(
            theta, exciter, initial - 0.1, initial, vref_offset=0.05
        )
    else:
        _set_limits(
            theta, exciter, initial, initial + 0.1, vref_offset=-0.05
        )
    descriptors = collect_limited_state_descriptors(psys, theta)
    modes = initialize_dynamic_limit_modes(descriptors)
    residual = np.zeros_like(z0)
    jacobian = preallocate_jacobian(psys)

    pinned, modes, activation_events = (
        dynamics._integrate_beuler_with_dynamic_limits(
            z0,
            theta,
            0.005,
            psys,
            residual,
            jacobian,
            descriptors=descriptors,
            modes=modes,
            state_tolerance=1e-8,
            release_tolerance=1e-10,
            max_active_set_iterations=20,
            endpoint_time=0.005,
            newton_tol=1e-10,
            newton_max_iter=50,
        )
    )
    assert pinned[state_index] == pytest.approx(initial, abs=1e-10)
    expected_mode = (
        DynamicLimitMode.UPPER_ACTIVE
        if side == "upper"
        else DynamicLimitMode.LOWER_ACTIVE
    )
    assert modes[state_index] == expected_mode

    theta[ptr + 7] = (
        base_vref - 0.05 if side == "upper" else base_vref + 0.05
    )
    released, modes, release_events = (
        dynamics._integrate_beuler_with_dynamic_limits(
            pinned,
            theta,
            0.005,
            psys,
            residual,
            jacobian,
            descriptors=descriptors,
            modes=modes,
            state_tolerance=1e-8,
            release_tolerance=1e-10,
            max_active_set_iterations=20,
            endpoint_time=0.01,
            newton_tol=1e-10,
            newton_max_iter=50,
        )
    )

    if side == "upper":
        assert released[state_index] < initial
    else:
        assert released[state_index] > initial
    assert modes[state_index] == DynamicLimitMode.FREE
    assert any(event["action"] == "activate" for event in activation_events)
    assert any(event["action"] == "release" for event in release_events)


def test_native_be_enforces_multiple_sexs_limits_together(data_dir):
    psys = load_psse(raw_filename=os.path.join(data_dir, "ieee9_v33.raw"))
    add_dyr(psys, os.path.join(data_dir, "ieee9bus_SEXS.dyr"))
    psys.createYbusComplex()
    psys.power_injection = False
    z0, theta, ctx = _initialize_context(psys)
    limited = []
    expected_buses = set()
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
        expected_buses.add(psys.buses[exciter.bus].id)
    ctx.set_theta(theta.copy())

    result = integrate_system(psys, _config(), ctx)

    for state_index, lower, upper in limited:
        values = result["history"][state_index]
        assert np.min(values) >= lower - 1e-12
        assert np.max(values) <= upper + 1e-12
    activation_buses = {
        event["bus"]
        for event in result["dynamic_limit_diagnostics"]["events"]
        if event["action"] == "activate"
    }
    assert activation_buses == expected_buses


def test_native_be_endpoint_is_dae_and_complementarity_consistent(data_dir):
    psys, ctx, state_index, _, upper = _biased_two_bus(data_dir, "upper")
    result = integrate_system(psys, _config(steps=1), ctx)
    zold = result["history"][:, 0]
    endpoint = result["history"][:, 1]
    theta = ctx.theta_user
    free_residual = np.zeros_like(endpoint)
    dynamics._assemble_beuler_residual(
        free_residual, endpoint, zold, 0.005, psys, theta
    )
    raw_residual = np.zeros_like(endpoint)
    residual_function(raw_residual, endpoint, theta, psys)

    assert endpoint[state_index] == pytest.approx(upper, abs=1e-10)
    assert free_residual[state_index] <= 1e-10
    assert np.linalg.norm(
        raw_residual[psys.num_dof_dif :], np.inf
    ) < 1e-8
    inactive_rows = np.ones(psys.num_dof_dif, dtype=bool)
    inactive_rows[state_index] = False
    assert np.linalg.norm(
        free_residual[: psys.num_dof_dif][inactive_rows], np.inf
    ) < 1e-8


def test_fault_projection_holds_differential_states_fixed(data_dir, monkeypatch):
    psys, ctx, state_index, lower, upper = _biased_two_bus(
        data_dir, "upper", fault=True
    )
    original_integrate = dynamics.integrate
    projection_pairs = []

    def recording_integrate(zold, theta, h, psys, *args, **kwargs):
        result = original_integrate(zold, theta, h, psys, *args, **kwargs)
        if h == 0.0:
            projection_pairs.append(
                (zold[: psys.num_dof_dif].copy(), result[0][: psys.num_dof_dif])
            )
        return result

    monkeypatch.setattr(dynamics, "integrate", recording_integrate)
    result = integrate_system(
        psys,
        _config(steps=3, ton=0.005, toff=0.01),
        ctx,
    )

    assert len(projection_pairs) == 2
    for before, after in projection_pairs:
        np.testing.assert_allclose(after, before, atol=1e-12)
    values = result["history"][state_index]
    assert np.min(values) >= lower - 1e-12
    assert np.max(values) <= upper + 1e-12


def test_native_be_failure_after_fault_application_restores_fault(
    data_dir, monkeypatch
):
    psys, ctx, _, _, _ = _biased_two_bus(
        data_dir, "upper", fault=True
    )
    original_step = dynamics._integrate_beuler_with_dynamic_limits
    calls = 0

    def fail_second_interval(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise DynamicLimitError(
                {
                    "enabled": True,
                    "phase": "runtime",
                    "method": "beuler",
                    "failure_reasons": ["active_set_cycle"],
                    "events": [],
                }
            )
        return original_step(*args, **kwargs)

    monkeypatch.setattr(
        dynamics,
        "_integrate_beuler_with_dynamic_limits",
        fail_second_interval,
    )

    with pytest.raises(DynamicLimitError):
        integrate_system(
            psys,
            _config(steps=2, ton=0.005, toff=0.02),
            ctx,
        )

    assert calls == 2
    assert psys.fault_events[0].active is False


def _run_failure_fixture(monkeypatch, updates, *, max_iterations=20):
    descriptor = _descriptor()
    psys = SimpleNamespace(num_dof_dif=1)
    residual = np.zeros(1)
    jacobian = csr_matrix(np.eye(1))

    monkeypatch.setattr(
        dynamics,
        "_solve_beuler_fixed_active_set",
        lambda *args, **kwargs: (np.asarray([0.5]), 3, 1e-12, True),
    )
    monkeypatch.setattr(
        dynamics,
        "_assemble_beuler_residual",
        lambda F, *args, **kwargs: F.fill(0.0),
    )
    calls = iter(updates)

    def fake_update(*args, **kwargs):
        mode = next(calls)
        return (
            {0: mode},
            True,
            {"consistent": False, "records": []},
            [],
        )

    monkeypatch.setattr(dynamics, "update_dynamic_limit_active_set", fake_update)
    return dynamics._integrate_beuler_with_dynamic_limits(
        np.asarray([0.5]),
        np.asarray([]),
        0.1,
        psys,
        residual,
        jacobian,
        descriptors=[descriptor],
        modes={0: DynamicLimitMode.FREE},
        state_tolerance=1e-8,
        release_tolerance=1e-10,
        max_active_set_iterations=max_iterations,
        endpoint_time=0.1,
        newton_tol=1e-10,
        newton_max_iter=10,
    )


def test_native_be_reports_active_set_cycle(monkeypatch):
    with pytest.raises(DynamicLimitError) as exc_info:
        _run_failure_fixture(
            monkeypatch,
            [DynamicLimitMode.UPPER_ACTIVE, DynamicLimitMode.FREE],
        )

    diagnostics = exc_info.value.diagnostics
    assert diagnostics["phase"] == "runtime"
    assert diagnostics["failure_reasons"] == ["active_set_cycle"]
    assert diagnostics["active_set_iterations"] == 2
    json.dumps(diagnostics, allow_nan=False)


def test_native_be_reports_active_set_iteration_limit(monkeypatch):
    with pytest.raises(DynamicLimitError) as exc_info:
        _run_failure_fixture(
            monkeypatch,
            [DynamicLimitMode.UPPER_ACTIVE],
            max_iterations=1,
        )

    diagnostics = exc_info.value.diagnostics
    assert diagnostics["failure_reasons"] == ["active_set_iteration_limit"]
    assert diagnostics["active_set_iterations"] == 1


def test_native_be_reports_fixed_set_newton_failure(monkeypatch):
    descriptor = _descriptor()
    monkeypatch.setattr(
        dynamics,
        "_solve_beuler_fixed_active_set",
        lambda *args, **kwargs: (np.asarray([0.5]), 4, np.inf, False),
    )

    with pytest.raises(DynamicLimitError) as exc_info:
        dynamics._integrate_beuler_with_dynamic_limits(
            np.asarray([0.5]),
            np.asarray([]),
            0.1,
            SimpleNamespace(num_dof_dif=1),
            np.zeros(1),
            csr_matrix(np.eye(1)),
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
    assert diagnostics["failure_reasons"] == ["newton_nonconvergence"]
    assert diagnostics["newton_iterations"] == 4
    assert diagnostics["residual_norm"] is None


def test_native_be_converges_toward_small_step_herk(data_dir):
    def run(method, dt, steps):
        psys, ctx, _, _, _ = _biased_two_bus(data_dir, "upper")
        result = integrate_system(
            psys,
            IntegrationConfig(
                method=method,
                petsc=False,
                dt=dt,
                steps=steps,
                ton=10.0,
                toff=11.0,
            ),
            ctx,
        )
        return result["history"][:, -1]

    reference = run("herk4", 0.00025, 80)
    coarse = run("beuler", 0.004, 5)
    fine = run("beuler", 0.002, 10)

    assert np.linalg.norm(fine - reference) < np.linalg.norm(coarse - reference)


def test_native_be_sexs_no_fault_trajectory_stays_flat(data_dir):
    psys = _build_two_bus(data_dir)

    result = integrate_system(
        psys,
        _config(steps=50, dt=0.001),
    )

    drift = np.max(np.abs(result["history"] - result["history"][:, [0]]))
    assert drift < 1e-8
    assert result["dynamic_limit_diagnostics"]["events"] == []


def test_disabled_dynamic_limits_preserve_legacy_beuler_path(
    data_dir, monkeypatch
):
    psys = _build_two_bus(data_dir)
    monkeypatch.setattr(
        dynamics,
        "_integrate_beuler_with_dynamic_limits",
        lambda *args, **kwargs: pytest.fail("limited BE path must not run"),
    )

    result = integrate_system(
        psys,
        _config(steps=1, enforce_dynamic_limits=False),
    )

    assert result["dynamic_limit_diagnostics"]["enabled"] is False
    assert result["dynamic_limit_diagnostics"]["events"] == []
