import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from uqgrid.io.parse import add_dyr, load_psse
from uqgrid.simulation import herk
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


def _build_two_bus_sexs(data_dir, *, fault=False):
    psys = load_psse(raw_filename=os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, os.path.join(data_dir, "2bus_SEXS.dyr"))
    psys.createYbusComplex()
    psys.power_injection = False
    if fault:
        psys.add_busfault(1, 1e-4)
    return psys


def _initialize_context(psys):
    pf_solution = runpf(psys, verbose=False)
    z0, theta = initialize_system(psys, pf_solution)
    ctx = IntegrationCtx()
    ctx.set_initial_conditions(z0.copy())
    ctx.set_theta(theta.copy())
    return z0, theta, ctx


def _set_sexs_limits(theta, exciter, lower, upper, *, vref_offset=0.0):
    par_ptr = exciter.par_ptr
    theta[par_ptr + 4] = lower
    theta[par_ptr + 5] = upper
    theta[par_ptr + 6] = 1.0
    theta[par_ptr + 7] += vref_offset


def _biased_two_bus_context(data_dir, side, *, width=0.01):
    psys = _build_two_bus_sexs(data_dir)
    z0, theta, ctx = _initialize_context(psys)
    exciter = psys.exc[0]
    state_index = exciter.dif_ptr + exciter.efd_idx
    initial_efd = float(z0[state_index])
    _set_sexs_limits(
        theta,
        exciter,
        initial_efd - width,
        initial_efd + width,
        vref_offset=0.05 if side == "upper" else -0.05,
    )
    ctx.set_theta(theta.copy())
    return psys, ctx, state_index, initial_efd - width, initial_efd + width


@pytest.mark.parametrize("method", ["herk2", "herk4"])
@pytest.mark.parametrize("side", ["upper", "lower"])
def test_herk_crossings_project_and_block_without_stored_overshoot(
    data_dir, method, side
):
    psys, ctx, state_index, lower, upper = _biased_two_bus_context(
        data_dir, side
    )
    config = IntegrationConfig(
        method=method,
        petsc=False,
        steps=4,
        dt=0.005,
        ton=10.0,
        toff=11.0,
    )

    result = integrate_system(psys, config, ctx)

    efd = result["history"][state_index]
    assert np.min(efd) >= lower - 1e-12
    assert np.max(efd) <= upper + 1e-12
    expected_bound = upper if side == "upper" else lower
    assert np.any(np.isclose(efd, expected_bound, atol=1e-12))
    events = result["dynamic_limit_diagnostics"]["events"]
    assert any(
        event["action"] == "project" and event["side"] == side
        for event in events
    )
    assert any(
        event["action"] == "activate" and event["side"] == side
        for event in events
    )
    assert not any(
        event["action"] == "block_outward_derivative" for event in events
    )
    assert all(event["time"] is not None for event in events)
    assert all(event["stage_or_endpoint"] is not None for event in events)
    json.dumps(events, allow_nan=False)


@pytest.mark.parametrize("method", ["herk2", "herk4"])
@pytest.mark.parametrize("side", ["upper", "lower"])
def test_herk_releases_immediately_for_inward_derivative(
    data_dir, method, side
):
    psys = _build_two_bus_sexs(data_dir)
    z0, theta, _ = _initialize_context(psys)
    exciter = psys.exc[0]
    state_index = exciter.dif_ptr + exciter.efd_idx
    par_ptr = exciter.par_ptr
    initial_efd = float(z0[state_index])
    base_vref = float(theta[par_ptr + 7])
    if side == "upper":
        _set_sexs_limits(
            theta,
            exciter,
            initial_efd - 0.1,
            initial_efd,
            vref_offset=0.05,
        )
    else:
        _set_sexs_limits(
            theta,
            exciter,
            initial_efd,
            initial_efd + 0.1,
            vref_offset=-0.05,
        )

    descriptors = collect_limited_state_descriptors(psys, theta)
    modes = initialize_dynamic_limit_modes(descriptors)
    F = np.zeros_like(z0)
    J = preallocate_jacobian(psys)
    A, b, c = herk.TABLEAUS[method]

    z_bound, modes, activation_events = herk._herk_step_with_limits(
        z0,
        theta,
        0.005,
        psys,
        A,
        b,
        c,
        F,
        J,
        1e-10,
        50,
        limit_descriptors=descriptors,
        limit_tolerance=1e-8,
        limit_modes=modes,
        t_start=0.0,
    )
    theta[par_ptr + 7] = (
        base_vref - 0.05 if side == "upper" else base_vref + 0.05
    )
    z_released, modes, release_events = herk._herk_step_with_limits(
        z_bound,
        theta,
        0.005,
        psys,
        A,
        b,
        c,
        F,
        J,
        1e-10,
        50,
        limit_descriptors=descriptors,
        limit_tolerance=1e-8,
        limit_modes=modes,
        t_start=0.005,
    )

    assert z_bound[state_index] == pytest.approx(initial_efd)
    if side == "upper":
        assert z_released[state_index] < initial_efd
    else:
        assert z_released[state_index] > initial_efd
    assert any(
        event["action"] == "activate" and event["side"] == side
        for event in activation_events
    )
    assert any(
        event["action"] == "release" and event["side"] == side
        for event in release_events
    )
    assert modes[state_index] == DynamicLimitMode.FREE


@pytest.mark.parametrize(
    "method, raw_derivatives, expected_endpoint",
    [
        ("herk2", [2.0, -1.0, 0.0], 0.5),
        ("herk4", [4.0, -2.0, 0.0, 0.0, 0.0], 0.0),
    ],
)
def test_projected_predictor_does_not_pin_weighted_endpoint(
    monkeypatch, method, raw_derivatives, expected_endpoint
):
    calls = iter(raw_derivatives)

    def fake_residual(F, z, theta, psys):
        F.fill(0.0)
        F[0] = next(calls)

    def fake_algebraic(X_i, y0, v0, *args, **kwargs):
        return np.asarray([]), np.asarray([]), 0

    monkeypatch.setattr(herk, "residual_function", fake_residual)
    monkeypatch.setattr(herk, "solve_stage_algebraic", fake_algebraic)
    descriptor = LimitedStateDescriptor(
        state_index=0,
        lower_bound=-10.0,
        upper_bound=1.0,
        device_type="SEXS",
        bus=1,
        device_id="1",
        enabled=True,
    )
    modes = initialize_dynamic_limit_modes([descriptor])
    psys = SimpleNamespace(num_dof_dif=1, num_dof_alg=0)
    A, b, c = herk.TABLEAUS[method]

    endpoint, modes, events = herk._herk_step_with_limits(
        np.asarray([0.0]),
        np.asarray([]),
        1.0,
        psys,
        A,
        b,
        c,
        np.zeros(1),
        None,
        1e-10,
        5,
        limit_descriptors=[descriptor],
        limit_tolerance=1e-8,
        limit_modes=modes,
        t_start=0.0,
    )

    assert endpoint[0] == pytest.approx(expected_endpoint)
    assert endpoint[0] < descriptor.upper_bound
    assert modes[0] == DynamicLimitMode.FREE
    assert any(
        event["action"] == "project"
        and event["stage_or_endpoint"].startswith("stage_")
        for event in events
    )
    assert not any(
        event["action"] == "project"
        and event["stage_or_endpoint"] == "endpoint"
        for event in events
    )


@pytest.mark.parametrize("method", ["herk2", "herk4"])
def test_multiple_sexs_limits_are_enforced_together(data_dir, method):
    psys = load_psse(raw_filename=os.path.join(data_dir, "ieee9_v33.raw"))
    add_dyr(psys, os.path.join(data_dir, "ieee9bus_SEXS.dyr"))
    psys.createYbusComplex()
    psys.power_injection = False
    z0, theta, ctx = _initialize_context(psys)
    state_indices = []
    bounds = []
    expected_buses = set()
    for index, exciter in enumerate(psys.exc):
        state_index = exciter.dif_ptr + exciter.efd_idx
        initial_efd = float(z0[state_index])
        _set_sexs_limits(
            theta,
            exciter,
            initial_efd - 0.01,
            initial_efd + 0.01,
            vref_offset=0.05 if index % 2 == 0 else -0.05,
        )
        state_indices.append(state_index)
        bounds.append((initial_efd - 0.01, initial_efd + 0.01))
        expected_buses.add(psys.buses[exciter.bus].id)
    ctx.set_theta(theta.copy())

    result = integrate_system(
        psys,
        IntegrationConfig(
            method=method,
            petsc=False,
            steps=4,
            dt=0.005,
            ton=10.0,
            toff=11.0,
        ),
        ctx,
    )

    for state_index, (lower, upper) in zip(state_indices, bounds):
        values = result["history"][state_index]
        assert np.min(values) >= lower - 1e-12
        assert np.max(values) <= upper + 1e-12
    activation_buses = {
        event["bus"]
        for event in result["dynamic_limit_diagnostics"]["events"]
        if event["action"] == "activate"
    }
    assert activation_buses == expected_buses


@pytest.mark.parametrize("method", ["herk2", "herk4"])
def test_projected_stages_and_endpoint_remain_algebraically_consistent(
    data_dir, method, monkeypatch
):
    psys, ctx, state_index, lower, upper = _biased_two_bus_context(
        data_dir, "upper"
    )
    original_solve = herk.solve_stage_algebraic
    algebraic_norms = []
    stage_values = []

    def recording_solve(X_i, y0, v0, theta, psys, F, J, tol, max_iter):
        y, v, iterations = original_solve(
            X_i, y0, v0, theta, psys, F, J, tol, max_iter
        )
        check = np.concatenate([X_i, y, v])
        residual = np.zeros_like(check)
        residual_function(residual, check, theta, psys)
        algebraic_norms.append(
            np.linalg.norm(residual[psys.num_dof_dif:], np.inf)
        )
        stage_values.append(float(X_i[state_index]))
        return y, v, iterations

    monkeypatch.setattr(herk, "solve_stage_algebraic", recording_solve)
    result = integrate_system(
        psys,
        IntegrationConfig(
            method=method,
            petsc=False,
            steps=1,
            dt=0.005,
            ton=10.0,
            toff=11.0,
        ),
        ctx,
    )

    assert len(algebraic_norms) == len(herk.TABLEAUS[method][1]) + 1
    assert max(algebraic_norms) < 1e-8
    assert min(stage_values) >= lower - 1e-12
    assert max(stage_values) <= upper + 1e-12
    assert lower - 1e-12 <= result["history"][state_index, -1] <= upper + 1e-12


@pytest.mark.parametrize("method", ["herk2", "herk4"])
def test_sexs_no_fault_trajectory_stays_flat_without_limit_events(
    data_dir, method
):
    psys = _build_two_bus_sexs(data_dir)
    result = integrate_system(
        psys,
        IntegrationConfig(
            method=method,
            petsc=False,
            steps=50,
            dt=0.001,
            ton=10.0,
            toff=11.0,
        ),
    )

    differential = result["history"][:psys.num_dof_dif]
    assert np.max(np.abs(differential - differential[:, [0]])) < 1e-8
    assert result["dynamic_limit_diagnostics"]["events"] == []


@pytest.mark.parametrize("method", ["herk2", "herk4"])
def test_faulted_sexs_trajectory_is_limited_and_post_switch_consistent(
    data_dir, method
):
    psys = _build_two_bus_sexs(data_dir, fault=True)
    z0, theta, ctx = _initialize_context(psys)
    exciter = psys.exc[0]
    state_index = exciter.dif_ptr + exciter.efd_idx
    initial_efd = float(z0[state_index])
    lower = initial_efd - 0.001
    upper = initial_efd + 0.001
    _set_sexs_limits(theta, exciter, lower, upper)
    ctx.set_theta(theta.copy())
    result = integrate_system(
        psys,
        IntegrationConfig(
            method=method,
            petsc=False,
            steps=8,
            dt=0.005,
            ton=0.01,
            toff=0.02,
        ),
        ctx,
    )

    efd = result["history"][state_index]
    assert np.min(efd) >= lower - 1e-12
    assert np.max(efd) <= upper + 1e-12
    assert any(
        event["action"] == "activate"
        for event in result["dynamic_limit_diagnostics"]["events"]
    )
    fault = psys.fault_events[0]
    residual = np.zeros(result["history"].shape[0])
    fault.apply()
    on_index = int(np.flatnonzero(np.isclose(result["tvec"], 0.01))[0])
    residual_function(residual, result["history"][:, on_index], theta, psys)
    assert np.linalg.norm(residual[psys.num_dof_dif:], np.inf) < 1e-8
    fault.remove()
    off_index = int(np.flatnonzero(np.isclose(result["tvec"], 0.02))[0])
    residual_function(residual, result["history"][:, off_index], theta, psys)
    assert np.linalg.norm(residual[psys.num_dof_dif:], np.inf) < 1e-8
    assert fault.active is False


def _limited_final_differential(data_dir, method, dt):
    psys, ctx, _, _, _ = _biased_two_bus_context(data_dir, "upper")
    result = integrate_system(
        psys,
        IntegrationConfig(
            method=method,
            petsc=False,
            tend=0.05,
            steps=-1,
            dt=dt,
            ton=10.0,
            toff=11.0,
        ),
        ctx,
    )
    return result["history"][:psys.num_dof_dif, -1]


@pytest.mark.parametrize("method", ["herk2", "herk4"])
def test_limited_herk_trajectory_converges_as_dt_decreases(data_dir, method):
    reference = _limited_final_differential(data_dir, method, 0.0003125)
    coarse = _limited_final_differential(data_dir, method, 0.01)
    fine = _limited_final_differential(data_dir, method, 0.005)

    coarse_error = np.linalg.norm(coarse - reference, np.inf)
    fine_error = np.linalg.norm(fine - reference, np.inf)
    assert coarse_error > 0.0
    assert fine_error < coarse_error


def test_herk_helper_failure_has_runtime_stage_context(data_dir, monkeypatch):
    psys = _build_two_bus_sexs(data_dir)
    _, _, ctx = _initialize_context(psys)
    state_index = psys.exc[0].dif_ptr + psys.exc[0].efd_idx
    original_residual = herk.residual_function

    def non_finite_efd_derivative(F, z, theta, psys):
        original_residual(F, z, theta, psys)
        F[state_index] = np.nan

    monkeypatch.setattr(herk, "residual_function", non_finite_efd_derivative)

    with pytest.raises(DynamicLimitError) as exc_info:
        integrate_system(
            psys,
            IntegrationConfig(
                method="herk2",
                petsc=False,
                steps=1,
                dt=0.005,
                ton=10.0,
                toff=11.0,
            ),
            ctx,
        )

    diagnostics = exc_info.value.diagnostics
    assert diagnostics["phase"] == "runtime"
    assert diagnostics["method"] == "herk2"
    assert diagnostics["backend"] == "native"
    assert diagnostics["operation"] == "derivative_projection"
    assert diagnostics["failure_reasons"] == ["non_finite_derivative"]
    assert diagnostics["time"] == pytest.approx(0.0)
    assert diagnostics["stage_or_endpoint"] == "stage_1"
    json.dumps(diagnostics, allow_nan=False)


def test_herk_failure_after_fault_application_restores_fault(
    data_dir, monkeypatch
):
    psys = _build_two_bus_sexs(data_dir, fault=True)
    z0, theta, ctx = _initialize_context(psys)
    exciter = psys.exc[0]
    state_index = exciter.dif_ptr + exciter.efd_idx
    initial_efd = float(z0[state_index])
    _set_sexs_limits(
        theta,
        exciter,
        initial_efd - 0.01,
        initial_efd + 0.01,
        vref_offset=0.05,
    )
    ctx.set_theta(theta.copy())
    original_step = herk._herk_step_with_limits
    calls = 0

    def fail_second_interval(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise DynamicLimitError(
                {
                    "message": "forced HERK limiter failure",
                    "operation": "derivative_projection",
                    "failure_reasons": ["non_finite_derivative"],
                    "events": [],
                }
            )
        return original_step(*args, **kwargs)

    monkeypatch.setattr(herk, "_herk_step_with_limits", fail_second_interval)

    with pytest.raises(DynamicLimitError) as exc_info:
        integrate_system(
            psys,
            IntegrationConfig(
                method="herk2",
                petsc=False,
                steps=2,
                dt=0.005,
                ton=0.005,
                toff=0.02,
            ),
            ctx,
        )

    assert calls == 2
    assert exc_info.value.diagnostics["phase"] == "runtime"
    assert exc_info.value.diagnostics["events"]
    assert psys.fault_events[0].active is False


@pytest.mark.parametrize("method", ["herk2", "herk4"])
def test_disabled_dynamic_limits_skip_herk_projection(
    data_dir, method, monkeypatch
):
    psys = _build_two_bus_sexs(data_dir)
    monkeypatch.setattr(
        herk,
        "project_limited_states",
        lambda *args, **kwargs: pytest.fail("projection must remain disabled"),
    )

    result = integrate_system(
        psys,
        IntegrationConfig(
            method=method,
            petsc=False,
            steps=1,
            dt=0.001,
            ton=10.0,
            toff=11.0,
            enforce_dynamic_limits=False,
        ),
    )

    assert result["dynamic_limit_diagnostics"]["enabled"] is False
    assert result["dynamic_limit_diagnostics"]["events"] == []
