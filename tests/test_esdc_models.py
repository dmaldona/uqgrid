import os
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from uqgrid.io.parse import add_dyr, load_psse
from uqgrid.models import ExcESDC1A, ExcESDC2A
from uqgrid.simulation import dynamics
from uqgrid.simulation.config import IntegrationConfig, IntegrationCtx
from uqgrid.simulation.dynamic_limits import (
    DynamicLimitError,
    DynamicLimitMode,
    LimitedStateDescriptor,
    collect_limited_state_descriptors,
    evaluate_limited_state_bounds,
    project_limited_states,
)
from uqgrid.simulation.dynamics import (
    initialize_system,
    integrate_system,
    preallocate_jacobian,
)
from uqgrid.simulation.jacobian import residual_jacobian
from uqgrid.simulation.pflow import runpf
from uqgrid.simulation.residual import residual_function


@pytest.fixture(params=["ESDC1A", "ESDC2A"])
def esdc_case(tmp_path, request):
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    model = request.param
    dyr = tmp_path / f"{model.lower()}.dyr"
    dyr.write_text(
        "1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 "
        "0.291 0.39 0.1733 0.0787 0.0 0.0 /\n"
        f"1 '{model}' 1 0.02 40.0 0.05 0.1 0.04 10.0 -10.0 "
        "1.0 0.9 0.05 1.0 0.0 2.3 0.12 3.1 0.36 /\n"
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, str(dyr))
    psys.createYbusComplex()
    solution = runpf(psys, verbose=False)
    state, theta = initialize_system(psys, solution)
    return psys, state, theta


def test_esdc_parser_initialization_and_descriptor(esdc_case):
    psys, state, theta = esdc_case
    exc = psys.exc[0]
    expected_class = ExcESDC2A if exc.device_type == "ESDC2A" else ExcESDC1A
    assert isinstance(exc, expected_class)
    assert exc.Tb == pytest.approx(0.1)
    assert exc.Tc == pytest.approx(0.04)
    assert exc.efd_idx == 3

    residual = np.zeros_like(state)
    residual_function(residual, state, theta, psys)
    rows = slice(exc.dif_ptr, exc.dif_ptr + exc.dif_dim)
    np.testing.assert_allclose(residual[rows], 0.0, atol=1e-9)

    descriptor = collect_limited_state_descriptors(psys, theta)[0]
    assert descriptor.state_index == exc.dif_ptr + 2
    assert descriptor.bound_scale == (
        "terminal_voltage" if exc.device_type == "ESDC2A" else None
    )


@pytest.mark.parametrize("model", ["ESDC1A", "ESDC2A"])
def test_esdc_parser_requires_exact_field_count(tmp_path, model):
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    dyr = tmp_path / "invalid_esdc.dyr"
    dyr.write_text(
        "1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 "
        "0.291 0.39 0.1733 0.0787 0.0 0.0 /\n"
        f"1 '{model}' 1 " + " ".join("1" for _ in range(15)) + " /\n"
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))

    with pytest.raises(ValueError, match=rf"{model}.*requires 16 parameters"):
        add_dyr(psys, str(dyr))


def test_esdc_analytical_jacobian_matches_finite_difference(esdc_case):
    psys, state, theta = esdc_case
    exc = psys.exc[0]
    state = state.copy()
    state[exc.dif_ptr + 3] += 0.03
    jacobian = preallocate_jacobian(psys)
    residual_jacobian(jacobian, state, theta, psys)
    analytical = jacobian.toarray()
    rows = list(range(exc.dif_ptr, exc.dif_ptr + exc.dif_dim))

    for column in range(state.size):
        step = 1e-6 * max(1.0, abs(state[column]))
        increased = state.copy()
        decreased = state.copy()
        increased[column] += step
        decreased[column] -= step
        f_increased = np.zeros_like(state)
        f_decreased = np.zeros_like(state)
        residual_function(f_increased, increased, theta, psys)
        residual_function(f_decreased, decreased, theta, psys)
        finite_difference = (f_increased[rows] - f_decreased[rows]) / (2.0 * step)
        np.testing.assert_allclose(
            analytical[rows, column], finite_difference,
            rtol=2e-4, atol=2e-6, err_msg=f"column {column}",
        )


def test_voltage_scaled_bounds_and_active_row_derivatives():
    descriptor = LimitedStateDescriptor(
        0, -2.0, 3.0, "ESDC2A", 1, "1", True,
        "terminal_voltage", (1, 2),
    )
    state = np.array([4.0, 0.8, 0.6])
    lower, upper, derivatives = evaluate_limited_state_bounds(state, descriptor)
    assert (lower, upper) == pytest.approx((-2.0, 3.0))
    assert tuple(item[0] for item in derivatives) == (1, 2)
    assert tuple(item[1] for item in derivatives) == pytest.approx((0.8, 0.6))
    projected, _ = project_limited_states(state, [descriptor])
    assert projected[0] == pytest.approx(3.0)

    residual = np.zeros(3)
    jacobian = csr_matrix(np.ones((3, 3)))
    dynamics._apply_beuler_active_rows(
        residual, jacobian, state, [descriptor],
        {0: DynamicLimitMode.UPPER_ACTIVE},
    )
    assert residual[0] == pytest.approx(1.0)
    np.testing.assert_allclose(jacobian.toarray()[0], [1.0, -2.4, -1.8])


def test_native_be_solves_active_voltage_scaled_constraint(esdc_case):
    psys, state, theta = esdc_case
    exc = psys.exc[0]
    if exc.device_type != "ESDC2A":
        pytest.skip("voltage-scaled constraint is specific to ESDC2A")
    descriptor = collect_limited_state_descriptors(psys, theta)[0]
    modes = {descriptor.state_index: DynamicLimitMode.UPPER_ACTIVE}
    residual = np.zeros_like(state)
    jacobian = preallocate_jacobian(psys)
    endpoint, _, _, converged = dynamics._solve_beuler_fixed_active_set(
        state,
        theta,
        0.005,
        psys,
        residual,
        jacobian,
        descriptors=[descriptor],
        modes=modes,
        newton_tol=1e-10,
        newton_max_iter=20,
    )
    _, upper, _ = evaluate_limited_state_bounds(endpoint, descriptor)
    assert converged
    assert endpoint[descriptor.state_index] == pytest.approx(upper, abs=1e-10)


@pytest.mark.parametrize("method", ["beuler", "herk2"])
def test_esdc_unperturbed_native_steps_are_flat(esdc_case, method):
    psys, state, theta = esdc_case
    context = IntegrationCtx()
    context.set_initial_conditions(state)
    context.set_theta(theta)
    result = integrate_system(
        psys,
        IntegrationConfig(
            method=method, petsc=False, steps=2, dt=1.0 / 120.0,
            ton=10.0, toff=11.0,
        ),
        context,
    )
    np.testing.assert_allclose(result["history"][:, -1], state, atol=2e-8)


@pytest.mark.parametrize("method", ["beuler", "herk2"])
def test_esdc2_fault_samples_respect_voltage_scaled_bounds(esdc_case, method):
    psys, state, theta = esdc_case
    exc = psys.exc[0]
    if exc.device_type != "ESDC2A":
        pytest.skip("voltage-scaled constraint is specific to ESDC2A")
    psys.add_busfault(exc.bus, 0.05)
    context = IntegrationCtx()
    context.set_initial_conditions(state)
    context.set_theta(theta)

    result = integrate_system(
        psys,
        IntegrationConfig(
            method=method, petsc=False, steps=4, dt=1.0 / 120.0,
            ton=1.0 / 120.0, toff=2.0 / 120.0,
        ),
        context,
    )

    history = result["history"]
    voltage_offset = psys.num_dof_dif + psys.num_dof_alg + 2 * exc.bus
    voltage = np.hypot(history[voltage_offset], history[voltage_offset + 1])
    regulator = history[exc.dif_ptr + 2]
    assert np.all(regulator >= exc.Vrmin * voltage - 1e-8)
    assert np.all(regulator <= exc.effective_vrmax * voltage + 1e-8)


def test_native_topology_projection_honors_configured_iteration_limit(
    monkeypatch,
):
    descriptor = LimitedStateDescriptor(
        0,
        -2.0,
        3.0,
        "ESDC2A",
        1,
        "1",
        True,
        "terminal_voltage",
        (1, 2),
    )
    state = np.array([4.0, 0.8, 0.6])
    modes = {0: DynamicLimitMode.FREE}
    solve_count = 0

    def restore_violation(projected, *args, **kwargs):
        nonlocal solve_count
        solve_count += 1
        endpoint = projected.copy()
        endpoint[0] = 4.0
        return endpoint, None, None, None

    monkeypatch.setattr(dynamics, "integrate", restore_violation)

    with pytest.raises(DynamicLimitError) as exc_info:
        dynamics._resolve_limited_topology_change(
            state,
            np.array([]),
            object(),
            None,
            None,
            [descriptor],
            modes,
            time=0.1,
            stage_or_endpoint="fault_on",
            newton_tol=1e-10,
            newton_max_iter=20,
            max_dynamic_limit_iterations=3,
            verbose=False,
        )

    diagnostics = exc_info.value.diagnostics
    assert solve_count == 4
    assert diagnostics["failure_reasons"] == ["projection_iteration_limit"]
    assert diagnostics["active_set_iterations"] == 3
    assert len(diagnostics["events"]) == 3


def test_herk_topology_projection_honors_configured_iteration_limit(
    monkeypatch,
):
    from uqgrid.simulation import herk

    psys = type("PsystemShape", (), {"num_dof_dif": 1})()
    event = {
        "device_type": "ESDC2A",
        "device_id": "1",
        "bus": 1,
        "state_index": 0,
        "side": "upper",
        "action": "project",
        "time": 0.1,
        "stage_or_endpoint": "fault_on",
    }
    monkeypatch.setattr(
        herk,
        "solve_stage_algebraic",
        lambda x, y, v, *args, **kwargs: (y, v, 0),
    )
    monkeypatch.setattr(
        herk,
        "project_limited_states",
        lambda state, *args, **kwargs: (state.copy(), [event.copy()]),
    )

    with pytest.raises(DynamicLimitError) as exc_info:
        herk._solve_algebraics_with_dynamic_bounds(
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0, 0.0]),
            np.array([]),
            psys,
            None,
            None,
            1e-10,
            20,
            [],
            time=0.1,
            stage="fault_on",
            max_dynamic_limit_iterations=2,
        )

    diagnostics = exc_info.value.diagnostics
    assert diagnostics["failure_reasons"] == ["projection_iteration_limit"]
    assert diagnostics["active_set_iterations"] == 2
    assert len(diagnostics["events"]) == 2


def test_herk_topology_without_moving_bounds_solves_algebraics_once(
    monkeypatch,
):
    from uqgrid.simulation import herk

    psys = type("PsystemShape", (), {"num_dof_dif": 1})()
    solve_count = 0

    def solve_once(x, y, v, *args, **kwargs):
        nonlocal solve_count
        solve_count += 1
        return y, v, 0

    monkeypatch.setattr(herk, "solve_stage_algebraic", solve_once)

    _, _, _, endpoint, events = herk._solve_algebraics_with_dynamic_bounds(
        np.array([0.0]),
        np.array([0.0]),
        np.array([1.0, 0.0]),
        np.array([]),
        psys,
        None,
        None,
        1e-10,
        20,
        [],
        time=0.1,
        stage="fault_on",
        max_dynamic_limit_iterations=2,
    )

    assert solve_count == 1
    np.testing.assert_array_equal(endpoint, [0.0, 0.0, 1.0, 0.0])
    assert events == []


def test_native_topology_algebraic_failure_is_structured(monkeypatch):
    def fail_newton(*args, **kwargs):
        raise NameError("N-R solver did not converge")

    monkeypatch.setattr(dynamics, "integrate", fail_newton)

    with pytest.raises(DynamicLimitError) as exc_info:
        dynamics._resolve_limited_topology_change(
            np.array([0.0]),
            np.array([]),
            object(),
            None,
            None,
            [],
            {},
            time=0.1,
            stage_or_endpoint="fault_on",
            newton_tol=1e-10,
            newton_max_iter=2,
            max_dynamic_limit_iterations=2,
            verbose=False,
        )

    diagnostics = exc_info.value.diagnostics
    assert diagnostics["failure_reasons"] == ["algebraic_nonconvergence"]
    assert diagnostics["operation"] == "topology_change_algebraic_solve"
    assert diagnostics["time"] == pytest.approx(0.1)
    assert diagnostics["stage_or_endpoint"] == "fault_on"


def test_herk_topology_algebraic_failure_is_structured(monkeypatch):
    from uqgrid.simulation import herk

    def fail_newton(*args, **kwargs):
        raise RuntimeError("algebraic Newton did not converge")

    monkeypatch.setattr(herk, "solve_stage_algebraic", fail_newton)

    with pytest.raises(DynamicLimitError) as exc_info:
        herk._solve_algebraics_with_dynamic_bounds(
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0, 0.0]),
            np.array([]),
            object(),
            None,
            None,
            1e-10,
            2,
            [],
            time=0.1,
            stage="fault_off",
            max_dynamic_limit_iterations=2,
        )

    diagnostics = exc_info.value.diagnostics
    assert diagnostics["failure_reasons"] == ["algebraic_nonconvergence"]
    assert diagnostics["operation"] == "topology_change_algebraic_solve"
    assert diagnostics["time"] == pytest.approx(0.1)
    assert diagnostics["stage_or_endpoint"] == "fault_off"


@pytest.mark.parametrize(
    "method,petsc",
    [
        ("beuler", False),
        ("herk2", False),
    ],
)
def test_topology_projection_failure_has_common_runtime_context(
    esdc_case, method, petsc, monkeypatch,
):
    from uqgrid.simulation import herk

    psys, state, theta = esdc_case
    exc = psys.exc[0]
    if exc.device_type != "ESDC2A":
        pytest.skip("voltage-scaled constraint is specific to ESDC2A")
    psys.add_busfault(exc.bus, 0.05)
    context = IntegrationCtx()
    context.set_initial_conditions(state)
    context.set_theta(theta)
    local_event = {
        "device_type": "ESDC2A",
        "device_id": "1",
        "bus": 1,
        "state_index": exc.dif_ptr + 2,
        "side": "upper",
        "action": "project",
        "time": 1.0 / 120.0,
        "stage_or_endpoint": "fault_on",
    }
    fault_limit_calls = []

    if method.startswith("herk"):
        original = herk._solve_algebraics_with_dynamic_bounds

        def fail_fault_projection(*args, **kwargs):
            if kwargs["stage"] != "fault_on":
                return original(*args, **kwargs)
            fault_limit_calls.append(kwargs["max_dynamic_limit_iterations"])
            raise DynamicLimitError(
                {
                    "message": "forced topology projection failure",
                    "operation": "topology_change_projection",
                    "time": float(kwargs["time"]),
                    "stage_or_endpoint": kwargs["stage"],
                    "failure_reasons": ["projection_iteration_limit"],
                    "events": [local_event],
                }
            )

        monkeypatch.setattr(
            herk, "_solve_algebraics_with_dynamic_bounds", fail_fault_projection
        )
    else:
        original = dynamics._resolve_limited_topology_change

        def fail_fault_projection(*args, **kwargs):
            if kwargs["stage_or_endpoint"] != "fault_on":
                return original(*args, **kwargs)
            fault_limit_calls.append(kwargs["max_dynamic_limit_iterations"])
            raise DynamicLimitError(
                {
                    "message": "forced topology projection failure",
                    "operation": "topology_change_projection",
                    "time": float(kwargs["time"]),
                    "stage_or_endpoint": kwargs["stage_or_endpoint"],
                    "failure_reasons": ["projection_iteration_limit"],
                    "events": [local_event],
                }
            )

        monkeypatch.setattr(
            dynamics,
            "_resolve_limited_topology_change",
            fail_fault_projection,
        )

    with pytest.raises(DynamicLimitError) as exc_info:
        integrate_system(
            psys,
            IntegrationConfig(
                method=method,
                petsc=petsc,
                steps=2,
                dt=1.0 / 120.0,
                ton=1.0 / 120.0,
                toff=2.0 / 120.0,
                max_dynamic_limit_iterations=3,
            ),
            context,
        )

    diagnostics = exc_info.value.diagnostics
    assert fault_limit_calls == [3]
    assert diagnostics["phase"] == "runtime"
    assert diagnostics["method"] == method
    assert diagnostics["backend"] == ("petsc" if petsc else "native")
    assert diagnostics["time"] == pytest.approx(1.0 / 120.0)
    assert diagnostics["stage_or_endpoint"] == "fault_on"
    assert diagnostics["failure_reasons"] == ["projection_iteration_limit"]
    assert diagnostics["events"].count(local_event) == 1
    assert psys.fault_events[0].active is False


@pytest.mark.parametrize("method", ["beuler", "cn"])
def test_petsc_topology_projection_failure_has_common_runtime_context(
    method, monkeypatch,
):
    descriptor = LimitedStateDescriptor(
        0,
        -1.0,
        1.0,
        "ESDC2A",
        1,
        "1",
        True,
        "terminal_voltage",
        (1, 2),
    )
    local_event = {
        "device_type": "ESDC2A",
        "device_id": "1",
        "bus": 1,
        "state_index": 0,
        "side": "upper",
        "action": "project",
        "time": 0.1,
        "stage_or_endpoint": "fault_on",
    }
    destroyed = []

    class Workspace:
        def __init__(self, *args, **kwargs):
            pass

        def destroy(self):
            destroyed.append(True)

    class Fault:
        active = False

        def apply(self):
            self.active = True

        def remove(self):
            self.active = False

    fault = Fault()
    config = SimpleNamespace(
        newton_tol=1e-10,
        newton_max_iter=20,
        dynamic_limit_tolerance=1e-8,
        dynamic_limit_release_tolerance=1e-10,
        max_dynamic_limit_iterations=4,
        verbose=False,
    )
    psys = SimpleNamespace(num_dof_dif=1)
    schedule = SimpleNamespace(fault_on_index=1, fault_off_index=None)
    diagnostics = {"events": [{"action": "prior"}]}

    def fail_projection(*args, **kwargs):
        assert kwargs["max_dynamic_limit_iterations"] == 4
        raise DynamicLimitError(
            {
                "message": "forced topology projection failure",
                "operation": "topology_change_projection",
                "time": float(kwargs["time"]),
                "stage_or_endpoint": kwargs["stage_or_endpoint"],
                "failure_reasons": ["projection_iteration_limit"],
                "events": [local_event],
            }
        )

    monkeypatch.setattr(
        dynamics,
        "_resolve_limited_topology_change",
        fail_projection,
    )

    with pytest.raises(DynamicLimitError) as exc_info:
        dynamics._integrate_system_petsc_implicit_limits(
            None,
            psys,
            config,
            np.array([]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.1]),
            schedule,
            fault,
            np.zeros(3),
            None,
            [descriptor],
            diagnostics,
            workspace_class=Workspace,
            interval_stepper=lambda z, *args, **kwargs: (
                z.copy(),
                {0: DynamicLimitMode.FREE},
                [],
            ),
            method=method,
        )

    error = exc_info.value.diagnostics
    assert error["phase"] == "runtime"
    assert error["method"] == method
    assert error["backend"] == "petsc"
    assert error["time"] == pytest.approx(0.1)
    assert error["stage_or_endpoint"] == "fault_on"
    assert error["events"] == [{"action": "prior"}, local_event]
    assert fault.active is False
    assert destroyed == [True]
