import os

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from uqgrid.io.parse import add_dyr, load_psse
from uqgrid.models import ExcESDC1A, ExcESDC2A
from uqgrid.simulation import dynamics
from uqgrid.simulation.config import IntegrationConfig, IntegrationCtx
from uqgrid.simulation.dynamic_limits import (
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
