import os

import numpy as np
import pytest

from uqgrid.io.parse import add_dyr, load_psse
from uqgrid.simulation.config import IntegrationConfig, IntegrationCtx
from uqgrid.simulation.dynamics import initialize_system, integrate_system, preallocate_jacobian
from uqgrid.simulation.dynamic_limits import collect_limited_state_descriptors
from uqgrid.simulation.jacobian import residual_jacobian
from uqgrid.simulation.jacobian_check import compare_jacobians
from uqgrid.simulation.pflow import runpf
from uqgrid.simulation.residual import residual_function


@pytest.fixture
def data_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
GOVERNOR_FIXTURES = {
    "TGOV1": ("2bus_TGOV1.dyr", 1, 2, 3, 7, 8),
    "GAST": ("2bus_GAST.dyr", 0, 6, 7, 9, 10),
    "HYGOV": ("2bus_HYGOV.dyr", 1, 6, 7, 13, 14),
    "IEEEG1": ("2bus_IEEEG1.dyr", 1, 29, 30, 31, 28),
}


@pytest.mark.parametrize("model", GOVERNOR_FIXTURES)
def test_parsed_governor_full_system_jacobian_matches_finite_difference(data_dir, model):
    dyr_name = GOVERNOR_FIXTURES[model][0]
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, os.path.join(data_dir, dyr_name))
    psys.createYbusComplex()
    solution = runpf(psys, verbose=False)
    state, theta = initialize_system(psys, solution)
    jacobian = preallocate_jacobian(psys)
    residual_jacobian(jacobian, state, theta, psys)

    assert compare_jacobians(
        psys, state, theta, jacobian, eps=1e-6, top_k=10, tol=1e-5
    ) == []


@pytest.mark.parametrize("model", GOVERNOR_FIXTURES)
@pytest.mark.parametrize(
    ("method", "petsc"),
    [
        pytest.param("beuler", False, id="native-be"),
        pytest.param("herk2", False, id="herk2"),
        pytest.param("herk4", False, id="herk4"),
        pytest.param("beuler", True, id="petsc-be"),
        pytest.param("cn", True, id="petsc-cn"),
    ],
)
def test_governor_position_limits_use_shared_integrator_path(
    data_dir, model, method, petsc
):
    if petsc:
        pytest.importorskip("petsc4py")
    dyr_name, state_offset, upper_offset, lower_offset, enabled_offset, pref_offset = (
        GOVERNOR_FIXTURES[model]
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, os.path.join(data_dir, dyr_name))
    psys.createYbusComplex()
    solution = runpf(psys, verbose=False)
    state, theta = initialize_system(psys, solution)
    governor = psys.gov[0]
    state_index = governor.dif_ptr + state_offset
    initial = state[state_index]
    width = 0.001 if model == "HYGOV" else 0.01
    theta[governor.par_ptr + lower_offset] = initial - width
    theta[governor.par_ptr + upper_offset] = initial + width
    theta[governor.par_ptr + enabled_offset] = 1.0
    theta[governor.par_ptr + pref_offset] += 0.2
    context = IntegrationCtx()
    context.set_initial_conditions(state.copy())
    context.set_theta(theta.copy())

    descriptors = collect_limited_state_descriptors(psys, theta)
    assert len(descriptors) == 1
    result = integrate_system(
        psys,
        IntegrationConfig(
            method=method,
            petsc=petsc,
            steps=20,
            dt=0.005,
            ton=1.0,
            toff=2.0,
        ),
        context,
    )

    values = result["history"][state_index]
    upper = initial + width
    assert np.max(values) <= upper + 1e-8
    assert any(
        event["action"] == "activate"
        and event["device_type"] == model
        and event["side"] == "upper"
        for event in result["dynamic_limit_diagnostics"]["events"]
    )
    if model in {"HYGOV", "IEEEG1"}:
        assert np.max(values) == pytest.approx(upper, abs=1e-8)
        assert values[-1] == pytest.approx(upper, abs=1e-8)


@pytest.mark.parametrize("model", ("HYGOV", "IEEEG1"))
@pytest.mark.parametrize(
    ("method", "petsc"),
    [
        pytest.param("beuler", False, id="native-be"),
        pytest.param("herk2", False, id="herk2"),
        pytest.param("herk4", False, id="herk4"),
        pytest.param("beuler", True, id="petsc-be"),
        pytest.param("cn", True, id="petsc-cn"),
    ],
)
def test_corrected_governors_activate_and_hold_lower_position_limit(
    data_dir, model, method, petsc
):
    if petsc:
        pytest.importorskip("petsc4py")
    dyr_name, state_offset, upper_offset, lower_offset, enabled_offset, pref_offset = (
        GOVERNOR_FIXTURES[model]
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, os.path.join(data_dir, dyr_name))
    psys.createYbusComplex()
    solution = runpf(psys, verbose=False)
    state, theta = initialize_system(psys, solution)
    governor = psys.gov[0]
    state_index = governor.dif_ptr + state_offset
    initial = state[state_index]
    width = 0.001 if model == "HYGOV" else 0.01
    lower = initial - width
    theta[governor.par_ptr + lower_offset] = lower
    theta[governor.par_ptr + upper_offset] = initial + width
    theta[governor.par_ptr + enabled_offset] = 1.0
    theta[governor.par_ptr + pref_offset] -= 0.2
    context = IntegrationCtx()
    context.set_initial_conditions(state.copy())
    context.set_theta(theta.copy())

    result = integrate_system(
        psys,
        IntegrationConfig(
            method=method,
            petsc=petsc,
            steps=20,
            dt=0.005,
            ton=1.0,
            toff=2.0,
        ),
        context,
    )

    values = result["history"][state_index]
    assert np.min(values) >= lower - 1e-8
    assert np.min(values) == pytest.approx(lower, abs=1e-8)
    assert values[-1] == pytest.approx(lower, abs=1e-8)
    assert any(
        event["action"] == "activate"
        and event["device_type"] == model
        and event["side"] == "lower"
        for event in result["dynamic_limit_diagnostics"]["events"]
    )
