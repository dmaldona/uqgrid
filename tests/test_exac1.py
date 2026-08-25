import os

import numpy as np
import pytest

from uqgrid.io.parse import add_dyr, load_psse
from uqgrid.models import ExcEXAC1
from uqgrid.models.exac1_imp import exac1_jac, exac1_resdiff
from uqgrid.simulation.config import IntegrationConfig, IntegrationCtx
from uqgrid.simulation.dynamics import (
    initialize_system,
    integrate_system,
    preallocate_jacobian,
)
from uqgrid.simulation.jacobian import residual_jacobian
from uqgrid.simulation.pflow import runpf
from uqgrid.simulation.residual import residual_function


@pytest.fixture(params=[(1.1, 0.4), (0.0, 0.0)])
def exac1_case(tmp_path, request):
    tb, tc = request.param
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    dyr = tmp_path / "exac1.dyr"
    dyr.write_text(
        "1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 0.291 0.39 0.1733 0.0787 0.0 0.0 /\n"
        f"1 'EXAC1' 1 0.02 {tb} {tc} 100.0 0.05 8.0 -8.0 1.0 "
        "0.5 1.0 0.1 0.1 1.0 3.0 0.01 4.0 0.02 /\n"
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, str(dyr))
    psys.createYbusComplex()
    solution = runpf(psys, verbose=False)
    state, theta = initialize_system(psys, solution)
    return psys, state, theta


def test_exac1_parser_and_initialization(exac1_case):
    psys, state, theta = exac1_case
    exc = psys.exc[0]
    assert isinstance(exc, ExcEXAC1)
    assert exc.KA == pytest.approx(100.0)
    assert exc.KD == pytest.approx(0.1)
    assert exc.SE2 == pytest.approx(0.02)
    assert psys.gen_efd_ctrl_col[0] == psys.num_dof_dif + exc.alg_ptr

    residual = np.zeros_like(state)
    residual_function(residual, state, theta, psys)
    rows = list(range(exc.dif_ptr, exc.dif_ptr + exc.dif_dim))
    rows.append(psys.num_dof_dif + exc.alg_ptr)
    np.testing.assert_allclose(residual[rows], 0.0, atol=1e-9)


def test_exac1_analytical_jacobian_matches_finite_difference(exac1_case):
    psys, state, theta = exac1_case
    exc = psys.exc[0]
    state = state.copy()
    state[exc.dif_ptr + 2] += 0.02
    state[exc.dif_ptr + 3] += 0.03

    jacobian = preallocate_jacobian(psys)
    residual_jacobian(jacobian, state, theta, psys)
    analytical = jacobian.toarray()
    rows = list(range(exc.dif_ptr, exc.dif_ptr + exc.dif_dim))
    rows.append(psys.num_dof_dif + exc.alg_ptr)

    for column in range(state.size):
        step = 1e-6 * max(1.0, abs(state[column]))
        increased_state = state.copy()
        decreased_state = state.copy()
        increased_state[column] += step
        decreased_state[column] -= step
        increased = np.zeros_like(state)
        decreased = np.zeros_like(state)
        residual_function(increased, increased_state, theta, psys)
        residual_function(decreased, decreased_state, theta, psys)
        finite_difference = (increased[rows] - decreased[rows]) / (2.0 * step)
        np.testing.assert_allclose(
            analytical[rows, column], finite_difference,
            rtol=2e-4, atol=2e-6, err_msg=f"column {column}",
        )


def test_exac1_numba_kernels_compile(exac1_case):
    psys, state, theta = exac1_case
    jacobian = preallocate_jacobian(psys)
    residual = np.zeros_like(state)
    residual_function(residual, state, theta, psys)
    residual_jacobian(jacobian, state, theta, psys)
    assert exac1_resdiff.signatures
    assert exac1_jac.signatures


def test_exac1_unperturbed_time_steps_are_flat(exac1_case):
    psys, state, theta = exac1_case
    context = IntegrationCtx()
    context.set_initial_conditions(state)
    context.set_theta(theta)
    result = integrate_system(
        psys,
        IntegrationConfig(steps=3, dt=1.0 / 120.0, verbose=False, petsc=False),
        context,
    )
    history = result["history"]
    np.testing.assert_allclose(history[:, -1], history[:, 0], atol=2e-9)


def test_exac1_rejects_nonzero_lead_with_zero_lag():
    with pytest.raises(ValueError, match="TC=0"):
        ExcEXAC1(
            "1", None, 0.0, 0.0, 0.1, 100.0, 0.05, 8.0, -8.0,
            1.0, 0.5, 1.0, 0.1, 0.1, 1.0, 3.0, 0.01, 4.0, 0.02,
        )
