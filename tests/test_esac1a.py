import os

import numpy as np
import pytest

from uqgrid.io.parse import add_dyr, load_psse
from uqgrid.models import ExcESAC1A
from uqgrid.models.esac1a_imp import (
    _rectifier,
    _signals,
    esac1a_jac,
    esac1a_resdiff,
)
from uqgrid.simulation.config import IntegrationConfig, IntegrationCtx
from uqgrid.simulation.dynamics import (
    initialize_system,
    integrate_system,
    preallocate_jacobian,
)
from uqgrid.simulation.jacobian import residual_jacobian
from uqgrid.simulation.pflow import runpf
from uqgrid.simulation.residual import residual_function


@pytest.fixture
def esac1a_case(tmp_path):
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    dyr = tmp_path / "esac1a.dyr"
    dyr.write_text(
        "1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 0.291 0.39 0.1733 0.0787 0.0 0.0 /\n"
        "1 'ESAC1A' 1 0.02 1.0 0.2 100.0 0.05 20.0 -20.0 0.8 0.1 1.0 "
        "0.1 0.2 1.0 3.0 0.01 4.0 0.02 9.0 -9.0 /\n"
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, str(dyr))
    psys.createYbusComplex()
    solution = runpf(psys, verbose=False)
    state, theta = initialize_system(psys, solution)
    return psys, state, theta


def _signals_for(exc, psys, state):
    idxs = np.array(
        [exc.dif_ptr, psys.num_dof_dif + exc.alg_ptr, exc.par_ptr, exc.bus],
        dtype=np.int64,
    )
    v = state[psys.num_dof_dif + psys.num_dof_alg:]
    gen_dp, gen_ap, *machine = exc._generator_args()
    return _signals(
        state, v, idxs, exc.bus, psys.power_injection, 0.0, exc.vref,
        gen_dp, gen_ap, exc.TR, exc.TB, exc.TC, exc.KA,
        exc.TE, exc.VRMIN, exc.effective_vrmax,
        exc.KC, exc.KD, exc.KE, exc.KF, exc.TF,
        exc.sat_a, exc.sat_b, *machine,
    )


def test_esac1a_parser_and_initialization(esac1a_case):
    psys, state, theta = esac1a_case
    exc = psys.exc[0]
    assert isinstance(exc, ExcESAC1A)
    assert exc.KA == pytest.approx(100.0)
    assert exc.VAMAX == pytest.approx(20.0)
    assert exc.KC == pytest.approx(0.1)
    assert exc.VRMAX == pytest.approx(9.0)
    assert psys.gen_efd_ctrl_col[0] == psys.num_dof_dif + exc.alg_ptr

    residual = np.zeros_like(state)
    residual_function(residual, state, theta, psys)
    rows = list(range(exc.dif_ptr, exc.dif_ptr + exc.dif_dim))
    rows.append(psys.num_dof_dif + exc.alg_ptr)
    assert np.linalg.norm(residual[rows], np.inf) < 1e-9


@pytest.mark.parametrize(
    "value, expected",
    [(-0.1, 1.0), (0.2, 1.0 - 0.577 * 0.2),
     (0.6, np.sqrt(0.75 - 0.36)), (0.9, 1.732 * 0.1), (1.1, 0.0)],
)
def test_esac1a_rectifier_regions(value, expected):
    output, _ = _rectifier(value)
    assert output == pytest.approx(expected)


def test_esac1a_field_equation_uses_regulator_state(esac1a_case):
    psys, state, theta = esac1a_case
    exc = psys.exc[0]
    state = state.copy()
    state[exc.dif_ptr + 2] += 0.2
    residual = np.zeros_like(state)
    residual_function(residual, state, theta, psys)
    assert residual[exc.dif_ptr + 3] == pytest.approx(0.2 / exc.TE)


@pytest.mark.parametrize(
    "va, expected, expected_slope",
    [
        (10.0, 9.0, 0.0),
        (-10.0, -9.0, 0.0),
        (0.5, 0.5, 1.0),
        (9.0, 9.0, 0.0),
        (-9.0, -9.0, 0.0),
    ],
)
def test_esac1a_regulator_signal_is_clamped(
    esac1a_case, va, expected, expected_slope,
):
    psys, state, theta = esac1a_case
    exc = psys.exc[0]
    state = state.copy()
    state[exc.dif_ptr + 2] = va

    signals = _signals_for(exc, psys, state)
    assert signals[10] == pytest.approx(expected)
    assert signals[11] == pytest.approx(expected_slope)

    residual = np.zeros_like(state)
    residual_function(residual, state, theta, psys)
    assert residual[exc.dif_ptr + 3] == pytest.approx(
        (expected - signals[2]) / exc.TE
    )


def test_esac1a_rejects_initial_regulator_signal_outside_vr_limits(
    tmp_path,
):
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    dyr = tmp_path / "esac1a_narrow_vr.dyr"
    dyr.write_text(
        "1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 0.291 0.39 0.1733 0.0787 0.0 0.0 /\n"
        "1 'ESAC1A' 1 0.02 1.0 0.2 100.0 0.05 20.0 -20.0 0.8 0.1 1.0 "
        "0.1 0.2 1.0 3.0 0.01 4.0 0.02 0.1 -0.1 /\n"
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, str(dyr))
    psys.createYbusComplex()

    with pytest.raises(ValueError, match="effective VRMIN/VRMAX"):
        initialize_system(psys, runpf(psys, verbose=False))


def test_esac1a_saturation_and_rectifier_branches(esac1a_case):
    psys, state, _ = esac1a_case
    exc = psys.exc[0]
    state = state.copy()
    state[exc.dif_ptr + 3] = exc.sat_a - 0.1
    below = _signals_for(exc, psys, state)
    state[exc.dif_ptr + 3] = max(exc.sat_a + 0.2, 2.0)
    above = _signals_for(exc, psys, state)
    assert below[3] == pytest.approx(exc.KE)
    assert above[3] > exc.KE
    assert np.isfinite(below[9])
    assert np.isfinite(above[9])


@pytest.mark.parametrize("tr,tb", [(0.0, 1.0), (0.02, 0.0), (0.0, 0.0)])
def test_esac1a_zero_time_constant_branches(esac1a_case, tr, tb):
    psys, state, theta = esac1a_case
    exc = psys.exc[0]
    exc.TR = tr
    exc.TB = tb
    if tb == 0.0:
        exc.TC = 0.0
    state = state.copy()
    if tr == 0.0:
        state[exc.dif_ptr] = 0.0
    if tb == 0.0:
        state[exc.dif_ptr + 1] = 0.0
    exc.initialize_theta(theta)
    residual = np.zeros_like(state)
    residual_function(residual, state, theta, psys)
    if tr == 0.0:
        assert residual[exc.dif_ptr] == 0.0
    if tb == 0.0:
        assert residual[exc.dif_ptr + 1] == 0.0


@pytest.mark.parametrize(
    "va,tr,tb",
    [(0.5, 0.02, 1.0), (10.0, 0.02, 1.0), (-10.0, 0.02, 1.0),
     (0.5, 0.0, 1.0), (0.5, 0.02, 0.0)],
)
def test_esac1a_analytical_jacobian_matches_finite_difference(
    esac1a_case, va, tr, tb,
):
    psys, state, theta = esac1a_case
    exc = psys.exc[0]
    exc.TR = tr
    exc.TB = tb
    if tb == 0.0:
        exc.TC = 0.0
    state = state.copy()
    if tr == 0.0:
        state[exc.dif_ptr] = 0.0
    if tb == 0.0:
        state[exc.dif_ptr + 1] = 0.0
    state[exc.dif_ptr + 2] = va
    state[exc.dif_ptr + 3] += 0.03
    exc.initialize_theta(theta)

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
            analytical[rows, column], finite_difference, rtol=2e-4, atol=2e-6,
            err_msg=f"column {column}",
        )


def test_esac1a_numba_kernels_compile(esac1a_case):
    psys, state, theta = esac1a_case
    jacobian = preallocate_jacobian(psys)
    residual = np.zeros_like(state)
    residual_function(residual, state, theta, psys)
    residual_jacobian(jacobian, state, theta, psys)
    assert esac1a_resdiff.signatures
    assert esac1a_jac.signatures


@pytest.mark.parametrize(
    "method,petsc",
    [
        ("beuler", False),
        ("herk2", False),
        ("herk4", False),
        ("beuler", True),
        ("cn", True),
    ],
)
def test_esac1a_faulted_integration_is_finite(
    esac1a_case, method, petsc,
):
    psys, state, theta = esac1a_case
    if petsc:
        pytest.importorskip("petsc4py")
    exc = psys.exc[0]
    state = state.copy()
    theta = theta.copy()
    exc.effective_vrmax = state[exc.dif_ptr + 2] + 0.005
    state[exc.dif_ptr] -= 0.2
    exc.initialize_theta(theta)
    psys.add_busfault(exc.bus, 0.05)
    context = IntegrationCtx()
    context.set_initial_conditions(state)
    context.set_theta(theta)

    result = integrate_system(
        psys,
        IntegrationConfig(
            method=method,
            petsc=petsc,
            steps=4,
            dt=1.0 / 120.0,
            ton=1.0 / 120.0,
            toff=2.0 / 120.0,
        ),
        context,
    )

    assert np.all(np.isfinite(result["history"]))
    assert any(
        _signals_for(exc, psys, endpoint)[11] == 0.0
        for endpoint in result["history"].T
    )
