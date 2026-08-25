import os

import numpy as np
import pytest

from uqgrid.io.parse import add_dyr, load_psse
from uqgrid.models.exac2_imp import _rectifier, _signals, exac2_jac, exac2_resdiff
from uqgrid.simulation.dynamics import initialize_system, preallocate_jacobian
from uqgrid.simulation.jacobian import residual_jacobian
from uqgrid.simulation.pflow import runpf
from uqgrid.simulation.residual import residual_function


@pytest.fixture
def exac2_case(tmp_path):
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    dyr = tmp_path / "exac2.dyr"
    dyr.write_text(
        "1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 0.291 0.39 0.1733 0.0787 0.0 0.0 /\n"
        "1 'EXAC2' 1 0.02 1.1 0.4 100.0 0.05 100.0 -100.0 1.0 80.0 -80.0 "
        "1.3 4.0 0.5 0.1 1.0 0.1 0.1 1.0 0.0 3.0 0.01 4.0 0.02 /\n"
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, str(dyr))
    psys.createYbusComplex()
    solution = runpf(psys, verbose=False)
    state, theta = initialize_system(psys, solution)
    return psys, state, theta


def test_exac2_parser_and_initialization(exac2_case):
    psys, state, theta = exac2_case
    exc = psys.exc[0]
    assert exc.TB == pytest.approx(1.1)
    assert exc.TC == pytest.approx(0.4)
    assert exc.KC == pytest.approx(0.1)
    assert exc.SE2 == pytest.approx(0.02)
    assert psys.gen_efd_ctrl_col[0] == psys.num_dof_dif + exc.alg_ptr

    residual = np.zeros_like(state)
    residual_function(residual, state, theta, psys)
    rows = list(range(exc.dif_ptr, exc.dif_ptr + exc.dif_dim))
    rows.append(psys.num_dof_dif + exc.alg_ptr)
    assert np.linalg.norm(residual[rows], np.inf) < 1e-9


@pytest.mark.parametrize(
    "value, expected",
    [(-0.1, 1.0), (0.2, 1.0 - 0.577 * 0.2), (0.6, np.sqrt(0.75 - 0.36)),
     (0.9, 1.732 * 0.1), (1.1, 0.0)],
)
def test_exac2_rectifier_regions(value, expected):
    output, _ = _rectifier(value)
    assert output == pytest.approx(expected)


@pytest.mark.parametrize(
    "state_changes, vrmax, expected",
    [
        ({2: -50.0}, 80.0, (True, False)),
        ({2: 50.0}, 80.0, (False, False)),
        ({2: -50.0}, 1.0, (True, True)),
    ],
)
def test_exac2_selector_and_limit_branches(exac2_case, state_changes, vrmax, expected):
    psys, state, _ = exac2_case
    exc = psys.exc[0]
    state = state.copy()
    for offset, value in state_changes.items():
        state[exc.dif_ptr + offset] = value
    idxs = np.array(
        [exc.dif_ptr, psys.num_dof_dif + exc.alg_ptr, exc.par_ptr, exc.bus],
        dtype=np.int64,
    )
    v = state[psys.num_dof_dif + psys.num_dof_alg:]
    gen_dp, gen_ap, *machine = exc._generator_args()
    signals = _signals(
        state, v, idxs, exc.bus, psys.power_injection, 0.0, exc.vref,
        gen_dp, gen_ap, exc.TR, exc.TB, exc.TC, exc.KA, exc.VAMAX,
        exc.VAMIN, exc.KB, vrmax, -vrmax, exc.KL, exc.KH,
        exc.KF, exc.TF, exc.KC, exc.KD, exc.KE, exc.VLRx,
        exc.sat_a, exc.sat_b, *machine,
    )
    gate_high, clipped = signals[9], signals[10] == 0.0
    assert (gate_high, clipped) == expected


def test_exac2_zero_transducer_and_feedback_paths(exac2_case):
    psys, state, theta = exac2_case
    exc = psys.exc[0]
    exc.TR = 0.0
    exc.KF = 0.0
    state = state.copy()
    state[exc.dif_ptr] = 0.0
    state[exc.dif_ptr + 4] = 0.0
    exc.initialize_theta(theta)
    residual = np.zeros_like(state)
    residual_function(residual, state, theta, psys)
    assert residual[exc.dif_ptr] == 0.0
    assert residual[exc.dif_ptr + 4] == 0.0


def test_exac2_analytical_jacobian_matches_finite_difference(exac2_case):
    psys, state, theta = exac2_case
    exc = psys.exc[0]
    state = state.copy()
    state[exc.dif_ptr + 2] += 0.02
    state[exc.dif_ptr + 3] += 0.03

    jacobian = preallocate_jacobian(psys)
    residual_jacobian(jacobian, state, theta, psys)
    analytical = jacobian.toarray()
    baseline = np.zeros_like(state)
    residual_function(baseline, state, theta, psys)
    rows = list(range(exc.dif_ptr, exc.dif_ptr + exc.dif_dim))
    rows.append(psys.num_dof_dif + exc.alg_ptr)

    for column in range(state.size):
        step = 1e-6 * max(1.0, abs(state[column]))
        perturbed = state.copy()
        perturbed[column] += step
        increased = np.zeros_like(state)
        residual_function(increased, perturbed, theta, psys)
        perturbed[column] -= 2.0 * step
        decreased = np.zeros_like(state)
        residual_function(decreased, perturbed, theta, psys)
        finite_difference = (increased[rows] - decreased[rows]) / (2.0 * step)
        np.testing.assert_allclose(
            analytical[rows, column], finite_difference, rtol=2e-4, atol=2e-6,
            err_msg=f"column {column}",
        )


def test_exac2_numba_kernels_compile(exac2_case):
    psys, state, theta = exac2_case
    jacobian = preallocate_jacobian(psys)
    residual = np.zeros_like(state)
    residual_function(residual, state, theta, psys)
    residual_jacobian(jacobian, state, theta, psys)
    assert exac2_resdiff.signatures
    assert exac2_jac.signatures


def test_exac2_initialization_is_flat_when_source_vlr_controls(exac2_case):
    psys, state, theta = exac2_case
    exc = psys.exc[0]
    exc.VLR = 20.0
    exc.initialize(
        1.0, 0.0, 0.0, 0.0,
        state[:psys.num_dof_dif],
        state[psys.num_dof_dif:psys.num_dof_dif + psys.num_dof_alg],
        psys,
    )
    exc.initialize_theta(theta)
    assert exc.VLRx == exc.VLR
    residual = np.zeros_like(state)
    residual_function(residual, state, theta, psys)
    assert residual[exc.dif_ptr + 3] == pytest.approx(0.0, abs=1e-10)
