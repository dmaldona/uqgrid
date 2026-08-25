import os

import numpy as np
import pytest

from uqgrid.io.parse import add_dyr, load_psse
from uqgrid.models.esst4b_imp import esst4b_rectifier
from uqgrid.simulation.dynamics import initialize_system, preallocate_jacobian
from uqgrid.simulation.jacobian import residual_jacobian
from uqgrid.simulation.pflow import runpf
from uqgrid.simulation.residual import residual_function


@pytest.fixture
def initialized_case(tmp_path):
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    dyr = tmp_path / "esst4b.dyr"
    dyr.write_text(
        "1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 0.291 0.39 0.1733 0.0787 0.0 0.0 /\n"
        "1 'ESST4B' 1 0.02 4.0 2.0 5.0 -5.0 0.05 1.0 1.0 2.0 -2.0 0.1 6.0 0.1 10.0 0.08 0.02 5.0 /\n"
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, str(dyr))
    psys.createYbusComplex()
    z, theta = initialize_system(psys, runpf(psys, verbose=False))
    return psys, z, theta


def test_parser_and_initialization(initialized_case):
    psys, z, theta = initialized_case
    exc = psys.exc[0]
    assert exc.KPR == pytest.approx(4.0)
    assert exc.THETAP == pytest.approx(5.0)
    assert psys.gen_efd_ctrl_col[0] == psys.num_dof_dif + exc.alg_ptr

    residual = np.zeros_like(z)
    residual_function(residual, z, theta, psys)
    rows = list(range(exc.dif_ptr, exc.dif_ptr + 4)) + [psys.num_dof_dif + exc.alg_ptr]
    assert np.linalg.norm(residual[rows], np.inf) < 1e-10


def test_zero_transducer_and_lag_time_constants_are_bypassed(tmp_path):
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    dyr = tmp_path / "esst4b_bypass.dyr"
    dyr.write_text(
        "1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 0.291 0.39 0.1733 0.0787 0.0 0.0 /\n"
        "1 'ESST4B' 1 0 3.9436 3.9436 1 -1 0 1 0 1 -1 0 6.8885 0 8.7286 0.08 0 0 /\n"
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, str(dyr))
    psys.createYbusComplex()
    z, theta = initialize_system(psys, runpf(psys, verbose=False))
    residual = np.zeros_like(z)
    residual_function(residual, z, theta, psys)
    exc = psys.exc[0]
    rows = list(range(exc.dif_ptr, exc.dif_ptr + 4)) + [psys.num_dof_dif + exc.alg_ptr]
    assert np.linalg.norm(residual[rows], np.inf) < 1e-10


def test_runtime_uses_theta_parameters(initialized_case):
    psys, z, theta = initialized_case
    exc = psys.exc[0]
    residual = np.zeros_like(z)
    theta = theta.copy()
    theta[exc.par_ptr + 17] += 0.01
    residual_function(residual, z, theta, psys)
    assert residual[exc.dif_ptr + 1] == pytest.approx(exc.KIR * 0.01)


@pytest.mark.parametrize(
    "current_ratio, expected, slope",
    [
        (-0.1, 1.0, 0.0),
        (0.2, 1.0 - 0.577 * 0.2, -0.577),
        (0.6, np.sqrt(0.75 - 0.6**2), -0.6 / np.sqrt(0.75 - 0.6**2)),
        (0.9, 1.732 * 0.1, -1.732),
        (1.1, 0.0, 0.0),
    ],
)
def test_rectifier_branches(current_ratio, expected, slope):
    value, derivative = esst4b_rectifier(current_ratio)
    assert value == pytest.approx(expected)
    assert derivative == pytest.approx(slope)


def test_analytical_jacobian_matches_centered_difference(initialized_case):
    psys, z, theta = initialized_case
    exc = psys.exc[0]
    z = z.copy()
    z[exc.dif_ptr] += 0.01
    z[exc.dif_ptr + 1] += 0.02
    z[exc.dif_ptr + 2] += 0.03
    z[exc.dif_ptr + 3] -= 0.01

    jacobian = preallocate_jacobian(psys)
    residual_jacobian(jacobian, z, theta, psys)
    rows = list(range(exc.dif_ptr, exc.dif_ptr + 4)) + [psys.num_dof_dif + exc.alg_ptr]
    columns = exc._columns(np.array([
        exc.dif_ptr, psys.num_dof_dif + exc.alg_ptr,
        psys.num_dof_dif + psys.num_dof_alg,
    ]))
    for column in columns:
        step = 1e-6 * max(1.0, abs(z[column]))
        zp, zm = z.copy(), z.copy()
        zp[column] += step
        zm[column] -= step
        fp, fm = np.zeros_like(z), np.zeros_like(z)
        residual_function(fp, zp, theta, psys)
        residual_function(fm, zm, theta, psys)
        numerical = (fp[rows] - fm[rows]) / (2.0 * step)
        assert jacobian[rows, column].toarray().ravel() == pytest.approx(
            numerical, rel=2e-5, abs=2e-7
        )


def test_initialization_adjusts_required_inner_upper_limit(initialized_case):
    psys, z, theta = initialized_case
    exc = psys.exc[0]
    exc.VMMAX_original = 0.1
    exc.VMMAX = 0.1

    exc.initialize(
        1.0, 0.0, 0.0, 0.0,
        z[:psys.num_dof_dif],
        z[psys.num_dof_dif:psys.num_dof_dif + psys.num_dof_alg],
        psys,
    )
    exc.initialize_theta(theta)

    assert exc.VMMAX > 0.1
    assert exc.limit_initialization_diagnostics["upper_bound_adjusted"]
