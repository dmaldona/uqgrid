from pathlib import Path

import numpy as np
import pytest

from uqgrid.io.parse import add_dyr, load_psse
from uqgrid.models.genrou_imp import sat_coefficients, sat_se
from uqgrid.simulation.dynamics import initialize_system, preallocate_jacobian
from uqgrid.simulation.jacobian import residual_jacobian
from uqgrid.simulation.jacobian_check import compare_jacobians
from uqgrid.simulation.pflow import runpf
from uqgrid.simulation.residual import residual_function


@pytest.fixture
def saturated_genrou_case(tmp_path):
    data_dir = Path(__file__).resolve().parents[1] / "data"
    dyr_path = tmp_path / "saturated_GENROU.dyr"
    dyr_path.write_text(
        """1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 0.291 0.39 0.1733 0.0787 0.115 0.627 /\n"""
    )
    psys = load_psse(raw_filename=str(data_dir / "2bus_33.raw"))
    add_dyr(psys, str(dyr_path))
    psys.createYbusComplex()
    state, theta = initialize_system(psys, runpf(psys, verbose=False))
    return psys, state, theta


def _q_axis_terms(gen, state, psys):
    dp = gen.dif_ptr
    ap = psys.num_dof_dif + gen.alg_ptr
    e_qp, e_dp, phi_1d, phi_2q = state[dp:dp + 4]
    i_q = state[ap + 2]

    psi_de = (
        (gen.x_ddp - gen.xl) / (gen.x_dp - gen.xl) * e_qp
        + (gen.x_dp - gen.x_ddp) / (gen.x_dp - gen.xl) * phi_1d
    )
    psi_qe = (
        -(gen.x_qdp - gen.xl) / (gen.x_qp - gen.xl) * e_dp
        + (gen.x_qp - gen.x_qdp) / (gen.x_qp - gen.xl) * phi_2q
    )
    sat_a, sat_b = sat_coefficients(gen.S1, gen.S2)
    se = sat_se(np.hypot(psi_de, psi_qe), sat_a, sat_b)
    gqd = (gen.x_q - gen.xl) / (gen.x_d - gen.xl)
    unsaturated = (
        -e_dp
        + (
            i_q
            - (gen.x_qp - gen.x_qdp)
            * (e_dp + i_q * (gen.x_qp - gen.xl) + phi_2q)
            / (gen.x_qp - gen.xl) ** 2
        )
        * (gen.x_q - gen.x_qp)
    )
    return unsaturated, se * psi_qe * gqd


def test_genrou_q_axis_saturation_is_present_in_initialization_and_runtime(
    saturated_genrou_case,
):
    psys, state, theta = saturated_genrou_case
    gen = psys.gendyn[0]
    residual = np.zeros_like(state)
    residual_function(residual, state, theta, psys)
    unsaturated, saturation = _q_axis_terms(gen, state, psys)

    assert abs(saturation) > 1e-4
    assert residual[gen.dif_ptr + 1] == pytest.approx(
        (unsaturated + saturation) / gen.T_q0p, abs=1e-10
    )
    assert np.linalg.norm(residual, np.inf) < 1e-8


def test_genrou_saturated_q_axis_analytical_jacobian_matches_finite_difference(
    saturated_genrou_case,
):
    psys, state, theta = saturated_genrou_case
    gen = psys.gendyn[0]
    jacobian = preallocate_jacobian(psys)
    residual_jacobian(jacobian, state, theta, psys)

    q_row = gen.dif_ptr + 1
    assert abs(jacobian[q_row, gen.dif_ptr]) > 1e-6
    assert abs(jacobian[q_row, gen.dif_ptr + 2]) > 1e-6
    assert compare_jacobians(
        psys, state, theta, jacobian, eps=1e-6, top_k=10, tol=1e-5
    ) == []
