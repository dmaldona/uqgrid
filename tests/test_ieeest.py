import os

import numpy as np
import pytest

from uqgrid.io.parse import add_dyr, load_psse
from uqgrid.models import PssIEEEST
from uqgrid.models.ieeest_imp import ieeest_jac, ieeest_resdiff
from uqgrid.simulation.dynamics import initialize_system, preallocate_jacobian
from uqgrid.simulation.config import IntegrationConfig, IntegrationCtx
from uqgrid.simulation.dynamics import integrate_system
from uqgrid.simulation.jacobian import residual_jacobian
from uqgrid.simulation.jacobian_check import compare_jacobians
from uqgrid.simulation.pflow import runpf
from uqgrid.simulation.residual import residual_function


@pytest.fixture
def data_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _write_case(path, pss_record):
    path.write_text(
        "1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 0.291 0.39 0.1733 0.0787 0.0 0.0 /\n"
        "1 'SEXS' 1 0.1 10 100 0.05 -4 5 /\n"
        + pss_record + "\n"
    )


def _initialized_case(data_dir, tmp_path, record=None):
    if record is None:
        record = "1 'IEEEST' 1 1 0 1.013 0.013 0 0 1.013 0.113 10 0.02 0 0 1.65 1.65 3 0.1 -0.1 0 0 /"
    dyr = tmp_path / "ieeest.dyr"
    _write_case(dyr, record)
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, str(dyr))
    psys.createYbusComplex()
    state, theta = initialize_system(psys, runpf(psys, verbose=False))
    return psys, state, theta


def test_ieeest_parser_initialization_attachment_and_numba(data_dir, tmp_path):
    psys, state, theta = _initialized_case(data_dir, tmp_path)
    pss = psys.pss[0]

    assert isinstance(pss, PssIEEEST)
    assert pss.generator is psys.gendyn[0]
    assert pss.exciter is psys.exc[0]
    assert pss.exciter.pss_input_idx == psys.num_dof_dif + pss.alg_ptr
    residual = np.zeros_like(state)
    residual_function(residual, state, theta, psys)
    assert np.linalg.norm(residual, np.inf) < 1e-8

    jacobian = preallocate_jacobian(psys)
    residual_jacobian(jacobian, state, theta, psys)
    assert compare_jacobians(psys, state, theta, jacobian, eps=1e-6, tol=1e-5) == []
    assert ieeest_resdiff.signatures
    assert ieeest_jac.signatures


def test_ieeest_no_disturbance_trajectory_is_flat(data_dir, tmp_path):
    psys, state, theta = _initialized_case(data_dir, tmp_path)
    result = integrate_system(
        psys,
        IntegrationConfig(
            steps=3, dt=1.0 / 120.0, ton=10.0, toff=11.0,
            solve_powerflow_dynamics=False,
        ),
    )
    assert np.max(np.abs(result["history"] - result["history"][:, [0]])) < 1e-9


def test_ieeest_t5_zero_suppresses_output_but_not_washout_state(
    data_dir, tmp_path,
):
    record = (
        "1 'IEEEST' 1 1 0 1.013 0.013 0 0 1.013 0.113 10 0.02 "
        "0 0 0 1.65 3 0.1 -0.1 0 0 /"
    )
    psys, state, theta = _initialized_case(data_dir, tmp_path, record)
    pss = psys.pss[0]
    state = state.copy()
    state[pss.dif_ptr + 6] = 1.0
    output = psys.num_dof_dif + pss.alg_ptr

    residual = np.zeros_like(state)
    residual_function(residual, state, theta, psys)
    assert residual[pss.dif_ptr + 6] != 0.0
    assert residual[output] == pytest.approx(-state[output])

    jacobian = preallocate_jacobian(psys)
    residual_jacobian(jacobian, state, theta, psys)
    assert jacobian[output, pss.dif_ptr + 6] == 0.0
    assert jacobian[output, pss.w_idx] == 0.0

    context = IntegrationCtx()
    context.set_initial_conditions(state)
    context.set_theta(theta)
    result = integrate_system(
        psys,
        IntegrationConfig(
            steps=3,
            dt=1.0 / 120.0,
            ton=10.0,
            toff=11.0,
            solve_powerflow_dynamics=False,
        ),
        context,
    )
    np.testing.assert_allclose(result["history"][output], 0.0, atol=1e-12)


def test_ieeest_output_clamp_and_voltage_gate(data_dir, tmp_path):
    psys, state, theta = _initialized_case(data_dir, tmp_path)
    pss = psys.pss[0]
    output = psys.num_dof_dif + pss.alg_ptr
    residual = np.zeros_like(state)

    state[pss.dif_ptr + 6] = -1.0
    residual_function(residual, state, theta, psys)
    assert residual[output] == pytest.approx(0.1)

    state[pss.dif_ptr + 6] = 1.0
    residual_function(residual, state, theta, psys)
    assert residual[output] == pytest.approx(-0.1)

    network = psys.num_dof_dif + psys.num_dof_alg + 2 * pss.bus
    state[network:network + 2] = (1.0, 0.0)
    theta[pss.par_ptr + 15] = 1.0
    residual_function(residual, state, theta, psys)
    assert residual[output] == pytest.approx(0.0)

    theta[pss.par_ptr + 15] = 999.0
    theta[pss.par_ptr + 16] = 1.0
    residual_function(residual, state, theta, psys)
    assert residual[output] == pytest.approx(0.0)


def test_ieeest_clamped_and_voltage_gated_jacobian_matches_finite_difference(
    data_dir, tmp_path,
):
    psys, state, theta = _initialized_case(data_dir, tmp_path)
    pss = psys.pss[0]
    state[pss.dif_ptr + 6] = -1.0

    for gated in (False, True):
        theta[pss.par_ptr + 15] = 1.0 if gated else 999.0
        jacobian = preallocate_jacobian(psys)
        residual_jacobian(jacobian, state, theta, psys)
        assert compare_jacobians(
            psys, state, theta, jacobian, eps=1e-6, tol=1e-5
        ) == []


@pytest.mark.parametrize("mode", [0, 2, 3, 4, 5, 6])
def test_ieeest_rejects_unsupported_modes(data_dir, tmp_path, mode):
    record = f"1 'IEEEST' 1 {mode} 0 1.013 0.013 0 0 1.013 0.113 10 0.02 0 0 1.65 1.65 3 0.1 -0.1 0 0 /"
    dyr = tmp_path / f"ieeest_mode_{mode}.dyr"
    _write_case(dyr, record)
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    with pytest.raises(ValueError, match="unsupported"):
        add_dyr(psys, str(dyr))


def test_ieeest_rejects_remote_bus(data_dir, tmp_path):
    record = "1 'IEEEST' 1 1 2 1.013 0.013 0 0 1.013 0.113 10 0.02 0 0 1.65 1.65 3 0.1 -0.1 0 0 /"
    dyr = tmp_path / "ieeest_remote.dyr"
    _write_case(dyr, record)
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    with pytest.raises(ValueError, match="remote-bus"):
        add_dyr(psys, str(dyr))


def test_stabilizer_attachment_rejects_mismatched_generator_id(data_dir, tmp_path):
    psys, _, _ = _initialized_case(data_dir, tmp_path)
    pss = PssIEEEST(
        "2", 1, 0, 1.013, 0.013, 0, 0, 1.013, 0.113,
        10, 0.02, 0, 0, 1.65, 1.65, 3, 0.1, -0.1, 0, 0,
    )
    psys.gendyn[0].stabilizer = None

    with pytest.raises(ValueError, match="IDs must match"):
        psys.add_pss(psys.gendyn[0], pss)


def test_ieeest_attaches_when_record_precedes_exciter(data_dir, tmp_path):
    dyr = tmp_path / "ieeest_before_exciter.dyr"
    dyr.write_text(
        "1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 0.291 0.39 0.1733 0.0787 0.0 0.0 /\n"
        "1 'IEEEST' 1 1 0 1.013 0.013 0 0 1.013 0.113 10 0.02 0 0 1.65 1.65 3 0.1 -0.1 0 0 /\n"
        "1 'SEXS' 1 0.1 10 100 0.05 -4 5 /\n"
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, str(dyr))

    assert len(psys.pss) == 1
    assert psys.pss[0].generator is psys.gendyn[0]
    assert psys.pss[0].exciter is psys.exc[0]


def test_ieeest_without_eventual_exciter_raises(data_dir, tmp_path):
    dyr = tmp_path / "ieeest_without_exciter.dyr"
    dyr.write_text(
        "1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 0.291 0.39 0.1733 0.0787 0.0 0.0 /\n"
        "1 'IEEEST' 1 1 0 1.013 0.013 0 0 1.013 0.113 10 0.02 0 0 1.65 1.65 3 0.1 -0.1 0 0 /\n"
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))

    with pytest.raises(ValueError, match="requires an attached exciter"):
        add_dyr(psys, str(dyr))


def test_ieeest_without_dynamic_generator_warns_and_skips(
    data_dir, tmp_path, caplog,
):
    dyr = tmp_path / "ieeest_without_generator.dyr"
    dyr.write_text(
        "1 'IEEEST' 1 1 0 1.013 0.013 0 0 1.013 0.113 10 0.02 0 0 1.65 1.65 3 0.1 -0.1 0 0 /\n"
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, str(dyr))

    assert psys.pss == []
    assert "Cannot pair IEEEST" in caplog.text


def test_deferred_duplicate_ieeest_is_rejected(data_dir, tmp_path):
    record = (
        "1 'IEEEST' 1 1 0 1.013 0.013 0 0 1.013 0.113 10 0.02 "
        "0 0 1.65 1.65 3 0.1 -0.1 0 0 /\n"
    )
    dyr = tmp_path / "duplicate_ieeest.dyr"
    dyr.write_text(
        "1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 0.291 0.39 0.1733 0.0787 0.0 0.0 /\n"
        + record
        + "1 'SEXS' 1 0.1 10 100 0.05 -4 5 /\n"
        + record
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))

    with pytest.raises(ValueError, match="already has a stabilizer"):
        add_dyr(psys, str(dyr))
