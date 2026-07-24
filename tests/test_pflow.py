# tests/test_pflow.py
# Refactored using pytest for improved testing workflow

import os
import pytest
import cmath
import numpy as np
import scipy.io as sio
from uqgrid.io.parse import load_matpower, load_psse
from uqgrid.core.psydef import Bus
from uqgrid.simulation.pflow import _project_reactive_dispatch, runpf

EPS = 1e-8

@pytest.fixture
def data_dir():
    """Fixture to provide the absolute path to the data directory."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'
    )

@pytest.mark.parametrize("case, mat_file, sbus_file, volt_file", [
    ("case9", "case9.m", "sbus_case9.mat", "volt_case9.mat"),
    ("case14", "case14.m", "sbus_case14.mat", "volt_case14.mat"),
    ("case30", "case30.m", "sbus_case30.mat", "volt_case30.mat"),
])
def test_power_flow_from_matpower(case, mat_file, sbus_file, volt_file, data_dir):
    """
    Test power flow for different MATPOWER cases by comparing computed results
    against expected values from .mat files.
    """
    print(f"\tTesting MATPOWER {case} power flow..")
    mat_file_path = os.path.join(data_dir, mat_file)
    sbus_file_path = os.path.join(data_dir, sbus_file)
    volt_file_path = os.path.join(data_dir, volt_file)

    # Load power system from MATPOWER file
    psys = load_matpower(mat_file=mat_file_path)
    psys.createYbusComplex()

    # Run power flow
    res = runpf(psys, verbose=True)
    v = res.v_vector
    Sinj = res.s_inj_vector

    # Load expected results
    Sbus_expected = sio.loadmat(sbus_file_path)['Sbus']
    Vbus_expected = sio.loadmat(volt_file_path)['V']

    # Compare power injections
    for i in range(len(Sbus_expected)):
        computed_S = Sinj[2*i] + 1j * Sinj[2*i + 1]
        residual = np.abs(Sbus_expected[i] - computed_S)
        assert residual < EPS, f'Bus ({i}) power injection differs by {residual}'

    # Compare bus voltages
    for i in range(len(Vbus_expected)):
        computed_V = cmath.rect(v[2*i], v[2*i + 1])
        residual = np.abs(computed_V - Vbus_expected[i])
        assert residual < EPS, f'Bus ({i}) voltage differs by {residual}'

def test_power_flow_from_psse(data_dir):
    """
    Test PSSE case9 power flow by comparing specific voltage angles against expected values.
    """
    print("\tTesting PSSE case9 power flow (modified)")
    raw_filename = os.path.join(data_dir, "ieee9_v33_mod1.raw")
    
    # Load power system from PSSE file
    psys = load_psse(raw_filename=raw_filename)
    psys.createYbusComplex()

    # Run power flow
    res = runpf(psys, verbose=True)
    v = res.v_vector
    Sinj = res.s_inj_vector

    # Expected voltage angles in degrees
    expected_vangles = {
        0: 0.00,
        1: -6.72,
        6: -12.33,
        8: -15.02
    }

    rad2ang = 180.0 / np.pi
    rtol = 1e-3

    for bus_idx, expected_angle in expected_vangles.items():
        computed_angle = v[2 * bus_idx + 1] * rad2ang
        assert np.isclose(computed_angle, expected_angle, rtol=rtol), \
            f'Bus ({bus_idx}) voltage angle differs: expected {expected_angle}, got {computed_angle}'


def test_reactive_dispatch_projection_respects_individual_limits():
    dispatch = _project_reactive_dispatch(
        total_q=0.8,
        lower=np.array([-0.2, 0.0]),
        upper=np.array([0.2, 1.0]),
    )

    np.testing.assert_allclose(dispatch, [0.2, 0.6])
    assert np.sum(dispatch) == pytest.approx(0.8)


def test_power_flow_q_limit_enforcement_can_be_disabled(data_dir):
    psys = load_matpower(os.path.join(data_dir, "case9.m"))
    psys.gens[1].qgub = 0.02
    psys.createYbusComplex()

    result = runpf(psys, verbose=False, enforce_q_limits=False)

    assert result.gen_qsch[1] > psys.gens[1].qgub
    assert result.bus_types[1] == Bus.PV
    assert not result.q_limit_enforced
    assert result.q_limit_events == []


def test_power_flow_qmax_active_set_switches_pv_bus_to_pq(data_dir):
    psys = load_matpower(os.path.join(data_dir, "case9.m"))
    psys.gens[1].qgub = 0.02
    original_voltage_setpoint = psys.buses[1].v0m
    psys.createYbusComplex()

    result = runpf(psys, verbose=False)

    assert result.gen_qsch[1] == pytest.approx(psys.gens[1].qgub)
    assert result.bus_types[1] == Bus.PQ
    assert result.v_magnitudes[1] < original_voltage_setpoint
    assert result.q_limit_enforced
    assert result.q_limit_iterations == 2
    assert result.q_limit_events[0]["side"] == "upper"
    assert psys.buses[1].type == Bus.PV
    assert psys.gens[1].qsch != pytest.approx(psys.gens[1].qgub)


def test_power_flow_qmin_active_set_switches_pv_bus_to_pq(data_dir):
    psys = load_matpower(os.path.join(data_dir, "case9.m"))
    psys.gens[2].qglb = -0.05
    psys.createYbusComplex()

    result = runpf(psys, verbose=False, enforce_q_limits=True)

    assert result.gen_qsch[2] == pytest.approx(psys.gens[2].qglb)
    assert result.bus_types[2] == Bus.PQ
    assert result.q_limit_events[0]["side"] == "lower"


def test_power_flow_projects_q_across_generators_before_switching_bus(data_dir):
    psys = load_matpower(os.path.join(data_dir, "case9.m"))
    psys.gens[1].qgub = 0.02
    psys.add_gen(
        bus=1,
        idx_name="2",
        psch=0.0,
        qsch=0.0,
        pgub=100.0,
        pglb=0.0,
        qgub=100.0,
        qglb=-100.0,
    )
    psys.createYbusComplex()

    result = runpf(psys, verbose=False, enforce_q_limits=True)

    assert result.bus_types[1] == Bus.PV
    assert result.q_limit_events == []
    assert result.gen_qsch[1] == pytest.approx(psys.gens[1].qgub)
    assert result.gen_qsch[-1] > result.gen_qsch[1]
    assert np.sum(result.gen_qsch[[1, -1]]) == pytest.approx(
        result.s_inj_vector[2*1 + 1]
    )
