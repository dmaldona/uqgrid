# tests/test_pflow.py
# Refactored using pytest for improved testing workflow

import os
import pytest
import cmath
import numpy as np
import scipy.io as sio
from uqgrid.io.parse import load_matpower, load_psse
from uqgrid.simulation.pflow import runpf

EPS = 1e-8

@pytest.fixture
def data_dir():
    """Fixture to provide the absolute path to the data directory."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'
    )

@pytest.mark.parametrize("case, mat_file, sbus_file, volt_file", [
    ("case9", "case9.mat", "sbus_case9.mat", "volt_case9.mat"),
    ("case14", "case14.mat", "sbus_case14.mat", "volt_case14.mat"),
    ("case30", "case30.mat", "sbus_case30.mat", "volt_case30.mat"),
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
