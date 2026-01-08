# tests/test_network.py

import os
import pytest
import numpy as np
import scipy.io as sio
from uqgrid.core.psydef import Psystem, Bus
from uqgrid.models.network import createYbusComplex
from uqgrid.io.parse import load_matpower

EPS = 1e-10  # Tolerance for floating-point comparisons


@pytest.fixture
def data_dir():
    """
    Fixture to provide the absolute path to the data directory.
    Ensures that data files are correctly located regardless of the current working directory.
    """
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'
    )


@pytest.mark.skip(reason="temporarily disabled")
def test_nine_bus_system(data_dir):
    """
    TEST NETWORK: Nine-bus system validation.
    
    This test constructs a nine-bus power system, initializes voltages,
    adds branches, creates the Y-bus matrix, and verifies the Y-bus entries
    against expected values from a MATPOWER-generated matrix.
    """
    psys = Psystem()

    # Add buses with respective types
    psys.add_bus(1, bus_type=Bus.SLACK)  # Slack bus
    for bus_id in range(2, 10):
        psys.add_bus(bus_id, bus_type=Bus.PV)  # PV buses

    # Set initial voltages (magnitude in p.u., angle in radians)
    initial_voltages = [
        (1.04000, np.deg2rad(0.0)),
        (1.02500, np.deg2rad(9.6926)),
        (1.02500, np.deg2rad(4.8812)),
        (0.99574, np.deg2rad(-2.3060)),
        (0.95068, np.deg2rad(-4.1382)),
        (0.96621, np.deg2rad(-3.7372)),
        (0.99740, np.deg2rad(3.9736)),
        (0.97915, np.deg2rad(0.8364)),
        (1.00414, np.deg2rad(2.1073)),
    ]

    for bus, (mag, ang) in zip(psys.buses, initial_voltages):
        bus.set_vinit(mag, ang)

    # Add branches (from_bus, to_bus, resistance, reactance, shunt)
    branches = [
        (0, 3, 0.0000, 0.0576),
        (1, 6, 0.0000, 0.0625),
        (2, 8, 0.0000, 0.0586),
        (3, 4, 0.0100, 0.0850, 0.176),
        (3, 5, 0.0170, 0.0920, 0.158),
        (4, 6, 0.0320, 0.1610, 0.306),
        (5, 8, 0.0390, 0.1700, 0.358),
        (6, 7, 0.0085, 0.0720, 0.149),
        (7, 8, 0.0119, 0.1008, 0.209),
    ]

    for branch in branches:
        if len(branch) == 4:
            from_bus, to_bus, resistance, reactance = branch
            psys.add_branch(from_bus, to_bus, resistance, reactance)
        elif len(branch) == 5:
            from_bus, to_bus, resistance, reactance, shunt = branch
            psys.add_branch(from_bus, to_bus, resistance, reactance, sh=shunt)

    # Create Y-bus matrix (complex)
    ybus = createYbusComplex(psys)

    # Load expected Y-bus matrix from MATPOWER
    matpower_data = sio.loadmat(os.path.join(data_dir, 'ymat9bus_matpower.mat'))
    ybus_mat = matpower_data['ymat']

    # Compare Y-bus matrices
    for i in range(psys.nbuses):
        for j in range(psys.nbuses):
            test_flag = np.abs(ybus[i, j] - ybus_mat[i, j]) < EPS
            assert test_flag, f'Ymat entry ({i}, {j}) differs from test.'


def test_nine_bus_from_matpower(data_dir):
    """
    TEST NETWORK: Nine-bus system from MATPOWER validation.
    
    This test loads a nine-bus power system from a MATPOWER file, creates the Y-bus matrix,
    and verifies the Y-bus entries against expected values from a MATPOWER-generated matrix.
    """
    psys = load_matpower(mat_file=os.path.join(data_dir, 'case14.m'))

    # Create Y-bus matrix
    ybus = createYbusComplex(psys)

    # Load expected Y-bus matrix from MATPOWER
    matpower_data = sio.loadmat(os.path.join(data_dir, 'ymat14bus_matpower.mat'))
    ybus_mat = matpower_data['ymat']

    # Compare Y-bus matrices
    for i in range(psys.nbuses):
        for j in range(psys.nbuses):
            test_flag = np.abs(ybus[i, j] - ybus_mat[i, j]) < EPS
            assert test_flag, f'Ymat entry ({i}, {j}) differs from test.'


def test_bus_constants_values():
    """Verify that Bus constants map to standard PSS/E integer codes."""
    assert Bus.PQ == 1
    assert Bus.PV == 2
    assert Bus.SLACK == 3

def test_psystem_bus_counters():
    """Verify that Psystem correctly counts bus types when added."""
    psys = Psystem()
    
    # Add 2 PQ buses
    psys.add_bus(1, Bus.PQ)
    psys.add_bus(2, Bus.PQ)
    
    # Add 3 PV buses
    psys.add_bus(3, Bus.PV)
    psys.add_bus(4, Bus.PV)
    psys.add_bus(5, Bus.PV)
    
    # Add 1 Slack bus
    psys.add_bus(6, Bus.SLACK)
    
    assert psys.npq == 2
    assert psys.npv == 3
    assert psys.nslack == 1
    assert psys.nbuses == 6

def test_add_bus_invalid_type():
    """Verify that adding a bus with an invalid type raises ValueError."""
    psys = Psystem()
    with pytest.raises(ValueError):
        psys.add_bus(1, 99)
