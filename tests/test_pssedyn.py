# tests/test_pssedyn.py

import os
import pytest
import numpy as np
import scipy.io as sio
from uqgrid.core.psydef import Psystem, GenGENROU, ExcESDC1A, GovIEESGO, MotCIM5
from uqgrid.simulation.dynamics import integrate_system
from uqgrid.io.parse import load_psse, add_dyr
from uqgrid.simulation.pflow import runpf

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


def test_two_bus_system(data_dir):
    """
    TEST CASE 001: Two-bus system validation.
    
    This test loads a two-bus system from a PSSE raw file, applies a fault,
    adds generator dynamics, integrates the system, and compares the simulation
    results against expected PSSE data.
    """
    zfault = 1.0

    h = 1.0 / 120.0  # Integration step in seconds
    nsteps = 2000

    # Load power system from PSSE raw file
    psys = load_psse(raw_filename=os.path.join(data_dir, "2bus_33.raw"))
    psys.createYbusComplex()
    psys.add_busfault(1, zfault, 1.0)

    # Add generator dynamics
    psys.add_gen_dynamics(
        psys.gens[0],
        GenGENROU(
            0, 1.575, 1.512, 0.291, 0.39, 0.1733,
            0.0787, 3.38, 0.0, 6.1, 1.0, 0.05, 0.15
        )
    )

    # Integrate the system
    res = integrate_system(psys, comp_sens=False, tend=10.0)

    # Load expected PSSE data
    psse = np.loadtxt(os.path.join(data_dir, '2bus_GENROU.csv'), delimiter=',')

    # Retrieve PSSE values. Delete negative time steps and switching events
    time_p = np.delete(psse[0, :], [0, 1, 33, 52])
    volt1_p = np.delete(psse[1, :], [0, 1, 33, 52])
    volt2_p = np.delete(psse[3, :], [0, 1, 33, 52])
    eq_p = np.delete(psse[5, :], [0, 1, 33, 52])
    speed = np.delete(psse[9, :], [0, 1, 33, 52])

    history = res["history"]

    # Calculate errors
    error_volt1 = np.linalg.norm(np.abs((volt1_p - history[10, :]) / history[10, :]))
    error_volt2 = np.linalg.norm(np.abs((volt2_p - history[12, :]) / history[12, :]))
    error_eqp = np.linalg.norm(np.abs((eq_p - history[0, :]) / history[0, :]))
    error_speed = np.linalg.norm(np.abs((speed - history[4, :]) / (history[4, :] + 1)))

    # Assertions
    assert error_volt1 < 0.01, 'Voltage 1 trajectory differs'
    assert error_volt2 < 0.01, 'Voltage 2 trajectory differs'
    assert error_eqp < 0.01, 'E\'q trajectory differs'
    assert error_speed < 0.01, 'Speed trajectory differs'