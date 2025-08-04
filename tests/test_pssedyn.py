import os
import pytest
import numpy as np
from uqgrid.core.psydef import Psystem
from uqgrid.models import GenGENROU
from uqgrid.simulation.dynamics import integrate_system
from uqgrid.io.parse import load_psse, add_dyr
from uqgrid.simulation.pflow import runpf
from uqgrid.simulation.config import IntegrationConfig

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
    adds generator dynamics, integrates the system using IntegrationConfig,
    and compares the simulation results against expected PSSE data.
    """
    zfault = 1.0  # Fault impedance (p.u.)
    fault_time = 1.0  # Fault initiation time in seconds

    # Define the integration configuration using IntegrationConfig
    config = IntegrationConfig(
        tend=10.0,                 # Integration end time in seconds
        dt=1.0 / 120.0,            # Time step in seconds
        steps=-1,                # Number of integration steps (-1 for automatic)
        power_injection=True,     # Adjust based on your requirements
        verbose=False,             # Disable verbose output for testing
        comp_sens=False,           # Disable sensitivity computation
        fsolve=False,              # Disable fsolve for nonlinear equations
        petsc=False,               # Disable PETSc integration
        zfault=1.0,                # Fault impedance (p.u.)
        ton=0.25,                   # Fault activation time
        toff=0.4                    # Fault deactivation time
        # Include other necessary fields if added to IntegrationConfig
    )

    # Load power system from PSSE raw file
    psys = load_psse(raw_filename=os.path.join(data_dir, "2bus_33.raw"))
    psys.createYbusComplex()
    psys.add_busfault(1, zfault, fault_time)

    # Add generator dynamics
    psys.add_gen_dynamics(
        psys.gens[0],
        GenGENROU(
            0, 1.575, 1.512, 0.291, 0.39, 0.1733,
            0.0787, 3.38, 0.0, 6.1, 1.0, 0.05, 0.15
        )
    )

    # Integrate the system using IntegrationConfig
    res = integrate_system(psys, config=config)

    # Load expected PSSE data
    psse = np.loadtxt(os.path.join(data_dir, '2bus_GENROU.csv'), delimiter=',')

    # Retrieve PSSE values. Delete erroneous or switching event indices
    indices_to_remove = [0, 1, 33, 52]
    time_p = np.delete(psse[0, :], indices_to_remove)
    volt1_p = np.delete(psse[1, :], indices_to_remove)
    volt2_p = np.delete(psse[3, :], indices_to_remove)
    eq_p = np.delete(psse[5, :], indices_to_remove)
    speed = np.delete(psse[9, :], indices_to_remove)

    # Extract simulation history
    history = res["history"]

    # Calculate relative errors
    error_volt1 = np.linalg.norm(np.abs((volt1_p - history[12, :]) / history[12, :]))
    error_volt2 = np.linalg.norm(np.abs((volt2_p - history[14, :]) / history[14, :]))
    error_eqp = np.linalg.norm(np.abs((eq_p - history[0, :]) / history[0, :]))
    error_speed = np.linalg.norm(np.abs((speed - history[4, :]) / (history[4, :] + 1)))

    # Assertions to verify simulation accuracy
    assert error_volt1 < 0.01, 'Voltage 1 trajectory differs'
    assert error_volt2 < 0.01, 'Voltage 2 trajectory differs'
    assert error_eqp < 0.01, 'E\'q trajectory differs'
    assert error_speed < 0.01, 'Speed trajectory differs'