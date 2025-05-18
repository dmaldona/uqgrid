# tests/test_sens.py

import os
import pytest
import numpy as np
from uqgrid.simulation.dynamics import integrate_system
from uqgrid.io.parse import load_psse, add_dyr
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


def test_forward_sensitivities(data_dir):
    """
    TEST CASE 001: Two-bus system sensitivities.
    
    This test evaluates first and second-order sensitivities of a two-bus system by perturbing load alphas and comparing finite differences with analytical sensitivities.
    """
    h = 1.0 / 120.0  # Integration step in seconds
    nsteps = 1000
    eps = 1e-4

    # Create system
    psys = load_psse(raw_filename=os.path.join(data_dir, "ieee9_v33.raw"))
    psys.add_busfault(1, 1.0, 0.1)
    psys.createYbusComplex()
    add_dyr(psys, os.path.join(data_dir, "ieee9bus.dyr"))

    alpha = 0.5
    alpha2 = 0.5 + eps
    alpha3 = 0.5 - eps

    var1 = 4
    var2 = 5
    var3 = 10

    # Define the integration configuration for nominal trajectory
    config_nominal = IntegrationConfig(
        tend=10.0,                 # Integration end time in seconds
        dt=h,                      # Time step in seconds
        steps=-1,                  # Number of integration steps (-1 for automatic)
        power_injection=True,     # Adjust based on your requirements
        verbose=False,             # Disable verbose output for testing
        comp_sens=True,            # Enable sensitivity computation
        fsolve=False,              # Disable fsolve for nonlinear equations
        petsc=False,               # Disable PETSc integration
        ton=0.25,                  # Fault activation time
        toff=0.4                   # Fault deactivation time
    )

    # Integrate nominal trajectory
    for load in psys.loads[:3]:
        load.set_alpha(alpha)
    res = integrate_system(psys, config=config_nominal)

    history = res["history"]
    tvec = res["tvec"]
    history_u = res["history_u"]
    history_v = res["history_v"]
    history_m = res["history_m"]

    config_perturbed = IntegrationConfig(
        tend=10.0,
        dt=h,
        steps=-1,
        power_injection=True,
        verbose=False,
        comp_sens=False,
        fsolve=False,
        petsc=False,
        ton=0.25,
        toff=0.4,
        solve_powerflow_dynamics=False,
    )

    # TEST FIRST-ORDER SENSITIVITIES
    for load_idx in range(3):
        psys.loads[load_idx].set_alpha(alpha2)
        res2 = integrate_system(psys, config=config_perturbed)
        history2 = res2["history"]

        fd = (history2[var1, :] - history[var1, :]) / eps
        analytic = history_u[var1, load_idx, :]
        assert np.allclose(fd, analytic), f'Load {load_idx} var1 sensitivity differs'

        fd = (history2[var2, :] - history[var2, :]) / eps
        analytic = history_u[var2, load_idx, :]
        assert np.allclose(fd, analytic, atol=1.e-5), f'Load {load_idx} var2 sensitivity differs'

        fd = (history2[var3, :] - history[var3, :]) / eps
        analytic = history_u[var3, load_idx, :]
        assert np.allclose(fd, analytic), f'Load {load_idx} var3 sensitivity differs'

        psys.loads[load_idx].set_alpha(alpha)

    # TEST SECOND-ORDER SENSITIVITIES
    for load_idx in range(3):
        for load in psys.loads[:3]:
            load.set_alpha(alpha)

        # Integrate perturbed trajectories
        psys.loads[load_idx].set_alpha(alpha2)
        res2 = integrate_system(psys, config=config_perturbed)
        history2 = res2["history"]

        # Integrate perturbed trajectories
        psys.loads[load_idx].set_alpha(alpha3)
        res3 = integrate_system(psys, config=config_perturbed)
        history3 = res3["history"]

        fd = (history2[var1, :] - 2 * history[var1, :] + history3[var1, :]) / (eps ** 2.0)
        analytic = history_v[var1, load_idx, :]
        assert np.allclose(fd, analytic, rtol=1.e-4, atol=1.e-5), f'Load {load_idx} var1 second-order sensitivity differs'

        fd = (history2[var2, :] - 2 * history[var2, :] + history3[var2, :]) / (eps ** 2.0)
        analytic = history_v[var2, load_idx, :]
        assert np.allclose(fd, analytic, rtol=1.e-4, atol=1.e-4), f'Load {load_idx} var2 second-order sensitivity differs'

        fd = (history2[var3, :] - 2 * history[var3, :] + history3[var3, :]) / (eps ** 2.0)
        analytic = history_v[var3, load_idx, :]
        assert np.allclose(fd, analytic, rtol=1.e-4, atol=1.e-5), f'Load {load_idx} var3 second-order sensitivity differs'

        for load in psys.loads[:3]:
            load.set_alpha(alpha)

    # TEST FIRST MIXED SENSITIVITIES

    # Integrate perturbed trajectories
    psys.loads[0].set_alpha(alpha2)
    psys.loads[1].set_alpha(alpha2)
    psys.loads[2].set_alpha(alpha)
    resa = integrate_system(psys, config=config_perturbed)
    hisA = resa["history"]

    # Integrate perturbed trajectories
    psys.loads[0].set_alpha(alpha2)
    psys.loads[1].set_alpha(alpha3)
    psys.loads[2].set_alpha(alpha)
    resb = integrate_system(psys, config=config_perturbed)
    hisB = resb["history"]

    # Integrate perturbed trajectories
    psys.loads[0].set_alpha(alpha3)
    psys.loads[1].set_alpha(alpha2)
    psys.loads[2].set_alpha(alpha)
    resc = integrate_system(psys, config=config_perturbed)
    hisC = resc["history"]

    # Integrate perturbed trajectories
    psys.loads[0].set_alpha(alpha3)
    psys.loads[1].set_alpha(alpha3)
    psys.loads[2].set_alpha(alpha)
    resd = integrate_system(psys, config=config_perturbed)
    hisD = resd["history"]

    fd_var1 = (hisA[var1, :] - hisB[var1, :] - hisC[var1, :] + hisD[var1, :]) / (4 * eps ** 2.0)
    analytic_var1 = history_m[var1, 0, :]
    assert np.allclose(fd_var1, analytic_var1), 'Var1 mixed sensitivities differ'

    fd_var2 = (hisA[var2, :] - hisB[var2, :] - hisC[var2, :] + hisD[var2, :]) / (4 * eps ** 2.0)
    analytic_var2 = history_m[var2, 0, :]
    assert np.allclose(fd_var2, analytic_var2, rtol=1.e-4, atol=1.e-4), 'Var2 mixed sensitivities differ'

    fd_var3 = (hisA[var3, :] - hisB[var3, :] - hisC[var3, :] + hisD[var3, :]) / (4 * eps ** 2.0)
    analytic_var3 = history_m[var3, 0, :]
    assert np.allclose(fd_var3, analytic_var3), 'Var3 mixed sensitivities differ'