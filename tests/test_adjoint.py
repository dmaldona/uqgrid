"""
Simple tests for adjoint sensitivity analysis validation.

Run with: pytest test_adjoint.py --adjoint-tests
Or skip with: pytest test_adjoint.py (skips by default due to speed)
"""

import pytest
pytest.importorskip("petsc4py", reason="PETSc required for adjoint tests")

import numpy as np
from uqgrid.core.psydef import Psystem
from uqgrid.simulation.dynamics import integrate_system, initialize_system
from uqgrid.io.parse import load_psse, add_dyr
from uqgrid.simulation.pflow import runpf
from uqgrid.simulation.config import IntegrationConfig, IntegrationCtx


def pytest_addoption(parser):
    parser.addoption(
        "--adjoint-tests", action="store_true", default=False,
        help="run slow adjoint validation tests"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "adjoint: marks tests as adjoint validation tests (slow)"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--adjoint-tests"):
        skip_adjoint = pytest.mark.skip(reason="need --adjoint-tests option to run")
        for item in items:
            if "adjoint" in item.keywords:
                item.add_marker(skip_adjoint)


@pytest.fixture(scope="module")
def test_system():
    """Setup IEEE 9-bus test system"""
    psys = load_psse(raw_filename="data/ieee9_v33.raw")
    add_dyr(psys, "data/ieee9bus.dyr")
    
    zfault = 0.01
    psys.add_busfault(1, zfault)
    psys.createYbusComplex()
    psys.set_load_parameters(np.zeros(psys.nloads))
    
    return psys


@pytest.fixture
def test_config():
    """Standard test configuration"""
    return IntegrationConfig(
        tend=2.0,
        dt=1.0/120.0,
        ton=0.1,
        toff=0.15,
        power_injection=False,
        verbose=False,
        comp_sens=True,
        enforce_dynamic_limits=False,
        petsc=True,
        method="cn",
    )


@pytest.mark.adjoint
def test_lambda_final_validation(test_system, test_config):
    """Test that λᵢ matches finite differences"""
    
    psys = test_system
    
    # Get baseline solution
    pf_solution = runpf(psys, verbose=False)
    z0_baseline, theta_baseline = initialize_system(psys, pf_solution)
    
    # Run adjoint simulation
    results = integrate_system(psys, test_config)
    lambda_petsc = results["lambda_final"]
    
    # Test a few components with finite differences
    eps = 1e-7
    sample_indices = [0, 4, len(z0_baseline)//2]  # Test first, middle, and a few others
    
    for idx in sample_indices:
        # Forward perturbation
        z0_plus = z0_baseline.copy()
        z0_plus[idx] += eps
        ctx_plus = IntegrationCtx()
        ctx_plus.set_initial_conditions(z0_plus)
        ctx_plus.set_theta(theta_baseline)
        results_plus = integrate_system(psys, test_config, ctx_plus)
        cost_plus = results_plus["adjoint_cost"]
        
        # Backward perturbation
        z0_minus = z0_baseline.copy()
        z0_minus[idx] -= eps
        ctx_minus = IntegrationCtx()
        ctx_minus.set_initial_conditions(z0_minus)
        ctx_minus.set_theta(theta_baseline)
        results_minus = integrate_system(psys, test_config, ctx_minus)
        cost_minus = results_minus["adjoint_cost"]
        
        # Finite difference
        lambda_fd = (cost_plus - cost_minus) / (2 * eps)
        
        # Check agreement
        if abs(lambda_petsc[idx]) > 1e-10:
            rel_error = abs(lambda_petsc[idx] - lambda_fd) / abs(lambda_petsc[idx])
            assert rel_error < 1e-3, f"λᵢ[{idx}] relative error {rel_error:.2e} too large"
        else:
            abs_error = abs(lambda_petsc[idx] - lambda_fd)
            assert abs_error < 1e-6, f"λᵢ[{idx}] absolute error {abs_error:.2e} too large"


@pytest.mark.adjoint  
def test_mu_trajectory_validation(test_system, test_config):
    """Test that μᵢ matches finite differences"""
    
    psys = test_system
    
    # Get baseline solution
    pf_solution = runpf(psys, verbose=False)
    z0_baseline, theta_baseline = initialize_system(psys, pf_solution)
    
    # Run adjoint simulation
    results = integrate_system(psys, test_config)
    mu_petsc = results["adjoint_gradient_trajectory"]
    
    # Find load parameter indices in theta
    load_indices = []
    for load in psys.loads:
        load_indices.extend([load.par_ptr, load.par_ptr + 1])  # P and Q
    
    # Test with finite differences
    eps = 1e-7
    
    for i, theta_idx in enumerate(load_indices):
        # Forward perturbation
        theta_plus = theta_baseline.copy()
        theta_plus[theta_idx] += eps
        ctx_plus = IntegrationCtx()
        ctx_plus.set_initial_conditions(z0_baseline)
        ctx_plus.set_theta(theta_plus)
        results_plus = integrate_system(psys, test_config, ctx_plus)
        cost_plus = results_plus["adjoint_cost"]
        
        # Backward perturbation
        theta_minus = theta_baseline.copy()
        theta_minus[theta_idx] -= eps
        ctx_minus = IntegrationCtx()
        ctx_minus.set_initial_conditions(z0_baseline)
        ctx_minus.set_theta(theta_minus)
        results_minus = integrate_system(psys, test_config, ctx_minus)
        cost_minus = results_minus["adjoint_cost"]
        
        # Finite difference
        mu_fd = (cost_plus - cost_minus) / (2 * eps)
        
        # Check agreement
        if abs(mu_petsc[i]) > 1e-10:
            rel_error = abs(mu_petsc[i] - mu_fd) / abs(mu_petsc[i])
            assert rel_error < 1e-3, f"μᵢ[{i}] relative error {rel_error:.2e} too large"
        else:
            abs_error = abs(mu_petsc[i] - mu_fd)
            assert abs_error < 1e-8, f"μᵢ[{i}] absolute error {abs_error:.2e} too large"


@pytest.mark.adjoint
def test_complete_gradient_validation(test_system, test_config):
    """Test that complete gradient matches finite differences"""
    
    psys = test_system
    
    # Run adjoint simulation
    results = integrate_system(psys, test_config)
    complete_grad_petsc = results["adjoint_gradient_complete"]
    
    # Get baseline load parameters
    p_loads_base, q_loads_base = psys.get_load_pq()
    baseline_params = np.zeros(2 * psys.nloads)
    baseline_params[::2] = p_loads_base
    baseline_params[1::2] = q_loads_base
    
    # Test with finite differences
    eps = 1e-7
    complete_grad_fd = np.zeros_like(baseline_params)
    
    for i in range(len(baseline_params)):
        # Create perturbed system - forward
        psys_plus = load_psse(raw_filename="data/ieee9_v33.raw")
        add_dyr(psys_plus, "data/ieee9bus.dyr")
        psys_plus.add_busfault(1, 0.01)
        psys_plus.createYbusComplex()
        psys_plus.set_load_parameters(np.zeros(psys_plus.nloads))
        
        pert_plus = baseline_params.copy()
        pert_plus[i] += eps
        p_plus = pert_plus[::2]
        q_plus = pert_plus[1::2]
        psys_plus.set_load_pq(p_plus, q_plus)
        
        results_plus = integrate_system(psys_plus, test_config)
        cost_plus = results_plus["adjoint_cost"]
        
        # Create perturbed system - backward
        psys_minus = load_psse(raw_filename="data/ieee9_v33.raw")
        add_dyr(psys_minus, "data/ieee9bus.dyr")
        psys_minus.add_busfault(1, 0.01)
        psys_minus.createYbusComplex()
        psys_minus.set_load_parameters(np.zeros(psys_minus.nloads))
        
        pert_minus = baseline_params.copy()
        pert_minus[i] -= eps
        p_minus = pert_minus[::2]
        q_minus = pert_minus[1::2]
        psys_minus.set_load_pq(p_minus, q_minus)
        
        results_minus = integrate_system(psys_minus, test_config)
        cost_minus = results_minus["adjoint_cost"]
        
        # Finite difference
        complete_grad_fd[i] = (cost_plus - cost_minus) / (2 * eps)
    
    # Check agreement
    for i in range(len(complete_grad_petsc)):
        if abs(complete_grad_petsc[i]) > 1e-10:
            rel_error = abs(complete_grad_petsc[i] - complete_grad_fd[i]) / abs(complete_grad_petsc[i])
            assert rel_error < 1e-3, f"Complete grad[{i}] relative error {rel_error:.2e} too large"
        else:
            abs_error = abs(complete_grad_petsc[i] - complete_grad_fd[i])
            assert abs_error < 1e-6, f"Complete grad[{i}] absolute error {abs_error:.2e} too large"


@pytest.mark.adjoint
def test_gradient_consistency(test_system, test_config):
    """Test that μᵢ + λᵢ(∂y₀/∂p) = complete gradient"""
    
    psys = test_system
    
    results = integrate_system(psys, test_config)
    
    mu_petsc = results["adjoint_gradient_trajectory"]
    lambda_dy0dp_petsc = results["adjoint_gradient_initial"] 
    complete_grad_petsc = results["adjoint_gradient_complete"]
    
    # Check that components sum correctly
    manual_sum = mu_petsc + lambda_dy0dp_petsc
    
    np.testing.assert_allclose(
        manual_sum, complete_grad_petsc,
        rtol=1e-12, atol=1e-15,
        err_msg="Gradient components don't sum to complete gradient"
    )


def test_adjoint_dependencies():
    """Quick test that adjoint dependencies are available"""
    import petsc4py  # noqa: F401
    from uqgrid.simulation.dynamics import integrate_system

    assert callable(integrate_system)


if __name__ == "__main__":
    # Run tests with adjoint flag
    import sys
    sys.exit(pytest.main([__file__, "--adjoint-tests", "-v"]))
