"""
Numerical verification of adjoint gradient computation.

This compares the adjoint gradient with finite difference gradients.
"""

import numpy as np
import copy
from uqgrid.simulation.dynamics import integrate_system
from uqgrid.simulation.config import IntegrationConfig
from uqgrid.simulation.pflow import runpf


def evaluate_cost_function(psys_template, params, config):
    """
    Evaluate the cost function J(p) = ∫₀ᵀ Σᵢ ωᵢ²(t) dt
    
    Args:
        psys_template: Template power system
        params: Load parameters [P0, Q0, P1, Q1, ...]
        config: Integration configuration
        
    Returns:
        float: Cost function value
    """
    # Deep copy to avoid modifying original
    psys = copy.deepcopy(psys_template)
    
    # Apply parameters
    p_loads = params[::2]   # Even indices: P values
    q_loads = params[1::2]  # Odd indices: Q values
    psys.set_load_pq(p_loads, q_loads)
    
    # Run integration without adjoint (faster)
    config_eval = IntegrationConfig(
        tend=config.tend,
        dt=config.dt,
        comp_sens=False,  # No adjoint needed for evaluation
        petsc=True,      # Use regular integrator
        verbose=False,
        power_injection=False,
        solve_powerflow_dynamics=True,
        ton=config.ton,
        toff=config.toff
    )
    
    try:
        results = integrate_system(psys, config_eval)
        
        # Compute cost: integral of generator speed deviations squared
        history = results["history"]
        tvec = results["tvec"]
        dt = tvec[1] - tvec[0] if len(tvec) > 1 else config.dt
        
        # Get generator speed indices
        speed_indices = psys.genspeed_idx_set()
        
        # Integrate ∫ Σᵢ ωᵢ²(t) dt using trapezoidal rule
        cost = 0.0
        for k in range(len(tvec)):
            speed_sum = sum(history[idx, k]**2 for idx in speed_indices)
            if k == 0 or k == len(tvec) - 1:
                cost += 0.5 * dt * speed_sum  # Endpoints get half weight
            else:
                cost += dt * speed_sum
        
        return cost
        
    except Exception as e:
        print(f"Error evaluating cost at params {params}: {e}")
        return np.inf


def compute_finite_difference_gradient(psys_template, nominal_params, config, epsilon=1e-5):
    """
    Compute gradient using centered finite differences.
    
    Args:
        psys_template: Template power system
        nominal_params: Nominal parameter values [P0, Q0, P1, Q1, ...]
        config: Integration configuration
        epsilon: Finite difference step size
        
    Returns:
        numpy.ndarray: Finite difference gradient
    """
    n_params = len(nominal_params)
    fd_gradient = np.zeros(n_params)
    
    print(f"Computing finite difference gradient with {n_params} parameters...")
    print(f"Nominal cost evaluation...")
    
    # Baseline cost (optional, for debugging)
    cost_nominal = evaluate_cost_function(psys_template, nominal_params, config)
    print(f"Nominal cost: {cost_nominal:.6e}")
    
    # Centered differences: (f(x+h) - f(x-h)) / (2h)
    for i in range(n_params):
        if i % 2 == 0:
            param_name = f"P{i//2}"
        else:
            param_name = f"Q{i//2}"
            
        print(f"  Computing ∂J/∂{param_name} ({i+1}/{n_params})...")
        
        # Forward perturbation
        params_plus = nominal_params.copy()
        params_plus[i] += epsilon
        cost_plus = evaluate_cost_function(psys_template, params_plus, config)
        
        # Backward perturbation
        params_minus = nominal_params.copy()
        params_minus[i] -= epsilon
        cost_minus = evaluate_cost_function(psys_template, params_minus, config)
        
        # Centered difference
        fd_gradient[i] = (cost_plus - cost_minus) / (2 * epsilon)
        
        print(f"    J(+ε) = {cost_plus:.6e}, J(-ε) = {cost_minus:.6e}")
        print(f"    ∂J/∂{param_name} ≈ {fd_gradient[i]:.6e}")
    
    return fd_gradient


def test_adjoint_gradient_numerical(psys, config_template, epsilon_fd=1e-5, verbose=True):
    """
    Test adjoint gradient against finite differences.
    
    Args:
        psys: Power system object
        config_template: Base integration configuration
        epsilon_fd: Finite difference step size
        verbose: Print detailed results
        
    Returns:
        dict: Test results and comparison metrics
    """
    print("="*60)
    print("NUMERICAL VERIFICATION OF ADJOINT GRADIENT")
    print("="*60)
    
    # Get nominal parameters
    p_loads, q_loads = psys.get_load_pq()
    nominal_params = np.zeros(2 * psys.nloads)
    nominal_params[::2] = p_loads
    nominal_params[1::2] = q_loads
    
    print(f"Power system: {psys.nbuses} buses, {psys.nloads} loads")
    print(f"Parameter vector length: {len(nominal_params)}")
    print(f"Nominal parameters: {nominal_params}")
    print()
    
    # 1. Compute adjoint gradient
    print("1. Computing adjoint gradient...")
    config_adjoint = IntegrationConfig(
        tend=config_template.tend,
        dt=config_template.dt,
        comp_sens=True,
        petsc=True,
        use_load_pq_params=True,  # Use new parameter type
        verbose=False,
        power_injection=config_template.power_injection,
        solve_powerflow_dynamics=config_template.solve_powerflow_dynamics,
        ton=config_template.ton,
        toff=config_template.toff
    )
    
    results_adjoint = integrate_system(psys, config_adjoint)
    
    if "adjoint_gradient_complete" in results_adjoint:
        adjoint_gradient = results_adjoint["adjoint_gradient_complete"]
        adjoint_cost = results_adjoint["adjoint_cost"]
        print(f"   Adjoint cost: {adjoint_cost:.6e}")
        print(f"   Adjoint gradient norm: {np.linalg.norm(adjoint_gradient):.6e}")
        if verbose:
            print(f"   Adjoint gradient: {adjoint_gradient}")
    else:
        print("   ERROR: No complete adjoint gradient found!")
        return {"success": False, "error": "No adjoint gradient computed"}
    
    print()
    
    # 2. Compute finite difference gradient
    print("2. Computing finite difference gradient...")
    fd_gradient = compute_finite_difference_gradient(
        psys, nominal_params, config_template, epsilon_fd
    )
    print(f"   FD gradient norm: {np.linalg.norm(fd_gradient):.6e}")
    if verbose:
        print(f"   FD gradient: {fd_gradient}")
    print()
    
    # 3. Compare gradients
    print("3. Comparison:")
    print("-" * 40)
    
    # Absolute and relative errors
    abs_error = np.abs(adjoint_gradient - fd_gradient)
    max_abs_error = np.max(abs_error)
    avg_abs_error = np.mean(abs_error)
    
    # Relative error (avoid division by zero)
    fd_norm = np.linalg.norm(fd_gradient)
    adjoint_norm = np.linalg.norm(adjoint_gradient)
    
    if fd_norm > 1e-12:
        rel_error_norm = np.abs(adjoint_norm - fd_norm) / fd_norm
        rel_error_vec = np.linalg.norm(abs_error) / fd_norm
    else:
        rel_error_norm = np.inf
        rel_error_vec = np.inf
    
    print(f"Max absolute error:     {max_abs_error:.2e}")
    print(f"Average absolute error: {avg_abs_error:.2e}")
    print(f"Relative error (norm):  {rel_error_norm:.2e}")
    print(f"Relative error (vector):{rel_error_vec:.2e}")
    
    # Component-wise comparison
    if verbose:
        print("\nComponent-wise comparison:")
        print("i | Param |   Adjoint   |    FD       |  Abs Error  | Rel Error")
        print("-"*70)
        for i in range(len(adjoint_gradient)):
            param_name = f"P{i//2}" if i % 2 == 0 else f"Q{i//2}"
            adj_val = adjoint_gradient[i]
            fd_val = fd_gradient[i]
            abs_err = abs(adj_val - fd_val)
            rel_err = abs_err / max(abs(fd_val), 1e-12)
            print(f"{i:2d}| {param_name:5s} | {adj_val:10.3e} | {fd_val:10.3e} | {abs_err:10.3e} | {rel_err:8.1e}")
    
    # Test success criteria
    success_criteria = {
        "max_abs_error < 1e-4": max_abs_error < 1e-4,
        "avg_abs_error < 1e-5": avg_abs_error < 1e-5,
        "rel_error_norm < 0.01": rel_error_norm < 0.01,  # 1% relative error
        "rel_error_vec < 0.01": rel_error_vec < 0.01
    }
    
    all_passed = all(success_criteria.values())
    
    print(f"\nTest Results:")
    for criterion, passed in success_criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {criterion}: {status}")
    
    overall_status = "PASS" if all_passed else "FAIL"
    print(f"\nOverall: {overall_status}")
    
    # Return results
    return {
        "success": all_passed,
        "adjoint_gradient": adjoint_gradient,
        "fd_gradient": fd_gradient,
        "max_abs_error": max_abs_error,
        "avg_abs_error": avg_abs_error,
        "rel_error_norm": rel_error_norm,
        "rel_error_vec": rel_error_vec,
        "adjoint_cost": adjoint_cost,
        "criteria": success_criteria
    }


def quick_gradient_test(psys, tend=1.0, dt=1.0/120.0):
    """Quick test function with reasonable defaults."""
    config = IntegrationConfig(
        tend=tend,
        dt=dt,
        power_injection=False,
        solve_powerflow_dynamics=True,
        ton=0.25,
        toff=0.4
    )
    
    return test_adjoint_gradient_numerical(psys, config, epsilon_fd=1e-5, verbose=True)


from uqgrid.io.parse import load_psse, add_dyr

# Load system
psys = load_psse("data/ieee9_v33.raw")
psys.createYbusComplex()
add_dyr(psys, "data/ieee9bus.dyr")

# Add a small fault for more interesting dynamics
psys.add_busfault(4, 0.01, 0.25)  # Fault at bus 4
psys.set_load_parameters(np.zeros(psys.nloads))

# Run test
results = quick_gradient_test(psys, tend=2.0)

if results["success"]:
    print("✅ Adjoint gradient verification PASSED!")
else:
    print("❌ Adjoint gradient verification FAILED!")
    print("This indicates a bug in the adjoint implementation.")