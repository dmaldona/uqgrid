# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import copy
import os
import sys
from typing import Dict, Tuple, Any, Optional
import time

# --- UQGrid Imports ---
# (Imports remain the same)
try:
    from uqgrid.core.psydef import Psystem
    from uqgrid.simulation.dynamics import integrate_system
    from uqgrid.io.parse import load_psse, add_dyr
    from uqgrid.simulation.config import IntegrationConfig
except ImportError as e:
    # ... (error handling as before) ...
    print(f"Error importing uqgrid: {e}")
    sys.exit(1)


# --- Configuration ---
RAW_FILE_PATH = "../data/2bus_33.raw"
DYR_FILE_PATH = "../data/GENROU.dyr"

#RAW_FILE_PATH = "../data/ieee9_v33.raw"
#DYR_FILE_PATH = "../data/ieee9bus_gov.dyr"

# --- LDT/Simulation Configuration ---
# *** USER: Set the desired threshold for the event functional F(theta) ***
# Let's try the threshold that worked better with grid search
Z_THRESHOLD = 7.5e-4
# *************************************
FAULT_BUS_IDX = 1
FAULT_Z = 0.5
FAULT_ON_TIME = 0.25
FAULT_OFF_TIME = 0.40
INTEGRATION_TEND = 5.0
INTEGRATION_DT = 1.0 / 120.0
I_THRESHOLD = 1e-7 # Min I(theta*) for FORM probability calculation

# --- LDT Math Functions ---
def rate_function_I(theta: np.ndarray, theta_mean: np.ndarray, inv_covariance_diag: np.ndarray) -> float:
    r"""Objective: I(theta) = 0.5 * (theta - theta_mean)^T * C^-1 * (theta - theta_mean)"""
    delta = theta - theta_mean
    return 0.5 * np.sum(delta * inv_covariance_diag * delta)

def grad_rate_function_I(theta: np.ndarray, theta_mean: np.ndarray, inv_covariance_diag: np.ndarray) -> np.ndarray:
    r"""Gradient of Objective: grad I(theta) = C^-1 * (theta - theta_mean)"""
    delta = theta - theta_mean
    return inv_covariance_diag * delta

def calculate_form_probability(I_theta_star: float) -> float:
    r"""P_FORM(z) \approx [ (2*pi) * (2*I(theta*)) ]^(-1/2) * exp(-I(theta*))"""
    if I_theta_star < I_THRESHOLD: return np.nan
    denominator = np.sqrt(2.0 * np.pi * 2.0 * I_theta_star)
    if denominator < 1e-15: return np.nan
    prefactor = 1.0 / denominator
    probability = prefactor * np.exp(-I_theta_star)
    return probability

# --- System Setup ---
def setup_system(raw_path: str, dyr_path: str, fault_bus_idx: int, fault_z: float) -> Psystem:
    """Loads RAW and DYR files, adds fault, and returns the Psystem object."""
    print("Setting up base power system object...")
    psys = load_psse(raw_filename=raw_path)
    add_dyr(psys, dyr_path)
    try: internal_fault_bus = psys.ext2int[fault_bus_idx]
    except Exception as e: print(f"Error getting internal bus for {fault_bus_idx}: {e}"); sys.exit(1)
    psys.add_busfault(internal_fault_bus, fault_z, 0.01)
    print("System setup complete.")
    return psys

# --- Simulation Runner ---
# Cache now stores the full result dictionary (cost and gradient)
simulation_cache: Dict[Tuple[float, ...], Dict[str, Any]] = {}

def run_simulation(
    theta: np.ndarray,
    base_psys: Psystem,
    config: IntegrationConfig # Must have comp_sens=True, petsc=True
    ) -> Dict[str, Any]:
    """
    Runs a single simulation FORCING SENSITIVITY CALCULATION.
    Returns dictionary with 'cost' (F(theta)) and 'v_mu' (grad F(theta)).
    Handles caching and errors.
    """
    # Use a tuple key, rounding theta slightly for cache robustness
    # Ensure theta is 1D array before creating tuple
    theta_flat = np.atleast_1d(theta).flatten()
    theta_tuple = tuple(np.round(theta_flat, 8))

    if not all(np.isfinite(x) for x in theta_tuple):
         return {"cost": np.inf, "v_mu": np.zeros_like(theta_flat)}

    if theta_tuple in simulation_cache:
        return simulation_cache[theta_tuple]

    # Check config - essential for getting 'cost' and 'v_mu'
    if not config.comp_sens or not config.petsc:
        raise ValueError("Configuration must have comp_sens=True and petsc=True.")

    # Make a deep copy to ensure thread-safety if parallelizing later
    # and avoid modifying base_psys
    psys_copy = copy.deepcopy(base_psys)
    # Clip theta within bounds just before simulation if necessary
    theta_clipped = np.clip(theta_flat, 0.01, 0.99)
    psys_copy.set_load_parameters(theta_clipped) # Pass the 1D array

    results_dict = {"cost": np.inf, "v_mu": np.zeros_like(theta_flat)} # Default failure
    try:
        psys_copy.createYbusComplex() # Create Ybus for the copied system
        results = integrate_system(psys_copy, config) # Run simulation
        cost = results.get("cost")
        v_mu = results.get("v_mu")

        # Validate results before caching
        if cost is not None and v_mu is not None:
            if isinstance(cost, np.ndarray): cost = cost.item() # Ensure scalar cost
            # Ensure v_mu has correct shape (matches theta_flat)
            if v_mu.shape == theta_flat.shape and np.isfinite(cost) and np.all(np.isfinite(v_mu)):
                results_dict["cost"] = cost
                results_dict["v_mu"] = v_mu.flatten() # Ensure v_mu is flat
            # else: # Debugging non-finite or shape mismatch
            #     print(f"DEBUG: Invalid sim results. Cost: {cost}, v_mu shape: {v_mu.shape}, theta shape: {theta_flat.shape}")

    except Exception as e:
        print(f"Warning: Sim exception for theta={theta_flat}: {e}")
        pass # Keep default failure result

    simulation_cache[theta_tuple] = results_dict
    return results_dict


# --- Functions for Scipy Optimizer (Constrained) ---

def objective_func(theta: np.ndarray, theta_mean: np.ndarray, inv_covariance_diag: np.ndarray) -> float:
    """The objective function for the optimizer is I(theta)."""
    # Ensure theta is treated as a 1D array if scalar is passed
    theta = np.atleast_1d(theta)
    theta_mean = np.atleast_1d(theta_mean)
    inv_covariance_diag = np.atleast_1d(inv_covariance_diag)
    return rate_function_I(theta, theta_mean, inv_covariance_diag)

def objective_grad(theta: np.ndarray, theta_mean: np.ndarray, inv_covariance_diag: np.ndarray) -> np.ndarray:
    """The gradient of the objective function is grad I(theta)."""
    theta = np.atleast_1d(theta)
    theta_mean = np.atleast_1d(theta_mean)
    inv_covariance_diag = np.atleast_1d(inv_covariance_diag)
    return grad_rate_function_I(theta, theta_mean, inv_covariance_diag)

def constraint_eq(theta: np.ndarray, base_psys: Psystem, config: IntegrationConfig, z_threshold: float) -> float:
    """Equality constraint function: c(theta) = F(theta) - z = 0"""
    results = run_simulation(theta, base_psys, config)
    F_theta = results.get("cost", np.inf) # Default to infinity if sim fails
    # print(f"Debug Constraint: theta={theta}, F={F_theta:.4e}, c={F_theta - z_threshold:.4e}") # Debug print
    return F_theta - z_threshold

def constraint_eq_jac(theta: np.ndarray, base_psys: Psystem, config: IntegrationConfig, z_threshold: float) -> np.ndarray:
    """Jacobian of the equality constraint: grad c(theta) = grad F(theta)"""
    results = run_simulation(theta, base_psys, config)
    grad_F = results.get("v_mu", np.zeros_like(np.atleast_1d(theta))) # Default to zero vector
    # Ensure correct shape/type if simulation failed
    theta_flat = np.atleast_1d(theta).flatten()
    if grad_F is None or grad_F.shape != theta_flat.shape:
        # print(f"Debug Warning: grad_F shape mismatch or None. theta={theta}, grad_F={grad_F}")
        grad_F = np.zeros_like(theta_flat)
    # print(f"Debug Constraint Jac: theta={theta}, grad_F={grad_F}") # Debug print
    return grad_F.flatten() # Ensure 1D


# --- Main Execution Logic ---
if __name__ == "__main__":
    # --- Parameter Distribution Setup ---
    base_psys_setup = load_psse(raw_filename=RAW_FILE_PATH)
    num_params = base_psys_setup.nloads
    if num_params != 1: sys.exit("Error: Script currently requires nloads=1.")
    del base_psys_setup
    theta_mean = np.array([0.5])
    theta_std_dev = np.array([0.1])
    inv_covariance_diag = 1.0 / (theta_std_dev**2)

    # --- Base System Setup ---
    base_psys = setup_system(RAW_FILE_PATH, DYR_FILE_PATH, FAULT_BUS_IDX, FAULT_Z)

    # --- Define Config with Sensitivities Enabled ---
    config_sens_enabled = IntegrationConfig(
        tend=INTEGRATION_TEND, dt=INTEGRATION_DT,
        ton=FAULT_ON_TIME, toff=FAULT_OFF_TIME,
        power_injection=False, verbose=False,
        comp_sens=True, fsolve=False, petsc=True # Must be True
    )

    # --- LDT Optimization (Constrained using SLSQP) ---
    print(f"\nPerforming LDT constrained optimization (Minimize I subject to F={Z_THRESHOLD:.4e})...")
    initial_guess = theta_mean
    # Bounds for the single parameter theta[0]
    bounds = [(0.01, 0.99)] * num_params # List of tuples

    # Define constraint dictionary for SLSQP, providing analytical Jacobians
    constraints = ({'type': 'eq',
                    'fun': constraint_eq,
                    'jac': constraint_eq_jac,
                    'args': (base_psys, config_sens_enabled, Z_THRESHOLD)}) # Args passed only to fun/jac

    simulation_cache.clear() # Clear cache before optimization
    start_time_opt = time.time()

    optimization_result = scipy.optimize.minimize(
        objective_func,
        initial_guess,
        args=(theta_mean, inv_covariance_diag), # Args for objective func/grad
        method='SLSQP',                         # Constrained solver
        jac=objective_grad,                     # Provide gradient of objective
        constraints=constraints,                # Provide constraint(s) (with Jacobian)
        bounds=bounds,
        options={'disp': True, 'maxiter': 100, 'ftol': 1e-8} # Enable solver output
    )

    print(f"Optimization took {time.time() - start_time_opt:.2f} seconds.")

    # --- Final Evaluation and Results ---
    if not optimization_result.success:
        print(f"\nWarning: LDT optimization did not converge: {optimization_result.message}")
        # Decide how to handle failure - exit, use last point, etc.
        # For now, proceed using the result but be aware it might not be optimal/feasible
        theta_star = optimization_result.x
    else:
         print("\nOptimization terminated successfully.")
         theta_star = optimization_result.x


    print(f"Optimal parameters theta*: {theta_star}")

    simulation_cache.clear() # Clear cache for final reliable evaluation
    print("Running final simulation with optimized theta*...")
    final_results = run_simulation(theta_star, base_psys, config_sens_enabled)
    F_theta_star = final_results.get("cost", np.inf)
    I_theta_star = objective_func(theta_star, theta_mean, inv_covariance_diag) # Use objective_func

    if np.isinf(F_theta_star):
        print("Error: Final simulation failed for optimal theta*. Cannot calculate probability.")
    else:
        P_ldt_form = calculate_form_probability(I_theta_star)
        print(f"\nLDT Result (Constrained Optimization):")
        print(f"  Target z:           {Z_THRESHOLD:.6f}")
        print(f"  Found theta*:       {np.atleast_1d(theta_star)[0]:.6f}") # Extract scalar if needed
        print(f"  Achieved F(theta*): {F_theta_star:.6f}")
        print(f"  Constraint Viol.:   {abs(F_theta_star - Z_THRESHOLD):.3e}")
        print(f"  Minimized I(theta*):{I_theta_star:.6f}")
        print(f"  P_FORM(F>=z) ~      {P_ldt_form:.4e}")