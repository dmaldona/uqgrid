# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import cyipopt # Import cyipopt
import copy
import os
import sys
from typing import Dict, Tuple, Any, Optional
import time

# --- UQGrid Imports ---
try:
    from uqgrid.core.psydef import Psystem
    from uqgrid.simulation.dynamics import integrate_system
    from uqgrid.io.parse import load_psse, add_dyr
    from uqgrid.simulation.config import IntegrationConfig
except ImportError as e:
    print(f"Error importing uqgrid: {e}")
    print("Please ensure uqgrid is installed correctly and accessible in your Python environment.")
    print("You might need to install it (e.g., 'pip install .') or adjust your PYTHONPATH.")
    sys.exit(1)


# --- Configuration ---
# Select the system you want to run
RAW_FILE_PATH = "../data/2bus_33.raw"
DYR_FILE_PATH = "../data/GENROU.dyr"

RAW_FILE_PATH = "../data/ieee9_v33.raw"
DYR_FILE_PATH = "../data/ieee9bus_gov.dyr"

RAW_FILE_PATH = "../data/IEEE39_v33.raw"
DYR_FILE_PATH = "../data/IEEE39_gov.dyr"

# --- LDT/Simulation Configuration ---
Z_THRESHOLD = 7.5e-4
FAULT_BUS_IDX = 1
FAULT_Z = 0.01
FAULT_ON_TIME = 0.25
FAULT_OFF_TIME = 0.40
INTEGRATION_TEND = 5.0
INTEGRATION_DT = 1.0 / 120.0
I_THRESHOLD = 1e-7 # Min I(theta*) for FORM probability calculation

# --- LDT Math Functions ---
# (rate_function_I, grad_rate_function_I, calculate_form_probability remain the same)
def rate_function_I(theta: np.ndarray, theta_mean: np.ndarray, inv_covariance_diag: np.ndarray) -> float:
    r"""Objective: I(theta) = 0.5 * (theta - theta_mean)^T * C^-1 * (theta - theta_mean)"""
    theta = np.atleast_1d(theta)
    theta_mean = np.atleast_1d(theta_mean)
    inv_covariance_diag = np.atleast_1d(inv_covariance_diag)
    delta = theta - theta_mean
    return 0.5 * np.sum(delta * inv_covariance_diag * delta)

def grad_rate_function_I(theta: np.ndarray, theta_mean: np.ndarray, inv_covariance_diag: np.ndarray) -> np.ndarray:
    r"""Gradient of Objective: grad I(theta) = C^-1 * (theta - theta_mean)"""
    theta = np.atleast_1d(theta)
    theta_mean = np.atleast_1d(theta_mean)
    inv_covariance_diag = np.atleast_1d(inv_covariance_diag)
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
# (setup_system remains the same)
def setup_system(raw_path: str, dyr_path: str, fault_bus_idx: int, fault_z: float) -> Psystem:
    """Loads RAW and DYR files, adds fault, and returns the Psystem object."""
    print("Setting up base power system object...")
    if not os.path.isfile(raw_path):
        print(f"Error: RAW file not found at {raw_path}")
        sys.exit(1)
    if not os.path.isfile(dyr_path):
        print(f"Error: DYR file not found at {dyr_path}")
        sys.exit(1)

    psys = load_psse(raw_filename=raw_path)
    add_dyr(psys, dyr_path)
    try:
        if not hasattr(psys, 'ext2int') or not psys.ext2int:
             print("Error: External to internal bus mapping not created. Check load_psse.")
             sys.exit(1)
        internal_fault_bus = psys.ext2int[fault_bus_idx]
    except KeyError:
         print(f"Error: Fault bus ID {fault_bus_idx} not found in the system's external IDs.")
         sys.exit(1)
    except Exception as e:
         print(f"Error getting internal bus for {fault_bus_idx}: {e}")
         sys.exit(1)

    psys.add_busfault(internal_fault_bus, fault_z, 0.01) # Add fault using internal index
    print(f"Fault added to internal bus index: {internal_fault_bus} (External: {fault_bus_idx})")
    print("System setup complete.")
    return psys


# --- Simulation Runner ---
# (simulation_cache and run_simulation remain the same)
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
    theta_flat = np.atleast_1d(theta).flatten()
    theta_tuple = tuple(np.round(theta_flat, 8))

    if not all(np.isfinite(x) for x in theta_tuple):
         # print(f"Warning: Non-finite theta encountered: {theta_flat}. Returning failure.")
         return {"cost": np.inf, "v_mu": np.full_like(theta_flat, np.nan)}

    if theta_tuple in simulation_cache:
        # print(f"Cache hit for theta: {theta_tuple}")
        return simulation_cache[theta_tuple]
    # else:
    #     print(f"Cache miss for theta: {theta_tuple}")

    if not config.comp_sens or not config.petsc:
        raise ValueError("Configuration must have comp_sens=True and petsc=True.")

    psys_copy = copy.deepcopy(base_psys)
    theta_clipped = np.clip(theta_flat, 0.01, 0.99)
    # print(f"Running sim with clipped theta: {theta_clipped}")
    psys_copy.set_load_parameters(theta_clipped)

    results_dict = {"cost": np.inf, "v_mu": np.full_like(theta_flat, np.nan)}
    try:
        psys_copy.createYbusComplex()
        results = integrate_system(psys_copy, config)
        cost = results.get("cost")
        v_mu = results.get("v_mu")

        if cost is not None and v_mu is not None:
            if isinstance(cost, np.ndarray): cost = cost.item()
            if v_mu.shape == theta_flat.shape and np.isfinite(cost) and np.all(np.isfinite(v_mu)):
                results_dict["cost"] = cost
                results_dict["v_mu"] = v_mu.flatten()
                # print(f"Sim Success: theta={theta_flat}, cost={cost:.4e}, v_mu shape={results_dict['v_mu'].shape}")
            # else:
            #      print(f"DEBUG: Invalid sim results. Cost: {cost}, v_mu shape: {v_mu.shape if v_mu is not None else 'None'}, v_mu finite: {np.all(np.isfinite(v_mu)) if v_mu is not None else 'N/A'}, theta shape: {theta_flat.shape}")

    except Exception as e:
        # print(f"Warning: Simulation exception for theta={theta_flat}: {e}")
        pass

    simulation_cache[theta_tuple] = results_dict
    return results_dict


# --- IPOPT Problem Definition Class ---
class LDTProblem:
    def __init__(self, base_psys, config, theta_mean, inv_covariance_diag, z_threshold):
        self.base_psys = base_psys
        self.config = config
        self.theta_mean = np.atleast_1d(theta_mean)
        self.inv_covariance_diag = np.atleast_1d(inv_covariance_diag)
        self.z_threshold = z_threshold
        self.n_vars = len(self.theta_mean)
        self.n_con = 1 # Single equality constraint F(theta) - z = 0

    def objective(self, theta):
        """Return the objective function I(theta)"""
        return rate_function_I(theta, self.theta_mean, self.inv_covariance_diag)

    def gradient(self, theta):
        """Return the gradient of the objective function grad I(theta)"""
        return grad_rate_function_I(theta, self.theta_mean, self.inv_covariance_diag)

    def constraints(self, theta):
        """Return the constraint function value c(theta) = F(theta) - z"""
        # theta might be modified by the optimizer, ensure it's clipped if needed
        theta_clipped = np.clip(np.atleast_1d(theta), 0.01, 0.99)
        results = run_simulation(theta_clipped, self.base_psys, self.config)
        F_theta = results.get("cost", np.inf) # Default to infinity if sim fails
        # IPOPT expects constraints as a list or 1D array
        return np.array([F_theta - self.z_threshold])

    def jacobian(self, theta):
        """Return the Jacobian of the constraints grad c(theta) = grad F(theta)."""
         # theta might be modified by the optimizer, ensure it's clipped if needed
        theta_clipped = np.clip(np.atleast_1d(theta), 0.01, 0.99)
        results = run_simulation(theta_clipped, self.base_psys, self.config)
        grad_F = results.get("v_mu") # Default is NaN set in run_simulation on failure

        theta_flat = np.atleast_1d(theta_clipped).flatten()
        if grad_F is None or grad_F.shape != theta_flat.shape or not np.all(np.isfinite(grad_F)):
            # print(f"Debug Warning: Using zeros for constraint Jacobian. theta={theta_clipped}, grad_F={grad_F}")
            # IPOPT needs numerical values, NaNs will cause errors.
            # Returning zeros might be safer than NaNs if simulation failed.
            grad_F = np.zeros_like(theta_flat)
        # IPOPT expects the Jacobian values as a flattened array matching the structure
        return grad_F.flatten()

    def jacobianstructure(self):
        """Return the structure of the Jacobian (indices of nonzeros).
           For one constraint and n variables, the Jacobian is 1xn.
           Assuming it's dense, all entries are non-zero.
        """
        rows = np.zeros(self.n_vars, dtype=int) # Row index is 0 for the single constraint
        cols = np.arange(self.n_vars, dtype=int) # Column indices 0 to n-1
        return (rows, cols)

    def hessian(self, theta, lagrange, obj_factor):
        """Return the Hessian of the Lagrangian.
           L = obj_factor * I(theta) + lagrange[0] * c(theta)
           Hess(L) = obj_factor * Hess(I) + lagrange[0] * Hess(c)
                   = obj_factor * C^-1 + lagrange[0] * Hess(F)
           Calculating Hess(F) is complex (requires 2nd order sensitivities
           of the simulation cost). For now, we let IPOPT approximate it.
        """
        # Return only Hess(I), letting IPOPT handle the constraint part
        # Hess(I) is the diagonal matrix inv_covariance_diag
        # IPOPT expects values corresponding to lower triangle non-zeros
        hess_I_vals = obj_factor * self.inv_covariance_diag
        return hess_I_vals # Only diagonal elements needed for hessianstructure below

    def hessianstructure(self):
        """Return the structure of the Hessian (lower triangle indices).
           For now, only provide structure for Hess(I), which is diagonal.
        """
        rows = np.arange(self.n_vars)
        cols = np.arange(self.n_vars)
        return (rows, cols)

    # Optional: Intermediate callback
    # def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
    #                  d_norm, regularization_size, alpha_du, alpha_pr,
    #                  ls_trials):
    #     print(f"Iter: {iter_count}, Obj: {obj_value:.4e}, CPU: {time.time() - self.start_time:.2f}")


# --- Main Execution Logic ---
if __name__ == "__main__":
    # --- Parameter Distribution Setup ---
    try:
        base_psys_for_params = load_psse(raw_filename=RAW_FILE_PATH)
        num_params = base_psys_for_params.nloads
        print(f"Detected {num_params} loads (parameters).")
        if num_params == 0: sys.exit("Error: No loads found in the system.")
        del base_psys_for_params
    except Exception as e:
        print(f"Error loading system to determine number of parameters: {e}")
        sys.exit(1)

    theta_mean = np.full(num_params, 0.5)
    theta_std_dev = np.full(num_params, 0.1)
    inv_covariance_diag = 1.0 / (theta_std_dev**2)

    # --- Base System Setup ---
    base_psys = setup_system(RAW_FILE_PATH, DYR_FILE_PATH, FAULT_BUS_IDX, FAULT_Z)

    # --- Define Config with Sensitivities Enabled ---
    config_sens_enabled = IntegrationConfig(
        tend=INTEGRATION_TEND, dt=INTEGRATION_DT,
        ton=FAULT_ON_TIME, toff=FAULT_OFF_TIME,
        power_injection=False, verbose=False,
        comp_sens=True, fsolve=False, petsc=True
    )

    # --- LDT Optimization (Constrained using IPOPT) ---
    print(f"\nPerforming LDT constrained optimization with cyipopt (Minimize I subject to F={Z_THRESHOLD:.4e})...")

    # Create problem instance
    ldt_problem = LDTProblem(base_psys, config_sens_enabled, theta_mean, inv_covariance_diag, Z_THRESHOLD)

    # Define bounds for variables and constraints
    lb = [0.01] * num_params  # Lower bounds for theta
    ub = [0.99] * num_params  # Upper bounds for theta
    cl = [0.0]                # Lower bound for constraint F(theta) - z = 0
    cu = [0.0]                # Upper bound for constraint F(theta) - z = 0

    # Create IPOPT problem object
    nlp = cyipopt.Problem(
        n=num_params,          # Number of variables
        m=1,                   # Number of constraints
        problem_obj=ldt_problem, # Class providing callbacks
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu
    )

    # --- Set IPOPT Options (Optional) ---
    # nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('max_iter', 100)
    nlp.add_option('tol', 1e-7)
    # Use limited-memory approximation for Hessian (good default)
    nlp.add_option('hessian_approximation', 'limited-memory')
    # To use the analytical Hessian (requires correct hessian callback):
    # nlp.add_option('hessian_approximation', 'exact')
    # nlp.add_option('limited_memory_max_history', 50) # If using limited-memory

    # --- Solve the Problem ---
    simulation_cache.clear() # Clear cache before optimization
    print("Starting IPOPT solve...")
    ldt_problem.start_time = time.time() # For timing in intermediate callback if used

    theta_star, info = nlp.solve(theta_mean.copy()) # Solve starting from mean

    print(f"IPOPT optimization took {time.time() - ldt_problem.start_time:.2f} seconds.")
    print(f"IPOPT final status: {info['status_msg']}")


    # --- Final Evaluation and Results ---
    print(f"\nIPOPT Result:")
    print(f"Optimal parameters theta*: {theta_star}")
    print(f"Optimal objective I(theta*): {info['obj_val']:.6f}")

    # Verify the result by running simulation one last time
    simulation_cache.clear()
    print("Running final simulation with optimized theta*...")
    theta_star_clipped = np.clip(np.atleast_1d(theta_star), 0.01, 0.99)
    final_results = run_simulation(theta_star_clipped, base_psys, config_sens_enabled)
    F_theta_star = final_results.get("cost", np.inf)
    # Recalculate I_theta_star for consistency, though info['obj_val'] should be the same
    I_theta_star_recalc = ldt_problem.objective(theta_star_clipped)

    if np.isinf(F_theta_star):
        print("\nError: Final simulation failed for optimal theta*. Cannot calculate probability.")
        print(f"  Final theta* used:  {theta_star_clipped}")
        print(f"  Minimized I(theta*):{I_theta_star_recalc:.6f} (from potentially infeasible point)")
    else:
        P_ldt_form = calculate_form_probability(I_theta_star_recalc)
        constraint_violation = abs(F_theta_star - Z_THRESHOLD)

        print(f"\nLDT Result Verification:")
        print(f"  Target z:           {Z_THRESHOLD:.6f}")
        print(f"  Found theta*:       {theta_star_clipped}")
        print(f"  Achieved F(theta*): {F_theta_star:.6f}")
        print(f"  Constraint Viol.:   {constraint_violation:.3e} (F(theta*) - z)")
        print(f"  Recalculated I(theta*):{I_theta_star_recalc:.6f}")
        print(f"  P_FORM(F>=z) ~      {P_ldt_form:.4e}")

        if info['status'] < 0 or constraint_violation > 1e-5 : # Check IPOPT status and constraint
             print("\nWarning: IPOPT may not have converged optimally or constraint violation is high.")
             print("The probability estimate might be less accurate.")