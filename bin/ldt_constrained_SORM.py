# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import cyipopt # Import cyipopt
import copy
import os
import sys
from typing import Dict, Tuple, Any, Optional
import time
import scipy.linalg # For SORM: eigh, null_space, qr

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
# RAW_FILE_PATH = "../data/2bus_33.raw"
# DYR_FILE_PATH = "../data/GENROU.dyr"
# FAULT_BUS_IDX = 1 # External Bus ID for fault

RAW_FILE_PATH = "../data/ieee9_v33.raw"
DYR_FILE_PATH = "../data/ieee9bus_gov.dyr"
FAULT_BUS_IDX = 1 # External Bus ID for fault (e.g., Bus 7 in IEEE9 RAW)

#RAW_FILE_PATH = "../data/IEEE39_v33.raw"
#DYR_FILE_PATH = "../data/IEEE39_gov.dyr"
#FAULT_BUS_IDX = 1 # External Bus ID for fault (e.g., Bus 3 in IEEE39 RAW)

# --- LDT/Simulation Configuration ---
Z_THRESHOLD = 7.5e-4 # Constraint F(theta) = Z_THRESHOLD
Z_THRESHOLD = 1.5e-4 # Constraint F(theta) = Z_THRESHOLD

# FAULT_BUS_IDX defined above
FAULT_Z = 0.01
FAULT_ON_TIME = 0.25
FAULT_OFF_TIME = 0.40
INTEGRATION_TEND = 5.0
INTEGRATION_DT = 1.0 / 120.0
I_THRESHOLD = 1e-7 # Min I(theta*) for FORM probability calculation
FINITE_DIFF_EPSILON = 1e-7 # Epsilon for finite difference Hessian-vector products
SORM_STABILITY_TOL = 1e-9 # For checking positivity of eigenvalues in SORM product

# --- LDT Math Functions ---
def rate_function_I(theta: np.ndarray, theta_mean: np.ndarray, inv_covariance_diag: np.ndarray) -> float:
    theta = np.atleast_1d(theta)
    theta_mean = np.atleast_1d(theta_mean)
    inv_covariance_diag = np.atleast_1d(inv_covariance_diag)
    delta = theta - theta_mean
    return 0.5 * np.sum(delta * inv_covariance_diag * delta)

def grad_rate_function_I(theta: np.ndarray, theta_mean: np.ndarray, inv_covariance_diag: np.ndarray) -> np.ndarray:
    theta = np.atleast_1d(theta)
    theta_mean = np.atleast_1d(theta_mean)
    inv_covariance_diag = np.atleast_1d(inv_covariance_diag)
    delta = theta - theta_mean
    return inv_covariance_diag * delta

def calculate_form_probability(I_theta_star: float) -> float:
    if not np.isfinite(I_theta_star) or I_theta_star < I_THRESHOLD: return np.nan
    if I_theta_star <= 0: return np.nan # I(theta*) should be positive
    denominator_factor = 2.0 * I_theta_star
    if denominator_factor <= 1e-15: return np.nan
    
    prefactor_val = (1.0 / np.sqrt(2.0 * np.pi * denominator_factor))
    probability = prefactor_val * np.exp(-I_theta_star)
    return probability

# --- System Setup ---
def setup_system(raw_path: str, dyr_path: str, fault_bus_idx_ext: int, fault_z: float) -> Psystem:
    print("Setting up base power system object...")
    if not os.path.isfile(raw_path): print(f"Error: RAW file not found at {raw_path}"); sys.exit(1)
    if not os.path.isfile(dyr_path): print(f"Error: DYR file not found at {dyr_path}"); sys.exit(1)

    psys = load_psse(raw_filename=raw_path)
    add_dyr(psys, dyr_path)
    try:
        if not hasattr(psys, 'ext2int') or not psys.ext2int:
             print("Error: External to internal bus mapping not created."); sys.exit(1)
        internal_fault_bus = psys.ext2int[int(fault_bus_idx_ext)] # Ensure int
    except KeyError:
         print(f"Error: Fault bus ID {fault_bus_idx_ext} not found in system's external IDs: {list(psys.ext2int.keys())}"); sys.exit(1)
    except Exception as e:
         print(f"Error getting internal bus for {fault_bus_idx_ext}: {e}"); sys.exit(1)

    psys.add_busfault(internal_fault_bus, fault_z, 0.01)
    print(f"Fault added to internal bus index: {internal_fault_bus} (External: {fault_bus_idx_ext})")
    print("System setup complete.")
    return psys

# --- Simulation Runner ---
simulation_cache: Dict[Tuple[float, ...], Dict[str, Any]] = {}
simulation_count_global = 0

def run_simulation(
    theta: np.ndarray,
    base_psys_sim: Psystem, # Renamed to avoid clash
    config_sim: IntegrationConfig, # Renamed
    caller_info: str = "unknown"
    ) -> Dict[str, Any]:
    global simulation_count_global
    simulation_count_global += 1
    # print(f"Sim call #{simulation_count_global} from {caller_info} for theta: {np.array(theta)[:min(3,len(np.array(theta)))]}...")
    
    theta_flat = np.atleast_1d(theta).flatten()
    theta_tuple = tuple(np.round(theta_flat, 8)) # Cache key

    if not all(np.isfinite(x) for x in theta_tuple):
         return {"cost": np.inf, "v_mu": np.full_like(theta_flat, np.nan)}
    if theta_tuple in simulation_cache:
        return simulation_cache[theta_tuple]

    if not config_sim.comp_sens or not config_sim.petsc:
        raise ValueError("Configuration must have comp_sens=True and petsc=True.")

    psys_copy = copy.deepcopy(base_psys_sim)
    theta_clipped = np.clip(theta_flat, 0.01, 0.99)
    psys_copy.set_load_parameters(theta_clipped)

    results_dict = {"cost": np.inf, "v_mu": np.full_like(theta_flat, np.nan)}
    try:
        psys_copy.createYbusComplex()
        results = integrate_system(psys_copy, config_sim)
        cost = results.get("cost")
        v_mu = results.get("v_mu")

        if cost is not None and v_mu is not None:
            cost_item = cost.item() if isinstance(cost, np.ndarray) else cost
            v_mu_flat = np.atleast_1d(v_mu).flatten()
            if v_mu_flat.shape == theta_flat.shape and np.isfinite(cost_item) and np.all(np.isfinite(v_mu_flat)):
                results_dict["cost"] = cost_item
                results_dict["v_mu"] = v_mu_flat
    except Exception: # Minimal printing during repeated calls
        pass
    simulation_cache[theta_tuple] = results_dict
    return results_dict

# --- Hessian-vector product and Explicit Hessian for F(theta) ---
def compute_hessian_F_vec_prod(
    theta_at: np.ndarray, vector_v: np.ndarray,
    base_psys_hvp: Psystem, config_hvp: IntegrationConfig, epsilon_hvp: float
    ) -> np.ndarray:
    """Computes (grad^2 F(theta_at)) @ vector_v using central finite differences."""
    theta_flat = np.atleast_1d(theta_at).flatten()
    v_flat = np.atleast_1d(vector_v).flatten()

    theta_plus = theta_flat + epsilon_hvp * v_flat
    grad_F_plus_dict = run_simulation(theta_plus, base_psys_hvp, config_hvp, caller_info="HVP_plus")
    grad_F_plus = grad_F_plus_dict.get("v_mu")

    theta_minus = theta_flat - epsilon_hvp * v_flat
    grad_F_minus_dict = run_simulation(theta_minus, base_psys_hvp, config_hvp, caller_info="HVP_minus")
    grad_F_minus = grad_F_minus_dict.get("v_mu")
    
    if grad_F_plus is None or not np.all(np.isfinite(grad_F_plus)) or \
       grad_F_minus is None or not np.all(np.isfinite(grad_F_minus)):
        print(f"Warning HVP: Failed to get valid gradients. theta_at={theta_at[:3]}, v={v_flat[:3]}")
        return np.full_like(theta_flat, np.nan)
    
    return (grad_F_plus - grad_F_minus) / (2.0 * epsilon_hvp)

def compute_hessian_F_explicit(
    theta_at: np.ndarray, num_params_hess: int,
    base_psys_hess: Psystem, config_hess: IntegrationConfig, epsilon_hess: float
    ) -> np.ndarray:
    """Computes the full Hessian matrix grad^2 F(theta_at) explicitly."""
    print(f"Computing explicit {num_params_hess}x{num_params_hess} Hessian of F via {num_params_hess} HVP calls...")
    H_F = np.zeros((num_params_hess, num_params_hess))
    for j in range(num_params_hess):
        e_j = np.zeros(num_params_hess)
        e_j[j] = 1.0
        H_F_col_j = compute_hessian_F_vec_prod(theta_at, e_j, base_psys_hess, config_hess, epsilon_hess)
        if not np.all(np.isfinite(H_F_col_j)):
            print(f"Warning: NaN column in Hessian F at index {j}. Filling with zeros.")
            H_F[:, j] = 0.0 # Or handle error more gracefully
        else:
            H_F[:, j] = H_F_col_j
        if (j + 1) % max(1, num_params_hess // 10) == 0 or j == num_params_hess - 1:
            print(f"  Hessian F: Computed column {j+1}/{num_params_hess}")
            
    # Symmetrize due to potential numerical inaccuracies from finite differences
    H_F = 0.5 * (H_F + H_F.T)
    print("Explicit Hessian of F computation complete.")
    return H_F

# --- SORM Probability Calculation (Explicit Hessian) ---
def calculate_sorm_probability_explicit_hessian(
    theta_star_sorm: np.ndarray, I_theta_star_sorm: float, lambda_LDT_sorm: float,
    P_form_sorm: float, base_psys_sorm: Psystem, config_sorm: IntegrationConfig,
    theta_mean_sorm: np.ndarray, theta_std_dev_sorm: np.ndarray,
    num_params_sorm: int, epsilon_hess_sorm: float
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Calculates SORM probability using explicitly formed Hessian of F."""
    print(f"\n--- Starting SORM Calculation (Explicit Hessian) ---")
    sorm_eigvals_for_prod = np.array([]) # For returning
    sorm_M_sub_eigvals_all = np.array([])


    if not (np.isfinite(I_theta_star_sorm) and I_theta_star_sorm > 0 and \
            np.isfinite(lambda_LDT_sorm) and np.isfinite(P_form_sorm)):
        print("Warning SORM: Invalid inputs (I*, lambda, P_FORM). Cannot compute SORM.")
        return np.nan, sorm_eigvals_for_prod, sorm_M_sub_eigvals_all

    H_F_at_theta_star = compute_hessian_F_explicit(
        theta_star_sorm, num_params_sorm, base_psys_sorm, config_sorm, epsilon_hess_sorm
    )
    if not np.all(np.isfinite(H_F_at_theta_star)):
        print("Warning SORM: Hessian of F contains NaNs. Cannot compute SORM.")
        return np.nan, sorm_eigvals_for_prod, sorm_M_sub_eigvals_all

    S_diag = theta_std_dev_sorm # S_ii = C_ii^(1/2)
    M_S = (S_diag[:, np.newaxis] * H_F_at_theta_star) * S_diag[np.newaxis, :] # S H_F S

    # Matrix for SORM product based on paper's Thm 4.2: (I - lambda_LDT * M_S_projected)
    # M_S_projected has eigenvalues kappa_prime_j. We need prod (1 - lambda_LDT * kappa_prime_j)^(-1/2)
    # This is equivalent to (det(P (I - lambda_LDT M_S) P)_tangent)^(-1/2)
    
    M_comb = np.eye(num_params_sorm) - lambda_LDT_sorm * M_S

    # Projection part:
    v_std = (theta_star_sorm - theta_mean_sorm) / theta_std_dev_sorm
    norm_v_std = np.linalg.norm(v_std)
    if norm_v_std < 1e-9:
        print("Warning SORM: Norm of v_std (for projection) is near zero.")
        return np.nan, sorm_eigvals_for_prod, sorm_M_sub_eigvals_all
    u_proj_vec = v_std / norm_v_std # Normalized C^(-1/2)(theta*-theta0)

    # Construct orthonormal basis Q where Q[:,0] = u_proj_vec
    Q_basis = np.zeros((num_params_sorm, num_params_sorm))
    Q_basis[:, 0] = u_proj_vec
    if num_params_sorm > 1:
        # Basis for orthogonal complement of u_proj_vec
        orth_comp_basis = scipy.linalg.null_space(u_proj_vec[:, np.newaxis].T)
        if orth_comp_basis.shape[1] != num_params_sorm - 1:
            # Fallback if null_space gives unexpected shape (e.g. num_params_sorm=1)
            # This can happen if u_proj_vec is an axis vector or similar simple cases
            # A robust way is full QR from a perturbed Identity or by creating a Householder reflector
            # For now, simple completion for small N, might not be robust for all u_proj_vec
            print(f"SORM: null_space dimension {orth_comp_basis.shape[1]} != {num_params_sorm -1}. Using QR fallback.")
            # Create a matrix where first col is u_proj_vec, rest are from Identity
            # Ensure linear independence for QR
            temp_mat_for_qr = np.eye(num_params_sorm)
            # Ensure first vector is u_proj_vec
            temp_mat_for_qr[:,0] = u_proj_vec
            # Make other columns orthogonal to u_proj_vec for robustness before QR
            for i_qr in range(1, num_params_sorm):
                 if np.allclose(temp_mat_for_qr[:,i_qr], u_proj_vec) or \
                    np.allclose(temp_mat_for_qr[:,i_qr], -u_proj_vec): # if a column is parallel to u
                     # find another vector from identity not parallel
                     for k_qr in range(num_params_sorm):
                         candidate_col = np.eye(num_params_sorm)[:,k_qr]
                         if not (np.allclose(candidate_col,u_proj_vec) or np.allclose(candidate_col,-u_proj_vec)):
                             temp_mat_for_qr[:,i_qr] = candidate_col
                             break
            Q_basis_from_qr, _ = scipy.linalg.qr(temp_mat_for_qr)
            Q_basis = Q_basis_from_qr # Q_basis has u_proj_vec (or -u_proj_vec) as first column
            # Verify and adjust sign if needed for Q_basis[:,0] to be u_proj_vec
            if np.dot(Q_basis[:,0], u_proj_vec) < 0:
                Q_basis[:,0] *= -1

        else:
             Q_basis[:, 1:] = orth_comp_basis


    M_comb_rotated = Q_basis.T @ M_comb @ Q_basis
    M_sub = M_comb_rotated[1:, 1:] # This is (P (I - lambda_LDT M_S) P) in tangent space
    sorm_M_sub_eigvals_all = np.linalg.eigh(M_sub)[0] # Get all eigenvalues of M_sub
    
    print(f"SORM: Eigenvalues of M_sub (size {M_sub.shape[0]}x{M_sub.shape[0]}): {sorm_M_sub_eigvals_all[:min(10, len(sorm_M_sub_eigvals_all))]}...")

    sorm_eigvals_for_prod = sorm_M_sub_eigvals_all[sorm_M_sub_eigvals_all > SORM_STABILITY_TOL]
    
    if len(sorm_eigvals_for_prod) < num_params_sorm - 1:
        num_excluded = (num_params_sorm - 1) - len(sorm_eigvals_for_prod)
        print(f"Warning SORM: {num_excluded} eigenvalue(s) of M_sub were <= {SORM_STABILITY_TOL} and excluded from product.")

    if not sorm_eigvals_for_prod.size: # No positive eigenvalues
        print("Warning SORM: No positive eigenvalues for SORM product. Correction factor is NaN.")
        sorm_correction_factor = np.nan
    else:
        sorm_correction_factor = np.prod(sorm_eigvals_for_prod**(-0.5))
        sorm_correction_factor = np.real(sorm_correction_factor) # Ensure real

    if not np.isfinite(sorm_correction_factor):
        P_sorm = np.nan
    else:
        P_sorm = P_form_sorm * sorm_correction_factor
    print(f"SORM: Correction factor = {sorm_correction_factor:.4e}")
    print(f"--- SORM Calculation Finished (Explicit Hessian) ---")
    return P_sorm, sorm_eigvals_for_prod, sorm_M_sub_eigvals_all


# --- IPOPT Problem Definition Class ---
class LDTProblem:
    def __init__(self, base_psys_cls: Psystem, config_cls: IntegrationConfig, theta_mean_cls: np.ndarray, 
                 inv_covariance_diag_cls: np.ndarray, z_threshold_cls: float): # Renamed args
        self.base_psys = base_psys_cls
        self.config = config_cls
        self.theta_mean = np.atleast_1d(theta_mean_cls)
        self.inv_covariance_diag = np.atleast_1d(inv_covariance_diag_cls)
        self.z_threshold = z_threshold_cls
        self.n_vars = len(self.theta_mean)
        self.n_con = 1
        self.eval_count_obj = 0
        self.eval_count_con = 0
        self.eval_count_grad = 0
        self.eval_count_jac = 0


    def objective(self, theta):
        self.eval_count_obj +=1
        return rate_function_I(theta, self.theta_mean, self.inv_covariance_diag)

    def gradient(self, theta):
        self.eval_count_grad += 1
        return grad_rate_function_I(theta, self.theta_mean, self.inv_covariance_diag)

    def constraints(self, theta):
        self.eval_count_con +=1
        theta_clipped = np.clip(np.atleast_1d(theta), 0.01, 0.99)
        results = run_simulation(theta_clipped, self.base_psys, self.config, caller_info="IPOPT_constraints")
        F_theta = results.get("cost", np.inf if self.eval_count_con > 1 else 1e6) # Default large
        return np.array([F_theta - self.z_threshold])

    def jacobian(self, theta):
        self.eval_count_jac += 1
        theta_clipped = np.clip(np.atleast_1d(theta), 0.01, 0.99)
        results = run_simulation(theta_clipped, self.base_psys, self.config, caller_info="IPOPT_jacobian")
        grad_F = results.get("v_mu")
        theta_flat = np.atleast_1d(theta_clipped).flatten()
        if grad_F is None or not np.all(np.isfinite(grad_F)) or grad_F.shape != theta_flat.shape :
            grad_F = np.zeros_like(theta_flat)
        return grad_F.flatten()

    def jacobianstructure(self):
        rows = np.zeros(self.n_vars, dtype=int)
        cols = np.arange(self.n_vars, dtype=int)
        return (rows, cols)
    
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        current_time = time.time()
        elapsed_time = current_time - getattr(self, 'start_time_iter', current_time)
        if iter_count == 0 : self.start_time_iter = current_time
        print(f"Iter: {iter_count:3d}, Obj: {obj_value:9.2e}, PrimalInf: {inf_pr:9.2e}, DualInf: {inf_du:9.2e}, "
              f"Time: {elapsed_time:.2f}s, Sims: {simulation_count_global} (Opt Evals C:{self.eval_count_con} J:{self.eval_count_jac})")

# --- Main Execution Logic ---
if __name__ == "__main__":
    # --- Parameter Distribution Setup ---
    try:
        _psys_temp = load_psse(raw_filename=RAW_FILE_PATH)
        num_params = _psys_temp.nloads
        print(f"Detected {num_params} loads (parameters).")
        if num_params == 0: print("Error: No loads found."); sys.exit(1)
        del _psys_temp
    except Exception as e:
        print(f"Error loading system for num_params: {e}"); sys.exit(1)

    theta_mean = np.full(num_params, 0.5)
    theta_std_dev = np.full(num_params, 0.1)
    inv_covariance_diag = 1.0 / (theta_std_dev**2)

    base_psys_main = setup_system(RAW_FILE_PATH, DYR_FILE_PATH, FAULT_BUS_IDX, FAULT_Z)
    config_main = IntegrationConfig(
        tend=INTEGRATION_TEND, dt=INTEGRATION_DT,
        ton=FAULT_ON_TIME, toff=FAULT_OFF_TIME,
        power_injection=False, verbose=False,
        comp_sens=True, fsolve=False, petsc=True
    )

    print(f"\nPerforming LDT constrained optimization (F={Z_THRESHOLD:.4e})...")
    ldt_problem_obj = LDTProblem(base_psys_main, config_main, theta_mean, inv_covariance_diag, Z_THRESHOLD)
    
    # IPOPT setup
    lb = [0.01] * num_params; ub = [0.99] * num_params
    cl = [0.0]; cu = [0.0] # F(theta) - z = 0
    nlp = cyipopt.Problem(n=num_params, m=1, problem_obj=ldt_problem_obj, lb=lb, ub=ub, cl=cl, cu=cu)
    nlp.add_option('max_iter', 100); nlp.add_option('tol', 1e-6)
    nlp.add_option('hessian_approximation', 'limited-memory')
    nlp.add_option('limited_memory_max_history', 20)
    # nlp.add_option('print_level', 0) # Suppress IPOPT output to see custom intermediate callback

    simulation_cache.clear(); simulation_count_global = 0
    print("Starting IPOPT solve...")
    ldt_problem_obj.start_time_iter = time.time()
    theta_star_opt, info_opt = nlp.solve(theta_mean.copy())
    total_opt_time = time.time() - ldt_problem_obj.start_time_iter
    
    print(f"IPOPT optimization took {total_opt_time:.2f}s using {simulation_count_global} simulations (for IPOPT).")
    print(f"IPOPT final status: {info_opt['status_msg']} (code: {info_opt['status']})")
    print(f"Optimal objective I(theta*): {info_opt['obj_val']:.6f}")

    simulation_cache.clear() # Clear cache before final verification and SORM
    current_sim_count_before_verify = simulation_count_global

    print("Running final simulation with optimized theta* for verification...")
    theta_star_final = np.clip(np.atleast_1d(theta_star_opt), 0.01, 0.99)
    final_results_dict = run_simulation(theta_star_final, base_psys_main, config_main, caller_info="Final_Verification")
    F_at_theta_star = final_results_dict.get("cost", np.inf)
    I_at_theta_star = rate_function_I(theta_star_final, theta_mean, inv_covariance_diag) # Use final clipped theta
    
    print(f"\nLDT Result Verification:")
    print(f"  Target z:               {Z_THRESHOLD:.6f}")
    print(f"  Found theta*:           {theta_star_final[:min(5,num_params)]}...")
    print(f"  Achieved F(theta*):     {F_at_theta_star:.6f}")
    constraint_violation = abs(F_at_theta_star - Z_THRESHOLD) if np.isfinite(F_at_theta_star) else np.inf
    print(f"  Constraint Viol. (abs): {constraint_violation:.3e}")
    print(f"  I(theta*):              {I_at_theta_star:.6f} (IPOPT obj: {info_opt['obj_val']:.6f})")

    lambda_LDT_val = np.nan
    if 'mult_g' in info_opt and len(info_opt['mult_g']) > 0:
        mu_F_val = info_opt['mult_g'][0]
        lambda_LDT_val = -mu_F_val # Since constraint is F-z=0 and KKT is grad_I + mu_F * grad_F = 0
                                  # lambda_LDT for I - lambda_LDT * F means grad_I = lambda_LDT * grad_F
        print(f"  Lagrange mult mu_F:     {mu_F_val:.4e}")
        print(f"  Effective lambda_LDT:   {lambda_LDT_val:.4e}")

    P_form_val = calculate_form_probability(I_at_theta_star)
    print(f"  P_FORM(F>=z) ~          {P_form_val:.4e}")

    # --- SORM Probability (using explicit Hessian) ---
    P_sorm_val = np.nan
    sorm_spectrum = np.array([])
    sorm_eigvals_used = np.array([])

    if info_opt['status'] >= 0 and constraint_violation < 1e-4 and np.isfinite(lambda_LDT_val):
        sorm_start_time = time.time()
        current_sim_count_before_sorm = simulation_count_global
        
        P_sorm_val, sorm_eigvals_used, sorm_spectrum = calculate_sorm_probability_explicit_hessian(
            theta_star_sorm=theta_star_final,
            I_theta_star_sorm=I_at_theta_star,
            lambda_LDT_sorm=lambda_LDT_val,
            P_form_sorm=P_form_val,
            base_psys_sorm=base_psys_main,
            config_sorm=config_main,
            theta_mean_sorm=theta_mean,
            theta_std_dev_sorm=theta_std_dev,
            num_params_sorm=num_params,
            epsilon_hess_sorm=FINITE_DIFF_EPSILON
        )
        sorm_time = time.time() - sorm_start_time
        sorm_sim_count = simulation_count_global - current_sim_count_before_sorm
        print(f"SORM (explicit Hessian) calculation took {sorm_time:.2f}s using {sorm_sim_count} simulations.")
        print(f"  P_SORM_expl(F>=z) ~     {P_sorm_val:.4e}")
        if sorm_spectrum.size > 0:
             print(f"  Full spectrum of M_sub (first 10): {sorm_spectrum[:min(10,len(sorm_spectrum))]}")
        if sorm_eigvals_used.size > 0:
             print(f"  Eigenvalues used in SORM product (first 10): {sorm_eigvals_used[:min(10,len(sorm_eigvals_used))]}")
    else:
        print("\nSORM calculation (explicit Hessian) skipped due to IPOPT status, constraint violation, or unavailable lambda_LDT.")

    if info_opt['status'] < 0 or constraint_violation > 1e-5 :
         print("\nWarning: IPOPT may not have converged optimally or constraint violation is high.")

    print(f"\nTotal simulations for optimization (from IPOPT obj/jac calls): "
          f"C:{ldt_problem_obj.eval_count_con}, J:{ldt_problem_obj.eval_count_jac}")
    print(f"Total global simulation_count (includes verification & SORM HVP): {simulation_count_global}")