# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import cyipopt
import copy
import os
import sys
import time
import multiprocessing
from typing import Dict, Tuple, Any, List, Optional

# --- UQGrid Imports (assuming they are accessible) ---
try:
    from uqgrid.core.psydef import Psystem
    from uqgrid.simulation.dynamics import integrate_system
    from uqgrid.io.parse import load_psse, add_dyr
    from uqgrid.simulation.config import IntegrationConfig
except ImportError as e:
    print(f"Error importing uqgrid: {e}. Please ensure uqgrid is installed and accessible.")
    sys.exit(1)

# --- Import functions from your previous script ---
# Ensure 'ldt_constrained_SORM.py' is in the same directory or PYTHONPATH
try:
    import ldt_constrained_SORM as ldt_sorm_module
except ImportError:
    print("Error: Could not import 'ldt_constrained_SORM.py'. Make sure it's in the current directory or PYTHONPATH.")
    sys.exit(1)

# --- Script Configuration ---
# System and Fault Parameters (should match those used for LDT/SORM consistency)
RAW_FILE_PATH = "../data/ieee9_v33.raw" # Example, adjust as needed
DYR_FILE_PATH = "../data/ieee9bus_gov.dyr"
FAULT_BUS_IDX = 1  # External Bus ID
FAULT_Z = 0.01
FAULT_ON_TIME = 0.25
FAULT_OFF_TIME = 0.40
INTEGRATION_TEND = 5.0
INTEGRATION_DT = 1.0 / 120.0

# Z_THRESHOLDS to test (start with values that might yield non-zero MC results)
# Adjust these based on your system and expected probabilities
Z_THRESHOLDS_TO_TEST = np.array([2.5e-4, 3.0e-4, 3.5e-4, 4.0e-4, 4.5e-4]) # Example values

# Monte Carlo Configuration
N_MC_SAMPLES = 500  # Number of Monte Carlo samples per Z_THRESHOLD
N_PROCESSES_MC = max(1, multiprocessing.cpu_count() - 2) # Number of parallel processes for MC

# LDT/SORM Configuration (can be imported or redefined if needed)
I_THRESHOLD_LDT = ldt_sorm_module.I_THRESHOLD # Min I(theta*) for FORM
FINITE_DIFF_EPSILON_SORM = ldt_sorm_module.FINITE_DIFF_EPSILON
SORM_STABILITY_TOL_SORM = ldt_sorm_module.SORM_STABILITY_TOL

# --- Helper function for MC worker ---
# This version of run_simulation is simplified for MC: no sensitivities, no caching by default
# It uses a distinct base_psys and config to avoid interference
def run_simulation_for_mc_worker(args: Tuple[np.ndarray, str, str, int, float, float, float, float, float]) -> Optional[float]:
    """
    Worker function for Monte Carlo simulation.
    Runs a single simulation without sensitivities.
    Returns the cost F(theta), or np.inf on failure.
    """
    theta_sample, raw_path_mc, dyr_path_mc, fault_bus_mc, fault_z_mc, \
    ton_mc, toff_mc, tend_mc, dt_mc = args

    try:
        # Minimal psys setup for this single run
        psys_mc = ldt_sorm_module.load_psse(raw_filename=raw_path_mc)
        ldt_sorm_module.add_dyr(psys_mc, dyr_path_mc)
        internal_fault_bus_mc = psys_mc.ext2int[int(fault_bus_mc)]
        psys_mc.add_busfault(internal_fault_bus_mc, fault_z_mc, 0.01) # fault_x is hardcoded in setup_system

        config_mc = IntegrationConfig(
            tend=tend_mc, dt=dt_mc,
            ton=ton_mc, toff=toff_mc,
            power_injection=False, verbose=False, # MC should be quiet
            comp_sens=True,
            fsolve=False, petsc=True
        )
        
        theta_clipped_mc = np.clip(np.atleast_1d(theta_sample).flatten(), 0.01, 0.99)
        psys_mc.set_load_parameters(theta_clipped_mc)
        psys_mc.createYbusComplex()
        results_mc = ldt_sorm_module.integrate_system(psys_mc, config_mc)
        cost = results_mc.get("cost")
        
        if cost is not None and np.isfinite(cost):
            return cost.item() if isinstance(cost, np.ndarray) else cost
        return np.inf # Indicate failure
    except Exception:
        # print(f"MC Worker Exception: {e} for theta {theta_sample[:2]}")
        return np.inf

def run_monte_carlo_parallel(
    z_target: float,
    n_samples: int,
    theta_mean_mc: np.ndarray,
    theta_std_dev_mc: np.ndarray,
    num_processes: int,
    base_raw_path: str, base_dyr_path: str, base_fault_bus: int, base_fault_z: float,
    mc_ton: float, mc_toff: float, mc_tend: float, mc_dt: float
) -> Tuple[float, float, float]:
    """
    Runs Monte Carlo simulation in parallel to estimate P(F(theta) >= z_target).
    Returns estimated probability, and lower/upper bounds of Wilson score interval.
    """
    print(f"\n--- Starting Monte Carlo for Z_THRESHOLD = {z_target:.4e} with {n_samples} samples ---")
    start_time_mc = time.time()

    # Generate theta samples from the prior Gaussian distribution
    num_params_mc = len(theta_mean_mc)
    theta_samples = np.random.normal(loc=theta_mean_mc, scale=theta_std_dev_mc, size=(n_samples, num_params_mc))
    
    # Prepare arguments for worker function
    worker_args = [
        (
            theta_samples[i, :], base_raw_path, base_dyr_path, base_fault_bus, base_fault_z,
            mc_ton, mc_toff, mc_tend, mc_dt
        ) for i in range(n_samples)
    ]

    costs_f_theta = []
    # Use try-finally to ensure pool is closed
    pool = None 
    try:
        if num_processes > 1:
            pool = multiprocessing.Pool(processes=num_processes)
            # Using map and handling results
            chunksize = max(1, n_samples // (num_processes * 4)) # Heuristic for chunksize
            print(f"MC: Distributing {n_samples} tasks to {num_processes} processes with chunksize ~{chunksize}...")
            costs_f_theta = pool.map(run_simulation_for_mc_worker, worker_args, chunksize=chunksize)
        else: # Serial execution for debugging or if num_processes = 1
            print(f"MC: Running {n_samples} tasks serially...")
            for i, arg_set in enumerate(worker_args):
                costs_f_theta.append(run_simulation_for_mc_worker(arg_set))
                if (i+1) % (n_samples // 10 if n_samples >=10 else 1) == 0:
                    print(f"  MC serial progress: {i+1}/{n_samples}")

    except Exception as e_pool:
        print(f"Error during MC parallel execution: {e_pool}")
    finally:
        if pool:
            pool.close()
            pool.join()
            
    valid_costs = [c for c in costs_f_theta if c is not None and np.isfinite(c)]
    num_valid_sims = len(valid_costs)
    
    if num_valid_sims < n_samples * 0.8: # If too many simulations failed
        print(f"Warning MC: Only {num_valid_sims}/{n_samples} simulations were successful.")
    if num_valid_sims == 0:
        print("Warning MC: All Monte Carlo simulations failed. Probability is undefined.")
        return np.nan, np.nan, np.nan

    num_exceeding_z = np.sum(np.array(valid_costs) >= z_target)
    prob_mc = num_exceeding_z / num_valid_sims

    # Wilson score interval for binomial proportion
    # (https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval)
    n = num_valid_sims
    p_hat = prob_mc
    z_alpha_half = 1.96  # For 95% confidence
    
    denominator = 1 + (z_alpha_half**2 / n)
    center_adjusted_p = p_hat + (z_alpha_half**2 / (2 * n))
    term = z_alpha_half * np.sqrt((p_hat * (1 - p_hat) / n) + (z_alpha_half**2 / (4 * n**2)))
    
    lower_bound = (center_adjusted_p - term) / denominator
    upper_bound = (center_adjusted_p + term) / denominator
    
    # Ensure bounds are within [0, 1]
    lower_bound = max(0, lower_bound)
    upper_bound = min(1, upper_bound)

    end_time_mc = time.time()
    print(f"MC for Z={z_target:.4e}: Hits={num_exceeding_z}/{num_valid_sims}, Prob={prob_mc:.4e}, "
          f"CI=[{lower_bound:.4e}, {upper_bound:.4e}], Time={end_time_mc - start_time_mc:.2f}s")
    print(f"--- Monte Carlo Finished for Z_THRESHOLD = {z_target:.4e} ---")
    return prob_mc, lower_bound, upper_bound


# --- Main Script Logic ---
if __name__ == "__main__":
    all_results = [] # To store results for each Z_THRESHOLD

    # --- Parameter Distribution Setup (once) ---
    try:
        _psys_temp = ldt_sorm_module.load_psse(raw_filename=RAW_FILE_PATH)
        num_params = _psys_temp.nloads
        print(f"Detected {num_params} loads (parameters).")
        if num_params == 0: print("Error: No loads found."); sys.exit(1)
        del _psys_temp
    except Exception as e:
        print(f"Error loading system for num_params: {e}"); sys.exit(1)

    theta_mean_main = np.full(num_params, 0.5)
    theta_std_dev_main = np.full(num_params, 0.1)
    inv_covariance_diag_main = 1.0 / (theta_std_dev_main**2)

    # --- Base System and Config for LDT/SORM (once) ---
    base_psys_ldt_sorm = ldt_sorm_module.setup_system(
        RAW_FILE_PATH, DYR_FILE_PATH, FAULT_BUS_IDX, FAULT_Z
    )
    config_ldt_sorm = IntegrationConfig(
        tend=INTEGRATION_TEND, dt=INTEGRATION_DT,
        ton=FAULT_ON_TIME, toff=FAULT_OFF_TIME,
        power_injection=False, verbose=False,
        comp_sens=True, fsolve=False, petsc=True
    )

    for z_val in Z_THRESHOLDS_TO_TEST:
        print(f"\n\n======================================================================")
        print(f"Processing Z_THRESHOLD = {z_val:.6e}")
        print(f"======================================================================")
        
        current_results = {"Z": z_val, "P_FORM": np.nan, "P_SORM": np.nan,
                           "P_MC": np.nan, "MC_CI_low": np.nan, "MC_CI_high": np.nan,
                           "I_theta_star": np.nan, "opt_time": np.nan, "sorm_time": np.nan,
                           "mc_time": np.nan, "opt_status": -99}

        # --- 1. LDT Optimization ---
        print(f"\n--- LDT Optimization for Z = {z_val:.4e} ---")
        ldt_sorm_module.simulation_cache.clear()
        ldt_sorm_module.simulation_count_global = 0 # Reset counter in imported module

        ldt_problem_ipopt = ldt_sorm_module.LDTProblem(
            base_psys_ldt_sorm, config_ldt_sorm, theta_mean_main,
            inv_covariance_diag_main, z_val
        )
        lb_ipopt = [0.01] * num_params; ub_ipopt = [0.99] * num_params
        cl_ipopt = [0.0]; cu_ipopt = [0.0]
        nlp_ipopt = cyipopt.Problem(
            n=num_params, m=1, problem_obj=ldt_problem_ipopt,
            lb=lb_ipopt, ub=ub_ipopt, cl=cl_ipopt, cu=cu_ipopt
        )
        nlp_ipopt.add_option('max_iter', 150) # Increased iterations
        nlp_ipopt.add_option('tol', 1e-7)    # Standard tolerance
        nlp_ipopt.add_option('hessian_approximation', 'limited-memory')
        nlp_ipopt.add_option('limited_memory_max_history', 20)
        nlp_ipopt.add_option('print_level', 0) # Suppress IPOPT's own prints

        opt_start_time = time.time()
        ldt_problem_ipopt.start_time_iter = opt_start_time # For intermediate callback
        
        theta_star_ipopt, info_ipopt = nlp_ipopt.solve(theta_mean_main.copy())
        current_results["opt_time"] = time.time() - opt_start_time
        current_results["opt_status"] = info_ipopt['status']
        print(f"IPOPT for Z={z_val:.4e} took {current_results['opt_time']:.2f}s. Status: {info_ipopt['status_msg']}")

        I_theta_star_val = info_ipopt['obj_val']
        current_results["I_theta_star"] = I_theta_star_val
        
        # Verification of F(theta_star)
        ldt_sorm_module.simulation_cache.clear() # Clear before verification
        theta_star_final_ipopt = np.clip(np.atleast_1d(theta_star_ipopt), 0.01, 0.99)
        final_results_ipopt = ldt_sorm_module.run_simulation(
            theta_star_final_ipopt, base_psys_ldt_sorm, config_ldt_sorm, caller_info="LDT_Verify"
        )
        F_at_theta_star_ipopt = final_results_ipopt.get("cost", np.inf)
        constraint_violation_ipopt = abs(F_at_theta_star_ipopt - z_val) if np.isfinite(F_at_theta_star_ipopt) else np.inf
        print(f"  Verified: I(theta*)={I_theta_star_val:.4f}, F(theta*)={F_at_theta_star_ipopt:.6e}, Constraint Viol={constraint_violation_ipopt:.3e}")

        # --- 2. FORM Probability ---
        P_form_val = ldt_sorm_module.calculate_form_probability(I_theta_star_val)
        current_results["P_FORM"] = P_form_val
        print(f"  P_FORM = {P_form_val:.4e}")

        # --- 3. SORM Probability ---
        lambda_LDT_ipopt = np.nan
        if 'mult_g' in info_ipopt and len(info_ipopt['mult_g']) > 0:
            mu_F_ipopt = info_ipopt['mult_g'][0]
            lambda_LDT_ipopt = -mu_F_ipopt
        
        if info_ipopt['status'] >= 0 and constraint_violation_ipopt < 1e-4 and np.isfinite(lambda_LDT_ipopt):
            print(f"\n--- SORM Calculation for Z = {z_val:.4e} ---")
            ldt_sorm_module.simulation_cache.clear() # Clear before Hessian computation
            ldt_sorm_module.simulation_count_global = 0 # Reset for SORM specific count

            sorm_start_time = time.time()
            P_sorm_val, _, _ = ldt_sorm_module.calculate_sorm_probability_explicit_hessian(
                theta_star_sorm=theta_star_final_ipopt,
                I_theta_star_sorm=I_theta_star_val,
                lambda_LDT_sorm=lambda_LDT_ipopt,
                P_form_sorm=P_form_val,
                base_psys_sorm=base_psys_ldt_sorm,
                config_sorm=config_ldt_sorm,
                theta_mean_sorm=theta_mean_main,
                theta_std_dev_sorm=theta_std_dev_main,
                num_params_sorm=num_params,
                epsilon_hess_sorm=FINITE_DIFF_EPSILON_SORM
            )
            current_results["sorm_time"] = time.time() - sorm_start_time
            current_results["P_SORM"] = P_sorm_val
            print(f"SORM for Z={z_val:.4e} took {current_results['sorm_time']:.2f}s. P_SORM = {P_sorm_val:.4e}")
        else:
            print(f"SORM calculation skipped for Z={z_val:.4e} due to IPOPT status/constraint/lambda.")
            current_results["P_SORM"] = np.nan
            current_results["sorm_time"] = 0.0
            
        # --- 4. Monte Carlo Simulation ---
        mc_start_time = time.time()
        P_mc_val, mc_low, mc_high = run_monte_carlo_parallel(
            z_target=z_val,
            n_samples=N_MC_SAMPLES,
            theta_mean_mc=theta_mean_main,
            theta_std_dev_mc=theta_std_dev_main,
            num_processes=N_PROCESSES_MC,
            base_raw_path=RAW_FILE_PATH, base_dyr_path=DYR_FILE_PATH,
            base_fault_bus=FAULT_BUS_IDX, base_fault_z=FAULT_Z,
            mc_ton=FAULT_ON_TIME, mc_toff=FAULT_OFF_TIME,
            mc_tend=INTEGRATION_TEND, mc_dt=INTEGRATION_DT
        )
        current_results["mc_time"] = time.time() - mc_start_time
        current_results["P_MC"] = P_mc_val
        current_results["MC_CI_low"] = mc_low
        current_results["MC_CI_high"] = mc_high
        
        all_results.append(current_results)

    # --- Plotting Results ---
    print("\n\n--- Plotting Results ---")
    Zs = np.array([res["Z"] for res in all_results])
    Is = np.array([res["I_theta_star"] for res in all_results])
    P_forms = np.array([res["P_FORM"] for res in all_results])
    P_sorms = np.array([res["P_SORM"] for res in all_results])
    P_mcs = np.array([res["P_MC"] for res in all_results])
    mc_cis_low = np.array([res["MC_CI_low"] for res in all_results])
    mc_cis_high = np.array([res["MC_CI_high"] for res in all_results])

    # Filter out NaNs for plotting (e.g., if SORM failed for some Z)
    valid_sorm_mask = np.isfinite(P_sorms)
    valid_mc_mask = np.isfinite(P_mcs)
    
    # Calculate MC error bars for plotting
    # yerr_lower = P_mcs[valid_mc_mask] - mc_cis_low[valid_mc_mask]
    # yerr_upper = mc_cis_high[valid_mc_mask] - P_mcs[valid_mc_mask]
    # mc_error_bars = np.array([yerr_lower, yerr_upper])
    # Ensure error bars are non-negative
    yerr_lower = np.maximum(0, P_mcs[valid_mc_mask] - mc_cis_low[valid_mc_mask])
    yerr_upper = np.maximum(0, mc_cis_high[valid_mc_mask] - P_mcs[valid_mc_mask])
    # Clip error bars if P_MC is 0, so lower bound doesn't go negative on log plot
    yerr_lower[P_mcs[valid_mc_mask] == 0] = 0 # Effectively, error bar starts at 0
    # For log plot, if P_MC is 0, we can't plot it directly with error bars.
    # We can plot the upper CI bound as a point, or skip plotting 0-prob MC.
    # For now, let's plot non-zero MC points with error bars.
    
    plot_mc_mask = valid_mc_mask & (P_mcs > 0) # Only plot MC if probability is > 0 for log scale

    plt.figure(figsize=(10, 7))
    plt.plot(Zs, P_forms, 'o-', label='FORM Probability', color='blue')
    if np.any(valid_sorm_mask):
        plt.plot(Zs[valid_sorm_mask], P_sorms[valid_sorm_mask], 's--', label='SORM Probability (Explicit Hessian)', color='green')
    
    if np.any(plot_mc_mask):
        plt.errorbar(Zs[plot_mc_mask], P_mcs[plot_mc_mask], 
                     yerr=[yerr_lower[P_mcs[valid_mc_mask]>0], yerr_upper[P_mcs[valid_mc_mask]>0]],
                     fmt='x', label=f'Monte Carlo ({N_MC_SAMPLES} samples, 95% CI)', color='red', capsize=5, elinewidth=1, markeredgewidth=1)
    
    # Plot MC points that were 0 as a different marker at a very low value (e.g., bottom of y-axis) or just their upper CI
    zero_mc_mask = valid_mc_mask & (P_mcs == 0)
    if np.any(zero_mc_mask):
        # For log scale, can't plot 0. Plot upper CI bound.
        # Or plot at a fixed low y-value if upper CI is also 0 or too small.
        # Let's plot the upper CI bound if it's > 0
        upper_ci_for_zero_mc = mc_cis_high[zero_mc_mask]
        plot_upper_ci_mask = upper_ci_for_zero_mc > 1e-18 # Arbitrary small number for plotting
        if np.any(plot_upper_ci_mask):
             plt.scatter(Zs[zero_mc_mask][plot_upper_ci_mask], upper_ci_for_zero_mc[plot_upper_ci_mask],
                        marker='v', color='red', label=f'MC=0 (Upper CI shown)', s=50, alpha=0.7)


    plt.yscale('log')
    plt.xlabel('Z_THRESHOLD (Constraint Value F(theta))')
    plt.ylabel('Probability P(F(theta) >= Z_THRESHOLD)')
    plt.title(f'Comparison of FORM, SORM, and Monte Carlo ({os.path.basename(RAW_FILE_PATH)})')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"probability_comparison_{os.path.basename(RAW_FILE_PATH).split('.')[0]}.png")
    print(f"\nPlot saved to probability_comparison_{os.path.basename(RAW_FILE_PATH).split('.')[0]}.png")
    plt.show()

    # Print summary table
    print("\n--- Summary Table ---")
    print(f"{'Z_THRESHOLD':<12} | {'I(theta*)':<10} | {'P_FORM':<12} | {'P_SORM':<12} | {'P_MC':<12} | {'MC_CI_Low':<12} | {'MC_CI_High':<12} | {'OptTime':<8} | {'SORMTime':<9} | {'MCTime':<8}")
    print("-" * 120)
    for res in all_results:
        print(f"{res['Z']:.4e} | {res['I_theta_star']:.3f}    | {res['P_FORM']:.4e} | {res['P_SORM']:.4e} | {res['P_MC']:.4e} | {res['MC_CI_low']:.4e} | {res['MC_CI_high']:.4e} | {res['opt_time']:.1f}s   | {res['sorm_time']:.1f}s    | {res['mc_time']:.1f}s")

