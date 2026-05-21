#!/usr/bin/env python
"""
Utility Functions for Visualizing TSI (Transient Stability Index) and data statistics.

This module provides convenient functions for loading and visualizing TSI data
from power grid simulation datasets. It creates publication-quality histograms
showing the distribution of TSI values across scenarios and fault conditions.

The TSI (Transient Stability Index) measures power system stability following
disturbances:
- TSI > 0 : Stable system response
- TSI < 0 : Unstable system response
- TSI = 100 : Maximum stability (no rotor angle deviation)
- TSI = -100 : Severe instability

Features
--------
- Load TSI datasets from compressed NumPy archives (.npz)
- Plot aggregate histograms across all scenarios and fault conditions
- Plot per-scenario histograms for detailed analysis
- Automatic statistics annotation (mean, std, sample counts)
- Optional figure saving with customizable output paths
- Display comprehensive dataset information including power variable statistics

Data Format
-----------
Expected input file structure (from export_probml_dataset):
    - Y : ndarray (N, F, Z)
        TSI values at the last time step where:
        - N = number of operating condition samples
        - F = number of fault locations
        - Z = number of fault impedance values
    - X : ndarray (N, 2, Ngen+Nload) when concat_generators_and_loads=True
        Input features where:
        - X[:, 0, :Ngen] = pg (generator active power)
        - X[:, 1, :Ngen] = qg (generator reactive power)
        - X[:, 0, Ngen:] = pl (load active power)
        - X[:, 1, Ngen:] = ql (load reactive power)
    - X_gen : ndarray (N, 2, Ngen) when concat_generators_and_loads=False
    - X_load : ndarray (N, 2, Nload) when concat_generators_and_loads=False

Usage
-----
Command-line execution with input file::

    $ python TSI_histogram_utils.py my_dataset.npz

    # Display info for a specific scenario
    $ python TSI_histogram_utils.py my_dataset.npz -s 5

    # Generate histograms
    $ python TSI_histogram_utils.py my_dataset.npz --histogram

    # Generate histograms without interactive display
    $ python TSI_histogram_utils.py my_dataset.npz --histogram --no-show

    # Show per-unit statistics
    $ python TSI_histogram_utils.py my_dataset.npz --per-unit

    # Full analysis
    $ python TSI_histogram_utils.py my_dataset.npz -s 0 --histogram --per-unit

    # Show help
    $ python TSI_histogram_utils.py --help

Programmatic usage::

    from TSI_histogram_utils import plot_histogram_all_samples, plot_histogram_single_scenario

    # Plot histogram of all TSI values in the dataset
    fig1 = plot_histogram_all_samples(
        "tsi_probml_fullinputs.npz",
        save_path="all_tsi_histogram.png"
    )

    # Plot histogram for a specific operating condition (scenario)
    fig2 = plot_histogram_single_scenario(
        scenario_idx=5,
        filepath="tsi_probml_fullinputs.npz",
        save_path="scenario_5_histogram.png"
    )

    # Display comprehensive dataset information
    display_dataset_info("tsi_probml_fullinputs.npz")

    # Display info for a specific scenario
    display_dataset_info("tsi_probml_fullinputs.npz", scenario_idx=5)

Output Files
------------
When save_path is specified:
- histogram_all_tsi.png : Distribution of all TSI values
- histogram_scenario_N.png : Distribution for scenario N

Dependencies
------------
- numpy : Data loading and numerical operations
- matplotlib : Histogram plotting and figure generation

See Also
--------
- TSI_analysis.py : Generates the TSI datasets visualized by this module
- export_probml_dataset() : Creates the .npz files consumed by these functions

Examples
--------
Analyze stability distribution across a simulation campaign:

    >>> fig = plot_histogram_all_samples("my_simulation.npz")
    >>> # Check if most scenarios are stable (TSI > 0)
    >>> plt.show()

Compare stability for different operating conditions::

    >>> for scenario_idx in [0, 10, 20]:
    ...     fig = plot_histogram_single_scenario(scenario_idx, "my_simulation.npz")
    ...     plt.show()

Display dataset statistics::

    >>> display_dataset_info("my_simulation.npz")
    >>> display_dataset_info("my_simulation.npz", scenario_idx=10)

"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Tuple
import time
from pathlib import Path
import scipy.io as scio

# =============================================================================
# Data Loading
# =============================================================================

def load_tsi_data(filepath: str = "tsi_probml_fullinputs.npz") -> dict:
    """
    Load TSI dataset from a compressed NumPy archive (.npz) file.

    Parameters
    ----------
    filepath : str, default='tsi_probml_fullinputs.npz'
        Path to the .npz file containing TSI data. Expected to contain
        at minimum a 'Y' array with TSI values.

    Returns
    -------
    dict
        Dictionary containing all arrays from the .npz file. Keys typically
        include:
        - 'Y' : TSI values array (N, F, Z)
        - 'X' or 'X_flat' : Input features (if saved)
        - 'sample_idx' : Sample indices
        - 'fault_locations' : Fault location bus numbers
        - 'fault_impedances' : Fault impedance values
        - 'meta' : Metadata dictionary

    Notes
    -----
    Uses allow_pickle=True to load metadata dictionaries stored as
    object arrays. The returned dictionary provides direct access to
    all arrays without keeping the file handle open.

    Examples
    --------
    >>> data = load_tsi_data("my_dataset.npz")
    >>> Y = data["Y"]
    >>> print(f"Dataset shape: {Y.shape}")
    Dataset shape: (1000, 50, 3)
    """
    # Load with allow_pickle for metadata object arrays
    data = np.load(filepath, allow_pickle=True)

    # Convert to regular dict to close file handle and enable easy access
    return {key: data[key] for key in data.files}


# =============================================================================
# Data Extraction Utilities
# =============================================================================

def extract_power_variables(
    data: dict,
    scenario_idx: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Extract pg, qg, pl, ql power variables from the dataset.

    This function handles both concatenated and separate storage modes
    based on the metadata in the dataset.

    Parameters
    ----------
    data : dict
        Dictionary containing dataset arrays, as returned by load_tsi_data().
    scenario_idx : int, optional
        If provided, extract variables for a single scenario only.
        If None, extract for all scenarios.

    Returns
    -------
    dict
        Dictionary with keys 'pg', 'qg', 'pl', 'ql', each containing
        the corresponding power values as numpy arrays.
        - If scenario_idx is None: shape is (N, Ngen) or (N, Nload)
        - If scenario_idx is provided: shape is (Ngen,) or (Nload,)

    Raises
    ------
    ValueError
        If the dataset does not contain the expected X arrays or metadata.
    IndexError
        If scenario_idx is out of bounds.

    Notes
    -----
    The data layout depends on the concat_generators_and_loads setting
    used during dataset creation:

    When concat_generators_and_loads=True (default):
        - X shape: (N, 2, Ngen+Nload)
        - X[:, 0, :Ngen] = pg (generator active power)
        - X[:, 1, :Ngen] = qg (generator reactive power)
        - X[:, 0, Ngen:] = pl (load active power)
        - X[:, 1, Ngen:] = ql (load reactive power)

    When concat_generators_and_loads=False:
        - X_gen shape: (N, 2, Ngen) with X_gen[:, 0, :] = pg, X_gen[:, 1, :] = qg
        - X_load shape: (N, 2, Nload) with X_load[:, 0, :] = pl, X_load[:, 1, :] = ql

    Examples
    --------
    >>> data = load_tsi_data("tsi_data.npz")
    >>> powers = extract_power_variables(data)
    >>> print(f"pg shape: {powers['pg'].shape}")

    >>> # For a single scenario
    >>> powers_s0 = extract_power_variables(data, scenario_idx=0)
    >>> print(f"pg for scenario 0: {powers_s0['pg']}")
    """
    # Get metadata
    if "meta" not in data:
        raise ValueError("Dataset does not contain 'meta' field")

    meta = data["meta"]
    if isinstance(meta, np.ndarray):
        meta = meta.item() if meta.ndim == 0 else meta[0]

    Ngen = meta.get("Ngen")
    Nload = meta.get("Nload")
    concat_mode = meta.get("concat_generators_and_loads", True)

    if Ngen is None or Nload is None:
        raise ValueError("Metadata missing 'Ngen' or 'Nload' fields")

    # Extract based on storage mode
    if concat_mode:
        if "X" not in data:
            raise ValueError("Dataset missing 'X' array (expected for concatenated mode)")

        X = data["X"]  # Shape: (N, 2, Ngen+Nload)
        N = X.shape[0]

        # Validate scenario_idx
        if scenario_idx is not None:
            if scenario_idx < 0 or scenario_idx >= N:
                raise IndexError(
                    f"scenario_idx {scenario_idx} out of bounds. Valid range: [0, {N-1}]"
                )
            # Extract for single scenario
            pg = X[scenario_idx, 0, :Ngen]
            qg = X[scenario_idx, 1, :Ngen]
            pl = X[scenario_idx, 0, Ngen:]
            ql = X[scenario_idx, 1, Ngen:]
        else:
            # Extract for all scenarios
            pg = X[:, 0, :Ngen]
            qg = X[:, 1, :Ngen]
            pl = X[:, 0, Ngen:]
            ql = X[:, 1, Ngen:]

    else:
        # Separate storage mode
        if "X_gen" not in data or "X_load" not in data:
            raise ValueError(
                "Dataset missing 'X_gen' or 'X_load' arrays "
                "(expected for separate storage mode)"
            )

        X_gen = data["X_gen"]    # Shape: (N, 2, Ngen)
        X_load = data["X_load"]  # Shape: (N, 2, Nload)
        N = X_gen.shape[0]

        # Validate scenario_idx
        if scenario_idx is not None:
            if scenario_idx < 0 or scenario_idx >= N:
                raise IndexError(
                    f"scenario_idx {scenario_idx} out of bounds. Valid range: [0, {N-1}]"
                )
            # Extract for single scenario
            pg = X_gen[scenario_idx, 0, :]
            qg = X_gen[scenario_idx, 1, :]
            pl = X_load[scenario_idx, 0, :]
            ql = X_load[scenario_idx, 1, :]
        else:
            # Extract for all scenarios
            pg = X_gen[:, 0, :]
            qg = X_gen[:, 1, :]
            pl = X_load[:, 0, :]
            ql = X_load[:, 1, :]

    return {"pg": pg, "qg": qg, "pl": pl, "ql": ql}


def compute_variable_statistics(arr: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive statistics for a numeric array.

    Parameters
    ----------
    arr : np.ndarray
        Input array (can be multi-dimensional, will be flattened).

    Returns
    -------
    dict
        Dictionary containing:
        - 'min': Minimum value
        - 'max': Maximum value
        - 'range': max - min
        - 'mean': Arithmetic mean
        - 'median': Median value
        - 'std': Standard deviation
        - 'q25': 25th percentile
        - 'q75': 75th percentile
        - 'count': Number of valid (non-NaN) elements

    Examples
    --------
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> stats = compute_variable_statistics(arr)
    >>> print(f"Mean: {stats['mean']}, Std: {stats['std']}")
    """
    flat = arr.flatten()
    valid = flat[~np.isnan(flat)]

    if len(valid) == 0:
        return {
            "min": np.nan, "max": np.nan, "range": np.nan,
            "mean": np.nan, "median": np.nan, "std": np.nan,
            "q25": np.nan, "q75": np.nan, "count": 0
        }

    return {
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "range": float(np.max(valid) - np.min(valid)),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "std": float(np.std(valid)),
        "q25": float(np.percentile(valid, 25)),
        "q75": float(np.percentile(valid, 75)),
        "count": len(valid)
    }


# =============================================================================
# Dataset Information Display
# =============================================================================

def display_dataset_info(
    filepath: str = "tsi_probml_fullinputs.npz",
    scenario_idx: Optional[int] = None,
    print_output: bool = True
) -> Dict[str, Any]:
    """
    Display comprehensive information about a TSI dataset.

    This function provides a complete overview of the dataset including:
    - All metadata fields from the 'meta' dictionary
    - Array shapes and data types for all stored arrays
    - Statistics for Y (TSI values): range, mean, median, std
    - Statistics for power variables (pg, qg, pl, ql) across all scenarios
    - Optionally, statistics for a specific scenario

    Parameters
    ----------
    filepath : str, default='tsi_probml_fullinputs.npz'
        Path to the .npz file containing TSI data.
    scenario_idx : int, optional
        If provided, also display statistics for this specific scenario.
        Must be in range [0, N-1] where N is the number of scenarios.
    print_output : bool, default=True
        If True, print formatted output to stdout.
        If False, only return the info dictionary.

    Returns
    -------
    dict
        Dictionary containing all extracted information:
        - 'filepath': Input file path
        - 'arrays': Dict of {array_name: {'shape': tuple, 'dtype': str}}
        - 'meta': Metadata dictionary from the file
        - 'Y_stats': Statistics dictionary for Y values
        - 'power_stats_all': Dict with statistics for pg, qg, pl, ql (all scenarios)
        - 'power_stats_scenario': Dict with per-scenario stats (if scenario_idx provided)

    Raises
    ------
    IndexError
        If scenario_idx is out of bounds.
    FileNotFoundError
        If the specified file does not exist.

    Notes
    -----
    Power variable layout (from TSI_analysis.py export_probml_dataset):

    When concat_generators_and_loads=True (default):
        - X shape: (N, 2, Ngen+Nload)
        - Channel 0 (P): [pg_1, ..., pg_Ngen, pl_1, ..., pl_Nload]
        - Channel 1 (Q): [qg_1, ..., qg_Ngen, ql_1, ..., ql_Nload]

    When concat_generators_and_loads=False:
        - X_gen shape: (N, 2, Ngen) - channels are [pg, qg]
        - X_load shape: (N, 2, Nload) - channels are [pl, ql]

    Examples
    --------
    >>> # Display info for entire dataset
    >>> info = display_dataset_info("tsi_probml_fullinputs.npz")

    >>> # Display info including a specific scenario
    >>> info = display_dataset_info("tsi_probml_fullinputs.npz", scenario_idx=5)

    >>> # Get info without printing (for programmatic use)
    >>> info = display_dataset_info("tsi_probml_fullinputs.npz", print_output=False)
    >>> print(info['meta']['Ngen'])
    """
    # Load dataset
    data = load_tsi_data(filepath)

    # Initialize result dictionary
    result: Dict[str, Any] = {
        "filepath": filepath,
        "arrays": {},
        "meta": {},
        "Y_stats": {},
        "power_stats_all": {},
    }

    # -------------------------------------------------------------------------
    # Section 1: Array information
    # -------------------------------------------------------------------------
    for key in data.keys():
        arr = data[key]
        if isinstance(arr, np.ndarray):
            result["arrays"][key] = {
                "shape": arr.shape,
                "dtype": str(arr.dtype)
            }

    # -------------------------------------------------------------------------
    # Section 2: Metadata
    # -------------------------------------------------------------------------
    if "meta" in data:
        meta = data["meta"]
        if isinstance(meta, np.ndarray):
            meta = meta.item() if meta.ndim == 0 else meta[0]
        result["meta"] = dict(meta) if isinstance(meta, dict) else {}

    # -------------------------------------------------------------------------
    # Section 3: Y (TSI) statistics
    # -------------------------------------------------------------------------
    if "Y" in data:
        Y = data["Y"]
        fault_locations_all = data["fault_locations"]
        fault_locations, inv = np.unique(fault_locations_all, return_inverse=True)
        assert len(fault_locations) == Y.shape[1], "fault_locations length must match Y.shape[1] (number of fault locations)"

        result["Y_stats"] = compute_variable_statistics(Y)

        # Add stability breakdown
        Y_flat = Y.flatten()
        Y_valid = Y_flat[~np.isnan(Y_flat)]
        print(f"y val = {Y_valid}")
        
        if len(Y_valid) > 0:
            n_stable = np.sum(Y_valid > 0)
            n_unstable = np.sum(Y_valid < 0)
            n_marginal = len(Y_valid) - n_stable - n_unstable
            result["Y_stats"]["n_stable"] = int(n_stable)
            result["Y_stats"]["n_unstable"] = int(n_unstable)
            result["Y_stats"]["n_marginal"] = int(n_marginal)
            result["Y_stats"]["pct_stable"] = 100.0 * n_stable / len(Y_valid)
            result["Y_stats"]["pct_unstable"] = 100.0 * n_unstable / len(Y_valid)
            
            # Sum over N and Z -> per fault location (F,)
            axis = (0, 2)
            mask_vaild_Y = ~np.isnan(Y)
            mask_stable_Y   = mask_vaild_Y & (Y > 0)
            mask_unstable_Y = mask_vaild_Y & (Y < 0)
            mask_marginal_Y = mask_vaild_Y & (Y == 0)
            n_valid_f    = mask_vaild_Y.sum(axis=axis)
            n_stable_f   = mask_stable_Y.sum(axis=axis)
            n_unstable_f = mask_unstable_Y.sum(axis=axis)
            n_marginal_f = mask_marginal_Y.sum(axis=axis)

            pct_stable_f = np.where(n_valid_f > 0, 100.0 * n_stable_f / n_valid_f, np.nan)
            pct_unstable_f = np.where(n_valid_f > 0, 100.0 * n_unstable_f / n_valid_f, np.nan)
            pct_marginal_f = np.where(n_valid_f > 0, 100.0 * n_marginal_f / n_valid_f, np.nan)


            min_vals = np.nanmin(Y, axis=1)         # min tsi over fault locations
            print(f"Ymin shape {np.shape(min_vals)}")
#            print(f"Ymin {min_vals}")
#            print(f"Ymin {np.where(min_vals < 0)}")
            unstable_idx = np.where(min_vals < 0)[0]
            print(f"unstable sample index: {unstable_idx}")
            print(f"Ymin(unstable) = {min_vals[unstable_idx]}")
    
            n_stable_sample = (min_vals > 0).sum()
#            print(f"unstable_idx: {unstable_idx}")
            
            n_unstable_sample = len(unstable_idx)
            n_marginal_sample = (min_vals == 0).sum()
            n_valid_sample = n_stable_sample + n_unstable_sample + n_marginal_sample
            result["Y_stats"]["n_stable_sample"] = int(n_stable_sample)
            result["Y_stats"]["n_unstable_sample"] = int(n_unstable_sample)
            result["Y_stats"]["pct_stable_sample"] = 100.0 * n_stable_sample / n_valid_sample
            result["Y_stats"]["pct_unstable_sample"] = 100.0 * n_unstable_sample / n_valid_sample

            result["Y_stats"]["fault_locations"] =  list(fault_locations)
            result["Y_stats"]["n_stable_f"] = n_stable_f.astype(int).tolist()
            result["Y_stats"]["n_unstable_f"] = n_unstable_f.astype(int).tolist()
            result["Y_stats"]["n_marginal_f"] = n_marginal_f.astype(int).tolist()
            result["Y_stats"]["pct_stable_f"] = pct_stable_f.tolist()
            result["Y_stats"]["pct_unstable_f"] = pct_unstable_f.tolist()
            result["Y_stats"]["pct_marginal_f"] = pct_marginal_f.tolist()

    # -------------------------------------------------------------------------
    # Section 4: Power variable statistics (all scenarios)
    # -------------------------------------------------------------------------
    try:
        powers_all = extract_power_variables(data, scenario_idx=None)
        for var_name in ["pg", "qg", "pl", "ql"]:
            result["power_stats_all"][var_name] = compute_variable_statistics(
                powers_all[var_name]
            )
    except (ValueError, KeyError) as e:
        result["power_stats_all"]["error"] = str(e)

    # -------------------------------------------------------------------------
    # Section 5: Power variable statistics (single scenario, if requested)
    # -------------------------------------------------------------------------
    if scenario_idx is not None:
        result["power_stats_scenario"] = {"scenario_idx": scenario_idx}
        try:
            powers_scenario = extract_power_variables(data, scenario_idx=scenario_idx)
            for var_name in ["pg", "qg", "pl", "ql"]:
                result["power_stats_scenario"][var_name] = compute_variable_statistics(
                    powers_scenario[var_name]
                )

            # Also add Y stats for this scenario if available
            if "Y" in data:
                Y = data["Y"]
                N = Y.shape[0]
                if 0 <= scenario_idx < N:
                    Y_scenario = Y[scenario_idx, :, :]
                    result["power_stats_scenario"]["Y"] = compute_variable_statistics(
                        Y_scenario
                    )
        except (ValueError, KeyError, IndexError) as e:
            result["power_stats_scenario"]["error"] = str(e)

    # -------------------------------------------------------------------------
    # Print formatted output
    # -------------------------------------------------------------------------
    if print_output:
        _print_dataset_info(result)

    return result
    
def create_training_samples(
    filepath: str = "tsi_probml_fullinputs.npz",
    output_path: Optional[str] = None,
) -> str:
    """
    Create training samples in MATLAB format for machine learning.

    Exports simulation results as a MATLAB .mat file with features (power
    setpoints) and labels (TSI values) suitable for training ML models.

    Parameters
    ----------
    filepath : str, default='tsi_probml_fullinputs.npz'
        Path to the .npz file containing TSI data.
    output_path : str, optional
        Output .mat file path or directory. If omitted, writes a timestamped
        .mat file in the current working directory. If a directory is provided,
        writes the timestamped file inside that directory.

    Returns
    -------
    str
        Path to the written .mat file.
    """
    # Load dataset
    data = load_tsi_data(filepath)

    # Get metadata
    if "meta" not in data:
        raise ValueError("Dataset does not contain 'meta' field")

    meta = data["meta"]
    if isinstance(meta, np.ndarray):
        meta = meta.item() if meta.ndim == 0 else meta[0]

    Ngen = meta.get("Ngen")
    Nload = meta.get("Nload")
    concat_mode = meta.get("concat_generators_and_loads", True)

    if Ngen is None or Nload is None:
        raise ValueError("Metadata missing 'Ngen' or 'Nload' fields")

    if "Y" not in data:
        raise ValueError("Dataset does not contain 'Y' field")
    Y = data["Y"]  
    N_y, F, Z = Y.shape

    # Min over BOTH fault location and impedance
    Y_flat = Y.reshape(N_y, -1)                 # (N, F*Z)
    all_nan = np.all(np.isnan(Y_flat), axis=1)
    TSI_per_sample = np.nanmin(Y_flat, axis=1)
    TSI_per_sample[all_nan] = np.nan

    argmin_flat = np.zeros(N_y, dtype=int)
    valid = ~all_nan
    argmin_flat[valid] = np.nanargmin(Y_flat[valid], axis=1)

    fault_loc_idx = argmin_flat // Z
    fault_imp_idx = argmin_flat % Z

    # MATLAB-friendly 1-based indices:
    fault_loc = fault_loc_idx + 1
    fault_impedance = fault_imp_idx + 1
    sample_id = np.arange(N_y) + 1

    if concat_mode:
        if "X" not in data:
            raise ValueError("Dataset missing 'X' array (expected for concatenated mode)")

        X = data["X"]  # Shape: (N, 2, Ngen+Nload)
        if X.shape[0] != N_y:
            raise ValueError(f"Mismatch: X has N={X.shape[0]} but Y has N={N_y}")
        N = N_y

        # Extract for all samples
        pg = X[:, 0, :Ngen]
        qg = X[:, 1, :Ngen]
        pl = X[:, 0, Ngen:]
        ql = X[:, 1, Ngen:]
    else:
        # Separate storage mode
        if "X_gen" not in data or "X_load" not in data:
            raise ValueError(
                "Dataset missing 'X_gen' or 'X_load' arrays "
                "(expected for separate storage mode)"
            )

        X_gen = data["X_gen"]    # Shape: (N, 2, Ngen)
        X_load = data["X_load"]  # Shape: (N, 2, Nload)
        if X_gen.shape[0] != N_y:
            raise ValueError(f"Mismatch: X_gen has N={X_gen.shape[0]} but Y has N={N_y}")
        N = N_y

        # Extract for all scenarios
        pg = X_gen[:, 0, :]
        qg = X_gen[:, 1, :]
        pl = X_load[:, 0, :]
        ql = X_load[:, 1, :]
        
    Data = np.hstack([
        pg,                         # (N, Ngen)
        pl,                         # (N, Nload)
        ql,                         # (N, Nload)
        TSI_per_sample[:, None],    # (N, 1)
    ])
    DataMisc = np.column_stack([fault_loc, fault_impedance, sample_id])
    
    # ---- Column names (matches Data layout) ----
    col_name = (
        [f'pg_{i+1}' for i in range(Ngen)] +
        [f'pl_{i+1}' for i in range(Nload)] +
        [f'ql_{i+1}' for i in range(Nload)] +
        ['tsi']
    )


    timestamp = time.strftime("%Y%m%d_%H%M%S")
    default_filename = f"uqgrid_{N}_samples_{timestamp}.mat"
    if output_path is None:
        filename = Path(default_filename)
    else:
        output = Path(output_path)
        if output.suffix.lower() == ".mat":
            filename = output
            filename.parent.mkdir(parents=True, exist_ok=True)
        else:
            output.mkdir(parents=True, exist_ok=True)
            filename = output / default_filename

    print(f"Saving samples to {filename}")
    scio.savemat(str(filename), {
            'Data': Data,
            'DataPlus': DataMisc,
            'col_name': np.array(col_name, dtype=object)  # saves as cellstr-like
        })
    return str(filename)



def _print_dataset_info(info: Dict[str, Any]) -> None:
    """
    Print formatted dataset information to stdout.

    Parameters
    ----------
    info : dict
        Information dictionary as returned by display_dataset_info().
    """
    print("\n" + "=" * 80)
    print(f"DATASET INFORMATION: {info['filepath']}")
    print("=" * 80)

    # --- Arrays ---
    print("\n--- STORED ARRAYS ---")
    for arr_name, arr_info in info["arrays"].items():
        print(f"  {arr_name:20s} shape={str(arr_info['shape']):25s} dtype={arr_info['dtype']}")

    # --- Metadata ---
    print("\n--- METADATA ---")
    if info["meta"]:
        for key, value in info["meta"].items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
    else:
        print("  (no metadata found)")

    # --- Y Statistics ---
    print("\n--- TSI (Y) STATISTICS ---")
    if info["Y_stats"]:
        stats = info["Y_stats"]
        print(f"  Count:     {stats.get('count', 'N/A'):,}")
        print(f"  Range:     [{stats.get('min', 'N/A'):.4f}, {stats.get('max', 'N/A'):.4f}]")
        print(f"  Mean:      {stats.get('mean', 'N/A'):.4f}")
        print(f"  Median:    {stats.get('median', 'N/A'):.4f}")
        print(f"  Std:       {stats.get('std', 'N/A'):.4f}")
        print(f"  Q25:       {stats.get('q25', 'N/A'):.4f}")
        print(f"  Q75:       {stats.get('q75', 'N/A'):.4f}")
        if "n_stable" in stats:
            total_scenarios = stats['n_stable']+stats['n_unstable']+stats['n_marginal']
            total_samples = info['arrays']['X']['shape'][0]
            print(f"\n  Scenario Stability breakdown:")
            print(f"    Total Scenarios:    {total_scenarios:,}")
            print(f"    Stable (TSI > 0):   {stats['n_stable']:,} ({stats['pct_stable']:.2f}%)")
            print(f"    Unstable (TSI < 0): {stats['n_unstable']:,} ({stats['pct_unstable']:.2f}%)")
            print(f"    Marginal (TSI = 0): {stats['n_marginal']:,}")

            print("\nStability breakdown per fault location:")
            print(f"    Total Samples:    {total_samples:,}")
            print(f"    Stable (TSI > 0):   {stats['n_stable_sample']:,} ({stats['pct_stable_sample']:.2f}%)")
            print(f"    Unstable (TSI < 0): {stats['n_unstable_sample']:,} ({stats['pct_unstable_sample']:.2f}%)")
            print("Fault Loc| n_stable | n_unstable | n_marginal |  %stable  | %unstable")
            print("---------|----------|------------|------------|-----------|-----------")

            for f in range(len(stats['fault_locations'])):
                print(
                    f"{stats['fault_locations'][f]:8d} | "
                    f"{stats['n_stable_f'][f]:8d} | "
                    f"{stats['n_unstable_f'][f]:10d} | "
                    f"{stats['n_marginal_f'][f]:10d} | "
                    f"{stats['pct_stable_f'][f]:9.2f} | "
                    f"{stats['pct_unstable_f'][f]:9.2f}"
                )


    else:
        print("  (Y array not found)")

    # --- Power Variables (All Scenarios) ---
    print("\n--- POWER VARIABLE STATISTICS (ALL SCENARIOS) ---")
    if "error" in info.get("power_stats_all", {}):
        print(f"  Error: {info['power_stats_all']['error']}")
    else:
        _print_power_variable_table(info["power_stats_all"])

    # --- Power Variables (Single Scenario) ---
    if "power_stats_scenario" in info:
        scenario_info = info["power_stats_scenario"]
        scenario_idx = scenario_info.get("scenario_idx", "?")
        print(f"\n--- POWER VARIABLE STATISTICS (SCENARIO {scenario_idx}) ---")
        if "error" in scenario_info:
            print(f"  Error: {scenario_info['error']}")
        else:
            # Filter out non-variable keys
            var_stats = {k: v for k, v in scenario_info.items()
                        if k in ["pg", "qg", "pl", "ql"]}
            _print_power_variable_table(var_stats)

            # Print Y stats for this scenario
            if "Y" in scenario_info:
                print(f"\n  TSI for scenario {scenario_idx}:")
                y_stats = scenario_info["Y"]
                print(f"    Range:  [{y_stats['min']:.4f}, {y_stats['max']:.4f}]")
                print(f"    Mean:   {y_stats['mean']:.4f}")
                print(f"    Median: {y_stats['median']:.4f}")
                print(f"    Std:    {y_stats['std']:.4f}")

    print("\n" + "=" * 80)


def _print_power_variable_table(power_stats: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted table of power variable statistics.

    Parameters
    ----------
    power_stats : dict
        Dictionary mapping variable names ('pg', 'qg', 'pl', 'ql') to
        their statistics dictionaries.
    """
    # Header
    print(f"  {'Variable':<10} {'Min':>12} {'Max':>12} {'Range':>12} "
          f"{'Mean':>12} {'Median':>12} {'Std':>12}")
    print("  " + "-" * 82)

    var_labels = {
        "pg": "Pg (gen)",
        "qg": "Qg (gen)",
        "pl": "Pl (load)",
        "ql": "Ql (load)"
    }

    for var_name in ["pg", "qg", "pl", "ql"]:
        if var_name in power_stats:
            stats = power_stats[var_name]
            label = var_labels.get(var_name, var_name)
            print(f"  {label:<10} "
                  f"{stats['min']:>12.4f} "
                  f"{stats['max']:>12.4f} "
                  f"{stats['range']:>12.4f} "
                  f"{stats['mean']:>12.4f} "
                  f"{stats['median']:>12.4f} "
                  f"{stats['std']:>12.4f}")


def display_per_unit_statistics(
    filepath: str = "tsi_probml_fullinputs.npz",
    scenario_idx: Optional[int] = None,
    print_output: bool = True
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Display per-unit (per generator/load) statistics for power variables.

    This function provides detailed statistics for each individual generator
    and load across all scenarios, showing how power values vary for each unit.

    Parameters
    ----------
    filepath : str, default='tsi_probml_fullinputs.npz'
        Path to the .npz file containing TSI data.
    scenario_idx : int, optional
        If provided, compute statistics for only this scenario.
        If None, compute statistics across all scenarios.
    print_output : bool, default=True
        If True, print formatted output to stdout.

    Returns
    -------
    dict
        Nested dictionary with structure:
        {
            'generators': {
                0: {'pg': {...stats...}, 'qg': {...stats...}},
                1: {'pg': {...stats...}, 'qg': {...stats...}},
                ...
            },
            'loads': {
                0: {'pl': {...stats...}, 'ql': {...stats...}},
                ...
            }
        }

    Examples
    --------
    >>> stats = display_per_unit_statistics("tsi_data.npz")
    >>> # Get stats for generator 0
    >>> gen0_pg_mean = stats['generators'][0]['pg']['mean']
    """
    data = load_tsi_data(filepath)
    powers = extract_power_variables(data, scenario_idx=scenario_idx)

    # Get dimensions
    meta = data["meta"]
    if isinstance(meta, np.ndarray):
        meta = meta.item() if meta.ndim == 0 else meta[0]

    Ngen = meta.get("Ngen", 0)
    Nload = meta.get("Nload", 0)

    result = {"generators": {}, "loads": {}}

    # Handle shape differences between single scenario and all scenarios
    pg, qg, pl, ql = powers["pg"], powers["qg"], powers["pl"], powers["ql"]

    # Generator statistics
    for i in range(Ngen):
        if scenario_idx is not None:
            # Single scenario: pg shape is (Ngen,)
            pg_i = np.array([pg[i]])
            qg_i = np.array([qg[i]])
        else:
            # All scenarios: pg shape is (N, Ngen)
            pg_i = pg[:, i]
            qg_i = qg[:, i]

        result["generators"][i] = {
            "pg": compute_variable_statistics(pg_i),
            "qg": compute_variable_statistics(qg_i)
        }

    # Load statistics
    for i in range(Nload):
        if scenario_idx is not None:
            pl_i = np.array([pl[i]])
            ql_i = np.array([ql[i]])
        else:
            pl_i = pl[:, i]
            ql_i = ql[:, i]

        result["loads"][i] = {
            "pl": compute_variable_statistics(pl_i),
            "ql": compute_variable_statistics(ql_i)
        }

    if print_output:
        _print_per_unit_statistics(result, Ngen, Nload, scenario_idx)

    return result


def _print_per_unit_statistics(
    stats: Dict,
    Ngen: int,
    Nload: int,
    scenario_idx: Optional[int]
) -> None:
    """Print formatted per-unit statistics."""
    scope = f"Scenario {scenario_idx}" if scenario_idx is not None else "All Scenarios"
    print(f"\n--- PER-UNIT STATISTICS ({scope}) ---")

    # Generators
    print(f"\n  GENERATORS ({Ngen} units):")
    print(f"  {'Unit':<6} {'Pg_min':>10} {'Pg_max':>10} {'Pg_mean':>10} "
          f"{'Qg_min':>10} {'Qg_max':>10} {'Qg_mean':>10}")
    print("  " + "-" * 70)

    for i in range(min(Ngen, 20)):  # Limit output for large systems
        pg_stats = stats["generators"][i]["pg"]
        qg_stats = stats["generators"][i]["qg"]
        print(f"  {i:<6} "
              f"{pg_stats['min']:>10.4f} {pg_stats['max']:>10.4f} {pg_stats['mean']:>10.4f} "
              f"{qg_stats['min']:>10.4f} {qg_stats['max']:>10.4f} {qg_stats['mean']:>10.4f}")

    if Ngen > 20:
        print(f"  ... ({Ngen - 20} more generators not shown)")

    # Loads
    print(f"\n  LOADS ({Nload} units):")
    print(f"  {'Unit':<6} {'Pl_min':>10} {'Pl_max':>10} {'Pl_mean':>10} "
          f"{'Ql_min':>10} {'Ql_max':>10} {'Ql_mean':>10}")
    print("  " + "-" * 70)

    for i in range(min(Nload, 20)):  # Limit output for large systems
        pl_stats = stats["loads"][i]["pl"]
        ql_stats = stats["loads"][i]["ql"]
        print(f"  {i:<6} "
              f"{pl_stats['min']:>10.4f} {pl_stats['max']:>10.4f} {pl_stats['mean']:>10.4f} "
              f"{ql_stats['min']:>10.4f} {ql_stats['max']:>10.4f} {ql_stats['mean']:>10.4f}")

    if Nload > 20:
        print(f"  ... ({Nload - 20} more loads not shown)")


# =============================================================================
# Histogram Plotting Functions
# =============================================================================

def plot_histogram_all_samples(
    filepath: str = "tsi_probml_fullinputs.npz",
    bins: int = 50,
    figsize: tuple = (10, 6),
    title: str = "Histogram of All TSI Values",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a histogram of all TSI values across the entire dataset.

    Creates a histogram showing the aggregate distribution of TSI values
    from all scenarios, fault locations, and fault impedances combined.
    This provides an overview of the overall stability characteristics
    of the simulation campaign.

    Parameters
    ----------
    filepath : str, default='tsi_probml_fullinputs.npz'
        Path to the .npz file containing TSI data.
    bins : int, default=50
        Number of histogram bins. More bins provide finer resolution
        but may be noisy for small datasets.
    figsize : tuple, default=(10, 6)
        Figure size as (width, height) in inches.
    title : str, default='Histogram of All TSI Values'
        Plot title displayed at the top of the figure.
    save_path : str, optional
        If provided, saves the figure to this path. Supports any format
        recognized by matplotlib (png, pdf, svg, etc.).

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object, which can be further customized
        or displayed with plt.show().

    Notes
    -----
    - The histogram is restricted to the range [-100, 100] since TSI
      values outside this range indicate numerical issues.
    - NaN values (from incomplete fault grids) are automatically removed.
    - Statistics box shows total samples, valid samples in range,
      mean, and standard deviation.

    Examples
    --------
    >>> fig = plot_histogram_all_samples("tsi_data.npz")
    >>> plt.show()

    >>> # Save high-resolution figure for publication
    >>> fig = plot_histogram_all_samples(
    ...     "tsi_data.npz",
    ...     bins=100,
    ...     figsize=(12, 8),
    ...     save_path="figures/tsi_distribution.pdf"
    ... )
    """
    # Load dataset
    data = load_tsi_data(filepath)
    Y = data["Y"]  # Shape: (N, F, Z) - samples × fault_locations × fault_impedances

    # Flatten all TSI values into a 1D array for histogram
    all_tsi_values = Y.flatten()

    # Remove NaN values (occur when require_complete_grid=False)
    all_tsi_values = all_tsi_values[~np.isnan(all_tsi_values)]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram with fixed range for consistency across datasets
    ax.hist(
        all_tsi_values,
        bins=bins,
        range=(-100, 100),  # TSI theoretical range
        edgecolor="black",
        alpha=0.7
    )

    # Configure axes
    ax.set_xlim(-100, 100)
    ax.set_xlabel("TSI Value")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add statistics annotation box
    valid_in_range = all_tsi_values[
        (all_tsi_values >= -100) & (all_tsi_values <= 100)
    ]
    stats_text = (
        f"Total samples: {len(all_tsi_values):,}\n"
        f"In range [-100, 100]: {len(valid_in_range):,}\n"
        f"Mean: {np.mean(all_tsi_values):.2f}\n"
        f"Std: {np.std(all_tsi_values):.2f}"
    )
    ax.text(
        0.02, 0.98,  # Position in axes coordinates (top-left)
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


def plot_histogram_single_scenario(
    scenario_idx: int,
    filepath: str = "tsi_probml_fullinputs.npz",
    bins: int = 50,
    figsize: tuple = (10, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a histogram of TSI values for a single scenario (operating condition).

    Creates a histogram showing the distribution of TSI values across all
    fault locations and impedances for one specific operating condition
    (sample_idx). This reveals how stability varies with fault parameters
    for a fixed system state.

    Parameters
    ----------
    scenario_idx : int
        Index of the scenario (sample) to plot. Corresponds to the first
        dimension of the Y array with shape (N, F, Z). This typically
        represents a unique combination of generator dispatch and load
        conditions.
    filepath : str, default='tsi_probml_fullinputs.npz'
        Path to the .npz file containing TSI data.
    bins : int, default=50
        Number of histogram bins.
    figsize : tuple, default=(10, 6)
        Figure size as (width, height) in inches.
    title : str, optional
        Plot title. If None, auto-generates a title based on scenario_idx:
        "Histogram of TSI Values for Scenario {scenario_idx}"
    save_path : str, optional
        If provided, saves the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.

    Raises
    ------
    IndexError
        If scenario_idx is out of bounds for the dataset. The error message
        includes the valid range of indices.

    Notes
    -----
    - Each scenario contains F × Z TSI values (one per fault condition).
    - The histogram shows how stability varies across the fault parameter
      space for a single operating point.
    - Useful for identifying operating conditions that are marginally
      stable (TSI values clustered near zero).

    Examples
    --------
    >>> # Plot for the first scenario
    >>> fig = plot_histogram_single_scenario(0, "tsi_data.npz")
    >>> plt.show()

    >>> # Compare two different operating conditions
    >>> fig1 = plot_histogram_single_scenario(
    ...     scenario_idx=10,
    ...     title="Low Load Condition"
    ... )
    >>> fig2 = plot_histogram_single_scenario(
    ...     scenario_idx=50,
    ...     title="High Load Condition"
    ... )

    >>> # Save with custom filename
    >>> fig = plot_histogram_single_scenario(
    ...     scenario_idx=42,
    ...     save_path="scenario_42_analysis.png"
    ... )
    """
    # Load dataset
    data = load_tsi_data(filepath)
    Y = data["Y"]  # Shape: (N, F, Z)

    # Extract dimensions
    N, F, Z = Y.shape

    # Validate scenario index
    if scenario_idx < 0 or scenario_idx >= N:
        raise IndexError(
            f"scenario_idx {scenario_idx} is out of bounds. "
            f"Valid range: [0, {N - 1}]"
        )

    # Extract TSI values for the specified scenario (all fault conditions)
    scenario_tsi_values = Y[scenario_idx, :, :].flatten()

    # Remove NaN values if present
    scenario_tsi_values = scenario_tsi_values[~np.isnan(scenario_tsi_values)]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Generate default title if not provided
    if title is None:
        title = f"Histogram of TSI Values for Scenario {scenario_idx}"

    # Plot histogram with distinctive color for single-scenario plots
    ax.hist(
        scenario_tsi_values,
        bins=bins,
        range=(-100, 100),
        edgecolor="black",
        alpha=0.7,
        color="steelblue"  # Different color to distinguish from aggregate plots
    )

    # Configure axes
    ax.set_xlim(-100, 100)
    ax.set_xlabel("TSI Value")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add statistics annotation box with scenario-specific info
    valid_in_range = scenario_tsi_values[
        (scenario_tsi_values >= -100) & (scenario_tsi_values <= 100)
    ]
    stats_text = (
        f"Scenario index: {scenario_idx}\n"
        f"Grid size (F×Z): {F}×{Z} = {F * Z}\n"
        f"Valid samples: {len(scenario_tsi_values)}\n"
        f"In range [-100, 100]: {len(valid_in_range)}\n"
        f"Mean: {np.mean(scenario_tsi_values):.2f}\n"
        f"Std: {np.std(scenario_tsi_values):.2f}"
    )
    ax.text(
        0.02, 0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """
    Main entry point for TSI histogram utilities.

    Provides command-line interface for displaying dataset information,
    generating histograms, and analyzing power variable statistics.

    For programmatic usage, import the functions directly::

        from TSI_histogram_utils import (
            load_tsi_data,
            plot_histogram_all_samples,
            plot_histogram_single_scenario,
            display_dataset_info,
            extract_power_variables
        )

        # Load data for custom analysis
        data = load_tsi_data("tsi_probml_fullinputs.npz")
        Y = data["Y"]

        # Display comprehensive dataset info
        info = display_dataset_info("tsi_probml_fullinputs.npz")

        # Display info for a specific scenario
        info = display_dataset_info("tsi_probml_fullinputs.npz", scenario_idx=5)

        # Extract power variables for custom analysis
        powers = extract_power_variables(data)
        pg_mean = powers['pg'].mean()

        # Plot histogram of all TSI values
        fig1 = plot_histogram_all_samples("tsi_probml_fullinputs.npz")

        # Plot histogram for a specific scenario
        fig2 = plot_histogram_single_scenario(5, "tsi_probml_fullinputs.npz")
    """
    import argparse
    import sys
    import os

    parser = argparse.ArgumentParser(
        description="TSI Histogram Utilities - Display dataset info and generate histograms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display dataset information only
  python TSI_histogram_utils.py my_dataset.npz

  # Display info for a specific scenario
  python TSI_histogram_utils.py my_dataset.npz -s 5

  # Generate histograms without displaying (save only)
  python TSI_histogram_utils.py my_dataset.npz --histogram --no-show

  # Generate histogram for specific scenario
  python TSI_histogram_utils.py my_dataset.npz -s 10 --histogram

  # Show per-unit statistics
  python TSI_histogram_utils.py my_dataset.npz --per-unit

  # Export MATLAB training samples
  python TSI_histogram_utils.py my_dataset.npz --export-mat

  # Export MATLAB training samples to a specific file
  python TSI_histogram_utils.py my_dataset.npz --export-mat --mat-output samples.mat

  # Full analysis with all options
  python TSI_histogram_utils.py my_dataset.npz -s 0 --histogram --per-unit
        """
    )

    parser.add_argument(
        "filepath",
        type=str,
        help="Path to the .npz file containing TSI data"
    )
    parser.add_argument(
        "-s", "--scenario",
        type=int,
        default=None,
        metavar="IDX",
        help="Scenario index to analyze (default: None, shows aggregate stats only)"
    )
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Generate histogram plots"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots interactively (only save to files)"
    )
    parser.add_argument(
        "--per-unit",
        action="store_true",
        help="Display per-unit (per generator/load) statistics"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=".",
        metavar="DIR",
        help="Directory for output files (default: current directory)"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress dataset info output (only show histograms/per-unit stats if requested)"
    )
    parser.add_argument(
        "--export-mat",
        action="store_true",
        help="Export MATLAB training samples from the TSI dataset"
    )
    parser.add_argument(
        "--mat-output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "MATLAB output .mat file or directory. If provided without "
            "--export-mat, MATLAB export is enabled."
        )
    )

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.filepath):
        print(f"Error: Dataset file '{args.filepath}' not found.")
        sys.exit(1)

    # Create output directory if needed
    if args.output_dir != "." and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # Display dataset information
    if not args.quiet:
        print("\n" + "=" * 80)
        print("DATASET INFORMATION")
        print("=" * 80)
        info = display_dataset_info(args.filepath, scenario_idx=args.scenario)

    # Display per-unit statistics if requested
    if args.per_unit:
        print("\n" + "=" * 80)
        print("PER-UNIT STATISTICS")
        print("=" * 80)
        display_per_unit_statistics(args.filepath, scenario_idx=args.scenario)

    # Generate histograms if requested
    if args.histogram:
        print("\n" + "=" * 80)
        print("GENERATING HISTOGRAMS")
        print("=" * 80)

        # Aggregate histogram
        save_path_all = os.path.join(args.output_dir, "histogram_all_tsi.png")
        print(f"\nGenerating aggregate histogram...")
        fig1 = plot_histogram_all_samples(
            filepath=args.filepath,
            bins=args.bins,
            save_path=save_path_all
        )

        # Per-scenario histogram if scenario specified
        if args.scenario is not None:
            save_path_scenario = os.path.join(
                args.output_dir,
                f"histogram_scenario_{args.scenario}.png"
            )
            print(f"\nGenerating histogram for scenario {args.scenario}...")
            fig2 = plot_histogram_single_scenario(
                scenario_idx=args.scenario,
                filepath=args.filepath,
                bins=args.bins,
                save_path=save_path_scenario
            )

        # Show plots unless --no-show specified
        if not args.no_show:
            plt.show()

    # Export MATLAB training samples only when explicitly requested.
    if args.export_mat or args.mat_output is not None:
        create_training_samples(args.filepath, output_path=args.mat_output)
        
    print("\nDone.")


if __name__ == "__main__":
    main()
