#!/usr/bin/env python
r"""
Transient Stability Index (TSI) Analysis for Power Grid Simulations.

This module provides comprehensive tools for computing and analyzing the Transient
Stability Index (TSI) from power grid simulation data. TSI is a widely-used metric
for assessing the dynamic stability of power systems following disturbances such
as faults.

The TSI is computed based on generator rotor angle deviations (delta states):

    $$TSI = (2\pi - \Delta_max) / (2\pi + \Delta_max) x 100$$

Where $\Delta_max$ is the maximum spread between generator rotor angles. A positive TSI
indicates stable operation, while a negative TSI indicates instability.

Features
--------
- Load and filter simulation scenarios based on various criteria
- Extract state variable time series from simulation data
- Compute TSI (both scalar and time series) for large scenario sets
- Create training datasets for machine learning applications
- Export datasets in formats suitable for probabilistic ML models
- Progress tracking with ETA estimation for long-running operations

Main Functions
--------------
ComputeTSI() : Compute TSI for all scenarios (memory-efficient, slower)
ComputeTSI_fast() : Optimized TSI computation (vectorized, faster)
create_training_samples() : Export data to MATLAB format for ML training
export_probml_dataset() : Build datasets for probabilistic ML models

Data Flow
---------
1. Load simulation log (scenario metadata) and state metadata
2. Filter scenarios based on convergence status and other criteria
3. Extract generator rotor angle (delta) time series
4. Compute TSI metrics (scalar and time-varying)
5. Export to desired format (MATLAB .mat or NumPy .npz)

Required Input Files
--------------------
- simulation_log.json : Metadata for all simulation scenarios including
  file paths, fault parameters, and convergence status
- state_metadata.json : Description of all state variables in the simulation
  including model type, device number, bus number, and state names
- simulation_data/scenario_*.npz : Individual scenario data files containing
  state history, time vectors, and power setpoints

Output Files
------------
- data_record.mat : MATLAB format training data (create_training_samples)
- tsi_probml_dataset.npz : NumPy format dataset for ML (export_probml_dataset)
- gen1_speed_comparison.png : Example visualization output

Dependencies
------------
- numpy : Numerical computations and array operations
- matplotlib : Visualization and plotting
- scipy : MATLAB file I/O
- seaborn (optional) : Enhanced statistical visualizations

Usage
-----
Command-line execution with default settings (final TSI)::

    $ python TSI_analysis.py

Use minimum TSI across all time steps::

    $ python TSI_analysis.py --tsi-mode min

Custom output path::

    $ python TSI_analysis.py -o my_dataset.npz --tsi-mode min

Show all CLI options::

    $ python TSI_analysis.py --help

Programmatic usage::

    from TSI_analysis import ComputeTSI_fast, export_probml_dataset

    # Compute TSI for all scenarios
    post_data = ComputeTSI_fast()

    # Export dataset for ML (using final TSI - default)
    result = export_probml_dataset(
        out_path="my_dataset.npz",
        require_complete_grid=False,
        concat_generators_and_loads=True,
        tsi_mode="final"  # or "min" for minimum TSI over time
    )

Performance Notes
-----------------
- ComputeTSI_fast() is significantly faster than ComputeTSI() due to
  vectorized operations and single-pass data loading
- Memory-mapped file access (mmap_mode='r') reduces memory footprint
- Progress tracking adds minimal overhead (~1% of total runtime)

See Also
--------
- generate_scenarios.py : Main simulation script that produces input data
- recovery_tool.py : Recovery utilities for failed simulations
- monitor.py : Real-time simulation monitoring

"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import scipy.io as scio
from collections import defaultdict
import time


# =============================================================================
# Utility Classes and Functions
# =============================================================================

class ProgressTracker:
    """
    Utility class for tracking and reporting progress with timing and ETA.

    Provides real-time feedback during long-running operations by displaying
    completion percentage, elapsed time, processing rate, and estimated time
    to completion.

    Parameters
    ----------
    total : int
        Total number of items to process.
    description : str, default='Processing'
        Label to display in progress reports.
    report_interval : int, default=100
        How often to print progress updates (every N items).

    Attributes
    ----------
    total : int
        Total items to process.
    description : str
        Progress label.
    report_interval : int
        Update frequency.
    start_time : float
        Timestamp when tracking began.
    count : int
        Current number of processed items.
    last_report_time : float
        Timestamp of last progress report.

    Examples
    --------
    >>> tracker = ProgressTracker(1000, "Loading files", report_interval=100)
    >>> for item in items:
    ...     process(item)
    ...     tracker.update()
    >>> tracker.finish()
    """

    def __init__(self, total: int, description: str = "Processing", report_interval: int = 100):
        self.total = total
        self.description = description
        self.report_interval = report_interval
        self.start_time = time.time()
        self.count = 0
        self.last_report_time = self.start_time

    def update(self, increment: int = 1) -> None:
        """
        Update progress counter and print report if at interval.

        Parameters
        ----------
        increment : int, default=1
            Number of items completed since last update.
        """
        self.count += increment
        if self.count % self.report_interval == 0 or self.count == self.total:
            self._print_progress()

    def _print_progress(self) -> None:
        """Print progress report with timing and ETA."""
        elapsed = time.time() - self.start_time
        pct = (self.count / self.total) * 100 if self.total > 0 else 0
        rate = self.count / elapsed if elapsed > 0 else 0

        # Calculate ETA if we have a meaningful rate
        if rate > 0 and self.count < self.total:
            remaining = (self.total - self.count) / rate
            eta_str = format_time(remaining)
        else:
            eta_str = "N/A"

        print(f"  [{self.description}] {self.count:,}/{self.total:,} ({pct:.1f}%) | "
              f"Elapsed: {format_time(elapsed)} | Rate: {rate:.1f}/s | ETA: {eta_str}")

    def finish(self) -> float:
        """
        Mark processing as complete and return total elapsed time.

        Returns
        -------
        float
            Total elapsed time in seconds.
        """
        elapsed = time.time() - self.start_time
        print(f"  [{self.description}] COMPLETE: {self.count:,} items in {format_time(elapsed)}")
        return elapsed


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.

    Parameters
    ----------
    seconds : float
        Time duration in seconds.

    Returns
    -------
    str
        Formatted time string (e.g., "45.2s", "3m 21s", "2h 15m").
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def print_stage_header(stage_name: str) -> float:
    """
    Print a formatted stage header and return the start time.

    Parameters
    ----------
    stage_name : str
        Name of the processing stage.

    Returns
    -------
    float
        Start timestamp for timing the stage.
    """
    print("\n" + "=" * 70)
    print(f"STAGE: {stage_name}")
    print("=" * 70)
    return time.time()


def print_stage_complete(stage_name: str, start_time: float) -> None:
    """
    Print stage completion message with timing.

    Parameters
    ----------
    stage_name : str
        Name of the completed stage.
    start_time : float
        Timestamp when the stage began.
    """
    elapsed = time.time() - start_time
    print(f"\n✓ {stage_name} completed in {format_time(elapsed)}")
    print("-" * 70)


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_simulation_log(file_path: str = 'simulation_log.json') -> Dict:
    """
    Load the simulation log containing metadata about all scenarios.

    Parameters
    ----------
    file_path : str, default='simulation_log.json'
        Path to the JSON log file.

    Returns
    -------
    Dict
        Dictionary mapping scenario IDs to their metadata including:
        - file : str, path to scenario data file
        - diverged : bool, whether simulation diverged
        - fault_location : int, bus number where fault occurred
        - fault_impedance : float, fault impedance value
        - sample_idx : int, sample index for this scenario
    """
    print(f"  Loading simulation log from '{file_path}'...", end=" ", flush=True)
    start = time.time()
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"done ({len(data):,} scenarios, {format_time(time.time() - start)})")
    return data


def load_state_metadata(file_path: str = 'state_metadata.json') -> Dict:
    """
    Load the state metadata that describes all state variables.

    Parameters
    ----------
    file_path : str, default='state_metadata.json'
        Path to the JSON metadata file.

    Returns
    -------
    Dict
        Dictionary mapping state indices to their metadata including:
        - model : str, dynamic model name (e.g., 'GenGENROU')
        - device_number : str, device identifier
        - bus_num : int, bus number
        - state_name : str, name of state variable (e.g., 'delta', 'w')
    """
    print(f"  Loading state metadata from '{file_path}'...", end=" ", flush=True)
    start = time.time()
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"done ({len(data):,} states, {format_time(time.time() - start)})")
    return data


# =============================================================================
# Filtering and Indexing Functions
# =============================================================================

def filter_scenarios(
    simulation_log: Dict,
    sample_idx: Optional[int] = None,
    fault_location: Optional[int] = None,
    fault_impedance: Optional[float] = None,
    diverged: Optional[bool] = None
) -> Dict:
    """
    Filter scenarios based on specified criteria.

    Parameters
    ----------
    simulation_log : Dict
        Full simulation log dictionary.
    sample_idx : int, optional
        Filter by sample index.
    fault_location : int, optional
        Filter by fault location (bus number).
    fault_impedance : float, optional
        Filter by fault impedance value.
    diverged : bool, optional
        Filter by convergence status (True=diverged, False=converged).

    Returns
    -------
    Dict
        Filtered dictionary containing only matching scenarios.

    Examples
    --------
    >>> # Get all converged scenarios at bus 42
    >>> filtered = filter_scenarios(log, fault_location=42, diverged=False)
    """
    filtered_log = {}

    for scenario_id, data in simulation_log.items():
        match = True

        # Apply each filter if specified
        if sample_idx is not None and data.get('sample_idx') != sample_idx:
            match = False
        if fault_location is not None and data['fault_location'] != fault_location:
            match = False
        if fault_impedance is not None and data['fault_impedance'] != fault_impedance:
            match = False
        if diverged is not None and data['diverged'] != diverged:
            match = False

        if match:
            filtered_log[scenario_id] = data

    return filtered_log


def find_state_index(
    state_metadata: Dict,
    model: Optional[str] = None,
    device_number: Optional[str] = None,
    bus_num: Optional[int] = None,
    state_name: Optional[str] = None
) -> List[int]:
    """
    Find indices of states matching the specified criteria.

    Parameters
    ----------
    state_metadata : Dict
        State metadata dictionary.
    model : str, optional
        Dynamic model name (e.g., 'GenGENROU', 'GenGENSAL').
    device_number : str, optional
        Device identifier.
    bus_num : int, optional
        Bus number.
    state_name : str, optional
        State variable name (e.g., 'delta', 'w', 'Eqp').

    Returns
    -------
    List[int]
        List of state indices matching all specified criteria.

    Examples
    --------
    >>> # Find rotor angle states for all GENROU generators
    >>> delta_indices = find_state_index(metadata, model='GenGENROU', state_name='delta')
    """
    indices = []

    for idx, (state_idx, data) in enumerate(state_metadata.items()):
        match = True

        # Apply each filter if specified
        if model is not None and data.get('model') != model:
            match = False
        if device_number is not None and str(data.get('device_number')) != str(device_number):
            match = False
        if bus_num is not None and data.get('bus_num') != bus_num:
            match = False
        if state_name is not None and data.get('state_name') != state_name:
            match = False

        if match:
            indices.append(int(state_idx))

    return indices


# =============================================================================
# Data Extraction Functions
# =============================================================================

def load_scenario_data(scenario_id: str, simulation_log: Dict) -> Dict:
    """
    Load complete data for a specific scenario.

    Parameters
    ----------
    scenario_id : str
        Unique identifier for the scenario.
    simulation_log : Dict
        Simulation log containing file paths.

    Returns
    -------
    Dict
        Dictionary containing:
        - history : ndarray, state variable history (n_states, n_timesteps)
        - tvec : ndarray, time vector
        - metadata : dict, scenario metadata
        - p_gen_scaled : ndarray, generator active power setpoints (per-unit)
        - p_load_scaled : ndarray, load active power (per-unit)
        - q_load_scaled : ndarray, load reactive power (per-unit)

    Raises
    ------
    FileNotFoundError
        If the scenario data file doesn't exist.

    Notes
    -----
    Uses memory-mapped file access (mmap_mode='r') to reduce memory footprint
    when loading large scenario files.
    """
    file_path = simulation_log[scenario_id]['file']
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Scenario data file not found: {file_path}")

    # Use memory mapping to avoid loading entire file into RAM
    data = np.load(file_path, mmap_mode='r')
    return {
        'history': data['history'],
        'tvec': data['tvec'],
        'metadata': simulation_log[scenario_id],
        'p_gen_scaled': data['p_gen_scaled'],
        'p_load_scaled': data['p_load_scaled'],
        'q_load_scaled': data['q_load_scaled'],
    }


def get_state_timeseries(
    scenario_data: Dict,
    state_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a specific state variable time series from scenario data.

    Parameters
    ----------
    scenario_data : Dict
        Scenario data dictionary from load_scenario_data().
    state_idx : int
        Index of the state variable to extract.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - tvec : Time vector
        - state_values : State variable values over time

        Returns empty arrays if history is None.
    """
    tvec = scenario_data['tvec']
    history = scenario_data['history']

    if history is None:
        return np.array([]), np.array([])

    # Copy to ensure we have actual data, not a memmap view
    state_values = history[state_idx, :].copy()
    return tvec, state_values


def get_state_timeseries_all(
    simulation_log: Dict,
    state_metadata: Dict,
    model: Optional[str] = None,
    device_number: Optional[str] = None,
    bus_num: Optional[int] = None,
    state_name: Optional[str] = None,
    sample_idx: Optional[int] = None,
    fault_location: Optional[int] = None,
    fault_impedance: Optional[float] = None,
    diverged: Optional[bool] = False,
    show_progress: bool = True,
    progress_interval: int = 500
) -> Dict[str, Dict[str, Union[np.ndarray, Any]]]:
    """
    Extract a state variable from multiple scenarios with filtering.

    This is a convenience function that combines scenario filtering, state
    index lookup, and data extraction into a single operation.

    Parameters
    ----------
    simulation_log : Dict
        Full simulation log.
    state_metadata : Dict
        State metadata dictionary.
    model : str, optional
        Filter by dynamic model name.
    device_number : str, optional
        Filter by device number.
    bus_num : int, optional
        Filter by bus number.
    state_name : str, optional
        Filter by state variable name.
    sample_idx : int, optional
        Filter by sample index.
    fault_location : int, optional
        Filter by fault location.
    fault_impedance : float, optional
        Filter by fault impedance.
    diverged : bool, default=False
        Filter by convergence status.
    show_progress : bool, default=True
        Display progress updates.
    progress_interval : int, default=500
        Progress update frequency.

    Returns
    -------
    Dict[str, Dict[str, Union[np.ndarray, Any]]]
        Dictionary mapping scenario IDs to their extracted data:
        - tvec : Time vector
        - values : State variable values
        - metadata : Scenario metadata

    Raises
    ------
    ValueError
        If no states match the specified criteria.
    """
    # Filter scenarios based on simulation parameters
    filtered_scenarios = filter_scenarios(
        simulation_log,
        sample_idx,
        fault_location,
        fault_impedance,
        diverged
    )

    # Also get diverged scenarios for reporting (can be commented out for performance)
    filtered_scenarios_div = filter_scenarios(
        simulation_log,
        sample_idx,
        fault_location,
        fault_impedance,
        diverged=True
    )
    print(f"  Scenarios that did not diverge: {len(filtered_scenarios):,}; "
          f"scenarios that diverged: {len(filtered_scenarios_div):,}")

    # Free memory from diverged scenarios lookup
    filtered_scenarios_div = None

    # Find state indices matching the state criteria
    state_indices = find_state_index(
        state_metadata,
        model,
        device_number,
        bus_num,
        state_name
    )

    if not state_indices:
        raise ValueError(f"No states found matching criteria: model={model}, "
                         f"device_number={device_number}, state_name={state_name}")

    # Take the first matching state if multiple found
    state_idx = state_indices[0]

    # Extract data for each scenario with progress tracking
    results = {}
    total_scenarios = len(filtered_scenarios)

    if show_progress and total_scenarios > 0:
        tracker = ProgressTracker(total_scenarios, "Loading scenarios", progress_interval)

    for scenario_id, scenario_info in filtered_scenarios.items():
        try:
            scenario_data = load_scenario_data(scenario_id, simulation_log)
            tvec, values = get_state_timeseries(scenario_data, state_idx)

            results[scenario_id] = {
                'tvec': tvec,
                'values': values,
                'metadata': scenario_info
            }
        except Exception as e:
            print(f"  Error loading scenario {scenario_id}: {e}")

        if show_progress:
            tracker.update()

    if show_progress and total_scenarios > 0:
        tracker.finish()

    return results


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_state_comparison(
    results: Dict[str, Dict[str, Union[np.ndarray, Any]]],
    title: str = None,
    xlabel: str = 'Time (s)',
    ylabel: str = None,
    legend_key: str = 'fault_location'
):
    """
    Plot comparison of state variables from multiple scenarios.

    Creates an overlay plot showing the time evolution of a state variable
    across multiple simulation scenarios.

    Parameters
    ----------
    results : Dict
        Results dictionary from get_state_timeseries_all().
    title : str, optional
        Plot title.
    xlabel : str, default='Time (s)'
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    legend_key : str, default='fault_location'
        Metadata key to use for legend entries.

    Returns
    -------
    matplotlib.pyplot
        The pyplot module for further customization or saving.
    """
    plt.figure(figsize=(10, 6))

    for scenario_id, data in results.items():
        tvec = data['tvec']
        values = data['values']
        metadata = data['metadata']

        # Skip empty results (e.g., from diverged simulations)
        if len(tvec) == 0 or len(values) == 0:
            continue

        # Plot with semi-transparent gray lines for ensemble visualization
        plt.plot(tvec, values, color='gray', alpha=0.5)

    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()

    return plt


# =============================================================================
# Example and Demo Functions
# =============================================================================

def example_gen1_speed_deviation():
    """
    Example demonstrating extraction of generator 1 speed deviation.

    This function shows how to:
    1. Load simulation metadata
    2. Extract state variables for a specific generator
    3. Filter by simulation parameters
    4. Create comparison plots

    Outputs
    -------
    Creates two PNG files:
    - gen1_speed_comparison.png : All non-diverged scenarios
    - gen1_speed_comparison_sample1.png : Sample index 1 only

    Returns
    -------
    Dict
        Results dictionary containing extracted time series data.
    """
    stage_start = print_stage_header("Example: Generator 1 Speed Deviation")

    # Load metadata files
    simulation_log = load_simulation_log()
    state_metadata = load_state_metadata()

    # Get all generator 1 speed deviations from non-diverged simulations
    results = get_state_timeseries_all(
        simulation_log,
        state_metadata,
        model='GenGENROU',
        device_number='1',
        state_name='w',  # 'w' is the speed deviation state
        diverged=False
    )

    # Plot results grouped by base load
    plt = plot_state_comparison(
        results,
        title='Generator 1 Speed Deviation',
        ylabel='Speed Deviation (pu)',
        legend_key='base_load'
    )

    # Save figure
    plt.savefig('gen1_speed_comparison.png')
    plt.close()

    # Now filter by fault location and show for a specific load level
    results_filtered = get_state_timeseries_all(
        simulation_log,
        state_metadata,
        model='GenGENROU',
        device_number='1',
        state_name='w',
        sample_idx=1,
        diverged=False
    )

    plt = plot_state_comparison(
        results_filtered,
        title='Generator 1 Speed Deviation (Sample Index = 1)',
        ylabel='Speed Deviation (pu)',
        legend_key='fault_location'
    )

    plt.savefig('gen1_speed_comparison_sample1.png')

    print(f"Found {len(results)} non-diverged scenarios")
    print(f"Found {len(results_filtered)} non-diverged scenarios at sample_idx=1")

    print_stage_complete("Example: Generator 1 Speed Deviation", stage_start)
    return results


# =============================================================================
# TSI Computation Functions
# =============================================================================

def ComputeTSI_fast():
    """
    Compute Transient Stability Index (TSI) for all scenarios (optimized version).

    This is a performance-optimized implementation that:
    1. Loads each scenario file only once
    2. Uses vectorized NumPy operations for TSI calculation
    3. Processes all generator delta states in a single array operation

    The TSI is computed as:
        TSI(t) = (2π - spread(t)) / (2π + spread(t)) × 100

    Where spread(t) = max(δ) - min(δ) across all generators at time t.

    Returns
    -------
    Dict
        Dictionary containing:
        - tsi_per_scenario : Dict[str, float]
            Scalar TSI (minimum over time) for each scenario
        - tsi_ts_per_scenario : Dict[str, ndarray]
            TSI time series for each scenario
        - tsi_all : ndarray (N,)
            Scalar TSI values as array
        - tsi_all_time : ndarray (N, T)
            TSI time series as 2D array
        - pg_per_scenario : Dict[str, ndarray]
            Generator active power setpoints
        - pl_per_scenario : Dict[str, ndarray]
            Load active power values
        - ql_per_scenario : Dict[str, ndarray]
            Load reactive power values

    See Also
    --------
    ComputeTSI : Memory-efficient but slower implementation

    Notes
    -----
    This function is significantly faster than ComputeTSI() for large datasets
    because it avoids redundant file loads and uses vectorized operations.
    """
    stage_start = print_stage_header("ComputeTSI_fast")

    # Sub-stage 1: Load metadata
    substage_start = time.time()
    print("\n--- Sub-stage 1/4: Loading metadata ---")
    simulation_log = load_simulation_log()
    state_metadata = load_state_metadata()
    print(f"  Sub-stage 1 completed in {format_time(time.time() - substage_start)}")

    # Sub-stage 2: Find delta state indices for all generators
    substage_start = time.time()
    print("\n--- Sub-stage 2/4: Finding generator delta states ---")
    delta_state_idxs = [
        int(idx)
        for idx, meta in state_metadata.items()
        if meta.get('model') == 'GenGENROU' and meta.get('state_name') == 'delta'
    ]
    if not delta_state_idxs:
        raise RuntimeError("No GenGENROU delta states found in state_metadata.")
    print(f"  Found {len(delta_state_idxs)} generator delta states")
    print(f"  Sub-stage 2 completed in {format_time(time.time() - substage_start)}")

    # Sub-stage 3: Identify scenarios to process (non-diverged with existing files)
    substage_start = time.time()
    print("\n--- Sub-stage 3/4: Identifying valid scenarios ---")
    scenario_ids = [
        sid for sid, meta in simulation_log.items()
        if not meta.get('diverged', False) and os.path.exists(meta['file'])
    ]
    if not scenario_ids:
        raise RuntimeError("No non-diverged scenarios with existing files.")
    print(f"  Found {len(scenario_ids):,} valid scenarios to process")
    print(f"  Sub-stage 3 completed in {format_time(time.time() - substage_start)}")

    # Sub-stage 4: Main processing loop
    substage_start = time.time()
    print("\n--- Sub-stage 4/4: Processing all scenarios ---")

    # Get dimensions from the first scenario
    print("  Loading first scenario for dimensions...")
    first = load_scenario_data(scenario_ids[0], simulation_log)
    tvec0 = np.asarray(first['tvec'])
    T = tvec0.shape[0]  # Number of time steps
    N = len(scenario_ids)  # Number of scenarios
    twopi = 2.0 * np.pi
    print(f"  Time steps per scenario: {T}")
    print(f"  Total scenarios to process: {N:,}")

    # Preallocate output arrays for efficiency
    tsi_time = np.empty((N, T), dtype=float)    # TSI time series per scenario
    tsi_scalar = np.empty(N, dtype=float)       # Scalar TSI per scenario

    # Dictionaries for power setpoints (used by downstream functions)
    pg_per_scenario, pl_per_scenario, ql_per_scenario = {}, {}, {}

    # Progress tracking with ~20 updates during processing
    tracker = ProgressTracker(N, "Processing scenarios", report_interval=max(1, N // 20))

    # Main processing: load each scenario ONCE and compute TSI with vectorized ops
    for i, sid in enumerate(scenario_ids):
        sc = load_scenario_data(sid, simulation_log)
        H = sc['history']  # Shape: (n_states, T), memory-mapped view

        # Vectorized extraction of all generator delta values (no Python loop)
        deltas = H[delta_state_idxs, :]  # Shape: (n_generators, T)

        # Compute rotor angle spread at each time step
        spread = deltas.max(axis=0) - deltas.min(axis=0)  # Shape: (T,)

        # Compute TSI time series: TSI = (2π - spread) / (2π + spread) × 100
        tsi_t = (twopi - spread) / (twopi + spread) * 100.0
        tsi_time[i] = tsi_t

        # Scalar TSI is minimum over time (most critical stability point)
        tsi_scalar[i] = float(tsi_t.min())

        # Cache power setpoints for later use
        pg_per_scenario[sid] = np.asarray(sc['p_gen_scaled'])
        pl_per_scenario[sid] = np.asarray(sc['p_load_scaled'])
        ql_per_scenario[sid] = np.asarray(sc['q_load_scaled'])

        tracker.update()

    tracker.finish()
    print(f"  Sub-stage 4 completed in {format_time(time.time() - substage_start)}")

    # Build dictionary-based API for compatibility with other functions
    print("\n  Building result dictionaries...")
    tsi_ts_per_scenario = {sid: tsi_time[i] for i, sid in enumerate(scenario_ids)}
    tsi_per_scenario = {sid: tsi_scalar[i] for i, sid in enumerate(scenario_ids)}

    # Optional visualization using seaborn
    try:
        import seaborn as sns
        plt.figure(figsize=(10, 5))

        # Left plot: TSI distribution (minimum over time)
        plt.subplot(1, 2, 1)
        sns.histplot(tsi_scalar, bins=20, stat='density', kde=True)
        plt.xlabel('TSI at all times')
        plt.ylabel('Density')

        # Right plot: TSI at final time step
        plt.subplot(1, 2, 2)
        sns.histplot(np.squeeze(tsi_time[:, -1]), bins=20, stat='density', kde=True)
        plt.xlabel('TSI at final time')

        # Add stability threshold indicator (TSI=0 is stability boundary)
        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.axvline(0, color='k', ls='--', lw=1.5)
            plt.text(-1, plt.ylim()[1] * 0.95, 'unstable', ha='right', va='top', fontsize=10)
            plt.text(1, plt.ylim()[1] * 0.95, 'stable', ha='left', va='top', fontsize=10)

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("  Seaborn is not installed. Install with `pip install seaborn` for visualizations.")

    # Package all results
    post_data = {
        'tsi_per_scenario': tsi_per_scenario,
        'tsi_ts_per_scenario': tsi_ts_per_scenario,
        'tsi_all': tsi_scalar,
        'tsi_all_time': tsi_time,
        'pg_per_scenario': pg_per_scenario,
        'pl_per_scenario': pl_per_scenario,
        'ql_per_scenario': ql_per_scenario,
    }

    print_stage_complete("ComputeTSI_fast", stage_start)
    return post_data


def ComputeTSI():
    """
    Compute Transient Stability Index (TSI) for all scenarios (memory-efficient version).

    This implementation prioritizes memory efficiency over speed by:
    1. Loading generator delta data separately for each generator
    2. Processing scenarios one at a time
    3. Explicitly freeing intermediate data structures

    Use this version when memory is constrained. For faster processing,
    use ComputeTSI_fast() instead.

    Returns
    -------
    Dict
        Dictionary containing:
        - tsi_per_scenario : Dict[str, float]
            Scalar TSI for each scenario
        - tsi_ts_per_scenario : Dict[str, ndarray]
            TSI time series for each scenario
        - tsi_all : ndarray
            Scalar TSI values as array
        - tsi_all_time : ndarray
            TSI time series as 2D array
        - pg_per_scenario, pl_per_scenario, ql_per_scenario : Dict
            Power setpoint dictionaries

    See Also
    --------
    ComputeTSI_fast : Faster but more memory-intensive implementation
    """
    stage_start = print_stage_header("ComputeTSI")

    # Sub-stage 1: Load metadata
    substage_start = time.time()
    print("\n--- Sub-stage 1/4: Loading metadata ---")
    simulation_log = load_simulation_log()
    state_metadata = load_state_metadata()
    print(f"  Sub-stage 1 completed in {format_time(time.time() - substage_start)}")

    # Sub-stage 2: Find generator pairs (device_number, bus_num)
    substage_start = time.time()
    print("\n--- Sub-stage 2/4: Finding generator delta states ---")
    gen_pairs = {
        (str(data['device_number']), data['bus_num'])
        for data in state_metadata.values()
        if data.get('model') == 'GenGENROU'
        and data.get('state_name') == 'delta'
    }
    gen_list = sorted(gen_pairs, key=lambda x: (x[1], x[0]))  # Sort by bus, then device
    print(f"  Found {len(gen_list)} generators with delta states")
    print(f"  Sub-stage 2 completed in {format_time(time.time() - substage_start)}")

    # Sub-stage 3: Load generator delta data (one generator at a time)
    substage_start = time.time()
    print("\n--- Sub-stage 3/4: Loading generator delta data ---")
    print(f"  Loading data for {len(gen_list)} generators...")

    delta_dicts = {}
    gen_tracker = ProgressTracker(len(gen_list), "Loading generators", report_interval=1)

    for device_number, bus_num in gen_list:
        print(f"\n  Loading δ for GenGENROU device {device_number} on bus {bus_num}")
        d = get_state_timeseries_all(
            simulation_log,
            state_metadata,
            model='GenGENROU',
            device_number=device_number,
            bus_num=bus_num,
            state_name='delta',
            diverged=False,
            show_progress=True,
            progress_interval=1000
        )
        delta_dicts[(bus_num, device_number)] = d
        gen_tracker.update()

    gen_tracker.finish()
    print(f"  Sub-stage 3 completed in {format_time(time.time() - substage_start)}")

    if not delta_dicts:
        raise RuntimeError("No generator deltas were loaded!")

    # Find scenarios common to all generators
    scenario_sets = [set(d.keys()) for d in delta_dicts.values()]
    common_scenarios = sorted(set.intersection(*scenario_sets))
    if not common_scenarios:
        raise RuntimeError("No scenario is common to all generators!")

    print(f"\n  Found {len(common_scenarios):,} common scenarios across all generators")

    # Get dimensions from first scenario
    first_key = next(iter(delta_dicts))
    first_scenario = next(iter(delta_dicts[first_key]))
    tvec = delta_dicts[first_key][first_scenario]['tvec']
    T = len(tvec)
    G = len(delta_dicts)
    print(f"  Time steps: {T}, Generators: {G}")

    # Sub-stage 4: Process scenarios
    substage_start = time.time()
    print("\n--- Sub-stage 4/4: Computing TSI for all scenarios ---")

    # Initialize result dictionaries
    tsi_per_scenario = {}
    tsi_ts_per_scenario = {}
    pg_per_scenario = {}
    pl_per_scenario = {}
    ql_per_scenario = {}

    # Progress tracking
    scenario_tracker = ProgressTracker(
        len(common_scenarios), "Computing TSI",
        report_interval=max(1, len(common_scenarios) // 20)
    )

    # Process scenarios one by one (memory efficient)
    for s_idx, scenario_id in enumerate(common_scenarios):
        # Load scenario data once for power setpoints
        try:
            scenario_data = load_scenario_data(scenario_id, simulation_log)
            pg_per_scenario[scenario_id] = scenario_data['p_gen_scaled']
            pl_per_scenario[scenario_id] = scenario_data['p_load_scaled']
            ql_per_scenario[scenario_id] = scenario_data['q_load_scaled']
            del scenario_data  # Free immediately
        except Exception as e:
            print(f"  Error loading scenario {scenario_id}: {e}")
            continue

        # Extract delta values for this scenario from pre-loaded data
        delta_values = np.zeros((G, T))
        for g_idx, key in enumerate(delta_dicts):
            delta_values[g_idx, :] = delta_dicts[key][scenario_id]['values']

        # Compute TSI for this scenario
        spread_ts = delta_values.max(axis=0) - delta_values.min(axis=0)
        Delta_max = spread_ts.max()
        tsi_scalar = (2 * np.pi - Delta_max) / (2 * np.pi + Delta_max) * 100
        tsi_ts = (2 * np.pi - spread_ts) / (2 * np.pi + spread_ts) * 100

        tsi_per_scenario[scenario_id] = tsi_scalar
        tsi_ts_per_scenario[scenario_id] = tsi_ts

        # Free memory for this scenario's intermediate data
        del delta_values, spread_ts, tsi_ts

        scenario_tracker.update()

    scenario_tracker.finish()
    print(f"  Sub-stage 4 completed in {format_time(time.time() - substage_start)}")

    # Clear delta_dicts to free memory before creating final arrays
    print("\n  Freeing intermediate memory...")
    del delta_dicts

    # Create final arrays
    print("  Creating final arrays...")
    tsi_all = np.array([tsi_per_scenario[sc] for sc in common_scenarios])
    tsi_all_time = np.vstack([tsi_ts_per_scenario[sc] for sc in common_scenarios])

    # Package results
    post_data = {
        'tsi_per_scenario': tsi_per_scenario,
        'tsi_ts_per_scenario': tsi_ts_per_scenario,
        'tsi_all': tsi_all,
        'tsi_all_time': tsi_all_time,
        'pg_per_scenario': pg_per_scenario,
        'pl_per_scenario': pl_per_scenario,
        'ql_per_scenario': ql_per_scenario,
    }

    print(f'\n  TSI for all scenarios: {tsi_all.shape}')
    print(f'  TSI for all time scenarios: {tsi_all_time.shape}')

    # Optional visualization
    try:
        import seaborn as sns
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(tsi_all, bins=20, stat='density', kde=True)
        plt.xlabel('TSI at all times')
        plt.ylabel('Density')

        plt.subplot(1, 2, 2)
        sns.histplot(np.squeeze(tsi_all_time[:, -1]), bins=20, stat='density', kde=True)
        plt.xlabel('TSI at final time')

        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.axvline(0, color='k', ls='--', lw=1.5)
            plt.text(-1, plt.ylim()[1] * 0.95, 'unstable', ha='right', va='top', fontsize=10)
            plt.text(1, plt.ylim()[1] * 0.95, 'stable', ha='left', va='top', fontsize=10)

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("  Seaborn is not installed. Install with `pip install seaborn` for visualizations.")

    print_stage_complete("ComputeTSI", stage_start)
    return post_data


# =============================================================================
# Dataset Export Functions
# =============================================================================

def create_training_samples(post_data: Dict):
    """
    Create training samples in MATLAB format for machine learning.

    Exports simulation results as a MATLAB .mat file with features (power
    setpoints) and labels (TSI values) suitable for training ML models.

    Parameters
    ----------
    post_data : Dict
        Output from ComputeTSI() or ComputeTSI_fast() containing:
        - tsi_per_scenario : TSI values
        - pg_per_scenario : Generator active power
        - pl_per_scenario : Load active power
        - ql_per_scenario : Load reactive power

    Output Files
    ------------
    data_record.mat : MATLAB file containing:
        - Data : ndarray (n_scenarios, n_features + 1)
            Concatenated [pg, pl, ql, tsi] for each scenario
        - col_name : list of str
            Column names for the data array
    """
    stage_start = print_stage_header("Create Training Samples")

    # Extract data dictionaries
    tsi_dict = post_data['tsi_per_scenario']
    pg_dict = post_data['pg_per_scenario']
    pl_dict = post_data['pl_per_scenario']
    ql_dict = post_data['ql_per_scenario']

    scenario_ids = sorted(tsi_dict.keys())
    print(f"  Processing {len(scenario_ids):,} scenarios...")

    # Get dimensions from first scenario
    first_sid = scenario_ids[0]
    pg_len = len(pg_dict[first_sid])
    pl_len = len(pl_dict[first_sid])
    ql_len = len(ql_dict[first_sid])

    # Create column names for the output data
    col_name = (
        [f'pg_{i+1}' for i in range(pg_len)] +
        [f'pl_{i+1}' for i in range(pl_len)] +
        [f'ql_{i+1}' for i in range(ql_len)] +
        ['tsi']
    )

    # Build data rows
    rows = []
    tracker = ProgressTracker(
        len(scenario_ids), "Building rows",
        report_interval=max(1, len(scenario_ids) // 10)
    )

    for sid in scenario_ids:
        pg = pg_dict[sid]
        pl = pl_dict[sid]
        ql = ql_dict[sid]
        tsi = np.array([tsi_dict[sid]])

        # Concatenate all features and label into single row
        row = np.hstack((pg, pl, ql, tsi))
        rows.append(row)
        tracker.update()

    tracker.finish()

    # Stack all rows into 2D array
    Data = np.vstack(rows)

    # Save to MATLAB format
    print("  Saving samples to data_record.mat...")
    scio.savemat('data_record.mat', {'Data': Data, 'col_name': col_name})

    print_stage_complete("Create Training Samples", stage_start)


def export_probml_dataset(
    out_path: str = "tsi_probml_dataset.npz",
    require_complete_grid: bool = True,
    concat_generators_and_loads: bool = True,
    return_X_flat: bool = True,
    verbose: bool = True,
    tsi_mode: str = "final",
) -> Dict[str, Any]:
    """
    Build a dataset suitable for probabilistic ML models.

    Creates a structured dataset where each sample corresponds to a unique
    operating condition (sample_idx), with TSI outputs computed across a
    grid of fault scenarios (fault_location × fault_impedance).

    This format is designed for learning the distribution of TSI given
    operating conditions, enabling probabilistic predictions and uncertainty
    quantification.

    Parameters
    ----------
    out_path : str, default='tsi_probml_dataset.npz'
        Output file path for the compressed NumPy archive.
    require_complete_grid : bool, default=True
        If True, only include samples where all fault combinations converged.
        If False, missing entries are filled with NaN.
    concat_generators_and_loads : bool, default=True
        If True, concatenate generator and load data into single array X.
        If False, save separate X_gen and X_load arrays.
    return_X_flat : bool, default=True
        If True (and concat=True), also save flattened input array X_flat.
    verbose : bool, default=True
        Print detailed progress and summary information.
    tsi_mode : str, default='final'
        Method for extracting the scalar TSI value from the time series:
        
        - 'final': Use TSI at the last time step (default)
        - 'min': Use minimum TSI across all time steps (most conservative)
        
        The 'min' mode captures the worst-case stability during the entire
        simulation, while 'final' captures the steady-state stability.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing all saved data:
        - X or (X_gen, X_load) : Input features
        - X_flat : Flattened inputs (optional)
        - Y : TSI grid (N, F, Z) where F=fault locations, Z=impedances
        - sample_idx : Sample indices array
        - fault_locations : Fault location array
        - fault_impedances : Fault impedance array
        - scenario_ids : Scenario ID mapping
        - out_path : Output file path

    Output File Contents
    --------------------
    If concat_generators_and_loads=True:
        - X : float64 (N, 2, Ngen+Nload) - [P, Q] channels
        - X_flat : float64 (N, 2*(Ngen+Nload)) - flattened view

    If concat_generators_and_loads=False:
        - X_gen : float64 (N, 2, Ngen) - generator [Pg, Qg]
        - X_load : float64 (N, 2, Nload) - load [Pl, Ql]

    Always saved:
        - Y : float64 (N, F, Z) - TSI value (mode depends on tsi_mode parameter)
        - sample_idx : int (N,) - sample indices
        - fault_locations : int (F,) - fault bus numbers
        - fault_impedances : float64 (Z,) - impedance values
        - scenario_ids : object (N, F, Z) - scenario ID lookup
        - meta : dict - metadata including array shapes, meanings, and tsi_mode

    Notes
    -----
    The dataset structure enables:
    - Learning P(TSI | operating_condition, fault_scenario)
    - Uncertainty quantification over fault scenarios
    - Efficient batch training with consistent array shapes

    See Also
    --------
    ComputeTSI_fast : TSI computation used internally
    create_training_samples : Simpler flat format for basic ML
    """
    overall_start = print_stage_header("Export ProbML Dataset")

    # -------------------------------------------------------------------------
    # Sub-stage 1: Load simulation log
    # -------------------------------------------------------------------------
    substage_start = time.time()
    print("\n--- Sub-stage 1/4: Loading simulation log ---")
    simulation_log = load_simulation_log()
    print(f"  Sub-stage 1 completed in {format_time(time.time() - substage_start)}")

    # -------------------------------------------------------------------------
    # Sub-stage 2: Compute TSI time series
    # -------------------------------------------------------------------------
    substage_start = time.time()
    print("\n--- Sub-stage 2/4: Computing TSI time series ---")
    if verbose:
        print("  Computing TSI time series via ComputeTSI_fast() ...")
    post_data = ComputeTSI_fast()

    # Extract results from TSI computation
    tsi_ts_per_scenario: Dict[str, np.ndarray] = post_data["tsi_ts_per_scenario"]
    pg_per_scenario: Dict[str, np.ndarray] = post_data["pg_per_scenario"]
    pl_per_scenario: Dict[str, np.ndarray] = post_data["pl_per_scenario"]
    ql_per_scenario: Dict[str, np.ndarray] = post_data["ql_per_scenario"]
    print(f"  Sub-stage 2 completed in {format_time(time.time() - substage_start)}")

    # -------------------------------------------------------------------------
    # Sub-stage 3: Build index structures
    # -------------------------------------------------------------------------
    substage_start = time.time()
    print("\n--- Sub-stage 3/4: Building index structures ---")

    def _load_qg_vector_for_scenario(scenario_id: str, n_gen_expected: int) -> np.ndarray:
        """
        Load generator reactive power (Qg) vector for a scenario.

        Parameters
        ----------
        scenario_id : str
            Scenario identifier.
        n_gen_expected : int
            Expected number of generators (for validation).

        Returns
        -------
        np.ndarray
            Generator Qg values, padded/trimmed to expected length if needed.
        """
        fn = simulation_log[scenario_id]["file"]
        if not os.path.exists(fn):
            return np.full(n_gen_expected, np.nan, dtype=float)

        with np.load(fn, mmap_mode="r") as z:
            if "q_gen_scaled" in z.files:
                qg = z["q_gen_scaled"]
                # Ensure length matches expected
                if qg.shape[0] != n_gen_expected:
                    qg = np.asarray(qg, dtype=float).reshape(-1)
                    if qg.shape[0] != n_gen_expected:
                        # Pad or trim to match expected size
                        if qg.shape[0] < n_gen_expected:
                            qg = np.pad(qg, (0, n_gen_expected - qg.shape[0]),
                                        constant_values=np.nan)
                        else:
                            qg = qg[:n_gen_expected]
                return qg.astype(float)
            else:
                return np.full(n_gen_expected, np.nan, dtype=float)

    def _last_tsi(ts) -> float:
        """
        Extract TSI value at the last time step.

        Parameters
        ----------
        ts : array-like
            TSI time series.

        Returns
        -------
        float
            TSI value at the final time step.
        """
        arr = np.asarray(ts)
        last = arr[..., -1]  # Last element along time axis
        if np.size(last) == 1:
            return float(np.asarray(last).reshape(-1)[0])
        # If extra axes remain, average over them (fallback)
        return float(np.nanmean(last))

    def _min_tsi(ts) -> float:
        """
        Extract minimum TSI value across all time steps.

        This represents the worst-case stability during the simulation,
        which is more conservative than using the final value.

        Parameters
        ----------
        ts : array-like
            TSI time series.

        Returns
        -------
        float
            Minimum TSI value across all time steps.
        """
        arr = np.asarray(ts)
        return float(np.nanmin(arr))

    # Select TSI extraction function based on mode
    valid_tsi_modes = {"final", "min"}
    if tsi_mode not in valid_tsi_modes:
        raise ValueError(
            f"Invalid tsi_mode '{tsi_mode}'. Must be one of: {valid_tsi_modes}"
        )
    
    tsi_extractor = _last_tsi if tsi_mode == "final" else _min_tsi
    tsi_mode_description = (
        "TSI at last time step" if tsi_mode == "final" 
        else "Minimum TSI across all time steps"
    )
    
    if verbose:
        print(f"  TSI extraction mode: '{tsi_mode}' ({tsi_mode_description})")

    # Get available scenarios (those with computed TSI)
    available_sids = set(tsi_ts_per_scenario.keys())
    print(f"  Available scenarios with TSI: {len(available_sids):,}")

    # Extract unique parameter values
    fault_locations = sorted({simulation_log[sid]["fault_location"] for sid in available_sids})
    fault_impedances = sorted({simulation_log[sid]["fault_impedance"] for sid in available_sids})
    sample_indices = sorted({simulation_log[sid]["sample_idx"] for sid in available_sids})
    F, Z = len(fault_locations), len(fault_impedances)

    if verbose:
        print(f"  Unique sample_idx: {len(sample_indices):,}; "
              f"fault locations: {F}; fault impedances: {Z}")
        print(f"  Total grid size per sample: {F} x {Z} = {F*Z}")

    # Build index: (sample_idx, fault_location, fault_impedance) -> scenario_id
    print("  Building index map...")
    idx_map: Dict[int, Dict[int, Dict[float, Optional[str]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for sid in available_sids:
        s = simulation_log[sid]["sample_idx"]
        f = simulation_log[sid]["fault_location"]
        z = simulation_log[sid]["fault_impedance"]
        idx_map[s][f][z] = sid

    print(f"  Sub-stage 3 completed in {format_time(time.time() - substage_start)}")

    # -------------------------------------------------------------------------
    # Sub-stage 4: Process samples
    # -------------------------------------------------------------------------
    substage_start = time.time()
    print("\n--- Sub-stage 4/4: Processing samples ---")

    # Prepare output containers
    kept_sample_idx = []
    X_rows_concat = []   # For concatenated mode: (2, Ngen+Nload)
    X_rows_gen = []      # For separate mode: (2, Ngen)
    X_rows_load = []     # For separate mode: (2, Nload)
    Y_rows = []
    SID_grid_rows = []

    # Progress tracking (~20 updates)
    sample_tracker = ProgressTracker(
        len(sample_indices), "Processing samples",
        report_interval=max(1, len(sample_indices) // 20)
    )

    skipped_incomplete = 0
    skipped_no_data = 0

    # Track dimensions (set from first valid sample)
    Ngen = None
    Nload = None

    # Iterate over sample indices; build one row per sample
    for s in sample_indices:
        # Check completeness of the fault grid for this sample
        complete = True
        sid_grid = np.empty((F, Z), dtype=object)

        for i, floc in enumerate(fault_locations):
            for j, fz in enumerate(fault_impedances):
                sid = idx_map.get(s, {}).get(floc, {}).get(fz, None)
                sid_grid[i, j] = sid
                if require_complete_grid and sid is None:
                    complete = False

        if require_complete_grid and not complete:
            # Skip samples with incomplete fault grids
            skipped_incomplete += 1
            sample_tracker.update()
            continue

        # Find a representative scenario (any available) to read inputs
        first_sid = next(
            (sid_grid[i, j] for i in range(F) for j in range(Z) if sid_grid[i, j] is not None),
            None
        )
        if first_sid is None:
            skipped_no_data += 1
            sample_tracker.update()
            continue

        # Load per-unit power vectors
        pg_vec = np.asarray(pg_per_scenario[first_sid], dtype=float)  # (Ngen,)
        pl_vec = np.asarray(pl_per_scenario[first_sid], dtype=float)  # (Nload,)
        ql_vec = np.asarray(ql_per_scenario[first_sid], dtype=float)  # (Nload,)
        n_gen = pg_vec.shape[0]
        n_load = pl_vec.shape[0]
        qg_vec = _load_qg_vector_for_scenario(first_sid, n_gen)       # (Ngen,)

        # Validate dimensions are consistent across samples
        if Ngen is None:
            Ngen, Nload = n_gen, n_load
            print(f"  Detected Ngen={Ngen}, Nload={Nload}")
        else:
            if (n_gen != Ngen) or (n_load != Nload):
                raise RuntimeError(
                    f"Inconsistent unit counts: expected (Ngen={Ngen}, Nload={Nload}), "
                    f"got ({n_gen}, {n_load}) for sample_idx={s}"
                )

        # Construct input array based on concatenation mode
        if concat_generators_and_loads:
            # Concatenate: [generators, loads] for both P and Q
            P_concat = np.concatenate([pg_vec, pl_vec], axis=0)  # (Ngen+Nload,)
            Q_concat = np.concatenate([qg_vec, ql_vec], axis=0)  # (Ngen+Nload,)
            X_row = np.stack([P_concat, Q_concat], axis=0)       # (2, Ngen+Nload)
            X_rows_concat.append(X_row)
        else:
            # Separate: generators and loads in different arrays
            X_rows_gen.append(np.stack([pg_vec, qg_vec], axis=0))   # (2, Ngen)
            X_rows_load.append(np.stack([pl_vec, ql_vec], axis=0))  # (2, Nload)

        # Build output array: TSI for each (floc, fz) using selected mode
        Y = np.full((F, Z), np.nan, dtype=float)
        for i in range(F):
            for j in range(Z):
                sid = sid_grid[i, j]
                if sid is None:
                    continue
                tsi_ts = tsi_ts_per_scenario[sid]
                Y[i, j] = tsi_extractor(tsi_ts)

        Y_rows.append(Y)
        SID_grid_rows.append(sid_grid.copy())
        kept_sample_idx.append(s)

        sample_tracker.update()

    sample_tracker.finish()

    print(f"\n  Samples kept: {len(kept_sample_idx):,}")
    print(f"  Samples skipped (incomplete grid): {skipped_incomplete:,}")
    print(f"  Samples skipped (no data): {skipped_no_data:,}")
    print(f"  Sub-stage 4 completed in {format_time(time.time() - substage_start)}")

    if len(kept_sample_idx) == 0:
        raise RuntimeError(
            "No rows were constructed; check that ComputeTSI() ran successfully "
            "and that scenarios exist."
        )

    # -------------------------------------------------------------------------
    # Final assembly and saving
    # -------------------------------------------------------------------------
    print("\n--- Finalizing and saving ---")
    save_start = time.time()

    # Stack inputs and outputs into arrays
    data_to_save: Dict[str, Any] = {}
    result: Dict[str, Any] = {}

    if concat_generators_and_loads:
        X = np.stack(X_rows_concat, axis=0)  # (N, 2, Ngen+Nload)
        data_to_save["X"] = X
        result["X"] = X
        if return_X_flat:
            X_flat = X.reshape(X.shape[0], 2 * X.shape[2])  # (N, 2*(Ngen+Nload))
            data_to_save["X_flat"] = X_flat
            result["X_flat"] = X_flat
    else:
        X_gen = np.stack(X_rows_gen, axis=0)    # (N, 2, Ngen)
        X_load = np.stack(X_rows_load, axis=0)  # (N, 2, Nload)
        data_to_save["X_gen"] = X_gen
        data_to_save["X_load"] = X_load
        result["X_gen"] = X_gen
        result["X_load"] = X_load

    # Stack output arrays
    Y = np.stack(Y_rows, axis=0)               # (N, F, Z)
    SID_grid = np.stack(SID_grid_rows, axis=0)  # (N, F, Z), dtype=object

    # Convert to appropriate array types
    kept_sample_idx = np.asarray(kept_sample_idx, dtype=int)
    fault_locations_arr = np.asarray(fault_locations, dtype=int)
    fault_impedances_arr = np.asarray(fault_impedances, dtype=float)

    # Add remaining arrays and metadata
    data_to_save.update({
        "Y": Y,
        "sample_idx": kept_sample_idx,
        "fault_locations": fault_locations_arr,
        "fault_impedances": fault_impedances_arr,
        "scenario_ids": SID_grid,
        "meta": np.array([{
            "inputs": "full_per_unit",
            "channels": ["P", "Q"],
            "unit_axis_order": "generators_then_loads",
            "Ngen": int(Ngen),
            "Nload": int(Nload),
            "require_complete_grid": bool(require_complete_grid),
            "concat_generators_and_loads": bool(concat_generators_and_loads),
            "return_X_flat": bool(return_X_flat),
            "tsi_mode": tsi_mode,
            "meaning_Y": f"{tsi_mode_description} for each (fault_location, fault_impedance)",
            "axes_Y": {"axis0": "fault_location", "axis1": "fault_impedance"},
        }], dtype=object),
    })

    # Save to compressed NumPy archive
    print(f"  Saving to '{out_path}'...")
    np.savez_compressed(out_path, **data_to_save)
    print(f"  Save completed in {format_time(time.time() - save_start)}")

    # Print summary
    if verbose:
        print(f"\n  === DATASET SUMMARY ===")
        print(f"  Constructed dataset with N={kept_sample_idx.shape[0]:,} rows.")
        if concat_generators_and_loads:
            print(f"  X shape: {data_to_save['X'].shape}  (layout: (N, 2, Ngen+Nload))")
            if return_X_flat:
                print(f"  X_flat shape: {data_to_save['X_flat'].shape}")
        else:
            print(f"  X_gen shape: {data_to_save['X_gen'].shape}; "
                  f"X_load shape: {data_to_save['X_load'].shape}")
        print(f"  Y shape: {Y.shape} (axis0=fault_location, axis1=fault_impedance)")
        print(f"  Saved dataset to '{out_path}'.")

    # Prepare return dictionary
    result.update({
        "Y": Y,
        "sample_idx": kept_sample_idx,
        "fault_locations": fault_locations_arr,
        "fault_impedances": fault_impedances_arr,
        "scenario_ids": SID_grid,
        "out_path": out_path,
    })

    print_stage_complete("Export ProbML Dataset", overall_start)
    return result


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """
    Main entry point for TSI analysis.

    Supports command-line arguments for configuring the analysis.

    Command-line Arguments
    ----------------------
    --output, -o : str
        Output file path (default: tsi_probml_fullinputs.npz)
    --tsi-mode : str
        TSI extraction mode: 'final' (default) or 'min'
    --require-complete-grid : flag
        Only include samples with complete fault grids
    --no-concat : flag
        Save separate X_gen and X_load arrays instead of concatenated X
    --no-flat : flag
        Don't save flattened X_flat array

    Examples
    --------
    Use final TSI (default)::

        $ python TSI_analysis.py

    Use minimum TSI across all time steps::

        $ python TSI_analysis.py --tsi-mode min

    Custom output path with minimum TSI::

        $ python TSI_analysis.py -o my_dataset.npz --tsi-mode min
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="TSI Analysis - Compute and export Transient Stability Index datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
TSI Modes:
  final   Use TSI at the last time step (steady-state stability)
  min     Use minimum TSI across all time steps (worst-case stability)

Examples:
  python TSI_analysis.py                          # Default: final TSI
  python TSI_analysis.py --tsi-mode min           # Minimum TSI
  python TSI_analysis.py -o output.npz --tsi-mode min
        """
    )
    parser.add_argument(
        "--output", "-o",
        default="tsi_probml_fullinputs.npz",
        help="Output file path (default: tsi_probml_fullinputs.npz)"
    )
    parser.add_argument(
        "--tsi-mode",
        choices=["final", "min"],
        default="final",
        help="TSI extraction mode: 'final' (last time step) or 'min' (minimum over time)"
    )
    parser.add_argument(
        "--require-complete-grid",
        action="store_true",
        help="Only include samples where all fault combinations converged"
    )
    parser.add_argument(
        "--no-concat",
        action="store_true",
        help="Save separate X_gen and X_load arrays instead of concatenated X"
    )
    parser.add_argument(
        "--no-flat",
        action="store_true",
        help="Don't save flattened X_flat array"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("TSI ANALYSIS - STARTING")
    print("=" * 70)
    print(f"  Output file: {args.output}")
    print(f"  TSI mode: {args.tsi_mode}")
    print(f"  Require complete grid: {args.require_complete_grid}")
    print(f"  Concatenate gen/load: {not args.no_concat}")
    print(f"  Return X_flat: {not args.no_flat}")
    print("=" * 70)
    
    overall_start = time.time()

    ret = export_probml_dataset(
        out_path=args.output,
        require_complete_grid=args.require_complete_grid,
        concat_generators_and_loads=not args.no_concat,
        return_X_flat=not args.no_flat,
        verbose=not args.quiet,
        tsi_mode=args.tsi_mode,
    )

    print("\n" + "=" * 70)
    print(f"TSI ANALYSIS - COMPLETE")
    print(f"Total runtime: {format_time(time.time() - overall_start)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
