import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import scipy.io as scio
from collections import defaultdict
import time

class ProgressTracker:
    """Utility class for tracking and reporting progress with timing and ETA."""
    
    def __init__(self, total: int, description: str = "Processing", report_interval: int = 100):
        self.total = total
        self.description = description
        self.report_interval = report_interval
        self.start_time = time.time()
        self.count = 0
        self.last_report_time = self.start_time
        
    def update(self, increment: int = 1) -> None:
        """Update progress counter and print report if at interval."""
        self.count += increment
        if self.count % self.report_interval == 0 or self.count == self.total:
            self._print_progress()
    
    def _print_progress(self) -> None:
        """Print progress report with timing and ETA."""
        elapsed = time.time() - self.start_time
        pct = (self.count / self.total) * 100 if self.total > 0 else 0
        rate = self.count / elapsed if elapsed > 0 else 0
        
        if rate > 0 and self.count < self.total:
            remaining = (self.total - self.count) / rate
            eta_str = format_time(remaining)
        else:
            eta_str = "N/A"
        
        print(f"  [{self.description}] {self.count:,}/{self.total:,} ({pct:.1f}%) | "
              f"Elapsed: {format_time(elapsed)} | Rate: {rate:.1f}/s | ETA: {eta_str}")
    
    def finish(self) -> float:
        """Mark as complete and return total elapsed time."""
        elapsed = time.time() - self.start_time
        print(f"  [{self.description}] COMPLETE: {self.count:,} items in {format_time(elapsed)}")
        return elapsed


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
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
    """Print a stage header and return the start time."""
    print("\n" + "=" * 70)
    print(f"STAGE: {stage_name}")
    print("=" * 70)
    return time.time()


def print_stage_complete(stage_name: str, start_time: float) -> None:
    """Print stage completion with timing."""
    elapsed = time.time() - start_time
    print(f"\n✓ {stage_name} completed in {format_time(elapsed)}")
    print("-" * 70)


def load_simulation_log(file_path: str = 'simulation_log.json') -> Dict:
    """Load the simulation log containing metadata about all scenarios."""
    print(f"  Loading simulation log from '{file_path}'...", end=" ", flush=True)
    start = time.time()
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"done ({len(data):,} scenarios, {format_time(time.time() - start)})")
    return data

def load_state_metadata(file_path: str = 'state_metadata.json') -> Dict:
    """Load the state metadata that describes all state variables."""
    print(f"  Loading state metadata from '{file_path}'...", end=" ", flush=True)
    start = time.time()
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"done ({len(data):,} states, {format_time(time.time() - start)})")
    return data

def filter_scenarios(
    simulation_log: Dict, 
    sample_idx: Optional[int] = None,
    fault_location: Optional[int] = None,
    fault_impedance: Optional[float] = None,
    diverged: Optional[bool] = None
) -> Dict:
    """Filter scenarios based on criteria."""
    filtered_log = {}
    
    for scenario_id, data in simulation_log.items():
        match = True
        
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
    """Find indices of states matching the criteria."""
    indices = []
    
    for idx, (state_idx, data) in enumerate(state_metadata.items()):
        match = True
        
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

def load_scenario_data(scenario_id: str, simulation_log: Dict) -> Dict:
    """Load data for a specific scenario."""
    file_path = simulation_log[scenario_id]['file']
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Scenario data file not found: {file_path}")
    
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
    """Extract a specific state variable from a scenario."""
    tvec = scenario_data['tvec']
    history = scenario_data['history']
    
    if history is None:
        return np.array([]), np.array([])
    
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
    """Extract a state variable from multiple scenarios with filtering."""
    # Filter scenarios
    filtered_scenarios = filter_scenarios(
        simulation_log, 
        sample_idx,
        fault_location,
        fault_impedance,
        diverged
    )
    
    # Diverged scenarios - should comment this out for performance
    filtered_scenarios_div = filter_scenarios(
        simulation_log, 
        sample_idx,
        fault_location,
        fault_impedance,
        diverged=True
    )
    print(f"  Scenarios that did not diverge: {len(filtered_scenarios):,}; scenarios that diverged: {len(filtered_scenarios_div):,}")
    # Free up the memory
    filtered_scenarios_div = None

    # Find state indices
    state_indices = find_state_index(
        state_metadata, 
        model, 
        device_number, 
        bus_num,
        state_name
    )
    
    if not state_indices:
        raise ValueError(f"No states found matching criteria: model={model}, device_number={device_number}, state_name={state_name}")
    
    state_idx = state_indices[0]  # Take the first match if multiple
    
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

def plot_state_comparison(
    results: Dict[str, Dict[str, Union[np.ndarray, Any]]],
    title: str = None,
    xlabel: str = 'Time (s)',
    ylabel: str = None,
    legend_key: str = 'fault_location'
):
    """Plot comparison of state variables from multiple scenarios."""
    plt.figure(figsize=(10, 6))
    
    for scenario_id, data in results.items():
        tvec = data['tvec']
        values = data['values']
        metadata = data['metadata']
        
        # Skip empty results (e.g., from diverged simulations)
        if len(tvec) == 0 or len(values) == 0:
            continue
        
        plt.plot(tvec, values, color='gray', alpha=0.5)
    
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    
    return plt

# Example usage
def example_gen1_speed_deviation():
    """Example that extracts generator 1 speed deviation for all scenarios."""
    stage_start = print_stage_header("Example: Generator 1 Speed Deviation")
    
    # Load metadata
    simulation_log = load_simulation_log()
    state_metadata = load_state_metadata()
    
    # Get all generator 1 speed deviations from non-diverged simulations
    results = get_state_timeseries_all(
        simulation_log,
        state_metadata,
        model='GenGENROU',
        device_number='1',
        state_name='w',
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

def ComputeTSI_fast():
    stage_start = print_stage_header("ComputeTSI_fast")
    
    # Sub-stage 1: Load metadata
    substage_start = time.time()
    print("\n--- Sub-stage 1/4: Loading metadata ---")
    simulation_log = load_simulation_log()
    state_metadata = load_state_metadata()
    print(f"  Sub-stage 1 completed in {format_time(time.time() - substage_start)}")
    
    # Sub-stage 2: Find delta state indices
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

    # Sub-stage 3: Identify scenarios to process
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
    
    # Dimensions from the first scenario
    print("  Loading first scenario for dimensions...")
    first = load_scenario_data(scenario_ids[0], simulation_log)
    tvec0 = np.asarray(first['tvec'])
    T = tvec0.shape[0]
    N = len(scenario_ids)
    twopi = 2.0 * np.pi
    print(f"  Time steps per scenario: {T}")
    print(f"  Total scenarios to process: {N:,}")

    # Preallocate outputs (same shapes you build later)
    tsi_time = np.empty((N, T), dtype=float)  # per-scenario TSI(t)
    tsi_scalar = np.empty(N, dtype=float)     # scalar TSI per scenario

    # Keep P/Q vectors for later consumers, as before
    pg_per_scenario, pl_per_scenario, ql_per_scenario = {}, {}, {}

    # Progress tracking
    tracker = ProgressTracker(N, "Processing scenarios", report_interval=max(1, N // 20))
    
    # Main pass: load each scenario ONCE, compute all δ-based quantities vectorized
    for i, sid in enumerate(scenario_ids):
        sc = load_scenario_data(sid, simulation_log)
        H = sc['history']  # shape: (Nstates, T), memmap view

        # One vectorized slice for all generator δ rows (no Python loop, no extra copies)
        deltas = H[delta_state_idxs, :]              # (G, T) view
        spread = deltas.max(axis=0) - deltas.min(axis=0)  # (T,)

        tsi_t = (twopi - spread) / (twopi + spread) * 100.0  # (T,)
        tsi_time[i] = tsi_t
        tsi_scalar[i] = float(tsi_t.min())  # same as using Δ_max in your original

        # Cache P/Q vectors
        pg_per_scenario[sid] = np.asarray(sc['p_gen_scaled'])
        pl_per_scenario[sid] = np.asarray(sc['p_load_scaled'])
        ql_per_scenario[sid] = np.asarray(sc['q_load_scaled'])
        
        tracker.update()
    
    tracker.finish()
    print(f"  Sub-stage 4 completed in {format_time(time.time() - substage_start)}")

    # Rebuild the dict API you already use elsewhere
    print("\n  Building result dictionaries...")
    tsi_ts_per_scenario = {sid: tsi_time[i] for i, sid in enumerate(scenario_ids)}
    tsi_per_scenario    = {sid: tsi_scalar[i] for i, sid in enumerate(scenario_ids)}


    # Plotting code
    try:
        import seaborn as sns
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        sns.histplot(tsi_scalar, bins=20, stat='density', kde=True)
        plt.xlabel('TSI at all times')
        plt.ylabel('Density')

        plt.subplot(1,2,2)
        sns.histplot(np.squeeze(tsi_time[:,-1]), bins=20, stat='density', kde=True)
        plt.xlabel('TSI at final time')

        for i in range(2):
            plt.subplot(1,2,i+1)
            plt.axvline(0, color='k', ls='--', lw=1.5)  
            plt.text(-1, plt.ylim()[1] * 0.95, 'unstable', ha='right', va='top', fontsize=10)
            plt.text(1, plt.ylim()[1] * 0.95, 'stable', ha='left', va='top', fontsize=10)

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("  Seaborn is not installed. Please install it if you want to see cool stuff with `pip install seaborn`.")


    post_data = {
        'tsi_per_scenario': tsi_per_scenario,
        'tsi_ts_per_scenario': tsi_ts_per_scenario,
        'tsi_all': tsi_scalar,          # (N,)
        'tsi_all_time': tsi_time,       # (N, T)
        'pg_per_scenario': pg_per_scenario,
        'pl_per_scenario': pl_per_scenario,
        'ql_per_scenario': ql_per_scenario,
    }
    
    print_stage_complete("ComputeTSI_fast", stage_start)
    return post_data


def ComputeTSI():
    stage_start = print_stage_header("ComputeTSI")
    
    # Sub-stage 1: Load metadata
    substage_start = time.time()
    print("\n--- Sub-stage 1/4: Loading metadata ---")
    simulation_log = load_simulation_log()
    state_metadata = load_state_metadata()
    print(f"  Sub-stage 1 completed in {format_time(time.time() - substage_start)}")
    
    # Sub-stage 2: Find generator pairs
    substage_start = time.time()
    print("\n--- Sub-stage 2/4: Finding generator delta states ---")
    gen_pairs = {
        (str(data['device_number']), data['bus_num'])
        for data in state_metadata.values()
        if data.get('model') == 'GenGENROU'
        and data.get('state_name') == 'delta'
    }
    gen_list = sorted(gen_pairs, key=lambda x: (x[1], x[0]))
    print(f"  Found {len(gen_list)} generators with delta states")
    print(f"  Sub-stage 2 completed in {format_time(time.time() - substage_start)}")

    # Sub-stage 3: Load generator delta data
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

    # find common scenarios
    scenario_sets = [set(d.keys()) for d in delta_dicts.values()]
    common_scenarios = sorted(set.intersection(*scenario_sets))
    if not common_scenarios:
        raise RuntimeError("No scenario is common to all generators!")

    print(f"\n  Found {len(common_scenarios):,} common scenarios across all generators")

    # Get dimensions
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
    scenario_tracker = ProgressTracker(len(common_scenarios), "Computing TSI", report_interval=max(1, len(common_scenarios) // 20))

    # Process scenarios one by one (memory efficient)
    for s_idx, scenario_id in enumerate(common_scenarios):
        # Load scenario data once for pg, pl, ql
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

        # Compute TSI for this scenario only
        spread_ts = delta_values.max(axis=0) - delta_values.min(axis=0)
        Delta_max = spread_ts.max()
        tsi_scalar = (2*np.pi - Delta_max) / (2*np.pi + Delta_max) * 100
        tsi_ts = (2*np.pi - spread_ts) / (2*np.pi + spread_ts) * 100

        tsi_per_scenario[scenario_id] = tsi_scalar
        tsi_ts_per_scenario[scenario_id] = tsi_ts

        # Free memory for this scenario
        del delta_values, spread_ts, tsi_ts
        
        scenario_tracker.update()
    
    scenario_tracker.finish()
    print(f"  Sub-stage 4 completed in {format_time(time.time() - substage_start)}")

    # Clear the delta_dicts to free memory before creating final arrays
    print("\n  Freeing intermediate memory...")
    del delta_dicts

    # Create final arrays  
    print("  Creating final arrays...")
    tsi_all = np.array([tsi_per_scenario[sc] for sc in common_scenarios])
    tsi_all_time = np.vstack([tsi_ts_per_scenario[sc] for sc in common_scenarios])

    # Package results
    post_data = {}
    post_data['tsi_per_scenario'] = tsi_per_scenario
    post_data['tsi_ts_per_scenario'] = tsi_ts_per_scenario
    post_data['tsi_all'] = tsi_all
    post_data['tsi_all_time'] = tsi_all_time
    post_data['pg_per_scenario'] = pg_per_scenario
    post_data['pl_per_scenario'] = pl_per_scenario
    post_data['ql_per_scenario'] = ql_per_scenario

    print(f'\n  TSI for all scenarios: {tsi_all.shape}')
    print(f'  TSI for all time scenarios: {tsi_all_time.shape}')
    
    # Plotting code
    try:
        import seaborn as sns
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        sns.histplot(tsi_all, bins=20, stat='density', kde=True)
        plt.xlabel('TSI at all times')
        plt.ylabel('Density')

        plt.subplot(1,2,2)
        sns.histplot(np.squeeze(tsi_all_time[:,-1]), bins=20, stat='density', kde=True)
        plt.xlabel('TSI at final time')

        for i in range(2):
            plt.subplot(1,2,i+1)
            plt.axvline(0, color='k', ls='--', lw=1.5)  
            plt.text(-1, plt.ylim()[1] * 0.95, 'unstable', ha='right', va='top', fontsize=10)
            plt.text(1, plt.ylim()[1] * 0.95, 'stable', ha='left', va='top', fontsize=10)

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("  Seaborn is not installed. Please install it if you want to see cool stuff with `pip install seaborn`.")
    
    print_stage_complete("ComputeTSI", stage_start)
    return post_data

def create_training_samples(post_data: Dict):
    stage_start = print_stage_header("Create Training Samples")
    
    tsi_dict = post_data['tsi_per_scenario']
    pg_dict  = post_data['pg_per_scenario']
    pl_dict  = post_data['pl_per_scenario']
    ql_dict  = post_data['ql_per_scenario']

    scenario_ids = sorted(tsi_dict.keys())
    print(f"  Processing {len(scenario_ids):,} scenarios...")

    first_sid = scenario_ids[0]
    pg_len = len(pg_dict[first_sid])
    pl_len = len(pl_dict[first_sid])
    ql_len = len(ql_dict[first_sid])

    col_name = (
        [f'pg_{i+1}' for i in range(pg_len)] +
        [f'pl_{i+1}' for i in range(pl_len)] +
        [f'ql_{i+1}' for i in range(ql_len)] +
        ['tsi']
    )

    rows = []
    tracker = ProgressTracker(len(scenario_ids), "Building rows", report_interval=max(1, len(scenario_ids) // 10))
    
    for sid in scenario_ids:
        pg = pg_dict[sid]
        pl = pl_dict[sid]
        ql = ql_dict[sid]
        tsi = np.array([tsi_dict[sid]])
    
        row = np.hstack((pg, pl, ql, tsi))
        rows.append(row)
        tracker.update()
    
    tracker.finish()

    Data = np.vstack(rows)

    # save to .mat file
    print("  Saving samples to data_record.mat...")
    scio.savemat('data_record.mat', {'Data': Data, 'col_name': col_name})
    
    print_stage_complete("Create Training Samples", stage_start)


def export_probml_dataset(
    out_path: str = "tsi_probml_dataset.npz",
    require_complete_grid: bool = True,
    concat_generators_and_loads: bool = True,
    return_X_flat: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Build a dataset suitable for probabilistic ML, learning the distribution of
    TSI at the last time step given all per-unit inputs.

    Inputs (per row):
        - A 2-channel array with shape (2, N), where N = Ngen + Nload.
          Channel 0 holds active powers (P), channel 1 holds reactive powers (Q).
          The unit axis is the concatenation of generators followed by loads,
          preserving the original order within each group.

          If concat_generators_and_loads=False, return separate arrays:
            - X_gen:  (N, 2, Ngen)  with channels [P_g, Q_g]
            - X_load: (N, 2, Nload) with channels [P_l, Q_l]

        - A flattened view X_flat with shape (N, 2*(Ngen+Nload)).

    Outputs (per row):
        - Y: a grid of TSI at the last time step for all (fault_location, fault_impedance):
              Y.shape = (F, Z)

    Parameters
    ----------
    out_path : str
        Path to save the dataset (.npz). Saved contents depend on the concat option:
          If concat_generators_and_loads:
            - X:        float64 array (N, 2, Ngen+Nload)
            - X_flat:   float64 array (N, 2*(Ngen+Nload))  [if return_X_flat]
          Else:
            - X_gen:    float64 array (N, 2, Ngen)
            - X_load:   float64 array (N, 2, Nload)

          Saved:
            - Y:        float64 array (N, F, Z)  (TSI at last time step) TODO: make this an additional option
            - sample_idx: int array (N,)
            - fault_locations:  int array (F,)
            - fault_impedances: float64 array (Z,)
            - scenario_ids:     object array (N, F, Z)
            - meta:     dict with a few notes including Ngen/Nload and unit ordering

    require_complete_grid : bool
        If True, keep only sample_idx that have all (fault_location, fault_impedance)
        combinations available and non-diverged. If False, missing entries are NaN.
    concat_generators_and_loads : bool
        If True, build a single X of shape (N, 2, Ngen+Nload). If False, save X_gen and X_load separately.
    return_X_flat : bool
        If True (and concat_generators_and_loads=True), also save a flattened view X_flat: (N, 2*(Ngen+Nload)).
    verbose : bool
        Print progress and summary information.

    Returns
    -------
    Dict[str, Any]
        A dictionary with is saved to `out_path`.
    """
    overall_start = print_stage_header("Export ProbML Dataset")

    # Sub-stage 1: Load simulation log
    substage_start = time.time()
    print("\n--- Sub-stage 1/4: Loading simulation log ---")
    simulation_log = load_simulation_log()
    print(f"  Sub-stage 1 completed in {format_time(time.time() - substage_start)}")

    # Sub-stage 2: Compute TSI
    substage_start = time.time()
    print("\n--- Sub-stage 2/4: Computing TSI time series ---")
    if verbose:
        print("  Computing TSI time series via ComputeTSI_fast() ...")
    post_data = ComputeTSI_fast()
    tsi_ts_per_scenario: Dict[str, np.ndarray] = post_data["tsi_ts_per_scenario"]
    pg_per_scenario: Dict[str, np.ndarray]     = post_data["pg_per_scenario"]
    pl_per_scenario: Dict[str, np.ndarray]     = post_data["pl_per_scenario"]
    ql_per_scenario: Dict[str, np.ndarray]     = post_data["ql_per_scenario"]
    print(f"  Sub-stage 2 completed in {format_time(time.time() - substage_start)}")

    # Sub-stage 3: Build index structures
    substage_start = time.time()
    print("\n--- Sub-stage 3/4: Building index structures ---")
    
    # Load generator Q (qg) vector for a scenario from artifact NPZ.
    def _load_qg_vector_for_scenario(scenario_id: str, n_gen_expected: int) -> np.ndarray:
        fn = simulation_log[scenario_id]["file"]
        if not os.path.exists(fn):
            return np.full(n_gen_expected, np.nan, dtype=float)
        with np.load(fn, mmap_mode="r") as z:
            if "q_gen_scaled" in z.files:
                qg = z["q_gen_scaled"]
                # Safety: ensure length matches expected
                if qg.shape[0] != n_gen_expected:
                    qg = np.asarray(qg, dtype=float).reshape(-1)
                    if qg.shape[0] != n_gen_expected:
                        # pad or trim to match
                        if qg.shape[0] < n_gen_expected:
                            qg = np.pad(qg, (0, n_gen_expected - qg.shape[0]), constant_values=np.nan)
                        else:
                            qg = qg[:n_gen_expected]
                return qg.astype(float)
            else:
                return np.full(n_gen_expected, np.nan, dtype=float)

    # Extract last-time-step TSI robustly (works for 1D or ND with time on the last axis)
    def _last_tsi(ts) -> float:
        arr = np.asarray(ts)
        last = arr[..., -1]  # last element along the last axis (time)
        if np.size(last) == 1:
            return float(np.asarray(last).reshape(-1)[0])
        return float(np.nanmean(last))  # TODO: fix: if an extra axis remains, average over it

    # Consider only scenarios where we actually have TSI time series
    available_sids = set(tsi_ts_per_scenario.keys())
    print(f"  Available scenarios with TSI: {len(available_sids):,}")

    # Sorted sets
    fault_locations = sorted({simulation_log[sid]["fault_location"] for sid in available_sids})
    fault_impedances = sorted({simulation_log[sid]["fault_impedance"] for sid in available_sids})
    sample_indices = sorted({simulation_log[sid]["sample_idx"] for sid in available_sids})
    F, Z = len(fault_locations), len(fault_impedances)

    if verbose:
        print(f"  Unique sample_idx: {len(sample_indices):,}; fault locations: {F}; fault impedances: {Z}")
        print(f"  Total grid size per sample: {F} x {Z} = {F*Z}")

    # Build index: (sample_idx, fault_location, fault_impedance) -> scenario_id
    print("  Building index map...")
    idx_map: Dict[int, Dict[int, Dict[float, Optional[str]]]] = defaultdict(lambda: defaultdict(dict))
    for sid in available_sids:
        s = simulation_log[sid]["sample_idx"]
        f = simulation_log[sid]["fault_location"]
        z = simulation_log[sid]["fault_impedance"]
        idx_map[s][f][z] = sid
    
    print(f"  Sub-stage 3 completed in {format_time(time.time() - substage_start)}")

    # Sub-stage 4: Process samples
    substage_start = time.time()
    print("\n--- Sub-stage 4/4: Processing samples ---")
    
    # Prepare output containers
    kept_sample_idx = []
    X_rows_concat = []      # will hold (2, Ngen+Nload) if concat_generators_and_loads
    X_rows_gen = []         # will hold (2, Ngen)   if not concat
    X_rows_load = []        # will hold (2, Nload)  if not concat
    Y_rows = []
    SID_grid_rows = []

    # Progress tracking
    sample_tracker = ProgressTracker(len(sample_indices), "Processing samples", 
                                     report_interval=max(1, len(sample_indices) // 20))
    
    skipped_incomplete = 0
    skipped_no_data = 0

    # Iterate over sample_idx; build one row per sample
    Ngen = None
    Nload = None
    for s in sample_indices:
        # Check completeness of the grid for this sample s
        complete = True
        sid_grid = np.empty((F, Z), dtype=object)
        for i, floc in enumerate(fault_locations):
            for j, fz in enumerate(fault_impedances):
                sid = idx_map.get(s, {}).get(floc, {}).get(fz, None)
                sid_grid[i, j] = sid
                if require_complete_grid and sid is None:
                    complete = False
        if require_complete_grid and not complete:
            # skip incomplete rows for this sample
            skipped_incomplete += 1
            sample_tracker.update()
            continue

        # Representative sid (any available for this sample) to read inputs
        first_sid = next((sid_grid[i, j] for i in range(F) for j in range(Z) if sid_grid[i, j] is not None), None)
        if first_sid is None:
            skipped_no_data += 1
            sample_tracker.update()
            continue  # nothing to keep

        # Per-unit vectors (no reduction)
        pg_vec = np.asarray(pg_per_scenario[first_sid], dtype=float)  # (Ngen,)
        pl_vec = np.asarray(pl_per_scenario[first_sid], dtype=float)  # (Nload,)
        ql_vec = np.asarray(ql_per_scenario[first_sid], dtype=float)  # (Nload,)
        n_gen = pg_vec.shape[0]
        n_load = pl_vec.shape[0]
        qg_vec = _load_qg_vector_for_scenario(first_sid, n_gen)       # (Ngen,)

        # Validate sizes are consistent across rows
        if Ngen is None:
            Ngen, Nload = n_gen, n_load
            print(f"  Detected Ngen={Ngen}, Nload={Nload}")
        else:
            if (n_gen != Ngen) or (n_load != Nload):
                raise RuntimeError(f"Inconsistent unit counts across scenarios: expected (Ngen={Ngen}, Nload={Nload}), got ({n_gen}, {n_load}) for sample_idx={s}")

        # Construct per-sample input
        if concat_generators_and_loads:
            P_concat = np.concatenate([pg_vec, pl_vec], axis=0)  # (Ngen+Nload,)
            Q_concat = np.concatenate([qg_vec, ql_vec], axis=0)  # (Ngen+Nload,)
            X_row = np.stack([P_concat, Q_concat], axis=0)       # (2, Ngen+Nload)
            X_rows_concat.append(X_row)
        else:
            X_rows_gen.append(np.stack([pg_vec, qg_vec], axis=0))  # (2, Ngen)
            X_rows_load.append(np.stack([pl_vec, ql_vec], axis=0)) # (2, Nload)

        # Outputs (TSI at last time step for each (floc,fz))
        Y = np.full((F, Z), np.nan, dtype=float)
        for i in range(F):
            for j in range(Z):
                sid = sid_grid[i, j]
                if sid is None:
                    continue
                tsi_ts = tsi_ts_per_scenario[sid]
                Y[i, j] = _last_tsi(tsi_ts)
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
        raise RuntimeError("No rows were constructed; check that ComputeTSI() ran successfully and that scenarios exist.")

    # Final assembly and saving
    print("\n--- Finalizing and saving ---")
    save_start = time.time()
    
    # Stack inputs / outputs
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
        X_gen = np.stack(X_rows_gen, axis=0)      # (N, 2, Ngen)
        X_load = np.stack(X_rows_load, axis=0)    # (N, 2, Nload)
        data_to_save["X_gen"] = X_gen
        data_to_save["X_load"] = X_load
        result["X_gen"] = X_gen
        result["X_load"] = X_load

    Y = np.stack(Y_rows, axis=0)                  # (N, F, Z)
    SID_grid = np.stack(SID_grid_rows, axis=0)    # (N, F, Z), dtype=object
    kept_sample_idx = np.asarray(kept_sample_idx, dtype=int)
    fault_locations_arr = np.asarray(fault_locations, dtype=int)
    fault_impedances_arr = np.asarray(fault_impedances, dtype=float)

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
            "meaning_Y": "TSI at last time step for each (fault_location, fault_impedance)",
            "axes_Y": {"axis0": "fault_location", "axis1": "fault_impedance"},
        }], dtype=object),
    })

    # Save
    print(f"  Saving to '{out_path}'...")
    np.savez_compressed(out_path, **data_to_save)
    print(f"  Save completed in {format_time(time.time() - save_start)}")

    if verbose:
        print(f"\n  === DATASET SUMMARY ===")
        print(f"  Constructed dataset with N={kept_sample_idx.shape[0]:,} rows.")
        if concat_generators_and_loads:
            print(f"  X shape: {data_to_save['X'].shape}  (layout: (N, 2, Ngen+Nload))")
            if return_X_flat:
                print(f"  X_flat shape: {data_to_save['X_flat'].shape}")
        else:
            print(f"  X_gen shape: {data_to_save['X_gen'].shape}; X_load shape: {data_to_save['X_load'].shape}")
        print(f"  Y shape: {Y.shape} (axis0=fault_location, axis1=fault_impedance)")
        print(f"  Saved dataset to '{out_path}'.")

    # Prepare return dict
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


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TSI ANALYSIS - STARTING")
    print("=" * 70)
    overall_start = time.time()
    
    # (original) compute generator speeds (w)
    #example_gen1_speed_deviation()

    #post_data = ComputeTSI()
    #create_training_samples(post_data)

    ret = export_probml_dataset(
        out_path="tsi_probml_fullinputs.npz",
        require_complete_grid=False,  
        concat_generators_and_loads=True,  # => X: (N, 2, Ngen+Nload)
        return_X_flat=True,                # also save X_flat: (N, 2*(Ngen+Nload))
        verbose=True)
    
    print("\n" + "=" * 70)
    print(f"TSI ANALYSIS - COMPLETE")
    print(f"Total runtime: {format_time(time.time() - overall_start)}")
    print("=" * 70)
