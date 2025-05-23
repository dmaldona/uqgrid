import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any

def load_simulation_log(file_path: str = 'simulation_log.json') -> Dict:
    """Load the simulation log containing metadata about all scenarios."""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_state_metadata(file_path: str = 'state_metadata.json') -> Dict:
    """Load the state metadata that describes all state variables."""
    with open(file_path, 'r') as f:
        return json.load(f)

def filter_scenarios(
    simulation_log: Dict, 
    base_load: Optional[float] = None, 
    fault_location: Optional[int] = None,
    fault_impedance: Optional[float] = None,
    diverged: Optional[bool] = None
) -> Dict:
    """Filter scenarios based on criteria."""
    filtered_log = {}
    
    for scenario_id, data in simulation_log.items():
        match = True
        
        if base_load is not None and data['base_load'] != base_load:
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
    device_id: Optional[str] = None,
    bus_num: Optional[int] = None,
    state_name: Optional[str] = None
) -> List[int]:
    """Find indices of states matching the criteria."""
    indices = []
    
    for idx, (state_idx, data) in enumerate(state_metadata.items()):
        match = True
        
        if model is not None and data.get('model') != model:
            match = False
        if device_id is not None and str(data.get('device_id')) != str(device_id):
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
    
    data = np.load(file_path)
    return {
        'history': data['history'],
        'tvec': data['tvec'],
        'metadata': simulation_log[scenario_id]
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
    
    state_values = history[state_idx, :]
    return tvec, state_values

def get_state_timeseries_all(
    simulation_log: Dict,
    state_metadata: Dict,
    model: Optional[str] = None,
    device_id: Optional[str] = None,
    bus_num: Optional[int] = None,
    state_name: Optional[str] = None,
    base_load: Optional[float] = None,
    fault_location: Optional[int] = None,
    fault_impedance: Optional[float] = None,
    diverged: Optional[bool] = False
) -> Dict[str, Dict[str, Union[np.ndarray, Any]]]:
    """Extract a state variable from multiple scenarios with filtering."""
    # Filter scenarios
    filtered_scenarios = filter_scenarios(
        simulation_log, 
        base_load, 
        fault_location,
        fault_impedance,
        diverged
    )
    
    # Find state indices
    state_indices = find_state_index(
        state_metadata, 
        model, 
        device_id, 
        bus_num,
        state_name
    )
    
    if not state_indices:
        raise ValueError(f"No states found matching criteria: model={model}, device_id={device_id}, state_name={state_name}")
    
    state_idx = state_indices[0]  # Take the first match if multiple
    
    # Extract data for each scenario
    results = {}
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
            print(f"Error loading scenario {scenario_id}: {e}")
    
    return results

def plot_state_comparison(
    results: Dict[str, Dict[str, Union[np.ndarray, Any]]],
    title: str = None,
    xlabel: str = 'Time (s)',
    ylabel: str = None,
    legend_key: str = 'base_load'
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
    # Load metadata
    simulation_log = load_simulation_log()
    state_metadata = load_state_metadata()
    
    # Get all generator 1 speed deviations from non-diverged simulations
    results = get_state_timeseries_all(
        simulation_log,
        state_metadata,
        model='GenGENROU',
        device_id='1',
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
        device_id='1',
        state_name='w',
        base_load=1.0,
        diverged=False
    )
    
    plt = plot_state_comparison(
        results_filtered,
        title='Generator 1 Speed Deviation (Load Level = 1.0)',
        ylabel='Speed Deviation (pu)',
        legend_key='fault_location'
    )
    
    plt.savefig('gen1_speed_comparison_load1.png')
    
    print(f"Found {len(results)} non-diverged scenarios")
    print(f"Found {len(results_filtered)} non-diverged scenarios at base_load=1.0")
    
    return results

def ComputeTSI():
    # Load metadata
    simulation_log = load_simulation_log()
    state_metadata = load_state_metadata()
    
    # 1) find all (device_id, bus_num) pairs for GenGENROU δ-states
    gen_pairs = {
        (str(data['device_id']), data['bus_num'])
        for data in state_metadata.values()
        if data.get('model')     == 'GenGENROU'
        and data.get('state_name') == 'delta'
    }

    # sort so results are deterministic
    gen_list = sorted(gen_pairs, key=lambda x: (x[1], x[0]))  # sort by bus, then device

    # 2) pull the raw dict for each generator
    delta_dicts = {}
    for device_id, bus_num in gen_list:
        print(f"⟳ loading δ for GenGENROU device {device_id} on bus {bus_num}")
        d = get_state_timeseries_all(
            simulation_log,
            state_metadata,
            model='GenGENROU',
            device_id=device_id,
            bus_num=bus_num,
            state_name='delta',
            diverged=False
        )

        delta_dicts[(bus_num, device_id)] = d

    if not delta_dicts:
        raise RuntimeError("No generator deltas were loaded!")

    # find common scenarios
    scenario_sets    = [ set(d.keys()) for d in delta_dicts.values() ]
    common_scenarios = sorted(set.intersection(*scenario_sets))
    if not common_scenarios:
        raise RuntimeError("No scenario is common to all generators!")

    # pick one generator‐key and one scenario 
    first_key      = next(iter(delta_dicts))                # t(bus_num, device_id)
    first_scenario = next(iter(delta_dicts[first_key]))     # scenario_id string

    # get tvec safely
    tvec = delta_dicts[first_key][first_scenario]['tvec']
    T    = len(tvec)
    G    = len(delta_dicts)
    S    = len(common_scenarios)

    # build 3D δ-array
    delta_arr = np.zeros((G, T, S))
    for g_idx, key in enumerate(delta_dicts):
        for s_idx, scen in enumerate(common_scenarios):
            delta_arr[g_idx, :, s_idx] = delta_dicts[key][scen]['values']



    # TSI (scalar) for each scenario 
    tsi_per_scenario = {}
    for s_idx, scen in enumerate(common_scenarios):
        spread_ts   = delta_arr[:, :, s_idx].max(axis=0) - delta_arr[:, :, s_idx].min(axis=0)
        Delta_max       = spread_ts.max()                      # maximum spread (radians) over time
        tsi_scalar  = (2*np.pi - Delta_max) / (2*np.pi + Delta_max) * 100
        tsi_per_scenario[scen] = tsi_scalar

    # TSI time-series for each scenario 
    tsi_ts_per_scenario = {}
    for s_idx, scen in enumerate(common_scenarios):
        spread_ts = delta_arr[:, :, s_idx].max(axis=0) - delta_arr[:, :, s_idx].min(axis=0)
        tsi_ts    = (2*np.pi - spread_ts) / (2*np.pi + spread_ts) * 100  # shape (T,)
        tsi_ts_per_scenario[scen] = tsi_ts

    # TSI for all scenarios (scalar array) 
    tsi_all        = np.array([tsi_per_scenario[sc] for sc in common_scenarios])  # shape (S,) in the same order

    # TSI time-series for all scenarios (2D array)
    # shape (S, T), row s is the time-series for scenario common_scenarios
    tsi_all_time   = np.vstack([tsi_ts_per_scenario[sc] for sc in common_scenarios])

    #  tsi_per_scenario    : dict { scenario_id -> scalar TSI }
    #  tsi_ts_per_scenario : dict { scenario_id -> array of TSI over time }
    #  tsi_all             : np.ndarray, shape (S,), all scalar TSI in order
    #  tsi_all_time        : np.ndarray, shape (S, T), all TSI time-series in order   
    post_data={}
    post_data['tsi_per_scenario']=tsi_per_scenario
    post_data['tsi_ts_per_scenario']=tsi_ts_per_scenario
    post_data['tsi_all']=tsi_all
    post_data['tsi_all_time']=tsi_all_time

    print(f'TSI for all scenarios: {tsi_all.shape}')
    print(f'TSI for all time scenarios: {tsi_all_time.shape}')
    
    try:
        # Plotting the TSI
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
        print("Seaborn is not installed. Please install it if you want to see cool stuff with `pip install seaborn`.")
        
    return post_data

if __name__ == "__main__":
    # (original) compute generator speeds (ω)
    #example_gen1_speed_deviation()

    ComputeTSI()
