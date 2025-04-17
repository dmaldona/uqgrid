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
    state_name: Optional[str] = None,
    bus_num: Optional[int] = None
) -> List[int]:
    """Find indices of states matching the criteria."""
    indices = []
    
    for idx, (state_idx, data) in enumerate(state_metadata.items()):
        match = True
        
        if model is not None and data.get('model') != model:
            match = False
        if device_id is not None and str(data.get('device_id')) != str(device_id):
            match = False
        if state_name is not None and data.get('state_name') != state_name:
            match = False
        if bus_num is not None and data.get('bus_num') != bus_num:
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
    state_name: Optional[str] = None,
    bus_num: Optional[int] = None,
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
        state_name,
        bus_num
    )
    
    if not state_indices:
        criteria = {k: v for k, v in {
            "model": model, 
            "device_id": device_id, 
            "state_name": state_name,
            "bus_num": bus_num
        }.items() if v is not None}
        
        raise ValueError(f"No states found matching criteria: {criteria}")
    
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

def get_states_by_bus(
    simulation_log: Dict,
    state_metadata: Dict,
    bus_num: int,
    scenario_id: Optional[str] = None,
    state_type: Optional[str] = None,
    diverged: Optional[bool] = False
) -> Dict[str, Dict]:
    """Get all states associated with a specific bus for a given scenario.
    
    Args:
        simulation_log: Dictionary of simulation metadata
        state_metadata: Dictionary of state variable metadata
        bus_num: Bus number to filter by
        scenario_id: Optional specific scenario to analyze
        state_type: Optional filter by state type ('Differential', 'Algebraic', 'Network Voltage')
        diverged: Whether to include diverged simulations
        
    Returns:
        Dictionary of state data indexed by state description
    """
    # Find all states belonging to the specified bus
    states = {}
    for state_idx, data in state_metadata.items():
        if data.get('bus_num') == bus_num:
            if state_type is None or data.get('type') == state_type:
                states[state_idx] = data
    
    if not states:
        raise ValueError(f"No states found for bus {bus_num}")
    
    # If no specific scenario provided, use the first non-diverged scenario
    if scenario_id is None:
        filtered_scenarios = filter_scenarios(simulation_log, diverged=diverged)
        if not filtered_scenarios:
            raise ValueError("No suitable scenarios found")
        scenario_id = next(iter(filtered_scenarios))
    
    # Get data for the scenario
    scenario_data = load_scenario_data(scenario_id, simulation_log)
    
    # Extract all relevant states
    results = {}
    for state_idx, state_info in states.items():
        tvec, values = get_state_timeseries(scenario_data, int(state_idx))
        
        # Create a descriptive key
        if 'model' in state_info and 'state_name' in state_info:
            key = f"{state_info['model']}_{state_info['state_name']}"
        else:
            key = f"{state_info['type']}_{state_info['state_name']}"
            
        results[key] = {
            'tvec': tvec,
            'values': values,
            'metadata': state_info
        }
    
    return results

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

def example_bus_analysis():
    """Plot generator speed deviations at bus 2."""
    simulation_log = load_simulation_log()
    state_metadata = load_state_metadata()

    # Find all generator speed deviation states at bus 2
    speed_states = {}
    for state_idx, data in state_metadata.items():
        # Ensure bus_num is an int and matches 2
        try:
            bus_match = int(data.get('bus_num', -1)) == 2
        except Exception:
            bus_match = False

        # Flexible generator and speed state matching
        is_generator = 'model' in data and 'gen' in data['model'].lower()
        is_speed = 'state_name' in data and any(
            key in data['state_name'].lower() for key in ['w', 'omega', 'speed', 'dw', 'delta_omega']
        )

        if bus_match and is_generator and is_speed:
            speed_states[state_idx] = data

    if not speed_states:
        print("No speed deviation states found for generators at bus 2")
        return

    # Use the first non-diverged scenario
    filtered_scenarios = filter_scenarios(simulation_log, diverged=False)
    if not filtered_scenarios:
        print("No suitable scenarios found")
        return

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))

    for scenario_id in filtered_scenarios:
        scenario_data = load_scenario_data(scenario_id, simulation_log)
        for state_idx, state_info in speed_states.items():
            tvec, values = get_state_timeseries(scenario_data, int(state_idx))
            label = f"{state_info.get('model', 'Unknown')} {state_info.get('device_id', 'Unknown')} ({state_info.get('state_name', 'Unknown')}) | Scenario {scenario_id}"
            plt.plot(tvec, values, label=label, alpha=0.5, color='gray')

    plt.title("Generator Speed Deviations at Bus 2 (All Scenarios)")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed Deviation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('bus2_speed_deviations_all_scenarios.png')
    print(f"Plotted speed deviations for {len(filtered_scenarios)} scenarios and {len(speed_states)} states.")
    return speed_states

if __name__ == "__main__":
    example_gen1_speed_deviation()
    example_bus_analysis()