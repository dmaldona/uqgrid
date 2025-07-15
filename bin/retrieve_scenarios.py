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
    fault_location: Optional[int] = None,
    fault_impedance: Optional[float] = None,
    sample_idx: Optional[int] = None,
    diverged: Optional[bool] = None
) -> Dict:
    """Filter scenarios based on criteria."""
    filtered_log = {}
    
    for scenario_id, data in simulation_log.items():
        match = True
        
        if fault_location is not None and data['fault_location'] != fault_location:
            match = False
        if fault_impedance is not None and data['fault_impedance'] != fault_impedance:
            match = False
        if sample_idx is not None and data['sample_idx'] != sample_idx:
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
    state_name: Optional[str] = None,
    bus_num: Optional[int] = None
) -> List[int]:
    """Find indices of states matching the criteria."""
    indices = []
    
    for idx, (state_idx, data) in enumerate(state_metadata.items()):
        match = True
        
        if model is not None and data.get('model') != model:
            match = False
        if device_number is not None and str(data.get('device_number')) != str(device_number):
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
    device_number: Optional[str] = None,
    state_name: Optional[str] = None,
    bus_num: Optional[int] = None,
    fault_location: Optional[int] = None,
    fault_impedance: Optional[float] = None,
    sample_idx: Optional[int] = None,
    diverged: Optional[bool] = False
) -> Dict[str, Dict[str, Union[np.ndarray, Any]]]:
    """Extract a state variable from multiple scenarios with filtering."""
    # Filter scenarios
    filtered_scenarios = filter_scenarios(
        simulation_log, 
        fault_location,
        fault_impedance,
        sample_idx,
        diverged
    )
    
    # Find state indices
    state_indices = find_state_index(
        state_metadata, 
        model, 
        device_number, 
        state_name,
        bus_num
    )
    
    if not state_indices:
        criteria = {k: v for k, v in {
            "model": model, 
            "device_number": device_number, 
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
    legend_key: str = 'fault_location'  # Changed from 'base_load' to 'fault_location'
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

def example_all_generator_speeds():
    """Example that plots speed deviations for all generators."""
    simulation_log = load_simulation_log()
    state_metadata = load_state_metadata()
    
    # Get speed deviations for all generators from non-diverged simulations
    results = get_state_timeseries_all(
        simulation_log,
        state_metadata,
        model='GenGENROU',
        state_name='w',
        diverged=False
    )
    
    plt = plot_state_comparison(
        results,
        title='All Generator Speed Deviations',
        ylabel='Speed Deviation (pu)',
        legend_key='fault_location'
    )
    
    plt.savefig('all_generator_speeds.png')
    plt.close()
    
    print(f"Found {len(results)} non-diverged scenarios with generator speed data")
    return results

def example_generator4_states():
    """Example that plots speed and angle for generator with device ID 4."""
    simulation_log = load_simulation_log()
    state_metadata = load_state_metadata()
    
    # Get speed (w) and angle (delta) for generator 4
    speed_results = get_state_timeseries_all(
        simulation_log,
        state_metadata,
        model='GenGENROU',
        device_number='4',
        state_name='w',
        diverged=False
    )
    
    angle_results = get_state_timeseries_all(
        simulation_log,
        state_metadata,
        model='GenGENROU',
        device_number='4',
        state_name='delta',
        diverged=False
    )
    
    # Create subplot figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot speed deviations
    plt.sca(ax1)
    for scenario_id, data in speed_results.items():
        tvec = data['tvec']
        values = data['values']
        if len(tvec) > 0 and len(values) > 0:
            ax1.plot(tvec, values, color='blue', alpha=0.3)
    
    ax1.set_title('Generator 4 Speed Deviation')
    ax1.set_ylabel('Speed Deviation (pu)')
    ax1.grid(True)
    
    # Plot rotor angles
    plt.sca(ax2)
    for scenario_id, data in angle_results.items():
        tvec = data['tvec']
        values = data['values']
        if len(tvec) > 0 and len(values) > 0:
            ax2.plot(tvec, values, color='red', alpha=0.3)
    
    ax2.set_title('Generator 4 Rotor Angle')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Rotor Angle (rad)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('generator4_speed_angle.png')
    plt.close()
    
    print(f"Found {len(speed_results)} scenarios with Generator 4 speed data")
    print(f"Found {len(angle_results)} scenarios with Generator 4 angle data")
    
    return speed_results, angle_results

if __name__ == "__main__":
    print("Running generator analysis examples...")
    
    print("1. All generator speed deviations")
    example_all_generator_speeds()
    
    print("2. Generator 4 speed and angle analysis")
    example_generator4_states()
    
    print("Analysis complete! Check the generated PNG files.")
