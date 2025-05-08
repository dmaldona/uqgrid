import itertools
import numpy as np
import uuid
import json
import os
import copy
from joblib import Parallel, delayed

def sample_scenarios(load_range, fault_locations, fault_impedances):
    scenarios = list(itertools.product(load_range, fault_locations, fault_impedances))
    return scenarios

def generate_metadata(scenarios):
    metadata = {}
    for scenario in scenarios:
        scenario_id = str(uuid.uuid4())
        metadata[scenario_id] = {
            'base_load': scenario[0],
            'fault_location': scenario[1],
            'fault_impedance': scenario[2]
        }
    with open('scenario_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    return metadata

import numpy as np
from uqgrid.simulation.dynamics import integrate_system
from uqgrid.simulation.config import IntegrationConfig
from uqgrid.io.parse import load_psse, add_dyr

def run_single_scenario(base_psys, scenario, scenario_id, base_p_load, base_q_load):
    # Make a deep copy of the base system
    psys = copy.deepcopy(base_psys)
    #TODO modify here to perturb P and Q separately 
    # Get the base load values and scale them according to scenario
    p_load_scaled = base_p_load * scenario['base_load']
    q_load_scaled = base_q_load * scenario['base_load']
    
    # Set the scaled loads
    psys.set_load_pq(p_load_scaled, q_load_scaled)
    
    psys.add_busfault(scenario['fault_location'], scenario['fault_impedance'], 0.25)
    psys.createYbusComplex()
    
    config = IntegrationConfig(
        tend=10.0,
        dt=1.0/120.0,
        power_injection=False,
        ton=0.25,
        toff=0.4,
        verbose=False,
        petsc=True
    )

    try:
        results = integrate_system(psys, config)
        divergence_flag = False
    except Exception as e:
        results = {'history': None, 'tvec': None}
        divergence_flag = True

    # Create simulation_data directory if it doesn't exist
    os.makedirs('simulation_data', exist_ok=True)
    
    filename = f"simulation_data/scenario_{scenario_id}.npz"
    np.savez_compressed(filename, history=results['history'], tvec=results['tvec'])
    
    return divergence_flag, filename

def run_simulation_driver(raw, dyr, scenarios_metadata):
    # Create base psys object once
    base_psys = load_psse(raw)
    add_dyr(base_psys, dyr)
    base_psys.export_state_metadata()
    
    # Get the base load values
    base_p_load, base_q_load = base_psys.get_load_pq()
    
    simulation_log = {}
    for scenario_id, params in scenarios_metadata.items():
        print(f"Running scenario {scenario_id}...")
        divergence, filename = run_single_scenario(
            base_psys, params, scenario_id, base_p_load, base_q_load
        )
        simulation_log[scenario_id] = {
            'file': filename,
            'diverged': divergence,
            **params
        }

        # Save periodically if large dataset
        with open('simulation_log.json', 'w') as f:
            json.dump(simulation_log, f, indent=4)

def run_simulation_driver_batched(raw, dyr, scenarios_metadata, n_jobs=-1, batch_size=10):
    """Run scenarios in parallel using joblib, with batching for memory management"""
    scenario_ids = list(scenarios_metadata.keys())
    num_scenarios = len(scenario_ids)
    simulation_log = {}
    
    for i in range(0, num_scenarios, batch_size):
        print(f"Processing batch {i//batch_size + 1} of {(num_scenarios + batch_size - 1)//batch_size}")
        batch_ids = scenario_ids[i:i+batch_size]
        
        # Create fresh psys object for each batch
        base_psys = load_psse(raw)
        add_dyr(base_psys, dyr)
        base_psys.export_state_metadata()
        
        # Get the base load values
        base_p_load, base_q_load = base_psys.get_load_pq()
        
        # Prepare batch arguments
        batch_args = [
            (base_psys, scenarios_metadata[sid], sid, base_p_load, base_q_load)
            for sid in batch_ids
        ]
        
        # Run batch in parallel
        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(run_single_scenario)(*args) for args in batch_args
        )
        
        # Process batch results
        for j, (divergence, filename) in enumerate(batch_results):
            scenario_id = batch_ids[j]
            simulation_log[scenario_id] = {
                'file': filename,
                'diverged': divergence,
                **scenarios_metadata[scenario_id]
            }
            
        # Save progress after each batch
        with open('simulation_log.json', 'w') as f:
            json.dump(simulation_log, f, indent=4)
            
        # Clean up memory
        del base_psys
        
    return simulation_log

def main():
    raw = "data/ieee9_v33.raw"
    dyr = "data/ieee9bus_gov.dyr"
    
    #load_range = np.linspace(0.75, 1.25, 5)  # Example load scaling
    #TODO add separrate P Q scaling
    load_range = np.random.uniform(low=0.75, high=1.25, size=50)

    fault_locations = [1, 2, 3, 4, 5, 6, 7, 8, 9]         # Example bus indices
    fault_impedances = [0.0001]   # Example fault impedances
    
    scenarios = sample_scenarios(load_range, fault_locations, fault_impedances)
    scenarios_metadata = generate_metadata(scenarios)

    #run_simulation_driver(raw, dyr, scenarios_metadata)
    run_simulation_driver_batched(raw, dyr, scenarios_metadata, n_jobs=5, batch_size=10)


if __name__ == "__main__":
    main()