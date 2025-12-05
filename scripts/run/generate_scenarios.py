import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import itertools
import numpy as np
import uuid
import json
import gc
import copy
import time
import traceback
from datetime import timedelta
from joblib import Parallel, delayed

from uqgrid.simulation.dynamics import integrate_system
from uqgrid.simulation.config   import IntegrationConfig
from uqgrid.io.parse            import load_psse, add_dyr


def generate_perturbations(base_p, base_q,
                                *,
                                noise_type="normal", var=0.1,
                                rng=None, return_noise=False):
    """
    Apply per-bus noise -> return scaled loads.
    If return_noise=True, also return (p_noise, q_noise)

        P_scaled = base_p * (1 + p_noise)
        Q_scaled = base_q * (1 + q_noise)
    """
    rng = np.random.default_rng() if rng is None else rng

    if noise_type == "normal":
        p_noise = rng.normal(0.0, var, size=base_p.shape)
        q_noise = rng.normal(0.0, var, size=base_q.shape)
    elif noise_type == "uniform":
        half = np.sqrt(3 * var)              # Var(U[-a,a]) = var
        p_noise = rng.uniform(-half, half, size=base_p.shape)
        q_noise = rng.uniform(-half, half, size=base_q.shape)
    elif noise_type == "none":
        p_noise = q_noise = np.zeros_like(base_p)
    else:
        raise ValueError(f"Unknown noise_type '{noise_type}'")

    p_scaled = base_p * (1.0 + p_noise)
    q_scaled = base_q * (1.0 + q_noise)

    if return_noise:
        return p_scaled, q_scaled, p_noise, q_noise
    return p_scaled, q_scaled


def sample_scenarios(n_samples, fault_locations, fault_impedances):
    return list(itertools.product(range(n_samples), fault_locations, fault_impedances))


def generate_metadata(scenarios):
    """
    Creates one UUID per scenario.
    """
    metadata = {}
    for sample_idx, floc, fz in scenarios:
        sid = str(uuid.uuid4())
        metadata[sid] = {
            "sample_idx"     : sample_idx,
            "fault_location" : floc,
            "fault_impedance": fz,
        }
    with open("scenario_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    return metadata


def run_single_scenario_worker(
        raw_file, dyr_file, scenario, scenario_id,
        noise_type="normal", noise_var=0.1,
        global_seed=0,
        balance_generation=False):
    """
    Modified to load the model inside the worker process to avoid MPI issues.
    Each worker loads its own copy of the model.
    """
    
    try:
        # Load fresh model in worker process - fix MPI communicator issues
        psys = load_psse(raw_file)
        add_dyr(psys, dyr_file)
        
        # Get base loads and generation
        base_p_load, base_q_load = psys.get_load_pq()
        base_p_gen, base_q_gen = psys.get_gen_pq()
        
        # Set up RNG
        ss = np.random.SeedSequence([global_seed, scenario["sample_idx"]])
        rng_load, rng_gen = [np.random.default_rng(s) for s in ss.spawn(2)]
        
        # Draw noise and obtain scaled loads
        pL_scaled, qL_scaled, pL_noise, qL_noise = generate_perturbations(
            base_p_load, base_q_load,
            noise_type=noise_type, var=noise_var, rng=rng_load,
            return_noise=True)
        
        pG_scaled, qG_scaled, pG_noise, qG_noise = generate_perturbations(
            base_p_gen, base_q_gen,
            noise_type=noise_type, var=noise_var, rng=rng_gen,
            return_noise=True)
        
        if balance_generation:
            sum_pL = np.sum(pL_scaled)
            sum_qL = np.sum(qL_scaled)
            sum_pG = np.sum(pG_scaled)
            sum_qG = np.sum(qG_scaled)
            
            if sum_pG != 0: pG_scaled *= (sum_pL / sum_pG)
            if sum_qG != 0: qG_scaled *= (sum_qL / sum_qG)
        
        psys.set_load_pq(pL_scaled, qL_scaled)
        psys.set_gen_pq(pG_scaled, qG_scaled)
        
        psys.add_busfault(scenario["fault_location"],
                          scenario["fault_impedance"])
        psys.createYbusComplex()
        
        cfg = IntegrationConfig(
            tend=10.0, dt=1/120.0, power_injection=False,
            ton=0.25, toff=0.4, verbose=False, petsc=True
        )
        
        try:
            sim = integrate_system(psys, cfg)
            diverged = False
        except Exception as e:
            print(f"Simulation failed for scenario {scenario_id}: {str(e)}")
            sim = {"history": None, "tvec": None}
            diverged = True
        
        # Save results
        os.makedirs("simulation_data", exist_ok=True)
        fn = f"simulation_data/scenario_{scenario_id}.npz"
        np.savez_compressed(
            fn,
            history=sim["history"],
            tvec=sim["tvec"],
            # loads
            p_load_scaled=pL_scaled, q_load_scaled=qL_scaled,
            p_load_noise=pL_noise, q_load_noise=qL_noise,
            # generators
            p_gen_scaled=pG_scaled, q_gen_scaled=qG_scaled,
            p_gen_noise=pG_noise, q_gen_noise=qG_noise,
        )
        
        # Clean up - important for PETSc/MPI resources
        del sim
        del psys
        gc.collect()
        
        return {"file": fn, "diverged": diverged}
        
    except Exception as e:
        print(f"Worker error for scenario {scenario_id}: {str(e)}")
        traceback.print_exc()
        return {"file": None, "diverged": True, "error": str(e)}


def run_simulation_driver_batched(
        raw, dyr, scenarios_metadata,
        *, noise_type="normal", noise_var=0.1,
        balance_generation=True,
        n_jobs=-1, batch_size=10, 
        checkpoint_interval=100):
    """
    Batched driver with error handling and checkpointing.
    """
    
    scenario_ids = list(scenarios_metadata.keys())
    simulation_log = {}
    
    # Load checkpoint if it exists
    checkpoint_file = "simulation_checkpoint.json"
    start_batch = 0
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)
                simulation_log = checkpoint.get("simulation_log", {})
                start_batch = checkpoint.get("last_batch", 0) + 1
                print(f"Resuming from batch {start_batch}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    
    t0 = time.time()
    total_batches = int(np.ceil(len(scenario_ids) / batch_size))
    
    for batch_idx, batch_start in enumerate(range(start_batch * batch_size, 
                                                   len(scenario_ids), 
                                                   batch_size), 
                                             start=start_batch):
        batch_ids = scenario_ids[batch_start : batch_start + batch_size]
        
        # Progress tracking
        elapsed = time.time() - t0
        if batch_idx > start_batch:
            est_total = elapsed * (total_batches - start_batch) / (batch_idx - start_batch)
            remaining = max(0.0, est_total - elapsed)
        else:
            est_total = remaining = 0
        
        print(
            f"Processing batch {batch_idx + 1} / {total_batches} | "
            f"elapsed {timedelta(seconds=int(elapsed))} | "
            f"est total {timedelta(seconds=int(est_total))} | "
            f"ETA {timedelta(seconds=int(remaining))}"
        )
        
        # Export state metadata once per batch (outside parallel region)
        base_psys = load_psse(raw)
        add_dyr(base_psys, dyr)
        base_psys.export_state_metadata()
        del base_psys  # Clean up immediately
        
        # Prepare arguments for parallel execution
        batch_args = [
            (raw, dyr, scenarios_metadata[sid], sid,
             noise_type, noise_var, 1234, balance_generation)
            for sid in batch_ids
        ]
        
        try:
            #TODO make surethe timeout does not limit the scenario run
            # Use loky backend with timeout protection
            batch_out = Parallel(n_jobs=n_jobs, backend='loky', timeout=600)(
                delayed(run_single_scenario_worker)(*args) for args in batch_args)
            
            # Update log
            for sid, out in zip(batch_ids, batch_out):
                simulation_log[sid] = {**scenarios_metadata[sid], **out}
            
        except Exception as e:
            print(f"Batch {batch_idx + 1} failed: {str(e)}")
            # Mark failed scenarios
            for sid in batch_ids:
                if sid not in simulation_log:
                    simulation_log[sid] = {
                        **scenarios_metadata[sid],
                        "file": None,
                        "diverged": True,
                        "error": f"Batch failure: {str(e)}"
                    }
        
        # Save progress
        with open("simulation_log.json", "w") as f:
            json.dump(simulation_log, f, indent=4)
        
        # Checkpoint periodically
        if (batch_idx + 1) % checkpoint_interval == 0:
            with open(checkpoint_file, "w") as f:
                json.dump({
                    "last_batch": batch_idx,
                    "simulation_log": simulation_log
                }, f, indent=4)
            print(f"Checkpoint saved at batch {batch_idx + 1}")
        
        # Force garbage collection between batches
        gc.collect()
    
    # Clean up checkpoint on completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    return simulation_log


def main():
    PowerGridModel = "IEEE-9"
    
    # Scenario sampling configuration
    SAMPLES_PER_FAULT_LOCATION = 5     # Samples per fault location
    FAULT_IMPEDANCES = [0.00001]         # Fault impedance values [p.u]
    
    # ADJUSTED PARAMETERS FOR STABILITY
    N_JOBS = 5                           
    BATCH_SIZE = 10                       
    CHECKPOINT_INTERVAL = 5              # Save checkpoint every 50 batches
    
    if PowerGridModel == "IEEE-9":
        raw = "../data/ieee9_v33.raw"
        dyr = "../data/ieee9bus_gov.dyr"
        n_bus = 9
    elif PowerGridModel == "IEEE-39":
        raw = "data/IEEE39_v33.raw"
        dyr = "data/IEEE39_gov.dyr"
        n_bus = 39
    elif PowerGridModel == "ACTIVSg200":
        raw = "data/ACTIVSg200.raw"
        dyr = "data/ACTIVSg200.dyr"
        n_bus = 200
    elif PowerGridModel == "ACTIVSg500":
        raw = "data/ACTIVSg500.raw"
        dyr = "data/ACTIVSg500.dyr"
        n_bus = 500
    else:
        raise RuntimeError(f"{PowerGridModel} is an invalid model!")
    
    fault_locations = list(range(0, n_bus))
    
    # Calculate total scenarios
    total_scenarios = SAMPLES_PER_FAULT_LOCATION * len(fault_locations) * len(FAULT_IMPEDANCES)
    print(f"Configuration: {total_scenarios} total scenarios")
    print(f"  - {SAMPLES_PER_FAULT_LOCATION} noise samples per fault location")
    print(f"  - {len(fault_locations)} fault locations.")
    print(f"  - {len(FAULT_IMPEDANCES)} fault impedances: {FAULT_IMPEDANCES}")
    print(f"  - N_JOBS: {N_JOBS}, BATCH_SIZE: {BATCH_SIZE}")
    print(f"  - Checkpoint interval: {CHECKPOINT_INTERVAL} batches")
    
    scenarios = sample_scenarios(
        SAMPLES_PER_FAULT_LOCATION, fault_locations, FAULT_IMPEDANCES)
    metadata = generate_metadata(scenarios)
    
    # Noise settings
    noise_type = "normal"
    noise_var = 0.25
    balance_generation = True
    
    run_simulation_driver_batched(
        raw, dyr, metadata,
        noise_type=noise_type, noise_var=noise_var,
        balance_generation=balance_generation,
        n_jobs=N_JOBS, batch_size=BATCH_SIZE,
        checkpoint_interval=CHECKPOINT_INTERVAL)


if __name__ == "__main__":
    main()
