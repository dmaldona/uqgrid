import itertools
import numpy as np
import uuid
import json
import os
import copy
from joblib import Parallel, delayed

from uqgrid.simulation.dynamics import integrate_system
from uqgrid.simulation.config   import IntegrationConfig
from uqgrid.io.parse            import load_psse, add_dyr


def generate_load_perturbations(base_p, base_q,
                                *,
                                noise_type="normal", var=0.1,
                                rng=None, return_noise=False):
    """
    Apply per-bus noise -> return scaled loads.
    If return_noise=True, also return (p_noise, q_noise)

        P_scaled = base_p * (1 + p_noise)
        Q_scaled = base_q * (1 + q_noise)
    TODO: may need to change it so it's more flexible.
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
    Creates one UUID per scenario.  Metadata no longer contains 'base_load'.
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


def run_single_scenario(
        base_psys, scenario, scenario_id,
        base_p_load, base_q_load,
        noise_type="normal", noise_var=0.1):

    psys = copy.deepcopy(base_psys)

    #  Draw noise and obtain positive, scaled loads
    p_scaled, q_scaled, p_noise, q_noise = generate_load_perturbations(
        base_p_load, base_q_load,
        noise_type=noise_type, var=noise_var,
        return_noise=True)

    psys.set_load_pq(p_scaled, q_scaled)

    psys.add_busfault(scenario["fault_location"],
                      scenario["fault_impedance"], 0.25)
    psys.createYbusComplex()

    cfg = IntegrationConfig(
        tend=10.0, dt=1/120.0, power_injection=False,
        ton=0.25, toff=0.4, verbose=False, petsc=True
    )

    try:
        sim       = integrate_system(psys, cfg)
        diverged  = False
    except Exception:
        sim       = {"history": None, "tvec": None}
        diverged  = True

    os.makedirs("simulation_data", exist_ok=True)
    fn = f"simulation_data/scenario_{scenario_id}.npz"
    np.savez_compressed(
        fn,
        history=sim["history"],
        tvec=sim["tvec"],
        p_scaled=p_scaled,
        q_scaled=q_scaled,
        p_noise=p_noise,
        q_noise=q_noise
    )
    return {"file": fn, "diverged": diverged}


def run_simulation_driver_batched(
        raw, dyr, scenarios_metadata,
        *, noise_type="normal", noise_var=0.1,
        n_jobs=-1, batch_size=10):

    scenario_ids   = list(scenarios_metadata.keys())
    simulation_log = {}

    for batch_start in range(0, len(scenario_ids), batch_size):
        batch_ids = scenario_ids[batch_start : batch_start+batch_size]
        print(f"Processing batch {batch_start//batch_size + 1}"
              f" / {int(np.ceil(len(scenario_ids)/batch_size))}")

        base_psys = load_psse(raw)
        add_dyr(base_psys, dyr)
        base_psys.export_state_metadata()

        base_p, base_q = base_psys.get_load_pq()

        batch_args = [
            (base_psys, scenarios_metadata[sid], sid,
             base_p, base_q, noise_type, noise_var)
            for sid in batch_ids
        ]

        batch_out = Parallel(n_jobs=n_jobs)(
            delayed(run_single_scenario)(*args) for args in batch_args)

        for sid, out in zip(batch_ids, batch_out):
            simulation_log[sid] = {**scenarios_metadata[sid], **out}

        with open("simulation_log.json", "w") as f:
            json.dump(simulation_log, f, indent=4)

        del base_psys

    return simulation_log

def main():
    PowerGridModel = "IEEE-9"
    if PowerGridModel == "IEEE-9":
        raw = "data/ieee9_v33.raw"
        dyr = "data/ieee9bus_gov.dyr"
        n_bus = 9
    elif PowerGridModel == "IEEE-39":
        raw = "data/IEEE39_v33.raw"
        dyr = "data/IEEE39_gov.dyr"
        n_bus = 39
    else:
        raise RuntimeError(f"{PowerGridModel} is an invalid model!")

    number_of_samples = 50             
    fault_locations   = list(range(1, n_bus + 1))
    fault_impedances  = [0.0001]

    scenarios = sample_scenarios(
        number_of_samples, fault_locations, fault_impedances)
    metadata  = generate_metadata(scenarios)

    # noise settings
    noise_type = "normal"   # "normal", "uniform", "none", 
    noise_var  = 0.10       # variance of the chosen distribution TODO: need to change this to be more flexible

    run_simulation_driver_batched(
        raw, dyr, metadata,
        noise_type=noise_type, noise_var=noise_var,
        n_jobs=5, batch_size=10)


if __name__ == "__main__":
    main()
