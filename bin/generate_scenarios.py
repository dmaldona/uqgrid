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


def generate_load_perturbations(base_p, base_q, *, noise_type="normal", var=0.1, rng=None):
    """
    Return two arrays (p_noise, q_noise) with the same shapes as base_p / base_q

    The numbers represent *relative* perturbations, i.e. P -> P*(1+noise)
    """
    rng = np.random.default_rng() if rng is None else rng

    if noise_type == "normal":
        p_noise = rng.normal(loc=0.0, scale=var, size=base_p.shape)
        q_noise = rng.normal(loc=0.0, scale=var, size=base_q.shape)
    elif noise_type == "uniform":
        half_width = np.sqrt(3 * var)       # so that Var(U[-a,a]) = var
        p_noise = rng.uniform(-half_width,  half_width, size=base_p.shape)
        q_noise = rng.uniform(-half_width,  half_width, size=base_q.shape)
    elif noise_type == "none":
        p_noise = np.zeros_like(base_p)
        q_noise = np.zeros_like(base_q)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    return p_noise, q_noise
# ────────────────────────────────────────────────────────────────────────────────


def sample_scenarios(load_range, fault_locations, fault_impedances):
    return list(itertools.product(load_range, fault_locations, fault_impedances))


def generate_metadata(scenarios):
    metadata = {}
    for load_scale, floc, fz in scenarios:
        sid = str(uuid.uuid4())
        metadata[sid] = {
            "base_load"      : load_scale,
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

    p_noise, q_noise = generate_load_perturbations(
        base_p_load, base_q_load, noise_type=noise_type, var=noise_var)


    p_load_scaled = base_p_load * scenario["base_load"] * (1.0 + p_noise)
    q_load_scaled = base_q_load * scenario["base_load"] * (1.0 + q_noise)
    psys.set_load_pq(p_load_scaled, q_load_scaled)

    psys.add_busfault(scenario["fault_location"], scenario["fault_impedance"], 0.25)
    psys.createYbusComplex()

    cfg = IntegrationConfig(
        tend=10.0, dt=1/120.0, power_injection=False,
        ton=0.25, toff=0.4, verbose=False, petsc=True
    )

    try:
        sim = integrate_system(psys, cfg)
        diverged = False
    except Exception:
        sim      = {"history": None, "tvec": None}
        diverged = True

    os.makedirs("simulation_data", exist_ok=True)
    fn = f"simulation_data/scenario_{scenario_id}.npz"
    np.savez_compressed(fn, history=sim["history"], tvec=sim["tvec"])

    return {
        "file"     : fn,
        "diverged" : diverged,
        "p_noise"  : p_noise.tolist(),
        "q_noise"  : q_noise.tolist(),
    }


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
    PowerGridModel = "IEEE-9"           # or "IEEE-9"
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

    load_range       = np.random.uniform(0.75, 1.25, size=50)
    fault_locations  = list(range(1, n_bus + 1))
    fault_impedances = [0.0001]
    scenarios        = sample_scenarios(load_range, fault_locations, fault_impedances)
    metadata         = generate_metadata(scenarios)

    # noise settings 
    noise_type = "normal"   # "normal", "uniform", "none", …
    noise_var  = 0.10       # variance of the chosen distribution

    run_simulation_driver_batched(
        raw, dyr, metadata,
        noise_type=noise_type, noise_var=noise_var,
        n_jobs=5, batch_size=10)


if __name__ == "__main__":
    main()
