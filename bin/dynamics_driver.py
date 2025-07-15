import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from uqgrid.core.psydef import Psystem
from uqgrid.simulation.dynamics import integrate_system
from uqgrid.io.parse import load_psse, add_dyr
from uqgrid.simulation.pflow import runpf
from uqgrid.simulation.config import IntegrationConfig

def parse_args():
    # Find the index of '--' if it exists
    try:
        split_index = sys.argv.index('--')
        script_args = sys.argv[1:split_index]
        petsc_args = sys.argv[split_index+1:]
    except ValueError:
        script_args = sys.argv[1:]
        petsc_args = []

    # Your original argument parser
    parser = argparse.ArgumentParser(description="Run power system dynamics integration.")
    parser.add_argument('--raw', type=str, required=True, help='Path to RAW file.')
    parser.add_argument('--dyr', type=str, required=True, help='Path to DYR file.')
    parser.add_argument('--zfault', type=float, default=0.03, help='Perturbation fault.')
    parser.add_argument('--tend', type=float, default=10.0, help='Integration end time in seconds.')
    parser.add_argument('--dt', type=float, default=1.0/120.0, help='Time step in seconds.')
    parser.add_argument('--steps', type=int, default=-1, help='Number of integration steps.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--comp_sens', action='store_true', help='Compute sensitivities.')
    parser.add_argument('--fsolve', action='store_true', help='Use fsolve for nonlinear equations.')
    parser.add_argument('--ton', type=float, default=0.25, help='Fault activation time.')
    parser.add_argument('--toff', type=float, default=0.4, help='Fault deactivation time.')
    parser.add_argument('--petsc', action='store_true', help='Enable PETSc integration.')

    # Parse only the script arguments
    args = parser.parse_args(script_args)

    # If PETSc is enabled, add its arguments to sys.argv
    if args.petsc and petsc_args:
        sys.argv = [sys.argv[0]] + petsc_args
    
    try:
        config = IntegrationConfig(
            tend=args.tend,
            dt=args.dt,
            ton=args.ton,
            toff=args.toff,
            steps=args.steps,
            power_injection=False,
            verbose=args.verbose,
            comp_sens=args.comp_sens,
            fsolve=args.fsolve,
            petsc=args.petsc
        )
    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)

    return args.raw, args.dyr, args.zfault, config

def main():
    raw, dyr, zfault, config = parse_args()

    # Check if data files exist
    if not os.path.isfile(raw):
        print(f"Error: Data file not found at {raw}")
        sys.exit(1)
    if not os.path.isfile(dyr):
        print(f"Error: Data file not found at {dyr}")
        sys.exit(1)

    # Load static file
    psys = load_psse(raw_filename=raw)

    # Add dynamics
    add_dyr(psys, dyr)

    # Add fault and create initial data structures
    psys.add_busfault(1, zfault, 0.01)
    psys.createYbusComplex()

    # Set up parameters
    print(f"Number of loads (parameters): {psys.nloads}")
    pmax = np.ones(psys.nloads)
    pmin = np.zeros(psys.nloads)

    pnom = pmin + 0.5 * (pmax - pmin)
    psys.set_load_parameters(pnom)

    # Run integration
    results = integrate_system(psys, config)

    bus_idx = psys.genspeed_idx_set()

    for bus in bus_idx:
        label = "generator at bus %d" % (bus)
        plt.plot(results["tvec"], results["history"][bus,:], label=label, color='blue')
    plt.show()
if __name__ == "__main__":
    main()