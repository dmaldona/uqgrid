import argparse
import os
import sys
from typing import List, Optional

import numpy as np

from uqgrid.core.psydef import Psystem
from uqgrid.io.parse import add_dyr, load_psse
from uqgrid.simulation.config import IntegrationConfig
from uqgrid.simulation.dynamics import integrate_system


def _split_script_and_petsc_args(argv: List[str]):
    if '--' in argv:
        idx = argv.index('--')
        return argv[1:idx], argv[idx + 1 :]
    return argv[1:], []


def _parse_index_list(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return []
    parts = [part.strip() for part in raw.replace(',', ' ').split() if part.strip()]
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid integer value in index list: '{raw}'") from exc


def parse_args(argv: List[str]):
    script_args, petsc_args = _split_script_and_petsc_args(argv)

    parser = argparse.ArgumentParser(
        description="Run power system dynamics with configurable ARKIMEX partitions."
    )
    parser.add_argument('--raw', required=True, help='Path to RAW file.')
    parser.add_argument('--dyr', required=True, help='Path to DYR file.')
    parser.add_argument('--zfault', type=float, default=0.03, help='Fault impedance in pu.')
    parser.add_argument('--tend', type=float, default=1.0, help='Simulation end time in seconds.')
    parser.add_argument('--dt', type=float, default=1.0 / 120.0, help='Time step in seconds.')
    parser.add_argument('--steps', type=int, default=-1, help='Number of fixed steps to run.')
    parser.add_argument('--ton', type=float, default=0.1, help='Fault activation time.')
    parser.add_argument('--toff', type=float, default=0.2, help='Fault clearing time.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging.')
    parser.add_argument('--petsc', action='store_true', help='Enable PETSc integrator backend.')
    parser.add_argument('--arkimex', action='store_true', help='Use ARKIMEX time integrator.')
    parser.add_argument('--slow-diff', default=None, help='Comma or space separated list of slow differential indices.')
    parser.add_argument('--fast-diff', default=None, help='Comma or space separated list of fast differential indices.')
    parser.add_argument('--plot', action='store_true', help='Plot generator speed traces after simulation.')

    args = parser.parse_args(script_args)

    slow_list = _parse_index_list(args.slow_diff)
    fast_list = _parse_index_list(args.fast_diff)

    if slow_list and fast_list:
        print("Error: specify only one of --slow-diff or --fast-diff.")
        sys.exit(1)

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
            petsc=args.petsc,
            arkimex=args.arkimex,
            arkimex_slow_differential=slow_list,
            arkimex_fast_differential=fast_list,
        )
    except ValueError as exc:
        print(f"Configuration Error: {exc}")
        sys.exit(1)

    return args, config


def _prepare_system(raw_path: str, dyr_path: str, zfault: float) -> Psystem:
    psys = load_psse(raw_filename=raw_path)
    add_dyr(psys, dyr_path)
    psys.add_busfault(1, zfault)
    psys.createYbusComplex()
    return psys


def _initialize_loads(psys: Psystem):
    print(f"Number of loads (parameters): {psys.nloads}")
    pmax = np.ones(psys.nloads)
    pmin = np.zeros(psys.nloads)
    pnom = pmin + 0.5 * (pmax - pmin)
    psys.set_load_parameters(pnom)


def main(argv: Optional[List[str]] = None):
    argv = argv or sys.argv
    args, config = parse_args(argv)

    if not os.path.isfile(args.raw):
        print(f"Error: RAW data file not found at {args.raw}")
        sys.exit(1)
    if not os.path.isfile(args.dyr):
        print(f"Error: DYR data file not found at {args.dyr}")
        sys.exit(1)

    psys = _prepare_system(args.raw, args.dyr, args.zfault)
    _initialize_loads(psys)

    results = integrate_system(psys, config)

    if not args.plot:
        return

    import matplotlib.pyplot as plt

    speed_indices = psys.genspeed_idx_set()
    for idx in speed_indices:
        plt.plot(results['tvec'], results['history'][idx, :], label=f'ω[{idx}]')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
