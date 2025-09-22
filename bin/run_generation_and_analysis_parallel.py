import os
import sys
import time
from datetime import datetime
import importlib
import numpy as np

# Configuration hyperparameters
# CONFIG = {
#     'PowerGridModel': 'ACTIVSg500',
#     'SAMPLES_PER_FAULT_LOCATION': 10,
#     'FAULT_IMPEDANCES': [0.0001, 0.01],
#     'NOISE_TYPE': 'uniform',
#     'NOISE_VAR': 1.0,
#     'BALANCE_GENERATION': True,
#     'N_JOBS': 4,
#     'BATCH_SIZE': 8
# }



# -------------------------------------------------
# 0) built-in defaults   (kept exactly as before)
# -------------------------------------------------
DEFAULT_CONFIG = {
    'PowerGridModel': 'IEEE-9',
    'SAMPLES_PER_FAULT_LOCATION': 2,
    'FAULT_IMPEDANCES': [0.0001, 0.01],
    'NOISE_TYPE': 'uniform',
    'NOISE_VAR': 1.0,
    'BALANCE_GENERATION': True,
    'N_JOBS': 4,
    'BATCH_SIZE': 8
}

# -------------------------------------------------
# 1) add 3 lightweight imports
# -------------------------------------------------
import argparse
import yaml
from pathlib import Path      # already std-lib

# -------------------------------------------------
# 2) parse command-line flags
# -------------------------------------------------
parser = argparse.ArgumentParser(
    description="Scenario generation + TSI analysis pipeline")
parser.add_argument(
    "--config",
    help="Path to YAML file with hyper-parameters (overrides defaults)")
parser.add_argument(
    "--skipsim",
    action="store_true",
    help="Skip scenario generation and run TSI analysis only"
)
parser.add_argument(
    "--skiptsi",
    action="store_true",
    help="Skip TSI analysis and run scenario generation only"
)
parser.add_argument(
    "--set",
    nargs="*",
    metavar="KEY=VAL",
    help="Override individual keys, e.g.  --set NOISE_VAR=0.2 BATCH_SIZE=4")
cli_args, unknown = parser.parse_known_args()

sys.argv = [sys.argv[0]]

# -------------------------------------------------
# 3) begin with defaults → layer YAML → layer --set
# -------------------------------------------------
CONFIG = dict(DEFAULT_CONFIG)          # start with defaults

# (a) YAML file, if provided
if cli_args.config:
    cfg_path = Path(cli_args.config).expanduser()
    with cfg_path.open() as f:
        yaml_cfg = yaml.safe_load(f) or {}
    CONFIG.update(yaml_cfg)

# (b) one-off KEY=VAL overrides
if cli_args.set:
    for pair in cli_args.set:
        if "=" not in pair:
            raise ValueError(f"--set argument '{pair}' is missing '='")
        k, v = pair.split("=", 1)
        # basic automatic type casting
        if k not in CONFIG:
            raise KeyError(f"Unknown CONFIG key: {k}")
        old_type = type(CONFIG[k])
        CONFIG[k] = old_type(eval(v)) if old_type is bool else old_type(v)

# -------------------------------------------------
# 4) print final config for sanity
# -------------------------------------------------
print("=== Active CONFIG ===")
for k, v in CONFIG.items():
    print(f"{k:>25}: {v}")
print("=====================")



def run_generate_scenarios():
    """Run generate_scenarios.py with specified hyperparameters."""
    # Dynamically import the module
    try:
        generate_scenarios = importlib.import_module('generate_scenarios')
    except ImportError:
        print("Error: Could not import generate_scenarios.py")
        sys.exit(1)
    
    # Override the configuration in generate_scenarios
    generate_scenarios.main = lambda: None  # Disable original main
    generate_scenarios.PowerGridModel = CONFIG['PowerGridModel']
    generate_scenarios.SAMPLES_PER_FAULT_LOCATION = CONFIG['SAMPLES_PER_FAULT_LOCATION']
    generate_scenarios.FAULT_IMPEDANCES = CONFIG['FAULT_IMPEDANCES']
    generate_scenarios.N_JOBS = CONFIG['N_JOBS']
    generate_scenarios.BATCH_SIZE = CONFIG['BATCH_SIZE']
    PATH = "/p/lustre1/chiang7/tsi_simulation/data" 
    # Set up model-specific parameters
    if CONFIG['PowerGridModel'] == 'IEEE-9':
        raw = f"{PATH}/ieee9_v33.raw"
        dyr = f"{PATH}/ieee9bus_gov.dyr" 
        n_bus = 9
    elif CONFIG['PowerGridModel'] == 'IEEE-39':
        raw = f"{PATH}/IEEE39_v33.raw"
        dyr = f"{PATH}/IEEE39_gov.dyr"
        n_bus = 39
    elif CONFIG['PowerGridModel'] == 'ACTIVSg200':
        raw = f"{PATH}/ACTIVSG/ACTIVSg200.raw"
        dyr = f"{PATH}/ACTIVSG/ACTIVSg200.dyr"
        n_bus = 200
    elif CONFIG['PowerGridModel'] == 'ACTIVSg500':
        raw = f"{PATH}/ACTIVSG/ACTIVSg500.raw"
        dyr = f"{PATH}/ACTIVSG/ACTIVSg500.dyr"
        n_bus = 500
    else:
        raise ValueError(f"Unsupported PowerGridModel: {CONFIG['PowerGridModel']}")
    
    #fault_locations = list(range(0, n_bus))
    fault_locations = [142, 143, 144, 495,  86, 337, 458,  62, 497,  87, 361, 338, 422,
       124, 140, 423,  61, 141,  81]
    
    # Calculate total scenarios
    total_scenarios = (CONFIG['SAMPLES_PER_FAULT_LOCATION'] * 
                      len(fault_locations) * 
                      len(CONFIG['FAULT_IMPEDANCES']))
    print(f"Running generate_scenarios with {total_scenarios} total scenarios")
    print(f"  - Model: {CONFIG['PowerGridModel']}")
    print(f"  - Samples per fault location: {CONFIG['SAMPLES_PER_FAULT_LOCATION']}")
    print(f"  - Fault locations: {len(fault_locations)}")
    print(f"  - Fault impedances: {CONFIG['FAULT_IMPEDANCES']}")
    print(f"  - Noise type: {CONFIG['NOISE_TYPE']}")
    print(f"  - Noise variance: {CONFIG['NOISE_VAR']}")
    print(f"  - Balance generation: {CONFIG['BALANCE_GENERATION']}")
    
    # Run the simulation
    scenarios = generate_scenarios.sample_scenarios(
        CONFIG['SAMPLES_PER_FAULT_LOCATION'],
        fault_locations,
        CONFIG['FAULT_IMPEDANCES']
    )
    metadata = generate_scenarios.generate_metadata(scenarios)
    
    generate_scenarios.run_simulation_driver_batched(
        raw,
        dyr,
        metadata,
        noise_type=CONFIG['NOISE_TYPE'],
        noise_var=CONFIG['NOISE_VAR'],
        balance_generation=CONFIG['BALANCE_GENERATION'],
        n_jobs=CONFIG['N_JOBS'],
        batch_size=CONFIG['BATCH_SIZE'],
        fix_samp_per_scen = True,
    )

def run_tsi_analysis():
    """Run TSI_analysis_parallel.py and save output with hyperparameter-based filename."""
    try:
        TSI_analysis = importlib.import_module('TSI_analysis_parallel')
    except ImportError:
        print("Error: Could not import TSI_analysis_parallel.py")
        sys.exit(1)
    
    # Run ComputeTSI
    post_data = TSI_analysis.ComputeTSI()
    
    # Generate filename with hyperparameters and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"{CONFIG['PowerGridModel']}_"
        f"samples{CONFIG['SAMPLES_PER_FAULT_LOCATION']}_"
        f"fi{len(CONFIG['FAULT_IMPEDANCES'])}_"
        f"noise{CONFIG['NOISE_TYPE']}_"
        f"var{CONFIG['NOISE_VAR']}_"
        f"{timestamp}.mat"
    )
    fi_str = '_'.join(f"{fi:.4f}" for fi in CONFIG['FAULT_IMPEDANCES'])
    filename = (
            f"{CONFIG['PowerGridModel']}_"
            f"samples{CONFIG['SAMPLES_PER_FAULT_LOCATION']}_"
            f"fi_{fi_str}_"
            f"noise{CONFIG['NOISE_TYPE']}_"
            f"var{CONFIG['NOISE_VAR']}_"
            f"balgen{CONFIG['BALANCE_GENERATION']}_timeseries_"
            f"{timestamp}.mat"
    )

    # Run the modified create_training_samples
    TSI_analysis.create_training_samples(post_data,filename=filename)

def main():
    print("Starting simulation and analysis pipeline...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run generate_scenarios
    if not cli_args.skipsim:
        print("\n=== Running generate_scenarios ===")
        start_time = time.time()
        run_generate_scenarios()
        print(f"generate_scenarios completed in {time.time() - start_time:.2f} seconds")
    else:
        print("\n=== Skipping scenario generation (--skipsim set) ===")

    # Run TSI_analysis
    if not cli_args.skiptsi:
        print("\n=== Running TSI_analysis in parallel ===")
        start_time = time.time()
        run_tsi_analysis()
        print(f"TSI_analysis completed in {time.time() - start_time:.2f} seconds")
    else:
        print("\n=== Skipping TSI computation (--skiptsi set) ===")
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
