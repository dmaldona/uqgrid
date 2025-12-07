# Power Grid Simulation Scripts

This folder contains scripts for generating, monitoring, and recovering power grid transient stability simulation scenarios.

## Overview

The simulation pipeline generates perturbed operating scenarios for power grid models, runs transient stability simulations for various fault conditions, and produces datasets suitable for machine learning applications.

```
scripts/run/
├── generate_scenarios.py    # Main scenario generation and simulation
├── monitor.py               # Real-time simulation progress monitoring
├── recovery_tool.py         # Recovery utilities for failed scenarios
└── config/                  # Configuration files
    ├── config_IEEE-9.json
    ├── config_IEEE-39.json
    ├── config_ACTIVSg200.json
    └── config_ACTIVSg500.json
```

## Quick Start

```bash
# 1. Run a simulation campaign
python generate_scenarios.py config/config_IEEE-9.json

# 2. (Optional) Monitor progress in another terminal
python monitor.py

# 3. Analyze results and compute TSI
python TSI_analysis.py
```

---

## Scripts

### generate_scenarios.py

Main script for generating perturbed scenarios and running transient stability simulations.

**Features:**
- Multiplicative perturbations with configurable noise distributions (normal/uniform)
- Independent control over load and generator perturbations
- Power factor preservation
- Generator limit enforcement (clamping)
- Generation-load balance maintenance
- Parallel execution with checkpointing
- Continuation mode to add more samples to existing runs

**Usage:**

```bash
# Run with a configuration file
python generate_scenarios.py config/config_IEEE-9.json

# Generate default config files for all supported models
python generate_scenarios.py --generate-configs --config-dir config/

# Continue from existing simulation, adding more samples
python generate_scenarios.py config/config_IEEE-9.json --continue --additional-samples 10
```

**Output Files:**
- `simulation_data/scenario_*.npz` - Per-scenario simulation results
- `simulation_log.json` - Simulation outcomes and metadata
- `scenario_metadata.json` - Scenario parameter definitions

---

### monitor.py

Real-time dashboard for monitoring simulation progress.

**Features:**
- Live progress tracking with ETA estimation
- CPU and memory utilization monitoring
- Success/failure rate statistics
- Failure analysis by fault location

**Usage:**

```bash
# Run in a separate terminal while simulation is running
python monitor.py

# Custom refresh interval (default: 5 seconds)
python monitor.py --interval 10

# Monitor a specific log file
python monitor.py --log-file path/to/simulation_log.json
```

---

### recovery_tool.py

Interactive tool for analyzing and recovering from simulation failures.

**Features:**
- Analyze simulation logs to identify failed scenarios
- Group failures by fault location for pattern identification
- Verify data file integrity
- Generate retry scripts for failed scenarios only

**Usage:**

```bash
# Run interactive recovery menu
python recovery_tool.py
```

---

### TSI_analysis.py

Is located in `/scripts/postprocessing`

Computes Transient Stability Index (TSI) from simulation results and exports datasets for machine learning.

**TSI Formula:**
```
TSI = (2π - Δ_max) / (2π + Δ_max) × 100
```
where Δ_max is the maximum spread between generator rotor angles.

- **TSI > 0**: Stable operation
- **TSI < 0**: Unstable operation

**Usage:**

```bash
# Default: use final TSI (steady-state stability)
python TSI_analysis.py

# Use minimum TSI across all time steps (worst-case stability)
python TSI_analysis.py --tsi-mode min

# Custom output path
python TSI_analysis.py -o my_dataset.npz --tsi-mode min

# Show all options
python TSI_analysis.py --help
```

**Output:**
- `.npz` file containing input features (P, Q) and TSI values for ML training

---

## Configuration

Configuration files are stored in `config/` and define all simulation parameters.

### Structure

```json
{
    "model": {
        "name": "IEEE-9",
        "raw": "../data/ieee9_v33.raw",
        "dyr": "../data/ieee9bus_gov.dyr",
        "n_bus": 9
    },
    "scenarios": {
        "samples_per_fault_location": 5,
        "fault_impedances": [0.00001],
        "fault_locations": "all"
    },
    "execution": {
        "n_jobs": 5,
        "batch_size": 10,
        "checkpoint_interval": 5
    },
    "perturbation": {
        "load_noise_type": "normal",
        "load_noise_var": 0.25,
        "gen_noise_type": "normal",
        "gen_noise_var": 0.25,
        "balance_generation": true,
        "perturb_loads": true,
        "perturb_gens": true,
        "keep_power_factor": true,
        "clamp_gens": true
    }
}
```

### Configuration Options

| Section | Parameter | Description |
|---------|-----------|-------------|
| **model** | `name` | Model identifier |
| | `raw` | Path to PSS/E RAW file |
| | `dyr` | Path to PSS/E DYR file |
| | `n_bus` | Number of buses in the model |
| **scenarios** | `samples_per_fault_location` | Number of perturbation samples per fault |
| | `fault_impedances` | List of fault impedance values [p.u.] |
| | `fault_locations` | `"all"` or list of bus indices |
| **execution** | `n_jobs` | Number of parallel workers |
| | `batch_size` | Scenarios per batch |
| | `checkpoint_interval` | Checkpoint every N batches |
| **perturbation** | `load_noise_type` | `"normal"` or `"uniform"` |
| | `load_noise_var` | Noise variance for loads |
| | `gen_noise_type` | `"normal"` or `"uniform"` |
| | `gen_noise_var` | Noise variance for generators |
| | `balance_generation` | Rebalance Pg to match Pl |
| | `perturb_loads` | Apply perturbations to loads |
| | `perturb_gens` | Apply perturbations to generators |
| | `keep_power_factor` | Maintain Q/P ratio |
| | `clamp_gens` | Enforce generator limits |

### Available Models

| Model | Buses | Description |
|-------|-------|-------------|
| IEEE-9 | 9 | Small test system |
| IEEE-39 | 39 | New England test system |
| ACTIVSg200 | 200 | Synthetic 200-bus system |
| ACTIVSg500 | 500 | Synthetic 500-bus system |

---

## Recommended Settings

### For High-Performance Computing

For a dual-socket server (e.g., 2× Intel Xeon with 32 cores total, 192GB RAM):

```json
"execution": {
    "n_jobs": 30,
    "batch_size": 100,
    "checkpoint_interval": 10
}
```

### For Desktop/Workstation

```json
"execution": {
    "n_jobs": 4,
    "batch_size": 20,
    "checkpoint_interval": 5
}
```

---

## Workflow Examples

### Complete Simulation Campaign

```bash
# 1. Generate scenarios and run simulations
python generate_scenarios.py config/config_IEEE-39.json

# 2. Monitor progress (in another terminal)
python monitor.py

# 3. If failures occurred, analyze and retry
python recovery_tool.py

# 4. Compute TSI and export dataset
python TSI_analysis.py -o ieee39_dataset.npz
```

### Adding More Samples to Existing Run

```bash
# Initial run with 5 samples per fault location
python generate_scenarios.py config/config_IEEE-9.json

# Later, add 10 more samples
python generate_scenarios.py config/config_IEEE-9.json --continue --additional-samples 10

# Add another 20 samples
python generate_scenarios.py config/config_IEEE-9.json --continue --additional-samples 20
```

### Excluding Problematic Buses

If certain buses cause failures, specify fault locations explicitly in the config:

```json
"scenarios": {
    "samples_per_fault_location": 5,
    "fault_impedances": [0.00001],
    "fault_locations": [0, 1, 2, 3, 5, 6, 7, 8]
}
```

---

## Output Data Format

### simulation_log.json

```json
{
    "scenario-uuid": {
        "sample_idx": 0,
        "fault_location": 5,
        "fault_impedance": 0.00001,
        "file": "simulation_data/scenario_xxx.npz",
        "diverged": false
    }
}
```

### scenario_*.npz

Each scenario file contains:
- `state_history` - Time series of all state variables
- `time` - Time vector
- `p_load_scaled`, `q_load_scaled` - Perturbed load values
- `p_gen_scaled`, `q_gen_scaled` - Perturbed generator values

### TSI Dataset (.npz)

- `X` - Input features: shape `(N, 2, Ngen+Nload)` with channels [P, Q]
- `Y` - TSI values: shape `(N, F, Z)` for fault locations × impedances
- `fault_locations`, `fault_impedances` - Parameter arrays
- `meta` - Metadata dictionary

---

## Dependencies

- Python 3.8+
- numpy
- joblib
- matplotlib (for visualization)
- scipy (for MATLAB export)
- uqgrid (power system simulation library)

---

## Troubleshooting

### High Failure Rate

1. Check for mismatched RAW/DYR files (generators without dynamic models)
2. Reduce noise variance (`load_noise_var`, `gen_noise_var`)
3. Try excluding problematic fault locations

### Out of Memory

1. Reduce `batch_size`
2. Reduce `n_jobs`

### Slow Performance

1. Increase `n_jobs` (up to number of physical cores)
2. Increase `batch_size`
3. Ensure numerical libraries use single-threaded mode (set automatically)

---
