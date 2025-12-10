# Power Grid Simulation Postprocessing

This folder contains scripts for analyzing simulation results and generating datasets for machine learning applications.

## Overview

The postprocessing pipeline takes raw simulation outputs from `/scripts/run` and transforms them into analysis-ready formats, including Transient Stability Index (TSI) computation and visualization.

```
scripts/postprocessing/
├── TSI_analysis.py          # TSI computation and ML dataset export
└── TSI_histogram_utils.py   # TSI distribution visualization
```

## Quick Start

```bash
# Navigate to the directory containing simulation results
cd /path/to/simulation/output

# Compute TSI and export dataset
python /path/to/scripts/postprocessing/TSI_analysis.py

# Display dataset information and statistics
python /path/to/scripts/postprocessing/TSI_histogram_utils.py tsi_probml_fullinputs.npz

# Generate and display TSI histograms
python /path/to/scripts/postprocessing/TSI_histogram_utils.py tsi_probml_fullinputs.npz --histogram
```

---

## Scripts

### TSI_analysis.py

Computes Transient Stability Index (TSI) from simulation results and exports structured datasets for machine learning.

**TSI Formula:**

```
TSI = (2π - Δ_max) / (2π + Δ_max) × 100
```

where Δ_max is the maximum angular spread between generator rotors.

| TSI Value | Interpretation |
|-----------|----------------|
| TSI > 0 | Stable operation |
| TSI < 0 | Unstable operation |
| TSI = 100 | Maximum stability (no deviation) |
| TSI → -100 | Severe instability |

**Features:**
- Load and filter simulation scenarios
- Extract generator rotor angle time series
- Compute TSI (scalar or time series)
- Export datasets in NumPy (.npz) or MATLAB (.mat) format
- Progress tracking with ETA estimation
- Configurable TSI extraction modes

**Usage:**

```bash
# Default: use final TSI (steady-state stability)
python TSI_analysis.py

# Use minimum TSI across all time steps (worst-case stability)
python TSI_analysis.py --tsi-mode min

# Custom output path
python TSI_analysis.py -o my_dataset.npz --tsi-mode final

# Require complete fault grids (no NaN values)
python TSI_analysis.py --require-complete-grid

# Show all options
python TSI_analysis.py --help
```

**CLI Options:**

| Option | Description |
|--------|-------------|
| `-o, --output` | Output file path (default: `tsi_probml_fullinputs.npz`) |
| `--tsi-mode` | `final` (last time step) or `min` (minimum over time) |
| `--require-complete-grid` | Only include samples with all fault combinations |
| `--no-concat` | Save separate X_gen and X_load arrays |
| `--no-flat` | Don't save flattened X_flat array |
| `-q, --quiet` | Reduce output verbosity |

**Required Input Files:**
- `simulation_log.json` - Scenario metadata and outcomes
- `state_metadata.json` - State variable descriptions
- `simulation_data/scenario_*.npz` - Per-scenario simulation results

**Output Format:**

```python
# Load the exported dataset
import numpy as np
data = np.load("tsi_probml_fullinputs.npz", allow_pickle=True)

X = data["X"]           # Shape: (N, 2, Ngen+Nload) - [P, Q] channels
Y = data["Y"]           # Shape: (N, F, Z) - TSI values
X_flat = data["X_flat"] # Shape: (N, 2*(Ngen+Nload)) - flattened

fault_locations = data["fault_locations"]   # (F,) bus indices
fault_impedances = data["fault_impedances"] # (Z,) impedance values
meta = data["meta"].item()                  # Metadata dictionary
```

---

### TSI_histogram_utils.py

Visualization and analysis utilities for exploring TSI distributions and dataset statistics.

**Features:**
- Publication-quality histogram plots
- Comprehensive dataset information display
- Power variable statistics (Pg, Qg, Pl, Ql) extraction and analysis
- Aggregate and per-scenario statistics
- Per-unit (per generator/load) statistics
- Automatic statistics annotation (mean, std, counts, stability breakdown)
- Flexible figure saving (PNG, PDF, SVG)

**Usage:**

```bash
# Display dataset information
python TSI_histogram_utils.py my_dataset.npz

# Display info for a specific scenario
python TSI_histogram_utils.py my_dataset.npz -s 5

# Generate histograms
python TSI_histogram_utils.py my_dataset.npz --histogram

# Generate histograms without interactive display (save only)
python TSI_histogram_utils.py my_dataset.npz --histogram --no-show

# Show per-unit statistics (per generator/load)
python TSI_histogram_utils.py my_dataset.npz --per-unit

# Full analysis with custom output directory
python TSI_histogram_utils.py my_dataset.npz -s 0 --histogram --per-unit -o ./output

# Quiet mode (suppress info, only generate histograms)
python TSI_histogram_utils.py my_dataset.npz -q --histogram

# Show all options
python TSI_histogram_utils.py --help
```

**CLI Options:**

| Option | Description |
|--------|-------------|
| `filepath` | Path to the .npz file (required) |
| `-s, --scenario IDX` | Scenario index to analyze |
| `--histogram` | Generate histogram plots |
| `--no-show` | Don't display plots interactively (save only) |
| `--per-unit` | Display per-unit (per generator/load) statistics |
| `-o, --output-dir DIR` | Directory for output files (default: current) |
| `--bins N` | Number of histogram bins (default: 50) |
| `-q, --quiet` | Suppress dataset info output |

**Programmatic Usage:**

```python
from TSI_histogram_utils import (
    load_tsi_data,
    plot_histogram_all_samples,
    plot_histogram_single_scenario,
    display_dataset_info,
    extract_power_variables,
    display_per_unit_statistics
)

# Display comprehensive dataset information
info = display_dataset_info("tsi_probml_fullinputs.npz")

# Display info for a specific scenario
info = display_dataset_info("tsi_probml_fullinputs.npz", scenario_idx=5)

# Load dataset for custom analysis
data = load_tsi_data("tsi_probml_fullinputs.npz")
Y = data["Y"]  # Shape: (N, F, Z)

# Extract power variables for analysis
powers = extract_power_variables(data)
print(f"Pg mean: {powers['pg'].mean():.4f}")
print(f"Pl range: [{powers['pl'].min():.4f}, {powers['pl'].max():.4f}]")

# Extract power variables for a single scenario
powers_s0 = extract_power_variables(data, scenario_idx=0)

# Display per-unit statistics
display_per_unit_statistics("tsi_probml_fullinputs.npz")

# Plot aggregate histogram (all scenarios combined)
fig1 = plot_histogram_all_samples(
    "tsi_probml_fullinputs.npz",
    bins=100,
    save_path="all_tsi_distribution.png"
)

# Plot histogram for a specific operating condition
fig2 = plot_histogram_single_scenario(
    scenario_idx=42,
    filepath="tsi_probml_fullinputs.npz",
    title="High Load Operating Condition",
    save_path="scenario_42_histogram.pdf"
)

plt.show()
```

**Functions:**

| Function | Description |
|----------|-------------|
| `load_tsi_data(filepath)` | Load TSI dataset from .npz file |
| `display_dataset_info(filepath, scenario_idx, print_output)` | Display comprehensive dataset information |
| `extract_power_variables(data, scenario_idx)` | Extract Pg, Qg, Pl, Ql from dataset |
| `compute_variable_statistics(arr)` | Compute min, max, mean, median, std, percentiles |
| `display_per_unit_statistics(filepath, scenario_idx)` | Show per-generator and per-load statistics |
| `plot_histogram_all_samples(...)` | Aggregate histogram across all scenarios |
| `plot_histogram_single_scenario(...)` | Histogram for one operating condition |

**Example Output (display_dataset_info):**

```
================================================================================
DATASET INFORMATION: tsi_probml_fullinputs.npz
================================================================================

--- STORED ARRAYS ---
  X                    shape=(1000, 2, 30)           dtype=float64
  Y                    shape=(1000, 50, 3)           dtype=float64
  ...

--- METADATA ---
  Ngen: 10
  Nload: 20
  tsi_mode: final
  ...

--- TSI (Y) STATISTICS ---
  Count:     150,000
  Range:     [-100.0000, 100.0000]
  Mean:      45.2341
  Median:    52.1234
  Std:       35.6789

  Stability breakdown:
    Stable (TSI > 0):   120,000 (80.00%)
    Unstable (TSI < 0): 30,000 (20.00%)

--- POWER VARIABLE STATISTICS (ALL SCENARIOS) ---
  Variable          Min          Max        Range         Mean       Median          Std
  ----------------------------------------------------------------------------------
  Pg (gen)       0.1000       1.5000       1.4000       0.8500       0.8200       0.2100
  Qg (gen)      -0.3000       0.5000       0.8000       0.1200       0.1000       0.1500
  Pl (load)      0.2000       2.0000       1.8000       1.1000       1.0500       0.3200
  Ql (load)      0.0500       0.8000       0.7500       0.3500       0.3200       0.1800
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    /scripts/run/                            │
│  generate_scenarios.py → simulation_data/scenario_*.npz     │
│                       → simulation_log.json                 │
│                       → state_metadata.json                 │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               /scripts/postprocessing/                      │
│                                                             │
│  TSI_analysis.py                                            │
│    ├── Load simulation results                              │
│    ├── Extract rotor angles (delta states)                  │
│    ├── Compute TSI time series                              │
│    └── Export ML dataset (.npz)                             │
│              │                                              │
│              ▼                                              │
│  TSI_histogram_utils.py                                     │
│    ├── Display dataset info & statistics                    │
│    ├── Extract power variables (Pg, Qg, Pl, Ql)             │
│    ├── Compute per-scenario & per-unit statistics           │
│    └── Generate visualization plots                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Output Dataset Structure

### Array Dimensions

| Array | Shape | Description |
|-------|-------|-------------|
| `X` | `(N, 2, Ngen+Nload)` | Input features with P and Q channels |
| `X_flat` | `(N, 2*(Ngen+Nload))` | Flattened input features |
| `X_gen` | `(N, 2, Ngen)` | Generator features only (if `--no-concat`) |
| `X_load` | `(N, 2, Nload)` | Load features only (if `--no-concat`) |
| `Y` | `(N, F, Z)` | TSI values per fault condition |
| `sample_idx` | `(N,)` | Original sample indices |
| `fault_locations` | `(F,)` | Fault location bus numbers |
| `fault_impedances` | `(Z,)` | Fault impedance values |

### Metadata Dictionary

```python
meta = {
    "inputs": "full_per_unit",
    "channels": ["P", "Q"],
    "unit_axis_order": "generators_then_loads",
    "Ngen": 10,
    "Nload": 20,
    "tsi_mode": "final",  # or "min"
    "meaning_Y": "TSI at last time step for each (fault_location, fault_impedance)",
    "axes_Y": {"axis0": "fault_location", "axis1": "fault_impedance"}
}
```

---

## Workflow Examples

### Basic Analysis Pipeline

```bash
# 1. Run simulations (in /scripts/run/)
python generate_scenarios.py config/config_IEEE-39.json

# 2. Compute TSI and export dataset
python TSI_analysis.py -o ieee39_tsi.npz

# 3. Display dataset information
python TSI_histogram_utils.py ieee39_tsi.npz

# 4. Generate and save histograms
python TSI_histogram_utils.py ieee39_tsi.npz --histogram --no-show -o ./figures

# 5. Full analysis for specific scenario
python TSI_histogram_utils.py ieee39_tsi.npz -s 0 --histogram --per-unit
```

### Comparing TSI Extraction Modes

```bash
# Final TSI (steady-state)
python TSI_analysis.py -o tsi_final.npz --tsi-mode final

# Minimum TSI (worst-case)
python TSI_analysis.py -o tsi_min.npz --tsi-mode min
```

```python
# Compare distributions
import numpy as np
import matplotlib.pyplot as plt

final = np.load("tsi_final.npz")["Y"].flatten()
minimum = np.load("tsi_min.npz")["Y"].flatten()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(final[~np.isnan(final)], bins=50, range=(-100, 100))
axes[0].set_title("Final TSI (Steady-State)")
axes[1].hist(minimum[~np.isnan(minimum)], bins=50, range=(-100, 100))
axes[1].set_title("Minimum TSI (Worst-Case)")
plt.show()
```

### Machine Learning Integration

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = np.load("tsi_probml_fullinputs.npz", allow_pickle=True)
X = data["X_flat"]  # (N, features)
Y = data["Y"]       # (N, F, Z)

# Predict average TSI across all fault conditions
y_mean = np.nanmean(Y, axis=(1, 2))  # (N,)

# Remove samples with NaN targets
valid = ~np.isnan(y_mean)
X_valid, y_valid = X[valid], y_mean[valid]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_valid, y_valid, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"R² score: {model.score(X_test, y_test):.3f}")
```

---

## Performance Notes

- **TSI Computation**: `ComputeTSI_fast()` uses vectorized operations and is significantly faster than the memory-efficient `ComputeTSI()` alternative.
- **Memory Usage**: Memory-mapped file access (`mmap_mode='r'`) reduces memory footprint for large datasets.
- **Progress Tracking**: Adds minimal overhead (~1% of total runtime).

---

## Troubleshooting

### "File not found" Errors

Ensure you're running from the directory containing simulation outputs:
```bash
ls simulation_log.json state_metadata.json simulation_data/
```

### Empty or NaN-filled Dataset

1. Check that simulations completed successfully:
   ```bash
   python -c "import json; log=json.load(open('simulation_log.json')); print(f'Diverged: {sum(1 for s in log.values() if s.get(\"diverged\"))}')"
   ```

2. Try with `--require-complete-grid` disabled (default) to include partial data.

### Memory Issues with Large Datasets

- Use `ComputeTSI()` instead of `ComputeTSI_fast()` for memory-efficient processing
- Process in smaller batches

---

## Dependencies

- Python 3.8+
- numpy
- matplotlib
- scipy (for MATLAB export)

---

## Related Scripts

| Location | Script | Description |
|----------|--------|-------------|
| `/scripts/run/` | `generate_scenarios.py` | Generate simulation data |
| `/scripts/run/` | `monitor.py` | Monitor simulation progress |
| `/scripts/run/` | `recovery_tool.py` | Recover failed simulations |

---
