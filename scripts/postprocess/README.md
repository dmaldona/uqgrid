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

# Visualize TSI distributions
python /path/to/scripts/postprocessing/TSI_histogram_utils.py
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

Visualization utilities for exploring TSI distributions across simulation campaigns.

**Features:**
- Publication-quality histogram plots
- Aggregate statistics across all scenarios
- Per-scenario analysis for detailed inspection
- Automatic statistics annotation (mean, std, counts)
- Flexible figure saving (PNG, PDF, SVG)

**Usage:**

```bash
# Generate example histograms
python TSI_histogram_utils.py
```

**Programmatic Usage:**

```python
from TSI_histogram_utils import (
    load_tsi_data,
    plot_histogram_all_samples,
    plot_histogram_single_scenario
)

# Load dataset for custom analysis
data = load_tsi_data("tsi_probml_fullinputs.npz")
Y = data["Y"]  # Shape: (N, F, Z)

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
| `plot_histogram_all_samples(...)` | Aggregate histogram across all scenarios |
| `plot_histogram_single_scenario(...)` | Histogram for one operating condition |

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
│    ├── Load TSI dataset                                     │
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

# 3. Visualize distributions
python -c "
from TSI_histogram_utils import plot_histogram_all_samples
import matplotlib.pyplot as plt
plot_histogram_all_samples('ieee39_tsi.npz', save_path='ieee39_hist.png')
plt.show()
"
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
