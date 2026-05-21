# Power Grid Simulation Scripts

This folder contains scripts for generating, monitoring, and recovering power grid transient stability simulation scenarios.
The commands below assume they are run from `scripts/run/`.

## Overview

The simulation pipeline generates perturbed operating scenarios for power grid models, runs transient stability simulations for various fault conditions, and produces datasets suitable for machine learning applications.

```
scripts/run/
├── generate_scenarios.py    # Main scenario generation and simulation
├── monitor.py               # Real-time simulation progress monitoring
├── recovery_tool.py         # Recovery utilities for failed scenarios
└── config/                  # Configuration files
    ├── config_IEEE-9.json
    ├── config_IEEE-9_stress.json
    ├── config_IEEE-39.json
    └── config_IEEE-39_stress.json
```

## Quick Start

```bash
# 1. Run a simulation campaign
python generate_scenarios.py config/config_IEEE-9.json

# 2. (Optional) Monitor progress in another terminal
python monitor.py

# 3. Analyze results and compute TSI
python ../postprocess/TSI_analysis.py
```

---

## Scripts

### generate_scenarios.py

Main script for generating perturbed scenarios and running transient stability simulations.

**Features:**
- Multiplicative perturbations with configurable noise distributions (`normal`, `uniform`, or `none`)
- Independent control over load and generator perturbations
- Power factor preservation
- Deterministic load stress via `load_scale` and `load_mean_shift`
- Generator limit enforcement (clamping)
- Generation-load balance maintenance
- Optional PF-aware operating-point preparation before dynamics
- Optional non-slack generation rebalance and slack mismatch redistribution
- Optional PF-loss compensation using automatic generator participation policies
- Optional rejection of non-converged or over-stressed operating points
- Optional accepted operating-point target mode with complete fault grids
- Per-scenario diagnostics for voltage, slack burden, generator limits, branch loading, and PF residual
- Parallel execution with checkpointing
- Continuation mode to add more samples to existing runs
- Configurable integration parameters (simulation time, fault timing, solver options)

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
- `scenario_diagnostics.jsonl` - Per-scenario operating-point diagnostics when stress/PF screening is active; name is configurable
- `scenario_diagnostics_summary.json` - Compact summary of diagnostics and rejection reasons; name is configurable
- `simulation_checkpoint.json` - Temporary checkpoint for interrupted fixed-grid runs; removed after successful completion

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

Located in `scripts/postprocess/`.

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
python ../postprocess/TSI_analysis.py

# Use minimum TSI across all time steps (worst-case stability)
python ../postprocess/TSI_analysis.py --tsi-mode min

# Custom output path
python ../postprocess/TSI_analysis.py -o my_dataset.npz --tsi-mode min

# Show all options
python ../postprocess/TSI_analysis.py --help
```

**Output:**
- `.npz` file containing input features (P, Q) and TSI values for ML training

---

## Configuration

Configuration files are stored in `config/` and define all simulation parameters.
The baseline configs, such as `config_IEEE-9.json` and `config_IEEE-39.json`,
intentionally keep the original minimal schema. They do not need the newer
`operating_point`, `load_scale`, `target_accepted_scenarios`, or related stress
keys because `generate_scenarios.py` supplies backward-compatible defaults.

Use separate `*_stress.json` configs when you want to opt into the newer
PF-aware workflow: deterministic load scaling, AC PF screening, loss
compensation, Q-limit mitigation, diagnostics, and accepted operating-point
target mode. Keeping both files is recommended:

- `config_*.json`: legacy/default behavior and broad compatibility.
- `config_*_stress.json`: explicit stressed operating-point generation.

### Structure

The example below shows the expanded schema with optional stress/PF fields. A
baseline config may omit those optional fields and still run.

```json
{
    "model": {
        "name": "IEEE-9",
        "raw": "../../data/ieee9_v33.raw",
        "dyr": "../../data/ieee9bus_gov.dyr",
        "n_bus": 9
    },
    "scenarios": {
        "samples_per_fault_location": 5,
        "fault_impedances": [0.00001],
        "fault_locations": "all",
        "target_accepted_scenarios": null,
        "max_total_attempts": null
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
        "clamp_gens": true,
        "load_scale": 1.0,
        "load_mean_shift": 0.0,
        "generation_dispatch_init": "perturbed"
    },
    "operating_point": {
        "enabled": false,
        "run_power_flow": true,
        "rebalance_non_slack": true,
        "redistribute_slack_mismatch": true,
        "rebalance_policy": "headroom",
        "loss_compensation": false,
        "loss_compensation_tolerance_pu": 1e-4,
        "loss_compensation_policy": "headroom",
        "q_limit_mitigation": false,
        "q_limit_mitigation_tolerance_pu": 1e-6,
        "q_limit_mitigation_max_passes": 10,
        "q_limit_mitigation_top_n": 10,
        "max_iterations": 5,
        "max_attempts_per_scenario": 50,
        "pf_residual_tol": 1e-8,
        "slack_p_tolerance_pu": 1e-4,
        "max_slack_p_deviation_fraction_of_load": 0.02,
        "voltage_min": 0.90,
        "voltage_max": 1.10,
        "gen_limit_tolerance": 1e-6,
        "branch_loading_max": 1.0,
        "diagnostics_file": "scenario_diagnostics.jsonl",
        "diagnostics_summary_file": "scenario_diagnostics_summary.json"
    },
    "integration": {
        "tend": 10.0,
        "dt": 0.008333333333333333,
        "power_injection": false,
        "ton": 0.25,
        "toff": 0.4,
        "verbose": false,
        "petsc": true
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
| | `target_accepted_scenarios` | Optional target count of complete accepted operating-point groups |
| | `max_total_attempts` | Optional cap on candidate operating-point attempts in target mode |
| **execution** | `n_jobs` | Number of parallel workers |
| | `batch_size` | Scenarios per batch |
| | `checkpoint_interval` | Checkpoint every N batches |
| **perturbation** | `load_noise_type` | `"normal"`, `"uniform"`, or `"none"` |
| | `load_noise_var` | Load noise parameter: standard deviation for `"normal"`, target variance for `"uniform"` |
| | `gen_noise_type` | `"normal"`, `"uniform"`, or `"none"` |
| | `gen_noise_var` | Generator noise parameter: standard deviation for `"normal"`, target variance for `"uniform"` |
| | `balance_generation` | Rebalance Pg to match Pl |
| | `perturb_loads` | Apply perturbations to loads |
| | `perturb_gens` | Apply perturbations to generators |
| | `keep_power_factor` | Maintain Q/P ratio |
| | `clamp_gens` | Enforce generator limits |
| | `load_scale` | Deterministic multiplicative load scaling before random perturbations |
| | `load_mean_shift` | Deterministic relative load shift before random perturbations |
| | `generation_dispatch_init` | `"perturbed"` or `"base"` generator dispatch before rebalance |
| **operating_point** | `enabled` | Enable PF-aware operating-point preparation and rejection |
| | `run_power_flow` | Run UQGrid PF before dynamics |
| | `rebalance_non_slack` | Rebalance active power on non-slack generators only |
| | `redistribute_slack_mismatch` | Move residual active slack burden to available non-slack generators |
| | `rebalance_policy` | Automatic participation policy for initial non-slack rebalance |
| | `loss_compensation` | Redistribute PF-estimated active losses away from slack before final screening |
| | `loss_compensation_tolerance_pu` | Stop loss compensation when `abs(slack_p_solved - slack_p_target)` is below this [p.u.]; strict generator P screening also caps the effective tolerance |
| | `loss_compensation_policy` | Automatic participation policy for loss compensation |
| | `q_limit_mitigation` | Clamp violating non-slack PV generator Q and switch those buses to PQ before final screening |
| | `q_limit_mitigation_tolerance_pu` | Q-limit tolerance used before PV-to-PQ switching [p.u.] |
| | `q_limit_mitigation_max_passes` | Maximum PV-to-PQ mitigation passes per scenario attempt |
| | `q_limit_mitigation_top_n` | Number of generator Q violators stored in diagnostics |
| | `max_iterations` | PF/rebalance iterations per attempt |
| | `max_attempts_per_scenario` | Resampling attempts before rejecting a scenario |
| | `pf_residual_tol` | Maximum accepted AC PF residual norm |
| | `slack_p_tolerance_pu` | Absolute slack active-power mismatch tolerance [p.u.] |
| | `max_slack_p_deviation_fraction_of_load` | Slack active mismatch tolerance as fraction of total load |
| | `voltage_min` | Minimum accepted bus voltage [p.u.] |
| | `voltage_max` | Maximum accepted bus voltage [p.u.] |
| | `gen_limit_tolerance` | Maximum accepted generator P/Q limit violation [p.u.] |
| | `branch_loading_max` | Maximum accepted branch loading ratio using `rateA` |
| | `diagnostics_file` | JSONL path for per-scenario diagnostics |
| | `diagnostics_summary_file` | JSON path for aggregate diagnostics summary |
| **integration** | `tend` | Simulation end time [s] |
| | `dt` | Integration time step [s] (default: 1/120) |
| | `power_injection` | Use power injection model |
| | `ton` | Fault onset time [s] |
| | `toff` | Fault clearing time [s] |
| | `verbose` | Enable verbose solver output |
| | `petsc` | Use PETSc solver |

### Perturbation and Operating-Point Workflow

The script builds one pre-fault operating point per `sample_idx`. That same
operating point is reused across all fault locations and fault impedances for
that sample. This is why diagnostics often repeat for multiple fault locations.

For each scenario attempt, the script does the following:

1. Load a fresh RAW/DYR model.
2. Read base load and generator schedules from the RAW file.
3. Apply deterministic load stress:
   ```text
   load_factor = load_scale * (1 + load_mean_shift)
   P_load_center = P_load_base * load_factor
   Q_load_center = Q_load_base * load_factor
   ```
4. If `perturb_loads=true`, apply random load perturbations around the stressed
   load center. If `keep_power_factor=true`, `P_load` and `Q_load` use the same
   multiplicative perturbation so the load `Q/P` ratio is preserved.
5. Initialize generator active-power dispatch:
   - `"perturbed"`: start from randomly perturbed generator `P`.
   - `"base"`: start from the RAW-file generator `P`.
6. If `clamp_gens=true`, clamp scheduled generator `P` to `Pmin/Pmax`.
7. If `operating_point.enabled=false`, use the legacy behavior:
   - if `balance_generation=true`, rebalance total generation across all
     generators to match total load;
   - compute generator `Q` from the generator `Q/P` ratio when
     `keep_power_factor=true`;
   - run the dynamic simulation.
8. If `operating_point.enabled=true`, use the PF-aware behavior:
   - rebalance active power across non-slack generators only;
   - run UQGrid AC power flow;
   - measure slack active/reactive burden, PF residual, voltage range,
     generator limit violations, and branch loading;
   - optionally redistribute residual active slack mismatch to non-slack
     generators with available headroom/footroom;
   - optionally compensate PF-estimated active losses by redistributing
     `slack_p_solved - slack_p_target` to non-slack generators before final
     generator-limit screening;
   - repeat until tolerances are met or `max_iterations` is reached;
   - reject and resample the scenario up to `max_attempts_per_scenario`.
9. For accepted operating points, run the transient simulation with the
   configured fault onset (`ton`), clearing time (`toff`), and fault impedance.

### Accepted Operating-Point Target Mode

If `scenarios.target_accepted_scenarios` is set, the script switches from a
fixed fault-level grid to target mode. In this mode, one scenario means one
accepted load/generator operating point. Each accepted operating point is then
run through every configured `fault_locations × fault_impedances` combination
using the same `P_load`, `Q_load`, `P_gen`, and `Q_gen` arrays.

- PF-rejected candidates do not run any faults and do not count toward the
  target.
- A candidate counts only if every fault simulation succeeds and writes a
  trajectory file.
- A fresh target-mode run samples candidate operating points sequentially from
  sample index `0`, then continues with higher sample indices until the target
  is reached or `max_total_attempts` is exhausted. In target mode,
  `samples_per_fault_location` is retained for configuration compatibility and
  summary display; the stopping criterion is `target_accepted_scenarios`.
- With `--continue`, `target_accepted_scenarios` is interpreted as the desired
  total number of complete accepted operating points, not the number of new
  operating points to add. Existing complete operating-point groups in
  `simulation_log.json` count toward the target, and new sampling starts after
  the largest existing `sample_idx`.
- Fault-level `scenario_metadata.json` and `simulation_log.json` stay compatible
  with postprocessing; each row also records `operating_point_id` and
  `accepted_operating_point_index`.

### Power Factor Behavior

`keep_power_factor=true` preserves the load and generator `Q/P` ratio while
building the scheduled operating point. It does not force generator `Q` to stay
fixed after AC PF. During PF:

- PV-bus generator `P` is fixed and generator `Q` is solved.
- Slack generator `P` and `Q` are solved.
- PQ-bus load `P` and `Q` are fixed.

Therefore, a scenario can pass PF convergence but still violate generator
reactive limits. Those violations are reported as `gen_q_violation_max` and are
rejected when they exceed `gen_limit_tolerance`.

### Feasibility Screening

The operating-point filter is a screening tool, not an ACOPF solver. It checks
whether the prepared operating point satisfies AC PF and user-defined stress
limits. It does not optimize generation cost or solve an OPF.

Common interpretations:

- `pf_residual` near zero means the AC PF equations converged.
- `voltage_min < voltage_min` rejects as `voltage_low`.
- `voltage_max > voltage_max` rejects as `voltage_high`.
- `gen_p_violation_max` or `gen_q_violation_max` above
  `gen_limit_tolerance` rejects as a generator limit violation.
- `branch_loading_max > branch_loading_max` rejects as `branch_overload`.
- `slack_p_deviation` above the slack tolerance rejects as
  `slack_p_deviation`.

For ACOPF-like feasibility screening, use strict limits:

```json
"operating_point": {
    "enabled": true,
    "voltage_min": 0.90,
    "voltage_max": 1.10,
    "gen_limit_tolerance": 1e-6,
    "branch_loading_max": 1.0
}
```

Some RAW cases may already violate strict reactive-power or branch-rating
limits under AC PF. In that case, strict screening can reject every scenario
even when PF converges.

For stressed dynamic simulations where you want PF-converged but overloaded
points, loosen the stress limits deliberately:

```json
"operating_point": {
    "enabled": true,
    "voltage_min": 0.85,
    "voltage_max": 1.10,
    "gen_limit_tolerance": 10.0,
    "branch_loading_max": 2.0
}
```

This accepts points with larger voltage stress, generator reactive violations,
or up to 200% branch loading. These points are useful for stress testing, but
they should not be described as ACOPF-feasible.

### Automatic Participation Policies

`rebalance_policy` and `loss_compensation_policy` use automatic weights only:

- `headroom`: increases use `Pmax - Pg`; decreases use `Pg - Pmin`.
- `capacity`: uses `Pmax - Pmin`, clipped iteratively to available margin.
- `current_dispatch`: uses current `Pg`, clipped iteratively.
- `base_dispatch`: uses RAW/base-case `Pg`, clipped iteratively.
- `equal`: splits equally across eligible non-slack generators, then clips.

`headroom` is the default because it maximizes the chance of staying inside
generator active-power limits. These policies do not solve ACOPF; they only
prepare and screen AC PF operating points.

### Q-Limit Mitigation

`q_limit_mitigation=true` applies a local PF heuristic before final screening:
non-slack PV generators that exceed `Qmin/Qmax` are clamped at the violated
limit, and their buses are temporarily switched from PV to PQ for the next PF
solve. This trades fixed voltage control for fixed reactive output at the
generator limit. The temporary bus-type changes are restored before dynamics
are launched. It is not ACOPF and does not modify core UQGrid PF behavior.

### Available Models

| Model | Buses | Description |
|-------|-------|-------------|
| IEEE-9 | 9 | Small test system |
| IEEE-39 | 39 | New England test system |
| ACTIVSg200 | 200 | Synthetic 200-bus system |
| ACTIVSg500 | 500 | Synthetic 500-bus system |

The default-config generator knows about the ACTIVSg systems, but the tracked
example configs in this directory are the IEEE configs above. ACTIVSg runs
require the corresponding RAW/DYR files to be available locally.

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

### Adjusting Simulation Duration

For longer transient analysis (e.g., 20 seconds with smaller time step):

```json
"integration": {
    "tend": 20.0,
    "dt": 0.005,
    "ton": 0.5,
    "toff": 0.65
}
```

### Load-Stress Scenario Generation

To create a stressed-load run, copy an existing config and change only the
fields you need:

```bash
cp config/config_IEEE-9.json config/config_IEEE-9_stress.json
```

Example: increase all loads by 25%, preserve load power factor, and screen the
resulting operating point before dynamics:

```json
"perturbation": {
    "load_scale": 1.25,
    "load_mean_shift": 0.0,
    "perturb_loads": true,
    "keep_power_factor": true,
    "generation_dispatch_init": "perturbed"
},
"operating_point": {
    "enabled": true,
    "run_power_flow": true,
    "rebalance_non_slack": true,
    "redistribute_slack_mismatch": true,
    "rebalance_policy": "headroom",
    "loss_compensation": true,
    "loss_compensation_tolerance_pu": 1e-4,
    "loss_compensation_policy": "headroom",
    "voltage_min": 0.90,
    "voltage_max": 1.10,
    "gen_limit_tolerance": 1e-6,
    "branch_loading_max": 1.0
}
```

Then run:

```bash
python generate_scenarios.py config/config_IEEE-9_stress.json
```

The random perturbation is still applied when `perturb_loads=true`. To run only
the deterministic load scaling, set:

```json
"perturbation": {
    "load_scale": 1.25,
    "load_mean_shift": 0.0,
    "perturb_loads": false
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
python ../postprocess/TSI_analysis.py -o ieee39_dataset.npz
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

For legacy fixed-grid mode, `--additional-samples` controls how many new sample
indices are appended per fault location.

For target accepted operating-point mode, keep
`scenarios.target_accepted_scenarios` at the desired final total. The script
counts existing complete operating-point groups from `simulation_log.json`,
starts new candidate sampling at the next `sample_idx`, and stops when the
configured total is reached. `--additional-samples` is still required by the CLI
when using `--continue`, but target mode uses `target_accepted_scenarios` and
`max_total_attempts` to decide how much additional sampling to perform.

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
        "diverged": false,
        "rejected": false,
        "diagnostics": {
            "accepted": true,
            "attempts": 1,
            "pf_residual": 1.0e-12,
            "voltage_min": 0.95,
            "voltage_max": 1.04,
            "slack_p_deviation": 0.001,
            "gen_q_violation_max": 0.0,
            "branch_loading_max": 0.80
        }
    }
}
```

### scenario_*.npz

Each scenario file contains:
- `history` - Time series of all state variables
- `tvec` - Time vector
- `p_load_scaled`, `q_load_scaled` - Perturbed load values
- `p_gen_scaled`, `q_gen_scaled` - Perturbed generator values
- `p_load_noise`, `q_load_noise` - Effective load perturbation factors
- `p_gen_noise`, `q_gen_noise` - Effective generator perturbation factors
- `load_scale`, `load_mean_shift` - Deterministic load-stress settings
- `operating_point_id`, `accepted_operating_point_index` - Present for target
  accepted operating-point mode

### scenario_diagnostics.jsonl

When deterministic load stress or operating-point screening is active, the
script writes one JSON object per line to `diagnostics_file`. Resampling
attempts are recorded individually, so rejection summaries include failed
internal attempts rather than only the final accepted/rejected scenario. JSONL
is used because large Monte Carlo runs can append diagnostics incrementally
without rewriting one huge JSON array. If a run stops early, all completed
lines are still readable.

Typical fields:

- `scenario_id`, `sample_idx`, `fault_location`, `fault_impedance`
- `record_type`, `operating_point_id`, `accepted_operating_point_index`
- `accepted`, `reject_reason`, `attempts`, `rebalance_iterations`
- target mode only: `target_accepted_scenarios`, `total_candidate_attempts`,
  `faults_required`, `faults_attempted`, `faults_successful`
- `load_scale`, `load_mean_shift`, `total_p_load`, `total_q_load`
- `pf_converged`, `pf_residual`
- `slack_p_target`, `slack_p_solved`, `slack_p_deviation`
- `slack_q_target`, `slack_q_solved`, `slack_q_deviation`
- `slack_p_limit_violation`
- `loss_compensation_enabled`, `loss_compensation_policy`
- `loss_compensation_requested_pu`, `loss_compensation_applied_pu`,
  `loss_compensation_unresolved_pu`, `loss_compensation_iterations`
- `loss_compensation_effective_tolerance_pu`
- `q_limit_mitigation_enabled`, `q_limit_mitigation_applied`,
  `q_limit_mitigation_passes`
- `q_limit_mitigation_switched_buses`,
  `q_limit_mitigation_switched_generators`, `q_limit_mitigation_events`
- `non_slack_headroom_remaining`, `non_slack_footroom_remaining`
- `voltage_min`, `voltage_max`
- `gen_p_violation_max`, `gen_q_violation_max`
- `gen_q_violation_count`, `gen_q_violation_total_abs`,
  `gen_q_violation_argmax`, `gen_q_violation_top`
- `branch_loading_available`, `branch_loading_max`,
  `branch_overloaded_count`, `branch_loading_argmax`

You can inspect it quickly with:

```bash
# Count accepted/rejected records
python - <<'PY'
import json
from collections import Counter
with open("scenario_diagnostics.jsonl") as f:
    records = [json.loads(line) for line in f]
print("records:", len(records))
print("accepted:", sum(r["accepted"] for r in records))
print("reject reasons:", Counter(r.get("reject_reason") for r in records))
PY
```

### scenario_diagnostics_summary.json

This file contains aggregate counts and min/mean/max values for key stress
metrics. It is the fastest place to check acceptance rate, dominant rejection
reason, voltage range, maximum PF residual, maximum branch loading, and target
mode candidate/fault counts when enabled.

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
2. Reduce noise parameters (`load_noise_var`, `gen_noise_var`)
3. Try excluding problematic fault locations
4. Decrease fault clearing time (`toff`) to reduce fault severity, or increase
   it only when intentionally creating harder stability cases

### Zero Accepted Operating Points

If `operating_point.enabled=true` and every scenario is rejected:

1. Open `scenario_diagnostics_summary.json`.
2. Check `reject_reasons`.
3. If the reason is `voltage_low` or `voltage_high`, inspect
   `voltage_min`/`voltage_max` and adjust voltage thresholds only if those
   voltage levels are acceptable for your study.
4. If the reason is `gen_q_limit`, the AC PF solution requires generator
   reactive power outside RAW-file Q limits. This can happen even in the base
   case for some systems. Use strict tolerance for ACOPF-like feasibility, or
   loosen `gen_limit_tolerance` only for stressed dynamic simulations.
5. If the reason is `branch_overload`, compare `branch_loading_max` to your
   acceptable thermal loading level. `1.0` means 100% of `rateA`.
6. If the reason is `slack_p_deviation`, there was not enough non-slack
   generator headroom/footroom to move active-power mismatch away from slack.
7. If the reason is `gen_p_limit` and `slack_p_limit_violation` is also high,
   enable `loss_compensation=true` or inspect `non_slack_headroom_remaining`.
   If unresolved loss compensation remains nonzero, the non-slack generators
   do not have enough active-power margin under the selected policy.

Remember: loosening limits makes the dynamic simulation run, but it does not
make the operating point ACOPF-feasible.

### Repeated Diagnostics Across Fault Locations

This is expected. The pre-fault operating point is generated per `sample_idx`.
Fault location and impedance affect the dynamic fault simulation, not the
pre-fault PF operating point. Therefore, diagnostics can repeat for every bus
fault associated with the same sample.

### Common Warnings

- `Transformer Magnetizing Impedance not Implemented`: existing parser warning;
  the branch transformer magnetizing impedance is ignored.
- SciPy `_nonlin.py` warnings during PF iterations: usually harmless if the
  final `pf_residual` is small and `pf_converged=true`.

### Suppress Known Parser Warnings

By default, warnings are shown. For production runs where the transformer
magnetizing-impedance warning is expected and too noisy, suppress just that
warning with Python's warning filter:

```bash
PYTHONWARNINGS="ignore:Transformer Magnetizing Impedance not Implemented:UserWarning:uqgrid.io.parse,ignore:invalid value encountered in scalar divide::scipy.optimize._nonlin" \
python generate_scenarios.py config/config_IEEE-39_stress.json
```

If the module-specific filter does not catch it in your environment, use the
same message-level filter without the module qualifier:

```bash
PYTHONWARNINGS="ignore:Transformer Magnetizing Impedance not Implemented:UserWarning,ignore:invalid value encountered in scalar divide::scipy.optimize._nonlin" \
python generate_scenarios.py config/config_IEEE-39_stress.json
```

For repeated production runs, export the filter for the session and unset it
afterward:

```bash
export PYTHONWARNINGS="ignore:Transformer Magnetizing Impedance not Implemented:UserWarning:uqgrid.io.parse,ignore:invalid value encountered in scalar divide::scipy.optimize._nonlin"
python generate_scenarios.py config/config_IEEE-39_stress.json
unset PYTHONWARNINGS
```

### Out of Memory

1. Reduce `batch_size`
2. Reduce `n_jobs`
3. Increase `dt` (larger time step = fewer data points)

### Slow Performance

1. Increase `n_jobs` (up to number of physical cores)
2. Increase `batch_size`
3. Ensure numerical libraries use single-threaded mode (set automatically)
4. Increase `dt` for faster (but less accurate) simulations

### Numerical Instability

1. Reduce `dt` for better accuracy
2. Ensure `petsc` is enabled for robust solving
3. Check that fault timing (`ton`, `toff`) is reasonable

---
