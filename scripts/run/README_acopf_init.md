# ACOPF-Initialized Scenario Generator

`generate_scenarios_acopf_init.py` builds ProbML-compatible transient
stability datasets from UQGrid operating-point candidates that are initialized
through ExaJuGO ACOPF before dynamic replay.

This generator is separate from `generate_scenarios.py`. The original generator
does not import ExaJuGO and its behavior is unchanged.

## Workflow

Production mode runs this pipeline for each accepted operating point:

1. Generate a PF-screened UQGrid operating-point candidate using the same
   target-mode candidate logic as `generate_scenarios.py`.
2. Patch an ExaJuGO RAW case with the candidate loads.
3. Run ExaJuGO ACOPF with the patched RAW and base ROP.
4. Import `acopf_system/Basecase_solution.txt` back into a fresh UQGrid system.
5. Validate post-ACOPF power flow in UQGrid.
6. Replay every configured fault in parallel.
7. Compute final and minimum TSI during replay.
8. Append one row to final and min ProbML NPZ files only after all faults pass.

Dense dynamic histories are discarded by default.

## Required Inputs

- A standard UQGrid scenario JSON config with model RAW/DYR paths.
- A Python environment that can import UQGrid and run the existing scenario
  generator helpers.
- A Julia executable.
- An ExaJuGO checkout containing `ACOPF.jl`, `Project.toml`, and
  `Manifest.toml`.
- ExaJuGO base `case.raw` and `case.rop` files for the same system.

Model RAW/DYR paths are resolved from the config path, current directory,
`--uqgrid-root`, `UQGRID_ROOT`, the installed UQGrid package root, then the
script repository root.

ACOPF path settings use this precedence:

1. CLI flags.
2. `acopf_initialization` config section.
3. Environment variables.
4. Built-in defaults where available.

The ACOPF settings are:

| Setting | CLI | Config key | Environment |
| --- | --- | --- | --- |
| Julia executable | `--julia` | `julia` | `JULIA` |
| ExaJuGO root | `--exajugo-root` | `exajugo_root` | `EXAJUGO_ROOT` |
| Base RAW | `--exajugo-base-raw` | `base_raw` | `EXAJUGO_BASE_RAW` |
| Base ROP | `--exajugo-base-rop` | `base_rop` | `EXAJUGO_BASE_ROP` |
| ACOPF timeout | `--acopf-timeout-s` | `acopf_timeout_s` | none |

The timeout defaults to `300` seconds.

## ExaJuGO Dependency Check

Run this once on a new local machine or cluster environment:

```bash
cd "$EXAJUGO_ROOT"
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=. -e 'using Ipopt, JuMP, SCACOPFSubproblems; println("ExaJuGO deps OK")'
```

The generator invokes ExaJuGO as:

```bash
"$JULIA" --project="$EXAJUGO_ROOT" "$EXAJUGO_ROOT/ACOPF.jl" \
  "$case_dir/case.raw" "$case_dir/case.rop" \
  "$case_dir/acopf_solution" "$case_dir/acopf_system"
```

UQGrid replay uses:

```text
<case_dir>/acopf_system/Basecase_solution.txt
```

## Local Commands

These examples use the local paths from the ACOPF handoff. Adjust them on other
machines.

### ACOPF Smoke

Runs one PF-screened candidate through ExaJuGO and post-ACOPF UQGrid PF
validation. It does not run dynamic replay or write ProbML rows.

```bash
/Users/emconsta/venvs/cnf-uqgrid-py313/bin/python \
  /Users/emconsta/Research/REPO/uqgrid-feature-acopf-init/scripts/run/generate_scenarios_acopf_init.py \
  /Users/emconsta/Research/REPO/base-surrogates-power-grid/CNF/838_tsi/config_ACTIVSg500_L1_04_C8.json \
  --smoke-acopf \
  --output-dir /private/tmp/uqgrid_acopf_stage2_smoke \
  --julia /Users/emconsta/.juliaup/bin/julia \
  --exajugo-root /Users/emconsta/Research/REPO/exajugo \
  --exajugo-base-raw /Users/emconsta/Research/REPO/tsSLOPE/example/ACTIVSg500/case.raw \
  --exajugo-base-rop /Users/emconsta/Research/REPO/tsSLOPE/example/ACTIVSg500/case.rop \
  --target-accepted-scenarios 1 \
  --uqgrid-root /Users/emconsta/Research/REPO/uqgrid
```

### Replay Smoke

Runs one ACOPF-initialized operating point, two fault buses, and one impedance.
It writes one-row final and min NPZ files.

```bash
/Users/emconsta/venvs/cnf-uqgrid-py313/bin/python \
  /Users/emconsta/Research/REPO/uqgrid-feature-acopf-init/scripts/run/generate_scenarios_acopf_init.py \
  /Users/emconsta/Research/REPO/base-surrogates-power-grid/CNF/838_tsi/config_ACTIVSg500_L1_04_C8.json \
  --smoke-replay \
  --output-dir /private/tmp/uqgrid_acopf_stage3_smoke \
  --julia /Users/emconsta/.juliaup/bin/julia \
  --exajugo-root /Users/emconsta/Research/REPO/exajugo \
  --exajugo-base-raw /Users/emconsta/Research/REPO/tsSLOPE/example/ACTIVSg500/case.raw \
  --exajugo-base-rop /Users/emconsta/Research/REPO/tsSLOPE/example/ACTIVSg500/case.rop \
  --target-accepted-scenarios 1 \
  --fault-locations 142,143 \
  --n-jobs 2 \
  --uqgrid-root /Users/emconsta/Research/REPO/uqgrid
```

Expected smoke NPZ shapes for ACTIVSg500 with two fault buses and one
impedance:

```text
X       (1, 2, 262)
X_flat  (1, 524)
Y       (1, 2, 1)
```

### One-OP All-Bus Validation

This is the Stage 4 local validation command. It runs one accepted operating
point over all 500 ACTIVSg500 fault buses with six parallel replay workers.

```bash
/Users/emconsta/venvs/cnf-uqgrid-py313/bin/python \
  /Users/emconsta/Research/REPO/uqgrid-feature-acopf-init/scripts/run/generate_scenarios_acopf_init.py \
  /Users/emconsta/Research/REPO/base-surrogates-power-grid/CNF/838_tsi/config_ACTIVSg500_L1_04_C8.json \
  --output-dir /private/tmp/uqgrid_acopf_stage4_allbus_one_op_njobs6 \
  --julia /Users/emconsta/.juliaup/bin/julia \
  --exajugo-root /Users/emconsta/Research/REPO/exajugo \
  --exajugo-base-raw /Users/emconsta/Research/REPO/tsSLOPE/example/ACTIVSg500/case.raw \
  --exajugo-base-rop /Users/emconsta/Research/REPO/tsSLOPE/example/ACTIVSg500/case.rop \
  --target-accepted-scenarios 1 \
  --fault-locations all \
  --n-jobs 6 \
  --uqgrid-root /Users/emconsta/Research/REPO/uqgrid
```

The completed validation produced:

```text
X       (1, 2, 262)
X_flat  (1, 524)
Y       (1, 500, 1)
sample_idx [0]
```

## Production / Cluster Run

On a cluster, prefer environment variables for machine-specific paths and put
outputs on scratch storage.

```bash
export JULIA=/path/to/julia
export EXAJUGO_ROOT=/path/to/exajugo
export EXAJUGO_BASE_RAW=/path/to/ACTIVSg500/case.raw
export EXAJUGO_BASE_ROP=/path/to/ACTIVSg500/case.rop
export UQGRID_ROOT=/path/to/uqgrid

python scripts/run/generate_scenarios_acopf_init.py \
  /path/to/config_ACTIVSg500_L1_04_C8.json \
  --output-dir /scratch/$USER/uqgrid_acopf_init_C8 \
  --fault-locations all \
  --n-jobs 64
```

For the C8 config, the production defaults are:

```text
target_accepted_scenarios = 1000
fault_locations = all
fault_impedances = [1e-4]
probml_basename = tsi_probml_fullinputs_ACTIVSg500
```

Use `--target-accepted-scenarios 1` for validation runs. Do not use that flag
for the full 1000-row dataset unless intentionally limiting the run.

## Restart and Status

Resume a production output directory after interruption:

```bash
python scripts/run/generate_scenarios_acopf_init.py \
  /path/to/config_ACTIVSg500_L1_04_C8.json \
  --output-dir /scratch/$USER/uqgrid_acopf_init_C8 \
  --continue
```

`--continue` validates that final/min NPZ files, progress, scenario metadata,
and simulation log row counts agree before appending more rows. If they do not
agree, the run stops with a repair instruction instead of guessing.

Read status without running ACOPF, candidate generation, or replay:

```bash
python scripts/run/generate_scenarios_acopf_init.py \
  --status \
  --output-dir /scratch/$USER/uqgrid_acopf_init_C8 \
  --probml-basename tsi_probml_fullinputs_ACTIVSg500
```

## Outputs

Production writes these files under `--output-dir`:

| File | Purpose |
| --- | --- |
| `<basename>_final.npz` | Final-time TSI ProbML dataset. |
| `<basename>_min.npz` | Minimum-over-time TSI ProbML dataset. |
| `acopf_init_progress.json` | Accepted count, next sample index, target, latest status. |
| `acopf_init_diagnostics.jsonl` | Per-step diagnostics and rejection records. |
| `acopf_init_diagnostics_summary.json` | Compact rejection and progress summary. |
| `scenario_metadata.json` | Per-fault scenario metadata with explicit TSI fields. |
| `simulation_log.json` | Per-fault replay log with `file: null` in normal mode. |
| `state_metadata.json` | UQGrid state metadata used to select generator angle states. |

Both NPZ files contain these keys:

```text
X
X_flat
Y
sample_idx
fault_locations
fault_impedances
scenario_ids
meta
```

For ACTIVSg500 with all 500 buses and one impedance, each accepted operating
point appends:

```text
X       (n, 2, 262)
X_flat  (n, 524)
Y       (n, 500, 1)
```

`X[i]` stores:

```text
[[pg_acopf_nonzero, pl_acopf],
 [qg_acopf_nonzero, ql_acopf_uqgrid_sign]]
```

The final and min NPZ files intentionally store different scalar TSI targets.

## Stability Interpretation

TSI is computed from the spread in generator rotor angles:

```text
TSI = (2*pi - delta_max) / (2*pi + delta_max) * 100
```

- Stable fault: `TSI > 0`.
- Unstable fault: `TSI <= 0`.
- Stable operating point/scenario: all tested faults are stable.
- Unstable operating point/scenario: at least one tested fault is unstable.

## Disk and Debug Behavior

- Dense histories are discarded by default.
- Final and minimum TSI are computed immediately during replay.
- `simulation_log.json` uses `file: null` in normal mode.
- Accepted ACOPF case directories are deleted by default in production after a
  successful row write.
- Failed ACOPF case directories are kept by default for debugging.

Useful flags:

| Flag | Effect |
| --- | --- |
| `--keep-intermediate-acopf-cases` | Keep accepted ACOPF case directories. |
| `--keep-failed-acopf-cases` / `--no-keep-failed-acopf-cases` | Control failed case retention. |
| `--keep-fault-histories` | Write dense per-fault histories for debugging. |
| `--debug-tracebacks` | Include dynamic replay tracebacks in worker diagnostics. |

## Common Failure Modes

- ExaJuGO returns a nonzero exit code, times out, or does not write
  `Basecase_solution.txt`.
- The post-ACOPF UQGrid PF validation fails or exceeds the configured residual
  tolerance.
- Load or generator ordering does not match between ExaJuGO output and UQGrid.
- Load Q sign conventions are inconsistent.
- Shunt compensation or bus voltage initialization is missing.
- A dynamic fault replay fails; in production, any failed fault rejects the
  whole operating point and no NPZ row is appended.

Use `acopf_init_diagnostics.jsonl`, retained failed ACOPF case directories, and
the captured `acopf_stdout.txt` / `acopf_stderr.txt` files to debug rejected
candidates.
