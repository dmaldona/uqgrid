# OSL Dataset Scripts

Lightweight scripts for generating Oscillation Source Location style PMU data.

This script is for batch data generation: many cases, no plots, plus a manifest
for training loaders.

## Generate A Small Dataset

Run from the repository root:

```bash
python scripts/osl/generate_dataset.py scripts/osl/configs/activsg200_small.json
```

The sample config uses `data/ACTIVSg200.raw` and `data/ACTIVSg200.dyr`, injects
governor forced oscillations at PSSE bus 49, and sweeps:

- frequencies: `0.6`, `0.8`, `1.0` Hz
- amplitudes: `0.10`, `0.20`
- colored load noise: enabled
- PMU buses: all buses, for stable channel order across cases

Without a config, pass the RAW/DYR paths explicitly:

```bash
python scripts/osl/generate_dataset.py \
  --raw data/ACTIVSg200.raw \
  --dyr data/ACTIVSg200.dyr \
  --outdir outputs/osl_dataset \
  --tend 8.0
```

For a quick smoke run:

```bash
python scripts/osl/generate_dataset.py scripts/osl/configs/activsg200_small.json \
  --outdir /tmp/osl_smoke --tend 0.5 --fo-start 0.1 --limit 1 --overwrite
```

## Output Layout

```text
outputs/osl_dataset/
  manifest.jsonl
  cases/
    case_0000.npz
    case_0000.json
    case_0001.npz
    case_0001.json
```

Each `.npz` stores PMU arrays such as `V_mag`, `V_ang`, `I_mag`, `I_ang`,
`observed_buses_psse`, `branches`, masks, PMU classes, and the PMU time vector.
Each `.json` stores metadata: source label, target, frequency, amplitude,
simulation settings, noise settings, and PMU settings.

`manifest.jsonl` has one row per case with the relative `.npz`/`.json` paths and
the main labels needed by a training script.

## Useful Options

Config files and command-line flags can be combined. Values are applied in this
order:

```text
built-in defaults < JSON config < command-line flags
```

```bash
# Generate only three cases
python scripts/osl/generate_dataset.py scripts/osl/configs/activsg200_small.json --limit 3 --overwrite

# Use 50% randomly observed buses instead of all buses
python scripts/osl/generate_dataset.py scripts/osl/configs/activsg200_small.json --observed-buses "50%" --overwrite

# Sweep two known governor buses
python scripts/osl/generate_dataset.py scripts/osl/configs/activsg200_small.json --fo-buses 49 50 --freqs 0.8 1.0 --overwrite

# Disable colored load noise
python scripts/osl/generate_dataset.py scripts/osl/configs/activsg200_small.json --no-noise --overwrite
```

For plain tensor models, prefer `--observed-buses all` or an explicit
comma-separated bus list so channel order is stable across cases.

## Config File

The sample config is:

```text
scripts/osl/configs/activsg200_small.json
```

It uses the same key names as the command-line options, with underscores instead
of dashes. For example, `fo_start` matches `--fo-start`, and `pmu_rate_hz`
matches `--pmu-rate-hz`.
