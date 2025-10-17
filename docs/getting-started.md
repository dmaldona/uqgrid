# Getting Started

This guide walks you through installing UQGrid, preparing input data, and
running your first dynamic simulation.

## Prerequisites

- Python 3.8 or newer (Python 3.10 recommended).
- A working C compiler toolchain (for PETSc builds).

Optional components:

- [PETSc](https://petsc.org/release/) + `petsc4py` for adjoint sensitivities.
- [MkDocs](https://www.mkdocs.org/) for local documentation builds.

## Installation

Clone the repository and install UQGrid in editable mode:

```bash
git clone https://github.com/dmaldona/uqgrid.git
cd uqgrid
python -m pip install --upgrade pip
pip install -e .
```

### Development extras

Install additional tooling for linting, tests, and profiling:

```bash
pip install -e ".[dev]"
```

### PETSc optional support

Enable the PETSc-backed integrators and adjoint solvers:

```bash
pip install -e ".[petsc]"
```

PETSc must be compiled with complex-number support and shared libraries. Refer
to the PETSc installation guide for platform-specific instructions.

### Documentation toolchain

Install the documentation extras to build this site locally:

```bash
pip install -e ".[docs]"
```

## Quick start example

The `bin/dynamics_driver.py` script demonstrates a typical workflow. Supply a
RAW/DYR pair (sample files are included under `data/`) and optional PETSc flags:

```bash
python bin/dynamics_driver.py \
  --raw data/IEEE39_v33.raw \
  --dyr data/IEEE39.dyr \
  --ton 0.05 --toff 0.10 --tend 0.5 --dt 0.01 \
  --petsc \
  -- -ts_monitor -snes_monitor
```

Key steps embedded in the script:

1. Load the static network with `load_psse` and enrich it with dynamics via
   `add_dyr`.
2. Register a bus fault using `psys.add_busfault(bus_id, rfault)`.
3. Configure integration parameters with `IntegrationConfig`.
4. Run `integrate_system` to obtain the state history and sensitivities.

## Next steps

- Learn more about configuration options in the
  [Dynamics Simulation guide](user-guide/dynamics-simulation.md).
- For contributing or extending UQGrid, see the
  [Developer Guide](developer-guide/contributing.md).
