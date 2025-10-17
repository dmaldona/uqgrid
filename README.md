# UQGrid

[![CI](https://img.shields.io/github/actions/workflow/status/dmaldona/uqgrid/ci.yml?branch=main&label=CI)](https://github.com/dmaldona/uqgrid/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue)](https://dmaldona.github.io/uqgrid/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Uncertainty quantification for the electrical grid.**

UQGrid is a Python framework for transient-stability simulation, adjoint-based
sensitivities, and probabilistic analysis of PSS®E-style power-system models.
Documentation is now available at
[dmaldona.github.io/uqgrid](https://dmaldona.github.io/uqgrid/).

## Features

- Power system modeling with PSS/E format files
- Dynamic simulation with DAE integration
- First and second-order parameter sensitivities
- Adjoint methods for efficient gradient computation (requires PETSc)
- Generator, exciter, governor, and load models

## Installation

```bash
git clone https://github.com/dmaldona/uqgrid.git
cd uqgrid
python -m pip install --upgrade pip
pip install -e .
```

Optional extras:

- `pip install -e ".[dev]"` — linting, testing, and profiling tools.
- `pip install -e ".[petsc]"` — PETSc/TS integrator and adjoint sensitivities.
- `pip install -e ".[docs]"` — MkDocs + mkdocstrings documentation toolchain.

## Quick Start

```python
import numpy as np
from uqgrid.io.parse import load_psse, add_dyr
from uqgrid.simulation.config import IntegrationConfig
from uqgrid.simulation.dynamics import integrate_system

# Load a power system and attach dynamics
psys = load_psse("data/ieee9_v33.raw")
add_dyr(psys, "data/ieee9bus.dyr")

# Add a shunt fault at bus 1
psys.add_busfault(1, 0.01)
psys.createYbusComplex()
psys.set_load_parameters(np.zeros(psys.nloads))

# Configure and run the simulation
config = IntegrationConfig(
    tend=2.0,
    dt=1.0 / 120.0,
    ton=0.10,
    toff=0.15,
    petsc=True,
)

results = integrate_system(psys, config)
```

## Examples

Explore the `/bin/` directory for runnable scripts:

- `dynamics_driver.py` — Full PETSc-based workflow with plotting hooks.
- `generate_scenarios.py` — Monte Carlo batch execution utilities.

Each script accepts `--help` for usage details.

## Documentation

The MkDocs site hosts tutorials, developer guides, and reference material:

- **Online**: [dmaldona.github.io/uqgrid](https://dmaldona.github.io/uqgrid/)
- **Local build**:

    ```bash
    pip install -e ".[docs]"
    make docs-serve
    ```

The site rebuilds automatically during development and is deployed from the
`main` branch via GitHub Actions.

## Testing

```bash
make test        # full pytest suite
make test-fast   # skip PETSc-heavy adjoint tests
```

Adjoint tests require PETSc (`pip install -e ".[petsc]"`).

## Requirements

- Python 3.8+
- NumPy, SciPy, Numba, NetworkX, Matplotlib, Pydantic
- PETSc4py (optional, for adjoint sensitivity analysis)
- MkDocs + mkdocs-material (optional, for documentation builds)

## License

MIT License - see LICENSE file for details.

## Contact

**Author**: D. Adrian Maldonado  
**Email**: maldonadod AT anl.gov
**Institution**: Argonne National Laboratory

## Acknowledgements

This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.
