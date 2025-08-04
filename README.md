# UQGrid

**Uncertainty quantification for the electrical grid.**

UQGrid is a Python package for performing uncertainty quantification-type computations in power grid systems. It provides tools for integrating differential-algebraic equations (DAEs) and computing first and second-order sensitivities with respect to system parameters.

## Features

- Power system modeling with PSS/E format files
- Dynamic simulation with DAE integration
- First and second-order parameter sensitivities
- Adjoint methods for efficient gradient computation (requires PETSc)
- Generator, exciter, governor, and load models

## Installation

### For Development

```bash
git clone https://github.com/dmaldona/uqgrid.git
cd uqgrid
pip install -e .
```

### With PETSc Support (for adjoint sensitivity analysis)

```bash
pip install -e ".[petsc]"
```

### With Development Tools

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
import uqgrid

# Load a power system
psys = uqgrid.load_psse("data/ieee9_v33.raw")
uqgrid.add_dyr(psys, "data/ieee9bus.dyr")

# Add a fault
psys.add_busfault(1, 0.01, 0.01)
psys.createYbusComplex()
psys.set_load_parameters(np.zeros(psys.nloads))

# Configure and run simulation
config = uqgrid.IntegrationConfig(
    tend=2.0,
    dt=1.0/120.0,
    ton=0.1,    # fault on time
    toff=0.15,  # fault off time
    petsc=True  # use PETSc for adjoint (if available)
)

results = uqgrid.integrate_system(psys, config)
```

## Examples

The `/bin/` directory contains examples:
- IEEE 9-bus test case
- New England test case  
- Sensitivity analysis examples

## Testing

```bash
# Run all tests
make test

# Run fast tests only (skip slow adjoint tests)
make test-fast
```

For adjoint tests (requires PETSc):
```bash
pytest --adjoint-tests
```

## Requirements

- Python 3.8+
- NumPy, SciPy, Numba, NetworkX, Matplotlib, Pydantic
- PETSc4py (optional, for adjoint sensitivity analysis)

## License

MIT License - see LICENSE file for details.

## Contact

**Author**: D. Adrian Maldonado  
**Email**: maldonadod AT anl.gov
**Institution**: Argonne National Laboratory

## Acknowledgements

This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.
