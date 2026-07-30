# API Overview

This section highlights core modules and entry points. Use the interactive
search above or `mkdocstrings` annotations for deeper dives.

## `uqgrid.io.parse`

```python
from uqgrid.io.parse import load_psse, add_dyr
```

- `load_psse(raw_filename: str) -> Psystem`: Parse PSS®E RAW network files.
- `add_dyr(psys: Psystem, dyr_path: str)`: Attach DYR-based dynamic models to an
  existing system.

## `uqgrid.simulation.config`

```python
from uqgrid.simulation.config import (
    IntegrationConfig,
    IntegrationCtx,
    PowerFlowValidationConfig,
)
from uqgrid.simulation.dynamic_limits import (
    DYNAMIC_LIMIT_EVENT_ACTIONS,
    DYNAMIC_LIMIT_EVENT_FIELDS,
    DynamicLimitError,
    DynamicLimitMode,
    LimitedStateDescriptor,
)
```

- `IntegrationConfig`: Validated settings for time integration.
- `PowerFlowValidationConfig`: Optional strict operating-point checks performed
  before dynamic initialization.
- `IntegrationCtx`: Optional container for custom initial conditions and
  parameter overrides.
- `LimitedStateDescriptor`: JSON-safe identity and bounds for a limited state.
- `DynamicLimitMode`: Free, lower-active, or upper-active limiter mode.
- `DynamicLimitError`: Structured dynamic-limit failure with a `.diagnostics`
  dictionary.
- `DYNAMIC_LIMIT_EVENT_FIELDS`: Required keys returned by every limiter event.
- `DYNAMIC_LIMIT_EVENT_ACTIONS`: Supported projection and mode-transition
  action names.

The dynamic-limit module also provides pure state projection, directional
derivative projection, complementarity evaluation, and active-set update
helpers for the supported integrators.

`IntegrationConfig` also includes optional Jacobian diagnostics for non-PETSc
runs:

- `check_jacobian`: Run a finite-difference Jacobian check.
- `jacobian_check_tol`: Absolute tolerance for reporting mismatches.
- `jacobian_check_top_k`: Number of mismatches to report.
- `jacobian_check_csv`: Optional CSV path for mismatch report.

## `uqgrid.simulation.dynamics`

```python
from uqgrid.simulation.dynamics import integrate_system
```

- `integrate_system(psys, config, ctx=None) -> dict`: Core routine for DAE
  integration, supporting pure Python and PETSc backends.
- Helper utilities (preallocation, Jacobians) support custom solvers and are
  covered in developer documentation.

## `uqgrid.simulation.pflow`

```python
from uqgrid.simulation.pflow import (
    PowerFlowValidationError,
    runpf,
    validate_power_flow_solution,
)
```

- `runpf(psys, verbose=False) -> PowerFlowSolution`: Compute steady-state
  voltages and injections before dynamics.
- `validate_power_flow_solution`: Return JSON-safe operating-point diagnostics
  without modifying the solution.
- `PowerFlowValidationError`: Raised before initialization when enabled
  validation fails.
- `compute_pinj_alt`: Alternative formulation for power-injection Jacobians.

## `uqgrid.core.psydef`

The heart of system modeling, including:

- `Psystem`: Container for buses, generators, loads, and faults.
- `BusFault`: Representation of a shunt fault toggled during integration.
- Device and shunt classes that back dynamic models.

!!! note "API stability"
    Modules in `uqgrid.core` and `uqgrid.simulation` follow semantic versioning.
    Experimental utilities in `bin/` scripts may change without notice.

## `uqgrid.models`

Reusable dynamic device implementations (generator, exciter, governor, loads).
Each model follows a consistent interface so they can be mixed and matched in
DYR assemblies.

## Extending the API

- Add new device models under `uqgrid/models/` and register them in
  `uqgrid/core/psydef.py`.
- Expose higher-level workflows via `uqgrid/__init__.py` (not currently done to
  keep import times minimal).
- Document new public functions in this reference section for quick discovery.
