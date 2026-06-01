# Dynamics Simulation

UQGrid models power-system dynamics as differential-algebraic equations (DAEs).
This guide covers the core APIs for preparing a system, configuring disturbances,
executing time integration, and interpreting the results.

## Load a system

```python
from uqgrid.io.parse import load_psse, add_dyr

psys = load_psse(raw_filename="data/IEEE39_v33.raw")
add_dyr(psys, "data/IEEE39.dyr")
```

`load_psse` parses the steady-state network description while `add_dyr` attaches
dynamic device models (generators, exciters, governors, loads, and monitors).

## Configure disturbances

Add shunt faults or other events before integration. Fault timing is controlled
externally via the integration settings.

```python
psys.add_busfault(bus=1, rfault=0.01)
```

The first fault registered is toggled automatically using `IntegrationConfig.ton`
(fault-on time) and `IntegrationConfig.toff` (fault clearing time). Multiple
faults can be registered when more elaborate sequencing is needed.

## Initialize with a power flow

```python
from uqgrid.simulation.pflow import runpf

psys.createYbusComplex()
v_init, s_inj = runpf(psys, verbose=False)
```

The solved voltages seed the dynamic state vector and ensure algebraic
constraints are satisfied at the start of the simulation.

## Build an integration configuration

`IntegrationConfig` encapsulates solver and timing options. Validation is handled
by [Pydantic](https://docs.pydantic.dev/).

```python
from uqgrid.simulation.config import IntegrationConfig

config = IntegrationConfig(
    tend=10.0,
    dt=1.0 / 120.0,
    ton=0.25,
    toff=0.40,
    steps=-1,
    verbose=False,
    comp_sens=False,
    fsolve=False,
    petsc=True,
    check_jacobian=False,
    jacobian_check_tol=1e-6,
    jacobian_check_top_k=10,
    jacobian_check_csv=None,
)
```

Key fields:

- **tend**: Final simulation time (seconds).
- **dt**: Integration step size (seconds).
- **steps**: Optional fixed number of steps (overrides `tend` when positive).
- **ton/toff**: Fault activation and removal times.
- **comp_sens**: Enable adjoint-based sensitivities (requires PETSc).
- **petsc**: Switch to PETSc-backed integrators for improved robustness and
  adjoint capabilities.
- **check_jacobian**: Run a finite-difference Jacobian check (non-PETSc only).
- **jacobian_check_tol**: Absolute tolerance for reporting FD mismatches.
- **jacobian_check_top_k**: Number of mismatches to report.
- **jacobian_check_csv**: Optional CSV file path for mismatch report.

### Jacobian diagnostics (optional)

When running without PETSc, you can enable a finite-difference Jacobian check:

```python
config = IntegrationConfig(
    check_jacobian=True,
    jacobian_check_tol=1e-6,
    jacobian_check_top_k=10,
    jacobian_check_csv="jacobian_mismatches.csv",
)
```

The solver prints the top mismatches (row/column labels included) and can
optionally write a CSV report.

## Run the integrator

```python
from uqgrid.simulation.dynamics import integrate_system

results = integrate_system(psys, config)
```

The solver returns a dictionary with time stamps (`tvec`), state trajectory
(`history`), and optional adjoint outputs when sensitivities are enabled.

When PETSc is unavailable, UQGrid falls back to the pure-Python integrator with
the same interface.

## Interpret the trajectory

Generator speed deviations, bus voltages, and other state variables can be
plotted directly from the result arrays. Fault transitions occur immediately
after the step in which the simulation time first exceeds `ton` or `toff`; a
zero-step Newton solve re-establishes the algebraic manifold at each transition.

```python
import matplotlib.pyplot as plt

speed_idx = psys.genspeed_idx_set()
plt.plot(results["tvec"], results["history"][speed_idx, :].T)
plt.xlabel("Time [s]")
plt.ylabel("Speed deviation [p.u.]")
plt.title("Generator speed response after Bus 1 fault")
plt.show()
```

## Sensitivities and post-processing

Setting `comp_sens=True` and `petsc=True` activates the adjoint solve and adds:

- `adjoint_cost`: Scalar performance index over the simulated interval.
- `adjoint_gradient_trajectory`: Contribution from the trajectory (`μᵢ`).
- `adjoint_gradient_initial`: Contribution from the initial condition (`λᵢ ∂y₀/∂p`).
- `adjoint_gradient_complete`: Sum of trajectory and initial-condition terms.

Example: rank loads by their influence on the monitored quantity.

```python
import numpy as np

grad = results["adjoint_gradient_complete"]
per_load = grad.reshape(psys.nloads, 2)
magnitudes = np.linalg.norm(per_load, axis=1)
for load, value in zip(psys.loads, magnitudes):
    print(f"{load.name:>12}: {value: .3e}")
```

## Batch studies

Use `bin/generate_scenarios.py` for Monte Carlo sweeps that sample load
perturbations, execute dynamics simulations in parallel, and store summary
statistics. Customize the script or import its helpers when building larger
workflows.

!!! tip "Data management"
    Large sweeps can create multi-gigabyte traces. Use the script's output
    options (e.g., `--outdir`) to segment runs by study.
