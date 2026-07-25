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
(fault-on time) and `IntegrationConfig.toff` (fault clearing time). The automatic
scheduler currently supports one fault event.

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
    method="cn",
    ton=0.25,
    toff=0.40,
    steps=-1,
    verbose=False,
    comp_sens=False,
    fsolve=False,
    petsc=True,
    enforce_q_limits=True,
    q_limit_tolerance=1e-8,
    max_q_limit_iterations=None,
    power_flow_validation={
        "enabled": False,
        "residual_tolerance": 1e-8,
        "generator_limit_tolerance": 1e-6,
        "voltage_min": None,
        "voltage_max": None,
        "branch_loading_max": None,
        "branch_limit_tolerance": 1e-5,
        "active_set_voltage_tolerance": 1e-6,
    },
    check_jacobian=False,
    jacobian_check_tol=1e-6,
    jacobian_check_top_k=10,
    jacobian_check_csv=None,
)
```

Key fields:

- **tend**: Final simulation time (seconds).
- **dt**: Integration step size (seconds).
- **steps**: Number of nominal advances. A positive value overrides `tend` and
  produces `steps + 1` base samples, including `t=0`.
- **method**: `"beuler"`, `"cn"`, `"herk2"`, or `"herk4"`.
- **ton/toff**: Fault activation and removal times.
- **comp_sens**: Enable adjoint-based sensitivities (requires PETSc and
  `method="cn"`).
- **petsc**: Switch to PETSc-backed integrators for improved robustness and
  adjoint capabilities.
- **enforce_q_limits**: Enforce non-slack PV generator reactive-power limits
  during the power flow used for dynamic initialization.
- **q_limit_tolerance**: Per-unit tolerance for activating a generator
  reactive-power limit.
- **max_q_limit_iterations**: Optional cap on active-set power-flow solves.
- **power_flow_validation**: Optional final operating-point checks performed
  after PF convergence and before dynamic device initialization.
- **check_jacobian**: Run a finite-difference Jacobian check (non-PETSc only).
- **jacobian_check_tol**: Absolute tolerance for reporting FD mismatches.
- **jacobian_check_top_k**: Number of mismatches to report.
- **jacobian_check_csv**: Optional CSV file path for mismatch report.

### Integration methods

| Configuration | Backend |
|---|---|
| `method="beuler", petsc=False` | Native backward Euler |
| `method="beuler", petsc=True` | PETSc `TSBEULER` |
| `method="cn", petsc=True` | PETSc `TSCN` |
| `method="herk2", petsc=False` | HERK2 |
| `method="herk4", petsc=False` | HERK4 |
| `arkimex=True, petsc=True` | PETSc ARKIMEX |

`cn` is explicitly mapped to PETSc `TSCN`. This is not the same as selecting
the midpoint form of `TSTHETA`; PETSc distinguishes endpoint Crank-Nicolson
from its theta-method midpoint formulation for DAEs. See the
[PETSc TSTHETA documentation](https://petsc.org/release/manualpages/TS/TSTHETA/).

`cn` requires PETSc, HERK requires the native backend, and ARKIMEX cannot be
combined with `cn` or HERK. The library default remains `method="beuler"`.
Configurations that previously omitted `method` therefore select backward
Euler. PETSc command-line options may tune solver internals, but options that
override the configured TS type, time step, horizon, exact-final-time policy,
or adaptivity are rejected.

### Initial reactive-power limits

Q-limit enforcement is enabled by default and projects generator reactive
dispatch onto its RAW/MATPOWER bounds before dynamic initialization. Set
`enforce_q_limits=False` only when an unconstrained legacy operating point is
required. If a non-slack PV bus cannot hold its voltage setpoint within
aggregate generator Q capability, the power flow switches that bus to PQ and
solves again. The same initialization is used for PETSc, backward Euler,
HERK2, and HERK4.

This constrains the operating point only. It does not impose generator
reactive-power or exciter field-voltage limits during the dynamic trajectory.

### Final operating-point validation

Validation is disabled by default. When enabled, UQGrid checks the final Newton
residual, finite voltages, generator P/Q limits, optional voltage and branch
loading bounds, Q-limit active-set consistency, and one slack bus per electrical
island. A failed check raises `PowerFlowValidationError` before
`initialize_system()` or PETSc setup.

Successful backward Euler, PETSc, HERK2, and HERK4 runs include the JSON-safe
diagnostics under `results["power_flow_diagnostics"]`.

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
(`history`), and optional adjoint outputs when sensitivities are enabled. The
first column of `history` is the initialized state at `t=0`.

## Interpret the trajectory

Generator speed deviations, bus voltages, and other state variables can be
plotted directly from the result arrays. With `steps=N`, the regular grid runs
from `0` through `N*dt` and contains `N+1` samples. With `steps=-1`, integration
ends exactly at `tend`; a shortened final interval is added when needed.

Exact `ton`, `toff`, and `tend` values are inserted when they are off the regular
grid, so fault events can add samples beyond the base count. Timestamps are
strictly increasing and are stored once. The fault is active on `[ton, toff)`.
At each transition, UQGrid holds differential states fixed, solves the
algebraic equations for the new topology, and stores that post-switch state.
`ton=0` is rejected when a fault is registered because both the initialized and
post-switch states cannot occupy the single `t=0` sample.

```python
import matplotlib.pyplot as plt

speed_idx = psys.genspeed_idx_set()
plt.plot(results["tvec"], results["history"][speed_idx, :].T)
plt.xlabel("Time [s]")
plt.ylabel("Speed deviation [p.u.]")
plt.title("Generator speed response after Bus 1 fault")
plt.show()
```

This normalized time contract changes the axes and potentially the TSI values of
trajectories produced by older releases. Do not append corrected trajectories to
an existing dense-history dataset created with the previous convention.

## Sensitivities and post-processing

Setting `comp_sens=True`, `petsc=True`, and `method="cn"` activates the
adjoint solve and adds:

- `adjoint_cost`: Scalar performance index over the simulated interval.
- `adjoint_gradient_trajectory`: Contribution from the trajectory (`μᵢ`).
- `adjoint_gradient_initial`: Contribution from the initial condition (`λᵢ ∂y₀/∂p`).
- `adjoint_gradient_complete`: Sum of trajectory and initial-condition terms.

Faulted adjoints replay the saved trajectory by time segment and apply the
fixed-differential algebraic projection jump at each topology transition.
Backward Euler remains supported for forward PETSc simulation, but its integral
adjoint is rejected because that combination does not satisfy the repository's
finite-difference validation.

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
